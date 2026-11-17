// Original to blis-vllm.

package latencykernel_test

import (
	"testing"
	"time"

	. "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-substrate/contracts/kernel"

	"github.com/inference-sim/blis-vllm/compat/lowering"
	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// SUB-6, SUB-11 (inference-sim/blis-vllm#161): the kernel prices a Muse Spark FULL-graph decode
// step with the launches vLLM v0.33.0's fused kernels leave, not the graph's nodes. The lowering
// states -3 launches a layer up to 128 tokens and -2 above (compat/lowering
// TestMuseSparkOnGB200StatesWhatItsFusedKernelsRemove), so against the same pool lowered
// without the statement the step is cheaper by three launches a layer at 8 and 100 decodes and
// by two at 200 and 300: the per-launch term cancels in the ratio, which is 3 to 2.
func TestMuseSparkDecodeStepsAreCheaperByTheLaunchesItsFusedKernelsRemove(t *testing.T) {
	in := fixtureInputs(t, "muse-gb200-tp8ep8.yaml")
	editEngine(t, &in, func(e *engineconfig.Engine) { e.MaxNumSeqs = 512 })
	fused := build(t, in)

	eng, err := vllmsettings.OfPool(in.Deployment, in.PoolIndex)
	if err != nil {
		t.Fatal(err)
	}
	resolveMemoryUtilization(&eng, in.Scenario.EngineRelease())
	low, err := lowering.Lower(lowering.Input{Scenario: in.Scenario, Deployment: in.Deployment,
		PoolIndex: in.PoolIndex, Model: in.Model, Chip: in.Chip, Rules: in.Rules, Settings: eng})
	if err != nil {
		t.Fatal(err)
	}
	if len(low.Plan.Launches) == 0 {
		t.Fatal("the lowering states no change in Muse Spark's launches")
	}
	low.Plan.Launches = nil
	in.Lowered = low
	graph, err := New(in)
	if err != nil {
		t.Fatal(err)
	}

	decodes := func(n int) kernel.Batch {
		b := kernel.Batch{DecodeThreshold: 1}
		for range n {
			b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096})
		}
		return b
	}
	saved := func(n int) time.Duration {
		return graph.StepTime(decodes(n)).NoOverlap - fused.StepTime(decodes(n)).NoOverlap
	}
	three, two := saved(8), saved(200)
	if three <= 0 || two <= 0 {
		t.Fatalf("saved %v at 8 decodes and %v at 200: the fused kernels must make a step cheaper", three, two)
	}
	near := func(a, b time.Duration) bool { return a-b < 3*time.Nanosecond && b-a < 3*time.Nanosecond }
	if got := saved(100); !near(got, three) {
		t.Errorf("100 decodes save %v, 8 save %v: both are below 128 tokens", got, three)
	}
	if got := saved(300); !near(got, two) {
		t.Errorf("300 decodes save %v, 200 save %v: both are above 128 tokens", got, two)
	}
	if !near(three*2, two*3) {
		t.Errorf("saved %v at 8 decodes and %v at 200: want a ratio of 3 to 2", three, two)
	}
}
