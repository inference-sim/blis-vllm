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

	graph := openWithoutLaunches(t, in)

	saved := func(n int) time.Duration {
		return graph.StepTime(launchDecodes(n)).NoOverlap - fused.StepTime(launchDecodes(n)).NoOverlap
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

func launchDecodes(n int) kernel.Batch {
	b := kernel.Batch{DecodeThreshold: 1}
	for range n {
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096})
	}
	return b
}

// openWithoutLaunches opens the pool of in as the lowering resolves it, less the statement of how
// the engine's fused kernels change its launches: the graph's own count.
func openWithoutLaunches(t *testing.T, in Inputs) *Kernel {
	t.Helper()
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
		t.Fatal("the lowering states no change in the pool's launches")
	}
	low.Plan.Launches = nil
	in.Lowered = low
	k, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// SUB-6, SUB-11: vLLM v0.33.0 compiles MiniMax-M2.5 with FlashInfer's all-reduce and RMSNorm fusion
// (compat/lowering TestTheLoweringStatesTheAllReduceNormFusion): at tensor parallelism 8 on an
// H200 it serves steps of up to 85 tokens, two launches fewer a layer. Against the same pool lowered
// without the statement, a FULL-graph decode step of 64 or 80 requests is cheaper, by the same
// amount (two launches a layer at the per-launch term), and a step of 88 or 128, past the bound, costs
// the same.
func TestTheAllReduceNormFusionMakesStepsUpToItsBoundCheaper(t *testing.T) {
	in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
	editEngine(t, &in, func(e *engineconfig.Engine) { e.MaxNumSeqs = 512 })
	fused := build(t, in)
	graph := openWithoutLaunches(t, in)
	saved := func(n int) time.Duration {
		return graph.StepTime(launchDecodes(n)).NoOverlap - fused.StepTime(launchDecodes(n)).NoOverlap
	}
	near := func(a, b time.Duration) bool { return a-b < 3*time.Nanosecond && b-a < 3*time.Nanosecond }
	at64, at80 := saved(64), saved(80)
	if at64 <= 0 || !near(at64, at80) {
		t.Errorf("64 decodes save %v and 80 save %v: want the same, positive amount", at64, at80)
	}
	for _, n := range []int{88, 128} {
		if got := saved(n); got != 0 {
			t.Errorf("%d decodes save %v: past the 85-token bound the graph's launches stand", n, got)
		}
	}
}
