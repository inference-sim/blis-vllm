// Original to blis-vllm.

package lowering

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	blisschemas "github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"

	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
	vllmrules "github.com/inference-sim/blis-vllm/rules"
)

// lowerFor lowers a catalog model on a catalog chip at the layout, under the release's own
// rules, on as many nodes as the layout's GPUs need.
func lowerFor(t *testing.T, release, mdl, chipName, parallel string, gpus int, s vllmsettings.Settings) *plan.Lowered {
	t.Helper()
	chip, err := data.Chip(chipName)
	if err != nil {
		t.Fatal(err)
	}
	perNode := chip.GPUsPerNode
	extra := ""
	if perNode == 0 {
		perNode = 8
		extra = "  gpus_per_node: 8\n"
	}
	nodes := (gpus + perNode - 1) / perNode
	body := fmt.Sprintf("kind: Scenario\nname: l\nmodel: %s\ncoefficients: [cost-model-primitives]\n"+
		"engine_version: %q\ncluster:\n  hardware: %s\n  fabric: ib-400g\n  nodes: %d\n%s---\n"+
		"kind: Deployment\nname: d\npools:\n  - role: colocated\n    nodes: %d\n    parallel: {%s}\n",
		mdl, release, chipName, nodes, extra, nodes, parallel)
	path := filepath.Join(t.TempDir(), "s.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	sc, dep, err := blisschemas.LoadBundle(path)
	if err != nil {
		t.Fatal(err)
	}
	g, err := data.Model(mdl)
	if err != nil {
		t.Fatal(err)
	}
	l, err := Lower(Input{Scenario: sc, Deployment: dep, Model: g, Chip: chip,
		Rules: vllmrules.Lookup(release), Settings: s})
	if err != nil {
		t.Fatalf("%s on %s (%s): %v", mdl, chipName, parallel, err)
	}
	return l
}

const museNVFP4 = "muse-spark-open-816b-a42b-nvfp4"

// SUB-6: the lowering states how Muse Spark's three fused kernels change each layer's launches
// at v0.33.0 (inference-sim/blis-vllm#161). GB200 at TP8 with expert parallelism spans two
// 4-GPU trays, so vLLM shards the unpooler from 32 tokens and fuses the finalize all-reduce up
// to 128 (_MULTI_NODE_SHARDED_UNPOOLER_MIN_TOKENS, _MULTI_NODE_FUSED_ALL_REDUCE_MAX_TOKENS,
// model.py:154, muse_moe_runner.py:152). Per layer, up to 128 tokens: the QK-norm/RoPE/KV insert
// takes the cache write (-1), the finalize takes the post-expert norm (-1) and the combine
// all-reduce (-1), and the gather folds into the boundary (0): -3. From 129 tokens the
// all-reduce is its own launch again: -2, at every larger step.
func TestMuseSparkOnGB200StatesWhatItsFusedKernelsRemove(t *testing.T) {
	l := lowerFor(t, "0.33.0", museNVFP4, "gb200-nvl72", "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true", 8, vllmsettings.Settings{})
	for _, c := range []struct{ tokens, perLayer int }{
		{1, -3}, {31, -3}, {32, -3}, {128, -3}, {129, -2}, {256, -2}, {257, -2}, {100000, -2},
	} {
		if got := l.Plan.LaunchChangeAt(units.Tokens(c.tokens)); got != c.perLayer {
			t.Errorf("%d tokens: %d launches a layer beyond the graph's, want %d", c.tokens, got, c.perLayer)
		}
	}
	checkLaunchRanges(t, l.Plan.Launches)
}

// checkLaunchRanges: what the plan requires of the ranges a lowering states.
func checkLaunchRanges(t *testing.T, ls []plan.LaunchChange) {
	t.Helper()
	for i, c := range ls {
		if c.PerLayer == 0 || c.FromTokens < 1 || c.ToTokens < c.FromTokens {
			t.Errorf("range %d %+v is empty, starts below 1 or changes nothing", i, c)
		}
		if i > 0 && c.FromTokens <= ls[i-1].ToTokens {
			t.Errorf("range %d %+v overlaps %+v", i, c, ls[i-1])
		}
	}
	if len(ls) > 0 && ls[len(ls)-1].ToTokens < math.MaxInt32 {
		t.Errorf("the last range ends at %d tokens; a step larger than it would launch the graph's count", ls[len(ls)-1].ToTokens)
	}
}

// The same TP8+EP layout states a more negative change on GB200 at PP1 than at PP2, and than on
// SM90. At PP2 vLLM does not fuse across layers (fuse_across_layers, model.py:1533), so the
// sharded unpooler's gather runs as gather_columns (an all-gather and a copy): +2 a layer from
// 32 to 256 tokens, which nets the change to -1 and then 0. An H200 has no tep_default
// (model.py:505-517): no sharded unpooler and no finalize fusion, so only the KV insert and the
// post-expert norm go: -2 at every step size.
func TestMuseSparkFusesMoreOnGB200AtPP1ThanAtPP2OrOnHopper(t *testing.T) {
	const ep = "tp: 8, pp: %d, dp: 1, enable_expert_parallel: true"
	pp1 := lowerFor(t, "0.33.0", museNVFP4, "gb200-nvl72", fmt.Sprintf(ep, 1), 8, vllmsettings.Settings{})
	pp2 := lowerFor(t, "0.33.0", museNVFP4, "gb200-nvl72", fmt.Sprintf(ep, 2), 16, vllmsettings.Settings{})
	h200 := lowerFor(t, "0.33.0", museNVFP4, "h200", fmt.Sprintf(ep, 1), 8, vllmsettings.Settings{})
	for _, c := range []struct {
		tokens         int
		pp1, pp2, h200 int
	}{
		{1, -3, -3, -2},
		{64, -3, -1, -2},
		{128, -3, -1, -2},
		{200, -2, 0, -2},
		{300, -2, -2, -2},
	} {
		at := units.Tokens(c.tokens)
		if got := pp1.Plan.LaunchChangeAt(at); got != c.pp1 {
			t.Errorf("GB200 PP1, %d tokens: %d, want %d", c.tokens, got, c.pp1)
		}
		if got := pp2.Plan.LaunchChangeAt(at); got != c.pp2 {
			t.Errorf("GB200 PP2, %d tokens: %d, want %d", c.tokens, got, c.pp2)
		}
		if got := h200.Plan.LaunchChangeAt(at); got != c.h200 {
			t.Errorf("H200 PP1, %d tokens: %d, want %d", c.tokens, got, c.h200)
		}
	}
	for _, l := range []*plan.Lowered{pp1, pp2, h200} {
		checkLaunchRanges(t, l.Plan.Launches)
	}
}

// An NVFP4 KV cache keeps QK-norm, RoPE and the cache write apart (SUPPORTED_KV_CACHE_DTYPES,
// fused_qk_norm_rope_kv.py:17-20), so the cache write is not removed: one launch a layer fewer
// than the stated NVFP4-free layout.
func TestMuseSparkWithAnNVFP4CacheKeepsItsCacheWrite(t *testing.T) {
	const parallel = "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true"
	fused := lowerFor(t, "0.33.0", museNVFP4, "gb200-nvl72", parallel, 8, vllmsettings.Settings{})
	unfused := lowerFor(t, "0.33.0", museNVFP4, "gb200-nvl72", parallel, 8, vllmsettings.Settings{CacheDType: "nvfp4"})
	for _, tokens := range []int{1, 64, 200} {
		at := units.Tokens(tokens)
		if got, want := unfused.Plan.LaunchChangeAt(at), fused.Plan.LaunchChangeAt(at)+1; got != want {
			t.Errorf("%d tokens: %d with an NVFP4 cache, want %d (the cache write is not fused)", tokens, got, want)
		}
	}
}

// A model that is not Muse Spark, and a release that is not v0.33.0, state no change: the graph's
// launch count stands.
func TestOnlyMuseSparkAtV033StatesFusedLaunches(t *testing.T) {
	const parallel = "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true"
	for _, c := range []struct {
		name, release, model string
	}{
		{"another MoE model", "0.33.0", "qwen3-30b-a3b"},
		{"a dense model", "0.33.0", "llama-3.1-70b-instruct"},
		{"Muse Spark at an earlier release", "0.32.0", museNVFP4},
	} {
		l := lowerFor(t, c.release, c.model, "gb200-nvl72", parallel, 8, vllmsettings.Settings{})
		if len(l.Plan.Launches) != 0 {
			t.Errorf("%s: states %+v, want none", c.name, l.Plan.Launches)
		}
	}
}
