// Original to blis-vllm.

package lowering

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	blisschemas "github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"

	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
	"github.com/inference-sim/blis-vllm/internal/upstream"
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

// Before v0.33.0 the lowering states no change in launches: Muse Spark's fused kernels and the
// compile passes are resolved for that release only.
func TestNoFusedLaunchesAreStatedBeforeV033(t *testing.T) {
	const parallel = "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true"
	for _, model := range []string{"qwen3-30b-a3b", "llama-3.1-70b-instruct", museNVFP4} {
		l := lowerFor(t, "0.32.0", model, "gb200-nvl72", parallel, 8, vllmsettings.Settings{})
		if len(l.Plan.Launches) != 0 {
			t.Errorf("%s at 0.32.0: states %+v, want none", model, l.Plan.Launches)
		}
	}
}

// passCases is vLLM v0.33.0's own resolution of its default compile passes for each case
// (oracle/launchpasses/oracle.py, run on a CPU over vLLM's VllmConfig).
type passCases struct {
	Commit           string           `json:"commit"`
	BackboneCompiled map[string]*bool `json:"backbone_compiled"`
	Cases            []struct {
		Case struct {
			Model      string `json:"model"`
			Cap        int    `json:"cap"`
			Minor      int    `json:"minor"`
			TP         int    `json:"tp"`
			Eager      bool   `json:"enforce_eager"`
			Batched    int    `json:"max_num_batched_tokens"`
			SpansNodes bool   `json:"spans_nodes"`
		} `json:"case"`
		Mode         string          `json:"mode"`
		Passes       map[string]bool `json:"passes"`
		FusionTokens int             `json:"all_reduce_fusion_tokens"`
		DType        string          `json:"dtype"`
	} `json:"cases"`
}

func loadPassCases(t *testing.T) passCases {
	t.Helper()
	b, err := os.ReadFile("testdata/launch_passes.json")
	if err != nil {
		t.Fatal(err)
	}
	var pc passCases
	if err := json.Unmarshal(b, &pc); err != nil {
		t.Fatal(err)
	}
	if pc.Commit != upstream.Commit || len(pc.Cases) == 0 {
		t.Fatalf("launch_passes.json is for %q with %d cases; the port models %s", pc.Commit, len(pc.Cases), upstream.Commit)
	}
	return pc
}

// The chip that stands for each compute capability the oracle resolves.
var chipOfCapability = map[[2]int]string{{9, 0}: "h200", {10, 0}: "b200", {10, 3}: "b300"}

// SUB-6, FID-1: the all-reduce and RMSNorm fusion the lowering states is vLLM v0.33.0's own
// decision. For every case of the oracle (a real VllmConfig on a CPU: mode, pass_config, compile
// ranges, FlashInfer's size table and backend choice), the lowering states a fused range exactly
// where vLLM builds the pass, up to the token bound vLLM computes, and states none where vLLM
// compiles nothing or builds no pass. The cases that need a tensor-parallel group on more hosts
// than the packaging's 8 GPUs a node give (tp 16) are the multi-host ones.
func TestTheCompilePassResolutionsAreVLLMs(t *testing.T) {
	pc := loadPassCases(t)
	for _, name := range []string{"llama-3.1-70b-instruct", "gpt-oss-120b", "minimax-m2.5", "kimi-k2.5-nvfp4", "qwen3-30b-a3b"} {
		g, err := data.Model(name)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := compiledBackbones[g.Global.Architecture], pc.BackboneCompiled[name]; want == nil || got != *want {
			t.Errorf("%s (%s): compiled backbone %v in the lowering, vLLM %v", name, g.Global.Architecture, got, want)
		}
	}
	ran := 0
	for _, c := range pc.Cases {
		in := c.Case
		if in.SpansNodes != (in.TP > 8) {
			continue // not a layout the 8-GPU packaging of the test chips can state
		}
		chip, ok := chipOfCapability[[2]int{in.Cap, in.Minor}]
		if !ok || c.DType != "bfloat16" {
			t.Fatalf("case %+v resolves dtype %s on capability %d.%d; the lowering assumes bf16 activations on a chip it has", in, c.DType, in.Cap, in.Minor)
		}
		g, _ := data.Model(in.Model)
		s := vllmsettings.Settings{MaxNumBatchedTokens: in.Batched, EnforceEager: in.Eager}
		l := lowerFor(t, "0.33.0", in.Model, chip, fmt.Sprintf("tp: %d, pp: 1, dp: 1", in.TP), in.TP, s)
		_ = g
		if c.FusionTokens == 0 || c.Mode != "VLLM_COMPILE" {
			if len(l.Plan.Launches) != 0 {
				t.Errorf("%+v: vLLM builds no all-reduce fusion (mode %s, %d tokens); the lowering states %+v", in, c.Mode, c.FusionTokens, l.Plan.Launches)
			}
			continue
		}
		ran++
		if len(l.Plan.Launches) != 1 || l.Plan.Launches[0].FromTokens != 1 || int(l.Plan.Launches[0].ToTokens) != c.FusionTokens {
			t.Errorf("%+v: vLLM fuses up to %d tokens; the lowering states %+v", in, c.FusionTokens, l.Plan.Launches)
		}
		checkLaunchRanges(t, nil)
	}
	if ran < 30 {
		t.Fatalf("only %d cases built the pass; the grid lost its cases", ran)
	}
}

// The launches vLLM v0.33.0's FlashInfer all-reduce and RMSNorm fusion removes, by deployment.
// Per layer it replaces the all-reduce and the norm after it with one launch, for the layer's
// attention output and its MLP or MoE output (which feeds the next layer's input norm): -2. The
// token bound is FlashInfer's size limit over the hidden width, capped at max_num_batched_tokens
// (allreduce_rms_fusion.py:104-127, :1022-1027), so it shrinks with the width and grows on
// Blackwell. Dense and MoE stacks alike; a width, part or mode with no fused kernel states none.
func TestTheLoweringStatesTheAllReduceNormFusion(t *testing.T) {
	const mnbt = 8192
	for _, c := range []struct {
		name, model, chip string
		tp                int
		eager             bool
		backend           string
		tokens, perLayer  int
	}{
		{"dense on Hopper, 2 MiB over 8192 x 2 bytes", "llama-3.1-70b-instruct", "h200", 4, false, "", 128, -2},
		{"dense on Hopper at tp 8: 0.5 MiB", "llama-3.1-70b-instruct", "h200", 8, false, "", 32, -2},
		{"dense on B200 at tp 8: 1 MiB", "llama-3.1-70b-instruct", "b200", 8, false, "", 64, -2},
		{"dense on B300 at tp 4: 64 MiB", "llama-3.1-70b-instruct", "b300", 4, false, "", 4096, -2},
		{"gpt-oss on B200 at tp 2: 64 MiB over 2880 x 2 bytes, capped at the batch", "gpt-oss-120b", "b200", 2, false, "", mnbt, -2},
		{"MiniMax-M2.5 (MoE) on H200 at tp 8", "minimax-m2.5", "h200", 8, false, "", 85, -2},
		{"gpt-oss on B200 at tp 8", "gpt-oss-120b", "b200", 8, false, "", 182, -2},
		{"Kimi-K2.5 (MLA, a dense layer and MoE layers) on B300 at tp 8: 4 MiB", "kimi-k2.5-nvfp4", "b300", 8, false, "", 292, -2},
		{"Kimi-K2.5 on B200 at tp 4", "kimi-k2.5-nvfp4", "b200", 4, false, "", 2340, -2},
		{"tp 1 has no all-reduce", "llama-3.1-70b-instruct", "h200", 1, false, "", 0, 0},
		{"enforce_eager compiles nothing", "llama-3.1-70b-instruct", "h200", 4, true, "", 0, 0},
		{"DeepSeek-V4 runs breakable graphs, not compiled", "deepseek-v4-pro", "b200", 8, false, "", 0, 0},
		{"a MoE-EP backend that reduces in its kernel: the MoE layers have no graph all-reduce", "kimi-k2.5-nvfp4", "b300", 8, false, "flashinfer_moe_ep_cutedsl", 0, 0},
	} {
		l := lowerFor(t, "0.33.0", c.model, c.chip, fmt.Sprintf("tp: %d, pp: 1, dp: 1", c.tp), c.tp,
			vllmsettings.Settings{MaxNumBatchedTokens: mnbt, EnforceEager: c.eager, MoEBackend: c.backend})
		if c.perLayer == 0 {
			if len(l.Plan.Launches) != 0 {
				t.Errorf("%s: states %+v, want none", c.name, l.Plan.Launches)
			}
			continue
		}
		want := []plan.LaunchChange{{FromTokens: 1, ToTokens: units.Tokens(c.tokens), PerLayer: c.perLayer}}
		if len(l.Plan.Launches) != 1 || l.Plan.Launches[0] != want[0] {
			t.Errorf("%s: states %+v, want %+v", c.name, l.Plan.Launches, want)
		}
	}
}

// A tensor-parallel group on two hosts has FlashInfer's fused kernel only on Blackwell (its size
// table has no width 16 for Hopper, and _resolve_fi_ar_backend, flashinfer_all_reduce.py:152-177,
// finds no backend for a multi-host group below capability 10): two 8-GPU H200 hosts at tp 16 fuse
// nothing, and the same width on B200 fuses up to the 64 MiB the table gives.
func TestAMultiHostGroupFusesAllReduceOnBlackwellOnly(t *testing.T) {
	const parallel = "tp: 16, pp: 1, dp: 1"
	hopper := lowerFor(t, "0.33.0", "llama-3.1-70b-instruct", "h200", parallel, 16, vllmsettings.Settings{MaxNumBatchedTokens: 8192})
	if len(hopper.Plan.Launches) != 0 {
		t.Errorf("H200 tp 16 across two hosts: %+v, want none", hopper.Plan.Launches)
	}
	blackwell := lowerFor(t, "0.33.0", "llama-3.1-70b-instruct", "b200", parallel, 16, vllmsettings.Settings{MaxNumBatchedTokens: 8192})
	if got := blackwell.Plan.LaunchChangeAt(units.Tokens(4096)); got != -2 {
		t.Errorf("B200 tp 16: %d launches a layer at 4096 tokens, want -2", got)
	}
}
