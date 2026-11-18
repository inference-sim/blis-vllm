// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/compilation/passes/fusion/allreduce_rms_fusion.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5):
// FI_ALLREDUCE_FUSION_MAX_SIZE_MB (:104-127) and AllReduceFusionPass.__init__ (:984-1027); with
// vllm/config/vllm.py (enable_allreduce_rms_fusion, :183-205; OPTIMIZATION_LEVEL_02, :297-317;
// _set_compile_ranges, :2658-2732) and vllm/config/compilation.py (PassConfig.flashinfer_max_size,
// :182-196).

package lowering

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/units"

	corelower "github.com/inference-sim/blis-vllm/core/lower"
	"github.com/inference-sim/blis-vllm/internal/upstream"
)

// compiledBackbones are the architectures whose backbone class vLLM v0.33.0 builds with
// @support_torch_compile and not @ignore_torch_compile, so that its default mode, VLLM_COMPILE,
// runs the compile passes over it. oracle/launchpasses/oracle.py imports each class from the
// release and reports it (backbone_compiled), and TestTheCompilePassResolutionsAreVLLMs compares
// this set with it. An architecture not here is not decided: it states no pass.
// The Kimi-K2.5 checkpoint serves its text decoder with DeepseekV2Model.
var compiledBackbones = map[string]bool{
	"LlamaForCausalLM":                true,
	"GptOssForCausalLM":               true,
	"MiniMaxM2ForCausalLM":            true,
	"KimiK25ForConditionalGeneration": true,
	"Qwen3MoeForCausalLM":             true,
}

// fiAllReduceFusionMaxMiB is FI_ALLREDUCE_FUSION_MAX_SIZE_MB: the largest tensor, in MiB, that
// FlashInfer's fused all-reduce and RMSNorm takes, by device capability (10 x major + minor) and
// tensor-parallel width (allreduce_rms_fusion.py:104-127). A width not listed has no fused kernel.
var fiAllReduceFusionMaxMiB = map[int]map[int]float64{
	90:  {2: 64, 4: 2, 8: 0.5},
	100: {2: 64, 4: 32, 8: 1, 16: 64},
	103: {2: 64, 4: 64, 8: 4, 16: 64},
	107: {2: 64, 4: 64, 8: 2},
}

// activationBytes is the width of the activations the fused all-reduce carries: every catalog
// model computes in bf16 or fp16 (the oracle reports the dtype of each case it resolves).
const activationBytes = 2

// emits answers a node's emit condition for the layout, as the cost model's resolver does
// (kernel/internal/resolve Layout.Emits); the second result is false for a condition it does not
// know, which a caller must not guess.
func (l layout) emits(c model.EmitCondition) (emitted, known bool) {
	switch c {
	case model.EmitAlways:
		return true, true
	case model.EmitTensorParallel:
		return l.TP > 1, true
	case model.EmitExpertParallel:
		return l.moeDispatches(), true
	case model.EmitTensorParallelUnlessSequenceParallelMoE:
		return l.TP > 1 && !l.SequenceParallelMoE, true
	case model.EmitSequenceParallel:
		return l.ModelSequenceParallel, true
	case model.EmitTensorParallelUnlessSequenceParallel:
		return l.TP > 1 && !l.ModelSequenceParallel, true
	case model.EmitTensorParallelWithoutExpertParallel:
		return l.TP > 1 && l.ExpertWidth <= 1, true
	case model.EmitExpertParallelWithoutDispatch:
		return l.ExpertWidth > 1 && l.TP > 1 && !l.moeDispatches(), true
	}
	return false, false
}

// allReduceNormFusion states how vLLM v0.33.0's all-reduce and RMSNorm fusion
// (compilation_config.pass_config.fuse_allreduce_rms) changes each layer's launches: where a
// layer's all-reduce feeds the RMSNorm that follows it, FlashInfer's fused kernel replaces the
// two launches with one, for steps up to a token bound.
//
// The pass runs when vLLM compiles the model (mode VLLM_COMPILE: not enforce_eager, not an
// architecture that runs breakable CUDA graphs, whose mode is NONE), tensor parallelism is above 1
// on Hopper (capability 9.0) or a Blackwell part (family 10), and FlashInfer has a kernel for the
// width and capability (fiAllReduceFusionMaxMiB; its Hopper rows stop at 8, and a group on more
// hosts has a backend only on Blackwell, flashinfer_all_reduce.py:_resolve_fi_ar_backend,
// :152-177, so no case of the table is lost to it). The fused kernel serves compile ranges that end at or below max_token_num =
// max_size / (hidden x bytes), no more than max_num_batched_tokens (AllReduceFusionPass,
// :1022-1027; is_applicable_for_range, :1117-1122); a larger step runs the graph the pass did not
// touch. The ranges are compiled by the padded step size, which is what Plan.Launches counts.
//
// It counts only pairs of nodes the catalog graph has: an emitted AllReduce node followed by an
// Elementwise norm, within a layer or from the layer's last node to the next layer's input_norm.
// A MoE layer's mlp_out all-reduce is a graph node only when the experts' combine does not reduce
// inside the kernel, which a stated FlashInfer MoE-EP backend does (output_is_reduced,
// fused_moe/flashinfer_moe_ep.py:366); that case, and a stack whose layer kinds fuse a different
// number of pairs (Plan.Launches states one change for every layer), state nothing and say so.
// The embedding's all-reduce before layer 0 is not a layer's launch and is not counted.
func allReduceNormFusion(g *model.Graph, c *lowerContext, lay layout, in Input, enforceEager bool,
	maxBatched int, release string) ([]plan.LaunchChange, error) {
	if release != upstream.Version || corelower.IsMuseSpark(g) {
		return nil, nil
	}
	arch := g.Global.Architecture
	if enforceEager || corelower.Breakable(arch) {
		return nil, nil // the model is not compiled: no pass runs
	}
	if !compiledBackbones[arch] {
		return nil, nil // not decided: no pass is stated for it (inference-sim/blis-vllm#191)
	}
	major, minor := capability(c.chip)
	capInt := 10*major + minor
	if lay.TP <= 1 || (capInt != 90 && major != 10) {
		return nil, nil
	}
	maxMiB, ok := fiAllReduceFusionMaxMiB[capInt][lay.TP]
	if !ok {
		return nil, nil
	}
	hidden := g.Global.HiddenSize
	if in.Draft != nil {
		hidden = max(hidden, in.Draft.Global.HiddenSize) // the workspace is sized for the widest model
	}
	if hidden <= 0 {
		return nil, nil
	}
	tokens := min(int(maxMiB*1024*1024)/(hidden*activationBytes), maxBatched)
	if tokens < 1 {
		return nil, nil
	}

	per, uniform, err := pairsPerLayer(g, lay, in.Settings.MoEBackend)
	if err != nil {
		return nil, err
	}
	if !uniform {
		return nil, nil // Plan.Launches states one change for every layer (inference-sim/blis-vllm#191)
	}
	if per == 0 {
		return nil, nil
	}
	return []plan.LaunchChange{{FromTokens: 1, ToTokens: units.Tokens(tokens), PerLayer: -per}}, nil
}

// pairsPerLayer is how many all-reduce and norm pairs one layer of the stack fuses, and whether
// every layer kind of the stack agrees.
func pairsPerLayer(g *model.Graph, lay layout, statedMoEBackend string) (int, bool, error) {
	counts := map[int]bool{}
	var only int
	for _, lk := range stackKinds(g.LayerKinds, g.Stack) {
		var nodes []model.Node
		moe := false
		for _, n := range lk.Nodes {
			emitted, known := lay.emits(n.Emit)
			if !known {
				return 0, false, fmt.Errorf("layer kind %q: node %s carries emit condition %q, which the "+
					"lowering cannot evaluate", lk.ID, n.Role, n.Emit)
			}
			if emitted {
				nodes = append(nodes, n)
			}
			moe = moe || n.Op == model.OpGroupedGEMM
		}
		inKernelReduce := moe && strings.HasPrefix(strings.ToLower(strings.ReplaceAll(statedMoEBackend, "-", "_")),
			"flashinfer_moe_ep")
		isNorm := func(n model.Node) bool { return n.Op == model.OpElementwise && strings.HasSuffix(n.Role, "norm") }
		n := 0
		for i, a := range nodes {
			if a.Op != model.OpAllReduce || (inKernelReduce && moe && a.Role == "mlp_out") {
				continue
			}
			switch {
			case i+1 < len(nodes):
				if isNorm(nodes[i+1]) {
					n++
				}
			case len(nodes) > 0 && nodes[0].Role == "input_norm" && isNorm(nodes[0]):
				n++ // the layer's last all-reduce feeds the next layer's input norm
			}
		}
		counts[n], only = true, n
	}
	return only, len(counts) == 1, nil
}
