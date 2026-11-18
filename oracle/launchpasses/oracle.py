# Original to blis-vllm.
"""The compile passes vLLM v0.33.0 runs by default, and which of them change a decode step's
launches (inference-sim/blis-vllm#161, #191).

vLLM resolves a deployment's pass_config, compilation mode, CUDA-graph mode and the token ranges the
all-reduce fusion applies to in VllmConfig.__post_init__ (config/vllm.py:_apply_optimization_level_
defaults, :_set_compile_ranges, and the per-model hooks of model_executor/models/config.py). This
oracle builds a real VllmConfig on a CPU for each case of CASES and prints what it resolved. The
GPU is replaced: the platform is vLLM's CudaPlatform with the case's compute capability; the
compiled-op namespaces are stand-ins; FlashInfer is taken as installed (requirements/cuda.txt pins
it); the number of devices is large enough for any case's world size; and the number of nodes the
tensor-parallel group spans is the case's. Every decision is vLLM's own code.

Printed per case:
  mode, cudagraph_mode          the resolved CompilationMode and CUDAGraphMode (names)
  passes                        every pass_config flag a CUDA platform reads
  compile_endpoints             compilation_config.compile_ranges_endpoints (:_set_compile_ranges)
  all_reduce_fusion_tokens      the largest compile range end the all-reduce + RMSNorm pass applies
                                to, from AllReduceFusionPass.__init__ (compilation/passes/fusion/
                                allreduce_rms_fusion.py:984-1027, the lines INLINE below) and
                                _resolve_fi_ar_backend; 0 when the pass is not built
and per model, backbone_compiled: the backbone class carries @support_torch_compile and not
@ignore_torch_compile (compilation/decorators.py), imported from the release.

Usage:
    python3 -I oracle/launchpasses/oracle.py <vllm checkout> json            print the cases
    python3 -I oracle/launchpasses/oracle.py <vllm checkout> check <file>    fail unless <file> equals what it prints
The models' config.json is read from $BLIS_ORACLE_CATALOG/<model>/config.json, or, for a model whose
licence does not let the catalog redistribute it, fetched and checked as the block-size oracle does.
"""
import itertools
import json
import os
import shutil
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "blocksize"))

COMMIT = "4ba787ec9959e566d3b587a796dff07673324da5"
CATALOG = os.environ.get("BLIS_ORACLE_CATALOG", "")

# model -> (backbone module, backbone class) the architecture's text decoder is built with.
MODELS = {
    "llama-3.1-70b-instruct": ("vllm.model_executor.models.llama", "LlamaModel"),
    "gpt-oss-120b": ("vllm.model_executor.models.gpt_oss", "GptOssModel"),
    "minimax-m2.5": ("vllm.model_executor.models.minimax_m2", "MiniMaxM2Model"),
    "minimax-m2.5-nvfp4": ("vllm.model_executor.models.minimax_m2", "MiniMaxM2Model"),
    "kimi-k2.5-nvfp4": ("vllm.model_executor.models.deepseek_v2", "DeepseekV2Model"),
    "deepseek-v4-pro": ("vllm.models.deepseek_v4.model", "DeepseekV4Model"),
    "qwen3-30b-a3b": ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeModel"),
}
CAPS = ((9, 0), (10, 0), (10, 3))
TPS = (1, 2, 4, 8, 16)
BATCHED = (2048, 8192)
REPLACEMENTS = []

PASS_FLAGS = ("fuse_norm_quant", "fuse_act_quant", "fuse_attn_quant", "fuse_allreduce_rms",
              "enable_qk_norm_rope_fusion", "fuse_rope_kvcache_cat_mla")


def cases():
    for model, cap, tp, eager, batched, spans in itertools.product(
            MODELS, CAPS, TPS, (False, True), BATCHED, (False, True)):
        if spans and tp == 1:
            continue  # a group of one rank spans nothing
        yield dict(model=model, cap=cap[0], minor=cap[1], tp=tp, enforce_eager=eager, max_num_batched_tokens=batched,
                   spans_nodes=spans)


def load(vllm_dir):
    warnings.simplefilter("ignore")
    import oracle as blocksize  # oracle/blocksize/oracle.py

    v = blocksize.load_vllm(vllm_dir)
    REPLACEMENTS.extend(v.stubs.REPLACEMENTS)
    from unittest import mock

    import torch

    for ns in ("_C", "_C_cache_ops", "_moe_C", "_C_custom_ar", "_rocm_C"):
        setattr(torch.ops, ns, mock.MagicMock(name="torch.ops." + ns))
    REPLACEMENTS.append(("torch.ops._C, _C_cache_ops, _moe_C, _C_custom_ar, _rocm_C",
                         "compiled ops that the pass modules look up when imported; none feeds a decision"))
    v.Sim.device_count = classmethod(lambda cls: 64)
    REPLACEMENTS.append(("Sim.device_count", "64, so ParallelConfig's check that the world size fits the "
                         "node's GPUs (config/parallel.py:1066-1075) passes for any case"))
    import vllm.utils.flashinfer as fi

    fi.has_flashinfer = lambda: True
    REPLACEMENTS.append(("vllm.utils.flashinfer.has_flashinfer",
                         "true: FlashInfer is a CUDA requirement (requirements/cuda.txt)"))
    return v


def model_dir(model):
    import modelconfig

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "wb") as f:
        f.write(modelconfig.config_bytes(model, os.path.join(CATALOG, model)))
    return d


def backbone_compiled(model):
    import importlib

    from vllm.compilation.decorators import _should_ignore_torch_compile
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    mod, cls = MODELS[model]
    try:
        c = getattr(importlib.import_module(mod), cls)
    except (ImportError, AttributeError):
        return None  # the class cannot be imported on a CPU: not decided
    return TorchCompileWithNoGuardsWrapper in c.__bases__ and not _should_ignore_torch_compile(c)


def resolve(v, d, directories):
    import vllm.distributed.device_communicators.flashinfer_all_reduce as far
    from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig

    v.Sim.cap = (d["cap"], d["minor"])
    # VllmConfig sets this variable for a breakable-graph architecture and keeps it for the process
    # (config/vllm.py:_maybe_enable_breakable_cudagraph); each case is an engine of its own.
    os.environ.pop("VLLM_USE_BREAKABLE_CUDAGRAPH", None)
    model = d["model"]
    if model not in directories:
        directories[model] = model_dir(model)
    mc = ModelConfig(model=directories[model], skip_tokenizer_init=True, trust_remote_code=False,
                     enforce_eager=d["enforce_eager"])
    sc = SchedulerConfig(max_num_batched_tokens=d["max_num_batched_tokens"], max_model_len=d["max_num_batched_tokens"],
                         max_num_seqs=256, is_encoder_decoder=False)
    vc = VllmConfig(model_config=mc, parallel_config=ParallelConfig(tensor_parallel_size=d["tp"]), scheduler_config=sc)
    cc = vc.compilation_config
    pc = cc.pass_config
    row = dict(case=d, mode=cc.mode.name, cudagraph_mode=cc.cudagraph_mode.name,
               passes={k: bool(getattr(pc, k)) for k in PASS_FLAGS},
               compile_endpoints=list(cc.compile_ranges_endpoints or []), all_reduce_fusion_tokens=0,
               dtype=str(mc.dtype).replace("torch.", ""))
    # AllReduceFusionPass.__init__ (allreduce_rms_fusion.py:984-1027), up to the workspace.
    # INLINE: the workspace itself is FlashInfer's and is assumed creatable, except where
    # _resolve_fi_ar_backend returns no backend for the group.
    if pc.fuse_allreduce_rms and d["tp"] > 1 and cc.mode.name == "VLLM_COMPILE":
        far._node_count = lambda group: 2 if d["spans_nodes"] else 1
        backend, _ = far._resolve_fi_ar_backend(None)
        size = pc.flashinfer_max_size(d["tp"])
        if backend is not None and size is not None:
            hidden = mc.get_hidden_size()
            row["all_reduce_fusion_tokens"] = min(size // (hidden * mc.dtype.itemsize), d["max_num_batched_tokens"])
    return row


def run(vllm_dir):
    v = load(vllm_dir)
    directories = {}
    try:
        rows = []
        compiled = {}
        for d in cases():
            rows.append(resolve(v, d, directories))
        for m in MODELS:
            compiled[m] = backbone_compiled(m)
    finally:
        for p in directories.values():
            shutil.rmtree(p, ignore_errors=True)
    return dict(commit=COMMIT, backbone_compiled=compiled,
                replacements=[dict(what=w, why=y) for w, y in REPLACEMENTS], cases=rows)


def main(argv):
    if len(argv) < 3 or argv[2] not in ("json", "check"):
        print(__doc__, file=sys.stderr)
        return 2
    out = json.dumps(run(argv[1]), indent=1, sort_keys=True) + "\n"
    if argv[2] == "json":
        sys.stdout.write(out)
        return 0
    with open(argv[3]) as f:
        if f.read() != out:
            print(f"{argv[3]} differs from what vLLM {COMMIT} decides; regenerate it", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
