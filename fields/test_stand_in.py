# Original to blis-vllm.
"""Hold fields/stand_ins.json to the upstream's own resolution (design 11.2: a stand-in is exact only where
the upstream resolves the stated value and the unset field to the same behaviour).

Run: python3 -I fields/test_stand_in.py           (with VLLM_SOURCE=<checkout of vLLM v0.32.0>)

The upstream functions are compiled from the pinned source file and run with their globals replaced by
stubs (the node count, the device capability, the environment); nothing here restates what they decide.
"""
import ast
import itertools
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stand_in  # noqa: E402

SOURCE = os.environ.get("VLLM_SOURCE")


def function_from(path, name, env):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    mod = ast.Module(body=[node], type_ignores=[])
    ns = dict(env)
    exec(compile(ast.fix_missing_locations(mod), path, "exec"), ns)
    return ns[name]


class Log:
    def debug_once(self, *a, **k): pass
    info_once = warning_once = error_once = debug_once


@unittest.skipUnless(SOURCE, "VLLM_SOURCE names a vLLM v0.32.0 checkout")
class FlashInferBackend(unittest.TestCase):
    PATH = os.path.join(SOURCE or "", "vllm/distributed/device_communicators/flashinfer_all_reduce.py")

    def resolve(self, value, nodes, cap):
        envs = types.SimpleNamespace(VLLM_FLASHINFER_ALLREDUCE_BACKEND=value)
        plat = types.SimpleNamespace(has_device_capability=lambda c: stand_in._cap(cap)[0] * 10 >= c)
        f = function_from(self.PATH, "_resolve_fi_ar_backend",
                          {"envs": envs, "_node_count": lambda g: nodes, "current_platform": plat, "logger": Log()})
        return f(object())

    def test_a_stated_value_is_exact_exactly_where_it_picks_the_backend_auto_picks(self):
        for value, nodes, cap in itertools.product(["auto", "mnnvl", "trtllm", "nccl"], [1, 2, 4], ["8.0", "9.0", "10.0", "10.3"]):
            try:
                stated = self.resolve(value, nodes, cap)[0]
            except Exception:
                stated = "error"
            auto = self.resolve("auto", nodes, cap)[0]
            want = stated == auto
            got = stand_in.exact("VLLM_FLASHINFER_ALLREDUCE_BACKEND", value, tp_group_nodes=nodes, compute_capability=cap)
            self.assertEqual(got, want, f"value {value} on {nodes} node(s), cc {cap}: upstream auto={auto} stated={stated}")


@unittest.skipUnless(SOURCE, "VLLM_SOURCE names a vLLM v0.32.0 checkout")
class RCacheLimit(unittest.TestCase):
    PATH = os.path.join(SOURCE or "", "vllm/distributed/nixl_utils.py")

    def test_a_stated_value_is_exact_exactly_where_it_is_the_one_vllm_sets(self):
        for value in ["1024", "0", "inf", "2048"]:
            env = {}
            f = function_from(self.PATH, "_maybe_set_ucx_rcache_limit",
                              {"os": types.SimpleNamespace(environ=env), "sys": types.SimpleNamespace(modules={}), "logger": Log()})
            f()
            self.assertEqual(stand_in.exact("UCX_RCACHE_MAX_UNRELEASED", value), value == env["UCX_RCACHE_MAX_UNRELEASED"], value)


class Grammar(unittest.TestCase):
    def test_a_fact_the_caller_did_not_give_makes_the_clause_false(self):
        self.assertFalse(stand_in.exact("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl"))
        self.assertTrue(stand_in.exact("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "auto"))

    def test_transport_lists(self):
        for v, want in [("cuda_copy,rc", True), ("cuda_copy,cuda_ipc,tcp", False), ("^rc", False), ("all", True), ("tcp", False)]:
            self.assertEqual(stand_in.exact("UCX_TLS", v), want, v)

    def test_an_unlisted_field_has_no_exact_stand_in(self):
        self.assertFalse(stand_in.exact("SOME_OTHER_VARIABLE", "1"))


if __name__ == "__main__":
    unittest.main()
