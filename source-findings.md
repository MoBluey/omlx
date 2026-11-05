# Combined marginal candidate source - 6 November 2026

Base: installed r3 source `8e8425ad1b7d80c25e2880018d3d73f9df400685`.
Only selector `9e1806d2` and pair-gather `87634965` are integrated. Their
kernel source is unchanged; installed masked attention and idle code remain
unchanged. No speed claim, installation or merge follows from these checks.

Both new switches default off. Change them only at an idle request boundary
in `mlx_vlm.models.qwen4_exp.qsa_fast`:

| Switch | Successful dispatch counter | Failure state |
| --- | --- | --- |
| `_DECODE_SELECT_REDUCED_BARRIERS_ENABLED` | `_DECODE_SELECT_REDUCED_BARRIERS_STATS['mask'/'tokens']` | stats `failures`; reduced signatures in `_DECODE_SELECT_FAILED` |
| `_FUSED_DECODE_KV_GATHER` | `_FUSED_DECODE_KV_GATHER_CALLS` | `_FUSED_DECODE_KV_GATHER_DISABLED` |

Delivery-source preparation adds one startup opt-in:
`OMLX_EXPERIMENTAL_QWEN_COMBINED_MARGINAL=1` initialises both switches to true
and `context_copy._MAX_MATCH` to 32. Unset and every other value keep false,
false and 64. GDN wider projections remain the separate existing model
setting `qwen4_gdn_decode_wide_proj`; managed activation belongs to main.
The r3 masked/idle controls and all candidate arithmetic remain unchanged.
This preparation does not imply final confirmation or installation approval.

CPU-only configuration coverage executes the actual three assignment ASTs
for unset, empty, `0`, `true`, `1` and space-prefixed `1`, without importing
MLX/native libraries. All 18 control/value comparisons passed. Command:
`python3.12 -BIS tests/test_qwen_combined_process_config.py` (absolute paths
and the pinned Python 3.12.15 executable used below).

Counters increment only when the candidate operator returns. First signatures
are evaluated before being marked valid/proven. Later results remain lazy,
as in the original candidates. Gather fallback and unsupported geometry do
not increment its counter. `_FUSED_DECODE_KV_GATHER_PROVEN` remains available.

Real M4 result: **94 passed in 8.62 seconds**, process exit 0. Selector 20,
gather 24, combined 6, existing masked 41, existing wide-GDN arithmetic 1,
existing Qwen4 context-copy 2. Combined tests compare exact output/cache bits,
accepted-prefix trim, raw prefix restore and continuation at 9K/64K/79K with
masked attention enabled in both arms. Rank-two positions engage gather;
rank-three positions engage masked attention. Candidate success counters and
clear failure states are asserted. Context-copy tests use `_MAX_MATCH=32`;
GDN/context-copy implementations are unchanged. These are focused component
and attention integration checks; main owns full-model combined engagement
and performance qualification.

No native-library setup was required for these focused tests. Source-cli
serving requires all nine ignored native files from the r3 sealed site:
four `_ext.cpython-312-darwin.so` extensions and five `.metallib` resources
under `omlx/custom_kernels/{lowbit,quant_matmul,attention,sparse_index}`.
`quant_matmul` includes both its regular and `_nax` libraries. Main copied
all nine byte-for-byte and recorded their hashes in `../native-closure.json`;
the run preflight verifies every hash before model startup.

Import availability alone is insufficient: `source-cli --check` imported
the four extensions, but the first full-model prefill then failed because
attention/quant_matmul `.metallib` resources were absent. Future source
worktrees must verify the full native resource closure before serving.
Do not recompile or change the pinned closure.

Exact executed command (Python isolated mode; no bytecode/cache writes):

```sh
/opt/homebrew/Cellar/python@3.12/3.12.15/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -BIS - <<'PY'
import sys
repo_path = '/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/benchmarks/qwen-combined-gains-20261106/source'
sys.path[:0] = [repo_path, repo_path + '/tests', '/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/runtime/omlx-0.7.3-m4-max-r3-sealed-site', '/tmp/qwen-idle-tests-20261106']
import pytest
class Copy32:
    def pytest_runtest_setup(self, item):
        if item.name.startswith('test_context_copy_'):
            from omlx.patches.mlx_lm_mtp import context_copy
            context_copy._MAX_MATCH = 32
    def pytest_runtest_teardown(self, item):
        if item.name.startswith('test_context_copy_'):
            from omlx.patches.mlx_lm_mtp import context_copy
            context_copy._MAX_MATCH = 64
nodes = ['test_qwen4_qsa_long_selector_trial.py', 'test_qwen4_decode_kv_pair.py', 'test_qwen4_combined_marginal_trials.py', 'test_qwen4_masked_s_trial.py', 'test_qwen35_gdn_prework.py::test_qwen4_decode_wide_projections_are_bit_exact_either_way', 'test_mlx_lm_mtp_patch.py::test_context_copy_drafts_keep_greedy_output[qwen4]', 'test_mlx_lm_mtp_patch.py::test_context_copy_width_follows_the_targets_verify_window[qwen4-15]']
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider'] + [repo_path + '/tests/' + node for node in nodes], plugins=[Copy32()]))
PY
```
