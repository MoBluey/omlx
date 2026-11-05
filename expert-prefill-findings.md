# Qwen expert prefill profiling and tile archive

Both experimental tiles are archived. Neither is connected to serving. The Q4
pair's measured latency reduction was 1.53%, too small to justify a custom
serving kernel. No full-model TTFT improvement was measured or claimed.

## Runtime and scope

- Apple M4 Max, 128 GiB, `applegpu_g16s`; sealed source `0713dc8e`.
- MLX 0.32.4, existing weights, BF16 activations, affine Q4/Q8 group 64.
- Production thinking/sampling profiles stayed unchanged. The 28K-token
  `prefill-trace` capture request overrode thinking off and maximum output to
  64 tokens; normal-thinking TTFT screens were separate. Fixed-depth-1 MTP,
  admission and host-swap guards stayed in place. No installation, merge or
  runtime library replacement.
- Context7 was unavailable. Library work used pinned installed declarations,
  installed Metal headers, MLX v0.32.4 source and its official documentation.

The loaded Qwen4 model inherits the **mlx-vlm** switch layers. The first capture
hook wrapped mlx-lm only, so it collected native timings but no expert tensors.
The corrected hook wraps both independent classes. Captures in
`baseline-state-expert-b` and `prepared-state-q4` contain the actual sorted
gate/up and down operands from a real second 4096-token chunk.

Existing `qwen35_moe_gate_up` fusion already applies to 49 loaded layers,
including the model's additional MTP layer. It concatenates quantised output
rows and issues one gate/up projection. Sorting is also already enabled. These
were verified from the live log and source; neither is a new candidate here.

## Native timing feasibility and attribution

An isolated MLX Release/debug-label build was attempted. Xcode's `metal` launcher
exists, but its optional Metal Toolchain is missing. Shader compilation and JIT
header preprocessing both fail. The build was stopped; no generated substitute
or rebuilt MLX library was used and no global toolchain was changed.

The replacement diagnostic is a small Objective-C++ shim compiled with ordinary
Clang. Server-only `ctypes.CDLL` loading records the actual `MTLFunction.name`
and public Metal `GPUStartTime`/`GPUEndTime` of completed command buffers. Clients
and monitoring processes do not inherit it. A small real-Metal capture produced
a `.gputrace`, correct matrix output, actual kernel names and nonzero GPU times.
This is timing evidence, not Metal logging.

The full-model diagnostic in `expert-diagnostic-retry` used
`MLX_MAX_OPS_PER_BUFFER=0`, profiling only the second actual 4096-token chunk.
Its lifecycle swap guard recorded zero new writes. An earlier load attempt
stopped at the unchanged 1 GiB guard before prefill and contributes no timings.

The selected 5.666520 s window contains 4,709 records from one server PID. After
excluding 634 invalid/zero intervals, there are 4,075 valid buffers. Their summed
durations are 8.148712 s, interval union 5.628728 s and overlap 2.519984 s. The
large overlap is retained, not silently counted as exclusive GPU time.

| Mixed operation group | Buffers | Summed buffer duration | Share of summed durations |
| --- | ---: | ---: | ---: |
| `gather_mm_offsets` + Q4 sorted expert QMM | 92 | 1.966786 s | 24.136% |
| `gather_mm_offsets` + Q8 sorted expert QMM | 4 | 0.081631 s | 1.002% |
| Both expert groups | 96 | 2.048417 s | 25.138% |

The actual matrix route is
`affine_gather_qmm_rhs_nt_bfloat16_t_gs_64_b_{4,8}_bm_16_bn_32_bk_32_wm_1_wn_2_align_N_t_align_K_t`.
It is the M4 steel SIMD-matrix path, not M5 NAX. Each offset/matrix pair remains
one indivisible mixed group. The table gives shares of summed buffer durations,
not exclusive GPU utilisation, TTFT shares or a rigorous Amdahl bound. Splitting
the buffer policy and adding native callbacks changes scheduling; these timings
are diagnostic attribution only.

## Two tile variants and real-input checks

The archived experiment reads the installed MLX0.32.4 headers. It reuses the
original dequantisation, `BlockMMA`, K loop, row-tile scheduling and offset binary
search, with safe math. The two variants change only output tile extent:

- `m32`: BM32/BN32, retaining BK32/WM1/WN2.
- `n64`: BM16/BN64, retaining BK32/WM1/WN2.

The real operands have 512 experts and 40,960 sorted routes (4096 tokens × 10).
Fused gate/up is N1280/K2560; down is N2560/K640. Both variants matched every
output's raw 16-bit representation on each captured Q4 and Q8 case: eight exact
comparisons, all zero maximum absolute error.

Candidate replay used normal MLX buffer policy, not the diagnostic ops=0 policy.
Each operation had three warm-ups and 12 rotating/reversing interleaved timing
rounds. The evidence preserves every sample and paired-round ratios. Times
include the candidate's separate offset prepass, dispatch and synchronisation.

| Real operation | Stock median | m32 median | n64 median | m32 speedup | n64 speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8 gate/up | 27.898 ms | 26.406 ms | 32.298 ms | 1.0565× | 0.8637× |
| Q8 down | 13.512 ms | 12.624 ms | 14.825 ms | 1.0703× | 0.9115× |
| Q4 gate/up | 24.601 ms | 24.544 ms | 25.989 ms | 1.0023× | 0.9466× |
| Q4 down | 13.555 ms | 13.029 ms | 13.053 ms | 1.0403× | 1.0385× |

The m32 Q4 pair sums to 37.573 ms versus 38.156 ms stock, a 1.528% reduction in
the sum of these isolated medians. Q4 dominates the model's routed matrices.
Q8's larger isolated gain affects only four diagnostic buffers. Neither result
justifies introducing this custom serving path. No percentage extrapolation is
presented as measured whole-model benefit.

The Q4 replay explicitly sampled cumulative host swap writes from process
startup through every measured round: 216306057216 bytes before and after,
zero new writes, with the same 1 GiB limit. Q8 replay predates that dedicated
sampling code; its recorded post-run `vm_stat` counter matches both surrounding
successful capture runs. This distinction is retained in the evidence.

## Verification and limits

- Native shim compiles; real smoke capture and full-model labelled native
  timestamps succeeded.
- CPU attribution check covers invalid intervals, overlaps and mixed groups.
- Real Q4/Q8 checks and 12-round samples are in `expert-prefill-evidence.json`.
- Candidate code is experimental and archived, with no serving entry point.
  Full-model candidate TTFT, rollback, cache continuation and sampled coding/tool
  qualification were not run because neither variant was retained. The component
  checks do not substitute for those gates.
- No captured tensors, keys, binary libraries or GPU capture bundles are in this
  commit. The evidence includes tool/runtime hashes and original artifact paths.

Sources checked: [MLX0.32.4 Metal debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html),
[MLX v0.32.4 sorted QMM dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp),
[pinned quantised kernel](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/quantized.h).
Public Metal timing declarations were checked in the local macOS SDK's
`MTLCommandBuffer.h`; kernel APIs and math-mode options in installed MLX stubs.
