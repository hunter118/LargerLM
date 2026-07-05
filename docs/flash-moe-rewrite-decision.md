# Flash-MoE Rewrite Decision

## Decision

The current LargerLM Python-orchestrated runtime should not be treated as the
final high-performance inference path.

Keep it as:

- a GLM-5.2 packer and layout validator,
- a launch-audit and memory-safety gate,
- a correctness oracle for small smokes and layer-level comparisons,
- a source of reusable Metal kernels and prepared artifacts.

Build a new Flash-MoE-style GLM runtime as the performance path: one persistent
Objective-C/C++/Metal process with direct expert reads into reusable Metal
buffers, layer-level command-buffer scheduling, GPU-side residual/combine/norm
boundaries, and no Python/file handoff inside the token loop.

## Evidence

Flash-MoE's source is a specialized single-process engine. Its hot path keeps
non-expert weights mmap'd, wraps them as Metal buffers, reads selected expert
slots via `pread` into 2MB-aligned shared Metal buffers, and batches each
layer into a small CMD1/CMD2/CMD3 pipeline. CMD3 contains expert forward,
shared expert, combine, residual, and next-layer norm, then commits without a
wait so the next layer can submit immediately.

The local Flash-MoE runtime trial is intentionally skipped until the packed Qwen
weights are available. The source review is still decisive enough for the GLM
path: Flash-MoE's retained wins are direct `pread`, no application expert cache,
OS page-cache reuse, 2MB-aligned I/O buffers, GPU-side combine/norm, and
row-tiled Metal matvec kernels with SIMD reductions. Its own discarded list is
also relevant here: expert `mmap`, dispatch_io, routing prediction, compression,
and read-ahead hints are not the next place to spend time.

LargerLM's current GLM-5.2 smoke is safe but not efficient enough:

- 126 prompt tokens plus one generated token: 141.647s on the same-window
  baseline rerun, about 0.89 prompt tok/s.
- Historical selected 126-token MLA-server baseline: 132.394s.
- Prepared memory profile: about 18.65GB estimated live working set, requiring
  at least 24GB free unified memory before loading.
- Routed MoE on the same-window baseline: 32.686s total, while runner timing
  accounts for 15.786s and staged-MoE wall phases account for 16.753s. The
  measured outer residual is about 15.933s.
- The current minimal `glm_moe_infer` one-token decode/logits artifact,
  `smoke-decode-1tok-metal-logits-result.json`, is runnable and safe but not
  yet usable-speed: it generated `[15]` in `6.535s`, with `6.004s` in 78 decode
  layers (`3.911s` attention, `2.084s` MLP), and estimated `12.538GB` of reads
  per generated token (`12.032GB` routed experts plus `0.506GB` logits).
- The current prepared HTTP/Metal runtime bridge is now runnable for a guarded
  1-token token-id request and reuses the in-process MLA KV-B cache across
  HTTP requests. With a 16GiB live cap, a 24GiB free-unified-memory guard, and a
  4608MiB MLA KV-B cache cap, two sequential prompt `[0]` requests generated
  `[15]`; after the context=1 MLA fast path and default `o_proj` resident simd
  matvec, the first stored 78 cache entries and took `2.418540s` decode time,
  while the second hit 78/78 cache entries and took `1.209470s` decode time.
  This is materially better than the first 6.535s smoke, but still about
  `0.827 tok/s` on the second request, so it is minimum-runnable evidence rather
  than useful chat-speed evidence.
- The large Flash-MoE weight replay is skipped for now at operator request.
  Instead, prepared-server health now computes the config-derived MLA KV-B F32
  cache plan and, for Metal runtime generation, emits an opt-in launch-profile
  suggestion. Public GLM-5.2 needs `4,580,179,968` bytes (`4368MiB`) of cached
  layer data, rounded to a `4608MiB` live-estimate cap; this remains disabled
  unless launch args explicitly include the suggested flags.

The residual is not a small tuning artifact. It comes from the design shape:
Python orchestration, router JSON files, staged files, compact-stage layout
files, per-boundary validation, scatter/gather files, and multiple runner
interfaces. Those are valuable for safety and auditability, but they are the
wrong granularity for a production token loop.

The single-process `glm_moe_infer` path is the right replacement direction, but
its current decode-layer implementation still contains probe-era boundaries:
subprobes historically recreated Metal libraries, pipelines, command queues,
buffers, and NSData/file handoffs inside the layer loop, and routed expert slots
are not yet scheduled like Flash-MoE's K-way `pread` hot path. The next work
therefore belongs in the persistent C/Metal runtime context, not in the older
Python/file orchestration layer.

GLM-5.2 is also intrinsically heavier than Flash-MoE's Qwen target:

- Flash-MoE Qwen path: hidden 4096, 60 layers, K=4, about 7.08MB per 4-bit
  expert slot.
- GLM-5.2 path: hidden 6144, 78 layers with 75 MoE layers, K=8, about
  20.05MB per routed expert slot.
- Per-token cold routed expert bytes for GLM are roughly
  `75 * 8 * 20.05MB`, about 12GB before OS page-cache reuse.
- Routed expert math is about 7.5x heavier than Flash-MoE's Qwen expert path
  under the same top-level matvec accounting.

That means a Flash-MoE-style GLM engine should be much faster than the current
Python path, but it should not be expected to match Flash-MoE's quoted Qwen
4.36 tok/s without major cache locality, reduced routing, or model-specific
shortcuts. A realistic first target is to make decode/pre-fill bounded, stable,
and near the hardware lower bounds, then measure.

## New Runtime Shape

Target binary:

- `metal/glm_moe_infer` or equivalent, separate from the safety-first Python
  CLI.

Core loop:

1. mmap resident GLM weights and wrap the backing file in one Metal buffer.
2. open all packed routed expert layer files once.
3. allocate reusable 2MB-aligned expert staging buffers with
   `newBufferWithBytesNoCopy`.
4. keep decode/cache state resident in the process.
5. run each layer as a small fixed command-buffer pipeline:
   attention projections, MLA/DSA attention, post-attention norm/router/shared
   work, direct expert reads, routed expert forward, shared/routed combine,
   residual add, and next-layer norm.
6. eliminate prompt-token router JSON, compact-layout JSON, per-layer output
   files, and subprocess boundaries from the hot loop.

Current bring-up status: `glm_moe_infer` now has a shared direct routed-expert
read primitive instead of separate probe-only read loops. It range-checks
selected expert ids, submits a batched `pread` dispatch through persistent
workers, reads into reusable 2MB-aligned `newBufferWithBytesNoCopy` Metal
buffers, disables fd readahead for packed expert layer files, and exposes
dispatch/task/worker telemetry from both `--probe-expert-read` and the layer
MoE consumer. The tiny MXFP4 smoke verifies the pooled read path and rejects an
out-of-range expert before any slot read. Resident MXFP4 linear probes can also
use the Flash-MoE-style resident path with
`--mmap-resident --wrap-resident-metal`: weight and scale tensors are addressed
by offsets inside one mmap-backed Metal buffer, with no explicit matrix staging
`pread`, and the tiny resident-linear smoke checks equality against the staging
fallback plus the reduced scratch estimate.

M5 acceleration:

- Treat Metal/MPS tensor operations as an execution backend for large resident
  GEMMs and prefill matrix batches.
- Do not make the first milestone depend on ANE-style acceleration. The
  exposed and controllable path must first be a correct single-process Metal
  pipeline; M5 neural/tensor acceleration can then be added behind backend
  selection for eligible prefill shapes.

## Safety Requirements

The rewrite must keep the no-OOM contract:

- preflight estimates from the existing prepared package still gate startup,
- memory allocation sizes are fixed and reported before weight loading,
- no application-level expert cache is enabled by default,
- OS page cache is trusted unless a measured replay proves otherwise,
- all large buffers are bounded and reused,
- long generation starts only after free unified memory and disk checks pass.

## Milestones

1. Single-process GLM weight loader:
   mmap resident weights, open packed expert files, initialize Metal, report
   allocations, then exit.
2. One-layer routed MoE parity:
   direct read selected GLM experts into aligned Metal buffers and compare
   against the existing `run_staged_routed_moe_batch` output on a tiny fixture.
3. One-layer fused GLM MLP:
   post-attention norm, router, routed experts, optional shared expert,
   residual add, and output in one process.
4. Full one-token decode skeleton:
   embeddings, all layers, cache state, final logits, one generated token,
   compared against the existing smoke output `[15]`.
5. Prompt prefill path:
   batch where it is actually beneficial, but without file intermediates in
   the layer loop.
6. M5 tensor backend:
   add MPS/Metal tensor-op backend for eligible resident prefill GEMMs after
   the single-process baseline is measured.

## Promotion Gate

Do not promote the new engine until it satisfies all of:

- generated token parity on the selected GLM-5.2 smoke,
- no startup when memory/disk safety checks fail,
- no residual runner processes after failure,
- lower total latency than the current selected 126-token baseline,
- benchmark output with per-layer timing and live allocation summary.

## Current Progress

- `metal/glm_moe_infer` is now the first executable skeleton for the rewrite.
  It initializes Metal, loads the prepared resident/expert layout JSON, validates
  `resident.bin`, opens all routed expert layer files, allocates reusable
  2MB-aligned expert buffers, optionally mmaps resident weights, reports the
  live envelope, and exits without entering inference.
- On the local M5 Max prepared GLM-5.2 MXFP4 package:
  `metal/glm_moe_infer --prepared artifacts/glm-5.2-mxfp4/largerlm-prepared
  --max-live-working-set-mib 1024 --min-free-unified-memory-gib 24` passed,
  opened 75 expert files, reported 358.594GiB of routed expert files,
  9.366GiB of resident weights, 8 reusable 20MiB expert buffers, and only
  0.156GiB estimated live bytes without resident mmap.
- The resident mmap gate also passed:
  `--mmap-resident --max-live-working-set-mib 20480 --json` reported schema
  `largerlm.glm_moe_infer_loader.v1`, `device_name="Apple M5 Max"`,
  `resident_mmap=true`, `expert_files_opened=75`, and
  `estimated_live_working_set_bytes=10224752128`.
- The first direct expert-read probe now runs against the real prepared GLM
  files. Command:
  `metal/glm_moe_infer --prepared artifacts/glm-5.2-mxfp4/largerlm-prepared
  --probe-expert-read --probe-layer 67
  --probe-experts 34,36,89,149,152,183,201,206 --expert-buffer-count 8
  --max-live-working-set-mib 1024 --json`. It does not mmap resident weights,
  keeps the estimated live envelope at `167772160` bytes, opens the 75 expert
  files once, and reads eight `20054016` byte slots into eight aligned Metal
  shared buffers. The local run read `160432128` bytes in `0.012959s`, reporting
  `11.53GiB/s`, with per-expert sample checksums recorded in the JSON output.
  This validates the Flash-MoE-style direct slot I/O path.
- A full 75-layer expert-read lower-bound probe also passes:
  `--probe-all-layers --probe-experts 34,36,89,149,152,183,201,206
  --expert-buffer-count 8 --max-live-working-set-mib 1024`. It reuses the same
  eight aligned buffers across all MoE layers, does not mmap resident weights,
  keeps estimated live bytes at `167772160`, reads `11475.000MiB` in
  `0.997703s`, and reports `11.23GiB/s`. This is the most important current
  efficiency signal: the GLM-5.2 cold expert-I/O lower bound is about one second
  per decode token before routed expert math, attention/cache work, final logits,
  and sampling. The new path is still worth pursuing because it removes the
  current Python/file/subprocess overhead, but GLM-5.2 should not be benchmarked
  against Flash-MoE's smaller Qwen MoE speed without accounting for this much
  larger per-token routed payload.
- `metal/glm_moe_infer` now has the first self-contained expert-compute probe:
  `--probe-layer-moe` reads selected MXFP4 expert slots and runs a group32
  Metal SwiGLU + down weighted-add path inside the new runtime process. The
  new `metal/glm_moe_infer_mxfp4_moe_smoke.py` builds the tiny layer-1 MXFP4
  fixture, runs experts `1,0` with weights `2/3,1/3`, and checks the full
  32-float output against the known value. Local result:
  `output[0]=16.4258575`, expected `16.425862`, max output diff about
  `4.6e-6`, `expert_bytes_read=3264`, `estimated_live_working_set_bytes=4194688`,
  `expert_read_dispatch_count=1`, `expert_read_task_count=2`,
  `expert_read_max_worker_count=2`, `expert_read_pool_dispatch_count=1`, and
  `ok=true` under an 8MiB cap.
- Milestone 2 is partially satisfied for the tiny MXFP4 fixture: direct slot
  read plus routed expert compute now runs in the new single process.
- Milestone 2 is also satisfied for a controlled real GLM-5.2 routed expert
  compute slice. `metal/glm_moe_infer_real_layer_moe_smoke.py` generates a
  deterministic 6144-float input, runs layer 67 experts
  `34,36,89,149,152,183,201,206` with weights
  `0.18,0.15,0.13,0.12,0.11,0.1,0.11,0.1`, uses the existing
  `metal/largerlm-runner --run-moe` fused MXFP4 path as the oracle, and then
  runs `glm_moe_infer --probe-layer-moe` on the same real packed expert file.
  Local result: all 6144 output floats match exactly (`max_abs_diff=0.0`), the
  new runtime reads `160432128` expert bytes, reports `0.019753s` elapsed,
  `0.008267s` expert read, `0.011118s` kernel time, and keeps estimated live
  bytes at `21028864` under a 256MiB cap.
- The first resident router gate slice also has production-semantics oracle
  parity. `metal/glm_moe_infer_real_router_smoke.py` generates the same
  deterministic layer-67 hidden vector, runs `metal/largerlm-runner
  --run-router` as the oracle, then runs `glm_moe_infer --probe-router` using
  the prepared GLM router metadata: sigmoid scores, F32 correction bias,
  normalized top-k, `n_group=1`, `topk_group=1`, and routed scale `2.5`. Local
  result: top-k experts `240,96,27,174,243,108,109,101`, scaled weights, and
  all 256 logits match exactly (`max_weight_diff=0.0`, `max_logit_diff=0.0`);
  the selected weights sum to `2.5000001788139343`. Router-only probes now set
  the active expert buffer count to zero, so the default eight requested expert
  buffers do not inflate the live envelope; the smoke reports `4221952`
  estimated live bytes under a 64MiB cap.
- Router selection is now wired directly into the single-process routed expert
  compute path. `metal/glm_moe_infer_real_router_moe_smoke.py` runs the old
  runner oracle as `--run-router` plus `--run-moe`, then runs
  `glm_moe_infer --probe-router-moe` on the same layer-67 hidden vector. Local
  result: router logits and weights still match exactly, all 6144 routed-MoE
  output floats match exactly (`max_abs_diff=0.0`), the new runtime reads
  `160432128` expert bytes, reports `0.023560s` MoE elapsed,
  `0.010267s` expert read, `0.013029s` expert kernels, activates one reusable
  expert buffer, and keeps estimated live bytes at `25250816` under a 64MiB
  cap.
- The full one-layer GLM MLP block now has oracle parity, including the
  resident MXFP4 shared expert path. `--probe-mlp-block
  --include-shared-expert` runs post-attention RMSNorm from
  `model.layers.67.post_attention_layernorm.weight`, production router,
  routed expert compute, resident shared expert compute, and residual add in
  `glm_moe_infer`. `metal/glm_moe_infer_real_mlp_block_smoke.py
  --include-shared-expert` compares it with `metal/largerlm-runner
  --run-mlp-block --include-shared-expert`. Local result: router logits and
  weights match exactly, all 6144 output floats match within
  `1.0477378964424133e-09`, the new runtime reads the 12KiB BF16 norm vector
  plus `160432128` routed expert bytes and `20054016` shared expert bytes,
  reports `0.004909s` RMSNorm elapsed, `0.026132s` MLP elapsed,
  `0.001452s` shared expert read, `0.001212s` shared expert kernels, activates
  one reusable expert buffer, and keeps estimated live bytes at `25336832`
  under a 64MiB cap.
- After the Flash-MoE source review, `glm_moe_infer` now has a row-tiled fast
  MXFP4 expert kernel path for supported GLM-5.2 group32 expert shapes. It is
  now enabled automatically for those shapes and can be forced back to the
  scalar parity path with `LARGERLM_GLM_MOE_INFER_FAST_MXFP4=0` (or `off` /
  `scalar`). The fast path uses 256-thread groups and SIMD reductions for
  group32 expert SwiGLU/down-add. The tiny MXFP4 fixture passes with
  `fast_mxfp4_kernel=true`. On real layer 67, the routed MoE fast smoke compares
  against `metal/largerlm-runner` with max output diff
  `1.199040866595169e-14`, reads `160432128` routed bytes, and reports
  `0.007381s` expert kernel time versus the earlier scalar `0.011118s`. The
  router+shared MLP block fast smoke also passes: router logits/weights match
  exactly, final hidden max diff is `1.4551915228366852e-09`, routed kernel time
  is `0.007613s`, and shared kernel time is `0.000774s` under the same 64MiB live
  cap.
- The `glm_moe_infer` process now caches its compiled Metal library, compute
  pipeline states, and command queue per device instead of rebuilding them at
  every probe boundary. This is a first hot-loop cleanup toward the persistent
  Flash-MoE-style runtime. Validation after the change: `make -C metal
  glm_moe_infer`, `python3 metal/glm_moe_infer_mxfp4_moe_smoke.py`, and
  `python3 metal/glm_moe_infer_real_layer_moe_smoke.py` all pass; the real
  layer-67 parity smoke still matches all 6144 floats exactly under a 256MiB
  cap.
- Routed expert slot reads in `glm_moe_infer --probe-layer-moe` now use a
  K-way parallel `pread` helper when the caller provides multiple reusable
  expert buffers. This keeps the live-memory envelope governed by
  `--expert-buffer-count` instead of adding any application cache. Validation:
  `python3 metal/glm_moe_infer_real_layer_moe_smoke.py`
  matches all 6144 layer-67 output floats exactly, uses eight 20MiB slots under
  a 256MiB cap, reads `160432128` routed bytes, reports `0.004472s` expert-read
  wall time, and encodes the eight expert forwards in one command buffer. Its
  read telemetry reports one dispatch, eight read tasks, max read batch 8, max
  persistent workers 8, one pooled dispatch, and zero serial fallbacks. The
  helper now uses a process-lifetime pthread worker pool plus fixed stack arrays
  for read tasks and slot indices instead of per-batch heap allocation or
  per-layer thread creation. `--expert-buffer-count` is capped at 64 so a bad
  launch cannot expand the hot-loop stack or reusable slot envelope silently;
  `--expert-buffer-count 65` is rejected before heavy loading. The loader JSON
  now exposes pooled-read telemetry for MoE probes and decode-layer summaries:
  dispatch count, total read task count, max tasks per dispatch, max persistent
  worker count, pooled dispatch count, and serial fallback count. The real
  router->MoE, MLP-block, decoder-layer, and decode-layers smokes now assert
  this telemetry on their performance-path top-k expert buffers instead of
  validating only the one-buffer serial-safe path.
- The first decoder-attention building block is now in the new runtime:
  `--probe-resident-linear --resident-tensor-name ...` reads a resident MXFP4
  `.weight/.scales` pair, runs a single-process Metal matvec, writes F32
  output, and participates in the same live working-set cap. The tiny
  `metal/glm_moe_infer_resident_mxfp4_linear_smoke.py` fixture matches
  `metal/largerlm-runner --run-resident-linear` exactly with zero expert
  buffers and `2097408` estimated live bytes under an 8MiB cap. The real GLM
  `metal/glm_moe_infer_real_resident_linear_smoke.py` gate compares
  `model.layers.67.self_attn.q_a_proj.weight`; local result: all 2048 output
  floats match exactly (`max_abs_diff=0.0`), the new runtime reads `6684672`
  resident bytes, reports `0.003977s` elapsed, `0.000495s` read time,
  `0.003230s` kernel time, activates zero expert buffers, and keeps estimated
  live bytes at `8421376` under a 16MiB cap.
- That resident-linear primitive is now composed into a real single-token
  attention projection probe. `--probe-attn-projections` runs input RMSNorm,
  q_a/q_b/kv_a resident MXFP4 projections, q_a RMSNorm, and kv_a-prefix RMSNorm
  inside `glm_moe_infer`, writing the old runner's projection files:
  `attn_input_norm.f32`, `attn_q_a.f32`, `attn_q_a_norm.f32`,
  `attn_q_b.f32`, `attn_kv_a.f32`, and `attn_kv_a_norm.f32`.
  `metal/glm_moe_infer_real_attn_projections_smoke.py` compares layer 67
  against `metal/largerlm-runner --run-attn-projections`. Local result: all six
  output files match exactly (`max_abs_diff=0.0`), the new runtime reads
  `26407936` resident bytes, reports `0.038256s` elapsed, activates zero expert
  buffers, and keeps estimated live bytes at `29615360` under a 64MiB cap.
- The next RoPE split/rotation gate is also in the new runtime.
  `--probe-rope-split` consumes `attn_q_b.f32` and a `kv_a` RoPE suffix, writes
  `q_nope.f32`, `q_rope.f32`, rotated `q_rope`, and rotated `k_rope`, and
  participates in the live working-set cap. `metal/glm_moe_infer_real_rope_split_smoke.py`
  first runs the verified attention projection probe, slices the 64-dim
  `k_rope` suffix from `attn_kv_a.f32`, then compares the new runtime against
  `metal/largerlm-runner --run-rope-split-batch` with GLM's local decode dims
  `num_heads=64`, `qk_nope_dim=192`, `rope_dim=64`. Local result: `q_nope`,
  `q_rope`, rotated `q_rope`, and rotated `k_rope` all match exactly
  (`max_abs_diff=0.0`), the new runtime reports `0.003411s` elapsed, activates
  zero expert buffers, and keeps estimated live bytes at `147968` under an
  8MiB cap.
- KV-cache movement is no longer an unverified gap. `--probe-attn-projections`
  now accepts `--cache-layout`, `--cache-file`, `--position`, and
  `--max-cache-file-mib`, validates the decode-cache layout and backing file,
  and appends the single-token KV-A row as BF16 or F32. `metal/glm_moe_infer_real_attn_cache_smoke.py`
  compares layer 67 against `metal/largerlm-runner --run-attn-projections` with
  cache append enabled. Local result: the six projection files still match
  exactly (`max_abs_diff=0.0`), the BF16 cache write is byte-identical, the new
  runtime writes `1152` cache bytes at position 2 in `0.000116s`, activates zero
  expert buffers, and keeps estimated live bytes at `29617664` under a 64MiB
  cap.
- Single-token MLA attention also has oracle parity in the new runtime.
  `--probe-mla-attention` reads the bounded decode-cache rows, builds the GLM
  absorbed-alias value view from `embed_q`/`unembed_out`, and runs the score/value
  Metal kernel under the live working-set cap. `metal/glm_moe_infer_real_mla_attention_smoke.py`
  composes the verified projection, cache, and RoPE gates, then compares layer 67
  against `metal/largerlm-runner --run-mla-attention`. Local result: all 16384
  attention value floats match exactly (`max_abs_diff=0.0`), the new runtime
  reads `2304` F32 cache bytes plus `7798784` stored resident value bytes,
  reports `0.012330s` MLA kernel time, `0.016223s` value-read time, activates zero
  expert buffers, and keeps estimated live bytes at `125374976` under a 192MiB
  cap.
- Single-token attention output now has oracle parity too. `--probe-attn-output`
  applies the resident MXFP4 `self_attn.o_proj.weight` to the MLA value row and
  adds the residual hidden vector with a small Metal F32 add kernel.
  `metal/glm_moe_infer_attn_output_smoke.py` checks the tiny semantics fixture,
  and `metal/glm_moe_infer_real_attn_output_smoke.py` reuses the verified real
  MLA output before comparing layer 67 against `metal/largerlm-runner
  --run-attn-output`. Local result: both the pure projection and final hidden
  output match exactly (`max_abs_diff=0.0`), the new runtime reads `53477376`
  resident bytes, reports `0.003824s` `o_proj` kernel time, `0.002943s`
  residual-add time, activates zero expert buffers, and keeps estimated live bytes
  at `54640640` under a 192MiB cap.
- The one-token decoder-layer skeleton is now present in `glm_moe_infer`.
  `--probe-decoder-layer` composes attention projections with cache append, RoPE
  split, MLA attention, attention output, post-attention RMSNorm, router, routed
  MoE, optional shared expert, and residual add in one process while keeping the
  same live working-set cap. `metal/glm_moe_infer_real_decoder_layer_smoke.py
  --include-shared-expert` compares real layer 67 against
  `metal/largerlm-runner --run-decoder-layer`. Local result: router logits and
  weights match exactly, all 6144 final hidden floats match within
  `1.2516975402832031e-06`, the new runtime reads `160432128` routed expert bytes
  plus `20054016` shared-expert bytes, writes `2304` F32 cache bytes, activates
  one reusable expert buffer, reports `0.126133s` decoder-layer elapsed, and keeps
  estimated live bytes at `235118080` under a 512MiB cap.
- Dense-prefix MLP support for GLM layers 0-2 is now covered by the new runtime.
  `--probe-dense-mlp-block` runs post-attention RMSNorm, resident MXFP4
  gate/up/down projections, F32 SwiGLU, and residual add without opening expert
  files or allocating expert buffers. `metal/glm_moe_infer_real_dense_mlp_smoke.py`
  compares real layer 0 against `metal/largerlm-runner --run-dense-mlp-block`.
  Local result: all 6144 output floats match exactly (`max_diff=0`), the new
  runtime reads `120336384` resident bytes, activates zero expert buffers,
  reports `0.109437s` dense-MLP elapsed, and keeps estimated live bytes at
  `126418944` under a 512MiB cap.
- The dense-prefix full decoder-layer path is also covered now.
  `--probe-dense-decoder-layer` composes attention projections with cache append,
  RoPE split, MLA attention, attention output, and the resident dense MLP in one
  `glm_moe_infer` process. `metal/glm_moe_infer_real_dense_decoder_layer_smoke.py`
  compares real layer 0 against `metal/largerlm-runner --run-dense-decoder-layer`.
  Local result: all 6144 final hidden floats match exactly (`max_abs_diff=0`),
  the cache writes are byte-identical, the new runtime opens zero expert files,
  activates zero expert buffers, reads `120336384` dense-MLP resident bytes,
  reports `0.135031s` dense-decoder elapsed, and keeps estimated live bytes at
  `336200192` under a 512MiB cap.
- The first continuous layer-list decode driver is present in `glm_moe_infer`.
  `--probe-decode-layers --decode-layers CSV` preflights each requested layer,
  classifies layers as dense when no packed expert layer exists and MoE when a
  packed expert layer exists, uses bounded multi-slot expert staging for MoE
  layers, and passes hidden-state files from one layer to the next inside one
  `glm_moe_infer` process. `metal/glm_moe_infer_real_decode_layers_smoke.py
  --quiet-commands` compares real layers `0,3` against
  `metal/largerlm-runner --run-decoder-layers --dense-layers 0`, then uses the
  same new-runtime invocation to stream final RMSNorm plus chunked MXFP4
  `lm_head` top-k from the layer-list output. Local result: the dense-to-MoE
  final hidden vector matches within `1.4901161193847656e-08`, cache writes are
  byte-identical, final top-k token ids are identical
  (`140366,83824,103983,105455,33166,60863,40520,33189`), logits match within
  `2.86102294921875e-06`, the new runtime reads `120336384` dense-MLP resident
  bytes, `160432128` routed expert bytes, and `505540608` final-logits bytes,
  activates nine reusable expert buffers for top-k 8 plus shared expert, feeds
  the second layer from the prior layer's in-memory hidden vector
  (`memory_chain_bytes=24576`,
  `memory_input_layer_count=1`), and keeps each layer's hot
  q_b/K-RoPE/q_nope/q_rope/attention-value bridge in memory
  (`hot_intermediate_memory_bytes=393728`). With
  `--skip-debug-intermediates`, the same smoke asserts no files are written
  under the new runtime work directory (`debug_intermediates_written=false`).
  Final logits now also consumes the decode layer-list's in-memory final hidden
  instead of re-reading `--output-f32` (`source=decode_layers_memory_output`).
  The no-debug path fuses the pre-cache attention projection phase into one
  command buffer per layer (`attn_projection_command_buffers=[1,1]`), fuses
  RoPE split plus MLA attention into one command buffer per layer
  (`rope_mla_command_buffers=[1,1]`), and fuses attention output `o_proj`
  matvec plus residual add into one command buffer per layer
  (`attn_output_command_buffers=[1,1]`). The dense-prefix MLP now runs
  RMSNorm, gate/up MXFP4 matvecs, SwiGLU, down MXFP4 matvec, and residual add
  inside one command buffer (`dense_mlp_command_buffers=[1]`). The MoE MLP path
  now uses bounded multi-slot expert staging: with nine active expert buffers
  for top-k 8 plus shared expert, it reads the selected slots first and encodes
  all routed/shared expert work plus residual add into one command buffer
  (`moe_mlp_command_buffers=[1]`, `expert_buffer_count=9`). It reports
  `0.124153s` for the two-layer decode probe plus `0.056369s` for final top-k,
  and keeps estimated live bytes at `542163472` under a 768MiB cap. The
  follow-up final-logits-only mmap path, `--mmap-final-logits`, maps only
  `lm_head` weight/scale ranges as Metal buffers and leaves the rest of
  resident weights on the staged/read path. The bounded two-layer smoke with a
  2048MiB live cap reports `1031513536` estimated live bytes,
  `final_logits_lm_head_bytes_read=0`, `final_logits_bytes_read=12288`, and a
  normal `0.020533s` attention-output time while final top-k takes `0.055212s`.
  This replaces the tempting but wrong blanket `--wrap-resident-metal` result:
  full resident wrap cut final top-k to `0.004331s` but pushed attention output
  to `1.471s` from cold mmap-backed resident projection access.
  The `probe_decode_layers` JSON now carries top-level MoE timing/read
  aggregation: the same run reports `160432128` routed expert bytes,
  `120336384` dense-MLP resident bytes, `0.004041s` total expert-read time,
  `0.003600s` MoE kernel time, and one pooled 8-task read dispatch with zero
  serial fallbacks.
  The same smoke now also has a `--all-layers --new-only` safety mode. It runs
  all 78 layers (`0,1,2` dense and 75 MoE layers) through the new runtime
  without the old-runner oracle, keeps all layer-local fused command-buffer
  invariants true, reads `12032409600` routed expert bytes plus `361009152`
  dense-MLP resident bytes, feeds 77 later layers from in-memory hidden states,
  keeps `expert_buffer_count=9`, reports `542163472` estimated live bytes under
  the same 768MiB cap, and now measures `2.829287s` for all-layer single-token
  decode plus `0.050884s` for final top-k on the current hot-cache M5 Max run.
  The first all-layer telemetry showed the remaining bottleneck was no longer
  expert streaming, but also exposed a Flash-MoE-style rewrite target in MLA:
  the generic context=1 kernel recomputed the same head score for every value
  dimension even though a one-element softmax always has weight 1. The
  `glm_mla_attention_context1_f32` fast path skips QK/rope scoring for that
  shape. Real layer-67 MLA validation matched the old output exactly while
  cutting the single-layer MLA kernel from about `0.021057s` to `0.003682s`;
  the real decoder-layer smoke reports `mla_attention_kernel_seconds=0.001020`
  with output max diff below `3e-7`. The follow-up resident MXFP4 simdgroup
  matvec is defaulted only for `self_attn.o_proj`; the broader resident fast
  path remains behind `LARGERLM_GLM_MOE_INFER_FAST_RESIDENT_MXFP4=1` because
  enabling it for all resident projections moved the real decoder-layer router
  logit oracle past the existing `1e-5` threshold. This keeps the current
  single-process Metal direction, but it confirms that further progress should
  come from shape-specific kernels and command scheduling, not Python-side
  patching.
  `scripts/decode_telemetry_report.py` summarizes raw `glm_moe_infer` JSON,
  smoke `dims` JSON, `probe_generate.steps`, or the two-request prepared HTTP
  wrapper into the same bottleneck/pooled-read verdict, including the newer MoE
  output-write and per-layer overhead split.
- The requested `danveloper/flash-moe` large-weight local trial was skipped on
  July 5, 2026 because the operator explicitly did not want to download another
  large checkpoint for this step. The efficiency conclusion still changed based on local
  real-weight telemetry: the Flash-MoE-style expert streaming path is no longer
  the dominant cost in the new GLM runtime, while absorbed MLA KV-B
  materialization is repeated per token. `glm_moe_infer` therefore now has an
  explicit opt-in in-process cache for absorbed MLA KV-B F32 views:
  `--cache-mla-kv-b-f32 --max-mla-kv-b-cache-mib N`. It is disabled by default,
  keyed by resident path, layer, tensor offsets/sizes, and dimensions, and its
  configured cap is included in `estimated_live_working_set_bytes` before
  admission. A full 78-layer 2-token GLM-5.2 MXFP4 smoke with a 4608MiB cache
  cap and 16GiB free-memory guard stored all 78 layer views on step 0, hit all
  78 on step 1, used `4580179968` cache bytes, generated `[15,11]`, and measured
  decode times `3.357781s` then `2.731560s` under a `5374387216` byte live
  estimate. The no-cache control generated the same `[15,11]` with
  `2.832896s` then `3.464909s` decode times under a `542549008` byte live
  estimate. This is useful for longer decode runs after warmup, but it does not
  make very short completions fast by itself.
- The same full 78-layer path now also passes the old-runner oracle mode without
  downloading additional model weights. The full hidden output differs by at
  most `0.0002593994140625` under the `5e-4` accumulated-drift threshold, F32
  cache rows differ by at most `1.0967254638671875e-05`, final top-k ids are
  identical (`16,15,17,18,19,21,20,23`), and final top-k logits differ by at
  most `4.1961669921875e-05`. `--output-token-json` now writes the argmax
  generated token from the streamed top-k result without materializing
  full-vocab logits; the full-layer oracle reports `generated_token_id=16`.
  `--output-next-input-f32` now immediately streams that argmax token's MXFP4
  embedding row into a 6144-float next-step input, reads only `3264` embedding
  bytes, and matches the Python embedding oracle exactly (`max_abs_diff=0.0`).
  The latest oracle run measured `4.904478s` for all-layer decode plus
  `0.057639s` for final top-k while preserving a feedback-enabled
  `542163472` byte live envelope and the one-command-buffer invariants for
  attention projection, RoPE+MLA, attention output, dense MLP, and MoE MLP.
- `metal/glm_moe_infer_real_two_step_decode_smoke.py` now exercises the first
  cross-token feedback loop without downloading more weights. It runs two
  consecutive decode calls against the same cache file, feeds step 0's argmax
  embedding into step 1, and verifies cache writes at positions `0` and `1`.
  On all 78 layers it generated `[16,18]`, wrote all 78 layer rows at both
  cache positions, kept generated-token embedding parity exact for both steps,
  and measured `4.923045s` then `5.480701s` decode time with live envelopes
  `542163472` and `542168080` bytes.
- `glm_moe_infer` now has a first in-process greedy loop via
  `--generate-steps`. The new
  `metal/glm_moe_infer_real_generate_steps_smoke.py --all-layers` smoke runs
  two decode steps inside one runtime invocation, reuses the same cache file,
  feeds step 0's generated embedding into step 1 without returning to Python for
  another binary launch, and emits `probe_generate.steps[]` telemetry. On all
  78 layers it generated `[16,18]`, wrote all 78 layer rows at cache positions
  `0` and `1`, kept the final generated-token embedding exact against the
  Python oracle, and measured `4.956343s` then `5.490593s` decode time under a
  `542168080` byte live envelope. This removes the immediate process boundary
  from the two-step decode loop; the next runtime step is turning the probe loop
  into a serving entry point with tokenizer/prompt plumbing.
- The first minimal user-facing wrapper around that loop is now in
  `python3 -m largerlm generate-metal-token-ids`. A real prepared GLM-5.2 MXFP4
  run with `--prompt-token-ids 0 --max-new-tokens 2
  --max-live-working-set-mib 768 --keep-work-dir --json` generated `[15,11]`,
  measured `4.864067s` and `5.453250s` decode time, `10.575633s` end-to-end
  wall time, wrote a persistent work dir, and kept the estimated live working
  set at `542168080` bytes. This remains the current minimum runnable
  single-token-prompt decode path; multi-token prompts now have a separate
  runtime prompt-prefill entry through `prompt_token_ids`.
- Text-level plumbing is now present in
  `python3 -m largerlm generate-metal-text`. The command defaults the tokenizer
  to the prepared manifest's `model_dir`, encodes the prompt locally, calls the
  same bounded Metal token loop, and decodes both generated and full text. It
  also accepts `--chat-messages` and `--chat-messages-file`; those are rendered
  through the local tokenizer chat template and then encoded without adding
  another special-token layer. Multi-token text prompts now automatically use
  the safe runtime prefill path unless the caller explicitly chooses the old
  decode-only risk path; the explicit
  `--allow-decode-only-multi-token-prompt` flag keeps old experimental behavior
  available by using only the last prompt token embedding. The lower-level
  token-id CLI keeps `--prefill-prompt` explicit. That prefill path now uses the
  single-process runtime `prompt_token_ids` path by default: the JSONL service
  embeds each prompt token, writes decode-cache rows through the service memory
  backend, takes first-token logits from the final prompt state, and then
  continues generation in the same runtime request without required `.f32` or
  generated-json handoff files. This is a correct minimum runtime fold, but it
  is still sequential per prompt token rather than the final Flash-MoE-style
  batched/overlapped prompt prefill. `--python-prefill-bridge`
  keeps the older bridge available for comparison. Telemetry reports
  prompt-prefill elapsed time separately from Metal generation elapsed time, so
  the Metal live cap is not misread as the full prefill+decode peak.
  `--prefill-max-live-working-set-mib` can now raise or lower only the bridge
  prefill live cap; the Metal logits/decode cap remains governed by
  `--max-live-working-set-mib` when `--python-prefill-bridge` is selected.
  A real GLM tokenizer smoke with prompt `你好` and `--max-new-tokens 1`
  generated token id `[23]`, decoded it as `"8"`, measured `5.096510s` decode
  time and `5.292902s` end-to-end wall time, wrote a persistent work dir, and
  kept the live envelope at `542163472` bytes under a 768MiB cap. A real
  chat-template smoke with `[{"role":"user","content":"你好"}]` renders to 13
  prompt tokens; the text wrapper now selects runtime prompt prefill by default
  for that shape, while the explicit decode-only risk flag still bypasses it
  for experiments.
- The first MoE router/top-k CPU boundary has been moved out of the decode
  layer hot path for the GLM-5.2 shape. `glm_moe_infer` now has a restricted
  `glm_router_topk_256` Metal kernel that consumes the router logits plus
  optional correction bias and emits only the selected expert ids and weights
  for `n_group=1/topk_group=1`, `num_experts<=256`, and `top_k<=64`. Standalone
  router probes still keep the full-logits CPU oracle path for debugging and
  parity JSON. Validation:
  `metal/glm_moe_infer_real_decode_layers_smoke.py --quiet-commands` reports
  `moe_router_topk_backends=["metal"]` while matching
  the old runner on the `0,3` dense-to-MoE slice (`max_abs_diff=1.49e-08`, final
  top-k ids identical), and `metal/glm_moe_infer_real_router_smoke.py` still
  reports `topk_backend="cpu"` with exact logits/weights parity.
- The prefill bridge now avoids the extra Metal process boundary when the
  caller asks for more than one generated token after prompt prefill.
  `glm_moe_infer --generate-token-ids --generate-steps ...` accepts
  `--generate-first-from-input-logits`, treating `--input-f32` as an already
  prefilling-computed final hidden state for step 0, then feeding the generated
  token embedding into normal decode layers for step 1 and later. The Python
  `generate_metal_token_ids(..., prefill_prompt=True)` wrapper uses this mode
  for `max_new_tokens>1`, so prompt prefill still runs through the existing
  bridge but first-token logits plus continuation decode run in one
  `glm_moe_infer` invocation. Validation:
  `metal/glm_moe_infer_real_generate_steps_smoke.py --layers 0,3 --dense-layers 0 --steps 2 --first-from-input-logits --quiet-commands`
  generated two tokens,
  reported decode times `[0.0, 0.11468]`, wrote only cache position 0 for the
  continuation decode, and kept the `542168080` byte live envelope. The same
  smoke without the flag still passes the ordinary decode-only generate path.
- The decode-only token entry has also moved one more step into
  `glm_moe_infer`. The runtime accepts `--generate-token-ids` as the formal
  greedy generation entry and `--input-token-id N` to decode the initial MXFP4
  embedding row into an in-memory hidden state before feeding the layer loop.
  The Python `generate_metal_token_ids` wrapper now uses this formal entry and
  passes the last prompt token id to Metal instead of first writing `input.f32`
  from Python. Validation:
  `metal/glm_moe_infer_real_generate_steps_smoke.py --layers 0,3 --dense-layers 0 --steps 2 --input-token-id 0 --quiet-commands`
  reports `runtime_entry="generate_token_ids"`,
  `input_token_embedding_bytes_read=3264`, keeps the same `542168080` byte live
  envelope, and verifies the final generated-token embedding against the Python
  oracle. The same smoke without `--input-token-id` still passes the legacy
  `--input-f32` fixture path.
- The formal generation entry can now be launched from a single structured
  request file. `--generate-request-json PATH` loads the decode layer list,
  input source, output paths, cache bounds, GLM attention dimensions, top-k,
  expert buffer count, and live-memory caps, then enters the same
  `generate_token_ids` runtime path. Validation:
  `metal/glm_moe_infer_real_generate_steps_smoke.py --layers 0,3
  --dense-layers 0 --steps 2 --input-token-id 0 --request-json
  --quiet-commands` reports `runtime_entry="generate_token_ids"`, generates
  `[30423,11093]`, keeps `estimated_live_working_set_bytes=542168080`, and
  verifies the final generated-token embedding exactly. This is a narrow but
  useful step toward a persistent serving protocol because the hot generation
  request is no longer spread across a long probe-style argv.
- The Python-facing token generator now uses formal generation requests instead
  of constructing long probe-style argv. Decode-only generation and the
  prompt-prefill continuation path for `max_new_tokens>1` now send request
  objects through `glm_moe_infer --generate-server-jsonl` by default, with no
  required final hidden/generated/next-input debug files and
  `in_memory_decode_cache=true` so decode uses the persistent service's memory
  cache backend. The prefill continuation request sets `input_f32` to
  `prefill_last_hidden.f32` and `generate_first_from_input_logits=true`.
  `--no-generate-server-jsonl` keeps the legacy direct
  `glm_moe_infer --generate-request-json ... --json` path for compatibility
  and explicit smoke coverage. Prompt prefill itself is still the existing
  bridge. Validation:
  `python3 -m py_compile largerlm/metal_generate.py tests/test_metal_generate.py
  tests/test_metal_text_generator.py`, `python3 -m pytest
  tests/test_metal_generate.py tests/test_metal_text_generator.py`, and the
  same two-layer real request JSON smoke all pass. This keeps the user-facing
  CLI stable while moving the internal boundary toward the durable JSONL
  serving protocol.
- The C/Metal runtime admission guard now enforces free unified-memory reserve
  inside `glm_moe_infer` itself. `--min-free-unified-memory-gib` used to be only
  reported; it now reads macOS `host_statistics64` free+inactive+speculative
  pages, requires `available >= estimated_live_working_set_bytes + reserve`,
  and refuses probes/generation before resident mmap or expert staging-buffer
  allocation when either the live cap or min-free reserve fails. The JSON now
  reports `admission_ok`, `available_unified_memory_ok`,
  `system_available_memory_bytes`, `required_available_memory_bytes`,
  `resident_mmap_actual`, and `expert_buffer_count_allocated`. Validation:
  `metal/glm_moe_infer --prepared ... --max-live-working-set-mib 1 --json`
  and `--min-free-unified-memory-gib 1000000 --json` both fail admission with
  `expert_buffer_count_allocated=0`; the normal two-layer request JSON smoke
  still passes and asserts `admission_ok` plus allocated expert-buffer count.
- The Python/CLI Metal generation path now carries that reserve all the way to
  the runtime. `generate_metal_token_ids` defaults
  `min_free_unified_memory_gib` from the prepared manifest recommendation,
  writes it into formal generate request JSON, passes it through the prompt
  prefill bridge, and includes it in the remaining final-logits argv path. The
  CLI exposes the same override for `generate-metal-token-ids` and
  `generate-metal-text`, while `MetalTokenGenerationResult` records admission
  telemetry (`admission_ok`, available/required unified memory, and allocated
  expert-buffer count). Unit coverage now checks decode-only request JSON,
  prefill continuation request JSON, the single-token prefill final-logits
  path, manifest defaults, and text CLI forwarding; the real request JSON smoke
  accepts `--min-free-unified-memory-gib` and reports the same admission fields.
- The current efficiency decision is now reproducible from metadata instead of
  living only in notes. `scripts/flash_moe_efficiency_envelope.py` reads only
  the prepared manifest, model config, and expert layout; it does not open
  model weight payloads or start Metal. On the local GLM-5.2 MXFP4 prepared
  package it reports 75 expert layers, top-k 8, `11.206GiB` routed expert
  reads/token, a `7.08x` routed-read ratio against the Flash-MoE Qwen reference
  shape, and a `1.895s/token` routed-read lower bound at the manifest cold-read
  calibration of `5.915GiB/s`. Re-running the same static envelope with the
  previous direct all-layer probe rate, `--io-gib-per-second 11.23`, gives a
  more optimistic pure expert-I/O lower bound of `0.998s/token`. The decision
  is therefore explicit: current Metal decode is runnable but not usable-speed;
  a Flash-MoE-shaped GLM runtime is still the right direction, but GLM-5.2
  should be treated as roughly 1 tok/s-class before attention/logits/math
  optimizations, not as a model that can inherit Flash-MoE's Qwen 4.36 tok/s
  headline.
- `glm_moe_infer` now has a formal generation JSONL service entry wired to a
  persistent runtime context. `--generate-server-jsonl` resolves the prepared
  resident/expert layouts, initializes the Metal device, opens all routed
  expert layer files once, emits a `runtime_context_ready` JSONL `ready` event,
  accepts formal generation request objects on stdin, applies the same request
  parser used by `--generate-request-json`, validates required
  decode/cache/dimension/memory fields, and exits on `{"command":"quit"}` with
  a JSONL `done` event. The C execution path is now split into
  `run_glm_moe_infer_once(LoaderOptions)` for normal CLI requests and
  `run_glm_moe_infer_with_runtime(LoaderOptions, GlmMoeRuntimeContext *)` for
  server requests, so JSONL execution reuses the loaded layouts, Metal device,
  one open resident-weight fd, 75 open expert file handles, a lazily allocated
  2MB-aligned expert staging-buffer pool, and an optional request-scoped
  decode-cache backend instead of rebuilding them per request.
  Service stdout remains strict JSONL by capturing the executor's pretty JSON
  stdout through an in-memory pipe, parsing it, and returning key fields such
  as `probe_generate` and `generated_token_ids` inline in the JSONL response.
  `executor_stdout_json` is now optional and only writes a debug copy;
  `dry_run=true` keeps a protocol-only path for cheap checks.
  Validation:
  `python3 metal/glm_moe_infer_generate_server_smoke.py` checks ready/request/
  done JSONL without decode; `python3
  metal/glm_moe_infer_generate_server_smoke.py --execute` runs the real
  two-layer GLM-5.2 generation executor through the service without sending
  `output_f32`, `output_generated_json`, `output_next_input_f32`, or
  `executor_stdout_json`; it keeps final decode hidden states, generated-step
  payloads, next-input embeddings, and decode-cache contents in memory,
  preserves
  `runtime_entry="generate_token_ids"` and `min_free_unified_memory_gib`,
  reports `execution_status="executed_once"` plus `runtime_reused=1`, parses
  the captured executor payload without any required stdout file, and verifies
  generated ids `[30423,11093]`. `--execute --write-executor-stdout` also
  covers the optional executor debug-file path, while
  `--execute --write-output-f32 --write-generated-files` covers legacy final
  hidden-state and generated-result files. `--execute --file-decode-cache`
  covers the legacy request-scoped decode-cache fd path.
  The smoke supports repeated requests;
  `--execute --requests 2` keeps nine runtime expert buffers allocated after
  the first request and verifies that the second executor payload reports
  `expert_buffer_pool_reused=1`. The server ready event also
  reports `resident_fd_opened=1`, while default execute responses report
  `decode_cache_backend="memory"`, `decode_cache_fd_opened=0`, and
  `decode_cache_memory_bytes` included in the admission-estimated live working
  set. File-cache compatibility responses report `decode_cache_backend="file"`,
  `decode_cache_fd_opened=1`, and an increasing `decode_cache_fd_open_count`
  when repeated requests use different cache files. Resident
  vector/matrix/tensor reads now prefer the runtime fd instead of opening
  `resident.bin` per read, and decode-cache reads/writes prefer the runtime
  memory backend or runtime cache fd instead of opening the cache file per
  segment.
- Runtime prompt prefill is now minimally folded into `glm_moe_infer`.
  Formal generation requests accept `prompt_token_ids` as an input source.
  In JSONL server mode, the executor embeds each prompt token, runs the normal
  decode-layer stack to populate cache rows, keeps the final prompt hidden state
  in memory, emits first-token logits from that state, and continues greedy
  generation without required `output_f32`, `output_generated_json`,
  `output_next_input_f32`, or `executor_stdout_json` files. The Python
  `generate_metal_token_ids(..., prefill_prompt=True)` and
  `generate-metal-text --prefill-prompt` paths now use this runtime path by
  default; `--python-prefill-bridge` keeps the older external prefill bridge
  available. Prepared serving can now carry the opt-in in-process MLA KV-B
  cache through the same persistent runtime with
  `--metal-runtime-cache-mla-kv-b-f32` plus
  `--metal-runtime-max-mla-kv-b-cache-mib N`; `/health` exposes the configured
  state and suggested launch-profile argv. When Metal runtime generation is
  enabled, `/health` now also recommends `--metal-runtime-mmap-final-logits`
  for the selective lm_head-only mmap path, distinct from the negative
  whole-resident `--wrap-resident-metal` experiment. If a validated context=1
  `o_proj*B_v` cache is configured, `/health` also carries
  `metal_runtime_context1_o_proj_cache_flags` inside `suggested_launch_profile`,
  so profile replay preserves the attention-output collapse instead of silently
  falling back to full `o_proj` streaming. Responses expose per-step MLA
  value-cache hit/store counts plus resident cache bytes. Validation on the
  existing prepared GLM package:
  `python3 metal/glm_moe_infer_generate_server_smoke.py --execute --requests 1
  --prompt-token-ids 0,1` reports `input_source="prompt_token_ids"`,
  `prompt_prefill.ok=1`, `first_step_from_input_logits=1`, generated ids
  `[30423,11093]`, memory decode-cache backend, no required debug files, and
  `estimated_live_working_set_bytes=542186512` under the 768MiB live cap. This
  is a correct minimum serving path, but still sequential per prompt token; the
  remaining performance gap is batching/overlap and Flash-MoE-style command
  scheduling, not more Python bridge work.
  A real prepared HTTP bridge smoke with the same persistent runtime and the
  opt-in MLA KV-B cache is also passing. The launch uses
  `--metal-runtime-generation --metal-runtime-cache-mla-kv-b-f32
  --metal-runtime-max-mla-kv-b-cache-mib 4608
  --max-cache-file-mib 16384 --max-cache-read-mib 1
  --max-live-working-set-mib 16384 --min-free-unified-memory-gib 24`; two
  sequential `/generate-token-ids` requests for prompt `[0]` generated `[15]`.
  The latest July 5, 2026 rerun, with the operator explicitly skipping the
  `danveloper/flash-moe` large-weight local trial, stored 78 MLA KV-B views on
  the first request in `2.114527s` decode time (`3.234041s` HTTP wall)
  and hit 78/78 views on the second request in `0.954184s` decode time
  (`1.497896s` HTTP wall). The cache-hit request therefore measured
  `1.048 tok/s` raw decode, `1.012 tok/s` including final logits, and
  `0.668 tok/s` through the HTTP wrapper on the current
  power-limited local machine. The hot path now encodes RoPE split, MLA
  attention, and attention output in one command buffer for the dense prefix,
  and also includes post-attention RMSNorm/router/top-k in that command buffer
  on all 75 MoE layers (`rope_mla_attn_output_norm_router_fused_count=75`), so
  the step reports `rope_mla_command_buffer_count=0`,
  `post_attn_norm_router_command_buffer_count=0`,
  `router_command_buffer_count=0`, and `post_attn_norm_command_buffer_count=0`.
  The attention projection outputs are now passed directly into that fused stage
  as Metal buffers (`rope_mla_input_buffer_direct_count=78`), attention output is
  passed directly as a Metal buffer inside the same fused stage
  (`attn_output_buffer_direct_count=75`), and the fused RMSNorm/router output is
  passed directly as a Metal buffer into MoE (`moe_mlp_input_buffer_direct_count=75`),
  removing the CPU/NSData handoffs from this part of the hot path. The layer
  output itself is now also handed to the next layer as a Metal buffer
  (`layer_input_buffer_direct_count=77`), so only the first token embedding and
  final layer output are materialized across the CPU boundary. Its main decode
  components include MLP `0.372377s`, expert read `0.166112s`, final
  logits `0.033497s`, and layer-boundary overhead `0.042428s`. The step now
  reports 78 layers
  (`3` dense plus `75` MoE), `12032409600` routed expert bytes, 600 expert read
  tasks, 8 persistent read workers, and a corrected all-stage 234 command
  buffers / 156 estimated synchronous waits. The smoke artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-direct-current-kva-latest.json`.
  A same-day wait audit then added per-layer attention-projection wait telemetry
  and reran the same guarded two-request smoke. The audit first showed that the
  previous cache-hit path generated `[15]` while still reporting 234 command
  buffers and 234 real synchronous waits: attention projection had to
  materialize KV-A on the CPU, append it to the decode cache, and let MLA read
  that same row back for the current token. `glm_moe_infer` now removes that
  context=1 CPU cache boundary. Fused MLA now consumes the current KV-A Metal
  buffer directly, then the runtime appends KV-A to the decode cache after the
  fused router wait so future tokens still see the row. For `contextLength>1`,
  it reads previous cache rows into a contiguous F32 view and blits the current
  KV-A row into the final slot inside the same command buffer before MLA. The
  guarded two-request HTTP smoke still generates `[15]`; on the cache-hit
  request it reports `0.954184s` decode time, `0.033632s` final logits,
  `1.012 tok/s` including final logits, `1.497896s` HTTP wall, 234 command
  buffers, and 156 real synchronous waits. This is the first measured
  Flash-MoE-shaped scheduling win past the 1 tok/s line on the local
  power-limited GLM-5.2 path. A guarded 2-token HTTP smoke validates the
  generalized path. A follow-up small-context MLA kernel now covers
  `contextLength=2..32` and avoids recomputing the same attention scores for
  every value dimension by assigning one threadgroup per head, computing the
  softmax once, and sharing it across value lanes. Prompt `[0]` still generated
  `[15,11]`; step 1 decode drops to `1.635305s`, and the aggregate report shows
  `468` command buffers / `312` waits with `0.511 tok/s` including logits for
  two generated tokens. The 3-token smoke validates the `contextLength=3` path:
  prompt `[0]` generated `[15,11,15]`, step 2 decode drops from the baseline
  `3.054386s` to `1.918706s`, and the aggregate report shows `702` command
  buffers / `468` waits with `0.515 tok/s` including logits for three generated
  tokens. A guarded 5-token comparison against the previous `contextLength<=4`
  small-kernel baseline generated the same `[15,11,15,15,21]`; aggregate
  throughput improved from `0.398 tok/s` to `0.467 tok/s` including logits, and
  the contextLength=5 step dropped from `4.400035s` decode / `3.732975s`
  attention to `2.576679s` decode / `1.994487s` attention. A 9-token comparison
  then exposed the next fallback: with the `contextLength<=8` limit, the
  contextLength=9 step took `7.066513s` decode / `6.425403s` attention; after
  extending the same small-context kernel to `contextLength<=32`, the identical
  `[15,11,15,15,21,15,15,15,15]` output took `3.897080s` decode / `3.316559s`
  attention on that step, and aggregate throughput improved from `0.325 tok/s`
  to `0.367 tok/s` including logits. This confirms the remaining long-context
  attention problem was repeated score computation, not SSD bandwidth.
  `glm_moe_infer` now routes `contextLength=2..32` through a small-context
  kernel and `contextLength>32` through a streaming kernel; both use one
  threadgroup per head and threadgroup-parallel qk/RoPE partials so each score
  is computed once. The streaming path applies an online softmax without
  allocating context-sized threadgroup storage. The probe-only `--mla-kv-b-f32` input and
  `metal/glm_moe_infer_mla_streaming_smoke.py` now validate that branch on tiny
  direct-F32 tensors at `contextLength=9`, `32`, and `33`, for both default and
  interleaved RoPE; the max absolute differences versus the Python reference
  stay below `8e-9`, with a `2460` byte live estimate at `contextLength=33`. The GLM-shaped
  direct-F32 microbench uses the real GLM-5.2 MLA dimensions (`64` heads,
  `kv_lora=512`, `qk_nope=192`, `rope=64`, `v=256`) without reading model
  weights. The small-context baseline measured `0.041593s` at
  `contextLength=9` and `0.140647s` at `contextLength=32`; after parallelizing
  the small-context scores, those medians drop to `0.005918s` and `0.006985s`
  respectively, while the streaming branch remains about `0.008656s` at
  `contextLength=33` and `0.010446s` at `contextLength=64`, with about `59MiB`
  estimated live working set. The real GLM guarded 9-token smoke now generates
  the same `[15,11,15,15,21,15,15,15,15]` as the previous smallctx32 run while
  improving throughput from `0.367 tok/s` to `0.872 tok/s` including logits;
  total attention time drops from `17.847006s` to `3.747658s`, and the
  contextLength=9 step drops from `3.897080s` decode / `3.316559s` attention to
  `0.995520s` decode / `0.420962s` attention. A guarded 17-token run stays in
  the small-context path, reads `190.503GiB` of routed experts at `64.487GiB/s`,
  and reports `0.908 tok/s` including logits with the last step at `1.038267s`
  decode / `0.452894s` attention. A guarded 33-token run reaches the streaming
  branch at `contextLength=33`, reads `369.800GiB` of routed experts at
  `64.844GiB/s`, and reports `0.889 tok/s` including logits. Total attention
  time is now `15.759542s`; the first streaming step reports `1.213384s`
  decode / `0.624476s` attention, so the streaming MLA path is no longer a
  catastrophic fallback, but attention output remains the largest aggregate
  bottleneck at `43.8%` of decode time. `scripts/decode_telemetry_report.py`
  now labels this fused timing as
  `rope_mla_attn_output_norm_router_fused` instead of plain `attn_output` when
  the command buffer includes RoPE, MLA, attention output, norm, and router.
  The first GLM version of Flash-MoE's deferred CMD3 scheduling is also in
  place for decode-layer MoE blocks: a non-final MoE layer can commit its single
  expert command buffer without waiting, return the output `MTLBuffer`, and let
  the next layer's attention command on the same Metal queue provide the
  synchronization point. Standalone MoE probes and final layers still wait for
  CPU readback. A real three-layer oracle (`layers=0,3,4`, dense layer `0`)
  matches the old runner within `1.430511e-6` hidden max diff and
  `1.144409e-5` top-k max diff, while reporting MoE waits `[0,1]`; the tiny
  standalone MoE smoke still reports `async_submitted=0` and
  `synchronous_wait_count=1`. A guarded direct CLI 9-token rerun after the
  telemetry propagation generated the same
  `[15,11,15,15,21,15,15,15,15]` token sequence under a 16GiB live cap and a
  24GiB free-unified-memory guard. The latest rerun reports `10.326609s`
  summed decode, `0.454387s` summed final logits, `0.835 tok/s` including
  logits, `100.854GiB` routed expert reads in `1.535937s` (`65.663GiB/s`),
  `34.963GiB` resident attention-output weight reads in `1.692658s`, `1.978GiB`
  router reads in `0.098004s`, `5375752720` estimated live bytes, and
  admission/memory checks both true. The end-to-end MoE command wait count is
  now `9 / 675`, so deferred MoE scheduling is active in the real generation
  path; the remaining primary bottleneck is still the fused
  `rope_mla_attn_output_norm_router_fused` stage at `46.4%` of decode time,
  with `3.096497s` inside the fused attention-output projection command.
  `scripts/decode_telemetry_report.py` now also emits an
  `optimization_frontier` block: on the current 9-token artifact,
  attention-output read+projection is `0.453s/token` versus routed expert read
  at `0.164s/token`, with `234` command buffers and `82` estimated sync waits
  per token. Detailed `probe_decode_layers` payloads now also get a
  `layer_frontier` with the hottest per-layer attention-output, overhead,
  command-buffer, and wait entries, while decode-layer aggregates expose
  attention-projection sync waits versus async submissions and
  `sync_waits_by_stage`. This keeps the next runtime work aimed at
  attention-output fusion and command-buffer/wait reduction before more
  expert-I/O tuning.
  A follow-up runtime change lets non-final dense MLP layers submit their fused
  RMSNorm/gate/up/SwiGLU/down/residual command buffer without an immediate host
  wait and pass the output `MTLBuffer` directly to the next layer on the same
  command queue. Standalone and final dense probes keep synchronous readback.
  The guarded two-layer real smoke
  `metal/glm_moe_infer_real_decode_layers_smoke.py --new-only --layers 0,3
  --dense-layers 0 --max-live-working-set-mib 2048` reports
  `dense_mlp_async_submitted=[1]`, `dense_mlp_synchronous_waits=[0]`, and the
  same generated token `140366`. This trims dense-prefix waits; the larger
  rewrite pressure remains the MoE attention-output/router boundary.
  The large-weight Flash-MOe trial is intentionally skipped for now because the
  user does not have time to download another huge model. The rewrite decision
  therefore moved to a smaller but directly relevant overlap test on the
  existing GLM artifact: in fully fused MoE decode layers, the runtime commits
  RoPE/MLA/attention-output/RMSNorm/router/top-k, then preads the fixed shared
  expert into its existing `top_k` staging slot while the host waits for router
  completion. This does not make routed expert I/O speculative; routed expert
  reads still depend on GPU top-k ids. It also does not allocate a new large
  buffer, because the prefetch only activates when the current reusable expert
  buffer set already has `top_k + shared` slots. The guarded two-layer smoke
  now asserts `shared_prefetch_used` on the MoE layer and aggregate
  `shared_prefetch_used_count=1`. The telemetry fields `shared_bytes_read`,
  `shared_read_seconds`, `shared_prefetch_seconds`, and
  `shared_prefetch_used_count` now propagate through raw Metal JSON, Python
  generation results, server step payloads, CLI summaries, and
  `scripts/decode_telemetry_report.py`.
  A follow-up resident-Metal experiment wired `--wrap-resident-metal` through
  the continuous MoE decode path and validated it with
  `metal/glm_moe_infer_real_decode_layers_smoke.py --new-only --layers 3,4
  --dense-layers '' --wrap-resident-metal`. The corrected path maps
  `resident.bin` as one Metal buffer and uses offsets for `o_proj`, router, and
  router correction bias; post-attention RMSNorm still uses the small
  BF16-to-F32 staging path because the existing RMSNorm kernel consumes F32.
  The smoke passed and reduced explicit resident reads to
  `attn_output_bytes_read=0`, `router_bytes_read=0`, and
  `router_correction_bias_bytes_read=0`, but two-layer attention-output kernel
  time regressed from about `0.016889s` on the staged default path to a stable
  `0.545517s` when the GPU read the file-backed mmap directly. This is a clear
  negative result for whole-file mmap-as-Metal-buffer as a hot-path strategy:
  keep it as an opt-in diagnostic, but keep the production direction on
  Flash-MoE-style bounded staging buffers, overlap, and command scheduling.
  The diagnostic path now has tighter tiny coverage: standalone fused
  attention-output, dense decoder fallback, and `--probe-decode-layers
  --skip-debug-intermediates` hot paths all verify identical output with
  `attn_output_resident_mmap_backed=1` and `attn_output_bytes_read=0`.
  `attn_output_resident_mmap_backed_count` is propagated through raw runtime
  JSON, Python generation results, server step payloads, CLI summaries, and
  `scripts/decode_telemetry_report.py` so future real artifacts can prove
  whether a layer used the diagnostic resident-backed path.
  The follow-up staged-path experiment removed redundant zero-fill of fully
  overwritten 2MiB-aligned staging buffers and dispatches `o_proj.weight`,
  `o_proj.scales`, router, and router correction-bias resident reads through
  the existing persistent parallel `pread` pool. This keeps the same live-memory
  envelope and generated the same `[15,11,15,15,21,15,15,15,15]` under the
  16GiB live cap and 24GiB free-unified-memory guard. The saved artifact
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-parallel-staging-9tok-latest.json`
  reports `9.815746s` summed decode, `0.423321s` summed final logits,
  `0.879 tok/s` including logits, `100.854GiB` routed expert reads in
  `1.527887s`, `34.963GiB` attention-output resident reads in `1.570976s`, and
  `1.978GiB` router reads in `0.087877s`. The primary bottleneck remains
  `rope_mla_attn_output_norm_router_fused` at `4.542971s` (`46.3%`) with
  `2.971820s` inside the attention-output projection command, so this is useful
  cleanup rather than the final Flash-MoE-class overlap.
  The next kernel-level cleanup made `fused_matvec_add` literal by adding the
  residual inside the MXFP4 attention-output matvec kernel instead of launching
  a separate F32 add kernel in the same command buffer. The guarded artifact
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-parallel-staging-matvecadd-9tok-latest.json`
  reports `9.296515s` summed decode, `0.389798s` summed final logits,
  `0.929 tok/s` including logits, `100.854GiB` routed expert reads in
  `1.474182s`, `34.963GiB` attention-output resident reads in `1.564848s`, and
  `1.978GiB` router reads in `0.087628s`. The fused
  RoPE/MLA/attention-output/norm/router stage dropped to `4.072766s` (`43.8%`),
  with `2.507744s` inside the attention-output command-buffer timing.
  The requested flash-moe trial was later skipped because downloading/running
  the large Qwen weights locally was not acceptable at the time, but the
  `danveloper/flash-moe` source was reviewed directly. Its core lessons line up
  with our local evidence: use bounded 2MiB-aligned staging buffers, parallel
  `pread()`, the OS page cache, and deferred GPU expert command submission;
  avoid whole-file mmap as a hot Metal source and avoid custom expert caches
  that shrink page-cache headroom. The apparent speed gap is also structural:
  flash-moe reports a Qwen MoE path that streams on the order of hundreds of
  MiB per token, while this GLM-5.2 MXFP4 path currently streams about
  `11.2GiB/token` of routed experts, `3.9GiB/token` of attention-output
  weights, `0.22GiB/token` router weights, and `0.34GiB/token` dense MLP
  weights. That makes flash-moe-level `4+ tok/s` unreachable by SSD bandwidth
  improvements alone unless the GLM-specific per-token weight traffic is
  reduced.
  A shared-input tiled MXFP4 attention-output kernel was prototyped as the most
  direct flash-moe-kernel carryover. It passed the real two-layer oracle smoke
  (`output.max_abs_diff=6.7e-08`, `topk.max_abs_diff=6.2e-06`) but did not
  improve the full guarded 9-token run. With
  `LARGERLM_GLM_MOE_INFER_TILED_ATTN_OUTPUT_MXFP4=1` and a 2048-float tile,
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-tiled-matvecadd-9tok-latest.json`
  reports `9.263535s` decode, `0.411329s` final logits, `0.930 tok/s`
  including logits, and `2.522796s` attention-output command-buffer timing,
  essentially neutral against the default `2.507744s` timing. A 4096-float tile
  in
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-tiled4096-matvecadd-9tok-latest.json`
  regressed to `0.876 tok/s` and `3.034180s` attention-output command timing,
  likely from lower occupancy/shared-memory pressure. The tiled kernel remains
  available only as an opt-in experiment; production defaults stay on staged
  SIMD matvec-add.
  The next viable rewrite is therefore GLM-specific rather than a tiled-kernel
  transplant. For context=1 decode, MLA's current value path can be rewritten
  from `o_proj(B_v * latent)` into `(o_proj * B_v) * latent`, so the hot path no
  longer streams the full attention-output matrix each token. The read-only
  planner `scripts/glm_context1_o_proj_collapse_plan.py` confirms this against
  the current GLM-5.2 MXFP4 artifact without loading large payloads:
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-collapse-plan-latest.json`
  reports 78/78 collapsible layers, `3978.0 MiB/token` current `o_proj`
  storage reads, a `468.0 MiB` total BF16 collapsed cache (`936.0 MiB` F32),
  and an observed cache/read ratio of `0.118`. The cost is a one-time
  offline/resumable `4.02T FMA` build. The cache schema and guarded builder
  entry point now exist as `largerlm.context1_o_proj_cache` and
  `python -m largerlm context1-o-proj-cache`; default dry-run mode writes no
  cache and the real artifact report
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-bv-cache-dry-run-latest.json`
  confirms `490733568` bytes and `4020089389056` FMA. The reference backend is
  intentionally capped and refuses the full build under `--execute`. A
  companion `validate-context1-o-proj-cache` gate now checks schema, backing
  file size, non-overlapping tensor spans, numeric layer order, dtype/dim
  consistency, optional prepared-artifact compatibility, and complete builder
  progress by default. `--allow-incomplete-progress` is reserved for reporting
  partial chunk-build status, not for runtime admission. The Metal side now has
  standalone one-layer builder and consumer probes:
  `--build-context1-o-proj-cache-layer` multiplies resident MXFP4 `o_proj` by
  resident MXFP4 `unembed_out` into a selected BF16/F32 collapsed-cache span
  under an explicit GFMA cap, and `--probe-context1-o-proj-cache-output` reads
  only the selected collapsed-cache layer for BF16/F32 matvec+residual. The path
  is covered by `metal/context1_o_proj_cache_build_smoke.py` and
  `metal/context1_o_proj_cache_output_smoke.py`. The Python builder CLI now has
  a `--backend metal` mode that calls `metal/glm_moe_infer` one layer at a time,
  updates the same progress file, and is covered by
  `metal/context1_o_proj_cache_cli_metal_backend_smoke.py`. It also supports
  `--build-layers 0,7-9`, which executes only a subset while keeping the full
  cache layout/progress intact for bounded resumable real-GLM builds.
  `--build-next-layers N` is the safer resumable form because it selects the
  next incomplete layers from the progress file, or the first `N` layers when
  no progress exists, and still only executes under `--execute`. The real GLM
  metal dry-run/build reports now also include a `selected_build` block, so
  a one-layer or chunked invocation records the exact requested layers,
  cache/source bytes, total FMA, max per-layer FMA, and minimum
  `--max-build-fma` before any payload-heavy work starts. The CLI also accepts
  `--max-build-gfma` as the friendlier form for real GLM chunks, e.g. roughly
  `51.54` GFMA for one current GLM-5.2 layer. The same reports now include
  `disk_budget`; `--execute` refuses to create or extend the cache when
  `total_bytes + --disk-margin-gib` would exceed free space, and the CLI keeps a
  16GiB free-disk margin by default. The Metal builder path now also reports
  `max_estimated_metal_builder_live_bytes`, Python refuses
  `--backend metal --execute` above `--max-metal-builder-live-mib`, and the
  standalone `metal/glm_moe_infer --build-context1-o-proj-cache-layer` enforces
  `--max-live-working-set-mib` before allocating Metal buffers. The tiny Metal
  smoke now asserts the accepted path and a deliberately rejected low-live-cap
  path, and the Python CLI Metal smoke asserts the reported builder live
  estimate/cap. Executed builds now persist per-layer `layer_results` in both
  the build report and progress file, carrying source/cache bytes, FMA,
  read/kernel/write timing, live estimate, and device name for Metal layers.
  `layer_result_summary` rolls those measurements into GFMA/s, completed FMA
  fraction, and estimated full/remaining build seconds so a real one-layer GLM
  build can immediately decide whether the full offline cache build is
  practical. `validate-context1-o-proj-cache --allow-incomplete-progress` now
  exposes the same progress summary plus `next_missing_layer` and
  `missing_layer_count`, so interrupted real builds can be resumed with
  `context1-o-proj-cache --backend metal --build-next-layers N` without
  manually inspecting progress JSON. Incomplete progress also carries
  `suggested_resume_build`, a one-next-layer dry-run/execute argv pair with the
  minimum `--max-build-gfma`; it remains explicit and still flows through the
  normal disk/FMA/live-memory guards. The real GLM
  metal dry-run artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-bv-cache-metal-dry-run-latest.json`;
  full or single-layer execution still requires intentionally raising
  `--max-build-fma` or `--max-build-gfma`, and one real layer is refused by the
  default cap because it requires `51,539,607,552` FMA, about `51.54` GFMA. The
  first runtime consumer has also landed in
  `glm_moe_infer`: `--context1-o-proj-cache-layout` lets the `context_length=1`
  dense and MoE decoder-layer paths replace MLA value projection plus resident
  `o_proj` with collapsed-cache matvec from the current KV-A latent. It is
  covered by `metal/glm_moe_infer_context1_dense_decoder_smoke.py`,
  `metal/glm_moe_infer_context1_moe_decoder_smoke.py`, and
  `metal/glm_moe_infer_context1_decode_layers_smoke.py`, which compare against
  tiny baselines and assert `attn_output_context1_o_proj_cache` telemetry. The
  Python Metal generation payload and prepared HTTP runtime now carry the same
  opt-in through `context1_o_proj_cache_layout` /
  `--metal-runtime-context1-o-proj-cache-layout`; prepared HTTP startup validates
  the context1 layout, backing file size, prepared-artifact dims/config match,
  and any progress-file completion before serving requests, then reports the
  validated summary in `/health`.
  Generation responses and `scripts/decode_telemetry_report.py` now propagate
  `attn_output_context1_o_proj_cache_count` so real benchmark artifacts can prove
  that the collapsed-cache branch was actually used.
  `scripts/glm_metal_viability_report.py --decode-telemetry` now folds the
  collapse plan into existing decode telemetry: context=1-only is expected to
  be a small aggregate win on multi-token decode. The new
  `scripts/glm_latent_value_collapse_plan.py` checks the exact all-context
  extension and rejects the naive per-head cache shape for GLM-5.2: 64
  independent attention heads exceed the 8 break-even shared-head groups, BF16
  per-head cache reads would be `29.250 GiB/token` (`7.53x` current `o_proj`),
  and even an ideal int4 floor would still read `7.312 GiB/token` (`1.88x`).
  The next step is therefore a guarded real context=1 build/measurement plus
  continued Flash-MoE-shaped fusion/scheduling work, not a general per-head
  collapsed-cache rewrite.
  A guarded 1-token runtime smoke also compiled the updated Metal library and
  still generated `[15]`. The direct CLI deferred-MoE artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-deferred-moe-9tok-latest.json`;
  the
  real 33-token artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-small-parallel-33tok-latest.json`;
  real 17-token artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-small-parallel-17tok-latest.json`;
  the real 9-token small-parallel artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-small-parallel-9tok-latest.json`;
  the optimized GLM-shaped MLA microbench artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-shaped-mla-small-parallel-latest.json`;
  the small-vs-streaming baseline artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-shaped-mla-small-vs-streaming-baseline-latest.json`;
  the runtime compile smoke artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-streaming-library-compile-1tok-latest.json`;
  the 9-token smallctx32 artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-smallctx32-9tok-latest.json`;
  the 9-token smallctx8 baseline artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-smallctx8-9tok-baseline-latest.json`;
  the 5-token smallctx8 artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-smallctx8-5tok-latest.json`;
  the 5-token smallctx4 baseline artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-smallctx4-5tok-baseline-latest.json`;
  the 3-token artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-smallctx-mla-fast-3tok-latest.json`;
  the 2-token artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-context2-mla-fast-latest.json`;
  the 1-token artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-direct-current-kva-latest.json`.
- The prepared HTTP server can now opt in to the same runtime generation path
  without replacing the mature default Python batch-prefill server. A new
  `PreparedServerConfig.metal_runtime_generation` flag and
  `serve-prepared --metal-runtime-generation --metal-binary ...` route
  `/generate-token-ids` and `/generate-text` through `generate_metal_token_ids`
  / `generate_metal_text` after the normal request check, memory preflight, and
  prepared run lock. The server lazily starts one persistent
  `glm_moe_infer --generate-server-jsonl` session per prepared app and reuses it
  across HTTP requests; health reports `metal_runtime_session_started` and
  `metal_runtime_session_request_count`. If a request trips a
  `MetalGenerateError`, the prepared app closes and clears that session so the
  next request starts a fresh runtime instead of reusing a poisoned pipe or
  crashed child process. Responses report
  `runtime="glm_moe_infer"` plus Metal admission/live-envelope telemetry. The
  mode is deliberately greedy-only and rejects non-zero temperature,
  non-default top-p, and seed requests until the C/Metal runtime implements
  matching sampling semantics. It also refuses to claim prefill-acceleration
  requirements because runtime prompt prefill is still sequential. The
  `MetalTokenGenerationResult`, CLI summary, and HTTP response step payloads
  now carry per-step decode telemetry from `glm_moe_infer`: layer counts,
  routed expert bytes, dense-MLP resident bytes, expert-read seconds, MoE
  kernel seconds, MoE output/write overhead, pooled/serial expert-read dispatch
  counts, command-buffer counts, the current synchronous-wait estimate, and the
  explicit MoE command wait count. `scripts/decode_telemetry_report.py` now
  also accepts flat `generate-metal-token-ids` artifacts in addition to nested
  HTTP/probe step payloads, and only prints MoE wait counts when the source
  artifact actually carried that field.
- The same persistent runtime is now exposed as an explicit Python text API for
  non-HTTP harnesses. `MetalTextGenerationSession` loads the tokenizer once,
  starts one `glm_moe_infer --generate-server-jsonl` child, and forwards that
  `generate_server_session` into every `session.generate(...)` call. The
  one-shot `generate_metal_text(...)` API also has an explicit
  `generate_server_session` parameter instead of relying on hidden `**kwargs`
  passthrough. `generate_metal_text_batch(...)` and the
  `generate-metal-text-batch --prompts-jsonl FILE` CLI now use the same session
  for a sequence of prompt objects (`{"prompt": "..."}` per JSONL line), and
  report total prompt tokens, generated tokens, runtime startup, elapsed time,
  aggregate generated-token throughput, max live envelope, max prefill
  envelope, min available memory, max required memory, and admission/memory-ok
  summaries across requests. This does not solve GLM's per-token I/O floor or
  make prompt prefill batched, but it prevents benchmark and local scripting
  code from accidentally returning to per-request runtime launches while the
  C/Metal hot path is being optimized.
- The July 5, 2026 source-only recheck of `danveloper/flash-moe` confirms the
  rewrite direction rather than a need to return to the old Python pipeline.
  The local large-weight trial was intentionally skipped at operator request,
  but the source and paper are enough to explain the speed gap. Flash-MoE's
  speed comes from a smaller active GLM-equivalent workload (Qwen hidden 4096,
  60 layers, K=4, roughly 7.08MB per 4-bit expert slot) and a tightly fused
  single-process engine: resident weights mmap'd once, selected experts read by
  K-way `pread` into 2MB-aligned `newBufferWithBytesNoCopy` Metal shared
  buffers, no application expert cache, GPU-side combine/residual/norm, and a
  deferred CMD3 expert command buffer. The local GLM target is substantially
  heavier (hidden 6144, 78 layers with 75 MoE layers, K=8, roughly 20.05MB per
  routed expert slot). The static envelope makes that difference mechanical:
  GLM reads `12032409600` routed expert bytes/token versus Flash-MoE's
  documented `1698693120`, a `7.08x` ratio before any attention or logits work.
  At Flash-MoE's reported 4-bit `4.36 tok/s`, byte-normalized GLM would land
  around `0.616 tok/s`; the current HTTP wrapper measured `0.668 tok/s`, and
  the raw cache-hit decode measured `1.048 tok/s` (`1.012 tok/s` including
  final logits). The latest direct CLI deferred-MoE run lands at `0.835 tok/s`
  including logits with MoE waits reduced to `9 / 675`, so the current
  direction is not methodologically broken, but the GLM-5.2 target should be
  treated as roughly 0.6-1.0 tok/s-class until GLM-specific command scheduling,
  attention/output projection fusion, and possibly lower-bit expert I/O reduce
  the per-token work. A Flash-MoE-shaped GLM runtime should improve the current
  path but should not be expected to match Flash-MoE's Qwen tokens/second on
  equal hardware. The remaining engineering should therefore focus on the
  missing Flash-MoE-shaped pieces inside `glm_moe_infer`: persistent serving,
  prompt prefill inside the same process, deeper deferred command-buffer
  scheduling beyond the MoE MLP block, and keeping the next-layer
  hidden/RMSNorm path resident on GPU where it is safe for GLM's MLA/cache
  structure.
- What remains for minimum runnable token-id decode is no longer a single-layer
  semantic gap, an all-layer correctness gap, or the service interface itself:
  the MoE layer path, dense-prefix path, mixed dense-to-MoE layer-list path,
  full 78-layer single-token decode, and the JSONL two-token generation service
  all have real-weight coverage. The current minimum serving loop is runnable
  for token-id input, text input, and explicit multi-token `--prefill-prompt`
  input with memory guards and default in-memory decode cache, including an
  opt-in prepared HTTP server bridge. What remains before useful
  end-to-end GLM text generation is not another bridge: it is making prompt
  prefill batched/overlapped, reducing the fused MLA/attention-output/router
  stage, and deepening Flash-MoE-style deferred command-buffer scheduling where
  GLM's MLA/cache structure allows it.
