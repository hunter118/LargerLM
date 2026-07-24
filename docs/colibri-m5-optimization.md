# Colibri M5 Max Optimization

Date: 2026-07-24

## Scope

This document records a feasibility result for GLM-5.2 Colibri int4 on one
Apple M5 Max with 128 GB unified memory. It is not a general performance claim
and it does not make the experimental route quality-equivalent to GLM-5.2.

The tested model package is
`mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`, approximately 357 GiB on disk.
The machine was running with a power-adapter limitation, so the results are
useful engineering measurements rather than a peak hardware benchmark.

## Architecture

GLM-5.2 has 78 layers, 75 routed MoE layers, 256 experts per routed layer, and
selects 8 experts per token. Keeping the whole model in 128 GB is impossible.
The working runtime therefore has four tiers:

1. Dense weights and active tensors stay in unified memory.
2. A learned 46 GiB expert set is locked in physical memory.
3. A bounded per-layer LRU retains roughly another 46 GiB of experts.
4. Remaining experts are read from the internal SSD on demand.

The measured process RSS is about 97 GiB. This is why the runtime now gives far
more than 80 GiB to expert residency without approaching the 128 GB failure
edge: about 92 GiB is expert data, while dense weights, KV state, activations,
Metal resources, macOS, and a pressure reserve occupy the rest.

The included frequency profile contains 259,200 routed selections. It is fixed
during benchmark runs so A/B tests use the same initial placement.

## Quality-Preserving Path

The default `quality` mode does not alter router choices. Its main M5
optimization is a Metal residency set:

- resident expert buffers are registered once;
- pending additions are committed to an `MTLResidencySet`;
- every MoE command buffer references that set;
- thousands of per-submit `useResource:` declarations are avoided.

On the same warm expert profile, this changed 64-token decode from about
`1.82 tok/s` to `2.74 tok/s`. A 256-token stability run sustained
`2.69 tok/s`, with:

- `73.5%` expert hit rate;
- `96.95 GiB` peak RSS;
- `15%` minimum macOS pressure-free reading;
- zero Metal fallbacks.

This is the best validated output-preserving configuration. Static held-out
analysis explains its ceiling: 4,906 resident expert slots cover only `80.53%`
of the held-out route oracle, while a 5 tok/s quality path would need roughly
`90-92%` hit rate. More I/O scheduling alone cannot close that gap.

## Experimental Fast Path

The optional fast mode changes routing to prefer experts already resident in
memory. For each layer and token, its Metal selector:

1. computes the exact router top 32;
2. always keeps the true top 2 experts;
3. fills the remaining six slots from resident experts within that top 32;
4. falls back to normal rank order if fewer than eight choices were found;
5. separately returns the original top 8 for fidelity telemetry.

The selector is one 32-lane SIMD threadgroup. A CPU reference test validates
`M=12` and `M=32`, and a real CPU/GPU A/B produced exactly the same 64 token IDs
and route telemetry. Moving selection from CPU to Metal reduced router time from
`1.515s` to `0.006s`.

The 64-token result for `J=2, M=32` was:

| Metric | Result |
| --- | ---: |
| Full decode | `4.92 tok/s` |
| First 32 tokens | `5.21 tok/s` |
| Expert hit rate | `93.6%` |
| Expert traffic | `711.7 MiB/token` |
| Changed route slots | `34.9%` |
| Top-8 overlap | `65.1%` |
| Route KL diagnostic | `6.3676` |
| Peak RSS | `96.31 GiB` |
| Metal fallback | `0` |

An `M=40` test reached `4.96 tok/s`, but reduced top-8 overlap to `63.5%`.
That is too little speed gain for the additional route distortion, so it is not
the recommended preset.

The generated Chinese sample remained coherent, but one prompt is not a quality
evaluation. Fast mode must be treated as an approximate model variant.

### Persistent Web Decode

Colibri's OpenAI-compatible server normally sends even one active request
through its ragged multi-slot decode function. That path cannot use the
contiguous full-layer Metal kernel, so a single-user browser session was slower
than the command-line runner.

The included patch specializes the `one configured slot, one active request`
case. It binds that slot's KV cache and uses the same contiguous decode step as
the command-line path. Multi-user and multi-slot requests retain Colibri's
original ragged batching behavior.

A real 26-token Chinese prompt followed by 64 generated tokens measured:

| Metric | Result |
| --- | ---: |
| Decode throughput | `5.31 tok/s` |
| Decode wall time | `12.063 s` |
| UI end-to-end rate | `3.6 tok/s` |
| Time to first token | `5.6 s` |
| Expert disk service | `6.009 s` |
| Expert I/O wait | `4.219 s` |
| Expert matmul | `3.714 s` |
| Attention | `3.534 s` |

The UI rate includes prompt prefill and first-token latency; it is not directly
comparable to decode-only measurements. The server is intentionally configured
with one KV slot and one queued request because this machine is optimized for
one interactive user and because additional simultaneous contexts increase
memory risk.

## M5 Neural Acceleration

The repository separately validates MPP TensorOps and can use the system MPP
path for eligible large dense F32/BF16/F16 prefill matrices. This is useful for
long-prompt prefill.

The routed expert weights are int4, so the decode hot path still uses custom
Metal quantized kernels. The new cache selector is a Metal SIMD kernel, not an
Apple Neural Engine call. No public API currently lets this project send the
whole quantized GLM decode graph directly to the ANE. The implementation avoids
claiming otherwise.

## Rejected Optimizations

The following ideas were measured and removed:

- Metal I/O command queues: cold shared-buffer reads reached about
  `12.7-13.0 GB/s`, below an eight-worker `F_NOCACHE pread` result of
  `14.1 GB/s`.
- Splitting each 19 MiB expert into two reads: real decode was unchanged at
  `2.76 tok/s`.
- A larger route horizon (`M=40`): negligible gain and greater semantic drift.
- Static residency alone: the held-out coverage ceiling is insufficient for
  output-preserving 5 tok/s.

These negative results matter because they keep the active path small and
prevent page-cache benchmark artifacts from being mistaken for SSD throughput.

## Build

The setup script clones Colibri at commit
`81f08a09e5651ce52616dc720f68810f9021c0be`, applies
`patches/colibri-m5-cache-route.patch`, builds with `METAL=1 ARCH=native`, and
runs the Metal tests:

```bash
./scripts/setup_colibri_m5.sh
```

To use another checkout:

```bash
./scripts/setup_colibri_m5.sh /path/to/new/colibri
```

The target path must not already exist. This avoids silently patching an unknown
source revision.

## Model

Download the Colibri-specific package:

```bash
hf download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp \
  --local-dir artifacts/colibri-glm5.2-int4
```

This is not GGUF, AWQ, GPTQ, or MLX format. The weights are excluded from Git.
Keep about 400 GB free and use the internal SSD.

## Usage

Default quality-preserving mode:

```bash
.venv/bin/python scripts/run_colibri_m5.py \
  "Write a short proof that sqrt(2) is irrational." \
  --mode quality --ngen 128 --profile
```

Experimental fast mode:

```bash
.venv/bin/python scripts/run_colibri_m5.py \
  "Write a short proof that sqrt(2) is irrational." \
  --mode experimental-fast --ngen 128 --profile
```

### Browser Chat

Build the official Colibri React interface after the main setup:

```bash
./scripts/setup_colibri_web.sh
```

Start a persistent, guarded model process:

```bash
.venv/bin/python scripts/run_colibri_m5.py \
  --web --detach --mode experimental-fast --ngen 512 \
  --profile --preserve-usage --sample-seconds 1
```

Open `http://127.0.0.1:8000`. The launcher only accepts loopback addresses, sets
one KV slot and one request queue slot, and keeps the model loaded across
questions. The browser's maximum-output control limits each answer; `128` is a
reasonable interactive starting point.

Stop the complete guarded process group and release about 97 GiB:

```bash
.venv/bin/python scripts/run_colibri_m5.py --stop-web
```

The detached monitor writes `runtime/colibri-web.log` and
`runtime/colibri-web.pid`. `runtime/` is excluded from Git.

Useful options:

- `--usage-profile PATH` selects a learned placement snapshot.
- `--preserve-usage` restores the model's `.coli_usage` after the run.
- `--capture-usage PATH` saves the run's final usage counters.
- `--ram-gib` may lower, but not raise, the 110 GiB Colibri budget.
- `--pin-gib` may lower, but not raise, the 46 GiB locked tier.
- `--max-rss-gib` may lower the 105 GiB emergency stop.
- `--min-pressure-free-percent` may raise the default 10% stop threshold.

The launcher also requires 24 GiB of reclaimable memory before starting. It
samples the entire Colibri process group, sends an interrupt on a guard breach,
then escalates to termination only if needed. This cannot make all macOS memory
failures impossible, but it is materially safer than an unbounded GUI launch.
The same monitor owns the detached Web child process, handles termination
signals, and refuses to stop a PID whose command line does not match the
guarded launcher.

## Verification

Python safety and policy tests:

```bash
.venv/bin/python -m pytest -q tests/test_colibri_m5.py \
  tests/test_colibri_usage_analysis.py
```

Colibri CPU and Metal tests:

```bash
make -C third_party/colibri/c test-c
make -C third_party/colibri/c metal-test
```

For comparable measurements, keep the prompt, generation length, temperature,
power condition, and usage profile fixed. Report the full-window decode rate,
not only the first 16-token line.

## Limits

- The exact quality path remains below 5 tok/s.
- The approximately 5 tok/s path changes expert routing and needs real
  evaluation before practical use.
- Prompt prefill and decode have different bottlenecks.
- SSD temperature, free space, background I/O, macOS cache state, and adapter
  power can materially change results.
- This work validates feasibility on one M5 Max 128 GB machine only.
