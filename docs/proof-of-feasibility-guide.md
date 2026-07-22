# LargerLM Feasibility Guide

LargerLM is a proof-of-feasibility project for running GLM-style MoE models on
Apple Silicon when the full model is larger than unified memory. It is not a
production inference engine and it is not a high-throughput serving stack. The
original local GLM-5.2 MXFP4 route was sealed as a minimum runnable version:
bounded generation works, but the measured and projected speed is below the
`5 tok/s` continuation target. Performance exploration resumed on 2026-07-22
for learned hot-expert residency and M5 prefill acceleration.

The useful result is the engineering map: how to split a large MoE checkpoint
into resident tensors and SSD-backed routed experts, how to keep memory bounded,
which Metal paths are viable, and where the GLM-5.2 layout stops being efficient.

## Device Requirements

Recommended test machine:

- Apple Silicon Mac with 128 GiB unified memory.
- Fast internal SSD, preferably Apple M-series Max/Ultra class.
- macOS with Xcode command line tools and Metal support.
- Hundreds of GB of free disk for GLM-5.2 MXFP4 weights and prepared artifacts.

Practical notes:

- The repository does not include model weights or prepared artifacts.
- Local artifacts are intentionally ignored by git under `artifacts/`.
- Use the guarded commands in this document. Do not run old uncapped experiments
  on a machine you care about; memory pressure can make macOS unstable.

## Current Status

The current GLM-5.2 MXFP4 evidence says:

- Real decode artifacts are around `0.9-1.0 tok/s`.
- The best evidence-backed context=1 projection is `1.485 tok/s`.
- The `5 tok/s` target is still `3.37x` away.
- Routed expert traffic is about `11.206 GiB/token`; `5 tok/s` would need about
  `56 GiB/s` of useful routed-expert reads before attention, kernels, logits, or
  scheduling overhead.

These numbers remain the baseline for the original all-streaming route. The
reopened work must beat them by reducing actual SSD misses, not by projecting
faster storage. See `docs/colibri-hot-expert-notes.md` for the active direction
and `docs/minimal-usable-seal.md` for the historical stop decision.

## Principle

Large MoE models are not uniformly expensive at decode time. A token only uses a
small number of routed experts per MoE layer, but the model still contains many
expert weights that cannot all fit comfortably in unified memory. LargerLM uses
that structure:

1. Scan checkpoint headers without loading the full tensors.
2. Classify tensors into resident weights and routed expert weights.
3. Pack resident tensors into a memory-mappable layout.
4. Pack routed experts into per-layer expert slots on SSD.
5. During decode, keep the resident path and small live buffers in memory while
   streaming only selected expert slots from disk.
6. Enforce live-memory, free-memory, disk, and read-volume guards before running
   real-weight work.

The runtime experiments follow a Flash-MOE-shaped pattern: keep orchestration
bounded, use persistent C/Metal processes where possible, avoid huge Python
object graphs, and prefer explicit staged reads over accidental whole-file
materialization.

## Runtime Pieces

The Python package provides planning, artifact inspection, packing, safety gates,
and HTTP/CLI wrappers:

- `largerlm.prepare` and related CLI paths build prepared artifacts.
- `largerlm.server` exposes guarded local serving.
- `largerlm.metal_generate` bridges prepared artifacts to the Metal runtime.
- `largerlm.metal_viability` summarizes whether the current route is worth
  further optimization.
- `largerlm.context1_o_proj_cache` handles the narrow context=1 collapsed
  attention-output cache.

The `metal/` directory contains Objective-C/Metal sources and smoke tests for
the low-level kernels and runtime probes. Built binaries are local artifacts and
are not committed.

## Safety Model

The important safety rule is that every real-weight path should be bounded
before it opens large files or allocates large buffers.

The recurring guard knobs are:

- `--max-live-working-set-mib`
- `--min-free-unified-memory-gib`
- `--metal-runtime-max-mla-kv-b-cache-mib`
- `--max-build-fma` or `--max-build-gfma`
- `--max-metal-builder-live-mib`
- disk safety margins on cache builders

For the local 128 GiB M5 Max experiments, the safe serving shape used a 16 GiB
Metal live cap and a 24 GiB free unified-memory admission guard. Those values
are conservative; keep them conservative unless you are deliberately measuring a
new envelope.

The reopened residency plan uses 44 GiB of hard-pinned hot experts and up to 36
GiB of evictable expert cache. The evictable tier must shrink before the 24 GiB
free-memory guard is crossed; 80 GiB is a conditional total expert working set,
not an unconditional pinned allocation.

Build the no-weight plan with:

```bash
python3 scripts/expert_usage_plan.py route-stats.jsonl \
  --expert-layout experts/layout.json \
  --m5-max-128g-safe \
  --write-profile expert-usage-profile.json \
  --write-plan expert-pin-plan.json
```

## Installation

Create a Python environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

Build Metal binaries when needed:

```bash
make -C metal
```

Run the lightweight Python tests:

```bash
python3 -m pytest tests/test_metal_viability.py tests/test_context1_o_proj_cache.py -q
```

The full test suite is broader and includes many smoke/prototype checks. Do not
run real-weight Metal smoke scripts unless you understand their guards.

## Preparing A Model

The intended model family is GLM-style `glm_moe_dsa` with 4-bit routed experts.
The local experiments used `mlx-community/GLM-5.2-mxfp4`, but weights are not
distributed here.

The high-level flow is:

1. Download or place the checkpoint under a local artifact directory.
2. Fetch/verify safetensors headers.
3. Run the prepared-artifact planner and packer.
4. Validate the prepared artifact before any runtime call.
5. Use a locked launch profile or explicit guards for every run.

Example paths in older local logs use:

```text
artifacts/glm-5.2-mxfp4/
artifacts/glm-5.2-mxfp4/largerlm-prepared/
```

Those paths are examples, not repository contents.

## Viability Gate

Before treating a run as useful, check the evidence-backed target gate:

```bash
python3 scripts/glm_metal_viability_report.py \
  artifacts/glm-5.2-mxfp4/largerlm-prepared \
  --decode-telemetry artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-tiled-matvecadd-9tok-latest.json \
  --context1-collapse-plan artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-collapse-plan-latest.json \
  --target-tok-s 5 \
  --require-below-target
```

For the sealed GLM-5.2 route this command should pass because the best supported
throughput remains below the target. If it fails in the future, the evidence has
changed and the stop decision should be revisited.

To save the report:

```bash
python3 scripts/glm_metal_viability_report.py \
  artifacts/glm-5.2-mxfp4/largerlm-prepared \
  --decode-telemetry artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-tiled-matvecadd-9tok-latest.json \
  --context1-collapse-plan artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-collapse-plan-latest.json \
  --target-tok-s 5 \
  --json > artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-metal-viability-5tps-gate-latest.json
```

## Safe Smoke

After a prepared artifact exists, use the audited smoke wrapper rather than
hand-building a long command:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-text-safe.sh \
  --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-text-latest.json
```

This is intentionally small. It is a sanity check that the local prepared
artifact and guarded Metal path still work, not a benchmark.

## Local Serving Shape

The guarded serving shape is:

```bash
python3 -m largerlm serve-prepared artifacts/glm-5.2-mxfp4/largerlm-prepared \
  --metal-runtime-generation \
  --metal-binary metal/glm_moe_infer \
  --metal-runtime-cache-mla-kv-b-f32 \
  --metal-runtime-max-mla-kv-b-cache-mib 4608 \
  --max-live-working-set-mib 16384 \
  --min-free-unified-memory-gib 24
```

This starts a local HTTP server and routes greedy generation through a lazily
started persistent `glm_moe_infer --generate-server-jsonl` child. Sampling is not
the focus of this proof-of-feasibility path.

## Context=1 Collapsed Attention Cache

GLM MLA has a useful context=1 identity: the current value can be expressed as
`B_v * latent`. For context=1 decode, LargerLM can precompute `o_proj * B_v` per
layer and replace a large attention-output projection with a smaller latent
matvec.

This is a narrow optimization:

- It validates that the algebra and cache format work.
- It can reduce context=1 attention-output cost.
- It does not solve general long-context decode.

Plan or dry-run the cache first:

```bash
python3 -m largerlm context1-o-proj-cache \
  artifacts/glm-5.2-mxfp4/largerlm-prepared \
  --backend metal \
  --build-next-layers 1
```

Validate an existing cache:

```bash
python3 -m largerlm validate-context1-o-proj-cache \
  artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-bv-cache/layout.json \
  --allow-incomplete-progress
```

Incomplete progress emits a `suggested_resume_build` block with dry-run and
explicit execute argv. Actual building still requires `--execute` plus the usual
FMA, disk, and live-memory guards.

## What This Project Proves

It proves that:

- A GLM-style MoE checkpoint can be split into resident and SSD-backed routed
  parts on Apple Silicon.
- Real-weight decode can be run under explicit memory guards without loading the
  whole model into unified memory.
- The Metal/runtime boundary can execute useful GLM components.
- The current GLM-5.2 MXFP4 layout is traffic-heavy enough that this route is
  not a 5 tok/s solution on the tested M5 Max 128 GiB machine.

It does not prove that:

- LargerLM is production-ready.
- The current route is competitive with optimized engines for models that fit in
  memory.
- SSD-backed GLM-5.2 decode can reach 5 tok/s without a major model/layout or
  runtime change.

## Restart Criteria

Reopen performance work only if at least one of these changes:

- A model or packing layout cuts routed expert traffic by roughly `5x`.
- A new engine demonstrates real GLM-family decode at `>=5 tok/s` on comparable
  Mac hardware while staying memory-safe.
- The target changes from high-throughput decode to a polished low-throughput
  local demo.
