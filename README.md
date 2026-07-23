# LargerLM

LargerLM is a proof-of-feasibility Apple Silicon inference project for
GLM-style MoE models that are larger than unified memory. It explores
SSD-backed routed expert streaming, bounded Metal execution, and safety gates
for 4-bit GLM-5.2-style checkpoints.

This repository is not a production inference engine. The original GLM-5.2
MXFP4 route reached guarded generation at around `0.9-1.0 tok/s`, with a best
evidence-backed projection of `1.485 tok/s`. Performance work has been reopened
to test Colibri-style learned hot-expert residency, bounded adaptive caching,
and M5 GPU Neural Accelerator prefill paths. No local model weights are
currently included.

## Start Here

- [Proof-of-feasibility guide](docs/proof-of-feasibility-guide.md): device
  requirements, design principle, safety model, and usage commands.
- [Colibri hot-expert notes](docs/colibri-hot-expert-notes.md): the reopened
  performance direction and the 128 GB M5 Max memory envelope.
- [Minimum runnable seal](docs/minimal-usable-seal.md): the historical stop
  decision for the original streaming route.
- [Architecture notes](docs/architecture.md): deeper implementation details.
- [Flash-MOE rewrite decision](docs/flash-moe-rewrite-decision.md): comparison
  with Flash-MOE-style streaming and the final efficiency call.
- [Development log](docs/development-log.md): the long historical README moved
  out of the GitHub landing page.

## What It Proves

LargerLM demonstrates that a GLM-style MoE checkpoint can be split into:

- resident tensors that stay memory-mapped or staged locally,
- routed expert slots streamed from SSD,
- bounded Metal kernels and persistent runtime processes,
- admission checks that cap live memory, free unified memory, disk usage, and
  routed read volume.

The project also provides a quality-preserving expert-usage profiler, pin
planner, and bounded runtime expert cache. It can learn a hot set from
LargerLM telemetry, router JSON/JSONL, or Colibri `.coli_usage` files.
On M5-class Macs, `mpp-f32` is an opt-in MPP TensorOps backend for resident
F32/BF16/F16 prefill GEMMs; routed MXFP4 experts remain on the bounded custom
Metal path.

## Device Requirements

Recommended for reproducing the local experiments:

- Apple M5 Max with 128 GB unified memory for the current target profile.
- Fast internal SSD with hundreds of GB free.
- macOS 26 with Metal 4; the full Xcode Metal Toolchain is needed for direct MPP
  TensorOps development.
- Python 3.9+.

Model weights and prepared artifacts are not included. Local artifacts are
ignored under `artifacts/`.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

Build local Metal binaries when needed:

```bash
make -C metal
```

Run a lightweight sanity subset:

```bash
python3 -m pytest tests/test_metal_viability.py tests/test_context1_o_proj_cache.py -q
```

## Safety

Use guarded commands only. Important knobs include:

- `--max-live-working-set-mib`
- `--min-free-unified-memory-gib`
- `--metal-runtime-max-mla-kv-b-cache-mib`
- `--max-build-fma` / `--max-build-gfma`
- `--max-metal-builder-live-mib`

The local proof-of-feasibility route used conservative guards such as a 16 GiB
Metal live cap and a 24 GiB free unified-memory admission guard.

The default M5 Max path uses no application-owned expert cache and lets macOS
manage reusable expert pages. A 44 GiB hard tier plus a 36 GiB adaptive tier is
available only as an experimental upper-bound comparison. Its adaptive tier
must shrink before the 24 GiB free-memory guard is crossed; it is not the
recommended default until a real GLM route replay beats the OS page cache.

Build a no-weight expert profile and pin plan with:

```bash
python3 scripts/expert_usage_plan.py route-stats.jsonl \
  --expert-layout experts/layout.json \
  --m5-max-128g-safe \
  --write-profile expert-usage-profile.json \
  --write-plan expert-pin-plan.json
```

Pass the result to the Metal runtime with
`--expert-pin-plan expert-pin-plan.json`. It preloads the hard-pinned set and
uses a memory-guarded per-layer LRU for the adaptive tier.

## Viability Gate

When local prepared artifacts exist, the sealed GLM-5.2 route can be checked
with:

```bash
python3 scripts/glm_metal_viability_report.py \
  artifacts/glm-5.2-mxfp4/largerlm-prepared \
  --decode-telemetry artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-tiled-matvecadd-9tok-latest.json \
  --context1-collapse-plan artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-collapse-plan-latest.json \
  --target-tok-s 5 \
  --require-below-target
```

For the sealed route this gate passes because the best observed/projected
throughput remains below the target. If it fails in the future, the evidence has
changed and the stop decision should be revisited.

## Repository Layout

- `largerlm/`: Python planning, packing, safety, runtime, and server code.
- `metal/`: Objective-C/Metal runtime sources and smoke tests.
- `scripts/`: analysis, viability, and artifact tooling.
- `tests/`: Python test coverage for planners, guards, formats, and reports.
- `docs/`: architecture, feasibility, and historical notes.

## Current Status

Active feasibility work has resumed. Runtime route telemetry, expert profiling,
hard-pinned residency, adaptive per-layer LRU caching, and memory-pressure
refusal are implemented and pass synthetic Metal tests. Real-weight speed
validation remains. The previous `<1.5 tok/s` evidence is still the baseline,
and no `5 tok/s` claim is made.
