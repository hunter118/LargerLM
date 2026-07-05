# LargerLM

LargerLM is a proof-of-feasibility Apple Silicon inference project for
GLM-style MoE models that are larger than unified memory. It explores
SSD-backed routed expert streaming, bounded Metal execution, and safety gates
for 4-bit GLM-5.2-style checkpoints.

This repository is not a production inference engine. The current GLM-5.2 MXFP4
route is sealed as a minimum runnable proof of feasibility: guarded generation
works, but measured decode is around `0.9-1.0 tok/s` and the best
evidence-backed projection is `1.485 tok/s`, below the `5 tok/s` continuation
target.

## Start Here

- [Proof-of-feasibility guide](docs/proof-of-feasibility-guide.md): device
  requirements, design principle, safety model, and usage commands.
- [Minimum runnable seal](docs/minimal-usable-seal.md): why the GLM-5.2 route
  was stopped at a minimum usable version.
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

The project also shows the limit of this approach for the tested GLM-5.2 MXFP4
layout: routed expert traffic is about `11.206 GiB/token`, so reaching `5 tok/s`
would require roughly `56 GiB/s` of useful expert traffic before attention,
logits, kernels, or scheduling overhead.

## Device Requirements

Recommended for reproducing the local experiments:

- Apple Silicon Mac, ideally Max/Ultra class.
- 128 GiB unified memory for the GLM-5.2 experiments.
- Fast internal SSD with hundreds of GB free.
- macOS with Xcode command line tools and Metal support.
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

The project is archived as a minimum runnable proof of feasibility. Continue
only if the model layout changes substantially, a comparable Mac demonstrates
real GLM-family `>=5 tok/s` decode safely, or the goal changes from throughput
to a polished low-throughput local demo.
