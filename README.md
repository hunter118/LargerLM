# LargerLM

LargerLM is a proof-of-feasibility Apple Silicon runtime for GLM-style MoE
models larger than unified memory. It combines SSD-backed expert streaming,
bounded Metal execution, M5 prefill acceleration, and strict memory admission.

It is not a production inference engine.

## Status

Real GLM-5.2 MXFP4 generation now works on an M5 Max with 128 GB unified
memory. A learned 10 GiB expert set improved held-out steady decode from
`0.858` to `1.083 tok/s` while preserving generated tokens and a 24 GiB system
memory reserve. The best evidence-backed context=1 projection is `1.427 tok/s`,
well below the `5 tok/s` continuation gate, so the project is sealed at a
minimum runnable version.

The detailed measurements and the rejected 80 GiB cache experiment are in
[GLM-5.2 MXFP4 validation](docs/real-glm-validation.md).

## Requirements

- Apple M5 Max with 128 GB unified memory for the validated profile.
- macOS 26 with Metal 4.
- Fast internal SSD and about 400 GB for a prepared GLM-5.2 package.
- Python 3.9+ and the local Metal build tools.

Weights and generated artifacts are excluded from Git.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
make -C metal
```

Generate with an existing prepared package:

```bash
python3 -m largerlm generate-metal-token-ids \
  artifacts/glm-5.2-mxfp4/largerlm-prepared \
  --expert-pin-plan expert-pin-plan.json \
  --prompt-token-ids 150000 \
  --max-new-tokens 8 \
  --mmap-final-logits \
  --cache-mla-kv-b-f32 \
  --max-mla-kv-b-cache-mib 4608 \
  --max-live-working-set-mib 16384 \
  --min-free-unified-memory-gib 24
```

Build the measured M5 Max expert plan from route telemetry:

```bash
python3 scripts/expert_usage_plan.py route-generated.json \
  --expert-layout artifacts/glm-5.2-mxfp4/largerlm-prepared/experts/layout.json \
  --m5-max-128g-safe \
  --write-profile expert-usage-profile.json \
  --write-plan expert-pin-plan.json
```

The safe profile pins at most 10 GiB and leaves the remaining reusable expert
pages to macOS. Do not raise the live cap merely because physical unified
memory is available.

## Documentation

- [Validation report](docs/real-glm-validation.md)
- [Proof-of-feasibility guide](docs/proof-of-feasibility-guide.md)
- [Minimum runnable seal](docs/minimal-usable-seal.md)
- [Colibri hot-expert analysis](docs/colibri-hot-expert-notes.md)
- [Flash-MoE rewrite decision](docs/flash-moe-rewrite-decision.md)
- [Architecture](docs/architecture.md)
- [Development log](docs/development-log.md)

## Test

```bash
.venv/bin/python -m pytest -q
```
