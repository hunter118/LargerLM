# LargerLM

Proof-of-feasibility GLM-5.2 inference on an Apple M5 Max with 128 GB unified
memory. The active runtime is a guarded, M5-tuned
[Colibri](https://github.com/JustVugg/colibri) build that streams routed experts
from the internal SSD.

This is experimental research code, not a production inference engine.

## Current Result

| Mode | Decode | Routing | Peak RSS |
| --- | ---: | --- | ---: |
| `quality` | `2.69 tok/s` over 256 tokens | Original GLM top-8 | `96.95 GiB` |
| `experimental-fast` | `4.92 tok/s` over 64 tokens | Cache-aware `J=2, M=32` | `96.31 GiB` |
| `experimental-fast` Web (32K) | `5.48 tok/s` over 64 tokens | Single-slot persistent API | about `97 GiB` |

The fast mode reached `5.21 tok/s` over its first 32 tokens on the
power-constrained test machine. It changes about 35% of routed expert slots and
is not output-equivalent to the original model. The quality mode is the default.

## Requirements

- Apple M5 Max with 128 GB unified memory.
- macOS 26, Xcode command-line tools, and the fast internal SSD.
- About 400 GB free for the 357 GB Colibri GLM-5.2 package.
- At least 24 GiB reclaimable memory before launch.

## Setup

Build the pinned Colibri revision and apply the tested M5 patch:

```bash
./scripts/setup_colibri_m5.sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

Download the supported model:

```bash
hf download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp \
  --local-dir artifacts/colibri-glm5.2-int4
```

## Web Chat

Build Colibri's official UI, then start the guarded loopback-only server:

```bash
./scripts/setup_colibri_web.sh
.venv/bin/python scripts/run_colibri_m5.py \
  --web --detach --mode experimental-fast --profile --preserve-usage
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Stop the model and release
its memory with:

```bash
.venv/bin/python scripts/run_colibri_m5.py --stop-web
```

The UI is persistent between questions. Its headline speed includes prefill and
time to first token; the Performance view reports decode throughput separately.
The guarded Web profile provides a 32,768-token total context and output cap.

Run with the original model routing:

```bash
.venv/bin/python scripts/run_colibri_m5.py \
  "请用三点解释量子纠缠为什么不能用于超光速通信。" \
  --ngen 64 --profile
```

Opt in to the experimental approximately 5 tok/s route:

```bash
.venv/bin/python scripts/run_colibri_m5.py \
  "请用三点解释量子纠缠为什么不能用于超光速通信。" \
  --mode experimental-fast --ngen 64 --profile
```

The launcher refuses unsafe starts and terminates Colibri before process-group
RSS exceeds 105 GiB or macOS memory pressure falls below 10% free. Do not bypass
these guards on a 128 GB machine.

## Documentation

- [M5 optimization and usage guide](docs/colibri-m5-optimization.md)
- [Real GLM validation history](docs/real-glm-validation.md)
- [Architecture](docs/architecture.md)
- [Development log](docs/development-log.md)

## Test

```bash
.venv/bin/python -m pytest -q
make -C third_party/colibri/c metal-test
```
