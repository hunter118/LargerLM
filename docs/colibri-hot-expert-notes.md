# Colibri-Inspired Hot Expert Work

Date: 2026-07-22

## Status

The performance project is active again for one narrow hypothesis: GLM expert
usage may be skewed enough that a learned resident hot set removes much of the
SSD traffic seen by the original LargerLM route.

This is not yet a speed result. The profiler, runtime route telemetry, and
residency planner work without full model weights, but the hot store and
adaptive cache still need to be integrated and measured with GLM-5.2 weights.

## What Colibri Changes

[Colibri](https://github.com/JustVugg/colibri) keeps dense weights resident and
streams routed experts. Its important additions are:

- a persistent expert-frequency file and learned pinned hot set;
- per-layer LRU caches and safe turn-boundary repinning;
- batch union so a selected expert is read once for all active tokens;
- adjacent expert matrices loaded with one bounded read;
- layer-ahead prefetch and an RSS guard that can shrink caches;
- optional cache-aware routing, which is disabled by default because it can
  change model output.

The quality-preserving pieces are directly relevant to LargerLM. Cache-aware
routing is not part of the current plan. See Colibri's
[tuning guide](https://github.com/JustVugg/colibri/blob/main/docs/tuning.md),
[cache-aware routing note](https://github.com/JustVugg/colibri/blob/main/docs/CACHE_ROUTE.md),
and [Metal notes](https://github.com/JustVugg/colibri/blob/main/docs/metal.md).

Colibri reports `2.06 tok/s` on an M5 Max with a learned pinned set of about
46.9 GB. That is useful evidence that expert popularity matters, but it is not
evidence that LargerLM or GLM-5.2 will reach `5 tok/s`.

## 128 GB Memory Envelope

The `m5-max-128g-safe` profile separates expert memory into two tiers:

- 44 GiB hard-pinned hot experts;
- up to 36 GiB of evictable expert LRU;
- 16 GiB maximum runtime live working set;
- 24 GiB minimum available unified-memory guard.

This permits an 80 GiB expert working set only while memory pressure is low.
Pinning the full 80 GiB would be unsafe: adding dense weights, Metal/KV scratch,
the OS, drivers, page tables, and the 24 GiB guard would exceed the machine's
128 GB capacity. The runtime integration must evict LRU entries before crossing
the guard and must refuse new allocations if pressure remains high.

## Usage Profiler And Plan

`largerlm.expert_usage` accepts:

- Colibri-style `layer expert count` files;
- JSON or JSONL route events with `layer` and `experts`;
- per-layer LargerLM router JSON with `--default-layer`;
- LargerLM prefill hotspot results, de-duplicated across copy/range views.

The persistent GLM runtime now includes `selected_experts` in standalone decode
layer results and `expert_routes` in prompt/decode generation steps. These are
complete top-k route observations and do not disable the fused router path.

Hotspot summaries are marked as incomplete telemetry. Use full route events for
a meaningful projected cache-hit fraction.

Create only a profile:

```bash
python3 scripts/expert_usage_plan.py route-stats.jsonl \
  --write-profile expert-usage-profile.json \
  --write-colibri-usage .coli_usage
```

Create the bounded M5 Max plan after a prepared expert layout exists:

```bash
python3 scripts/expert_usage_plan.py route-stats.jsonl \
  --expert-layout artifacts/glm-5.2-mxfp4/largerlm-prepared/experts/layout.json \
  --m5-max-128g-safe \
  --write-profile expert-usage-profile.json \
  --write-plan expert-pin-plan.json
```

The plan is quality preserving: it changes residency priority, not router
selection. Selection is greedy by observed count per expert byte, so layers
with different expert slot sizes use the RAM budget efficiently.

## M5 Neural Accelerators

Apple documents a Neural Accelerator in each M5 GPU core. Metal Performance
Primitives TensorOps is the direct low-level interface, while MPSGraph and MLX
can use the hardware through optimized framework paths. See Apple's
[M5 GPU ML talk](https://developer.apple.com/videos/play/tech-talks/111432/) and
[MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).

On this 40-core M5 Max, the existing LargerLM probe confirms Metal 4 tensors and
MPSGraph execution. The resident batch-linear smoke also passes with automatic
selection of `mpsgraph-f32` for a 128-token batch.

Direct MPP TensorOps is not yet validated. The installed Xcode 26.5 SDK contains
the MPP headers, but the separate Metal Toolchain is absent, and
`xcodebuild -downloadComponent MetalToolchain` currently fails because the local
Xcode plug-ins and system developer frameworks do not match. Fix that Xcode
installation before building an offline MPP kernel.

Neural Accelerators principally help compute-heavy batched prefill. They do not
remove SSD latency from batch-one decode, so hot-expert residency remains the
central decode optimization.

## Next Runtime Milestones

1. Materialize the planned 44 GiB hot set as a bounded resident slab.
2. Add a 36 GiB maximum evictable per-layer LRU with memory-pressure shrinkage.
3. Add layer-ahead prefetch without altering selected experts.
4. Re-download or externally copy GLM-5.2 weights and measure hit rate, SSD
   bytes/token, prefill time, decode tok/s, and peak memory.
