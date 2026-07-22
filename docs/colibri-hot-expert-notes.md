# Colibri-Inspired Hot Expert Work

Date: 2026-07-22

## Status

The performance project is active again for one narrow hypothesis: GLM expert
usage may be skewed enough that a learned resident hot set removes much of the
SSD traffic seen by the original LargerLM route.

This is not yet a real-model speed result. The profiler, runtime route
telemetry, residency planner, hard-pinned hot store, and adaptive per-layer LRU
are integrated and pass synthetic Metal tests. They still need to be measured
with GLM-5.2 weights.

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
It is an experimental ceiling, not the default. Pinning the full 80 GiB would
be unsafe: adding dense weights, Metal/KV scratch, the OS, drivers, page tables,
and the 24 GiB guard would exceed the machine's 128 GB capacity. Even below
that ceiling, application-owned buffers displace the macOS file cache and can
make expert reads slower. The runtime evicts adaptive entries before crossing
the guard and refuses execution or new cache allocations if pressure remains
high.

## Flash-MoE Cache Result

A fresh review of Flash-MoE's retained fast path changes the benchmark order.
Its author measured the OS page cache at about a 71% hit rate and found that
removing a custom Metal LRU improved throughput by 38%. Temporal expert
prediction was 18% slower, while `F_RDADVISE` reduced expert I/O but increased
concurrent GPU time by 73% because SSD DMA and Metal share the unified-memory
fabric. Its final path uses parallel `pread` directly into aligned shared Metal
buffers, overlaps only work that does not require route prediction, defers the
expert command buffer, and otherwise trusts macOS.

LargerLM already has the same quality-preserving direct-read, aligned reusable
buffer, parallel-worker, and deferred-submit foundations. Its custom cache is
therefore opt-in. Real GLM-5.2 testing must compare, in this order:

1. no application cache, allowing the largest OS page cache;
2. a smaller learned hot set with a bounded adaptive tier;
3. the 44+36 GiB upper bound.

The 80 GiB case is retained because GLM routing may be more skewed than
Flash-MoE's Qwen routing. It wins only if the measured reduction in SSD bytes
outweighs lost page-cache capacity, memory pressure, and lookup/allocation cost.

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

The plan is consumed directly by `glm_moe_infer`:

```bash
make -C metal glm_moe_infer

metal/glm_moe_infer \
  --prepared artifacts/glm-5.2-mxfp4/largerlm-prepared \
  --expert-pin-plan expert-pin-plan.json \
  --generate-token-ids \
  ...
```

At first execution the runtime validates the complete pin plan and available
memory before allocating anything large. Planned experts are loaded into
stable shared `MTLBuffer` objects. A route hit passes that buffer directly to
the MXFP4 kernel; it does not copy the expert into a staging buffer. Misses use
the existing bounded parallel `pread` path and are admitted to an evictable
cache only when both the byte budget and memory guard permit it.

Allocation requires both enough reclaimable pages to preserve the configured
minimum and a normal macOS VM-pressure level. Warning or critical pressure
refuses preload/cache growth. This is intentionally conservative for a 128 GB
unified-memory machine where excessive pressure can make the whole desktop
unresponsive.

The adaptive budget is divided evenly across routed layers. Each layer has its
own LRU so the normal layer-by-layer decode scan cannot evict early-layer
experts merely because later layers ran more recently. Under actual system
memory pressure the runtime may evict the globally oldest adaptive entry from
any layer. Pinned entries are never selected for eviction.

Runtime JSON includes `expert_resident_cache` plus per-layer and aggregate
`expert_cache_hit_count`, `expert_cache_miss_count`, `expert_cache_hit_bytes`,
SSD read counters, and a post-execution available-memory/VM-pressure snapshot.
These fields are the basis for the later real-model decision.

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

## Validation And Next Milestones

The no-weight Metal smoke now verifies pinned hits, adaptive hits, per-layer LRU
eviction, unchanged MXFP4 output, reduced SSD task count, and refusal when the
minimum-free-memory guard cannot be met. The composed context-1 decode smoke
also passes.

Remaining work:

1. Finish the GLM-5.2 download and prepare a 4096-token package; this keeps the
   initial decode cache near 372 MiB instead of about 16 GiB.
2. Measure the no-cache baseline, then collect complete route telemetry and
   compare smaller residency plans with the 44+36 GiB upper bound.
3. Promote a cache policy only when it improves steady-state tok/s without
   crossing the 24 GiB free-memory guard or raising VM pressure.
4. Repair/install the Xcode Metal Toolchain and compare direct MPP TensorOps
   against the working MPSGraph prefill path.
