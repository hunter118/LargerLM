# Colibri-Inspired Hot Expert Work

Date: 2026-07-23

> Historical note: this report covers the first LargerLM-native cache path.
> The active Colibri runtime and July 24 results supersede its final decision;
> see [Colibri M5 Max Optimization](colibri-m5-optimization.md).

## Final Status

The Colibri hypothesis was tested with real GLM-5.2 MXFP4 weights on the target
M5 Max 128 GB machine. Learned expert residency helps, but the safe improvement
is not large enough to approach `5 tok/s`.

- macOS page-cache baseline: `0.858 tok/s` steady.
- Safe learned 10 GiB expert set: `1.083 tok/s` steady.
- Generated tokens were identical.
- The accepted cache reduced steady expert reads from `11.206 GiB/token` to
  `5.94-7.17 GiB/token`.

See [the validation report](real-glm-validation.md) for the complete A/B.

## What Was Reused

[Colibri](https://github.com/JustVugg/colibri) keeps dense weights resident and
streams routed experts. The quality-preserving parts relevant to LargerLM are:

- persistent expert-frequency telemetry;
- a learned per-layer hot set;
- bounded per-layer LRU policy;
- batch union and adjacent range reads;
- memory-pressure refusal;
- routing remains unchanged.

LargerLM accepts Colibri `layer expert count` files, generic JSON/JSONL route
events, native generation `expert_routes`, and prefill hotspot reports. The
planner ranks observations by count per expert byte and emits a runtime
consumable `largerlm.expert_pin_plan.v1`.

Cache-aware routing is deliberately excluded because it changes model
semantics.

## What Flash-MoE Changed

A code review of [Flash-MoE](https://github.com/danveloper/flash-moe) showed
that its retained fast path relies heavily on the macOS file cache. Its own
experiments found a custom LRU slower, temporal prediction slower, and read
advice capable of increasing concurrent GPU time on unified memory.

That changed LargerLM's default:

1. trust the OS page cache;
2. add only a small measured hot set;
3. use parallel `pread` into bounded reusable Metal staging buffers;
4. avoid speculative expert prefetch that depends on an unconfirmed route.

This also explains why a weaker machine can report a better speed on a
different model. GLM-5.2 reads `11.206 GiB/token` of routed experts, about
`7.08x` the Qwen-shaped Flash-MoE comparison used by the viability report, and
still performs substantial MLA and attention-output work.

## Why The Profile Is 10 GiB

The first experimental profile allowed 44 GiB hard residency plus 36 GiB
adaptive residency. That was a physical-memory envelope, not a GPU working-set
envelope.

The target M5 Max reports only about 17.4 GiB as its recommended Metal working
set. The real runtime already needs about 5.5 GiB for MLA KV-B, staging,
activations, and logits. Real tests found:

- 1 GiB expert residency stayed correct and roughly neutral;
- 10 GiB produced a 15.91 GiB live estimate and the best accepted result;
- 11 GiB produced a 16.95 GiB live estimate and stayed correct;
- 12 GiB was refused by the normal 17.4 GiB live cap;
- a manually raised cap with 44 GiB pinned plus 16.1 GiB adaptive became
  slower and changed generated tokens.

The published `m5-max-128g-safe` profile therefore uses:

- 10 GiB hard-pinned experts;
- no application-owned adaptive expert tier;
- 16 GiB total Metal live cap, including the expert cache;
- 24 GiB minimum system-available memory;
- macOS page cache for the rest.

The remaining unified memory is still useful. It backs file-cache pages,
resident model data, drivers, the desktop, and other system allocations. It
just should not be exposed as tens of GiB of simultaneously active
`MTLBuffer` resources.

## Build A Plan

Collect complete routes from one or more representative prompts, then run:

```bash
python3 scripts/expert_usage_plan.py route-a.json route-b.json \
  --expert-layout artifacts/glm-5.2-mxfp4/largerlm-prepared/experts/layout.json \
  --m5-max-128g-safe \
  --write-profile expert-usage-profile.json \
  --write-plan expert-pin-plan.json \
  --write-colibri-usage .coli_usage
```

Pass the plan to the persistent Metal runtime:

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

Runtime telemetry includes cache hits, misses, hit bytes, stores, evictions,
pressure rejections, allocation size, and post-run system memory.

## M5 Neural Accelerators

Direct MPP TensorOps is now validated on this M5 Max through the system
framework. The 32x32x32 half-matmul probe passes with zero maximum absolute
error. `auto-mpp` selects MPP for eligible large resident F32/BF16/F16 prefill
matrices; MXFP4 and int4 remain on custom Metal.

For this GLM shape, only the 75 router matrices are directly eligible, about
225 MiB and roughly 0.5% of router plus routed-expert FLOPs. The feature is
useful for long-prompt prefill coverage, not batch-one decode throughput.

## Decision

The Colibri work improved the safe minimum runtime and answered the memory
question, but it did not change the product decision. The optimistic
context=1 projection is `1.427 tok/s`, still `3.50x` below the requested gate.
Further work requires a different model/layout or independent `>=5 tok/s`
evidence, not a larger application-owned expert cache.
