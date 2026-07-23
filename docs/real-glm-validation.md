# GLM-5.2 MXFP4 Validation

Date: 2026-07-23

## Scope And Verdict

This is a proof of feasibility, not a production inference engine.

LargerLM successfully downloaded, packed, admitted, and generated with the real
GLM-5.2 MXFP4 checkpoint on an Apple M5 Max with 128 GB unified memory. The
single-process Metal runtime is usable for experiments, but it does not meet
the requested `5 tok/s` continuation gate.

- Safe held-out decode with a learned 10 GiB expert set: `1.083 tok/s` steady.
- No application-owned expert cache: `0.858 tok/s` steady.
- Optimistic context=1 `o_proj*B_v` projection: `1.427 tok/s`.
- Gap to `5 tok/s`: `3.50x`.

The project is therefore sealed at a minimum runnable version. Reopen it only
for a model/layout change or independent evidence of real GLM-family
`>=5 tok/s` decode on comparable hardware.

## Reproduction Machine

- Apple M5 Max, 40 GPU cores, 128 GiB unified memory.
- macOS 26 and Metal 4.
- Internal SSD measured at `11.076 GiB/s` on a 4 GiB sequential read.
- Tests used a 24 GiB minimum available-memory guard.
- The power adapter could not sustain full machine power. macOS reported AC
  attached while the battery was not charging, so all throughput numbers are
  tagged power-limited.

The prepared package targets 4096 context tokens. Its logical contents are:

- routed experts: `385,037,107,200` bytes;
- resident tensors: `10,056,979,968` bytes;
- decode cache: `390,070,272` bytes;
- total: `395,484,157,440` bytes.

The 76 downloaded source shards were removed after the prepared package passed
storage validation and real generation. This reclaimed about 369 GiB. Model
weights and prepared artifacts remain excluded from Git.

## How It Works

The packer separates the checkpoint into a small resident file and 75
layer-local expert files. Decode keeps bounded activation, attention, MLA, and
logits state in a persistent `glm_moe_infer` process. Each MoE layer routes to
eight of 256 experts, reads only those slots, and dispatches custom MXFP4 Metal
kernels. Dense weights and reusable file pages are left to macOS virtual
memory.

The fast path includes:

- persistent JSONL execution instead of per-layer Python subprocesses;
- parallel `pread` into aligned shared Metal buffers;
- deferred MoE command-buffer waits;
- an in-process 4.27 GiB MLA KV-B F32 cache;
- mmap-backed final logits;
- a quality-preserving hot-expert plan learned from complete route telemetry.

MPP TensorOps is available through the `auto-mpp` prefill backend. A real M5
probe passes, but GLM's routed MXFP4 experts remain on custom Metal. Only 75
router matrices, about 225 MiB and roughly 0.5% of router plus routed-expert
FLOPs, are directly eligible. MPP can help long-prompt prefill; it cannot fix
batch-one decode.

## Why Not Pin 80 GiB?

The 128 GiB unified-memory size is not the same as the recommended active Metal
working set. This machine reports about 17.4 GiB for the latter.

A 44 GiB hard plus 36 GiB adaptive experiment was run only after explicitly
raising the live cap. It reached 44 GiB pinned plus 16.1 GiB adaptive, reduced
late expert misses to as little as `0.71 GiB/token`, but slowed steady decode to
`0.524 tok/s` and changed generated tokens after the first step. The result is
rejected.

A 12 GiB static plan was correctly refused by the normal 17.4 GiB admission
cap. An 11 GiB boundary plan stayed correct at `1.073 tok/s`. The published M5
profile uses 10 GiB, which produces a 15.91 GiB total live estimate and leaves
the rest of reusable expert storage to the macOS page cache.

This matches Flash-MoE's retained design: its OS page cache beat a custom LRU,
while extra read advice and prediction could contend with GPU work on unified
memory.

## Measured A/B

Both accepted runs used unseen input token `150000`, eight generated tokens,
the MLA KV-B cache, mmap final logits, and a 24 GiB free-memory guard.

| Run | Steady decode | All decode | Expert hit rate after warmup | Expert read after warmup |
| --- | ---: | ---: | ---: | ---: |
| macOS page cache only | `0.858 tok/s` | `0.743 tok/s` | n/a | `11.206 GiB/token` |
| learned 10 GiB static set | `1.083 tok/s` | `0.947 tok/s` | `36.0-47.0%` | `5.94-7.17 GiB/token` |

Both accepted runs generated exactly:

```text
19, 17, 19, 19, 15, 19, 19, 24
```

The 10 GiB plan improved steady throughput by about 26%. It kept VM pressure
normal, reported about 73.4 GiB system memory available after execution, and
used no adaptive allocations.

The audited Python path also generated a real token, but took `125.8s` because
`118.7s` was spent in CPU final-logits scanning. It is a validation path, not a
performance path.

## Safe Use

Build the runtime:

```bash
make -C metal
```

Collect complete routes with a direct debug replay, then build the safe plan:

```bash
python3 scripts/expert_usage_plan.py route-generated.json \
  --expert-layout artifacts/glm-5.2-mxfp4/largerlm-prepared/experts/layout.json \
  --m5-max-128g-safe \
  --write-profile expert-usage-profile.json \
  --write-plan expert-pin-plan.json
```

Run token generation:

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

For text, use `generate-metal-text`; multi-token prompts enter the bounded
runtime prefill path automatically.

Never remove the live cap or the 24 GiB reserve on a desktop machine you care
about. The large-cache failure above is evidence that system memory availability
alone is not a sufficient GPU residency guard.
