# LargerLM Minimum Runnable Seal

Date: 2026-07-05

## Decision

Seal the current GLM-5.2 MXFP4 Apple Silicon route as a minimum runnable version
and stop speculative optimization toward the 5 tok/s target.

The current evidence-backed ceiling is not close enough:

- Best real GLM decode artifacts are around `0.9-1.0 tok/s`.
- The viability gate with the context=1 `o_proj*B_v` collapse projects
  `1.485 tok/s`, about `1.60x` over the measured 9-token decode artifact.
- The requested continuation target is `5.000 tok/s`; the best supported
  projection is still `3.37x` short.
- The routed expert read floor is still `11.206 GiB/token`; reaching
  `5 tok/s` would require roughly `56 GiB/s` of sustained useful routed-expert
  traffic before counting attention, kernels, logits, scheduling, or HTTP
  overhead.

The matching gate command is:

```bash
python3 scripts/glm_metal_viability_report.py \
  artifacts/glm-5.2-mxfp4/largerlm-prepared \
  --decode-telemetry artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-tiled-matvecadd-9tok-latest.json \
  --context1-collapse-plan artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-collapse-plan-latest.json \
  --target-tok-s 5 \
  --require-below-target
```

Expected target decision:

```text
stop_at_minimal_usable
```

## Minimum Usable Scope

Keep these paths working and guarded:

- Header/prepared-artifact inspection, sizing, disk, and launch-profile checks.
- Safe text smoke for the local prepared GLM artifact.
- Bounded Metal runtime generation with explicit live-memory and free-memory
  guards.
- Context=1 `o_proj*B_v` cache planning, validation, and resumable one-layer
  build suggestions. This remains a narrow verified fast path, not a general
  decode-speed solution.
- Viability reporting with an explicit throughput target.

Safe smoke:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-text-safe.sh \
  --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-text-latest.json
```

Safe prepared server shape:

```bash
python3 -m largerlm serve-prepared artifacts/glm-5.2-mxfp4/largerlm-prepared \
  --metal-runtime-generation \
  --metal-binary metal/glm_moe_infer \
  --metal-runtime-cache-mla-kv-b-f32 \
  --metal-runtime-max-mla-kv-b-cache-mib 4608 \
  --max-live-working-set-mib 16384 \
  --min-free-unified-memory-gib 24
```

## Do Not Continue

Do not spend more time on these under the current GLM-5.2 MXFP4 layout:

- Chasing 5 tok/s with additional small attention-output kernel tweaks.
- Building a general per-head collapsed cache: the modeled exact BF16 cache is
  `29.250 GiB/token` and the int4 floor is still `7.312 GiB/token`.
- Running uncapped real-weight experiments.
- Running large Flash-MOE weight trials unless the user explicitly resumes that
  comparison with downloaded weights available.
- Downloading more huge model shards for this sealed route.

## Restart Criteria

Only reopen the performance project if at least one condition changes:

- A model/layout reduces routed expert traffic by about `5x` while preserving the
  target model quality.
- A new packed expert format proves sustained useful routed-expert throughput
  near the required range on the local machine.
- A different engine demonstrates real GLM-family `>=5 tok/s` decode on this
  class of Mac with comparable memory safety.
- The user lowers the target below the current evidence-backed ceiling and wants
  a polished local demo rather than a 5 tok/s system.
