# LargerLM Minimum Runnable Seal

Date: 2026-07-23

> Historical decision, reopened on 2026-07-24 after direct Colibri testing.
> See [Colibri M5 Max Optimization](colibri-m5-optimization.md) for the active
> quality-preserving and experimental approximately 5 tok/s paths.

## Decision

Seal the GLM-5.2 MXFP4 M5 Max route as a minimum runnable proof of feasibility
and stop speculative work toward `5 tok/s`.

The final real-weight evidence is:

- no application expert cache: `0.858 tok/s` steady decode;
- safe learned 10 GiB expert set: `1.083 tok/s` steady decode;
- identical eight-token held-out output between those runs;
- optimistic context=1 `o_proj*B_v` projection: `1.427 tok/s`;
- gap from that projection to `5 tok/s`: `3.50x`;
- power-limited test machine, but no evidence for the required 3.5x full-pipeline
  improvement at normal adapter power.

The 44+36 GiB expert-cache hypothesis was also tested. After manually raising
the Metal live cap it became slower and changed output, so it is rejected. The
published M5 profile uses 10 GiB static residency and lets macOS manage the
remaining reusable file pages.

The matching viability gate is:

```bash
python3 scripts/glm_metal_viability_report.py \
  artifacts/glm-5.2-mxfp4/largerlm-prepared \
  --decode-telemetry \
    artifacts/glm-5.2-mxfp4/largerlm-prepared/metal-heldout-cache10g-safe-p150000-8t.json \
  --context1-collapse-plan \
    artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-collapse-plan-current.json \
  --target-tok-s 5 \
  --require-below-target
```

Expected decision:

```text
stop_at_minimal_usable
```

## Preserved Scope

Keep these paths working:

- checkpoint and prepared-artifact validation;
- 4096-token bounded package preparation;
- persistent Metal token and text generation;
- 10 GiB learned expert residency with unchanged routing;
- MLA KV-B caching and mmap final logits;
- MPP/MPSGraph prefill probes and `auto-mpp`;
- explicit Metal live-memory and 24 GiB system reserve guards;
- viability reporting against a user-supplied throughput target.

Safe token generation:

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

## Stop Scope

Do not continue these under the current model layout:

- chasing `5 tok/s` with additional small kernel changes;
- increasing application-owned expert residency beyond the Metal working-set
  admission cap;
- enabling a large adaptive cache by raising `--max-live-working-set-mib`;
- building the 4.02T-FMA context=1 cache as though it were a sustained decode
  solution;
- downloading a second huge checkpoint only to reproduce another engine's
  model-specific headline.

## Restart Criteria

Reopen performance work only when at least one condition changes:

- a GLM-compatible model/layout reduces routed expert traffic by several times;
- another engine demonstrates real comparable GLM-family `>=5 tok/s` decode on
  this class of Mac with memory guards;
- Apple exposes a materially different MXFP4 execution path for the M5 Neural
  Accelerators;
- the throughput requirement is lowered to roughly the measured 1 tok/s class.

The full experiment record is in
[GLM-5.2 MXFP4 validation](real-glm-validation.md).
