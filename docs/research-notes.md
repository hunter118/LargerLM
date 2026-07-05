# Research Notes

## Flash-MoE

Reference: <https://github.com/danveloper/flash-moe>

The useful pieces for LargerLM are:

- Store routed experts in per-layer packed files with fixed-size expert slots.
- Use `pread` into scratch buffers instead of `mmap` for cold or mixed cold/warm
  expert reads.
- Use OS page cache as the primary expert cache. Large custom Metal or malloc
  caches can reduce the memory available to the kernel page cache.
- Use 2 MB aligned destination buffers and `newBufferWithBytesNoCopy` so the
  same memory is efficient for SSD DMA and Metal reads.
- Keep one layer file descriptor open across the selected expert reads in a
  MoE forward, and use macOS `F_RDADVISE` as a hinting layer before reaching for
  larger application-owned prefetch buffers.
- Calibrate `plan --cold-read-gib-s` with bounded sequential `pread` on an
  existing packed layer file; avoid synthetic large-file generation or `mmap`
  when measuring the expert streaming path.
- Keep routed expert kernels simple and bandwidth-aware: affine dequant plus
  matvec, fused gate/up/SwiGLU where shape permits, and a fused final combine.

Flash-MoE's most important systems result is that SSD DMA and GPU compute both
compete for the unified memory fabric. Naive async prefetching can slow the GPU
more than it hides I/O.
The repository also gives a useful hardware reference point: a MacBook Pro M3
Max with 48 GB unified memory and roughly 17.5 GB/s measured sequential SSD
read bandwidth can stream a 209 GB 4-bit expert set on demand. The M5 Max target
should have more headroom, but only if reads remain coalesced and memory
pressure leaves room for the OS page cache. Planner and prepare budget inputs
now fail early when a negative reserve, zero runtime buffer, non-finite value, or
out-of-range page-cache fraction would make that headroom estimate optimistic.

Current LargerLM mapping:

- The source-only Flash-MoE comparison now points to a structural GLM fix for
  decode rather than another resident/tiled attention-output kernel. For
  context=1, precomputing `o_proj * B_v` turns the current `3978.0 MiB/token`
  attention-output read into a `468.0 MiB` total BF16 collapsed cache on the
  current GLM-5.2 MXFP4 artifact, with a one-time offline/resumable `4.02T FMA`
  build. The planner artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-collapse-plan-latest.json`.
  The follow-up cache builder skeleton is
  `python -m largerlm context1-o-proj-cache`; it currently provides the
  production cache schema, dry-run/report path, resumable progress contract, and
  a tiny-fixture reference backend while refusing the full GLM reference build
  unless a deliberately raised FMA cap is provided. Runtime integration should
  go through `python -m largerlm validate-context1-o-proj-cache`, which rejects
  missing backing files, overlapping spans, non-numeric layer order drift,
  dtype/dim mismatches, cache/prepared-artifact mismatches, and incomplete
  builder progress before Metal code maps the cache. `--allow-incomplete-progress`
  reports partial chunk-build status for diagnostics without admitting the cache
  to runtime use. `metal/glm_moe_infer` now also has a standalone
  `--build-context1-o-proj-cache-layer` one-layer builder and
  `--probe-context1-o-proj-cache-output` consumer. The builder multiplies
  resident MXFP4 `o_proj` and `unembed_out` into a selected BF16/F32 cache span
  under an explicit GFMA cap, and the consumer validates the runtime side with
  BF16/F32 Metal matvec+residual kernels. `python -m largerlm
  context1-o-proj-cache --backend metal --execute` now wraps that builder in the
  resumable cache-builder CLI, with default dry-run behavior and a default FMA
  cap that refuses even one real GLM layer unless raised intentionally.
  `--build-layers` can execute a bounded subset against the full cache layout,
  and serving rejects incomplete progress before using the cache. The path is
  now folded into the runtime decode layer: `glm_moe_infer` uses
  `--context1-o-proj-cache-layout` to route `context_length=1` dense and MoE
  decoder-layer attention output through the collapsed cache, and the
  dense/MoE/decode-layers context1 smoke tests verify identical tiny baseline
  output/cache writes plus `attn_output_context1_o_proj_cache` telemetry. The
  Python Metal generation payload and prepared HTTP runtime also expose the
  opt-in, with prepared HTTP startup validation and `/health` summary for the
  layout/backing/prepared match. Generation responses and decode telemetry
  reports carry `attn_output_context1_o_proj_cache_count` for benchmark proof;
  real GLM cache execution and benchmarking are still pending.
- Expert files remain on disk; selected experts are copied into bounded compact
  stages per batch/layer.
- The single-process `glm_moe_infer` route now has a reusable direct
  routed-expert read primitive for the Flash-MoE-style hot path. It opens
  packed expert layer files once, disables fd readahead, range-checks selected
  expert ids, reads a batch of selected slots through the persistent `pread`
  worker pool into reusable 2 MiB-aligned `newBufferWithBytesNoCopy` Metal
  buffers, and reports dispatch/task/worker telemetry from both
  `--probe-expert-read` and layer MoE. The tiny MXFP4 smoke verifies the pooled
  read path and the out-of-range expert guard without touching real GLM
  weights.
- The same runtime now has a minimal mmap-backed resident MXFP4 linear path.
  With `--mmap-resident --wrap-resident-metal`, resident weight/scales are
  addressed by offsets inside one read-only resident Metal buffer instead of
  being staged with a separate `pread` into an aligned matrix buffer. The tiny
  resident-linear smoke compares the mmap path against the staging fallback and
  runner output, verifies `bytes_read=0` for the explicit staging read, and
  confirms the live estimate drops to resident mmap plus input/output scratch.
  The same diagnostic wiring now reaches attention-output composition: tiny
  standalone, dense decoder fallback, and decode-layers hot-path smokes verify
  `attn_output_resident_mmap_backed=1`, `attn_output_bytes_read=0`, and
  identical outputs. This is kept as an opt-in diagnostic because real GLM MoE
  testing showed file-backed mmap GPU reads are slower than bounded staging.
- `stage-batch-experts` coalesces selected expert slots into aligned ranges.
  The Python stage copy now issues non-fatal macOS `F_RDADVISE` hints for those
  coalesced ranges and writes read-advice telemetry into the stage manifest.
  Staged execution hardlinks the compact layer when the stage file already has
  exactly the selected slots in compact order, and falls back to the existing
  bounded copy path otherwise. Actual prompt telemetry now separates logical
  compact expert bytes from `compact_stage_materialized_bytes`, so hardlinked
  runs report zero extra compact-stage storage while preserving runner read
  volume.
- Stage manifests and prompt prefill results report serial-vs-unique-vs-planned
  expert read bytes, coalesced range counts, savings, read amplification, and
  read-advice attempted/call/byte/failure counters so long runs can verify SSD
  reads are still coalesced and hinted where macOS accepts the hint. Router JSON
  and packed layer-file paths are validated before staging, and staged execution
  now checks stage byte counts, coalesced range coverage, and slot source/stage
  offsets before compacting unique experts. Malformed routing telemetry cannot
  redirect SSD reads, silently duplicate a wrong expert slot, or inject NaN
  weights into later kernels. If actual stage-copy elapsed time exceeds the
  configured seconds cap, staging now fails before the routed MoE Metal command
  runs. Successful stages also write actual copy elapsed seconds, effective
  staged-file GiB/s, and copy-seconds cap status into `io_summary`, and the
  staged MoE, prompt prefill, generation, and benchmark summaries preserve
  those fields when present so M5 SSD tuning can compare measured copy
  throughput with planned read-second budgets.
- Safetensors scanning now validates known dtype byte widths against every
  tensor shape and byte span before scan/preflight/pack consumers trust the
  metadata, keeping corrupt GLM checkpoint shards from producing optimistic
  resident or routed byte budgets.
- The same scanner keeps `weight_map` authoritative: shard paths must remain
  relative to the model directory, and every tensor-like header entry in a
  referenced shard must be listed in the map so scan budgets cannot silently
  omit stray tensors.
- Referenced shard payloads must be fully covered by non-overlapping indexed
  tensor spans, preventing corrupt offsets or trailing unindexed data from being
  trusted by the planner or packers.
- Safetensors index `metadata.total_size`, when present, must match the summed
  indexed tensor byte spans, so a stale or partial index cannot understate the
  checkpoint budget before GLM preflight or packing.
- GLM 4bit readiness now also checks exact packed expert layer-file sizes
  against `num_experts * expert_slot_bytes` and reports the actual summed layer
  file bytes, so oversized or duplicated expert files fail before a launch
  script accepts the package's SSD-volume estimates.
- The same readiness gate now validates the decode-cache layout against the GLM
  config before launch: `mla_kv` segments must cover every model layer with the
  config-derived MLA cache width, full DSA indexer layers must have matching
  `dsa_index` segments, unexpected or duplicate segments fail closed, and the
  cache backing file must match the layout total.
- Batch prefill admission now has a separate routed-read amplification guard:
  after auto chunk sizing resolves the prompt chunk, a positive
  `prefill_max_routed_read_amplification` rejects chunks that save activation
  memory by multiplying expert SSD reads too aggressively. A companion
  `prefill_max_routed_read_gib` guard caps the absolute planned routed expert
  read volume for prompts whose ratio is acceptable but whose SSD budget is not.
  With a measured SSD read GiB/s, the same estimate can be converted into a
  seconds cap so prompt admission follows a user-visible latency budget. When a
  cap rejects a prompt, the checker now reports the minimum chunk size that
  would satisfy the routed-read limits, or marks the cap impossible if even a
  full-prompt chunk is too large.
- Batch expert staging can now carry the same measured SSD speed into
  `plan-batch-expert-io` and `stage-batch-experts`. The plan/manifest record
  planned read seconds, and `--max-read-seconds` rejects a stage before copying
  when the coalesced expert read plan would exceed the latency budget.
  `stage-batch-experts` and the staged routed prefill wrapper can also reject
  excessive raw or coalesced range counts before copying. Those range caps are
  disabled by default, but successful stage manifests record the configured
  caps and pass/fail booleans so GLM-5.2 SSD tuning can distinguish a small
  byte budget from an actually coalesced read plan. Prompt prefill and
  generation now roll those observed raw/coalesced counts into their actual
  prefill summaries, so a launch profile can prove both "not too many bytes"
  and "not too fragmented" for the concrete router output.
- Prompt prefill/generation now forward
  `prefill_ssd_read_gib_per_second` and `prefill_max_routed_read_seconds` to the
  actual staged expert copy, so admission-time routed-read estimates are
  rechecked against the concrete router-output stage plan before expert bytes
  are copied. The prompt prefill executor applies this as a cumulative cap,
  passing only the remaining read-time budget to each staged MoE call so
  multi-layer or multi-chunk prompts cannot exceed the total prompt budget one
  individually-safe stage at a time.
- The staged execution manifest is also strict: offsets, lengths, layer ids,
  selected experts, token ids, and compact expert ids must be integers, token
  routes cannot repeat, and route weights must be finite before compact routes
  are written for the runner.
- Runner estimates include assignment tables, token-seen bitmaps, token-block
  buffers, and compact expert allocation before work starts.
- Free-disk checks run before stage, compact-stage, static-capacity route, and
  prompt work files are written. Shared chunk/heap/disk-budget helpers now use
  strict integer validation so direct Python callers cannot turn booleans into
  one-byte chunks or one-byte disk margins.
- Expert I/O planning applies the same strict integer policy to selected
  expert ids, read-advice merge/alignment bytes, fixed static capacity, and
  stage disk margins before staging work begins.
- The first GLM-5.2 17-token copy-chunk experiment shows that the default
  8 MiB stage-copy chunk leaves measurable SSD throughput on the table. With
  the same locked launch/audit guard and generated token id 11, the baseline
  `seventeen-token-prefill-smoke-accel-keycache-profile-replay-coverage.json`
  ran in 177.569s with 11.332s of expert stage copy at 2.401 GiB/s. A locked
  32 MiB profile (`launch-profile-17tok-copy32-accel-keycache.json`) reduced
  the run to 159.199s and stage copy to 9.731s at 2.797 GiB/s. A locked
  64 MiB profile (`launch-profile-17tok-copy64-accel-keycache.json`) reduced
  the run further to 153.466s and stage copy to 9.125s at 2.982 GiB/s. After
  switching the Python copy loop to a reusable buffer with positional `preadv`
  where available, the same locked 64 MiB profile
  (`seventeen-token-prefill-smoke-copy64-preadv-accel-keycache.json`) staged
  the experts in 8.494s at 3.204 GiB/s and finished in 155.674s. A follow-up
  128 MiB run (`seventeen-token-prefill-smoke-copy128-preadv-accel-keycache.json`)
  finished in 163.807s with 8.973s of stage copy at 3.033 GiB/s, so it did not
  beat the 64 MiB setting on this workload. Within the key-cache copy-chunk
  experiment, the 17-token recommendation is therefore 64 MiB plus the reusable
  `preadv` copy path, while longer prompts still need their own guarded replay
  because cache warmth and router distributions can shift the non-copy phases.
- A synthetic 17-token resident-linear calibration over GLM-like shapes showed
  custom Metal beating MPSGraph for the small `6144x256` router gate shape, but
  the real locked GLM smoke contradicted using custom Metal as the launch
  policy. The first explicit custom profile
  `launch-profile-17tok-copy64-custom-keycache.json` was a useful negative
  control but did not actually carry `--prefill-mla-key-cache`; result summaries
  now print `MLA key cache: enabled=N/M` to make this visible. It generated the
  same token id 11 but finished in 230.016s with `enabled=0/78` and MLA
  attention at 42.261s. The corrected custom/key-cache profile
  `launch-profile-17tok-copy64-custom-mla-keycache.json`, audited with
  `--allow-non-accelerated-prefill-launch-audit`, enabled 78/78 MLA key caches
  and improved to 201.993s with MLA attention at 18.568s. It still regressed
  1.298x versus the 64 MiB reusable-`preadv` accelerated run at 155.674s;
  projection time was 59.581s versus 45.220s and stage copy was 11.106s versus
  8.494s. For that backend-split experiment, the recommendation therefore
  remains the accelerated/key-cache profile; synthetic calibration is useful for
  candidate selection but must not replace locked GLM replay.
- A fresh 1-token policy-aware bakeoff now gives a narrower but useful update:
  `custom-metal-vs-accel-keycache-policy-bakeoff-latest.json` selects
  `seventeen-token-prefill-copy64-custom-mla-keycache-baseline-rerun-latest.json`
  over `seventeen-token-prefill-keycache-baseline-replay-latest.json`
  (96.012s versus 110.205s, token `[11]`, replay files ready). The generated
  `selected-replay-custom-metal-keycache-1tok-policy-bakeoff-latest.json` and
  `.sh` preserve that scoped winner. The same conclusion does not apply to
  max2: `max2-custom-metal-policy-bakeoff-latest.json` retains
  `direct-selected-warm-cache-max2-mxfp4-fused-latest.json` because the custom
  candidate generated `[15, 18]` instead of `[18, 18]` and was slower.
- After local shard verification, the scoped selected replay was re-run with
  bounded SSD checking and wrote
  `selected-replay-custom-metal-keycache-1tok-policy-bakeoff-rerun-latest.json`.
  It generated `[11]`, finished in 95.017s, staged 27.21 GiB of routed experts
  with 8.284s of copy at 3.285 GiB/s, reported about 79.64 GiB system memory
  available at launch, and stayed under the 17.37 GiB live-working-set cap.
  `selected-replay-check` still reports `ok=true` for the scoped replay,
  including current memory, stage-temp disk, SSD speed, launch profile hash, and
  launch audit binding checks.
- The fresh custom-metal/key-cache copy128 replay is now audited separately as
  `launch-profile-17tok-copy128-custom-mla-keycache.json` plus
  `launch-audit-17tok-copy128-custom-mla-keycache.json`. It kept the same token
  `[11]`, 17.37 GiB live-working-set cap, and replay-ready launch binding, but
  `copy128-custom-mla-keycache-bakeoff-latest.json` retained copy64: copy128
  finished in 95.925s versus 96.012s for fresh copy64, a 0.999x tie inside the
  two-percent promotion band. Do not promote the larger copy chunk on this
  workload.
- Result summaries now also aggregate real elapsed tensor records by
  layer-normalized tensor suffix and backend. On the current best 64 MiB
  reusable-`preadv` accelerated/key-cache smoke, the largest resident tensor
  groups are `self_attn.o_proj.weight` at 10.368s across 78 custom-Metal calls,
  `mlp.gate.weight` at 8.064s across 75 MPSGraph calls,
  `self_attn.q_b_proj.weight` at 7.861s, `self_attn.q_a_proj.weight` at 7.056s,
  and `self_attn.kv_a_proj_with_mqa.weight` at 6.813s. On the corrected
  custom/key-cache run, the same groups rise to 13.480s, 10.242s, 9.310s, and
  8.977s, with shared-expert projections also slower. This makes the next
  MXFP4 prefill work concrete: optimize the recurrent MLA/output projection
  shapes first, while keeping the router gate on the accelerated policy until
  a locked full replay proves a better backend split. `largerlm result-compare`
  now includes the same tensor suffix rows, so future profile A/Bs can promote
  or reject backend changes from the CLI output without manually diffing the
  raw result JSON. The compare output also emits `profile_recommendation`:
  comparing the current best accelerated/key-cache run against the corrected
  custom/key-cache run yields `decision="prefer_baseline"` and
  `candidate_promotable=false`, with large tensor regressions in
  `self_attn.o_proj.weight`, `self_attn.q_b_proj.weight`,
  `self_attn.q_a_proj.weight`, `mlp.shared_experts.down_proj.weight`, and
  `self_attn.kv_a_proj_with_mqa.weight`. This gives later autonomous profile
  selection a conservative no-promotion gate instead of relying on a human to
  remember which GLM replay was the current safe default. The same decision is
  scriptable with `result-compare --require-candidate-promotable`, which returns
  non-zero for the custom/key-cache candidate while preserving the compare
  diagnostics in stdout/stderr. `result-bakeoff` extends this to multiple
  candidates. It now has a `--promote-only-replay-files-ready` policy, and
  selected replay JSON/script generation enables that policy automatically, so
  a faster candidate whose profile/audit files are missing or stale cannot
  become the replay winner. Using the 64 MiB reusable-`preadv`
  accelerated/key-cache run as baseline, the score-cache MLA result
  `seventeen-token-prefill-smoke-accel-scorecache-mla.json` is the current
  file-ready 17-token winner at 128.811s (`ratio=0.827x`, `delta=-26.863s`),
  while 32 MiB, 128 MiB, and custom/key-cache candidates do not beat the
  baseline and a stale key-cache profile-replay artifact is not promotable under
  the file-ready policy. The bakeoff result now carries `launch_binding` for
  the retained baseline or selected winner; for the copy64 baseline it records
  `launch-profile-17tok-copy64-accel-keycache.json`,
  `launch-audit-17tok-copy64-accel-keycache.json`, the applied profile hash, and
  `safe_to_replay=true`. Because the result includes the original prompt token
  ids and `max_new_tokens`, the same binding now reports `replay_ready=true` and
  a complete locked `generate-prepared-token-ids` argv for replaying that
  measured 17-token smoke. Bakeoff output now has a single `selected` entry; for
  the current file-ready bakeoff that includes score-cache MLA it selects
  `role="candidate"` and writes a locked replay command for
  `launch-profile-17tok-chunk16-accel.json` plus
  `launch-audit-17tok-chunk16-accel.json`. The same bakeoff command can write
  `largerlm.selected_replay.v1` JSON and an executable replay script. Generated
  replay scripts now include `--check-ssd-read-speed` by default, so the
  persistent entry point rechecks current SSD health before model loading. The
  replay JSON itself also records `pre_run_checks.ssd_read_speed`: GLM-scale
  artifacts inherit the prepared manifest's cold-read benchmark byte count and
  enable the check automatically, while tiny debug benchmarks are recorded but
  left disabled unless the CLI flag explicitly requests them. Those
  replay outputs now require `replay_files_ready=true`: the current manifest,
  launch profile, and launch audit must exist, and the launch profile SHA-256
  must still match the measured result's applied-profile hash. The file-ready
  gate now also validates the audit itself: `launch_audit.ok` must be true, and
  the audit's `applied_launch_profile.path`/`sha256` must match the selected
  locked profile. `largerlm selected-replay-check` now provides the same
  no-weight validation as a standalone pre-run check for
  `largerlm.selected_replay.v1`: it reloads the selected result, recomputes the
  current launch binding, verifies the stored/current replay argv match, and
  fails if the launch profile file hash, audit binding, or audit-vs-file hash
  has drifted. It also rechecks the current system memory snapshot against the
  audited request runtime-preflight `required_available_memory_bytes`, failing
  closed before weight loading when current free memory is below the audited
  no-OOM envelope. The same standalone check now rechecks the audited prompt
  prefill stage-temp disk budget: the launch audit's
  `prefill_stage_temp_disk_free.path` and `required_free_bytes` are compared
  with current `disk_usage`, so a replay cannot start after `/private/tmp` has
  fallen below the stage/compact/static-capacity scratch requirement. The same
  no-weight check can now run `--check-ssd-read-speed`, a bounded sequential
  read against the prepared manifest's cold-read benchmark file. The default
  reads 1GiB in 8MiB chunks and requires at least 75% of the manifest's
  `prepare_cold_read_gib_per_second`, failing before any weight loading when
  the current SSD state is too slow for the selected replay. JSON
  `selected-replay-run --dry-run` now withholds `run_argv`/`run_command` until
  every check passes, so automation cannot accidentally exec a replay after a
  failed SSD gate; text-mode failures now include actual/required/baseline
  GiB/s, ratio, requested/measured bytes, chunk size, and benchmark path for the
  same gate. The same no-weight check now verifies the selected launch
  still preserves audited prefill acceleration: required acceleration, the
  MPSGraph runtime probe, request-level accelerated matrix coverage,
  `--require-prefill-acceleration`,
  `--run-mpsgraph-probe`, and replayable MPSGraph threshold flags must remain
  present before `selected-replay-run` can exec the heavy command.
  `largerlm selected-replay-run` wraps that check and then execs the locked
  replay argv, while `--dry-run` prints the checked command without loading
  weights. Non-dry-run execution now acquires
  `.largerlm-selected-replay.lock` in the prepared package before pre-run
  checks and keeps the lock across the final `exec`, so accidental concurrent
  selected replays fail before overlapping SSD reads, page-cache churn, or
  unified-memory allocation. Direct `generate-prepared-token-ids` and
  `generate-prepared-text` now acquire the same lock immediately before
  generation, and selected replay passes an inherited-lock marker through
  `exec` so its child process does not relock against itself. `serve-prepared`
  keeps its in-process request serialization and now uses the same
  prepared-package file lock around each generation request, covering multiple
  server processes and offline commands with one exclusion boundary.
  `bench-prepared-token-ids` and the benchmark API acquire the same lock before
  token generation, so telemetry probes cannot run concurrently with serving or
  replay on the same prepared package. Prepared server `/health` now reports the
  lock path, availability, busy state, and inherited selected-replay marker, so
  operators can diagnose an occupied prepared package without loading weights.
  `/health` also reports `prefill_acceleration_requirement` whenever acceleration
  is required, including the configured backend, MPSGraph probe status, selected
  accelerated backends, and reason code; the safe GLM-5.2 server smoke now shows
  that gate as `ok=true` with `mpsgraph-f32` after startup.
  It can now append
  an output-only `--write-result PATH` for checked `generate-prepared-token-ids`
  replays and can append `--quiet-runner` as a
  logging-only wrapper; JSON dry-runs expose the resulting `run_argv` and
  `run_command` separately from the stored replay argv, so automation can
  verify the exact output path and quiet mode without mutating the selected
  replay artifact. Generated replay scripts now call this wrapper instead of
  directly invoking `generate-prepared-token-ids`. The current persistent replay
  artifacts,
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/selected-replay.json` and
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/selected-replay.sh`, select the
  score-cache MLA winner and pass with `selected replay ok: True`. The script
  now resolves its own directory, returns to the repository root for
  repo-relative result/profile/audit paths, and forwards `--dry-run`, which was
  verified from `/private/tmp` without loading weights. On the latest check,
  current available memory was about 88.4GB against a 44.4GB audited
  requirement, `/private/tmp` had about 424.3GB free against a 5.13GB audited
  stage-temp requirement, and the launch audit recorded 75 accelerated prefill
  matrices on `mpsgraph-f32`. Tightening that gate caught a stale
  `launch-audit-17tok-copy64-accel-keycache.json` that still referenced the
  chunk16 key-cache profile; it was regenerated with the copy64 locked profile,
  17-token request check, runtime preflight, and metal-final-logits check before
  writing the selected replay artifacts.
- The prepared package now also carries
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-selected-safe.sh`, a
  conservative local `serve-prepared` launcher for the same selected 17-token /
  1-new-token envelope. It applies the selected launch profile and launch audit,
  locks the profile, keeps the memory/SSD/GLM/public-shape/prefill acceleration
  guards, enables Metal final logits, pins the local tokenizer backend to
  `tokenizers`, and defaults to `127.0.0.1:8080`. This is a safe API smoke
  entry point, not a broader chat-serving envelope. The script
  now runs `selected-replay-check --check-ssd-read-speed` against the same
  selected replay artifact before starting the server, so current machine and
  SSD state must pass the no-weight replay gate before the API begins accepting
  requests. A real
  host-visible server smoke on 2026-07-02 started it on `127.0.0.1:18080` with
  the selected envelope: `/health` returned `ok=true`, lock `available=true`,
  memory guard `available_ok=true`, GLM 4bit and public GLM-5.2 shape ok,
  launch-audit server caps within envelope, `require_prefill_acceleration=true`,
  and `prefill_acceleration_requirement.ok=true` with MPSGraph probe/runtime ok
  and `mpsgraph-f32` validated. `/v1/models` returned
  `glm-5.2-mxfp4-largerlm-selected-safe`, and an 18-token
  `/generate-token-ids` request was rejected at admission with HTTP 400 because
  the server cap is 17 tokens, proving the smoke server refuses out-of-envelope
  requests before generation. A same-envelope real HTTP `/generate-token-ids`
  smoke then generated token id 11 from the selected 17-token prompt, matching
  the offline selected replay workload. The client observed 456.537s elapsed,
  and the saved generation response reports 448.410s total elapsed. The original
  comparison treated the prompt/generation ids as comparable and flagged
  `possible_system_slowdown=true`; the current stricter `result-compare` also
  checks prompt-prefill plan signatures and correctly marks this old HTTP smoke
  as not directly comparable with the selected baseline because the baseline
  records 780 custom-Metal linear calls while the HTTP artifact records 1230.
  Treat
  `server-http-selected-safe-smoke.json` and
  `server-http-selected-safe-response.json` as API-path correctness evidence,
  not a promoted performance baseline. Successful server token/text responses
  now also include the request admission check and launch-audit envelope, so
  future HTTP smoke artifacts carry the guard evidence that admitted generation.
- The prepared package now also carries
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-canonical-safe.sh`, a
  conservative local `serve-prepared` launcher for the package-level
  `launch-profile.json`/`launch-audit.json` custom-Metal safety baseline. It
  runs a local `checkpoint-status --verify-local-headers --require-complete
  --require-clean` check before server startup, then applies and locks the
  canonical profile, requires the launch audit, prepared memory profile, GLM
  4-bit and public-shape gates, caps requests at 2048 prompt tokens and one new
  token, and defaults to `127.0.0.1:8081`. A bounded HTTP smoke against that
  launcher generated token id 15 from prompt token 0 in 12.224s through
  `/generate-token-ids`; the response is saved as
  `server-http-canonical-safe-smoke.json` and includes a launch-audit server
  envelope with `server_memory_guard_ok=true`, 2048 audited prompt tokens, one
  audited new token, 128GiB total memory, and the canonical profile sha
  `b9477c40416d01cd85820a3c6e943f8db601ad5ddc105ebad3cee4890c005056`.
- The text/OpenAI serving path now supports GLM's local `chat_template.jinja`
  without executing tokenizer remote code. `render_chat_prompt` first uses a
  tokenizer-provided chat renderer when available, then falls back to a sandboxed
  local Jinja renderer for `chat_template.jinja` or `tokenizer_config.json`
  `chat_template`; the rendered prompt is still encoded by the real tokenizer
  backend. A real 2026-07-02 `/v1/chat/completions` smoke through
  `serve-selected-safe.sh` used the prompt `hi hi hi hi hi`, which renders to
  exactly 17 GLM chat tokens, requested one new token, and returned HTTP 200 in
  480.587s. The saved artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-selected-safe-openai-chat-smoke.json`;
  it reports `object=chat.completion`, assistant content `3`, generated token
  id 18, `usage.total_tokens=18`, `request_check.ok=true`,
  `chat_template_backend=local-jinja`, `tokenizer_backend=tokenizers`, and a
  bound `launch_audit_envelope`. This proves the OpenAI-compatible chat wrapper
  is functional inside the audited 17/1 envelope, while also confirming the
  current service path is still far too slow for interactive use.
- `result-summary` and `result-compare` now unwrap saved HTTP smoke artifacts
  directly. They understand both the newer OpenAI-style
  `response.largerlm.token_result` shape and the older `/generate-token-ids`
  smoke where `response` is itself the token result. Re-running
  `result-summary` on the OpenAI chat artifact reports client elapsed 480.587s,
  token-result elapsed 469.210s, stage-copy throughput 1.779GiB/s, and about
  238.513s unattributed. Re-running `result-compare` directly on the older
  `/generate-token-ids` HTTP wrapper versus the selected 128.811s baseline now
  reports `prefill_plan_match=false` even though prompt tokens, max-new tokens,
  and generated ids match. `result-summary` now exposes the plan as top-level
  `prefill_plan_signature`, and text compare prints the mismatch directly:
  the selected baseline records `custom-metal=780,fused-metal=312,mpsgraph-f32=75`
  for the 2x16 prefill plan, while the HTTP smoke records
  `custom-metal=1230,fused-metal=312,mpsgraph-f32=75` with the same expert-read
  and stage-byte envelope. The elapsed ratios remain useful diagnostics rather
  than promotion evidence: total elapsed is 3.481x slower, custom-Metal linear
  time 5.068x slower, MPSGraph linear time 4.104x slower, fused-Metal linear time
  3.566x slower, and stage-copy time 3.479x slower. Future promotable smokes
  need both token-level workload equality and prompt-prefill plan equality.
- A current replay through `selected-replay-run --write-result
  artifacts/glm-5.2-mxfp4/largerlm-prepared/seventeen-token-prefill-smoke-current-safe-replay.json`
  confirmed the selected artifact is still runnable and memory-safe on the M5
  Max: the guarded run generated token id 11, kept the applied launch profile
  locked, finished with about 82.40GiB system memory available, and recorded the
  same 1713 unique runner command signatures as the 128.811s selected result.
  It took 444.213s, however, and `result-compare` against the selected
  score-cache MLA baseline reported `possible_system_slowdown=True` with a
  3.543x median sentinel ratio. Treat this as current-system slow-path
  evidence, not a promoted performance result. The slow-path summary shows
  stage-copy throughput at 2.016GiB/s versus 6.569GiB/s in the selected result;
  adjacent bounded `disk-read-benchmark` probes measured 1GiB expert reads at
  3.158GiB/s with 8MiB chunks and 4.656GiB/s with 32MiB chunks, below the
  prepared manifest's 5.9145GiB/s cold-read calibration. Future full replays
  should run `selected-replay-check --check-ssd-read-speed` when the machine is
  thermally or I/O loaded, and should avoid promoting candidates whose sentinel
  slowdown points to host state rather than algorithmic changes. A default
  pre-run SSD check later passed on the selected replay envelope with the
  manifest baseline at 5.9145GiB/s, a 4.4359GiB/s requirement, and a bounded
  1GiB read at 10.256GiB/s; a forced slow-gate dry-run returned 1 and omitted
  `run_argv`, confirming the wrapper fails closed before the heavy command is
  exposed.
- A stricter request-profile/audit path now protects mixed auto prefill
  backends. `inspect-prepared --write-launch-profile` no longer writes an
  explicit `--prefill-linear-backend mpsgraph-f32` into request launch profiles
  when the checked request's configured/effective backend is `auto`; it keeps
  `--require-prefill-acceleration`, `--run-mpsgraph-probe`, and the MPSGraph
  threshold flags while recording `prefill_linear_backend_policy="auto"` in
  `sections.prefill_acceleration_flags`. That prevents MXFP4 resident matrices
  from being forced through MPSGraph while still requiring router BF16 GEMMs to
  prove accelerated coverage.
- The canonical GLM-5.2 prepared `launch-profile.json`/`launch-audit.json` was
  refreshed as the conservative custom-Metal fallback path after BF16
  calibration: the locked audit is bound to profile sha
  `b9477c40416d01cd85820a3c6e943f8db601ad5ddc105ebad3cee4890c005056`, allows
  non-accelerated prefill evidence explicitly, keeps the prepared memory profile
  guard, and now audits a 2048-token / 8-new-token envelope with chunk=17. The audit
  must be generated with host `sysctl hw.memsize` visible so
  `system_total_memory_bytes` is populated for strict `--require-launch-audit`
  replay. The refreshed audit records 128GiB total memory, about 81.63GiB
  available, about 17.37GiB live set, and about 41.37GiB required available
  memory. Worst-case routed prefill I/O is still about 22.41TiB, so this is a
  safety baseline rather than the selected performance replay. A current
  `minimal-smoke.json` replay against that canonical profile generated token id
  15 from prompt token 0 in 12.442s, read about 11.68GiB, and is file-ready
  against the refreshed launch profile/audit binding. The canonical HTTP
  launcher intentionally still defaults to a 1-new-token service cap inside the
  wider audit envelope; setting `LARGERLM_CANONICAL_MAX_NEW_TOKENS_CAP=8`
  enables the wider cap for explicit experiments. A 3-token HTTP attempt stayed
  within the audit envelope but timed out at the client after several minutes,
  so multi-token API serving remains a performance target rather than a
  promoted path. The refreshed 1-token HTTP smoke generated token id 15 with
  `audited_max_new_tokens=8`, `server_max_new_tokens_cap=1`, and
  `server_memory_guard_ok=true`, but took 78.923s while a bounded 1GiB SSD probe
  measured only 4.49GiB/s against the 5.9145GiB/s prepared cold-read baseline.
  `serve-prepared --check-ssd-read-speed` now exposes that bounded startup gate
  directly: it benchmarks the prepared cold-read file before server startup and
  fails closed below the configured baseline ratio. `serve-canonical-safe.sh`
  enables the gate by default at 0.75x of the prepared baseline; a later
  startup check measured 5.247GiB/s and allowed the server to start.
  `serve-prepared` and the canonical launcher now forward expert
  read-advise hints to routed decode/prompt-prefill reads; the canonical
  defaults are `merge_gap=128KiB` and `align=4KiB`. A follow-up
  `server-http-canonical-readadvise-smoke.json` replay confirmed the health
  path reports those values, generated token id 15, stayed within the launch
  envelope with `server_memory_guard_ok=true`, and read the same 11.206GiB of
  routed experts, but still took 80.003s generation time and 84.335s over HTTP.
  A same-envelope direct CLI replay,
  `direct-canonical-readadvise-smoke.json`, also generated token id 15 and took
  79.460s with the read-advise flags present in the recorded runner command, so
  this is current direct decode/runner/I/O evidence rather than HTTP-wrapper
  overhead. A no-read-advise direct control,
  `direct-canonical-no-readadvise-smoke.json`, took 79.875s with the same token
  and read volume, so the read-advise hints are neither the main regression nor
  a promoted performance fix. The older 12.442s `minimal-smoke.json` remains
  correctness evidence, but current performance work should instrument the
  batched decoder runner's per-layer read/dispatch/compute time before treating
  it as an active speed baseline. The runner now writes optional per-layer
  decoder stage timings into `--output-report-json`, and
  `direct-canonical-instrumented-smoke.json` recorded the pre-cache breakdown:
  79.791s total, 76.891s summed layer time, 47.826s attention sub-stages, and
  29.032s MLP. Across 78 layers, average stage time is about 0.323s MLA
  attention, 0.179s attention projections, 0.086s attention output, 0.023s
  RoPE, and 0.381s MoE MLP for MoE layers. The runner now caches the shared
  `expert_kernel_source()` library and compute pipeline states within each
  process; `direct-canonical-cached-pipeline-smoke.json` generated the same
  token id 15 and reduced the same canonical direct replay to 19.881s total,
  18.812s summed layer time, 12.537s attention sub-stages, and 6.269s MLP.
  Cached MoE layers average 0.246s, with about 0.076s MLA attention, 0.055s
  attention projections, 0.022s attention output, and 0.083s MLP. The next
  `server-http-canonical-cached-pipeline-smoke.json` replay confirmed the same
  improvement through the local API: token id 15, 19.949s generation time,
  20.789s HTTP wall time, `server_memory_guard_ok=true`, and
  `server_caps_within_envelope=true`. A follow-up shared-command-queue pass
  reuses the runner's `MTLCommandQueue` in the same process:
  `direct-canonical-shared-queue-smoke.json` reduced the canonical direct replay
  to 12.509s total with 11.750s summed layer time, while
  `server-http-canonical-shared-queue-smoke.json` reached 9.756s generation
  time and 10.566s HTTP wall time with the same token id 15 and both server
  safety gates true. A follow-up runner housekeeping pass caches read-only
  layout/cache JSON dictionaries by path within the same process, avoiding
  repeated parsing of the resident, expert, and decode-cache metadata without
  adding a large application-owned expert cache.
  `direct-canonical-json-cache-smoke.json` reduced the same replay to 10.348s
  total with 9.746s summed layer time, 6.450s attention sub-stages, 3.292s MLP,
  the same token id 15, and the same 9.48GiB live peak under the 17.37GiB cap.
  The local API replay
  `server-http-canonical-json-cache-smoke.json` reached 9.933s generation time
  and 10.753s HTTP wall time; its `request_check` was ok with the launch-audit
  envelope, decode routed-read guard, and available-memory guard all true.
  Caching resident tensor metadata indexes by layout pointer and layer/name
  removed another repeated JSON scan from each attention/MLP subcommand without
  caching weight payloads:
  `direct-canonical-tensor-index-cache-smoke.json` reached 7.418s total with
  6.864s summed layer time, 4.469s attention sub-stages, 2.391s MLP, and the
  same token id 15. The matching local API replay
  `server-http-canonical-tensor-index-cache-smoke.json` reached 7.282s
  generation time and 8.101s HTTP wall time, with `request_check.ok`,
  launch-audit envelope, decode routed-read guard, and available-memory guard
  all true. Caching validated resident backing paths in the same process
  removed repeated resident span sorting/stat checks while preserving the same
  fail-closed first validation. `direct-canonical-weightpath-cache-smoke.json`
  reached 6.559s total with 6.013s summed layer time, 3.924s attention
  sub-stages, 2.086s MLP, and the same token id 15; the matching API replay
  `server-http-canonical-weightpath-cache-smoke.json` reached 6.577s generation
  time and 7.402s HTTP wall time with the same request guards true.
  Decode-cache layout validation, segment lookup, and cache-file stat caches
  were neutral on the current 1-token smoke:
  `direct-canonical-cachelayout-cache-smoke.json` reached 6.576s total and
  `server-http-canonical-cachelayout-cache-smoke.json` reached 6.554s
  generation time / 7.374s HTTP wall time, with the same token and request
  guards. Keep this as repeated-validation cleanup for multi-token and
  cache-heavy paths rather than a promoted 1-token speedup. A first small
  fused/streamed attention cleanup now splits `attn_q_b.f32` and applies RoPE
  for `q_rope` in CPU memory inside the decoder-layer runner, removing the
  old `q_rot`/dummy-k/dummy-k-RoPE temp files and one tiny RoPE kernel dispatch
  per layer while leaving standalone `--run-rope` unchanged.
  `direct-canonical-cpu-rope-smoke.json` reached 6.512s total, 5.966s summed
  layer time, and 3.850s attention sub-stages with the same token id 15 and
  memory envelope. The matching API replay
  `server-http-canonical-cpu-rope-smoke.json` reached 6.525s generation time
  and 7.355s HTTP wall time, again with request guards true. The next small
  file-boundary cleanup keeps the MLA attention value tensor in memory for the
  attention-output projection instead of writing and rereading `attn_value.f32`
  inside each decoder layer. `direct-canonical-attn-value-memory-smoke.json`
  reached 6.504s total with 5.959s summed layer time and 3.847s attention
  sub-stages; `server-http-canonical-attn-value-memory-smoke.json` reached
  6.463s generation time and 7.279s HTTP wall time, with the same token id 15
  and request guards true. A follow-up intra-layer residual boundary cleanup
  passes the attention-output vector directly into the MLP block, avoiding the
  `attn_out.f32` write/read boundary while keeping standalone attention-output
  and MLP commands file-compatible. The direct
  `direct-canonical-attn-out-memory-smoke.json` replay reached 6.525s total,
  5.975s summed layer time, 3.861s attention sub-stages, and 2.111s MLP
  sub-stages with token id 15. The matching
  `server-http-canonical-attn-out-memory-smoke.json` replay reached 6.518s
  generation time and 7.357s HTTP wall time with request guards and decode
  read-time guards true. This is primarily a temporary-file surface reduction:
  the one-token timing difference is inside normal run-to-run noise. The same
  layer-internal cleanup was extended to pass CPU-split `q_nope`/`q_rope`
  tensors directly into MLA attention, avoiding those two temporary f32 files
  as well. `direct-canonical-q-memory-smoke.json` reached 6.560s total,
  6.000s summed layer time, 3.869s attention sub-stages, 2.128s MLP sub-stages,
  and reduced summed `split_q` time to 0.003s with token id 15. The matching
  `server-http-canonical-q-memory-smoke.json` replay reached 6.517s generation
  time and 7.392s HTTP wall time with request guards and decode read-time
  guards true. The q-memory path improves the tiny split/RoPE boundary but does
  not move whole-token latency under the current SSD/Metal scheduling noise. A
  final small projection-boundary cleanup returns `attn_q_b` from the
  single-token attention projection path in memory and runs the same in-process
  split/RoPE without writing the projection debug directory. The direct
  `direct-canonical-proj-q-memory-smoke.json` replay reached 6.427s total,
  5.889s summed layer time, 3.758s attention sub-stages, 2.128s MLP sub-stages,
  0.465s summed projection time, and 0.002s summed `split_q` time with token id
  15. The matching `server-http-canonical-proj-q-memory-smoke.json` replay
  reached 6.464s generation time and 7.311s HTTP wall time, with request guards
  and decode read-time guards true. This gives a modest projection-stage
  reduction while preserving the same one-token memory/read envelope. The next
  layer-boundary cleanup threads intermediate hidden vectors through
  `run_decoder_layers` in memory, so non-final layers no longer write
  `layer_XXXX.f32` just for the following layer to read it back; the final
  hidden vector is still written for logits and CLI compatibility. The direct
  `direct-canonical-layer-hidden-memory-smoke.json` replay reached 6.403s
  total, 5.874s summed layer time, 3.768s attention sub-stages, 2.099s MLP
  sub-stages, and token id 15. The matching
  `server-http-canonical-layer-hidden-memory-smoke.json` replay reached 6.338s
  generation time and 7.211s HTTP wall time, with request guards and decode
  read-time guards true. The optimization only keeps one hidden f32 vector live
  between layers, so it does not change the recorded large-model memory/read
  envelope.
  The next performance target is therefore fused/streamed
  attention-projection/MLA/MLP execution and late-layer I/O/thermal stability,
  not HTTP overhead, repeated metadata parsing/scanning, or macOS read-ahead
  hints.
- The strict GLM-5.2 replay
  `seventeen-token-prefill-smoke-accel-strict-latest.json` completed with
  `launch-profile-17tok-chunk16-accel-strict.json` plus
  `launch-audit-17tok-chunk16-accel-strict.json`: generated token id 11, Metal
  final logits, applied-profile lock match, 17-token prompt, one generated
  token, MPSGraph on the 75 router BF16 GEMMs, and custom Metal on MXFP4
  resident/expert paths. Runtime preflight stayed inside the no-OOM guard
  (17.37GiB live peak; about 82.41GiB available at generation admission), and
  the run read 36.79GiB during prefill with the audited 33.82s read cap still
  marked ok. The run took 426.567s; compared with the score-cache MLA selected
  replay at 128.811s, `result-compare` reported `possible_system_slowdown`
  with median sentinel ratio 3.448 and `candidate_promotable=false`.
  `result-bakeoff --promote-only-replay-files-ready
  --require-selected-replay-ready` therefore retained the score-cache MLA
  selected replay and treated the strict run as replay-gate evidence rather
  than a new performance baseline.
- `result-summary` now reports recorded runner command groups. On the current
  score-cache MLA selected replay, the 128.811s result exposes 2103 command
  records but 1713 unique command signatures: 855
  `--run-resident-linear-batch`, 390 `--run-rmsnorm-batch`, 156
  `--run-mla-attention-batch`, and 78 each for `--run-attn-projections`,
  `--run-rope-batch`, `--run-attn-output`, and `python-rope-singleton`. This
  makes the next optimization target measurable: layer/block fusion should
  reduce unique command signatures and subprocess/Metal setup churn before
  another full smoke is promoted as a performance result.
- `result-summary` now also promotes routed-MoE MXFP4 split-kernel diagnostics
  when a run was captured with `LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING=1`.
  It reports aggregate gate/up/SwiGLU versus down/weighted-add seconds and
  shows those sub-kernel timings in the slow-layer rows. The same hint now
  emits a bounded `glm_moe_tile_sweep.py` command for the hottest routed layer,
  with selected expert ids when available, so the next M5 routed expert kernel
  experiment can start from the standard summary output and still require full
  replay/bakeoff proof before promotion. The sweep JSON now carries
  `schema=largerlm.glm_moe_tile_sweep.v2` plus `config_comparison`, so bounded
  microbench wins can be promoted into the full-replay queue without changing
  defaults prematurely.
- A fair layer-19 repeat-4 vector-SwiGLU check against the current default
  group32/auto path is recorded at
  `glm-moe-layer19-vector-swiglu-vs-default-smoke-17tok-repeat4-latest.json`.
  The latest sample had small microbench leads for both group32-off variants:
  `tile1_vector_silu` about 0.02226s, `tile1_scalar_silu` about 0.02230s, and
  default auto about 0.02382s. A policy-compliant custom-metal/keycache full
  replay bakeoff (`moe-kernel-custom-mla-keycache-bakeoff-latest.json`) retained
  the fresh baseline, however: baseline 96.012s, group32-off 97.139s, and
  vector+group32-off 100.119s, all generating `[11]`. No routed-MoE kernel
  default changes.
- Batch `--run-attn-projections` is now wired through `largerlm-runner` for the
  standard GLM attention projection suffixes, but it is intentionally
  experimental. The Python prefill path keeps the proven single-token fused
  projection as the default and only enables staged batch projection fusion when
  `LARGERLM_EXPERIMENTAL_BATCH_FUSED_ATTN_PROJECTIONS=1` is set, while still
  writing the older `q_b_proj.f32`/`kv_a_proj_with_mqa.f32` aliases expected by
  CLI smoke tools. A 17-token GLM-5.2 MXFP4 replay written to
  `seventeen-token-prefill-smoke-accel-batch-fused-attn-proj.json` generated
  token id 11 and reduced unique runner command signatures from 1713 to 1323
  (`--run-attn-projections` unique signatures became 156), but it took
  411.103s and `result-compare` flagged `possible_system_slowdown` with a
  3.416x median sentinel ratio. Do not promote that artifact; keep the
  128.811s score-cache MLA selected replay as the baseline. The next fused
  attempt should avoid staging the projection as several helper primitives
  inside one runner process, or should fuse farther across projection/RoPE/MLA
  to reduce both command signatures and Metal setup/kernel churn.

## Apple LLM In A Flash

Reference: <https://arxiv.org/abs/2312.11514>

This paper is the earlier work that made the flash-memory direction concrete.
The key ideas to keep available are windowing and row-column bundling: use
activation sparsity to read fewer model parameters, and arrange on-disk data so
random small reads become larger useful reads.

For GLM MoE, the direct analog is expert bundling rather than dense-neuron
bundling. The runtime should pack the full gate/up/down triplet of each expert
contiguously, then optionally cluster experts that co-activate in the same
layer.
`LLMSCAP1` static-capacity route tables are the current fixed-shape boundary:
the Metal runner can consume expert-major assignments directly without
rebuilding per-token JSON routes, and prompt/generation commands can request
`auto` static capacity bounded by the active prompt chunk. The production
prompt/generation path now writes only the compact binary route table by
default, leaving static-capacity JSON as a low-level debug artifact so large
chunks do not allocate a huge Python JSON payload before execution. The binary
table is still validated in Python after writing and before runner dispatch,
including header, length, route-record, and plan-count checks, so the JSON-free
path does not become an unchecked Metal input. Offline request inspection
includes those binary route-table bytes in the routed stage/compact
temporary-disk estimate, so long prompts expose the route-table footprint before
staging begins.

## FlexGen

Reference: <https://arxiv.org/abs/2303.06865>

FlexGen is the earlier general offload system to keep in mind. It frames LLM
inference as a placement and scheduling problem over GPU, CPU, and disk memory,
then searches for high-throughput tensor movement plans under constrained GPU
memory. It also combines offload with 4-bit weight/cache compression.

The LargerLM target is different: local interactive GLM MoE on Apple unified
memory, not high-throughput batched OPT serving on a discrete GPU. The useful
lesson is still the same boundary discipline: every tensor movement tier needs
an explicit budget, and batch/prompt chunk sizing must be chosen together with
the offload schedule. LargerLM should therefore keep the current pattern of
request-time admission checks, modeled SSD read volume, cache write volume,
stage-file bytes, and live working-set limits rather than adding opportunistic
prefetch that is invisible to the planner.

## oMLX

Reference: <https://github.com/jundot/omlx>

Useful components and patterns:

- `glm_moe_dsa` support already exists as a patch around the pinned `mlx-lm`
  runtime. It confirms the GLM path has DSA/indexer sharing semantics.
- MLA-style cache accounting is different from regular MHA/GQA cache math. A
  planner must use `kv_lora_rank + qk_rope_head_dim` plus optional indexer cache
  dimensions, not `num_kv_heads * head_dim`.
- Existing SSD cache code is good reference for bounded async disk writes and
  cache observability, but expert weights should not be treated like KV cache.
- oQ quantization code has MoE/router-aware policies that are useful when
  deciding which resident tensors must avoid aggressive quantization.
- Generation now has a top-level live working-set guard in addition to
  per-kernel scratch checks; the first `serve-prepared` layer reuses that
  policy for admission control and serializes access to the shared decode
  cache. Batch-prefill admission counts cache read/write and bounded stage-copy
  phases in that live estimate so large prompt cache writes are rejected before
  staging begins, passes the same live-memory caps into direct prompt prefill,
  rechecks before every prompt chunk and every selected prefill layer, and
  rechecks after prefill or decode layers before final logits. If that guard
  trips inside a user-provided prompt work directory, chunk scratch directories
  are cleaned unless they contain protected output paths. Runtime cap inputs are
  now validated as finite positive or non-negative values before MiB-to-byte
  conversion, preventing bad CLI/server limits from weakening the guard.
- Plan/preflight/prepare/pack/decode-cache CLI GiB/MiB budgets now use the same
  finite-value check before conversion, so malformed `nan`/`inf` inputs return
  clean CLI errors instead of uncaught Python conversion failures.
- Prepared manifest loading now preserves the validated expert, resident,
  decode-cache layout, and cache-file byte counts. `inspect-prepared` and
  `/health` expose those storage totals plus recommended available-memory
  budget before any model-weight mapping, which makes GLM-5.2 launch checks
  easier to automate without risking a large allocation. The prepared runtime
  profile now treats a known shortfall against that recommended live+reserve
  available-memory envelope as a pre-launch failure, not just advisory output.
  The same backing validation now rejects overlapping expert-component or
  resident-tensor byte spans and decode-cache backing files whose size does not
  exactly match the cache layout, catching corrupt or stale prepared artifacts
  before any runner reads the backing files.
- Direct prepared server generation and the prepared benchmark Python API now
  enforce the prepared runtime profile before running, matching the CLI guard
  that rejects packages prepared for a larger unified-memory envelope or reserve
  when the current machine is below that profile. Recorded prepared memory
  profiles now also fail closed when current system memory cannot be inspected,
  rather than treating an unverified envelope as safe.
- Prepared health and inspection now emit `suggested_launch_guard_flags` when a
  manifest records live-working-set and minimum-free-unified-memory
  recommendations, making the prepare-time 128 GiB safety envelope directly
  reusable as launch arguments.
- When GLM 4bit readiness passes, prepared health also emits
  `suggested_decode_guard_flags` from the config-derived per-token routed expert
  read, so decode SSD caps can be pinned before a request-specific preflight.
- Prepared health now emits `suggested_launch_profile`, a de-duplicated argv
  profile that combines manifest memory launch guards, prepared SSD read-speed
  defaults, backend probe flags, tuned prefill runtime policy, selectable prefill
  acceleration flags, GLM 4bit readiness guards, and model-level decode guards.
  The tuned prefill runtime policy now includes explicit
  `--prefill-linear-backend` and, when enabled,
  `--prefill-router-hybrid-margin-threshold` entries, so calibration-derived
  backend/router fallback decisions are preserved when
  `inspect-prepared --write-launch-profile` turns them into a prepared-bound
  safe profile.
  `inspect-prepared` adds a `request_launch_profile` after a successful
  prompt/chat request check, folding in the prompt-specific prefill read/stage
  guard profile.
- `prefill-plan` can now save the same tuned prefill runtime policy into its
  prefill-only launch profile, so config-only GLM-5.2 planning runs can seed
  MPSGraph thresholds and accelerated-FLOP coverage gates before a prepared
  package exists.
- `inspect-prepared --write-launch-profile` now writes that best available
  profile to JSON, and prepared generation/benchmark/serve/inspect commands can
  use `--apply-launch-profile` to expand the saved argv before later explicit
  command-line overrides. This makes the safe preflight profile reusable instead
  of relying on copy-paste from JSON output. Saved profiles carry prepared
  identity metadata, and apply-time validation rejects profiles whose layout
  bytes, context, quantization, or model-config hash do not match the current
  prepared package. New profiles also bind plan-derived prepare flags by
  recording `prepare_flags_applied`, `prepare_flags_source`, and
  `prepare_flags_sha256` in the prepared identity; local prepare-flags paths stay
  out of the identity so moved prepared packages can still replay safely.
  Profiles now expose `prepared.identity_strength` and
  `identity_warnings`, so legacy packages without layout config hashes are
  visible as weakly bound instead of silently looking equivalent to modern
  `prepare-glm` outputs. Required launch audits now additionally reject weak
  prepared identities, so production GLM-5.2 launch gates must be hash-bound even
  though normal profile replay can still inspect old packages.
- `--lock-launch-profile` now rejects command-line overrides that change any
  argv flag carried by an applied profile, giving real larger-than-memory GLM
  launches an opt-in exact replay mode for measured safety envelopes.
- The same programmatic surfaces now expose GLM 4bit readiness gates:
  `PreparedServerConfig(require_glm_4bit=True)` and
  `benchmark_prepared_token_ids(..., require_glm_4bit=True)` reject prepared
  artifacts whose expert layout does not match the config-derived affine-int4
  GLM MoE shape. Readiness treats expert-layout `num_layers` as the total
  model layer count and validates the packed expert `layers` array against only
  the config-derived MoE layer ids. It also checks the exact affine-int4
  `component_order` and every expert component's packed offset within the slot,
  so dense-prefix GLM packages are accepted without losing coverage for any
  routed layer or allowing shifted scale/bias regions.
- GLM 4bit readiness also binds the expert and resident layout `model_type`
  fields to the loaded config, matching the existing decode-cache model-type
  check so the three runner-facing layouts cannot silently drift apart.
- GLM 4bit readiness now requires resident backing to be exact-sized as well:
  `resident_weight_file_bytes` must match `resident_layout_total_bytes`, and the
  evidence flows into launch audits with `resident_weight_file_exact_size`.
- Public GLM-5.2 shape is now an optional launch gate as well:
  `--require-public-glm-5-2-shape`,
  `PreparedServerConfig(require_public_glm_5_2_shape=True)`, and
  `benchmark_prepared_token_ids(..., require_public_glm_5_2_shape=True)` require
  GLM 4bit readiness plus an exact match to the public GLM-5.2 config shape.
  The strict public GLM-5.2 gate now rejects the debug-only missing DSA indexer
  override across CLI, server config, and benchmark API entry points. It also
  checks public GLM-5.2 runner semantics such as `bfloat16`, `hidden_act=silu`,
  `attention_bias=false`, and zero attention dropout, while the broader GLM
  4bit readiness gate rejects explicit unsupported attention bias or non-SiLU
  activations before launch.
  `preflight-glm` now reports the same public shape diagnostic from config-only
  metadata and can enforce it with `--require-public-glm-5-2-shape` as the
  earliest no-write GLM-5.2 gate. That hard gate also requires the preparation
  target to remain 4bit (`quant_bits=4`), matching the GLM-5.2 bring-up target
  instead of accepting a larger expert format. When the gate fails, preflight
  stops before safetensors metadata scans, layout estimates, tokenizer loading,
  or disk-budget checks, so a wrong public-shape or non-4bit target cannot
  accidentally start a heavier checkpoint inspection path.
  `prepare-glm` reports the same public shape diagnostic during dry-run and
  execute; its matching `--require-public-glm-5-2-shape` flag fails before
  writing when a preparation run is meant specifically for GLM-5.2.
  The `serve-prepared` CLI now front-loads the same public-shape gate before
  calling the server launcher, matching generation and benchmark entry points.
  Benchmark API failures now reuse the same mismatch-field preview as the
  server gate, so automated GLM-5.2 sweeps can report config drift without
  parsing the full readiness JSON.
  Prepared health emits the matching launch-profile section only when that shape
  check passes, so a GLM-5.2-specific profile is not silently reused for another
  GLM MoE package.
- GLM 4bit readiness now also validates resident layout metadata for the
  tensors generation needs before it reaches decode: global embedding/final
  norm, attention projections/norms, MoE routers, dense/shared MLPs, and full
  DSA indexer tensors where configured. It rejects resident layouts that still
  contain config-declared routed expert tensors, so expert weights cannot be
  accidentally packed into the resident working set. It also respects
  `tie_word_embeddings=false`, requiring a real resident `lm_head.weight`
  instead of silently falling back to `embed_tokens.weight`. This catches
  partially or incorrectly packed resident layouts without reading large weight
  files.
- `prepare-glm --execute` now runs that same GLM 4bit readiness gate after the
  manifest/backing-file validation and before returning success. If the final
  self-check fails, newly written expert, resident, cache, and manifest files
  are cleaned up like any other prepare-stage failure.
- Prepared manifests now retain prepare-time public GLM-5.2 target evidence:
  whether the strict gate was required, whether the public shape matched, and
  the mismatched field list. Manifest loading rejects self-contradictory public
  shape metadata, and prepared health exposes the same fields in
  `prepared_storage`.
- Raw BF16/F16/F32 expert quantization now streams each component as row blocks
  and writes generated weight/scales/biases blocks directly into the packed
  slot. The heap model adds only the largest generated row-block output to the
  normal metadata/copy-chunk allowance, so local 4bit conversion no longer
  requires a whole raw expert matrix or whole generated slot to fit in heap.
- Raw fused gate/up expert tensors (`gate_up_proj`, `gate_up`, or `w13`) are
  now sliced into separate gate/up source windows during discovery when their
  per-expert shape matches `[2 * moe_hidden_size, hidden_size]`. This keeps
  real-checkpoint compatibility in the streaming quantizer without widening the
  runner's affine-int4 slot contract.
- Pre-quantized fused gate/up expert tensors now get the same split-window
  treatment for `weight`, `scales`, and `biases`, including F16/BF16 metadata,
  so MLX-style fused affine-int4 checkpoints can still publish the standard
  nine-component runner slot.
- Resident packing now applies the same split-window idea to dense/shared
  fused gate/up tensors, publishing separate gate/up resident layout entries
  while copying bounded slices from the original safetensors shard. Preflight
  uses that normalized resident view, so dry-run evidence matches the prepared
  layout the runner will actually consume. Prepare manifests and health output
  retain the source/expanded tensor counts and expanded byte total for this
  resident rewrite. Resident dense/shared `w1/w3/w2` component aliases now
  normalize to `gate_proj/up_proj/down_proj` through the same metadata-only
  path, with source/renamed counts and bytes retained in the executed prepare
  manifest and `prepared_storage` health. Strict launch audit now includes a
  resident alias rewrite check, so passing artifacts must keep those one-to-one
  alias counts and fused gate/up expansion counts self-consistent when present.
  Launch-profile prepared identity now binds the same resident rewrite fields
  plus the raw expert-pack heap envelope whenever they carry recorded evidence,
  preventing stale GLM launch profiles from replaying across prepare-safety
  drift. Required launch-audit consumers also compare the resident rewrite
  check payload back to the current prepared manifest, so an `ok` audit artifact
  cannot be hand-edited away from the actual prepared evidence. Manifest
  loading now rejects resident alias/fused rewrite byte totals that exceed the
  validated resident layout size, closing another stale or hand-edited evidence
  path before launch-profile matching runs.
- The generated affine-int4 row-block buffers now stay mutable through the
  `pwrite` boundary instead of being copied into immutable `bytes` objects,
  removing an unmodeled transient duplicate of weight/scales/biases output
  during GLM-5.2 raw expert conversion.
- Executed `prepare-glm` manifests now persist the expert-pack heap envelope:
  chunk size, estimated peak heap, max heap, raw source block, generated output
  block, extra modeled heap, and rows per block. Manifest loading validates the
  pack peak against its cap, rejects raw blocks larger than that peak, and
  requires the extra heap to cover generated output plus any source-block
  overflow beyond the copy chunk. Internal `largerlm-affine-int4` expert layouts
  also require the complete pack-heap evidence set at manifest-load time, using
  layout quantization as the effective type if the manifest-level field is
  missing. Prepared health exposes the fields so raw GLM-5.2 conversion safety
  is still visible after preparation. Required
  launch-audit replay now treats the same envelope as hard safety evidence for
  internal `largerlm-affine-int4` packs and rejects stale artifacts whose
  audited pack peak, cap, raw block sizes, or row-block count no longer match
  the current manifest.
- Direct token generation and prepared benchmarks now reject non-integer token
  IDs and integer controls before layout reads, closing a quiet `int()` coercion
  path that could hide bad prompt data in offline experiments.
- Direct prompt prefill now enforces the same rule on prompt token ids, chunk
  size, start position, layer sets, attention/DSA dimensions, static capacity,
  and stage disk margins before creating work files.
- The lower-level routed prefill wrappers now do the same for batch tokens,
  top-k/max-k, router group counts, stage/read-advise controls, static capacity,
  and stage disk margins before runner launch.
- Config loading now rejects non-integer model shape fields, non-numeric scalar
  fields, non-boolean flags, malformed GLM-5.2 DSA schedule freq/offset fields,
  and full/shared DSA schedules that omit `index_head_dim`, `index_n_heads`,
  `index_topk`, or `q_lora_rank` before they can influence planning or packing.
- Preflight now rejects global tensor shape drift from safetensors headers:
  embedding and output-head hidden dimensions must match `hidden_size`, and the
  final norm must be exactly `[hidden_size]`. When `vocab_size` is present,
  embedding and output-head row counts must match it too, catching bad resident
  metadata before packing opens weight payloads.
- Safetensors metadata scanning now accepts a single unindexed `.safetensors`
  file for tiny GLM smoke checkpoints while still requiring
  `model.safetensors.index.json` for multi-shard directories. That gives the
  importer a low-friction fixture path without letting incomplete large GLM-5.2
  downloads guess shard ownership.
- Preflight now records config-level quantization metadata from MLX-style
  `quantization`, Transformers-style `quantization_config`, and common top-level
  bit/group fields. Pre-quantized affine-int4 imports fail closed when declared
  bits or group size disagree with the requested `--quant-bits`/`--group-size`,
  while raw conversion keeps those config fields as warning-only source
  evidence. If the config is silent but routed expert tensor names look like
  GPTQ/AWQ/bitsandbytes (`qweight`, `qzeros`, `g_idx`, `quant_state`, `absmax`,
  or `quant_map`), header-only preflight now emits a specific unsupported
  expert quantization layout error before the generic packer coverage failure.
- Generation admission now carries the same config-derived `hidden_size` and
  `vocab_size` expectations into prompt embedding, decode embedding, runtime
  guard, and CPU/Metal final logits, so stale resident layouts fail before
  large reads, chunks, or runner launches.
- Runtime preflight now requires a resident embedding tensor and checks
  `max_embedding_row_mib` before the generation work directory is created;
  request inspection reports the embedding row/output bytes for this guard.
- Direct decode-layer execution now rejects non-integer position/context,
  layer, attention/DSA, cache-dtype, and read-advise controls before work files
  or runner processes are created.
- `decode-layers` now removes automatically-created work directories on runner
  failure unless `--keep-work-dir` or an explicit work directory requests
  preservation, keeping failed layer tests from leaving large f32 intermediates.
- Prompt embedding batches validate token ids before opening output files and
  remove partial f32 prompt embeddings if a row read/write fails.
- Final-logits paths validate RMSNorm epsilon and Metal top-k JSON records, so
  malformed runner output cannot feed boolean token ids back into generation.
- The generation runtime and live-memory guard now use the same strict integer
  policy for context, layer, DSA, logits, cache dtype, and live-memory byte
  fields, so JSON booleans cannot become 0/1 in admission control.
- Planner, preflight, `prepare-glm`, and `prefill-plan` apply that strict
  integer policy to context, byte budget, bit-width, group-size, M5 tile,
  activation-byte, chunk/heap, and runner-scratch controls before deriving cache
  budgets, backend candidates, or staged MoE capacity estimates.
- `prefill-plan` and `prefill-plan-calibrate` now also treat
  `--require-public-glm-5-2-shape` as a 4bit target lock: public GLM-5.2 plans
  with non-4bit `expert_bits` fail before calibration dispatch, and non-4bit
  exploratory plans no longer emit the public-shape launch guard suggestion.
- The prepared HTTP/OpenAI-compatible boundary uses the same strict integer
  policy for JSON controls, returning 400-style admission errors instead of
  truncating floats or accepting booleans/strings. Float sampling controls must
  be finite so request inspection cannot quietly accept NaN/Inf.
- Raw GLM expert quantization now reports its row-block memory envelope:
  maximum raw source block, maximum generated affine-int4 output block,
  extra modeled heap, and maximum rows per block. This makes prepare/pack dry
  runs auditable before a full GLM-5.2 conversion touches large expert files.
- Routed expert discovery now rejects per-expert tensor ids outside the
  config-declared expert range, so preflight/prepare fail closed on
  checkpoint/config mismatches instead of silently ignoring extra expert tensors.
- Raw expert quantization now rejects NaN/Inf source values as `PackerError`s,
  keeping execute-mode partial layer cleanup on the controlled error path.
- Continuous batching and prefix sharing should consume low-level runner
  telemetry rather than guessing memory use.

## GLM-5.1 Public Checkpoint Shape

Reference: <https://huggingface.co/zai-org/GLM-5.1>

The public GLM-5.1 config uses `model_type="glm_moe_dsa"`, 78 layers,
`hidden_size=6144`, `moe_intermediate_size=2048`, 256 routed experts, and
top-k 8 experts per token. It sets `scoring_func="sigmoid"`,
`norm_topk_prob=true`, `routed_scaling_factor=2.5`, `n_group=1`, and
`topk_group=1`. Its safetensors index names routed experts as
`model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight`; router
weights are `model.layers.N.mlp.gate.weight`; correction-bias buffers are
`model.layers.N.mlp.gate.e_score_correction_bias`; shared experts are under
`model.layers.N.mlp.shared_experts.*`.

The index does not expose pre-quantized `.scales` or `.biases` tensors for
routed experts, so LargerLM uses its streamed raw-weight affine-int4 quantizer
unless an external quantized checkpoint already provides packed weights. The
packer handles both per-expert tensors and fused expert-major `[num_experts,
out, in]` tensors by slicing one expert matrix at a time. It also accepts
common `w1/w3/w2` MoE component aliases as gate/up/down when their shapes match
the GLM config, while duplicate canonical components are rejected.

## GLM-5.2 Public Checkpoint Shape

Reference: <https://huggingface.co/zai-org/GLM-5.2>
Reference implementation:
<https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py>

The public GLM-5.2 config is still `model_type="glm_moe_dsa"` with 78 layers,
`hidden_size=6144`, `moe_intermediate_size=2048`, 256 routed experts, and top-k
8 experts per token. It makes the first three layers dense with explicit
`mlp_layer_types`, then uses sparse MoE layers 3..77; LargerLM now validates
those labels at config load so a typo cannot silently move a MoE layer into the
resident dense path. The same load step rejects non-positive scalar dimensions,
epsilon/theta values, expert/top-k counts, and impossible top-k/context
relationships before planner estimates are produced. It also exposes a full
78-entry `indexer_types` schedule, with 21 full-indexer layers, and raises
`max_position_embeddings` to 1,048,576. The official config uses
`kv_lora_rank=512`, `q_lora_rank=2048`, `qk_rope_head_dim=64`,
`qk_nope_head_dim=192`, `v_head_dim=256`, `rope_theta=8000000`,
`index_head_dim=128`, `index_n_heads=32`, `index_topk=2048`,
`index_topk_freq=4`, `index_skip_topk_offset=3`,
`indexer_rope_interleave=true`, `scoring_func="sigmoid"`,
`norm_topk_prob=true`, `routed_scaling_factor=2.5`, `n_group=1`, and
`topk_group=1`.
Prepared readiness reports `matches_public_glm_5_2_shape`; launch commands can
promote that diagnostic to a hard pre-run gate with
`--require-public-glm-5-2-shape`. The diagnostic now carries per-field
`public_glm_5_2_shape.checks` and `mismatched_fields`, including attention/MLA
dimensions, the DSA indexer schedule, and router semantic fields, so a failed
public-shape gate points at the actual drift instead of only returning a bool.
Router/RoPE semantic config fields (`scoring_func`, `topk_method`,
`norm_topk_prob`, `routed_scaling_factor`, `n_group`, `topk_group`, and
`rope_interleave`) are now parsed into `ModelConfig`, with invalid types,
non-positive groups/scales, and `topk_group > n_group` rejected at config-load
time. GLM `eos_token_id` values are also parsed as a typed, de-duplicated tuple
of non-negative stop-token ids, and `dtype`/`torch_dtype` is parsed into a
validated resident-estimate byte width. Resident packing writes
`resident/layout.json` router metadata from the same typed fields, so
whitespace-normalized strings and validated booleans, scales, and group counts
are preserved consistently in prepared artifacts. The GLM 4bit readiness gate
now treats that metadata as required and compares it with `config.json`, which
prevents an old prepared package from reusing expert files under drifted router
semantics. The same gate also requires expert and resident layout
`config_sha256` identity, so hashless legacy layouts are rejected for guarded
GLM-5.2 launches. Manifest `expert_quantization` and `expert_group_size`
fields are likewise mandatory and must match the expert layout.
The public safetensors index names full-indexer resident tensors as
`model.layers.N.self_attn.indexer.{wk,wq_b,weights_proj}.weight` plus
`model.layers.N.self_attn.indexer.k_norm.{weight,bias}`. The published index
also includes MTP layer-78 indexer tensors; LargerLM's decode-cache and preflight
paths intentionally ignore layers at or above `num_hidden_layers`. Resident
packing uses the same runtime-layer boundary, so MTP/extra-layer tensors do not
inflate `resident.bin` or the modeled resident-byte budget.
The first metadata-only probe of `mlx-community/GLM-5.2-mxfp4` fetched only
HTTP Range safetensors headers and produced a 76-shard, 2,489-tensor manifest
with `metadata.total_size=395094087168`. Using the public GLM-5.2 config
fixture, preflight now reports public-shape match plus full attention, dense,
shared-expert, router, indexer, and resident coverage from headers alone. The
artifact is not affine-int4: resident and routed matrices are MXFP4-style
`U32 .weight` plus `U8 .scales` without affine `.biases`, and attention uses
absorbed `embed_q`/`unembed_out` tensors instead of `kv_b_proj.weight`. The
packer now derives `quantization="mlx-mxfp4"` per-layer slots directly from
those headers: 75 MoE layers x 256 experts, 20,054,016 bytes per expert slot,
and 385,037,107,200 routed bytes in total. Routed expert MXFP4 Metal execution
now has direct layer-MoE and expert-major batch smoke coverage, including E2M1
value decode and E8M0 scale-byte handling. Resident MXFP4 execution now covers
bounded embedding rows, 2-D custom-metal linear/batch projections, router
gates, bounded Python DSA indexer matrices, attention q/kv/o projections,
shared/dense MLP triplets, and chunked final-logits top-k. The low-level
attention runners now also cover the absorbed `embed_q`/`unembed_out` form:
attention projections skip absent `kv_b_proj`, while single, batch, and indexed
MLA attention synthesize a bounded f32 KV-B view from the aliases; a tiny
`--run-decoder-layer` command smoke covers the same absorbed path end to end for
one layer, and the Python batch MLA prefill wrapper now accepts aliases when
estimating value-source storage and f32 expansion. Python batch projection and
composed attention-block wrappers now carry the same alias source through
bounded JSON-visible results and skip the absent `kv_b_proj.f32` intermediate.
A metadata-only bring-up in `artifacts/glm-5.2-mxfp4` completed header fetch,
capped small-file fetch, M5 prefill-backend reporting,
`preflight-glm --metadata-only`, and `prepare-glm --metadata-only` with
`--group-size 32`, then advanced to the 76-shard `download_weights` handoff. The
current local artifact has now moved past that boundary:
`checkpoint-status --verify-local-headers --require-complete --require-clean`
reports 76/76 complete shards, `download_complete_proven=true`,
`local_headers_ok=true`, `artifact_clean=true`, `issues=[]`,
`expected_safetensors_file_bytes=395094391502`, and `next_bringup_step=null`.
The remaining hard readiness boundary is now policy/performance and serving
hardening, not missing GLM-5.2 metadata, weight download, routed expert packing,
high-level alias orchestration, or low-level alias runtime.
The reference `GlmMoeDsaIndexer` computes `wq_b(q_resid)` and `wk(hidden)`,
applies LayerNorm plus non-interleaved RoPE to the index keys, stores those keys
in an indexer cache, scores cached positions with ReLUed dot products, and uses
`weights_proj(hidden)` to combine index heads. Full-indexer layers compute new
top-k token ids, while shared-indexer layers reuse the previous full layer's
top-k selection. LargerLM therefore rejects schedules, or selected layer subsets,
where a shared-indexer layer appears before any full-indexer layer in the same
walk.

For a 4-bit affine expert slot with group size 64, one routed expert is
21,233,664 bytes. A decode token that activates 75 layers x 8 experts reads
about 12.74 GB of routed expert slots before OS page-cache reuse. A long prefill
with enough tokens to touch all 256 experts per MoE layer can sweep about
407.69 GB of routed expert slots once per prompt. The BF16 MLA/DSA cache
estimate is 95,232 bytes per token, so the full 1,048,576-token advertised
context is about 99.86 GB of decode cache alone and does not fit the current
M5 Max memory target without strict cache/file guards.
Decode-layer records now preserve the per-layer preflighted routed-expert read,
attention/cache read, and peak scratch estimates. Generation attaches those
records to each generated step and server token payloads expose the same
summary, so a real GLM-5.2 run can explain aggregate SSD read pressure by layer
instead of only reporting one step-level total.
Decode admission now has an explicit per-token routed expert read guard:
`--decode-max-routed-read-gib-per-token` caps the deterministic
`top_k * expert_slot_bytes` sum across selected MoE layers, and
`--decode-max-routed-read-seconds-per-token` uses the prepared or supplied SSD
GiB/s estimate to bound that read time. Setting either cap forces the necessary
runtime preflight even if broad runtime preflight was otherwise disabled.
Request checks and prepared benchmarks now surface `suggested_decode_guard_flags`
with argv-style `--decode-max-routed-read-*` caps and 5% headroom, so a measured
or inspected GLM run can be replayed without widening the decode SSD envelope.
128 GiB safety budget after system reserve, runtime buffers, and target page
cache.
Prepared GLM 4bit readiness now reports those config-derived expert byte
quantities directly and rejects a prepared package whose validated expert bytes
do not match the affine-int4 total, so launch checks cannot accidentally accept
an underestimated SSD working set.
Planner/preflight now exposes resident-memory budget, pressure, and headroom
explicitly, making this 128 GiB tradeoff visible before packing: GLM-5.2 can fit
resident/runtime/reserve headroom, but advertised full-context cache must be
shrunk or capped by `--auto-context-from-budget`.

## Metal 4 MPP Programming Guide

Reference:
<https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf>

The March 2026 guide introduces Metal tensor resources and
`mpp::tensor_ops`, which can invoke GPU neural accelerators on M5 for GEMM-like
operations. Practical implications:

- Prefill should use MPP tensor ops for dense tiled GEMMs where output tiles are
  large enough to be compute-bound.
- Decode routed expert matvecs are usually low arithmetic-intensity and
  I/O-bound, so custom streaming Metal kernels are still likely better.
- Start GEMM tuning with 2x2 simdgroups per threadgroup and 32x32 simdgroup
  tiles for 16-bit operands, then benchmark.
- Use locality-preserving threadgroup walk order, such as Morton order, for
  large GEMMs.
- Use cooperative tensors for postfix fusion, such as bias, residual, norm, or
  activation when it avoids a device-memory round trip.

Current local status on the target M5 Max:

- 2026-06-28 host check: `/usr/sbin/sysctl -n machdep.cpu.brand_string`
  reports `Apple M5 Max`, and `hw.memsize` reports 137438953472 bytes
  (128 GiB).
- Codex's sandboxed backend probe can return a false `no Metal device`; the same
  `prefill-backend --run-mpsgraph-probe --run-mpp-probe --probe-timeout-seconds
  30 --json` command run outside that sandbox sees the Apple M5 Max Metal device,
  passes the tiny MPSGraph matmul probe, reports
  `metal4_ml_runtime_available=true`, and keeps MPP non-selectable because the
  public SDK still lacks the `mpp::tensor_ops` shader symbols/includes needed by
  the current probe.
- `prefill-backend` reports Metal 4 family support, Metal 4 machine-learning
  host APIs, `MTLTensor`, tensor size/alignment selector support, int4 tensor
  datatypes, and tiny ML tensor allocation.
- The backend JSON exposes derived `mps_graph_runtime_available`,
  `metal4_ml_runtime_available`, and `mpp_runtime_available` booleans so tuning
  scripts can distinguish raw SDK declarations from runtime-probed MPSGraph,
  Metal 4/MTLTensor, and compiled MPP tensor-op paths.
- `prefill-backend --run-mpsgraph-probe` now executes a 2x2 float32 MPSGraph
  matmul and exposes `mps_graph_probe_requested`, `mps_graph_probe_ran`,
  `mps_graph_probe_ok`, and `mps_graph_probe_error`. When requested,
  `mps_graph_runtime_available` requires that tiny runtime matmul to pass, so
  M5 bring-up can distinguish header availability from an executable MPSGraph
  path without loading model weights.
- On the current Codex-managed Mac environment, the restricted sandbox cannot
  see a Metal device and reports `mps_graph_probe_error="no Metal device"`.
  Running the same no-weight probe with host access sees `device_name="Apple M5
  Max"`, `metal4_ml_runtime_available=true`, `mps_graph_probe_ok=true`, and
  validates `mpsgraph-f32` as the selectable accelerated prefill backend. Public
  SDK MPP tensor-op symbols are still absent, so the neural-accelerator status
  remains `missing_public_mpp_symbols` and recommends the MPSGraph fallback.
- `prefill-backend --compile-mpp-probe` now requires the host probe and records
  `mpp_compile_probe_requested` separately from `mpp_compile_probe_ran` and
  `mpp_compile_probe_ok`, so MPP readiness reports cannot imply a compile probe
  ran when host probing was disabled.
- It also exposes `prefill_acceleration_runtimes` separately from
  `selectable_accelerated_prefill_backends`, so MPP can be tracked as a ready
  runtime candidate without letting generation or benchmark gates claim it is a
  selectable execution backend before that path exists.
- The same payload now includes `suggested_prefill_acceleration_flags` when a
  selectable accelerated backend exists, currently emitting
  `--prefill-linear-backend mpsgraph-f32 --require-prefill-acceleration` while
  keeping MPP behind the future selectable-backend boundary.
- Those suggested flag blocks now also carry
  `validated_accelerated_prefill_backends` and the full
  `prefill_neural_accelerator_status`, so health and prefill-plan launch profiles
  preserve whether MPSGraph has runtime-proof and whether the planned MPP/neural
  path is unavailable, probe-gated, visible-but-not-selectable, or eventually
  ready.
- Probe path, request status, and compact host-probe failure details are now
  exposed in the same capability payload, including the tiny tensor allocation
  error when Metal reports one, which keeps SDK/toolchain failures diagnosable
  during M5/MPP bring-up without loading model weights.
- Prepared health and launch-audit acceleration evidence now also carry the
  host probe requested/ran/ok booleans, probe path, and probe timeout. Strict
  required-audit replay rejects required acceleration artifacts that dropped
  that host-probe identity, so an MPSGraph pass cannot be replayed as a bare
  boolean with no executable boundary.
- The MPP run probe now records concrete execution evidence when it reaches a
  pipeline: kernel include variant, `32x32x32` tile shape, half dtype, and the
  `mpp::tensor_ops::matmul2d` primitive. These fields flow through
  `prefill_neural_accelerator_status` while MPP remains non-selectable for
  generation.
- `validated_accelerated_prefill_backends` now separates "generation can select
  this path" from "the required runtime probe proved it on this host"; required
  launch-audit consumers reject selectable acceleration evidence with no
  validated backend.
- Server request summaries now include `total_matrix_scratch_bytes` in addition
  to peak scratch and raw-conversion bytes. The peak value remains the memory
  admission cap, while the total exposes MPSGraph F32-conversion churn across a
  full resident-GEMM prefill sweep.
- The resident batch-linear runner now prints matrix scratch, F32 conversion
  bytes, raw-conversion bytes, estimated peak, backend elapsed time,
  matrix-F32 materialization time, and accelerator/graph dispatch time for each
  command. `prefill-linear-batch --json` parses those lines into
  `runner_backend_elapsed_seconds`, `runner_matrix_f32_elapsed_seconds`, and
  `runner_accelerator_elapsed_seconds`, so real GLM replays can distinguish CPU
  materialization from MPSGraph/MPSMatrix/custom-Metal execution without
  opening raw runner logs. The MPSGraph execution block has a local
  autorelease pool so temporary graph objects drain before the command returns.
  A single real GLM-5.2 router probe on layer 10 `.mlp.gate.weight`
  (`BF16 [256,6144]`, 16 zero tokens, one 3 MiB resident matrix) measured
  custom Metal at 0.0085s runner backend time, MPSGraph at 0.0506s
  (`matrix_f32=0.0024s`, `accelerator=0.0467s`), and MPSMatrix at 0.2402s
  (`matrix_f32=0.0023s`, `accelerator=0.2379s`). For this current 17-token
  router shape, the immediate bottleneck is graph/dispatch overhead rather than
  BF16-to-F32 materialization, so the next profile experiment should be a
  plan-signature-guarded custom-router candidate rather than assuming MPSGraph
  wins every BF16 resident GEMM.
- Required launch-audit consumers now reject a passing MPP run-probe claim if
  that concrete execution evidence is missing or inconsistent, so an audit file
  cannot say "MPP ran" without preserving the exact bring-up boundary.
- `metal/largerlm-runner --self-test-mpp` now performs the same tiny 32x32 half
  MPP `matmul2d` check inside the runner binary, giving the eventual prefill
  path an executable-boundary smoke test before MPP is allowed to become a
  selectable generation backend.
- The host probe emits explicit selector booleans and an MPP compile error even
  when `MTLCreateSystemDefaultDevice()` returns no device, so bring-up can
  distinguish "no Metal device visible" from missing JSON fields.
- The Python capability summary collapses that no-device case into an explicit
  `host probe could not create a default Metal device` reason while preserving
  the raw selector booleans for automation.
- Prepared offline generation and serving now resolve
  `prefill_linear_backend=auto` against the local MPSGraph runtime capability
  and pass `custom-metal` at runtime when the host probe cannot verify MPSGraph
  support, matching the health warning instead of leaving the lower-level
  shape-only auto selector to choose an unavailable backend.
- Prepared benchmarks now reuse that request-admission runtime backend for the
  actual measured runner call when `prefill_linear_backend=auto`, keeping
  benchmark execution aligned with the probed MPSGraph availability boundary.
- Required launch audits now fail when the checked request's effective prefill
  backend differs from the runtime-resolved backend, so an audited GLM launch
  cannot replay an `auto` backend across a changed MPSGraph availability state.
- The command-line `metal` tool currently reports a missing Metal Toolchain
  component.
- `prefill-backend --compile-mpp-probe` does not find an MSL include that
  exposes `mpp::tensor_ops`: `metal_tensor` exists but does not declare `mpp`,
  while `metal_mpp` and `metal_performance_primitives` are not found. Treat the
  MPP shader path as an optional fast path until the matching Metal Toolchain or
  SDK component is installed; `recommended_backend` stays on custom Metal when
  the host probe cannot create a Metal device, uses MPSGraph when matmul support
  is declared and the requested host/runtime probes have not failed, and moves
  to MPP only after an MPP compile probe succeeds and the full Metal 4 ML runtime
  selector set is present. The runner-side `--self-test-mpp` currently reports
  the same missing `metal_mpp` / `metal_performance_primitives` headers, so the
  inference binary has a direct retest once the matching SDK is installed.
- The neural-accelerator status now prefers the root SDK gap over the secondary
  run-probe failure: on this M5 Max it reports
  `status="missing_public_mpp_symbols"` while preserving the failed MPP
  compile/run probe details.
- The Metal runner now mirrors the GLM readiness affine-int4 expert-slot
  contract for direct MoE calls: `parse_expert_layout` rejects wrong
  `component_order`, component dtype drift, shifted offsets, size drift, and
  slot-span mismatches before allocating/running expert compute. It also treats
  expert layer ids, group sizes, component offsets, sizes, and shape dimensions
  as strict JSON integers so booleans or floats cannot be coerced into slot
  geometry. The `metal/expert_layout_contract_smoke.py` script exercises those
  bottom-layer failures without relying on the Python server gate.
- Direct runner MoE entrypoints now also validate the surrounding expert layer
  file contract before creating a Metal device: `layer_file` must be a relative
  file name, `num_experts` and `expert_slot_bytes` must be positive integers,
  `num_experts * expert_slot_bytes` must not overflow, and the actual layer file
  size must match the layout. The contract smoke mutates those fields as well.
- Direct runner expert/MoE command buffers now fail closed on Metal command
  errors after `waitUntilCompleted`: dequant/expert self-tests, direct
  `--run-expert`, direct `--run-moe` accumulator clear, and each routed expert
  command check `MTLCommandBufferStatusError`, with the open expert fd closed
  before returning on per-expert failure.
- The same low-level expert/router path now also rejects nil Metal command
  buffers before encoding work. Dequant/expert self-tests, shared expert
  commands, direct expert/MoE, MoE-batch blocks, and router/router-batch calls
  emit an explicit error and release any open files or temporary tables instead
  of letting Objective-C nil messaging silently turn the command into a no-op.
- Metal encode helpers now also propagate nil compute-encoder failures instead
  of silently encoding nothing. Resident matvec/batch prefill, RMSNorm,
  final-logit chunks, RoPE, MLA attention, routed expert, shared expert, and
  router paths all stop before commit if `computeCommandEncoder` cannot be
  created, preserving the no-bad-output contract under Metal resource pressure.
- Direct runner resident-weight entrypoints now validate `resident/layout.json`
  backing files before tensor reads: `weight_file` must stay within the
  resident layout directory, `total_bytes` must be a positive integer, and the
  actual file size must match. They also validate every resident tensor span
  before reads: tensor integer fields must be strict JSON integers, dtype must
  be one of the runner-supported resident dtypes, shape-derived byte size must
  match `size`, offsets must stay inside `total_bytes`, and tensor spans must
  not overlap. `metal/resident_layout_contract_smoke.py` exercises those
  failures through the router path.
- Direct runner decode-cache entrypoints now validate cache layout/backing
  files before cache reads or writes: layout version must be 1, top-level and
  segment integer fields must be strict JSON integers rather than booleans,
  segment dtype/stride/total/span metadata must match the runner's contiguous
  row interpretation, `kind:layer` segments must be unique, and the cache file
  logical size must match layout `total_bytes`. `metal/decode_cache_contract_smoke.py`
  exercises those failures through `--validate-cache-backing` without requiring
  a Metal device, and also verifies the direct MLA, attention-projection, and
  decoder-layer commands reject bad cache backing before Metal device creation.
- Direct runner batch-MoE route entrypoints now validate `--routes-json` and
  `LLMSCAP1` static-capacity `--routes-bin` inputs before creating a Metal
  device. JSON route `batch_tokens` and expert ids are strict integers, route
  weights must be finite numbers, per-token duplicate experts are rejected, and
  merged binary active/overflow assignments are checked for non-finite weights
  and duplicate token/expert pairs. `metal/moe_route_contract_smoke.py`
  exercises those failures without requiring a Metal device.
- The direct expert/MoE/MoE-batch runner CLI now uses strict full-string numeric
  parsing for layer/expert ids, `--max-slot-mib`, `--max-runner-scratch-mib`,
  `--max-k`, `--batch-tokens`, `--moe-token-block`, read-advice KiB caps, direct
  MoE expert lists, direct MoE weights, and shared router scaling/group options.
  The same strict boundary now covers final-logits chunk/scratch caps,
  attention-projection positions and RMS eps, dense/resident matrix caps,
  resident batch token counts, resident RMSNorm eps, decode/cache/MLA/RoPE
  dimensions, and router/MoE block caps. Decode-facing MiB flags use strict
  finite decimal parsing and byte conversion rather than integer-prefix
  truncation. The `metal/runner_cli_contract_smoke.py` rejects malformed
  suffixes, invalid router groups, and non-finite weights/eps/theta values
  before Metal device creation, keeping the bottom-level runner from silently
  coercing unsafe direct invocations.
- Current executable boundary: resident batch-linear prefill commands have an
  opt-in `mpsgraph-f32` backend for F32/BF16/F16 resident matrix batches and an
  explicit `mps-matrix-f32` backend using Metal Performance Shaders
  `MPSMatrixMultiplication`. The `auto` policy still selects only MPSGraph for
  large prompt GEMMs and keeps small shapes on custom Metal, giving prefill
  benchmarks a supported ML stack comparison point while the MPSMatrix path is
  measured explicitly. Its batch-token and matrix-dimension thresholds are now
  runtime-tunable, so M5 crossover experiments do not require code edits.
  The resident batch-linear smoke now exercises custom Metal, MPSGraph, and
  MPSMatrix on the M5 Max for both F32 and BF16 resident matrices under the same
  scratch cap. A sandbox-external M5 Max backend probe confirms Metal 4 ML and
  MPSGraph matmul are runtime-visible, while the public SDK still does not
  expose `metal_mpp`/`mpp::tensor_ops` symbols; MPP tensor-ops prefill therefore
  remains a gated future path, with MPSGraph as the selectable Apple ML fallback.
- `prefill-linear-calibrate` now generates bounded synthetic resident GEMM cases
  and measures custom Metal, MPSGraph, and MPSMatrix through the same runner
  wrapper. It supports square sweeps and explicit rectangular `INxOUT` shapes,
  reports per-case elapsed times, speedup, winner, and scratch estimates, then
  emits replayable `--prefill-mpsgraph-min-*` launch-profile flags only when the
  measured grid supports a threshold that excludes all observed MPSGraph-slower
  cases. MPSMatrix is kept as an explicit comparison backend for now, and the
  result now includes `backend_comparison`: total backend elapsed time,
  estimated TFLOP/s, speedup versus custom Metal, winner counts, winner FLOPs,
  and conservative explicit-backend flags when a backend should replace `auto`.
  Non-custom backends must clear the requested speedup on every measured case
  and win some measured FLOPs; custom Metal can now be recommended explicitly
  when Apple ML comparison backends lose and no finer MPSGraph threshold applies.
  Those flags are now folded into standalone calibration launch profiles and
  into `prefill-plan-calibrate` profiles when the static plan has not already
  forced a non-`auto` backend. The work-directory budget includes MPSMatrix's
  extra output files across repeats before creating the directory. The execution
  layer checks the target volume has a configurable free-space margin beyond
  that estimate. Calibration can now synthesize BF16 resident matrices with
  `--matrix-dtype BF16`, and both standalone and plan-driven calibration budgets
  account for the smaller matrix payload. A BF16 M5 Max sweep at 16 prompt
  tokens over GLM-style `6144x256` and `6144x2048` matrices favored custom
  Metal overall and now emits a replayable `--prefill-linear-backend
  custom-metal` launch profile accepted by `inspect-prepared
  --apply-launch-profile`. Small M5 Max smokes over 32x32 and 48x32 matrices
  showed custom Metal faster, and bounded F32 sweeps up through
  1024-token/2048-dim cases still favored custom Metal.
  A planner-derived BF16 sweep at 128 prompt tokens over the top eight GLM-5.2
  resident GEMM shapes covered 98.1% of planned candidate FLOPs; custom Metal
  won 7/8 cases, MPSMatrix won only `2048x16384`, and MPSGraph won none. Totals
  were custom 0.474s, MPSMatrix 0.660s, and MPSGraph 0.663s, so the calibration
  should be replayed as explicit custom Metal at the requested 1.1x threshold.
  The first measured MPSGraph win in the bounded local grid was the
  2048-token/4096-dim case, so the default `auto` MPSGraph threshold is now
  2048 batch tokens and a 4096 minimum matrix dimension while calibration
  profiles can still lower that policy for a different SDK or model shape.
- `prefill-plan` now bridges static GLM shape planning into that calibration
  path by exporting `prefill_linear_calibration_shapes` from the highest-FLOP
  unique resident GEMM candidates and a separate
  `suggested_prefill_linear_calibration_flags` argv fragment for
  `prefill-linear-calibrate`. The fragment includes bounded case, resident
  matrix, and runner scratch caps computed with headroom, while staying out of
  `suggested_launch_profile` so measurement-only flags cannot be replayed as
  inference launch policy.
- `prefill-plan-calibrate` now consumes those planned shapes directly: it builds
  the static GLM prefill plan, applies the planner-derived calibration caps to
  `prefill-linear-calibrate`, and can write both the applied calibration flags
  and a merged prefill launch profile. The merged profile preserves planner
  guard sections and uses measured calibration thresholds when available,
  without dropping planner-side acceleration gates such as required acceleration
  or a minimum accelerated-FLOP fraction. It now accepts `--ssd-read-gib-s` too,
  so the same calibration artifact can carry routed expert read-second guards
  derived from measured local SSD throughput. It also records `--matrix-dtype`
  in the written applied calibration flags, keeping BF16 sweeps replayable
  without silently reverting to F32 synthetic matrices. Effective cap bytes now
  follow any MiB override in that artifact too. For BF16 plan-driven sweeps, the
  planner-derived runner scratch cap is now raised automatically when needed to
  cover Apple ML comparison backends that keep both the raw BF16 resident matrix
  and the converted F32 matrix in scratch.
  Default `--max-auto-*` calibration limits block unexpectedly large planned
  cases before the runner is dispatched, and the command now separately budgets
  calibration work-directory bytes for matrix/input files plus both backend
  outputs across repeats. The execution layer also checks real target-volume
  free space before creating the work directory. This preserves the
  no-OOM/no-disk-spike bring-up path for GLM-5.2/M5 experiments.
  Planner-driven calibration artifacts now include
  `calibration_candidate_coverage`, which records the candidate ranks, op names,
  layer counts, FLOPs, and weight bytes represented by each deduplicated matrix
  shape. This keeps GLM-5.2/M5 prefill calibration tied to the actual hot
  planner candidates instead of anonymous synthetic GEMM dimensions.
- `checkpoint-status` now gives GLM-5.2 downloads a no-OOM/no-payload-read
  bring-up checkpoint before preflight: it reads only `config.json`,
  safetensors index/header-manifest JSON, and shard `stat()` results, then
  reports missing/truncated/complete shard counts plus replayable `hf download`,
  header-fetch, no-weight prefill-backend probe, preflight, and prepare dry-run
  commands. It also computes
  remaining expected shard bytes and checks the target volume against a
  configurable download disk margin, so GLM-5.2 downloads can fail early before
  filling the SSD. It can also write a missing/truncated-shard JSON manifest and
  one-URL-per-line list for external download machines, plus a full
  `--write-status-json` audit artifact for the complete shard table, local
  header-check result, bring-up plan, and readiness gates from the run. This
  avoids manual shard transcription and lets large-model safety checks leave a
  stable file instead of only stdout. `--require-download-disk-ok` promotes that
  disk-budget report to a non-zero exit gate for download scripts, and
  `download_precheck_command` prints that replayable invocation. The generated
  header-fetch command now includes capped small-file fetching for `config.json`
  and tokenizer metadata, reducing manual setup before metadata-only preflight.
  The generated no-weight `prefill-backend --write-report` step writes
  `largerlm-prepared/prefill-backend-report.json` before shard download, and
  metadata `preflight-glm` plus `prepare-glm --metadata-only` dry-run gates are
  also ordered before shard download. That captures M5 MPSGraph runtime proof,
  public MPP availability, GLM-5.2 shape compatibility, layout planning, cache
  budget, and disk budget without touching GLM-5.2 payloads. After copied
  shards return to the Mac, `--verify-local-headers` reads only local
  safetensors headers and `stat()` data, validates tensor names, dtype/shape
  byte spans, manifest header equality, and data-start/file sizes, and fails
  closed before packing if a size-correct shard is corrupt or from the wrong
  artifact.
  On 2026-07-02 the local `mlx-community/GLM-5.2-mxfp4` artifact was verified
  complete with this gate: 76/76 safetensors shards present, 0 missing/partial
  shards, 395094391502 expected and present safetensors file bytes,
  395094087168 expected tensor bytes, all 76 local headers checked, and
  `artifact_clean=true` with no reported issues. That closes the Hugging Face
  download handoff as an active blocker for the current M5 Max package.
  `--require-clean` promotes the aggregate
  no-missing/no-partial/no-extra/no-header-error state to a non-zero exit gate
  for post-copy automation, and `post_copy_check_command` prints that exact
  safe check. Once that verified-header state is present, the same report emits
  `prepare_execute_command`, `inspect_prepared_command`, `launch_audit_command`,
  and `minimal_smoke_command`, giving GLM-5.2 bring-up scripts a conservative
  sequence from final shard validation to bounded prepare, prepared request
  admission, locked launch-profile auditing, and the first real 1-token run
  guarded by the audited launch profile. The generated preflight/prepare
  commands now explicitly pin the first-pass M5 Max profile with
  `--group-size 32 --max-cache-gib 16 --disk-margin-gib 32
  --unified-memory-gib 128`, so scripts do not silently fall back to a weaker
  MXFP4/cache/disk/memory envelope.
  The generated inspect/audit commands now also run the tiny MPSGraph and MPP
  tensor-ops probes with a 30 second backend-probe timeout, preserving M5 neural
  accelerator runtime evidence before the first audited token is generated. The
  generic checkpoint-status launch audit and minimal-smoke commands no longer
  force `--require-prefill-acceleration`: the default bring-up path stays on the
  token-stable custom-Metal profile and carries
  `--allow-non-accelerated-prefill-launch-audit` as explicit audit evidence,
  while MPSGraph/router-gate-only experiments remain in dedicated wrappers with
  `--allow-router-gate-only-prefill-acceleration`.
  `bringup_plan` records the same path as ordered structured steps with
  prerequisites, `step_status` values (`complete`, `ready`, or `blocked`),
  blocked reasons, command availability, and explicit reads-weights/writes/
  runs-model safety labels. `next_bringup_step` exposes the first ready command
  after currently complete steps for conservative single-step automation without
  pretending to prove external prerequisite execution. `checkpoint-status` can
  now write that exact next step as JSON or a quoted executable shell script,
  reducing manual flag-copying risk while still leaving execution under explicit
  operator/script control. It can also write an external safetensors download
  handoff JSON plus a resumable `curl` script with target paths, expected byte
  sizes, exact post-download size checks, outer `curl` failure retries, and
  transient `Unsupported content type` body retries for machines that can fetch
  the GLM-5.2 shards more reliably than the target Mac. The generated script can
  now limit a batch by shard index range or planned download bytes using
  `LARGERLM_DOWNLOAD_START_INDEX`, `LARGERLM_DOWNLOAD_END_INDEX`, and
  `LARGERLM_DOWNLOAD_MAX_BYTES`, making first-shard network probes and batched
  off-machine transfers resumable without editing the script. Configurable curl
  connect and low-speed timeouts prevent long zero-byte hangs while leaving
  partial files resumable. The completed state for
  prefill backend reports, preflight reports, prepare dry-run reports, prepared
  manifests, launch profiles, and launch audits now use bounded small-JSON
  validation, so empty/corrupt/failed/stale artifacts cannot masquerade as
  finished GLM-5.2 bring-up steps.
  On 2026-07-03 a refreshed local status wrote
  `artifacts/glm-5.2-mxfp4/checkpoint-status-latest.json` with
  `download_complete=true`, `download_complete_proven=true`,
  `local_headers_ok=true`, 76/76 shards present and complete,
  `artifact_clean=true`, `expected_safetensors_file_bytes=395094391502`,
  `present_safetensors_file_bytes=395094391502`, `launch_profile_valid=true`,
  `launch_audit_valid=true`, `minimal_smoke_result_valid=true`, and no next
  bring-up step. The refreshed
  canonical `minimal-smoke.json` generated `[15]` in 17.000s with no
  prompt-prefill phase while replaying the launch-audit MPP/MPSGraph probe
  flags.
- The generated minimal-smoke command now writes a schema-tagged
  `minimal-smoke.json` result via `generate-prepared-token-ids --write-result`.
  `checkpoint-status` validates that small result, including bindings to the
  current prepared manifest, locked launch profile, and launch audit, before
  marking the final one-token smoke complete. This makes the last bring-up step
  resumable without trusting stdout logs, stale copied results, or bare file
  existence.
- `preflight-glm --metadata-only` and `prepare-glm --metadata-only` now prefer
  `largerlm.safetensors.headers.json` even when partial local shard files are
  present. That keeps GLM-5.2 metadata/layout/cache/disk-budget checks usable
  during a long download while preventing dry-run metadata from being confused
  with executable packing; `prepare-glm --metadata-only --execute` is rejected.
- The header-only fetcher now retries transient
  `{"detail":"Unsupported content type"}` HTTP error bodies and 200-response
  bodies for index JSON, safetensors Range requests, and capped small-file
  downloads, so intermittent Hugging Face or proxy content negotiation failures
  do not abort the metadata-only GLM-5.2 bring-up path on the first attempt.
- Prepared commands now accept `prefill_linear_calibration` and
  `prefill_plan_calibration` profiles through the same prefill-only section/argv
  whitelist used for `prefill_plan`, and the launch-profile loader can extract
  `combined_launch_profile` from the full `prefill-plan-calibrate --json`
  payload. Unsafe non-prefill flags still require prepared identity metadata, so
  config-only calibration artifacts cannot widen the launch surface. Standalone
  `prefill_linear_calibration` artifacts are limited to the runtime-policy
  section for identity-less replay.
- Request checks now expose `prefill_acceleration_coverage`, and prepared
  token-id generation, prepared text generation, prepared token-id benchmarks,
  and `serve-prepared` enforce it when `--require-prefill-acceleration` is set.
  Text and OpenAI-compatible service requests first do a lightweight tokenizer
  encode so the gate uses the real prompt token count. This catches the
  practical failure mode where MPSGraph is available but a particular prompt
  chunk or threshold setting still resolves all resident GEMMs to custom Metal.
- The same requirement is now checked after actual prompt prefill as well:
  prepared generation, benchmark, and server paths reject a run whose
  `prompt_prefill.prefill_acceleration_coverage` does not show an accelerated
  resident GEMM, preventing optimistic preflight results from masking runtime
  fallback.
- Router-gate-only MPSGraph coverage is now treated as an explicit experiment,
  not as the default acceleration success condition. If every accelerated
  matrix is a MoE router gate, `--require-prefill-acceleration` fails unless
  `--allow-router-gate-only-prefill-acceleration` is passed. The 2026-07-03
  shortchat KV-B smoke
  `server-http-shortchat64-max4-mpsgraph13x32-kvbcache-routergate-allow-guard-latest.json`
  returned HTTP 200 / `8333` / `[23, 18, 18, 18]` and recorded
  `allow_router_gate_only_acceleration=true`,
  `accelerated_router_gate_only=true`, `router_gate_accelerated_matrix_count=75`,
  and `non_router_accelerated_estimated_flops=0`.
  Coverage and result summaries now also carry the non-router acceleration gap:
  `non_router_matrix_count`, `non_router_estimated_flops`,
  `non_router_unaccelerated_matrix_count`,
  `non_router_unaccelerated_estimated_flops`, and
  `non_router_unaccelerated_flop_fraction`. They additionally break that gap
  into `non_router_unaccelerated_streamed_routed_expert_*`,
  `non_router_unaccelerated_non_streamed_*`, and per-backend unaccelerated
  FLOPs. On that same allow-guard smoke, `result-summary` reports
  `non_router_unaccelerated_estimated_flops=985045401600` and
  `non_router_unaccelerated_flop_fraction=0.996896`, with streamed
  routed-expert FLOPs `588880281600`, non-streamed FLOPs `396165120000`,
  backend FLOPs `custom-metal=606546690048`, and `other=378498711552`. The M5
  acceleration frontier is therefore mostly the streamed/custom-Metal routed
  expert path, followed by the non-streamed fused resident/shared path, not the
  router gate.
- The router-hybrid margin threshold is now explicit and replayable through
  `--prefill-router-hybrid-margin-threshold` rather than only the legacy
  `LARGERLM_PREFILL_ROUTER_HYBRID_MARGIN_THRESHOLD` environment fallback. A
  2026-07-03 driftx4 shortchat64/max4 experiment wrote
  `launch-profile-shortchat-64tok-auto-mpsgraph13x32-routerhybrid-driftx4-keycache-memory-accumulator-decode-keycache-request-locked.json`
  (SHA `1d83ef285f25b7a3e2e288984bfa8b22fdb0a67f4a874ded378e3ec862c300ee`),
  `launch-audit-shortchat-64tok-auto-mpsgraph13x32-routerhybrid-driftx4-keycache-memory-accumulator-decode-keycache-request-max4-locked.json`
  (SHA `1be9bfe26a7fa7fd9c18720d1e869be507d2d71a800e97063c72f2e1f484a70b`),
  and
  `server-http-shortchat64-max4-routerhybrid-driftx4-kvbcache-memory-accumulator-decode-keycache-openai-chat-latest.json`
  (SHA `408a46068d7eaf52d36fd53b36578c37ba7a71843e835fcbabe1f3f5d88f3c7d`).
  The smoke was memory-admitted and audit-bound, but returned HTTP 200 /
  `8333` / `[23, 18, 18, 18]` with 56.400s request time, so it remains a
  negative experiment rather than a replacement for the baseline
  `0003` / `[15, 15, 15, 18]` custom-Metal profile.
- The shortchat64 max4 serving path now treats the in-memory MoE output
  accumulator and decode MLA key-cache as launch-profile contracts instead of
  operator environment variables.
  `launch-profile-shortchat-64tok-auto-keycache-memory-accumulator-decode-keycache-request-locked.json`
  records both `--prefill-moe-output-accumulator memory` and
  `--decode-mla-key-cache`, and the paired
  `launch-audit-shortchat-64tok-auto-keycache-memory-accumulator-decode-keycache-request-max4-locked.json`
  is bound to that profile SHA. The reproducible HTTP smoke
  `server-http-shortchat64-max4-kvbcache-memory-accumulator-smoke-safe.sh`
  starts the audited KV-B-cache wrapper, runs the deterministic `你好` OpenAI
  chat with `max_tokens=4`, and stops the server. The latest artifact
  `server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-latest.json`
  returned HTTP 200 / `0003` / `[15, 15, 15, 18]`, with
  `audit_bound=True`, `safe_to_replay=True`, `replay_ready=True`,
  `files_ready=True`, and `decode_mla_key_cache=true` in both health and
  request-check evidence. The server token result took 55.381s: the first
  prefill-heavy step took 45.876s and the remaining decode steps took about
  3.35s, 3.05s, and 3.08s. This fixes the earlier profile-pinned artifact
  `server-http-shortchat64-max4-kvbcache-memory-accumulator-openai-chat-latest.json`,
  whose decode steps regressed to roughly 9s each after the shell
  `LARGERLM_MLA_KEY_CACHE` export was removed. This keeps the real-chat entry
  point fail-closed if the profile, audit, model files, memory envelope,
  temp-disk headroom, decode key-cache policy, or KV-B cache files drift.
  Server startup now enforces this strictly: an audit whose embedded request
  profile is replay-compatible but whose applied-profile SHA differs from the
  current server profile is rejected before `run_prepared_server` is called.
  Offline replay can still use the request-profile path where appropriate, but
  serving artifacts must be profile/audit file-ready.
- `result-summary` now emits a compact `optimization_targets` list in both JSON
  and text output. Re-reading the current 128-token GLM-5.2 MXFP4
  custom/key-cache memory-accumulator artifact does not launch the model, but it
  ranks the next work items as: non-router prefill acceleration gap, blocked
  `mpp_tensor_ops_prefill` frontier, routed MoE, MLA attention, resident
  projections, RoPE, expert stage copy, and cache write. This keeps future M5
  experiments tied to measured artifacts instead of manually re-ranking hot
  spots from prose notes.
- The routed-MoE optimization target now carries a fallback bounded
  `glm_moe_tile_sweep.py` experiment when split-kernel hints alone do not
  select a candidate. Running the generated layer-77/128-token command with
  240 MiB stage and compact caps plus a 256 MiB runner scratch cap produced
  `glm-moe-layer77-optimization-target-128tok.json`. It staged about
  200.5 MB, copied at about 8.42 GiB/s, and rejected all tested variants:
  `tile1_auto_silu` remained the fastest kernel mean at about 0.0698s, with
  no `candidate_for_full_replay`. This closes that routed-MoE branch without a
  memory-risky full replay.
- Routed-MoE summaries now compute non-runner boundary time as routed MoE
  elapsed minus the runner's own `timing total`. The previous 128-token
  artifact had routed MoE at about 14.151s, runner total at about 3.951s, and
  non-runner boundary time at 72.1%, so it was classified as
  `process_boundary`. The locked-profile persistent runner replay
  `smoke-prefill-128tok-persistent-moe-server-tiled-profile-replay-result.json`
  generated the same token id 15, lowered total elapsed from 63.739s to
  59.688s, kept the same 17.37 GiB live-memory envelope, and recorded
  `moe_plan_server_plan_count=75` with `routed_moe_runner_command_count=0`.
  The follow-up strict
  `smoke-prefill-128tok-persistent-moe-server-tiled-memory-accumulator-strict-audit-result.json`
  pins `--prefill-moe-output-accumulator memory` in the launch profile instead
  of relying on an environment override. Its launch binding is file-ready
  (`audit_bound=True`, `files_ready=True`), it generated token id 15 in
  60.122s, kept the same 17.37 GiB live cap with about 78.48 GiB available at
  admission, and reports `accum=memory` on the routed MoE layer records.
  Routed MoE elapsed is 11.481s, expert stage copy is 1.980s at 5.744 GiB/s,
  and `result-summary` classifies the remaining routed target as
  `moe_orchestration_overhead`, not process launch overhead. A current bakeoff
  against the strict result wrote
  `prefill-128-persistent-moe-memory-accumulator-bakeoff-current.json` plus
  `selected-replay-prefill-128-persistent-moe-memory-accumulator.json` and
  `.sh`; the selected replay now records `required_environment={}` because the
  launch profile pins `--prefill-moe-output-accumulator memory`. Both
  `selected-replay-check` and `selected-replay-run --dry-run` passed on the
  local machine with about 77.84 GiB available memory versus 41.37 GiB required
  and about 21 GiB/s bounded SSD read speed versus the 4.44 GiB/s replay
  threshold. The strict smoke wrapper now executes that selected replay through
  `selected-replay-run --check-ssd-read-speed --write-result` instead of
  calling `generate-prepared-token-ids` directly. The localhost
  `serve-prefill-128-memory-accumulator-safe.sh` wrapper was also moved to the
  same profile/audit/selected-replay gate and no longer exports
  `LARGERLM_MOE_BATCH_ACCUMULATOR`. The new
  `server-http-prefill128-persistent-memory-accumulator-smoke-safe.sh` script
  starts that wrapper, waits for `/health`, posts the 128-token
  `/generate-token-ids` request, saves the raw response, and stops the server.
  It wrote
  `server-http-prefill128-persistent-memory-accumulator-smoke-latest.json`,
  generated token id 15, completed in 60.295s server elapsed with 60.188s
  prompt prefill, and `result-compare` marks it workload-comparable with the
  offline strict replay at 1.003x total elapsed without system slowdown.
  Running the latest generated bounded targets wrote
  `glm-moe-layer19-optimization-target-128tok.json`,
  `glm-mla-layer37-cache-sweep-128tok.json`, and
  `glm-attn-proj-layer37-fusion-sweep-128tok.json`; all three reported
  `candidate_for_full_replay=false`. `tile1_auto_silu` remained the fastest MoE
  kernel at 0.0439s mean, key+value MLA cache remained fastest at 0.0552s total
  mean, and the fused attention-projection path remained fastest at 0.0797s
  wall mean while separate projections were about 3.23x slower.
- Staged-MoE execution now records wall-clock subphases on successful runs:
  compact-stage construction, route writing, static-capacity route
  materialization, runner subprocess wall time, output validation, and total
  wall time. `result-summary` prints these as a `routed moe wall:` line and now
  also reports residual elapsed time outside the recorded wall split. On the
  current 126-token MLA-server baseline rerun, routed MoE took 32.686s: the
  runner-reported total was 15.786s, recorded staged-MoE wall phases summed to
  16.753s, and the remaining outer Python/file orchestration residual was
  15.933s. That makes the next MoE target explicit: instrument and reduce the
  outer orchestration boundary before promoting more local tile/vector/group32
  kernel toggles.
- The opt-in process-boundary prototype is implemented through full prompt
  prefill:
  `metal/largerlm-runner --run-moe-batch-plan --batch-plan-json PATH` reads
  `largerlm.staged_routed_moe_batch_plan.v1` and executes multiple
  already-materialized staged routed-MoE batch jobs inside one runner process.
  The Python `run-staged-routed-moe-batch-plan` wrapper validates expected
  outputs, and the staged MoE Metal smoke now checks this entry point against
  the static-capacity route path. A JSONL server mode,
  `--run-moe-batch-plan-server-jsonl`, now keeps the runner process alive across
  sequential plan submissions, and
  `StagedRoutedMoEBatchPlanServerSession` provides the Python context-manager
  client. This matches the full GLM prefill dependency shape better than one
  giant pre-materialized plan because each routed layer depends on the previous
  layer output. Low-level staged MoE, tiled staged MoE, staged routed MLP,
  prompt prefill, generation, inspection, and serving can now opt into the
  session. Tiled expert stages retain their per-tile stage/compact/scratch
  guards while submitting each tile plan to the same persistent runner.
- Resident o_proj group32 sweeps now write a `config_comparison` gate with the
  same sample-count, drift, and full-replay-bakeoff policy. The generated
  layer-62/128-token attention-output target,
  `glm-resident-o-proj-layer62-optimization-target-128tok.json`, used 256 MiB
  resident-matrix and runner-scratch caps. It kept `auto`/group32 as fastest
  backend mean at about 0.0220s versus about 0.0283s with group32 disabled, so
  no resident projection candidate should be promoted.
- MLA attention summaries now keep `mla_attention_layers` so the slowest
  attention layer remains visible even when routed-MoE records dominate the
  generic top-record list. The MLA target emits a bounded cache-mode sweep. The
  generated layer-53/128-token run,
  `glm-mla-layer53-cache-sweep-128tok.json`, used 256 MiB cache-read,
  resident-matrix, and runner-scratch caps. Current key+value cache stayed
  fastest at about 0.054s total mean; key-only, value-only, and no-cache were
  much slower, so no MLA cache-mode candidate should enter full replay.
- `result-summary` now closes the loop on these bounded experiments: when a
  suggested `--write-result` JSON exists, it reads the small
  `config_comparison` block back into the corresponding `optimization_targets`
  entry and prints the candidate/baseline/fastest verdict. On the current
  128-token artifact, routed MoE, MLA cache mode, attention projections,
  resident o_proj, RoPE split, and cache write all report `candidate=False`,
  matching the separate microbench files and avoiding duplicate reruns of
  already rejected branches.
- Attention projection targets now emit a bounded fused-vs-separate sweep based
  on the slowest `projections_elapsed_seconds` record. The generated layer-64
  run, `glm-attn-proj-layer64-fusion-sweep-128tok.json`, used 256 MiB
  resident-matrix and runner-scratch caps. The current fused path stayed fastest
  at about 0.0825s mean wall time versus about 0.2436s for the separate
  seven-command path, with zero measured output drift, so no
  attention-projection candidate should enter full replay.
- RoPE split targets now emit a bounded fused-vs-old sweep based on the slowest
  `rope_elapsed_seconds` record and GLM-5.2 rope metadata. The generated
  layer-37/128-token run,
  `glm-rope-split-layer37-fusion-sweep-128tok.json`, used 256 MiB runner
  scratch, `qk_nope_head_dim=192`, `rope_theta=8000000`, and interleaved GLM
  rotation. The current fused path stayed fastest at about 0.04195s mean wall
  time versus about 0.04749s for the old split plus RoPE path, with zero
  measured output drift, so no RoPE split candidate should enter full replay.
- Cache write targets now emit a bounded synthetic chunk-size sweep from the
  slowest `cache_write_elapsed_seconds` record. The generated layer-0/128-token
  run, `glm-cache-write-layer0-chunk-sweep-128tok.json`, reads only the
  prepared cache layout metadata, writes a small temporary BF16 cache, and keeps
  the real `decode_cache.bin` untouched. The new chunked bitcast encoder kept
  the default single-chunk path fastest at about 0.00682s mean wall time,
  one-row writes averaged about 0.00751s, all modes were byte-identical, and no
  cache-write candidate should enter full replay.
- Request checks also expose `prefill_acceleration_frontier`, a bounded chunk
  candidate table that shows MPSGraph/custom-Metal matrix counts and emits a
  reusable `--prefill-prompt-chunk-tokens` suggestion when a larger safe chunk
  would satisfy the acceleration gate. Acceleration-gate failures now keep the
  same structured coverage/frontier diagnostics in `request_check`, and the
  suggested argv preserves `--require-prefill-acceleration` and
  `--prefill-min-accelerated-flop-fraction` when those gates are configured.
- `prefill-plan` now emits `suggested_guard_flags` from its chunked routed-read
  estimate, including `--prefill-prompt-chunk-tokens`,
  `--prefill-max-routed-read-amplification`, `--prefill-max-routed-read-gib`,
  and optional SSD-time caps with 5% headroom, so GLM-5.2 launch profiles can
  reuse the plan's bounded chunk/read budget directly.
- The same plan emits `suggested_stage_temp_guard_flags` from the resolved
  chunk, expert slot size, and `--expert-stage-align-kib`, exposing
  `--prefill-max-stage-mib`, `--prefill-max-compact-stage-mib`,
  `--prefill-max-stage-raw-ranges`, and
  `--prefill-max-stage-coalesced-ranges` suggestions before any routed expert
  stage files are created.
- Prepared request checks and required launch-audit replay now bind those
  range-count caps as part of the prompt-specific request profile. Replayed
  profiles must carry the range flags and keep them within the audited maximum
  raw/coalesced range counts plus the standard 5% headroom, preventing a later
  launch from silently accepting more fragmented SSD reads than the checked
  request proved safe.
- Prepared request checks now add `prefill_stage_temp_disk_free`, comparing the
  largest per-chunk routed stage/compact/static-capacity temp requirement plus
  `prefill_stage_disk_margin_mib` with current `/private/tmp` free space.
  Batch-prefill requests fail before generation if that temp disk headroom is
  insufficient or cannot be verified.
- Prepared server token-id, text, OpenAI completion, and OpenAI chat generation
  now run the same request admission before serialized generation, giving live
  service calls the offline `inspect-prepared` safety envelope for chunk sizing,
  routed reads, temp disk, and acceleration coverage.
- `generate-prepared-token-ids` and prepared text generation now run that
  token-count request admission before calling the generator as well, so direct
  prepared launch scripts fail before creating runner work directories when
  their prompt envelope is unsafe.
- Prepared text generation now treats pre-admission prompt tokenization as a
  required safety step. If the tokenizer cannot be loaded or the prompt cannot
  be encoded, the command fails before request admission and before any runner
  call, rather than launching with an unknown prompt size.
- Offline request admission now uses the exact generation overrides that the
  runner will receive, including layer filters, top-k, runtime guard caps, and
  DSA settings, avoiding routed-read or memory estimates based on stale server
  defaults.
- Prepared token-id benchmarks now run the same request admission before
  dispatching `generate_token_ids`, including direct
  `benchmark_prepared_token_ids(...)` API callers, so benchmark sweeps cannot
  bypass prompt chunk, routed-read, temp-disk, decode, or acceleration guards.
- Generation and benchmark launch paths now treat a structured admission result
  with `ok: false` as a hard failure, matching exception-based admission errors
  and preventing future soft-fail checks from reaching the runner.
- Compact-stage fallback copies now use positional chunked reads and reject
  short writes as well as short reads, so the staged MoE runner cannot consume
  silently truncated compact expert files when hardlinking is unavailable.
- `suggested_prefill_guard_flags` combines those plan-level read and stage
  guard suggestions into one deduplicated argv list, avoiding repeated
  `--prefill-prompt-chunk-tokens` entries in GLM-5.2 launch profiles.
- The same combine helper now backs prepared request checks and prepared
  benchmarks, so `inspect-prepared`, `/health`, and actual benchmark telemetry
  all expose a single `suggested_prefill_guard_flags` argv list alongside the
  read/stage detail fields.
- Prefill acceleration coverage now carries FLOPs-weighted diagnostics in
  addition to matrix counts. The estimate uses `2 * prompt_chunk_tokens * rows *
  cols` per resident matrix and reports total, accelerated, custom-Metal,
  unsupported-MPSGraph, and fractional accelerated FLOPs. This gives M5/MPSGraph
  tuning a more honest frontier: accelerating a small matrix no longer looks
  equivalent to accelerating the dominant resident GEMMs.
- The same coverage now reports the non-router unaccelerated gap by routed path
  and backend: streamed routed-expert FLOPs, non-streamed FLOPs, and
  `custom-metal`/`unsupported-mpsgraph`/`other` buckets. This keeps the next
  M5 kernel target visible even when a request technically has router-gate-only
  MPSGraph coverage.
- Prompt prefill now also aggregates actual resident-linear elapsed seconds and
  estimated TFLOP/s by backend from the executed runner calls. This gives M5
  MPSGraph threshold and future MPP comparisons a post-run timing signal that
  includes graph setup, conversion scratch work, and custom-Metal fallback
  overhead instead of only static FLOP coverage. Benchmark launch profiles now
  store the same evidence in `sections.prefill_actual_linear_backend`, and
  applied launch profiles/launch audits preserve it for later calibration
  review. Required launch-audit replay keeps the section optional for older
  profiles, but rejects malformed backend timing evidence when present. When a
  benchmark profile also carries actual acceleration coverage, strict replay now
  cross-checks the two sections so matrix counts, FLOP buckets, accelerated
  backend names, and accelerated FLOP fraction cannot describe different
  prefill runs.
- `--prefill-min-accelerated-flop-fraction` now turns that weighted diagnostic
  into an enforceable bring-up gate. Values above zero imply
  `--require-prefill-acceleration`, and both request preflight and actual
  prompt-prefill coverage must meet the configured accelerated FLOP share.
- Prepared benchmark summaries now include actual
  `prefill_acceleration_coverage` and `prefill_acceleration_frontier` fields
  from the prompt-prefill run, so measured throughput can be read together with
  the resident GEMM backend mix that produced it. The frontier now includes
  layout-derived candidate chunks as well as the actual resolved chunk, so a
  benchmark can suggest the next chunk size that would trigger MPSGraph.
- Actual prompt-prefill and benchmark coverage now preserve
  `mpp_tensor_ops_candidate_*` counts/FLOPs alongside the MPP candidate policy,
  matching request inspection. This keeps M5 neural-accelerator bring-up visible
  in real prompt evidence while still requiring an actually selectable backend
  to satisfy `--require-prefill-acceleration`.
- Benchmark-derived `source="benchmark_actual"` launch profiles now carry the
  prepared memory launch guards and prepared/configured SSD read-speed baseline
  in addition to measured prefill/decode guards, so small measured runs replay
  the same no-OOM envelope on the next launch.
- Benchmark summaries and prepared serving payloads now expose cumulative
  prompt-prefill stage read seconds, the SSD GiB/s estimate, the configured
  seconds cap, the actual cap result, and actual raw/coalesced stage range
  counts alongside staged expert byte telemetry. This makes measured GLM prompt
  runs auditable for read-time budget adherence and read-fragmentation, not
  only byte volume.
- Benchmark-derived launch profiles now preserve that measured cumulative
  read-time and range-count evidence in `sections.prefill_actual_read_time`,
  while keeping the replayable argv under the prefill/decode guard sections.
  Required launch-audit replay now also rejects benchmark profiles whose actual
  routed-expert stage copy exceeded the configured prefill seconds cap, even
  when the planned read-time estimate passed.
- Benchmark-derived launch profiles also preserve actual prefill acceleration
  coverage/frontier evidence. When prefill acceleration is required, strict
  launch-audit replay now rejects artifacts that omit that benchmark-actual
  coverage or carry coverage that no longer passes the configured accelerated
  FLOP-fraction gate. Replay also rejects artifacts whose actual coverage
  disagrees with `sections.prefill_actual_linear_backend`, preventing a profile
  from mixing an MPSGraph coverage claim with custom-Metal backend timing.
- Benchmark profiles now also preserve `sections.decode_actual_read_time` when
  decode step telemetry is present, recording actual decoded expert-read bytes
  and SSD seconds against the profile's suggested decode seconds/token cap, plus
  whether actual bytes stayed within runtime-preflight planned bytes.
- Prepared token/text generation and prepared benchmark admission now pass the
  effective runtime-preflight setting into `inspect_token_request`, so the
  default `preflight_runtime=True` path produces current live-memory/layout
  evidence before runner dispatch instead of only relying on the later generator
  guard. Tests that use intentionally tiny prepared fixtures now opt into
  `--no-runtime-preflight`/`preflight_runtime=False` to keep that debug-only
  behavior explicit.
- Prepared-bound launch profiles can now explicitly replay and lock
  `--no-runtime-preflight`, so debug profiles preserve the same admission and
  generator behavior on reuse. Identity-less `plan`/`prefill_plan` profiles
  still cannot carry that flag, preventing config-only profiles from lowering
  the runtime memory/layout preflight gate.
- `serve-prepared --require-launch-audit` now fail-closes on the current server
  memory guard before starting the process. It reapplies prepared memory defaults,
  compares live-working-set plus free-memory reserve against the current system
  memory snapshot, and records the current server guard next to audited runtime
  memory evidence in `launch_audit_envelope`.
- Bound server launch-audit envelopes are now enforced per request as well as
  at startup. `inspect_token_request` reports the matched prompt/generation
  envelope, and generation rejects over-envelope prompt or max-new-token values
  before chunk planning or runner execution.
- Token generation and serving payloads now expose top-level
  `prefill_actual_read_time` and `decode_actual_read_time` summaries when the
  corresponding stages have SSD-time evidence, so non-benchmark runs can still
  audit actual routed expert read seconds. The prefill summary now also carries
  the prompt-prefill staging counters for serial-vs-unique bytes, alignment
  waste, coalesced savings, read-advice attempts/calls/bytes/failures, read
  amplification, and stage-budget utilization, making short chat smokes useful
  for diagnosing whether SSD time is fragmentation, overread, hinting, or copy
  throughput.
- The tiny prepared generation fixture now exercises batch prompt-prefill
  followed by a real decode step, proving that first-token prefill read-time
  evidence and subsequent decode read-time evidence can coexist in one
  `generate_token_ids` result without bypassing dense+MoE decode records.
- Required launch-audit consumers now validate the applied benchmark profile's
  preserved cumulative prefill/decode read-time evidence, so a benchmark-derived
  profile cannot lose or tamper with its measured SSD-time cap result before
  replay.
- The same benchmark profile also preserves active GLM 4-bit/public-shape
  gates and M5 prefill acceleration/probe gates, so a strict validation run
  cannot be replayed later as a looser launch by accident.
- Prepared runtime-profile health now exposes recommended max-live and
  min-free guard components separately, and new required launch-audit artifacts
  bind those manifest-derived values alongside the effective unified-memory
  budget, reserve, and required-available sum. Strict replay rejects artifacts
  whose recorded GLM-5.2/M5 memory envelope no longer matches the prepared
  manifest, before any runner work starts.
- Prepared SSD read-speed provenance is now launch-audit bound too. New
  artifacts record `prepared_ssd_read_profile_valid` with the prepared
  cold-read speed/source and benchmark metadata when present, and strict replay
  rejects stale artifacts after those manifest fields drift. Runtime overrides
  of `--prefill-ssd-read-gib-s` remain possible, but the prepared package's
  original SSD evidence can no longer be silently rewritten underneath a saved
  GLM-5.2 request envelope.
- Runtime preflight now charges validated resident backing bytes against the
  live working-set budget in addition to the largest non-resident stage peak.
  This is deliberately conservative for GLM-5.2: shared experts, embeddings,
  attention projections, routers, and final logits live in `resident.bin`, so a
  request that fits the per-layer scratch cap can still be rejected before
  runner startup if the resident file plus scratch would exceed the configured
  max-live or min-free unified-memory envelope. `prepare-glm` now writes the
  same resident-inclusive value into the prepared manifest's recommended
  max-live guard, preventing generated launch profiles from recommending an
  obsolete scratch-only cap.
- Execute-mode `prepare-glm` now performs a live-memory admission after dry-run
  sizing and before writing any prepared output. It requires current available
  memory to cover the larger expert/resident packer heap estimate plus the
  prepare system reserve, and records the estimate, requirement, observed system
  memory, and probe source in the manifest.
- The same execute-mode admission now includes a combined prepared-output disk
  budget for packed experts, resident weights, and decode cache plus margin,
  avoiding GLM-5.2-scale runs where separate per-output checks succeed but the
  total prepared package would fill the volume mid-write.
- Low-level `run-staged-routed-moe-batch` results now surface the stage
  manifest's SSD-read telemetry directly: serial assignment bytes, unique
  requested bytes, planned/staged bytes, waste, raw/coalesced range counts,
  amplification, utilization, and read-advice counters. Standalone stage
  experiments can therefore be checked against the same no-surprise envelope as
  prompt-prefill and benchmark runs.
- Expert stage copy now uses explicit bounded positional reads rather than
  file-object seek/read loops: on macOS/Python builds with `os.preadv`, the
  stage command reuses one bytearray-backed buffer per copy loop and falls back
  to bounded `pread` chunks otherwise. This keeps the same no-`mmap`,
  no-whole-range-read memory envelope as the standalone disk benchmark.
- Prompt-prefill results now carry the same actual acceleration coverage and a
  `prompt_prefill_actual` chunk frontier, and prepared generation/server
  payloads expose it under `prompt_prefill`. Small real runs can therefore
  verify whether their resolved chunk reached MPSGraph before moving to larger
  GLM manifests.
- Launch audits now store full request prefill acceleration coverage evidence
  instead of only the pass/fail reason. Required-audit replay validates the
  matrix counts, FLOP buckets, accelerated backend list, and FLOP fraction when
  those fields are present, so a GLM-5.2 launch artifact cannot claim M5
  acceleration coverage with internally inconsistent telemetry.
- Runtime-preflight launch audits now also validate optional
  `prefill_live_memory` evidence. The prompt batch, runner scratch, cache
  read/write, and stage-copy byte caps must reproduce the recorded extra
  live-working-set estimate, keeping M5 prompt-prefill memory caps auditable
  before any runner process starts.
- Required launch-audit replay now validates optional `prefill_cache_io`
  evidence as well. MLA cache and DSA index-cache read/write byte buckets must
  sum to the recorded totals, making long-prompt GLM-5.2 cache I/O pressure
  auditable alongside routed expert SSD pressure.
- Routed prompt chunk-frontier evidence is now strict-audit checked when
  present. Candidate chunk sizes must be sorted and include the resolved and
  max-safe chunks, while read amplification and stage/static byte totals must
  remain internally consistent. This keeps GLM-5.2 prompt-chunk tuning evidence
  replayable instead of advisory-only.
- Required launch-audit replay now also requires batch-prefill
  `prefill_prompt_chunk_plan` evidence. The request evidence, audit check, and
  `request_launch_profile` section must match, and the max-safe plan must carry
  `max_matrix_scratch_bytes`/`next_token_matrix_scratch_bytes`, so MPSGraph F32
  conversion scratch is bound before a saved GLM-5.2 launch envelope can be
  reused.
- Stage-temp launch-audit evidence is now strict about the full prompt prefill
  temp envelope: required replay validates stage, compact, and static-capacity
  route-table bytes together, checks the combined
  `max_stage_plus_compact_plus_static_bytes` and total fields, and requires
  `serve-prepared` to replay `--prefill-static-capacity-per-expert` just like
  offline prepared generation. This prevents a GLM-5.2 audit artifact from
  budgeting the staged expert files while silently omitting the fixed-shape
  `LLMSCAP1` route table.
- Batched decoder memory-link telemetry is now wired through end to end. The
  runner writes real JSON booleans for `input_in_memory`/`output_in_memory`,
  `run_decode_layers`, token generation, CLI summaries, and server payloads
  preserve them, and strict tests reject non-boolean report values. On the
  canonical GLM-5.2 MXFP4 1-token run,
  `direct-canonical-layer-memory-telemetry-smoke.json` generated token `[15]`
  in 6.328s with 78 decode layers, 77 in-memory inputs, and 77 in-memory
  outputs. The HTTP smoke
  `server-http-canonical-layer-memory-telemetry-smoke.json` matched the same
  token and memory-link counts in 6.397s. Both kept the decode routed-read cap
  passing and the live-memory guard under its 17.37 GiB cap.
- Metal final logits can now fuse behind batched decoder layers. Passing
  `--output-topk-json` to `--run-decoder-layers` keeps the final hidden vector in
  memory, calls Metal final logits in the same runner process, and stores
  logits elapsed/read metadata in the top-k JSON. Guarded token generation uses
  this path when the batched decoder branch is eligible. The canonical GLM-5.2
  MXFP4 direct smoke `direct-canonical-fused-final-logits-smoke.json` generated
  token `[15]` in 6.311s with logits at 0.052s, 77 in-memory inputs, and all 78
  layer outputs in memory. The HTTP smoke
  `server-http-canonical-fused-final-logits-smoke.json` matched token `[15]` in
  6.355s with the same memory-link counts. Both kept the strict live-memory and
  decode routed-read gates passing.
- A post-telemetry shortchat64 max4 replay wrote
  `server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-telemetry-latest.json`
  (SHA `b3d6ccc35ccf5e73c0603336884a8b5db15d35e377c430dadb0455d3e0d4e928`)
  and preserved the baseline output `0003` / `[15, 15, 15, 18]`. The new
  top-level `prefill_actual_read_time` fields show 145.66 GiB serial assignment
  reads, 34.664 GiB unique/planned reads, 0 alignment waste, 111.03 GiB
  coalesced savings, read advice on all 1,678 ranges with no failures, unique
  read amplification 1.0, and expert stage copy at 6.214 GiB/s. This confirms
  the current shortchat SSD plan is already perfectly unique/coalesced for the
  selected routes; the next M5 work should target range-count/copy overhead and
  fused streamed-expert execution rather than chasing alignment waste.
  `result-summary` now prints the same data as an `expert stage io:` line so
  future smoke logs do not need ad hoc JSON extraction.
- Copy-call telemetry is now wired from `stage-batch-experts` through prompt
  prefill, token generation, server payloads, and `result-summary`, including
  stage-time counterfactual read-call estimates for 8/16/32/64/128 MiB copy
  chunks. The bounded shortchat64 max4 replay
  `server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-copycounter-latest.json`
  (SHA `db5907384b58ad1d2b48a0e6910b7460d5d24ca9353a7b318635b66ea68c22d8`)
  preserved `0003` / `[15, 15, 15, 18]` while reporting 5,409 read calls,
  5,409 writes, and 6.562 MiB average copy calls for the 34.664 GiB stage. The
  counterfactual estimates are 5,409 reads at 8 MiB, 3,534 at 16 MiB, 1,837 at
  32 MiB, 1,679 at 64 MiB, and 1,678 at 128 MiB. Runtime preflight stayed
  inside the no-OOM envelope with about 83.583 GiB available versus 44.417 GiB
  required. The evidence narrows the next SSD experiment: copy32/copy64 can
  remove most extra copy calls, while larger chunks are already at the range
  floor and route/layout changes are needed for further range-count reduction.
- A locked copy64 A/B was generated from the current shortchat64 max4
  memory-accumulator profile. The profile
  `launch-profile-shortchat-64tok-copy64-keycache-memory-accumulator-decode-keycache-request-locked.json`
  has SHA `49b491d91fc75ad1448cd2901fb40506ba803e290b2c5d88c55f74b2e311414d`;
  the audit
  `launch-audit-shortchat-64tok-copy64-keycache-memory-accumulator-decode-keycache-request-max4-locked.json`
  has SHA `4de2a00d79c7f88c2b46571d5ac5a91058b6a26e62e7be8b6af9abf162af0860`
  and passes with the explicit non-accelerated prefill audit allowance. New
  wrappers
  `serve-shortchat-64-max4-kvbcache-memory-accumulator-copy64-safe.sh` and
  `server-http-shortchat64-max4-kvbcache-memory-accumulator-copy64-smoke-safe.sh`
  keep the profile/audit binding reproducible without changing the default
  server. The smoke
  `server-http-shortchat64-max4-kvbcache-memory-accumulator-copy64-decode-keycache-openai-chat-latest.json`
  (SHA `eff7549c0fcdd5cdde43531ecb9280c7440614a562bd4dd5fbe04be2ea91bf27`)
  returned HTTP 200, `0003`, and `[15, 15, 15, 18]`. It reduced stage-copy
  syscalls from 5,409 read/write calls to 1,679 read/write calls, with
  21.141 MiB average calls and 5.643s at 6.143 GiB/s for the same 34.664 GiB
  stage. Compared with the 5.610s default copy-counter run this is a syscall
  success but not yet an end-to-end speed win, so copy64 remains an opt-in
  candidate until repeated low-noise runs justify promotion.
- A copy32 A/B was generated to test the smaller-memory near-floor alternative.
  The profile
  `launch-profile-shortchat-64tok-copy32-keycache-memory-accumulator-decode-keycache-request-locked.json`
  has SHA `92685163105ea6d59d83a198e4c9f2e3c583210b7468557c082caab84b279c61`;
  the audit
  `launch-audit-shortchat-64tok-copy32-keycache-memory-accumulator-decode-keycache-request-max4-locked.json`
  has SHA `5f27865e1d818161925f4f5013bd9a5daf657361de34b90ee50495beec5dac79`
  and passed with runtime preflight at about 83.47 GiB available versus
  44.42 GiB required. New wrappers
  `serve-shortchat-64-max4-kvbcache-memory-accumulator-copy32-safe.sh` and
  `server-http-shortchat64-max4-kvbcache-memory-accumulator-copy32-smoke-safe.sh`
  keep it opt-in. The smoke
  `server-http-shortchat64-max4-kvbcache-memory-accumulator-copy32-decode-keycache-openai-chat-latest.json`
  (SHA `72466004662d97424cc06f8b09338d8f63dc919f0b1e90ad0214ee38e9bb4133`)
  returned HTTP 200, `0003`, and `[15, 15, 15, 18]`, but stage copy slowed to
  7.336s at 4.725 GiB/s despite reducing copy calls to 1,837 read/write calls.
  Do not promote copy32 from this evidence; it confirms that syscall count is
  not currently the dominant first-token SSD bottleneck once the range floor is
  approached.
- Stage/layer IO hotspot telemetry is now present in actual prefill results.
  Prompt prefill records the top five copy-time and coalesced-range hotspots
  across routed expert stages, and token/server result payloads expose them as
  `expert_stage_copy_hotspots` and `expert_stage_range_hotspots` under
  `prefill_actual_read_time`. `result-summary` prints those rows as
  `expert stage hotspots:` with chunk/layer/tile, copy seconds, raw/coalesced
  ranges, planned bytes, and copy calls. The next range/layout experiment can
  therefore start from the worst layers or tiles rather than another global
  copy-chunk sweep.
- The first real hotspot smoke with selected expert ids,
  `server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-hotspots-selected-latest.json`
  (SHA `b81f1e26ca84ea538e18d492af0565491c510d529f695318573a5369879999df`),
  preserved HTTP 200, `0003`, and `[15, 15, 15, 18]` under the default 8 MiB
  copy profile. It reported 75 routed expert stage calls, 34.664 GiB planned
  stage reads, 5.589s copy at 6.203 GiB/s, and no residual server process
  after the smoke. Copy hotspots ranked layers 67, 55, 62, 65, and 59; range
  hotspots ranked layers 67, 46, 65, 62, and 57. Layer 67 is the first concrete
  target because it is worst by both copy time and range count: 36 coalesced
  ranges, 726.750 MiB planned bytes, 112 copy calls, and 0.151s stage copy.
  Its 38 selected expert ids include only two adjacent pairs in the current
  expert-id order, with max gap 22 and mean gap 6.78, so a layer-local
  coactivation-aware expert order is a plausible next range-reduction
  experiment.
- `scripts/stage_hotspot_layout_analysis.py` now converts bounded hotspot rows
  into selected-expert locality diagnostics and coactivation-order candidates
  without loading weights. The first v2 analysis artifact,
  `stage-hotspot-layout-analysis-shortchat64-hotspots-selected-v2-latest.json`
  (SHA `db78abbe2a7eab0208d32981466f139066a992d9748109609db701f60642928c`),
  dedupes copy/range hotspots into seven stage targets and emits full
  layer-local expert orders from the observed coactivation graph. Its top five
  layout targets are layers 67, 65, 46, 62, and 57. For layer 67, the current
  order has 36 runs for 38 selected experts; the candidate order clusters those
  observed experts into one simulated range, a 35-range reduction for this
  single prompt. Treat that as an upper-bound locality signal, not as a
  production reorder policy, until multi-prompt sampling proves the order is
  stable.
- Two additional safe smokes were collected with short prompts
  `请用一句话介绍杭州` and `Explain AI in one sentence.`, writing
  `server-http-shortchat64-max4-hotspots-selected-hangzhou-latest.json`
  (SHA `4dcd40304abf7d365d57e869c566628290d83c3e6a378c358525d9ef8e551075`,
  output `3333` / `[18, 18, 18, 18]`) and
  `server-http-shortchat64-max4-hotspots-selected-explain-ai-latest.json`
  (SHA `03a7c64a2627d4cbed4bad9d328a19b8b6738a3aa662061113d2535131d280a9`,
  output `0303` / `[15, 18, 15, 18]`). Both ended with no residual `largerlm`
  process. The three-prompt coactivation analysis
  `stage-hotspot-layout-analysis-shortchat64-hotspots-selected-3prompt-latest.json`
  (SHA `da7edd7084915c3a424679ef970b80f2692b7b6294437fe3aa4e92179d6d7d05`)
  keeps layer 67 as the top target: 3 hotspot samples, 44 observed experts,
  109 current ranges, 4 simulated candidate ranges, and 0.485s cumulative copy
  time. Layers 62, 46, 57, and 65 follow. This is now strong enough to justify
  a dry-run repack/indirection planner, but not yet enough to rewrite the
  prepared expert files by default.
- Expert IO planning and staging now accept an optional per-layer
  `expert_order` field for future repacked layouts. The field lists logical
  expert ids in physical slot order; router outputs remain logical ids, while
  `plan_expert_io` maps them to physical offsets and `stage_batch_experts`
  copies from those offsets. Identity layouts keep the old behavior. Unit
  coverage verifies range reduction for reordered metadata, rejects malformed
  orders, and confirms staged bytes come from the expected physical slots.
- `scripts/expert_order_repack_plan.py` now converts a coactivation analysis
  artifact into a dry-run repack/indirection manifest. The first real plan,
  `expert-order-repack-plan-shortchat64-3prompt-top5-dryrun-latest.json`
  (SHA `a91fed3a207b192f053f5303d4a51047f3d0fc7ea12afad903651f46a19f2ed6`),
  selects layers 67, 62, 46, 57, and 65. Each layer file is 4.781 GiB, so the
  top-5 dry run estimates 23.906 GiB read, 23.906 GiB write, and 47.812 GiB
  total repack IO. The manifest is intentionally marked
  `safe_to_apply_to_existing_layout=false` and `requires_layer_file_repack=true`
  because adding `expert_order` metadata without physically reordering the layer
  file would corrupt expert reads.
- `scripts/expert_order_repack_execute.py` is now the bounded executor for that
  manifest. It defaults to dry-run; an explicit `--execute` is required before
  it creates an output experts directory. On execute it rewrites only selected
  layer files slot-by-slot with bounded `pread`/`pwrite` chunks and hardlinks
  unchanged layer files. The real top-5 plan was run through executor dry-run as
  `expert-order-repack-execute-shortchat64-top5-dryrun-latest.json`
  (SHA `79ba27d4e657d1457440e383d9aaa2787192232948aae63b32bb87c60f50504e`).
  The dry run did not create `experts-repacked-shortchat64-top5-dryrun`; it
  reports 23.906 GiB repack read, 23.906 GiB repack write, and 334.688 GiB of
  unchanged layer files that would be hardlinked. Small-fixture tests verify
  selected layer slot reordering, unchanged-layer hardlinking, non-empty output
  rejection, and dry-run no-output behavior.
- The executor now has a default write guard: `--execute` refuses plans above
  8 GiB of repack writes unless explicitly overridden, and it requires at least
  20 GiB free after the planned write. A bounded top1 run for layer 67 passed
  that guard and created `experts-repacked-shortchat64-top1-layer67` without
  modifying the original `experts` directory. The execute manifest
  `expert-order-repack-execute-shortchat64-top1-layer67-executed-latest.json`
  (SHA `515c69cac65d53348d4e5939ce7dbdc452bad899c863aa873f564fafb1b25aee`)
  reports `executed=true`, 4.781 GiB repack read/write, layer-67 output SHA
  `cd64b4766b4611c18e44ae61e0f2c8a6584c961992a0646963780a6c6895b06e`, and
  hardlinks for unchanged layers. Validation
  `expert-order-repack-validate-shortchat64-top1-layer67-latest.json`
  (SHA `39c0a8321606b55cd008597e4d73fd7c87648a1b21e92fe472f6303db4246250`)
  sampled nine layer-67 slots successfully and recomputed the three observed
  layer-67 hotspot rows from 109 old ranges to 4 new ranges.
- A validated shadow prepared manifest now tests that repacked layer without
  altering the original package: `manifest-repacked-layer67.json` (SHA
  `c7f7c465ea1d7426fedeb86f48c7c23c00519c7354506f6eb32aafbf78561223`)
  points `experts_layout` at
  `experts-repacked-shortchat64-top1-layer67/layout.json`. The matching locked
  profile/audit are `launch-profile-repacked-layer67-shortchat64.json` (SHA
  `e10a7028a35e54c3032a45443fcee35c990b1b4b7144d4f82df276e60ef01bc1`) and
  `launch-audit-repacked-layer67-shortchat64.json` (SHA
  `116bd17129efea8a12030d228e9e505aa0a358afad3f58d2285915f20e822595`). The
  first audited direct-generation smoke,
  `repacked-layer67-shortchat64-max1-generate-tokenids-latest.json` (SHA
  `31090658257b9b485e7c0ade414289425d6baa59c7d7dc0fe3133c01add266da`),
  generated token id `[15]` and exited with no residual runner. That run also
  forced a runtime integration fix: `run-staged-routed-moe-batch` now validates
  stage slot `source_offset` against `batch_plan.io_plan.expert_physical_slots`
  when present, preserving strict identity-layout validation otherwise. In the
  real smoke, layer 67 staged 38 selected experts as one raw/coalesced range
  and no longer appeared in the top copy/range hotspot tables.
- The bounded executor has now also materialized the full top-5 locality plan
  as `experts-repacked-shortchat64-top5`, rewriting layers 67, 62, 46, 57, and
  65 under a 32 GiB write cap with a 20 GiB free-space floor. Execution artifact
  `expert-order-repack-execute-shortchat64-top5-executed-latest.json` (SHA
  `65828bef845fd1cdd19f32c66a175b96161e030211c7cf3ff31237799ee90ab3`)
  reports 23.906 GiB read/write and output layout SHA
  `b46af7baaf958e833c1606652f80774822722911ee6ca9fba78f4238de0d7a23`.
  Validation
  `expert-order-repack-validate-shortchat64-top5-latest.json` (SHA
  `5e7f2833e0a6e28403f04490dfcdb268132b0c135f5a4124b3306f3fec717c83`)
  sampled all five repacked layers and reduced the three-prompt analysis from
  692 old coalesced ranges to 254 new ranges, a 438-range reduction. The
  top-5 shadow manifest/profile/audit are `manifest-repacked-top5.json` (SHA
  `a7f723d5e7c1f554e0976896aa344543ef8fa682b7355891364342506ec7e2e1`),
  `launch-profile-repacked-top5-shortchat64.json` (SHA
  `d1ccde3793e799b819744fdc270cbb67a0b2e0a41cb2a2092c1c10c367cc9fec`), and
  `launch-audit-repacked-top5-shortchat64.json` (SHA
  `3437710bb4801757fdb4de68d1f81e0b37b1184def220f3e106db0e7e7f221f0`).
  The first audited direct-generation smoke,
  `repacked-top5-shortchat64-max1-generate-tokenids-latest.json` (SHA
  `1022c4e89c7c2740d8f635320d59ce98067f9480324697bce3d5deb2240ce76a`),
  generated `[15]`, left no residual runner process, and confirmed the target
  layers in real execution: layers 46, 57, 65, and 67 each staged as one range,
  while layer 62 staged as two. Total prompt expert-stage coalesced ranges
  dropped from 1,643 in the layer67-only shadow run to 1,526.
- The same top-5 shadow manifest has an experimental MPSGraph 13x32
  profile/audit:
  `launch-profile-repacked-top5-shortchat64-mpsgraph13x32.json` (SHA
  `e3c9c2433aa4495abcd5be29edfdeb383e5c02e15509309b362adefdd8d43450`) and
  `launch-audit-repacked-top5-shortchat64-mpsgraph13x32.json` (SHA
  `43558ef3809f4f4a6fe871fb0d04053fc8ba4a8cf3dd8a9a9ab9874163f245e5`).
  Run those MPSGraph/Metal probes outside the restricted command sandbox; the
  sandbox can report `no Metal device` even when the host MPSGraph probe passes.
  The audited direct-generation smoke
  `repacked-top5-shortchat64-max1-mpsgraph13x32-generate-tokenids-latest.json`
  (SHA `83c36a1292b1458873f0adb5e198f4cf5c987cacb6f8de5f9d758cf40f9ba74d`)
  proved actual MPSGraph execution on 75 router-gate matrices, with actual
  backend counts `custom-metal=234`, `fused-metal=387`, and
  `mpsgraph-f32=75`. Its accelerated FLOP fraction was only
  0.0031039834454216243, `accelerated_router_gate_only=true`, and the generated
  token changed from the custom-metal top-5 `[15]` to `[23]`. The changed router
  path also shifted real stage locality: layers 46, 57, 62, 65, and 67 staged as
  1, 3, 4, 8, and 8 coalesced ranges respectively, with 1,565 total prompt
  expert-stage ranges. Keep this profile as gated M5 acceleration bring-up
  evidence only; the safe top-5 default remains the custom-metal profile until
  router parity is solved.
- A follow-up locality-only planning wave used the custom-metal top-5 smoke as
  the source and the top-5 repacked layout as the current physical order. The
  analysis artifact
  `stage-hotspot-layout-analysis-repacked-top5-shortchat64-latest.json` (SHA
  `e02648ead64b2fc435328dd25fe6eb04c19a31674f4278221be0ce1b53e9fd5a`) points
  at layers 76, 54, 68, 47, and 55 as the next range targets. The dry-run plan
  `expert-order-repack-plan-repacked-top5-shortchat64-next5-dryrun-latest.json`
  (SHA `ccc031d892fb05bd3d05fa42ab5fdd52c8fe892fb03d863f28f81de24bbc8db5`)
  would reduce the observed rows from 30/30/29/28/28 ranges to one range each,
  but remains `safe_to_apply_to_existing_layout=false` and
  `requires_layer_file_repack=true`. The executor guard dry-run
  `expert-order-repack-execute-repacked-top5-shortchat64-next5-dryrun-latest.json`
  (SHA `99ee097a164f45aec9626715d2c45ddfabd1a39abe61449e42381a415c8a87f6`)
  left `executed=false`, did not create `experts-repacked-shortchat64-top10`,
  and reports 23.906 GiB of repack read/write, 334.688 GiB of hardlink-referenced
  unchanged layer files, `copy_chunk_bytes=67108864`, and a passing write guard
  with a 32 GiB write cap and 20 GiB post-write free-space floor. This is a
  candidate top10 plan, not a promoted layout: it is based on one post-top5
  prompt and should either collect more prompt evidence or be explicitly
  accepted as a bounded 23.906 GiB follow-up experiment.
- `expert_order_repack_plan.py` now has a `--min-sample-count` evidence gate so
  large expert rewrites can require repeated hotspot evidence before selection.
  Replaying the three original shortchat64 hotspot prompts against the top-5
  physical layout wrote
  `stage-hotspot-layout-analysis-repacked-top5-shortchat64-3prompt-latest.json`
  (SHA `0981212556403247e8bd72b7ca4f089b3691698528f80c11c458f60fc4e4cd44`).
  With `--min-sample-count 2`, the guarded plan
  `expert-order-repack-plan-repacked-top5-shortchat64-3prompt-minsample2-dryrun-latest.json`
  (SHA `1b963e2663ee9ecc7d92fb488a2b9c57978a256623e812f221519e152488cf0b`)
  skipped the already-applied top-5 layers and selected only layer 55. The
  executor dry-run (SHA
  `bb6fb6fa1ab310484b436e170e571d10e7c77abec44f366b1bf41d80ac98fd5c`) passed
  the default 8 GiB write cap and 20 GiB free-space floor, so the bounded
  follow-up executed as `experts-repacked-shortchat64-top6-minsample2-layer55`.
  Execution SHA is
  `a75c81bfd3f71ad4cb887e9f0c5486512b74a23b67423ed97c46725506d37ef5`, and the
  output layout SHA is
  `b756215dd142494605b266d6ecbf453b016a17fbacad3115c2b5cb7f947c5cff`.
- The repack validator now maps logical expert ids through the source layout's
  existing `expert_order`, fixing incremental source-layout validation. With
  that fix, `expert-order-repack-validate-top6-minsample2-layer55-latest.json`
  (SHA `6123328260b60550890663fc4bc52bb84a20c4b1eefbf6e8839e21c8d808565b`)
  reports `slot_validation_ok=true` for layers 46, 55, 57, 62, 65, and 67. The
  three-prompt range total drops from 254 to 199, with layer55 rows going
  29->1 and 28->1 ranges.
- The top6 shadow manifest/profile/audit are
  `manifest-repacked-top6-minsample2-layer55.json` (SHA
  `5b3a1478fe4e1f80a5907e6372a677c6bc0075aa257ed6355c18922a80d34288`),
  `launch-profile-repacked-top6-minsample2-layer55-shortchat64.json` (SHA
  `52311c0018d3acbf7f96ec542daa70763f8dd94e9ac57b510b1343a768298126`), and
  `launch-audit-repacked-top6-minsample2-layer55-shortchat64.json` (SHA
  `7ed4453f331fae83a6d43b3390c8cbfb83472bcf548d971e56e0126ddeffc046`). The
  audited max1 smoke
  `repacked-top6-minsample2-layer55-shortchat64-max1-generate-tokenids-latest.json`
  (SHA `b958e52d56d41724160ad4a96895b34349627ccd187ddb9206b10c91de44763d`)
  generated `[15]`, left no residual runner process, reduced real prompt
  expert-stage ranges from 1,526 to 1,499, and staged 34.664 GiB in 5.786s at
  5.991 GiB/s. `result-compare` against top5 reports the same generated token
  and comparable plan, total 238.804s versus 242.862s (`0.983x`), no system
  slowdown, and `candidate_promotable=false` because the end-to-end gain is
  within the two-percent tie band. Treat top6 as a validated candidate rather
  than the new default until another replay or bakeoff clears the promotion
  policy.
- The top6 13-token optimization-target queue has been swept under bounded
  caps. `glm-moe-layer55-optimization-target-13tok.json` (SHA
  `b6c9a11c93c1b5a86165c6a5321959c0be16f16e824bf2b4c85dd05ee0cbe9c4`) kept
  `tile1_auto_silu` as both baseline and fastest routed-MoE config
  (`0.04106175s` kernel, `0.076913s` runner total) after staging 681,836,544
  bytes across 28 coalesced ranges at 5.0205 GiB/s.
  `glm-mla-layer57-cache-sweep-13tok.json` (SHA
  `65087953d11cdac33574667c79867bd4b7957b3553403872215c0409bdfd7e6f`) kept
  `key-value` fastest (`0.04116s` total, `0.00500875s` kernel), while
  `key-only`, `value-only`, and `none` were slower despite exact agreement.
  `glm-attn-proj-layer0-fusion-sweep-13tok.json` (SHA
  `0f42075808d09a48dfd131676b682bfc1eb35097d2c620833488b83d9ced03bb`) kept
  `fused` fastest (`0.060548s` versus `0.233784s` separate). The layer57
  attention-output group32 sweep (SHA
  `aea85d6e98a39a12342f546e462c80d629b45dc4889b912c759c9f2c611bdca0`) kept
  `auto` fastest (`0.046369s` wall, `0.013480s` backend). The layer27
  cache-write sweep (SHA
  `00bd527358e37f4f0aff88a9952f696276a1f2f5c6f113c52230c1fc2bc90042`) found
  `1MiB` slightly fastest (`0.000788s`) but below the two-percent promotion
  threshold versus default (`0.000795s`), so it is not a replay candidate.
- The same queue produced one microbench candidate: the RoPE split sweep
  `glm-rope-split-layer57-fusion-sweep-13tok.json` (SHA
  `45924ee3eacf0fdf53fa8503bc20a64a90624bca25c17947c9cea74c564aaf94`) found
  the older Python split plus `--run-rope-batch` path at `0.034296s` versus
  fused split at `0.035488s`, with exact agreement. Full replay with
  `LARGERLM_DISABLE_FUSED_ROPE_SPLIT_BATCH=1` wrote
  `repacked-top6-minsample2-layer55-oldrope-shortchat64-max1-generate-tokenids-latest.json`
  (SHA `1c00f6d3394dd99d11c3ae0e7b316db3c79522d402e121cc6b8490ce13592334`),
  generated `[15]`, and finished in 235.592s versus 238.804s for top6 fused.
  `result-compare` called the workload comparable and no-system-slowdown, but
  `profile recommendation: tie` because the ratio was only `0.987x`. The
  policy bakeoff `top6-oldrope-policy-bakeoff-latest.json` (SHA
  `1aad770c0d10fa37bd9fa8a514bd67764c4772a7097c01d8858c03d5f18e625f`)
  retained the fused top6 baseline; selected replay JSON/script SHAs are
  `5d03a9a3c134a45ba5413e0b38da17dc12b86272f6e5b9752358dca59b525dc6` and
  `dea363f7a6ae1ff23f73ad625a82eeab35fefdef7b4d672719c3368ba974d5f3`.
- Added `scripts/multi_prompt_replay_plan.py` to build a safe replay matrix
  from existing HTTP/generation result JSONs. It extracts `prompt_token_ids`
  from prepared results or OpenAI-style `raw_response.largerlm.token_result`,
  verifies each variant's prepared manifest/profile/audit files, records
  prompt-token limits, can reuse existing results by variant plus prompt-token
  match, and writes guarded commands that refuse to run if an existing
  LargerLM/Metal runner process is present. The current top5-vs-top6 plan,
  `multi-prompt-replay-plan-top5-vs-top6-latest.json` (SHA
  `6908904cff5a042ce0237a66743101da7028483829ce69e8385bcefa9820689a`), covers
  the 13-token original prompt, 17-token Hangzhou prompt, and 19-token
  explain-ai prompt across top5 and top6. All six tasks are ready; the refreshed
  plan reuses the existing 13-token top5/top6 results and records the Hangzhou
  and explain-ai pairs as already present. The generated script SHA is
  `5e3c7f1afee1c5c3f9189f7ae28dd184b5194405e0add50cf3a3697359e2f88b`.
- Executed the first new matrix pair on the 17-token Hangzhou prompt under the
  locked top5/top6 audits. Top5 wrote
  `multi-prompt-replay-top5-vs-top6/shortchat64-top5-vs-top6-top5-server-http-shortchat64-max4-hotspots-selected-hangzhou-latest-max1-generate-tokenids.json`
  (SHA `559009e3ee1ce80714ea7f25611dc7cd97a6837b948946d3d99d3289fe82b665`) and
  top6 wrote the matching `top6-...hangzhou...` result (SHA
  `20f84813c46ac0f9eba23c7cdcba6f09b277b3ab9c878aab537635f082ff7a55`). Both
  generated `[18]` with no residual runner afterward. Top6 reduced prompt
  prefill from 88.403s to 87.220s and expert-stage copy from 7.852s to 6.497s,
  but total elapsed regressed from 279.733s to 281.503s because final logits
  rose from 191.324s to 194.277s. `result-compare` reported a comparable
  strict-prefill-plan workload, no system slowdown, and a 1.006x tie. The
  bakeoff `multi-prompt-top5-vs-top6-hangzhou-bakeoff-latest.json` (SHA
  `ce53d5ddeb5432f6bb310986bb2a9c9f1e1e7937ea0e2521f924e095b877e5bb`)
  retained top5; selected replay JSON/script SHAs are
  `096a3058dc728fa2bb2243c19cb36e6eeb87a791cc589aa9421ee8eee5f45910` and
  `97f47ee865f519d0c2ac70956fe79b52827e1db909e76057deeff2a7f2ff9915`.
- Added `scripts/multi_prompt_bakeoff.py` to aggregate the replay matrix with
  the existing `result_bakeoff_files` promotion policy. The current aggregate
  `multi-prompt-top5-vs-top6-bakeoff-latest.json` (SHA
  `96ef82b762d7de912a5acbf4b299f34dbe13a08e29dab2a5eef27da16336f12c`) now
  reports all three prompt pairs complete, with top6 at zero wins, two ties, and
  one inconclusive pair. The overall decision is `retain_baseline`, selected
  variant remains top5, and top6 cannot be promoted under the same replay
  readiness gates.
- Completed the 19-token explain-ai pair under the locked top5/top6 audits.
  Top5 wrote
  `multi-prompt-replay-top5-vs-top6/shortchat64-top5-vs-top6-top5-server-http-shortchat64-max4-hotspots-selected-explain-ai-latest-max1-generate-tokenids.json`
  (SHA `ef22c1144a3f4d72d6c8f14d86c40d2f673ff1cc352d712162aa6fa55c2da350`) and
  top6 wrote the matching top6 result (SHA
  `3e66ca65f36197659552634a0f57c5cfb8bf260c014dcb6cbd6754c9e33fbb9d`). Both
  generated `[15]`. Top5 finished in 339.474s; top6 finished in 536.748s.
  `result-compare` called the workload comparable but flagged possible
  system-level slowdown (`median_sentinel_ratio=2.151x`, total ratio `1.581x`),
  so the profile recommendation was `inconclusive` and not promotable. The
  prompt bakeoff
  `multi-prompt-top5-vs-top6-explain-ai-bakeoff-latest.json` (SHA
  `eb74fef6ef55b8f5163635bf2a74ab9c735c329d1810c29a67eb6e1d20cdfa40`) retained
  top5; selected replay JSON/script SHAs are
  `cfd65ce9bce2a10e6d5d228334957768bb0203668defef42e186225e053ee3e9` and
  `5fae54acf236a34105305402fc474b158c2d71d1dc1794e669e998b019b58052`.
- Runtime/live-memory guard failures now preserve structured diagnostics instead
  of collapsing to a string at the HTTP boundary. `GenerationGuardError`
  carries fields such as `code`, `required_available_memory_bytes`,
  `system_available_memory_bytes`, `system_memory_source`, and
  `available_memory_ok=false`; `inspect_token_request` wraps those into
  `request_check.runtime_preflight`; and HTTP 400 responses for request-check
  failures include the `request_check` payload. This keeps the no-OOM behavior
  auditable for UI/API clients: a refused request can be reported or retried
  as low-memory admission failure without confusing it with a server crash.
- The top-5 `result-summary` MLA-attention suggestion was tested with the
  bounded single-layer sweep
  `glm-mla-layer77-cache-sweep-13tok.json` (SHA
  `479876356e851b3f256215ac97d86288cc3a7469243fab12cfbb42d654efee45`). It ran
  `key-value`, `key-only`, `value-only`, and `none` with four interleaved
  repeats, 13 batch/context tokens, and 256 MiB-class cache/read/resident/scratch
  caps. All modes matched numerically, but `key-value` stayed fastest
  (`0.04148s` total mean, `0.0051505s` kernel mean); `key-only` was 1.078x
  slower by total mean, and `value-only` / `none` were about 2.61x / 2.63x
  slower. The artifact therefore records `candidate_for_full_replay=false` with
  reasons `no_candidate_met_total_speedup_and_drift_policy` and
  `baseline_has_fastest_total_mean`. Keep the current MLA key-value cache mode
  for the 13-token top-5 path.
- The paired attention-projection fusion sweep
  `glm-attn-proj-layer54-fusion-sweep-13tok.json` (SHA
  `86c6bc49f35f476a3352bb79c0dd928c3cd83b1a1865d7ecbf3ce4fbda186d83`) compared
  fused and separate layer-54 projection paths with four interleaved repeats and
  the same 256 MiB-class caps. Outputs matched exactly. `fused` stayed fastest
  at `0.05924713575s` mean wall time and one command, while `separate` took
  `0.23221686425s` mean wall time and seven command-equivalent calls
  (`3.919x` slower). The result records `candidate_for_full_replay=false`, so
  the current fused projection path remains the 13-token top-5 policy.
- Four more top-5 13-token result-summary targets have bounded evidence. Routed
  MoE layer 67,
  `glm-moe-layer67-optimization-target-13tok.json` (SHA
  `ddb6c44b801832fe7e82d6b678ed984726f36063e3346a0809d074cf4a8b0c3c`), kept
  `tile1_auto_silu` fastest (`0.0436305s` kernel mean,
  `0.08263s` runner-total mean); tile2 and vector-SwiGLU variants were slower,
  so `candidate_for_full_replay=false`. Attention output layer 8,
  `glm-resident-o-proj-layer8-optimization-target-13tok.json` (SHA
  `f9b6fec4ad7bb01677074de51778dacf18b70ab67fa2a8092e5637d44700aa92`), kept
  `auto`/group32 fastest (`0.01742075s` backend mean), with `off` 1.120x slower
  by backend mean. RoPE split,
  `glm-rope-split-layer14-fusion-sweep-13tok.json` (SHA
  `ebcf15e3ca8e5c9a4bbd685621f927370d1787d3d3f50d3eae38c15e126ca152`), kept
  fused fastest (`0.04801492725s` wall mean), with the old path 1.037x slower.
  Cache write,
  `glm-cache-write-layer12-chunk-sweep-13tok.json` (SHA
  `6226efa59d03ae948a91bd289491b09c4d700fdb0692946d1a586beaa6082931`), is the
  only new microbench candidate: `1MiB` chunking was byte-identical and 0.975x
  the default wall mean, but the absolute mean delta is only about 38 us on this
  13-token probe. It still requires full replay and bakeoff before any policy
  change, and is lower priority than larger end-to-end bottlenecks.
- Long-prompt admission is now measured by
  `scripts/long_prompt_readiness_matrix.py`, which wraps `inspect-prepared`
  with runtime preflight, GLM-5.2 shape checks, backend probes, and optional raw
  inspect JSONs without starting generation or reading model tensor payloads.
  The current baseline matrix
  `long-prompt-readiness-matrix-latest.json` (SHA
  `8b166e0d1afaaa9962e75aaba53032775d845ccf831805ed1ccc7722e47c81dc`) admits
  128, 512, 2048, and 4096 prompt tokens for one generated token. The auto
  prompt chunk policy now preserves a near-tile safety cap instead of always
  rounding down to the previous 64-token tile when that would throw away more
  than 25% of the safe limit; the 2048/4096-token baseline now resolves to
  113-token chunks instead of the old 64-token alignment. Planned routed-expert
  reads remain large, about 6611.57/13223.14 GiB for 2048/4096 prompts, but the
  request is no-OOM admitted with about 41.37 GiB required available memory and
  about 77-78 GiB visible at admission. The
  `--max-cache-read-mib 320` counterfactual
  `long-prompt-readiness-matrix-cache320-latest.json` (SHA
  `2bce43355b78a4b8d88545afa937889330ea6fd7ae99456841e60bffa7da2c58`) is also
  admitted, reaches 128-token chunks for 2048/4096 prompts, restores 75
  MPSGraph router-gate matrices, and lowers read amplification to 16/32. It is
  a launch-profile candidate, not a promoted default; 2048/4096-token locked
  replays and bakeoffs are still needed before promoting that long-prompt
  policy.
- The first locked 512-token GLM-5.2 MXFP4 replay for that cache320 candidate
  has now run successfully. The explicit router-gate-only profile
  `launch-profile-prefill-512tok-cache320-auto-mpsgraph13x32-routergate-tiled-memory-keycache.json`
  has SHA
  `d7594ad01175fe2e058d34e580a7a742df6e52a178f92cb20d440e290dc94d5d`; the
  bound audit
  `launch-audit-prefill-512tok-cache320-auto-mpsgraph13x32-routergate-tiled-memory-keycache.json`
  has SHA
  `ec06cad0be67aa6cb0064413e54271ed08e34d4480038d44c8725c195dd2a14a` and
  `ok=true`. The real generation result
  `prefill-512-cache320-routergate-memory-keycache-latest.json` has SHA
  `8a1168a81e09f4822defee0a747c8dd2b4c3ce67c571084e0f451da20400edef`,
  generated token id `[15]`, took `262.906s`, kept the profile live peak at
  `17.37 GiB`, and observed about `77.68 GiB` system memory available. The
  result is audit-bound, replay-ready, and file-ready. Its routed expert stage
  read only the unique selected ranges, `11.411 GiB` planned/actual versus the
  `5737.500 GiB` serial counterfactual, with zero waste, 599/599 read-advice
  successes, 3.757s copy time, and about 3.038 GiB/s copy throughput. The
  standard selected replay
  `selected-replay-prefill-512-cache320-routergate-memory-keycache.json` has SHA
  `8e70326a4140203f68a7101c3042e3cf4cc9b4b5c72cac42fa2e8d8decf09949`; both
  `selected-replay-check --check-ssd-read-speed` and
  `selected-replay-run --dry-run --check-ssd-read-speed` passed with no failed
  checks. A non-router-gate profile audit failed as intended, so required
  acceleration cannot silently count router-gate-only MPSGraph coverage unless
  `--allow-router-gate-only-prefill-acceleration` is present.
- The process-fusion path now has separate 512-token custom-metal fallback
  evidence for host states where the router-gate MPSGraph profile is not
  selectable. The process-fusion profile
  `launch-profile-prefill-512tok-custom-metal-processfusion-rope-server-memory-keycache.json`
  has SHA
  `9392aca50086d53f96e7451955003a7139fea301562a62ad59dfa90842b968a8`; its audit
  `launch-audit-prefill-512tok-custom-metal-processfusion-rope-server-memory-keycache.json`
  has SHA
  `859cc98794b9e30d4ef5280176fbcc03d20f46f437b4702062be85cad48e9424` and
  `ok=true`. The real result
  `prefill-512-custom-metal-processfusion-rope-server-memory-keycache-result.json`
  has SHA
  `f5c14f34eacb90d496bfcd16fe8054da604be931c7329523b27fe9371f26abab`,
  generated `[15]`, took `441.960s`, and kept the 17.37 GiB live cap with about
  77 GiB available memory. It is not comparable with the faster historical
  `cache320` selected replay because that plan used 75 `mpsgraph-f32` router-gate
  matrices and the current probe resolved the fallback to custom/fused Metal
  only. Against the same-plan custom-metal MoE-server control
  `prefill-512-custom-metal-moe-server-memory-keycache-result.json` (SHA
  `0b7005244615008eb2922f2934ef85262432c7c04baf66c6e68b18d687520c13`,
  `500.707s`), strict `result-compare` reports workload comparable and selects
  the process-fusion candidate at `0.883x`, `-58.747s`. The bakeoff
  `prefill-512-custom-metal-processfusion-rope-server-vs-moe-server-bakeoff-current.json`
  has SHA
  `a279a8cf623ea03718524a58d73dae880fe1665c2c6c5d3dbc7c17e3cff04abc`;
  the fallback selected replay
  `selected-replay-prefill-512-custom-metal-processfusion-rope-server-memory-keycache.json`
  has SHA
  `889f082ab0da73d22e53e3e0dba5f7973be527ab9bdd941f16ff6a6baf831e80`,
  and `selected-replay-check --check-ssd-read-speed` passed with no failed
  checks. Keep the historical MPSGraph router-gate 512 replay as the fastest
  selected evidence when its probe is valid; use this artifact as the locked
  custom-metal fallback.
- A non-sandbox MPSGraph recheck confirms the router-gate profile is still
  executable on the current host, but the old selected replay wrapper rejected
  before weight load when its locked stage cap drifted from the current profile
  calculation. A current-host profile
  `launch-profile-prefill-512tok-cache320-auto-mpsgraph13x32-routergate-tiled-memory-keycache-current.json`
  has SHA
  `5875805c980ceaaaeffce926a48345ec564add13020767052b58e73aa35859d9`; its audit
  `launch-audit-prefill-512tok-cache320-auto-mpsgraph13x32-routergate-tiled-memory-keycache-current.json`
  has SHA
  `0f000d3b548cbf0e8be76d6706cb36e326232f35d08e30657f2bc2a2659b3bb9`. The real
  replay
  `prefill-512-cache320-routergate-memory-keycache-current-mpsgraph-rerun.json`
  has SHA
  `113af4fa512734e9e3e5602764100da4e5b1748a8bc64e0e5b025830ba9dd42c`,
  generated `[15]`, and took `495.513s`. It is strict-plan comparable with the
  historical `262.906s` replay, but `result-compare` flags likely system-wide
  slowdown (`median sentinel ratio=1.858x`), so this is current-host
  reproducibility evidence rather than a new baseline. The combined MPSGraph +
  process-fusion profile
  `launch-profile-prefill-512tok-cache320-auto-mpsgraph13x32-routergate-processfusion-rope-server-memory-keycache-current.json`
  has SHA
  `a484b0cbab799335e4f694149e73438bc194f908533f2b12b477f4c0eda593ff`; its audit
  has SHA
  `d80af87de155f20dc857685e74f65ed11e8438ae33818d6ad7da3245f252c00d`. The
  replay
  `prefill-512-cache320-routergate-processfusion-rope-server-memory-keycache-current-result.json`
  has SHA
  `30d8654c6309d703aaba7f12e12b0adfe186b69445ac4ec6ba3a3f8e74f459a7`,
  generated `[15]`, took `472.264s`, and kept the same 17.37 GiB live cap. It is
  strict-plan comparable with the current non-fused MPSGraph replay and wins at
  `0.953x`, `-23.249s`; the scoped bakeoff
  `prefill-512-cache320-mpsgraph-processfusion-rope-server-vs-current-mpsgraph-bakeoff-current.json`
  has SHA
  `b46cbf7d28419d2ec4cc7ab5164c640070b36e9e3cc2b3f518bce70c21aba4a5`, and
  selected replay
  `selected-replay-prefill-512-cache320-mpsgraph-processfusion-rope-server-memory-keycache-current.json`
  has SHA
  `d54b467a36ffc19fad11ad9c85f8565390db2bb1947db110f8d907a990861201` with
  `selected-replay-check --check-ssd-read-speed` passing. It still loses to the
  custom-metal process-fusion fallback (`1.069x`) and to the historical 512
  MPSGraph selected replay, so keep it scoped to current-host MPSGraph
  process-fusion evidence.
- The 2048-token cache320 gate now has both locked launch evidence and a
  completed real replay. Profile
  `launch-profile-prefill-2048tok-cache320-auto-mpsgraph13x32-routergate-tiled-memory-keycache.json`
  has SHA
  `b30c9a00ba93f34c3ac4f0ab1cf7a47861d0c0af0fb6413e22dd9a0c0373e05a`;
  audit
  `launch-audit-prefill-2048tok-cache320-auto-mpsgraph13x32-routergate-tiled-memory-keycache.json`
  has SHA
  `a478f615b0c5b175a0ae2d1ca322391c48037ab640eaa6f1efb9209e47057a1e`
  and `ok=true`. The request resolves to 128-token chunks, requires about
  `41.37 GiB` available memory, saw about `77.52 GiB` available during audit,
  plans 16x read amplification, and preserves the 75 router-gate MPSGraph
  matrices. A real replay attempt was interrupted after `2312.617s` with no OOM
  after file progress reached `chunk_0000/layer_0054`; the traceback was inside
  `compute_dsa_topk_batch -> _matvec`. Diagnostic
  `prefill-2048-cache320-routergate-abort-diagnostic-latest.json` (SHA
  `e22133097384798a09d700316124c338762c2b05e9af2ea5335834e29084a51f`)
  records that pre-optimization evidence and the post-fix validation. In
  response, `largerlm/dsa_indexer.py`
  now uses optional NumPy fast paths for BF16/F16 resident tensor conversion,
  DSA matvec, and batched prefix scoring, while retaining the pure-Python path
  when NumPy is unavailable. Focused validation after the change passed
  `tests/test_dsa_indexer.py` (`30 passed`) and the wider DSA/prefill/token
  set (`354 passed`). A real 512-token GLM replay with the DSA fast path wrote
  `prefill-512-cache320-routergate-memory-keycache-dsa-fastpath-latest.json`
  (SHA `4ce1f9ea07fac280246f3f2ea1bc10bc048c89ee68a88a79e58a5e5744f96f9e`),
  generated `[15]`, and stayed replay-ready with matching selected-replay
  inputs. Its total time was `266.802s`, or within 2% of the prior
  `262.906s` baseline, so it is compatibility evidence rather than a promoted
  speedup. The follow-up DSA batch fast path adds vectorized cache writes and
  per-batch top-k scoring; validation after this extension passed
  `tests/test_dsa_indexer.py` (`32 passed`) and the wider DSA/prefill/token set
  (`356 passed`). The real 128-token replay
  `smoke-prefill-128tok-persistent-moe-dsa-batch-fastpath-latest.json` (SHA
  `35d5838c91f148b629e8e39a3422d0533fdef27f03ff59967650f16ed4a6f1b3`)
  generated `[15]` in `60.039s`. The real 2048-token replay
  `prefill-2048-cache320-routergate-memory-keycache-dsa-batch-fastpath-latest.json`
  (SHA `2fd14f86610040dbddcae6ceb1618b809151d812fedf4eeb081be7cbae3a2af4`)
  completed all 16 chunks, generated `[15]`, took `3744.429s`, preserved
  `replay_ready=true`, `replay_files_ready=true`, and `launch_audit_ok=true`,
  and reported a 17.37 GiB live cap with 77.66 GiB available at admission.
  Diagnostic
  `prefill-2048-cache320-routergate-dsa-batch-fastpath-diagnostic-latest.json`
  (SHA `a31573919e382d1d3d7d2d2ed9d901a4a430c006587f8ec36894223bbb7b28e0`)
  records the DSA target-size benchmark (SHA
  `9a5782fbf4a8db7fe862b16c7fbc16e6a41521c826c9d3acca0b1686a0e1f096`)
  and the successful 128/2048 replays. The 2048 run is now runnable evidence,
  but not a throughput finish line: DSA indexer time is still `1585.739s` of
  `3744.429s`. The next DSA change skips prompt top-k generation when the
  visible context is fully covered by `index_topk`, while still writing the DSA
  cache needed by future decode. Target-size benchmarking in
  `dsa-visible-context-skip-benchmark-latest.json` (SHA
  `4e33c2b77653a52875fbd96bdc974b7748179efa123775b817ffbc0cbaaf1ca6`)
  measured cache-write-only DSA at `0.232s` versus `1.879s` with top-k for a
  128-token GLM-sized chunk, an `8.11x` local branch speedup. A real direct
  128-token `prefill-prompt` validation
  `prefill-128tok-visible-context-dsa-cachewrite-skip-topk-latest.json` (SHA
  `af16068ce89ac3e995b03efcb72617bcaef36f9dff28db6fe5b9baeb7411164b`)
  completed in `65.703s`, produced `dsa_topk_path_count=0`, and spent `3.398s`
  in DSA cache writes across 78 attention layers under the same 17.37 GiB live
  working-set cap and 24 GiB free-memory guard. Diagnostic
  `prefill-2048-cache320-routergate-dsa-visible-context-skip-diagnostic-latest.json`
  (SHA `4b80808fbeaf824120c2b4e81ccc1acb6ec9b8d9c91c142d9a4bf6c2174bcee0`)
  now records the code change, tests, microbench, direct-prefill proof, and the
  full promotion replay. The locked 2048-token replay after the skip,
  `prefill-2048-cache320-routergate-memory-keycache-dsa-visible-context-skip-latest.json`
  (SHA `d10afd6d76d77e58507bc6363689b123697a1b190208c0d08a61162cada27f09`),
  completed all 16 chunks, generated `[15]`, kept the same locked profile/audit,
  and reduced total time from `3744.429s` to `2748.777s` (`1.36x`). Its compact
  summary
  `prefill-2048-cache320-routergate-memory-keycache-dsa-visible-context-skip-summary-latest.json`
  (SHA `fd02d141e10533a0101ab7368d665ce1850fd6ab44e1c70c3f7727ee2b8e43ad`)
  reports prompt DSA time at `193.168s`, down from `1585.739s`, with
  `prompt_dsa_topk_path_count=0`. The new largest prompt-side targets are routed
  MoE (`450.813s`), MLA attention (`401.685s`), resident linear backends
  (`custom-metal 470.176s`, `fused-metal 342.187s`, `mpsgraph-f32 161.328s`),
  and attention output/projection/RoPE work. `result-compare` now treats this
  visible-context replay as promotable despite local tensor timing regressions:
  `compare-2048-dsa-batch-fastpath-vs-visible-context-skip-latest.json` (SHA
  `a684b1ef7c1e6d7731ec3694a7394d9de290d3647ad1a9f12a08c2567d5acdb9`)
  reports `candidate_promotable=true`, `total_ratio=0.7341`, and reason
  `candidate_large_total_win_overrides_tensor_regressions`. The follow-up
  bakeoff `bakeoff-prefill-2048-cache320-routergate-visible-context-skip-latest.json`
  (SHA `d44c5652eef81a940e2a0cab850d24cd803b5356556b66fdb67f2548fb19e643`)
  set `baseline_retained=false` and wrote the selected replay
  `selected-replay-prefill-2048-cache320-routergate-visible-context-skip.json`
  (SHA `1ed0289ed7e5d7eb0e376f8b5cf9eeefd0c431a7392b16098b44fb1238368cfe`)
  plus shell wrapper (SHA
  `31ff40fb4304c782a5013cdb17148fe1e5d9ce12b5aec5f51d2ac638ae725932`).
  `selected-replay-check --check-ssd-read-speed --json` returned `ok=true`
  with 41/41 checks passing, including replay-file readiness, launch audit
  binding, current memory/disk guards, MPSGraph acceleration coverage, and
  current bounded SSD read speed. Two bounded post-DSA probes did not find
  another runtime promotion: routed-MoE tile/vector
  sweep `glm-moe-layer36-optimization-target-128tok.json` (SHA
  `8856d87f8c2caa87bf7236b23f84ea2579c4d9c01a8a070839f81f0acd8f1363`)
  kept baseline `tile1_auto_silu` fastest, and MLA cache-mode sweep
  `glm-mla-layer3-cache-sweep-128tok.json` (SHA
  `52b98441399af3b07aba755d9e6123f6d0646d46e10d3e2ce765fbe61f1a9926`)
  kept `key-value` fastest under its 256 MiB scratch cap. The combined
  diagnostic `post-dsa-optimization-target-benchmarks-latest.json` (SHA
  `b70644166493d755b231b926acb857b1d8ab9327f8d7473dae004effc34ae7e7`)
  records those no-candidate probes and the promotable 2048 comparison. The MLA
  sweep script now records when a requested value cache is disabled by scratch
  cap so bounded cache-mode artifacts are not mistaken for full-profile
  value-cache evidence. Router-hybrid follow-up is now a measured negative for
  this long-prompt line. The policy analysis script has a low-memory
  `--stream-result` mode for large replay JSONs; it read the 625 MiB 2048 replay
  without materializing the full payload and wrote
  `glm-router-hybrid-policy-analysis-2048-visible-context-latest.json` (SHA
  `c553264273153c25776e562f3f9e30a48bf813ae8a1e97feddc2bfd0016bef8e`).
  That analysis found `min_effective_score_margin=3.24e-08`, so global custom
  router promotion remains blocked by the driftx4 margin gate, while an online
  driftx4 hybrid would route 1147/1200 layer-chunks through custom and fall back
  on 53. A locked 512-token validation was then audited with
  `launch-profile-prefill-512tok-cache320-routerhybrid-driftx4-tiled-memory-keycache.json`
  (SHA `59c515a054183d2b630e5c494af67750c024f07505b21046ec3a88f4983576ab`) and
  `launch-audit-prefill-512tok-cache320-routerhybrid-driftx4-tiled-memory-keycache.json`
  (SHA `184f32568631f34b61db340f6f3e4043b555193bbf0b1492f49e77ad2efd7b9a`,
  `ok=true`). The real candidate
  `prefill-512-cache320-routerhybrid-driftx4-memory-keycache-latest.json` (SHA
  `f8f7a7ee5a0dba8fc6b5a5de9dd97e168c043879f15775fdfe34c06942f2ace6`) kept
  generated token `[15]` and memory guards, but slowed total time from
  `262.906s` to `451.455s` (`1.717x` slower). It selected custom router output
  on 71/75 routed layers and MPSGraph fallback on 4, yet routed-MoE wall time
  grew from `71.753s` to `126.029s`. The comparison
  `compare-512-routergate-vs-routerhybrid-driftx4-latest.json` (SHA
  `3eeeeabd548725d1bead36c5012c0b57b9b44e1fb729a5d28d57375d48af0d94`) is
  comparable under `--allow-prefill-policy-change`, reports matching `[15]`,
  and chooses `prefer_baseline` with reason `candidate_total_elapsed_slower`.
  Do not spend a 2048-token long run on this router-hybrid branch unless a new
  runner-level implementation changes the MoE side effects.
- `result-summary` now promotes high runner-command cardinality into an explicit
  `runner_process_fusion` optimization target. The 2048 visible-context replay
  reports `runner command records: unique=8784 records=15024
  duplicate_records=6240`; the new top target is
  `runner_process_fusion[runner_process_orchestration]=8784cmds`. The largest
  unique command groups are `--run-resident-linear-batch=1344`,
  `--run-attn-output-batch=1248`, `--run-mla-attention-batch=1248`,
  `--run-attn-projections=1248`, `--run-rope-split-batch=1248`, and
  `--run-rmsnorm-batch=1248`. This makes the next implementation boundary
  concrete: prototype persistent runner/plan-server fusion around those prompt
  command groups under 128/512 locked replays before attempting another 2048
  long run.
- Resident-linear process fusion has its first opt-in executable prototype.
  `metal/largerlm-runner --run-resident-linear-batch-plan-server-jsonl` accepts
  one bounded resident batch-linear request per JSONL line, reuses the same
  Metal process, and exits on `{"command":"quit"}`. The Python
  `ResidentBatchLinearServerSession` can drive `run_resident_batch_linear`
  while keeping the existing resident layout, input/output byte, backend, and
  scratch-limit checks in place. It is now threaded through full prompt prefill
  as an explicit opt-in: raw `prefill-prompt` uses
  `--persistent-resident-linear-server`, while prepared generation, inspection,
  serving, and launch profiles use
  `--prefill-persistent-resident-linear-server`. Validation so far is still
  deliberately bounded: `python -m pytest tests/test_prefill_execute.py -q`
  passed 103 tests, `python -m pytest tests/test_prompt_prefill.py -q` passed
  87 tests including the new prompt-prefill session case,
  `python -m pytest tests/test_token_generator.py tests/test_server.py -q`
  passed 278 tests with 3 skips, `make -C metal largerlm-runner` is clean, and
  `python metal/resident_linear_batch_smoke.py` confirms two resident-linear
  JSONL server requests complete in one runner process on Apple M5 Max. The
  next gate is a 128/512 locked replay that proves command-count reduction
  without token drift or memory growth.
- Resident-linear process fusion has now passed the first real locked GLM-5.2
  replay gate, but only as bring-up evidence. The locked chunk128 profile
  `launch-profile-prefill-128tok-persistent-moe-resident-linear-server-tiled-memory-accumulator.json`
  (SHA `8f374c9bedfc43eb7baacbcd11f572c758f15d565ae8149288c8b4960d86680b`)
  and audit
  `launch-audit-prefill-128tok-persistent-moe-resident-linear-server-tiled-memory-accumulator.json`
  passed and generated `[15]` in
  `smoke-prefill-128tok-persistent-moe-resident-linear-server-tiled-memory-accumulator-strict-audit-result.json`.
  `result-summary` reports `persistent_linear_server=yes`, one 128-token chunk,
  a replay-ready launch binding, 17.37 GiB live cap, 196.051s total, and fewer
  runner command signatures (`unique=466`, `records=864`). The selected replay
  `selected-replay-prefill-128-persistent-moe-resident-linear-memory-accumulator.json`
  passes `selected-replay-check --check-ssd-read-speed` with zero failed checks.
  Comparing this result to the old 60.122s artifact alone looked like a
  `3.261x` slowdown, but a same-window rerun of the old memory-accumulator
  baseline,
  `smoke-prefill-128tok-persistent-moe-server-tiled-memory-accumulator-rerun-after-resident-candidate.json`,
  also took 196.818s. The same-window bakeoff
  `prefill-128-resident-linear-server-vs-rerun-bakeoff-current.json` therefore
  keeps the old baseline and marks the resident-linear candidate as a
  `total_elapsed_within_two_percent` tie (`0.996x`, generated `[15]`,
  replay-ready). Do not spend a 512/2048 replay on this resident-linear server
  path until it shows a real win under repeated same-window bakeoffs or more
  non-runner orchestration is fused.
- Attention-projection process fusion is wired and has passed a real locked
  128-token GLM-5.2 replay, but it is not promoted. The low-level runner exposes
  `--run-attn-projections-server-jsonl`, accepting one prompt-prefill fused
  projection request per JSONL line and deliberately rejecting cache-append
  fields so decode cache writes stay on the existing one-shot
  `--run-attn-projections` path. Python drives it through
  `AttentionProjectionsServerSession`; raw prompt prefill can opt in with
  `--persistent-attention-projection-server`, while prepared generation,
  inspection, serving, and launch profiles use
  `--prefill-persistent-attention-projection-server`. Validation now includes
  `python -m pytest tests/test_prefill_execute.py tests/test_prompt_prefill.py
  tests/test_result_summary.py -q` passing 267 tests, `python -m pytest
  tests/test_token_generator.py tests/test_server.py -q` passing 280 tests with
  3 skips, `make -C metal largerlm-runner`, and
  `python metal/attn_projections_smoke.py` on Apple M5 Max. The locked profile
  `launch-profile-prefill-128tok-persistent-moe-resident-linear-attnproj-server-tiled-memory-accumulator.json`
  (SHA `3d4dd88308d0171249a660ae9e06f446daa3308b36d7baf713f42fddd46680a8`)
  and matching audit generated `[15]` in
  `smoke-prefill-128tok-persistent-moe-resident-linear-attnproj-server-tiled-memory-accumulator-strict-audit-result.json`,
  stayed inside the 17.37 GiB live cap, and reports
  `persistent_attention_projection_server=true`. Result-summary confirms the
  projection process boundary collapsed to one
  `--run-attn-projections-server-jsonl` unique group with 468 records, and
  `projections_elapsed_seconds` improved sharply versus the old memory rerun
  (18.228s to 8.445s). Total elapsed still did not win: 200.255s versus the
  old memory rerun at 196.818s (`1.017x`, tie) and versus resident-linear-only
  at 196.051s (`1.021x`, prefer baseline). The bakeoff
  `prefill-128-attnproj-server-vs-rerun-bakeoff-current.json` keeps the old
  memory baseline and marks both resident-linear and attention-projection
  process-fusion candidates as non-promotable.
- RoPE split process fusion is now implemented as an opt-in tiny server.
  `metal/largerlm-runner --run-rope-split-batch-server-jsonl` accepts one
  bounded fused q_b split plus RoPE request per JSONL line and exits on
  `{"command":"quit"}`. Python drives it with
  `RopeSplitBatchServerSession`; raw prompt prefill uses
  `--persistent-rope-split-server`, while prepared generation, inspection,
  serving, and launch profiles use
  `--prefill-persistent-rope-split-server`. Validation now includes
  `python -m pytest tests/test_prefill_execute.py tests/test_prompt_prefill.py
  tests/test_result_summary.py -q` passing 271 tests, `python -m pytest
  tests/test_token_generator.py tests/test_server.py -q` passing 282 tests with
  3 skips, `make -C metal largerlm-runner`, and
  `python metal/rope_split_batch_smoke.py` on Apple M5 Max confirming one-shot
  `--run-rope-split-batch` and the JSONL server produce matching
  q_nope/q_rope/rotated outputs. The real locked GLM candidate uses
  `launch-profile-prefill-128tok-persistent-moe-resident-linear-attnproj-rope-split-server-tiled-memory-accumulator.json`
  (SHA `81d9e6888344b5013e24e32ea14d8e0370e91e83960d088acf3a94eb71d85ba9`) and
  matching audit, generated `[15]`, stayed inside the 17.37 GiB live cap, and
  reports `persistent_rope_split_server=true`. Its same-window baseline rerun
  without the RoPE server took 148.846s; the RoPE server candidate took
  114.796s, collapsed RoPE split runner groups from 78 to one
  `--run-rope-split-batch-server-jsonl` group, and reduced
  `rope_elapsed_seconds` from 8.506s to 0.949s. The bakeoff
  `prefill-128-attnproj-rope-split-server-vs-rerun-bakeoff-current.json`
  selected the candidate (`0.771x`, `-34.050s`) and wrote
  `selected-replay-prefill-128-persistent-moe-resident-linear-attnproj-rope-split-memory-accumulator.json`;
  `selected-replay-check --check-ssd-read-speed` passes with zero failed
  checks. Treat this as the new 128-token process-fusion baseline, while
  512/2048-token promotion still needs separate locked replay evidence.
- Resident batch RMSNorm process fusion is now implemented as the next small
  opt-in server. `metal/largerlm-runner --run-rmsnorm-batch-server-jsonl`
  accepts one bounded RMSNorm request per JSONL line and exits on
  `{"command":"quit"}`. Python drives it with
  `ResidentBatchRMSNormServerSession`; raw prompt prefill uses
  `--persistent-rmsnorm-server`, while prepared generation, inspection,
  serving, and launch profiles use
  `--prefill-persistent-rmsnorm-server`. Validation now includes
  `python -m py_compile` over the touched modules/tests,
  `python -m pytest tests/test_prefill_execute.py tests/test_prompt_prefill.py
  tests/test_token_generator.py tests/test_server.py tests/test_result_summary.py
  -q` passing 559 tests with 3 skips, `make -C metal largerlm-runner`, and
  `python metal/rmsnorm_batch_smoke.py` on Apple M5 Max confirming one-shot
  `--run-rmsnorm-batch`, the JSONL server, and the CLI wrapper produce matching
  outputs. The first locked GLM gate is useful negative evidence rather than a
  promotion: a 128-token candidate was rejected by request admission because
  `prefill_prompt_chunk_tokens 128` exceeds the safety-capped maximum of 126,
  and the 126-token same-window bakeoff retained the RoPE split baseline. Both
  126-token runs generated `[15]` inside the 17.37 GiB live cap, but the RMSNorm
  server candidate took 220.614s versus 211.945s for the baseline
  (`1.041x`, `+8.670s`, `candidate_total_elapsed_slower`). The selected replay
  remains
  `selected-replay-prefill-126-persistent-moe-resident-linear-attnproj-rope-split-memory-accumulator.json`,
  and `selected-replay-check --check-ssd-read-speed` passes with SSD read speed
  at 5.506 GiB/s (`0.931x` of calibration). The next process-fusion target
  should merge a larger boundary than standalone RMSNorm, such as attention
  output/MLA or a layer-level server.
- MLA attention process fusion is now implemented and promoted for the
  126-token custom-metal fallback envelope. `metal/largerlm-runner
  --run-mla-attention-batch-server-jsonl` accepts bounded JSONL requests for
  contiguous or indexed batch MLA attention and exits on `{"command":"quit"}`.
  Python drives it with `MLAAttentionBatchServerSession`; raw prompt prefill
  uses `--persistent-mla-attention-server`, while prepared generation,
  inspection, serving, and launch profiles use
  `--prefill-persistent-mla-attention-server`. Validation included
  `python -m py_compile` over the touched Python modules/tests, `python -m
  pytest tests/test_prefill_execute.py tests/test_prompt_prefill.py
  tests/test_token_generator.py tests/test_server.py tests/test_result_summary.py
  tests/test_prepare.py -q` passing 907 tests with 3 skips, `make -C metal
  largerlm-runner`, and `python metal/mla_attention_batch_smoke.py` on Apple M5
  Max confirming one-shot versus JSONL server equivalence. The locked GLM
  candidate used profile SHA
  `f7f1cbd90c2ce28c191607165431b925ca1dbf043511f8a9f2165b6578d7a172`, audit SHA
  `d7bcb0a7e555ed2d7ac1614b68d8e4275f8219175e7076453107f6c6dcaef302`, generated
  `[15]`, stayed inside the 17.37 GiB live cap with about 77.40 GiB available,
  and finished in 132.394s with 132.173s prompt prefill. The same-window RoPE
  split baseline was 211.945s, so
  `prefill-126-mla-attention-server-vs-rope-split-baseline-bakeoff-current.json`
  selected the MLA server candidate (`0.625x`, `-79.551s`,
  `candidate_large_total_win_overrides_tensor_regressions`) and wrote
  `selected-replay-prefill-126-persistent-moe-resident-linear-attnproj-rope-split-mla-server-memory-accumulator.json`
  (SHA `0ed3e32aab894d170bbf232cf2badaf0b2c9ea66d0272b1839f9b0745bf6c7ee`) plus
  the matching script (SHA
  `122ad0ab336da6b416fb24fddf294589919ae275dbae294e56d4c46b1fb47403`).
  `selected-replay-check --check-ssd-read-speed` passes with current bounded SSD
  speed 18.831 GiB/s (`3.184x` of calibration). Because this envelope uses the
  explicit custom-metal fallback, the launch audit is generated with
  `--allow-non-accelerated-prefill-launch-audit`; this does not relax the
  memory, temp-disk, SSD, profile, or GLM-shape gates.
- Attention-output process fusion is now implemented but remains opt-in after
  negative locked GLM evidence. `metal/largerlm-runner
  --run-attn-output-batch-server-jsonl` accepts one bounded batch `o_proj +
  residual` JSONL request per line and exits on `{"command":"quit"}`. Python
  drives it with `AttentionOutputBatchServerSession`; raw prompt prefill uses
  `--persistent-attention-output-server`, while prepared generation,
  inspection, serving, and launch profiles use
  `--prefill-persistent-attention-output-server`. Validation included `python
  -m py_compile` over the touched modules/tests, `python -m pytest
  tests/test_prefill_execute.py tests/test_prompt_prefill.py
  tests/test_token_generator.py tests/test_server.py tests/test_result_summary.py
  tests/test_prepare.py -q` passing 913 tests with 3 skips, `make -C metal
  largerlm-runner`, and `python metal/attention_output_smoke.py` on Apple M5
  Max confirming one-shot versus JSONL server equivalence. The locked
  126-token GLM candidate used profile SHA
  `7164678692bf11fb7be6be9b90d2202670a89a56930e4fc6e04c51cfb4873679`, audit SHA
  `9658a02a5df93143377c554a2515eb0b47ff270b3df6b3fabc5764446fbc72d2`, generated
  `[15]`, stayed inside the 17.37 GiB live cap with about 76.98 GiB available,
  and collapsed attention-output runner groups to one
  `--run-attn-output-batch-server-jsonl` group. It improved
  `self_attn.o_proj.weight` time from 12.882s to 9.561s, but total latency
  regressed to 194.082s versus the selected MLA baseline at 132.394s
  (`1.466x`, `+61.688s`). The bakeoff
  `prefill-126-attnout-server-vs-mla-attention-server-bakeoff-current.json`
  (SHA `95365d155593eb191dbee8d6287e0dde64de1ed1d0792f77d1298b3a3be84d71`)
  retained the MLA selected replay (`candidate_total_elapsed_slower`).
  `selected-replay-check --check-ssd-read-speed` still passes for the current
  selected replay, with current bounded SSD speed 5.808 GiB/s (`0.982x` of the
  prepared cold-read calibration).
- The current 126-token selected-replay optimization-target sweeps are now
  closed with no promotable microbench candidate: routed MoE layer 4
  `glm-moe-layer4-optimization-target-126tok.json` (SHA
  `d4b38681f0ac2b8a6fabc5822e1de017040701ebdd56bfa508766071981a01a9`), MLA
  cache layer 2 (SHA
  `5afb1de4b2e7262b8a8619f82a5bd3f3152d9c0c24b9e9d2b103878dbbc5f80e`),
  attention output layer 1 (SHA
  `2308eed541e63f21595d70c698b0f980297c041ba20a2912919f08022cce3ca9`),
  attention projection layer 2 (SHA
  `be8835fdd68d82f6ced208f2ed7834087b6c5f20ef1d56c379960d3a8dc4ae2a`), cache
  write layer 60 (SHA
  `14ed0a396ed35a11dd44a5d6730bd6ca8ace6cd6cc08541d78c344cb103a77fb`), and
  RoPE split layer 2 (SHA
  `4559d5800dd120640fdd54314e04bb3d73ab6752af20236a2dab8dff3172e0b5`) all
  report `candidate=false`. The selected 126-token profile therefore keeps its
  current MoE tile, MLA key+value cache, fused projection, fused RoPE, default
  cache-write, and group32 attention-output choices until a full replay proves
  otherwise.
- Shared-expert process fusion is implemented and verified but not promoted.
  `metal/largerlm-runner --run-shared-expert-batch-server-jsonl` now accepts
  bounded JSONL shared-expert batch requests, and Python exposes it through
  `ResidentSharedExpertBatchServerSession`; raw prompt prefill uses
  `--persistent-shared-expert-server`, while prepared generation, inspection,
  serving, and launch profiles use
  `--prefill-persistent-shared-expert-server`. Validation included `python -m
  py_compile` over the touched modules/tests, `python -m pytest
  tests/test_prefill_execute.py tests/test_prompt_prefill.py
  tests/test_token_generator.py tests/test_server.py tests/test_prepare.py -q`
  passing 839 tests with 3 skips, `make -C metal largerlm-runner`, and `python
  metal/shared_expert_batch_smoke.py` on Apple M5 Max confirming one-shot versus
  JSONL server equivalence. The locked GLM candidate used profile SHA
  `14e98a7445957aa0441f3afac1dfb7401d28fb3d1f0911509f2336ea3d981767`, audit SHA
  `48cf0d26b037133bb8795e40b71a3c727d1fb95ba8bc14d6ef599296e7c45bed`, generated
  `[15]`, stayed inside the 17.37 GiB live cap, and finished in 133.308s with
  133.064s prompt prefill. It cut `mlp.shared_experts` time from 8.503s to
  2.839s and dropped unique runner command groups from 235 to 161, but total
  latency remained a tie/slight regression versus the selected MLA baseline at
  132.394s. The bakeoff
  `prefill-126-shared-expert-server-vs-mla-attention-server-bakeoff-current.json`
  (SHA `e395ac891ae4c2837ac05857bb4bde31b5242af9ceaf1d3838a3585c6a975f04`)
  retained the baseline with `total_elapsed_within_two_percent`.
- A same-window rerun after the shared-server bring-up confirmed the baseline
  retention. The current MLA-server baseline rerun
  `smoke-prefill-126tok-mla-server-baseline-rerun-after-shared-server-current.json`
  (SHA `4a8986bb90f81a76aceddc245db838de4daf4051d318aba57aad0b55c08e8ead`)
  generated `[15]` in 141.647s. The shared-server rerun
  `smoke-prefill-126tok-shared-server-rerun-after-baseline-current.json` (SHA
  `c35d2486e8dda805e73d949f212bed600633b8f1219cbdf220c14739acfca013`)
  also generated `[15]` but took 194.353s. `result-compare` kept the workloads
  comparable and did not flag system slowdown; the rerun bakeoff
  `prefill-126-shared-expert-server-rerun-vs-baseline-bakeoff-current.json`
  (SHA `6c5f7fdd88f4b705c109161ef9b05788bf081361639a763e29aa0e95b0968fbb`)
  retained the baseline (`1.372x`, `+52.705s`,
  `candidate_total_elapsed_slower`). The shared-server path is therefore
  retained only as opt-in evidence and infrastructure.
- The hottest routed-MoE layer from that baseline rerun, layer 67, was checked
  with the bounded target sweep suggested by `result-summary`. The artifact
  `glm-moe-layer67-optimization-target-126tok.json` (SHA
  `6f7fc4dfdbbd484368a255e7234edfbfa5e0f9ef6e7b49b7d4f44d9eafaf42bb`) reports
  `candidate_for_full_replay=false`: `tile1_auto_silu` remains both the
  baseline and fastest kernel, while tile2, scalar, and vector-SwiGLU variants
  all miss the kernel-speedup/drift policy. This closes another local MoE
  toggle path and points the next MoE work toward structural orchestration or
  kernel changes.
- The latest local GLM-5.2 MXFP4 artifact no longer needs another Hugging Face
  weight download: `checkpoint-status --verify-local-headers
  --require-complete --require-clean` checked all 76 local shard headers, found
  no extra or corrupt files, and kept `next_bringup_step=null`. The current M5
  backend report
  `prefill-backend-report-latest.json` (SHA
  `d3be34e55ed80db05418a21ca606ea48a3f22c58c05a7601f8694ac7f566f378`) validates
  `mpsgraph-f32` but still reports `missing_public_mpp_symbols` for the
  MPP/Neural-Accelerator path, so MPSGraph remains the only selectable
  accelerated prefill backend in the public-SDK build.

## Near-Term Design Rules

- Keep the default path conservative: no unbounded caches, no implicit overflow,
  and no route representation that scales worse than the configured prompt
  chunk.
- Prefer binary route/assignment formats once routing leaves Python.
- Treat M5 Max SSD bandwidth as useful only when reads are coalesced and memory
  pressure is controlled.
- Tune prompt chunk caps against both activation memory and routed expert read
  amplification; a safe activation cap can still double SSD traffic.
- Use the `prefill_backend_candidates` priority list for large prefill GEMMs
  first. ANE/Neural Accelerator work stays behind the MPP backend selector, and
  each candidate should be measured against MPSGraph/custom Metal before it
  becomes default.
- Treat MPP candidate coverage as launch-critical evidence. Request-level
  launch audits now bind the candidate policy, candidate matrix count, candidate
  FLOPs, and candidate FLOP fraction; this keeps M5 Neural Accelerator bring-up
  measurable even while MPSGraph remains the selectable fallback.
