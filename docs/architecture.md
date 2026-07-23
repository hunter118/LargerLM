# Architecture

## Target Machine

Initial profile:

- Apple M5 Max
- 128 GiB unified memory
- 40-core GPU
- High-bandwidth internal SSD

The runtime should not try to fill unified memory with expert weights. Memory is
more valuable as OS page cache, KV/DSA cache, and temporary Metal buffers.

## Current GLM-5.2 M5 Max Baseline

The prepared GLM-5.2 MXFP4 package at
`artifacts/glm-5.2-mxfp4/largerlm-prepared` passed storage validation and real
generation. The final held-out Metal A/B measured `0.858 tok/s` without an
application expert cache and `1.083 tok/s` with the safe learned 10 GiB set.
See `docs/real-glm-validation.md`.
For opt-in MLA key-cache experiments, use
`launch-profile-17tok-chunk16-accel-keycache.json`; it is a request-checked,
safe-to-replay launch profile that carries the same memory, SSD, shape, chunk,
and prefill acceleration guards plus `--prefill-mla-key-cache` and
`--metal-final-logits`.

Longer prompts are currently gated by admission evidence rather than promoted
serving defaults. `scripts/long_prompt_readiness_matrix.py` runs
`inspect-prepared` across several prompt lengths without loading tensor payloads
or starting generation. The current matrix admits 128, 512, 2048, and 4096
prompt tokens for one generated token with about 41.37 GiB required available
memory and about 77-78 GiB actually visible at admission. The conservative
baseline resolves long prompts to 113-token chunks for 2048/4096 tokens after
preserving a near-tile safety cap; a `--max-cache-read-mib 320` counterfactual
reaches 128-token chunks, restores the 75-matrix MPSGraph router-gate path, and
lowers routed-expert read amplification to 16/32 for 2048/4096 prompts.

That cache-read profile now has a locked 512-token generation replay, but it is
still not a promoted serving default. The profile
`launch-profile-prefill-512tok-cache320-auto-mpsgraph13x32-routergate-tiled-memory-keycache.json`
(SHA `d7594ad01175fe2e058d34e580a7a742df6e52a178f92cb20d440e290dc94d5d`)
and audit
`launch-audit-prefill-512tok-cache320-auto-mpsgraph13x32-routergate-tiled-memory-keycache.json`
(SHA `ec06cad0be67aa6cb0064413e54271ed08e34d4480038d44c8725c195dd2a14a`)
are bound to
`prefill-512-cache320-routergate-memory-keycache-latest.json` (SHA
`8a1168a81e09f4822defee0a747c8dd2b4c3ce67c571084e0f451da20400edef`). The
run generated token id 15, completed in 262.906s, held the 17.37 GiB live peak,
and saw about 77.68 GiB memory available. Its selected replay artifact,
`selected-replay-prefill-512-cache320-routergate-memory-keycache.json` (SHA
`8e70326a4140203f68a7101c3042e3cf4cc9b4b5c72cac42fa2e8d8decf09949`), passes
`selected-replay-check --check-ssd-read-speed` and
`selected-replay-run --dry-run --check-ssd-read-speed`. This profile requires
`--allow-router-gate-only-prefill-acceleration`: MPSGraph covers the 75 router
gate GEMMs only, while the main non-router prefill frontier remains custom
Metal now and future MPP/Neural-Accelerator work.

When that router-gate MPSGraph path is not selectable by the current host probe,
there is now a locked 512-token custom-metal fallback replay. The process-fusion
profile
`launch-profile-prefill-512tok-custom-metal-processfusion-rope-server-memory-keycache.json`
(SHA `9392aca50086d53f96e7451955003a7139fea301562a62ad59dfa90842b968a8`) and
audit
`launch-audit-prefill-512tok-custom-metal-processfusion-rope-server-memory-keycache.json`
(SHA `859cc98794b9e30d4ef5280176fbcc03d20f46f437b4702062be85cad48e9424`) bind
the same 512-token prompt to resident-linear, attention-projection, and RoPE
split JSONL servers. The real replay
`prefill-512-custom-metal-processfusion-rope-server-memory-keycache-result.json`
generated `[15]`, completed in 441.960s, and stayed inside the 17.37 GiB live cap.
It should not replace the faster historical MPSGraph selected replay, but it does
beat the same-plan custom-metal MoE-server control at 500.707s. The scoped
fallback selected replay
`selected-replay-prefill-512-custom-metal-processfusion-rope-server-memory-keycache.json`
passes `selected-replay-check --check-ssd-read-speed`.
A current-host non-sandbox MPSGraph probe also validates the router-gate path:
`prefill-512-cache320-routergate-memory-keycache-current-mpsgraph-rerun.json`
generated `[15]` and completed in 495.513s, but strict comparison with the
historical 262.906s result flags likely system-wide slowdown. Combining that
MPSGraph router-gate path with the process-fusion servers produced
`prefill-512-cache320-routergate-processfusion-rope-server-memory-keycache-current-result.json`
at 472.264s. That scoped candidate beats the same-window non-fused MPSGraph
replay, but remains slower than the custom-metal process-fusion fallback, so it
is retained as current-host MPSGraph process-fusion evidence rather than a
default.

The 2048-token cache320 candidate is now launch-audit locked and has a
completed real replay. Its request-bound profile
`launch-profile-prefill-2048tok-cache320-auto-mpsgraph13x32-routergate-tiled-memory-keycache.json`
(SHA `b30c9a00ba93f34c3ac4f0ab1cf7a47861d0c0af0fb6413e22dd9a0c0373e05a`)
and audit
`launch-audit-prefill-2048tok-cache320-auto-mpsgraph13x32-routergate-tiled-memory-keycache.json`
(SHA `a478f615b0c5b175a0ae2d1ca322391c48037ab640eaa6f1efb9209e47057a1e`)
pass the same memory, SSD, GLM-5.2 shape, stage-temp, and explicit
router-gate-only acceleration gates. The attempted real replay was stopped
after 2312.617s while still in `chunk_0000/layer_0054`, proving that the next
long-prompt blocker was DSA indexer orchestration rather than memory pressure.
`prefill-2048-cache320-routergate-abort-diagnostic-latest.json` (SHA
`e22133097384798a09d700316124c338762c2b05e9af2ea5335834e29084a51f`) records the
interruption, traceback, and post-fix 512-token compatibility replay. The
immediate fix is an optional NumPy fast path in
`largerlm/dsa_indexer.py` for BF16/F16 tensor conversion, DSA matvec, and
batched prefix scoring, with pure-Python fallback for minimal environments. The
512-token replay after this change generated `[15]` from the same selected
inputs and remained within 2% of the previous baseline, so it validates
compatibility but is not yet a long-prompt performance promotion.

The later DSA batch fast path extends that fix to cache writes and whole-batch
top-k scoring. The 2048-token replay
`prefill-2048-cache320-routergate-memory-keycache-dsa-batch-fastpath-latest.json`
(SHA `2fd14f86610040dbddcae6ceb1618b809151d812fedf4eeb081be7cbae3a2af4`)
completed all 16 chunks, generated `[15]`, and stayed under the 17.37 GiB live
working-set cap with 77.66 GiB available at admission. The compact diagnostic
`prefill-2048-cache320-routergate-dsa-batch-fastpath-diagnostic-latest.json`
(SHA `a31573919e382d1d3d7d2d2ed9d901a4a430c006587f8ec36894223bbb7b28e0`)
records the DSA benchmark and replay summaries. This makes 2048-token
`cache320` a runnable long-prompt profile, while the next architecture target is
still DSA throughput: it consumed 1585.739s of the 3744.429s 2048-token run.
The following visible-context DSA path avoids generating prompt top-k indices
when `context_length <= index_topk`; MLA already covers the full visible context
in that case, but the DSA cache is still written for later decode. A target-size
microbench shows the branch dropping from 1.879s with top-k to 0.232s without
top-k for a 128-token chunk, and a real 128-token direct prefill produced no
`dsa_topk` files while keeping the 17.37 GiB live-memory cap. The full
2048-token `cache320` promotion replay after this change completed with the
same generated token `[15]`, reduced total time from 3744.429s to 2748.777s,
and cut prompt DSA time from 1585.739s to 193.168s with zero prompt `dsa_topk`
files. The selected replay wrapper for this path now passes
`selected-replay-check --check-ssd-read-speed` with 41/41 checks, so it is the
current minimum runnable 2048-token gate. DSA top-k orchestration is no longer
the first architecture bottleneck. The next work should focus on routed MoE,
MLA attention, and resident projection/output kernels under the same locked
replay gates. Immediate
bounded probes did not find a shallow promotion: the routed-MoE tile/vector
sweep kept `tile1_auto_silu` fastest, and the MLA cache-mode sweep kept
`key-value` fastest under the bounded scratch cap, so the next meaningful work
is deeper orchestration/kernel profiling rather than flipping existing knobs.
The streamed 2048 router-hybrid analysis plus locked 512-token driftx4
validation also rule out router hybrid as a shallow follow-up: it preserved the
generated token but slowed the 512 replay from 262.906s to 451.455s. The next
architecture target should therefore be runner/process fusion for attention,
shared expert, and routed-MoE orchestration rather than another router policy
toggle. `result-summary` now exposes this as
`runner_process_fusion[runner_process_orchestration]=8784cmds` on the 2048
replay, with resident-linear, attention-output, MLA attention, attention
projection, RoPE, and RMSNorm command groups as the first fusion boundary. The
first executable prototype is the opt-in resident-linear batch JSONL server,
which keeps one Metal runner alive across sequential resident projection
requests while preserving the old per-request scratch caps.

The current selected 126-token process-fusion replay remains the MLA attention
server baseline. Its historical locked result completed a 126-token prompt plus
one generated token in 132.394s, while a later same-window rerun under the
current host conditions took 141.647s total with 141.419s in prompt prefill
(`smoke-prefill-126tok-mla-server-baseline-rerun-after-shared-server-current.json`),
or about 0.89 prompt tokens/s. That rerun is the most honest answer to "can it
run now?": yes, with strict launch/audit binding and the same generated token
`[15]`, but not yet at interactive speed. Its prepared memory profile estimates
18.65GB live working set and requires at least 24GB free unified memory before
loading, so short guarded smokes are safe while long free-running chat should
wait for more orchestration work and a full-power system state.

The best current end-to-end smoke result is
`seventeen-token-prefill-smoke-accel-scorecache-mla.json`: total elapsed
128.811 s, prompt prefill 128.667 s, generated token id 11, preflight live peak
17.37 GiB, and system available memory about 83 GiB at admission. Its expert
stage copied 36.793 GiB in 5.601 s (6.569 GiB/s). The preceding two-stage MLA
baseline was `seventeen-token-prefill-smoke-accel-twostage-mla.json` at
135.317 s total and 39.435 s of MLA attention time; the score-cache weights
kernel reduced MLA attention to 32.665 s while preserving the generated token.
A later checked replay written to
`seventeen-token-prefill-smoke-current-safe-replay.json` proves the selected
profile is still runnable under the no-OOM/no-temp-disk guard, but it is not a
new performance baseline: it generated the same token id 11 with about
82.40 GiB memory still available, yet took 444.213 s and `result-compare`
flagged likely system slowdown versus the 128.811 s selected result. Its stage
copy throughput fell to 2.016 GiB/s, and adjacent bounded disk probes measured
3.158 GiB/s with 8 MiB read chunks and 4.656 GiB/s with 32 MiB chunks, below the
prepared manifest's 5.9145 GiB/s calibration. Keep the 128.811 s score-cache
MLA result as the selected baseline until a low-noise replay beats it without a
sentinel-wide slowdown signal.

The current minimum usable OpenAI-compatible chat path is the shortchat64 max4
KV-B-cache server wrapper:
`serve-shortchat-64-max4-kvbcache-memory-accumulator-safe.sh`. Its launch
profile now pins `--prefill-moe-output-accumulator memory` directly via
`launch-profile-shortchat-64tok-auto-keycache-memory-accumulator-decode-keycache-request-locked.json`
and starts only when
`launch-audit-shortchat-64tok-auto-keycache-memory-accumulator-decode-keycache-request-max4-locked.json`
passes the 64-prompt-token / 4-new-token no-OOM envelope. The reproducible HTTP
smoke `server-http-shortchat64-max4-kvbcache-memory-accumulator-smoke-safe.sh`
returned HTTP 200 for `messages=[{"role":"user","content":"你好"}]`,
generated `0003` from token ids `[15, 15, 15, 18]`, and wrote
`server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-latest.json`.
The new profile also pins `--decode-mla-key-cache`, so the request reports
`decode_mla_key_cache=true` without relying on `LARGERLM_MLA_KEY_CACHE`. Its
server token result took 55.381s with post-prefill decode steps near 3.35s,
3.05s, and 3.08s, restoring the fast path lost by the previous profile-pinned
artifact at 72.549s. The result carries `audit_bound=True`,
`safe_to_replay=True`, `replay_ready=True`, and `files_ready=True`. This is the
current safety/reproducibility baseline for real chat; the older
environment-backed max4 artifact remains useful historical performance evidence
but no longer owns the policy contract.
A later telemetry replay,
`server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-telemetry-latest.json`,
preserved the same output (`0003`, token ids `[15, 15, 15, 18]`) while proving
the promoted `prefill_actual_read_time` fields on a real request. Its staged
expert path reduced 145.66 GiB of serial per-token expert-slot reads to
34.664 GiB of unique/planned coalesced reads with zero alignment waste,
1.0 unique read amplification, read-advice on all 1,678 ranges, and 5.578s of
copy at 6.214 GiB/s. That makes the shortchat SSD bottleneck a scheduling/copy
overhead problem rather than a bad coalescing or overread problem.
A copy-counter replay,
`server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-copycounter-latest.json`,
kept the same output and shows that the default 8 MiB copy chunk turned those
1,678 ranges into 5,409 read calls and 5,409 writes, averaging 6.562 MiB per
call. The stage-time counterfactual estimates 3,534 reads at 16 MiB, 1,837 at
32 MiB, 1,679 at 64 MiB, and 1,678 at 128 MiB. The copy-chunk tuning envelope
is therefore bounded: copy32/copy64 can remove most syscall overhead, while a
separate route or layout change is needed to reduce the coalesced-range floor.
The copy64 profile/audit pair now has a reproducible wrapper,
`serve-shortchat-64-max4-kvbcache-memory-accumulator-copy64-safe.sh`, and smoke
harness,
`server-http-shortchat64-max4-kvbcache-memory-accumulator-copy64-smoke-safe.sh`.
The smoke result
`server-http-shortchat64-max4-kvbcache-memory-accumulator-copy64-decode-keycache-openai-chat-latest.json`
preserved `0003` / `[15, 15, 15, 18]`, passed the same no-OOM launch envelope,
and reduced copy calls to 1,679 reads plus 1,679 writes. Its measured stage
copy time was 5.643s, essentially tied with the default 8 MiB run, so copy64 is
kept as an experimental SSD/syscall candidate rather than the default serving
policy.
The companion copy32 profile/audit/smoke is also reproducible, but is retained
as negative evidence rather than a candidate default. Its smoke preserved the
same `0003` / `[15, 15, 15, 18]` output and passed the no-OOM envelope, yet
1,837 read/write calls still measured a slower 7.336s stage copy at
4.725 GiB/s. This reinforces that larger copy chunks only remove syscall
overhead; once calls approach the 1,678-range floor, real progress needs fewer
ranges, a better streamed-expert layout, or less staged expert materialization.
Actual prefill summaries now also carry bounded hotspot tables for the routed
expert stage path: `expert_stage_copy_hotspots` ranks the slowest copy stages
and `expert_stage_range_hotspots` ranks the most fragmented coalesced-range
stages. Each row identifies chunk/layer/tile plus copy time, raw/coalesced
range counts, planned bytes, and copy calls. The summary remains small enough
for HTTP responses, but gives the next layout or fused streamed-expert change a
layer/tile target instead of only a global stage-copy aggregate.
The first real hotspot smoke with selected expert ids preserved the shortchat64
max4 output and points at layer 67 as the initial layout target: it is both the
top copy hotspot and the top range hotspot, with 36 coalesced ranges,
726.750 MiB planned bytes, 112 copy calls, and 0.151s stage copy. Its 38
selected expert ids have only two adjacent pairs in the current expert-id
order, so a layer-local coactivation-aware expert order is the next plausible
range-reduction experiment.
The current offline analysis script can now emit those candidate orders from
one or more result JSON files; for the first shortchat64 sample it simulates
layer 67 from 36 ranges to one range, but that is explicitly an observed-sample
upper bound. Production use requires multi-prompt evidence plus a repack or
indirection design that preserves routing semantics.
The first three-prompt aggregation keeps layer 67 first and simulates its
sampled hotspot ranges from 109 down to 4 with a coactivation candidate order.
That moves the next step from measurement to planning: produce a dry-run
repack/indirection manifest and verify the range-count delta before copying
large expert files or changing the default serving layout.
The expert IO planner now understands an optional per-layer `expert_order`
metadata field. The field is the physical slot order expressed as logical
expert ids; router outputs remain logical ids, and the planner maps them to
physical source offsets. This is only valid when the corresponding layer file
has actually been repacked in the same physical order. Existing layouts without
`expert_order` keep identity order and previous behavior.
The first dry-run top-5 repack manifest estimates 47.812 GiB of total file IO
to rewrite layers 67, 62, 46, 57, and 65. It is deliberately not an executable
layout patch: `safe_to_apply_to_existing_layout=false` and
`requires_layer_file_repack=true` remain the guardrails until a bounded repack
tool copies and verifies the new layer files.
That bounded executor now exists and defaults to no-output dry-run. The execute
path rewrites only selected layer files with bounded positional reads/writes and
hardlinks unchanged layers, so a top-5 GLM-5.2 experiment would write about
23.906 GiB rather than duplicating the full expert directory. The latest real
executor dry-run confirms no output directory was created.
The first executed layout change is deliberately smaller: only layer 67 was
physically repacked into `experts-repacked-shortchat64-top1-layer67`, with the
unchanged layers hardlinked and the original `experts` directory left intact. A
shadow manifest, `manifest-repacked-layer67.json`, points only
`experts_layout` at that repacked layout. Its locked shortchat64 launch
profile/audit passed, and the first audited direct-generation smoke produced
token id `[15]` from the 13-token chat prompt. That run validated the runtime
integration too: staged MoE now checks stage slot source offsets through the
optional `expert_physical_slots` mapping from the batch plan, so repacked
layouts remain strict without assuming logical expert id equals physical slot.
In the smoke, layer 67 staged 38 selected experts as one raw/coalesced range,
removing it from the top hotspot tables.
The next bounded layout milestone repacks the top five measured locality
targets, producing `experts-repacked-shortchat64-top5` and a shadow
`manifest-repacked-top5.json`. The three-prompt offline validator drops those
rows from 692 to 254 coalesced ranges, and the first audited direct-generation
smoke confirms the runtime effect for the 13-token chat prompt: layers 46, 57,
65, and 67 each become one stage range, layer 62 becomes two, and total prompt
expert-stage ranges fall from 1,643 in the layer67-only shadow run to 1,526.
The copy time does not fall monotonically yet because the current staged MoE
path still materializes compact stages and uses custom-metal prefill fallback;
after the high-fragmentation layers are clustered, the dominant remaining work
shifts toward MPP/MPSGraph prefill coverage, MLA attention, resident projection
fusion, and reducing staged materialization overhead.
An experimental top-5 MPSGraph 13x32 audit is also runnable, but it is not a
default-safe serving path: the smoke proves only router-gate MPSGraph coverage
on 75 matrices, accelerates about 0.31% of estimated prefill FLOPs, and changes
the generated token from the custom-metal `[15]` to `[23]`. Because that router
drift also changes downstream expert locality, the architecture policy keeps
the custom-metal top-5 profile as the reproducible layout baseline. MPP/Metal ML
tensor ops remain a future acceleration path until the local SDK exposes public
`metal_mpp` / `mpp::tensor_ops` symbols and the compile/run probes pass with
real Metal access; restricted sandboxes can report `no Metal device` and should
not be used as negative MPSGraph evidence.
The next top10 locality plan exists only as a guarded dry-run. It selects
layers 76, 54, 68, 47, and 55 from the custom-metal top-5 smoke and would write
another 23.906 GiB if executed, while hardlinking the remaining layer files
from the top-5 layout. Because that evidence is currently one post-top5 prompt,
the architecture keeps it as a candidate planning artifact instead of a serving
layout.
For repeated evidence, the planner now supports a minimum hotspot sample count.
With three prompt artifacts and `--min-sample-count 2`, only layer 55 remains as
a post-top5 candidate; that bounded top6 layout has been physically repacked,
validated, and smoked. It preserves token `[15]` and reduces real prompt ranges
to 1,499, but `result-compare` calls the 238.804s run a tie versus top5 because
the total win is under 2%. The architecture therefore treats top6 as a validated
candidate and keeps promotion gated by another replay or bakeoff.
Bounded optimization-target sweeps may open replay candidates, but they do not
change defaults on their own. For the top6 13-token smoke, routed MoE, MLA cache
mode, attention projections, attention output, and cache-write chunking all kept
their existing policies. RoPE split was the lone microbench candidate: the old
Python split plus `--run-rope-batch` path beat fused split by about 3.4% in the
single-layer sweep. The required full replay still stayed inside the two-percent
tie band (`235.592s` versus `238.804s`, token `[15]`), so bakeoff retained the
fused-RoPE top6 baseline. This is the intended promotion flow: microbench
candidate, locked full replay, then bakeoff, with ties retained as evidence but
not as defaults.
Multi-prompt promotion evidence is now planned explicitly. The replay matrix
planner consumes already-audited result artifacts, extracts exact prompt token
ids, verifies each candidate's locked launch binding, and writes per-prompt
commands guarded by a no-existing-runner check; it can also reuse existing
results when their variant and prompt tokens match the matrix entry. The
multi-prompt bakeoff aggregator applies the same result-bakeoff policy per
prompt and requires the configured prompt count before making a promotion
decision. The current top5-vs-top6 aggregate has all three required prompt
pairs complete: the original 13-token pair and the 17-token Hangzhou pair are
ties, and the 19-token explain-ai pair is inconclusive because the top6 run
looked like a system-level slowdown. Hangzhou retained top5 despite top6
improving prompt-prefill and expert-stage copy because total elapsed was
281.503s versus top5's 279.733s; explain-ai also retained top5 after top6 took
536.748s versus 339.474s. This keeps top6 in the candidate bucket and confirms
that layout promotion needs several prompt-level wins, not a single locality
improvement.
The current top5 default has also closed its six 13-token result-summary target
sweeps. MLA cache, attention projection fusion, routed MoE tiling/vector-SwiGLU,
attention-output group32, and RoPE split all retained their existing default
policies. Cache-write chunking found a byte-identical `1MiB` microbench
candidate, but its mean wall delta is only about 38 us on the 13-token probe and
still requires locked full replay plus bakeoff. The architecture therefore keeps
top5 custom-Metal as the serving default and treats cache-write chunking as
deferred low-risk tuning evidence, not a promotion.

For a real layer-0 GLM chunk, the non-indexed MLA batch command measures about
0.104-0.106 s kernel time for the 16-token chunk, down from the earlier
two-stage 0.194 s and head-level 0.313 s baselines. The default batch command
also uses the same two-stage score-cache path for 1-token decode-like calls,
measuring about 0.108 s kernel time versus about 0.166 s for the legacy
singleton kernel on the same layer-0 chunk. Set
`LARGERLM_MLA_LEGACY_SINGLETON=1` to force the old singleton path for bisection.
An attempted value-projection cache for MLA is present only as an explicit
experiment behind `LARGERLM_MLA_VALUE_CACHE=1`. It is numerically exact for the
layer-0 16-token probe, but it is slower for the current GLM-5.2 M5 Max shape
(about 0.219 s kernel time after warmup versus about 0.105 s for the default
values kernel), so the production default leaves it disabled.
The score-side key-projection cache is the promising companion experiment:
`LARGERLM_MLA_KEY_CACHE=1` precomputes `K_nope = kv_b_key * latent` for the
current context before the weights kernel. The low-level Python API and CLI
expose this as `mla_key_cache=True` and `prefill-mla-attention-batch
--mla-key-cache`; prompt prefill, token generation, prepared server config, and
launch profiles expose the higher-level `--prefill-mla-key-cache` flag. Result
JSON records both `mla_key_cache` and `mla_key_cache_bytes`, and the runner
reports `key cache bytes` for every MLA batch call. Result summaries report MLA
key-cache coverage as `enabled=N/M` plus total scratch bytes, and they aggregate
elapsed tensor records by layer-normalized tensor suffix and backend so resident
projection hot spots can be compared across full GLM replays. On the same
layer-0 16-token probe it is numerically exact and previously measured
0.016-0.072 s kernel time versus about 0.105 s for the default score-cache path,
with 768 KiB of extra scratch. A follow-up CLI probe on 2026-07-02 measured
key-cache at 0.015-0.017 s kernel time with exact output matches; the adjacent
default controls were noisier at 0.225-0.445 s. The full 17-token smoke
`seventeen-token-prefill-smoke-accel-keycache-mla.json` also generated token id
11 correctly with key-cache enabled on the 78 first-chunk MLA layers, using
58.50 MiB total key-cache scratch across those layer calls. That full run took
266.028 s and `result-compare` flagged likely system slowdown
(`median_sentinel_ratio` 2.223, 11 of 13 sentinels slow), so it proves the
full-workload plumbing and correctness but not an end-to-end speedup. A second
run, `seventeen-token-prefill-smoke-accel-keycache-mla-rerun.json`, reproduced
the same token id 11 and the same 78 key-cache layer calls but again showed
system-level slowdown: 268.748 s total, `median_sentinel_ratio` 2.239, and 11
of 13 sentinels slow. Immediately after that run, layer-0 16-token probes still
measured the expected local behavior: default score-cache MLA kernel 0.106 s,
key-cache MLA kernel 0.0156 s, and exact 1.00 MiB output match. Key-cache
therefore remains opt-in until a low-noise full smoke confirms the benefit under
the launch profile.
For the current top-5 13-token path, a fresh bounded layer-77 cache-mode sweep
keeps the existing key-value MLA cache policy: it was fastest by total and
kernel mean, while key-only, value-only, and no-cache modes produced no
full-replay candidate.
The adjacent layer-54 attention-projection sweep likewise keeps the fused
projection path: separate projection calls matched numerically but were about
3.92x slower on mean wall time.

The profile replay smoke
`seventeen-token-prefill-smoke-accel-keycache-profile-replay.json` verifies that
the key-cache launch profile and audit are sufficient on their own: the command
applied `launch-profile-17tok-chunk16-accel-keycache.json` with
`--lock-launch-profile`, `--require-locked-launch-profile`, and
`--require-launch-audit`, without manually passing `--prefill-mla-key-cache` or
`--metal-final-logits`. It generated token id 11, applied 23 profile flags,
used key-cache on the same 78 first-chunk MLA layers with 58.50 MiB total
scratch, and kept final logits on the Metal path (`logits=0.184 s`). Its total
270.288 s runtime is again marked as likely system-level slowdown, so it is
profile-plumbing evidence rather than speedup evidence.

The stricter replay/audit smoke
`seventeen-token-prefill-smoke-accel-strict-latest.json` verifies the current
full request gate with `launch-profile-17tok-chunk16-accel-strict.json` and
`launch-audit-17tok-chunk16-accel-strict.json`: it generated token id 11, kept
final logits on Metal, replayed `--require-prefill-acceleration` and the
MPSGraph probe, and preserved request-level accelerated prefill coverage while
leaving the mixed prefill backend policy on `auto`. The actual run used
MPSGraph for the 75 router BF16 resident GEMMs and custom Metal for MXFP4
resident/expert paths. It is correctness and replay-gate evidence, not the
performance baseline: total runtime was 426.567 s and `result-compare` against
`seventeen-token-prefill-smoke-accel-scorecache-mla.json` flagged likely
system-level slowdown with median sentinel ratio 3.448, so `result-bakeoff`
retains the score-cache MLA selected replay.

One later full smoke,
`seventeen-token-prefill-smoke-accel-scorecache-singleton-mla.json`, completed
correctly but took 363.846 s while linear kernels, MPSGraph kernels, expert
copy, rope, and MLA all slowed by roughly the same factor. Low-output MLA
probes immediately before and after that run still measured the expected
0.104-0.108 s kernel times, so that run should be treated as system
load/I/O/thermal contamination rather than a proven algorithmic regression.
`largerlm result-compare` is the lightweight no-weight evidence gate for this:
it compares two result JSON files, checks generated-token/workload match,
checks a prompt-prefill plan signature including chunking, linear backend
counts/FLOPs, expert read bytes, and MLA key-cache evidence,
reports largest elapsed deltas and ratios, and flags likely system-level
slowdowns when the median sentinel ratio and most major timing sentinels rise
together. `result-summary` exposes the same signature as top-level
`prefill_plan_signature`, and text `result-compare` prints the baseline and
candidate signatures whenever they differ, so stale fast baselines cannot be
silently treated as comparable to the current runtime path. The comparison also
includes layer-normalized tensor suffix elapsed rows and resident-linear runner
subphase sentinels (`runner_backend_elapsed_seconds`,
`runner_matrix_f32_elapsed_seconds`, and `runner_accelerator_elapsed_seconds`),
so backend/profile experiments can show whether `o_proj`, MLA projection
matrices, shared experts, router gates, CPU materialization, or Metal/MPSGraph
dispatch caused the delta. It now emits a `profile_recommendation` block as a
conservative profile-promotion gate:
candidate results are promotable only when prompt/generation tokens, generated
ids, and prompt-prefill plan shape match, the run is not flagged as a likely
system slowdown, total elapsed time improves by at least 2%, and no large
tensor-suffix regression is present. Pass
`--require-candidate-promotable` to make `result-compare` return non-zero when a
candidate should not be promoted by automation. For multi-profile experiments,
`largerlm result-bakeoff` compares several candidate result JSON files against
one baseline, selects the fastest promotable candidate, and otherwise reports
`baseline_retained=True`; `--require-winner` turns "no promotable candidate" into
a non-zero script gate. `--promote-only-replay-files-ready` narrows promotion to
candidates whose locked launch binding is file-ready, including a current
launch audit bound to the same profile; selected replay JSON/script generation
enables that policy automatically so automation falls back to a safe baseline
instead of failing on a faster but stale candidate. Bakeoffs always expose a
single `selected` result:
`role="candidate"` when a candidate wins, or `role="baseline"` when the current
baseline is retained. `--require-selected-replay-ready` makes the command fail
unless that selected result has a locked replay-ready launch binding. Result
summaries and bakeoffs also expose
`launch_binding`, which extracts the prepared manifest, locked launch profile,
launch audit, applied profile hash, and `safe_to_replay` status from the result
artifact. That lets automation choose a measured profile and then replay the
same audited profile instead of guessing from filenames. When the result also
contains prompt token ids and max-new-token count, `launch_binding` emits
`replay_generate_token_ids_argv` and `replay_ready=True`, giving scripts the
locked `generate-prepared-token-ids` replay argv without re-parsing the raw
result artifact. `result-bakeoff --write-selected-replay-json` writes that
selected replay envelope as a compact artifact, and
`--write-selected-replay-script` writes an executable shell script that invokes
`largerlm selected-replay-run`, so the selected replay is revalidated before the
exact locked replay argv is executed. Generated replay scripts include
`--check-ssd-read-speed` by default, making the SSD state check part of the
persistent replay entry point rather than a manual operator step. The selected
replay JSON now also carries `pre_run_checks.ssd_read_speed`, so direct
`selected-replay-run selected-replay.json` honors the saved SSD preflight policy
when the prepared manifest's cold-read benchmark is large enough to be stable;
generated scripts still pass the flag so older artifacts fail through the same
gate. Both outputs require the selected result to be replay-ready and
file-ready: the prepared manifest, launch profile, and launch audit must exist,
and the launch profile file's SHA-256 must still match the hash recorded in the
measured result. File-ready replay also parses the launch
audit and requires `schema="largerlm.launch_audit.v1"`, `launch_audit.ok=true`,
and an audit `applied_launch_profile` path/SHA-256 that matches the selected
profile, so a stale audit or renamed profile cannot silently produce a replay
script. `largerlm selected-replay-check` is the no-weight pre-run gate for those
artifacts: it reloads the selected result, recomputes the current launch
binding, checks profile/audit hashes, argv stability, and the current available
memory against the launch audit's request runtime-preflight requirement. It also
rechecks the audited prompt prefill stage-temp disk path and required free bytes,
so SSD-backed expert staging fails before weight loading if the temp volume no
longer has the audited headroom. With `--check-ssd-read-speed`, the same
no-weight gate runs a bounded sequential read against the prepared manifest's
cold-read benchmark file before replay. The default reads 1GiB in 8MiB chunks
and requires the current GiB/s to be at least 75% of the manifest's
`prepare_cold_read_gib_per_second`, so a selected replay can fail closed when
the SSD or filesystem cache state is too slow for the audited profile. New
selected replay artifacts inherit the check size from
`prepare_cold_read_benchmark_requested_bytes`; very small test/debug benchmark
sizes are recorded but not auto-enabled because syscall overhead dominates their
throughput signal. Text-mode failures print the current, required, baseline,
ratio, byte count, chunk size, and benchmark path so operators can tell a true
slow-disk state from a missing benchmark or short read without opening JSON. It
also verifies that audited prefill
acceleration evidence is still replayable: a required accelerated launch must
still have a passing audit gate/probe, request-level accelerated matrix
coverage, `--require-prefill-acceleration`, the audited MPSGraph runtime probe,
and the MPSGraph threshold flags in the current launch profile. The command
returns non-zero before a heavy
replay can start if any referenced file has drifted, the machine is currently
below the audited no-OOM envelope, or the temp disk budget is no longer
available; JSON `selected-replay-run --dry-run` only reports `run_argv` and
`run_command` after all checks pass.
`largerlm selected-replay-run` wraps the same check and only then execs the
locked command; `--dry-run` prints the checked command without loading model
weights. Non-dry-run execution acquires `.largerlm-selected-replay.lock` in the
prepared package before the pre-run checks and keeps that advisory lock across
the final `exec`, so a second selected replay for the same prepared package
fails before SSD reads, page-cache pressure, or runner allocation can overlap
the first one. Direct `generate-prepared-token-ids` and `generate-prepared-text`
acquire the same prepared-package lock immediately before entering generation;
when launched by `selected-replay-run`, an internal inherited-lock marker keeps
the child generation process from trying to relock itself while still preserving
the cross-process exclusion. `serve-prepared` keeps its existing in-process
request lock and now acquires the same prepared-package file lock around each
generation request, so multiple server processes and offline generation commands
cannot overlap on the same prepared GLM package. `bench-prepared-token-ids` and
the underlying benchmark API also acquire this lock before invoking token
generation, so performance probes cannot accidentally run alongside live
serving or replay. Prepared server `/health` reports the lock path,
availability, busy state, and inherited selected-replay marker, giving operators
a no-weight way to see that a package is currently occupied before starting a
request. The same health payload now reports the prefill acceleration
requirement gate when acceleration is required, including the configured backend,
runtime/probe booleans, selected accelerated backends, and reason code, so a
server can prove MPSGraph readiness from `/health` after startup instead of only
failing before startup. Successful server token/text responses include the
request admission check and launch-audit envelope as well, so saved HTTP
generation artifacts can prove which guard envelope admitted the request. It can
append an output-only `--write-result PATH`
to a checked `generate-prepared-token-ids` replay and can append `--quiet-runner` as a
logging-only wrapper, and JSON dry-runs report the resulting
`run_argv`/`run_command` separately from the immutable selected replay argv so
automation can see the exact command without mutating the replay artifact. When
an older launch audit carries a request profile with explicit acceleration
guard flags, `selected-replay-run` materializes the missing request-profile
guards into the run argv; if the audited request recorded
`prefill_linear_backend.configured/effective` as `auto`, the stale
`--prefill-linear-backend mpsgraph-f32` request-profile flag is treated as an
auto-policy artifact rather than forcing MXFP4 resident matrices through an
unsupported MPSGraph path. The
current GLM-5.2 prepared package also carries
`selected-replay.json` and `selected-replay.sh` as the persistent replay entry
for the best file-ready 17-token smoke; the script resolves its own directory,
returns to the repository root for repo-relative result/profile/audit paths, and
forwards `--dry-run` to the no-weight check. Full
GLM-5.2 MXFP4
smokes should include `--metal-final-logits`; an
interrupted key-cache run without that flag fell back to the Python MXFP4 logits
loop in `compute_final_logits`, which is too slow for routine end-to-end probes.
Request-checked launch profiles now record `--metal-final-logits` when
`inspect-prepared --check-metal-final-logits` succeeds, so applying the key-cache
profile is enough to replay the safe logits path.
The package-level `launch-profile.json`/`launch-audit.json` is a separate
canonical safety profile for current-machine admission. On the M5 Max bring-up
package it is bound to the explicit `custom-metal` prefill fallback, requires
the prepared memory profile, GLM-5.2 public-shape, and GLM 4-bit guards, and
records an allow-listed non-accelerated launch audit for a 2048-token /
8-new-token envelope.
That canonical profile is suitable for conservative admission and server
bring-up; generate the audit with host total-memory visibility so strict replay
can validate `system_total_memory_bytes`. `selected-replay.json` remains the
measured 17-token replay entry.
`serve-canonical-safe.sh` is the matching local API launcher for that canonical
profile. It verifies local shard headers first, then starts `serve-prepared`
with the locked package-level launch profile and audit, request caps of 2048
prompt tokens and one new token by default, and the explicit `custom-metal`
prefill fallback. The wider audited decode cap is opt-in via
`LARGERLM_CANONICAL_MAX_NEW_TOKENS_CAP=8`. The launcher also runs a bounded SSD
read-speed gate before startup and fails closed when the current expert-pack
read speed falls below the prepared cold-read baseline ratio.

The selected prepared package also includes `serve-selected-safe.sh`, a
conservative local API launcher for the same audited 17-token / 1-new-token
envelope. It runs `selected-replay-check --check-ssd-read-speed` before starting
`serve-prepared`, so the service does not begin accepting requests unless the
current memory, artifact, launch-audit, prefill-acceleration, and bounded SSD
read-speed gates are still satisfied. It also pins `--tokenizer-backend
tokenizers` for the GLM-5.2 MXFP4 package, avoiding a failed transformers
AutoTokenizer path while preserving local-tokenizer encode/decode semantics.

## Runtime Split

### Prefill

Prefill has large prompt-token batches and therefore real GEMM shapes. It should
be the first place to use Metal 4 MPP tensor ops:

- Q/K/V/O projections
- DSA indexer projections
- dense first layers
- shared expert or dense MLP paths when batch dimensions are large

The prefill kernel path should use static tile extents for full tiles, dynamic
extents only at edges, and benchmark tile sizes per M5 Max.
`prefill-plan` is the current no-allocation planning boundary for this work: it
derives GLM attention, dense MLP, shared expert, router, and DSA/indexer GEMM
shapes from `config.json`, estimates FLOPs and activation peaks, and labels
large prompt-token GEMMs as MPP tensor-op candidates. It also estimates routed
expert assignments, unique experts per MoE layer, and expert slot bytes that
would be streamed from SSD during prefill. For GLM MLA/DSA attention it
separately estimates prompt-cache read/write bytes, with DSA-indexed layers
using the configured `index_topk` cap and full-indexer layers accounting for
their index-key cache reads. It reports the public GLM-5.2 shape diagnostic
and `--require-public-glm-5-2-shape` turns that diagnostic into a pre-profile
gate for both `prefill-plan` and `prefill-plan-calibrate`. That gate also locks
the routed expert planning target to `expert_bits=4`, and matching public
GLM-5.2 plans carry the same guard in `suggested_launch_profile` only when the
plan is using 4bit experts, so prepared launches can reject shape drift without
blessing an incompatible expert-bit estimate. Routed MoE plans include static
capacity hints for future fixed-shape expert execution: balanced capacity gives
the efficient average assignment slots per expert and explicitly requires an
overflow path, while spill-free capacity uses the per-prompt-token expert bound
to validate kernels without dropping assignments. Both variants report
per-layer f32 activation bytes for the compact input/output and gate/up slots
so static expert kernels can be budgeted before allocating buffers. With
`--max-runner-scratch-mib`, it also estimates the current staged MoE runner's
assignment table, token-seen bitmap, auto token-block buffers, aligned expert
slot allocation, and per-layer scratch peak. If `--max-prefill-activation-mib`
enables chunking, the plan also reports chunk-major routed expert read bytes,
since a bounded prompt driver may stage the same layer's expert slots once per
chunk rather than once for the whole prompt. `routed_expert_read_cost_plan`
turns that into baseline bytes, planned bytes, extra bytes, read
amplification, and optional read seconds when `--ssd-read-gib-s` is supplied
from a measured `disk-read-benchmark` run.
`prefill-backend` is the matching local capability boundary: it inspects the
selected macOS SDK for Metal 4 machine-learning, `MTLTensor`, int4 tensor data
types, MPSGraph matmul, and visible MPP tensor-op symbols, then optionally runs
a tiny Metal device probe. The host probe only checks selectors and attempts a
64-element ML tensor allocation, including the tensor size/alignment selector
needed before creating ML tensor-backed buffers, so it is safe to run before
any large model work. MPP compile probes require that host probe and report
`mpp_compile_probe_requested` separately from ran/ok fields. The stricter
`--run-mpp-probe` path compiles and runs one 32x32 half-precision MPP
`matmul2d`, verifies the output, and records `mpp_run_probe_*` fields; when it
is requested, `mpp_runtime_available` requires that tiny execution probe to
pass. `metal/largerlm-runner --self-test-mpp` runs the same tiny MPP matmul
inside the inference runner binary as an additional executable-boundary check,
without loading weights. A separate
opt-in MPSGraph runtime probe runs only a 2x2 float32 matmul and records
`mps_graph_probe_*` fields; when requested, `mps_graph_runtime_available`
requires that tiny matmul to succeed. Prepared commands use a configurable
backend probe timeout, defaulting to 5 seconds so first-run MPSGraph/Metal
compiler cold starts on M5-class hosts do not look like missing acceleration.
The report carries the probe path, whether a probe was requested, host-probe
pass/fail state, the probe timeout, and a short failure detail when the probe
binary is missing, times out, exits non-zero, emits invalid JSON, or Metal
returns a tiny tensor allocation error; `prefill-backend --write-report` can
atomically save the same capability JSON for bring-up evidence.
If the host cannot create a default Metal device, the summary reports that as a
single explicit reason while keeping the raw selector booleans in JSON.
If Metal 4 ML is available but public MPP tensor-op symbols are missing, the
neural-accelerator status reports `missing_public_mpp_symbols` even when an
explicit MPP run probe also fails, while preserving the run-probe error fields
for diagnosis.
`mps_graph_runtime_available` distinguishes SDK-level MPSGraph matmul headers
from a runtime that passed the host probe; prepared serving uses that runtime
field for `auto` backend admission. The same report also separates
`prefill_acceleration_runtimes` from
`selectable_accelerated_prefill_backends`. MPP appears as the selectable
`mpp-f32` backend after its compile probe succeeds; hard acceleration gates
additionally require the execution probe. MPSGraph remains the portable
accelerated fallback. `suggested_prefill_acceleration_flags` prefers a validated
backend and emits either `--prefill-linear-backend mpp-f32
--require-prefill-acceleration --run-mpp-probe` or the equivalent MPSGraph
flags.
`validated_accelerated_prefill_backends` is the hard-gate subset of the
selectable set: `mpp-f32` and `mpsgraph-f32` enter it only when their respective
runtime probes were requested, ran, and passed. Launch-audit consumers use this
field to reject artifacts that advertise selectable acceleration without
preserving runtime proof. The current `mpp-f32` implementation converts
resident F32/BF16/F16 matrices into a bounded F32 working buffer and dispatches
32x32 MPP TensorOps tiles. Quantized routed experts continue to use the existing
MXFP4 kernels.
`prefill_neural_accelerator_status` summarizes the same MPP/Metal-ML path as a
machine-readable bring-up state: runtime-visible, selectable, compile/run-probe
status, planned `mpp_tensor_ops_gpu_neural_accelerator` execution path, and the
reason generation still cannot select it when it remains a gap. A successful
run probe also records the MPP kernel variant, tile shape, dtype, and execution
primitive, giving the future resident-GEMM backend concrete shape evidence
without marking MPP selectable prematurely. Prepared health and
`inspect-prepared` surface that same status inside
`prefill_backend.capability`.
When `prefill-plan --inspect-backend` uses a non-default probe timeout, the
saved launch profile rewrites it to the prepared-command flag
`--prefill-backend-probe-timeout-seconds`, keeping M5 first-run Metal/MPSGraph
compiler cold-start behavior reproducible.
Prepared commands that require prefill acceleration also require that selectable
MPSGraph path to have a passing runtime probe, so SDK-visible symbols alone are
not treated as proven acceleration. The shared acceleration gate emits a stable
`reason_code` alongside the human-readable reason, keeping M5 launch audit and
serving checks machine-readable while MPP remains a visible but not-yet
selectable runtime.
`prefill-plan --inspect-backend` attaches that
report and converts raw candidate counts into effective Metal 4 ML and strict
MPP-symbol candidates for the current machine. `recommended_backend` remains an
implementation recommendation:
it reports `mpp_tensor_ops_prefill` only when public MPP symbols are visible,
the local compile probe succeeds, the full Metal 4 ML runtime selector set is
available, and a tiny ML tensor allocation succeeds; otherwise it falls back to
MPSGraph only when matmul support is present, the requested host probe has not
failed, and any requested MPSGraph runtime probe has succeeded.
The plan also carries `prefill_backend_candidates`, a FLOP-sorted implementation
priority list for candidate prefill GEMMs. Each entry includes the exact
`M x K x N` shape, layer count, total FLOPs and resident weight bytes,
arithmetic intensity, static-tile and edge-tile status, and the current
preferred execution path: `mpp_tensor_ops_gpu_neural_accelerator` when MPP is
ready, `mpsgraph_gpu_matmul` when the supported fallback is safer, or
`custom_metal_gpu_fallback` when neither public ML path is available.
Candidate GEMMs also carry a tile plan derived from the Apple M5 starting
heuristics: 32x32 simdgroup output tiles, 2x2 simdgroups per threadgroup, and
BK/K tiles of 128. The plan reports grid size, full versus edge threadgroup
tiles, K alignment, and whether a full static-extent kernel can be used. For
the 4096-token GLM-5.2 fixture, 14 of the 15 current prefill candidates are
full static tiles with no edge threadgroups; the small
`dsa.index_weights_proj` candidate keeps an N-edge tile because its output
width is below the default 64-column threadgroup tile.
To keep prefill memory bounded, `--max-prefill-activation-mib` computes a
tile-aligned prompt chunk size from the largest per-token GEMM activation. For
example, the 4096-token GLM-5.2 fixture with a 128 MiB activation cap recommends
2240-token chunks, preserving 64-token threadgroup alignment while keeping the
largest intermediate below the cap; the same plan reports a 2.0x routed expert
SSD read amplification for that cap. Runtime entry points can enforce the same
tradeoff with `--prefill-max-routed-read-amplification`: a positive value
rejects resolved prompt chunks that would multiply routed expert SSD reads
beyond the cap before work directories or stage files are created. A separate
`--prefill-max-routed-read-gib` cap rejects prompts whose absolute planned
routed expert SSD reads are too large even when the amplification ratio is
acceptable. When a measured SSD throughput is available,
`--prefill-ssd-read-gib-s` converts the same byte estimate to seconds and
`--prefill-max-routed-read-seconds` enforces a latency-oriented admission cap.
`prefill-plan` now also emits `suggested_guard_flags`, an argv-style copy of
that chunk/read profile with 5% headroom, so a planning run can directly seed
generation, benchmark, inspection, or serving admission guards. The same plan
emits `suggested_stage_temp_guard_flags` from the resolved chunk, expert slot
size, and `--expert-stage-align-kib` value, giving launch scripts matching
`--prefill-max-stage-mib`, `--prefill-max-compact-stage-mib`,
`--prefill-max-stage-raw-ranges`, and
`--prefill-max-stage-coalesced-ranges` caps before any expert files are staged.
`suggested_prefill_guard_flags` combines the read and stage suggestions into
one deduplicated argv profile, so the chunk size only appears once when copied
into a generation, benchmark, inspect, or serving command.
The first executable prefill fallback primitives are batch RMSNorm and resident
linear projection. `--run-rmsnorm-batch` reads one resident norm vector,
validates an `M x hidden` f32 input batch, and emits an `M x hidden` normalized
batch. `--run-resident-linear-batch` reads one resident projection matrix and
emits an `M x out_dim` f32 batch. Both enforce scratch limits before Metal
buffers are allocated. The Python `prefill-rmsnorm-batch` and
`prefill-linear-batch` commands wrap these primitives with resident layout
validation, input/output byte checks, and runner subprocess reporting. The
resident batch-linear smoke now covers forced MPSGraph, forced MPSMatrix, and
CLI `auto` selection for 128-token F32/BF16 32x32 projections with explicit
lowered test thresholds, plus an opt-in JSONL resident-linear server path that
processes two bounded requests in one runner process, so the supported M5
prefill fallback paths have a real executable gate beyond tiny matrices. The
`prefill-attention-prefix` command composes those checked stages into the first
GLM prefill prefix, emitting `input_layernorm.f32`, `q_a_proj.f32`, and
`kv_a_proj_with_mqa.f32` for a bounded prompt chunk without bypassing per-stage
memory guards. `prefill-attention-projections` continues that batch flow through
`q_a_layernorm`, `q_b_proj`, `kv_a_layernorm`, and `kv_b_proj`; the KV-A
latent/RoPE split is streamed row-by-row so the split does not allocate another
full activation buffer. `prefill-cache-write` then streams those f32 KV-A rows
into the `mla_kv` decode-cache segment at a prompt position range, encoding
BF16/F32 per row and validating cache layout/file size before writing. The
default linear primitive uses a simple Metal GEMM kernel today. It now also has
opt-in `mpsgraph-f32` and `mps-matrix-f32` batch backends for F32/BF16/F16
resident matrices. `mpsgraph-f32` uses MPSGraph matmul; `mps-matrix-f32` uses
Metal Performance Shaders `MPSMatrixMultiplication` and is currently explicit
opt-in so it can be benchmarked without changing the conservative `auto`
selection policy. `auto` uses MPSGraph only for large resident GEMMs. The default
thresholds are 2048 prompt tokens and a 4096 minimum matrix dimension, based on
bounded M5 Max crossover calibration, but generation, benchmark, inspect,
serving, and the low-level batch-linear command can tune them with
`prefill_mpsgraph_min_batch_tokens` and `prefill_mpsgraph_min_matrix_dim`. The
MPSGraph and MPSMatrix paths compute in F32, converting BF16/F16 resident
matrices into a bounded F32 matrix buffer and counting that conversion in the
scratch estimate; small `auto` shapes stay on the custom Metal path. Prepared
offline generation and serving resolve `auto` to
`custom-metal` when local backend inspection cannot see MPSGraph matmul support,
and serving exposes both configured and effective backends in health/request
inspection. Prepared benchmarks reuse the same request-admission runtime backend
for the measured runner call when the configured backend is `auto`, so benchmark
telemetry and execution cannot drift across MPSGraph availability boundaries.
Request launch profiles include a `prefill_backend_policy_flags` section with
the checked request's effective backend, making explicit custom-Metal,
MPSGraph, and MPSMatrix A/B experiments replayable without inheriting a
different global backend recommendation. Strict launch audits require the
checked request's effective prefill backend to match the runtime-resolved
backend, making that fallback boundary replayable.
The same selector is threaded through the
attention, dense MLP, and staged routed-MLP prefill wrappers, giving a supported
Apple ML stack comparison point while the public SDK still lacks the
`mpp::tensor_ops` shader symbols needed for the stricter MPP path. Resident
batch-linear telemetry exposes matrix scratch, logical F32 matrix bytes, and
raw BF16/F16 conversion bytes separately so GLM-5.2 prompt runs can tune the
runner scratch cap without guessing where MPSGraph overhead comes from. Prompt
prefill also rolls up actual resident-linear elapsed seconds and estimated
TFLOP/s by backend, so M5 MPSGraph threshold tuning can compare the real
custom-Metal versus MPSGraph execution mix instead of relying only on planned
FLOPs. Result summaries also report MLA key-cache coverage as
`enabled=N/M` plus total key-cache bytes, so backend and copy experiments cannot
accidentally compare a cached MLA run against an uncached one. Prepared
benchmarks preserve that timing summary in
`sections.prefill_actual_linear_backend` inside the suggested launch profile,
so a saved M5 calibration artifact carries both replayable backend policy and
the actual backend mix that produced the run. Strict launch-audit replay accepts
older profiles without this optional section, but rejects malformed backend
timing evidence whenever the section is present.
`prefill-linear-calibrate` reuses the same wrapper on synthesized bounded
identity GEMMs, including rectangular `INxOUT` projection shapes, measuring
custom Metal, MPSGraph, and MPSMatrix elapsed time and emitting replayable
MPSGraph auto-threshold profile flags only when the measured grid has no
included MPSGraph-slower case. The synthesized resident matrix dtype defaults
to F32 and can be set to BF16 with `--matrix-dtype BF16`, while inputs and
outputs remain F32 for backend comparison. MPSMatrix is reported as an explicit
comparison backend and participates in per-case winners, but it does not change
the conservative `auto` policy yet. Each calibration result now includes a
`backend_comparison` summary with total elapsed time, estimated TFLOP/s,
speedup versus custom Metal, winner counts, winner-weighted FLOPs, and a
conservative `recommended_explicit_backend_policy_flags` entry when a backend
should replace `auto`: non-custom backends must meet the requested speedup on
every measured case and win some measured FLOPs, while custom Metal can be
recommended when the Apple ML comparison backends lose and no finer MPSGraph
threshold applies. That explicit-backend recommendation is folded into the
calibration `suggested_launch_profile`, so a standalone calibration artifact can
be replayed with `--apply-launch-profile`. In `prefill-plan-calibrate`, the
merged profile uses the calibrated explicit backend only when the plan did not
already force a non-`auto` backend. The command estimates the full work-directory
footprint for all matrix/input files plus custom, MPSGraph, and MPSMatrix
outputs across repeats before creating any directory, so manual sweeps retain
the same no-surprise disk guard as plan-driven calibration. The same preflight
checks both the configured `--max-calibration-work-dir-mib` cap and the target
volume's free bytes plus `--calibration-work-dir-free-margin-mib`.
`prefill-plan` now exports those rectangular calibration shapes directly from
its highest-FLOP unique resident GEMM candidates, using the resolved prefill
chunk as the batch size and computing bounded calibration-case, resident-matrix,
and runner-scratch caps with headroom. The resulting
`suggested_prefill_linear_calibration_flags` are deliberately separate from the
generation `suggested_launch_profile` so calibration-only flags cannot be
replayed as inference guards. `prefill-plan-calibrate` is the automation bridge:
it rebuilds the static plan, applies those shape/cap suggestions to
`prefill-linear-calibrate`, writes optional calibration flags and runtime-policy
launch profiles, and refuses to dispatch the runner when planner-derived caps
exceed its explicit `--max-auto-*` calibration limits. It also estimates the
work-directory bytes needed for the generated matrix/input files plus custom
Metal, MPSGraph, and MPSMatrix outputs for every repeat, rejecting sweeps above
`--max-calibration-work-dir-mib` before any runner dispatch; the lower execution
layer then checks the target volume's real free space before creating the
directory. That estimate follows the selected calibration matrix dtype, and the
written applied calibration flags include `--matrix-dtype` so BF16 calibration
artifacts are replayed as BF16 rather than reverting to F32. For BF16 sweeps,
the planner-derived runner scratch cap is raised automatically when the
MPSGraph/MPSMatrix comparison path needs both the raw BF16 matrix and its F32
conversion resident at once. Its written launch profile is merged for prefill
replay: static
planner guard sections are carried forward, and any measured calibration
runtime policy replaces the static MPSGraph threshold values while preserving
planner-side acceleration gates such as `--require-prefill-acceleration` and
`--prefill-min-accelerated-flop-fraction`. When `--ssd-read-gib-s` is provided,
the same merged profile also carries the planner's routed-expert read-second
guards, tying the calibration artifact back to measured local SSD throughput.
The MPSGraph batch-linear path skips custom Metal batch-kernel library
compilation, so it avoids unrelated shader compile latency and cannot be
blocked by custom-kernel compile failures.
Resident matrix validation rejects zero dimensions and uint64 byte-size
wraparound, and the direct runner batch-linear command stats the input file
before reading it so scratch-limit failures happen before prompt-batch input
allocation.
Prepared health exposes the same auto-policy thresholds, and offline
`inspect-prepared` request checks estimate the MPSGraph/custom-Metal resident
matrix mix, peak matrix scratch, and batch-prefill MLA/DSA cache read/write
traffic for the resolved request. The request check also returns a bounded
`top_matrices` list sorted by estimated FLOPs, so prepared-model hot spots can
be selected for MPSGraph/MPP work without loading resident weight bytes.
Request summaries now also report `total_matrix_scratch_bytes` alongside peak
scratch and raw-conversion bytes; peak scratch remains the safety cap while the
total exposes MPSGraph F32-conversion churn across all resident GEMMs.
The same request summary marks `mpp_tensor_ops_candidate` for matrices that
meet the planner's MPP policy (`>=128` prompt-chunk tokens and both GEMM
dimensions `>=32`) and reports candidate counts/FLOPs separately from
selectable accelerated backends, so the MPP path can be brought up against real
prepared layouts without claiming it is already executable.
`prefill_acceleration_coverage` wraps that matrix mix into a request-level gate:
when acceleration is required, a checked request must resolve at least one
resident matrix to the current selectable accelerated backend. Prepared text
generation and server text/OpenAI-compatible requests run a lightweight
tokenizer encode first so the same coverage gate uses the real prompt token
count. This keeps prepared generation, prepared token-id benchmarks, and M5
bring-up checks from passing merely because MPSGraph or MPP probes are
available while the actual prompt chunk falls below the auto thresholds.
Coverage is weighted by estimated prefill linear FLOPs as well as matrix count:
resident candidates contribute `2 * prompt_chunk_tokens * rows * cols`, and
streamed routed expert MLP work contributes
`6 * routed_assignments * hidden_size * moe_hidden_size` for the fused
gate/up/down expert path. The result includes total, accelerated, custom-Metal,
unsupported-MPSGraph, fractional accelerated FLOPs, and explicit
`streamed_routed_expert_*` fields. It also reports
`non_router_matrix_count`, `non_router_estimated_flops`,
`non_router_unaccelerated_matrix_count`,
`non_router_unaccelerated_estimated_flops`, and
`non_router_unaccelerated_flop_fraction`, making router-gate-only experiments
show the remaining real acceleration gap directly. It further divides that
gap into `non_router_unaccelerated_streamed_routed_expert_*`,
`non_router_unaccelerated_non_streamed_*`, and
`unaccelerated_backend_*` maps, so M5 backend work can distinguish streamed
MXFP4 routed experts from non-streamed shared/resident fused paths and
unsupported conversion cases. The weighted view is used for diagnostics and
frontier reporting so a small accelerated matrix cannot hide larger resident or
streamed routed-expert work that still falls back to custom Metal.
Request coverage, actual prompt-prefill coverage, and benchmark coverage now
also carry `mpp_tensor_ops_candidate_*` counts/FLOPs plus the MPP candidate
policy. These fields mark M5 neural-accelerator opportunities without changing
the hard acceleration definition: only a selectable backend that actually ran,
currently MPSGraph, satisfies the acceleration gate.
The optional `prefill_min_accelerated_flop_fraction` setting promotes that
weighted view into a gate. Values greater than zero imply the acceleration
requirement for CLI, server, and benchmark paths; request checks and actual
prompt-prefill results must both meet the configured accelerated FLOP fraction.
Generation, benchmark, and serving paths also re-check the actual
`prompt_prefill.prefill_acceleration_coverage` before returning success, so a
runtime fallback to custom Metal is reported even if the request preflight was
optimistic.
`prefill_acceleration_frontier` evaluates a bounded set of candidate prompt
chunks against the same resident matrix policy and reports the minimum safe
chunk that would use MPSGraph, when one exists. Coverage failures include that
minimum chunk as a reusable guard suggestion, and each frontier row mirrors the
MPP candidate matrix count and FLOPs for that chunk.

### Decode

Decode is routed-expert streaming:

1. Run attention/indexer/router from resident weights.
2. Select top-k routed experts.
3. `pread` only those experts from the packed per-layer expert file.
4. Run custom Metal affine-dequant matvec kernels.
5. Add the resident shared expert path for GLM MoE layers.
6. Fuse expert combine, residual, and RMSNorm where possible.

Router selection follows the GLM/DeepSeek-style noaux path when the resident
layout carries router metadata: compute scores from logits, add
`gate.e_score_correction_bias` only for expert selection, optionally restrict to
top groups, gather weights from the original scores, normalize top-k weights
when `norm_topk_prob` is true, then multiply by `routed_scaling_factor`.
High-level decode, prompt-prefill, and generation CLIs derive those router
defaults from `--model-config`, while explicit CLI flags remain authoritative.

Shared experts are resident tensors, not SSD-routed experts. The runner reads
only the current layer's shared gate/up/down matrices, one matrix at a time,
computes the shared MLP, and adds it to the routed MoE accumulation buffer.

The current MLP block wrapper covers the decoder feed-forward half after
attention: read `post_attention_layernorm.weight`, apply RMSNorm to the
post-attention residual hidden state, run routed/shared MoE on the normalized
state, then add the residual back to the output. It remains a separate command
so the attention boundary can be tested before fused full-layer execution.

The resident linear runner is the first attention-side primitive. It locates a
current-layer resident matrix by suffix, enforces matrix/scratch byte limits,
reads only that matrix from `resident.bin`, and runs a Metal matvec for F32,
BF16, or F16 weights. This is the reusable path for GLM attention and DSA
projections such as `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`,
`o_proj`, and indexer weights.

The attention projection wrapper covers the GLM MLA prefix for one layer:
`input_layernorm`, `q_a_proj`, `q_a_layernorm`, `q_b_proj`,
`kv_a_proj_with_mqa`, `kv_a_layernorm`, and `kv_b_proj`. The KV-A output is
split by the `kv_a_layernorm` width: the prefix becomes the compressed KV latent
for `kv_b_proj`, and the remaining suffix is the RoPE-side dimension for the
next milestone. The runner still reads only one resident matrix at a time and
keeps `--max-resident-matrix-mib` plus `--max-runner-scratch-mib` checks before
allocating Metal buffers. Batch `--run-attn-projections --batch-tokens N` is
available as an experimental staged helper, but Python prompt prefill enables it
only when `LARGERLM_EXPERIMENTAL_BATCH_FUSED_ATTN_PROJECTIONS=1` is set; the
default keeps the proven single-token fused path and the decomposed batch path.

When given a `plan-cache` layout, `--run-attn-projections` can append the
current token's KV-A latent+RoPE vector into the decode cache file. This is a
single bounded BF16/F32 `pwrite` at `segment.offset + position * stride`; it
does not allocate or initialize the whole cache file.

The standalone RoPE primitive applies rotary embeddings to f32 Q-rot/K-rot
vectors on Metal. It supports DeepSeek's default half-split `rotate_half`
semantics and the interleaved GLM variant. The Python CLIs derive
`rope_interleave` from `--model-config` when the config carries that GLM flag.
DSA full-indexer cache writes and query projection top-k use their own
`indexer_rope_interleave` config flag, so GLM-5.2's indexer RoPE layout can
match the checkpoint independently of the main MLA attention RoPE path.
`--run-rope-batch` extends the same
kernel family to prompt chunks by rotating `[tokens, heads, rope_dim]` Q and
`[tokens, rope_dim]` K rows at `start_position + token_index`. This keeps RoPE
independently testable before it is fused into the attention decode or prefill
paths.
The Python `prefill-rope-batch` command streams `q_b_proj.f32` one token row at
a time into `q_nope.f32` and `q_rope.f32`, then calls the Metal batch RoPE
runner for `q_rope.f32` plus the KV-A RoPE suffix emitted by
`prefill-attention-projections`.
`--run-mla-attention-batch` and the Python `prefill-mla-attention-batch` wrapper
complete the current prompt-chunk attention value path. Each output element
computes a causal softmax over cache rows `0..start_position+token` using the
resident `kv_b_proj.weight`; the implementation is still serial per output
element, but it validates cache read bytes, resident matrix size, and total
scratch before allocating Metal buffers.

The single-token MLA attention primitive connects those pieces for decode
correctness: it reads a bounded prefix of the current layer's MLA cache,
converts the current layer's `kv_b_proj.weight` to f32, rotates each cached
K-rot vector by its token position, expands K-nope/value through KV-B, computes
a stable softmax per head, and emits the per-head value output. This kernel is
intentionally serial inside each output element; it is a correctness milestone
before optimized reductions, output projection, and residual fusion.

The attention output wrappers apply the current layer's resident `o_proj` to
the per-head value output and add the original residual hidden state. The
single-token runner finishes the decode attention residual boundary; the Python
`prefill-attention-output-batch` wrapper reuses bounded batch resident linear
for `o_proj`, then streams the residual add one hidden row at a time for prompt
chunks.
The prompt-prefill attention-output JSONL server keeps that same batch
`o_proj + residual` primitive in one Metal process across layers. It preserves
the resident-layout, tensor-size, residual/input/output byte, and scratch-cap
checks, and is exposed as `AttentionOutputBatchServerSession` plus the
`--persistent-attention-output-server` /
`--prefill-persistent-attention-output-server` opt-in flags. It is not a
default path: the first locked 126-token GLM bakeoff improved the local
`o_proj` tensor time but retained the faster MLA-server selected replay.
`prefill-attention-block-batch` composes the current prompt-chunk attention
path across those bounded pieces: projection, cache write, Q/K RoPE, causal MLA
attention values, `o_proj`, and streaming residual add. It reports the maximum
substep peak instead of introducing a larger fused allocation.
`prefill-dense-mlp-block-batch` adds the matching prompt-chunk dense MLP path
for GLM-5.2 dense-prefix layers. It applies post-attention RMSNorm with the
batch norm primitive, runs resident gate/up/down projections with the bounded
batch linear primitive, streams SwiGLU one intermediate row at a time, and
streams the final residual add one hidden row at a time. The default dense
suffixes fall back to `mlp.switch_mlp.*` names.
`prefill-routed-mlp-block-batch` adds the first routed MoE prompt-chunk path.
It is intentionally a safe serial baseline: the wrapper preflights the packed
layer with `check-runtime` logic, then feeds each hidden row through the
existing bounded `--run-mlp-block` runner and appends the row output. It can
emit per-token router JSON for later grouping. This validates prompt-chunk
correctness and memory guards before the optimized scheduler batches router
decisions and coalesces SSD reads for shared expert slots across tokens.
`plan-batch-expert-io` consumes those per-token router JSON files and turns
them into the next scheduler input: token-to-expert routes, expert-to-token
assignments, and a coalesced aligned expert-slot read plan. It reports both the
serial per-token read cost and the planned unique-slot read cost so SSD
prefetch policy can be tuned before a fused routed batch kernel exists.
`stage-batch-experts` materializes the same plan into a stage file plus
manifest. It streams each coalesced aligned range from the packed layer file to
the stage file using a bounded copy buffer, after first attempting non-fatal
macOS `F_RDADVISE` hints for those coalesced ranges. The manifest records the
read-advice support, call count, hinted bytes, and any error so SSD hinting is
visible without becoming a correctness requirement. It then records every
selected expert's source offset and stage offset. It also checks free disk plus an
optional stage-file margin before opening the output, so a large prompt chunk
fails before any giant partial file is written. The bounded copy itself uses a
single reusable buffer with positional `preadv` when available, falling back to
bounded `pread` chunks; it never uses `mmap` or a moving file cursor, so the
copy cap is also the largest source buffer held in Python. If the bounded copy
or manifest write
fails after opening the output, the partial stage file is removed before
returning the error. The manifest itself is written through a temporary file and
atomic replace, so a failed manifest write does not leave a partial success
marker beside the stage file. This creates the next execution boundary for a
batch expert runner: compute can consume staged unique slots without re-reading
the original layer file per token.
Prompt prefill rolls those per-stage hints up into attempted range, `fcntl`
call, hinted-byte, and non-fatal failure counters. Benchmark and serving payloads
preserve the same counters plus cumulative planned read seconds, the SSD GiB/s
used for the estimate, the configured read-time cap, and the cap result, so SSD
hint coverage and read-time budget can be compared against planned expert reads
during long-prompt experiments. A stage whose measured copy time exceeds the
remaining seconds cap is rejected before the staged routed MoE Metal command is
launched. Successful stage manifests record actual stage-copy elapsed seconds,
effective staged-file GiB/s, and whether that copy stayed inside the same
seconds cap when one was configured, keeping measured SSD behavior separate
from the planned read-time budget.
Expert read-advice merge/alignment bytes and stage disk-margin bytes are strict
integers after CLI unit conversion, so direct Python callers cannot pass
booleans or floats into SSD read planning or free-disk admission.
`run-staged-routed-moe-batch` consumes that manifest and emits the first
execution boundary over staged unique slots. It compacts the staged slots into
a temporary expert layout containing only the batch's selected experts, maps
per-token original expert ids to compact ids, writes a compact routes JSON, and
calls the bounded `--run-moe-batch` runner once for the batch. When the stage
file already contains exactly those selected slots in compact order, the
compact layer is hardlinked instead of copied; the result reports
`compact_stage_storage` as `hardlink` or `copy`, and hardlink failure falls
back to the existing guarded copy path. `compact_stage_bytes` is still the
logical expert byte count consumed by the runner; the separate
`compact_stage_materialized_bytes` field records additional compact-file
storage written by this run and is zero for hardlinks. Compact-stage fallback
copy uses the same reusable-buffer positional read policy and rejects short
reads or short writes; copy, layout-write, and routes-write failures remove
partial compact artifacts before surfacing the error. Compact layout and route
JSON files are written via
temporary files and atomic replace, matching the stage-manifest policy. Stage
manifest parsing is strict at this boundary: layer ids, selected experts, stage
byte counts, coalesced range coverage, slot source/stage offsets, slot lengths,
token ids, and compact expert ids must be real integers and mutually
consistent; selected experts and token ids cannot repeat; route weights must be
finite. With compact routes JSON, the runner flattens routes into a small
assignment table and sorts by compact expert. With the `LLMSCAP1` binary
static-capacity route table, the
runner loads the expert-major assignment table directly and skips sorting when
the table is already monotonic. Prompt/generation paths write this binary table
without the debug JSON slot table by default, so large prompt chunks do not
construct a huge static-route JSON object in Python. It then reads each staged
expert slot once and
row-streams input plus output accumulation through a sparse f32 output file.
Within one expert, assignments are processed in bounded `--moe-token-block`
chunks by 4bit batch-row dequant, SwiGLU, and weighted-add kernels, so command
dispatch scales with expert chunks rather than individual assignments. In
`auto` mode, the runner chooses the largest token block that fits the scratch
cap and the largest routed assignment group for the current batch.
The first assignment for each token clears the one-row accumulator buffer
instead of reading a known-zero sparse output row; later assignments for the
same token reload the row. The assignment table, token-seen bitmap, and
token-block buffers are included in the runner scratch estimate before
allocation. The Python staged wrapper parses runner telemetry back into result
JSON, including the effective token block, max expert-token group, token-block
buffer bytes, and estimated runner peak. This cuts per-token process/setup
overhead and per-token staged-slot rereads while preserving explicit scratch
caps, compact-stage disk checks, and avoiding a full-batch activation
allocation. Expert staging manifests also include an `io_summary` with the
serial assignment read baseline, unique expert-slot lower bound, planned/staged
read bytes, optional SSD GiB/s read-time estimates and read-time cap result,
alignment waste, raw-to-coalesced range counts, savings ratios, read
amplification, and stage-budget utilization. `stage-batch-experts` can reject a
coalesced plan before opening the stage file when `--max-read-seconds` is set
and the planned read seconds exceed the cap at the measured `--ssd-read-gib-s`.
Prompt prefill and generation forward their routed-read seconds budget into
that same staging boundary, giving the request-admission estimate a second
check against the concrete router JSON emitted for the current prompt chunk
before any expert stage file is copied. The forwarded cap is cumulative across
the full prompt; each staged MoE call is checked against the remaining read-time
budget after earlier chunks/layers have consumed their planned seconds.
The low-level staged MoE runner mirrors that summary and read-advice counters
into its result/CLI JSON, so standalone SSD staging runs expose the same
read-amplification and read-time envelope as prompt-prefill and benchmark
summaries. When present, the mirrored stage-copy elapsed seconds and throughput
make it possible to compare M5-class SSD staging behavior against the configured
prefill SSD speed without widening the memory envelope.
Token generation also promotes those prompt-prefill staging counters into the
top-level `prefill_actual_read_time` payload, including serial-vs-unique bytes,
waste, coalesced savings, read-advice counters, read amplification, and stage
budget utilization. This keeps server/chat smokes auditable without reopening
the per-layer stage manifests.
For process-boundary experiments, staged MoE also has an explicit
materialized-job plan layer. `largerlm.staged_routed_moe_batch_plan.v1` lists
compact layouts, route tables, input/output f32 files, and per-job scratch/slot
caps; `metal/largerlm-runner --run-moe-batch-plan --batch-plan-json PATH`
executes those jobs sequentially inside one runner process. The JSONL server
variant, `--run-moe-batch-plan-server-jsonl`, leaves that process alive and
accepts one plan path per line, which lets Python submit per-layer plans in
dependency order without recreating the Metal process. The plan does not stage
new expert bytes or change route semantics by itself, so it can be gated
separately. Low-level staged MoE, tiled staged MoE, the staged routed MLP
wrapper, prompt prefill, generation, inspection, and serving can all hold an
optional session. `--prefill-persistent-moe-plan-server` keeps one MoE runner
alive across routed layers and expert-stage tiles; the raw `prefill-prompt`
command exposes the same behavior as `--persistent-moe-plan-server`. MoE output
accumulation is also profile-addressable: low-level commands accept
`--moe-output-accumulator`, while prepared generation, inspection, and serving
accept `--prefill-moe-output-accumulator {env,file,memory}`. `env` preserves the
legacy `LARGERLM_MOE_BATCH_ACCUMULATOR` override; `file` and `memory` pin the
child runner mode for launch-profile/audit replay. The resident-linear side now
has the same low-level server shape through
`--run-resident-linear-batch-plan-server-jsonl` and
`ResidentBatchLinearServerSession`. Raw `prefill-prompt` exposes it as
`--persistent-resident-linear-server`; prepared generation, inspection, and
serving expose it as `--prefill-persistent-resident-linear-server`, and launch
profiles can carry the flag. The old subprocess path remains the default
locked baseline while the resident-linear server is validated under the
`runner_process_fusion` target. The 128-token GLM-5.2 tiled locked-profile replay
`smoke-prefill-128tok-persistent-moe-server-tiled-profile-replay-result.json`
submitted 75 MoE plans through one runner, recorded zero routed MoE runner
launches, preserved the 17.37 GiB live-memory envelope, generated the same
token id 15, and completed in 59.688s versus the prior 63.739s baseline.
The newer strict memory-accumulator replay
`smoke-prefill-128tok-persistent-moe-server-tiled-memory-accumulator-strict-audit-result.json`
uses
`launch-profile-prefill-128tok-persistent-moe-server-tiled-memory-accumulator.json`
and its matching launch audit, reports `audit_bound=True` and
`files_ready=True`, generated token id 15, stayed inside the same 17.37 GiB
live cap with about 78.48 GiB available at admission, and completed in 60.122s.
The resident-linear server opt-in has also cleared a real locked GLM-5.2 replay:
`smoke-prefill-128tok-persistent-moe-resident-linear-server-tiled-memory-accumulator-strict-audit-result.json`
uses one 128-token chunk, generated token id 15 with
`persistent_linear_server=yes`, a replay-ready launch binding, and the same
17.37 GiB live cap. It is not promoted yet: when the old baseline was rerun in
the same hot-system window it took 196.818s, while the resident-linear server
candidate took 196.051s, a `total_elapsed_within_two_percent` tie.
Attention projection fusion follows the same opt-in pattern with a narrower
server: `--run-attn-projections-server-jsonl` reuses one Metal process for
prompt-prefill fused projection requests, `AttentionProjectionsServerSession`
drives it from Python, raw `prefill-prompt` exposes
`--persistent-attention-projection-server`, and prepared generation,
inspection, serving, plus launch profiles expose
`--prefill-persistent-attention-projection-server`. This server intentionally
rejects cache layout/file/position fields; decode cache append remains on the
one-shot `--run-attn-projections` path until a separate server contract is
validated. The first locked 128-token GLM-5.2 candidate used profile SHA
`3d4dd88308d0171249a660ae9e06f446daa3308b36d7baf713f42fddd46680a8`,
generated token id 15, and kept the 17.37 GiB live cap. It proved the process
fusion shape by reducing attention-projection unique runner groups to one, but
it did not win total latency: the candidate took 200.255s, while the old
same-window memory baseline took 196.818s and the resident-linear-only
candidate took 196.051s. The path therefore remains a replay-ready candidate,
not a selected default.
RoPE split process fusion is the next bounded prototype:
`--run-rope-split-batch-server-jsonl` keeps one Metal process alive for the
existing fused q_b split plus RoPE kernel. `RopeSplitBatchServerSession` drives
it from Python, raw `prefill-prompt` exposes `--persistent-rope-split-server`,
and prepared generation, inspection, serving, plus launch profiles expose
`--prefill-persistent-rope-split-server`. It is deliberately narrower than a
full attention-block server: q_b/k inputs remain file-backed, Python still
checks expected byte counts and output sizes, and the Metal side enforces the
same scratch limit as `--run-rope-split-batch`. The tiny Metal smoke covers
one-shot versus JSONL server output equivalence. The first same-window locked
128-token GLM bakeoff selected the RoPE split server candidate:
`114.796s` versus `148.846s` for the no-RoPE-server rerun, with matching
generated token id 15, a bound launch audit, and replay files ready. This makes
it the 128-token process-fusion baseline; 512/2048-token replay evidence is
still required before treating it as a long-context default.
Its summary confirms `accum=memory` for routed MoE layers, routed MoE elapsed
11.481s, and expert stage copy 1.980s at 5.744 GiB/s.
Resident batch RMSNorm now has the same narrow JSONL process-fusion shape.
`--run-rmsnorm-batch-server-jsonl` keeps one Metal runner alive for sequential
resident RMSNorm batch requests, while preserving the one-shot command's
resident-layout lookup, layer/suffix resolution, input/output byte checks,
RMSNorm epsilon validation, and scratch-cap accounting. Python drives it with
`ResidentBatchRMSNormServerSession`; raw `prefill-prompt` exposes
`--persistent-rmsnorm-server`, and prepared generation, inspection, serving,
plus launch profiles expose `--prefill-persistent-rmsnorm-server`. This server
is intentionally smaller than a full layer server: attention and MLP orchestration
remain in Python, and the server only replaces repeated subprocess launches for
`input_layernorm`, q/kv latent norms, and `post_attention_layernorm`. The tiny
Metal smoke confirms one-shot and JSONL server numerical equivalence. The first
locked GLM bakeoff rejected promotion: the 128-token request was stopped by the
safety-capped admission limit of 126 prompt-prefill tokens, and the 126-token
candidate generated the same `[15]` token as baseline but ran slower
(`220.614s` versus `211.945s`) while staying inside the same 17.37 GiB live cap.
Standalone RMSNorm is available for experiments but should not be enabled by
default.
MLA attention now has a wider JSONL process-fusion boundary than the standalone
RMSNorm server. `--run-mla-attention-batch-server-jsonl` keeps one Metal runner
alive for sequential prompt MLA attention batches and reuses the same
`run_mla_attention_batch` / `run_mla_attention_indexed_batch` implementations as
the one-shot commands. The server protocol preserves resident-layout,
decode-cache, q_nope/q_rope, output-size, RoPE, cache-read, resident-matrix, and
runner-scratch validation; Python drives it with
`MLAAttentionBatchServerSession`. Raw `prefill-prompt` exposes
`--persistent-mla-attention-server`, and prepared generation, inspection,
serving, plus launch profiles expose
`--prefill-persistent-mla-attention-server`. The tiny Metal smoke now validates
one-shot versus JSONL server equivalence for both default and interleaved RoPE.
The first locked GLM bakeoff selected this boundary for the 126-token
custom-metal fallback envelope: the candidate generated `[15]`, stayed inside
the 17.37 GiB live cap, finished in `132.394s`, and beat the previous RoPE split
baseline at `211.945s` (`0.625x`, `-79.551s`). The selected replay is now
`selected-replay-prefill-126-persistent-moe-resident-linear-attnproj-rope-split-mla-server-memory-accumulator.json`,
with launch profile SHA
`f7f1cbd90c2ce28c191607165431b925ca1dbf043511f8a9f2165b6578d7a172` and audit
SHA `d7bcb0a7e555ed2d7ac1614b68d8e4275f8219175e7076453107f6c6dcaef302`.
Because this profile is an explicit custom-metal fallback, the launch audit uses
`--allow-non-accelerated-prefill-launch-audit`; replay still requires the normal
memory, temp-disk, SSD, locked-profile, launch-audit, and GLM-shape checks.
Shared-expert batch process fusion is available as the next opt-in JSONL
boundary but is not part of the selected profile. The Metal runner command
`--run-shared-expert-batch-server-jsonl` preserves the one-shot
`--run-shared-expert-batch` contract for resident layout, layer, input/output
files, batch-token count, resident-matrix cap, and runner-scratch cap, while
serving one JSONL request per line until `{"command":"quit"}`. Python drives it
with `ResidentSharedExpertBatchServerSession`; raw `prefill-prompt` exposes
`--persistent-shared-expert-server`, and prepared generation, inspection,
serving, plus launch profiles expose
`--prefill-persistent-shared-expert-server`. The tiny Metal smoke validates
one-shot versus server byte-equivalence. The locked 126-token GLM candidate used
profile SHA `14e98a7445957aa0441f3afac1dfb7401d28fb3d1f0911509f2336ea3d981767`,
generated `[15]`, stayed inside the same 17.37 GiB live cap, and reduced
`mlp.shared_experts` time from 8.503s to 2.839s. End-to-end latency remained a
tie/slight regression, 133.308s versus the selected MLA baseline at 132.394s,
so `prefill-126-shared-expert-server-vs-mla-attention-server-bakeoff-current.json`
retains the baseline. A same-window rerun strengthened that negative result:
the baseline replay finished in 141.647s, the shared-server replay in 194.353s,
and `prefill-126-shared-expert-server-rerun-vs-baseline-bakeoff-current.json`
again retained the baseline. The shared-expert server remains useful plumbing
and a local tensor-time win, but it is not a selected process-fusion boundary.
Separately, the earlier 128-token memory-accumulator strict result is baked into
`selected-replay-prefill-128-persistent-moe-memory-accumulator.json` and
`selected-replay-prefill-128-persistent-moe-memory-accumulator.sh`. Because the
launch profile itself pins `--prefill-moe-output-accumulator memory`, the
selected replay has an empty `required_environment` map instead of relying on
the legacy accumulator environment variable. The smoke wrapper
`smoke-prefill-128-persistent-moe-memory-accumulator-safe.sh` uses
`selected-replay-run --check-ssd-read-speed --write-result`, so replay always
passes the no-weight memory, temp-disk, SSD, profile-hash, and launch-audit
checks before loading GLM weights.
The matching localhost wrapper
`serve-prefill-128-memory-accumulator-safe.sh` now uses the same profile,
launch audit, and selected replay gate before starting `serve-prepared`; it no
longer depends on `LARGERLM_MOE_BATCH_ACCUMULATOR`.
`server-http-prefill128-persistent-memory-accumulator-smoke-safe.sh` is the
matching reproducible HTTP smoke: it starts the server wrapper, waits for
`/health`, sends a 128-token `/generate-token-ids` request, saves the raw
response, and stops the server. The latest response
`server-http-prefill128-persistent-memory-accumulator-smoke-latest.json`
generated token id 15 in 60.295s server elapsed, reports 60.188s prompt
prefill and `prefill_moe_output_accumulator=memory`, and remains
workload-comparable with the offline strict replay at 1.003x total elapsed.
The latest bounded follow-ups,
`glm-moe-layer19-optimization-target-128tok.json`,
`glm-mla-layer37-cache-sweep-128tok.json`, and
`glm-attn-proj-layer37-fusion-sweep-128tok.json`, all report
`candidate_for_full_replay=false`; the current tile1 MoE kernel, key+value MLA
cache, and fused attention-projection path remain selected.
`prefill-staged-routed-mlp-block-batch` composes the staged boundary into a
prompt-chunk MLP path: batch post-attention RMSNorm, single-process
`--run-router-batch` JSON emission, bounded stage materialization, staged
routed MoE, and streaming residual add. When `--include-shared-expert` is set,
shared gate/up/down projections run
through bounded resident batch matvecs and streaming SwiGLU before the final
residual add, so routed and shared memory caps stay separate.
With `--expert-stage-tiling`, the same wrapper replaces the single routed stage
with the tiled staged MoE executor. Each expert tile has its own stage file,
compact-stage file, route table, and runner invocation; tile outputs are
scatter-added into one routed MLP output before shared-expert and residual work
continue. The returned layer record aggregates staged bytes, compact-stage
bytes, static-capacity slots, route-table bytes, copy telemetry, runner calls,
and peak estimates across all tiles while preserving the complete per-tile
records under `tiled_staged_moe`.
`prefill-prompt` is the first chunked multi-layer prompt driver built from
those pieces. It streams token embedding rows into `[chunk, hidden]` f32 files,
runs each chunk through the full selected layer stack, writes that layer's MLA
cache rows before causal batch attention, and extracts only the final prompt
hidden row for the logits boundary. Intermediate hidden states stay on disk only
for the active chunk unless `--keep-work-dir` is set, and the driver keeps
separate caps for prompt batch bytes, resident matrices, cache read/write bytes,
runner scratch, staged expert files, and compact-stage files. Stage and
compact-stage writes check free disk before copying. Each chunk also budgets its
main f32 work files before embedding is written, and generation can reserve an
extra work-directory margin with `--prefill-stage-disk-margin-mib`.
Direct prompt prefill forwards `--expert-stage-tiling` into routed layers and
aggregates prompt-level read bytes, range counts, read-advice counters,
routed-assignment counts, static-capacity totals, MoE block telemetry, and
custom-metal elapsed time over every tile rather than over only the first tile.
Prepared generation, benchmark, inspect, and server paths expose the paired
`--prefill-expert-stage-tiling` flag. Their request checks keep the original
full-layer stage estimates for diagnostics, but admission and temp-disk checks
use the effective maximum per-tile stage/compact footprint when tiling is
enabled. Request profiles also carry non-default cache read/write caps that
influence auto chunk sizing. This matters for large GLM-5.2 checks: a profile
that selects a 448-token tiled prefill chunk must replay its
`--max-cache-read-mib 1024` cap, otherwise the default cache-read cap lowers the
current max-safe chunk and the locked profile is rejected before generation.
Automatically-created temporary prompt/generation work directories are removed
on failure unless `--keep-work-dir` or an explicit work directory asks to
preserve artifacts for debugging.
MoE layers forward `--moe-token-block auto` to the staged runner by default, so
the runner picks the largest expert-local token block that fits
`--max-runner-scratch-mib`. `--static-capacity-per-expert auto` expands to the
current chunk's token count, which is a conservative no-overflow cap because an
expert cannot receive more assignments than there are tokens in that prompt
chunk. A fixed strict capacity below the prompt chunk size is rejected before
prompt/generation work files are created unless overflow is explicitly enabled
for analysis. The prompt result keeps top-level max telemetry for effective MoE
token block, max tokens per expert, MoE batch buffer bytes, MoE runner peak
bytes, static-capacity route-table slots/bytes, and staged expert I/O totals;
generation exposes the same fields under `prompt_prefill` when
`--batch-prefill-prompt` is enabled. When generation resolves an automatic
prompt chunk, top-level `auto_prefill_prompt_chunk_plan` carries the actual
cap-table, limiting-cap, and next-token scratch evidence used for the run.
Top-level `max_safe_prefill_prompt_chunk_plan` is emitted for any batch-prefill
generation, including fixed launch-profile chunks, so replays can compare the
chosen chunk against the current safety-capped maximum.
Prompt-prefill results also carry actual
`prefill_acceleration_coverage` and a `prompt_prefill_actual`
`prefill_acceleration_frontier`, so CLI JSON and server responses can confirm
whether the real run reached MPSGraph instead of relying only on preflight
inspection. The actual acceleration gate can also require a minimum accelerated
FLOP fraction, matching the request-side
`prefill_min_accelerated_flop_fraction` setting. Those payloads include
`linear_backend_flops` and scalar
`total_linear_estimated_flops`, `accelerated_linear_estimated_flops`,
`custom_linear_estimated_flops`, `unsupported_linear_estimated_flops`, and
`accelerated_linear_flop_fraction` fields for comparing matrix-count coverage
with the FLOPs-weighted view.
Prepared token generation mirrors that evidence at the token-result level as
`prefill_actual_acceleration_coverage`,
`prefill_actual_acceleration_frontier`, and
`prefill_actual_linear_backend`, all sourced from the actual prompt-prefill run.
This keeps standalone `--write-result` smokes and benchmark/audit artifacts on
the same evidence shape without scraping nested prompt-prefill payloads.
`largerlm result-summary` consumes the same top-level actual evidence, reporting
accelerated FLOP fraction, accelerated/total FLOPs, and streamed routed-expert
FLOPs directly in the summary output. The same summary path unwraps saved HTTP
smoke artifacts before analysis, including both OpenAI-style
`response.largerlm.token_result` payloads and older direct-token-result
`response` payloads, so local API smokes can be compared against offline replay
baselines without a manual extraction step. It also aggregates
`routed_moe_elapsed_seconds` from staged prompt-prefill layers into a routed-MoE
elapsed/TFLOP/s/custom-elapsed-share line and emits the slowest routed-MoE
layers with assignments, selected expert count, materialized stage bytes,
compact-stage storage mode, effective token block, and static-capacity usage.
When `LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING=1` was used, it also reports the
MXFP4 split between gate/up/SwiGLU and down/weighted-add kernels, both as an
aggregate hint and in each slow-layer timing row.
When one split side dominates, the bottleneck hints include a bounded
`glm_moe_tile_sweep.py` experiment command for the hottest routed layer,
including selected expert ids when the result artifact preserved them. These
commands are microbench leads only; promotion still requires full replay and
result-bakeoff evidence. Sweep artifacts use the
`largerlm.glm_moe_tile_sweep.v2` schema and add `config_comparison`, which
marks a candidate only after it clears the configured kernel-speedup and output
drift thresholds with enough timing samples, then keeps it gated behind full
replay/bakeoff.
This gives M5/MPP routed-expert work a stable per-layer optimization target
instead of burying it in custom-Metal totals. The summary also emits routed-MoE
bottleneck hints, including whether the effective token block already saturates
per-expert fanout, how large expert stage copy time is relative to routed runner
time, and whether materialized stage+compact bytes are concentrated in a single
layer or spread across many layers. When stage copy is a large share of the
routed-MoE path and measured throughput is low with small copy chunks, the
summary suggests bounded `--prefill-copy-chunk-mib` experiments for the next
guarded replay. The same summary now reports recorded runner command groups
using both raw record count and deduplicated command count. For the current
score-cache MLA 17-token smoke, that exposes 2103 command records but 1713
unique command signatures: 855 resident-linear calls, 390 RMSNorm calls, 156
MLA attention calls, and 78 calls each for attention projections, RoPE batch,
attention output, and singleton RoPE. That makes the next performance target
explicit: reduce Python/runner round trips by fusing
attention/projection/RMSNorm/layer blocks, not merely retune staged-expert copy
chunks.
Non-default prompt-prefill copy chunks are now emitted as a
`prefill_copy_policy_flags` launch-profile section, so successful experiments
such as `--prefill-copy-chunk-mib 64` can be locked, audited, replayed, and
compared like the other request-profile guard flags.
Direct prompt prefill uses the same top-level live working-set guard as
generation, checked before creating prompt work files. The prefill estimate is
phase-aware: it takes the maximum of prompt-batch bytes, cache-read bytes,
cache-write bytes, and expert-stage copy bytes, each paired with runner scratch,
instead of only counting the embedding batch. When a minimum free
unified-memory reserve is configured, direct prompt prefill rechecks that
live-memory guard before each prompt chunk. Generation passes the same caps
into batch prompt prefill and rechecks the guard after prefill or decode-layer
work before final logits, keeping the next large allocation behind a fresh
memory-pressure check. Its memory, cache, scratch, stage, and disk-margin caps
must be finite positive or non-negative values before any
prompt work directory is created. Generation rechecks the same guard before
prompt/decode stages so long requests can stop if system headroom drops after
admission.
`prefill-prompt --prompt-chunk-tokens auto` and generation's
`--prefill-prompt-chunk-tokens auto` use the same chunk-sizer, including that
single-chunk disk budget in the chosen chunk size. The scheduling is
chunk-major: chunk 0 completes all layers before chunk 1 starts, so later chunks
can safely attend to every earlier cache row at each layer without loading the
full prompt into unified memory.
`generate-token-ids --batch-prefill-prompt` now uses that driver for the prompt
phase, computes logits from the final prompt hidden row to select the first
generated token, and then resumes the existing single-token decode loop for
new tokens. `--prefill-moe-token-block` and
`--prefill-static-capacity-per-expert` override only that prompt MoE phase; the
default generation path remains unchanged for debugging and comparisons.

`dsa-indexer-batch` establishes the first GLM DSA correctness boundary. For a
full-indexer layer it streams hidden rows through `indexer.wk`, applies the
indexer LayerNorm and half-split RoPE, writes the result into that layer's
`dsa_index` cache segment, then computes causal top-k token ids from
`indexer.wq_b(q_resid)`, cached index keys, and `indexer.weights_proj(hidden)`.
It can also write a compact row-major u32 file with `count + padded indices`.
The optional JSON top-k artifact and compact u32 sidecar are published through
temporary files and atomic replace so a failed publish does not overwrite the
previous artifacts.
`prefill-mla-attention-batch --indices-u32 --index-topk` consumes that file via
the indexed Metal MLA path: the host stages only selected cache rows into a
bounded F32 buffer, and the kernel computes softmax/value over those selected
positions while still using the original token positions for RoPE. Shared-indexer
layers reuse the most recent full layer's top-k indices inside
`prefill-prompt`: each prompt chunk resets that state, full layers write a
compact `dsa_topk.u32`, and shared layers pass that file to indexed MLA without
recomputing the indexer. `generate-token-ids --batch-prefill-prompt` forwards
the same DSA schedule and `indexer_rope_interleave` mode derived from the GLM
config. The single-token `decode-layers` loop also consumes that schedule: full
layers switch to the composed attention block to update `dsa_index` and produce
a one-row top-k u32 file, while shared layers reuse the most recent full layer's
file for indexed MLA before continuing through the regular MLP boundary.
The DSA decode guard budgets full layers as `dsa_index` scan bytes plus selected
MLA cache rows, and shared layers as selected MLA cache rows only, so the safety
check follows the bounded indexed path instead of assuming a full MLA prefix
read.

The composed attention smoke chains the current pieces across process
boundaries: projection, cache append, Q split, current-Q RoPE, MLA score/value,
output projection, and residual add. It is intentionally still a harness rather
than a fused runtime command, but it verifies the current tensor interfaces.

The decoder-layer smoke feeds that attention output into `--run-mlp-block`,
covering a tiny full current-token layer path across attention and the
routed/shared MLP. This is still a correctness harness assembled from bounded
commands, not the final decode scheduler.

The `--run-decoder-layer` wrapper moves that same sequence behind one runner
command: attention projection/cache append, Q split, current-Q RoPE, MLA
score/value, output projection, post-attention RMSNorm, routed/shared MoE, and
residual add. `--run-dense-decoder-layer` shares the same attention path but
finishes with resident dense gate/up/down MLP matrices for GLM-5.2's initial
dense layers. Both commands intentionally reuse the existing bounded primitives
and a private work directory for intermediate f32 tensors. That reduces
command-line orchestration without pretending to be the final fused in-memory
decode loop.

The Python `decode-layers` driver is the first multi-layer scheduler. It reads
the selected packed layer IDs, preflights each layer with the decoder-layer
runtime budget, then launches `--run-decoder-layer` or
`--run-dense-decoder-layer` sequentially with f32 hidden-state files between
layers. With `--model-config`, it derives the GLM MLA dimensions, top-k,
RMSNorm epsilon, RoPE theta, router score, shared-expert inclusion, and
`mlp_layer_types` dense/MoE split from the Hugging Face config unless a CLI flag
overrides them. Config loading normalizes known dense/sparse aliases and rejects
unknown `mlp_layer_types` values before they can misclassify a layer. It also
rejects non-positive required dimensions, expert counts, top-k counts,
RoPE/RMSNorm scalars, and contradictory relationships such as
`num_experts_per_tok > n_routed_experts` or `index_topk` above the configured
context window. Config integer, float, and boolean scalars are type-strict,
including GLM-5.2 DSA schedule freq/offset fields, so booleans and floats
cannot be truncated into model shape metadata. Full/shared DSA schedules must
also include `index_head_dim`, `index_n_heads`, `index_topk`, and `q_lora_rank`,
so cache and indexer planning cannot silently use a zero-width placeholder. The
driver also validates memory, cache, scratch, and
expert-read-advise caps before creating its private work directory, so malformed
limits cannot weaken per-layer runtime preflight. Automatically-created
decode-layer work directories are removed on failure unless `--keep-work-dir`
or an explicit work directory asks to preserve them for debugging. This is still
below the full model API boundary: tokenization,
embeddings, sampling, and a server loop remain separate milestones.

The `final-logits` command covers the next boundary after decoder layers:
final RMSNorm followed by `lm_head.weight` or tied embedding top-k. It streams
the head matrix in bounded row chunks from `resident.bin` instead of loading the
whole vocab-by-hidden matrix into unified memory. Runtime preflight and prepared
readiness disable that tied-embedding fallback when the model config declares
`tie_word_embeddings=false`, so GLM packages with separate output heads must
carry `lm_head.weight` in the resident layout. The default implementation is a
safety-first Python path. Passing `--runner metal/largerlm-runner` uses a
chunked Metal top-k path: final RMSNorm and each `lm_head` chunk matvec run on
Metal, while the CPU keeps only a small top-k heap. Full-vocab logits output
remains on the Python path for now. The top-k JSON output and optional
full-vocab logits file are published through temporary files and atomic replace.
When generation has a model config, runtime preflight and final-logits calls
also compare the resident head/embedding row count with `vocab_size` and the
head hidden dimension with `hidden_size` before selecting a token. The
embedding readers receive the same config-derived expectations, so prompt
prefill and decode reject a stale resident embedding layout before streaming
token rows from disk. Runtime preflight also requires the resident embedding
tensor to be present and checks its row size against the request's
`max_embedding_row_mib`; prepared request inspection reports the resulting
embedding row/output bytes next to layer, cache, logits, and live-memory
budgets.

The `embed-token` command provides the matching single-token input boundary by
streaming one `embed_tokens.weight` row from `resident.bin` into an f32 hidden
vector. `embed-tokens-batch` uses the same bounded row reads for prompt chunks,
writing a `[tokens, hidden]` f32 batch without loading the embedding matrix.
The current tiny loop is therefore token id -> embedding row -> packed decoder
layers -> final logits top-k, while prompt prefill now has a token-id batch ->
hidden batch entry point.

The `generate-token-ids` command wraps that loop into a greedy or top-k sampled
token-id generator. It consumes prompt token ids one at a time, advances cache
positions, checks the requested prompt+generation length against the cache
layout context budget, emits selected next-token ids, and reports estimated
embedding, routed-expert, cache, and logits read bytes separately from the total
read count. If a direct layout
entry point also receives `--model-config`, the CLI verifies that any packed
layout `config_sha256` still matches that config before deriving GLM dimensions,
router metadata, or DSA schedules. Runtime backing validation also rejects
expert and resident layouts that both record different config hashes or
overlapping byte spans. Before creating the per-token work directory or
launching the runner, its default runtime preflight
checks the decode cache layout/file size, resident/expert backing file sizes,
resident tensor span/shape metadata, worst requested context length, selected
layer scratch and cache-read budgets, and final-logits chunk peak memory. The
Metal runner mirrors the resident backing checks at the executable boundary:
resident `weight_file` paths, `total_bytes`, actual file size, tensor
dtype/shape/size consistency, tensor offsets, and overlapping spans are
validated before direct router, resident-matrix, or attention reads. The runner
also strictly parses direct final-logits, decode, MLA, RoPE, router,
attention-projection, dense/resident linear, resident batch, and RMSNorm numeric
caps before Metal setup so malformed suffixes or non-finite eps/chunk/theta
values cannot be coerced into unsafe budgets. Decode-facing MiB caps accept
finite decimal values and convert them to bytes directly, avoiding prefix
integer truncation. It also mirrors the cache backing contract at the executable
boundary:
`--validate-cache-backing` and the
MLA/cache read-write helpers require layout version 1, strict non-boolean
integer fields, segment stride/total/span consistency, unique
`kind:layer` segments, and an actual cache file whose logical size equals
layout `total_bytes`. Direct MLA, attention-projection, and decoder-layer
commands run that backing check before creating a Metal device or doing
projection work, so stale cache files cannot hide behind later GPU setup
errors. If the decode cache layout contains DSA `dsa_index` segments, generation
requires a DSA schedule from `indexer_types`; otherwise it refuses to run unless
`--allow-missing-dsa-indexer` is set for explicit debugging. The guard rejects
unknown DSA schedule values, verifies full-indexer layers have `dsa_index` cache
segments, and can derive `index_head_dim` from the cache layout. Config loading
normalizes DSA schedule aliases and rejects `indexer_types` or
`index_topk_pattern` schedules whose length differs from `num_hidden_layers` or
whose values are unknown, and also rejects shared-indexer layers before any full
indexer. Runtime admission repeats that full-before-shared check for the
selected layer walk, so a layer subset cannot start from a shared DSA layer
without the full layer that produces its top-k file. This keeps planner/cache
estimates from silently underbudgeting full-indexer cache rows. The planner
budgets full-indexer `wk`, `wq_b`, and `weights_proj` as separate resident
GEMMs; `wq_b` output width is `index_n_heads * index_head_dim`, while the cache
segment remains `index_head_dim` wide. GLM preflight validates those full-indexer
resident tensor shapes, and the main q_a/q_b/kv_a/kv_b/o_proj attention shapes,
before packing. It also checks dense MLP, shared expert, router gate, and router
correction-bias resident shapes against the config, so malformed GLM weights
fail before runtime cache/index work begins. Batch prompt
prefill accepts automatic chunk sizing: the chosen token count is bounded by the
prompt f32 batch cap, MLA/DSA cache read/write caps, resident matrix scratch
headroom for the selected resident linear backend, worst-case routed expert
stage/compact-stage caps, and the prompt length before the chunked driver
starts. Direct `prefill-prompt` sizing also includes any nonzero
`--start-position` in the cache-read context bound, so continuing an existing
cache prefix cannot underbudget DSA/MLA reads. The same direct path rejects
`start_position + prompt_length` beyond the decode-cache context before creating
work files or appending cache rows, and rechecks the live-memory guard before
each selected layer inside a chunk so a long GLM prompt does not run all
remaining layers after available memory falls below the configured reserve. For
MPSGraph BF16/F16 resident matrices,
that scratch estimate includes
the bounded F32 conversion buffer. The work-directory disk cap also includes a
worst-case routed stage plus compact-stage increment per selected routed layer,
so auto sizing does not ignore stage files that live until the prompt chunk is
cleaned up. Auto sizing rejects malformed resident layouts rather than using a
fixed fallback, because the resident embedding and matrix shapes define both the
hidden dimension and scratch cap; malformed decode-cache layouts are rejected
for the same reason instead of skipping cache read/write caps. Direct
`prefill-prompt --json` exposes the auto chunk plan's cap table, limiting cap
names, resolved-chunk scratch, and next-token scratch estimate, making MPSGraph
scratch cliffs auditable without increasing the selected chunk. Prompt prefill records
`linear_backend_counts` so JSON output exposes the actual custom Metal versus
MPSGraph resident GEMM mix, and aggregates resident linear matrix scratch,
logical F32 matrix bytes, and raw conversion bytes across the prompt. It also
records routed assignment counts, unique expert slots, and max tokens per expert per chunk, which are the capacity
signals needed to replace `prefill-plan`'s static lower/upper capacity bounds
with calibrated fixed-shape ANE/NPU expert batches.
At the per-batch level, `plan_static_expert_capacity` converts a
`BatchExpertIOPlan` into fixed `[expert, capacity]` token slots plus explicit
overflow assignments. This keeps future static expert kernels honest: a
balanced capacity can be tested without silently dropping the skewed tail.
`plan-batch-expert-io --static-capacity-output-json` writes the same contract as
a padded JSON artifact: every selected expert has exactly `capacity` slots, empty
slots are marked inactive, and overflow remains separate. Artifact writes reject
overflow by default, with an explicit analysis-only override.
`run-staged-routed-moe-batch` and the staged routed prefill wrapper can emit the
same artifact after compact staging. In that path, the artifact uses compact
expert ids aligned with `compact_layout.json`; the compact layout remains the
source of truth for mapping those ids back to original GLM expert ids.
The companion `.bin` artifact starts with the `LLMSCAP1` little-endian header,
then selected expert ids, then an expert-major fixed slot table of
`token_index:u32, weight:f32, active:u32` records, followed by overflow records.
The Metal runner consumes this table through `--routes-bin` without rebuilding
per-token JSON routes. No-overflow static tables are already expert-major and
pre-sorted; overflow or malformed ordering can still fall back to sorting before
execution. After writing the binary, the Python staging path validates the
header, file length, sorted expert table, active slot flags, token bounds,
finite weights, overflow records, and plan-matching counts before dispatching
the runner; validation failures remove the generated static route artifacts.
The same checker is exposed as `validate-static-capacity-bin` for standalone
launch-script sanity checks.
Static-capacity JSON and binary writes use temporary files and atomic
replace, so failed artifact writes do not leave partial route tables. Future
static expert kernels can use the same file without JSON parsing, while the JSON
artifact remains available for inspection.
`generate-text` adds local tokenizer encode/decode over the same bounded loop.
Text-generation CLI prompt strings and prompt files are capped by
`--max-prompt-bytes` before tokenization, so a mistaken huge prompt file is
rejected before layout validation, work-directory setup, or runner launch.
`serve-prepared` exposes the prepared path through a small local JSON API while
preserving manifest validation, runtime preflight, live working-set admission,
batch prompt prefill defaults, and serialized access to the shared decode-cache
file. It inherits GLM router defaults and DSA indexer metadata from the
prepared model config, and request-level `"batch_prefill_prompt": false`
disables the text server's automatic batch-prefill switch for that request. It
also accepts `--prefill-linear-backend`, so a long-lived local service can pin
prefill resident GEMMs to `auto`, `mpsgraph-f32`, `mps-matrix-f32`, or
`custom-metal` without changing request payloads. The server exposes the same
key prompt-prefill
memory and disk caps for prompt chunking, prompt batch bytes, cache writes,
stage files, copy chunks, stage disk margin, and prompt MoE token block, keeping
low-memory service profiles explicit at process start. It
computes an effective context limit from the decode-cache layout, prepared
manifest, and `max_position_embeddings` from the model config, reports all three
sources plus the effective limit through `/health`, and rejects token-id, text,
completion, and chat requests whose prompt plus requested generation would
exceed that limit before entering token generation.
It
also exposes `GET /v1/models`, a non-streaming OpenAI-compatible
`/v1/completions` endpoint over the same text generation path, plus
`/v1/chat/completions` when the local tokenizer can render the model's real
chat template. Chat rendering first tries a tokenizer-provided renderer and can
fall back to a sandboxed local Jinja renderer for `chat_template.jinja` or a
`tokenizer_config.json` `chat_template`; the rendered prompt is still encoded by
the selected tokenizer backend before the same prepared request admission runs.
It intentionally
avoids a large application-owned weight cache; richer continuous batching,
streaming, tool calling, and prefix sharing remain a later scheduling layer.

The default serving profile prefers the OS page cache over a large
application-owned cache. The measured M5 Max 128 GB profile may consume a
`largerlm.expert_pin_plan.v1` with at most 10 GiB of hard residency and no
adaptive application tier. The 16 GiB Metal live cap includes those buffers,
and the runtime preserves at least 24 GiB of available unified memory.

Pinned experts live in stable shared `MTLBuffer` objects and are passed directly
to the MXFP4 kernels on route hits. Misses use the existing bounded parallel
`pread` path and remain eligible for the macOS page cache. The older adaptive
per-layer LRU remains implemented for experiments, but it is disabled in the
published M5 profile. Routing and weights are unchanged.

## Packed Expert Format

The first format should be shape-driven, not hard-coded to one model:

```text
experts/
  layout.json
  layer_000.bin
  layer_001.bin
  ...
```

`layout.json` records:

- model type and config hash
- quantization type, group size, scale/bias dtype
- number of layers and experts
- per-layer expert slot size
- component offsets for gate/up/down weight, scales, and biases

The packer has two input modes. It can copy already-quantized affine int4 expert
tensors, or it can quantize raw BF16/F16/F32 GLM expert weights into the same
affine int4 slot format. Existing affine-int4 slots are validated against the
config-derived GLM gate/up/down dimensions, group size, U32 packed-weight shape,
and 16-bit BF16/F16 scale/bias metadata. The raw quantizer validates that each
raw expert slice's dtype and logical shape match its byte count and the
config-derived GLM gate/up/down dimensions before deriving a packed slot layout.
Pre-quantized fused gate/up tensors such as `gate_up_proj.{weight,scales,biases}`
are split along the fused row dimension into the standard gate/up components
before bounded copies, so the runner-facing layout keeps the same component
order for separate and fused checkpoints.
Common MoE
component aliases `w1/w3/w2` are normalized to `gate_proj/up_proj/down_proj`,
and duplicate canonical components are rejected during discovery instead of
being silently overwritten. Resident dense/shared MLP tensors apply the same
one-to-one component alias normalization before layout construction while
keeping the original safetensors shard offsets for bounded streaming copies. Raw
fused `gate_up_proj.weight`, `gate_up.weight`, and `w13.weight` inputs are
accepted only when they contain exactly
`[2 * moe_hidden_size, hidden_size]` rows per expert; discovery exposes those
halves as separate gate/up slices so the runtime layout and dequant kernels keep
their existing nine-component affine-int4 contract. Per-expert tensor ids are
rejected when they fall
outside the config-declared expert range, so a GLM checkpoint/config mismatch
cannot silently omit an extra expert. The quantizer then streams each expert
component as row blocks with bounded `pread` calls and writes the generated
weight/scales/biases row blocks directly into the packed slot without converting
those generated buffers into duplicate immutable `bytes` objects. Non-finite raw
values fail as controlled packer errors, preserving the same partial-file cleanup
path as short reads or writes. Its modeled heap
adds only the largest generated quantized row-block output to the normal
safetensors metadata/copy-chunk allowance before checking
`--max-pack-heap-mib`, so a tight cap still fails before output files are
created without requiring a whole raw matrix or whole expert slot to fit in
heap.
Malformed safetensors headers are rejected during dry-run/preflight
and execution avoids a second chunk-join copy. It is currently a Python
correctness path; for full GLM-5-class checkpoints it should be
replaced or assisted by Metal/Accelerate kernels before serious end-to-end
packing runs. Packed expert and resident layout JSON files are published through
temporary files and atomic replace; on a failed forced publish, the previous
layout metadata remains intact.

Resident packing is config-aware. Tensors that match routed expert naming are
excluded only for layers that the config marks as MoE; dense-prefix MLP matrices
remain resident even if the checkpoint uses a `switch_mlp` style name.
Preflight uses the same classification, so resident bytes and page-cache budgets
include dense-prefix MLP weights while expert packing ignores those dense
tensors. Dense/shared resident MLP component aliases `w1/w3/w2` are normalized
to `gate_proj/up_proj/down_proj`, and raw fused dense/shared gate-up resident
tensors are normalized into separate gate/up tensor layout entries during
resident packing when their config-derived shape matches. This preserves
compatibility with checkpoints that use alias or fused storage while keeping
the runner and readiness contract on independent `gate_proj.weight`,
`up_proj.weight`, and `down_proj.weight` resident tensors. Executed prepare
manifests and prepared health preserve the component-alias source/renamed
counts and bytes alongside the fused source tensor count, expanded tensor
count, and expanded byte total for resident alias expansion. Tensors from
layers at or beyond `num_hidden_layers`, such as GLM-5.2
MTP layer-78 indexer artifacts, are categorized as ignored extra-layer tensors
and are not copied into the runtime resident blob.

Each `layer_N.bin` stores all routed experts for that layer in fixed-size slots.
Fixed-size slots make expert offset calculation trivial:

```text
offset = expert_id * layer_layout.expert_slot_bytes
```

If future checkpoints use different expert shapes per layer, `layout.json` can
record a per-layer slot size while preserving fixed expert offsets inside each
layer.

## Cache Policy

Use three tiers:

- Resident: non-expert weights, router/indexer weights, embeddings, lm head,
  small lookup tables, and current KV/DSA cache.
- OS page cache: packed expert files. Leave as much free memory as possible.
- Scratch: 2 MB aligned expert read buffers, double-buffered only for measured
  prediction experiments.

Config-only planning must still reserve space for resident weights. LargerLM
therefore estimates GLM resident attention, dense-prefix MLP, shared expert,
router, DSA indexer, embedding, and lm-head bytes from `config.json` when no
safetensors scan is available. That estimate uses a typed, validated
`dtype`/`torch_dtype` byte width rather than defaulting unknown values to bf16;
`--scan-safetensors` replaces the estimate with exact shard-header byte counts.
The scan uses the same MoE/runtime-layer classification as `preflight-glm`, so
GLM MTP or extra-layer tensors do not inflate resident or routed expert budgets.

Avoid:

- Long-lived Metal buffers for many cached experts.
- `mmap` as the cold expert read path.
- Read-ahead hints until profiling proves a specific GLM routing pattern
  benefits from them.

## Safety Defaults

All large-file tooling must default to non-destructive dry-runs. The expert
packer follows these rules:

- `checkpoint-status` is the first no-payload-read gate for real checkpoint
  artifacts. It reads only `config.json`, `model.safetensors.index.json`, the
  optional metadata-only `largerlm.safetensors.headers.json`, and shard
  `stat()` results, then reports expected/present/complete/missing/partial shard
  counts. When source repo metadata is known, it emits replayable `hf download`,
  header-fetch, no-weight prefill-backend probe, preflight, and prepare dry-run
  commands. It also computes the
  remaining expected shard bytes from the header manifest and checks target
  volume free space against a configurable download margin. Optional JSON and
  URL-list outputs capture the missing/truncated shard set for external download
  machines, and `--require-download-disk-ok` turns the free-space calculation
  into a hard exit gate for scripts; `download_precheck_command` exposes that
  replayable invocation in JSON and text reports. `--verify-local-headers` is
  the post-copy guard for those external downloads: it opens only safetensors
  headers, compares local data-start/file sizes and manifest headers, and runs
  the same tensor-name and dtype/shape span checks used by metadata preflight.
  The generated header-fetch command now uses `--fetch-small-files`, so
  `config.json`, tokenizer metadata, and other capped small files can arrive in
  the metadata bundle without opening any weight payloads.
  The no-weight `prefill-backend --write-report` step writes
  `largerlm-prepared/prefill-backend-report.json` before shard download, and
  the metadata `preflight-glm` plus `prepare-glm --metadata-only` dry-run gates
  are also ordered before shard download. That captures M5 MPSGraph runtime
  proof, public MPP availability, GLM-5.2 shape compatibility, layout planning,
  cache budget, and disk budget without opening any model payloads.
  `--require-clean` makes the aggregate
  no-missing/no-partial/no-extra/no-header error state a hard gate, and
  `post_copy_check_command` prints that exact safe check. Only after that
  local-header-verified state does the report expose
  `prepare_execute_command`, `inspect_prepared_command`, `launch_audit_command`,
  and `minimal_smoke_command`; those commands form the conservative final
  bring-up path from writing packed artifacts, to prepared request admission, to
  a locked launch-profile audit, to the first 1-token prepared generation
  guarded by that audit. The generated preflight/prepare commands also pin the
  first-pass M5 Max envelope with
  `--group-size 32 --max-cache-gib 16 --disk-margin-gib 32
  --unified-memory-gib 128`, aligning status-driven bring-up with the documented
  GLM-5.2 MXFP4 safety budget. The
  generated inspect and audit commands also run the tiny MPSGraph and MPP
  tensor-ops probes with a 30 second backend-probe timeout, so M5 launch
  artifacts capture runtime proof or failure evidence for the planned neural
  accelerator path before the first token is generated. The default
  checkpoint-status bring-up path does not require prefill acceleration; it
  keeps the first audited token on the token-stable custom-Metal profile and
  carries `--allow-non-accelerated-prefill-launch-audit` in the audit evidence while
  dedicated MPSGraph wrappers carry the explicit router-gate-only experiment
  flag.
  `bringup_plan` exposes the whole sequence as ordered structured steps with
  prerequisites, `step_status` values (`complete`, `ready`, or `blocked`),
  blocked reasons, command-availability booleans, and explicit labels for steps
  that read weight payloads, write artifacts, or run the model.
  `next_bringup_step` is the first ready command after currently complete steps
  in that ordered sequence, meant as a conservative automation hint rather than
  a durable state-machine cursor. `checkpoint-status` can also write that exact
  next step as machine-readable JSON or a quoted executable shell script without
  running it, reducing manual flag-copying errors during GLM-5.2 bring-up.
  For off-machine downloads, the same status pass can write an external
  download handoff JSON and resumable `curl` script for missing or partial
  safetensors shards; each entry carries the resolved URL, target path, and
  expected byte size, the script accepts a `LARGERLM_MODEL_DIR` override for the
  external machine's download directory, wraps `curl` failures in an outer retry
  loop, supports shard-range and planned-byte caps via
  `LARGERLM_DOWNLOAD_START_INDEX`, `LARGERLM_DOWNLOAD_END_INDEX`, and
  `LARGERLM_DOWNLOAD_MAX_BYTES`, applies configurable connect and low-speed
  timeouts to prevent zero-byte hangs, and fails unless the completed file size
  is exact.
  Prefill backend report, preflight report, prepare dry-run report, prepared
  manifest, launch profile, and launch audit completion use bounded small-JSON
  validation instead of path existence.
  The generated preflight and dry-run commands atomically write
  `preflight-report.json` and `prepare-dry-run-report.json`, allowing a resumed
  status run to skip metadata-only gates only when their reports still bind to
  the current model and prepared output paths.
  The minimal smoke command also writes a schema-tagged `minimal-smoke.json`,
  and status validation requires the first-token result to bind to the current
  prepared manifest, locked launch profile, and launch audit, then contain the
  expected prompt, one-token request, and generated token before marking the
  final step complete. Corrupt, stale, or failed downstream artifacts therefore
  do not let automation skip ahead.
  This gives GLM-5.2 downloads and partial downloads a safe readiness check
  before any safetensors shard payload is opened or the SSD is filled by
  accident.
- `preflight-glm` is the first tensor-metadata gate for real checkpoints. It reads only
  `config.json`, safetensors headers, and tokenizer metadata; it validates GLM
  attention/router/global tensor coverage, dense-prefix MLP gate/up/down
  coverage, router correction-bias coverage, full DSA indexer tensor coverage,
  routed expert packing coverage, decode-cache budget, packed-output disk
  budget, and tokenizer presence before any packing or Metal execution. It
  reports public GLM-5.2 shape diagnostics and can enforce them with
  `--require-public-glm-5-2-shape` before any output path is touched. It
  warns when `topk_method=noaux_tc` has incomplete
  `gate.e_score_correction_bias` coverage. The safetensors scanner treats
  weight-map/header mismatches, shard paths that escape the model directory,
  shard-header tensors omitted from `weight_map`, out-of-bounds tensor offsets,
  overlapping or gapped shard spans, unsupported dtypes, dtype/shape byte-span
  mismatches, and mismatched `metadata.total_size` values as hard errors. Dtype
  size checks accept common long/lowercase aliases (`uint32`, `float16`,
  `bfloat16`, and related integer/float names) without rewriting the recorded
  dtype, so later packer/server diagnostics still show the checkpoint's source
  metadata. A single unindexed `.safetensors` file is accepted for minimal smoke
  checkpoints, but multiple shards without `model.safetensors.index.json` fail
  closed, so partial or corrupt checkpoint downloads fail before packing. A
  metadata-only `largerlm.safetensors.headers.json` manifest can stand in for
  absent shard files during preflight; it records each shard header, data-start
  offset, and remote file size fetched through HTTP Range requests, letting
  GLM-5.2 tensor naming, dtype/shape coverage, and routed/resident byte
  accounting be audited before the full quantized weights are downloaded. The
  header fetcher retries transient `{"detail":"Unsupported content type"}`
  HTTP error bodies and 200-response bodies for index, Range, and capped
  small-file requests, keeping metadata-only bring-up resilient to intermittent
  Hugging Face or proxy content negotiation failures.
  `--metadata-only` extends that manifest-first path to partial downloads where
  local shard files already exist, and the same flag on `prepare-glm` is limited
  to dry-run layout/cache/disk-budget checks so executable packing still
  requires real shard payloads. The
  first real probe against `mlx-community/GLM-5.2-mxfp4` produced a 76-shard,
  2,489-tensor manifest with `395094087168` indexed tensor bytes. With the
  public GLM-5.2 config fixture, preflight accepts its absorbed
  `embed_q`/`unembed_out` attention form and resident MXFP4 logical shapes; the
  routed expert packer now emits `quantization="mlx-mxfp4"` layouts with
  32-element scale blocks. The runner now executes routed MXFP4 experts for
  single-token and expert-major batch MoE smoke cases. Resident MXFP4 execution
  now covers bounded embedding rows, 2-D custom-metal linear/batch projection,
  router gates, bounded Python DSA indexer matrices, attention q/kv/o
  projections, shared/dense MLP triplets, and chunked final-logits top-k. The
  low-level attention runner now covers absorbed `embed_q`/`unembed_out`
  aliases by skipping absent `kv_b_proj` in projection-only runs and building a
  bounded f32 KV-B view for single, batch, and indexed MLA attention. A tiny
  `--run-decoder-layer` command smoke covers the absorbed path end to end for
  one layer, and Python batch MLA prefill checks accept the same aliases; the
  Python batch projection and composed attention-block wrappers now carry that
  value source through their bounded result objects. A live metadata-only
  bring-up in `artifacts/glm-5.2-mxfp4` now completes header fetch, capped
  small-file fetch, M5 prefill-backend reporting, metadata preflight, and
  metadata prepare dry-run with `--group-size 32`; the remaining hard boundary
  starts at full shard download and continues through post-copy local-header
  verification, `prepare-glm --execute`, launch audit, and first-token smoke.
  Preflight also inspects config-level quantization metadata from
  `quantization`, `quantization_config`, and common top-level bit/group fields.
  Pre-quantized MLX affine-int4 runs fail closed when declared bits or group
  size disagree with the requested packer settings, while raw BF16/F16/F32
  conversion treats mismatched declarations as source metadata and warns. If
  the config already describes MLX affine-int4 and raw conversion is requested,
  preflight warns that raw conversion expects BF16/F16/F32 expert weights; if
  the source tensors are already `uint32` affine weights, packer diagnostics
  tell the user to remove the raw-conversion flag. When a checkpoint omits
  config-level quantization metadata but routed expert tensor names expose
  GPTQ/AWQ/bitsandbytes conventions such as `qweight`, `qzeros`,
  `g_idx`, `quant_state`, `absmax`, or `quant_map`, preflight adds a specific
  unsupported expert quantization layout error before the generic expert
  coverage failure. Direct expert layout construction uses the same header-only
  name check before reporting missing routed expert sources. Preflight and
  resident packing now validate non-routed resident `uint32 .weight` tensors
  with `.scales`/`.biases` companions, and the resident linear custom-metal
  path executes attention, dense, and indexer-style affine-int4 projections
  without expanding the full matrix to F32. The router custom-metal path also
  executes affine-int4 or MXFP4 gate weights directly for routed expert
  selection.
  Routed experts execute either affine-int4 or MLX MXFP4 directly. Shared-expert
  and dense-MLP blocks execute resident affine-int4 or MXFP4 gate/up/down
  triplets directly, attention q/kv/o projections dispatch resident MXFP4
  matrices by logical shape, the Python DSA indexer baseline dequantizes MXFP4
  `wk/wq_b/weights_proj` behind its matrix and f32-expansion caps, and final
  logits stream affine-int4 or MXFP4 `lm_head.weight` in bounded row chunks for
  top-k selection. Absorbed `embed_q`/`unembed_out` aliases are now covered in
  low-level projection and MLA attention runners, including a single-layer
  decoder command smoke, and Python batch MLA prefill budget checks accept the
  alias value source. Higher-level batch projection and composed attention-block
  wrappers now skip the absent `kv_b_proj` output and report the same
  `absorbed-alias` source. The remaining fully MLX-quantized GLM-5.2 boundary
  for `mlx-community/GLM-5.2-mxfp4` is the full shard download, post-copy
  local-header verification, `prepare-glm --execute`, launch audit, and an
  end-to-end prompt smoke against the selected quantized artifact.
  Routed expert packing accepts both per-expert tensors and fused expert-major
  raw tensors by slicing each expert's logical `[out, in]` matrix before
  affine-int4 quantization; `w1/w3/w2` expert aliases map to the standard
  gate/up/down components, and per-expert ids outside `n_routed_experts` fail
  during expert coverage instead of being ignored. Pre-quantized affine-int4
  layouts keep checkpoint dtype strings for provenance while accepting
  `U32`/`uint32` weights and `BF16`/`bfloat16`/`BFLOAT16` or
  `F16`/`float16`/`FLOAT16` metadata in both prepared readiness checks and the
  Metal runner layout contract. Planner and preflight byte/context/group
  controls use strict integer validation before deriving packed layouts,
  decode-cache budgets, or live-memory recommendations.
- `prepare-glm` composes the safe preparation flow. It is a dry-run by default:
  preflight, expert/resident packing estimates, decode-cache layout planning,
  output collision checks, and disk budgeting happen before any file is written.
  Its cache alignment, chunk, heap, disk, cache, and memory budget controls are
  strict integers before either the dry-run or execute path starts.
  Reports include the public GLM-5.2 shape diagnostic, and
  `--require-public-glm-5-2-shape` turns that diagnostic into a pre-write gate
  for GLM-5.2-specific preparation runs.
  With `--execute`, it streams resident and expert tensors with the packers'
  chunk/heap limits, writes the cache layout, creates the cache backing file via
  sparse `ftruncate`, records a manifest, and validates the written prepared
  output before returning success. The manifest records whether the public
  GLM-5.2 gate was required at prepare time, whether the shape matched, and the
  mismatched field list for audit/health output. It then runs the same GLM
  4bit readiness gate used by launch commands, so a prepare run cannot leave a
  package that immediately fails the routed-expert/resident/decode-cache launch
  checks. The cache-layout JSON and prepared manifest use temporary files and atomic
  replace.
- Expert packing requires every expert in a packed layer to have the same
  component names, dtypes, logical shapes, and byte sizes. Shape mismatches are
  rejected even when malformed tensors have the same raw byte count.
- Preflight validates global resident metadata as well as layer-local tensors:
  `embed_tokens.weight` and `lm_head.weight` must be 2-D with hidden dimension
  matching `config.hidden_size`; when `vocab_size` is present, their row counts
  must match it, and the final norm must be `[hidden_size]`. These checks run
  from safetensors headers before resident packing reads weight bytes.
- Model-config-driven generation repeats the same global embedding/head shape
  expectations at runtime admission: embedding reads check `vocab_size` and
  `hidden_size`, and final-logits guards check head rows and input dimension
  before allocating chunks or launching Metal. Runtime preflight also validates
  the resident embedding row cap up front, so missing or unexpectedly large
  embeddings fail before generation work directories are created.
- `prepare-glm --auto-context-from-budget` chooses the largest decode-cache
  context that fits the modeled cache budget by validating the actual aligned
  cache layout, not just the per-token byte estimate. Auto context is also
  capped at the model config's `max_position_embeddings`, and explicit
  `--max-context-tokens` requests above that limit are preflight errors. Planner
  budget inputs are validated before auto context sizing: runtime buffer and
  quantization shape controls must be positive, explicit memory/cache budgets
  must be non-negative, and page-cache fraction must stay in [0, 1]. Plans also
  expose resident-memory budget, pressure, headroom, and fit status after
  subtracting system reserve and runtime buffer from unified memory, and
  preflight warns when those resident assumptions exceed the selected memory
  profile. Preflight reports detected chip/memory/GPU-core metadata, the
  effective unified-memory bytes used for planning, and whether that value came
  from explicit CLI input or hardware detection. Prepare reports and executed
  manifests record the auto-context decision, requested/resolved context,
  modeled decode-cache budget, safe context under that budget, effective
  `max_cache_bytes` guard, and cache dtype/alignment, so the context budget is
  auditable after the original dry run has scrolled away.
  Planner JSON also emits `suggested_prepare_flags` when the decode-cache
  budget is usable: it is an argv-style fragment for `prepare-glm` that pins
  auto-context sizing, the cache budget, unified-memory/reserve/runtime inputs,
  page-cache fraction, and group size without guessing the checkpoint's raw-vs-
  prequantized expert format. The `plan --write-prepare-flags` option writes
  that fragment as a small JSON artifact, leaving model path, output directory,
  and execute/dry-run choice explicit at prepare time. `prepare-glm
  --apply-prepare-flags` replays only plan-authored, safe-to-replay,
  allowlisted budget flags from that artifact before local CLI arguments.
  Dry-run reports, executed manifests, and prepared health output retain the
  artifact path and SHA-256 digest so budget provenance remains auditable.
  Launch audit includes a `prepare_flags_provenance_ok` check when that
  provenance is present, without requiring legacy prepared packages to have it.
  Prepared manifests also preserve prepare-time public GLM-5.2 target evidence:
  whether the strict gate was required, whether the public shape matched, and
  the recorded mismatched fields. Loaded manifests reject self-contradictory
  public-shape metadata, and prepared health exposes the same fields under
  `prepared_storage`. Launch-profile prepared identity treats the same fields
  as optional comparables, so newly generated profiles reject prepared
  directories whose public GLM-5.2 prepare evidence has drifted while older
  compatibility profiles remain replayable when they omit the optional fields.
  Launch-profile identity also binds prepare-time expert-pack heap envelope and
  resident alias rewrite evidence when those safety fields are present, so
  raw-conversion or resident compatibility rewrites cannot drift under a stale
  profile while old packages with no recorded rewrite evidence keep replaying.
  Manifest loading also bounds resident alias/fused rewrite byte counts by the
  validated resident layout size, rejecting impossible compatibility evidence
  before a launch profile or audit artifact can trust it.
  Executed manifests also retain the expert-pack heap envelope used during
  raw 4-bit conversion: chunk size, estimated peak heap, max heap, raw source
  block, generated output block, extra modeled heap, and rows per block.
  Manifest loading rejects self-contradictory pack-heap evidence: raw blocks
  cannot exceed the recorded peak, and the extra heap must cover generated
  output plus any source-block overflow beyond the copy chunk. For internal
  `largerlm-affine-int4` expert layouts, manifest loading now requires every
  pack-heap evidence field and uses the expert layout quantization as the
  effective type when the top-level manifest omits it, preventing a stale or
  stripped artifact from bypassing the later audit path. Prepared health also
  reports the validated layout quantization/group size separately from
  manifest-level expert metadata, and pack-heap audits report the effective
  quantization used to decide whether the envelope is required. Prepared health
  exposes the fields for later GLM-5.2 bring-up audits. Required
  launch-audit replay also carries a dedicated
  `prepare_expert_pack_heap_envelope_ok` check: internal
  `largerlm-affine-int4` expert packs must have complete heap evidence, the
  estimated pack peak must stay within the recorded cap, raw byte evidence must
  have a positive row-block count, and replay compares the audited fields with
  the current manifest before runner work starts.
- `plan-cache` applies the same `max_position_embeddings` cap before writing a
  standalone decode-cache layout, so an oversized sparse cache cannot be
  initialized from an invalid plan.
- `prefill-plan --prompt-tokens` is also capped by `max_position_embeddings`
  before reporting MPP/SSD work estimates.
- `inspect-prepared`, `generate-prepared-token-ids`, and
  `generate-prepared-text` consume that manifest directly. They validate that
  the recorded layouts/cache/model path still exist, the decode-cache layout is
  parseable, the cache backing file exactly matches its layout size, the
  resident/expert backing files are large enough for their layouts,
  resident/expert byte spans do not overlap, recorded byte counts match layout
  metadata, and any packed layout `config_sha256` still matches the model
  `config.json` before starting generation. This reduces the
  chance of mixing an expert pack, resident pack, cache file, or config from
  different runs. Loaded manifests keep the validated expert, resident,
  decode-cache layout, and decode-cache file byte counts, letting
  `inspect-prepared` and `/health` report prepared storage totals and the
  recommended available-memory budget without scanning or mapping model
  weights. `/health` also reports a `memory_guard` block with the configured
  required available bytes, best-effort current available bytes, and an
  `available_ok` flag. `inspect-prepared` stops at the shared health
  calculation, so it can check effective context limits, live guard defaults,
  current system memory, and prefill backend warnings without starting a server
  or creating work files; its text output includes the configured required
  memory and guard ok status. `--write-launch-profile` writes the best available
  prepared/request launch profile JSON, and prepared generation, benchmark,
  serving, and inspection commands can later use `--apply-launch-profile` to
  expand that profile as safety defaults before explicit command-line overrides.
  `--lock-launch-profile` turns that replay into an exact argv lock, rejecting
  later command-line overrides that change any flag carried by the profile.
  New launch profiles also carry prepare-time hardware identity fields when the
  manifest has them, including Apple Silicon generation/tier; those fields are
  optional for old profiles, but once present they must match the prepared
  manifest before any runner dispatch.
  `--require-locked-launch-profile` makes prepared commands fail unless a
  launch profile is both applied and locked, which gives production launch
  scripts a single switch for refusing ad-hoc memory and prefill guard limits.
  Applied profile summaries report `locked`, `lock_required`,
  `profile_flag_count`, and `lock_checked_flags`, so JSON and text health
  output show whether exact replay was actually enforced before any runner
  process starts.
  `inspect-prepared --require-launch-audit` layers a strict startup checklist on
  top of those fields. It returns non-zero unless the applied profile is locked,
  the prepared identity is hash-bound, the expert and resident layout backing
  files were loader-validated, the decode-cache backing file exactly matches the
  declared layout size, the prepared memory profile is required and verified, the
  prepared decode-cache context budget is present and verified, the current
  runtime memory profile and configured memory guard pass, GLM 4-bit and public
  GLM-5.2 gates pass, prefill acceleration is both required and available, the
  current selectable MPSGraph prefill path has a passing runtime probe, the
  locked launch profile itself replays that probe, and a checked request passes
  admission, prefill routed expert SSD read-speed/seconds-budget checks, decode
  routed expert bytes/token and seconds/token checks, routed stage temp
  cap/free-space checks, runtime preflight, and produces a safe request launch
  profile. This gives launch scripts a
  runner-free gate before they map any large weights.
  The prefill acceleration check stores the selected MPSGraph fallback, visible
  acceleration runtimes, runtime gaps, host probe identity/timeout, MPP
  compile/run probe fields, and `prefill_neural_accelerator_status`, so an M5
  launch artifact records whether the planned MPP/Neural Accelerator route is
  unavailable,
  probed-but-not-selectable, or ready for a future selectable backend.
  Request launch-audit checks also preserve the full prefill acceleration
  coverage evidence, including matrix counts, estimated FLOPs, accelerated
  backend names, and FLOP fractions. They also bind the MPP tensor-ops candidate
  policy plus candidate matrix count, FLOPs, and FLOP fraction, so an M5 launch
  artifact cannot silently lose the evidence that large prefill GEMMs were
  eligible for the future Neural Accelerator path. Strict replay rejects
  malformed or internally inconsistent coverage when those fields are present.
  If that status claims a passing MPP run probe, required-audit consumers
  require the concrete kernel variant, `32x32x32` half tile shape, and
  `mpp::tensor_ops::matmul2d` primitive to replay consistently.
  Required-audit consumers also require the prefill acceleration backend probe
  itself to have been requested, run, passed, and recorded with a non-empty
  probe path plus positive timeout before accepting the MPSGraph fallback as
  proven acceleration.
  The GLM 4-bit check stores the accepted expert quantization/group size, packed
  expert layer file counts and exact-size status, config-derived expert bytes,
  per-token routed expert read bytes, full-prompt expert sweep bytes, and
  decode-cache segment counts in the audit artifact. That keeps the accepted SSD
  streaming envelope visible even when the later serving command only consumes a
  saved audit file.
  `--write-launch-audit` saves that checklist with the prepared identity and
  applied launch-profile SHA-256. `serve-prepared --require-launch-audit PATH`
  revalidates the artifact before server startup: the audit must have passed,
  every current strict checklist item must be present and passing, the prepared
  identity must be strong and match the package being served, and the current
  command must apply and lock the same launch-profile bytes that were audited.
  Artifact consumers require the GLM 4-bit SSD-envelope fields, the checked
  request's prefill/decode routed expert SSD read-budget evidence, routed stage
  temp cap/free-space evidence, runtime preflight live-memory evidence, and the
  prefill acceleration/M5 probe evidence fields, re-run the low-memory
  readiness pass, and reject mismatched expert/read/cache envelope values,
  missing request read/temp/memory-budget evidence, or contradictory
  acceleration probe states before invoking the runner. When runtime preflight
  includes prompt-prefill live-memory details, strict replay also validates the
  prompt batch, runner scratch, cache read/write, and stage-copy byte caps
  against the recorded live-working-set formula. Prompt-prefill cache I/O
  evidence is likewise checked when present: MLA and DSA index cache read/write
  byte buckets must sum to the recorded totals before the audit can be replayed.
  Runtime-memory evidence also includes the resident backing byte count and the
  non-resident peak, so replay rejects old or tampered artifacts that omit
  resident pressure from `live_working_set_bytes`.
  Prefill acceleration coverage evidence now similarly includes the MPP
  candidate policy and candidate FLOP buckets. Replay verifies the candidate
  fraction formula, count bounds, and expected MPP execution path before
  trusting the checked request.
  Routed chunk-frontier evidence is also validated when present: candidate
  chunks must be sorted, cover the resolved/max-safe chunk values, and keep
  read-amplification plus stage/static byte totals internally consistent.
  The artifact must carry a safe `request_launch_profile`, and that nested
  profile must also carry the same strong prepared identity before its
  prompt-specific guard flags are trusted. When the artifact records
  prefill/decode routed-read budget evidence, the request profile must carry
  the matching `--prefill-*` and `--decode-max-routed-read-*` replay flags, and
  those caps must not be wider than the evidence-derived 5% headroom profile.
  Routed stage temp byte caps, static-capacity mode, and raw/coalesced stage
  range-count caps are bound to the same request profile. Replay rejects
  missing `--prefill-max-stage-raw-ranges` or
  `--prefill-max-stage-coalesced-ranges` flags, and also rejects values wider
  than the audited request's maximum range counts plus the same 5% headroom.
  If the applied profile came from a benchmark and carried
  `sections.prefill_actual_read_time`, required-audit consumers also verify
  that cumulative read-time evidence before launching, including the measured
  stage-copy seconds guard. Decode actual read-time evidence is likewise
  verified when present. New launch-audit artifacts also
  carry the prepared runtime-profile envelope
  (`prepare_effective_unified_memory_bytes`, reserve, recommended max-live,
  recommended min-free, and their required-available sum); strict replay
  compares those manifest-derived values with the current prepared package, so
  a saved GLM-5.2/M5 memory audit cannot be reused after the prepared memory
  profile is widened or rewritten. Prepared SSD cold-read evidence is handled
  the same way: when an audit records `prepare_cold_read_*` provenance, replay
  rejects packages whose manifest speed, source, benchmark path, benchmark
  byte counts, or elapsed time no longer match.
  The server's `max_prompt_tokens` and `max_new_tokens_cap` must also fit inside
  the audited request envelope. Server startup rechecks the current system memory
  against the applied live-working-set plus free-unified-memory guard, and
  `/health` reports both the audited runtime memory evidence and the current
  server guard binding in `launch_audit_envelope`. Server request admission
  also rechecks each prompt and `max_new_tokens` value against that envelope
  before chunk planning, runtime preflight, or runner execution. Prepared direct generation
  and benchmark commands accept the same artifact; before invoking the runner
  they verify that the current prompt token count and `max_new_tokens` do not
  exceed that envelope.
  Server startup and direct generation/benchmark commands require the current
  effective guard flags to replay that request profile. Passing the audit
  artifact itself to
  `--apply-launch-profile` works because the profile loader extracts
  `request_launch_profile`.
  Saved profiles carry prepared-package identity metadata, and apply-time
  validation rejects mismatched layout byte counts, context, quantization, or
  model-config hashes before generation or serving starts. New prepared-bound
  profiles also record whether plan-derived prepare flags were applied, their
  source, and their sha256 digest; local prepare-flags paths are intentionally
  omitted from the identity so package moves do not invalidate an otherwise
  identical safety profile. The profile records
  `prepared.identity_strength`; profiles without a model-config hash or
  routed-expert format metadata are marked `weak` and include
  `identity_warnings`, while normal `prepare-glm` outputs are hash-bound and
  format-bound. Weak identities remain visible for compatibility in normal
  profile replay, but they cannot satisfy a required launch audit.
  Config-only `prefill_plan`, `prefill_linear_calibration`, and
  `prefill_plan_calibration` profiles are the exception: they have no prepared
  identity, but `--apply-launch-profile` accepts them only when every section
  and argv flag is in the prefill-only whitelist. Standalone
  `prefill_linear_calibration` profiles are narrower still: identity-less replay
  accepts only their runtime-policy section. `--no-runtime-preflight` is
  intentionally not part of that identity-less whitelist; disabling the startup
  preflight can only be replayed from a prepared-bound profile and is still
  covered by `--lock-launch-profile`.
  The same loader can extract `combined_launch_profile` from a full
  `prefill-plan-calibrate --json` payload, so calibration artifacts can be
  replayed without hand-editing JSON.
  The payload also carries `calibration_candidate_coverage`, mirrored from the
  planner, which maps each deduplicated calibration matrix shape back to the
  concrete prefill backend candidate ranks, op names, layer counts, FLOPs, and
  weight bytes it represents. That makes an M5/GLM-5.2 calibration artifact
  auditable: a later launch can tell whether the measured shapes covered the
  hot planner-ranked MPP/MPSGraph candidates or only a truncated tail.
  When given `--check-prompt-tokens` and
  `--check-max-new-tokens`,
  it also runs the prepared token-id admission check and automatic prefill chunk
  sizing for that request, including the same batch-prefill MLA/DSA cache I/O
  estimate reported by `prefill-plan`. `--check-runtime-preflight` extends that
  request check with the same deterministic layer/cache/final-logits/live-working-set
  budget used before generation, plus required available memory, detected
  available/total memory, source, and an `available_memory_ok` flag. The
  preflight runs for decode work and for batch-prefill-only requests, so a
  `max_new_tokens=0` prompt check still admits the prefill scratch/cache working
  set before an audit can pass.
  Runtime/live-memory failures now travel as structured request-check failures:
  the guard raises a diagnostic payload with a failure `code`,
  required/available memory, memory source, and `available_memory_ok=false`, and
  HTTP server paths include that payload under `request_check.runtime_preflight`
  in the 400 response. This keeps "low memory, refused before runner" distinct
  from a process crash or opaque tokenizer/request error.
  `--require-launch-audit` consumes that same request evidence and requires the
  request-level prefill acceleration coverage and runtime memory result to pass,
  so the audited envelope is tied to an actual prompt size instead of only
  package-wide defaults. The audit's `public_glm_5_2_shape_ok` check also
  records the public-shape mismatch list, the raw DSA schedule checks, and the
  derived full-indexer layer count/list, so a saved launch audit proves which
  GLM-5.2 DSA schedule was accepted.
  `--check-prompt` and `--check-prompt-file`
  load only the local tokenizer to count a real text prompt before the same
  admission check. `--check-chat-messages` and
  `--check-chat-messages-file` additionally render the local chat template,
  including the same local Jinja fallback used by the server when a tokenizer
  backend cannot expose `apply_chat_template`, and encode the rendered prompt
  with `add_special_tokens=false`, matching the prepared chat endpoint. All
  offline prompt/chat check inputs are capped by
  `--max-request-bytes` before tokenization. The request check also reports the
  selected prefill linear backend, auto-policy thresholds, and resident matrix
  backend mix for the resolved prompt chunk, including a bounded `top_matrices`
  list sorted by estimated FLOPs. It also reports
  `prefill_routed_expert_read`, computed from the prepared expert layout, so a
  request check exposes routed expert read amplification before any stage files
  are created. `prefill_prompt_chunk_plan` carries the resolved-auto and
  max-safe auto chunk cap tables, limiting cap names, and next-token scratch
  estimate, so prepared/server preflight can explain MPSGraph scratch cliffs
  before creating prompt work files. Required launch-audit replay now treats
  that plan as hard evidence for batch prefill: the request evidence, audit
  check, and `request_launch_profile.sections.prefill_prompt_chunk_plan` must
  match, and both the max-safe plan and matrix-scratch fields must be present.
  A positive
  `--prefill-max-routed-read-amplification` turns that
  estimate into an admission limit, and
  `--prefill-max-routed-read-gib` caps the absolute planned routed expert read
  bytes; `--prefill-ssd-read-gib-s` and
  `--prefill-max-routed-read-seconds` add the matching time estimate/cap. The
  same bottom-level token generator rechecks all routed-read limits before
  creating prompt-prefill work files. Request inspection also reports
  `minimum_chunk_tokens_for_limits` when a larger prompt chunk could satisfy the
  active routed-read caps, and distinguishes impossible caps whose baseline
  full-prompt expert read already exceeds the limit. Successful checks include
  `suggested_guard_flags`, an argv-style routed-read guard profile with 5%
  headroom for replaying the same safety envelope in generation, benchmark, or
  server commands. Decode has a matching per-token safety valve:
  `--decode-max-routed-read-gib-per-token` caps the deterministic
  `read_bytes_per_token` estimate from selected MoE layers, while
  `--decode-max-routed-read-seconds-per-token` converts that estimate through
  `--prefill-ssd-read-gib-s`. Either decode cap forces the runtime preflight
  needed to compute it before generation work files are created.
  Successful request checks and prepared benchmarks expose
  `suggested_decode_guard_flags`, an argv-style `--decode-max-routed-read-*`
  profile with 5% headroom for replaying the inspected per-token SSD envelope.
  `suggested_stage_temp_guard_flags` similarly profiles
  `--prefill-max-stage-mib`, `--prefill-max-compact-stage-mib`, and the
  inspected `--prefill-static-capacity-per-expert` mode from the routed staging
  footprint. `suggested_prefill_guard_flags` combines the prefill read and
  stage profiles into one deduplicated argv list for replaying the inspected
  prompt envelope. When `inspect-prepared` runs a
  successful prompt/chat request check, `request_launch_profile` combines that
  prompt-specific prefill profile with the prepared launch, acceleration, and
  decode sections into one replayable argv list. It also keeps
  `sections.prefill_prompt_chunk_plan` as audit metadata while leaving replay
  `argv` limited to guard flags. Applying that profile echoes the same metadata
  in `applied_launch_profile.prefill_prompt_chunk_plan` for post-run drift
  checks and emits `prefill_prompt_chunk_plan_drift`, comparing profile
  max-safe chunk evidence with the current request/run's max-safe plan.
  Offline prepared generation and `serve-prepared` request checks reject the
  replay before runner work starts when the current max-safe chunk is lower
  than the profiled max-safe chunk, the selected chunk exceeds the current
  max-safe chunk, or the current max-safe evidence cannot be verified. Request
  inspection also reports
  `prefill_routed_stage_temp_disk`, the resolved prompt chunk's routed expert
  stage + compact temporary disk estimate plus the static-capacity binary route
  bytes, with both the single-layer peak and total bytes across the checked
  prompt. Admission rejects the request when the single-layer stage or compact
  peak exceeds the configured caps. `prefill_stage_temp_disk_free` then compares
  the stage + compact + static-capacity peak
  plus the configured stage disk margin with the selected prompt work
  directory's free space, defaulting to `/private/tmp`, and fails closed when
  the temp volume is too small or cannot be inspected.
  It also reports
  `prefill_routed_chunk_frontier`, a small candidate frontier for prompt chunk
  sizing. The frontier includes powers of two, the expert-saturation threshold,
  the resolved chunk, and the full prompt, then records each candidate's
  worst-case routed read amplification, planned SSD read bytes/seconds when an
  SSD GiB/s is configured, and stage + compact + static-route temporary bytes.
  It reports
  `request_check` and returns non-zero if the request would be rejected.
  The prepared server's `/generate-token-ids`, `/generate-text`, OpenAI
  completion, and OpenAI chat paths invoke the same request admission before
  entering the serialized generation section, and force runtime memory
  preflight for that admission, so service calls fail on unsafe prompt chunk,
  routed-read, temp-disk, free-memory, or acceleration envelopes before any
  runner work directory is created.
  The offline `generate-prepared-token-ids` and `generate-prepared-text` CLIs
  invoke the same token-count admission before calling the generator, closing
  the gap between service and one-off launch scripts.
  `generate-prepared-text` must successfully tokenize the prompt before that
  admission step; tokenizer initialization or encoding failures are terminal, so
  text launch scripts cannot continue to the runner with an unverified prompt
  token count.
  Offline admission is fed the same generation overrides used by the eventual
  runner call, including layer filters, top-k, runtime guard caps, and DSA
  settings, so routed-read and memory estimates are not computed from stale
  server defaults.
  Prepared token-id benchmarks invoke that same admission before runner
  dispatch, including the direct `benchmark_prepared_token_ids(...)` Python API,
  so benchmark sweeps cannot skip prompt chunk, routed-read, temp-disk, decode,
  or acceleration guards.
  All generation and benchmark entry points require the returned admission
  payload to be structured and to carry `ok: true`; a soft `ok: false` result is
  treated like an admission exception and cannot fall through to the runner.
  Required launch-audit replay validates the same stage-temp evidence, including
  `max_stage_plus_compact_plus_static_bytes`, static-capacity route-table bytes,
  raw/coalesced range counts and limits, and the replayed
  `--prefill-static-capacity-per-expert` mode, so a launch artifact cannot omit
  route table budget or read-fragmentation evidence while still satisfying the
  routed stage caps.
  Prepared token-id, text, and benchmark entry points automatically switch
  prompts longer than one token to chunked batch prefill unless explicitly
  disabled, so the safer manifest path also avoids accidental prompt replay.
  The chunk size defaults to automatic safety-cap sizing rather than the old
  fixed 64-token fallback. Prepared prompt MoE also defaults to
  `--prefill-static-capacity-per-expert auto`, so the binary `LLMSCAP1` route
  table is used without repeating flags on every prepared run, after the same
  Python-side binary validation used by the low-level staged MoE command.
  Prepared
  manifests also carry `recommended_max_live_working_set_bytes` and
  `recommended_min_free_unified_memory_bytes`; prepared generation, benchmark,
  and serving entry points inherit those guard defaults unless the caller
  explicitly supplies guard options. They also record the packed routed-expert
  quantization and group size plus the prepare-time hardware profile
  (`prepare_hardware_*`, `prepare_effective_unified_memory_*`, and
  `prepare_system_reserve_bytes`). The hardware profile includes parsed Apple
  Silicon M-series generation and tier, so M5-specific launch/backend policy can
  key off structured manifest evidence instead of matching the chip string.
  Execute-mode preparation reuses the runtime memory probe after dry-run sizing
  and before the first output write, requiring current available memory to cover
  the larger expert/resident packer heap estimate plus the prepare system
  reserve. The executed manifest records the estimated prepare live set, the
  required available bytes, observed system available/total bytes, and probe
  source, and prepared server health exposes that record as a
  `prepare_live_memory` block with a current-available-memory comparison.
  The same pre-write gate checks a combined disk budget for packed experts,
  resident weights, and decode cache plus margin, preventing large prepares from
  passing separate per-output checks while exceeding the volume as a whole.
  Prepared manifests also retain the
  decode-cache context-budget metadata from `prepare-glm`; loading rejects
  manifests whose recorded resolved context, cache dtype/alignment, or cache
  budget no longer agrees with the decode-cache layout. When `prepare-glm` is given
  `--cold-read-gib-s`, the manifest records
  `prepare_cold_read_gib_per_second`; prepared generation, benchmark, serving,
  and inspection inherit it as the default `--prefill-ssd-read-gib-s` unless
  `--no-prepared-ssd-read-default` is set. `inspect-prepared` and `/health`
  surface those fields under `prepared_storage`, and launch-audit generation
  records a `prepared_ssd_read_profile_valid` check with the prepared cold-read
  speed, source, and benchmark provenance when present. Strict replay compares
  those prepared-manifest SSD fields with the current package before trusting a
  saved request I/O budget. This makes it possible to audit that a GLM-5.2
  package was prepared with the intended 4-bit expert format, memory budget,
  and SSD read-speed basis before mapping any large model files. They also expose
  `prepared_runtime_profile`, which compares the current system total and
  available memory with that prepare-time budget and reserve so a smaller or
  memory-pressured machine is flagged before generation starts. The profile
  also reports the prepared package's recommended required available memory
  (`recommended_max_live_working_set_bytes` plus
  `recommended_min_free_unified_memory_bytes`) and treats a known shortfall or
  inability to inspect current memory as a profile failure. The offline
  `inspect-prepared` command returns non-zero when that profile is false or
  unverifiable, making the memory envelope a scriptable pre-launch gate. The
  profile exposes both recommended guard components, not only their sum, so
  launch-audit artifacts can bind the prepared max-live and min-free envelope
  separately.
  Prepared health also exposes `suggested_launch_guard_flags`, an
  argv-style replay of the manifest's recorded live-working-set and
  minimum-free-unified-memory launch guards. When GLM readiness passes, prepared
  health also emits `suggested_decode_guard_flags` from the config-derived
  per-token routed expert read, giving launch scripts a model-level decode SSD
  envelope before request-specific checks run. `suggested_launch_profile`
  combines those prepared launch guards, selectable prefill acceleration flags,
  any explicit non-`auto` prefill linear backend, the GLM 4bit readiness guard,
  the public GLM-5.2 shape guard when applicable, and model-level decode guards
  into one de-duplicated argv list while retaining each section payload for
  audit. This lets an identity-less calibration profile such as explicit
  `custom-metal` be applied once and then written back as a prepared-bound safe
  launch profile with the memory, SSD, and GLM guards still attached. Prepared
  GLM 4bit readiness also requires the resident backing file to be exact-sized:
  `resident_weight_file_bytes` must match `resident_layout_total_bytes`, and the
  result is preserved as `resident_weight_file_exact_size` in readiness and
  launch-audit evidence. Prepared
  token-id generation, prepared text generation, prepared benchmarks, and
  `serve-prepared` enforce the same profile check before invoking generation or
  starting the server. Direct `PreparedGenerationApp` generation
  calls and the `benchmark_prepared_token_ids` Python API re-check it before
  running too, so programmatic use keeps the same memory-envelope guard.
  `--require-prepared-memory-profile` tightens that gate for real
  larger-than-memory GLM runs by rejecting manifests that lack the full
  prepare-time memory envelope before generic defaults can be used. Manifest
  loading also verifies any recorded expert quantization and group size against
  the expert layout metadata before the package can participate in launch
  profile matching or GLM readiness checks. The same command
  can be run with `--require-glm-4bit`, which promotes
  `glm_4bit_readiness.ok` into a failure condition after checking the
  expert and resident layout `config_sha256` identity, expert/resident
  layout `model_type` identity, expert layout
  model-layer count, manifest-recorded expert quantization/group metadata,
  config-derived GLM MoE layer set, routed expert count,
  affine-int4 quantization/group metadata, the exact `component_order`,
  each gate/up/down
  component's dtype, shape, byte size, and exact packed offset in the expert
  layout, plus resident embedding/final-norm, attention, router,
  dense/shared MLP, and full-DSA indexer tensor metadata. The readiness gate
  also checks the resident layout's
  router metadata against the typed config fields that drive decode and prefill
  routing, so stale packages with different scoring, top-k grouping, normalized
  top-k, or routed scaling semantics fail closed. The expert layout `num_layers` is the total model
  layer count, while the packed `layers` array is the MoE-only expert file set;
  this keeps dense-prefix GLM packages valid while still requiring every MoE
  layer to have SSD-backed expert slots.
  The Metal runner repeats the expert slot geometry checks at the executable
  boundary: layer ids, slot bytes, group size, component offsets, component
  sizes, and component shape dimensions must be strict integers, and
  affine-int4 component spans must match the exact gate/up/down packed order
  before any expert bytes are read into Metal buffers.
  It also rejects resident layouts that contain config-declared routed expert
  tensors, keeping routed experts on the SSD-backed path instead of silently
  growing the resident working set. When `config.json` declares
  `tie_word_embeddings=false`, the gate requires a real resident
  `lm_head.weight`; tied embedding fallback is only valid for configs that omit
  or enable tied embeddings. If `vocab_size` is present, resident embedding and
  output-head row counts must match that config value.
  The same payload reports config-derived expert slot bytes, per-layer packed
  expert bytes, total expert bytes, decode-token routed expert read bytes, and
  full-prompt all-experts sweep bytes. Prepared expert bytes must match the
  config-derived affine-int4 total, keeping SSD volume estimates honest before
  a launch script accepts the package. GLM readiness also stats each packed
  expert layer file and requires exact `num_experts * expert_slot_bytes`
  lengths, so oversized or duplicated layer files cannot masquerade as a clean
  prepared package.
  It also validates the decode-cache layout against the GLM config: every layer
  must have exactly one `mla_kv` segment with the config-derived MLA cache
  width, full DSA indexer layers must have one `dsa_index` segment with
  `index_head_dim`, unexpected or duplicate segments are rejected, and the
  decode-cache file size must match the layout total. Health JSON reports both
  byte counts so stale cache files are visible before any runner launch.
  Prepared token-id generation, prepared
  text generation, prepared benchmarks, and `serve-prepared` expose the same
  flag so the readiness check can travel with the actual launch command. The
  opt-in `--require-public-glm-5-2-shape` flag adds an exact public-GLM-5.2
  config-shape requirement on top of GLM 4bit readiness, preventing a tuned
  GLM-5.2 launch profile from being replayed against another GLM MoE package.
  That shape report checks the public model's core dimensions, dense/MoE layer
  schedule, attention and MLA projection dimensions, DSA indexer schedule,
  the raw DSA schedule fields (`index_topk_freq=4` and
  `index_skip_topk_offset=3`), `num_nextn_predict_layers`, context window,
  vocabulary, router grouping, normalized top-k behavior, and routed scaling
  factor. The report also exposes the full-indexer layer count and layer list
  so the GLM-5.2 DSA schedule can be audited without expanding the whole config.
  `public_glm_5_2_shape.mismatched_fields` and per-field `checks` are included
  in health/inspection JSON and summarized in text output.
  Programmatic use can set `PreparedServerConfig(require_glm_4bit=True)`,
  `PreparedServerConfig(require_public_glm_5_2_shape=True)`, or
  `benchmark_prepared_token_ids(..., require_public_glm_5_2_shape=True)` for the
  same pre-run gates.
  `preflight-glm`, `prepare-glm`, `prefill-plan`, and
  `prefill-plan-calibrate` also treat the public GLM-5.2 gate as a 4bit target
  lock: with `--require-public-glm-5-2-shape`, non-4bit `quant_bits` or
  `expert_bits` fail before safetensors metadata scans, prepared-file writes,
  or prefill calibration runner dispatch.
  The public GLM-5.2 gate is incompatible with the debug-only
  `allow_missing_dsa_indexer` escape hatch across CLI, server config, and
  benchmark API paths, keeping strict launches from silently omitting required
  DSA indexer residency checks.
  The `serve-prepared` CLI front-loads the same public-shape gate before
  starting the local server, so wrapper code cannot bypass it by replacing the
  server launch function.
  They also expose `--require-prefill-acceleration`, an opt-in M5 bring-up gate
  that fails before generation/server startup unless the selected prefill
  backend has an accelerated runtime available. Today that means the MPSGraph
  fallback path with a passing runtime probe; the same gate will accept the MPP
  tensor-ops path once it is selectable and the SDK/runtime probe reports it
  ready. Prepared generation, benchmark, and
  serving then verify the actual prompt-prefill coverage before reporting
  success. Router-gate-only MPSGraph coverage is rejected by default and must be
  opted into with `--allow-router-gate-only-prefill-acceleration`, keeping
  routing-drift experiments separate from the default safe path.
  `--prefill-min-accelerated-flop-fraction` makes that same gate
  require a minimum FLOPs-weighted accelerated share; values greater than zero
  imply the acceleration requirement even without `--require-prefill-acceleration`.
  `--prefill-router-hybrid-margin-threshold` is also part of the explicit
  prefill runtime policy now: positive values make the auto router gate try
  custom Metal first and keep it only when the effective margin clears the
  threshold, while preserving the value in health, request checks, and launch
  profiles for audit/replay. The legacy
  `LARGERLM_PREFILL_ROUTER_HYBRID_MARGIN_THRESHOLD` environment fallback is only
  used when the explicit value is zero.
- `pack-experts` writes nothing unless `--execute` is present.
- Copying uses bounded chunks, defaulting to 8 MiB.
- Raw expert quantization fills one preallocated component-slice buffer instead
  of accumulating chunk lists. Its heap estimate includes metadata/copy chunk
  plus the largest raw source row-block and generated quantized row-block, and
  pack reports expose those block sizes directly for dry-run audits.
- Estimated packer heap must stay under 512 MiB by default.
- GiB/MiB CLI budget arguments on plan, preflight, prepare, pack, and
  decode-cache setup commands are checked for finite values before byte
  conversion, so malformed `nan`/`inf` inputs fail before any output path is
  touched.
- Shared packer/resident/decode-cache safety helpers then re-check chunk,
  heap, required-byte, and disk-margin values with strict integer semantics, so
  direct Python callers cannot pass floats or booleans into write guards.
- Expert I/O planning also treats expert ids, read-advice byte controls,
  static-capacity slots, stage range-count caps, and stage disk margins as
  strict integers before staging or fixed-route artifacts are created.
- Direct prompt prefill applies that rule to prompt token ids, start positions,
  chunk tokens, attention/DSA dimensions, layer sets, static capacity, and
  stage disk margins before creating its work directory.
- Low-level routed prefill wrappers validate batch tokens, top-k/max-k, router
  groups, finite stage/read-advise controls, static capacity, stage range-count
  caps, and stage disk margins before checking layouts or launching the runner.
- Staged MoE manifest and compact-route fields are type-checked before runner
  invocation: integer fields cannot be floats or booleans, route weights must
  be finite, and duplicate selected experts or token ids are rejected.
- Decode-cache layouts and prepared manifests reject boolean values in integer
  fields, so JSON `true`/`false` cannot masquerade as byte counts, offsets, or
  context sizes.
- Runtime preflight and execution-side layout readers apply the same rule to
  expert layers, resident tensor shapes/sizes/offsets, DSA indexer matrices,
  embeddings, final logits, and prompt-prefill auto sizing before any runner is
  launched.
- GLM launch readiness rejects configs that explicitly request unsupported
  attention bias or non-SiLU MLP activations. The stricter public GLM-5.2 gate
  also binds dtype, activation, attention-bias, and dropout assumptions to the
  published config shape, so same-shape semantic drift cannot satisfy a
  production GLM-5.2 launch audit.
- Direct decode-layer execution validates position/context, layer sets,
  attention and DSA dimensions, cache dtype, read-advice integer controls, and
  finite numeric knobs before work-directory creation or runner launch.
- Prompt embedding batches validate every token id before opening the output
  file, and remove partial prompt embedding output if a row read/write fails.
- Final-logits execution validates RMSNorm epsilon as finite and non-negative;
  the Metal top-k path rejects malformed runner JSON token ids or logits before
  returning candidates to generation.
- Direct token generation rejects non-integer token/control ids, invalid cache
  dtype widths, non-finite sampling/router numeric controls, and invalid
  prompt-prefill stage limits before reading layouts or creating work files.
- Disk free space must cover output bytes plus a safety margin, defaulting to
  16 GiB.
- Expert packing, resident packing, decode-cache initialization, and expert
  staging remove partial output files when a write fails after an output has
  been opened. Prepare metadata, expert stage, compact layout, and compact
  route JSON manifests use a temporary file and atomic replace.
- Non-`--force` `prepare-glm --execute` behaves transactionally for new output
  files: if a later prepare step or final manifest validation fails, newly
  written prepared files are removed before the error returns.
- `prepare-glm` rejects an output directory that resolves to the source
  checkpoint directory itself, while still allowing a dedicated subdirectory
  such as `checkpoint/largerlm_packed`.
- Routed prompt prefill work files, stage files, and compact-stage files run
  pre-write free-disk checks. Their default extra margin is 0 MiB to keep tiny
  tests frictionless, but `--stage-disk-margin-mib`,
  `--compact-stage-disk-margin-mib`, and `--prefill-stage-disk-margin-mib` are
  available for real checkpoint runs. Failed stage and compact-stage copies
  remove partial output files before returning an error.
- Expert stage copies also have default-off raw/coalesced range-count caps.
  `stage-batch-experts` exposes `--max-raw-ranges` and
  `--max-coalesced-ranges`; the staged prefill wrapper exposes the matching
  `--expert-stage-max-raw-ranges` and
  `--expert-stage-max-coalesced-ranges`, and generation/server request profiles
  expose the prepared-run forms `--prefill-max-stage-raw-ranges` and
  `--prefill-max-stage-coalesced-ranges`. Rejections happen before stage files
  or manifests are created, while successful manifests, prompt-prefill results,
  and generation actual-prefill summaries keep the cap, observed counts, and ok
  fields.
- Existing packed files are not overwritten unless `--force` is present.
- `check-runtime` reads only layout JSON and estimates the one-layer runner peak
  before Metal execution. It uses the same top-k, router-size, expert-slot, and
  scratch-budget limits exposed by the runner CLI. The runner also enforces
  `--max-runner-scratch-mib` immediately before allocating Metal buffers.
- With `--include-attention-projections`, `check-runtime` also validates the
  current layer's GLM attention projection dimensions and resident matrix cap.
- Generation preflight carries a top-level live working-set budget across
  decode, final logits, and optional batch prompt prefill. The deterministic
  `--max-live-working-set-mib` guard defaults to 8 GiB, matching the default
  preflight/plan runtime buffer; the optional
  `--min-free-unified-memory-gib` guard probes system memory before any work
  directory is created and rejects runs that would leave too little unified
  memory headroom. A positive reserve forces the runtime guard even when
  `--no-runtime-preflight` is supplied, preserving the memory gate for debugging
  launches that disable broader preflight. Prepared token/text generation and
  prepared benchmarks now pass that runtime-preflight setting into request
  admission as well as the eventual generator, so launch-audit evidence and
  current runner startup checks cannot diverge. Prepared manifests inherit the
  preflight system reserve; when
  no explicit reserve is supplied, the default is adaptive, using 24 GiB on
  96+ GiB unified-memory machines and 16 GiB otherwise. The prepared
  `recommended_max_live_working_set_bytes` is the runtime buffer plus the
  resident layout's packed byte estimate, so generated launch profiles replay
  the same resident-inclusive memory formula used by runtime preflight.
  The live-working-set estimate now adds the validated resident backing bytes
  from `resident.bin` to the largest non-resident runtime peak. That keeps
  shared experts, attention projections, embeddings, and other resident tensors
  inside the same memory guard instead of treating them as separate layout-only
  metadata. `runtime_preflight` reports `resident_backing_bytes`,
  `nonresident_peak_bytes`, and `extra_live_working_set_bytes`, and strict
  launch-audit replay requires
  `live_working_set_bytes == resident_backing_bytes + nonresident_peak_bytes`.
- Batch-prefill admission includes the configured prompt-batch cap,
  cache-read/write caps, stage copy chunk, and runner scratch cap in the live
  estimate. `inspect-prepared --check-runtime-preflight` reports this prefill
  live-memory breakdown alongside the final admitted live set.
- With `--include-decoder-layer`, it additionally budgets bounded MLA cache
  prefix reads, KV-B expansion scratch, attention output projection, and the
  routed/shared MLP peak before a decoder-layer runner invocation is attempted.
- `plan-expert-io` reads only packed expert layout JSON and maps a selected
  routed expert set to coalesced, optionally aligned layer-file read ranges. It
  reports requested bytes, read bytes, wasted bytes, and amplification so SSD
  experiments can compare exact slot reads against larger sequential reads
  without introducing an application-owned weight cache. Routed router JSON is
  validated for integer expert ids, finite weights, and duplicate experts before
  planning; packed layout `layer_file` paths are kept relative to the layout
  directory.
- `disk-read-benchmark` is the matching hardware probe for packed layer files:
  it performs bounded sequential `pread` over an existing file, reports GiB/s,
  and never creates a synthetic large file or maps the file into memory. Its
  JSON result can calibrate `plan --cold-read-gib-s`. The planner converts the
  configured runtime buffer and system reserve into
  `suggested_launch_guard_flags`, and converts the resulting SSD speed and
  config-derived routed expert bytes/token into `suggested_decode_guard_flags`.
  That gives later decode/generation commands a replayable
  `--require-prepared-memory-profile`, `--max-live-working-set-mib`,
  `--min-free-unified-memory-gib`, and `--decode-max-routed-read-*` envelope
  before any prepared package exists.
  `plan --write-launch-profile` writes that memory+decode profile with
  `source="plan"`; prepared commands accept it without prepared identity
  metadata because the whitelist is limited to plan-derived memory and decode
  guard flags.
- The Metal MoE path now opens each packed layer file once per MoE forward
  instead of once per selected expert slot. When
  `--expert-read-advise-align-kib` is non-zero, the runner also issues macOS
  `F_RDADVISE` hints for the selected expert ranges, optionally coalescing
  nearby ranges with `--expert-read-advise-merge-gap-kib`. This nudges the OS
  page cache and SSD readahead without pinning multi-GB expert caches in unified
  memory.
- `plan --max-context-tokens` estimates total MLA/DSA decode cache use and can
  compare it against an explicit `--max-cache-gib` budget. When a cache budget
  is known, it also reports the maximum cache context that fits the current
  safety model.
- `prefill-plan` estimates prompt-prefill GEMM shapes, MPP tensor-op
  candidates, MLA/DSA prompt-cache read/write traffic, routed expert SSD
  streaming cost, and balanced/spill-free static expert capacities without
  loading checkpoint tensors. It can also check the staged MoE runner's auto
  token-block scratch model against a supplied runner cap. Its prompt, bit,
  group, tile, activation-byte, and runner-scratch controls use strict integer
  validation before planning. This keeps M5 prefill backend work measurable
  before introducing Metal 4 MPP code.
- `plan-cache` emits a JSON layout for MLA/DSA decode cache offsets and strides
  without creating or mapping the backing cache data file. Its context,
  alignment, and max-cache controls are strict integers.
- `init-cache` creates the cache backing file with `ftruncate` after checking
  the layout size, optional max-cache budget, and disk safety margin. Those
  budget controls are also strict integers. It is a sparse logical allocation,
  not a multi-GB zero-fill step.
- `generate-text` is a local-tokenizer wrapper over the token-id loop. It does
  not download tokenizer files and keeps logits memory bounded by the
  `final-logits` chunk and top-k limits. Non-greedy sampling samples only from
  the streamed top-k candidates, so users must raise `--logits-top-k`
  deliberately before increasing diversity.
- The token runtime guard validates memory cap inputs before converting MiB to
  bytes: cache/read/scratch/logits chunk caps must be finite and positive,
  live-memory and min-free guards must be finite and non-negative, and extra
  prefill live-memory estimates cannot be negative. Integer guard fields use
  strict integer semantics, so booleans are rejected instead of becoming
  0/1 context, layer, DSA, logits, or live-memory values.
- Direct token generation validates prompt/eos token IDs and integer control
  values with strict integer semantics before layout reads. This keeps
  benchmark and offline Python callers from silently truncating floats or
  accepting booleans as token IDs.
- Token generation defaults to deleting per-position embedding/hidden f32 files
  and per-layer intermediate work directories as soon as they are no longer
  needed. `--keep-work-dir` is a debugging mode, not a long-context default,
  because retaining every token/layer intermediate can exhaust disk even when
  unified memory is safe.
- Chunked batch prefill follows the same default cleanup policy: completed
  prompt chunk directories are removed unless `--keep-work-dir` is set, so the
  work-directory disk peak is a single chunk rather than the whole prompt.
  Automatically-created work directories are also removed on failure, avoiding
  large abandoned stage files in the default `/private/tmp` path.
  Manual `--prefill-prompt-chunk-tokens N` overrides are rejected before the
  generation work directory is created if `N` exceeds the same prompt/cache/
  scratch/disk cap used by automatic chunk sizing.
- Token generation records low-overhead telemetry in the Python scheduler:
  wall time per emitted token plus estimated embedding, routed-expert,
  decode-cache, and logits-head bytes read. Batch-prefill benchmarks also carry
  resolved prompt chunk counts, prefill peak bytes, embedding bytes,
  staged/compact-stage totals, stage-plus-compact temporary bytes, and stage
  read amplification. These are budget-derived estimates,
  not a replacement for Instruments, but they give an immediate sanity check
  before optimizing SSD prefetch, Metal buffer reuse, or MPP prefill.
- `bench-prepared-token-ids` packages that telemetry into a stable prepared
  manifest benchmark. It is the first performance gate for comparing M5 Max
  SSD/page-cache behavior, expert slot sizes, logits chunking, and later Metal
  4 MPP prefill changes without changing the inference API. When batch prompt
  prefill is active, the benchmark also exposes routed MoE counts, effective
  token block, estimated MoE runner peak bytes, static-capacity route table
  slots/bytes, aggregate resident linear scratch/F32 conversion bytes, and
  actual `prefill_acceleration_coverage` /
  `prefill_acceleration_frontier` fields for the resolved chunk and a bounded
  set of layout-derived candidate chunks. Frontier suggestions preserve
  acceleration requirements and minimum accelerated-FLOP fractions when those
  gates are configured, so copied argv keeps both the chunk and the reason the
  chunk was selected. It also
  reports stage-plus-compact temporary-byte totals/peaks, plus
  `suggested_guard_flags` derived from the actual staged routed-expert read
  profile and `suggested_stage_temp_guard_flags` derived from the actual
  staged/compact/static-capacity peaks and raw/coalesced stage range counts,
  both with 5% headroom.
  `suggested_prefill_guard_flags` combines those actual-prefill read and stage
  profiles into one replayable argv list. `suggested_launch_profile` merges that
  measured prefill envelope with measured decode guard flags, prepared memory
  launch guards, the prepared/configured SSD read-speed baseline, and prepared
  identity. It also preserves GLM 4-bit/public-shape gates plus M5 prefill
  acceleration/runtime-probe gates that were active during the benchmark;
  when cumulative prefill read-time telemetry is available, the profile stores
  planned read seconds, measured copy seconds/cap status, configured caps, and
  raw/coalesced range-count evidence in `sections.prefill_actual_read_time` for
  audit while leaving replay argv
  ownership with the guard sections. Decode step telemetry is stored similarly
  in `sections.decode_actual_read_time`, recording actual decoded
  expert-read bytes and SSD seconds against the profile's decode seconds/token
  cap, plus whether actual bytes stayed within runtime-preflight planned bytes.
  Benchmark-derived launch profiles also preserve
  `sections.prefill_actual_acceleration_coverage` and
  `sections.prefill_actual_acceleration_frontier` when batch prefill reports
  them, carrying the actual resident-GEMM backend counts, accelerated FLOP
  fraction, MPP candidate diagnostics, and candidate chunk frontier alongside
  the measured run. Strict launch-audit replay cross-checks actual coverage
  against `sections.prefill_actual_linear_backend` when both are present, so the
  measured backend counts, FLOP buckets, accelerated backend names, and FLOP
  fraction cannot be spliced from different runs.
  Launch-audit generation still requires prefill acceleration by default. For
  bounded backend A/B experiments where acceleration is intentionally disabled,
  `--allow-non-accelerated-prefill-launch-audit` marks the acceleration gate as
  non-required while preserving the same prepared identity, memory, runtime
  preflight, SSD read, stage-temp, decode, and request-envelope checks.
  `bench-prepared-token-ids --write-launch-profile` writes the same
  `source="benchmark_actual"` JSON for the next launch. It also includes
  `routed_chunk_frontier` for the same prompt when batch prefill ran, so
  benchmark telemetry and `inspect-prepared` request checks can compare chunk
  choices on the same SSD-read/stage-temp surface.
- `--metal-final-logits` is available on token/text/prepared/benchmark
  generation commands. It keeps the same streamed top-k contract while moving
  final RMSNorm and chunked `lm_head` matvecs to the Metal runner.
- `serve-prepared` packages the prepared generation path as a conservative
  local HTTP boundary. Each request goes through the prepared manifest and
  runtime guards, including a forced runtime memory preflight during request
  admission, requests are serialized around the shared decode cache, and the
  process does not retain a model-wide expert cache outside the OS page cache.
  Safety and generation admission failures are returned as HTTP 400 JSON
  errors rather than HTTP 500, so callers can lower prompt/token limits and
  retry. JSON integer controls are strict: floats, booleans, and strings are
  rejected rather than coerced for `max_new_tokens`, `logits_top_k`, `seed`,
  OpenAI `max_tokens`, and `n`. JSON float controls such as `temperature` and
  `top_p` must be finite. `/health` includes effective context/prompt limits,
  configured live guards, `memory_guard.available_ok`,
  decode/cache/scratch runtime caps, prepared storage totals, any bound
  `launch_audit_envelope`, a
  lightweight prefill-backend capability summary with warnings, and a
  best-effort system memory snapshot for pre-request headroom checks. When a
  launch audit envelope is bound, per-request inspection reports the matched
  prompt/generation envelope and rejects over-envelope requests before runner
  work can begin. Token
  generation responses include top-level `prefill_actual_read_time` when batch
  prefill records routed expert read seconds. They also include
  `decode_actual_read_time` when decode steps run under a runtime guard with a
  positive SSD GiB/s estimate, exposing actual decoded expert-read bytes/seconds
  and the configured cap result, plus whether actual bytes stayed within
  runtime-preflight planned bytes. The
  OpenAI-compatible completion endpoint is single-candidate and non-streaming
  for the same reason. The chat endpoint uses tokenizer-provided templates or a
  local model-shipped Jinja chat template only; it does not synthesize
  model-specific chat formatting.

These constraints are intentionally conservative. They protect unified memory
and keep the OS page cache available for actual inference.

## Baseline Format

MLX and Metal verification share a small raw-tensor baseline format:

```text
baseline/
  manifest.json
  tensors/
    000001_layer.1.router_logits.bin
    000002_layer.1.hidden_post_norm.bin
```

The manifest records model type, prompt/generated tokens, semantic record kind,
layer/token indices, tensor dtype, shape, byte size, and SHA-256. Tensor payloads
are raw little-endian bytes. This avoids a NumPy dependency in the Metal side
while still being easy for Python/MLX to write. `validate-baseline` rejects
malformed manifests, absolute or parent-traversal tensor paths, invalid shapes,
bad hashes, byte-count mismatches, and SHA-256 mismatches before any runner uses
those payloads. Baseline tensor payloads and the manifest are written through
temporary files and atomic replace so a failed tiny-baseline export cannot leave
partial artifacts that validate later by accident.

## Baseline Export Policy

`export-mlx-baseline` is intentionally gated:

- Without `--execute`, it only runs a preflight and never imports MLX or loads a
  model.
- With `--execute`, it refuses checkpoints above the configured whole-model load
  limit, defaulting to 64 GiB.
- It writes at most 8 generated-token steps and bounds each captured tensor by
  `--max-tensor-mib`.

This command is for small correctness harnesses. It must not be used to load
GLM-5.2 as a whole model. The production GLM path needs streaming baselines that
load one layer or one packed component at a time.

## Milestones

1. Config and checkpoint planner.
2. GLM expert packer that emits `layout.json` and per-layer files.
3. MLX baseline loader using oMLX/`mlx-lm` GLM patch to verify tokenizer,
   numerics, and routing against the original checkpoint.
4. Metal decode runner for one layer, then full greedy decode. The first
   runner milestone only validates layout parsing, bounded expert-slot reads,
   2 MB aligned buffers, `newBufferWithBytesNoCopy`, and a small 4-bit affine
   dequant matvec kernel. It can also read baseline manifests and verify tensor
   payload sizes before numerical comparison is added.
   The next runner milestone runs a single routed expert from a packed slot:
   gate projection, up projection, SwiGLU, and down projection, using component
   offsets from `layout.json`. The runner now validates the affine-int4
   `component_order`, component dtypes, packed offsets, byte sizes, and slot
   span total before executing an expert. It also rejects escaped layer-file
   paths, non-integer slot metadata, byte-size overflows, and expert layer files
   whose actual size differs from `num_experts * expert_slot_bytes`, so direct
   runner invocations get the same packed-slot contract enforced by prepared
   GLM readiness. Resident-weight runner paths apply the same backing-file
   policy to `resident/layout.json`: `weight_file` must be a relative file name,
   `total_bytes` must be a positive integer, and the actual resident file size
   must match before router, attention, dense MLP, or final-logits kernels read
   resident tensors.
   The current MoE milestone streams several top-k experts through one reusable
   aligned slot buffer, computes each expert output, and accumulates weighted
   outputs into one hidden-size buffer. It deliberately avoids long-lived expert
   caches until routing telemetry proves a cache is worth its memory cost.
   The router milestone reads only the current layer's resident router tensor
   from `resident.bin`, computes logits on Metal, and performs CPU top-k plus
   normalized routing weights. It does not map or read the full resident blob.
   The current layer-MoE milestone combines those pieces: `--run-layer-moe`
   computes router top-k, validates router/expert shape agreement, streams only
   the selected expert slots through one reusable 2 MB aligned buffer, and
   writes the hidden-size MoE output. Its CLI keeps explicit `--max-k`,
   `--max-router-mib`, and `--max-slot-mib` limits so single-layer tests cannot
   accidentally become whole-model loads.
   The router semantics milestone adds resident-layout router metadata,
   correction-bias-aware selection, group top-k, `norm_topk_prob`, and routed
   scaling factor support. The same router/RoPE semantic fields are typed in
   `ModelConfig`, so bad config values are rejected before decode or prefill
   creates work files.
   The shared-expert milestone adds `--include-shared-expert`, which runs the
   current layer's resident shared MLP and includes its scratch/matrix peak in
   `check-runtime`.
   The MLP block milestone adds `--run-mlp-block`, covering
   post-attention RMSNorm, routed/shared MoE, and residual add for one layer.
   The resident-linear milestone adds `--run-resident-linear`, a bounded
   F32/BF16/F16 current-layer matvec path for attention/indexer projections.
   The attention-projection milestone adds `--run-attn-projections`, covering
   input RMSNorm, Q low-rank projection, KV-A projection, Q/KV latent RMSNorms,
   and Q/KV-B projection.
   The decode-cache layout milestone adds `plan-cache`, a non-allocating JSON
   metadata pass for MLA/DSA cache offsets, per-token strides, and total byte
   budget.
   The cache-append milestone connects `--run-attn-projections` to that layout
   and writes the current token KV-A vector into the cache with a small bounded
   `pwrite`.
   The RoPE milestone adds `--run-rope`, covering standalone Metal rotary
   embedding for Q-rot/K-rot with both default and interleaved rotate semantics.
   The MLA attention milestone adds `--run-mla-attention`, a bounded
   correctness kernel for single-token score/value accumulation from the
   compressed cache and resident KV-B matrix.
   The attention-output milestone adds `--run-attn-output`, covering resident
   output projection plus residual add after attention.
   The attention-output server milestone adds
   `--run-attn-output-batch-server-jsonl` and the matching Python/prompt/
   prepared opt-in flags for process-fusion experiments; locked GLM evidence
   keeps it off the selected replay path for now.
   The attention-block smoke milestone verifies the attention-side commands can
   be composed end to end for one tiny token without loading a model.
   The decoder-layer smoke milestone verifies that the attention residual output
   can feed the bounded routed/shared MLP block for one tiny token.
   The decoder-layer command milestone adds `--run-decoder-layer`, a single
   runner entry point for the same bounded one-layer path.
   The dense-MLP milestone adds `--run-dense-mlp-block` for GLM-5.2's initial
   dense layers, reusing resident gate/up/down matvecs, SwiGLU, and residual
   add under the same matrix/scratch caps as the shared expert path.
   The prefill dense-MLP milestone adds `prefill-dense-mlp-block-batch`, a
   prompt-chunk wrapper around batch resident norm/linear, row-streamed SwiGLU,
   and row-streamed residual add for those same dense-prefix layers.
   The prefill routed-MLP milestone adds `prefill-routed-mlp-block-batch`, a
   serial but memory-bounded prompt-chunk routed/shared MoE baseline that
   preserves router semantics and read-advice flags while exposing per-token
   router JSON for the future coalesced expert scheduler.
   The batch expert-I/O milestone adds `plan-batch-expert-io`, which aggregates
   those router JSON files by expert and reuses the expert range planner to
   quantify coalesced SSD reads across the prompt chunk.
   The batch expert-staging milestone adds `stage-batch-experts`, which copies
   those planned ranges into a bounded stage file and manifest for the future
   batched expert executor.
   The staged routed-MoE execution milestone adds
   `run-staged-routed-moe-batch`, which consumes that manifest, builds a compact
   selected-expert layout, and runs a bounded routed-only batch output through
   one `--run-moe-batch` runner invocation that executes compact slots
   expert-major with bounded token-block 4bit kernels while row-streaming
   hidden states and output accumulation.
   The staged prefill routed-MLP milestone adds
   `prefill-staged-routed-mlp-block-batch`, which wraps batch router, staging,
   staged routed MoE, and residual add around batch RMSNorm for prompt chunks.
   The dense-decoder milestone adds `--run-dense-decoder-layer` and a mixed
   dense/MoE Python scheduler so GLM-5.2 style layer 0-2 dense plus routed MoE
   layers can be preflighted and executed in one ordered decode pass.
   The config-aware packing milestone keeps dense `switch_mlp` tensors resident,
   excludes them from routed expert packing, and reports dense MLP coverage in
   preflight before any large copy begins.
   The decode-layers milestone adds a Python scheduler that chains multiple
   packed decoder layers with per-layer runtime preflight before each Metal
   invocation. Its records now carry the preflighted routed-expert read bytes,
   attention/cache read bytes, and peak scratch estimate for each layer, and
   token generation attaches those records to each generated step for server
   telemetry.
   The final-logits milestone adds bounded row-chunk streaming for final
   RMSNorm plus lm_head/tied-embedding top-k.
   The embed-token milestone adds bounded row streaming for token embeddings,
   closing a tiny token-id-to-top-k decode loop over packed weights. The
   embed-tokens-batch milestone extends that input boundary to prompt chunks
   by streaming selected embedding rows into one `[tokens, hidden]` f32 batch.
   The generate-token-ids milestone adds a greedy token-id loop that advances
   decode-cache positions and emits generated ids with the same bounded
   primitives.
5. MPP prefill kernels for resident dense/attention/indexer operations.
6. Routing telemetry and optional expert clustering.
7. Continuous batching, prefix sharing, and richer server scheduling.
