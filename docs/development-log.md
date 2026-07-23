# LargerLM

LargerLM is a local inference stack for Apple Silicon machines that need to run
MoE LLMs larger than unified memory.

The first target is GLM-style `glm_moe_dsa` checkpoints, with 4-bit routed
experts stored on SSD and only the active experts streamed per token. The
current GLM-5.2 MXFP4 route is sealed as a minimum runnable version rather than
an active 5 tok/s optimization project: measured decode is around `0.9-1.0 tok/s`
and the best evidence-backed projection is `1.485 tok/s`, below the `5 tok/s`
continuation threshold. See `docs/minimal-usable-seal.md` for the stop decision,
safe runnable entry points, and restart criteria.

For a concise publishable guide with device requirements, principles, safety
guards, and usage commands, start with
[`docs/proof-of-feasibility-guide.md`](docs/proof-of-feasibility-guide.md).

## Current Status

- Hardware-aware model budget planner.
- GLM/MoE config parser.
- Safetensors header scanner for resident-vs-routed byte accounting.
- Header-only GLM checkpoint preflight for config, tokenizer, tensor coverage,
  expert packing coverage, decode-cache budget, and disk budget.
- Low-memory expert and resident packers.
- Metal runner smoke path for router top-k followed by streamed top-k MoE on a
  single layer.
- Bounded Metal attention projection smoke path for GLM MLA-style
  `q_a/q_b/kv_a/kv_b` resident weights.
- Tiny decoder-layer smoke that composes attention projection/cache/RoPE/MLA
  score-value/output projection with the routed/shared MLP block.
- Expert-major staged MoE batch execution that reads each selected compact
  expert slot once while streaming prompt-token rows through bounded buffers.
- Locked 512-token GLM-5.2 MXFP4 prompt-prefill replay evidence for the
  `--max-cache-read-mib 320` candidate: the audit-bound replay generated one
  token without exceeding the 17.37 GiB live-working-set cap, and the matching
  selected replay passes no-weight memory, disk, SSD, profile, and audit checks.
  This remains an explicit router-gate-only MPSGraph experiment, not the default
  serving profile.
- The 2048-token `cache320` long-prompt replay now completes under the locked
  launch profile/audit: 16 128-token chunks, generated token `[15]`, 3744.429s
  total, and the same 17.37 GiB live-working-set cap with 77.66 GiB available
  at admission. The DSA indexer now has optional NumPy batch fast paths for
  cache writes and top-k scoring with pure-Python fallback. This moves 2048 from
  "audited but interrupted" to runnable evidence, while DSA remains the next
  performance target at 1585.7s of the 2048-token run. The follow-up
  visible-context DSA optimization now skips prompt top-k generation when
  `context_length <= index_topk` while still writing the future decode DSA cache;
  target-size microbenching shows an 8.11x DSA chunk speedup for that branch,
  and a real 128-token direct prefill produced no `dsa_topk` files with 3.40s
  total DSA cache-write time. The full 2048-token locked replay after this skip
  completed, generated `[15]`, and dropped total time to 2748.777s while prompt
  DSA time fell to 193.168s with zero prompt `dsa_topk` files. `result-bakeoff`
  now selects this replay as the current minimum runnable 2048 path, and
  `selected-replay-check --check-ssd-read-speed` passes 41/41 pre-run checks.
  A follow-up driftx4 router-hybrid policy looked promising in streamed 2048
  telemetry, but the locked 512-token validation was 1.72x slower with the same
  `[15]` token, so router hybrid is not the next promotion path. `result-summary`
  now flags the real next orchestration target as
  `runner_process_fusion[runner_process_orchestration]=8784cmds` on the 2048
  replay, led by resident-linear, attention-output, MLA, projection, RoPE, and
  RMSNorm runner groups. The first opt-in prototype for that target is a
  resident-linear batch JSONL server, `--run-resident-linear-batch-plan-server-jsonl`,
  plus a Python `ResidentBatchLinearServerSession`; it is now wired as an
  explicit prompt-prefill/prepared/server opt-in via
  `--persistent-resident-linear-server` or
  `--prefill-persistent-resident-linear-server`, while the default path remains
  the old subprocess baseline for locked replay comparison. A real locked
  128-token replay with this opt-in now runs and reports
  `persistent_linear_server=yes`, keeps generated token `[15]`, and passes
  selected-replay/audit/SSD checks. In the current hot-system rerun window it
  tied the old resident-linear subprocess path (`196.051s` versus `196.818s`),
  so it is replay-ready bring-up evidence rather than a promoted fast path.
  The next narrower prototype, `--run-attn-projections-server-jsonl`, is now
  wired through Python, raw prompt prefill, prepared generation, serving, and
  launch profiles as `--persistent-attention-projection-server` /
  `--prefill-persistent-attention-projection-server`. It now has a real locked
  128-token GLM-5.2 replay: profile SHA
  `3d4dd88308d0171249a660ae9e06f446daa3308b36d7baf713f42fddd46680a8`
  generated `[15]`, stayed inside the 17.37 GiB live cap, and collapsed
  attention projection launches into one JSONL server group. The projection
  subphase improved (`18.228s -> 8.445s` versus the old memory rerun), but total
  time did not: `200.255s` versus `196.818s` old memory rerun and `196.051s`
  resident-linear-only. The bakeoff
  `prefill-128-attnproj-server-vs-rerun-bakeoff-current.json` keeps the old
  memory baseline; this remains useful bring-up evidence, not a promoted fast
  path. The next tiny process-fusion piece,
  `--run-rope-split-batch-server-jsonl`, is now implemented and wired as
  `--persistent-rope-split-server` / `--prefill-persistent-rope-split-server`.
  It reuses one Metal process for prompt-prefill fused RoPE split requests and
  has passed the Python prompt/token/server regressions, the tiny Metal smoke,
  and a same-window locked 128-token GLM replay bakeoff. The selected candidate
  profile SHA is
  `81d9e6888344b5013e24e32ea14d8e0370e91e83960d088acf3a94eb71d85ba9`; it
  generated `[15]`, stayed inside the 17.37 GiB live cap, collapsed RoPE split
  launches from 78 one-shot groups to one JSONL server group, and finished in
  `114.796s` versus the no-RoPE-server rerun at `148.846s` (`0.771x`).
- The next process-fusion boundary, resident batch RMSNorm, is now available as
  `--run-rmsnorm-batch-server-jsonl` with Python
  `ResidentBatchRMSNormServerSession` plumbing. Raw prompt prefill can opt in
  with `--persistent-rmsnorm-server`, while prepared generation, inspection,
  serving, and launch profiles use `--prefill-persistent-rmsnorm-server`.
  Validation now includes the targeted Python regressions, `make -C metal
  largerlm-runner`, the Apple M5 Max Metal smoke, and a real locked GLM replay
  gate. The 128-token candidate was rejected by the request admission guard
  because the current safety-capped maximum is 126 tokens, and the same-window
  126-token bakeoff kept the baseline: both runs generated `[15]` inside the
  17.37 GiB live cap, but RMSNorm server took `220.614s` versus `211.945s`
  (`1.041x`, `+8.670s`). Keep this as implemented negative evidence, not a
  promoted process-fusion fast path.
- MLA attention process fusion is now the selected 126-token GLM-5.2 MXFP4
  process-fusion replay. `metal/largerlm-runner
  --run-mla-attention-batch-server-jsonl` is wired through Python as
  `MLAAttentionBatchServerSession`; raw prompt prefill uses
  `--persistent-mla-attention-server`, and prepared generation, inspection,
  serving, plus launch profiles use
  `--prefill-persistent-mla-attention-server`. Validation includes the targeted
  Python regressions (`907 passed, 3 skipped`), `make -C metal largerlm-runner`,
  the Apple M5 Max MLA attention one-shot/server smoke, a locked launch
  audit/profile, and a real 126-token GLM replay. The candidate generated `[15]`,
  stayed inside the 17.37 GiB live cap with about 77.40 GiB available, finished
  in `132.394s`, and beat the RoPE split baseline at `211.945s` (`0.625x`,
  `-79.551s`). `result-bakeoff` selected it and wrote
  `selected-replay-prefill-126-persistent-moe-resident-linear-attnproj-rope-split-mla-server-memory-accumulator.json`;
  `selected-replay-check --check-ssd-read-speed` passes, with current bounded
  SSD read speed at 18.831 GiB/s (`3.184x` of calibration). The audit uses the
  explicit non-accelerated-prefill exception because this 126-token profile is
  the custom-metal fallback path, not the MPSGraph/MPP path.
- Attention-output process fusion is now implemented as an opt-in JSONL server,
  but it is not promoted. `metal/largerlm-runner
  --run-attn-output-batch-server-jsonl` reuses one Metal process for bounded
  batch `o_proj + residual` requests; Python drives it through
  `AttentionOutputBatchServerSession`, raw prompt prefill exposes
  `--persistent-attention-output-server`, and prepared generation, inspection,
  serving, plus launch profiles expose
  `--prefill-persistent-attention-output-server`. Validation includes
  `913 passed, 3 skipped` for the targeted Python regressions, `make -C metal
  largerlm-runner`, `python metal/attention_output_smoke.py`, and a locked
  126-token GLM replay. The candidate generated `[15]`, stayed inside the
  17.37 GiB live cap, and collapsed attention-output runner groups to one
  `--run-attn-output-batch-server-jsonl` group. It improved the `o_proj`
  tensor time (`12.882s -> 9.561s`), but the full replay was slower:
  `194.082s` versus the selected MLA baseline at `132.394s`
  (`1.466x`, `+61.688s`). The bakeoff
  `prefill-126-attnout-server-vs-mla-attention-server-bakeoff-current.json`
  retained the current MLA selected replay, and
  `selected-replay-check --check-ssd-read-speed` still passes for that replay
  with current bounded SSD speed at 5.808 GiB/s.
- Shared-expert process fusion is now implemented as an opt-in JSONL server,
  but it is also not promoted. `metal/largerlm-runner
  --run-shared-expert-batch-server-jsonl` reuses one Metal process for fused
  shared gate/up/down MXFP4 prompt-prefill requests; Python drives it through
  `ResidentSharedExpertBatchServerSession`, raw prompt prefill exposes
  `--persistent-shared-expert-server`, and prepared generation, inspection,
  serving, plus launch profiles expose
  `--prefill-persistent-shared-expert-server`. Validation includes the targeted
  Python regressions (`839 passed, 3 skipped`), `make -C metal largerlm-runner`,
  `python metal/shared_expert_batch_smoke.py`, and a locked 126-token GLM
  replay. The candidate generated `[15]`, stayed inside the 17.37 GiB live cap,
  and reduced `mlp.shared_experts` tensor time from `8.503s` to `2.839s` while
  dropping unique runner commands from 235 to 161. Total latency still landed at
  `133.308s` versus the selected MLA baseline at `132.394s` (`1.007x`,
  `+0.914s`), so
  `prefill-126-shared-expert-server-vs-mla-attention-server-bakeoff-current.json`
  retained the current baseline with `total_elapsed_within_two_percent`. A
  follow-up same-window rerun made the decision clearer: the baseline replay
  took `141.647s`, the shared-server replay took `194.353s`, and
  `prefill-126-shared-expert-server-rerun-vs-baseline-bakeoff-current.json`
  retained the baseline again (`1.372x`, `+52.705s`). The current routed-MoE
  layer-67 bounded sweep also found no tile/vector/group32 candidate:
  `glm-moe-layer67-optimization-target-126tok.json` kept `tile1_auto_silu` as
  the fastest kernel, so the next MoE work should target structural
  orchestration/kernel behavior rather than those local toggles. On the current
  same-window baseline rerun, the minimal 126-token GLM-5.2 MXFP4 smoke is
  runnable but slow: 126 prompt tokens plus one generated token took `141.647s`
  (`141.419s` prefill, about `0.89` prompt tokens/s) under the audited launch
  binding. The result's prepared memory profile estimates an `18.65GB` live
  working set while requiring at least `24GB` free unified memory before weight
  loading, so this is safe smoke evidence rather than a usable chat-speed
  target, especially on a power-limited adapter.
- The high-performance path is now being split out into a Flash-MoE-style
  single-process GLM runtime instead of continuing to optimize Python/file
  orchestration as the final token loop. The first executable skeleton,
  `metal/glm_moe_infer`, initializes Metal, validates the prepared GLM-5.2
  resident/expert layouts, opens all 75 routed expert layer files, allocates
  reusable 2MB-aligned expert buffers, optionally mmaps `resident.bin`, reports
  the live-memory envelope, and exits before inference. On the local M5 Max
  package it reports `Apple M5 Max`, `358.594GiB` of routed expert files,
  `9.366GiB` of resident weights, and a successful resident-mmap live envelope
  of `10224752128` bytes under a 20GiB cap. The first real expert-I/O probe also
  passes: layer 67 experts `34,36,89,149,152,183,201,206` were read directly
  into eight aligned Metal shared buffers with only `167772160` estimated live
  bytes, reading `160432128` bytes in `0.012959s` (`11.53GiB/s`). This validates
  the Flash-MoE-style slot-read path. A full 75-layer read probe with the same
  8-expert pattern read `11475.000MiB` in `0.997703s` (`11.23GiB/s`) while still
  holding the live envelope at `0.156GiB`; this puts the GLM-5.2 cold expert-I/O
  lower bound near 1 second per decode token before math, attention, and logits.
  The static comparison tool
  `scripts/flash_moe_efficiency_envelope.py` now makes that bound reproducible
  without opening weights: the local prepared GLM-5.2 package needs
  `12032409600` routed expert bytes/token (`11.206GiB`), which is `7.08x`
  Flash-MoE's documented Qwen reference shape. At the manifest cold-read
  calibration of `5.915GiB/s`, the pure expert-I/O floor is `1.895s/token`; at
  the direct all-layer probe rate of `11.23GiB/s`, it is `0.998s/token`.
  The runtime has also crossed the first compute parity gate:
  `--probe-layer-moe` runs a one-layer group32 MXFP4 routed expert forward in
  `glm_moe_infer` itself. `metal/glm_moe_infer_mxfp4_moe_smoke.py` compares the
  tiny layer-1 fixture against the known MXFP4 output and passes with max output
  error about `4.6e-6` under an 8MiB cap. That tiny gate now requests two
  reusable expert buffers, reports `4194688` estimated live bytes, and verifies
  that the routed read path used one pooled dispatch with two pread tasks and
  two persistent workers. The same path now also has a real GLM-5.2 layer
  parity smoke:
  `metal/glm_moe_infer_real_layer_moe_smoke.py` compares layer 67 experts
  `34,36,89,149,152,183,201,206` against `metal/largerlm-runner` and matches all
  6144 output floats exactly (`max_abs_diff=0.0`) while the new runtime reports
  `160432128` expert bytes read, `0.014685s` elapsed, `0.004472s` expert-read
  time, `0.009976s` kernel time, and `167829504` estimated live bytes under a
  256MiB cap. Its default path now uses eight reusable expert buffers and
  asserts one pooled read dispatch with eight pread tasks, eight persistent
  workers, and zero serial fallbacks. After reviewing Flash-MoE's row-tiled
  Metal kernels, an opt-in
  fast MXFP4 expert path is now the default for supported GLM-5.2 group32
  expert shapes; set `LARGERLM_GLM_MOE_INFER_FAST_MXFP4=0` (or `off` /
  `scalar`) to force the older scalar parity path. On the same real layer-67
  routed MoE smoke it matches the old runner within
  `1.199040866595169e-14` and reports `0.007381s` expert kernel time. The
  production router gate also has parity coverage now:
  `metal/glm_moe_infer_real_router_smoke.py` compares
  `glm_moe_infer --probe-router` with `metal/largerlm-runner --run-router` on
  the same deterministic layer-67 hidden vector, using the prepared GLM router
  metadata: sigmoid scores, correction bias, normalized top-k, and routed scale
  `2.5`. The top-k experts `240,96,27,174,243,108,109,101`, normalized and
  scaled weights, and all 256 logits match exactly (`max_logit_diff=0.0`,
  `max_weight_diff=0.0`). For router-only probes the new runtime now activates
  zero expert buffers despite the default request for eight, so the measured
  live envelope is only `4221952` bytes under a 64MiB cap. Router selection is
  now wired into routed expert compute in the same process:
  `metal/glm_moe_infer_real_router_moe_smoke.py` compares
  `glm_moe_infer --probe-router-moe` with the old runner oracle formed by
  `--run-router` plus `--run-moe`. The layer-67 router logits/weights and all
  6144 routed-MoE output floats match exactly (`max_abs_diff=0.0`), while the
  new path activates one expert buffer and reports `25250816` estimated live
  bytes under a 64MiB cap. `--probe-mlp-block` now adds post-attention RMSNorm,
  production routing, routed expert compute, optional resident MXFP4 shared
  expert, and residual add around that routed core.
  `metal/glm_moe_infer_real_mlp_block_smoke.py --include-shared-expert`
  compares it with `metal/largerlm-runner --run-mlp-block
  --include-shared-expert` on the same layer-67 input; router logits/weights
  match exactly and all 6144 output floats match within
  `1.0477378964424133e-09`. The shared MLP probe reads the 12KiB BF16 norm
  vector, `160432128` routed expert bytes, and `20054016` resident shared
  expert bytes, activates one reusable expert buffer, and reports `25336832`
  estimated live bytes under a 64MiB cap. With
  `LARGERLM_GLM_MOE_INFER_FAST_MXFP4=1`, the same shared MLP block keeps router
  logits/weights exact, matches final hidden within `1.4551915228366852e-09`,
  and reports `0.007613s` routed kernel plus `0.000774s` shared kernel under the
  same live cap. The same runtime now has a decoder
  attention building block: `--probe-resident-linear` runs a resident MXFP4
  matrix-vector projection without allocating expert buffers.
  `metal/glm_moe_infer_resident_mxfp4_linear_smoke.py` matches the old runner
  tiny fixture exactly under an 8MiB cap, and
  `metal/glm_moe_infer_real_resident_linear_smoke.py` matches the real
  `model.layers.67.self_attn.q_a_proj.weight` output exactly
  (`max_abs_diff=0.0`) while reading `6684672` resident bytes and reporting
  `8421376` estimated live bytes under a 16MiB cap. That primitive is now
  composed into `--probe-attn-projections`: input RMSNorm, q_a/q_b/kv_a
  resident projections, q_a RMSNorm, and kv_a prefix RMSNorm all run in
  `glm_moe_infer` and write the old runner's single-token attention projection
  files. `metal/glm_moe_infer_real_attn_projections_smoke.py` compares layer 67
  against `metal/largerlm-runner --run-attn-projections`; all six output files
  match exactly (`max_abs_diff=0.0`), the new runtime reads `26407936` resident
  bytes, activates zero expert buffers, and reports `29615360` estimated live
  bytes under a 64MiB cap. `--probe-rope-split` now takes those projection
  outputs and runs the q_b split plus q/k RoPE rotation in Metal.
  `metal/glm_moe_infer_real_rope_split_smoke.py` compares the real layer-67
  path against `metal/largerlm-runner --run-rope-split-batch`; `q_nope`,
  `q_rope`, rotated `q_rope`, and rotated `k_rope` all match exactly
  (`max_abs_diff=0.0`), with zero expert buffers and only `147968` estimated
  live bytes under an 8MiB cap. KV-cache movement is now covered by
  `--probe-attn-projections` as well: when `--cache-layout`, `--cache-file`,
  `--position`, and `--max-cache-file-mib` are supplied, it appends the
  single-token KV-A row to the decode cache under the same live cap.
  `metal/glm_moe_infer_real_attn_cache_smoke.py` compares layer 67 against the
  old runner and verifies the six projection files at `max_abs_diff=0.0` plus a
  byte-identical BF16 cache write of `1152` bytes at position 2. The cache
  append run activates zero expert buffers and reports `29617664` estimated live
  bytes under a 64MiB cap. `--probe-mla-attention` covers the next single-token
  MLA attention step. `metal/glm_moe_infer_real_mla_attention_smoke.py` chains
  the verified projection, cache, and RoPE outputs, then compares layer 67
  against `metal/largerlm-runner --run-mla-attention`; all 16384 attention value
  floats match exactly (`max_abs_diff=0.0`). The probe uses the absorbed
  `embed_q`/`unembed_out` value source, reads `2304` F32 cache bytes plus
  `7798784` stored resident value bytes, activates zero expert buffers, reports
  `125374976` estimated live bytes under a 192MiB cap, and measured
  `0.012330s` MLA kernel time with `0.016223s` value-read time on the local
  Apple M5 Max. `--probe-attn-output` now applies resident MXFP4 `o_proj` and
  residual add. `metal/glm_moe_infer_attn_output_smoke.py` covers the tiny
  semantics gate, and `metal/glm_moe_infer_real_attn_output_smoke.py` compares
  the real layer-67 projection and final hidden output against
  `metal/largerlm-runner --run-attn-output`; both files match exactly
  (`max_abs_diff=0.0`). The real probe reads `53477376` resident bytes,
  activates zero expert buffers, reports `54640640` estimated live bytes under a
  192MiB cap, and measured `0.003824s` `o_proj` kernel time with `0.002943s`
  residual-add time. `--probe-decoder-layer` now composes the single-token
  layer path in one `glm_moe_infer` process: attention projections with cache
  append, RoPE split, MLA attention, attention output, post-attention RMSNorm,
  router, routed MoE, optional shared expert, and residual add.
  `metal/glm_moe_infer_real_decoder_layer_smoke.py --include-shared-expert`
  compares real layer 67 against `metal/largerlm-runner --run-decoder-layer`;
  router logits/weights match exactly and all 6144 final hidden floats match
  within `1.2516975402832031e-06`. The current no-shared performance-path run
  activates eight reusable expert buffers, writes `2304` F32 cache bytes,
  reports `381918720` estimated live bytes under a 512MiB cap, and measures
  `0.057084s` decoder-layer elapsed with `0.009544s` MLA kernel,
  `0.001522s` `o_proj` kernel, `0.004394s` expert-read, and `0.003170s`
  routed-expert kernel time. Its MLP read telemetry reports one pooled dispatch,
  eight pread tasks, eight persistent workers, and zero serial fallbacks. The
  dense-prefix side is now covered too: `--probe-dense-mlp-block` runs GLM
  layers 0-2 style post-attention
  RMSNorm, resident MXFP4 gate/up/down, F32 SwiGLU, and residual add without
  opening expert files or allocating expert buffers.
  `metal/glm_moe_infer_real_dense_mlp_smoke.py` compares real layer 0 against
  `metal/largerlm-runner --run-dense-mlp-block`; all 6144 output floats match
  exactly (`max_diff=0`), the new runtime reads `120336384` resident bytes,
  reports `126418944` estimated live bytes under a 512MiB cap, and measured
  `0.109437s` dense-MLP elapsed. `--probe-dense-decoder-layer` then composes the
  full dense-prefix layer path: attention projections with cache append, RoPE
  split, MLA attention, attention output, and resident dense MLP in one
  `glm_moe_infer` process. `metal/glm_moe_infer_real_dense_decoder_layer_smoke.py`
  compares real layer 0 against `metal/largerlm-runner
  --run-dense-decoder-layer`; all 6144 final hidden floats match exactly
  (`max_abs_diff=0`), cache writes are byte-identical, expert files opened and
  expert buffers are both zero, estimated live bytes are `336200192` under a
  512MiB cap, and measured dense-decoder elapsed time is `0.135031s`. This is
  still component/single-layer evidence, not token generation. The first
  continuous layer-list driver is now present as `--probe-decode-layers
  --decode-layers CSV`. It preflights each layer, classifies dense layers by the
  absence of a packed expert layer, uses bounded multi-slot expert staging for
  MoE layers, and chains hidden states between layers in one `glm_moe_infer`
  process.
  `metal/glm_moe_infer_real_decode_layers_smoke.py --quiet-commands` compares
  real layers `0,3` against `metal/largerlm-runner --run-decoder-layers
  --dense-layers 0`, then asks the new runtime to stream final RMSNorm plus
  chunked MXFP4 `lm_head` top-k from the resulting hidden state in the same
  `glm_moe_infer` process. The latest local result: final hidden max diff is
  `1.4901161193847656e-08`, cache writes are byte-identical, final top-k token
  ids are identical (`140366,83824,103983,105455,33166,60863,40520,33189`),
  logits match within `2.86102294921875e-06`, the new runtime reads
  `120336384` dense-MLP resident bytes plus `160432128` routed expert bytes and
  `505540608` final-logits bytes, activates nine reusable expert buffers for
  top-k 8 plus shared expert, feeds the second layer from the prior layer's
  in-memory hidden vector
  (`memory_chain_bytes=24576`, `memory_input_layer_count=1`), and keeps each
  layer's hot q_b/K-RoPE/q_nope/q_rope/attention-value bridge in memory
  (`hot_intermediate_memory_bytes=393728`). With
  `--skip-debug-intermediates`, the same smoke asserts no files are written
  under the new runtime work directory (`debug_intermediates_written=false`).
  Final logits now also consumes the decode layer-list's in-memory final hidden
  instead of re-reading `--output-f32` (`source=decode_layers_memory_output`).
  The no-debug path fuses the pre-cache attention projection phase into one
  command buffer per layer (`attn_projection_command_buffers=[1,1]`), fuses
  RoPE split plus MLA attention into one command buffer per layer
  (`rope_mla_command_buffers=[1,1]`), and fuses attention output `o_proj`
  matvec plus residual add into one command buffer per layer
  (`attn_output_command_buffers=[1,1]`). The dense-prefix MLP now runs
  RMSNorm, gate/up MXFP4 matvecs, SwiGLU, down MXFP4 matvec, and residual add
  inside one command buffer (`dense_mlp_command_buffers=[1]`). The MoE MLP path
  now uses bounded multi-slot expert staging: with nine active expert buffers
  for top-k 8 plus shared expert, it reads the selected slots first and encodes
  all routed/shared expert work plus residual add into one command buffer
  (`moe_mlp_command_buffers=[1]`, `expert_buffer_count=9`). It reports
  `542163472` estimated live bytes under a 768MiB cap, and measured
  `0.124153s` for the two-layer decode probe plus `0.056369s` for final top-k.
  A follow-up final-logits-only mmap path, `--mmap-final-logits`, maps just the
  `lm_head` weight/scale ranges as Metal buffers instead of wrapping all
  resident weights. The bounded two-layer smoke with a 2048MiB live cap reports
  `1031513536` estimated live bytes, `final_logits_lm_head_bytes_read=0`,
  `final_logits_bytes_read=12288` for final-norm weight only, and preserves the
  normal attention-output timing (`0.020533s`) while final top-k takes
  `0.055212s`. A full `--wrap-resident-metal` experiment was negative evidence:
  it made final logits very fast (`0.004331s`) but made attention output cold
  mmap-backed and slow (`1.471s`), so the current resident mmap direction is
  selective mapping, not a blanket resident wrap.
  The `probe_decode_layers` JSON now also aggregates MoE telemetry across the
  layer list; the same run reports `160432128` routed expert bytes,
  `120336384` dense-MLP resident bytes, `0.004041s` total expert-read time,
  `0.003600s` MoE kernel time, and one pooled 8-task read dispatch with zero
  serial fallbacks.
  The same smoke now also has a `--all-layers --new-only` safety mode. It runs
  all 78 layers (`0,1,2` dense and 75 MoE layers) through the new runtime
  without the old-runner oracle, keeps all layer-local fused command-buffer
  invariants true, reads `12032409600` routed expert bytes plus `361009152`
  dense-MLP resident bytes, feeds 77 later layers from in-memory hidden states,
  keeps `expert_buffer_count=9`, reports `542163472` estimated live bytes under
  the same 768MiB cap, and now measures `2.829287s` for all-layer single-token
  decode plus `0.050884s` for final top-k on the current hot-cache M5 Max run.
  The first all-layer telemetry showed that expert streaming was no longer the
  dominant cost, and exposed a bad context=1 MLA shape: the old generic kernel
  recomputed the same head score for every value dimension. `glm_moe_infer` now
  dispatches a `glm_mla_attention_context1_f32` fast path that skips QK/rope
  scoring when the softmax has exactly one element. Real layer-67 validation
  matched the old MLA output exactly and cut the MLA kernel from about
  `0.021057s` to `0.003682s`; the real decoder-layer smoke reports
  `mla_attention_kernel_seconds=0.001020` with output max diff below `3e-7`.
  The next safe resident-matvec specialization adds a simdgroup group-32 MXFP4
  kernel for `self_attn.o_proj` by default. A broader resident-matvec fast path
  remains available behind `LARGERLM_GLM_MOE_INFER_FAST_RESIDENT_MXFP4=1`, but
  is not default because the real decoder-layer router-logit oracle drifted
  past the existing `1e-5` gate when every resident matvec used it. The
  remaining all-layer costs after KV-B cache hit are routed/dense MLP,
  attention projections, attention output, and the still-large expert read
  volume.
  A July 5, 2026 follow-up skipped the large `danveloper/flash-moe` weight trial
  at operator request, rechecked the source-only Flash-MoE implementation, and
  focused on this bottleneck instead. `glm_moe_infer`
  now has an explicit, opt-in in-process absorbed MLA KV-B F32 cache:
  `--cache-mla-kv-b-f32 --max-mla-kv-b-cache-mib N`. The cache is disabled by
  default, is keyed by resident path/layer/tensor offsets, is included in the
  live working-set admission estimate, and never evicts during an active run.
  Prepared-server health now derives the full-cache byte count from model
  config and, only when Metal runtime generation is selected, includes an
  opt-in launch-profile suggestion for these flags. For public GLM-5.2 this is
  `4368MiB` of layer data rounded to a conservative `4608MiB` cap; the server
  still leaves the cache off unless the launch args explicitly enable it.
  On the prepared GLM-5.2 MXFP4 package, a full 78-layer 2-token smoke with a
  4608MiB cache cap and 16GiB free-memory guard generated `[15,11]`, stored all
  78 layer value views on step 0, hit all 78 on step 1, used
  `4580179968` cache bytes, and reported decode times `3.357781s` then
  `2.731560s` under a `5374387216` byte live estimate. The no-cache control
  generated the same `[15,11]` with decode times `2.832896s` then `3.464909s`
  under a `542549008` byte live estimate. In other words, this cache is a real
  long-decode optimization, but very short generations pay the warmup cost.
  `scripts/decode_telemetry_report.py` can summarize either raw
  `glm_moe_infer` JSON, smoke `dims` JSON, `probe_generate.steps`, or the
  two-request prepared HTTP wrapper into the same bottleneck/pooled-read
  verdict. The report now also separates MoE expert read, MoE kernel, MoE
  output write, residual per-layer overhead, layer counts, command-buffer
  counts, the current synchronous-wait estimate, and explicit MoE command wait
  counts when the source artifact carries them. It now also accepts flat
  `generate-metal-token-ids` artifacts in addition to nested HTTP/probe step
  payloads. The all-layer old-runner oracle mode
  now also passes without downloading additional weights: the full 78-layer
  hidden output differs by at most `0.0002593994140625` under the `5e-4`
  accumulated-drift threshold, the F32 cache rows differ by at most
  `1.0967254638671875e-05`, final top-k ids are identical
  (`16,15,17,18,19,21,20,23`), and final top-k logits differ by at most
  `4.1961669921875e-05`. `--output-token-json` now writes the argmax generated
  token from that streamed top-k result without materializing full-vocab logits;
  the full-layer oracle reports `generated_token_id=16`. `--output-next-input-f32`
  now immediately streams that argmax token's MXFP4 embedding row into a
  6144-float next-step input; it reads only `3264` embedding bytes and matches
  the Python embedding oracle exactly (`max_abs_diff=0.0`). The latest oracle
  run measured `4.904478s` for all-layer decode plus `0.057639s` for final
  top-k while keeping the feedback-enabled live envelope at `542163472` bytes.
  A new two-step smoke,
  `metal/glm_moe_infer_real_two_step_decode_smoke.py`, reuses the same cache
  file across two decode calls: on all 78 layers it generates `[16,18]`, writes
  all 78 layer rows at cache positions `0` and `1`, keeps next-step embedding
  parity exact for both steps, and reports `4.923045s` then `5.480701s` decode
  time with live envelopes `542163472` and `542168080` bytes. This proves the
  cache/embedding feedback loop across tokens; folding the repeated calls into
  one persistent service process is now underway. `glm_moe_infer` also has a
  first in-process greedy loop via `--generate-steps`: the new
  `metal/glm_moe_infer_real_generate_steps_smoke.py --all-layers` run performs
  both decode steps inside one runtime invocation, generates `[16,18]`, writes
  all 78 cache rows at positions `0` and `1`, keeps the final generated-token
  embedding exact against the Python oracle, and reports `4.956343s` then
  `5.490593s` decode time under a `542168080` byte live envelope.
  The first Python CLI wrapper around that single-process loop is available as
  `python3 -m largerlm generate-metal-token-ids`. A local 2-token real-weight
  run on the prepared GLM-5.2 MXFP4 package generated `[15,11]`, reported
  `4.864067s` and `5.453250s` decode times, `10.575633s` end-to-end wall time,
  and kept the estimated live working set at `542168080` bytes under a 768MiB
  cap. The wrapper now sends decode requests through
  `glm_moe_infer --generate-server-jsonl` by default; pass
  `--no-generate-server-jsonl` to use the legacy direct request-json entry for
  debugging. Single-token prompts enter as `input_token_id`; multi-token prompts
  still require `--prefill-prompt`, but that flag now defaults to the runtime
  `prompt_token_ids` path instead of the older Python prefill bridge.
  `python3 -m largerlm generate-metal-text` now adds the first text-level entry:
  it loads the local tokenizer from the prepared manifest's model directory by
  default, encodes the prompt, runs the same bounded Metal loop, and decodes the
  generated ids. It also accepts `--chat-messages` / `--chat-messages-file` and
  renders the local tokenizer chat template before decode. Multi-token text
  prompts now automatically enable the safe runtime `prompt_token_ids` prefill
  path unless the caller explicitly opts into the old decode-only experiment
  with
  `--allow-decode-only-multi-token-prompt`, which uses only the last prompt
  token embedding. The lower-level token-id CLI still keeps
  `--prefill-prompt` explicit. The safe path sends
  `prompt_token_ids` to the persistent `glm_moe_infer --generate-server-jsonl`
  service, runs sequential prompt prefill inside that runtime, writes the
  decode cache through the service memory backend, takes the first generated
  token logits from the final prompt state, and continues generation without
  required `.f32` or generated-json debug files. This is a correct minimum
  runtime path, not the final Flash-MoE-style batched/overlapped prompt prefill
  path. Pass `--python-prefill-bridge` only to use the older bridge, where
  prompt prefill runs outside `glm_moe_infer` and hands
  `prefill_last_hidden.f32` back to generation. Results now report
  prompt-prefill elapsed time separately from the Metal generation elapsed time
  and Metal live envelope. Pass
  `--prefill-max-live-working-set-mib` to give the bridge prefill phase its own
  live cap while keeping `--max-live-working-set-mib` tight for logits/decode
  when using `--python-prefill-bridge`. Single-request token/text CLIs can opt
  into the new in-process MLA KV-B value cache with
  `--cache-mla-kv-b-f32 --max-mla-kv-b-cache-mib N`; prepared serving exposes
  the same persistent-runtime cache as `--metal-runtime-cache-mla-kv-b-f32`
  plus `--metal-runtime-max-mla-kv-b-cache-mib N`. `/health` reports the configured
  cache state and suggested launch-profile argv, and also recommends
  `--metal-runtime-mmap-final-logits` for the selective lm_head-only mmap path
  when Metal runtime generation is enabled. Metal runtime responses report
  per-step MLA value-cache hit/store counts plus resident cache bytes.
  The context=1 collapsed `o_proj*B_v` cache can be supplied to direct Metal
  generation with `--context1-o-proj-cache-layout PATH` and to prepared serving
  with `--metal-runtime-context1-o-proj-cache-layout PATH`; the optional
  `*-cache-file PATH` variants override the backing file recorded in the layout.
  Prepared serving validates the context1 cache layout, backing file size, and
  prepared-artifact dims/config match at startup, then reports the validated
  summary in `/health`. When the cache is configured with Metal runtime
  generation, `/health` also preserves the layout/file flags in
  `suggested_launch_profile` as `metal_runtime_context1_o_proj_cache_flags`,
  so replaying the profile keeps the attention-output collapse enabled. Metal
  generation/server responses and
  `scripts/decode_telemetry_report.py` now also carry
  `attn_output_context1_o_proj_cache_count`, so a benchmark can prove whether
  the collapsed cache was actually used.
  A prepared HTTP two-request smoke now confirms that cache reuse survives the
  HTTP boundary inside one persistent `glm_moe_infer` session. With
  `--metal-runtime-generation`, `--metal-runtime-cache-mla-kv-b-f32`,
  `--metal-runtime-max-mla-kv-b-cache-mib 4608`,
  `--max-live-working-set-mib 16384`, and
  `--min-free-unified-memory-gib 24`, two sequential
  `/generate-token-ids` requests for prompt token `[0]` both generated `[15]`.
  After the context=1 MLA fast path, the default `o_proj` resident simd matvec,
  direct `q_b`/`k_rope` Metal-buffer handoff, dense-prefix RoPE/MLA +
  attention-output fusion, cross-layer hidden-state Metal-buffer handoff, and
  direct current-token KV-A handoff into context=1 MLA,
  the latest July 5, 2026 rerun under the current power-limited local machine
  stored all 78 MLA KV-B views on the first request and took `2.114527s` decode
  time (`3.234041s` HTTP wall); the second request hit all 78 views and took
  `0.954184s` decode time / `1.497896s` HTTP wall, or `1.048 tok/s` raw decode,
  `1.012 tok/s` including final logits, and `0.668 tok/s` through the HTTP
  wrapper. The
  cache-hit breakdown is now explicit: attention projections, MLA attention,
  attention output plus fused RoPE/MLA/post-attention norm/router, MLP
  `0.390004s`, final logits `0.033632s`, and the now-shifted wait attribution
  from direct KV-A scheduling. Inside MoE MLP, expert read is `0.169454s`; the
  report keeps MoE kernel/write/overhead split in the artifact. The hot path
  now encodes RoPE split, MLA attention, and attention output in one command
  buffer for the dense prefix, and also includes post-attention RMSNorm/router/top-k
  in that command buffer on all 75 MoE layers
  (`rope_mla_attn_output_norm_router_fused_count=75`), so
  `rope_mla_command_buffer_count=0` and
  `post_attn_norm_router_command_buffer_count=0`, while
  `rope_mla_input_buffer_direct_count=78`,
  `attn_output_buffer_direct_count=75` and
  `moe_mlp_input_buffer_direct_count=75` confirm the intra-layer GPU-resident
  handoff. The new `layer_input_buffer_direct_count=77` confirms that every
  layer after the first now receives the previous layer's hidden state directly
  as a Metal buffer instead of materializing an `NSData` layer boundary.
  Each step reports `12032409600` routed expert bytes read, 600 expert read
  tasks, 8 persistent read workers, 78 layers (`3` dense plus `75` MoE), and a
  corrected all-stage 234 command buffers / 156 estimated synchronous waits.
  The direct routed-expert read path inside `glm_moe_infer` is now a shared
  runtime primitive: selected expert slots are range-checked, read with the
  persistent `pread` worker pool into reusable 2 MiB-aligned
  `newBufferWithBytesNoCopy` Metal buffers, marked modified once, and reported
  with dispatch/task/worker telemetry. The tiny MXFP4 smoke covers both the
  standalone `--probe-expert-read` path and the layer-MoE consumer, and rejects
  out-of-range expert ids before any slot read is issued. Resident MXFP4 linear
  probes can now opt into the same single-process resident-weight shape with
  `--mmap-resident --wrap-resident-metal`: the kernel reads weight/scales by
  offset from one mmap-backed Metal buffer, reports `resident_mmap_backed=1`
  and `bytes_read=0`, and the tiny resident-linear smoke verifies identical
  output against the staging fallback while reducing scratch from the rounded
  matrix staging buffer to input/output only. The diagnostic mmap-backed path
  now also reaches attention-output and dense decode composition: tiny smokes
  verify standalone fused `--probe-attn-output`, dense decoder fallback, and
  `--probe-decode-layers --skip-debug-intermediates` hot paths with
  `attn_output_resident_mmap_backed=1`, `attn_output_bytes_read=0`, and
  identical outputs. This is correctness and telemetry coverage, not a default
  performance choice; the real GLM MoE experiment below still shows direct GPU
  reads from a file-backed mmap are slower than bounded staging.
  A stricter
  6GiB live cap
  was correctly rejected before Metal generation because the resident backing
  alone is about 10.06GB; the successful smoke used a 16GiB live cap plus a
  24GiB free-unified-memory guard. The artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-direct-current-kva-latest.json`.
  A follow-up wait audit added per-layer attention-projection wait telemetry
  and confirmed that the previous cache-hit path still performed `234` real
  synchronous waits, not just `234` command-buffer submissions. The reason was
  structural: attention projection materialized KV-A on the CPU so it could
  append the current token to the decode cache before MLA read that same row.
  `glm_moe_infer` now feeds the current KV-A Metal buffer directly into fused
  MLA, then appends KV-A to the decode cache after the fused router wait for
  future tokens. For `contextLength>1`, it reads previous cache rows into a
  contiguous F32 view and blits the current KV-A row into the final slot inside
  the same command buffer before MLA. The guarded two-request HTTP smoke still
  generates `[15]`; on the cache-hit request it now reports `0.954184s` decode
  time, `0.033632s` final logits, `1.012 tok/s` including final logits, and
  `1.497896s` HTTP wall. Command buffers remain `234`, but real synchronous
  waits drop to `156`, and same-token MLA cache-read time for the context=1
  shape drops to microseconds because the current row is no longer written and
  read back through the CPU cache path. A guarded 2-token HTTP smoke validates
  the generalized path. A follow-up small-context MLA kernel now covers
  `contextLength=2..32` and avoids recomputing the same attention scores for
  every value dimension by assigning one threadgroup per head, computing the
  softmax once, and sharing it across value lanes. Prompt `[0]` still generated
  `[15,11]` in the 2-token smoke; step 1 decode drops to `1.635305s`, and the
  aggregate report shows `468` command buffers / `312` waits with `0.511 tok/s`
  including logits for two generated tokens. The 3-token smoke validates the
  `contextLength=3` path: prompt `[0]` generated `[15,11,15]`, step 2 decode
  drops from the baseline `3.054386s` to `1.918706s`, and the aggregate report
  shows `702` command buffers / `468` waits with `0.515 tok/s` including logits
  for three generated tokens. A guarded 5-token comparison against the previous
  `contextLength<=4` small-kernel baseline generated the same
  `[15,11,15,15,21]`; aggregate throughput improved from `0.398 tok/s` to
  `0.467 tok/s` including logits, and the contextLength=5 step dropped from
  `4.400035s` decode / `3.732975s` attention to `2.576679s` decode /
  `1.994487s` attention. A 9-token comparison then exposed the next fallback:
  with the `contextLength<=8` limit, the contextLength=9 step took `7.066513s`
  decode / `6.425403s` attention; after extending the same small-context kernel
  to `contextLength<=32`, the identical `[15,11,15,15,21,15,15,15,15]` output
  took `3.897080s` decode / `3.316559s` attention on that step, and aggregate
  throughput improved from `0.325 tok/s` to `0.367 tok/s` including logits.
  This confirms the remaining attention problem was repeated score computation,
  not SSD bandwidth. `glm_moe_infer` now routes `contextLength=2..32` through a
  small-context kernel and `contextLength>32` through a streaming kernel; both
  use one threadgroup per head and threadgroup-parallel qk/RoPE partials so each
  score is computed once. The streaming path applies an online softmax without
  allocating context-sized threadgroup storage. The probe-only
  `--mla-kv-b-f32` input and
  `metal/glm_moe_infer_mla_streaming_smoke.py` now validate that branch on tiny
  direct-F32 tensors at `contextLength=9`, `32`, and `33`, for both default and
  interleaved RoPE; the max absolute differences versus the Python reference
  stay below `8e-9`, with a `2460` byte live estimate at `contextLength=33`. The GLM-shaped
  direct-F32 microbench uses the real GLM-5.2 MLA dimensions
  (`64` heads, `kv_lora=512`, `qk_nope=192`, `rope=64`, `v=256`) without reading
  model weights. The small-context baseline measured `0.041593s` at
  `contextLength=9` and `0.140647s` at `contextLength=32`; after parallelizing
  the small-context scores, those medians drop to `0.005918s` and `0.006985s`
  respectively, while the streaming branch remains about `0.008656s` at
  `contextLength=33` and `0.010446s` at `contextLength=64`, with about `59MiB`
  estimated live working set. The real GLM guarded 9-token smoke now generates
  the same `[15,11,15,15,21,15,15,15,15]` as the previous smallctx32 run while
  improving throughput from `0.367 tok/s` to `0.872 tok/s` including logits;
  total attention time drops from `17.847006s` to `3.747658s`, and the
  contextLength=9 step drops from `3.897080s` decode / `3.316559s` attention to
  `0.995520s` decode / `0.420962s` attention. A guarded 17-token run stays in
  the small-context path, reads `190.503GiB` of routed experts at `64.487GiB/s`,
  and reports `0.908 tok/s` including logits with the last step at `1.038267s`
  decode / `0.452894s` attention. A guarded 33-token run reaches the streaming
  branch at `contextLength=33`, reads `369.800GiB` of routed experts at
  `64.844GiB/s`, and reports `0.889 tok/s` including logits. Total attention
  time is now `15.759542s`; the first streaming step reports `1.213384s`
  decode / `0.624476s` attention, so the streaming MLA path is no longer a
  catastrophic fallback, but attention output remains the largest aggregate
  bottleneck at `43.8%` of decode time. `scripts/decode_telemetry_report.py`
  now labels this fused timing as
  `rope_mla_attn_output_norm_router_fused` instead of plain `attn_output` when
  the command buffer includes RoPE, MLA, attention output, norm, and router.
  The first GLM version of Flash-MoE's deferred CMD3 scheduling is also in
  place for decode-layer MoE blocks: a non-final MoE layer can commit its single
  expert command buffer without waiting, return the output `MTLBuffer`, and let
  the next layer's attention command on the same Metal queue provide the
  synchronization point. Standalone MoE probes and final layers still wait for
  CPU readback. A real three-layer oracle (`layers=0,3,4`, dense layer `0`)
  matches the old runner within `1.430511e-6` hidden max diff and
  `1.144409e-5` top-k max diff, while reporting MoE waits `[0,1]`; the tiny
  standalone MoE smoke still reports `async_submitted=0` and
  `synchronous_wait_count=1`. A guarded direct CLI 9-token rerun after the
  telemetry propagation generated the same
  `[15,11,15,15,21,15,15,15,15]` under a 16GiB live cap and a 24GiB
  free-unified-memory guard. The latest rerun reports `10.326609s` summed
  decode, `0.454387s` summed final logits, `0.835 tok/s` including logits,
  `100.854GiB` routed expert reads in `1.535937s` (`65.663GiB/s`),
  `34.963GiB` resident attention-output weight reads in `1.692658s`,
  `1.978GiB` router reads in `0.098004s`, `5375752720` estimated live bytes,
  and admission/memory checks both true. The end-to-end MoE command wait count
  is now `9 / 675`, so the Flash-MoE-like deferred MoE scheduling is active in
  the real generation path; the remaining primary bottleneck is still the fused
  `rope_mla_attn_output_norm_router_fused` stage at `46.4%` of decode time,
  with `3.096497s` inside the fused attention-output projection command.
  `scripts/decode_telemetry_report.py` now also emits an
  `optimization_frontier` block: on the current 9-token artifact,
  attention-output read+projection is `0.453s/token` versus routed expert read
  at `0.164s/token`, with `234` command buffers and `82` estimated sync waits
  per token. Detailed `probe_decode_layers` payloads now also get a
  `layer_frontier` with the hottest per-layer attention-output, overhead,
  command-buffer, and wait entries, while decode-layer aggregates expose
  attention-projection sync waits versus async submissions and
  `sync_waits_by_stage`. This keeps the next runtime work aimed at
  attention-output fusion and command-buffer/wait reduction before more
  expert-I/O tuning.
  A follow-up runtime change lets non-final dense MLP layers submit their fused
  RMSNorm/gate/up/SwiGLU/down/residual command buffer without an immediate host
  wait, then chains the output `MTLBuffer` into the next layer on the same
  command queue. Standalone/final dense probes still wait for readback. A
  guarded two-layer real smoke (`--layers 0,3 --dense-layers 0 --new-only`,
  `2048MiB` live cap) reports `dense_mlp_async_submitted=[1]` and
  `dense_mlp_synchronous_waits=[0]`, while preserving the generated token
  `140366`. This removes the dense-prefix waits from the full decode wait
  budget; the hot MoE attention-output/router boundary remains the larger
  blocker.
  The Flash-MOe large-weight trial was explicitly skipped because the local
  network/time budget is not available for another huge download. The current
  main-line experiment instead keeps using the prepared GLM artifact and moves
  one Flash-MOe-shaped overlap into the real decode path: for fully fused
  MoE layers with `top_k + shared` reusable expert buffers, the runtime now
  commits the RoPE/MLA/attention-output/RMSNorm/router/top-k command buffer
  and preads the fixed shared expert into its existing staging slot while the
  host waits for router completion. Routed expert reads still wait for GPU
  top-k ids, so correctness is unchanged and peak staging memory does not grow.
  The guarded two-layer smoke above now asserts `shared_prefetch_used` on the
  MoE layer and aggregate `shared_prefetch_used_count=1`; the new telemetry
  fields `shared_bytes_read`, `shared_read_seconds`,
  `shared_prefetch_seconds`, and `shared_prefetch_used_count` are carried
  through raw Metal JSON, Python generation results, server step payloads,
  CLI summaries, and `scripts/decode_telemetry_report.py`.
  A July 5, 2026 resident-Metal experiment then wired `--wrap-resident-metal`
  through the continuous MoE decode path and validated it with
  `metal/glm_moe_infer_real_decode_layers_smoke.py --new-only --layers 3,4
  --dense-layers '' --wrap-resident-metal`. The corrected path maps the whole
  `resident.bin` as one Metal buffer and uses offsets for `o_proj`, router, and
  router correction bias while still staging post-attention RMSNorm because the
  resident norm tensor is BF16 and the current RMSNorm kernel consumes F32.
  The smoke passed and reported `attn_output_bytes_read=0`,
  `router_bytes_read=0`, and `router_correction_bias_bytes_read=0`, but the
  two-layer attention-output projection kernel regressed from about
  `0.016889s` on the staged default path to a stable `0.545517s` when the GPU
  read the file-backed mmap directly. So `--wrap-resident-metal` remains an
  experimental diagnostic, not a default optimization; the main line should
  keep Flash-MoE-style bounded staging buffers and overlap rather than making
  Metal kernels stream directly from the mmap'd file. The telemetry now carries
  `attn_output_resident_mmap_backed_count` through raw `glm_moe_infer` JSON,
  Python generation results, server step payloads, CLI summaries, and
  `scripts/decode_telemetry_report.py` so future artifacts can prove exactly
  which layers took the diagnostic resident-backed path.
  The next staged-path experiment removed redundant zero-fill of fully
  overwritten 2MiB-aligned staging buffers and changed the fused MoE
  attention-output/router path to read `o_proj.weight`, `o_proj.scales`,
  router, and router correction bias through the existing persistent parallel
  `pread` pool. This keeps the same live-memory envelope and produced the same
  `[15,11,15,15,21,15,15,15,15]` sequence under the 16GiB/24GiB guards. The
  saved artifact
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-parallel-staging-9tok-latest.json`
  reports `9.815746s` summed decode, `0.423321s` summed final logits,
  `0.879 tok/s` including logits, `100.854GiB` routed expert reads in
  `1.527887s`, `34.963GiB` attention-output resident reads in `1.570976s`, and
  `1.978GiB` router reads in `0.087877s`. This is a small staged-read
  improvement over the previous telemetry run, but the primary bottleneck is
  still `rope_mla_attn_output_norm_router_fused` at `4.542971s` (`46.3%`) with
  `2.971820s` inside the attention-output projection command.
  A follow-up kernel fusion made the `fused_matvec_add` path literal: the
  attention-output MXFP4 matvec kernels now add the residual in-kernel instead
  of launching a separate F32 add kernel in the same command buffer. The guarded
  9-token artifact
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-parallel-staging-matvecadd-9tok-latest.json`
  reports `9.296515s` summed decode, `0.389798s` summed final logits,
  `0.929 tok/s` including logits, `100.854GiB` routed expert reads in
  `1.474182s`, `34.963GiB` attention-output resident reads in `1.564848s`, and
  `1.978GiB` router reads in `0.087628s`. The fused
  RoPE/MLA/attention-output/norm/router stage dropped to `4.072766s`
  (`43.8%`), with `2.507744s` in the attention-output command-buffer timing.
  A follow-up code review of `danveloper/flash-moe` was done without running
  its large weights locally. The important distinction is not just hardware:
  flash-moe's Qwen MoE path streams roughly hundreds of MiB per token and keeps
  only a small non-expert mmap plus bounded expert staging buffers live, while
  the current GLM path streams about `11.2GiB/token` of routed experts plus
  `3.9GiB/token` of attention-output weights. Its source also matches our
  resident-Metal negative result: bulk `pread()` into 2MiB-aligned staging and
  OS page cache are preferred, while mmap expert reads and custom caches were
  measured as harmful. A shared-input tiled attention-output MXFP4 experiment
  was therefore kept as an opt-in kernel instead of a default rewrite. With
  `LARGERLM_GLM_MOE_INFER_TILED_ATTN_OUTPUT_MXFP4=1` and a 2048-float tile, the
  guarded 9-token artifact
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-tiled-matvecadd-9tok-latest.json`
  was essentially neutral (`9.263535s` decode, `0.930 tok/s` including logits,
  `2.522796s` attention-output command-buffer timing). A 4096-float tile
  regressed to `0.876 tok/s` in
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-tiled4096-matvecadd-9tok-latest.json`.
  Default execution remains on the proven staged SIMD matvec-add path. The
  GLM-specific route is now concrete for context=1 decode: because MLA emits
  the current value as `B_v * latent`, the runtime can precompute
  `o_proj * B_v` per layer and replace the hot attention-output projection with
  a much smaller latent matvec. The read-only planner
  `scripts/glm_context1_o_proj_collapse_plan.py` validates this on the current
  GLM-5.2 MXFP4 prepared artifact without reading the large weight payloads:
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-collapse-plan-latest.json`
  reports 78/78 supported layers, `3978.0 MiB/token` current `o_proj` reads,
  a `468.0 MiB` total BF16 collapsed cache (`936.0 MiB` F32), and an observed
  hot-read ratio of `0.118`. The cache format and safe builder entry point now
  live in `largerlm.context1_o_proj_cache` and
  `python -m largerlm context1-o-proj-cache`. Its default mode is a no-payload
  dry run; the real artifact report
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/context1-o-proj-bv-cache-dry-run-latest.json`
  confirms a `468.0 MiB` BF16 cache and `4.02T FMA` build without reading the
  resident weight payload. `--execute` is guarded by `--max-build-fma` and
  refuses the full GLM build on the reference backend, so no accidental
  multi-teraflop Python job or memory-heavy cache build starts. The companion
  `python -m largerlm validate-context1-o-proj-cache` command validates the
  cache schema, backing file, tensor spans, numeric layer order, dtype/dims, and
  optional prepared-artifact match before runtime code is allowed to trust the
  file. It also rejects an incomplete builder progress file by default; use
  `--allow-incomplete-progress` only to report partial progress during bounded
  chunk builds. The Metal runtime now has the first standalone builder/consumer
  probes.
  `--build-context1-o-proj-cache-layer` builds one selected collapsed-cache layer
  from resident MXFP4 `o_proj` and `unembed_out` spans under an explicit GFMA
  cap; `metal/context1_o_proj_cache_build_smoke.py` verifies the tiny BF16 build
  and immediately consumes it. The Python builder CLI can now drive that path
  with `python -m largerlm context1-o-proj-cache --backend metal --execute`,
  calling `metal/glm_moe_infer` one layer at a time and updating the same
  resumable progress file. `--build-layers 0,7-9` can execute a small subset
  while keeping the full cache layout/progress intact, so a real GLM cache can
  be built in bounded resumable chunks under a per-invocation FMA cap.
  `--build-next-layers N` is the safer resumable form: it selects the next
  incomplete layers from the progress file, or the first `N` layers when no
  progress exists, and still only executes under `--execute`. Dry-run and build
  reports now include a `selected_build` block with the requested
  layer subset, cache/source bytes, total FMA, max per-layer FMA, and the
  minimum `--max-build-fma` needed for that invocation. The CLI also accepts
  `--max-build-gfma` as a human-readable alternative, so a one-layer GLM
  execution can be capped around `51.54` GFMA instead of spelling out
  `51539607552` FMA. The builder now also reports a `disk_budget` block and,
  under `--execute`, refuses to create or
  extend the cache unless `total_bytes + --disk-margin-gib` fits on the target
  volume; the CLI default keeps a 16GiB free-disk margin. The Metal builder
  path has the same shape of live-memory guard: reports include
  `max_estimated_metal_builder_live_bytes`, Python refuses
  `--backend metal --execute` above `--max-metal-builder-live-mib`, and
  `metal/glm_moe_infer --build-context1-o-proj-cache-layer` enforces
  `--max-live-working-set-mib` directly. `metal/context1_o_proj_cache_build_smoke.py`
  now asserts both the accepted tiny build and an intentionally rejected
  too-low live cap, while `metal/context1_o_proj_cache_cli_metal_backend_smoke.py`
  asserts the Python CLI exposes the same builder-live estimate/cap. Executed
  builds now persist per-layer `layer_results` in both the build report and
  progress file, including source/cache bytes, FMA, read/kernel/write timing,
  live estimate, and device name for Metal layers. The report also adds
  `layer_result_summary`, which rolls measured layers into GFMA/s, completed
  FMA fraction, and estimated full/remaining build seconds so a real one-layer
  GLM build can immediately decide whether the full offline cache build is
  practical. `validate-context1-o-proj-cache --allow-incomplete-progress` now
  exposes the same progress summary plus `next_missing_layer` and
  `missing_layer_count`, so an interrupted real build can be resumed with
  `context1-o-proj-cache --backend metal --build-next-layers N` without manually
  inspecting progress JSON. Incomplete progress now also emits
  `suggested_resume_build`, a conservative one-next-layer dry-run/execute argv
  pair with the minimum `--max-build-gfma`; it still requires explicit
  `--execute` and the normal disk/FMA/live-memory guards. The real GLM artifact
  `context1-o-proj-bv-cache-metal-dry-run-latest.json` confirms that the Metal
  backend plan is still a dry run by default; an execute attempt for one real
  layer is refused under the default `50,000,000` FMA cap because that layer
  requires `51,539,607,552` FMA, or about `51.54` GFMA. The consumer path,
  `--probe-context1-o-proj-cache-output`, which reads one validated BF16/F32
  collapsed-cache layer and runs latent matvec+residual without opening
  resident/expert weights; `metal/context1_o_proj_cache_output_smoke.py` checks
  a tiny BF16 fixture (`output0=64.5`) against the new kernel. The first
  decode-layer runtime integration is now in `glm_moe_infer`: when
  `--context1-o-proj-cache-layout` is supplied and `context_length == 1`, the
  dense decoder-layer path can bypass MLA value projection plus resident
  `o_proj` and consume the collapsed cache from the current KV-A latent instead.
  `metal/glm_moe_infer_context1_dense_decoder_smoke.py`,
  `metal/glm_moe_infer_context1_moe_decoder_smoke.py`, and
  `metal/glm_moe_infer_context1_decode_layers_smoke.py` compare that opt-in path
  against tiny dense/MoE/decode-layers baselines, verify identical output/cache
  writes, and check `attn_output_context1_o_proj_cache` telemetry. The Python
  Metal generation payload, `serve-prepared --metal-runtime-generation`, and
  `/health` now expose the same opt-in with
  `--context1-o-proj-cache-layout` or
  `--metal-runtime-context1-o-proj-cache-layout`. Prepared serving validates the
  layout/backing/prepared match at startup, rejects an incomplete builder
  progress file, and exposes the validated summary in `/health`; generation
  responses and telemetry reports surface
  `attn_output_context1_o_proj_cache_count` for benchmark proof. Full production
  use still needs a deliberately executed real GLM collapsed-cache build and
  benchmarking. `scripts/glm_metal_viability_report.py --decode-telemetry`
  now folds the collapse plan into existing decode telemetry: the current
  context=1-only integration is expected to be only a small aggregate speedup
  on multi-token decode. `scripts/glm_latent_value_collapse_plan.py` now checks
  the exact all-context extension and rejects the naive per-head cache shape for
  GLM-5.2: 64 independent attention heads exceed the 8 break-even shared-head
  groups, the BF16 per-head cache would read `29.250 GiB/token` (`7.53x`
  current `o_proj`), and even an ideal int4 floor would still read
  `7.312 GiB/token` (`1.88x`). Keep the context=1 cache as a narrow verified
  fast path; do not replace general decode with a per-head collapsed cache.
  A guarded 1-token runtime smoke also compiled the updated Metal library and
  still generated `[15]`. The direct CLI deferred-MoE artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/direct-cli-metal-runtime-kvbcache-deferred-moe-9tok-latest.json`;
  the
  real 33-token artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-small-parallel-33tok-latest.json`;
  real 17-token artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-small-parallel-17tok-latest.json`;
  the real 9-token small-parallel artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-small-parallel-9tok-latest.json`;
  the optimized GLM-shaped MLA microbench artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-shaped-mla-small-parallel-latest.json`;
  the small-vs-streaming baseline artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-shaped-mla-small-vs-streaming-baseline-latest.json`;
  the runtime compile smoke artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-streaming-library-compile-1tok-latest.json`;
  the 9-token smallctx32 artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-smallctx32-9tok-latest.json`;
  the 9-token smallctx8 baseline artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-smallctx8-9tok-baseline-latest.json`;
  the 5-token smallctx8 artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-smallctx8-5tok-latest.json`;
  the 5-token smallctx4 baseline artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-smallctx4-5tok-baseline-latest.json`;
  the 3-token artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-smallctx-mla-fast-3tok-latest.json`;
  the 2-token artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-context2-mla-fast-latest.json`;
  the 1-token artifact is
  `artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-metal-runtime-kvbcache-direct-current-kva-latest.json`.
  A 1-token real GLM
  tokenizer smoke with prompt `你好` generated token id `[23]` (`"8"`), reported
  `5.096510s` decode time and `5.292902s` end-to-end wall time, and kept the
  same `542163472` byte Metal live envelope. A chat-template smoke with
  `[{"role":"user","content":"你好"}]` renders to 13 prompt tokens and now uses
  runtime prefill by default in the text wrapper; the explicit decode-only risk
  flag still bypasses prefill for experiments. The runtime
  prompt-prefill service smoke now passes on the existing prepared GLM package:
  `python3 metal/glm_moe_infer_generate_server_smoke.py --execute --requests 1
  --prompt-token-ids 0,1` reports `input_source="prompt_token_ids"`,
  `prompt_prefill.ok=1`, `first_step_from_input_logits=1`, generated ids
  `[30423,11093]`, the memory decode-cache backend, and a live estimate of
  `542186512` bytes under the 768MiB cap. `MetalTokenGenerationResult`, the
  CLI summary, and the HTTP response step payloads now expose decode telemetry
  from `glm_moe_infer`: routed expert bytes, dense-MLP resident bytes, expert
  read seconds, MoE kernel seconds, and pooled/serial expert-read dispatch
  counts per generated step, plus command-buffer and explicit MoE wait counts.
  Python callers that need repeated local generations can now use
  `MetalTextGenerationSession` as a reusable context: it loads the tokenizer
  once, starts one persistent `glm_moe_infer --generate-server-jsonl` process,
  and passes the same `generate_server_session` through every
  `session.generate(...)` call while retaining the usual Metal request guards.
  The matching CLI entry is `generate-metal-text-batch --prompts-jsonl FILE`,
  where each JSONL line is `{"prompt": "..."}`; it is intended for small local
  benchmark/replay batches that should reuse one runtime process. Its JSON
  result includes total prompt tokens, generated tokens, runtime startup,
  elapsed time, aggregate generated-token throughput, max live envelope,
  max prefill envelope, min available memory, max required memory, and
  admission/memory-ok summaries across requests.
  `serve-prepared` now has an explicit greedy-only bridge to this runtime:
  pass `--metal-runtime-generation --metal-binary metal/glm_moe_infer` to route
  `/generate-token-ids` and `/generate-text` through a lazily started,
  persistent `glm_moe_infer --generate-server-jsonl` session after the normal
  request check and prepared run lock. The server health payload reports
  whether that Metal runtime session has started and how many requests it has
  handled, so repeated HTTP requests can confirm resident/expert fd reuse. If a
  Metal runtime request breaks the JSONL pipe or otherwise raises a
  `MetalGenerateError`, the server closes that session and the next request
  lazily starts a fresh one instead of reusing a poisoned child process.
  The default server path remains the older Python batch-prefill stack, and the
  Metal runtime server mode rejects sampling parameters until the C/Metal
  runtime has matching sampling semantics.
  See
  `docs/flash-moe-rewrite-decision.md`.
- The same resident-linear + attention-projection + RoPE process-fusion bundle
  now has a locked 512-token custom-metal fallback bakeoff for cases where the
  router-gate MPSGraph path is not selectable by the current host probe. The
  process-fusion replay generated `[15]`, stayed inside the 17.37 GiB live cap,
  and beat the same-plan custom-metal MoE-server control, `441.960s` versus
  `500.707s` (`0.883x`). It is recorded as
  `selected-replay-prefill-512-custom-metal-processfusion-rope-server-memory-keycache.json`
  and passes `selected-replay-check --check-ssd-read-speed`.
  A non-sandbox MPSGraph probe also validates the router-gate path on the current
  host. The scoped MPSGraph + process-fusion replay is replay-ready and beats the
  same-window non-fused MPSGraph replay, `472.264s` versus `495.513s`, but it is
  still slower than the custom-metal process-fusion fallback and the historical
  262.906s MPSGraph selected replay.
- Architecture notes for the M5 Max path, including Metal 4 MPP tensor ops.

## Quick Start

Analyze from only a Hugging Face `config.json`:

```bash
python -m largerlm plan /path/to/model-or-config --quant-bits 4 \
  --max-context-tokens 32768
```

Config-only planning now subtracts a conservative resident-weight estimate from
the unified-memory budget before reporting page-cache and decode-cache headroom.
Use `--scan-safetensors` on a real checkpoint for exact resident/routed byte
counts from shard headers. Planning rejects non-finite or unsafe budget inputs
up front: quant/group sizes and runtime buffer must be positive, explicit memory
and cache byte budgets must not be negative, and `--page-cache-fraction` must be
between 0 and 1. GiB/MiB budget arguments on the plan, preflight, prepare,
packing, and decode-cache setup commands must also be finite before they are
converted to bytes, so `nan`/`inf` inputs fail as CLI argument errors.
For GLM MLA/DSA configs with a modeled decode-cache budget, planner JSON also
emits `suggested_prepare_flags.argv`: a replayable prepare-argument fragment
with `--auto-context-from-budget`, `--max-cache-gib`, memory/reserve/runtime
budget flags, page-cache fraction, and group size. It intentionally does not
choose the raw-BF16 quantization conversion flag; use
`--quantize-bf16-affine-int4` only when the checkpoint weights are not already
in the expected 4-bit affine expert format.
Use `--write-prepare-flags /path/to/prepare-flags.json` to save that fragment
as a small JSON artifact before a real prepare run; add the checkpoint path,
`--output-dir`, and `--execute` explicitly when you are ready to write files.
`prepare-glm --apply-prepare-flags /path/to/prepare-flags.json` applies that
artifact before local arguments, and only accepts plan-authored allowlisted
budget flags. Dry-run reports and executed prepared manifests record the
artifact path plus SHA-256 digest, and `inspect-prepared` surfaces the same
provenance in health output. Launch audit records a
`prepare_flags_provenance_ok` check when that provenance is present.

Analyze a checkpoint directory when `model.safetensors.index.json` and shards
are present, or when a minimal smoke checkpoint has exactly one `.safetensors`
file:

```bash
python -m largerlm plan /path/to/checkpoint --scan-safetensors --json
```

For GLM configs, `--scan-safetensors` uses the same MoE-layer and runtime-layer
boundary as `preflight-glm`, so MTP/extra tensors at or above
`num_hidden_layers` are reported as ignored bytes instead of inflating resident
or routed expert budgets. Config loading validates and normalizes
`mlp_layer_types`, rejecting unknown dense/sparse labels before they can
misclassify a GLM MoE layer. It also rejects non-positive model dimensions,
expert/top-k counts, RoPE/RMSNorm scalars, and impossible config relationships
such as top-k above the routed expert count or DSA `index_topk` above the model
context window. Config scalar parsing is type-strict: integer, float, and
boolean fields reject booleans, floats, strings, and non-finite values where
they do not belong, including GLM-5.2 DSA schedule freq/offset fields. If a
config declares a full/shared DSA schedule, loading also requires
`index_head_dim`, `index_n_heads`, `index_topk`, and `q_lora_rank` so cache and
indexer costs cannot be planned from a zero-width fallback.

Estimate prefill GEMM shapes, MPP tensor-op candidates, and routed expert SSD
streaming cost without loading weights:

```bash
python -m largerlm prefill-plan /path/to/checkpoint/config.json \
  --prompt-tokens 4096 --dtype-bits 16 \
  --max-runner-scratch-mib 4096
```

The prefill plan also emits M5-oriented tile plans for candidate GEMMs. Defaults
follow the Apple MPP guide starting point: 32x32 simdgroup tiles, 2x2 simdgroups
per threadgroup, and K tiles of 128. Use `--simdgroup-tile-m`,
`--simdgroup-tile-n`, `--simdgroups-m`, `--simdgroups-n`, and `--k-tile` for
future tuning runs.
It also emits a `prefill_backend_candidates` list, sorted by total FLOPs, so
the first M5/MPP or MPSGraph work items are explicit. Each row records the GEMM
shape, arithmetic intensity, static-tile status, preferred backend, execution
path, local availability, and the reason for any fallback. The text output
shows the top three priorities; `--json` includes the full list.
`--prompt-tokens` is checked against the model config's
`max_position_embeddings` when present, so prefill planning cannot silently
exceed the advertised context window.
For GLM MLA/DSA attention, the same plan reports prompt-cache read/write
bytes, including capped indexed-attention rows and full DSA indexer cache rows,
so long-prompt SSD pressure is visible before any cache file or Metal buffer is
created. The plan also includes the public GLM-5.2 shape diagnostic; pass
`--require-public-glm-5-2-shape` to fail before calibration or profile writing
when a GLM-5.2 launch profile must not be derived from another GLM variant.
When the config does match public GLM-5.2, `suggested_launch_profile` carries
the same guard so replayed prepared launches can reject drift. The diagnostic
also audits the raw GLM-5.2 DSA schedule fields (`index_topk_freq=4`,
`index_skip_topk_offset=3`), `num_nextn_predict_layers=1`, and the derived
full-indexer layer list, so a near-shape config with a different indexer
schedule is not treated as the public target.
Pass `--max-prefill-activation-mib` to get a tile-aligned prompt chunk size that
keeps each prefill GEMM's activation footprint below a chosen cap.
For routed MoE models, chunking may re-read the same layer's expert slots per
chunk. `routed_expert_read_cost_plan` reports baseline bytes, planned bytes,
extra bytes, and read amplification; pass `--ssd-read-gib-s` with a measured
`disk-read-benchmark` result to estimate routed expert read seconds. Use
`--prefill-max-routed-read-amplification` on generation, benchmark, inspect, or
serving commands to turn that estimate into a hard admission guard; add
`--prefill-max-routed-read-gib` when you also want to cap the absolute planned
routed expert SSD read volume. If you have a measured `disk-read-benchmark`
throughput, pass `--prefill-ssd-read-gib-s` to estimate planned read seconds
and `--prefill-max-routed-read-seconds` to reject prompts that would exceed a
latency budget. `0` keeps each guard disabled. The plan emits
`suggested_guard_flags`, an argv-style version of the resolved chunk and
routed-read guard profile with 5% headroom, so GLM-5.2 launch scripts can reuse
the same prompt chunk/read budget without re-deriving it. It also emits
`suggested_stage_temp_guard_flags`, using `--expert-stage-align-kib` (default
4 KiB) to suggest matching `--prefill-max-stage-mib` and
`--prefill-max-compact-stage-mib`,
`--prefill-max-stage-raw-ranges`, and
`--prefill-max-stage-coalesced-ranges` caps for the resolved chunk. It also
records the static-capacity binary route-table peak/total bytes and replays
`--prefill-static-capacity-per-expert auto` by default, matching the prepared
batch-prefill runtime path.
`suggested_prefill_guard_flags` combines those read and stage profiles into one
deduplicated argv list for launch scripts. `suggested_launch_profile` then
wraps that prefill guard section together with explicit prefill runtime policy
flags such as `--prefill-mpsgraph-min-batch-tokens`,
`--prefill-mpsgraph-min-matrix-dim`, and
`--prefill-min-accelerated-flop-fraction`, backend acceleration flags from
`--inspect-backend`, and probe flags such as `--compile-mpp-probe` and
`--run-mpsgraph-probe`, so a prefill-plan result can be saved and replayed
through prepared generation, inspection, benchmark, or serving commands. Those
policy flags are only saved by the plan; the prepared command that applies the
profile performs the actual backend admission and request coverage checks. Use
`--write-launch-profile PATH` to write only that profile JSON without
post-processing the full plan output. The full plan also includes
`prefill_linear_calibration_shapes` and
`suggested_prefill_linear_calibration_flags`: a bounded
`prefill-linear-calibrate` argv fragment derived from the highest-FLOP unique
resident GEMM `INxOUT` shapes. These calibration flags intentionally stay out
of `suggested_launch_profile`; they are for measuring the local
custom-Metal/MPSGraph crossover before choosing runtime thresholds, not for
replaying a generation request. Use `--write-calibration-flags PATH` to save
just that calibration fragment for a follow-up `prefill-linear-calibrate` run,
or call `prefill-plan-calibrate RUNNER MODEL --prompt-tokens N` to build the
plan and immediately run the bounded custom-Metal/MPSGraph calibration over
those shapes. The combined command refuses to dispatch the runner if the
planner-derived case, resident-matrix, or scratch caps exceed its
`--max-auto-*` calibration limits, keeping accidental large-shape sweeps behind
an explicit opt-in. It also estimates the calibration work directory footprint
from the planned matrix/input files plus both backend outputs for every repeat,
refuses to dispatch if that exceeds `--max-calibration-work-dir-mib` (default
8192 MiB), and checks that the target volume still has
`--calibration-work-dir-free-margin-mib` free MiB beyond the estimate. Its
`--write-launch-profile` output is a merged prefill profile: planner-derived
chunk/read/stage guards are kept, while the measured calibration runtime policy
supplies the MPSGraph auto thresholds when the grid supports one. If you pass
`--require-prefill-acceleration` or
`--prefill-min-accelerated-flop-fraction` to `prefill-plan-calibrate`, those
planner-side acceleration gates are preserved in the merged runtime-policy
section rather than being overwritten by the calibration threshold result. Pass
`--ssd-read-gib-s` to the same command to carry measured SSD throughput into the
planner and include routed-expert read-second guards in the merged profile.
Because `prefill-plan` only sees the config, not a prepared package, these
`prefill_plan` and `prefill_plan_calibration` profiles are accepted as
prefill-only profiles without prepared identity metadata, after their sections
and argv flags pass the prefill-only whitelist. `--apply-launch-profile` can
read either the saved profile itself or a full `prefill-plan-calibrate --json`
payload containing `combined_launch_profile`. Profiles produced by
`inspect-prepared --write-launch-profile` still carry prepared-package identity
and are rejected if replayed against a different prepared output.
For routed MoE prefill, the plan also reports static expert capacity hints:
balanced capacity is the efficient fixed-shape lower bound that needs an
overflow path, while spill-free capacity is the conservative per-expert prompt
bound useful for validating ANE/ML tensor kernels without dropping routed
assignments. Both include per-layer f32 activation estimates so unsafe static
shapes can be rejected before any Metal buffers are allocated. With
`--max-runner-scratch-mib`, the plan also estimates the current staged MoE
runner's assignment table, token-seen bitmap, auto token-block buffer bytes,
and per-layer runner peak so a prompt chunk can be rejected before launching
Metal. Plan budget and tile controls are strict integers, so booleans and
floats cannot slip into prompt, bit-width, group-size, activation-byte, or
runner-scratch fields. When chunking is enabled, the plan separately reports
chunk-major routed expert read bytes so SSD cost is not underestimated.

Inspect the local Metal 4 / MTLTensor prefill backend without loading model
weights:

```bash
make -C metal prefill-backend-probe
python -m largerlm prefill-backend --json
python -m largerlm prefill-backend --compile-mpp-probe --json
python -m largerlm prefill-backend --run-mpp-probe --json
python -m largerlm prefill-backend \
  --run-mpsgraph-probe --run-mpp-probe --probe-timeout-seconds 30 \
  --write-report /path/to/m5-prefill-backend.json --json
python -m largerlm prefill-plan /path/to/checkpoint/config.json \
  --prompt-tokens 4096 --inspect-backend --run-mpsgraph-probe \
  --write-launch-profile /path/to/prefill_launch_profile.json --json
```

The backend JSON includes both raw probe fields and derived
`mps_graph_runtime_available`, `metal4_ml_runtime_available`, and
`mpp_runtime_available` booleans, plus
`prefill_acceleration_runtimes` and
`selectable_accelerated_prefill_backends`. The former lists hardware/OS
acceleration paths that probe as available; the latter lists accelerated
backends current LargerLM generation commands can actually select.
`validated_accelerated_prefill_backends` is stricter: it only lists selectable
backends with runtime proof strong enough for required acceleration gates, so
`mpsgraph-f32` appears there only after `--run-mpsgraph-probe` succeeds.
`prefill_acceleration_runtime_gaps` records visible runtimes that are not yet
selectable by generation. On M5-class hosts this keeps a probed
`mpp_tensor_ops_prefill` runtime distinct from the currently wired
`mpsgraph-f32` execution path.
`prefill_neural_accelerator_status` gives the same MPP/Metal-ML path a
machine-readable bring-up state, including whether the runtime is visible,
whether generation can select it yet, the compile/run-probe result, and the
planned `mpp_tensor_ops_gpu_neural_accelerator` execution path. `--run-mpp-probe`
runs a tiny 32x32 half-precision MPP `matmul2d` and verifies the output before
reporting execution evidence, including the kernel include variant, `32x32x32`
tile shape, dtype, and MPP execution primitive when the run reaches a pipeline;
it is an inspection/bring-up probe, not a generation backend selector. The same
status is included in prepared health/inspection output when the backend probe
runs. The earlier probe looked only for `metal_mpp` inside the selected SDK and
therefore missed the public
`MetalPerformancePrimitives/MetalPerformancePrimitives.h` system-framework
entry. After correcting that include/search path on the local M5 Max, both the
compile probe and the 32x32 execution probe pass with zero maximum absolute
error. LargerLM now exposes `mpp-f32` as an opt-in backend for resident
F32/BF16/F16 prefill GEMMs; MPSGraph remains the fallback and routed MXFP4
experts remain on custom Metal.
A local synthetic `128x4096` by `4096x4096` F32 comparison measured about
`7.8 ms` inside MPP versus `17.8 ms` inside MPSGraph. Including the bounded
matrix read, MPP took about `16.2 ms`, versus `25.3 ms` for MPSGraph and
`21.7 ms` for the existing custom batch kernel. These are bring-up numbers
under the current power-limited setup, not a full-model prefill claim.
Prompt prefill coverage, request health, and benchmark frontier rows also
report `mpp_tensor_ops_candidate_backend_counts` and
`mpp_tensor_ops_candidate_backend_flops`: these break down MPP-candidate GEMMs by
the backend that currently executes them, so M5 bring-up can prove when candidate
FLOPs move from `custom-metal` or `mpsgraph-f32` onto a future selectable
`mpp_tensor_ops_prefill` path.
`result-summary` now collapses those rows into a short MPP frontier verdict. On
the current 128-token GLM-5.2 MXFP4 custom/key-cache memory-accumulator artifact,
`smoke-prefill-128tok-custom-metal-keycache-memory-accumulator-bound-result.json`,
it reports
`prefill MPP frontier: status=mpp_backend_not_selectable candidate=546/621 flops=92.5% selectable=False policy=tokens>=128,dim>=32`,
with the candidate work currently split across `custom-metal=234` and
`fused-metal=312`. This means the prompt/chunk shapes are already large enough
for the planned M5 MPP/Neural Accelerator path; the remaining blocker is a real
selectable `mpp_tensor_ops_prefill` execution backend, not a larger prompt
chunk or another memory-risky model run.
The same summary JSON now emits `optimization_targets` so the next experiment is
machine-readable. On that artifact, the ranked targets are the non-router
prefill acceleration gap, the blocked MPP tensor-ops frontier, routed MoE, MLA
attention, resident projections, RoPE, expert stage copy, and cache write.
When a suggested bounded experiment has already written its small
`--write-result` JSON, `result-summary` now reads back its `config_comparison`
and reports whether it produced a candidate for full replay. The current
128-token artifact therefore shows routed MoE, MLA cache mode, attention
projections, resident attention-output, RoPE split, and cache write as
completed bounded experiments with `candidate=False`, so those branches do not
need to be rerun before picking the next target.
For routed MoE, the target also emits a bounded one-layer
`glm_moe_tile_sweep.py` command when the older split-kernel hint is not
selective enough. The generated layer-77 command used the slowest measured
128-token layer, capped stage and compact staging at 240 MiB each and runner
scratch at 256 MiB, then wrote
`glm-moe-layer77-optimization-target-128tok.json`. That microbench found no
candidate for full replay: `tile1_auto_silu` stayed the fastest kernel
configuration at about 0.0698s mean kernel time, while tile/group32-off/vector
variants were slower despite numerical agreement. Keep the current routed-MoE
default.
`result-summary` also separates runner-reported routed MoE time from the
surrounding Python/file orchestration. The old 128-token artifact classified
routed MoE as `process_boundary`: routed MoE totaled about 14.151s, runner
`timing total` summed to about 3.951s, and the non-runner boundary was 72.1% of
routed elapsed. The locked-profile persistent-runner replay,
`smoke-prefill-128tok-persistent-moe-server-tiled-profile-replay-result.json`,
keeps one MoE plan server alive across all 75 tiled routed layers. It generated
the same token id 15, reduced total elapsed from 63.739s to 59.688s, kept the
same 17.37 GiB live-memory envelope, and recorded
`moe_plan_server_plan_count=75` with `routed_moe_runner_command_count=0`. With
process launch removed, the routed target is now classified as
`moe_orchestration_overhead`; the next structural candidate is batching/fusing
tiled plan submissions and Python/file work around the persistent runner, gated
by locked replay.
New staged-MoE results also carry wall-clock subphase telemetry for that
boundary: compact-stage construction, route writing, static-capacity route
materialization, runner subprocess wall time, output validation, and total wall
time. When a fresh replay includes those fields, `result-summary` prints a
`routed moe wall:` line so the persistent-runner prototype can be scoped against
the largest measured boundary component instead of guessing from the aggregate
72.1% non-runner share.
The process-boundary prototype is explicit and opt-in:
`metal/largerlm-runner --run-moe-batch-plan --batch-plan-json PATH` consumes a
`largerlm.staged_routed_moe_batch_plan.v1` JSON file and executes multiple
already-materialized staged routed-MoE jobs in one runner process. The Python
wrapper `run-staged-routed-moe-batch-plan` checks that every planned output was
written, and `metal/staged_routed_moe_batch_smoke.py` verifies the new plan
entry point against the existing static-capacity staged-MoE path. A second
entry point, `--run-moe-batch-plan-server-jsonl`, keeps the same runner process
alive and accepts one plan path per JSONL request; the
`StagedRoutedMoEBatchPlanServerSession` Python context manager can submit plans
sequentially and wait for each `server request: ok` marker. This is the safer
shape for full prefill, where routed-MoE jobs cannot all be materialized up
front because each layer depends on the previous layer's output. The low-level
`run_staged_routed_moe_batch`, tiled staged MoE, staged routed MLP, prompt
prefill, generation, inspection, and serving paths can opt into that session
with `--prefill-persistent-moe-plan-server` (or
`--persistent-moe-plan-server` on the raw `prefill-prompt` command). Tiled
expert stages keep their per-tile stage/compact/scratch guards while submitting
each tile plan to the same persistent runner.
A second, narrower process-boundary prototype exists for resident projections:
`metal/largerlm-runner --run-resident-linear-batch-plan-server-jsonl` accepts
one bounded `--run-resident-linear-batch` equivalent per JSONL request and
reuses the same Metal process until `{"command":"quit"}`. The Python
`ResidentBatchLinearServerSession` can be passed to `run_resident_batch_linear`
to reuse the same runner while preserving the existing matrix-size,
input/output byte, backend, and scratch-limit checks. This is currently an
explicit experiment for the `runner_process_fusion` target. Raw `prefill-prompt`
can enable it with `--persistent-resident-linear-server`; prepared generation,
inspection, and serving use `--prefill-persistent-resident-linear-server`, and
launch-profile composition treats that flag as replay-safe. The default remains
the old resident-linear subprocess path unless the caller opts in.
A third, even narrower process-boundary prototype targets fused prompt
attention projections: `metal/largerlm-runner --run-attn-projections-server-jsonl`
accepts one JSONL request at a time and reuses the same Metal process for
`input_layernorm`, `q_a`, `q_b`, `kv_a`, and `kv_b` projection work. This server
rejects cache append fields by design, so decode cache writes remain on the
existing one-shot `--run-attn-projections` command. Raw `prefill-prompt` can
enable it with `--persistent-attention-projection-server`; prepared generation,
inspection, and serving use
`--prefill-persistent-attention-projection-server`, and launch-profile
composition can carry the flag. The default remains the one-shot fused
projection path: the first locked 128-token GLM-5.2 candidate generated the
same token id 15 and reduced the projection subphase, but total latency stayed
within/no better than the old same-window baseline, so the bakeoff retained the
baseline.
The fourth tiny prototype is `--run-rope-split-batch-server-jsonl`, which keeps
one Metal process alive for the existing fused prompt-prefill RoPE split kernel.
Python drives it through `RopeSplitBatchServerSession`; raw `prefill-prompt`
can enable it with `--persistent-rope-split-server`, and prepared generation,
inspection, serving, plus launch profiles expose
`--prefill-persistent-rope-split-server`. It preserves the existing q_b/k input
byte checks, output-size checks, and runner scratch cap in Python and Metal. The
first same-window 128-token GLM bakeoff,
`prefill-128-attnproj-rope-split-server-vs-rerun-bakeoff-current.json`, selected
the RoPE split server candidate (`114.796s` vs `148.846s`, `0.771x`) and wrote
`selected-replay-prefill-128-persistent-moe-resident-linear-attnproj-rope-split-memory-accumulator.json`.
Treat this as the new 128-token process-fusion baseline; 512/2048 replay still
needs separate validation before claiming long-context promotion.
The companion offline bakeoff
`prefill-128-current-bakeoff-latest.json` compares the current 128-token
custom/key-cache baseline against the already collected resident-group32,
fused-rope-split, and related variants. It retains the baseline: the fastest
candidate was only 0.007s faster end-to-end (`ratio=1.000x`) and is classified
as a tie, while the other variants are flat or slower. Treat the existing
128-token key-cache profile as the regression target until a candidate clears
the result-bakeoff total-latency gate.
`suggested_prefill_acceleration_flags` turns that selectable set into argv-style
`--prefill-linear-backend ... --require-prefill-acceleration` flags and carries
the required `--run-mpsgraph-probe` replay flag plus probe-satisfied status for
the current selectable MPSGraph path. This lets launch scripts opt into the best
wired acceleration path without treating a probed but unselectable MPP runtime
as usable. `prefill-plan --inspect-backend` also preserves a non-default
`--probe-timeout-seconds` as the prepared launch flag
`--prefill-backend-probe-timeout-seconds`, so M5 first-run compiler timeouts
remain reproducible when a plan is saved and replayed. On the
current M5 Max path,
MTLTensor/Metal 4 ML runtime can be available while MPP tensor ops remain
disabled if the installed SDK does not expose `mpp::tensor_ops`;
`mpp_runtime_available` requires those SDK symbols, a successful local compile
probe, the full Metal 4 ML runtime selector set, and a successful tiny ML tensor
allocation, including the tensor size/alignment selector needed before creating
ML tensor-backed buffers. If `--run-mpp-probe` is requested, the MPP runtime
also requires that tiny execution probe to pass. Probe path, request status,
host-probe pass/fail state, probe timeout, and short failure details are
included in JSON, health, and text output, including the tiny tensor allocation
error when Metal reports one, so SDK/toolchain bring-up failures are visible
without rerunning under a debugger.
When Metal 4 ML is available but the selected SDK lacks public MPP tensor-op
symbols, the neural-accelerator status reports
`missing_public_mpp_symbols` even if an explicitly requested MPP run probe also
fails, while preserving the compile/run error details for toolchain debugging.
Prepared generation, benchmark, inspection, and serving also accept
`--compile-mpp-probe`, `--run-mpp-probe`, `--run-mpsgraph-probe`, and
`--prefill-backend-probe-timeout-seconds` (default 5 seconds). They keep the
default fast health path unchanged, but let an M5/Metal 4 launch check ask the
same backend probe to compile the MPP tensor-op smoke kernel, run the tiny MPP
matmul execution probe, and run a 2x2 MPSGraph matmul without loading model
weights; the timeout can be raised for first-run MPSGraph/Metal compiler cold
starts. `--compile-mpp-probe` and `--run-mpp-probe` require the host probe, and
prepared health surfaces `mpp_compile_probe_requested`, `mpp_compile_probe_*`,
`mpp_run_probe_*`, and `mps_graph_probe_*` fields. Prepared commands that
set `--require-prefill-acceleration` or a positive
`--prefill-min-accelerated-flop-fraction` now require the selectable MPSGraph
path to have a passing `--run-mpsgraph-probe` result, so a launch cannot treat
SDK-visible MPSGraph symbols as proven acceleration. The resulting prepared
health gate includes a stable `reason_code` such as
`selectable_mpsgraph_runtime_probe_required`, `mpp_runtime_not_selectable`, or
`no_accelerated_backend` for launch scripts that should not parse prose.

Run one bounded resident projection over an `M x hidden` f32 prompt batch:

```bash
python -m largerlm prefill-rmsnorm-batch metal/largerlm-runner \
  /path/to/resident/layout.json \
  --layer 0 --norm-suffix .input_layernorm.weight \
  --input-f32 /tmp/prompt_chunk.f32 --batch-tokens 2240 \
  --output-f32 /tmp/prompt_chunk_norm.f32 \
  --max-runner-scratch-mib 4096

python -m largerlm prefill-linear-batch metal/largerlm-runner \
  /path/to/resident/layout.json \
  --layer 0 --tensor-suffix .self_attn.q_a_proj.weight \
  --input-f32 /tmp/prompt_chunk_norm.f32 --batch-tokens 2240 \
  --output-f32 /tmp/q_a_proj.f32 \
  --max-resident-matrix-mib 512 --max-runner-scratch-mib 4096

python -m largerlm prefill-attention-prefix metal/largerlm-runner \
  /path/to/resident/layout.json \
  --layer 0 --input-f32 /tmp/prompt_chunk.f32 --batch-tokens 2240 \
  --output-dir /tmp/layer0_prefix \
  --max-resident-matrix-mib 512 --max-runner-scratch-mib 4096

python -m largerlm prefill-attention-projections metal/largerlm-runner \
  /path/to/resident/layout.json \
  --layer 0 --input-f32 /tmp/prompt_chunk.f32 --batch-tokens 2240 \
  --output-dir /tmp/layer0_projections \
  --max-resident-matrix-mib 512 --max-runner-scratch-mib 4096

python -m largerlm prefill-cache-write \
  /tmp/decode_cache_layout.json /tmp/decode_cache.bin \
  --layer 0 --input-f32 /tmp/layer0_projections/kv_a_proj_with_mqa.f32 \
  --start-position 0 --batch-tokens 2240 \
  --max-cache-file-mib 32768 --max-cache-write-mib 4096

python -m largerlm dsa-indexer-batch \
  /path/to/resident/layout.json /tmp/decode_cache_layout.json /tmp/decode_cache.bin \
  --layer 0 --hidden-f32 /tmp/prompt_chunk.f32 \
  --q-resid-f32 /tmp/layer0_projections/q_a_layernorm.f32 \
  --output-indices-json /tmp/layer0_dsa_topk.json \
  --output-indices-u32 /tmp/layer0_dsa_topk.u32 \
  --start-position 0 --batch-tokens 2240 --context-length 2240 \
  --index-topk 2048 --index-n-heads 32 --qk-rope-dim 64 \
  --max-cache-file-mib 32768 --max-cache-write-mib 4096 \
  --max-cache-read-mib 4096 --max-runner-scratch-mib 4096

python -m largerlm prefill-rope-batch metal/largerlm-runner \
  --q-b-f32 /tmp/layer0_projections/q_b_proj.f32 \
  --k-rope-f32 /tmp/layer0_projections/kv_a_rope.f32 \
  --output-dir /tmp/layer0_rope \
  --batch-tokens 2240 --num-heads 96 --qk-nope-dim 128 --rope-dim 64 \
  --start-position 0 --max-runner-scratch-mib 4096

python -m largerlm prefill-mla-attention-batch metal/largerlm-runner \
  /path/to/resident/layout.json /tmp/decode_cache_layout.json /tmp/decode_cache.bin \
  --layer 0 --q-nope-f32 /tmp/layer0_rope/q_nope.f32 \
  --q-rope-f32 /tmp/layer0_rope/q_rope_rotated.f32 \
  --output-f32 /tmp/layer0_attn_value.f32 \
  --context-length 2240 --start-position 0 --batch-tokens 2240 \
  --indices-u32 /tmp/layer0_dsa_topk.u32 --index-topk 2048 \
  --num-heads 96 --qk-nope-dim 128 --rope-dim 64 --v-head-dim 128 \
  --max-cache-read-mib 4096 --max-runner-scratch-mib 4096

python -m largerlm prefill-attention-output-batch metal/largerlm-runner \
  /path/to/resident/layout.json \
  --layer 0 --attn-value-f32 /tmp/layer0_attn_value.f32 \
  --residual-f32 /tmp/prompt_chunk.f32 \
  --output-f32 /tmp/layer0_attn_hidden.f32 --batch-tokens 2240 \
  --max-resident-matrix-mib 512 --max-runner-scratch-mib 4096

python -m largerlm prefill-attention-block-batch metal/largerlm-runner \
  /path/to/resident/layout.json /tmp/decode_cache_layout.json /tmp/decode_cache.bin \
  --layer 0 --input-f32 /tmp/prompt_chunk.f32 \
  --output-dir /tmp/layer0_attention_block \
  --output-f32 /tmp/layer0_attn_hidden.f32 \
  --start-position 0 --batch-tokens 2240 \
  --num-heads 96 --qk-nope-dim 128 --rope-dim 64 --v-head-dim 128 \
  --max-cache-write-mib 4096 --max-cache-read-mib 256 \
  --max-resident-matrix-mib 512 --max-runner-scratch-mib 4096

python -m largerlm prefill-dense-mlp-block-batch metal/largerlm-runner \
  /path/to/resident/layout.json \
  --layer 0 --input-f32 /tmp/layer0_attn_hidden.f32 \
  --output-dir /tmp/layer0_dense_mlp \
  --output-f32 /tmp/layer0_hidden_out.f32 --batch-tokens 2240 \
  --max-resident-matrix-mib 512 --max-runner-scratch-mib 4096

python -m largerlm prefill-routed-mlp-block-batch metal/largerlm-runner \
  /path/to/experts/layout.json /path/to/resident/layout.json \
  --layer 3 --input-f32 /tmp/layer3_attn_hidden.f32 \
  --output-dir /tmp/layer3_routed_mlp \
  --output-f32 /tmp/layer3_hidden_out.f32 --batch-tokens 2240 \
  --top-k 8 --include-shared-expert \
  --expert-read-advise-align-kib 4 \
  --max-slot-mib 256 --max-router-mib 64 --max-runner-scratch-mib 4096

python -m largerlm prefill-staged-routed-mlp-block-batch metal/largerlm-runner \
  /path/to/experts/layout.json /path/to/resident/layout.json \
  --layer 3 --input-f32 /tmp/layer3_attn_hidden.f32 \
  --output-dir /tmp/layer3_staged_routed_mlp \
  --output-f32 /tmp/layer3_hidden_out.f32 --batch-tokens 2240 \
  --top-k 8 --expert-stage-align-kib 4 \
  --include-shared-expert --max-resident-matrix-mib 512 \
  --max-stage-mib 4096 --max-compact-stage-mib 4096 \
  --max-slot-mib 256 --max-router-mib 64 --max-runner-scratch-mib 4096 \
  --moe-token-block auto --static-capacity-per-expert 128
```

Add `--expert-stage-tiling` when the routed experts selected for a prompt chunk
cannot fit under the single-stage or compact-stage caps. That path uses the
tiled staged MoE executor internally: each tile stages a bounded expert subset,
gathers only the tokens that route to those experts, runs the existing staged
MoE kernel, and scatter-adds tile outputs into the full routed MLP output before
the shared-expert and residual-add phases. The JSON result keeps legacy
`stage_result`/`staged_moe` fields for the first tile and adds
`tiled_staged_moe` with all per-tile stage and runner records; top-level byte,
static-capacity, command-count, copy-time, and peak estimates are aggregated
across all tiles. Explicit `--static-capacity-output-json` and
`--static-capacity-output-bin` paths are intentionally rejected with tiling
because each tile writes its own route table.

The DSA top-k JSON output and compact u32 sidecar in that prompt path are
written through temporary files and atomic replace, so a failed publish does
not clobber previous artifacts.

After router JSON has been emitted, static expert capacity can be checked
without running a kernel:

```bash
python -m largerlm plan-batch-expert-io /path/to/experts/layout.json \
  --layer 3 --router-json-dir /tmp/layer3_staged_routed_mlp/router_json \
  --static-capacity-per-expert 128 \
  --static-capacity-output-json /tmp/layer3_static_capacity.json \
  --static-capacity-output-bin /tmp/layer3_static_capacity.bin --json
```

The static capacity report maps routed tokens into fixed `[expert, capacity]`
slots and lists overflow assignments explicitly, which is the boundary future
ANE/ML tensor expert kernels should use before allocating fixed-shape buffers.
Writing the artifact is strict by default: if the selected capacity overflows,
the command fails unless `--allow-static-capacity-overflow` is set for analysis.
Static capacity values must be real positive integers, so booleans and floats
cannot accidentally size a fixed route table.
The JSON and binary artifacts are written through temporary files and atomic
replace, so a failed write does not leave a half-written route table for the
runner to consume. Low-level staged MoE commands accept
`--no-static-capacity-json` to skip the debug JSON artifact and write only the
binary route table.
For prompt/generation wrappers, a fixed strict capacity below the prompt chunk
size is rejected before work files are created because it cannot guarantee an
overflow-free route table in the worst case; use `auto` for the conservative
no-overflow setting. Prompt/generation wrappers default to binary-only static
routes, avoiding a potentially large JSON slot table during real GLM-5.2 runs.
The staged routed MoE commands can emit the same artifact directly after compact
staging; that artifact uses compact expert ids matching `compact_layout.json`,
with the original expert mapping recorded there. The JSON file is for inspection;
the binary file is the fixed expert-major route table consumed by the Metal
runner through `--routes-bin` without rebuilding per-token JSON routes. Normal
no-overflow static tables stay pre-sorted by compact expert, so the runner can
skip the assignment sort on that path.

Resident batch-linear based prefill commands accept
`--prefill-linear-backend custom-metal|mpsgraph-f32|auto`. The low-level
`prefill-linear-batch` default remains `custom-metal`; `prefill-prompt` and the
generation wrappers default to `auto`. `mpsgraph-f32` forces the local MPSGraph
matmul prototype with F32 compute, converting BF16/F16 resident matrices into a
bounded F32 matrix buffer when needed. `auto` uses MPSGraph only for large
resident batch GEMMs (currently at least 2048 prompt tokens and a 4096 minimum
matrix dimension after bounded M5 Max calibration) and keeps smaller operations
on the custom Metal path; prepared offline generation and serving resolve `auto` to
`custom-metal` when local backend inspection cannot verify MPSGraph runtime
support. Prepared benchmarks reuse the same request-admission backend resolution
for the actual measured runner call, so `auto` cannot pass admission as a safe
fallback and then dispatch a different resident GEMM backend. Strict launch
audits also require the checked request's effective prefill backend to match the
runtime-resolved backend. Tune that selector with
`--prefill-mpsgraph-min-batch-tokens` and
`--prefill-mpsgraph-min-matrix-dim` when benchmarking M5/MPSGraph crossover
points. Resident batch-linear
JSON reports `matrix_scratch_bytes`, `matrix_f32_bytes`, and
`matrix_raw_conversion_bytes`, so MPSGraph BF16/F16 conversion overhead is
visible when tuning `--max-runner-scratch-mib`; it also reports
`elapsed_seconds` for the wrapped runner invocation so small MPSGraph/custom
Metal crossover sweeps have a timing signal. Use `prefill-linear-calibrate` to
run a bounded synthetic resident GEMM sweep over custom Metal and MPSGraph; pass
`--matrix-shapes INxOUT,...` to match rectangular GLM projection shapes. The
sweep also rejects total synthesized matrix/input/output temp files above
`--max-calibration-work-dir-mib` and requires the target volume to keep
`--calibration-work-dir-free-margin-mib` free MiB beyond that estimate before
creating the work directory. When the measured grid supports a safe threshold,
it emits `suggested_launch_profile` with replayable `--prefill-mpsgraph-min-*`
flags. The runner skips custom Metal
batch-kernel compilation on the MPSGraph path, keeping that fallback isolated
from custom shader compile latency or failures. Prepared `/health` reports the
same `auto` threshold policy plus the effective runtime backend, and
`inspect-prepared` request checks summarize the resident matrix mix that would
use MPSGraph versus custom Metal for the resolved prompt chunk. The same summary
includes a bounded `top_matrices` list, sorted by estimated FLOPs, with each
matrix's shape, dtype, resolved backend, scratch bytes, and raw conversion
bytes.
The runner validates resident matrix dimensions without integer wraparound and
stats resident batch-linear input files before reading them, so direct runner
use still rejects unsafe shapes or scratch limits before allocating the prompt
batch input.
`dsa-indexer-batch` is a bounded Python correctness baseline for GLM full-indexer
layers: it writes rotated `indexer.wk` keys into the `dsa_index` cache segment
and emits causal top-k token indices from `indexer.wq_b` plus
`indexer.weights_proj`. Its resident matrix reader supports F32/BF16/F16 and
MLX MXFP4 `U32 .weight` plus `U8 .scales` matrices, with both compressed bytes
and f32 expansion bytes counted before any matrix is expanded. It fixes the DSA
cache/top-k interface before that work moves fully into Metal/MPP.
`prefill-mla-attention-batch` accepts
`--indices-u32 --index-topk` to run the indexed Metal MLA path, staging only the
selected cache rows instead of converting the full context prefix to F32.
`prefill-prompt --model-config` now derives `indexer_types`, `index_topk`,
`index_n_heads`, `index_head_dim`, and `indexer_rope_interleave` from GLM
configs: full-indexer layers write a compact `dsa_topk.u32`, and shared-indexer
layers reuse the latest full layer's file. `prefill-plan` budgets the
full-indexer Q projection as
`index_n_heads * index_head_dim`, matching the executor's `indexer.wq_b` shape,
and separates the `wk`, `wq_b`, and `weights_proj` resident GEMMs.
`preflight-glm` also validates GLM attention tensor shapes from
`q_lora_rank`, `kv_lora_rank`, head dimensions, and hidden size, plus
full-indexer tensor shapes from `index_head_dim`, `index_n_heads`,
`hidden_size`, and `q_lora_rank`. Dense MLP, shared expert, router gate, and
router correction-bias shapes are checked against the same config before
packing. Config loading normalizes DSA schedule aliases and rejects
`indexer_types` or `index_topk_pattern` schedules that contain unknown values or
do not cover all hidden layers, or place a shared-indexer layer before any full
indexer, keeping decode-cache planning conservative. Request admission applies
the same full-before-shared rule to the selected layer walk, so running only a
shared layer without its previous full layer is rejected before work files are
created.
Use `--disable-dsa-indexer` to force the non-indexed attention path.

Run a readiness preflight before packing or launching decode:

```bash
python -m largerlm preflight-glm /path/to/checkpoint \
  --quantize-bf16-affine-int4 --group-size 64 \
  --max-context-tokens 32768 --max-cache-gib 16 \
  --output-dir /path/to/checkpoint/largerlm_packed \
  --disk-margin-gib 32 --unified-memory-gib 128
```

`preflight-glm` is header-only: it reads `config.json`, safetensors headers,
and tokenizer metadata, but it does not load model tensors into memory. Use
`--load-tokenizer` only when you want to instantiate the local tokenizer
backend as part of the check. For GLM DSA checkpoints, preflight also checks
that safetensors shard paths stay inside the model directory, every shard-header
tensor is listed in `weight_map`, and every tensor's dtype, shape, and byte span
agree. The scanner accepts the standard safetensors dtype names plus common
long/lowercase aliases such as `uint32`, `float16`, and `bfloat16`, while
preserving the source dtype string for later layout diagnostics. Each shard's
tensor spans must also cover the payload contiguously without overlap. When the
index includes `metadata.total_size`, it must also match the summed indexed
tensor bytes. A single unindexed `.safetensors` file is accepted for small smoke
checkpoints; multiple `.safetensors` shards without
`model.safetensors.index.json` fail closed instead of guessing shard ownership.
For very large checkpoints, generate a metadata-only bundle before downloading
hundreds of GiB of weights:

```bash
python scripts/fetch_safetensors_headers.py mlx-community/GLM-5.2-mxfp4 \
  --fetch-small-files \
  --output-dir /path/to/glm-5.2-mxfp4-headers
```

To inspect a downloaded, partially downloaded, or metadata-only artifact without
opening any shard payloads, run:

```bash
python -m largerlm checkpoint-status /path/to/GLM-5.2-mxfp4 \
  --repo mlx-community/GLM-5.2-mxfp4 \
  --write-missing-shards-json /path/to/missing-shards.json \
  --write-missing-shards-urls /path/to/missing-shards.txt \
  --write-external-download-json /path/to/external-download.json \
  --write-external-download-sh /path/to/external-download.sh \
  --write-status-json /path/to/checkpoint-status.json \
  --require-download-disk-ok \
  --json
```

`checkpoint-status` reads only small JSON files plus `stat()` results. It
reports expected/present/complete/missing/partial shard counts, total expected
file bytes when a header manifest is present, and next-step commands for
`hf download`, header fetching, download precheck, the no-weight
`prefill-backend --write-report` probe, post-copy checking, `preflight-glm`,
and a safe `prepare-glm` dry-run. It also checks the target volume's free space
against remaining
expected shard bytes plus `--download-disk-margin-gib` (default 16 GiB), so a
large GLM-5.2 download can fail early instead of filling the SSD. Add
`--require-download-disk-ok` when a script should stop unless the remaining
download bytes plus the safety margin fit on the target volume. Add
`--require-preflight-ready` for CI-style metadata readiness, or
`--require-complete` when the full shard download must be manifest-proven before
continuing. When a valid header manifest is present but shard download is not
manifest-proven complete, the suggested preflight and prepare dry-run commands
include `--metadata-only`. The optional missing-shard JSON and URL-list outputs
are meant for external download machines. `--write-external-download-json`
adds target paths, expected byte sizes, and safetensors header metadata when a
header manifest is available. The range downloader uses that metadata to reject
or discard stale partial files before appending bytes, so a wrong local prefix
cannot silently become a same-size corrupted shard. `--write-external-download-sh`
writes a resumable `curl` script that checks every completed shard's exact byte
count, wraps `curl` failures in an outer retry loop, and retries transient
`Unsupported content type` bodies before failing; set
`LARGERLM_MODEL_DIR=/download/path` when running that script on another machine,
and tune `LARGERLM_DOWNLOAD_RETRIES` or
`LARGERLM_DOWNLOAD_RETRY_SLEEP_SECONDS` for flaky links. For very large
checkpoints, the same script can be safely batched with
`LARGERLM_DOWNLOAD_START_INDEX`, `LARGERLM_DOWNLOAD_END_INDEX`, or
`LARGERLM_DOWNLOAD_MAX_BYTES`; completed shards are still exact-size checked and
skipped on later resumes. `LARGERLM_DOWNLOAD_CONNECT_TIMEOUT_SECONDS`,
`LARGERLM_DOWNLOAD_LOW_SPEED_LIMIT_BYTES_PER_SECOND`, and
`LARGERLM_DOWNLOAD_LOW_SPEED_TIME_SECONDS` prevent long zero-byte hangs on flaky
networks while preserving resumable partial files. Copy the completed
shards back with the original filenames, then re-run
`checkpoint-status --verify-local-headers --require-complete --require-clean`.
That extra check reads only safetensors headers and file sizes, not tensor
payloads, and rejects missing, partial, extra, wrong, or corrupt shards before
packing starts.
The JSON fields `download_precheck_command` and `post_copy_check_command`
contain those two safe `checkpoint-status` command lines directly, preserving
the source repo/revision/endpoint and download margin used by the status run.
Use `--write-status-json` to atomically preserve the full report as an audit
artifact; unlike the missing-shard and next-bringup files, it includes the
complete shard table, local header-check result, bring-up plan, and all current
readiness gates from that run.
`bringup_plan` also lists the full ordered path from header fetch through the
first audited 1-token smoke, with each step carrying its command availability,
`step_status` (`complete`, `ready`, or `blocked`), blocking reason,
prerequisites, and whether it writes artifacts, reads weight payloads, or runs
the model. The human-readable report prints the same plan as a compact
complete/ready/blocked checklist. `next_bringup_step` is the first ready command
after skipping currently complete steps. It is a safe automation starting point,
not proof that every earlier prerequisite has already been executed outside the
current status run. Use `--write-next-bringup-json PATH` to write that exact
next-step argv for automation, or `--write-next-bringup-sh PATH` to write a
quoted executable shell script for human review; both only write files and do
not execute the command. Downstream completion is based on bounded small-JSON
validation of `largerlm-prepared/prefill-backend-report.json`,
`preflight-report.json`, `prepare-dry-run-report.json`, `manifest.json`,
`launch-profile.json`, and `launch-audit.json`, not file existence alone;
corrupt, empty, unsafe, failed, stale, or oversized artifacts keep the
corresponding bring-up step ready instead of being treated as complete. The
prefill backend report is also bound to the default
`metal/prefill-backend-probe`, a successful host/MPSGraph runtime probe, an MPP
compile/run-probe attempt, and the GLM-5.2 bring-up 30 second probe timeout, so
older or foreign acceleration reports are not accepted as current M5 evidence. The
prefill backend CLI records the probe binary SHA-256 for new reports; when that
field is present, `checkpoint-status` also verifies it against the current
probe binary.
prefill backend report, metadata preflight report, and prepare dry-run report
are intentionally placed before weight download so M5 MPSGraph fallback/public
MPP availability, GLM-5.2 shape compatibility, layout planning, cache budget,
and disk budget can be recorded without touching large model payloads.
After a post-copy status run has proven local headers, JSON/text reports also
include `prepare_execute_command`, `inspect_prepared_command`,
`launch_audit_command`, and `minimal_smoke_command`. The execute command writes
`largerlm-prepared` with `prepare-glm --execute` and records a bounded cold-read
benchmark. The generated preflight/prepare commands explicitly use
`--group-size 32 --max-cache-gib 16 --disk-margin-gib 32 --unified-memory-gib
128`, matching the first-pass M5 Max 128 GiB GLM-5.2 MXFP4 bring-up envelope,
and atomically write
`preflight-report.json` plus `prepare-dry-run-report.json` so reruns can resume
past successful metadata-only gates without trusting stdout logs. The inspect
command checks the prepared memory/profile/request envelope without generating
tokens and also runs the tiny MPSGraph and MPP tensor-ops probes with a 30
second backend probe timeout before writing `launch-profile.json`; the launch
audit command re-runs inspect with that profile locked, replays those probes,
and writes `launch-audit.json`. The generic checkpoint bring-up path does not
force `--require-prefill-acceleration`; it explicitly passes
`--allow-non-accelerated-prefill-launch-audit` so the audit records that the
default custom-Metal path is intentional. MPSGraph/router-gate acceleration is
kept behind the dedicated shortchat wrappers and explicit
`--allow-router-gate-only-prefill-acceleration` gate. The minimal smoke command
is the first actual 1-token
prepared run, requires both the locked launch profile and audit artifact, and
atomically writes `minimal-smoke.json` with schema
`largerlm.prepared_token_generation_result.v1`. `checkpoint-status` validates
that small result JSON before marking the final bring-up step complete, including
the current prepared manifest path, locked launch-profile path, launch-audit
path, prompt `[0]`, one-token request, and generated token. A rerun can therefore
distinguish a completed first token from a missing, stale, empty, or failed smoke
artifact.
On 2026-07-03 the refreshed local status reported 76/76 complete shards,
`local_headers_ok=true`, `artifact_clean=true`, `launch_profile_valid=true`,
`launch_audit_valid=true`, `minimal_smoke_result_valid=true`, and
`next_bringup_step=null`. The latest real `minimal-smoke.json` generated token
`[15]` in 17.000s with no prompt-prefill phase while replaying the launch-audit
MPP/MPSGraph probe flags, so the default safe path is currently runnable
without requiring another Hugging Face download.

The script writes `model.safetensors.index.json` plus
`largerlm.safetensors.headers.json`; the latter contains only shard headers,
data-start offsets, and file sizes fetched through HTTP Range requests. With
`--fetch-small-files`, it also downloads common small repo files such as
`config.json`, tokenizer metadata, and `generation_config.json` under strict
per-file and total byte caps while skipping missing optional files. Header and
small-file fetches default to three HTTP attempts and automatically retry
transient `{"detail":"Unsupported content type"}` responses from Hugging Face or
a proxy; use `--http-retries` and `--retry-delay-seconds` to tune that behavior.
`preflight-glm` accepts that manifest when the real `.safetensors` shards are
not present. If partial shard files are already present because a download has
started, add `--metadata-only` to prefer the manifest over local shard files.
Tensor naming, dtype, shape, routed/resident byte accounting, and GLM-5.2
coverage can then be checked before the full model download completes. If a
repo uses uncommon tokenizer filenames, add them with repeated
`--small-file PATH` arguments for the strongest metadata-only readiness report.
`prepare-glm --metadata-only` provides the same
manifest-first behavior for dry-run layout/cache/disk-budget checks and is
intentionally incompatible with `--execute`.
A metadata-only probe of `mlx-community/GLM-5.2-mxfp4` confirmed 76 shards,
2,489 tensors, `395094087168` indexed tensor bytes, and the public GLM-5.2
shape. Its resident tensors use MXFP4-style `U32 .weight` plus `U8 .scales`
without affine `.biases`, and the attention path stores absorbed
`embed_q`/`unembed_out` tensors instead of `kv_b_proj.weight`; preflight now
accounts for those logical shapes. Expert packing now emits
`quantization="mlx-mxfp4"` layouts with 32-element scale blocks, and the Metal
runner has MXFP4 routed-expert single-token and batch MoE smoke coverage. The
non-routed resident MXFP4 path now covers bounded embedding rows, 2-D resident
linear/batch projections, router gates, DSA indexer `wk/wq_b/weights_proj`,
attention q/kv/o projections, shared/dense MLP gate/up/down triplets, and
chunked `lm_head` final-logits top-k. The runner also accepts the absorbed
`embed_q`/`unembed_out` attention form: projection runs skip the absent
`kv_b_proj`, and `--run-mla-attention`, batch MLA, and indexed-batch MLA build a
bounded f32 KV-B view from those aliases. A tiny `--run-decoder-layer` command
smoke covers that absorbed path end to end for a single layer, and the Python
batch MLA prefill wrapper now accepts the same aliases. The Python batch
projection and composed `prefill-attention-block-batch` paths also report
`attention_value_source="absorbed-alias"` and skip `kv_b_proj.f32` when the
aliases are present. A metadata-only bring-up in `artifacts/glm-5.2-mxfp4`
completed header fetch, capped small-file fetch, M5 prefill-backend reporting,
`preflight-glm --metadata-only`, and `prepare-glm --metadata-only` with
`--group-size 32`, then advanced to the 76-shard `download_weights` handoff. The
current local artifact has now moved past that boundary:
`checkpoint-status --verify-local-headers --require-complete --require-clean`
reports 76/76 complete shards, `download_complete_proven=true`,
`local_headers_ok=true`, `artifact_clean=true`, `issues=[]`,
`expected_safetensors_file_bytes=395094391502`, and `next_bringup_step=null`.
The remaining work is policy/performance and serving hardening, not another
Hugging Face weight download.
A follow-up local-header audit of `artifacts/glm-5.2-mxfp4` with
`--verify-local-headers --require-complete --require-clean` checked all 76
shard headers successfully. The companion backend probe report
The superseding
`largerlm-prepared/prefill-backend-report-mpp-system-header.json` has SHA-256
`ff4db46d186d87fd04319e5c9e16093f5fcaf41aa9db878d3101a6fdd7a044c2` and
records both `mpp-f32` and `mpsgraph-f32` as selectable and validated.
For prompt lengths beyond the existing short smokes, use
`scripts/long_prompt_readiness_matrix.py` to batch `inspect-prepared` admission
checks without running generation or reading tensor payloads. The current
baseline matrix
`largerlm-prepared/long-prompt-readiness-matrix-latest.json` (SHA-256
`8b166e0d1afaaa9962e75aaba53032775d845ccf831805ed1ccc7722e47c81dc`) admits
128, 512, 2048, and 4096 prompt tokens with one generated token under the
runtime preflight guard. It resolves chunks of 128, 448, 113, and 113 tokens,
requires about 41.37 GiB available memory, sees about 77-78 GiB available at
admission, and keeps expert stage plus compact temp under about 9.56 GiB. The
auto chunk selector now keeps a non-tiled safety cap when rounding down to a
64-token tile would discard more than 25% of safe capacity; on this GLM-5.2
artifact that raises the 2048/4096-token baseline chunk from the old 64-token
alignment to 113 tokens and cuts planned routed-expert read amplification from
the old 32/64 chunk-equivalents to 18.4375/36.875. A candidate
`--max-cache-read-mib 320` counterfactual matrix,
`long-prompt-readiness-matrix-cache320-latest.json` (SHA-256
`2bce43355b78a4b8d88545afa937889330ea6fd7ae99456841e60bffa7da2c58`), is also
admitted and reaches 128-token chunks for 2048/4096 prompts, enabling the
MPSGraph router-gate path for 75 matrices and reducing read amplification to
16/32. Treat that cache-read profile as a launch candidate until it passes a
locked full generation replay and result bakeoff.
The same preflight also reports quantization metadata declared in
`quantization`, `quantization_config`, or common top-level bit/group fields. For
pre-quantized MLX affine-int4 inputs, declared bit width and group size must
match `--quant-bits`/`--group-size`; conflicting config metadata fails before
any expert payload is read. Raw BF16/F16/F32 conversion keeps treating
`--quantize-bf16-affine-int4` as the output-format source of truth and reports
conflicting source metadata as a warning. If config metadata already describes
MLX affine-int4, preflight warns that raw conversion expects BF16/F16/F32 expert
weights, and direct layout construction reports a remove-the-flag error if the
expert tensors are already packed as `uint32` affine weights. If config
metadata is missing but routed expert tensor names look like
GPTQ/AWQ/bitsandbytes layouts
(`qweight`, `qzeros`, `g_idx`, `quant_state`, `absmax`, or `quant_map`),
preflight reports an explicit unsupported expert quantization layout error
instead of only surfacing generic missing `weight/scales/biases` components.
The same diagnostic is used by direct expert layout construction, so scripted
packer entry points fail with the same actionable reason.
Preflight and resident packing validate non-routed resident `uint32 .weight`
tensors with MLX-style affine `.scales`/`.biases` companions and MLX MXFP4
`.scales` companions. `--run-resident-linear` and the custom-metal
`--run-resident-linear-batch` path execute affine-int4 and MXFP4 2-D
projections without expanding the full matrix to F32. `--run-router` and
`--run-router-batch` also execute affine-int4 or MXFP4 router gates directly for
routed expert selection. Routed experts can use affine-int4 or MLX MXFP4; the
MXFP4 path decodes E2M1 values with E8M0 scale bytes and is covered by direct
layer-MoE and expert-major batch smoke tests. Shared-expert, dense-MLP, and
attention projection blocks execute resident MXFP4 2-D matrices directly,
`dsa-indexer-batch` dequantizes MXFP4 indexer matrices behind its existing
matrix/f32-expansion caps, and `final-logits` streams affine-int4 or MXFP4
`lm_head.weight` in bounded row chunks for top-k selection. Absorbed
`embed_q`/`unembed_out` aliases are covered in the low-level projection and MLA
attention runners with the same scratch caps, and Python batch MLA preflight
accepts that value source. The Python batch projection and composed attention
block wrappers now carry the same value-source accounting through JSON and
human-readable reports. For a fresh `mlx-community/GLM-5.2-mxfp4` artifact,
the bring-up sequence is still the full shard download, post-copy local-header
verification, `prepare-glm --execute`, launch audit, and an end-to-end prompt
smoke against the selected quantized artifact. On the local M5 Max 128 GiB
artifact under `artifacts/glm-5.2-mxfp4/largerlm-prepared`, those steps have a
complete prepared package and a conservative audited text smoke:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-text-safe.sh
```

That smoke uses the locked 1-token launch audit with `--metal-final-logits`,
the local `tokenizer.json`, the prepared memory profile, and the 24 GiB live
free-unified-memory guard. It is intentionally decode-only and should complete
without opening the long prompt-prefill envelope. Pass
`--write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-text-latest.json`
to keep the schema-tagged text result instead of reading the full JSON from
stdout. A current 2-token Metal-logits smoke is preserved at
`two-token-smoke-metal-logits-latest.json`; it generated `[15, 15]`, completed
in about 48s, kept the live working set at about 9.48 GiB under the 17.37 GiB
cap, and had about 80.9 GiB available memory at launch. Avoid using the older
non-`--metal-final-logits` MXFP4 smoke profiles for routine checks: a sampled
interrupted run spent its CPU time in Python `_mxfp4_e2m1_to_f32` final-logits
decode rather than in the Metal runner. The current selected replay result for
a minimal real prompt-prefill envelope is preserved at
`seventeen-token-prefill-keycache-kvbcache-warm-replay-fixed-latest.json`. It
replays the locked 17-token single-chunk key-cache profile with the identity-bound
MLA `kv_b` f32 cache directory
`mla-kv-b-cache-resident-d541388189fe9d64`, generated token id `11`, completed
in 107.103s, read about 54.93 GiB total, staged about 27.23 GiB of routed
experts, enabled MLA key and value caches on all 78 attention layers, and
launched with about 80.6 GiB system memory available under the prepared
17.37 GiB live cap. Its selected prefill plan is one 17-token chunk with backend
counts `custom-metal=234`, `fused-metal=387`, and `mpsgraph-f32=75`. Re-run it
quietly with the bounded SSD check and a fresh result artifact:

```bash
python -m largerlm selected-replay-run \
  artifacts/glm-5.2-mxfp4/largerlm-prepared/selected-replay.json \
  --quiet-runner --check-ssd-read-speed \
  --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/seventeen-token-prefill-keycache-kvbcache-warm-replay-fixed-latest.json
```

The generated `selected-replay.sh` also appends `--quiet-runner` by default so
long replays do not stream every low-level runner record to the terminal. The
selected replay JSON now records the selected result's prefill plan signature,
including chunking, backend counts, MLA cache state, and the `kv_b` cache
directory's file count and bytes. The selected cache currently has 78 files and
4,580,179,968 bytes; if that directory is removed or truncated,
`selected-replay-check` will no longer report replay files ready. The previous
no-`kv_b`-cache baseline remains at
`seventeen-token-prefill-keycache-baseline-replay-latest.json`; it completed in
110.205s with the same generated token and prefill plan. The warm `kv_b` cache
reduces MLA value-read time from 5.687s to 1.357s and MLA total time from 7.988s
to 3.670s. The same cache can now be configured on `serve-prepared` with
`--prefill-mla-kv-b-cache-dir artifacts/glm-5.2-mxfp4/largerlm-prepared/mla-kv-b-cache-resident-d541388189fe9d64`,
and prepared health output reports the configured directory before the server
starts. The selected safe server script now uses that same locked copy64
key-cache profile, launch audit, and warm cache directory:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-selected-safe.sh
```

The selected and canonical safe launchers place any extra user arguments before
their fixed audit, memory, prompt, and decode caps, so duplicate safety flags
resolve back to the audited script values. Use `LARGERLM_SERVER_HOST`,
`LARGERLM_SERVER_PORT`, `LARGERLM_CANONICAL_SERVER_HOST`, and
`LARGERLM_CANONICAL_SERVER_PORT` for the intended host/port overrides.

A localhost `/generate-token-ids` smoke through that server wrote
`server-http-selected-warm-cache-smoke-latest.json` and returned HTTP 200 with
token id `11`, one 17-token chunk, the same backend counts, MLA key/value caches
enabled on all 78 layers, runtime preflight `available_memory_ok=true`, and about
80.5 GiB available memory. The compact HTTP response now carries MLA cache
summaries, so `result-compare` treats it as workload-comparable with the selected
replay. A localhost OpenAI-compatible `/v1/chat/completions` smoke through the
same safe server wrote
`server-http-selected-warm-cache-openai-chat-smoke-latest.json` and returned
HTTP 200 for a 17-token rendered chat prompt with one completion token. The
OpenAI response usage was `prompt_tokens=17`, `completion_tokens=1`,
`total_tokens=18`; assistant content was `3`, and the attached
`largerlm.token_result` generated token id `[18]`, completed in 123.555s, used
the same one-chunk backend plan and 78/78 MLA key/value caches, and passed the
runtime memory guard with about 80.4 GiB available memory. Raw OpenAI responses
that carry `largerlm.token_result` are now accepted by `result-summary`, so they
show the same token counts, plan signature, warm-cache replay binding, and
launch-audit binding as wrapped HTTP smoke artifacts. A separate two-token
server envelope is preserved at
`launch-audit-17tok-copy64-accel-keycache-max2.json`, with its inspect payload
at `inspect-17tok-copy64-accel-keycache-max2.json`, and can be launched with:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-selected-safe-max2.sh
```

That script keeps the same selected warm-cache profile and raises only
`--max-new-tokens-cap` to 2. The max2 launch audit passed with 17 prompt tokens,
2 new tokens, `required_context_tokens=19`, MPSGraph-F32 coverage on 75 resident
prefill matrices, decode routed-read guards within the 11.7664 GiB/token and
1.98941s/token caps, and runtime memory guard `available_memory_ok=true`.
A real OpenAI chat smoke wrote
`server-http-selected-warm-cache-openai-chat-max2-latest.json`; it returned
HTTP 200 with usage `prompt_tokens=17`, `completion_tokens=2`, `total_tokens=19`,
assistant content `33`, generated token ids `[18, 18]`, and total elapsed
173.319s after switching decoder-layer MLA from the singleton kernel to the
existing batch-MLA weights/values path with `batch_tokens=1`. The first token
step took 120.377s with the warm-cache one-chunk prefill plan; the second token
decode took 52.908s. `result-summary` now reports decode-layer bottlenecks
directly: for this max2 run, one decode step covered 78 layers, with 51.411s in
decode layers, 46.768s in attention, and 4.623s in MLP. The previous singleton
MLA max2 artifact, `server-http-selected-warm-cache-openai-chat-max2-decode-kvbcache-diagnostic-latest.json`,
confirmed all 78 decode layers received the warm `--mla-kv-b-cache-dir` but
still spent 137.171s in decode layers and 131.206s in attention, so the speedup
came from avoiding repeated per-value softmax score work rather than from
cache-argv plumbing alone. A follow-up single-token value-cache experiment wrote
`server-http-selected-warm-cache-openai-chat-max2-decode-valuecache-latest.json`;
it stayed correct (`[18, 18]`, content `33`) but regressed the second decode
step to 57.182s in decode layers and 52.114s in attention, so decoder
`batch_tokens=1` value-cache materialization is not a default. It remains
available behind `LARGERLM_MLA_VALUE_CACHE_SINGLETON=1` for experiments. The
latest MLA-instrumented max2 run wrote
`server-http-selected-warm-cache-openai-chat-max2-mla-timing-latest.json` and
kept the same output (`[18, 18]`, content `33`) while lowering total elapsed to
141.881s. Its second-token decode layer step took 50.080s, with 45.636s in
attention and 4.423s in MLP. The new nested MLA timing shows that this is now a
Metal compute-kernel bottleneck rather than SSD or cache-file I/O:
`kernel=42.562s`, `total=43.325s`, `value_read=0.665s`,
`cache_read=0.004s`, `setup=0.011s`, and `write=0.071s`. The next performance
target is therefore the default batch MLA weights/values kernels for
`batch_tokens=1`; value-cache materialization remains opt-in because it slowed
this path down. A split-kernel diagnostic run,
`server-http-selected-warm-cache-openai-chat-max2-mla-split-timing-latest.json`,
confirmed the imbalance: `kernel=44.215s`, with `weights=43.891s` and
`values=0.322s`.
A follow-up score-side key-cache A/B wrote
`server-http-selected-warm-cache-openai-chat-max2-mla-keycache-split-timing-latest.json`;
it kept the same output (`[18, 18]`, content `33`), enabled key-cache on all 78
decode layers with about 62.156 MiB of per-runner scratch across the summed
records, and reduced second-token MLA kernel time to `1.862s`
(`weights=1.022s`, `values=0.839s`). The second-token decode layer step dropped
to 17.029s, with 7.193s in attention and 9.810s in MLP. The selected and
canonical safe server launchers now export `LARGERLM_MLA_KEY_CACHE=1` by
default for decode; set `LARGERLM_MLA_KEY_CACHE_DISABLE=1` to force the old
score path for regression checks. The post-change default max2 verification
artifact,
`server-http-selected-warm-cache-openai-chat-max2-keycache-default-latest.json`,
also returned content `33` and generated `[18, 18]`; without split timing
overhead it completed in 137.686s, with the second-token decode layer step at
16.485s, attention at 7.059s, MLP at 9.399s, and summed MLA kernel time at
1.773s. The follow-up MLP-instrumented default run,
`server-http-selected-warm-cache-openai-chat-max2-mlp-timing-latest.json`, kept
the same output and completed in 134.705s. Its second-token decode layer step
was 16.812s: attention 7.221s, MLP 9.564s. The new MLP stage telemetry shows
the next decode bottleneck is MoE work rather than routing or normalization:
`expert_kernel=4.385s`, `expert_read=2.367s`, `shared=1.752s`,
`router=0.591s`, `rmsnorm=0.361s`, `residual=0.050s`. The next candidate
optimization is therefore decode MoE expert execution/slot reading, with
read-advice and fused selected-expert execution as the likely first experiments.
The first read-advice A/B,
`server-http-selected-warm-cache-openai-chat-max2-mlp-readadvise-latest.json`,
added `--expert-read-advise-merge-gap-kib 128 --expert-read-advise-align-kib 4`
to the selected max2 server and stayed correct, but it is not a default:
`expert_read` only moved from 2.367s to 2.308s while MLP total rose from 9.562s
to 9.798s and total elapsed rose to 140.350s. Fusing selected-expert execution
or reducing per-expert command overhead is the better next direction. An
opt-in selected-expert preload experiment,
`LARGERLM_MOE_DECODE_PRELOAD_SELECTED=1`, now preloads all routed expert slots
for a decode layer into one scratch buffer and submits the routed expert work in
one Metal command buffer when it fits the runner scratch cap. The real max2 A/B
artifact,
`server-http-selected-warm-cache-openai-chat-max2-mlp-preload-latest.json`,
confirmed the path enabled on 75/78 MoE layers and stayed correct, but it also
is not a default: MLP total rose to 9.970s, `expert_kernel` to 4.659s, and
`expert_read` to 2.489s. The result points away from coarse command batching
and toward a true fused multi-expert decode kernel or shared/routed MLP fusion.
The next MXFP4 decode path now does the useful part of that fusion directly:
`LARGERLM_MOE_DECODE_MXFP4_FUSED=1` uses the existing group32 MXFP4 fused
SwiGLU and fused down+router-weight add kernels for single-token routed decode,
while keeping the scratch envelope unchanged. A safe single-layer layer_019 A/B
is preserved at `glm-layer19-decode-mxfp4-fused-ab-latest.json`; with nonzero
input it kept router JSON equal and output max error at `4.26e-13`, while
reducing routed expert kernel time from 19.096ms to 9.831ms. The full direct
17-token/2-new-token replay,
`direct-selected-warm-cache-max2-mxfp4-fused-latest.json`, generated the same
`[18, 18]` output and enabled fused decode on 75/78 layers. Its second-token
decode MLP fell from 9.564s to 7.824s, with `expert_kernel` down from 4.385s
to 2.554s; the safe selected/canonical server launchers now export the fused
path by default. Set `LARGERLM_MOE_DECODE_MXFP4_FUSED_DISABLE=1` to force the
old decode expert path for regression checks.
A follow-up short-prompt prefill A/B checked whether forcing the explicit
`custom-metal` prefill backend should replace the tiny MPSGraph router-gate
path for the selected max2 replay. It should not be the max2 default: the audited artifact
`direct-selected-warm-cache-max2-custommetal-prefill-mxfp4-fused-latest.json`
completed safely but generated `[15, 18]` instead of `[18, 18]`, and total time
rose to 136.195s. The policy-aware bakeoff
`max2-custom-metal-policy-bakeoff-latest.json` therefore retains the existing
max2 selected baseline. A separate 1-token policy bakeoff,
`custom-metal-vs-accel-keycache-policy-bakeoff-latest.json`, does select the
fresh custom-metal/key-cache replay: `selected-replay-custom-metal-keycache-1tok-policy-bakeoff-latest.json`
and `.sh` reproduce that scoped winner under the locked launch profile and SSD
read-speed check. Keep that as a short-replay optimization artifact rather than
replacing the max2 selected launcher. A post-download verification run wrote
`selected-replay-custom-metal-keycache-1tok-policy-bakeoff-rerun-latest.json`;
it generated token id `11`, finished in 95.017s, staged 27.21 GiB of routed
experts with 8.284s of copy at 3.285 GiB/s, reported about 79.64 GiB system
memory available at launch, and stayed under the 17.37 GiB live-working-set cap.
`selected-replay-check` still reports the scoped replay `ok=true`, including
current memory, stage-temp disk, SSD speed, launch profile hash, and launch
audit binding checks. The small router-only probe
`scripts/glm_router_gate_consistency.py` wrote
`glm-router-gate-layer19-consistency-latest.json`; layer 19 showed identical
custom fallback/custom-resident logits and only `5.48e-6` max logit drift
between custom and MPSGraph. The probe now records router selection margins as
well: on the latest 17-token layer-19 random-input run, the minimum effective
margin was about `4.56e-4` and only two tokens were under `1e-3`; top-k margins
were all above `7.18e-3`. This suggests that layer-19 alone is not a `1e-5`
near-tie, so the full token difference is likely accumulated cross-layer
near-tie drift rather than an obvious single-layer router bug. Keep custom
router gate as an experiment until a full replay shows unchanged generated
tokens and reports router margins comfortably above the observed backend drift
across routed layers. A full audited max2 replay with margin telemetry,
`direct-selected-warm-cache-max2-router-margins-mxfp4-fused-latest.json`, kept
the baseline `[18, 18]` tokens and launch-audit binding, but it also showed why
that promotion gate is not yet satisfied: across 75 routed layers, the minimum
effective/top-k router margin was only about `2.01e-6`, with 3 token-layer
selections at or below `1e-5` and 27 at or below `1e-4`. That is below the
observed custom-vs-MPSGraph drift from the layer-19 probe, so the default keeps
MPSGraph router gate for now.
`scripts/glm_router_hybrid_policy_analysis.py` is a bounded follow-up that
reads only compact replay telemetry and the router consistency probe, without
launching model kernels. Its current max2 report,
`glm-router-hybrid-policy-analysis-max2-latest.json`, uses the layer-19
`5.48e-6` observed logit drift and a 4x safety multiplier. That keeps global
custom router promotion blocked, but it does identify a possible layer-hybrid
experiment: 67/75 routed layers are above the drift*4 threshold, while layers
5, 7, 47, 55, 56, 60, 61, and 71 would fall back to MPSGraph for this replay.
Using the same probe's custom/MPSGraph elapsed ratio, the static layer-policy
upper-bound saves about 1.66s of the 7.62s router-gate segment, and an online
custom-first layer fallback still estimates about 1.17s savings. Those numbers
are planning evidence only: token-level fallback needs retained router
JSON/logits, and any online hybrid policy still needs a full audited replay
showing unchanged generated tokens before it can become a launch default.
An experimental online layer fallback is available only as an explicit opt-in:
pass `--prefill-router-hybrid-margin-threshold` with a positive effective
margin threshold while keeping `--prefill-linear-backend auto`. The router gate
then computes custom Metal logits first only when `auto` would otherwise choose
MPSGraph; layers above the threshold keep the custom route, and layers at or
below it rerun the router gate with MPSGraph and overwrite the final router
JSON. Prepared health, request checks, launch profiles, and `serve-prepared`
now preserve that threshold explicitly, so a hybrid experiment can be audited
and replayed without relying on shell state. `LARGERLM_PREFILL_ROUTER_HYBRID_MARGIN_THRESHOLD`
remains a legacy fallback for older direct runner scripts when the CLI/profile
value is zero. The default safe launchers leave both unset.
The first audited online hybrid max2 replay used that drift*4 threshold and
wrote `direct-selected-warm-cache-max2-router-hybrid-driftx4-latest.json`. It
kept the baseline `[18, 18]` generated tokens under the same launch-audit and
live-memory guards, with no retained work directory. The run selected custom
router output on 69/75 routed layers and MPSGraph fallback on layers 5, 7, 47,
55, 58, and 72. Result summary now reports the policy explicitly:
`router hybrid: layers=75 threshold=2.193e-05 decisions=custom-metal:69,mpsgraph-f32-fallback:6 elapsed=7.259s fallback_custom_probe=0.517s`.
Total elapsed was 117.003s with 95.467s prompt prefill, so this remains an
experiment rather than a default; promote it only after a same-envelope
result-bakeoff shows a total-latency win while preserving generated tokens.
The strict bakeoff against
`direct-selected-warm-cache-max2-router-margins-mxfp4-fused-latest.json` retained
that baseline: the hybrid candidate had matching `[18, 18]` tokens but
`total_ratio=1.1896`, `winner=null`, and a non-comparable backend signature
because the router policy intentionally changed the prefill backend mix. The
new `selected-replay-router-hybrid-driftx4-bakeoff-latest.json` and `.sh`
therefore replay the retained router-margin baseline, not the hybrid run.
`result-compare` and `result-bakeoff` now also have an explicit
`--allow-prefill-policy-change` mode for backend/routing policy experiments:
it still requires matching prompt shape and generated tokens, but it does not
reject a candidate merely because the prefill backend mix or routed policy
changed the prefill plan signature. The policy-aware hybrid bakeoff,
`router-hybrid-driftx4-policy-bakeoff-latest.json`, treated the same hybrid run
as comparable and still retained the baseline because `total_ratio=1.1896`
with reason `candidate_total_elapsed_slower`. The paired
`selected-replay-router-hybrid-driftx4-policy-bakeoff-latest.json` and `.sh`
are compact replay pointers for the retained baseline.
The same max2 replay with `--prefill-copy-chunk-mib 128`,
`direct-selected-warm-cache-max2-copy128-mxfp4-fused-latest.json`, stayed
correct and generated `[18, 18]`, but it is also not a new default: expert
stage copy improved from 14.652s to 13.839s while total elapsed stayed flat
at 134.792s versus 134.656s for copy64. The selected safe launchers therefore
remain on the copy64 audited profile. The structured bakeoff artifacts
`selected-replay-max2-fused-bakeoff-latest.json` and
`selected-replay-max2-fused-bakeoff-latest.sh` retain that baseline and can be
used as the current replay-ready regression target. `result-summary` still
lists larger copy chunks as possible A/B experiments when stage-copy throughput
looks low, but the text and JSON now mark them as `requires_bakeoff`; do not
promote a copy-chunk change unless `result-bakeoff` shows an end-to-end
total-latency win under the same safe replay/audit envelope.
The first cold materialization run wrote
`seventeen-token-prefill-keycache-kvbcache-cold-replay-latest.json` and took
114.201s because it created the 4.3G cache. A fresh `--prefill-copy-chunk-mib
128` replay with the same key-cache/fused plan wrote
`seventeen-token-prefill-copy128-keycache-fused-replay-latest.json` and is not
the default: it improved expert stage copy from 10.700s to 10.068s, but total
elapsed rose to 113.810s because routed MoE, projection, and attention-output
time regressed. A current old-RoPE-path replay,
`seventeen-token-prefill-keycache-oldrope-replay-latest.json`, is also not the
default: disabling fused split+RoPE changed the runner command back to
`--run-rope-batch`, but total elapsed was 111.146s and RoPE time rose from
7.224s to 7.452s.
The same copy128 check has now been repeated under the policy-compliant
custom-metal/key-cache profile. `launch-profile-17tok-copy128-custom-mla-keycache.json`
and `launch-audit-17tok-copy128-custom-mla-keycache.json` replay safely with the
same 17.37 GiB live-working-set cap. The generated result
`seventeen-token-prefill-copy128-custom-mla-keycache-latest.json` kept token
`[11]` and finished in 95.925s versus 96.012s for the fresh copy64 baseline, but
`copy128-custom-mla-keycache-bakeoff-latest.json` retained copy64 because the
0.999x ratio was within the two-percent tie band.
The wrapper-level value-cache debug switch is honored as well:
`LARGERLM_MLA_DISABLE_VALUE_CACHE=1` now disables the opportunistic MLA value
cache before runner launch. On the current one-chunk key-cache/fused plan,
`seventeen-token-prefill-keycache-novaluecache-replay-latest.json` generated
the same token id `11` with `MLA value cache: enabled=0/78`, but total elapsed
rose to 113.821s and MLA kernel time increased, so value-cache disable is a
debug mode rather than a faster default. A fixed-env verification run also wrote
`seventeen-token-prefill-selected-replay-quiet-novaluecache-fixed-latest.json`;
it reported `MLA value cache: enabled=0/156`, generated the same token id `11`,
and still completed in about 232s on the older two-chunk scorecache replay, so
that replay's slowdown versus its historical 128.8s candidate is not explained
by value-cache materialization alone.

A heavier local prompt-prefill smoke is
also available:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-prefill-128-safe.sh
```

That path uses a locked 128-token custom-Metal profile with expert stage tiling
and Metal final logits. The first successful local run completed in about
425 seconds, generated token id `15`, estimated 22.89 GiB total reads, reported
11.21 GiB actual prompt-prefill stage reads, and kept the live working set under
the prepared 17.37 GiB cap. A later component-instrumented run wrote
`smoke-prefill-128tok-custom-metal-component-stats-result.json` and
`smoke-prefill-128tok-custom-metal-component-stats-summary.json`. It generated
the same token id `15`, kept the same 17.37 GiB live cap, and reported the main
custom-Metal hotspots: `moe.routed_experts_streamed` took about 148s
for 5.80T estimated FLOPs, `attention.o_proj` took about 33s for 2.01T FLOPs,
and `attention.q_b_proj` took about 17s for 0.67T FLOPs. That makes streamed
routed expert execution the first optimization target, followed by resident
MXFP4 attention output projection. The same summary can be regenerated with:

```bash
python -m largerlm result-summary \
  artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-prefill-128tok-custom-metal-component-stats-result.json
```

The custom Metal expert dispatch now uses pipeline-sized 1-D/2-D Metal
threadgroups instead of one output element per threadgroup for the expert
dequant, MXFP4, SwiGLU, and weighted-add kernels. The MXFP4 batch expert path
also fuses the gate/up projections with SwiGLU activation and fuses
`down_proj` with the route-weighted accumulator add, so routed experts no
longer launch separate gate matvec, up matvec, activation, down-output, and
weighted-add kernels. A focused layer-19 GLM MoE microbench can be rerun with:

```bash
python scripts/glm_moe_layer_microbench.py \
  --layer 19 --batch-tokens 128 \
  --experts 11,79,92,103,154,212,236,254
```

Token-tile experiments can be compared with an interleaved repeated sweep over
the same staged experts and random input:

```bash
python scripts/glm_moe_tile_sweep.py \
  --repeat 6 --order interleave --layer 19 --batch-tokens 128 \
  --experts 11,79,92,103,154,212,236,254 \
  --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-moe-layer19-tile-sweep-random-128tok.json
```

On the local M5 Max artifact, the same 128-token/8-expert/1024-assignment
layer-19 microbench now runs in about 0.145-0.158s total after replacing the
MXFP4 E2M1 decode branch chain with a constant lookup table. The routed MoE
runner body reports about 0.076s and the fused Metal kernel section about
0.056-0.058s in the fast steady-state runs. The same timing report shows
expert-slot reads around 0.01s and accumulator read/write around 0.004s, so the
remaining routed-MoE bottleneck is kernel tiling/math throughput rather than
SSD streaming.

`LARGERLM_MOE_BATCH_ACCUMULATOR=memory` is available as a bounded A/B for the
same staged MoE path. On the local M5 Max 128-token layer-19 repeat5 artifact
`glm-moe-layer19-accumulator-bakeoff-128tok-repeat5-latest.json`, the direct
output-accumulator read/write mean fell from about 3.21ms to 1.30ms while adding
3.0 MiB to the runner peak estimate. Runner total also improved in that sample,
but kernel timing noise is large enough that the file-backed path remains the
runner default for maximum batch-size compatibility.
The paired replay-ready 128-token prompt-prefill artifacts
`smoke-prefill-128tok-custom-metal-keycache-file-accumulator-bound-result.json`
and
`smoke-prefill-128tok-custom-metal-keycache-memory-accumulator-bound-result.json`
both generated token id `15` under the same locked request profile and launch
audit. The bound-profile A/B shows 67.401s file-backed versus 63.739s
memory-backed total elapsed, with the same 17.37 GiB live cap and about 79 GiB
available system memory at launch. The stricter project `result-bakeoff`
artifact is `prefill-128-accumulator-bound-result-bakeoff-current.json`; it
selects the memory candidate at `0.946x` total elapsed and confirms strict
prefill plan compatibility and replay-file readiness. That earlier selected
replay records the required replay environment
`LARGERLM_MOE_BATCH_ACCUMULATOR=memory`; the prepared-generation surface now
also exposes the same choice as a launch-profile flag,
`--prefill-moe-output-accumulator memory`, so strict replays do not depend on a
shell environment variable. The selected replay
artifacts are
`selected-replay-prefill-128-memory-accumulator.json` and
`selected-replay-prefill-128-memory-accumulator.sh`; `selected-replay-check` and
`selected-replay-run --dry-run` both pass current memory, stage-temp disk, SSD
speed, launch-audit, and required-environment checks. Because this is still a
single prompt shape, the file-backed path remains the default and the memory
accumulator is the selected 128-token launch experiment. Rerun it with:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-prefill-128-keycache-memory-accumulator-safe.sh
```

The current strict 128-token minimal replay pins the memory accumulator directly
in the launch profile:
`launch-profile-prefill-128tok-persistent-moe-server-tiled-memory-accumulator.json`
plus
`launch-audit-prefill-128tok-persistent-moe-server-tiled-memory-accumulator.json`.
The bound result
`smoke-prefill-128tok-persistent-moe-server-tiled-memory-accumulator-strict-audit-result.json`
generated token id `15`, reports `audit_bound=True`,
`replay_ready=True`, and `files_ready=True`, and stayed inside the
17.37 GiB live-working-set cap with about 78.48 GiB system memory available at
admission. Total elapsed was 60.122s, prompt prefill 60.012s, routed MoE
11.481s, and expert stage copy 1.980s at 5.744 GiB/s. Result summary confirms
the routed MoE slow-layer records use `accum=memory`. The replay-ready selected
entry is now baked as
`selected-replay-prefill-128-persistent-moe-memory-accumulator.json` plus
`selected-replay-prefill-128-persistent-moe-memory-accumulator.sh`. Its
`required_environment` is `{}` because the launch profile pins
`--prefill-moe-output-accumulator memory`; `selected-replay-check` and
`selected-replay-run --dry-run` both pass current memory, stage-temp disk,
launch-audit, profile-hash, and SSD read-speed gates before any weights load.
On the current machine those gates saw about 77.84 GiB available memory versus
41.37 GiB required, and about 21 GiB/s bounded SSD read speed versus the
4.44 GiB/s replay threshold. The latest bounded target experiments also found
no full-replay candidate: `glm-moe-layer19-optimization-target-128tok.json`
kept `tile1_auto_silu` fastest at 0.0439s kernel mean,
`glm-mla-layer37-cache-sweep-128tok.json` kept key+value cache fastest at
0.0552s total mean, and `glm-attn-proj-layer37-fusion-sweep-128tok.json` kept
the fused attention-projection path fastest at 0.0797s wall mean while
separate projections were about 3.23x slower.
Rerun that profile-pinned strict path with:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-prefill-128-persistent-moe-memory-accumulator-safe.sh
```

That wrapper rechecks checkpoint completeness and then dispatches
`selected-replay-run --check-ssd-read-speed --write-result`, so the locked
selected replay gate remains the normal smoke entry point.

A conservative localhost server wrapper for the same profile-pinned strict
128-token memory-accumulator path is also available:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-prefill-128-memory-accumulator-safe.sh
```

It rechecks checkpoint completeness, selected-replay safety, SSD read speed,
the locked request launch profile, and the launch-audit envelope before
starting `serve-prepared`. The wrapper caps prompts at 128 tokens and completions
at one token by default, and the launch profile pins the custom-Metal/key-cache,
persistent MoE, tiled expert-stage, and in-memory accumulator policy. It also
refreshes `prefill-backend-m5-mpp-probe-latest.json` by default before startup;
the current local report shows `prefill_neural_accelerator_status.status` as
`missing_public_mpp_symbols`, MPSGraph probe success, and a clean fallback to
`runtime_prefill_linear_backend=custom-metal`. The reproducible HTTP smoke
entry point is:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-prefill128-persistent-memory-accumulator-smoke-safe.sh
```

It starts the wrapper, waits for `/health`, posts a 128-token
`/generate-token-ids` request, saves the raw response, and terminates the
server. The latest run wrote
`server-http-prefill128-persistent-memory-accumulator-smoke-latest.json`,
returned HTTP 200 with generated token id `[15]`, completed in 60.295s server
elapsed with 60.188s prompt prefill, copied 11.374 GiB of expert stage data in
1.243s at 9.149 GiB/s, and reports `audit_bound=True`, `safe_to_replay=True`,
and `files_ready=True`. `result-compare` marks it workload-comparable with the
offline strict replay at 1.003x total elapsed and no system slowdown.
The older `server-http-prefill128-memory-accumulator-smoke-latest.json` remains
historical evidence for the previous server wrapper.
The same wrapper also passed a real `/generate-text` smoke with prompt `你好`,
`max_new_tokens=1`, and deterministic sampling. It wrote
`server-http-prefill128-memory-accumulator-generate-text-latest.json`, returned
HTTP 200, encoded the prompt as one token, generated text `0` from token id
`[15]`, completed the token result in 5.543s, and passed the runtime memory
preflight. `result-summary` unwraps this text HTTP artifact through the nested
`token_result`, so it reports the same launch-audit binding plus decode-layer
timing for the 78-layer single-token decode.
The OpenAI-compatible chat route also works through this wrapper when short
chat prompts opt out of the locked 128-token batch-prefill chunk with
`"batch_prefill_prompt": false`. The smoke artifact
`server-http-prefill128-memory-accumulator-openai-chat-decodeonly-latest.json`
returned HTTP 200 for `messages=[{"role":"user","content":"你好"}]`,
`max_tokens=1`, usage `prompt_tokens=13`, `completion_tokens=1`,
assistant content `8`, generated token id `[23]`, and
`available_memory_ok=true`. Without that request-level opt-out, the current
128-token locked profile is correctly rejected for this short rendered chat
prompt because `prefill_prompt_chunk_tokens=128` exceeds the request's
safety-capped maximum of 13.

A short-chat server wrapper now covers that usability path without disabling
batch prefill:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-shortchat-64-memory-accumulator-safe.sh
```

It replays
`launch-profile-shortchat-64tok-auto-keycache-request-locked.json` with
`--prefill-prompt-chunk-tokens auto`, binds it to
`launch-audit-shortchat-64tok-auto-keycache-request-locked.json`, and caps the
server at 64 prompt tokens and one generated token. A localhost OpenAI chat
smoke wrote
`server-http-shortchat64-memory-accumulator-openai-chat-latest.json` and
returned HTTP 200 for `messages=[{"role":"user","content":"你好"}]`,
`max_tokens=1`, usage `prompt_tokens=13`, `completion_tokens=1`, assistant
content `0`, and generated token id `[15]`. The request used a 13-token
batch-prefill chunk under the audited 64-token envelope; the drift status is
`shorter_prompt`, the launch audit binding matches, `safe_to_replay=true`, and
the runtime memory guard reports `server_memory_guard_ok=true`.

A max2 variant now extends the same short-chat path to two generated tokens:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-shortchat-64-max2-memory-accumulator-safe.sh
```

It reuses the same locked shortchat64 profile SHA
`176bf4a8290ff4b68f6accd1aae768b6e798c1fb0646e4e490170b137e6bb5f3`, binds
startup to
`launch-audit-shortchat-64tok-auto-keycache-request-max2-locked.json`, and caps
the server at 64 prompt tokens and two generated tokens. The localhost OpenAI
chat smoke
`server-http-shortchat64-max2-memory-accumulator-openai-chat-latest.json`
returned HTTP 200 for the same `你好` chat, usage `prompt_tokens=13`,
`completion_tokens=2`, assistant content `00`, generated token ids `[15, 15]`,
and `replay_max_new_tokens=2`. The first step completed in about 45.79s, the
second decode step in about 5.57s with 78 decode layers, and the launch summary
reports `launch_audit_binding_matches=true`, `safe_to_replay=true`, and
`server_memory_guard_ok=true`.

An optional warm-cache max2 wrapper adds the resident MLA KV-B cache directory
to the same audited shortchat64 envelope:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-shortchat-64-max2-kvbcache-memory-accumulator-safe.sh
```

It wraps the max2 entry above, defaults to port 8085, sets the served model name
to `glm-5.2-mxfp4-largerlm-shortchat64-max2-kvbcache-memory-accumulator-safe`,
and uses
`mla-kv-b-cache-resident-d541388189fe9d64` unless
`LARGERLM_SHORTCHAT_MAX2_KVBCACHE_DIR` overrides it. A real wrapper smoke wrote
`server-http-shortchat64-max2-kvbcache-wrapper-openai-chat-latest.json` and
returned HTTP 200 for the same `你好` chat, usage `prompt_tokens=13`,
`completion_tokens=2`, assistant content `00`, and generated token ids
`[15, 15]`. The request took about 50.07s, with token steps about 43.78s and
3.31s; all 78 decode layers in the second step carried
`--prefill-mla-kv-b-cache-dir`. Its result summary reports
`launch_audit_ok=true`, `launch_audit_binding_matches=true`,
`safe_to_replay=true`, `replay_files_ready=true`, and
`replay_mla_kv_b_cache_ready=true` for the 78-file, 4.58 GB cache. The earlier
non-cache max2 smoke took about 57.14s with a 5.57s second token, so this route
is currently the fastest safe short-chat server path while remaining opt-in.

The same shortchat64 envelope is now extended to four generated tokens:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-shortchat-64-max4-memory-accumulator-safe.sh
artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-shortchat-64-max4-kvbcache-memory-accumulator-safe.sh
```

The current max4 server pins the memory accumulator and decode MLA key-cache in
the locked launch profile instead of relying on
`LARGERLM_MOE_BATCH_ACCUMULATOR` or `LARGERLM_MLA_KEY_CACHE`. It uses
`launch-profile-shortchat-64tok-auto-keycache-memory-accumulator-decode-keycache-request-locked.json`
(SHA `02267e9bf359987ceb049097e9a665d87d38e784ca02e453a43dc9282697db07`)
and binds startup to
`launch-audit-shortchat-64tok-auto-keycache-memory-accumulator-decode-keycache-request-max4-locked.json`
(SHA `4d59cd85210bd10bdb5be55acad2798f562efa5dd66c9848204e27897a4d87d5`).
That audit passes for a 64-token prompt plus four generated tokens with
`required_available_memory_bytes=44416718336`, about 78 GiB available at audit
time, GLM-5.2 public-shape and 4-bit guards, prefill routed-read/stage-temp
guards, decode-read guards, `decode_mla_key_cache=true`, and the same
custom-Metal prefill fallback policy as the current shortchat profile. The
KV-B wrapper defaults to port 8087 and served model name
`glm-5.2-mxfp4-largerlm-shortchat64-max4-kvbcache-memory-accumulator-safe`.
The reproducible HTTP smoke entry point is:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/server-http-shortchat64-max4-kvbcache-memory-accumulator-smoke-safe.sh
```

It starts the KV-B wrapper, waits for `/health`, posts the OpenAI-compatible
`你好` chat with `max_tokens=4`, writes
`server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-latest.json`,
and terminates the server. The latest run returned HTTP 200, usage
`prompt_tokens=13`, `completion_tokens=4`, assistant content `0003`, and
generated token ids `[15, 15, 15, 18]`. The server token result took 55.381s
with token steps about 45.88s, 3.35s, 3.05s, and 3.08s. Prompt prefill used one
13-token chunk, copied 34.664 GiB of expert stage data in 6.875s at
5.042 GiB/s, and the health/request checks report `decode_mla_key_cache=true`,
`server_memory_guard_ok=true`, and `runtime_preflight.available_memory_ok=true`
with about 83.7 GiB available for a 44.4 GiB required envelope. The result
summary reports `audit_bound=True`, `safe_to_replay=True`,
`replay_ready=True`, `files_ready=True`, `replay_max_new_tokens=4`, and
`replay_mla_kv_b_cache_ready=True`. This restores the decode fast path that was
lost in the earlier profile-pinned run
`server-http-shortchat64-max4-kvbcache-memory-accumulator-openai-chat-latest.json`
(72.549s, with roughly 9s decode steps). The older environment-backed max4
smoke
`server-http-shortchat64-max4-kvbcache-wrapper-openai-chat-latest.json` remains
historical evidence at 52.838s for the same generated ids, but it depended on
the shell environment for the decode key-cache policy. `serve-prepared
--require-launch-audit` now rejects server startup when the audit's applied
launch-profile SHA differs from the current server profile, even if the audit's
embedded request profile would otherwise be replay-compatible; server evidence
must be profile/audit file-ready at launch.
A telemetry replay written to
`server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-telemetry-latest.json`
(SHA `b3d6ccc35ccf5e73c0603336884a8b5db15d35e377c430dadb0455d3e0d4e928`)
kept the same HTTP 200, assistant content `0003`, and generated ids
`[15, 15, 15, 18]`. It proves the top-level `prefill_actual_read_time` now
includes serial/unique/planned/waste/coalesced-savings/read-advice/amplification
fields: serial assignment reads were 145.66 GiB, unique/planned reads were
34.664 GiB, alignment waste was 0, coalesced savings were 111.03 GiB,
read-advice covered all 1,678 coalesced ranges with 0 failures, unique read
amplification was 1.0, and expert stage copy took 5.578s at 6.214 GiB/s. This
suggests the current shortchat first-token SSD cost is not alignment overread;
the next stage-copy work should focus on reducing range count/copy overhead or
moving more routed-expert work into a resident/shared fused path without
widening the memory guard. `result-summary` now prints these fields as an
`expert stage io:` line for future real-chat smokes.
A follow-up copy-counter replay,
`server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-copycounter-latest.json`
(SHA `db5907384b58ad1d2b48a0e6910b7460d5d24ca9353a7b318635b66ea68c22d8`),
kept the same output and passed the same launch binding. It adds actual copy
syscall telemetry plus chunk-size counterfactuals: the 34.664 GiB stage was
copied through 5,409 read calls and 5,409 writes, averaging 6.562 MiB per call,
across the same 1,678 coalesced ranges. Estimated read calls by copy chunk are
8 MiB: 5,409; 16 MiB: 3,534; 32 MiB: 1,837; 64 MiB: 1,679; 128 MiB: 1,678.
Runtime preflight still reported `available_memory_ok=true` with 44.417 GiB
required and about 83.583 GiB available. This makes the next copy experiment
concrete: copy32/copy64 can remove most extra read calls, while anything above
64 MiB is already at the 1,678-range floor and needs route/layout changes to
improve further.
A locked copy64 A/B profile is now available through
`serve-shortchat-64-max4-kvbcache-memory-accumulator-copy64-safe.sh` and
`server-http-shortchat64-max4-kvbcache-memory-accumulator-copy64-smoke-safe.sh`.
It uses
`launch-profile-shortchat-64tok-copy64-keycache-memory-accumulator-decode-keycache-request-locked.json`
(SHA `49b491d91fc75ad1448cd2901fb40506ba803e290b2c5d88c55f74b2e311414d`) and
`launch-audit-shortchat-64tok-copy64-keycache-memory-accumulator-decode-keycache-request-max4-locked.json`
(SHA `4de2a00d79c7f88c2b46571d5ac5a91058b6a26e62e7be8b6af9abf162af0860`).
The latest copy64 smoke
`server-http-shortchat64-max4-kvbcache-memory-accumulator-copy64-decode-keycache-openai-chat-latest.json`
(SHA `eff7549c0fcdd5cdde43531ecb9280c7440614a562bd4dd5fbe04be2ea91bf27`)
preserved HTTP 200, assistant content `0003`, and token ids
`[15, 15, 15, 18]`. It reduced the same 34.664 GiB stage copy from 5,409
read/write calls to 1,679 read/write calls, averaging 21.141 MiB per call, but
single-run stage copy time stayed effectively flat at 5.643s versus 5.610s for
the default 8 MiB copy-counter run. Memory guard remained healthy with about
44.417 GiB required and 83.623 GiB available at runtime preflight. Keep copy64
as a reproducible SSD/syscall candidate until a low-noise replay shows a real
end-to-end win.
A copy32 A/B is also reproducible via
`serve-shortchat-64-max4-kvbcache-memory-accumulator-copy32-safe.sh` and
`server-http-shortchat64-max4-kvbcache-memory-accumulator-copy32-smoke-safe.sh`.
It uses
`launch-profile-shortchat-64tok-copy32-keycache-memory-accumulator-decode-keycache-request-locked.json`
(SHA `92685163105ea6d59d83a198e4c9f2e3c583210b7468557c082caab84b279c61`) and
`launch-audit-shortchat-64tok-copy32-keycache-memory-accumulator-decode-keycache-request-max4-locked.json`
(SHA `5f27865e1d818161925f4f5013bd9a5daf657361de34b90ee50495beec5dac79`).
The smoke
`server-http-shortchat64-max4-kvbcache-memory-accumulator-copy32-decode-keycache-openai-chat-latest.json`
(SHA `72466004662d97424cc06f8b09338d8f63dc919f0b1e90ad0214ee38e9bb4133`)
also preserved HTTP 200, `0003`, and `[15, 15, 15, 18]`, with both launch and
runtime memory guards passing. It reduced copy calls to 1,837 read/write calls
but measured a slower 7.336s stage copy at 4.725 GiB/s. Treat copy32 as
negative evidence for promotion: syscall reduction alone is not the bottleneck
on this path, and the next SSD work should target coalesced-range/layout
changes or fused streamed-expert execution.
Stage/layer IO hotspot telemetry is now wired into actual prefill results:
`prefill_actual_read_time` carries `expert_stage_io_stage_count`,
`expert_stage_copy_hotspots`, and `expert_stage_range_hotspots`, while
`result-summary` prints an `expert stage hotspots:` line with chunk/layer/tile,
copy seconds, raw/coalesced ranges, planned bytes, and copy-call count. This
keeps future layout or fused-streaming experiments focused on the specific
stages that dominate copy time or range fragmentation, instead of tuning global
copy chunk size blindly.
A real default-copy safe smoke with the selected-expert hotspot fields,
`server-http-shortchat64-max4-kvbcache-memory-accumulator-decode-keycache-openai-chat-hotspots-selected-latest.json`
(SHA `b81f1e26ca84ea538e18d492af0565491c510d529f695318573a5369879999df`),
again returned HTTP 200, `0003`, and `[15, 15, 15, 18]`. It recorded 75 routed
expert stage calls, 34.664 GiB of planned stage reads, and 5.589s of copy at
6.203 GiB/s. Top copy hotspots were layers 67, 55, 62, 65, and 59; range
hotspots were layers 67, 46, 65, 62, and 57. Layer 67 is worst on both axes:
36 coalesced ranges, 726.750 MiB planned bytes, 112 copy calls, and only two
adjacent expert-id pairs among 38 selected experts.
Use `scripts/stage_hotspot_layout_analysis.py` on that result to produce
`stage-hotspot-layout-analysis-shortchat64-hotspots-selected-v2-latest.json`, a
deduped layout-target and coactivation-order summary that currently ranks
layers 67, 65, 46, 62, and 57 as the first locality targets. In the single
observed prompt, each of those layer-local candidate orders clusters the
selected experts into one simulated range; treat that as an upper-bound signal
until multi-prompt sampling proves the order is stable.
A first three-prompt aggregation is now available at
`stage-hotspot-layout-analysis-shortchat64-hotspots-selected-3prompt-latest.json`
(SHA `da7edd7084915c3a424679ef970b80f2692b7b6294437fe3aa4e92179d6d7d05`).
It combines the original `你好` smoke with short Hangzhou and English AI prompts.
Layer 67 remains the top target across three samples: 44 observed experts,
109 current ranges, 4 simulated candidate ranges, and 0.485s cumulative copy
time over the sampled hotspots. The next safe step is to turn this candidate
order into a dry-run repack/indirection plan and verify range-count reduction
before copying any large expert layer files.
As the runtime-side prerequisite, `plan_expert_io` and `stage_batch_experts`
now understand an optional layer `expert_order` field. It maps logical expert
ids to physical slots for already-repacked layer files while preserving router
semantics; identity layouts remain unchanged.
`scripts/expert_order_repack_plan.py` turns the coactivation analysis into a
dry-run manifest without copying expert files. The first top-5 plan is
`expert-order-repack-plan-shortchat64-3prompt-top5-dryrun-latest.json`
(SHA `a91fed3a207b192f053f5303d4a51047f3d0fc7ea12afad903651f46a19f2ed6`):
layers 67, 62, 46, 57, and 65 would each require repacking a 4.781 GiB layer
file, for 23.906 GiB read plus 23.906 GiB write. The manifest is explicitly
`dry_run=true`, `safe_to_apply_to_existing_layout=false`, and
`requires_layer_file_repack=true`.
`scripts/expert_order_repack_execute.py` is the bounded executor for that plan:
without `--execute` it only emits an execution manifest, and with `--execute`
it repacks selected layers slot-by-slot while hardlinking unchanged layer files.
The real top-5 plan has been dry-run through the executor as
`expert-order-repack-execute-shortchat64-top5-dryrun-latest.json`
(SHA `79ba27d4e657d1457440e383d9aaa2787192232948aae63b32bb87c60f50504e`);
it did not create the requested output directory, and reports 23.906 GiB read,
23.906 GiB write, plus 334.688 GiB of unchanged layer files that would be
hardlinked during an explicit execute run.
A first bounded execute run has now repacked only layer 67 into
`experts-repacked-shortchat64-top1-layer67` using an 8 GiB write cap and 20 GiB
post-write free-space guard. The execute manifest
`expert-order-repack-execute-shortchat64-top1-layer67-executed-latest.json`
(SHA `515c69cac65d53348d4e5939ce7dbdc452bad899c863aa873f564fafb1b25aee`)
reports `executed=true`, 4.781 GiB repack read/write, and hardlinks for the
unchanged layers. Validation artifact
`expert-order-repack-validate-shortchat64-top1-layer67-latest.json`
(SHA `39c0a8321606b55cd008597e4d73fd7c87648a1b21e92fe472f6303db4246250`)
confirms sampled slot heads match and the three observed layer-67 hotspot rows
drop from 109 coalesced ranges to 4.
`scripts/prepared_manifest_variant.py` now writes a small validated shadow
manifest so this repacked expert directory can be tested without modifying the
original `manifest.json`. The first variant,
`manifest-repacked-layer67.json` (SHA
`c7f7c465ea1d7426fedeb86f48c7c23c00519c7354506f6eb32aafbf78561223`), points
only `experts_layout` at
`experts-repacked-shortchat64-top1-layer67/layout.json`. Its locked shortchat64
profile/audit are `launch-profile-repacked-layer67-shortchat64.json` (SHA
`e10a7028a35e54c3032a45443fcee35c990b1b4b7144d4f82df276e60ef01bc1`) and
`launch-audit-repacked-layer67-shortchat64.json` (SHA
`116bd17129efea8a12030d228e9e505aa0a358afad3f58d2285915f20e822595`).
The first audited direct-generation smoke,
`repacked-layer67-shortchat64-max1-generate-tokenids-latest.json` (SHA
`31090658257b9b485e7c0ade414289425d6baa59c7d7dc0fe3133c01add266da`),
generated token id `[15]` with no residual runner process. The repacked layer 67
stage used one raw/coalesced range for 38 selected experts, where the original
shortchat64 hotspot had 36 coalesced ranges.
The same bounded executor has also produced the top-5 layout
`experts-repacked-shortchat64-top5`, repacking layers 67, 62, 46, 57, and 65
under a 32 GiB write cap and 20 GiB post-write free-space guard. The executed
manifest `expert-order-repack-execute-shortchat64-top5-executed-latest.json`
(SHA `65828bef845fd1cdd19f32c66a175b96161e030211c7cf3ff31237799ee90ab3`)
reports 23.906 GiB read/write and layout SHA
`b46af7baaf958e833c1606652f80774822722911ee6ca9fba78f4238de0d7a23`.
Validation `expert-order-repack-validate-shortchat64-top5-latest.json` (SHA
`5e7f2833e0a6e28403f04490dfcdb268132b0c135f5a4124b3306f3fec717c83`)
sampled all five repacked layers successfully and reduced the three-prompt
analysis total from 692 to 254 coalesced ranges. The top-5 shadow manifest
`manifest-repacked-top5.json` (SHA
`a7f723d5e7c1f554e0976896aa344543ef8fa682b7355891364342506ec7e2e1`) has a
locked profile/audit (`d1ccde3793e799b819744fdc270cbb67a0b2e0a41cb2a2092c1c10c367cc9fec`
/ `3437710bb4801757fdb4de68d1f81e0b37b1184def220f3e106db0e7e7f221f0`).
Its first audited 1-token direct-generation smoke
`repacked-top5-shortchat64-max1-generate-tokenids-latest.json` (SHA
`1022c4e89c7c2740d8f635320d59ce98067f9480324697bce3d5deb2240ce76a`)
generated `[15]` with no residual runner. In that real run, layers 46, 57, 65,
and 67 staged as one coalesced range and layer 62 staged as two; total prompt
expert-stage coalesced ranges dropped from 1,643 in the layer67-only run to
1,526.
The top-5 layout also has an experimental MPSGraph 13x32 profile/audit,
`launch-profile-repacked-top5-shortchat64-mpsgraph13x32.json` (SHA
`e3c9c2433aa4495abcd5be29edfdeb383e5c02e15509309b362adefdd8d43450`) and
`launch-audit-repacked-top5-shortchat64-mpsgraph13x32.json` (SHA
`43558ef3809f4f4a6fe871fb0d04053fc8ba4a8cf3dd8a9a9ab9874163f245e5`). Run
Metal/MPSGraph probes outside the restricted sandbox; otherwise the probe can
report `no Metal device`. The audited smoke
`repacked-top5-shortchat64-max1-mpsgraph13x32-generate-tokenids-latest.json`
(SHA `83c36a1292b1458873f0adb5e198f4cf5c987cacb6f8de5f9d758cf40f9ba74d`)
proved actual MPSGraph coverage on 75 router-gate matrices, but generated `[23]`
instead of the custom-metal `[15]`. Keep this profile experimental until router
parity is solved; the safe default remains the custom-metal top-5 profile.
Using that safe top-5 smoke as the new locality source, the next dry-run
planning wave is
`stage-hotspot-layout-analysis-repacked-top5-shortchat64-latest.json` (SHA
`e02648ead64b2fc435328dd25fe6eb04c19a31674f4278221be0ce1b53e9fd5a`) plus
`expert-order-repack-plan-repacked-top5-shortchat64-next5-dryrun-latest.json`
(SHA `ccc031d892fb05bd3d05fa42ab5fdd52c8fe892fb03d863f28f81de24bbc8db5`).
It selects layers 76, 54, 68, 47, and 55; on the observed top-5 prompt these
would reduce the five rows from 30/30/29/28/28 ranges to one range each. The
executor guard dry-run
`expert-order-repack-execute-repacked-top5-shortchat64-next5-dryrun-latest.json`
(SHA `99ee097a164f45aec9626715d2c45ddfabd1a39abe61449e42381a415c8a87f6`)
kept `executed=false`, did not create `experts-repacked-shortchat64-top10`,
and reports 23.906 GiB read/write with the 32 GiB write cap and 20 GiB
post-write free-space floor satisfied. Treat this as the candidate top10
layout plan; execution should wait for either more prompt evidence or a
decision that another 23.906 GiB bounded repack is worth spending on this
single-prompt hotspot.
The repack planner now has an explicit `--min-sample-count` evidence gate.
Replaying the original three shortchat64 hotspot prompts against the top-5
physical layout produced
`stage-hotspot-layout-analysis-repacked-top5-shortchat64-3prompt-latest.json`
(SHA `0981212556403247e8bd72b7ca4f089b3691698528f80c11c458f60fc4e4cd44`).
With `--min-sample-count 2`, the guarded plan
`expert-order-repack-plan-repacked-top5-shortchat64-3prompt-minsample2-dryrun-latest.json`
(SHA `1b963e2663ee9ecc7d92fb488a2b9c57978a256623e812f221519e152488cf0b`)
selects only layer 55 after skipping already-applied layers 67, 62, 46, 57, and
65. The dry-run guard (SHA
`bb6fb6fa1ab310484b436e170e571d10e7c77abec44f366b1bf41d80ac98fd5c`) showed a
4.781 GiB write under the default 8 GiB cap, so this robust top6 step was
executed as `experts-repacked-shortchat64-top6-minsample2-layer55`. Execution
artifact SHA is
`a75c81bfd3f71ad4cb887e9f0c5486512b74a23b67423ed97c46725506d37ef5`, output
layout SHA is `b756215dd142494605b266d6ecbf453b016a17fbacad3115c2b5cb7f947c5cff`,
and validation SHA is
`6123328260b60550890663fc4bc52bb84a20c4b1eefbf6e8839e21c8d808565b`.
The validator now handles incremental repacks whose source layout already has
`expert_order`; all six reordered layers validate, and the three-prompt rows
drop from 254 to 199 coalesced ranges.
The top6 shadow manifest/profile/audit are
`manifest-repacked-top6-minsample2-layer55.json` (SHA
`5b3a1478fe4e1f80a5907e6372a677c6bc0075aa257ed6355c18922a80d34288`),
`launch-profile-repacked-top6-minsample2-layer55-shortchat64.json` (SHA
`52311c0018d3acbf7f96ec542daa70763f8dd94e9ac57b510b1343a768298126`), and
`launch-audit-repacked-top6-minsample2-layer55-shortchat64.json` (SHA
`7ed4453f331fae83a6d43b3390c8cbfb83472bcf548d971e56e0126ddeffc046`).
The audited max1 smoke
`repacked-top6-minsample2-layer55-shortchat64-max1-generate-tokenids-latest.json`
(SHA `b958e52d56d41724160ad4a96895b34349627ccd187ddb9206b10c91de44763d`)
kept token `[15]`, reduced real prompt expert-stage ranges from 1,526 to 1,499,
and copied the same 34.664 GiB stage plan in 5.786s at 5.991 GiB/s. Compared
with top5, `result-compare` reports a comparable workload, no system slowdown,
and total elapsed 238.804s versus 242.862s (`0.983x`), but classifies it as a
tie because the total win is inside the two-percent promotion band. Keep top6
as a validated candidate until another replay or multi-prompt bakeoff clears
the promotion gate.
The top6 result-summary optimization-target queue has now been closed for the
same 13-token prompt. Bounded sweeps for layer55 routed MoE
(`glm-moe-layer55-optimization-target-13tok.json`, SHA
`b6c9a11c93c1b5a86165c6a5321959c0be16f16e824bf2b4c85dd05ee0cbe9c4`),
layer57 MLA cache mode (`glm-mla-layer57-cache-sweep-13tok.json`, SHA
`65087953d11cdac33574667c79867bd4b7957b3553403872215c0409bdfd7e6f`),
layer0 attention projections (`glm-attn-proj-layer0-fusion-sweep-13tok.json`,
SHA `0f42075808d09a48dfd131676b682bfc1eb35097d2c620833488b83d9ced03bb`),
layer57 attention output
(`glm-resident-o-proj-layer57-optimization-target-13tok.json`, SHA
`aea85d6e98a39a12342f546e462c80d629b45dc4889b912c759c9f2c611bdca0`),
and layer27 cache writes (`glm-cache-write-layer27-chunk-sweep-13tok.json`,
SHA `00bd527358e37f4f0aff88a9952f696276a1f2f5c6f113c52230c1fc2bc90042`)
all kept the current default policy. The RoPE split sweep
(`glm-rope-split-layer57-fusion-sweep-13tok.json`, SHA
`45924ee3eacf0fdf53fa8503bc20a64a90624bca25c17947c9cea74c564aaf94`)
did mark the older Python split plus `--run-rope-batch` path as a microbench
candidate (`0.034296s` versus `0.035488s`), but the required full replay
`repacked-top6-minsample2-layer55-oldrope-shortchat64-max1-generate-tokenids-latest.json`
(SHA `1c00f6d3394dd99d11c3ae0e7b316db3c79522d402e121cc6b8490ce13592334`)
remained a tie: token `[15]`, total 235.592s versus 238.804s (`0.987x`), and
`result-bakeoff` retained the fused-RoPE top6 baseline because the win is still
inside the two-percent gate. The bakeoff and selected replay artifacts are
`top6-oldrope-policy-bakeoff-latest.json` (SHA
`1aad770c0d10fa37bd9fa8a514bd67764c4772a7097c01d8858c03d5f18e625f`),
`selected-replay-top6-oldrope-policy-bakeoff-latest.json` (SHA
`5d03a9a3c134a45ba5413e0b38da17dc12b86272f6e5b9752358dca59b525dc6`), and
`selected-replay-top6-oldrope-policy-bakeoff-latest.sh` (SHA
`dea363f7a6ae1ff23f73ad625a82eeab35fefdef7b4d672719c3368ba974d5f3`).
`scripts/multi_prompt_replay_plan.py` now turns existing HTTP/generation result
JSONs into a locked replay matrix without running the model, and can reuse
already completed result JSONs by matching variant plus exact prompt token ids.
`scripts/multi_prompt_bakeoff.py` aggregates the completed per-prompt bakeoffs
with the same replay-file readiness policy. The current top5-vs-top6 plan
`multi-prompt-replay-plan-top5-vs-top6-latest.json` (SHA
`6908904cff5a042ce0237a66743101da7028483829ce69e8385bcefa9820689a`) extracts
three shortchat64 prompts (13, 17, and 19 tokens), verifies both top5 and top6
variants are audit-ready, reuses the existing 13-token top5/top6 results, and
writes the runnable guarded script
`multi-prompt-replay-top5-vs-top6-latest.sh` (SHA
`5e3c7f1afee1c5c3f9189f7ae28dd184b5194405e0add50cf3a3697359e2f88b`).
The first additional matrix sample, the 17-token Hangzhou prompt, generated
`[18]` for both variants. Top5 took 279.733s (SHA
`559009e3ee1ce80714ea7f25611dc7cd97a6837b948946d3d99d3289fe82b665`), top6
took 281.503s (SHA
`20f84813c46ac0f9eba23c7cdcba6f09b277b3ab9c878aab537635f082ff7a55`), and
`result-compare` marked the candidate as a 1.006x tie. The Hangzhou bakeoff
`multi-prompt-top5-vs-top6-hangzhou-bakeoff-latest.json` (SHA
`ce53d5ddeb5432f6bb310986bb2a9c9f1e1e7937ea0e2521f924e095b877e5bb`) retained
top5; selected replay JSON/script SHAs are
`096a3058dc728fa2bb2243c19cb36e6eeb87a791cc589aa9421ee8eee5f45910` and
`97f47ee865f519d0c2ac70956fe79b52827e1db909e76057deeff2a7f2ff9915`.
This second full-replay sample strengthens the decision not to promote top6
yet, even though top6 reduced Hangzhou prompt-prefill time from 88.403s to
87.220s and expert-stage copy from 7.852s to 6.497s.
The aggregate multi-prompt report
`multi-prompt-top5-vs-top6-bakeoff-latest.json` (SHA
`96ef82b762d7de912a5acbf4b299f34dbe13a08e29dab2a5eef27da16336f12c`) now has
all three required prompt pairs complete. Top6 has zero wins, two ties, and one
inconclusive prompt caused by possible system slowdown, so the overall decision
is `retain_baseline` and the selected variant remains top5. The completed
19-token explain-ai pair generated `[15]` on both layouts: top5 took 339.474s
(SHA `ef22c1144a3f4d72d6c8f14d86c40d2f673ff1cc352d712162aa6fa55c2da350`),
top6 took 536.748s (SHA
`3e66ca65f36197659552634a0f57c5cfb8bf260c014dcb6cbd6754c9e33fbb9d`), and
`result-compare` marked the run as `possible_system_slowdown` with ratio
`1.581x`. Its bakeoff
`multi-prompt-top5-vs-top6-explain-ai-bakeoff-latest.json` (SHA
`eb74fef6ef55b8f5163635bf2a74ab9c735c329d1810c29a67eb6e1d20cdfa40`) retained
top5; selected replay JSON/script SHAs are
`cfd65ce9bce2a10e6d5d228334957768bb0203668defef42e186225e053ee3e9` and
`5fae54acf236a34105305402fc474b158c2d71d1dc1794e669e998b019b58052`.

An experimental MPSGraph-prefill max4 route is also available:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-shortchat-64-max4-mpsgraph-memory-accumulator-safe.sh
artifacts/glm-5.2-mxfp4/largerlm-prepared/serve-shortchat-64-max4-mpsgraph-kvbcache-memory-accumulator-safe.sh
```

It uses
`launch-profile-shortchat-64tok-auto-mpsgraph13x32-keycache-request-locked.json`
(SHA `c0cf57af1de2b0eaaad89c331ea220ed4c98d0535286aaaf0451685d5d67cdcd`)
and binds startup to
`launch-audit-shortchat-64tok-auto-mpsgraph13x32-keycache-request-max4-locked.json`
(SHA `45d0402b5a1d529a190deb6b62b97e5277aa526dad6b98dfcd6ce40b497056c8`).
This profile requires `--run-mpsgraph-probe`, `--require-prefill-acceleration`,
`--prefill-mpsgraph-min-batch-tokens 13`, and
`--prefill-mpsgraph-min-matrix-dim 32`. The wrapper now also passes
`--allow-router-gate-only-prefill-acceleration`, because this route is a
declared routing-drift experiment rather than a default-safe acceleration
policy. Both the 64-token audit request and the real 13-token `你好` chat
resolve 75 resident prefill matrices to
`mpsgraph-f32` with `prefill_acceleration_coverage.ok=true`. MPP/Metal ML tensor
ops remain unavailable on the current public SDK because `mpp::tensor_ops`
symbols are missing, so this is the current selectable M5 acceleration fallback.
A real KV-B wrapper smoke wrote
`server-http-shortchat64-max4-mpsgraph13x32-kvbcache-wrapper-openai-chat-latest.json`,
returned HTTP 200, usage `prompt_tokens=13`, `completion_tokens=4`, assistant
content `8333`, and generated token ids `[23, 18, 18, 18]`. The request took
about 55.85s, with token steps about 43.75s, 3.30s, 3.04s, and 3.04s, and its
summary reports `launch_audit_ok=true`, `launch_audit_binding_matches=true`,
`safe_to_replay=true`, `replay_max_new_tokens=4`, and
`replay_mla_kv_b_cache_ready=true`. This route is not the default yet: for the
same prompt it has no measurable speed win over the custom-Metal max4 smoke and
changes the generated ids, so it is retained as a gated M5 acceleration
bring-up artifact rather than a correctness-equivalent serving path.
Fresh result payloads now expose this distinction directly in
`prefill_acceleration_coverage`: `router_gate_matrix_count`,
`router_gate_accelerated_matrix_count`,
`router_gate_accelerated_estimated_flops`,
`non_router_matrix_count`, `non_router_estimated_flops`,
`non_router_accelerated_estimated_flops`,
`non_router_unaccelerated_estimated_flops`,
`non_router_unaccelerated_flop_fraction`,
`non_router_unaccelerated_streamed_routed_expert_*`,
`non_router_unaccelerated_non_streamed_*`,
`unaccelerated_backend_matrix_counts`, `unaccelerated_backend_estimated_flops`,
`accelerated_router_gate_flop_share`, and `accelerated_router_gate_only`. When
`accelerated_router_gate_only=true`, the MPSGraph coverage comes entirely from
MoE router gates; by default that no longer satisfies
`--require-prefill-acceleration`. Pass
`--allow-router-gate-only-prefill-acceleration` only for explicit
routing-drift experiments and keep the route behind launch-audit/token-match
gates instead of promoting it to the default.
The top-5 result-summary-suggested 13-token MLA cache microbench has also been
run as a bounded negative control:
`glm-mla-layer77-cache-sweep-13tok.json` (SHA
`479876356e851b3f256215ac97d86288cc3a7469243fab12cfbb42d654efee45`).
It compares `key-value`, `key-only`, `value-only`, and `none` on layer 77 with
four interleaved repeats and 256 MiB-class cache/read/resident/scratch caps.
All modes matched numerically, but `key-value` remained fastest by total and
kernel mean (`0.04148s` total, `0.0051505s` kernel). `key-only` was 1.078x
slower by total mean, while `value-only` and `none` were about 2.61x and 2.63x
slower. The artifact sets `candidate_for_full_replay=false`, so no MLA cache
mode change should be promoted from this 13-token top-5 workload.
The adjacent attention-projection fusion sweep,
`glm-attn-proj-layer54-fusion-sweep-13tok.json` (SHA
`86c6bc49f35f476a3352bb79c0dd928c3cd83b1a1865d7ecbf3ce4fbda186d83`), is
also negative for a policy change: `fused` stayed fastest at `0.059247s` mean
wall time, while `separate` took `0.232217s` (`3.919x` slower), with exact
output agreement and `candidate_for_full_replay=false`.
The remaining top-5 13-token result-summary targets are now also closed as
bounded experiments. The routed-MoE layer-67 sweep
`glm-moe-layer67-optimization-target-13tok.json` (SHA
`ddb6c44b801832fe7e82d6b678ed984726f36063e3346a0809d074cf4a8b0c3c`) kept
`tile1_auto_silu` fastest (`0.0436305s` kernel mean), while tile2 and
vector-SwiGLU variants were slower. The attention-output layer-8 sweep
`glm-resident-o-proj-layer8-optimization-target-13tok.json` (SHA
`f9b6fec4ad7bb01677074de51778dacf18b70ab67fa2a8092e5637d44700aa92`) kept
`auto`/group32 fastest (`0.01742075s` backend mean); `off` was 1.120x slower by
backend mean. The RoPE split sweep
`glm-rope-split-layer14-fusion-sweep-13tok.json` (SHA
`ebcf15e3ca8e5c9a4bbd685621f927370d1787d3d3f50d3eae38c15e126ca152`) kept the
fused path fastest (`0.048014927s` wall mean), with the old path 1.037x slower.
Only the cache-write chunk sweep
`glm-cache-write-layer12-chunk-sweep-13tok.json` (SHA
`6226efa59d03ae948a91bd289491b09c4d700fdb0692946d1a586beaa6082931`) produced a
microbench candidate: `1MiB` wrote byte-identical cache files and was 0.975x the
default wall mean, but the absolute mean delta is about 38 us on this 13-token
probe. Treat it as a full-replay-required low-priority candidate, not as a
default change.
Older smoke JSONs may not contain per-component telemetry, so
`result-summary` only infers this flag when the artifact itself carries enough
evidence. A fresh post-telemetry smoke,
`server-http-shortchat64-max4-mpsgraph13x32-kvbcache-routergate-telemetry-latest.json`,
returned the same HTTP 200 / `8333` / `[23, 18, 18, 18]` output and reports
`router_gate_accelerated_matrix_count=75`,
`router_gate_accelerated_estimated_flops=3067084800`,
`non_router_accelerated_estimated_flops=0`,
`accelerated_router_gate_flop_share=1.0`, and
`accelerated_router_gate_only=true`.
A second allow-guard smoke,
`server-http-shortchat64-max4-mpsgraph13x32-kvbcache-routergate-allow-guard-latest.json`,
returned HTTP 200, assistant content `8333`, and generated token ids
`[23, 18, 18, 18]`. Its request coverage records
`allow_router_gate_only_acceleration=true`,
`accelerated_router_gate_only=true`, and `ok=true`, confirming the stricter
default rejection and the explicit experimental override both work.
After promoting the online router-hybrid threshold from an environment variable
to a profile/audit flag, an explicit driftx4 experiment wrote
`launch-profile-shortchat-64tok-auto-mpsgraph13x32-routerhybrid-driftx4-keycache-memory-accumulator-decode-keycache-request-locked.json`
(SHA `1d83ef285f25b7a3e2e288984bfa8b22fdb0a67f4a874ded378e3ec862c300ee`),
`launch-audit-shortchat-64tok-auto-mpsgraph13x32-routerhybrid-driftx4-keycache-memory-accumulator-decode-keycache-request-max4-locked.json`
(SHA `1be9bfe26a7fa7fd9c18720d1e869be507d2d71a800e97063c72f2e1f484a70b`),
and
`server-http-shortchat64-max4-routerhybrid-driftx4-kvbcache-memory-accumulator-decode-keycache-openai-chat-latest.json`
(SHA `408a46068d7eaf52d36fd53b36578c37ba7a71843e835fcbabe1f3f5d88f3c7d`).
The profile pins `--prefill-router-hybrid-margin-threshold 2.19345e-05`,
`--prefill-moe-output-accumulator memory`, and `--decode-mla-key-cache`;
its request check passed with `available_memory_ok=true` and
`required_available_memory_bytes=44416718336`. The smoke returned HTTP 200 in
59.344s, request time 56.400s, token steps
`[44.112, 3.342, 3.093, 3.069]`, and `decode_mla_key_cache=true`, but it still
generated assistant content `8333` / token ids `[23, 18, 18, 18]`. Because the
baseline profile generates `0003` / `[15, 15, 15, 18]`, this explicit hybrid
route remains a negative M5 experiment and is not promoted to the default.
`result-summary` now prints this as a non-router acceleration gap. For the
allow-guard smoke, router-gate MPSGraph covers only 3,067,084,800 of
988,112,486,400 estimated prefill FLOPs, while
`non_router_unaccelerated_estimated_flops=985045401600` and
`non_router_unaccelerated_flop_fraction=0.996896`. The fallback breakdown is
`streamed=588880281600`, `non_streamed=396165120000`, and backend FLOPs
`custom-metal=606546690048`, `other=378498711552`. That makes the next M5
optimization target explicit: first the streamed routed expert/custom-Metal
path, then the non-streamed shared/resident fused path, not the already-gated
router projection fallback.

The current fastest audited 128-token local smoke enables the runner's
score-side MLA key cache, and the runner now enables the value-side MLA cache by
default whenever a non-indexed batch fits the configured scratch cap:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-prefill-128-keycache-safe.sh
```

That script replays
`launch-profile-prefill-128tok-custom-metal-keycache-request-locked.json` and
`launch-audit-prefill-128tok-custom-metal-keycache-request-locked.json`. The first
successful run generated token id `15`, kept the same 17.37 GiB live cap,
reported about 80 GiB available system memory at launch, and completed in about
451 seconds. Its summary shows `mla_key_cache` enabled on all 78 layers with
about 491 MB total key-cache buffers; `attention.mla_attention_elapsed_seconds`
dropped from about 271s in the uncached threadgroup run to about 61s, while
`moe.routed_experts_streamed` dropped to about 123s. After the fused MXFP4
expert kernels, the same locked smoke completed in about 445s, kept the
generated token id `15`, and reduced `moe.routed_experts_streamed` to about
118s. The final routed-MoE summary reports about 88.3s in fused Metal kernels
out of about 91.9s of runner-internal timing, so the next routed-expert work is
deeper MXFP4 matvec tiling rather than SSD or accumulator I/O. After the MXFP4
E2M1 lookup-table decode change, the same locked smoke completed in about
389s, kept token id `15`, reduced `moe.routed_experts_streamed` to about 62s,
and reduced routed MoE kernel timing to about 32s. The next global hotspot is
the MLA attention path plus resident MXFP4 projections such as
`attention.o_proj`. Follow-up timing-instrumented runs complete in about
388s and show `mla_attention_elapsed_seconds` around 72-83s, of which the MLA
runner reports about 65-76s internally and roughly 58-69s in MLA Metal kernels.
MLA value projection/materialization accounts for about 5.4-5.6s; cache/file
I/O is small. A key-rope pre-rotation cache experiment remains available behind
`LARGERLM_MLA_ROPE_CACHE=1`, but it regressed the full GLM smoke and is not
enabled by default.

After making the value-side MLA cache the default opportunistic path, the same
locked smoke completed in about 249s with token id `15`. The summary reported
`mla_value_cache` enabled on all 78 layers, with about 624 MiB summed across
per-layer value-cache buffers, or 8 MiB extra scratch for the active layer in
this 128-token GLM-5.2 run. Python and the runner both include that buffer in
their scratch estimates; if it would exceed `--max-runner-scratch-mib`, the
runner disables the cache for that call. Set
`LARGERLM_MLA_DISABLE_VALUE_CACHE=1` to force the old path for debugging.
`result-summary` now also keeps a dedicated `mla_attention_layers` slow-layer
table and emits a bounded cache-mode sweep for the MLA target. The generated
layer-53 128-token sweep,
`glm-mla-layer53-cache-sweep-128tok.json`, used 256 MiB cache-read,
resident-matrix, and runner-scratch caps. It kept the current key+value cache
mode as both baseline and fastest total mean at about 0.054s; key-only was
about 0.248s, value-only about 0.714s, and no-cache about 0.915s. No MLA
cache-mode candidate is eligible for full replay.

The batch attention-projection prefix is also fused into one staged runner call
per layer by default. This keeps the same bounded RMSNorm/MXFP4 kernels and
scratch checks, but removes most of the per-layer projection subprocess
boundaries. Set `LARGERLM_DISABLE_BATCH_FUSED_ATTN_PROJECTIONS=1` to force the
older multi-command projection path. With value-cache and default fused
projections, the locked 128-token smoke completes in about 216s with token id
`15`, unique runner commands drop from 1014 to 624, and
`projections_elapsed_seconds` drops from about 57.5s to about 18.8s.
`mla_attention_elapsed_seconds` is about 18.7s, routed MoE about 45.7s, and
`attention_output_elapsed_seconds` about 24.1s.
The target-driven 128-token projection fusion sweep,
`glm-attn-proj-layer64-fusion-sweep-128tok.json`, compares the current fused
path with the separate seven-command path under 256 MiB resident-matrix and
runner-scratch caps. Fused stayed fastest at about 0.0825s mean wall time versus
about 0.2436s for the separate path, with zero measured output drift, so no
attention-projection candidate is eligible for full replay.

Shared-expert MXFP4 gate/up/down projections are also fused by default for
custom-metal GLM layers through `--run-shared-expert-batch`. This replaces
three resident batch-linear subprocesses plus Python-side SwiGLU with one
bounded runner call that keeps gate/up/down resident buffers in the same scratch
budget. Set `LARGERLM_DISABLE_FUSED_SHARED_EXPERT_BATCH=1` to force the older
three-projection path. With value-cache, fused attention projections, and fused
shared experts, the locked 128-token smoke completed in about 202s with token
id `15`; unique runner commands dropped from 624 to 474, and
`--run-resident-linear-batch` calls dropped from 312 to 87. The larger remaining
hotspots were streamed routed MoE and `attention.o_proj`.

Batch attention output now fuses the resident `o_proj` projection with the
residual add through `--run-attn-output-batch` for the default GLM attention
suffix. The Python prefill path enables this for `metal/largerlm-runner` by
default and keeps the same resident matrix and scratch-limit checks; set
`LARGERLM_DISABLE_BATCH_FUSED_ATTN_OUTPUT=1` to force the older batch-linear
plus Python residual-add path. With value-cache, fused attention projections,
fused shared experts, and fused attention output, the locked 128-token smoke
completed in about 190s with token id `15`. `attention_output_elapsed_seconds`
dropped from about 25.1s to about 16.5s, `attention.o_proj` moved to 78
`fused-metal` calls, and `--run-resident-linear-batch` dropped from 87 calls to
9 while `--run-attn-output-batch` accounts for the 78 attention output calls.
The larger remaining hotspot was streamed routed MoE.

Resident GLM MXFP4 batch projections with group size 32 now also default to a
group32-specialized kernel when `LARGERLM_MXFP4_BATCH_TOKEN_TILE=1`. This covers
hot resident matrices such as `.self_attn.o_proj.weight`, keeps the same
bounded scratch estimate, and can be disabled with
`LARGERLM_MXFP4_GROUP32_SPECIALIZED=0`. Compare the default and disabled paths
on the real GLM layer-19 attention output projection with:

```bash
python scripts/glm_resident_mxfp4_group32_sweep.py \
  --repeat 10 --group32-modes off,auto --order interleave \
  --layer 19 --batch-tokens 128 \
  --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-resident-mxfp4-o-proj-group32-interleave-128tok-repeat10.json
```

The recorded run confirms `auto` selects `MXFP4 group32 path: yes`, keeps the
bounded peak at about 69 MB for the single runner call, and matches the old
projection/output within about 3.3e-7 max absolute drift. Despite short-run
system noise, backend timing averaged about 0.122s for the group32 path versus
about 0.153s for the disabled path. With resident group32 specialization
enabled by default, the locked 128-token GLM smoke again generated token id
`15`, kept the same 17.37 GiB live working-set cap with about 79.9 GiB
available memory at launch, and completed in about 179s. Its result and summary
are preserved as
`smoke-prefill-128tok-custom-metal-keycache-default-resident-group32-result.json`
and
`smoke-prefill-128tok-custom-metal-keycache-default-resident-group32-summary.txt`;
the summary reports `attention.o_proj` at about 13.8s, down from the previous
resident-default group's about 15.6s.
On the current 17-token minimal GLM-5.2 replay, the same resident `o_proj`
sweep is recorded at
`glm-resident-mxfp4-o-proj-group32-17tok-repeat6-latest.json`: default `auto`
again selects the group32 path, averages about 0.0165s backend time versus
about 0.0193s with the path disabled, and keeps max output drift around
3.3e-7.
The sweep JSON now also carries `config_comparison`, matching the routed-MoE
microbench promotion policy. A fresh target-driven layer-62 128-token o_proj
sweep, `glm-resident-o-proj-layer62-optimization-target-128tok.json`, used
256 MiB resident-matrix and runner-scratch caps. It kept `auto` as both the
baseline and fastest backend mean at about 0.0220s; group32-off averaged about
0.0283s, so no resident o_proj candidate is eligible for full replay.

The same group32 idea is available experimentally for fused resident shared
experts, but it is not enabled by default. Compare it with:

```bash
python scripts/glm_shared_mxfp4_group32_sweep.py \
  --repeat 6 --group32-modes off,auto --order interleave \
  --layer 19 --batch-tokens 128 \
  --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-shared-mxfp4-group32-experimental-interleave-128tok-repeat6.json
```

The layer-19 microbench confirms the experimental path selects
`MXFP4 group32 path: yes`, keeps the single-call peak around 31 MB, and matches
the old shared-expert output within about 4.4e-10 max absolute drift. Its
backend timing averaged about 0.010s versus about 0.013s for the disabled path.
However, a full 128-token GLM smoke with this path enabled changed later router
choices, increasing expert reads from about 22.75 GiB to about 29.29 GiB and
raising total time to about 181s. The default therefore keeps shared group32
off; the follow-up default smoke completed in about 177s, kept token id `15`,
and preserved the 17.37 GiB live working-set cap. That default-off result is
preserved as
`smoke-prefill-128tok-custom-metal-keycache-default-resident-group32-shared-gs32-off-result.json`
with a matching summary text file.
The current 17-token shared-expert sweep is preserved at
`glm-shared-mxfp4-group32-17tok-repeat6-latest.json`. Explicit `auto`/`on`
selects the group32 path and remains numerically close to the disabled path
with about 3.1e-10 max drift; backend timing averages about 0.0074s for
explicit `auto` versus about 0.0084s disabled. Because the earlier full smoke
showed router-choice sensitivity, this remains an experiment rather than a
default.

Prompt RoPE now defaults to a fused split+rotate runner command for batch
prefill. `--run-rope-split-batch` reads the full `q_b` projection and K-RoPE
rows once, writes the same `q_nope.f32`, `q_rope.f32`,
`q_rope_rotated.f32`, and `k_rope_rotated.f32` artifacts as the older Python
split plus `--run-rope-batch` path, and keeps the operation under the runner
scratch cap. Compare the two paths with:

```bash
python scripts/glm_rope_split_sweep.py \
  --repeat 4 --order interleave --batch-tokens 128 \
  --num-heads 64 --qk-nope-dim 192 --rope-dim 64 \
  --start-position 0 --max-runner-scratch-mib 256 \
  --rope-theta 8000000 --rope-interleave \
  --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-rope-split-layer37-fusion-sweep-128tok.json
```

The target-driven GLM-5.2 microbench matched all four outputs exactly against
the old path and measured about 0.04195s per call for fused split+RoPE versus
about 0.04749s for the old path, with estimated single-call peak rising only
from about 16.0 MiB to about 18.1 MiB. The `config_comparison` verdict keeps
the current fused path as both baseline and fastest, so no RoPE split candidate
is eligible for full replay. The guarded 128-token GLM smoke with this default
kept token id `15`, kept expert reads at about 22.75 GiB, preserved the
17.37 GiB live working-set cap, and completed in about 177s. Its summary
reports `rope_elapsed_seconds` at about 8.17s, down from the previous
default-off summary's about 8.91s. The result is preserved as
`smoke-prefill-128tok-custom-metal-keycache-default-fused-rope-split-result.json`
with a matching summary text file.

Prefill MLA KV cache writes now encode BF16 rows through a bounded chunked
bitcast path instead of one Python float row at a time. The default chunk cap is
64 MiB and can be overridden for debugging with
`LARGERLM_PREFILL_CACHE_WRITE_CHUNK_BYTES`, while the live memory estimate uses
the actual chunk peak rather than a single-row shortcut. The target-driven
cache write sweep,
`glm-cache-write-layer0-chunk-sweep-128tok.json`, uses a synthetic cache layout
with the measured GLM-5.2 width/dtype and does not touch the real
`decode_cache.bin`. It wrote 147456 cache bytes per run, kept all tested chunk
modes byte-identical, and retained the default single-chunk path as fastest at
about 0.00682s mean wall time. One-row writes averaged about 0.00751s, 16 KiB
about 0.00707s, 256 KiB about 0.00682s, and 1 MiB about 0.00725s, so no cache
write candidate is eligible for full replay.

Routed MXFP4 MoE also has experimental token tiles of 2 and 4 inside the fused
gate/up/SwiGLU and down/weighted-add kernels. These reuse each packed expert
weight row across multiple prompt tokens without increasing scratch. The safe
default remains `LARGERLM_MOE_MXFP4_BATCH_TOKEN_TILE=1`; set it to `2` or `4`
to benchmark the multi-token paths. With all of the fusions above plus routed
MoE token tile 2, the best observed locked 128-token smoke completed in about
183s with token id `15`; `routed_moe_elapsed_seconds` was about 44.5s, and the
run kept the same 17.37 GiB live working-set cap. A no-env verification run
generated token id `15` under the same memory guard and completed in about
192s. An interleaved random-input layer-19 sweep is recorded at
`glm-moe-layer19-tile-sweep-random-128tok.json`; it did not prove a stable
default win for tile 2, so the tile remains experimental rather than default.
The follow-up tile-4 sweep recorded at
`glm-moe-layer19-tile-sweep-random-128tok-tile4.json` verified numerical
agreement with tile 1, but did not show a stable speed win.
For GLM-style routed MXFP4 experts with group size 32 and token tile 1, the
runner now selects a group32-specialized fused SwiGLU/down-add path by default.
It unrolls each 32-value MXFP4 scale group into four packed words, keeps the
same scratch and I/O plan, and can be disabled with
`LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED=0`. Compare the default and disabled
paths with:

```bash
LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING=1 \
  python scripts/glm_moe_tile_sweep.py \
    --repeat 6 --tiles 1 --group32-modes off,on --order interleave \
    --layer 19 --batch-tokens 128 \
    --experts 11,79,92,103,154,212,236,254 \
    --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-moe-layer19-group32-interleave-128tok.json
```

The recorded layer-19 run keeps max absolute output drift around 1.3e-10 and
reduces mean fused kernel time from about 0.335s to 0.274s, with both the
gate/up/SwiGLU side and down/weighted-add side improving despite run-to-run
system noise. With group32 specialization enabled by default, the locked
128-token GLM smoke generated token id `15`, kept the 17.37 GiB live working
set cap with about 79.8 GiB available memory at launch, and completed in about
184s. Its result is preserved at
`smoke-prefill-128tok-custom-metal-keycache-default-group32-result.json`; the
summary reports `moe.routed_experts_streamed` at about 41.3s, down from the
previous default group's about 45.7s.
For the current 17-token minimal replay, the latest combined routed-MoE kernel
sweep is recorded at
`glm-moe-layer19-mxfp4-kernel-sweep-17tok-repeat4-latest.json`. It stages only
one layer's eight selected experts (about 153 MiB staged) and keeps the runner
peak around 22 MiB for each microbench call. The default tile-1 `auto` path
selects group32 and averages about 0.0243s fused-kernel time; forcing group32 is
within noise at about 0.0236s, while disabling group32 is slower at about
0.0287s. Tile 2, tile 4, and the experimental vector SwiGLU path do not win on
this short prompt, so the default remains tile 1 plus automatic group32. The
sweep script now skips unsupported combinations, such as forcing group32 on
tile 2/4, and records them in `skipped_configs` instead of aborting the run.
Sweep result JSON is versioned as `largerlm.glm_moe_tile_sweep.v2` and carries
a `config_comparison` block. That block compares each candidate against the
baseline tile-1 scalar/auto path, marks only numerically close kernel-speedup
candidates with at least three timing samples as `candidate_for_full_replay`,
and still requires a full replay/result-bakeoff win before any default changes.
For deeper kernel diagnosis, set
`LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING=1` on the same sweep to split the fused
MXFP4 MoE timing into gate/up/SwiGLU and down/weighted-add command buffers:

```bash
LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING=1 \
  python scripts/glm_moe_tile_sweep.py \
    --repeat 3 --tiles 1 --order interleave --layer 19 --batch-tokens 128 \
    --experts 11,79,92,103,154,212,236,254 \
    --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-moe-layer19-split-kernel-random-128tok.json
```

The recorded local run reports about 0.052s fused-kernel time for the
128-token/8-expert layer-19 microbench, with about 0.035s in the fused
gate/up/SwiGLU side and about 0.016s in down/weighted-add. That points the next
routed-MoE optimization pass at the SwiGLU-side MXFP4 dot products and SiLU
math rather than SSD reads or accumulator writes. A tile-1/2/4 split sample is
also recorded at `glm-moe-layer19-split-kernel-random-128tok-tile4.json`; it
kept the same numerical agreement and did not justify making tile 4 the
default. Full prompt-prefill artifacts captured with the same env var now show
the split directly in `largerlm result-summary`, so routed-MoE kernel A/B runs
can be compared without opening the raw nested layer JSON.
An experimental vector-dot variant for the default tile-1 SwiGLU kernel is
available behind `LARGERLM_MOE_MXFP4_VECTOR_SWIGLU=1`. Compare it against the
scalar kernel with:

```bash
LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING=1 \
  python scripts/glm_moe_tile_sweep.py \
    --repeat 6 --tiles 1 --vector-swiglu-modes off,on --order interleave \
    --layer 19 --batch-tokens 128 \
    --experts 11,79,92,103,154,212,236,254 \
    --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-moe-layer19-vector-swiglu-interleave-128tok.json
```

The recorded interleaved run verifies numerical agreement with max absolute
drift around 1.9e-10, but it does not show a stable speed win, so vector SwiGLU
remains an explicit experiment instead of the default path.
A bounded 17-token recommendation smoke on the same layer is recorded at
`glm-moe-layer19-vector-swiglu-recommendation-smoke-17tok-latest.json`. In that
scalar-off ablation, `tile1_vector_silu` cut kernel mean from about 0.0337s to
0.0210s with max absolute drift around 1.9e-10. That is not a default-path
comparison because the vector kernel cannot be combined with group32/auto. The
fair repeat-4 comparison against default auto is recorded at
`glm-moe-layer19-vector-swiglu-vs-default-smoke-17tok-repeat4-latest.json`:
`tile1_auto_silu` averaged about 0.02382s, `tile1_scalar_silu` about 0.02230s,
and `tile1_vector_silu` about 0.02226s. Both group32-off variants are only
microbench leads. The policy-compliant custom-metal/keycache full replay
bakeoff is recorded at `moe-kernel-custom-mla-keycache-bakeoff-latest.json`;
fresh baseline stayed fastest at about 96.012s, group32-off took about 97.139s,
and vector+group32-off took about 100.119s, all generating token `[11]`.
Baseline is therefore retained and no routed-MoE kernel default changes.
The SwiGLU activation cost can be isolated with
`LARGERLM_MOE_MXFP4_SWIGLU_ACTIVATION=fast-exp` or `linear`; the latter skips
SiLU and is diagnostic-only because it intentionally changes model math:

```bash
LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING=1 \
  python scripts/glm_moe_tile_sweep.py \
    --repeat 6 --tiles 1 --activation-modes silu,fast-exp,linear \
    --order interleave --layer 19 --batch-tokens 128 \
    --experts 11,79,92,103,154,212,236,254 \
    --write-result artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-moe-layer19-activation-diagnostic-128tok-repeat6.json
```

That diagnostic records no stable speed win from `fast-exp`, and the
`linear` skip-SiLU mode is also not faster. This keeps the next routed-MoE
optimization target on MXFP4 dot-product and memory scheduling rather than the
SiLU/exp activation.

An experimental mixed-backend smoke is available:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/smoke-prefill-128-mixed-safe.sh
```

That path keeps `--prefill-linear-backend auto`, lowers the MPSGraph auto
threshold to 128 batch tokens and 32 matrix dimension, and replays a locked
request profile/audit. It accelerated the 75 BF16 resident router/gate matrices
with MPSGraph while leaving 771 MXFP4/custom matrices on custom Metal, reported
zero unsupported MPSGraph matrices, generated token id `15`, and kept the same
17.37 GiB live cap. The first local run took about 589 seconds, so this profile
is a correctness and coverage experiment rather than the current fastest path.
A lighter calibration-only check is also recorded:

```bash
artifacts/glm-5.2-mxfp4/largerlm-prepared/calibrate-prefill-128-bf16-safe.sh
```

The first M5 Max 128 GiB calibration run wrote
`prefill-plan-calibration-128tok-bf16.json`,
`prefill-calibration-flags-128tok-bf16.json`, and
`launch-profile-prefill-128tok-calibrated-bf16.json`. It covered 8 BF16 GLM
resident GEMM shapes representing about 98.1% of planned candidate FLOPs. At a
1.25x required speedup, it recommended `custom-metal`: total calibrated time was
about 1.00s for custom Metal, 1.07s for MPS matrix, and 1.14s for MPSGraph.
That calibration result agrees with the full mixed smoke: current public
MPSGraph/MPS matrix paths should be treated as fallback/experiments for this
128-token GLM-5.2 MXFP4 profile, not as the default fast path.
Explicit `mpsgraph-f32` prefill is currently rejected for this MXFP4 package
when unsupported resident matrices would reach the F32 backend. Larger prompt
experiments should start from a locked request profile/audit and should not
bypass the live-memory and routed-read guards.
Preflight then checks
that every full-indexer layer declared by `indexer_types` has the expected
resident indexer tensors before packing starts. Global embedding, final norm,
and `lm_head.weight` metadata must also match the config hidden size; when
`vocab_size` is present, embedding and output-head rows must match it as well,
so a bad output head is rejected before resident packing. Preflight JSON also
includes the public GLM-5.2 shape diagnostic, and
`--require-public-glm-5-2-shape` turns that diagnostic into a not-ready result
before any output path is touched. The report also prints the
recommended generation live working-set cap from `--runtime-buffer-gib`
(default 8 GiB) and the optional live free-memory guard from
`--system-reserve-gib`. If this reserve is not specified, LargerLM uses an
adaptive default: 24 GiB on machines with at least 96 GiB unified memory, and
16 GiB otherwise. Planner output also reports the modeled resident-memory
budget, pressure, and headroom after subtracting system reserve and runtime
buffer; preflight emits a warning when resident bytes plus those guards exceed
the selected unified-memory profile. Preflight JSON also records the detected
chip, detected unified memory/GPU core counts when available, the effective
unified-memory bytes used for planning, and whether that profile came from
`--unified-memory-gib` or hardware detection. Preflight and planner budget fields use
strict integer semantics for byte/context/group controls, so booleans and
floats fail before any packed layout or cache budget is derived. For noaux
router configs, it reports how many MoE layers carry
`gate.e_score_correction_bias` and warns when the correction-bias coverage is
incomplete. When a checkpoint includes layers at or beyond
`num_hidden_layers`, the preflight JSON includes `checkpoint_ignored_bytes` and
the text report prints `checkpoint ignored`.

Prepare a checkpoint into LargerLM's packed layout. This defaults to a dry-run
and writes nothing:

```bash
python -m largerlm prepare-glm /path/to/checkpoint \
  --output-dir /path/to/checkpoint/largerlm_packed \
  --quantize-bf16-affine-int4 --group-size 64 \
  --max-context-tokens 32768 --max-cache-gib 16 \
  --disk-margin-gib 32 --unified-memory-gib 128
```

After the dry-run report is clean, add `--execute` to write:

```bash
python -m largerlm prepare-glm /path/to/checkpoint \
  --output-dir /path/to/checkpoint/largerlm_packed \
  --quantize-bf16-affine-int4 --group-size 64 \
  --max-context-tokens 32768 --max-cache-gib 16 \
  --disk-margin-gib 32 --unified-memory-gib 128 --execute
```

Use a dedicated prepared output directory; `prepare-glm` rejects an
`--output-dir` that resolves to the original checkpoint directory itself. A
subdirectory such as `/path/to/checkpoint/largerlm_packed` is supported.

On the real target machine, add `--auto-cold-read-benchmark` during execute to
measure a bounded sequential read from the largest packed expert layer and
record the result in the manifest:

```bash
python -m largerlm prepare-glm /path/to/checkpoint \
  --output-dir /path/to/checkpoint/largerlm_packed \
  --quantize-bf16-affine-int4 --group-size 64 \
  --max-context-tokens 32768 --max-cache-gib 16 \
  --disk-margin-gib 32 --unified-memory-gib 128 --execute \
  --auto-cold-read-benchmark \
  --cold-read-benchmark-mib 1024 --cold-read-benchmark-chunk-mib 8
```

The benchmark uses bounded chunked `pread` calls against the already-packed
expert file; it does not load the full model into unified memory. The chunk is
also capped by the prepare heap limit and a 512 MiB single-read safety guard, so
an accidental large chunk setting is rejected before the benchmark runs. It
conflicts with an explicit `--cold-read-gib-s`, which is still useful when you
want to pin a previously measured value instead of measuring during
preparation.

`prepare-glm` validates its own cache alignment, chunk size, max chunk, pack
heap, disk margin, cache, and memory budget controls as strict integers before
running the dry-run or execute path. In execute mode, after the dry-run budgets
are known but before any output file is written, it also probes current system
available memory and requires enough headroom for the larger of the expert or
resident packer heap estimate plus the configured system reserve. The report
and manifest record that prepare live-memory check so a failed first run can
distinguish bad model metadata from a temporarily crowded unified-memory pool.
The same pre-write admission also checks the combined prepared-output disk
budget, adding packed experts, resident weights, and the decode cache together
before applying the disk margin, so a large GLM-5.2 prepare cannot pass three
separate per-file checks and then fill the volume halfway through execution.

If you want the cache context chosen from the current memory/cache budget
instead of hand-picking it, use:

```bash
python -m largerlm prepare-glm /path/to/checkpoint \
  --output-dir /path/to/checkpoint/largerlm_packed \
  --quantize-bf16-affine-int4 --group-size 64 \
  --auto-context-from-budget --max-cache-gib 16 \
  --disk-margin-gib 32 --unified-memory-gib 128
```

Explicit `--max-context-tokens` values above the model config's
`max_position_embeddings` are rejected during preflight. Auto-context mode is
also capped by `max_position_embeddings`, so it chooses the largest context that
fits both the cache budget and the model's advertised context window.
Dry-run JSON, execute JSON, and the prepared manifest record the context-budget
decision as `prepare_auto_context_from_budget`,
`prepare_resolved_max_context_tokens`, `prepare_decode_cache_budget_bytes`, and
`prepare_decode_cache_safe_context_tokens`, so a later launch can audit that
the prepared decode cache was derived from the intended memory/cache budget.
The standalone `plan-cache` layout builder applies the same
`max_position_embeddings` cap before writing a decode-cache layout.

The prepare command writes `experts/layout.json`, per-layer expert files,
`resident/layout.json`, `resident/resident.bin`, `decode_cache_layout.json`, a
sparse `decode_cache.bin`, and a manifest. Existing outputs are refused unless
`--force` is present. Dry-run and execute reports include a public GLM-5.2
shape diagnostic; add `--require-public-glm-5-2-shape` to fail before writing
when the config is not the public GLM-5.2 checkpoint shape. Execute mode
validates the written manifest and backing
files before reporting success, including the packed layout `config_sha256`
against the current `config.json`, then runs the same GLM 4bit readiness gate
used by launch commands. If that post-prepare readiness check, expert packing,
resident packing, or decode-cache initialization fails after opening an output
file, LargerLM removes the partial file before returning the error. The cache
layout and manifest are
written through temporary files and atomic replace, so interrupted metadata
writes do not leave partial JSON. In normal non-`--force` execute mode, if a
later prepare step fails after earlier files were written, those new prepared
outputs are removed before the error is returned.
Preflight and preparation are config-aware: dense-prefix MLP weights stay in
`resident.bin`, while only config-declared MoE layers are packed into expert
slot files. Tensors whose `model.layers.N` index is at or above
`num_hidden_layers` are treated as extra/MTP tensors and skipped by resident
packing and runtime resident-byte budgeting.

Build a non-allocating decode cache layout JSON for a target context length:

```bash
python -m largerlm plan-cache /path/to/checkpoint \
  --max-context-tokens 32768 --output /tmp/decode_cache_layout.json
```

Create the sparse backing file after the layout passes budget checks:

```bash
python -m largerlm init-cache /tmp/decode_cache_layout.json /tmp/decode_cache.bin \
  --max-cache-gib 16 --disk-margin-gib 32
```

`plan-cache` and `init-cache` validate context, alignment, max-cache, and disk
margin values as strict integers before writing layout JSON or creating the
sparse cache file.

Dry-run routed expert packing:

```bash
python -m largerlm pack-experts /path/to/checkpoint
```

Actually write packed experts, after the dry-run looks sane:

```bash
python -m largerlm pack-experts /path/to/checkpoint --execute
```

Quantize raw BF16/F16/F32 GLM expert weights into LargerLM affine-int4 slots:

```bash
python -m largerlm pack-experts /path/to/checkpoint \
  --quantize-bf16-affine-int4 --group-size 64
```

Do not use `--quantize-bf16-affine-int4` for MLX 4-bit checkpoints whose expert
tensors are already `weight`/`scales`/`biases` affine-int4 components; pack them
without the raw-conversion flag.

The raw quantizer accepts both per-expert tensors such as
`mlp.experts.0.gate_proj.weight` and fused expert-major tensors such as
`mlp.experts.gate_proj.weight` with shape `[num_experts, out, in]`. It also
normalizes common MoE aliases `w1/w3/w2` to `gate_proj/up_proj/down_proj` when
the shapes match the config-derived dimensions, and rejects duplicate canonical
components instead of overwriting one source with another. Raw fused
`gate_up_proj.weight`/`gate_up.weight`/`w13.weight` tensors are accepted when
their logical shape is `[2 * moe_hidden_size, hidden_size]` per expert; the
packer slices them into separate gate/up row blocks before quantization while
keeping the packed runner layout unchanged. It reads one expert
component slice as bounded row blocks under the configured chunk/heap caps. For
raw quantization, the modeled packer heap includes the normal safetensors
metadata/copy chunk allowance plus the largest raw source row-block and
generated quantized row-block output, not a whole raw matrix or whole expert
slot, so GLM-5.2-scale raw conversion can stay inside a tight
`--max-pack-heap-mib`. Dry-run and execute reports expose
`raw_quantization_max_source_block_bytes`,
`raw_quantization_max_output_block_bytes`, `raw_quantization_extra_heap_bytes`,
and `raw_quantization_max_rows_per_block` for auditing that bound.
Pre-quantized fused gate/up tensors are also accepted when
`gate_up_proj.weight`, `.scales`, and `.biases` use the affine-int4 row layout;
the packer slices each fused tensor into the standard gate/up weight, scale, and
bias components before bounded copying, so the runner still sees the same
nine-component slot format. Pre-quantized affine-int4 layouts preserve source
dtype strings, and the prepared readiness checks plus Metal runner accept both
short and common long/lowercase forms such as `U32`/`uint32` and
`BF16`/`bfloat16`/`BFLOAT16`. Prepared
manifest loading rejects raw block evidence that exceeds the recorded pack peak
or whose extra heap does not cover generated output plus any source-block
overflow beyond the copy chunk. For internal `largerlm-affine-int4` expert
layouts, loading also requires the complete pack-heap evidence set even when
the top-level manifest omits `expert_quantization`, so an unsafe prepared
artifact is rejected before launch-profile replay. The packer also
rejects raw expert headers whose dtype/shape byte count does not match the
slice bytes or whose logical `[out, in]` shape does not match the
config-derived GLM gate/up/down dimensions. Per-expert tensor ids must also stay
inside the config-declared expert range, so a checkpoint/config mismatch cannot
silently drop extra experts during packing. Raw source values must be finite;
NaN/Inf rows fail as controlled packer errors and partial layer files are
removed before returning. Already-quantized
affine-int4 expert tensors are also checked against those dimensions, group
size, U32 packed-weight shape, and 16-bit BF16/F16 scale/bias metadata before
any large copy begins.
Packed expert and resident `layout.json` files are written through temporary
files and atomic replace, so a failed `--force` publish does not half-overwrite
or delete the previous layout metadata.

Dry-run resident non-expert weight packing:

```bash
python -m largerlm pack-resident /path/to/checkpoint
```

Resident packing also normalizes dense/shared MLP component aliases
`w1/w3/w2` to `gate_proj/up_proj/down_proj` and raw fused dense/shared gate-up
tensors such as `gate_up_proj.weight`, `gate_up.weight`, or `w13.weight` into
separate `gate_proj.weight` and `up_proj.weight` layout entries when the
config-derived shape matches. The copy still uses bounded `pread` slices from
the original safetensors shard, and the runner continues to consume the existing
independent gate/up/down resident tensor contract. Executed prepare manifests
and prepared health report the component-alias source/renamed counts and bytes
alongside the fused source tensor count, expanded tensor count, and expanded
resident bytes so real checkpoint compatibility rewrites remain auditable.
Prepared manifest loading rejects resident rewrite byte totals that exceed the
validated resident layout bytes, so corrupted or hand-edited compatibility
evidence fails before launch-profile or audit replay.

Preflight a tiny-checkpoint MLX baseline export:

```bash
python -m largerlm export-mlx-baseline /path/to/checkpoint --prompt "Hello"
```

The exporter does not load the model unless `--execute` is present, and it
refuses whole-model MLX loads above the configured safety limit. This command is
for small checkpoints and numerical harnesses; full GLM-5.2 baselines need the
streaming/layer-by-layer path. Baseline tensors and `manifest.json` are written
through temporary files and atomic replace.

Validate a baseline:

```bash
python -m largerlm validate-baseline /path/to/baseline
```

Validation checks the manifest schema, tensor path safety, byte counts, and
SHA-256 hashes before any Metal-side comparison uses the raw tensor payloads.

Check a packed layer before running the Metal decode path:

```bash
python -m largerlm check-runtime /path/to/experts/layout.json \
  /path/to/resident/layout.json --layer 3 --top-k 8 \
  --include-shared-expert --include-decoder-layer \
  --context-length 32768 --num-heads 64 --qk-nope-dim 128 \
  --rope-dim 64 --v-head-dim 128 \
  --max-cache-read-mib 256 --max-resident-matrix-mib 512 \
  --max-runner-scratch-mib 4096
```

Plan coalesced SSD read ranges for a routed expert set without touching the
layer file:

```bash
python -m largerlm plan-expert-io /path/to/experts/layout.json \
  --layer 3 --experts 4,19,20,27 --merge-gap-kib 0 --align-kib 4

python -m largerlm plan-batch-expert-io /path/to/experts/layout.json \
  --layer 3 --router-json-dir /tmp/layer3_routed_mlp/router_json \
  --merge-gap-kib 0 --align-kib 4 --json

python -m largerlm plan-batch-expert-io /path/to/experts/layout.json \
  --layer 3 --router-json-dir /tmp/layer3_routed_mlp/router_json \
  --merge-gap-kib 0 --align-kib 4 \
  --tile-max-stage-mib 2731.61 \
  --tile-max-compact-stage-mib 2731.05 --json

python -m largerlm stage-batch-experts /path/to/experts/layout.json \
  --layer 3 --router-json-dir /tmp/layer3_routed_mlp/router_json \
  --stage-file /tmp/layer3_experts.stage.bin \
  --merge-gap-kib 0 --align-kib 4 \
  --max-stage-mib 4096 --copy-chunk-mib 8 \
  --stage-disk-margin-mib 16384 --json

python -m largerlm run-staged-routed-moe-batch metal/largerlm-runner \
  --stage-manifest /tmp/layer3_experts.stage.bin.manifest.json \
  --input-f32 /tmp/layer3_normed_hidden.f32 \
  --output-dir /tmp/layer3_staged_moe \
  --output-f32 /tmp/layer3_routed_out.f32 \
  --max-compact-stage-mib 4096 --copy-chunk-mib 8 \
  --compact-stage-disk-margin-mib 16384 \
  --max-slot-mib 256 --max-runner-scratch-mib 4096 \
  --moe-token-block auto --json

python -m largerlm run-tiled-staged-routed-moe-batch metal/largerlm-runner \
  /path/to/experts/layout.json --layer 3 \
  --router-json-dir /tmp/layer3_routed_mlp/router_json \
  --input-f32 /tmp/layer3_normed_hidden.f32 \
  --output-dir /tmp/layer3_tiled_staged_moe \
  --output-f32 /tmp/layer3_tiled_routed_out.f32 \
  --merge-gap-kib 0 --align-kib 4 \
  --max-stage-mib 2731.61 --max-compact-stage-mib 2731.05 \
  --copy-chunk-mib 64 --max-slot-mib 256 \
  --max-runner-scratch-mib 4096 --moe-token-block auto --json

python -m largerlm validate-static-capacity-bin \
  /tmp/layer3_staged_moe/static_capacity.bin \
  --expect-overflow-records 0 --json
```

This reports requested slot bytes, planned aligned read bytes, waste bytes, and
read amplification. It is the dry-run experiment boundary for choosing
runner-side read-advice settings; it does not allocate a cache or load expert
weights.
The optional `--tile-max-stage-mib` and `--tile-max-compact-stage-mib` flags on
`plan-batch-expert-io` add `batch_expert_io_tiling_plan` to the JSON output.
This is planning-only for now: it shows how a routed batch's selected experts
can be split into stage/compact-stage-safe tiles before the runtime grows a
tiled manifest executor.
`run-tiled-staged-routed-moe-batch` is the experimental executor for that plan:
it stages each expert tile, gathers the active token subbatch, runs the existing
staged MoE kernel, and scatter-adds tile outputs back into the full batch output.
Measure bounded sequential `pread` throughput on an existing packed layer file
before choosing a `plan --cold-read-gib-s` value:

```bash
python -m largerlm disk-read-benchmark /path/to/experts/layer_003.bin \
  --bytes-mib 1024 --chunk-mib 8 --json
```

The benchmark reads only the requested byte window in fixed chunks, does not
create a large temporary file, and does not use `mmap`. A 512 MiB
`--max-chunk-mib` safety cap rejects oversized single reads unless you
explicitly lower or raise that cap. Use a large packed layer file for meaningful
SSD numbers; tiny files mostly measure cache and timer overhead. `plan` now
turns the configured runtime buffer and system reserve into
`suggested_launch_guard_flags`, and turns the config-derived routed expert
bytes/token plus an optional `--cold-read-gib-s` into
`suggested_decode_guard_flags` so later generation commands can reuse
`--require-prepared-memory-profile`, `--max-live-working-set-mib`,
`--min-free-unified-memory-gib`, `--decode-max-routed-read-gib-per-token`, and
`--decode-max-routed-read-seconds-per-token` with 5% SSD headroom. Add
`--write-launch-profile PATH` to save that memory+decode `source="plan"`
profile; prepared commands can replay it with `--apply-launch-profile` before
a prepared package has emitted its own identity-bound profile.
The batch variant consumes per-token router JSON from
`prefill-routed-mlp-block-batch`, aggregates token assignments by expert, and
reports the serial per-token read bytes versus the coalesced unique expert slot
read plan. This is the planning surface for the next SSD-backed routed expert
scheduler. Router JSON is validated before planning: expert ids must be real
integers, weights must be finite numbers, and repeated experts are rejected.
Packed expert layout `layer_file` entries must stay inside the layout directory,
so a malformed layout cannot make the stage command read an unrelated file.
Pass `--ssd-read-gib-s` to `plan-batch-expert-io` or `stage-batch-experts` to
include the planned SSD read seconds in the JSON/manifest. Add
`--max-read-seconds` to `stage-batch-experts` to fail before copying when the
coalesced expert-stage read plan would exceed the latency budget at that
measured SSD speed.
`prefill-prompt` and generation pass their
`--prefill-ssd-read-gib-s` / `--prefill-max-routed-read-seconds` budget into
the actual staged expert copy too, so a request that passed admission still gets
rechecked against the concrete router-output stage plan before expert bytes are
copied. The execution path treats that seconds cap as a cumulative prompt
budget: each staged expert copy receives only the remaining read-time headroom
after earlier chunks and layers. If the measured staging copy time exceeds the
remaining cap, staging fails before the routed MoE Metal path runs. Successful
stage manifests and higher-level prompt prefill summaries also record whether
measured staging copy elapsed time stayed inside the same cap, so
launch-profile/audit evidence can distinguish a safe read plan from an
unexpectedly slow SSD copy.
Stage copies use bounded `pread` chunks instead of `mmap` or whole-range reads,
so the maximum in-flight source buffer is the configured copy chunk while each
manifest still records the exact staged source offsets.
The staged MoE execution boundary keeps the same strictness when it consumes a
stage manifest: layer ids, selected experts, slot offsets/lengths, compact
route token ids, and compact expert ids must be real integers; selected experts
and token ids must be unique; route weights must be finite. If validation or a
compact artifact write fails, compact layout/layer/routes files are removed
before the error returns. `run-staged-routed-moe-batch` also mirrors the stage
manifest's `io_summary` and read-advice telemetry in its result and CLI JSON,
including serial, unique, planned, staged, waste, range-count, amplification,
stage-utilization, and read-advice counters, so low-level SSD staging experiments
can be audited without opening the manifest separately.
Runtime preflight and execution-side layout readers also reject JSON booleans in
integer fields such as expert layer ids, slot sizes, resident tensor
shapes/sizes/offsets, DSA indexer tensors, and prompt-prefill auto-sizing
metadata before launching a runner.
Direct decode-layer execution rejects boolean or float values for position,
context, layer, attention/DSA dimensions, cache dtype, and expert read-advice
integer controls before work-directory creation.
Prompt embedding batches validate every token id before opening the output file,
and remove partial output if a row read/write fails.
Final-logits paths validate RMSNorm epsilon as a finite non-negative number, and
the Metal top-k path rejects non-integer token ids in runner JSON before
returning candidates to generation.
Stage and compact-stage commands also check free disk before writing; use
`--stage-disk-margin-mib` and `--compact-stage-disk-margin-mib` to leave a
deliberate SSD safety margin during large prompt experiments. If a stage or
compact-stage copy fails after opening the output, the partial file is removed
before the error is returned so a retry does not leave large orphaned artifacts.
Expert read-advice merge/alignment bytes and stage disk margins are strict
integers after CLI unit conversion, matching layout and manifest validation.

Build and run the low-level Metal smoke tests. `--self-test-dequant` validates
the same affine slot convention emitted by the raw packer: low-to-high 4-bit
lanes in U32 words plus 16-bit BF16/F16 scale and bias metadata, where the
runner reconstructs `w = q * scale + bias`. `--self-test-mpp` is the runner-side
M5/Metal 4 bring-up gate: it compiles and executes a tiny 32x32 half MPP `matmul2d` and
checks the result without loading model weights or making MPP selectable for
generation. `expert_layout_contract_smoke.py` mutates a tiny packed expert
layout and verifies that the runner itself rejects wrong `component_order`,
component offsets, affine-int4 dtypes, unsafe layer-file paths, invalid slot
metadata, non-integer layer/component/group fields, or mismatched expert layer
file sizes before any direct MoE execution. `resident_layout_contract_smoke.py` applies the same idea to
resident weights, rejecting unsafe `weight_file`, invalid `total_bytes`, and
resident backing-file size drift before direct router or resident-matrix reads.
It also mutates tensor offsets, sizes, spans, and dtypes to verify resident
tensor metadata is strict before the runner reads any resident matrix/vector.
`decode_cache_contract_smoke.py` verifies the runner-side decode cache backing
contract through `--validate-cache-backing`, rejecting unsupported layout
versions, boolean integer fields, mismatched segment strides, duplicate
segments, segment spans past `total_bytes`, and cache files whose logical size
does not match the layout. The same smoke also calls the direct MLA,
attention-projection, and decoder-layer entry points with a bad cache file to
verify the failure happens before Metal device creation or projection work.
`moe_route_contract_smoke.py` exercises the `--run-moe-batch` route boundary,
rejecting coerced JSON route integers, non-finite route weights, duplicate
per-token experts, and duplicate token/expert assignments in `LLMSCAP1` static
capacity binaries before the runner creates a Metal device.
`runner_cli_contract_smoke.py` covers the direct runner CLI numeric boundary for
expert ids, MoE weights, batch tokens, `--max-k`, router scaling/group options,
read-advice caps, final-logits chunk/scratch caps, attention projection
positions, decode/cache/MLA/RoPE/router caps, resident-linear matrix/scratch
caps, and RMSNorm eps values so malformed values fail before layout work or
Metal device creation. MiB budget flags that reach decode/generation wrappers
accept strict finite decimal values and convert them directly to bytes instead
of truncating them through integer prefix parsing.

```bash
make -C metal
metal/largerlm-runner --self-test-dequant
metal/largerlm-runner --self-test-expert
metal/largerlm-runner --self-test-mpp
python metal/expert_layout_contract_smoke.py
python metal/resident_layout_contract_smoke.py
python metal/decode_cache_contract_smoke.py
python metal/moe_route_contract_smoke.py
python metal/runner_cli_contract_smoke.py
metal/largerlm-runner --layout /path/to/experts/layout.json --layer 3 --expert 0 --read
metal/largerlm-runner --layout /path/to/experts/layout.json --layer 3 --expert 0 \
  --run-expert --input-f32 /path/to/hidden.f32 --output-f32 /tmp/expert_out.f32
metal/largerlm-runner --layout /path/to/experts/layout.json --layer 3 \
  --run-moe --experts 4,19,27,88 --weights 0.4,0.3,0.2,0.1 \
  --input-f32 /path/to/hidden.f32 --output-f32 /tmp/moe_out.f32
metal/largerlm-runner --layout /tmp/layer3_staged_moe/compact_layout.json \
  --layer 3 --run-moe-batch --routes-json /tmp/layer3_staged_moe/compact_routes.json \
  --input-f32 /tmp/layer3_normed_hidden.f32 --batch-tokens 2240 \
  --output-f32 /tmp/layer3_routed_out.f32 --max-k 8
metal/largerlm-runner --layout /tmp/layer3_staged_moe/compact_layout.json \
  --layer 3 --run-moe-batch --routes-bin /tmp/layer3_staged_moe/static_capacity.bin \
  --input-f32 /tmp/layer3_normed_hidden.f32 --batch-tokens 2240 \
  --output-f32 /tmp/layer3_routed_out.f32 --max-k 8
metal/largerlm-runner --resident-layout /path/to/resident/layout.json --layer 3 \
  --run-router --input-f32 /path/to/hidden.f32 --top-k 8 \
  --output-router-json /tmp/router.json
metal/largerlm-runner --resident-layout /path/to/resident/layout.json --layer 3 \
  --run-router-batch --input-f32 /tmp/layer3_normed_hidden.f32 \
  --batch-tokens 2240 --top-k 8 \
  --output-router-json-dir /tmp/layer3_router_json
metal/largerlm-runner --layout /path/to/experts/layout.json \
  --resident-layout /path/to/resident/layout.json --layer 3 \
  --run-layer-moe --input-f32 /path/to/hidden.f32 --top-k 8 \
  --include-shared-expert \
  --output-f32 /tmp/layer_moe_out.f32 --output-router-json /tmp/router.json
metal/largerlm-runner --layout /path/to/experts/layout.json \
  --resident-layout /path/to/resident/layout.json --layer 3 \
  --run-mlp-block --input-f32 /path/to/post_attention_residual.f32 \
  --top-k 8 --include-shared-expert --output-f32 /tmp/mlp_block_out.f32
metal/largerlm-runner --resident-layout /path/to/resident/layout.json --layer 3 \
  --run-resident-linear --tensor-suffix .self_attn.q_a_proj.weight \
  --input-f32 /path/to/hidden.f32 --output-f32 /tmp/q_a.f32
metal/largerlm-runner --resident-layout /path/to/resident/layout.json --layer 3 \
  --run-attn-projections --input-f32 /path/to/hidden.f32 \
  --output-dir /tmp/attn_proj \
  --cache-layout /tmp/decode_cache_layout.json --cache-file /tmp/decode_cache.bin \
  --position 0 --max-resident-matrix-mib 512
metal/largerlm-runner --run-rope --q-f32 /tmp/q_rot.f32 --k-f32 /tmp/k_rot.f32 \
  --output-q-f32 /tmp/q_rot_rope.f32 --output-k-f32 /tmp/k_rot_rope.f32 \
  --num-heads 64 --rope-dim 64 --position 0 --rope-theta 10000
metal/largerlm-runner --resident-layout /path/to/resident/layout.json \
  --cache-layout /tmp/decode_cache_layout.json --cache-file /tmp/decode_cache.bin \
  --layer 3 --run-mla-attention --q-nope-f32 /tmp/q_nope.f32 \
  --q-rope-f32 /tmp/q_rope.f32 --context-length 1024 --num-heads 64 \
  --qk-nope-dim 128 --rope-dim 64 --v-head-dim 128 \
  --output-f32 /tmp/attn_value.f32
metal/largerlm-runner --resident-layout /path/to/resident/layout.json --layer 3 \
  --run-attn-output --input-f32 /tmp/attn_value.f32 \
  --residual-f32 /path/to/residual_hidden.f32 --output-f32 /tmp/attn_out.f32
metal/largerlm-runner --resident-layout /path/to/resident/layout.json --layer 3 \
  --run-attn-output-batch --input-f32 /tmp/attn_value_batch.f32 \
  --residual-f32 /path/to/residual_hidden_batch.f32 --batch-tokens 128 \
  --output-f32 /tmp/attn_out_batch.f32 --projection-f32 /tmp/o_proj_batch.f32
metal/largerlm-runner --layout /path/to/experts/layout.json \
  --resident-layout /path/to/resident/layout.json \
  --cache-layout /tmp/decode_cache_layout.json --cache-file /tmp/decode_cache.bin \
  --layer 3 --run-decoder-layer --input-f32 /path/to/hidden.f32 \
  --position 1023 --context-length 1024 --num-heads 64 \
  --qk-nope-dim 128 --rope-dim 64 --v-head-dim 128 \
  --top-k 8 --include-shared-expert --output-f32 /tmp/layer_out.f32
metal/largerlm-runner --validate-baseline /path/to/baseline
```

Run multiple packed decoder layers sequentially through the Python driver:

```bash
python -m largerlm embed-token /path/to/resident/layout.json \
  --token-id 123 --output-f32 /tmp/hidden_in.f32 --max-row-mib 64

python -m largerlm embed-tokens-batch /path/to/resident/layout.json \
  --token-ids-file /tmp/prompt_token_ids.json \
  --output-f32 /tmp/prompt_hidden.f32 \
  --max-row-mib 64 --max-output-mib 4096

python -m largerlm prefill-prompt /path/to/experts/layout.json \
  /path/to/resident/layout.json /tmp/decode_cache_layout.json /tmp/decode_cache.bin \
  --model-config /path/to/checkpoint/config.json \
  --runner metal/largerlm-runner --prompt-token-ids 123,456,789 \
  --output-last-hidden-f32 /tmp/prompt_last_hidden.f32 \
  --prompt-chunk-tokens auto --max-prompt-batch-mib 1024 \
  --max-stage-mib 4096 --max-compact-stage-mib 4096 \
  --moe-token-block auto --static-capacity-per-expert auto \
  --prefill-linear-backend auto --quiet-runner

python -m largerlm decode-layers /path/to/experts/layout.json \
  /path/to/resident/layout.json /tmp/decode_cache_layout.json /tmp/decode_cache.bin \
  --model-config /path/to/checkpoint/config.json \
  --runner metal/largerlm-runner --layers 0-3 \
  --input-f32 /tmp/hidden_in.f32 --output-f32 /tmp/hidden_out.f32 \
  --position 1023 --context-length 1024 --max-cache-read-mib 256 \
  --expert-read-advise-align-kib 4
```

With `--model-config`, the driver derives the GLM attention dimensions, router
settings, RoPE theta/interleave mode, shared-expert use, dense/MoE split, and
DSA full/shared `indexer_types`, including the DSA-specific
`indexer_rope_interleave` flag. Omitting `--layers` runs the union of resident
dense layers and packed expert layers. Full-indexer decode layers switch to the
composed attention path to write `dsa_index` cache rows and compact top-k u32
files; shared-indexer layers reuse the latest full layer's top-k file for
indexed MLA.
The decode preflight is DSA-aware: full-indexer layers budget the compact
`dsa_index` scan plus only the selected MLA cache rows, while shared-indexer
layers budget only the selected MLA rows. The driver validates schedule values,
requires full-indexer layers to have `dsa_index` cache segments, and can derive
`--dsa-index-head-dim` from the cache layout. Decode-layer memory, cache,
scratch, and expert-read-advise caps are validated as finite positive or
non-negative values before the private work directory is created.
Automatically-created decode-layer work directories are removed on runner
failure unless `--keep-work-dir` or an explicit `--work-dir` asks to preserve
them for debugging.
See [`docs/research-notes.md`](docs/research-notes.md) for the external papers
and projects guiding the flash-backed MoE design.

`prefill-prompt` is the current chunked prompt path. It streams token embedding
rows into bounded `[chunk, hidden]` f32 files, runs every selected layer through
batch attention plus dense or staged routed MLP blocks, writes the MLA cache for
each prompt position, and emits the final prompt token hidden row for logits.
The prompt batch, resident matrix, cache read/write, runner scratch, staged
expert, compact-stage, and binary static-route file sizes all have explicit
MiB caps or request-check estimates. MoE prompt
layers forward `--moe-token-block auto` into the staged runner, and
`--static-capacity-per-expert auto` writes an `LLMSCAP1` binary route table with
capacity equal to that chunk's token count without constructing the debug JSON
slot table. That binary is validated against the static-capacity plan before the
runner receives `--routes-bin`, so corrupt headers, mismatched lengths,
out-of-range token ids, or non-finite route weights fail in Python first. The
prompt result aggregates the
runner's effective block size, MoE batch-buffer and runner-peak telemetry, and
static-capacity slot/binary-route counts.
Pass `--expert-stage-tiling` to `prefill-prompt` to use the same bounded
expert-tile path inside every routed prompt layer. This keeps the top-level
prompt safety/reporting fields aggregated across all tiles while preserving
per-tile details under each layer's `staged_mlp.tiled_staged_moe` JSON field.
It is useful when `inspect-prepared` or `prefill-prompt --prompt-chunk-tokens
auto --json` shows that the prompt chunk is blocked only by routed expert
stage/compact-stage size.
`prefill-prompt` also applies the live working-set guard before creating its
work directory. The estimate is the maximum of the prompt-batch, cache-read,
cache-write, and expert-stage-copy phases, each paired with the runner scratch
cap, and it must fit under `--max-live-working-set-mib`.
All prompt-prefill memory, cache, scratch, stage, and disk-margin caps are
validated as finite positive or non-negative values at entry, before layout
reads create any prompt work files. Direct prompt-prefill integer controls,
including prompt token ids, start positions, chunk tokens, attention/DSA
dimensions, layer sets, static capacity, and stage disk margins, reject booleans
and floats before work-directory creation. `--min-free-unified-memory-gib` can
require additional system headroom. Long
prompt prefill rechecks the same live-memory guard before each chunk, so a run
can stop before starting the next chunk if another app consumes the reserved
unified-memory headroom. Generation forwards the same live-memory caps into
batch prompt prefill, and rechecks the guard after prefill or decode layers
before running final logits so a late memory-pressure change can stop the next
large allocation.
Prepared generation, benchmark, inspect, and server commands accept
`--prefill-expert-stage-tiling` for the same prompt-prefill path. Request checks
then validate the maximum per-tile stage/compact footprint instead of rejecting
solely because the full routed layer would not fit in one stage file, and the
combined suggested launch argv preserves that valueless flag. If a larger safe
chunk depends on non-default cache caps, the request profile also preserves
`--max-cache-read-mib` and `--prefill-max-cache-write-mib`, so locked profile
replay recomputes the same max-safe prompt chunk instead of falling back to the
default cache-read envelope.
The lower-level routed prefill wrappers share the same boundary for batch
tokens, top-k/max-k, router group counts, static capacity, SSD read-advice
controls, and stage disk margins before launching the runner.
For direct prompt prefill, `--prompt-chunk-tokens auto` picks a tile-aligned
chunk size bounded by prompt f32 output, MLA/DSA cache read/write caps,
resident matrix scratch headroom, worst-case routed expert stage/compact-stage
caps, available work-directory disk for one prompt chunk including routed
stage plus compact-stage temporary files, and the prompt length. When
`--start-position` is nonzero, the cache read cap is sized against the existing
cache prefix plus the new prompt, not just the prompt slice. Direct prefill also
rejects `start_position + prompt_length` beyond the decode-cache context before
creating its work directory or writing cache rows. Generation commands use the
same sizing logic through
`--prefill-prompt-chunk-tokens auto`, which is the default there.
Auto sizing refuses malformed resident layouts instead of falling back to a
fixed chunk size, because those tensors define the hidden dimension and resident
matrix scratch budget. It also refuses malformed decode-cache layouts rather
than skipping cache read/write caps.
When direct `prefill-prompt --json` resolves `auto`, it reports
`auto_prompt_chunk_plan` with every cap source, the limiting cap names, the
scratch used by the resolved chunk, and the next-token scratch estimate so
MPSGraph conversion cliffs are visible while keeping the chosen chunk bounded.
Prompt work files, stage files, and compact-stage writes all run a free-disk
check before copying; use `--stage-disk-margin-mib` on `prefill-prompt` and
`--prefill-stage-disk-margin-mib` on generation commands to reserve extra room
for the private work directory. Unless `--keep-work-dir` is set, each prompt
chunk's intermediate f32 work directory is deleted as soon as that chunk is no
longer needed. Automatically-created temporary work directories are also
removed on execution failure; pass an explicit `--work-dir` or `--keep-work-dir`
when you deliberately want to preserve failure artifacts for debugging.
When the model config declares GLM DSA `indexer_types`, `prefill-prompt` also
keeps the full/shared DSA top-k state inside each prompt chunk so indexed MLA
can stage only selected cache rows. Override with `--dsa-index-topk`,
`--dsa-index-n-heads`, `--dsa-index-head-dim`, `--dsa-qk-rope-dim`, or disable
it with `--disable-dsa-indexer`.

`--expert-read-advise-align-kib` asks the Metal runner to issue macOS
`F_RDADVISE` hints for selected expert slot ranges before the individual
`pread`s. `--expert-read-advise-merge-gap-kib` can merge nearby selected slots
into a larger hinted range. Both default to `0`; no application-owned expert
cache is created.

Stream final RMSNorm plus `lm_head` top-k without loading the whole head matrix:

```bash
python -m largerlm final-logits /path/to/resident/layout.json \
  --input-f32 /tmp/hidden_out.f32 --output-topk-json /tmp/topk.json \
  --top-k 8 --max-chunk-mib 64
```

Top-k JSON output and optional full-vocab f32 logits output are also published
with temporary files and atomic replace.

Use the Metal chunked path for the same top-k boundary:

```bash
python -m largerlm final-logits /path/to/resident/layout.json \
  --runner metal/largerlm-runner \
  --input-f32 /tmp/hidden_out.f32 --output-topk-json /tmp/topk.json \
  --top-k 8 --max-chunk-mib 64 --max-runner-scratch-mib 4096
```

The single-process rewrite runtime also has a bounded final-logits probe for
prepared GLM MXFP4 packages:

```bash
metal/glm_moe_infer --prepared /path/to/largerlm-prepared \
  --probe-final-logits --input-f32 /tmp/hidden_out.f32 \
  --output-topk-json /tmp/topk.json --top-k 8 --max-chunk-mib 64
```

Run a greedy token-id loop over packed weights:

```bash
python -m largerlm generate-token-ids /path/to/experts/layout.json \
  /path/to/resident/layout.json /tmp/decode_cache_layout.json /tmp/decode_cache.bin \
  --model-config /path/to/checkpoint/config.json \
  --runner metal/largerlm-runner --prompt-token-ids 123,456 \
  --max-new-tokens 16 --batch-prefill-prompt \
  --prefill-prompt-chunk-tokens 64 --prefill-moe-token-block auto \
  --prefill-static-capacity-per-expert auto \
  --max-cache-read-mib 256 --quiet-runner
```

Run the same path from text with a local tokenizer:

```bash
python -m largerlm generate-text /path/to/experts/layout.json \
  /path/to/resident/layout.json /tmp/decode_cache_layout.json /tmp/decode_cache.bin \
  --tokenizer /path/to/checkpoint --model-config /path/to/checkpoint/config.json \
  --runner metal/largerlm-runner --prompt "你好" --max-new-tokens 16 \
  --logits-top-k 8 --temperature 0 --max-cache-read-mib 256 --quiet-runner
```

The text command never downloads tokenizer files. It uses local
`tokenizer.json`, `tokenizer.model`, `spiece.model`, or an explicit simple test
tokenizer, with optional backend packages loaded only when present. Text
generation CLI prompt inputs are capped by `--max-prompt-bytes` before
tokenization, so an accidental huge `--prompt-file` is rejected before layout
validation or runner setup.

Token generation runs a runtime preflight by default before creating the
per-token work directory or launching the runner. The guard checks the decode
cache layout/file size, resident/expert backing file sizes, worst requested
context length, selected layer scratch and cache-read budgets, and final-logits
chunk peak memory. It also rejects requests whose estimated live
runner/prompt/logits working set exceeds `--max-live-working-set-mib` (default
8192). Runtime caps must be finite, and cache/read/scratch/logits chunk limits
must be positive, so malformed negative or NaN limits fail before any work files
or runner processes are created. Runtime/live-memory guard integer fields also
reject booleans instead of treating them as 0/1. Direct token generation also
rejects
non-integer prompt/eos token IDs, integer control values, invalid cache dtype
widths, and non-finite sampling/router numeric controls before layout reads, so
offline benchmarks do not silently coerce floats or booleans into a large run.
Resident/expert layout fields used by runtime preflight, prompt-prefill auto
sizing, embeddings, final logits, DSA indexer, and expert staging follow the
same strict integer rule.
Keep this enabled for real checkpoints;
`--no-runtime-preflight` is
mainly for debugging an already-known tiny fixture.
For real GLM-5.2 runs on a busy desktop, add a live headroom guard such as
`--min-free-unified-memory-gib 24`; generation will inspect available system
memory before creating the work directory and refuse to start if the live
working-set estimate plus that reserve does not fit. A positive
`--min-free-unified-memory-gib` still forces that runtime guard even when
`--no-runtime-preflight` is supplied, because the free-memory reserve is a
safety gate rather than an advisory hint. With that reserve enabled, the decode
loop also rechecks available memory before prompt/decode stages so a long
request can stop if another app starts consuming unified memory mid-generation.
When batch prompt prefill is enabled, an explicit
`--prefill-prompt-chunk-tokens N` is also checked against the same prompt,
cache, scratch, and disk caps used by `auto` before the work directory is
created. Runtime preflight also folds the prefill live estimate into the
top-level generation live set, so large cache-write caps are admitted before
any prompt staging begins. Prepared request inspection runs the same preflight
for batch-prefill-only checks too, even when `max_new_tokens=0`, because long
prompt prefill can still consume the live scratch/cache envelope. Prepared
request inspection reports the required
available memory, detected available/total memory, source, and `available_ok`
flag alongside the live estimate.
If the live-memory guard rejects a request, server paths return a structured
`request_check.runtime_preflight` diagnostic in the HTTP 400 body with fields
such as `code`, `required_available_memory_bytes`,
`system_available_memory_bytes`, `system_memory_source`, and
`available_memory_ok=false`. This does not relax the guard; it makes low-memory
refusals machine-readable so clients can lower prompt/new-token limits or retry
later instead of treating the failure as a crash.
Automatically-created generation work directories are removed on failure by
default, so a failed large prompt does not leave staged expert or f32 activation
files in `/private/tmp`; use an explicit `--work-dir` or `--keep-work-dir` when
you want to inspect the failed run.

If the checkpoint was prepared with `prepare-glm`, you can run from the
prepared output directory or its `manifest.json` without repeating all layout
paths:

```bash
python -m largerlm inspect-prepared /path/to/checkpoint/largerlm_packed \
  --prefill-linear-backend auto \
  --check-prompt-tokens 4096 --check-max-new-tokens 16 \
  --check-runtime-preflight \
  --write-launch-profile /path/to/checkpoint/largerlm_launch_profile.json \
  --json

python -m largerlm generate-prepared-text /path/to/checkpoint/largerlm_packed \
  --apply-launch-profile /path/to/checkpoint/largerlm_launch_profile.json \
  --prompt "你好" --max-new-tokens 16 --logits-top-k 8 \
  --metal-final-logits --temperature 0 --max-cache-read-mib 256 --quiet-runner \
  --write-result /path/to/checkpoint/largerlm_text_result.json
```

For a real larger-than-memory GLM-5.2 launch, run a strict audit immediately
before starting serving or generation:

```bash
python -m largerlm inspect-prepared /path/to/checkpoint/largerlm_packed \
  --apply-launch-profile /path/to/checkpoint/largerlm_launch_profile.json \
  --lock-launch-profile \
  --require-launch-audit \
  --require-prepared-memory-profile \
  --require-glm-4bit --require-public-glm-5-2-shape \
  --require-prefill-acceleration \
  --prefill-ssd-read-gib-s 16 --prefill-max-routed-read-seconds 5 \
  --decode-max-routed-read-gib-per-token 1 \
  --decode-max-routed-read-seconds-per-token 5 \
  --check-prompt-tokens 4096 --check-max-new-tokens 16 \
  --check-runtime-preflight \
  --write-launch-audit /path/to/checkpoint/largerlm_launch_audit.json \
  --json
```

`--require-launch-audit` on `inspect-prepared` returns non-zero unless the JSON
`launch_audit.ok` checklist proves the applied launch profile is locked, the
prepared identity is hash-bound, the expert/resident layout backing files and
decode-cache backing file were loader-validated, the prepared memory profile,
decode-cache context budget, and current memory guard pass, GLM 4-bit and public
GLM-5.2 shape gates pass, prefill acceleration is required and available, the
current selectable MPSGraph prefill path has a passing runtime probe, the locked
launch profile itself replays that probe, and the checked request passes
admission, prefill routed expert SSD read-speed/seconds-budget checks, decode
routed expert bytes/token and seconds/token checks, routed stage temp
cap/free-space checks, and runtime memory preflight.
The prefill acceleration audit records the selectable MPSGraph backend, visible
MPP runtimes, validated accelerated backend set, runtime gaps, host probe
identity/timeout, probe results, and `prefill_neural_accelerator_status` so an
M5 launch artifact explains whether the planned MPP/Neural Accelerator path is
unavailable, probed-but-not-selectable, or ready for future selection.
Required-audit consumers reject acceleration evidence that has selectable
backends but no validated accelerated backend.
They also reject required prefill acceleration evidence without a requested,
run, passing host probe, a non-empty probe path, and a positive probe timeout.
When an artifact claims a passing MPP run probe, required-audit consumers also
require the recorded kernel variant, `32x32x32` half tile shape, and
`mpp::tensor_ops::matmul2d` primitive to be present and consistent.
The prepared storage audit records whether the expert and resident layout
backings were present and whether the decode-cache file exactly matched the
declared layout size, so stale or partial cache files cannot satisfy the strict
startup gate. It also validates resident `w1/w3/w2` and fused gate/up rewrite
counts and bytes when those prepare-time compatibility rewrites are present.
The GLM 4-bit audit check records the accepted expert
quantization/group size, packed expert layer file counts and exact-size status,
config-derived expert bytes, per-token routed expert read bytes, full-prompt
expert sweep bytes, and decode-cache segment counts, so the saved artifact
explains the SSD streaming envelope it approved. The public GLM-5.2 audit check
records the raw DSA
schedule checks and derived full-indexer layer count/list, so the saved artifact
shows which GLM-5.2 indexer schedule was accepted. The optional
`--write-launch-audit PATH` artifact can then be required by
`serve-prepared`, `generate-prepared-token-ids`, `generate-prepared-text`, and
`bench-prepared-token-ids` to bind startup to that passing audit and the same
locked launch profile. Consumers reject audit artifacts that omit any current
strict checklist item, so older `ok: true` artifacts cannot bypass newer safety
gates. Consumers also require the GLM 4-bit SSD-envelope fields, the checked
request's prefill/decode routed expert SSD read-budget evidence, the routed
stage temp cap/free-space evidence, the runtime preflight live-memory evidence,
the applied benchmark profile's cumulative prefill/decode read-time evidence
when present, and the prefill acceleration/M5 probe evidence fields, re-run the
low-memory readiness pass, and reject any mismatch, missing request
read/temp/memory-budget evidence, missing benchmark read-time evidence, or
contradictory acceleration probe state before invoking the runner.
`serve-prepared` also refuses to start when
`--max-prompt-tokens` or `--max-new-tokens-cap` exceeds the audited request
envelope, and direct generation/benchmark commands reject prompts or
`max_new_tokens` values larger than that envelope. The artifact must carry a
safe `request_launch_profile`; `serve-prepared` plus direct generation and
benchmark commands verify that the current effective guard flags replay that
request profile, and the nested profile must carry the same strong prepared
identity as the audit artifact. When the artifact records prefill/decode
routed-read budget evidence, that request profile must also carry the matching
`--prefill-*` and `--decode-max-routed-read-*` guard flags, and those caps must
not be wider than the evidence-derived 5% headroom profile; routed stage temp
byte caps, static-capacity mode, and raw/coalesced stage range-count caps are
bound the same way. The audit artifact itself can be passed to
`--apply-launch-profile` because the profile loader extracts
`request_launch_profile` from it.

The prepared manifest records the expert layout, resident layout, decode cache
layout, cache backing file, original model directory, and the preflight
recommended live-memory guard. It also records the packed routed-expert
quantization (`expert_quantization` and `expert_group_size`) plus the
validated model-config SHA-256 (`model_config_sha256`) copied from the packed
expert/resident layouts, and the
prepare-time hardware and memory profile: detected chip name, unified-memory
bytes, GPU cores, effective unified-memory budget/source, and system reserve.
Executed prepares also record the live-memory admission used immediately before
writing files: estimated prepare live working set, required free memory,
observed system available/total memory, and the probe source. Prepared server
health exposes the same block as `prepare_live_memory`, including whether the
current available memory still meets that prepare-time requirement.
The manifest also records the combined prepared-output disk budget used by the
same pre-write admission.
It also records the prepare-time decode-cache context budget: whether
auto-context was used, the requested and resolved context lengths, the modeled
decode-cache budget, the safe context under that budget, the effective
`max_cache_bytes` guard, and cache dtype/alignment.
If preparation was given a measured `--cold-read-gib-s`, or was executed with
`--auto-cold-read-benchmark`, the manifest also records
`prepare_cold_read_gib_per_second` and its source.
That makes a prepared package auditable later, so you can tell whether a GLM
package was created with the intended 4-bit expert format and M5 Max memory
envelope before launching a larger-than-memory run. It is the safer default
entry point after preparation because it checks path existence, decode cache
layout integrity, exact cache backing file size, resident/expert backing files
large enough for their layouts, non-overlapping resident/expert byte spans, and
recorded byte counts before generation starts. Oversized cache files are
rejected as stale prepared artifacts instead of being silently accepted. When
the packed layouts record `config_sha256`, prepared loading also rejects a
changed `config.json` before generation starts, and a manifest-level
`model_config_sha256` must match that validated layout/config digest.
Prepared loading also rejects context-budget drift when the manifest's
`prepare_resolved_max_context_tokens`, cache dtype/alignment, or recorded cache
budget no longer agrees with the decode-cache layout.
Loaded manifests retain the validated expert, resident, decode-cache layout,
and decode-cache file byte counts, so `/health` and `inspect-prepared` can show
the prepared package size, expert quantization, prepare-time memory profile,
decode-cache context budget, and recommended available-memory budget without
scanning or mapping model weights.
They also expose `suggested_launch_guard_flags`, an argv-style replay of the
prepared `--max-live-working-set-mib` and `--min-free-unified-memory-gib`
envelope when those recommendations were recorded in the manifest.
When GLM 4bit readiness passes, prepared health also emits
`suggested_decode_guard_flags` from the config-derived per-token routed expert
read, so a launch script can pin the model-level decode SSD envelope before a
specific request is checked.
`suggested_launch_profile` combines the manifest launch guards, the prepared
SSD read-speed flag when available, prefill backend probe flags such as
`--compile-mpp-probe`, tuned MPSGraph auto thresholds and accelerated-FLOPs
coverage gates, selectable GLM 4-bit readiness guard, public GLM-5.2 shape
guard when the prepared config matches that checkpoint, prefill acceleration
flags, and model-level decode guard flags into one de-duplicated argv list while
preserving each source section for auditing.
When GLM readiness passes, the profile includes
`--require-glm-4bit`, so replayed launches keep the same config/layout/
quantization gate instead of relying on operator memory.
When the prepared config also matches the public GLM-5.2 shape, the profile
includes `--require-public-glm-5-2-shape`, so a GLM-5.2-specific launch profile
cannot be silently replayed against a different GLM MoE shape.
`inspect-prepared --write-launch-profile PATH` writes the best available
profile JSON, preferring `request_launch_profile` when a `--check-*` prompt or
chat check succeeds. Prepared generation, benchmark, inspection, and serving
commands accept `--apply-launch-profile PATH`; the profile argv is expanded
before later command-line flags, so an explicit flag on the launch command still
overrides the saved profile. Add `--lock-launch-profile` for real
larger-than-memory launches when every argv flag from the profile must remain
unchanged after explicit command-line arguments are parsed. Add
`--require-locked-launch-profile` to make prepared commands fail unless a
profile is applied and locked, which is useful for launch scripts that must not
fall back to ad-hoc memory limits. Saved profiles
include prepared-package identity metadata, and applying a profile to a package
with different validated layout bytes, context, quantization, or model-config
hash is rejected before generation or serving starts. Profiles also bind
prepare-time expert-pack heap envelope and resident alias rewrite evidence when
those fields are present, so a raw-conversion or checkpoint-compatibility
rewrite drift cannot silently reuse an older GLM launch profile. The profile's
`prepared.identity_strength` is `strong` when
a model-config hash plus routed-expert format metadata are available; old or
hand-written prepared packages without those fields are marked `weak` with
`identity_warnings`. Weak identities can still
be inspected or replayed through the normal profile path for compatibility, but
they cannot satisfy `--require-launch-audit`; regenerate the prepared package
with `prepare-glm` before using it as a strict larger-than-memory launch gate.
When a profile is applied, prepared health, generation, and benchmark JSON
include `applied_launch_profile` with the profile path, file SHA-256, source,
safe argv, `locked`, `lock_required`, `profile_flag_count`,
`lock_checked_flags`, prepared identity, section names, and
`matches_prepared: true`.
The human-readable CLI output prints the applied profile path, SHA-256, source,
lock status, flag count, and match result so real GLM-5.2 runs can be audited
even without saving JSON.
Prepared generation, benchmark, serving, and inspection inherit
`prepare_cold_read_gib_per_second` as the default `--prefill-ssd-read-gib-s`
for routed expert read-time estimates; pass
`--no-prepared-ssd-read-default` to keep that estimate disabled unless a command
sets `--prefill-ssd-read-gib-s` explicitly.
They also report `prepared_runtime_profile`, which compares the current system
total/available memory against the prepare-time effective unified-memory budget
and system reserve before any large model file is mapped. It also reports
`prepared_recommended_required_available_memory_bytes` and whether current
available memory meets that `recommended_max_live_working_set_bytes +
recommended_min_free_unified_memory_bytes` envelope. `inspect-prepared`
returns non-zero when any known profile check is false or a recorded profile
cannot be verified, so launch scripts can stop before a package prepared for a
larger memory envelope is used on a smaller, memory-pressured, or uninspectable
machine. Prepared token-id generation, prepared text generation, prepared
benchmarks, and `serve-prepared` enforce the same profile check before invoking
generation or starting the server. Direct
`PreparedGenerationApp` generation calls and the
`benchmark_prepared_token_ids` Python API re-check the same profile before
running as well, so bypassing the CLI does not bypass the memory-envelope
guard.
For larger-than-memory GLM runs, add `--require-prepared-memory-profile` to
prepared generation, benchmark, inspection, or serving commands. This fails
closed when the manifest lacks the full prepare-time memory envelope
(`prepare_effective_unified_memory_bytes`, `prepare_system_reserve_bytes`,
`recommended_max_live_working_set_bytes`, and
`recommended_min_free_unified_memory_bytes`) instead of silently falling back to
generic defaults from a hand-written or old manifest.
When a manifest does record routed-expert quantization or group size, manifest
loading also checks those fields against the expert layout before exposing the
package to profile matching or readiness checks, so stale hand edits cannot
make a package look like a different 4-bit format.
Prepared generation also uses that original
model directory as the default `--model-config`, so GLM-5.2 dense-prefix layers
are picked up without repeating `--dense-layers`; expert and resident layout
`model_type` fields must match that config before a strict GLM launch can
proceed, and router metadata
(`scoring_func`, `norm_topk_prob`, `routed_scaling_factor`, `n_group`,
`topk_group`) and GLM DSA indexer metadata are derived from the same config for
prepared generation and serving. Programmatic server use can set
`PreparedServerConfig(require_glm_4bit=True)`, and
`benchmark_prepared_token_ids(..., require_glm_4bit=True)` applies the same
readiness gate before generation. That gate verifies the affine-int4
`component_order` plus each gate/up/down component's dtype, shape, byte size,
and exact packed offset inside each SSD-backed expert slot, so hand-edited
layouts cannot shift scales or biases while still passing the size checks. Use
`require_public_glm_5_2_shape=True` on those APIs, or
`--require-public-glm-5-2-shape` on prepared CLI commands, when a profile or
launch script is meant specifically for the public GLM-5.2 checkpoint shape.
That strict public-shape gate rejects the debug-only
`--allow-missing-dsa-indexer` escape hatch, so production GLM-5.2 launches
cannot silently drop DSA indexer residency coverage.
`serve-prepared` front-loads that public-shape gate before starting the local
server, matching the prepared generation and benchmark commands.
Prepared token-id, text, and
benchmark commands automatically enable `--batch-prefill-prompt` for prompts
longer than one token; use `--no-batch-prefill-prompt` only when comparing
against the older token-by-token prompt replay path. Their batch-prefill chunk
size defaults to `auto`, so prepared generation scales up from the old 64-token
fallback only when the configured safety caps allow it. Auto sizing accounts
for the selected resident linear backend, including the bounded F32 conversion
buffer used by MPSGraph for BF16/F16 matrices. Prepared batch prefill also
defaults `--prefill-static-capacity-per-expert auto`, so routed prompt MoE uses
the binary `LLMSCAP1` route table unless explicitly disabled with
`--prefill-static-capacity-per-expert none`. If the manifest carries
`recommended_max_live_working_set_bytes` or
`recommended_min_free_unified_memory_bytes`, prepared generation and serving use
those as the default `--max-live-working-set-mib` and
`--min-free-unified-memory-gib`; pass an explicit `0` to either option only for
debugging when you want to disable that guard. The prepared server also reports
the effective guard values from `/health`, including a `memory_guard` block with
configured required available bytes, current system available bytes, and
`available_ok`. When started with `--require-launch-audit PATH`, `/health` also
includes `launch_audit_envelope`, the audited prompt/max-new-token envelope and
the server caps that were checked against it. Prepared text generation uses the
same `--max-prompt-bytes` prompt input cap before tokenization.
For M5/MPSGraph/MPP bring-up, prepared generation, benchmark, serving, and
inspection accept `--require-prefill-acceleration`. It returns non-zero before
generation or server startup unless the selected prefill backend has an
accelerated backend that the current command can actually select. Today that is
the MPSGraph resident GEMM path; MPP tensor ops are reported separately as a
visible acceleration runtime until a selectable MPP prefill backend is wired
into generation. This is opt-in so baseline `custom-metal` runs remain
available. Prepared health and `prefill-backend --json` expose
`suggested_prefill_acceleration_flags` when a selectable accelerated backend is
present, currently suggesting the MPSGraph resident GEMM path with the
acceleration gate and runtime probe replay flag enabled. When a request profile
is available, the same flag also checks
actual resident GEMM coverage: `inspect-prepared --check-prompt-tokens ...`
returns non-zero if the resolved prompt chunk has no matrices assigned to an
accelerated backend. Prepared token-id generation and prepared token-id
benchmarks run that same coverage preflight for their prompt token ids before
calling the generator. Prepared text generation does the same after a lightweight
tokenizer encode when the flag is set, and `serve-prepared` applies the same
guard to token-id, text, completions, and chat requests before generation. The
prepared generation, benchmark, and server paths also check the actual
`prompt_prefill.prefill_acceleration_coverage` after the run and fail before
returning a success payload if the real prompt prefill did not report an
accelerated resident GEMM.
Use `--prefill-min-accelerated-flop-fraction <0..1>` when you want that gate to
cover the dominant resident prefill work, not merely one matrix. Values greater
than zero imply `--require-prefill-acceleration`: request checks require the
resolved prompt chunk to meet the configured accelerated FLOP fraction, and
generation, benchmark, and serving reject the actual run if
`prompt_prefill.prefill_acceleration_coverage.accelerated_flop_fraction` falls
below the same threshold. This is useful for M5/MPSGraph bring-up and future
MPP/Neural Accelerator experiments where a small accelerated GEMM should not be
treated as evidence that prefill is meaningfully off the custom-Metal fallback.
When request checks find a larger prompt chunk that satisfies the acceleration
frontier, the suggested argv preserves `--require-prefill-acceleration` and
`--prefill-min-accelerated-flop-fraction` alongside the chunk size.
`inspect-prepared` runs the same prepared manifest/config/context validation
and health calculation without starting a server, creating work files, or
loading model weights, so it is the quick pre-launch check for backend warnings,
effective prompt/context limits, prepared storage bytes, live-memory guard
defaults, `memory_guard.available_ok`, and current system memory. With
`--require-glm-4bit`, it also requires the prepared artifact to be a
config-consistent affine-int4 GLM MoE package: the check validates the GLM
model type, verified layout `config_sha256` identity, manifest-recorded expert
quantization/group metadata, matching expert-layout quantization/group metadata,
MoE layer ids, routed expert count,
gate/up/down expert component dtype, shape, and byte sizes, plus resident
embedding/final-norm, attention, router, dense/shared MLP, and full-DSA
indexer tensor metadata from `config.json`. It also requires the resident
layout's router metadata to match the typed config fields used by decode and
prefill routing, so stale prepared packages with different top-k, scoring, or
normalization semantics fail before launch. The expert layout's `num_layers`
is checked against the model's total layer count, while the packed expert
`layers` array is checked against only the config-derived MoE layer ids, so
dense-prefix GLM layouts are accepted without pretending every layer has
SSD-backed experts. It also rejects resident layouts
that still contain config-declared routed expert tensors, because those experts
must remain SSD-backed rather than inflate the resident working set. When
`config.json` declares `tie_word_embeddings=false`, the readiness gate also
requires a resident `lm_head.weight`; tied embedding fallback is only allowed
for configs that omit or enable tied embeddings. If `vocab_size` is present,
resident embedding and output-head row counts must match it; generation also
reuses the config's `hidden_size` for embedding and output-head runtime guards
before streaming rows or launching final logits.
The same gate validates the decode-cache layout against the typed GLM config:
every model layer needs one `mla_kv` segment with the config-derived MLA cache
width, full DSA indexer layers need one `dsa_index` segment with
`index_head_dim`, unexpected or duplicate segments fail, and the cache file byte
count must match the layout total before the package is accepted. The payload
reports both values so stale cache files are visible in inspection JSON.
The same readiness payload reports config-derived expert slot bytes, per-layer
expert bytes, total packed expert bytes, a decode-token routed expert read
estimate, and the full-prompt all-experts sweep bytes. A mismatch between the
validated prepared expert bytes and the config-derived affine-int4 total is a
readiness failure, so launch checks cannot silently underestimate SSD expert
volume. The gate also stats every packed expert layer file and requires its
actual byte length to exactly match `num_experts * expert_slot_bytes`; oversized
or duplicated layer files are rejected before a GLM-5.2 launch script trusts the
prepared package. Resident backing is exact-sized too: `resident.bin` must match
the resident layout `total_bytes`, and readiness reports
`resident_weight_file_exact_size`, `resident_weight_file_bytes`, and
`resident_layout_total_bytes`.
This is the recommended gate before a real GLM-5.2 4-bit run because it catches
stale manifests or partially packed expert/resident layouts without mapping
model weights. The same flag is available on prepared token-id generation, prepared
text generation, prepared benchmarks, and `serve-prepared`, so production
launch commands can carry the gate directly. Add
`--require-public-glm-5-2-shape` when the launch should additionally require the
public GLM-5.2 dimensions, layer schedule, routed expert count, top-k, dense
prefix, vocabulary size, context window, MLA head dimensions, DSA indexer
schedule, dtype, activation, attention-bias/dropout assumptions, and router
semantics (`scoring_func`, group top-k, normalized top-k, and routed scaling).
The readiness payload includes
`public_glm_5_2_shape.checks`, `mismatched_fields`, and the DSA full-indexer
schedule summary, so a failed launch can show whether the drift is in tensor
shape, DSA metadata, runner semantic assumptions, or routing semantics. With
`--check-prompt-tokens N --check-max-new-tokens M`, it also runs
the same token-id request admission, auto chunk sizing, and batch-prefill
MLA/DSA cache read/write estimate used by prepared serving. Add
`--check-runtime-preflight` to include the deterministic
layer/cache/final-logits/live-working-set budget used before generation. The
JSON includes `request_check`, and the command exits non-zero when that request
would be rejected. You can also pass `--check-prompt TEXT` or
`--check-prompt-file PATH` to have `inspect-prepared` tokenize a real prompt
with the prepared model tokenizer before running the same request check, still
without starting the server or touching model weights. For chat serving, use
`--check-chat-messages '[{"role":"user","content":"..."}]'` or
`--check-chat-messages-file PATH`; the command renders the local chat template
and then encodes the rendered prompt with `add_special_tokens=false`, matching
the prepared chat endpoint's prompt-token accounting. Offline prompt and chat
check inputs are capped by `--max-request-bytes`, so an accidentally huge prompt
file is rejected after a bounded read. `--require-launch-audit` reuses the same
`request_check` evidence and fails
unless a locked launch profile, strong prepared identity, prepared memory
profile, GLM-5.2 shape gate, prefill acceleration gate, request admission,
runtime memory preflight, and request launch profile are all present and
passing. The request check's
`prefill_linear_backend.top_matrices` field identifies the hottest resident
matrices for the resolved prompt chunk without reading their weight bytes. The
same summary marks `mpp_tensor_ops_candidate` matrices when the resolved chunk
meets the planner's MPP policy (`>=128` tokens and both GEMM dimensions
`>=32`), and reports candidate counts/FLOPs separately from selectable
backends, so M5 MPP bring-up can target real prepared shapes without pretending
the MPP path is wired for execution. `prefill_routed_expert_read` reports
baseline routed expert bytes, planned
chunked bytes, extra bytes, and read amplification from the prepared expert
layout. `prefill_routed_stage_temp_disk` reports the resolved chunk's routed
expert stage + compact temporary disk estimate plus the `LLMSCAP1` static-route
binary bytes, including the single-layer peak and total bytes that would be
staged across the checked prompt.
For decode, `--decode-max-routed-read-gib-per-token` caps the estimated routed
expert slot bytes read by one generated token, using the same deterministic
runtime preflight that sums selected MoE layers' `top_k * expert_slot_bytes`.
`--decode-max-routed-read-seconds-per-token` adds a latency-oriented cap from
that byte count and `--prefill-ssd-read-gib-s` or the prepared package's SSD
read-speed default. Setting either decode cap forces the request/runtime
preflight even when the broader runtime preflight flag is off, so a launch
cannot skip the admission check by accident. When batch prefill records routed
expert read seconds, token generation and serving payloads include
`prefill_actual_read_time` as a top-level summary. That summary now carries the
same serial, unique, planned, waste, coalesced-savings, read-advice, and
read-amplification counters as prompt prefill, so a normal chat smoke can show
whether the first-token bottleneck came from overread, fragmentation, hinting,
or raw copy speed. When generation has decode
steps, a positive SSD GiB/s estimate, and a runtime guard, those payloads also
include `decode_actual_read_time` with actual decoded expert-read bytes/seconds
and the configured total decode read-time cap result; the summary also records
whether actual decode expert-read bytes stayed within the runtime preflight's
planned bytes.
`prefill_routed_chunk_frontier` lists a small offline frontier of candidate
prompt chunk sizes, including powers of two, the expert-saturation threshold,
the resolved chunk, and the full prompt. Each candidate reports worst-case
routed expert read amplification, planned SSD read bytes/seconds when an SSD
GiB/s is supplied, and stage + compact + static-route temporary bytes, making the
activation-vs-SSD tradeoff visible before model weights are mapped. When
`--prefill-max-routed-read-amplification` or
`--prefill-max-routed-read-gib`, or `--prefill-max-routed-read-seconds` is
positive, request checks reject prompts whose resolved chunk size would exceed
any routed-read cap before any prefill stage files are created. Passing
`--prefill-ssd-read-gib-s` without a seconds cap still adds planned
read-second estimates to `prefill_routed_expert_read`. Rejections include a
minimum `prefill_prompt_chunk_tokens` suggestion when a larger chunk can satisfy
the configured caps; otherwise they report that no chunk size can satisfy the
requested routed-read budget. Successful request checks also include
`suggested_guard_flags`, a machine-readable argv-style set of routed-read guard
flags with 5% headroom for pinning later generation, benchmark, or server runs
to the inspected prompt profile. `suggested_stage_temp_guard_flags` provides
the matching argv-style `--prefill-max-stage-mib` and
`--prefill-max-compact-stage-mib`,
`--prefill-max-stage-raw-ranges`, and
`--prefill-max-stage-coalesced-ranges` profile plus the inspected
`--prefill-static-capacity-per-expert` mode, also with 5% headroom, so a later
run can preserve the inspected routed stage/compact/static-capacity temporary
file envelope and read-fragmentation envelope;
request admission now rejects prompts whose resolved stage or compact-stage peak
exceeds those configured caps before checking disk free space.
`suggested_prefill_guard_flags` combines both prefill guard profiles into one
deduplicated argv list, so launch profiles can replay the inspected read and
stage budgets without repeating `--prefill-prompt-chunk-tokens`. Request checks
also include `prefill_stage_temp_disk_free`, a current free-space check for the
selected prompt work directory (default `/private/tmp`) against the largest
routed stage/compact/static-capacity temp peak plus
`--prefill-stage-disk-margin-mib`; generation is rejected when that temporary
disk requirement is unavailable or cannot be verified.
Prepared/server request checks also report `prefill_prompt_chunk_plan`, carrying
the resolved-auto and max-safe cap tables, limiting cap names, and next-token
scratch estimate for auditing prompt chunks before runner files are created.
The plan also reports `mpp_tensor_ops_candidate_reachable_under_caps`,
`mpp_tensor_ops_dimension_candidate_matrix_count`, and
`mpp_tensor_ops_candidate_blocking_cap_names`, plus the categorized
`mpp_tensor_ops_candidate_blocking_cap_summary`, so M5/MPP bring-up can distinguish
"there are no large GEMMs" from "large GEMMs exist, but current prompt/cache/
scratch/stage caps cannot safely reach the 128-token MPP threshold yet."
When expert staging is the blocker, the same plan includes
`mpp_tensor_ops_candidate_stage_tiling_plan`, non-expert blocker summaries,
`mpp_tensor_ops_candidate_streamed_stage_disk_cap_tokens`, and
`mpp_tensor_ops_candidate_stage_tiling_counterfactual_reachable`. These fields
are diagnostic only: they describe whether a future expert-stage tiling plus
streamed stage-temp cleanup path would make an MPP-sized chunk admissible while
leaving the current safety admission unchanged.
Rejected request checks caused by an explicit chunk above the current max-safe
limit still carry this max-safe plan in JSON, so `inspect-prepared` can explain
why a requested MPP-sized chunk was refused without creating runner work files or
loading model weights.
When `inspect-prepared` runs a successful request check, `request_launch_profile`
combines that prompt-specific prefill guard profile with the prepared launch,
acceleration, and decode sections into a single argv list for replaying the
checked request envelope. Its `sections.prefill_prompt_chunk_plan` keeps the
same auto/max-safe chunk-plan evidence as metadata without adding replay args.
When that profile is applied later, generation echoes the same metadata under
`applied_launch_profile.prefill_prompt_chunk_plan` and reports
`prefill_prompt_chunk_plan_drift`, comparing profile max-safe chunk evidence
with the current request/run's max-safe plan. Both offline prepared generation
and `serve-prepared` request checks reject profile replays before runner work
starts when that comparison shows the current max-safe chunk is lower than the
profiled max-safe chunk, when the selected chunk exceeds the current max-safe
chunk, or when the current max-safe evidence cannot be verified.
`generate-prepared-token-ids` and prepared text generation run the same
token-count request admission before calling the generator, so one-off offline
launch scripts get the inspected prompt chunk, routed-read, temp-disk, and
acceleration gate failures before runner work files are created.
Because prepared text must derive its request envelope from the exact tokenizer
output, `generate-prepared-text` fails closed when prompt tokenization cannot be
completed before request admission; it does not fall through to the runner with
an unverified prompt size.
That offline admission reuses the exact generation overrides that will be sent
to the runner, including layer filters, top-k, runtime guard caps, and DSA
settings, so the preflight envelope does not drift from the actual launch.
Prepared token-id benchmarks run the same admission before dispatching the
runner as well, including direct `benchmark_prepared_token_ids(...)` API calls,
so benchmark sweeps cannot bypass the prepared request envelope.
`suggested_decode_guard_flags` similarly converts the inspected per-token decode
routed expert read into `--decode-max-routed-read-*` argv flags with 5%
headroom, including the seconds cap when an SSD GiB/s profile is available.
Prepared serving computes an effective context limit from the decode-cache
layout, prepared manifest, and model `max_position_embeddings` when present.
`/health` reports `prepared_max_context_tokens`,
`decode_cache_context_tokens`, `model_context_tokens`, and
`effective_context_tokens`; token-id, text, completions, and chat requests are
rejected before token generation when `prompt + max_new_tokens` would exceed
that limit.
Prompt prefill telemetry reports
`linear_backend_counts`, so generation, serving, and benchmark JSON show how
many resident GEMMs actually used `custom-metal` versus `mpsgraph-f32`, plus
aggregate resident linear matrix scratch and F32-conversion bytes. New runs
also report `linear_backend_component_stats` inside the prompt-prefill result
and the top-level `prefill_actual_linear_backend` summary. That breakdown keeps
the same counts/FLOPs/elapsed schema but keys it by concrete components such as
`attention.o_proj`, `moe.shared_down_proj`, and
`moe.routed_experts_streamed`, making the next optimization target visible
without rerunning a profiler.
`prefill_acceleration_coverage` collapses those actual counts into
accelerated/custom matrix totals, and `prefill_acceleration_frontier` records
the actual resolved chunk plus layout predictions for candidate chunks such as
`1`, the MPSGraph auto threshold, and the full prompt length. Benchmark
telemetry also reports prompt chunk count, resolved chunk tokens, prefill peak
bytes, embedding bytes, staged expert bytes, compact-stage bytes, routed expert
assignment and unique-slot counts, including the maximum tokens assigned to one
expert in a chunk.
Benchmark summaries also expose the effective MoE token block, estimated MoE
runner peak bytes, staged expert planned/unique/waste bytes, read
amplification, stage-plus-compact temporary bytes, and static-capacity route
slot and binary byte counts. They also expose the prompt-prefill cumulative
stage read seconds, the SSD GiB/s used for that estimate, the configured
read-seconds cap, and whether the actual run stayed within it. Benchmark
summaries also include
`routed_chunk_frontier` when
batch prefill ran, using the same candidate chunk table as `inspect-prepared`
but tagged with `benchmark_actual_prefill`. Compare
`inspect-prepared`'s `prefill_routed_expert_read` prediction with benchmark
stage telemetry to tune prompt chunk sizes without pushing unified memory too
hard, accidentally doubling SSD traffic, or underestimating routed
stage/compact temporary disk pressure.
`inspect-prepared` request checks expose the same decision before a run in
`prefill_linear_backend` and `prefill_acceleration_coverage`, including the
total resident matrix count, MPSGraph count, custom-Metal count, whether any
resident matrix is accelerated, and whether all resident matrices are
accelerated. Coverage also reports FLOPs-weighted fields derived from
`2 * prompt_chunk_tokens * rows * cols`: `linear_backend_flops`,
`total_estimated_flops`, `accelerated_estimated_flops`, and
`accelerated_flop_fraction`. `prefill_linear_backend` also reports
`mpp_tensor_ops_candidate_matrix_count`,
`mpp_tensor_ops_candidate_estimated_flops`, and
`mpp_tensor_ops_candidate_flop_fraction`; the frontier candidate table mirrors
those fields for each candidate chunk size. This keeps M5/MPSGraph bring-up
from looking good
only because a tiny matrix used the accelerated path while the larger resident
GEMMs stayed on custom Metal. `--prefill-min-accelerated-flop-fraction` makes
that FLOPs-weighted view enforceable from CLI, server, and benchmark paths, and
the request payload records the configured minimum under
`min_accelerated_flop_fraction`. Coverage also separates the non-router
unaccelerated gap into streamed routed-expert FLOPs, non-streamed FLOPs, and
backend buckets (`custom-metal`, `unsupported-mpsgraph`, and `other`), so an
M5 run can show whether the next kernel should target streamed MXFP4 experts,
shared/resident fused paths, or an unsupported MPSGraph conversion case. The
same summaries include
`max_matrix_scratch_bytes`, `total_matrix_scratch_bytes`, and raw-conversion
bytes, making MPSGraph F32-conversion churn visible while peak scratch remains
the admission limit. They also include
`prefill_acceleration_frontier`, a small candidate table for prompt chunk sizes
such as the current chunk, the MPSGraph auto threshold, and the safety-capped
maximum. When a larger safe chunk would turn on MPSGraph, the frontier emits
argv-style `--prefill-prompt-chunk-tokens` guard flags that can be reused for
generation, benchmark, or serving.

Run a repeatable token-id benchmark from the same prepared manifest:

```bash
python -m largerlm bench-prepared-token-ids /path/to/checkpoint/largerlm_packed \
  --prompt-token-ids 123,456 --max-new-tokens 32 \
  --logits-top-k 1 --metal-final-logits \
  --max-cache-read-mib 256 --quiet-runner --json
```

The benchmark reports wall time, generated tok/s, estimated embedding, expert,
cache, and logits bytes read, plus prompt MoE capacity telemetry when batch
prefill runs. Batch-prefill summaries include resident linear backend counts,
FLOPs-weighted backend coverage, aggregate MPSGraph/custom-Metal matrix scratch
bytes, prompt chunk counts, prefill peak bytes, staged/compact-stage bytes,
stage-plus-compact temporary bytes, and stage read amplification.
When routed prefill telemetry is present, the benchmark also emits
`suggested_guard_flags` derived from the actual staged expert reads, so the next
generation or server run can reuse the observed safety envelope with 5%
headroom. It also emits `suggested_stage_temp_guard_flags` from the actual
staged/compact-stage peaks, giving matching `--prefill-max-stage-mib` and
`--prefill-max-compact-stage-mib`,
`--prefill-max-stage-raw-ranges`, and
`--prefill-max-stage-coalesced-ranges` arguments for replaying the observed
temporary-file and read-fragmentation envelope. The
`suggested_prefill_guard_flags` field combines those actual-read and
actual-stage suggestions into one argv list for replaying the measured prefill
envelope. When batch prefill ran, `routed_chunk_frontier`
provides the same offline candidate chunk table as `inspect-prepared`, making
measured runs and request checks comparable on the same SSD/stage tradeoff
surface.
The benchmark also emits `suggested_decode_guard_flags` from the actual decode
runtime guard, so the next launch can pin per-token routed expert read caps with
the same 5% headroom.
Use `bench-prepared-token-ids --write-launch-profile PATH` to save the merged
`source="benchmark_actual"` launch profile, including prepared-package identity,
prepared memory launch guards, the prepared/configured SSD read-speed baseline,
GLM 4-bit/public-shape gates, M5 prefill acceleration/probe gates, and the
measured prefill/decode guard sections, for a safer replay on the next
generation or server run. The same profile preserves
`sections.prefill_actual_read_time` with the measured cumulative routed expert
read seconds, SSD GiB/s assumption, configured cap, actual raw/coalesced stage
range counts, and cap result for audit. When batch prefill reports resident
GEMM coverage, the profile also preserves
`sections.prefill_actual_acceleration_coverage` and
`sections.prefill_actual_acceleration_frontier`, so a locked replay can prove
the measured MPSGraph/custom-Metal backend mix and accelerated FLOP fraction
that produced the benchmark instead of relying only on static request
predictions. When `sections.prefill_actual_linear_backend` is also present,
strict launch-audit replay cross-checks it against the coverage section so
backend counts, FLOP buckets, accelerated backend names, and accelerated FLOP
fraction all describe the same measured prefill run. When decode step telemetry
is present,
`sections.decode_actual_read_time`
records actual decode-step expert read bytes and SSD seconds against the
profile's decode seconds/token cap, plus whether actual bytes stayed within the
runtime-preflight planned bytes. The replay argv still comes from the guard
sections. Add
`--require-launch-audit PATH` together with
`--apply-launch-profile` and `--lock-launch-profile` when benchmarking should be
limited to the same prompt/max-new-token envelope that was audited.
It is deliberately lightweight and is intended as the first comparison point
before tuning SSD prefetch or Metal/MPP paths.

Serve a prepared manifest through a small local JSON API:

```bash
python -m largerlm serve-prepared /path/to/checkpoint/largerlm_packed \
  --apply-launch-profile /path/to/checkpoint/largerlm_launch_profile.json \
  --lock-launch-profile \
  --require-launch-audit /path/to/checkpoint/largerlm_launch_audit.json \
  --runner metal/largerlm-runner --host 127.0.0.1 --port 8000 \
  --served-model-name glm-5.2-local \
  --max-new-tokens-cap 256 --max-prompt-tokens 4096 \
  --prefill-linear-backend auto --max-live-working-set-mib 8192 \
  --expert-read-advise-merge-gap-kib 128 --expert-read-advise-align-kib 4 \
  --quiet-runner
```

The server exposes `GET /health`, `POST /generate-token-ids`,
`POST /generate-text`, `GET /v1/models`, and a non-streaming OpenAI-compatible
`POST /v1/completions` endpoint. `POST /v1/chat/completions` is also available
when the selected tokenizer backend can render the model's real chat template;
with `--tokenizer-backend auto`, the server tries the local Transformers
tokenizer first for that route. It is intentionally conservative: requests are
admitted through the prepared manifest, generation still runs the same runtime
preflight and live working-set guard, multi-token prompts default to chunked
batch prefill with static-capacity routes, and requests are serialized around
the shared decode-cache backing file. `POST /generate-token-ids`,
`POST /generate-text`, and the OpenAI-compatible completion/chat endpoints run
the same request admission used by `inspect-prepared --check-prompt-tokens`
with runtime memory preflight before entering the generation lock, so unsafe
prompt chunk sizes, routed-read caps, prefill stage temp disk requirements,
free-memory reserves, and acceleration gates fail before a runner work
directory is created. Prompt prefill rechecks the live-memory
reserve before each chunk and selected layer, cleaning chunk scratch directories
when the guard trips mid-run. It does not keep a large
application-owned expert cache in memory. Safety and generation admission
failures are returned as HTTP 400 JSON errors rather than internal server
errors, so clients can back off or reduce prompt/token limits. JSON integer
controls such as `max_new_tokens`, `logits_top_k`, `seed`, OpenAI `max_tokens`,
and `n` must be real integers; floats, booleans, and strings are rejected
instead of truncated. Float controls such as `temperature` and `top_p` must be
finite numbers. `POST /generate-text` also honors a
request-level `"batch_prefill_prompt": false` when a low-memory caller wants to
force token-by-token prompt replay for a single request. Use
`--prefill-linear-backend mpsgraph-f32` to force the MPSGraph resident GEMM
path for service prefill experiments, or `custom-metal` to keep the older
bounded Metal path while debugging. The service also accepts the same
low-memory prompt-prefill caps for chunk tokens, prompt batch bytes, cache-write
bytes, stage-file bytes, copy chunk bytes, stage disk margin, and prompt MoE
token block, MPSGraph auto-policy thresholds, a routed expert
read-amplification cap, an absolute routed-read GiB cap, and an SSD-read
seconds cap, so a long running server can be pinned below tighter memory/disk
ceilings than the defaults. Add `--check-ssd-read-speed` to make server startup
run the same bounded sequential read gate used by selected replays: by default
it reads 1GiB in 8MiB chunks from the prepared manifest's cold-read benchmark
file and requires at least 75% of the recorded
`prepare_cold_read_gib_per_second`, failing before the API starts if the
current SSD state is too slow.
`serve-prepared` also forwards `--expert-read-advise-merge-gap-kib` and
`--expert-read-advise-align-kib` to decode and prompt-prefill routed expert
reads, and `GET /health` reports their effective values. These hints do not
create an application-owned expert cache.
Generation responses include the same compact
prompt-prefill backend/scratch and routed-stage telemetry used by the benchmark.
Raw `/generate-token-ids` responses can be passed directly to
`python -m largerlm result-summary`; top-level launch-profile and
launch-audit-envelope fields are treated as replay-binding evidence in the same
way as wrapped HTTP smoke artifacts.
`GET /health` reports the effective context/prompt limits, configured live
guards, `memory_guard.available_ok`, decode/cache/scratch/runtime read-advise
caps, prepared
storage bytes, configured prefill backend plus non-blocking backend warnings,
and a best-effort current system memory snapshot so a client can check headroom
before sending a large prompt.
Streaming, tools, and
multi-candidate completions are deliberately not enabled until the server
scheduler can bound those modes.

Reproduce the tiny end-to-end layer smoke:

```bash
python metal/router_semantics_smoke.py
python metal/layer_moe_smoke.py
python metal/quantized_layer_moe_smoke.py
python metal/shared_layer_moe_smoke.py
python metal/dense_mlp_block_smoke.py
python metal/dense_decoder_layer_smoke.py
python metal/mlp_block_smoke.py
python metal/resident_linear_smoke.py
python metal/rmsnorm_batch_smoke.py
python metal/resident_linear_batch_smoke.py
python metal/resident_mxfp4_router_smoke.py
python metal/resident_mxfp4_linear_smoke.py
python metal/resident_mxfp4_attention_smoke.py
python metal/resident_mxfp4_final_logits_smoke.py
python metal/attention_projection_absorbed_smoke.py
python metal/attention_prefix_batch_smoke.py
python metal/attention_projection_batch_smoke.py
python metal/prefill_dense_mlp_batch_smoke.py
python metal/prefill_routed_mlp_batch_smoke.py
python metal/attn_projections_smoke.py
python metal/rope_smoke.py
python metal/rope_batch_smoke.py
python metal/mla_attention_smoke.py
python metal/mla_attention_batch_smoke.py
python metal/mla_attention_indexed_batch_smoke.py
python metal/mla_attention_absorbed_smoke.py
python metal/decode_cache_contract_smoke.py
python metal/attention_output_smoke.py
python metal/attention_block_smoke.py
python metal/decoder_layer_smoke.py
python metal/decoder_layer_command_smoke.py
python metal/decoder_layer_absorbed_command_smoke.py
python metal/mixed_decode_layers_smoke.py
python metal/decode_layers_smoke.py
python metal/final_logits_smoke.py
```

The runner defaults stay conservative: `--run-layer-moe` refuses top-k values
above `--max-k` (default 8, hard cap 64), refuses router tensors above
`--max-router-mib` (default 64), and refuses expert slots above `--max-slot-mib`
(default 256). It also estimates scratch allocations before Metal buffers are
created and refuses peaks above `--max-runner-scratch-mib` (default 4096). It
reads only the current layer's router tensor and one expert slot at a time.
When `--include-shared-expert` is present, it also reads the current layer's
resident shared expert matrices one at a time and adds that output to the routed
MoE result.
`--run-mlp-block` adds the decoder MLP block wrapper: resident
`post_attention_layernorm.weight` RMSNorm, routed/shared MoE, and residual add.
`--run-resident-linear` is the attention/indexer building block for resident
projections such as `.self_attn.q_a_proj.weight`,
`.self_attn.kv_a_proj_with_mqa.weight`, `.self_attn.kv_b_proj.weight`, and
`.self_attn.o_proj.weight`.
`--run-attn-projections` wraps the current GLM MLA projection prefix:
`input_layernorm`, `q_a_proj`, `q_a_layernorm`, `q_b_proj`,
`kv_a_proj_with_mqa`, `kv_a_layernorm`, and `kv_b_proj`. For GLM-5.2-style
absorbed attention layouts without `kv_b_proj.weight`, it accepts
`embed_q`/`unembed_out`, reports `attention_value_source="absorbed-alias"`, and
skips the `kv_b_proj.f32` intermediate. It reads only those current-layer
resident matrices or aliases one at a time, refuses matrices above
`--max-resident-matrix-mib`, and writes intermediate f32 tensors only when an
`--output-dir` is supplied. With `--cache-layout`, `--cache-file`, and
`--position`, it appends the current token's KV-A latent+RoPE vector to the
decode cache using a small BF16/F32 `pwrite`.
`--run-attn-projections-server-jsonl` keeps one Metal process alive for the
same prompt-prefill fused projection work, one JSONL request per line. It is a
deliberately narrower server contract: it rejects cache layout/file/position
fields, so decode cache append remains on the one-shot `--run-attn-projections`
path. Python exposes the server through `AttentionProjectionsServerSession`;
raw `prefill-prompt` can opt in with
`--persistent-attention-projection-server`, while prepared generation,
inspection, serving, and launch profiles use
`--prefill-persistent-attention-projection-server`.
`--run-rope` applies Metal RoPE to Q-rot and K-rot f32 vectors. It supports the
DeepSeek default half-split rotation and `--rope-interleave` for GLM-style
interleaved weights.
`--run-mla-attention` is a bounded single-token MLA attention score/value
primitive. It reads the current layer's cache prefix and either
`kv_b_proj.weight` or the absorbed `embed_q`/`unembed_out` aliases, rotates
cached K-rot positions, computes stable softmax scores, and writes the per-head
value output. This is a correctness-first kernel; optimized reductions and
fused layer execution are still future work.
`--run-attn-output` applies the resident `o_proj.weight` to one attention value
row and adds the original residual hidden state. `--run-attn-output-batch`
performs the same operation for prompt chunks, optionally writes the pre-residual
projection, and keeps the projection matrix, input, residual, and output inside
one bounded runner scratch estimate.
`--run-attn-output-batch-server-jsonl` keeps one Metal process alive for the
same prompt-prefill batch attention-output primitive, one bounded JSONL request
per line, and exits on `{"command":"quit"}`. Python exposes it as
`AttentionOutputBatchServerSession`; raw `prefill-prompt` can opt in with
`--persistent-attention-output-server`, while prepared generation, inspection,
serving, and launch profiles use
`--prefill-persistent-attention-output-server`. The default remains the
one-shot `--run-attn-output-batch` path because the first locked 126-token GLM
bakeoff improved `o_proj` time but regressed total latency.
The Python `prefill-attention-block-batch` command composes the prompt-chunk
attention path with the same guards: projections, cache write, Q/K RoPE,
causal MLA attention values, `o_proj`, and streaming residual add. Its nested
projection and MLA results report whether `kv_b_proj` or absorbed aliases
provided the attention value source. Non-indexed batch MLA with `batch_tokens>1`
also materializes a per-layer value cache by default when it fits
`--max-runner-scratch-mib`, and reports `mla_value_cache` /
`mla_value_cache_bytes` in JSON; pass
`--no-mla-value-cache` on the low-level batch commands, or set
`LARGERLM_MLA_DISABLE_VALUE_CACHE=1` for the runner, to force the old
recompute-values path. For decoder-style `batch_tokens=1`, set
`LARGERLM_MLA_VALUE_CACHE_SINGLETON=1` to opt into that value cache explicitly;
the full GLM max2 smoke regressed with it, so the default keeps the lean
weights/values batch path. Batch attention projections default to the staged
`--run-attn-projections` runner path for `metal/largerlm-runner`; set
`LARGERLM_DISABLE_BATCH_FUSED_ATTN_PROJECTIONS=1` to debug the older separate
RMSNorm/linear/split commands.
Batch attention output similarly defaults to `--run-attn-output-batch`; set
`LARGERLM_DISABLE_BATCH_FUSED_ATTN_OUTPUT=1` to debug the older resident
batch-linear plus Python residual-add path.
`--prefill-persistent-attention-output-server` can reduce process-launch
overhead for experiments, but the current selected GLM replay does not enable
it because same-window bakeoff retained the faster MLA-server baseline.
For resident GLM MXFP4 batch-linear calls such as batch attention output,
`LARGERLM_MXFP4_BATCH_TOKEN_TILE=2` or `4` selects experimental multi-token
kernels. The default token tile remains `1`; with group size 32, that default
also selects the resident group32-specialized kernel unless
`LARGERLM_MXFP4_GROUP32_SPECIALIZED=0` is set. Set
`LARGERLM_MXFP4_GROUP32_SPECIALIZED=1` to force the same path explicitly;
forcing it requires token tile 1 and group size 32.
For GLM shared experts stored as resident MXFP4 matrices, custom-metal prefill
uses `--run-shared-expert-batch` by default. That command runs shared
gate/up fused SwiGLU and shared down projection in one runner invocation while
including all three resident matrices, the activation buffer, input, and output
in the scratch estimate. Set `LARGERLM_DISABLE_FUSED_SHARED_EXPERT_BATCH=1` to
debug the older separate gate/up/SwiGLU/down path.
`--run-shared-expert-batch-server-jsonl` keeps one Metal process alive for the
same fused shared-expert batch primitive, one bounded JSONL request per line,
and exits on `{"command":"quit"}`. Python exposes it through
`ResidentSharedExpertBatchServerSession`; raw `prefill-prompt` can opt in with
`--persistent-shared-expert-server`, while prepared generation, inspection,
serving, and launch profiles use `--prefill-persistent-shared-expert-server`.
The default remains the one-shot `--run-shared-expert-batch` path because the
first locked 126-token GLM bakeoff cut shared-expert tensor time but did not
improve total replay latency over the selected MLA-server baseline.
`LARGERLM_SHARED_MXFP4_GROUP32_SPECIALIZED=1` or `auto` enables an experimental
group-size-32 specialized shared-expert kernel path. It is intentionally not
default because the full 128-token GLM smoke showed that tiny shared-expert
numeric drift can perturb later router choices enough to increase total expert
reads and wall time.
Batch prefill RoPE defaults to `--run-rope-split-batch`, which fuses Python's
old `q_b` split with the Metal RoPE runner call while preserving the same output
files. Set `LARGERLM_DISABLE_FUSED_ROPE_SPLIT_BATCH=1` to force the older
Python split plus `--run-rope-batch` path for debugging.
`--persistent-rope-split-server` on raw `prefill-prompt` and
`--prefill-persistent-rope-split-server` on prepared generation/serving keep one
RoPE split runner process alive across prompt-prefill fused split requests. The
128-token locked bakeoff selected this opt-in as the current process-fusion
baseline, and the 512-token custom-metal fallback bakeoff selected the same
process-fusion bundle over the same-plan MoE-server control. The faster
historical 512 `cache320` router-gate MPSGraph replay remains separate because
its prefill plan depends on MPSGraph availability; a current-host MPSGraph +
process-fusion replay is also recorded, but it did not beat the custom-metal
process-fusion fallback in the noisy current window.
`--run-rmsnorm-batch-server-jsonl` is the matching opt-in for resident batch
RMSNorm requests. Raw `prefill-prompt` exposes it as
`--persistent-rmsnorm-server`; prepared generation, inspection, serving, and
launch profiles expose it as `--prefill-persistent-rmsnorm-server`. It preserves
the existing per-request resident-layout validation, input/output byte checks,
RMSNorm epsilon validation, and scratch cap, but reuses one Metal process across
the many prompt-prefill `input_layernorm`, latent norm, and
`post_attention_layernorm` calls. The tiny smoke
`python metal/rmsnorm_batch_smoke.py` verifies one-shot versus JSONL server
equivalence on a 2-token fixture. Real GLM evidence is intentionally negative
for now: a 128-token candidate tripped the request safety cap
(`prefill_prompt_chunk_tokens 128` exceeds the capped maximum `126`), and the
locked 126-token bakeoff retained the RoPE split baseline after the RMSNorm
server candidate generated the same `[15]` token but ran slower (`220.614s`
versus `211.945s`, `1.041x`). RMSNorm alone therefore did not replace the
then-current RoPE split selected replay, though it remains available for
experiments.
`--run-mla-attention-batch-server-jsonl` is the next, larger process-fusion
boundary for prompt MLA attention. Raw `prefill-prompt` exposes it as
`--persistent-mla-attention-server`, while prepared generation, inspection,
serving, and launch profiles expose it as
`--prefill-persistent-mla-attention-server`. The server accepts bounded
one-shot-equivalent MLA attention JSONL requests, supports indexed and
contiguous prompt batches, preserves the cache/resident/layout/scratch guards,
and exits on `{"command":"quit"}`. The tiny smoke
`python metal/mla_attention_batch_smoke.py` now checks both one-shot
`--run-mla-attention-batch` and the JSONL server on Apple M5 Max. Real GLM
evidence is positive: the locked 126-token candidate generated the same `[15]`
token as the RoPE split baseline, stayed inside the 17.37 GiB live cap with
about 77.40 GiB available, collapsed MLA attention into one
`--run-mla-attention-batch-server-jsonl` runner group, and finished in
`132.394s` versus `211.945s` (`0.625x`, `-79.551s`). The new selected replay is
`selected-replay-prefill-126-persistent-moe-resident-linear-attnproj-rope-split-mla-server-memory-accumulator.json`,
with profile SHA
`f7f1cbd90c2ce28c191607165431b925ca1dbf043511f8a9f2165b6578d7a172`, audit SHA
`d7bcb0a7e555ed2d7ac1614b68d8e4275f8219175e7076453107f6c6dcaef302`, and
selected replay SHA
`0ed3e32aab894d170bbf232cf2badaf0b2c9ea66d0272b1839f9b0745bf6c7ee`.
For routed GLM MXFP4 experts, `--run-moe-batch` accepts
`LARGERLM_MOE_MXFP4_BATCH_TOKEN_TILE=2` or `4`, which evaluates multiple prompt
tokens per packed weight row in the fused expert kernels. The default remains
`1`; use the multi-token settings for explicit benchmarking.
`LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING=1` is a diagnostic-only timing mode for
the same path. It preserves the math but splits the fused MXFP4 MoE dispatch
into separate gate/up/SwiGLU and down/weighted-add command buffers, so use it to
find hotspots rather than to measure the default end-to-end fast path.
`result-summary` now surfaces those recorded split timings as
`routed moe MXFP4 split` plus per-layer `swiglu=` and `down_add=` timing fields
when a run was captured with the diagnostic enabled. If one split side clearly
dominates, the summary also emits a bounded `glm_moe_tile_sweep.py` experiment
command for the hottest routed layer, including selected expert ids when the
artifact preserved them; treat it as a microbench lead and promote only after a
full replay/result-bakeoff win.
The corresponding sweep artifacts include `config_comparison` recommendations:
`candidate_for_full_replay` means the microbench passed kernel-speedup and
drift thresholds, while `requires_full_replay_bakeoff` keeps the default path
unchanged until an end-to-end replay also wins.
`LARGERLM_MOE_MXFP4_VECTOR_SWIGLU=1` selects an experimental tile-1 SwiGLU
kernel that decodes each packed MXFP4 word into `float4` vectors and uses
vector dot products for the gate/up accumulation. It is numerically close to the
default scalar kernel but is not enabled by default because the layer-19
interleaved sweep did not prove a stable throughput gain.
`LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED=0` disables the default routed MXFP4
group-size-32 specialized kernels for A/B testing. Set it to `1` to force the
same path explicitly; forcing it requires token tile 1 and group size 32.
`LARGERLM_MOE_MXFP4_SWIGLU_ACTIVATION=fast-exp` swaps the exact SiLU exponent
for Metal's fast exponent path. `linear` or `skip-silu` skips the activation
entirely and is only for hotspot diagnosis; do not use it for generation.
The Python `prefill-dense-mlp-block-batch` command continues prompt chunks
through GLM-5.2 dense-prefix MLP layers: post-attention RMSNorm, dense gate/up
batch projections, row-streamed SwiGLU, dense down projection, and row-streamed
residual add. Default dense suffixes automatically fall back to
`mlp.switch_mlp.*` names.
The Python `prefill-routed-mlp-block-batch` command is the safe routed MoE
prompt-chunk baseline. It preflights the layer budget, then streams one hidden
row at a time through the existing bounded `--run-mlp-block` runner, appending
each row to the batch output and optionally writing per-token router JSON. This
keeps memory bounded while providing the router traces consumed by the staged
batch path.
`plan-batch-expert-io` consumes those router JSON files and builds the next
scheduler input: token assignments grouped by expert plus a coalesced aligned
expert-slot read plan.
`stage-batch-experts` materializes that read plan into a bounded stage file and
manifest. It streams each planned aligned range in configurable copy chunks, so
the stage file can be prepared without loading all selected expert slots into
memory, and checks free disk plus `--stage-disk-margin-mib` before opening the
stage file for writing. Before copying, it tries to issue macOS `F_RDADVISE`
hints for the already coalesced aligned ranges; unsupported platforms or hint
failures are non-fatal and recorded in the manifest's `read_advice` block. The
manifest also includes `io_summary` telemetry for the serial assignment read
baseline, unique expert-slot lower bound, planned/staged read bytes, aligned
waste, raw/coalesced range counts, savings, read amplification, and stage
budget utilization. Prompt prefill aggregates the same staged expert I/O counters at
top level, including total and per-layer peak stage-plus-compact temporary
bytes, plus read-advice attempted ranges, calls, hinted bytes, and non-fatal
failure count. Generation, benchmark, and serving expose the same key totals
under `prompt_prefill` when batch prefill is enabled.
`run-staged-routed-moe-batch` consumes the stage manifest, builds a compact
layout containing only the batch's unique routed experts, writes a compact
route JSON, and calls the bounded `--run-moe-batch` runner once for the batch.
If the stage file already contains exactly those expert slots in compact order,
the compact layer is a hardlink instead of a second copy; otherwise the
compact-stage file has the same pre-write free-disk guard via
`--compact-stage-disk-margin-mib`. The result reports `compact_stage_storage`
as `hardlink` or `copy`; `compact_stage_bytes` remains the logical expert bytes
the runner reads, while `compact_stage_materialized_bytes` is the extra compact
file storage actually written for this run, which is zero for hardlinks. It also
strictly validates integer manifest fields, finite route weights, stage byte
counts, coalesced range coverage, and slot source/stage offsets before handing
routes to the runner; failed validation or compact artifact writes clean up
compact layout, layer, and route files before returning.
Inside that runner, compact JSON routes are flattened into a small
`batch_tokens * top_k` assignment table and sorted by compact expert. Static
`LLMSCAP1` route tables loaded with `--routes-bin` go directly into the
expert-major assignment table and skip sorting when already monotonic. Each
staged slot is read once, while input and output accumulation stay row-streamed
through a sparse output file. Within one expert, assignments are processed in
bounded `--moe-token-block` chunks by 4bit batch-row dequant, SwiGLU, and
weighted-add kernels; `--moe-token-block auto` picks the largest block that fits
the scratch cap and that expert's assignment group. The first assignment for each token clears the one-row
accumulator instead of reading a known-zero output row; later assignments
reload that row. This avoids per-token process setup and per-token slot rereads
without loading the whole batch into memory. The assignment table, token-seen
bitmap, and token-block buffers are counted in the runner scratch estimate
before they are allocated. Set `--moe-output-accumulator memory` on low-level
staged MoE commands, or `--prefill-moe-output-accumulator memory` on prepared
generation/inspection/serving profiles, to pin the in-memory batch output
accumulator. `env` preserves the legacy `LARGERLM_MOE_BATCH_ACCUMULATOR`
override, while `file` and `memory` write an explicit child-process setting.
The default remains file-backed unless a profile pins the mode, and the full
accumulator byte size is added to the same scratch estimate before allocation.
The staged wrapper parses runner telemetry back into JSON fields such as
`effective_moe_token_block`, `moe_max_expert_tokens`,
`moe_batch_buffer_bytes`, `moe_estimated_peak_bytes`,
`moe_output_accumulator`, and `moe_output_accumulator_bytes`.
`prefill-staged-routed-mlp-block-batch` composes that boundary into a prompt
MLP block: batch post-attention RMSNorm, single-process batch router JSON,
bounded expert stage, staged routed MoE, optional shared expert, and row-streamed residual add.
Shared expert matrices remain under `--max-resident-matrix-mib`; routed expert
stage and compact-stage files remain under their separate caps and disk-margin
checks.
`prefill-prompt` is the first multi-layer prompt driver over those batch
pieces. It processes prompt token ids in bounded chunks, streams embedding rows
from resident weights, runs each chunk through the selected layer stack, writes
per-layer MLA cache rows at `start_position + token_index`, and extracts the
last prompt hidden row for logits. The driver is chunk-major rather than
layer-major, so each later chunk can attend to cache rows written by earlier
chunks at every layer. By default, completed chunk work directories are cleaned
immediately; `--keep-work-dir` is only for debugging.
`metal/attention_block_smoke.py` composes the attention-side pieces into one
tiny current-token flow: projection, cache append, Q split, RoPE, MLA
score/value, output projection, and residual add.
`metal/decoder_layer_smoke.py` feeds that attention residual boundary into
`--run-mlp-block`, giving a tiny full decoder-layer correctness harness while
still keeping every command bounded by explicit matrix/cache/scratch limits.
`--run-decoder-layer` wraps the same sequence in one runner invocation. The
current implementation still uses a private bounded work directory for
intermediate f32 tensors, which keeps the milestone low risk while preserving
the same slot/router/matrix/cache/scratch guards.
`decode-layers` is the first multi-layer hidden-state driver over those packed
layer commands. It runs selected layers sequentially and preflights every layer
with decoder cache/scratch budgets before launching the Metal runner. It does
not include tokenization, embeddings, logits, sampling, or a server loop.
Automatically-created decode-layer work directories are cleaned on failure by
default; `--keep-work-dir` is only for debugging failed layer runs.
When `--model-config` is supplied, it derives GLM MLA dimensions, top-k,
RMSNorm epsilon, RoPE theta/interleave mode, router score, and shared-expert
inclusion from the Hugging Face config unless an explicit CLI flag overrides
them; unknown `mlp_layer_types` values are rejected before dense/MoE layer
scheduling.
`final-logits` adds the next boundary: final RMSNorm followed by streaming
`lm_head.weight` or tied embedding top-k. It reads the head in bounded row
chunks controlled by `--max-chunk-mib`, and only writes the full f32 logits
vector when `--output-logits-f32` is explicitly requested. Model-config-driven
generation disables tied embedding fallback when `tie_word_embeddings=false`
and checks runtime embedding/head row counts against `vocab_size` plus
embedding/head hidden dimensions against `hidden_size` when the config declares
them. Runtime preflight also checks the resident embedding row against
`--max-embedding-row-mib`, and prepared request inspection exposes the
embedding row/output bytes alongside layer, logits, and live-memory budgets.
`embed-token` is the matching single-token input boundary: it streams one
`embed_tokens.weight` row for a token id into an f32 hidden vector, bounded by
`--max-row-mib`. `embed-tokens-batch` extends the same row-streaming behavior
to prompt chunks, writing a `[tokens, hidden]` f32 batch while guarding the
total output with `--max-output-mib`. Together, `embed-token`, `decode-layers`,
and `final-logits` form the current tiny token-id-to-top-k decode loop.
`generate-token-ids` and `generate-text` wrap that loop with sampling, local tokenizer encode/decode,
per-token cleanup, telemetry, and an additional whole-generation preflight.
Generation and server JSON split estimated reads into embedding, routed-expert,
cache, and logits bytes, so the reported total can be reconciled after GLM-5.2
experiments without parsing runner logs.
Each generated step now carries a compact `decode_layers` telemetry list with
the layer id, dense/MoE kind, estimated routed-expert read bytes, MLA/DSA cache
read bytes, peak scratch estimate, and runner stage timings when the batched
decoder report provides them. Server JSON returns the same per-step summary, so
GLM-5.2 runs can identify SSD-heavy or kernel-heavy layers without parsing
runner commands or loading model weights for postmortem accounting.
The Metal runner caches the shared `expert_kernel_source()` library, compute
pipeline states, command queue, read-only layout/cache JSON dictionaries, and
resident/decode-cache metadata and backing validation indexes within each
runner process, so batched layer execution avoids recreating the same Metal
runtime objects or rescanning the same small metadata files for every decoder
layer.
The batched decoder report now marks each layer boundary with
`input_in_memory` and `output_in_memory`, and the Python driver, CLI summaries,
generation JSON, and HTTP server payload preserve those booleans. Canonical
GLM-5.2 artifacts can therefore verify that intermediate hidden states stayed
inside the runner process without scraping temporary work directories.
When `--run-decoder-layers` also receives `--output-topk-json`, the runner keeps
the final hidden vector in memory, immediately runs Metal final logits on it,
and writes the top-k JSON with logits read/elapsed metadata. Guarded
token-generation requests use that fused path when the batched decoder branch is
eligible, avoiding the previous final-hidden file and second runner process.
Single-token decoder layers also keep the attention projection `q_b` result in
memory, split it and apply RoPE for `q_rope` directly in the runner process,
pass the resulting `q_nope`/`q_rope` tensors to MLA attention in memory, and
pass the small MLA attention value tensor to the attention-output projection in
memory. The attention-output residual vector can then be handed directly to the
MLP block without materializing `attn_out.f32` inside the same decoder layer.
That avoids the old temporary `q_rot`/dummy-k files, one tiny standalone RoPE
runner command per layer, plus the intra-layer `attn_q_b.f32`, `q_nope.f32`,
`q_rope.f32`, `attn_value.f32`, and `attn_out.f32` write/read boundaries. The
batched decoder loop also carries intermediate layer hidden states in memory
and only writes the final hidden vector when the fused final-logits path is not
selected, while
standalone `--run-rope`, `--run-mla-attention`, `--run-attention-output`,
`--run-mlp-block`, and `--run-dense-mlp-block` commands keep their file-based
interfaces for tests and batch/prompt plumbing.
When `--model-config` is provided and the packed expert/resident layouts record
`config_sha256`, direct generation refuses mismatched configs before deriving
GLM shapes or router/DSA defaults. Runtime backing validation also rejects
expert and resident layouts that both record different config hashes or
overlapping byte spans.
`--batch-prefill-prompt` makes generation use `prefill-prompt` for the prompt
phase, select the first generated token from the final prompt hidden row, and
then continue with the existing single-token decode loop for new tokens. This
keeps the old path available while avoiding per-token prompt replay on long
inputs. `--prefill-moe-token-block auto` keeps the prompt MoE runner under the
same scratch cap and reports the aggregated prompt-prefill MoE telemetry in the
generation JSON under `prompt_prefill`; `--prefill-static-capacity-per-expert auto`
uses binary static-capacity prompt routes instead of compact JSON routes.
When generation resolves `--prefill-prompt-chunk-tokens auto`, the JSON result
also includes `auto_prefill_prompt_chunk_plan` so actual runs retain the same
cap-table and next-token scratch evidence as request preflight. Batch-prefill
generation also reports `max_safe_prefill_prompt_chunk_plan`, including fixed
chunk runs, so launch-profile replays can compare the chosen chunk with the
current safety-capped maximum.
With a GLM config that declares DSA `indexer_types`, batch prefill also
computes full-indexer cache rows and reuses full-layer top-k files for shared
layers, and the following single-token decode loop continues using the same
full/shared DSA schedule and `indexer_rope_interleave` mode. Generation still
refuses `dsa_index` cache layouts when no DSA schedule is available; it also
rejects unknown DSA schedule values and full-indexer layers without `dsa_index`
cache segments. `--allow-missing-dsa-indexer` is only for explicit debugging
and should not be used for correctness runs.
`serve-prepared` exposes the prepared path through a local JSON API while
preserving the same manifest validation, runtime preflight, live memory guard,
batch prefill defaults, and bounded cleanup policy. It serializes requests
around the shared decode-cache file; continuous batching and prefix sharing are
still future work.
`generate-token-ids` wraps that trio into a greedy token-id loop. The default
path processes prompt token ids one at a time; `--batch-prefill-prompt` switches
the prompt phase to the chunked batch driver before continuing single-token
decode. Both modes update the decode cache by position, emit greedy next-token
ids, and refuse to run if prompt plus requested generation would exceed the
cache layout context budget. When `--model-config` provides a GLM-style
`eos_token_id` integer or list, the config loader validates that every id is a
non-negative integer and all listed ids are used as stop tokens unless
`--eos-token-id` explicitly overrides them.

Router execution reads GLM-style metadata from `resident/layout.json` when
present: `norm_topk_prob`, `routed_scaling_factor`, `n_group`, `topk_group`, and
optional `gate.e_score_correction_bias`. The correction bias affects top-k
selection; weights are gathered from the original router scores, normalized if
enabled, then multiplied by the routed scaling factor. Higher-level decode,
prompt-prefill, and generation commands also derive `scoring_func`,
`norm_topk_prob`, `routed_scaling_factor`, `n_group`, and `topk_group` from
`--model-config`; explicit CLI flags still override those defaults. Prepared
server requests inherit the same defaults from the prepared model config. These
router, RoPE, EOS, and config weight-dtype semantics are now parsed as typed
config fields, and resident packing writes router metadata from those parsed
values rather than ad hoc raw JSON coercions, so invalid types, unknown
`dtype`/`torch_dtype` values, non-positive routed scaling/groups, and
`topk_group > n_group` fail before any runner or work directory starts.

The config-only estimate is intentionally conservative: routed experts assume
MLX affine quantization, while resident non-expert weights are estimated from
config-declared GLM attention, dense-prefix MLP, shared expert, router, DSA
indexer, embedding, and lm-head shapes. Config scalar validation runs before
these estimates, so malformed GLM configs fail with a config error instead of
producing optimistic memory budgets.
When `--max-context-tokens` is supplied, `plan` also estimates total MLA/DSA
decode cache bytes and compares it with either `--max-cache-gib` or the cache
budget left after system reserve, runtime buffers, resident weights, and target
OS page cache. When a cache budget is known, it also reports the safe cache
context token count under that budget; use that number instead of the
checkpoint's advertised maximum context when memory headroom matters.

## Direction

The runtime should split the model into two execution regimes:

- Prefill: batched dense GEMMs, attention/indexer work, and long prompt compute.
  This is the part that should use Metal 4 machine-learning/tensor resources
  and MPP tensor ops on M5 GPU neural accelerators where the local SDK and
  device expose them. `prefill-backend` reports whether that path is currently
  available and which fallback should be used.
- Decode: routed expert matvecs dominated by SSD reads and unified memory
  bandwidth. This should use a custom Metal streaming path with OS page cache,
  aligned `pread`, and fused affine dequant matvec kernels.

See [docs/architecture.md](docs/architecture.md) and
[docs/research-notes.md](docs/research-notes.md).
