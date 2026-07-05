#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layer_moe_smoke import write_fixture


def _write_cache_fixture(root: Path) -> tuple[Path, Path]:
    layout_path = root / "cache_layout.json"
    cache_path = root / "decode_cache.bin"
    layout_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 2,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 16,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 0,
                        "width": 4,
                        "dtype": "BF16",
                        "dtype_bytes": 2,
                        "token_stride_bytes": 8,
                        "max_context_tokens": 2,
                        "total_bytes": 16,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cache_path.write_bytes(b"\0" * 16)
    return layout_path, cache_path


def _base_moe_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--layout",
        str(root / "experts" / "layout.json"),
        "--layer",
        "1",
        "--run-moe",
        "--experts",
        "0",
        "--weights",
        "1",
        "--input-f32",
        str(root / "input.f32"),
        "--output-f32",
        str(root / "out.f32"),
        "--max-slot-mib",
        "1",
    ]


def _base_batch_cmd(runner: Path, root: Path, routes_path: Path) -> list[str]:
    return [
        str(runner),
        "--layout",
        str(root / "experts" / "layout.json"),
        "--layer",
        "1",
        "--run-moe-batch",
        "--routes-json",
        str(routes_path),
        "--input-f32",
        str(root / "input.f32"),
        "--batch-tokens",
        "1",
        "--output-f32",
        str(root / "batch_out.f32"),
        "--max-k",
        "1",
        "--max-slot-mib",
        "1",
    ]


def _base_router_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-router",
        "--input-f32",
        str(root / "input.f32"),
        "--top-k",
        "1",
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_router_batch_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-router-batch",
        "--input-f32",
        str(root / "input.f32"),
        "--batch-tokens",
        "1",
        "--top-k",
        "1",
        "--output-router-json-dir",
        str(root / "router_batch"),
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_layer_moe_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--layout",
        str(root / "experts" / "layout.json"),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-layer-moe",
        "--input-f32",
        str(root / "input.f32"),
        "--top-k",
        "1",
        "--max-k",
        "1",
        "--router-score",
        "raw",
        "--max-slot-mib",
        "1",
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_mlp_block_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--layout",
        str(root / "experts" / "layout.json"),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-mlp-block",
        "--input-f32",
        str(root / "input.f32"),
        "--top-k",
        "1",
        "--max-k",
        "1",
        "--router-score",
        "raw",
        "--max-slot-mib",
        "1",
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_final_logits_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--run-final-logits",
        "--input-f32",
        str(root / "input.f32"),
        "--top-k",
        "1",
        "--max-chunk-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_attn_projections_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-attn-projections",
        "--input-f32",
        str(root / "input.f32"),
        "--max-resident-matrix-mib",
        "1",
        "--max-cache-file-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_dense_mlp_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-dense-mlp-block",
        "--input-f32",
        str(root / "input.f32"),
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_resident_linear_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-resident-linear",
        "--tensor-suffix",
        "mlp.gate.weight",
        "--input-f32",
        str(root / "input.f32"),
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_rmsnorm_batch_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-rmsnorm-batch",
        "--norm-suffix",
        "input_layernorm.weight",
        "--input-f32",
        str(root / "input.f32"),
        "--batch-tokens",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_resident_linear_batch_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-resident-linear-batch",
        "--tensor-suffix",
        "mlp.gate.weight",
        "--input-f32",
        str(root / "input.f32"),
        "--batch-tokens",
        "1",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_validate_cache_cmd(runner: Path, cache_layout: Path, cache_file: Path) -> list[str]:
    return [
        str(runner),
        "--validate-cache-backing",
        "--cache-layout",
        str(cache_layout),
        "--cache-file",
        str(cache_file),
        "--max-cache-file-mib",
        "1",
    ]


def _base_decoder_cmd(
    runner: Path,
    root: Path,
    cache_layout: Path,
    cache_file: Path,
) -> list[str]:
    return [
        str(runner),
        "--layout",
        str(root / "experts" / "layout.json"),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--cache-layout",
        str(cache_layout),
        "--cache-file",
        str(cache_file),
        "--layer",
        "1",
        "--run-decoder-layer",
        "--input-f32",
        str(root / "input.f32"),
        "--position",
        "0",
        "--context-length",
        "2",
        "--num-heads",
        "1",
        "--qk-nope-dim",
        "2",
        "--rope-dim",
        "2",
        "--v-head-dim",
        "2",
        "--output-f32",
        str(root / "decoder_out.f32"),
        "--top-k",
        "1",
        "--max-k",
        "1",
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-resident-matrix-mib",
        "1",
        "--max-slot-mib",
        "1",
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_dense_decoder_cmd(
    runner: Path,
    root: Path,
    cache_layout: Path,
    cache_file: Path,
) -> list[str]:
    cmd = _base_decoder_cmd(runner, root, cache_layout, cache_file)
    cmd.remove("--layout")
    cmd.remove(str(root / "experts" / "layout.json"))
    cmd[cmd.index("--run-decoder-layer")] = "--run-dense-decoder-layer"
    for flag in ("--top-k", "--max-k", "--max-slot-mib", "--max-router-mib"):
        idx = cmd.index(flag)
        del cmd[idx : idx + 2]
    return cmd


def _base_mla_cmd(
    runner: Path,
    root: Path,
    cache_layout: Path,
    cache_file: Path,
) -> list[str]:
    return [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--cache-layout",
        str(cache_layout),
        "--cache-file",
        str(cache_file),
        "--layer",
        "1",
        "--run-mla-attention",
        "--q-nope-f32",
        str(root / "input.f32"),
        "--q-rope-f32",
        str(root / "input.f32"),
        "--context-length",
        "2",
        "--num-heads",
        "1",
        "--qk-nope-dim",
        "2",
        "--rope-dim",
        "2",
        "--v-head-dim",
        "2",
        "--output-f32",
        str(root / "mla_out.f32"),
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_rope_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--run-rope",
        "--q-f32",
        str(root / "input.f32"),
        "--k-f32",
        str(root / "input.f32"),
        "--output-q-f32",
        str(root / "rope_q.f32"),
        "--output-k-f32",
        str(root / "rope_k.f32"),
        "--num-heads",
        "1",
        "--rope-dim",
        "2",
        "--position",
        "0",
        "--max-runner-scratch-mib",
        "64",
    ]


def _base_rope_batch_cmd(runner: Path, root: Path) -> list[str]:
    return [
        str(runner),
        "--run-rope-batch",
        "--q-f32",
        str(root / "input.f32"),
        "--k-f32",
        str(root / "input.f32"),
        "--output-q-f32",
        str(root / "rope_batch_q.f32"),
        "--output-k-f32",
        str(root / "rope_batch_k.f32"),
        "--num-heads",
        "1",
        "--rope-dim",
        "2",
        "--start-position",
        "0",
        "--batch-tokens",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _run_reject_case(cmd: list[str], *, name: str, expected_error: str) -> None:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    combined = completed.stdout + completed.stderr
    if completed.returncode == 0:
        raise SystemExit(f"{name}: runner accepted invalid CLI numeric input")
    if "Metal device unavailable" in combined:
        raise SystemExit(
            f"{name}: CLI rejection happened after Metal device creation:\n{combined}"
        )
    if expected_error not in combined:
        raise SystemExit(
            f"{name}: expected error {expected_error!r}; got:\n{combined}"
        )
    print(f"  rejected {name}: {expected_error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify runner-side strict CLI numeric parsing failures.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("largerlm-runner"),
        help="Path to the compiled Metal runner.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Fixture directory. Defaults to a new /private/tmp directory.",
    )
    args = parser.parse_args()

    root = args.root or Path(
        tempfile.mkdtemp(prefix="largerlm-runner-cli-contract-", dir="/private/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    cache_layout, cache_file = _write_cache_fixture(root)
    routes_path = root / "routes.json"
    routes_path.write_text(
        json.dumps(
            {
                "batch_tokens": 1,
                "routes": [{"experts": [0], "weights": [1.0]}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"fixture: {root}")

    cmd = _base_moe_cmd(args.runner, root)
    cmd[cmd.index("--experts") + 1] = "0x"
    _run_reject_case(
        cmd,
        name="moe-expert-suffix",
        expected_error="MoE expert must be an integer",
    )

    cmd = _base_moe_cmd(args.runner, root)
    cmd[cmd.index("--weights") + 1] = "nan"
    _run_reject_case(
        cmd,
        name="moe-weight-nan",
        expected_error="MoE weight must be a finite number",
    )

    cmd = _base_moe_cmd(args.runner, root)
    cmd.extend(["--expert-read-advise-align-kib", "4x"])
    _run_reject_case(
        cmd,
        name="read-advise-suffix",
        expected_error="--expert-read-advise-align-kib must be an integer",
    )

    cmd = _base_batch_cmd(args.runner, root, routes_path)
    cmd[cmd.index("--batch-tokens") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="batch-tokens-suffix",
        expected_error="--batch-tokens must be an integer",
    )

    cmd = _base_batch_cmd(args.runner, root, routes_path)
    cmd[cmd.index("--max-k") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="max-k-suffix",
        expected_error="--max-k must be an integer",
    )

    cmd = [
        str(args.runner),
        "--layout",
        str(root / "experts" / "layout.json"),
        "--layer",
        "1",
        "--expert",
        "0x",
        "--run-expert",
        "--input-f32",
        str(root / "input.f32"),
        "--max-slot-mib",
        "1",
    ]
    _run_reject_case(
        cmd,
        name="expert-suffix",
        expected_error="--expert must be an integer",
    )

    cmd = _base_router_cmd(args.runner, root)
    cmd.extend(["--routed-scaling-factor", "nan"])
    _run_reject_case(
        cmd,
        name="router-scale-nan",
        expected_error="--routed-scaling-factor must be a finite number",
    )

    cmd = _base_router_cmd(args.runner, root)
    cmd.extend(["--routed-scaling-factor", "0"])
    _run_reject_case(
        cmd,
        name="router-scale-zero",
        expected_error="--routed-scaling-factor must be positive",
    )

    cmd = _base_router_batch_cmd(args.runner, root)
    cmd.extend(["--router-n-group", "2x"])
    _run_reject_case(
        cmd,
        name="router-batch-group-suffix",
        expected_error="--router-n-group must be an integer",
    )

    cmd = _base_layer_moe_cmd(args.runner, root)
    cmd.extend(["--router-topk-group", "0"])
    _run_reject_case(
        cmd,
        name="layer-moe-topk-group-zero",
        expected_error="--router-topk-group must be in 1..",
    )

    cmd = _base_mlp_block_cmd(args.runner, root)
    cmd.extend(["--router-n-group", "0"])
    _run_reject_case(
        cmd,
        name="mlp-block-group-zero",
        expected_error="--router-n-group must be in 1..",
    )

    cmd = _base_validate_cache_cmd(args.runner, cache_layout, cache_file)
    cmd[cmd.index("--max-cache-file-mib") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="validate-cache-mib-suffix",
        expected_error="--max-cache-file-mib must be a finite number",
    )

    cmd = _base_decoder_cmd(args.runner, root, cache_layout, cache_file)
    cmd[cmd.index("--top-k") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="decoder-top-k-suffix",
        expected_error="--top-k must be an integer",
    )

    cmd = _base_decoder_cmd(args.runner, root, cache_layout, cache_file)
    cmd[cmd.index("--max-cache-read-mib") + 1] = "1.5"
    cmd.extend(["--rms-norm-eps", "nan"])
    _run_reject_case(
        cmd,
        name="decoder-fractional-mib-rms-nan",
        expected_error="--rms-norm-eps must be a finite number",
    )

    cmd = _base_decoder_cmd(args.runner, root, cache_layout, cache_file)
    cmd.extend(["--expert-read-advise-merge-gap-kib", "1x"])
    _run_reject_case(
        cmd,
        name="decoder-read-advise-suffix",
        expected_error="--expert-read-advise-merge-gap-kib must be an integer",
    )

    cmd = _base_dense_decoder_cmd(args.runner, root, cache_layout, cache_file)
    cmd[cmd.index("--context-length") + 1] = "2x"
    _run_reject_case(
        cmd,
        name="dense-decoder-context-suffix",
        expected_error="--context-length must be an integer",
    )

    cmd = _base_dense_decoder_cmd(args.runner, root, cache_layout, cache_file)
    cmd.extend(["--attention-scale", "nan"])
    _run_reject_case(
        cmd,
        name="dense-decoder-attention-scale-nan",
        expected_error="--attention-scale must be a finite number",
    )

    cmd = _base_mla_cmd(args.runner, root, cache_layout, cache_file)
    cmd[cmd.index("--max-cache-read-mib") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="mla-cache-read-suffix",
        expected_error="--max-cache-read-mib must be a finite number",
    )

    cmd = _base_mla_cmd(args.runner, root, cache_layout, cache_file)
    cmd[cmd.index("--max-cache-read-mib") + 1] = "1.5"
    cmd.extend(["--rope-theta", "nan"])
    _run_reject_case(
        cmd,
        name="mla-fractional-mib-rope-theta-nan",
        expected_error="--rope-theta must be a finite number",
    )

    cmd = _base_rope_cmd(args.runner, root)
    cmd[cmd.index("--position") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="rope-position-suffix",
        expected_error="--position must be an integer",
    )

    cmd = _base_rope_cmd(args.runner, root)
    cmd.extend(["--rope-theta", "nan"])
    _run_reject_case(
        cmd,
        name="rope-theta-nan",
        expected_error="--rope-theta must be a finite number",
    )

    cmd = _base_rope_batch_cmd(args.runner, root)
    cmd[cmd.index("--start-position") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="rope-batch-start-suffix",
        expected_error="--start-position must be an integer",
    )

    cmd = _base_router_cmd(args.runner, root)
    cmd[cmd.index("--top-k") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="router-top-k-suffix",
        expected_error="--top-k must be an integer",
    )

    cmd = _base_router_cmd(args.runner, root)
    cmd[cmd.index("--max-router-mib") + 1] = "1.5"
    cmd.extend(["--router-n-group", "0"])
    _run_reject_case(
        cmd,
        name="router-fractional-mib-group-zero",
        expected_error="--router-n-group must be in 1..",
    )

    cmd = _base_router_batch_cmd(args.runner, root)
    cmd[cmd.index("--batch-tokens") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="router-batch-tokens-suffix",
        expected_error="--batch-tokens must be an integer",
    )

    cmd = _base_final_logits_cmd(args.runner, root)
    cmd[cmd.index("--max-chunk-mib") + 1] = "nan"
    _run_reject_case(
        cmd,
        name="final-logits-chunk-nan",
        expected_error="--max-chunk-mib must be a finite number",
    )

    cmd = _base_final_logits_cmd(args.runner, root)
    cmd[cmd.index("--max-chunk-mib") + 1] = "0"
    _run_reject_case(
        cmd,
        name="final-logits-chunk-zero",
        expected_error="--max-chunk-mib must be positive and within range",
    )

    cmd = _base_final_logits_cmd(args.runner, root)
    cmd.extend(["--chunk-rows", "1x"])
    _run_reject_case(
        cmd,
        name="final-logits-chunk-rows-suffix",
        expected_error="--chunk-rows must be an integer",
    )

    cmd = _base_final_logits_cmd(args.runner, root)
    cmd.extend(["--rms-norm-eps", "nan"])
    _run_reject_case(
        cmd,
        name="final-logits-rms-nan",
        expected_error="--rms-norm-eps must be a finite number",
    )

    cmd = _base_attn_projections_cmd(args.runner, root)
    cmd.extend(["--rms-norm-eps", "nan"])
    _run_reject_case(
        cmd,
        name="attn-projections-rms-nan",
        expected_error="--rms-norm-eps must be a finite number",
    )

    cmd = _base_attn_projections_cmd(args.runner, root)
    cmd.extend(
        [
            "--cache-layout",
            str(root / "missing-cache-layout.json"),
            "--cache-file",
            str(root / "missing-cache.bin"),
            "--position",
            "1x",
        ]
    )
    _run_reject_case(
        cmd,
        name="attn-projections-position-suffix",
        expected_error="--position must be an integer",
    )

    cmd = _base_dense_mlp_cmd(args.runner, root)
    cmd[cmd.index("--max-resident-matrix-mib") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="dense-mlp-matrix-suffix",
        expected_error="--max-resident-matrix-mib must be an integer",
    )

    cmd = _base_dense_mlp_cmd(args.runner, root)
    cmd.extend(["--rms-norm-eps", "nan"])
    _run_reject_case(
        cmd,
        name="dense-mlp-rms-nan",
        expected_error="--rms-norm-eps must be a finite number",
    )

    cmd = _base_resident_linear_cmd(args.runner, root)
    cmd[cmd.index("--layer") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="resident-linear-layer-suffix",
        expected_error="--layer must be an integer",
    )

    cmd = _base_resident_linear_cmd(args.runner, root)
    cmd[cmd.index("--max-resident-matrix-mib") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="resident-linear-matrix-suffix",
        expected_error="--max-resident-matrix-mib must be an integer",
    )

    cmd = _base_rmsnorm_batch_cmd(args.runner, root)
    cmd[cmd.index("--batch-tokens") + 1] = "1x"
    _run_reject_case(
        cmd,
        name="rmsnorm-batch-tokens-suffix",
        expected_error="--batch-tokens must be an integer",
    )

    cmd = _base_rmsnorm_batch_cmd(args.runner, root)
    cmd.extend(["--rms-norm-eps", "nan"])
    _run_reject_case(
        cmd,
        name="rmsnorm-batch-rms-nan",
        expected_error="--rms-norm-eps must be a finite number",
    )

    cmd = _base_resident_linear_batch_cmd(args.runner, root)
    cmd[cmd.index("--max-runner-scratch-mib") + 1] = "64x"
    _run_reject_case(
        cmd,
        name="resident-linear-batch-scratch-suffix",
        expected_error="--max-runner-scratch-mib must be an integer",
    )

    print("  runner CLI contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
