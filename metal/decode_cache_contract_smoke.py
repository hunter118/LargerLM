#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


Mutator = Callable[[dict[str, object], Path], None]


def _base_layout() -> dict[str, object]:
    return {
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
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    layout_path = root / "cache_layout.json"
    cache_path = root / "decode_cache.bin"
    layout_path.write_text(json.dumps(_base_layout(), indent=2), encoding="utf-8")
    cache_path.write_bytes(b"\0" * 16)
    return layout_path, cache_path


def _run_expect(
    *,
    runner: Path,
    layout_path: Path,
    cache_path: Path,
    expect_ok: bool,
    expected_error: str = "",
) -> None:
    cmd = [
        str(runner),
        "--validate-cache-backing",
        "--cache-layout",
        str(layout_path),
        "--cache-file",
        str(cache_path),
        "--max-cache-file-mib",
        "1",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    combined = completed.stdout + completed.stderr
    if expect_ok:
        if completed.returncode != 0:
            raise SystemExit(f"valid cache backing was rejected:\n{combined}")
        return
    if completed.returncode == 0:
        raise SystemExit("runner accepted invalid decode cache backing")
    if expected_error and expected_error not in combined:
        raise SystemExit(
            f"expected error {expected_error!r}; got:\n{combined}"
        )
    if "Metal device unavailable" in combined:
        raise SystemExit(f"cache contract was checked after Metal init:\n{combined}")


def _run_reject_case(
    *,
    runner: Path,
    root: Path,
    name: str,
    mutator: Mutator,
    expected_error: str,
) -> None:
    case_root = root / name
    layout_path, cache_path = _write_fixture(case_root)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    mutator(layout, cache_path)
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    _run_expect(
        runner=runner,
        layout_path=layout_path,
        cache_path=cache_path,
        expect_ok=False,
        expected_error=expected_error,
    )
    print(f"  rejected {name}: {expected_error}")


def _use_boolean_total_bytes(layout: dict[str, object], _cache_path: Path) -> None:
    layout["total_bytes"] = True


def _use_unsupported_version(layout: dict[str, object], _cache_path: Path) -> None:
    layout["version"] = 2


def _use_boolean_segment_offset(layout: dict[str, object], _cache_path: Path) -> None:
    segment = list(layout["segments"])[0]
    assert isinstance(segment, dict)
    segment["offset"] = False


def _break_segment_stride(layout: dict[str, object], _cache_path: Path) -> None:
    segment = list(layout["segments"])[0]
    assert isinstance(segment, dict)
    segment["token_stride_bytes"] = 6


def _overflow_segment(layout: dict[str, object], _cache_path: Path) -> None:
    segment = list(layout["segments"])[0]
    assert isinstance(segment, dict)
    segment["offset"] = 8


def _duplicate_segment(layout: dict[str, object], _cache_path: Path) -> None:
    segments = layout["segments"]
    assert isinstance(segments, list)
    segments.append(dict(segments[0]))


def _truncate_cache_file(_layout: dict[str, object], cache_path: Path) -> None:
    cache_path.write_bytes(b"\0" * 8)


def _run_entrypoint_preflight_case(
    *,
    runner: Path,
    root: Path,
    name: str,
    extra_args: list[str],
) -> None:
    case_root = root / f"entry-{name}"
    layout_path, cache_path = _write_fixture(case_root)
    cache_path.write_bytes(b"\0" * 8)
    indices_path = case_root / "indices.u32"
    indices_path.write_bytes(b"\1\0\0\0\0\0\0\0")
    cmd = [
        str(runner),
        "--cache-layout",
        str(layout_path),
        "--cache-file",
        str(cache_path),
        *extra_args,
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    combined = completed.stdout + completed.stderr
    if completed.returncode == 0:
        raise SystemExit(f"{name}: runner accepted invalid cache backing")
    if "decode cache file size" not in combined:
        raise SystemExit(
            f"{name}: expected cache backing error before other work; got:\n{combined}"
        )
    if "Metal device unavailable" in combined:
        raise SystemExit(f"{name}: cache backing checked after Metal init:\n{combined}")
    print(f"  preflighted {name}: decode cache file size")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify runner-side decode cache layout and backing-file contracts.",
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
        tempfile.mkdtemp(prefix="largerlm-decode-cache-contract-", dir="/private/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    print(f"fixture: {root}")
    valid_layout, valid_cache = _write_fixture(root / "valid")
    _run_expect(
        runner=args.runner,
        layout_path=valid_layout,
        cache_path=valid_cache,
        expect_ok=True,
    )
    print("  accepted valid backing")
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="total-bytes-type",
        mutator=_use_boolean_total_bytes,
        expected_error="cache layout total_bytes must be a positive integer",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="unsupported-version",
        mutator=_use_unsupported_version,
        expected_error="unsupported cache layout version",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="segment-offset-type",
        mutator=_use_boolean_segment_offset,
        expected_error="cache segment offset must be a non-negative integer",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="segment-stride",
        mutator=_break_segment_stride,
        expected_error="cache segment token_stride_bytes does not match width*dtype_bytes",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="segment-overflow",
        mutator=_overflow_segment,
        expected_error="cache segment exceeds layout total_bytes",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="duplicate-segment",
        mutator=_duplicate_segment,
        expected_error="duplicate cache segment",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="cache-file-size",
        mutator=_truncate_cache_file,
        expected_error="decode cache file size",
    )
    common_attention = [
        "--resident-layout",
        str(root / "missing-resident-layout.json"),
        "--layer",
        "1",
        "--q-nope-f32",
        str(root / "missing-q-nope.f32"),
        "--q-rope-f32",
        str(root / "missing-q-rope.f32"),
        "--context-length",
        "1",
        "--num-heads",
        "1",
        "--qk-nope-dim",
        "1",
        "--rope-dim",
        "2",
        "--v-head-dim",
        "1",
        "--output-f32",
        str(root / "out.f32"),
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]
    _run_entrypoint_preflight_case(
        runner=args.runner,
        root=root,
        name="mla-attention",
        extra_args=["--run-mla-attention", *common_attention],
    )
    _run_entrypoint_preflight_case(
        runner=args.runner,
        root=root,
        name="mla-attention-batch",
        extra_args=[
            "--run-mla-attention-batch",
            *common_attention,
            "--start-position",
            "0",
            "--batch-tokens",
            "1",
        ],
    )
    _run_entrypoint_preflight_case(
        runner=args.runner,
        root=root,
        name="mla-attention-indexed-batch",
        extra_args=[
            "--run-mla-attention-indexed-batch",
            *common_attention,
            "--batch-tokens",
            "1",
            "--index-topk",
            "1",
            "--indices-u32",
            str(root / "entry-mla-attention-indexed-batch" / "indices.u32"),
        ],
    )
    _run_entrypoint_preflight_case(
        runner=args.runner,
        root=root,
        name="attn-projections",
        extra_args=[
            "--resident-layout",
            str(root / "missing-resident-layout.json"),
            "--layer",
            "1",
            "--run-attn-projections",
            "--input-f32",
            str(root / "missing-input.f32"),
            "--position",
            "0",
            "--max-cache-file-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
        ],
    )
    decoder_args = [
        "--resident-layout",
        str(root / "missing-resident-layout.json"),
        "--layer",
        "1",
        "--input-f32",
        str(root / "missing-input.f32"),
        "--position",
        "0",
        "--context-length",
        "1",
        "--num-heads",
        "1",
        "--qk-nope-dim",
        "1",
        "--rope-dim",
        "2",
        "--v-head-dim",
        "1",
        "--output-f32",
        str(root / "out.f32"),
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]
    _run_entrypoint_preflight_case(
        runner=args.runner,
        root=root,
        name="decoder-layer",
        extra_args=[
            "--layout",
            str(root / "missing-expert-layout.json"),
            "--run-decoder-layer",
            *decoder_args,
            "--max-router-mib",
            "1",
            "--max-slot-mib",
            "1",
        ],
    )
    _run_entrypoint_preflight_case(
        runner=args.runner,
        root=root,
        name="dense-decoder-layer",
        extra_args=["--run-dense-decoder-layer", *decoder_args],
    )
    print("  contract smoke:     ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
