#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-embedding-batch-", dir="/private/tmp"))
    resident = root / "resident"
    resident.mkdir()
    payload = struct.pack("<12f", 1, 2, 3, 4, 5, 6, -1, -2, -3, 7, 8, 9)
    (resident / "resident.bin").write_bytes(payload)
    layout = resident / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": [
                    {
                        "name": "model.embed_tokens.weight",
                        "offset": 0,
                        "size": len(payload),
                        "dtype": "F32",
                        "shape": [4, 3],
                        "category": "embeddings",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    token_ids = root / "tokens.json"
    token_ids.write_text(json.dumps([3, 0, 2]), encoding="utf-8")
    output = root / "prompt.f32"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "largerlm",
            "embed-tokens-batch",
            str(layout),
            "--token-ids-file",
            str(token_ids),
            "--output-f32",
            str(output),
            "--max-output-mib",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        return completed.returncode
    payload_json = json.loads(completed.stdout)
    if (
        payload_json["token_count"] != 3
        or payload_json["hidden_dim"] != 3
        or payload_json["read_bytes"] != 36
        or payload_json["output_bytes"] != 36
    ):
        raise SystemExit(f"unexpected embedding payload: {payload_json}")
    values = struct.unpack("<9f", output.read_bytes())
    expected = (7.0, 8.0, 9.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0)
    if values != expected:
        raise SystemExit(f"embedding values {values} != {expected}")
    print(f"fixture: {root}")
    print("  embedding batch: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
