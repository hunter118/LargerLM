from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .baseline import BaselineTensor, BaselineWriter
from .config import load_config
from .safetensors import CheckpointStats, SafetensorsError, scan_checkpoint


class MlxBaselineError(RuntimeError):
    """Raised when MLX baseline export cannot run safely."""


@dataclass(frozen=True)
class MlxBaselinePreflight:
    model_dir: Path
    output_dir: Path
    model_type: str
    checkpoint_total_bytes: int | None
    max_model_load_bytes: int
    safe_to_load: bool
    reason: str


@dataclass(frozen=True)
class MlxBaselineResult:
    output_dir: Path
    generated_tokens: tuple[int, ...]
    tensor_count: int


def preflight_mlx_baseline(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    max_model_load_bytes: int,
    allow_unknown_size: bool = False,
) -> MlxBaselinePreflight:
    root = Path(model_dir)
    cfg = load_config(root)
    total: int | None = None
    try:
        stats = scan_checkpoint(root)
        total = stats.total_bytes
    except SafetensorsError:
        if not allow_unknown_size:
            return MlxBaselinePreflight(
                model_dir=root,
                output_dir=Path(output_dir),
                model_type=cfg.model_type,
                checkpoint_total_bytes=None,
                max_model_load_bytes=max_model_load_bytes,
                safe_to_load=False,
                reason="checkpoint size unknown; pass allow_unknown_size only for tiny test models",
            )

    if total is not None and total > max_model_load_bytes:
        return MlxBaselinePreflight(
            model_dir=root,
            output_dir=Path(output_dir),
            model_type=cfg.model_type,
            checkpoint_total_bytes=total,
            max_model_load_bytes=max_model_load_bytes,
            safe_to_load=False,
            reason="checkpoint total bytes exceed safe MLX whole-model load limit",
        )

    return MlxBaselinePreflight(
        model_dir=root,
        output_dir=Path(output_dir),
        model_type=cfg.model_type,
        checkpoint_total_bytes=total,
        max_model_load_bytes=max_model_load_bytes,
        safe_to_load=True,
        reason="safe preflight passed",
    )


def _load_mlx_stack() -> tuple[Any, Callable[[str], tuple[Any, Any]]]:
    try:
        import mlx.core as mx
        from mlx_lm import load
    except Exception as exc:
        raise MlxBaselineError(
            "mlx and mlx_lm are required for --execute; install them in the "
            "runtime environment before exporting an MLX baseline"
        ) from exc
    return mx, load


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(prompt)
    else:
        raise MlxBaselineError("tokenizer does not expose encode(prompt)")
    return [int(t) for t in tokens]


def _call_model(model: Any, input_ids: Any, cache: Any | None) -> Any:
    try:
        return model(input_ids, cache=cache)
    except TypeError:
        return model(input_ids)


def _extract_logits(result: Any) -> Any:
    if isinstance(result, tuple):
        return result[0]
    return result


def _last_token_logits(logits: Any) -> Any:
    try:
        return logits[:, -1, :]
    except Exception:
        return logits


def _array_to_bytes(array: Any) -> tuple[bytes, str, tuple[int, ...]]:
    if all(hasattr(array, attr) for attr in ("tobytes", "shape", "dtype")):
        return (
            array.tobytes(),
            str(array.dtype),
            tuple(int(x) for x in array.shape),
        )

    try:
        import numpy as np
    except Exception as exc:
        raise MlxBaselineError("numpy is required to materialize baseline tensors") from exc

    host = np.asarray(array)
    contiguous = np.ascontiguousarray(host)
    return contiguous.tobytes(), str(contiguous.dtype), tuple(int(x) for x in contiguous.shape)


def _argmax_token(mx: Any, logits: Any) -> int:
    token = mx.argmax(logits, axis=-1)
    try:
        return int(token.item())
    except AttributeError:
        return int(token)


def export_mlx_baseline(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    prompt: str,
    max_tokens: int = 1,
    max_prompt_tokens: int = 4096,
    max_tensor_bytes: int = 32 * 1024**2,
    max_model_load_bytes: int = 64 * 1024**3,
    allow_unknown_size: bool = False,
    mx_module: Any | None = None,
    load_fn: Callable[[str], tuple[Any, Any]] | None = None,
) -> MlxBaselineResult:
    if max_tokens < 1:
        raise MlxBaselineError("max_tokens must be positive")
    if max_tokens > 8:
        raise MlxBaselineError("refusing to export more than 8 tokens per baseline")

    preflight = preflight_mlx_baseline(
        model_dir,
        output_dir,
        max_model_load_bytes=max_model_load_bytes,
        allow_unknown_size=allow_unknown_size,
    )
    if not preflight.safe_to_load:
        raise MlxBaselineError(preflight.reason)

    mx = mx_module
    loader = load_fn
    if mx is None or loader is None:
        mx, loader = _load_mlx_stack()

    model, tokenizer = loader(str(model_dir))
    prompt_tokens = _encode_prompt(tokenizer, prompt)
    if len(prompt_tokens) > max_prompt_tokens:
        raise MlxBaselineError(
            f"prompt has {len(prompt_tokens)} tokens, exceeding limit {max_prompt_tokens}"
        )

    cache = model.make_cache() if hasattr(model, "make_cache") else None
    generated: list[int] = []
    tensor_count = 0
    cfg = load_config(model_dir)

    with BaselineWriter(
        output_dir,
        model_type=cfg.model_type,
        prompt_tokens=prompt_tokens,
        metadata={
            "source": "mlx_lm",
            "max_tokens": max_tokens,
            "checkpoint_total_bytes": preflight.checkpoint_total_bytes,
        },
    ) as writer:
        input_ids = mx.array([prompt_tokens])
        for token_index in range(max_tokens):
            result = _call_model(model, input_ids, cache)
            logits = _last_token_logits(_extract_logits(result))
            try:
                mx.eval(logits)
            except Exception:
                pass

            data, dtype, shape = _array_to_bytes(logits)
            tensor: BaselineTensor = writer.write_tensor(
                name=f"token.{token_index}.last_logits",
                dtype=dtype,
                shape=shape,
                data=data,
                max_bytes=max_tensor_bytes,
            )
            tensor_count += 1
            next_token = _argmax_token(mx, logits)
            writer.add_generated_token(next_token)
            writer.add_record(
                kind="decode_logits",
                token_index=token_index,
                layer=None,
                tensors=[tensor],
                metadata={"next_token": next_token},
            )
            generated.append(next_token)
            input_ids = mx.array([[next_token]])

    return MlxBaselineResult(
        output_dir=Path(output_dir),
        generated_tokens=tuple(generated),
        tensor_count=tensor_count,
    )
