from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .token_generator import TokenGenerationResult, generate_token_ids
from .tokenizer import TokenizerError, load_tokenizer


class TextGenerationError(RuntimeError):
    """Raised when text generation cannot be prepared safely."""


@dataclass(frozen=True)
class TextGenerationResult:
    prompt: str
    generated_text: str
    full_text: str
    tokenizer_backend: str
    tokenizer_path: Path
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    eos_token_id: int | None
    token_result: TokenGenerationResult
    applied_launch_profile: dict[str, object] | None = None


def read_prompt_arg(
    *,
    prompt: str | None,
    prompt_file: str | Path | None,
) -> str:
    if prompt is not None and prompt_file is not None:
        raise TextGenerationError("--prompt and --prompt-file are mutually exclusive")
    if prompt_file is not None:
        path = Path(prompt_file)
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TextGenerationError(f"failed to read prompt file {path}: {exc}") from exc
    if prompt is None:
        raise TextGenerationError("--prompt or --prompt-file is required")
    return prompt


def generate_text(
    *,
    tokenizer_path: str | Path,
    prompt: str,
    max_new_tokens: int,
    tokenizer_backend: str = "auto",
    trust_remote_code: bool = False,
    add_special_tokens: bool = True,
    skip_special_tokens: bool = True,
    max_prompt_tokens: int | None = 4096,
    eos_token_id: int | None = None,
    use_tokenizer_eos: bool = True,
    **token_generation_kwargs: Any,
) -> TextGenerationResult:
    tokenizer = load_tokenizer(
        tokenizer_path,
        backend=tokenizer_backend,
        trust_remote_code=trust_remote_code,
    )
    try:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    except TokenizerError:
        raise
    except Exception as exc:  # pragma: no cover - optional tokenizer backends
        raise TextGenerationError(f"tokenizer failed to encode prompt: {exc}") from exc
    if not prompt_ids:
        raise TextGenerationError("prompt encoded to zero tokens")
    if max_prompt_tokens is not None and len(prompt_ids) > max_prompt_tokens:
        raise TextGenerationError(
            f"prompt token length {len(prompt_ids)} exceeds limit {max_prompt_tokens}"
        )
    auto_batch_prefill = bool(
        token_generation_kwargs.pop("auto_batch_prefill_prompt", False)
    )
    if (
        auto_batch_prefill
        and len(prompt_ids) > 1
        and not bool(token_generation_kwargs.get("batch_prefill_prompt", False))
    ):
        token_generation_kwargs["batch_prefill_prompt"] = True

    effective_eos = eos_token_id
    if effective_eos is None and use_tokenizer_eos:
        effective_eos = tokenizer.eos_token_id

    token_result = generate_token_ids(
        prompt_token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=effective_eos,
        **token_generation_kwargs,
    )
    generated_ids = token_result.generated_token_ids
    try:
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )
        full_text = tokenizer.decode(
            tuple(prompt_ids) + tuple(generated_ids),
            skip_special_tokens=skip_special_tokens,
        )
    except TokenizerError:
        raise
    except Exception as exc:  # pragma: no cover - optional tokenizer backends
        raise TextGenerationError(f"tokenizer failed to decode generated ids: {exc}") from exc

    return TextGenerationResult(
        prompt=prompt,
        generated_text=generated_text,
        full_text=full_text,
        tokenizer_backend=tokenizer.backend,
        tokenizer_path=tokenizer.path,
        prompt_token_ids=tuple(prompt_ids),
        generated_token_ids=generated_ids,
        eos_token_id=effective_eos,
        token_result=token_result,
    )
