from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .metal_generate import (
    MetalGenerateServerSession,
    MetalTokenGenerationResult,
    generate_metal_token_ids,
)
from .prepared import PreparedManifestError, load_prepared_manifest
from .tokenizer import TokenizerError, load_tokenizer


class MetalTextGenerationError(RuntimeError):
    """Raised when Metal text generation cannot be prepared safely."""


@dataclass(frozen=True)
class MetalTextGenerationResult:
    prompt: str
    generated_text: str
    full_text: str
    tokenizer_backend: str
    tokenizer_path: Path
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    token_result: MetalTokenGenerationResult


@dataclass(frozen=True)
class MetalTextGenerationBatchResult:
    prepared_dir: Path
    prompts: tuple[str, ...]
    results: tuple[MetalTextGenerationResult, ...]
    tokenizer_backend: str
    tokenizer_path: Path
    runtime_request_count: int
    elapsed_seconds: float
    total_prompt_tokens: int
    total_generated_tokens: int
    generated_tokens_per_second: float | None
    max_estimated_live_working_set_bytes: int
    all_admission_ok: bool | None
    all_available_unified_memory_ok: bool | None
    runtime_startup_elapsed_seconds: float | None = None
    max_prompt_prefill_estimated_live_working_set_bytes: int | None = None
    min_system_available_memory_bytes: int | None = None
    max_required_available_memory_bytes: int | None = None
    max_expert_buffer_count_allocated: int | None = None


def _default_tokenizer_path(prepared_dir: str | Path) -> Path:
    prepared = Path(prepared_dir)
    manifest_path = prepared / "manifest.json"
    if not manifest_path.exists():
        raise MetalTextGenerationError(f"prepared manifest not found: {manifest_path}")
    try:
        return load_prepared_manifest(manifest_path).model_dir
    except PreparedManifestError as exc:
        raise MetalTextGenerationError(
            f"failed to load prepared manifest for tokenizer default: {exc}"
        ) from exc


def _all_optional_true(values: Sequence[bool | None]) -> bool | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return all(present)


def _max_optional_int(values: Sequence[int | None]) -> int | None:
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


def _min_optional_int(values: Sequence[int | None]) -> int | None:
    present = [int(value) for value in values if value is not None]
    return min(present) if present else None


def _generate_metal_text_with_tokenizer(
    *,
    prepared_dir: str | Path,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    max_prompt_tokens: int | None,
    generation_kwargs: dict[str, Any],
) -> MetalTextGenerationResult:
    try:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    except TokenizerError:
        raise
    except Exception as exc:  # pragma: no cover - optional tokenizer backends
        raise MetalTextGenerationError(
            f"tokenizer failed to encode prompt: {exc}"
        ) from exc
    if not prompt_ids:
        raise MetalTextGenerationError("prompt encoded to zero tokens")
    if max_prompt_tokens is not None and len(prompt_ids) > max_prompt_tokens:
        raise MetalTextGenerationError(
            f"prompt token length {len(prompt_ids)} exceeds limit {max_prompt_tokens}"
        )

    if (
        len(prompt_ids) > 1
        and not generation_kwargs.get("prefill_prompt")
        and not generation_kwargs.get("allow_decode_only_multi_token_prompt")
    ):
        generation_kwargs["prefill_prompt"] = True

    token_result = generate_metal_token_ids(
        prepared_dir,
        prompt_token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        **generation_kwargs,
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
        raise MetalTextGenerationError(
            f"tokenizer failed to decode generated ids: {exc}"
        ) from exc

    return MetalTextGenerationResult(
        prompt=prompt,
        generated_text=generated_text,
        full_text=full_text,
        tokenizer_backend=tokenizer.backend,
        tokenizer_path=tokenizer.path,
        prompt_token_ids=tuple(prompt_ids),
        generated_token_ids=generated_ids,
        token_result=token_result,
    )


def generate_metal_text(
    *,
    prepared_dir: str | Path,
    tokenizer_path: str | Path | None = None,
    prompt: str,
    max_new_tokens: int,
    tokenizer_backend: str = "auto",
    trust_remote_code: bool = False,
    add_special_tokens: bool = True,
    skip_special_tokens: bool = True,
    max_prompt_tokens: int | None = 4096,
    generate_server_session: MetalGenerateServerSession | None = None,
    **metal_generation_kwargs: Any,
) -> MetalTextGenerationResult:
    resolved_tokenizer = (
        Path(tokenizer_path)
        if tokenizer_path is not None
        else _default_tokenizer_path(prepared_dir)
    )
    tokenizer = load_tokenizer(
        resolved_tokenizer,
        backend=tokenizer_backend,
        trust_remote_code=trust_remote_code,
    )
    generation_kwargs = dict(metal_generation_kwargs)
    if generate_server_session is not None:
        generation_kwargs["generate_server_session"] = generate_server_session

    return _generate_metal_text_with_tokenizer(
        prepared_dir=prepared_dir,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        add_special_tokens=add_special_tokens,
        skip_special_tokens=skip_special_tokens,
        max_prompt_tokens=max_prompt_tokens,
        generation_kwargs=generation_kwargs,
    )


class MetalTextGenerationSession:
    """Reusable tokenizer plus persistent glm_moe_infer JSONL session."""

    def __init__(
        self,
        *,
        prepared_dir: str | Path,
        binary: str | Path = "metal/glm_moe_infer",
        expert_pin_plan: str | Path | None = None,
        max_adaptive_expert_cache_gib: float = 0.0,
        tokenizer_path: str | Path | None = None,
        tokenizer_backend: str = "auto",
        trust_remote_code: bool = False,
        quiet: bool = True,
    ) -> None:
        self.prepared_dir = Path(prepared_dir)
        self.binary = Path(binary)
        self.expert_pin_plan = expert_pin_plan
        self.max_adaptive_expert_cache_gib = float(
            max_adaptive_expert_cache_gib
        )
        resolved_tokenizer = (
            Path(tokenizer_path)
            if tokenizer_path is not None
            else _default_tokenizer_path(self.prepared_dir)
        )
        self.tokenizer = load_tokenizer(
            resolved_tokenizer,
            backend=tokenizer_backend,
            trust_remote_code=trust_remote_code,
        )
        runtime_session_kwargs: dict[str, object] = {
            "binary": self.binary,
            "prepared_dir": self.prepared_dir,
            "quiet": quiet,
        }
        if self.expert_pin_plan is not None:
            runtime_session_kwargs["expert_pin_plan"] = self.expert_pin_plan
        if self.max_adaptive_expert_cache_gib > 0.0:
            runtime_session_kwargs["max_adaptive_expert_cache_gib"] = (
                self.max_adaptive_expert_cache_gib
            )
        self.generate_server_session = MetalGenerateServerSession(
            **runtime_session_kwargs,
        )

    @property
    def request_count(self) -> int:
        return int(self.generate_server_session.request_count)

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        add_special_tokens: bool = True,
        skip_special_tokens: bool = True,
        max_prompt_tokens: int | None = 4096,
        **metal_generation_kwargs: Any,
    ) -> MetalTextGenerationResult:
        if "generate_server_session" in metal_generation_kwargs:
            raise MetalTextGenerationError(
                "MetalTextGenerationSession owns generate_server_session"
            )
        generation_kwargs = dict(metal_generation_kwargs)
        generation_kwargs.setdefault("binary", self.binary)
        generation_kwargs.setdefault("expert_pin_plan", self.expert_pin_plan)
        generation_kwargs.setdefault(
            "max_adaptive_expert_cache_gib",
            self.max_adaptive_expert_cache_gib,
        )
        generation_kwargs["generate_server_session"] = self.generate_server_session
        return _generate_metal_text_with_tokenizer(
            prepared_dir=self.prepared_dir,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            add_special_tokens=add_special_tokens,
            skip_special_tokens=skip_special_tokens,
            max_prompt_tokens=max_prompt_tokens,
            generation_kwargs=generation_kwargs,
        )

    def close(self) -> None:
        self.generate_server_session.close()

    def __enter__(self) -> "MetalTextGenerationSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def generate_metal_text_batch(
    *,
    prepared_dir: str | Path,
    prompts: Sequence[str],
    max_new_tokens: int,
    binary: str | Path = "metal/glm_moe_infer",
    tokenizer_path: str | Path | None = None,
    tokenizer_backend: str = "auto",
    trust_remote_code: bool = False,
    add_special_tokens: bool = True,
    skip_special_tokens: bool = True,
    max_prompt_tokens: int | None = 4096,
    quiet: bool = True,
    **metal_generation_kwargs: Any,
) -> MetalTextGenerationBatchResult:
    prompt_tuple = tuple(str(prompt) for prompt in prompts)
    if not prompt_tuple:
        raise MetalTextGenerationError("prompts must not be empty")
    if any(prompt == "" for prompt in prompt_tuple):
        raise MetalTextGenerationError("prompts must not contain empty strings")
    if "generate_server_session" in metal_generation_kwargs:
        raise MetalTextGenerationError(
            "generate_metal_text_batch owns generate_server_session"
        )
    if len(prompt_tuple) > 1 and metal_generation_kwargs.get("work_dir") is not None:
        raise MetalTextGenerationError(
            "generate_metal_text_batch cannot reuse one work_dir for multiple prompts"
        )

    started = time.monotonic()
    with MetalTextGenerationSession(
        prepared_dir=prepared_dir,
        binary=binary,
        expert_pin_plan=metal_generation_kwargs.get("expert_pin_plan"),
        max_adaptive_expert_cache_gib=float(
            metal_generation_kwargs.get("max_adaptive_expert_cache_gib", 0.0)
        ),
        tokenizer_path=tokenizer_path,
        tokenizer_backend=tokenizer_backend,
        trust_remote_code=trust_remote_code,
        quiet=quiet,
    ) as session:
        results = tuple(
            session.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                add_special_tokens=add_special_tokens,
                skip_special_tokens=skip_special_tokens,
                max_prompt_tokens=max_prompt_tokens,
                **metal_generation_kwargs,
            )
            for prompt in prompt_tuple
        )
        startup_elapsed = getattr(
            session.generate_server_session,
            "startup_elapsed_seconds",
            None,
        )
        runtime_request_count = session.request_count
        tokenizer = session.tokenizer
    elapsed = time.monotonic() - started
    total_prompt_tokens = sum(len(item.prompt_token_ids) for item in results)
    total_generated_tokens = sum(len(item.generated_token_ids) for item in results)
    token_results = tuple(item.token_result for item in results)

    return MetalTextGenerationBatchResult(
        prepared_dir=Path(prepared_dir),
        prompts=prompt_tuple,
        results=results,
        tokenizer_backend=tokenizer.backend,
        tokenizer_path=tokenizer.path,
        runtime_request_count=runtime_request_count,
        elapsed_seconds=elapsed,
        total_prompt_tokens=total_prompt_tokens,
        total_generated_tokens=total_generated_tokens,
        generated_tokens_per_second=(
            total_generated_tokens / elapsed if elapsed > 0.0 else None
        ),
        max_estimated_live_working_set_bytes=max(
            item.estimated_live_working_set_bytes for item in token_results
        ),
        all_admission_ok=_all_optional_true(
            tuple(item.admission_ok for item in token_results)
        ),
        all_available_unified_memory_ok=_all_optional_true(
            tuple(item.available_unified_memory_ok for item in token_results)
        ),
        runtime_startup_elapsed_seconds=(
            float(startup_elapsed) if startup_elapsed is not None else None
        ),
        max_prompt_prefill_estimated_live_working_set_bytes=_max_optional_int(
            tuple(
                item.prompt_prefill_estimated_live_working_set_bytes
                for item in token_results
            )
        ),
        min_system_available_memory_bytes=_min_optional_int(
            tuple(item.system_available_memory_bytes for item in token_results)
        ),
        max_required_available_memory_bytes=_max_optional_int(
            tuple(item.required_available_memory_bytes for item in token_results)
        ),
        max_expert_buffer_count_allocated=_max_optional_int(
            tuple(item.expert_buffer_count_allocated for item in token_results)
        ),
    )
