from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TokenizerError(RuntimeError):
    """Raised when a local tokenizer cannot be loaded or used."""


@dataclass(frozen=True)
class EncodedPrompt:
    token_ids: tuple[int, ...]
    backend: str
    tokenizer_path: Path


@dataclass(frozen=True)
class RenderedChatPrompt:
    text: str
    backend: str
    tokenizer_path: Path
    tokenizer_backend: str | None = None

    def __post_init__(self) -> None:
        if self.tokenizer_backend is None:
            object.__setattr__(self, "tokenizer_backend", self.backend)


class LocalTokenizer:
    backend: str
    path: Path
    eos_token_id: int | None
    bos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = True) -> tuple[int, ...]:
        raise NotImplementedError

    def decode(
        self,
        token_ids: tuple[int, ...] | list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        raise NotImplementedError

    def render_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        raise TokenizerError(
            f"tokenizer backend {self.backend} does not expose a chat template"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TokenizerError(f"failed to read tokenizer JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TokenizerError(f"failed to parse tokenizer JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TokenizerError(f"tokenizer JSON {path} must be an object")
    return payload


def _token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def _special_from_config(root: Path, name: str) -> str | None:
    candidates = []
    if root.is_dir():
        candidates.extend([root / "tokenizer_config.json", root / "special_tokens_map.json"])
    for path in candidates:
        if not path.exists():
            continue
        payload = _read_json(path)
        token = _token_content(payload.get(name))
        if token is not None:
            return token
    return None


class SimpleJsonTokenizer(LocalTokenizer):
    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            raise TokenizerError("simple tokenizer JSON missing tokens object")
        vocab: dict[str, int] = {}
        for token, token_id in tokens.items():
            if not isinstance(token, str) or not isinstance(token_id, int):
                raise TokenizerError("simple tokenizer tokens must map strings to ints")
            if token_id < 0:
                raise TokenizerError("simple tokenizer token ids must be non-negative")
            vocab[token] = token_id
        if not vocab:
            raise TokenizerError("simple tokenizer vocab must not be empty")
        inverse: dict[int, str] = {}
        for token, token_id in vocab.items():
            if token_id in inverse:
                raise TokenizerError(f"duplicate simple tokenizer id {token_id}")
            inverse[token_id] = token

        split = str(payload.get("split") or "characters")
        if split not in {"characters", "whitespace"}:
            raise TokenizerError("simple tokenizer split must be characters or whitespace")

        self.backend = "simple"
        self.path = path
        self._vocab = vocab
        self._inverse = inverse
        self._split = split
        self._join = "" if split == "characters" else " "
        self._add_bos = bool(payload.get("add_bos_on_encode", False))
        self._add_eos = bool(payload.get("add_eos_on_encode", False))
        self.bos_token_id = self._resolve_special(payload, "bos")
        self.eos_token_id = self._resolve_special(payload, "eos")
        self._skip_ids = {
            token_id
            for token_id in (
                self.bos_token_id,
                self.eos_token_id,
                self._resolve_special(payload, "pad"),
            )
            if token_id is not None
        }

    def _resolve_special(self, payload: dict[str, Any], name: str) -> int | None:
        raw_id = payload.get(f"{name}_token_id")
        if isinstance(raw_id, int) and raw_id >= 0:
            return raw_id
        raw_token = payload.get(f"{name}_token")
        if isinstance(raw_token, str) and raw_token in self._vocab:
            return self._vocab[raw_token]
        return None

    def _pieces(self, text: str) -> list[str]:
        if self._split == "characters":
            return list(text)
        return text.split()

    def encode(self, text: str, *, add_special_tokens: bool = True) -> tuple[int, ...]:
        ids: list[int] = []
        if add_special_tokens and self._add_bos and self.bos_token_id is not None:
            ids.append(self.bos_token_id)
        for piece in self._pieces(text):
            try:
                ids.append(self._vocab[piece])
            except KeyError as exc:
                raise TokenizerError(f"simple tokenizer has no token for {piece!r}") from exc
        if add_special_tokens and self._add_eos and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return tuple(ids)

    def decode(
        self,
        token_ids: tuple[int, ...] | list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            tid = int(token_id)
            if skip_special_tokens and tid in self._skip_ids:
                continue
            try:
                pieces.append(self._inverse[tid])
            except KeyError as exc:
                raise TokenizerError(f"simple tokenizer has no id {tid}") from exc
        return self._join.join(pieces)


class TokenizersAdapter(LocalTokenizer):
    def __init__(self, root: Path, tokenizer_json: Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise TokenizerError(
                "tokenizers package is not installed; install it or choose another backend"
            ) from exc
        try:
            self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise TokenizerError(f"failed to load tokenizer.json {tokenizer_json}: {exc}") from exc
        self.backend = "tokenizers"
        self.path = root
        eos = _special_from_config(root, "eos_token")
        bos = _special_from_config(root, "bos_token")
        self.eos_token_id = self._tokenizer.token_to_id(eos) if eos else None
        self.bos_token_id = self._tokenizer.token_to_id(bos) if bos else None

    def encode(self, text: str, *, add_special_tokens: bool = True) -> tuple[int, ...]:
        encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return tuple(int(token_id) for token_id in encoded.ids)

    def decode(
        self,
        token_ids: tuple[int, ...] | list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return str(
            self._tokenizer.decode(
                [int(token_id) for token_id in token_ids],
                skip_special_tokens=skip_special_tokens,
            )
        )

class TransformersAdapter(LocalTokenizer):
    def __init__(self, root: Path, *, trust_remote_code: bool = False) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TokenizerError(
                "transformers package is not installed; install it or choose another backend"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise TokenizerError(f"failed to load local AutoTokenizer from {root}: {exc}") from exc
        self.backend = "transformers"
        self.path = root
        self.eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        self.bos_token_id = getattr(self._tokenizer, "bos_token_id", None)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> tuple[int, ...]:
        ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return tuple(int(token_id) for token_id in ids)

    def decode(
        self,
        token_ids: tuple[int, ...] | list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return str(
            self._tokenizer.decode(
                [int(token_id) for token_id in token_ids],
                skip_special_tokens=skip_special_tokens,
            )
        )

    def render_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        renderer = getattr(self._tokenizer, "apply_chat_template", None)
        if renderer is None:
            raise TokenizerError("transformers tokenizer does not expose apply_chat_template")
        try:
            rendered = renderer(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as exc:  # pragma: no cover - optional tokenizer backends
            raise TokenizerError(f"failed to render chat template: {exc}") from exc
        if not isinstance(rendered, str) or not rendered:
            raise TokenizerError("chat template rendered an empty prompt")
        return rendered


class SentencePieceAdapter(LocalTokenizer):
    def __init__(self, root: Path, model_path: Path) -> None:
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise TokenizerError(
                "sentencepiece package is not installed; install it or choose another backend"
            ) from exc
        try:
            self._processor = spm.SentencePieceProcessor(model_file=str(model_path))
        except TypeError:  # pragma: no cover - older sentencepiece API
            self._processor = spm.SentencePieceProcessor()
            self._processor.Load(str(model_path))
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise TokenizerError(f"failed to load SentencePiece model {model_path}: {exc}") from exc
        self.backend = "sentencepiece"
        self.path = root
        eos = int(self._processor.eos_id())
        bos = int(self._processor.bos_id())
        self.eos_token_id = eos if eos >= 0 else None
        self.bos_token_id = bos if bos >= 0 else None

    def encode(self, text: str, *, add_special_tokens: bool = True) -> tuple[int, ...]:
        del add_special_tokens
        ids = self._processor.encode(text, out_type=int)
        return tuple(int(token_id) for token_id in ids)

    def decode(
        self,
        token_ids: tuple[int, ...] | list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if skip_special_tokens:
            ids = [
                token_id
                for token_id in ids
                if token_id not in {self.eos_token_id, self.bos_token_id}
            ]
        return str(self._processor.decode(ids))


def _simple_tokenizer_file(path: Path) -> Path | None:
    candidates = [path] if path.is_file() else [path / "largerlm_tokenizer.json", path / "simple_tokenizer.json"]
    for candidate in candidates:
        if not candidate.exists() or candidate.suffix != ".json":
            continue
        payload = _read_json(candidate)
        if payload.get("type") == "largerlm-simple-vocab" or "tokens" in payload:
            return candidate
    return None


def _tokenizer_json_file(path: Path) -> Path | None:
    if path.is_file() and path.name == "tokenizer.json":
        return path
    candidate = path / "tokenizer.json"
    return candidate if candidate.exists() else None


def _sentencepiece_file(path: Path) -> Path | None:
    if path.is_file() and path.suffix in {".model", ".spm"}:
        return path
    for name in ("tokenizer.model", "spiece.model"):
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def _tokenizer_root(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _local_chat_template_source(path: Path) -> tuple[str, Path] | None:
    root = _tokenizer_root(path)
    template_path = root / "chat_template.jinja"
    if template_path.exists():
        try:
            return template_path.read_text(encoding="utf-8"), template_path
        except OSError as exc:
            raise TokenizerError(f"failed to read chat template {template_path}: {exc}") from exc

    config_path = root / "tokenizer_config.json"
    if config_path.exists():
        payload = _read_json(config_path)
        template = payload.get("chat_template")
        if isinstance(template, str) and template:
            return template, config_path
    return None


def _render_local_jinja_chat_template(
    tokenizer_path: Path,
    messages: list[dict[str, str]],
    *,
    tokenizer_backend: str,
    add_generation_prompt: bool,
) -> RenderedChatPrompt:
    template_source = _local_chat_template_source(tokenizer_path)
    if template_source is None:
        raise TokenizerError(
            f"local chat_template.jinja or tokenizer_config chat_template not found under "
            f"{_tokenizer_root(tokenizer_path)}"
        )
    template, source_path = template_source
    try:
        from jinja2.sandbox import SandboxedEnvironment
    except ImportError as exc:
        raise TokenizerError(
            "jinja2 package is not installed; install it to render local chat templates"
        ) from exc

    def tojson(value: Any, ensure_ascii: bool = False, **_: Any) -> str:
        return json.dumps(value, ensure_ascii=ensure_ascii)

    try:
        env = SandboxedEnvironment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.filters["tojson"] = tojson
        rendered = env.from_string(template).render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as exc:  # pragma: no cover - depends on user-provided templates
        raise TokenizerError(f"failed to render local chat template {source_path}: {exc}") from exc
    if not isinstance(rendered, str) or not rendered:
        raise TokenizerError(f"local chat template {source_path} rendered an empty prompt")
    return RenderedChatPrompt(
        text=rendered,
        backend="local-jinja",
        tokenizer_path=_tokenizer_root(tokenizer_path),
        tokenizer_backend=tokenizer_backend,
    )


def load_tokenizer(
    tokenizer_path: str | Path,
    *,
    backend: str = "auto",
    trust_remote_code: bool = False,
) -> LocalTokenizer:
    root = Path(tokenizer_path)
    if backend not in {"auto", "simple", "tokenizers", "transformers", "sentencepiece"}:
        raise TokenizerError(f"unsupported tokenizer backend {backend}")
    if not root.exists():
        raise TokenizerError(f"tokenizer path does not exist: {root}")

    errors: list[str] = []
    simple_file = _simple_tokenizer_file(root)
    tokenizer_json = _tokenizer_json_file(root)
    sp_model = _sentencepiece_file(root)

    if backend in {"auto", "simple"} and simple_file is not None:
        return SimpleJsonTokenizer(simple_file, _read_json(simple_file))
    if backend == "simple":
        raise TokenizerError(f"simple tokenizer JSON not found under {root}")

    if backend in {"auto", "tokenizers"} and tokenizer_json is not None:
        try:
            return TokenizersAdapter(root, tokenizer_json)
        except TokenizerError as exc:
            if backend == "tokenizers":
                raise
            errors.append(str(exc))

    if backend in {"auto", "transformers"} and root.is_dir():
        try:
            return TransformersAdapter(root, trust_remote_code=trust_remote_code)
        except TokenizerError as exc:
            if backend == "transformers":
                raise
            errors.append(str(exc))

    if backend in {"auto", "sentencepiece"} and sp_model is not None:
        try:
            return SentencePieceAdapter(root, sp_model)
        except TokenizerError as exc:
            if backend == "sentencepiece":
                raise
            errors.append(str(exc))

    detail = "; ".join(errors) if errors else "no supported tokenizer files found"
    raise TokenizerError(
        f"could not load a local tokenizer from {root}: {detail}. "
        "Use a directory with tokenizer.json/tokenizer.model, install the matching "
        "optional package, or pass --tokenizer-backend simple for a test tokenizer."
    )


def encode_prompt(
    tokenizer_path: str | Path,
    prompt: str,
    *,
    backend: str = "auto",
    trust_remote_code: bool = False,
    add_special_tokens: bool = True,
) -> EncodedPrompt:
    tokenizer = load_tokenizer(
        tokenizer_path,
        backend=backend,
        trust_remote_code=trust_remote_code,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if not ids:
        raise TokenizerError("prompt encoded to zero tokens")
    return EncodedPrompt(token_ids=ids, backend=tokenizer.backend, tokenizer_path=tokenizer.path)


def render_chat_prompt(
    tokenizer_path: str | Path,
    messages: list[dict[str, str]],
    *,
    backend: str = "auto",
    trust_remote_code: bool = False,
    add_generation_prompt: bool = True,
) -> RenderedChatPrompt:
    """Render chat messages with a local tokenizer-provided template."""

    root = Path(tokenizer_path)
    errors: list[str] = []
    backends = ("transformers", "auto") if backend == "auto" else (backend,)
    for candidate in backends:
        try:
            tokenizer = load_tokenizer(
                root,
                backend=candidate,
                trust_remote_code=trust_remote_code,
            )
            text = tokenizer.render_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
            )
        except TokenizerError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        return RenderedChatPrompt(
            text=text,
            backend=tokenizer.backend,
            tokenizer_path=tokenizer.path,
        )
    if backend in {"auto", "simple", "tokenizers", "sentencepiece"}:
        try:
            return _render_local_jinja_chat_template(
                root,
                messages,
                tokenizer_backend=backend,
                add_generation_prompt=add_generation_prompt,
            )
        except TokenizerError as exc:
            errors.append(f"local-jinja: {exc}")
    detail = "; ".join(errors) if errors else "no chat tokenizer backend tried"
    raise TokenizerError(f"could not render chat template: {detail}")
