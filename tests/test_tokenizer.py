from __future__ import annotations

import json
from pathlib import Path

import pytest

from largerlm.tokenizer import (
    TokenizerError,
    encode_prompt,
    load_tokenizer,
    render_chat_prompt,
)


def write_simple_tokenizer(path: Path) -> Path:
    path.write_text(
        """{
  "type": "largerlm-simple-vocab",
  "split": "characters",
  "tokens": {"A": 0, "B": 1, "C": 2, "<eos>": 3},
  "eos_token": "<eos>"
}
""",
        encoding="utf-8",
    )
    return path


def test_simple_tokenizer_round_trips_characters(tmp_path: Path) -> None:
    tokenizer_path = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")

    tokenizer = load_tokenizer(tokenizer_path, backend="simple")

    assert tokenizer.backend == "simple"
    assert tokenizer.eos_token_id == 3
    assert tokenizer.encode("AB") == (0, 1)
    assert tokenizer.decode((0, 2, 3)) == "AC"
    assert tokenizer.decode((0, 2, 3), skip_special_tokens=False) == "AC<eos>"


def test_encode_prompt_rejects_empty_prompt(tmp_path: Path) -> None:
    tokenizer_path = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")

    with pytest.raises(TokenizerError, match="zero tokens"):
        encode_prompt(tokenizer_path, "", backend="simple")


def test_simple_tokenizer_rejects_unknown_piece(tmp_path: Path) -> None:
    tokenizer_path = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path, backend="simple")

    with pytest.raises(TokenizerError, match="no token"):
        tokenizer.encode("Z")


def test_render_chat_prompt_requires_template_backend(tmp_path: Path) -> None:
    tokenizer_path = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")

    with pytest.raises(TokenizerError, match="could not render chat template"):
        render_chat_prompt(
            tokenizer_path,
            [{"role": "user", "content": "hi"}],
            backend="simple",
        )


def test_render_chat_prompt_uses_local_jinja_template_file(tmp_path: Path) -> None:
    tokenizer_path = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    (tmp_path / "chat_template.jinja").write_text(
        "[gMASK]<sop>{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}",
        encoding="utf-8",
    )

    rendered = render_chat_prompt(
        tokenizer_path,
        [{"role": "user", "content": "hi"}],
        backend="simple",
    )

    assert rendered.backend == "local-jinja"
    assert rendered.tokenizer_backend == "simple"
    assert rendered.tokenizer_path == tmp_path
    assert rendered.text == "[gMASK]<sop><|user|>hi<|assistant|>"


def test_render_chat_prompt_uses_tokenizer_config_chat_template(tmp_path: Path) -> None:
    tokenizer_path = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": (
                    "{{ messages[0].content | tojson(ensure_ascii=False) }}"
                    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
                )
            }
        ),
        encoding="utf-8",
    )

    rendered = render_chat_prompt(
        tokenizer_path,
        [{"role": "user", "content": "你好"}],
        backend="simple",
        add_generation_prompt=False,
    )

    assert rendered.backend == "local-jinja"
    assert rendered.tokenizer_backend == "simple"
    assert rendered.text == '"你好"'
