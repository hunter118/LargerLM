from __future__ import annotations

import json
from pathlib import Path

from largerlm.cli import main as cli_main
from largerlm.metal_generate import MetalTokenGenerationResult
from largerlm.metal_text_generator import (
    MetalTextGenerationBatchResult,
    MetalTextGenerationError,
    MetalTextGenerationResult,
    MetalTextGenerationSession,
    generate_metal_text_batch,
    generate_metal_text,
)
from test_tokenizer import write_simple_tokenizer


def _metal_token_result(
    root: Path,
    *,
    prompt_token_ids: tuple[int, ...] = (0,),
    generated_token_ids: tuple[int, ...] = (2,),
    estimated_live_working_set_bytes: int = 1024,
    admission_ok: bool | None = None,
    available_unified_memory_ok: bool | None = None,
    system_available_memory_bytes: int | None = None,
    required_available_memory_bytes: int | None = None,
    expert_buffer_count_allocated: int | None = None,
    prompt_prefill_estimated_live_working_set_bytes: int | None = None,
) -> MetalTokenGenerationResult:
    return MetalTokenGenerationResult(
        prepared_dir=root / "prepared",
        binary=root / "glm_moe_infer",
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        work_dir=root / "work",
        kept_work_dir=False,
        cache_layout_path=root / "decode_cache_layout.json",
        cache_file_path=root / "decode_cache.bin",
        input_f32_path=root / "input.f32",
        output_f32_path=root / "output.f32",
        generated_json_path=root / "generated.json",
        elapsed_seconds=2.0,
        decode_elapsed_seconds=(1.0,),
        final_logits_elapsed_seconds=(0.1,),
        estimated_live_working_set_bytes=estimated_live_working_set_bytes,
        max_live_working_set_mib=768,
        cache_total_bytes=64,
        note="decode-only path",
        admission_ok=admission_ok,
        available_unified_memory_ok=available_unified_memory_ok,
        system_available_memory_bytes=system_available_memory_bytes,
        required_available_memory_bytes=required_available_memory_bytes,
        expert_buffer_count_allocated=expert_buffer_count_allocated,
        prompt_prefill_estimated_live_working_set_bytes=(
            prompt_prefill_estimated_live_working_set_bytes
        ),
    )


def test_generate_metal_text_encodes_prompt_and_decodes_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    captured: dict[str, object] = {}

    def fake_generate_metal_token_ids(prepared_dir, **kwargs):
        captured["prepared_dir"] = Path(prepared_dir)
        captured.update(kwargs)
        return _metal_token_result(
            tmp_path,
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2, 2),
        )

    monkeypatch.setattr(
        "largerlm.metal_text_generator.generate_metal_token_ids",
        fake_generate_metal_token_ids,
    )

    result = generate_metal_text(
        prepared_dir=tmp_path / "prepared",
        tokenizer_path=tokenizer,
        tokenizer_backend="simple",
        prompt="AB",
        max_new_tokens=2,
        binary="/tmp/glm_moe_infer",
        max_live_working_set_mib=256,
        prefill_max_live_working_set_mib=2048.0,
        min_free_unified_memory_gib=4.0,
    )

    assert captured["prepared_dir"] == tmp_path / "prepared"
    assert captured["prompt_token_ids"] == (0, 1)
    assert captured["max_new_tokens"] == 2
    assert captured["binary"] == "/tmp/glm_moe_infer"
    assert captured["max_live_working_set_mib"] == 256
    assert captured["prefill_max_live_working_set_mib"] == 2048.0
    assert captured["min_free_unified_memory_gib"] == 4.0
    assert captured["prefill_prompt"] is True
    assert result.prompt_token_ids == (0, 1)
    assert result.generated_token_ids == (2, 2)
    assert result.generated_text == "CC"
    assert result.full_text == "ABCC"
    assert result.token_result.generated_token_ids == (2, 2)


def test_generate_metal_text_leaves_single_token_prompt_decode_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    captured: dict[str, object] = {}

    def fake_generate_metal_token_ids(prepared_dir, **kwargs):
        captured.update(kwargs)
        return _metal_token_result(
            tmp_path,
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
        )

    monkeypatch.setattr(
        "largerlm.metal_text_generator.generate_metal_token_ids",
        fake_generate_metal_token_ids,
    )

    result = generate_metal_text(
        prepared_dir=tmp_path / "prepared",
        tokenizer_path=tokenizer,
        tokenizer_backend="simple",
        prompt="A",
        max_new_tokens=1,
    )

    assert captured["prompt_token_ids"] == (0,)
    assert "prefill_prompt" not in captured
    assert result.generated_text == "C"


def test_generate_metal_text_honors_explicit_decode_only_risk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    captured: dict[str, object] = {}

    def fake_generate_metal_token_ids(prepared_dir, **kwargs):
        captured.update(kwargs)
        return _metal_token_result(
            tmp_path,
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
        )

    monkeypatch.setattr(
        "largerlm.metal_text_generator.generate_metal_token_ids",
        fake_generate_metal_token_ids,
    )

    generate_metal_text(
        prepared_dir=tmp_path / "prepared",
        tokenizer_path=tokenizer,
        tokenizer_backend="simple",
        prompt="AB",
        max_new_tokens=1,
        allow_decode_only_multi_token_prompt=True,
    )

    assert captured["prompt_token_ids"] == (0, 1)
    assert captured["allow_decode_only_multi_token_prompt"] is True
    assert "prefill_prompt" not in captured


def test_generate_metal_text_uses_explicit_generate_server_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    session = object()
    captured: dict[str, object] = {}

    def fake_generate_metal_token_ids(prepared_dir, **kwargs):
        captured["prepared_dir"] = Path(prepared_dir)
        captured.update(kwargs)
        return _metal_token_result(
            tmp_path,
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
        )

    monkeypatch.setattr(
        "largerlm.metal_text_generator.generate_metal_token_ids",
        fake_generate_metal_token_ids,
    )

    result = generate_metal_text(
        prepared_dir=tmp_path / "prepared",
        tokenizer_path=tokenizer,
        tokenizer_backend="simple",
        prompt="AB",
        max_new_tokens=1,
        generate_server_session=session,
    )

    assert captured["prepared_dir"] == tmp_path / "prepared"
    assert captured["generate_server_session"] is session
    assert captured["prefill_prompt"] is True
    assert result.generated_text == "C"


def test_metal_text_generation_session_reuses_runtime_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    runtime_sessions: list[object] = []
    captured_sessions: list[object] = []

    class FakeRuntimeSession:
        instances: list["FakeRuntimeSession"] = []

        def __init__(self, *, binary, prepared_dir, quiet=True):
            self.binary = Path(binary)
            self.prepared_dir = Path(prepared_dir)
            self.quiet = quiet
            self.request_count = 0
            self.closed = False
            FakeRuntimeSession.instances.append(self)
            runtime_sessions.append(self)

        def close(self):
            self.closed = True

    def fake_generate_metal_token_ids(prepared_dir, **kwargs):
        runtime = kwargs["generate_server_session"]
        runtime.request_count += 1
        captured_sessions.append(runtime)
        return _metal_token_result(
            tmp_path,
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
        )

    monkeypatch.setattr(
        "largerlm.metal_text_generator.MetalGenerateServerSession",
        FakeRuntimeSession,
    )
    monkeypatch.setattr(
        "largerlm.metal_text_generator.generate_metal_token_ids",
        fake_generate_metal_token_ids,
    )

    with MetalTextGenerationSession(
        prepared_dir=tmp_path / "prepared",
        tokenizer_path=tokenizer,
        tokenizer_backend="simple",
        binary="/tmp/glm_moe_infer",
        quiet=True,
    ) as session:
        first = session.generate(prompt="A", max_new_tokens=1)
        second = session.generate(
            prompt="AB",
            max_new_tokens=1,
            max_live_working_set_mib=256,
        )
        assert session.request_count == 2

    assert len(runtime_sessions) == 1
    runtime = runtime_sessions[0]
    assert runtime.binary == Path("/tmp/glm_moe_infer")
    assert runtime.prepared_dir == tmp_path / "prepared"
    assert runtime.quiet is True
    assert captured_sessions == [runtime, runtime]
    assert runtime.closed is True
    assert first.generated_text == "C"
    assert second.generated_text == "C"


def test_metal_text_generation_session_owns_generate_server_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")

    class FakeRuntimeSession:
        request_count = 0

        def __init__(self, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "largerlm.metal_text_generator.MetalGenerateServerSession",
        FakeRuntimeSession,
    )

    with MetalTextGenerationSession(
        prepared_dir=tmp_path / "prepared",
        tokenizer_path=tokenizer,
        tokenizer_backend="simple",
    ) as session:
        try:
            session.generate(
                prompt="A",
                max_new_tokens=1,
                generate_server_session=object(),
            )
        except MetalTextGenerationError as exc:
            assert "owns generate_server_session" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected MetalTextGenerationError")


def test_generate_metal_text_batch_reuses_one_runtime_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    runtime_sessions: list[object] = []
    captured: list[dict[str, object]] = []

    class FakeRuntimeSession:
        startup_elapsed_seconds = 0.125

        def __init__(self, *, binary, prepared_dir, quiet=True):
            self.binary = Path(binary)
            self.prepared_dir = Path(prepared_dir)
            self.quiet = quiet
            self.request_count = 0
            self.closed = False
            runtime_sessions.append(self)

        def close(self):
            self.closed = True

    def fake_generate_metal_token_ids(prepared_dir, **kwargs):
        runtime = kwargs["generate_server_session"]
        runtime.request_count += 1
        prompt_ids = tuple(kwargs["prompt_token_ids"])
        request_index = runtime.request_count
        captured.append(dict(kwargs))
        return _metal_token_result(
            tmp_path,
            prompt_token_ids=prompt_ids,
            generated_token_ids=(2,),
            estimated_live_working_set_bytes=1024 * request_index,
            admission_ok=True,
            available_unified_memory_ok=True,
            system_available_memory_bytes=10_000 - request_index,
            required_available_memory_bytes=1_000 * request_index,
            expert_buffer_count_allocated=request_index,
            prompt_prefill_estimated_live_working_set_bytes=(
                2048 if len(prompt_ids) > 1 else None
            ),
        )

    monkeypatch.setattr(
        "largerlm.metal_text_generator.MetalGenerateServerSession",
        FakeRuntimeSession,
    )
    monkeypatch.setattr(
        "largerlm.metal_text_generator.generate_metal_token_ids",
        fake_generate_metal_token_ids,
    )
    ticks = iter((10.0, 12.0))
    monkeypatch.setattr(
        "largerlm.metal_text_generator.time.monotonic",
        lambda: next(ticks),
    )

    result = generate_metal_text_batch(
        prepared_dir=tmp_path / "prepared",
        tokenizer_path=tokenizer,
        tokenizer_backend="simple",
        prompts=["A", "AB"],
        max_new_tokens=1,
        binary="/tmp/glm_moe_infer",
        max_live_working_set_mib=256,
    )

    assert len(runtime_sessions) == 1
    runtime = runtime_sessions[0]
    assert runtime.closed is True
    assert runtime.binary == Path("/tmp/glm_moe_infer")
    assert result.prompts == ("A", "AB")
    assert result.runtime_request_count == 2
    assert result.runtime_startup_elapsed_seconds == 0.125
    assert result.elapsed_seconds == 2.0
    assert result.total_prompt_tokens == 3
    assert result.total_generated_tokens == 2
    assert result.generated_tokens_per_second == 1.0
    assert result.max_estimated_live_working_set_bytes == 2048
    assert result.all_admission_ok is True
    assert result.all_available_unified_memory_ok is True
    assert result.min_system_available_memory_bytes == 9998
    assert result.max_required_available_memory_bytes == 2000
    assert result.max_expert_buffer_count_allocated == 2
    assert result.max_prompt_prefill_estimated_live_working_set_bytes == 2048
    assert len(result.results) == 2
    assert captured[0]["generate_server_session"] is runtime
    assert captured[1]["generate_server_session"] is runtime
    assert "prefill_prompt" not in captured[0]
    assert captured[1]["prefill_prompt"] is True
    assert captured[0]["max_live_working_set_mib"] == 256


def test_generate_metal_text_batch_rejects_shared_work_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")

    try:
        generate_metal_text_batch(
            prepared_dir=tmp_path / "prepared",
            tokenizer_path=tokenizer,
            tokenizer_backend="simple",
            prompts=["A", "B"],
            max_new_tokens=1,
            work_dir=tmp_path / "work",
        )
    except MetalTextGenerationError as exc:
        assert "cannot reuse one work_dir" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected MetalTextGenerationError")


def test_generate_metal_text_cli_passes_args_and_prints_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    captured: dict[str, object] = {}

    def fake_generate_metal_text(**kwargs):
        captured.update(kwargs)
        token_result = _metal_token_result(
            tmp_path,
            prompt_token_ids=(0,),
            generated_token_ids=(2,),
        )
        from largerlm.metal_text_generator import MetalTextGenerationResult

        return MetalTextGenerationResult(
            prompt=str(kwargs["prompt"]),
            generated_text="C",
            full_text="AC",
            tokenizer_backend="simple",
            tokenizer_path=tokenizer,
            prompt_token_ids=(0,),
            generated_token_ids=(2,),
            token_result=token_result,
        )

    monkeypatch.setattr("largerlm.cli.generate_metal_text", fake_generate_metal_text)

    status = cli_main(
        [
            "generate-metal-text",
            str(tmp_path / "prepared"),
            "--tokenizer",
            str(tokenizer),
            "--tokenizer-backend",
            "simple",
            "--prompt",
            "A",
            "--max-new-tokens",
            "1",
            "--binary",
            "/tmp/glm_moe_infer",
            "--max-live-working-set-mib",
            "256",
            "--min-free-unified-memory-gib",
            "12",
            "--allow-decode-only-multi-token-prompt",
            "--prefill-prompt",
            "--prefill-runner",
            "/tmp/largerlm-runner",
            "--prefill-prompt-chunk-tokens",
            "2",
            "--prefill-max-live-working-set-mib",
            "2048",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prepared_dir"] == str(tmp_path / "prepared")
    assert captured["tokenizer_path"] == str(tokenizer)
    assert captured["tokenizer_backend"] == "simple"
    assert captured["prompt"] == "A"
    assert captured["max_new_tokens"] == 1
    assert captured["binary"] == "/tmp/glm_moe_infer"
    assert captured["max_live_working_set_mib"] == 256
    assert captured["min_free_unified_memory_gib"] == 12.0
    assert captured["allow_decode_only_multi_token_prompt"] is True
    assert captured["prefill_prompt"] is True
    assert captured["prefill_runner"] == "/tmp/largerlm-runner"
    assert captured["prefill_prompt_chunk_tokens"] == 2
    assert captured["prefill_max_live_working_set_mib"] == 2048.0
    assert captured["use_generate_server_jsonl"] is True
    assert captured["quiet"] is True
    out = capsys.readouterr().out
    assert '"generated_text": "C"' in out
    assert '"generated_token_ids": [' in out


def test_generate_metal_text_cli_renders_chat_messages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    (tmp_path / "chat_template.jinja").write_text(
        "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_generate_metal_text(**kwargs):
        captured.update(kwargs)
        token_result = _metal_token_result(
            tmp_path,
            prompt_token_ids=(0,),
            generated_token_ids=(2,),
        )
        from largerlm.metal_text_generator import MetalTextGenerationResult

        return MetalTextGenerationResult(
            prompt=str(kwargs["prompt"]),
            generated_text="C",
            full_text="AC",
            tokenizer_backend="simple",
            tokenizer_path=tmp_path,
            prompt_token_ids=(0,),
            generated_token_ids=(2,),
            token_result=token_result,
        )

    monkeypatch.setattr("largerlm.cli.generate_metal_text", fake_generate_metal_text)

    status = cli_main(
        [
            "generate-metal-text",
            str(tmp_path / "prepared"),
            "--tokenizer",
            str(tokenizer),
            "--tokenizer-backend",
            "simple",
            "--chat-messages",
            json.dumps([{"role": "user", "content": "A"}]),
            "--max-new-tokens",
            "1",
            "--quiet-runner",
        ]
    )

    assert status == 0
    assert captured["prompt"] == "<|user|>A<|assistant|>"
    assert captured["tokenizer_path"] == str(tmp_path)
    assert captured["tokenizer_backend"] == "simple"
    assert captured["add_special_tokens"] is False


def test_generate_metal_text_batch_cli_reads_jsonl_and_prints_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    prompts_jsonl = tmp_path / "prompts.jsonl"
    prompts_jsonl.write_text(
        json.dumps({"prompt": "A"}) + "\n" + json.dumps({"prompt": "AB"}) + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_generate_metal_text_batch(**kwargs):
        captured.update(kwargs)
        first_token_result = _metal_token_result(
            tmp_path,
            prompt_token_ids=(0,),
            generated_token_ids=(2,),
        )
        second_token_result = _metal_token_result(
            tmp_path,
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
        )
        return MetalTextGenerationBatchResult(
            prepared_dir=Path(kwargs["prepared_dir"]),
            prompts=tuple(kwargs["prompts"]),
            results=(
                MetalTextGenerationResult(
                    prompt="A",
                    generated_text="C",
                    full_text="AC",
                    tokenizer_backend="simple",
                    tokenizer_path=tokenizer,
                    prompt_token_ids=(0,),
                    generated_token_ids=(2,),
                    token_result=first_token_result,
                ),
                MetalTextGenerationResult(
                    prompt="AB",
                    generated_text="C",
                    full_text="ABC",
                    tokenizer_backend="simple",
                    tokenizer_path=tokenizer,
                    prompt_token_ids=(0, 1),
                    generated_token_ids=(2,),
                    token_result=second_token_result,
                ),
            ),
            tokenizer_backend="simple",
            tokenizer_path=tokenizer,
            runtime_request_count=2,
            elapsed_seconds=2.0,
            total_prompt_tokens=3,
            total_generated_tokens=2,
            generated_tokens_per_second=1.0,
            max_estimated_live_working_set_bytes=2048,
            all_admission_ok=True,
            all_available_unified_memory_ok=True,
            runtime_startup_elapsed_seconds=0.125,
            max_prompt_prefill_estimated_live_working_set_bytes=2048,
            min_system_available_memory_bytes=9998,
            max_required_available_memory_bytes=2000,
            max_expert_buffer_count_allocated=2,
        )

    monkeypatch.setattr(
        "largerlm.cli.generate_metal_text_batch",
        fake_generate_metal_text_batch,
    )

    status = cli_main(
        [
            "generate-metal-text-batch",
            str(tmp_path / "prepared"),
            "--prompts-jsonl",
            str(prompts_jsonl),
            "--tokenizer",
            str(tokenizer),
            "--tokenizer-backend",
            "simple",
            "--max-new-tokens",
            "1",
            "--binary",
            "/tmp/glm_moe_infer",
            "--max-live-working-set-mib",
            "256",
            "--min-free-unified-memory-gib",
            "12",
            "--prefill-prompt",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prepared_dir"] == str(tmp_path / "prepared")
    assert captured["prompts"] == ("A", "AB")
    assert captured["tokenizer_path"] == str(tokenizer)
    assert captured["tokenizer_backend"] == "simple"
    assert captured["max_new_tokens"] == 1
    assert captured["binary"] == "/tmp/glm_moe_infer"
    assert captured["max_live_working_set_mib"] == 256
    assert captured["min_free_unified_memory_gib"] == 12.0
    assert captured["prefill_prompt"] is True
    assert captured["quiet"] is True
    out = capsys.readouterr().out
    assert '"runtime_request_count": 2' in out
    assert '"generated_tokens_per_second": 1.0' in out
    assert '"max_estimated_live_working_set_bytes": 2048' in out
    assert '"all_admission_ok": true' in out
    assert '"generated_text": "C"' in out
