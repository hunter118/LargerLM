from __future__ import annotations

from pathlib import Path

from largerlm.cli import main as cli_main
from largerlm.text_generator import generate_text
from largerlm.token_generator import TokenGenerationResult
from test_token_generator import write_config, write_fake_runner, write_fixture
from test_tokenizer import write_simple_tokenizer


def test_generate_text_encodes_prompt_and_decodes_output(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    cache_dir = tmp_path / "mla-kv-b-cache"
    write_fake_runner(runner)

    result = generate_text(
        tokenizer_path=tokenizer,
        tokenizer_backend="simple",
        prompt="A",
        max_new_tokens=2,
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layers={1},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        router_score="raw",
        rms_norm_eps=0.0,
        prefill_mla_kv_b_cache_dir=cache_dir,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        echo_runner_output=False,
    )

    assert result.prompt_token_ids == (0,)
    assert result.generated_token_ids == (2, 2)
    assert result.generated_text == "CC"
    assert result.full_text == "ACC"
    assert result.token_result.steps
    first_command = result.token_result.steps[0].decode_layers[0].command
    assert first_command[first_command.index("--mla-kv-b-cache-dir") + 1] == str(
        cache_dir
    )


def test_generate_text_can_auto_enable_batch_prefill_after_tokenization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr(
        "largerlm.text_generator.generate_token_ids",
        fake_generate_token_ids,
    )

    result = generate_text(
        tokenizer_path=tokenizer,
        tokenizer_backend="simple",
        prompt="AB",
        max_new_tokens=1,
        auto_batch_prefill_prompt=True,
    )

    assert captured["batch_prefill_prompt"] is True
    assert result.prompt_token_ids == (0, 1)
    assert result.generated_text == "C"
    assert result.full_text == "ABC"


def test_generate_text_cli_derives_decode_args_from_config(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    tokenizer = write_simple_tokenizer(tmp_path / "simple_tokenizer.json")
    config = tmp_path / "config.json"
    write_fake_runner(runner)
    write_config(config)

    status = cli_main(
        [
            "generate-text",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--tokenizer",
            str(tokenizer),
            "--tokenizer-backend",
            "simple",
            "--prompt",
            "A",
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--layers",
            "1",
            "--max-new-tokens",
            "1",
            "--max-k",
            "2",
            "--logits-top-k",
            "2",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--quiet-runner",
        ]
    )

    assert status == 0


def test_generate_text_cli_rejects_oversized_prompt_file(
    tmp_path: Path,
    capsys,
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("AB", encoding="utf-8")

    status = cli_main(
        [
            "generate-text",
            str(tmp_path / "experts.json"),
            str(tmp_path / "resident.json"),
            str(tmp_path / "cache_layout.json"),
            str(tmp_path / "cache.bin"),
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
            "--prompt-file",
            str(prompt_file),
            "--max-prompt-bytes",
            "1",
            "--max-new-tokens",
            "1",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prompt file" in err
    assert "exceeds --max-prompt-bytes" in err
