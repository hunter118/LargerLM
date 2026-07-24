from __future__ import annotations

from pathlib import Path

import pytest

from largerlm.colibri_usage_analysis import (
    ColibriUsageAnalysisError,
    analyze_usage_overlap,
    load_colibri_usage,
)


def test_load_colibri_usage_merges_duplicate_rows(tmp_path: Path) -> None:
    usage = tmp_path / "usage.txt"
    usage.write_text("3 4 5\n3 4 2\n4 1 3\n", encoding="ascii")

    assert load_colibri_usage(usage) == {(3, 4): 7, (4, 1): 3}


def test_analyze_usage_overlap_reports_heldout_and_oracle_coverage() -> None:
    training = {(3, 1): 10, (3, 2): 8, (3, 3): 1}
    heldout = {(3, 1): 2, (3, 2): 5, (3, 3): 9}

    result = analyze_usage_overlap(training, heldout, slots=2)

    assert result["training"]["coverage"] == pytest.approx(18 / 19)
    assert result["heldout"]["training_profile_coverage"] == pytest.approx(7 / 16)
    assert result["heldout"]["oracle_coverage"] == pytest.approx(14 / 16)
    assert result["selected_set_overlap"]["jaccard"] == pytest.approx(1 / 3)


def test_analyze_usage_overlap_rejects_nonpositive_slots() -> None:
    with pytest.raises(ColibriUsageAnalysisError, match="slots"):
        analyze_usage_overlap({(1, 1): 1}, {(1, 1): 1}, slots=0)
