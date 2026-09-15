"""Regression checks against results captured before the module refactor."""

import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from mate.cli import get_args
from mate.pipeline import main


@pytest.mark.parametrize("scenario", ["reference", "continuous", "pseudotime"])
def test_pipeline_matches_original(scenario, input_factory, tmp_path, monkeypatch):
    args = input_factory(scenario)
    monkeypatch.chdir(tmp_path)
    main(get_args(args))

    expected_path = Path(__file__).parent / "data" / f"{scenario}.json"
    expected = json.loads(expected_path.read_text())
    actual_files = sorted(path.name for path in tmp_path.rglob("result*.*") if path.is_file())
    assert actual_files == expected["files"]
    for filename, original_csv in expected["tables"].items():
        actual = pd.read_csv(tmp_path / filename)
        original = pd.read_csv(io.StringIO(original_csv))
        pd.testing.assert_frame_equal(actual, original, rtol=1e-9, atol=1e-11)
    status = (tmp_path / "result_granger_status.txt").read_text()
    assert "only 4 REAL measured time points" in status
    for path in tmp_path.rglob("*.png"):
        assert path.stat().st_size > 1000


@pytest.mark.parametrize("command", [["-m", "mate"], ["MATE_v3_0.py"]])
def test_cli_help(command):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, *command, "--help"], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "MATE v3.0.0:" in result.stdout
    assert "--trajectory_mode {reference,pseudotime}" in result.stdout
    assert "--continuous_trajectory" in result.stdout


def test_reference_mode_requires_reference(input_factory, tmp_path, monkeypatch):
    options = get_args(input_factory("reference"))
    options.reference_condition = ""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="reference_condition is required"):
        main(options)
