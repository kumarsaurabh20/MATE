"""Boundary checks for numerical conventions and input handling."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from mate.continuous import _fit_continuous_curve
from mate.fingerprints import calculate_sigma
from mate.io import load_clusters, normalize_feature_table
from mate.lag import _lag_similarity, _lag_time_value
from mate.trajectories import calculate_pseudotime_trajectory_descriptors


def test_fingerprint_sign_follows_mean_profile():
    matrix = pd.DataFrame({"feature": ["a", "b"], "s1": [-1, -2], "s2": [2, 4], "s3": [3, 6]})
    np.testing.assert_allclose(calculate_sigma(matrix), np.array([-1, 2, 3]) / np.sqrt(14))


def test_normalization_handles_constant_features():
    matrix = pd.DataFrame({"feature": ["constant", "varying"], "s1": [5., 1.], "s2": [5., 3.]})
    result = normalize_feature_table(matrix)
    np.testing.assert_allclose(result[["s1", "s2"]], [[0, 0], [-1, 1]])
    assert matrix.loc[0, "s1"] == 5


def test_positive_lag_means_transcript_leads():
    rna = np.array([0, 1, 3, 1, 0, 0], dtype=float)
    met = np.array([0, 0, 1, 3, 1, 0], dtype=float)
    similarity, error, overlap = _lag_similarity(rna, met, 1)
    assert similarity == pytest.approx(1)
    assert error == pytest.approx(0)
    assert overlap == 5
    assert similarity > _lag_similarity(rna, met, 0)[0]
    assert similarity > _lag_similarity(rna, met, -1)[0]
    assert _lag_time_value(["12", "24", "48"], 1) == (18, "12;24")


@pytest.mark.parametrize("model", ["linear", "pchip"])
def test_continuous_curve_passes_through_measured_points(model):
    observed = np.array([0., 2., 5.])
    values = np.array([0., 3., 1.])
    fitted, used_model = _fit_continuous_curve(observed, values, observed, model)
    np.testing.assert_allclose(fitted, values)
    assert used_model == model


def test_two_point_pseudotime_needs_no_reference():
    metadata = pd.DataFrame({"sample": ["a", "b"], "time": [1, 2]})
    fingerprints = pd.DataFrame({"ID": ["module"], "a": [1.], "b": [3.]})
    result = calculate_pseudotime_trajectory_descriptors(
        fingerprints, metadata, stratify_by="none", trajectory_metric="delta"
    )
    assert result.loc[0, "Pattern"] == "UP"
    assert result.loc[0, "NTimepoints"] == 2
    assert result.loc[0, "NTransitions"] == 1


def test_sqlite_loader_filters_clone_tables_and_decay_rate(tmp_path):
    path = tmp_path / "modules.sqlite"
    with sqlite3.connect(path) as connection:
        pd.DataFrame({"Cluster": ["a"], "Members": ["g1 g2"]}).to_sql(
            'Clone_DR_25 "quoted"', connection, index=False
        )
        pd.DataFrame({"Cluster": ["b"], "Members": ["g3"]}).to_sql(
            "clone_DR_50", connection, index=False
        )
        pd.DataFrame({"note": ["ignored"]}).to_sql("notes", connection, index=False)
    selected = load_clusters(path, 25)
    assert selected.to_dict("records") == [{"ID": "cluster001_DR_25_a", "Members": "g1 g2"}]
    assert len(load_clusters(path, 99)) == 2


def test_sqlite_loader_rejects_database_without_modules(tmp_path):
    path = tmp_path / "empty.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE notes (note TEXT)")
    with pytest.raises(ValueError, match="No tables were imported"):
        load_clusters(path, 25)
