"""Small reproducible input tables shared by the integration tests."""

import os
import sqlite3
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mate-test-mpl"))

import numpy as np
import pandas as pd
import pytest

SCENARIOS = {
    "reference": ["--trajectory_mode", "reference", "--reference_condition", "Mock"],
    "continuous": [
        "--trajectory_mode", "reference", "--reference_condition", "Mock",
        "--continuous_trajectory", "--continuous_embedding", "--trajectory_model", "pchip",
    ],
    "pseudotime": [
        "--trajectory_mode", "pseudotime", "--trajectory_model", "linear",
        "--continuous_embedding",
    ],
}


def make_inputs(directory, scenario):
    """Write two feature matrices, metadata, and independent module databases."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    conditions = ["Series"] if scenario == "pseudotime" else ["Mock", "Treatment"]
    records = [
        {"sample": f"{condition}_{time}_{rep}", "condition": condition,
         "time": time, "replicate": rep}
        for condition in conditions
        for time in [12, 24, 36, 48]
        for rep in [1, 2, 3]
    ]
    metadata = pd.DataFrame(records)
    metadata.to_csv(directory / "metadata.csv", index=False)
    args = ["--metadata", str(directory / "metadata.csv")]
    for modality, prefix, delay in [("transcriptomics", "g", 0), ("metabolomics", "m", 0.6)]:
        rows = []
        for feature_index in range(6):
            module = feature_index // 2
            row = {"feature": f"{prefix}{feature_index}"}
            for sample in records:
                time = sample["time"] / 12 - delay
                response = (module + 1) * np.sin(time + module * 0.9)
                if sample["condition"] == "Mock":
                    response = 0.05 * time
                row[sample["sample"]] = (
                    10 + feature_index + response * (1 + 0.1 * (feature_index % 2))
                    + rng.normal(0, 0.15)
                )
            rows.append(row)
        table = directory / f"{modality}.csv"
        pd.DataFrame(rows).to_csv(table, index=False)
        database = directory / f"{modality}.sqlite"
        modules = pd.DataFrame({
            "Cluster": [1, 2, 3],
            "Members": [f"{prefix}0 {prefix}1 absent", f"{prefix}2,{prefix}3", f"[{prefix}4;{prefix}5]"],
        })
        with sqlite3.connect(database) as connection:
            modules.to_sql("clone_DR_25", connection, index=False, if_exists="replace")
            modules.iloc[:1].to_sql("clone_DR_50", connection, index=False, if_exists="replace")
            pd.DataFrame({"note": ["ignored"]}).to_sql(
                "notes", connection, index=False, if_exists="replace"
            )
        args += [f"--{modality}_table", str(table), f"--{modality}_db", str(database)]
    return args + SCENARIOS[scenario] + [
        "--embedding_method", "none", "--continuous_bootstrap", "5",
        "--continuous_grid_step", "6", "--continuous_lag_max", "12",
        "--continuous_lag_step", "6", "--continuous_top_pairs", "5",
        "--lag_plot_top_n", "1", "--continuous_plot_top_n", "1", "--granger",
        "--outfile", "result",
    ]


@pytest.fixture
def input_factory(tmp_path):
    return lambda scenario: make_inputs(tmp_path / "inputs", scenario)
