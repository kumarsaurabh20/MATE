#!/usr/bin/env python3
"""Run the synthetic falcarindiol-layout example using the existing MATE CLI."""

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=example_dir / "results",
        help="Directory for generated SQLite databases and analysis outputs.",
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Also demonstrate linear continuous fitting and bootstrap intervals.",
    )
    options = parser.parse_args()
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, str(example_dir.parents[1] / "MATE_v3_0.py")]
    for modality in ["transcriptomics", "metabolomics"]:
        modules = pd.read_csv(example_dir / f"{modality}_modules.csv")
        database = output_dir / f"{modality}_modules.sqlite"
        # These are demo-specific databases; reruns refresh the same demo table.
        with sqlite3.connect(database) as connection:
            modules.to_sql("clone_DR_25", connection, index=False, if_exists="replace")
        command.extend([
            f"--{modality}_table", str(example_dir / f"{modality}_example.csv"),
            f"--{modality}_db", str(database),
        ])

    command.extend([
        "--metadata", str(example_dir / "metadata_example.csv"),
        "--trajectory_mode", "reference", "--reference_condition", "Mock",
        "--stratify_by", "condition", "--time_values", "12,24,48",
        "--embedding_method", "none", "--max_lag_steps", "1",
        "--top_lag_pairs", "9", "--lag_plot_top_n", "3", "--outfile", "example",
    ])
    if options.continuous:
        command.extend([
            "--continuous_trajectory", "--continuous_embedding",
            "--trajectory_model", "linear", "--continuous_bootstrap", "20",
            "--continuous_grid_step", "3", "--continuous_lag_max", "12",
            "--continuous_lag_step", "3", "--continuous_plot_top_n", "3",
        ])

    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    print("Running the synthetic example; outputs:", output_dir, flush=True)
    subprocess.run(command, cwd=output_dir, env=environment, check=True)


if __name__ == "__main__":
    main()
