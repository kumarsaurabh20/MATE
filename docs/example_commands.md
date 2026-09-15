# MATE 3.0 example commands

Run these commands from the MATE repository root unless stated otherwise.
The bundled falcarindiol-layout example uses **synthetic data** and predefined
modules; it contains no experimental falcarindiol measurements.

## Install with Conda

```bash
conda env create -f environment.yml
conda activate mate
python -m mate --help
```

The environment uses Python 3.11 and installs MATE in editable mode, including
UMAP, statsmodels, and pytest. It reads dependency definitions from
`pyproject.toml`. The environment file is a portable installation recipe, not a
fully pinned lockfile.

To update an existing `mate` environment after pulling changes:

```bash
conda env update -n mate -f environment.yml
conda activate mate
```

## Install with pip

In an existing Python environment:

```bash
python -m pip install -r requirements.txt
python -m mate --help
```

For development and regression tests, include the development extra:

```bash
python -m pip install -e ".[umap,stats,dev]"
python -m pytest -q
```

## Run the bundled reference-mode example

```bash
python examples/falcarindiol/run_example.py
```

This creates two SQLite module databases in `examples/falcarindiol/results/`
and runs the reference-aware analysis against the `Mock` condition. PCA is
always included; UMAP and t-SNE are disabled for this quick example.

## Include continuous trajectories

```bash
python examples/falcarindiol/run_example.py \
  --continuous \
  --output-dir examples/falcarindiol/results/continuous
```

This uses linear interpolation and 20 replicate-bootstrap resamples, plus a
second PCA of the fitted curves. The settings keep the demonstration small.
Outputs in different directories are kept separate; rerunning in the same
directory replaces the corresponding demo outputs and module tables.

## Use the command-line interface directly

First run the bundled reference example above to create its module databases.
Then set the following paths **from the repository root**:

```bash
mate_repo="$PWD"
mate_data="$mate_repo/examples/falcarindiol"
mate_results="$mate_data/results"
```

The subshells below write each analysis into its own directory and return to
the repository root afterward. `--outfile` is a filename prefix, not a path.

### Reference-aware trajectories

```bash
mkdir -p "$mate_results/reference_cli"
(
  cd "$mate_results/reference_cli" || exit 1
  python -m mate \
    --transcriptomics_table "$mate_data/transcriptomics_example.csv" \
    --metabolomics_table "$mate_data/metabolomics_example.csv" \
    --transcriptomics_db "$mate_results/transcriptomics_modules.sqlite" \
    --metabolomics_db "$mate_results/metabolomics_modules.sqlite" \
    --metadata "$mate_data/metadata_example.csv" \
    --trajectory_mode reference --reference_condition Mock \
    --stratify_by condition --time_values "12,24,48" \
    --embedding_method none --max_lag_steps 1 --lag_plot_top_n 3 \
    --outfile reference
)
```

For reference-aware continuous fitting, add these options to that command
(or use the bundled continuous example):

```text
--continuous_trajectory --continuous_embedding
--trajectory_model linear --continuous_bootstrap 20
--continuous_grid_step 3 --continuous_lag_max 12 --continuous_lag_step 3
```

### Reference-free ordered trajectories

```bash
mkdir -p "$mate_results/pseudotime_cli"
(
  cd "$mate_results/pseudotime_cli" || exit 1
  python -m mate \
    --transcriptomics_table "$mate_data/transcriptomics_example.csv" \
    --metabolomics_table "$mate_data/metabolomics_example.csv" \
    --transcriptomics_db "$mate_results/transcriptomics_modules.sqlite" \
    --metabolomics_db "$mate_results/metabolomics_modules.sqlite" \
    --metadata "$mate_data/metadata_example.csv" \
    --trajectory_mode pseudotime --stratify_by condition \
    --time_values "12,24,48" --trajectory_model linear \
    --continuous_bootstrap 20 --continuous_grid_step 3 \
    --continuous_lag_max 12 --continuous_lag_step 3 \
    --continuous_plot_top_n 3 --embedding_method none \
    --max_lag_steps 1 --lag_plot_top_n 3 --outfile pseudotime
)
```

Here `Mock` and `Treatment` are each analyzed as their own ordered series.
No reference subtraction occurs. This demonstrates the interface; MATE follows
the supplied time ordering and does not infer pseudotime from these tables.

To add nonlinear embeddings, replace `--embedding_method none` with `umap`,
`tsne`, or `both`; UMAP requires the corresponding optional dependency.

For your own data, replace the five input paths and the condition/time labels.
Consult the [input example](../examples/falcarindiol/README.md) and
[output column guide](output_columns.md) for formats and interpretations.

## Script launcher and installed command

These entry points accept the same analysis arguments:

```bash
python MATE_v3_0.py --help
python -m mate --help
mate --help
```

The versioned script command is run from the repository root. The package and
installed command can be used from any directory after installation.

Check the installed version with `python -m mate --version` or `mate --version`.
