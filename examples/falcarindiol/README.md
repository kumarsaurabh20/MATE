# Synthetic example

This directory demonstrates the requested falcarindiol example layout using
**synthetic data**. It contains no measured falcarindiol data, validated pathway
assignments, or real gene/metabolite identifiers.

## Inputs

| File | Contents |
| --- | --- |
| `metadata_example.csv` | Mock and Treatment conditions at 12, 24, and 48 hours, with three replicates per condition/time: 18 samples. |
| `transcriptomics_example.csv` | Six synthetic transcript features measured in those samples. |
| `metabolomics_example.csv` | Six synthetic metabolite features measured in those samples. |
| `transcriptomics_modules.csv` | Three predefined transcript modules, each containing two features. |
| `metabolomics_modules.csv` | Three independent metabolite modules, each containing two features. |

The values use simple rising, falling, and peaked response profiles with small
replicate variation (NumPy random seed 42). They demonstrate file formats and
exercise the workflow; they do not model a particular biological experiment.
Module definitions are supplied as readable CSV files. The runner creates the
two SQLite databases required by MATE, with a `clone_DR_25` table in each.

## Run

From the repository root, install and run:

```bash
python -m pip install -r requirements.txt
python examples/falcarindiol/run_example.py
```

Outputs and generated databases go to `examples/falcarindiol/results/`, which is
ignored by Git. Rerunning replaces the demo tables and outputs in that directory.

To include continuous fitting and choose a different output directory:

```bash
python examples/falcarindiol/run_example.py --continuous --output-dir /tmp/mate-example
```

The example uses PCA (`--embedding_method none` disables additional UMAP/t-SNE
embeddings). The optional continuous demonstration uses linear interpolation
and 20 bootstrap resamples for a quick run. The example therefore also works
with the minimal installation, `python -m pip install -e .`.

The runner invokes the `MATE_v3_0.py` launcher. For custom analyses,
use `python -m mate --help` and supply your own feature tables, metadata, and
module databases. MATE's `--outfile` is a filename prefix; outputs are written
in the current working directory. The runner sets that directory for you.

## Inspect the results

Start with:

- `example_RNA_trajectory_descriptors.csv` and
  `example_MET_trajectory_descriptors.csv`: one row per module/treatment.
- `example_joint_pca_embedding.png`: the joint module representation.
- `example_cross_omics_lag_top_pairs.csv`: ranked transcript–metabolite pairs.
- With `--continuous`, `example_RNA_continuous_trajectories.csv` and
  `example_continuous_lag_top_pairs.csv`: fitted curves and model-estimated lags.

See the [output column guide](../../docs/output_columns.md) and
[workflow diagram](../../docs/workflow.png).
