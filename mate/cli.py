"""Command-line arguments and entry point for MATE."""

import argparse
from typing import List, Optional

from . import __version__


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"MATE v{__version__}: Multi-omics Analysis through Trajectory Embeddings. "
            "Processes transcriptomics and metabolomics modules independently, "
            "then integrates their temporal behavior."
        )
    )
    parser.add_argument("--version", action="version", version=f"MATE {__version__}")

    # Required dual-omics inputs
    parser.add_argument(
        "--transcriptomics_table",
        required=True,
        help="Transcriptomics feature table: feature/gene IDs in first column, samples in remaining columns."
    )
    parser.add_argument(
        "--metabolomics_table",
        required=True,
        help="Metabolomics feature table: feature/metabolite IDs in first column, samples in remaining columns."
    )
    parser.add_argument(
        "--transcriptomics_db",
        required=True,
        help="SQLite database containing independently derived transcriptomics modules."
    )
    parser.add_argument(
        "--metabolomics_db",
        required=True,
        help="SQLite database containing independently derived metabolomics modules."
    )
    parser.add_argument(
        "-m", "--metadata",
        required=True,
        help="Metadata CSV containing sample and time/timepoint columns."
    )

    # General
    parser.add_argument("-o", "--outfile", default="MATE", help="Output prefix. Default: MATE.")
    parser.add_argument(
        "-dr", "--decay_rate", default=25, type=int,
        help="Decay rate used to select DR_<value> module tables. Default: 25."
    )
    parser.add_argument(
        "--no_zscore", action="store_true",
        help="Do not row-wise z-score features before metafingerprint calculation."
    )
    parser.add_argument(
        "--plot_format", default="png", choices=["png", "pdf", "svg"],
        help="Plot format. Default: png."
    )

    # Trajectories
    parser.add_argument(
        "--n_timepoints", default=0, type=int,
        help="Number of ordered measured time points. Minimum 2. Default 0 uses all resolved points."
    )
    parser.add_argument(
        "--time_values", default="",
        help="Comma-separated ordered time values, e.g. '12,24,48'."
    )
    parser.add_argument(
        "--time_column", default="",
        help="Metadata time column. If omitted, MATE tries time/timepoint/dpi."
    )
    parser.add_argument(
        "--trajectory_flat_threshold", default=0.10, type=float,
        help="Absolute delta threshold below which a transition is FLAT. Default: 0.10."
    )

    parser.add_argument(
        "--stratify_by",
        default="condition",
        help=(
            "Metadata column used to calculate separate trajectories for each "
            "experimental condition/treatment. Default: condition. Use 'none' "
            "to reproduce pooled trajectories."
        )
    )

    parser.add_argument(
        "--trajectory_mode",
        default="reference",
        choices=["reference", "pseudotime"],
        help=(
            "Trajectory design. 'reference' calculates treatment responses relative "
            "to a matched mock/reference at each time point. 'pseudotime' follows "
            "module fingerprints directly along the ordered time/pseudotime axis and "
            "does not require a mock/reference. Default: reference."
        )
    )

    parser.add_argument(
        "--reference_condition",
        default="",
        help=(
            "Value in --stratify_by identifying the matched reference/mock "
            "condition, e.g. MK, Mock, Control. Required only when "
            "--trajectory_mode reference."
        )
    )
    parser.add_argument(
        "--trajectory_metric",
        default="effect_size",
        choices=["effect_size", "delta"],
        help=(
            "Metric used to assign UP/DOWN/FLAT between adjacent time points. "
            "'effect_size' uses the standardized mean difference between replicate "
            "sample fingerprint loadings; 'delta' uses the difference between "
            "timepoint means. Default: effect_size."
        )
    )
    parser.add_argument(
        "--effect_size_threshold",
        default=0.50,
        type=float,
        help=(
            "Absolute standardized effect-size threshold for UP/DOWN classification "
            "when --trajectory_metric effect_size. Default: 0.50."
        )
    )

    # Continuous-time trajectory reconstruction
    parser.add_argument(
        "--continuous_trajectory", action="store_true",
        help=(
            "Enable model-assisted continuous-time reconstruction after the "
            "reference-aware observed trajectory has been calculated."
        )
    )
    parser.add_argument(
        "--trajectory_model", default="auto",
        choices=["auto", "linear", "pchip", "gam"],
        help=(
            "Continuous trajectory model. auto uses linear for <=4 measured "
            "time points and a low-complexity GAM for >=5. Default: auto."
        )
    )
    parser.add_argument(
        "--continuous_grid_step", default=3.0, type=float,
        help=(
            "Spacing of the modelled temporal grid in the same units as metadata "
            "time values. Default: 3.0."
        )
    )
    parser.add_argument(
        "--continuous_bootstrap", default=200, type=int,
        help=(
            "Number of replicate bootstrap resamples used for uncertainty bands. "
            "Set 0 to disable. Default: 200."
        )
    )
    parser.add_argument(
        "--continuous_ci", default=0.95, type=float,
        help="Bootstrap confidence level for continuous trajectories. Default: 0.95."
    )
    parser.add_argument(
        "--continuous_seed", default=42, type=int,
        help="Random seed for replicate bootstrap. Default: 42."
    )
    parser.add_argument(
        "--gam_min_timepoints", default=5, type=int,
        help=(
            "Minimum number of REAL measured time points required for GAM fitting. "
            "Default: 5."
        )
    )
    parser.add_argument(
        "--gam_splines", default=4, type=int,
        help="Spline basis size for GAM trajectories. Default: 4."
    )
    parser.add_argument(
        "--gam_alpha", default=1.0, type=float,
        help="GAM smoothness penalty. Larger values produce smoother curves. Default: 1.0."
    )
    parser.add_argument(
        "--continuous_lag_max", default=None, type=float,
        help=(
            "Maximum continuous lag to test in time units. Default is half the "
            "observed temporal span."
        )
    )
    parser.add_argument(
        "--continuous_lag_step", default=None, type=float,
        help=(
            "Lag search increment in time units. Default uses --continuous_grid_step."
        )
    )
    parser.add_argument(
        "--continuous_top_pairs", default=50, type=int,
        help="Number of top continuous-time RNA-MET pairs to save. Default: 50."
    )
    parser.add_argument(
        "--continuous_plot_top_n", default=12, type=int,
        help="Number of top continuous-time lag pairs to plot. Default: 12."
    )
    parser.add_argument(
        "--continuous_embedding", action="store_true",
        help=(
            "Generate a second joint embedding from the modelled response curves "
            "on the regular temporal grid."
        )
    )

    # Optional predictive directionality for genuinely dense measured time courses.
    parser.add_argument(
        "--granger", action="store_true",
        help=(
            "Run optional Granger predictive-direction tests on ORIGINAL measured "
            "time points only. Interpolated points are never used as observations."
        )
    )
    parser.add_argument(
        "--granger_min_timepoints", default=12, type=int,
        help=(
            "Minimum number of REAL measured time points required before Granger "
            "analysis is allowed. Default: 12."
        )
    )
    parser.add_argument(
        "--granger_max_lag", default=1, type=int,
        help="Maximum autoregressive lag order for Granger tests. Default: 1."
    )
    parser.add_argument(
        "--granger_transform", default="difference",
        choices=["none", "difference"],
        help=(
            "Transformation applied to observed trajectories before Granger tests. "
            "Default: first difference."
        )
    )
    parser.add_argument(
        "--granger_scope", default="top", choices=["top", "all"],
        help=(
            "Run Granger tests on top lag candidates or all RNA-MET pairs. "
            "Default: top."
        )
    )
    parser.add_argument(
        "--granger_top_n", default=100, type=int,
        help="Maximum candidate pairs used when --granger_scope top. Default: 100."
    )

    # Joint embeddings
    parser.add_argument(
        "--embedding_method", default="umap",
        choices=["none", "umap", "tsne", "both"],
        help="Additional nonlinear embedding(s). Joint PCA is always generated. Default: umap."
    )
    parser.add_argument(
        "--embedding_metric", default="euclidean",
        help="Distance metric for UMAP/t-SNE and related analyses. Default: euclidean."
    )
    parser.add_argument(
        "--embedding_color_by", default="Pattern",
        help="Annotation used to color joint embeddings. Default: Pattern."
    )
    parser.add_argument(
        "--embedding_size_by", default="DynamicRange",
        help="Descriptor used to scale point size. Default: DynamicRange."
    )
    parser.add_argument("--tsne_perplexity", default=30.0, type=float)
    parser.add_argument("--umap_neighbors", default=15, type=int)
    parser.add_argument("--umap_min_dist", default=0.10, type=float)

    # Lag-aware cross-omics comparison
    parser.add_argument(
        "--max_lag_steps", default=2, type=int,
        help="Maximum discrete timepoint shift tested. Positive lag means transcriptomics leads."
    )
    parser.add_argument(
        "--lag_direction", default="both",
        choices=["both", "transcript_leads"],
        help="Evaluate both lag directions or only lag 0/positive transcript-leading lags."
    )
    parser.add_argument("--top_lag_pairs", default=50, type=int)
    parser.add_argument("--lag_plot_top_n", default=12, type=int)
    parser.add_argument("--min_lag_similarity", default=0.0, type=float)

    # Module coherence
    parser.add_argument(
        "--module_coherence_threshold", default=0.50, type=float,
        help="Cosine threshold for fraction of module members agreeing with module trajectory."
    )

    return parser.parse_args(argv)


def run() -> None:
    """Parse the command line and run the analysis."""
    options = get_args()

    from .pipeline import main

    main(options)
