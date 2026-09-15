#!/usr/bin/env python3
"""
MATE v2.4
Multi-omics Analysis through Trajectory Embeddings

Clean dual-omics workflow:
input -> fingerprints -> trajectories -> embeddings -> lag analysis -> outputs

Required inputs:
  --transcriptomics_table
  --metabolomics_table
  --transcriptomics_db
  --metabolomics_db
  --metadata

Transcriptomics and metabolomics are processed independently until module
fingerprints and trajectory descriptors have been calculated. Integration then
occurs at the module-representation level through joint embeddings and lag-aware
cross-omics temporal comparison.

Positive lag means transcriptomics precedes metabolomics.
Supported trajectory designs: 2 or more measured time points.
MATE v2.4 supports two trajectory modes: reference-aware treatment response and
reference-free pseudotime/ordered-time trajectories. Continuous modelled grid
points are representations of fitted curves and are never treated as new
biological observations.
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from scipy.interpolate import PchipInterpolator

try:
    from statsmodels.gam.api import GLMGam, BSplines
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.stats.multitest import multipletests
except ImportError:
    GLMGam = None
    BSplines = None
    grangercausalitytests = None
    multipletests = None

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    Birch,
    AffinityPropagation,
    MeanShift,
    estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture

try:
    import umap.umap_ as umap
except ImportError:
    umap = None

import gizmos


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MATE v2.4: Multi-omics Analysis through Trajectory Embeddings. "
            "Processes transcriptomics and metabolomics modules independently, "
            "then integrates their temporal behavior."
        )
    )

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

    # Continuous-time trajectory reconstruction (MATE v2.4)
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

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def natural_key(value: object) -> List[object]:
    """Natural sort key: 2 before 10, T2 before T10."""
    text = str(value)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first matching column name from a candidate list, case-insensitive."""
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def safe_output_prefix(prefix: object) -> str:
    prefix = str(prefix) if prefix else "results"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)


def orient_vector(vec: np.ndarray) -> np.ndarray:
    """
    SVD vectors have arbitrary sign. Orient the vector so that the largest
    absolute entry is positive. This makes outputs more consistent.
    """
    vec = np.asarray(vec, dtype=float)
    if vec.size == 0 or np.all(np.isnan(vec)):
        return vec
    idx = int(np.nanargmax(np.abs(vec)))
    if vec[idx] < 0:
        vec = -vec
    return vec


def parse_comma_list(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


# -----------------------------------------------------------------------------
# Input data
# -----------------------------------------------------------------------------


def read_feature_table(feature_table: str) -> pd.DataFrame:
    """Read one omics feature table and normalize its identifier column to 'feature'."""
    df = pd.read_csv(feature_table)
    df.columns = [str(c).strip() for c in df.columns]

    if df.shape[1] < 2:
        raise ValueError(
            f"Feature table '{feature_table}' must contain one feature-ID column and at least one sample column."
        )

    if "feature" not in df.columns:
        if "gene" in df.columns:
            df = df.rename(columns={"gene": "feature"})
        elif "metabolite" in df.columns:
            df = df.rename(columns={"metabolite": "feature"})
        else:
            first_col = df.columns[0]
            print(
                f"WARNING: No feature/gene/metabolite column found in {feature_table}. "
                f"Using first column '{first_col}' as feature ID."
            )
            df = df.rename(columns={first_col: "feature"})

    df["feature"] = df["feature"].astype(str).str.strip()
    df = df[df["feature"].notna() & (df["feature"] != "")].copy()

    for col in df.columns:
        if col != "feature":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = ["feature"] + numeric_cols
    dropped = [c for c in df.columns if c not in keep_cols]
    if dropped:
        print(f"WARNING: Dropping non-numeric columns from {feature_table}: {dropped}")
    df = df[keep_cols]

    if df["feature"].duplicated().any():
        print(f"WARNING: Duplicate feature IDs found in {feature_table}; averaging duplicate rows.")
        df = df.groupby("feature", as_index=False).mean(numeric_only=True)

    print(
        f"Loaded feature table '{feature_table}': "
        f"{df.shape[0]} features x {df.shape[1] - 1} numeric sample columns"
    )
    return df


def normalize_feature_table(df: pd.DataFrame, do_zscore: bool = True) -> pd.DataFrame:
    """Row-wise z-score each feature across all numeric samples."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric sample columns found in feature table.")

    if do_zscore:
        values = out[numeric_cols].to_numpy(dtype=float)
        scaled = zscore(values, axis=1, nan_policy="omit")
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        out[numeric_cols] = scaled
        print("Applied row-wise z-score normalization across sample columns.")
    else:
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        print("Skipped z-score normalization; using numeric values as provided.")

    return out


def read_metadata(metadata_file: str) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_file)
    metadata.columns = [str(c).strip() for c in metadata.columns]

    sample_col = find_column(metadata, ["sample", "Sample", "sample_id", "SampleID", "sample_name"])
    if sample_col is None:
        raise ValueError("Metadata must contain a sample column, e.g. 'sample'.")

    if sample_col != "sample":
        metadata = metadata.rename(columns={sample_col: "sample"})

    metadata["sample"] = metadata["sample"].astype(str).str.strip()
    print(f"Loaded metadata: {metadata.shape[0]} rows")
    return metadata


def get_group_column(metadata: pd.DataFrame, group_flag: str) -> str:
    if group_flag == "timepoint":
        col = find_column(metadata, ["time", "timepoint", "dpi", "Time", "Timepoint"])
        if col is None:
            raise ValueError("For -g timepoint, metadata must contain a time/timepoint column.")
        return col

    col = find_column(metadata, ["condition", "treatment", "Treatment", "group", "Group"])
    if col is None:
        raise ValueError("For -g treatment, metadata must contain a condition/treatment column.")
    return col


def group_samples(metadata: pd.DataFrame, feature_df: pd.DataFrame, group_flag: str) -> Dict[str, pd.DataFrame]:
    """Create one feature matrix per timepoint/treatment group."""
    group_col = get_group_column(metadata, group_flag)
    print(f"Grouping samples by metadata column: {group_col}")

    metadata = metadata.copy()
    metadata[group_col] = metadata[group_col].astype(str).str.strip()

    feature_columns = set(feature_df.columns)
    groups = sorted(metadata[group_col].dropna().unique().tolist(), key=natural_key)
    grouped = {}

    for group in groups:
        subset = metadata[metadata[group_col] == group]
        requested_samples = subset["sample"].astype(str).str.strip().tolist()
        selected_samples = [s for s in requested_samples if s in feature_columns]
        missing_samples = [s for s in requested_samples if s not in feature_columns]

        if missing_samples:
            print(f"WARNING: group {group}: {len(missing_samples)} metadata samples not present in feature table.")

        if not selected_samples:
            print(f"WARNING: group {group}: no matching sample columns; skipping group.")
            continue

        grouped[group] = feature_df[["feature"] + selected_samples].copy()
        print(f"Group {group}: using {len(selected_samples)} samples")

    if not grouped:
        raise ValueError("No groups contained matching samples between metadata and feature table.")

    return grouped


# -----------------------------------------------------------------------------
# Cluster import and parsing
# -----------------------------------------------------------------------------


def load_clusters(sqlite_db_name: str, decay_rate: int) -> pd.DataFrame:
    """Import cluster tables from SQLite and keep clusters matching DR_<decay_rate> when available."""
    print(f"Importing clusters from SQLite database: {sqlite_db_name}")
    all_tables = gizmos.import_from_sql(
        sqlite_db_name,
        sqlite_tablename="",
        df_columns=[],
        conditions={},
        structures=False,
        clone=True,
    )

    if not all_tables:
        raise ValueError(f"No tables were imported from {sqlite_db_name}")

    frames = []
    for table_name, df in all_tables.items():
        df = df.copy()
        match = re.search(r"(DR_\d+)", str(table_name))
        df["Source"] = match.group(1) if match else "unknown"
        df["Table"] = table_name
        frames.append(df)

    frame = pd.concat(frames, axis=0, ignore_index=True)

    if "Cluster" not in frame.columns:
        raise ValueError("Cluster table must contain a 'Cluster' column.")
    if "Members" not in frame.columns:
        raise ValueError("Cluster table must contain a 'Members' column.")

    requested_source = f"DR_{decay_rate}"
    if requested_source in set(frame["Source"]):
        frame = frame[frame["Source"] == requested_source].copy()
        print(f"Filtered clusters to {requested_source}: {frame.shape[0]} clusters")
    else:
        print(f"WARNING: {requested_source} not found in table names. Using all imported clusters: {frame.shape[0]} clusters")

    frame = frame.reset_index(drop=True)
    frame["cluster_id"] = [f"cluster{i:03d}" for i in range(1, frame.shape[0] + 1)]
    frame["ID"] = frame["cluster_id"] + "_" + frame["Source"].astype(str) + "_" + frame["Cluster"].astype(str)

    clusters_frame = frame[["ID", "Members"]].copy()
    print(f"Prepared cluster frame: {clusters_frame.shape[0]} clusters")
    return clusters_frame


def parse_members(member_string: object) -> List[str]:
    """Parse cluster members separated by whitespace, comma, semicolon, or brackets."""
    if pd.isna(member_string):
        return []
    text = str(member_string).strip()
    text = text.replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ")
    text = text.replace("'", "").replace('"', "")
    return [x.strip() for x in re.split(r"[,;\s]+", text) if x.strip()]


# -----------------------------------------------------------------------------
# Fingerprints
# -----------------------------------------------------------------------------


def build_feature_lookup(feature_df: pd.DataFrame) -> pd.DataFrame:
    lookup = feature_df.set_index("feature", drop=True)
    if not lookup.index.is_unique:
        print("WARNING: feature index is still not unique after preprocessing. Keeping first occurrence.")
        lookup = lookup[~lookup.index.duplicated(keep="first")]
    return lookup


def extract_cluster_matrix(row: pd.Series, feature_lookup: pd.DataFrame) -> Tuple[str, pd.DataFrame, List[str]]:
    cluster_id = row["ID"]
    members = parse_members(row["Members"])

    found = []
    missing = []
    for member in members:
        if member in feature_lookup.index:
            found.append(member)
        else:
            missing.append(member)

    if not found:
        return cluster_id, pd.DataFrame(), missing

    extracted = feature_lookup.loc[found].copy()
    extracted.insert(0, "feature", found)
    return cluster_id, extracted, missing


def calculate_sigma(cluster_df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Calculate the first right singular vector of a cluster feature x sample matrix.

    SVD vectors have arbitrary sign. To make UP/DOWN interpretation reproducible,
    orient the first component toward the mean sample profile of the module.
    """
    if cluster_df.empty:
        return None

    numeric_cols = cluster_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None

    X = cluster_df[numeric_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    keep_rows = np.linalg.norm(X, axis=1) > 0
    X = X[keep_rows]

    if X.shape[0] == 0 or X.shape[1] == 0:
        return None

    if X.shape[0] == 1:
        return X.flatten().astype(float)

    try:
        _, _, VT = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    if VT.shape[0] == 0:
        return None

    sigma = VT[0].astype(float)
    mean_profile = np.nanmean(X, axis=0)

    if np.linalg.norm(mean_profile) > 0 and float(np.dot(sigma, mean_profile)) < 0:
        sigma = -sigma
    elif np.linalg.norm(mean_profile) == 0:
        sigma = orient_vector(sigma)

    return sigma


def calc_metafingerprint(row: pd.Series, feature_lookup: pd.DataFrame) -> Tuple[Optional[dict], List[dict]]:
    cluster_id, cluster_df, missing = extract_cluster_matrix(row, feature_lookup)

    missing_records = [{"ID": cluster_id, "missing_feature": m} for m in missing]

    sigma = calculate_sigma(cluster_df)
    if sigma is None:
        return None, missing_records

    numeric_cols = cluster_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) != len(sigma):
        return None, missing_records

    record = {
        "ID": cluster_id,
        "Members": row["Members"],
        "n_features_found": int(cluster_df.shape[0]),
        "n_features_missing": int(len(missing)),
    }
    for col, value in zip(numeric_cols, sigma):
        record[col] = float(value)

    return record, missing_records


def calculate_fingerprints_for_matrix(feature_df: pd.DataFrame, clusters_frame: pd.DataFrame, desc: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate fingerprints for a single feature matrix (either all samples or a group)."""
    feature_lookup = build_feature_lookup(feature_df)
    results = []
    all_missing_records = []

    for _, row in tqdm(clusters_frame.iterrows(), total=clusters_frame.shape[0], desc=desc):
        fp, missing_records = calc_metafingerprint(row, feature_lookup)
        all_missing_records.extend(missing_records)
        if fp is not None:
            results.append(fp)

    fingerprints_df = pd.DataFrame(results)
    missing_df = pd.DataFrame(all_missing_records)
    return fingerprints_df, missing_df


def calculate_fingerprints_by_group(grouped_data: Dict[str, pd.DataFrame], clusters_frame: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    fingerprints_by_group = {}
    missing_frames = []

    for group, group_df in grouped_data.items():
        print("\n" + "=" * 70)
        print(f"Processing group: {group}")
        print("=" * 70)

        fp_df, missing_df = calculate_fingerprints_for_matrix(group_df, clusters_frame, desc=f"Processing Group {group}")
        if not missing_df.empty:
            missing_df["Group"] = group
            missing_frames.append(missing_df)

        if fp_df.empty:
            print(f"WARNING: No valid fingerprints were generated for group {group}.")
            continue

        fingerprints_by_group[group] = fp_df
        print(f"Generated fingerprints for group {group}: {fp_df.shape[0]} clusters")

    combined_missing = pd.concat(missing_frames, ignore_index=True) if missing_frames else pd.DataFrame()
    return fingerprints_by_group, combined_missing



def resolve_stratify_column(metadata: pd.DataFrame, requested: str) -> Optional[str]:
    """
    Resolve the metadata column used to stratify trajectories.

    requested='none' disables stratification.
    """
    if not requested or str(requested).lower() in {"none", "pooled", "all"}:
        return None

    if requested in metadata.columns:
        return requested

    lower_map = {str(c).lower(): c for c in metadata.columns}
    if str(requested).lower() in lower_map:
        return lower_map[str(requested).lower()]

    if str(requested).lower() == "condition":
        col = find_column(metadata, ["condition", "treatment", "group"])
        if col is not None:
            return col

    raise ValueError(
        f"Requested stratification column '{requested}' not found in metadata."
    )


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen-like standardized mean difference between two replicate groups.

    Positive values indicate higher values at the later time point (b).
    For groups with insufficient/zero variance, fall back to the raw mean
    difference so perfectly reproducible changes are not silently classified FLAT.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        return np.nan

    mean_diff = float(np.mean(b) - np.mean(a))

    if a.size >= 2 and b.size >= 2:
        va = np.var(a, ddof=1)
        vb = np.var(b, ddof=1)
        denom_df = a.size + b.size - 2
        if denom_df > 0:
            pooled_var = ((a.size - 1) * va + (b.size - 1) * vb) / denom_df
            if pooled_var > 0 and np.isfinite(pooled_var):
                return float(mean_diff / np.sqrt(pooled_var))

    return mean_diff


def transition_label_metric(
    value: float,
    metric: str,
    flat_threshold: float,
    effect_size_threshold: float,
) -> str:
    """Convert a transition statistic into UP/DOWN/FLAT."""
    if pd.isna(value):
        return "NA"

    threshold = (
        float(effect_size_threshold)
        if metric == "effect_size"
        else float(flat_threshold)
    )

    if value > threshold:
        return "UP"
    if value < -threshold:
        return "DOWN"
    return "FLAT"


def condition_time_sample_groups(
    metadata: pd.DataFrame,
    available_columns: List[str],
    ordered_times: List[str],
    time_col: str,
    stratify_col: Optional[str],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build Condition -> Time -> sample-list mapping.

    If stratify_col is None, a single condition named 'ALL' is used.
    """
    md = metadata.copy()
    md["sample"] = md["sample"].astype(str).str.strip()
    md[time_col] = md[time_col].astype(str).str.strip()
    avail = set(available_columns)

    if stratify_col is None:
        condition_values = ["ALL"]
    else:
        md[stratify_col] = md[stratify_col].astype(str).str.strip()
        condition_values = sorted(
            md[stratify_col].dropna().unique().tolist(),
            key=natural_key,
        )

    result = {}
    for condition in condition_values:
        if stratify_col is None:
            sub = md
        else:
            sub = md[md[stratify_col] == str(condition)]

        groups = {}
        complete = True
        for t in ordered_times:
            samples = sub.loc[sub[time_col] == str(t), "sample"].tolist()
            samples = [s for s in samples if s in avail]
            if not samples:
                complete = False
                break
            groups[str(t)] = samples

        if complete:
            result[str(condition)] = groups
        else:
            print(
                f"WARNING: condition '{condition}' does not contain all selected "
                f"time points {ordered_times}; skipping this condition."
            )

    if not result:
        raise ValueError(
            "No condition contains a complete set of selected time points."
        )

    return result



def response_timing_label(index_zero_based: int, n_timepoints: int) -> str:
    """Return a readable timing label for the strongest absolute response."""
    if n_timepoints <= 1:
        return "NA"
    if index_zero_based == 0:
        return "EARLY"
    if index_zero_based == n_timepoints - 1:
        return "LATE"
    if n_timepoints == 3 and index_zero_based == 1:
        return "MID"
    return f"T{index_zero_based + 1}"


# -----------------------------------------------------------------------------
# Trajectory descriptors
# -----------------------------------------------------------------------------


def resolve_time_column(metadata: pd.DataFrame, requested_time_col: str = "") -> str:
    if requested_time_col:
        if requested_time_col in metadata.columns:
            return requested_time_col
        matches = [c for c in metadata.columns if c.lower() == requested_time_col.lower()]
        if matches:
            return matches[0]
        raise ValueError(f"Requested time column '{requested_time_col}' not found in metadata.")

    col = find_column(metadata, ["time", "timepoint", "dpi", "Time", "Timepoint"])
    if col is None:
        raise ValueError("No time column found. Provide --time_column or include time/timepoint/dpi in metadata.")
    return col


def resolve_time_values(
    metadata: pd.DataFrame,
    time_col: str,
    time_values_arg: str,
    n_timepoints: int,
    max_timepoints: int = 100,
) -> List[str]:
    """
    Resolve the ordered time values used for trajectory analysis.

    MATE supports 2 or more measured time points. A trajectory with N time points
    contains exactly N-1 transitions. For example:

      2 time points -> UP
      3 time points -> UP-DOWN
      4 time points -> UP-DOWN-UP
      5 time points -> UP-DOWN-UP-FLAT
    """
    metadata = metadata.copy()
    metadata[time_col] = metadata[time_col].astype(str).str.strip()

    if time_values_arg:
        values = parse_comma_list(time_values_arg)
    else:
        values = sorted(
            metadata[time_col].dropna().unique().tolist(),
            key=natural_key,
        )

    if n_timepoints and n_timepoints > 0:
        if n_timepoints < 2:
            raise ValueError("--n_timepoints must be at least 2 for trajectory analysis.")
        if n_timepoints > max_timepoints:
            raise ValueError(
                f"MATE currently supports a maximum of {max_timepoints} time points; "
                f"--n_timepoints={n_timepoints} was requested."
            )
        if len(values) < n_timepoints:
            raise ValueError(
                f"--n_timepoints={n_timepoints}, but only {len(values)} time values "
                f"are available/selected: {values}"
            )
        values = values[:n_timepoints]

    if len(values) < 2:
        raise ValueError("Trajectory descriptors need at least two time points.")

    if len(values) > max_timepoints:
        raise ValueError(
            f"MATE currently supports a maximum of {max_timepoints} time points, "
            f"but {len(values)} were resolved: {values}. "
            "Use --time_values and/or --n_timepoints to select an ordered subset of time points."
        )

    print(
        f"Trajectory design: {len(values)} time points -> "
        f"{len(values) - 1} transition state(s) per module."
    )
    return values


def fingerprint_sample_columns(fingerprints: pd.DataFrame) -> List[str]:
    meta_cols = {"ID", "Members", "Group", "n_features_found", "n_features_missing", "Cluster", "Cluster_Color"}
    return [c for c in fingerprints.select_dtypes(include=[np.number]).columns if c not in meta_cols]


def timepoint_means_for_row(row: pd.Series, sample_groups: Dict[str, List[str]]) -> List[float]:
    means = []
    for _, samples in sample_groups.items():
        vals = pd.to_numeric(row[samples], errors="coerce").to_numpy(dtype=float) if samples else np.array([], dtype=float)
        if vals.size == 0 or np.all(np.isnan(vals)):
            means.append(np.nan)
        else:
            means.append(float(np.nanmean(vals)))
    return means


def transition_label(delta: float, threshold: float) -> str:
    if pd.isna(delta):
        return "NA"
    if delta > threshold:
        return "UP"
    if delta < -threshold:
        return "DOWN"
    return "FLAT"


def calculate_trajectory_descriptors(
    fingerprints: pd.DataFrame,
    metadata: pd.DataFrame,
    time_column: str = "",
    time_values: str = "",
    n_timepoints: int = 0,
    flat_threshold: float = 0.10,
    stratify_by: str = "condition",
    trajectory_metric: str = "effect_size",
    effect_size_threshold: float = 0.50,
    reference_condition: str = "",
) -> pd.DataFrame:
    """
    Calculate matched-reference treatment-response trajectories.

    For each non-reference treatment and each selected time point:
      effect_size -> standardized mean difference(reference, treatment)
      delta       -> treatment mean - reference mean

    The chronological trajectory is then calculated on these response values:
      Delta_T1_to_T2 = Response_T2 - Response_T1
      Delta_T2_to_T3 = Response_T3 - Response_T2

    ResponseStatePattern has N states relative to reference.
    Pattern has N-1 temporal transitions in response strength.
    """
    if fingerprints.empty:
        return pd.DataFrame()

    if not reference_condition:
        raise ValueError("--reference_condition is required.")

    time_col = resolve_time_column(metadata, time_column)
    ordered_times = resolve_time_values(
        metadata, time_col, time_values, n_timepoints, max_timepoints=100
    )
    stratify_col = resolve_stratify_column(metadata, stratify_by)

    if stratify_col is None:
        raise ValueError(
            "Reference-aware trajectories require a treatment/condition stratification column."
        )

    available_cols = [
        c for c in fingerprints.columns
        if c not in {
            "ID", "Members", "Group", "n_features_found", "n_features_missing",
            "Cluster", "Cluster_Color", "OriginalID", "Modality"
        }
    ]

    condition_groups = condition_time_sample_groups(
        metadata,
        available_cols,
        ordered_times,
        time_col,
        stratify_col,
    )

    reference_condition = str(reference_condition).strip()
    if reference_condition not in condition_groups:
        raise ValueError(
            f"Reference condition '{reference_condition}' is not available with all "
            f"selected time points. Complete conditions: {list(condition_groups.keys())}"
        )

    reference_groups = condition_groups[reference_condition]
    treatments = [c for c in condition_groups if c != reference_condition]

    if not treatments:
        raise ValueError("No non-reference treatment conditions remain.")

    print(
        f"Reference-aware trajectories: reference={reference_condition}; "
        f"treatments={treatments}"
    )

    records = []
    time_labels = [f"T{i+1}_{t}" for i, t in enumerate(ordered_times)]

    for condition in treatments:
        treatment_groups = condition_groups[condition]
        print(f"Trajectory treatment {condition} vs {reference_condition}")

        for _, row in fingerprints.iterrows():
            response_values = []
            response_states = []

            rec = {
                "ID": row["ID"],
                "Condition": str(condition),
                "ReferenceCondition": reference_condition,
                "StateID": f"{row['ID']}::{condition}",
                "Members": row.get("Members", ""),
                "n_features_found": row.get("n_features_found", np.nan),
                "n_features_missing": row.get("n_features_missing", np.nan),
                "NTimepoints": int(len(ordered_times)),
                "NTransitions": int(len(ordered_times) - 1),
                "TrajectoryMetric": trajectory_metric,
                "TrajectoryMode": "reference",
            }

            for i, t in enumerate(ordered_times):
                tr_samples = treatment_groups[str(t)]
                ref_samples = reference_groups[str(t)]

                tr_vals = pd.to_numeric(row[tr_samples], errors="coerce").to_numpy(dtype=float)
                ref_vals = pd.to_numeric(row[ref_samples], errors="coerce").to_numpy(dtype=float)
                tr_vals = tr_vals[np.isfinite(tr_vals)]
                ref_vals = ref_vals[np.isfinite(ref_vals)]

                tr_mean = float(np.mean(tr_vals)) if tr_vals.size else np.nan
                ref_mean = float(np.mean(ref_vals)) if ref_vals.size else np.nan
                raw_response = (
                    float(tr_mean - ref_mean)
                    if np.isfinite(tr_mean) and np.isfinite(ref_mean)
                    else np.nan
                )

                if trajectory_metric == "effect_size":
                    response = standardized_mean_difference(ref_vals, tr_vals)
                else:
                    response = raw_response

                response_values.append(response)
                label = time_labels[i]

                # Main time columns now represent treatment response vs reference.
                rec[label] = response

                # Audit columns.
                rec[f"TreatmentMean_{label}"] = tr_mean
                rec[f"ReferenceMean_{label}"] = ref_mean
                rec[f"RawResponse_{label}"] = raw_response
                rec[f"NrepTreatment_T{i+1}"] = int(len(tr_vals))
                rec[f"NrepReference_T{i+1}"] = int(len(ref_vals))

                if trajectory_metric == "effect_size":
                    rec[f"TreatmentVsReferenceEffectSize_{label}"] = response

                state = transition_label_metric(
                    response,
                    metric=trajectory_metric,
                    flat_threshold=flat_threshold,
                    effect_size_threshold=effect_size_threshold,
                )
                rec[f"ResponseState_{label}"] = state
                response_states.append(state)

            values = np.asarray(response_values, dtype=float)

            if np.all(np.isnan(values)):
                rec.update({
                    "Peak": np.nan,
                    "PeakIndex": np.nan,
                    "AbsPeak": np.nan,
                    "AbsPeakIndex": np.nan,
                    "ResponseTiming": "NA",
                    "DominantResponse": "NA",
                    "ResponseStatePattern": "-".join(response_states),
                    "Delta_first_last": np.nan,
                    "OverallTrend": "NA",
                    "DynamicRange": np.nan,
                    "MaxAbsDelta": np.nan,
                    "Variance": np.nan,
                    "AUC": np.nan,
                    "Pattern": "NA",
                })
                records.append(rec)
                continue

            peak_idx = int(np.nanargmax(values))
            abs_peak_idx = int(np.nanargmax(np.abs(values)))
            response_deltas = np.diff(values)

            rec["Peak"] = str(ordered_times[peak_idx])
            rec["PeakIndex"] = peak_idx + 1
            rec["AbsPeak"] = str(ordered_times[abs_peak_idx])
            rec["AbsPeakIndex"] = abs_peak_idx + 1
            rec["ResponseTiming"] = response_timing_label(abs_peak_idx, len(ordered_times))

            dominant_value = values[abs_peak_idx]
            rec["DominantResponse"] = transition_label_metric(
                dominant_value,
                metric=trajectory_metric,
                flat_threshold=flat_threshold,
                effect_size_threshold=effect_size_threshold,
            )
            rec["ResponseStatePattern"] = "-".join(response_states)

            pattern_parts = []
            for i, delta in enumerate(response_deltas):
                rec[f"Delta_T{i+1}_to_T{i+2}"] = float(delta)
                rec[f"TransitionStat_T{i+1}_to_T{i+2}"] = float(delta)

                if trajectory_metric == "effect_size":
                    rec[f"EffectSizeChange_T{i+1}_to_T{i+2}"] = float(delta)

                pattern_parts.append(
                    transition_label_metric(
                        delta,
                        metric=trajectory_metric,
                        flat_threshold=flat_threshold,
                        effect_size_threshold=effect_size_threshold,
                    )
                )

            rec["Delta_first_last"] = float(values[-1] - values[0])
            rec["OverallTransitionStat"] = rec["Delta_first_last"]
            rec["OverallTrend"] = transition_label_metric(
                rec["Delta_first_last"],
                metric=trajectory_metric,
                flat_threshold=flat_threshold,
                effect_size_threshold=effect_size_threshold,
            )

            rec["Max"] = float(np.nanmax(values))
            rec["Min"] = float(np.nanmin(values))
            rec["DynamicRange"] = float(np.nanmax(values) - np.nanmin(values))
            rec["MaxAbsDelta"] = (
                float(np.nanmax(np.abs(response_deltas)))
                if response_deltas.size else 0.0
            )
            rec["Variance"] = float(np.nanvar(values, ddof=0))
            rec["AUC"] = (
                float(np.trapezoid(values, dx=1.0))
                if hasattr(np, "trapezoid")
                else float(np.trapz(values, dx=1.0))
            )

            rec["Pattern"] = "-".join(pattern_parts)
            records.append(rec)

    out = pd.DataFrame(records)
    print(
        f"Calculated {out.shape[0]} matched-reference module x treatment trajectories "
        f"across {len(treatments)} treatment(s)."
    )
    return out



def calculate_pseudotime_trajectory_descriptors(
    fingerprints: pd.DataFrame,
    metadata: pd.DataFrame,
    time_column: str = "",
    time_values: str = "",
    n_timepoints: int = 0,
    flat_threshold: float = 0.10,
    stratify_by: str = "condition",
    trajectory_metric: str = "effect_size",
    effect_size_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Calculate reference-free trajectories along an ordered time/pseudotime axis.

    T1_*, T2_*, ... contain the mean module fingerprint at each observed
    time/pseudotime position.

    Pattern contains N-1 transitions:
      effect_size -> standardized mean difference between adjacent replicate groups
      delta       -> later mean - earlier mean

    No treatment-vs-mock response state is inferred in this mode.
    """
    if fingerprints.empty:
        return pd.DataFrame()

    time_col = resolve_time_column(metadata, time_column)
    ordered_times = resolve_time_values(
        metadata, time_col, time_values, n_timepoints, max_timepoints=100
    )
    stratify_col = resolve_stratify_column(metadata, stratify_by)

    available_cols = [
        c for c in fingerprints.columns
        if c not in {
            "ID", "Members", "Group", "n_features_found", "n_features_missing",
            "Cluster", "Cluster_Color", "OriginalID", "Modality"
        }
    ]

    condition_groups = condition_time_sample_groups(
        metadata, available_cols, ordered_times, time_col, stratify_col
    )

    print(
        f"Pseudotime/ordered-time trajectories: {len(condition_groups)} "
        f"stratum/condition(s), {len(ordered_times)} observed positions."
    )

    records = []
    time_labels = [f"T{i+1}_{t}" for i, t in enumerate(ordered_times)]

    for condition, sample_groups in condition_groups.items():
        print(
            f"Trajectory stratum {condition}: "
            f"{sum(len(v) for v in sample_groups.values())} samples"
        )

        for _, row in fingerprints.iterrows():
            time_means = []
            time_replicates = []

            rec = {
                "ID": row["ID"],
                "Condition": str(condition),
                "ReferenceCondition": "",
                "StateID": f"{row['ID']}::{condition}",
                "Members": row.get("Members", ""),
                "n_features_found": row.get("n_features_found", np.nan),
                "n_features_missing": row.get("n_features_missing", np.nan),
                "NTimepoints": int(len(ordered_times)),
                "NTransitions": int(len(ordered_times) - 1),
                "TrajectoryMetric": trajectory_metric,
                "TrajectoryMode": "pseudotime",
                "ResponseStatePattern": "NA",
                "DominantResponse": "NA",
            }

            for i, t in enumerate(ordered_times):
                samples = sample_groups[str(t)]
                vals = pd.to_numeric(
                    row[samples], errors="coerce"
                ).to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                time_replicates.append(vals)

                mean_val = float(np.mean(vals)) if vals.size else np.nan
                time_means.append(mean_val)

                label = time_labels[i]
                rec[label] = mean_val
                rec[f"ObservedMean_{label}"] = mean_val
                rec[f"Nrep_T{i+1}"] = int(len(vals))

            values = np.asarray(time_means, dtype=float)

            if np.all(np.isnan(values)):
                rec.update({
                    "Peak": np.nan,
                    "PeakIndex": np.nan,
                    "AbsPeak": np.nan,
                    "AbsPeakIndex": np.nan,
                    "TrajectoryTiming": "NA",
                    "ResponseTiming": "NA",
                    "Delta_first_last": np.nan,
                    "OverallTrend": "NA",
                    "DynamicRange": np.nan,
                    "MaxAbsDelta": np.nan,
                    "Variance": np.nan,
                    "AUC": np.nan,
                    "Pattern": "NA",
                })
                records.append(rec)
                continue

            peak_idx = int(np.nanargmax(values))
            abs_peak_idx = int(np.nanargmax(np.abs(values)))
            raw_deltas = np.diff(values)

            rec["Peak"] = str(ordered_times[peak_idx])
            rec["PeakIndex"] = peak_idx + 1
            rec["AbsPeak"] = str(ordered_times[abs_peak_idx])
            rec["AbsPeakIndex"] = abs_peak_idx + 1
            rec["TrajectoryTiming"] = response_timing_label(
                peak_idx, len(ordered_times)
            )
            # compatibility with existing embedding annotations
            rec["ResponseTiming"] = rec["TrajectoryTiming"]

            pattern_parts = []
            for i in range(len(ordered_times) - 1):
                raw_delta = float(raw_deltas[i])

                if trajectory_metric == "effect_size":
                    stat = standardized_mean_difference(
                        time_replicates[i], time_replicates[i + 1]
                    )
                else:
                    stat = raw_delta

                rec[f"Delta_T{i+1}_to_T{i+2}"] = raw_delta
                rec[f"TransitionStat_T{i+1}_to_T{i+2}"] = stat
                if trajectory_metric == "effect_size":
                    rec[f"EffectSize_T{i+1}_to_T{i+2}"] = stat

                pattern_parts.append(
                    transition_label_metric(
                        stat,
                        metric=trajectory_metric,
                        flat_threshold=flat_threshold,
                        effect_size_threshold=effect_size_threshold,
                    )
                )

            if trajectory_metric == "effect_size":
                overall_stat = standardized_mean_difference(
                    time_replicates[0], time_replicates[-1]
                )
            else:
                overall_stat = float(values[-1] - values[0])

            rec["Delta_first_last"] = float(values[-1] - values[0])
            rec["OverallTransitionStat"] = overall_stat
            rec["OverallTrend"] = transition_label_metric(
                overall_stat,
                metric=trajectory_metric,
                flat_threshold=flat_threshold,
                effect_size_threshold=effect_size_threshold,
            )
            rec["Max"] = float(np.nanmax(values))
            rec["Min"] = float(np.nanmin(values))
            rec["DynamicRange"] = float(np.nanmax(values) - np.nanmin(values))
            rec["MaxAbsDelta"] = (
                float(np.nanmax(np.abs(raw_deltas))) if raw_deltas.size else 0.0
            )
            rec["Variance"] = float(np.nanvar(values, ddof=0))
            rec["AUC"] = (
                float(np.trapezoid(values, dx=1.0))
                if hasattr(np, "trapezoid")
                else float(np.trapz(values, dx=1.0))
            )
            rec["Pattern"] = "-".join(pattern_parts)
            records.append(rec)

    out = pd.DataFrame(records)
    print(
        f"Calculated {out.shape[0]} reference-free module x stratum trajectories."
    )
    return out

def plot_trajectory_module_counts(
    descriptors: pd.DataFrame,
    output_prefix: str,
    plot_format: str = "png",
    include_na: bool = False,
) -> pd.DataFrame:
    """
    Count the number of modules assigned to each trajectory Pattern and create
    a bar plot.

    Parameters
    ----------
    descriptors : pd.DataFrame
        Trajectory descriptor table returned by calculate_trajectory_descriptors().
        Must contain a 'Pattern' column.
    output_prefix : str
        Prefix used for the output CSV and figure.
    plot_format : str
        Figure format: png, pdf, or svg.
    include_na : bool
        If False (default), trajectories labelled 'NA' are excluded.

    Returns
    -------
    pd.DataFrame
        Table with columns 'Pattern' and 'ModuleCount', sorted by decreasing
        module count.
    """
    if descriptors.empty:
        print("WARNING: trajectory descriptor table is empty; skipping trajectory count plot.")
        return pd.DataFrame()

    if "Pattern" not in descriptors.columns:
        print("WARNING: 'Pattern' column not found; skipping trajectory count plot.")
        return pd.DataFrame()

    pattern_series = descriptors["Pattern"].fillna("NA").astype(str)

    if not include_na:
        pattern_series = pattern_series[pattern_series.str.upper() != "NA"]

    if pattern_series.empty:
        print("WARNING: no valid trajectory patterns available for plotting.")
        return pd.DataFrame()

    counts = (
        pattern_series
        .value_counts()
        .rename_axis("Pattern")
        .reset_index(name="ModuleCount")
        .sort_values(["ModuleCount", "Pattern"], ascending=[False, True])
        .reset_index(drop=True)
    )

    counts_file = f"{output_prefix}_trajectory_pattern_counts.csv"
    counts.to_csv(counts_file, index=False)
    print(f"Saved trajectory pattern count table: {counts_file}")

    # Horizontal bars remain readable when many trajectory classes are present.
    fig_height = max(5.0, 0.45 * len(counts) + 1.5)
    plt.figure(figsize=(9, fig_height))

    ax = sns.barplot(
        data=counts,
        y="Pattern",
        x="ModuleCount",
        order=counts["Pattern"].tolist(),
    )

    ax.set_title("Number of modules associated with each trajectory")
    ax.set_xlabel("Number of modules")
    ax.set_ylabel("Trajectory pattern")
    ax.grid(axis="x", alpha=0.25)

    # Print the exact module count at the end of every bar.
    max_count = counts["ModuleCount"].max()
    offset = max(0.15, max_count * 0.01)

    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(
            width + offset,
            y,
            f"{int(width)}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlim(0, max_count + max(1.0, max_count * 0.12))
    plt.tight_layout()

    plot_file = f"{output_prefix}_trajectory_pattern_counts.{plot_format}"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory pattern bar plot: {plot_file}")

    return counts




def get_trajectory_time_columns(descriptors: pd.DataFrame) -> List[str]:
    """
    Return ordered timepoint-mean columns generated by trajectory calculation,
    e.g. T1_12, T2_24, T3_48.
    """
    cols = [c for c in descriptors.columns if re.match(r"^T\d+_", str(c))]
    return sorted(
        cols,
        key=lambda c: int(re.match(r"^T(\d+)_", str(c)).group(1))
    )


def trajectory_time_labels(time_cols: List[str]) -> List[str]:
    """Convert T1_12, T2_24, ... into display labels 12, 24, ..."""
    labels = []
    for col in time_cols:
        match = re.match(r"^T\d+_(.*)$", str(col))
        labels.append(match.group(1) if match else str(col))
    return labels


def plot_trajectory_profiles(
    descriptors: pd.DataFrame,
    output_prefix: str,
    plot_format: str = "png",
    include_na: bool = False,
    max_patterns: int = 16,
) -> None:
    """
    Plot actual temporal profiles for each trajectory class.

    Each facet contains:
      - individual module trajectories as faint lines
      - the mean trajectory for the class as a thicker line

    The plot uses the timepoint means already present in the trajectory
    descriptor table; it does not affect trajectory assignment or embeddings.
    """
    if descriptors.empty or "Pattern" not in descriptors.columns:
        print("WARNING: no trajectory descriptors/Pattern column; skipping trajectory profile plot.")
        return

    time_cols = get_trajectory_time_columns(descriptors)
    if len(time_cols) < 2:
        print("WARNING: fewer than two trajectory time columns; skipping trajectory profile plot.")
        return

    tmp = descriptors.copy()
    tmp["Pattern"] = tmp["Pattern"].fillna("NA").astype(str)

    if not include_na:
        tmp = tmp[tmp["Pattern"].str.upper() != "NA"].copy()

    if tmp.empty:
        print("WARNING: no valid trajectory patterns available for trajectory profile plot.")
        return

    pattern_counts = tmp["Pattern"].value_counts()
    patterns = pattern_counts.index.tolist()

    if len(patterns) > max_patterns:
        print(
            f"WARNING: {len(patterns)} trajectory patterns found. "
            f"Plotting the {max_patterns} most frequent patterns."
        )
        patterns = patterns[:max_patterns]
        tmp = tmp[tmp["Pattern"].isin(patterns)].copy()

    n_patterns = len(patterns)
    ncols = min(4, max(1, n_patterns))
    nrows = int(np.ceil(n_patterns / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.3 * nrows),
        squeeze=False,
        sharex=False,
        sharey=True,
    )

    x = np.arange(len(time_cols))
    xlabels = trajectory_time_labels(time_cols)

    for ax, pattern in zip(axes.flat, patterns):
        subset = tmp[tmp["Pattern"] == pattern]
        Y = subset[time_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        for row in Y:
            ax.plot(x, row, alpha=0.14, linewidth=0.8)

        mean_profile = np.nanmean(Y, axis=0)
        ax.plot(
            x,
            mean_profile,
            linewidth=2.8,
            marker="o",
            markersize=5,
            label="Mean",
        )

        ax.axhline(0, linewidth=0.7, alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_title(f"{pattern}  (n={subset.shape[0]})")
        ax.set_xlabel("Time")
        mode_values = subset.get(
            "TrajectoryMode", pd.Series(["reference"])
        ).astype(str)
        mode = mode_values.iloc[0] if len(mode_values) else "reference"
        ax.set_ylabel(
            "Module fingerprint trajectory"
            if mode == "pseudotime"
            else "Treatment response vs reference"
        )
        ax.grid(True, alpha=0.2)

    for ax in axes.flat[n_patterns:]:
        ax.axis("off")

    trajectory_modes = tmp.get(
        "TrajectoryMode", pd.Series(["reference"])
    ).astype(str).unique()
    title = (
        "Reference-free module trajectories by temporal pattern"
        if len(trajectory_modes) == 1 and trajectory_modes[0] == "pseudotime"
        else "Reference-aware module trajectories by temporal response pattern"
    )
    fig.suptitle(title, y=1.01, fontsize=14)
    fig.tight_layout()

    output_file = f"{output_prefix}_trajectory_profiles.{plot_format}"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory profile plot: {output_file}")


def _permanova_pseudo_f(distance_matrix: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    One-factor PERMANOVA pseudo-F and R² from a precomputed distance matrix.

    Uses Gower centering of the squared distance matrix and a one-factor
    projection matrix. The grouping variable is the trajectory Pattern.
    """
    D = np.asarray(distance_matrix, dtype=float)
    labels = np.asarray(labels)

    n = D.shape[0]
    groups = pd.unique(labels)
    g = len(groups)

    if n < 3 or g < 2 or n <= g:
        return np.nan, np.nan

    A = -0.5 * (D ** 2)
    H = np.eye(n) - np.ones((n, n)) / n
    G = H @ A @ H

    X = np.zeros((n, g), dtype=float)
    group_to_col = {grp: i for i, grp in enumerate(groups)}
    for i, grp in enumerate(labels):
        X[i, group_to_col[grp]] = 1.0

    Hx = X @ np.linalg.pinv(X.T @ X) @ X.T

    ss_between = float(np.trace(Hx @ G))
    ss_total = float(np.trace(G))
    ss_within = ss_total - ss_between

    df_between = g - 1
    df_within = n - g

    if df_between <= 0 or df_within <= 0 or ss_within <= 0:
        return np.nan, np.nan

    pseudo_f = (ss_between / df_between) / (ss_within / df_within)
    r2 = ss_between / ss_total if ss_total > 0 else np.nan
    return float(pseudo_f), float(r2)


def trajectory_separation_statistics(
    fingerprints: pd.DataFrame,
    descriptors: pd.DataFrame,
    output_prefix: str,
    metric: str = "euclidean",
    permutations: int = 999,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Quantify whether modules with the same trajectory Pattern are more similar
    in the ORIGINAL scaled metafingerprint space than modules with different
    trajectory Patterns.

    Outputs:
      1. mean within-pattern and between-pattern distances
      2. observed separation = mean_between - mean_within
      3. permutation p-value for the distance separation
      4. one-factor PERMANOVA pseudo-F, R², and permutation p-value

    Importantly, PCA/UMAP/t-SNE coordinates are NOT used for this test.
    """
    if fingerprints.empty or descriptors.empty or "Pattern" not in descriptors.columns:
        print("WARNING: insufficient data for trajectory separation statistics.")
        return pd.DataFrame()

    annotation = descriptors[["ID", "Pattern"]].copy()
    annotation["Pattern"] = annotation["Pattern"].fillna("NA").astype(str)
    annotation = annotation[annotation["Pattern"].str.upper() != "NA"].copy()

    # For the current non-stratified trajectory workflow there should be one row
    # per ID. If duplicates exist, retain the first to avoid duplicating modules.
    if annotation["ID"].duplicated().any():
        print(
            "WARNING: multiple trajectory rows found for some module IDs. "
            "Trajectory-separation statistics use the first Pattern per ID."
        )
        annotation = annotation.drop_duplicates("ID", keep="first")

    merged = fingerprints[["ID"]].merge(annotation, on="ID", how="inner")
    if merged.shape[0] < 4 or merged["Pattern"].nunique() < 2:
        print("WARNING: not enough annotated modules/patterns for trajectory separation statistics.")
        return pd.DataFrame()

    fp_indexed = fingerprints.set_index("ID")
    fp_for_test = fp_indexed.loc[merged["ID"]].reset_index()

    X_scaled, used_cols = prepare_embedding_matrix(fp_for_test)
    labels = merged["Pattern"].to_numpy()

    try:
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X_scaled, metric=metric)
    except Exception as exc:
        print(
            f"WARNING: distance metric '{metric}' failed for trajectory statistics ({exc}). "
            "Falling back to euclidean."
        )
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X_scaled, metric="euclidean")
        metric = "euclidean"

    iu = np.triu_indices_from(D, k=1)
    pair_dist = D[iu]
    same = labels[iu[0]] == labels[iu[1]]

    within = pair_dist[same]
    between = pair_dist[~same]

    if within.size == 0 or between.size == 0:
        print("WARNING: no within- or between-pattern distance pairs; skipping trajectory statistics.")
        return pd.DataFrame()

    mean_within = float(np.mean(within))
    mean_between = float(np.mean(between))
    separation = mean_between - mean_within

    pseudo_f, r2 = _permanova_pseudo_f(D, labels)

    rng = np.random.default_rng(random_state)
    n_perm = max(0, int(permutations))
    perm_separation = []
    perm_f = []

    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_same = perm_labels[iu[0]] == perm_labels[iu[1]]

        if np.any(perm_same) and np.any(~perm_same):
            perm_within = float(np.mean(pair_dist[perm_same]))
            perm_between = float(np.mean(pair_dist[~perm_same]))
            perm_separation.append(perm_between - perm_within)

        pf, _ = _permanova_pseudo_f(D, perm_labels)
        if np.isfinite(pf):
            perm_f.append(pf)

    separation_p = (
        (1 + np.sum(np.asarray(perm_separation) >= separation)) /
        (1 + len(perm_separation))
        if perm_separation else np.nan
    )

    permanova_p = (
        (1 + np.sum(np.asarray(perm_f) >= pseudo_f)) /
        (1 + len(perm_f))
        if perm_f and np.isfinite(pseudo_f) else np.nan
    )

    result = pd.DataFrame([{
        "n_modules": int(merged.shape[0]),
        "n_trajectory_patterns": int(merged["Pattern"].nunique()),
        "distance_metric": metric,
        "n_fingerprint_columns": int(len(used_cols)),
        "mean_within_pattern_distance": mean_within,
        "mean_between_pattern_distance": mean_between,
        "between_minus_within_distance": separation,
        "distance_permutation_pvalue": separation_p,
        "PERMANOVA_pseudo_F": pseudo_f,
        "PERMANOVA_R2": r2,
        "PERMANOVA_pvalue": permanova_p,
        "permutations": n_perm,
    }])

    output_file = f"{output_prefix}_trajectory_separation_statistics.csv"
    result.to_csv(output_file, index=False)
    print(f"Saved trajectory separation statistics: {output_file}")

    pair_summary = pd.DataFrame({
        "Comparison": ["Within trajectory", "Between trajectories"],
        "MeanDistance": [mean_within, mean_between],
        "NumberOfPairs": [int(within.size), int(between.size)],
    })
    pair_file = f"{output_prefix}_trajectory_distance_summary.csv"
    pair_summary.to_csv(pair_file, index=False)
    print(f"Saved within/between trajectory distance summary: {pair_file}")

    return result


# -----------------------------------------------------------------------------
# Continuous-time trajectory reconstruction (MATE v2.4)
# -----------------------------------------------------------------------------


def _trajectory_response_threshold(
    trajectory_metric: str,
    flat_threshold: float,
    effect_size_threshold: float,
) -> float:
    return (
        float(effect_size_threshold)
        if trajectory_metric == "effect_size"
        else float(flat_threshold)
    )


def _select_continuous_model(
    requested: str,
    n_observed: int,
    gam_min_timepoints: int = 5,
) -> str:
    """
    Select a scientifically conservative continuous-time model.

    auto:
      <=4 REAL measured time points -> linear interpolation
      >=gam_min_timepoints          -> low-complexity GAM
    """
    requested = str(requested).lower()
    if requested == "auto":
        return "gam" if n_observed >= int(gam_min_timepoints) else "linear"

    if requested == "gam" and n_observed < int(gam_min_timepoints):
        raise ValueError(
            f"GAM requested with only {n_observed} real measured time points. "
            f"MATE requires at least {gam_min_timepoints}. Use --trajectory_model "
            "linear/pchip or provide a denser time course."
        )

    if requested == "pchip" and n_observed < 3:
        print("WARNING: PCHIP needs at least 3 measured points; using linear interpolation.")
        return "linear"

    return requested


def _make_continuous_grids(
    observed_times: np.ndarray,
    step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      regular_grid  - equally spaced model grid used for lag/embedding
      output_grid   - union of regular grid and original measured times for auditing
    """
    times = np.asarray(observed_times, dtype=float)
    times = np.sort(np.unique(times[np.isfinite(times)]))
    if times.size < 2:
        raise ValueError("Continuous trajectories require at least two numeric time points.")
    if step <= 0:
        raise ValueError("--continuous_grid_step must be > 0.")

    start = float(times[0])
    end = float(times[-1])
    regular = np.arange(start, end + step * 0.5, step, dtype=float)
    if regular.size == 0 or regular[-1] < end - 1e-9:
        regular = np.append(regular, end)
    elif regular[-1] > end + 1e-9:
        regular = regular[regular <= end + 1e-9]
        if regular.size == 0 or regular[-1] < end - 1e-9:
            regular = np.append(regular, end)

    # Remove numerical duplicates while retaining the real end point.
    regular = np.asarray(sorted(set(np.round(regular, 12))), dtype=float)
    output_grid = np.asarray(
        sorted(set(np.round(np.concatenate([regular, times]), 12))),
        dtype=float,
    )
    return regular, output_grid


def _fit_continuous_curve(
    observed_times: np.ndarray,
    observed_values: np.ndarray,
    prediction_times: np.ndarray,
    model_name: str,
    gam_splines: int = 4,
    gam_alpha: float = 1.0,
) -> Tuple[np.ndarray, str]:
    """Fit one continuous response trajectory and predict on prediction_times."""
    t = np.asarray(observed_times, dtype=float)
    y = np.asarray(observed_values, dtype=float)
    xnew = np.asarray(prediction_times, dtype=float)

    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < 2:
        return np.full(xnew.shape, np.nan, dtype=float), "insufficient"

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # Duplicate time values should not normally exist, but average them safely.
    if len(np.unique(t)) != len(t):
        tmp = pd.DataFrame({"t": t, "y": y}).groupby("t", as_index=False)["y"].mean()
        t = tmp["t"].to_numpy(dtype=float)
        y = tmp["y"].to_numpy(dtype=float)

    if model_name == "linear":
        return np.interp(xnew, t, y), "linear"

    if model_name == "pchip":
        try:
            model = PchipInterpolator(t, y, extrapolate=False)
            return np.asarray(model(xnew), dtype=float), "pchip"
        except Exception as exc:
            print(f"WARNING: PCHIP fit failed ({exc}); falling back to linear.")
            return np.interp(xnew, t, y), "linear_fallback"

    if model_name == "gam":
        if GLMGam is None or BSplines is None:
            raise ImportError(
                "statsmodels GAM support is unavailable. Install statsmodels or use "
                "--trajectory_model linear/pchip."
            )
        # Cubic smooth with intentionally low basis complexity for omics time courses.
        degree = 3
        df = int(max(degree + 1, gam_splines))
        # More spline basis functions than observations are not useful here.
        df = min(df, max(degree + 1, len(t) - 1))
        try:
            smoother = BSplines(t[:, None], df=[df], degree=[degree])
            model = GLMGam(
                y,
                exog=np.ones((len(y), 1), dtype=float),
                smoother=smoother,
                alpha=np.asarray([float(gam_alpha)], dtype=float),
            )
            result = model.fit()
            pred = result.predict(
                exog=np.ones((len(xnew), 1), dtype=float),
                exog_smooth=xnew[:, None],
            )
            return np.asarray(pred, dtype=float), "gam"
        except Exception as exc:
            print(f"WARNING: GAM fit failed ({exc}); falling back to linear.")
            return np.interp(xnew, t, y), "linear_fallback"

    raise ValueError(f"Unknown continuous trajectory model: {model_name}")


def _response_from_replicates(
    reference_values: np.ndarray,
    treatment_values: np.ndarray,
    trajectory_metric: str,
) -> float:
    ref = np.asarray(reference_values, dtype=float)
    tr = np.asarray(treatment_values, dtype=float)
    ref = ref[np.isfinite(ref)]
    tr = tr[np.isfinite(tr)]
    if ref.size == 0 or tr.size == 0:
        return np.nan

    if trajectory_metric == "effect_size":
        # standardized_mean_difference(a, b) returns mean(b)-mean(a), so the
        # sign is treatment relative to reference.
        return standardized_mean_difference(ref, tr)
    return float(np.mean(tr) - np.mean(ref))


def _bootstrap_continuous_curve(
    fingerprint_row: pd.Series,
    ordered_times: List[str],
    observed_numeric_times: np.ndarray,
    prediction_times: np.ndarray,
    treatment_groups: Dict[str, List[str]],
    reference_groups: Dict[str, List[str]],
    trajectory_metric: str,
    model_name: str,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
    gam_splines: int,
    gam_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Bootstrap biological replicate samples, recalculate matched-reference
    responses, and refit the trajectory. Interpolated grid points are never
    resampled as though they were independent observations.
    """
    if n_bootstrap <= 0:
        n = len(prediction_times)
        return np.full(n, np.nan), np.full(n, np.nan), 0

    curves = []
    for _ in range(int(n_bootstrap)):
        responses = []
        for t in ordered_times:
            tr = pd.to_numeric(
                fingerprint_row[treatment_groups[str(t)]], errors="coerce"
            ).to_numpy(dtype=float)
            ref = pd.to_numeric(
                fingerprint_row[reference_groups[str(t)]], errors="coerce"
            ).to_numpy(dtype=float)
            tr = tr[np.isfinite(tr)]
            ref = ref[np.isfinite(ref)]

            if tr.size == 0 or ref.size == 0:
                responses.append(np.nan)
                continue

            tr_b = rng.choice(tr, size=tr.size, replace=True)
            ref_b = rng.choice(ref, size=ref.size, replace=True)
            responses.append(
                _response_from_replicates(ref_b, tr_b, trajectory_metric)
            )

        response_array = np.asarray(responses, dtype=float)
        try:
            curve, _ = _fit_continuous_curve(
                observed_numeric_times,
                response_array,
                prediction_times,
                model_name=model_name,
                gam_splines=gam_splines,
                gam_alpha=gam_alpha,
            )
            if np.isfinite(curve).sum() >= 2:
                curves.append(curve)
        except Exception:
            continue

    if not curves:
        n = len(prediction_times)
        return np.full(n, np.nan), np.full(n, np.nan), 0

    arr = np.vstack(curves)
    alpha = max(0.0, min(0.5, (1.0 - float(ci_level)) / 2.0))
    lower = np.nanquantile(arr, alpha, axis=0)
    upper = np.nanquantile(arr, 1.0 - alpha, axis=0)
    return lower, upper, int(arr.shape[0])



def _bootstrap_pseudotime_curve(
    fingerprint_row: pd.Series,
    ordered_times: List[str],
    observed_numeric_times: np.ndarray,
    prediction_times: np.ndarray,
    sample_groups: Dict[str, List[str]],
    model_name: str,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
    gam_splines: int,
    gam_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Bootstrap biological replicate samples within each observed time/pseudotime
    position and refit the continuous curve.
    """
    if n_bootstrap <= 0:
        n = len(prediction_times)
        return np.full(n, np.nan), np.full(n, np.nan), 0

    curves = []
    for _ in range(int(n_bootstrap)):
        means = []

        for t in ordered_times:
            vals = pd.to_numeric(
                fingerprint_row[sample_groups[str(t)]], errors="coerce"
            ).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                means.append(np.nan)
                continue

            vals_b = rng.choice(vals, size=vals.size, replace=True)
            means.append(float(np.mean(vals_b)))

        try:
            curve, _ = _fit_continuous_curve(
                observed_numeric_times,
                np.asarray(means, dtype=float),
                prediction_times,
                model_name=model_name,
                gam_splines=gam_splines,
                gam_alpha=gam_alpha,
            )
            if np.isfinite(curve).sum() >= 2:
                curves.append(curve)
        except Exception:
            continue

    if not curves:
        n = len(prediction_times)
        return np.full(n, np.nan), np.full(n, np.nan), 0

    arr = np.vstack(curves)
    alpha = (1.0 - float(ci_level)) / 2.0
    lower = np.nanquantile(arr, alpha, axis=0)
    upper = np.nanquantile(arr, 1.0 - alpha, axis=0)
    return lower, upper, arr.shape[0]


def build_continuous_trajectories(
    fingerprints: pd.DataFrame,
    descriptors: pd.DataFrame,
    metadata: pd.DataFrame,
    modality: str,
    time_column: str,
    time_values: str,
    n_timepoints: int,
    stratify_by: str,
    reference_condition: str,
    trajectory_mode: str,
    trajectory_metric: str,
    flat_threshold: float,
    effect_size_threshold: float,
    trajectory_model: str,
    grid_step: float,
    n_bootstrap: int,
    ci_level: float,
    random_seed: int,
    gam_min_timepoints: int,
    gam_splines: int,
    gam_alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Reconstruct continuous curves for either MATE trajectory mode.

    reference:
        treatment-vs-matched-reference response trajectory.

    pseudotime:
        direct module fingerprint trajectory along the supplied ordered
        chronological/pseudotime axis; no mock/reference is required.
    """
    if descriptors.empty:
        return pd.DataFrame(), pd.DataFrame(), np.array([])

    trajectory_mode = str(trajectory_mode).strip().lower()
    if trajectory_mode not in {"reference", "pseudotime"}:
        raise ValueError(f"Unknown trajectory mode: {trajectory_mode}")

    time_col = resolve_time_column(metadata, time_column)
    ordered_times = resolve_time_values(
        metadata, time_col, time_values, n_timepoints, max_timepoints=100
    )
    observed_numeric = _numeric_times(ordered_times)

    if observed_numeric is None:
        raise ValueError(
            "Continuous modelling requires numeric time/pseudotime labels."
        )
    if not np.all(np.diff(observed_numeric) > 0):
        raise ValueError(
            "Continuous modelling requires strictly increasing time/pseudotime values."
        )

    regular_grid, output_grid = _make_continuous_grids(
        observed_numeric, float(grid_step)
    )
    model_selected = _select_continuous_model(
        trajectory_model,
        len(observed_numeric),
        gam_min_timepoints=gam_min_timepoints,
    )

    print(
        f"{modality} continuous trajectories: mode={trajectory_mode}; "
        f"{len(observed_numeric)} REAL observed positions; model={model_selected}; "
        f"grid step={grid_step}; bootstrap={n_bootstrap}."
    )

    stratify_col = resolve_stratify_column(metadata, stratify_by)
    sample_cols = fingerprint_sample_columns(fingerprints)
    groups = condition_time_sample_groups(
        metadata, sample_cols, ordered_times, time_col, stratify_col
    )

    reference_condition = str(reference_condition or "").strip()
    ref_groups = None

    if trajectory_mode == "reference":
        if not reference_condition:
            raise ValueError(
                "--reference_condition is required when --trajectory_mode reference."
            )
        if reference_condition not in groups:
            raise ValueError(
                f"Reference condition '{reference_condition}' is unavailable for "
                "continuous modelling."
            )
        ref_groups = groups[reference_condition]

    fp_lookup = fingerprints.set_index("ID", drop=False)
    time_cols = get_trajectory_time_columns(descriptors)

    if len(time_cols) != len(observed_numeric):
        raise ValueError(
            "Descriptor time columns do not match the selected observed positions."
        )

    threshold = _trajectory_response_threshold(
        trajectory_metric, flat_threshold, effect_size_threshold
    )
    rng = np.random.default_rng(int(random_seed))
    long_records = []
    summary_records = []

    for _, desc_row in tqdm(
        descriptors.iterrows(),
        total=descriptors.shape[0],
        desc=f"{modality} continuous modelling",
    ):
        state_id = str(desc_row["StateID"])
        module_id = desc_row["ID"]
        condition = str(desc_row["Condition"])

        if module_id not in fp_lookup.index or condition not in groups:
            continue

        observed_values = pd.to_numeric(
            desc_row[time_cols], errors="coerce"
        ).to_numpy(dtype=float)

        fitted, model_used = _fit_continuous_curve(
            observed_numeric,
            observed_values,
            output_grid,
            model_name=model_selected,
            gam_splines=gam_splines,
            gam_alpha=gam_alpha,
        )

        fp_row = fp_lookup.loc[module_id]
        if isinstance(fp_row, pd.DataFrame):
            fp_row = fp_row.iloc[0]

        if trajectory_mode == "reference":
            ci_lower, ci_upper, n_boot_ok = _bootstrap_continuous_curve(
                fp_row,
                ordered_times,
                observed_numeric,
                output_grid,
                groups[condition],
                ref_groups,
                trajectory_metric,
                model_selected,
                int(n_bootstrap),
                float(ci_level),
                rng,
                int(gam_splines),
                float(gam_alpha),
            )
        else:
            ci_lower, ci_upper, n_boot_ok = _bootstrap_pseudotime_curve(
                fp_row,
                ordered_times,
                observed_numeric,
                output_grid,
                groups[condition],
                model_selected,
                int(n_bootstrap),
                float(ci_level),
                rng,
                int(gam_splines),
                float(gam_alpha),
            )

        regular_pred, _ = _fit_continuous_curve(
            observed_numeric,
            observed_values,
            regular_grid,
            model_name=model_selected,
            gam_splines=gam_splines,
            gam_alpha=gam_alpha,
        )

        finite = np.isfinite(regular_pred)
        if finite.sum() < 2:
            continue

        rp = regular_pred[finite]
        rt = regular_grid[finite]
        peak_i = int(np.argmax(rp))
        abs_peak_i = int(np.argmax(np.abs(rp)))
        slope = np.gradient(rp, rt)

        # These are biological treatment-response descriptors and therefore
        # intentionally remain undefined when no reference condition exists.
        dominant_sign = "NA"
        onset_time = np.nan
        end_time = np.nan
        onset_left_censored = False
        duration = np.nan

        if trajectory_mode == "reference":
            dominant_sign = (
                "UP" if rp[abs_peak_i] > threshold
                else "DOWN" if rp[abs_peak_i] < -threshold
                else "FLAT"
            )

            if dominant_sign == "UP":
                active = rp > threshold
            elif dominant_sign == "DOWN":
                active = rp < -threshold
            else:
                active = np.zeros(len(rp), dtype=bool)

            if active.any():
                inds = np.where(active)[0]
                onset_time = float(rt[inds[0]])
                end_time = float(rt[inds[-1]])
                onset_left_censored = bool(inds[0] == 0)
                duration = float(end_time - onset_time)

        mean_ci_width = (
            float(np.nanmean(ci_upper - ci_lower))
            if np.isfinite(ci_upper - ci_lower).any()
            else np.nan
        )

        summary_records.append({
            "ID": module_id,
            "StateID": state_id,
            "Condition": condition,
            "ReferenceCondition": (
                reference_condition if trajectory_mode == "reference" else ""
            ),
            "TrajectoryMode": trajectory_mode,
            "Modality": modality,
            "NObservedTimepoints": int(len(observed_numeric)),
            "TrajectoryModelRequested": trajectory_model,
            "TrajectoryModelUsed": model_used,
            "ContinuousGridStep": float(grid_step),
            "BootstrapRequested": int(n_bootstrap),
            "BootstrapSuccessful": int(n_boot_ok),
            "ContinuousPeakTime": float(rt[peak_i]),
            "ContinuousPeakResponse": float(rp[peak_i]),
            "ContinuousAbsPeakTime": float(rt[abs_peak_i]),
            "ContinuousAbsPeakResponse": float(rp[abs_peak_i]),
            "ContinuousDominantResponse": dominant_sign,
            "ContinuousResponseOnsetTime": onset_time,
            "ContinuousResponseEndTime": end_time,
            "ContinuousResponseDuration": duration,
            "OnsetLeftCensored": onset_left_censored,
            "ContinuousMaxSlopeTime": float(rt[int(np.nanargmax(slope))]),
            "ContinuousMaxSlope": float(np.nanmax(slope)),
            "ContinuousMinSlopeTime": float(rt[int(np.nanargmin(slope))]),
            "ContinuousMinSlope": float(np.nanmin(slope)),
            "MeanCIWidth": mean_ci_width,
        })

        for j, t in enumerate(output_grid):
            obs_matches = np.where(
                np.isclose(observed_numeric, t, rtol=0, atol=1e-9)
            )[0]
            regular_match = bool(
                np.any(np.isclose(regular_grid, t, rtol=0, atol=1e-9))
            )
            observed_here = bool(len(obs_matches) > 0)
            observed_value = (
                float(observed_values[int(obs_matches[0])])
                if observed_here else np.nan
            )
            predicted_value = (
                float(fitted[j]) if np.isfinite(fitted[j]) else np.nan
            )

            long_records.append({
                "ID": module_id,
                "StateID": state_id,
                "Condition": condition,
                "ReferenceCondition": (
                    reference_condition if trajectory_mode == "reference" else ""
                ),
                "TrajectoryMode": trajectory_mode,
                "Modality": modality,
                "Time": float(t),
                "ObservedTimepoint": observed_here,
                "RegularGridPoint": regular_match,
                "ObservedTrajectoryValue": observed_value,
                "PredictedTrajectoryValue": predicted_value,
                # aliases retained so v2.3 downstream code remains compatible
                "ObservedResponse": observed_value,
                "PredictedResponse": predicted_value,
                "CI_Lower": (
                    float(ci_lower[j]) if np.isfinite(ci_lower[j]) else np.nan
                ),
                "CI_Upper": (
                    float(ci_upper[j]) if np.isfinite(ci_upper[j]) else np.nan
                ),
                "TrajectoryModel": model_used,
                "NObservedTimepoints": int(len(observed_numeric)),
            })

    return pd.DataFrame(long_records), pd.DataFrame(summary_records), regular_grid


def _continuous_profiles(
    continuous_long: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return StateID -> regular-grid times and predicted values."""
    times = {}
    profiles = {}
    if continuous_long.empty:
        return times, profiles
    regular = continuous_long[continuous_long["RegularGridPoint"] == True].copy()  # noqa: E712
    for state, sub in regular.groupby("StateID", sort=False):
        sub = sub.sort_values("Time")
        times[str(state)] = sub["Time"].to_numpy(dtype=float)
        profiles[str(state)] = sub["PredictedResponse"].to_numpy(dtype=float)
    return times, profiles


def _continuous_shift_similarity(
    times: np.ndarray,
    rna: np.ndarray,
    met: np.ndarray,
    lag_time: float,
) -> Tuple[float, float, float, int]:
    """
    Compare RNA(t) with MET(t + lag_time).

    Positive lag_time means RNA precedes metabolomics.
    Returns cosine, RMSE, Pearson-style cross-correlation, n aligned grid points.
    """
    t = np.asarray(times, dtype=float)
    r = np.asarray(rna, dtype=float)
    m = np.asarray(met, dtype=float)
    mask = np.isfinite(t) & np.isfinite(r) & np.isfinite(m)
    t = t[mask]
    r = r[mask]
    m = m[mask]
    if len(t) < 3:
        return np.nan, np.nan, np.nan, 0

    shifted = t + float(lag_time)
    valid = (shifted >= t.min() - 1e-12) & (shifted <= t.max() + 1e-12)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan, int(valid.sum())

    a = r[valid]
    b = np.interp(shifted[valid], t, m)
    good = np.isfinite(a) & np.isfinite(b)
    a = a[good]
    b = b[good]
    if len(a) < 3:
        return np.nan, np.nan, np.nan, int(len(a))

    an = _normalize_profile_for_lag(a)
    bn = _normalize_profile_for_lag(b)
    denom = np.linalg.norm(an) * np.linalg.norm(bn)
    cosine = float(np.dot(an, bn) / denom) if denom > 0 else np.nan
    rmse = float(np.sqrt(np.mean((an - bn) ** 2)))
    if np.std(a) > 0 and np.std(b) > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = np.nan
    return cosine, rmse, corr, int(len(a))


def continuous_cross_omics_lag_analysis(
    transcript_long: pd.DataFrame,
    metabolite_long: pd.DataFrame,
    transcript_desc: pd.DataFrame,
    metabolite_desc: pd.DataFrame,
    transcript_cont_desc: pd.DataFrame,
    metabolite_cont_desc: pd.DataFrame,
    output_prefix: str,
    lag_max: Optional[float],
    lag_step: Optional[float],
    default_grid_step: float,
    lag_direction: str,
    top_n: int,
    min_similarity: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-omics lag search on the regular modelled temporal grid."""
    r_times, r_profiles = _continuous_profiles(transcript_long)
    m_times, m_profiles = _continuous_profiles(metabolite_long)
    if not r_profiles or not m_profiles:
        return pd.DataFrame(), pd.DataFrame()

    r_ann = transcript_desc.set_index("StateID")
    m_ann = metabolite_desc.set_index("StateID")
    r_cont = transcript_cont_desc.set_index("StateID")
    m_cont = metabolite_cont_desc.set_index("StateID")

    common_states = set(r_profiles).union(set(m_profiles))
    # Derive global grid limits from any state; all continuous trajectories use
    # the same selected time grid in one MATE run.
    example_state = next(iter(r_profiles))
    grid = r_times[example_state]
    if len(grid) < 3:
        return pd.DataFrame(), pd.DataFrame()

    span = float(grid[-1] - grid[0])
    max_lag = float(lag_max) if lag_max is not None else span / 2.0
    step = float(lag_step) if lag_step is not None else float(default_grid_step)
    if step <= 0 or max_lag < 0:
        raise ValueError("Continuous lag step must be >0 and lag max must be >=0.")

    positive = np.arange(0.0, max_lag + step * 0.5, step)
    if lag_direction == "transcript_leads":
        lags = positive
    else:
        negative = -positive[1:][::-1]
        lags = np.concatenate([negative, positive])

    records = []
    conditions = sorted(
        set(transcript_desc["Condition"]).intersection(set(metabolite_desc["Condition"])),
        key=natural_key,
    )

    for condition in conditions:
        r_states = transcript_desc.loc[
            transcript_desc["Condition"] == condition, "StateID"
        ].astype(str).tolist()
        m_states = metabolite_desc.loc[
            metabolite_desc["Condition"] == condition, "StateID"
        ].astype(str).tolist()

        for r_state in tqdm(r_states, desc=f"Continuous lag {condition}"):
            if r_state not in r_profiles or r_state not in r_times:
                continue
            rt = r_times[r_state]
            rv = r_profiles[r_state]

            for m_state in m_states:
                if m_state not in m_profiles or m_state not in m_times:
                    continue
                mt = m_times[m_state]
                mv = m_profiles[m_state]
                if len(rt) != len(mt) or not np.allclose(rt, mt):
                    common_t = np.intersect1d(rt, mt)
                    if len(common_t) < 3:
                        continue
                    rv_use = np.interp(common_t, rt, rv)
                    mv_use = np.interp(common_t, mt, mv)
                    t_use = common_t
                else:
                    t_use, rv_use, mv_use = rt, rv, mv

                tested = []
                for lag in lags:
                    cos, rmse, corr, n_aligned = _continuous_shift_similarity(
                        t_use, rv_use, mv_use, float(lag)
                    )
                    tested.append((float(lag), cos, rmse, corr, n_aligned))

                valid = [x for x in tested if np.isfinite(x[1])]
                if not valid:
                    continue
                best = sorted(
                    valid,
                    key=lambda x: (-x[1], x[2] if np.isfinite(x[2]) else np.inf),
                )[0]
                zero = min(tested, key=lambda x: abs(x[0]))
                best_lag, best_cos, best_rmse, best_corr, n_align = best

                if best_lag > 1e-12:
                    relation = "Transcriptomics_leads"
                elif best_lag < -1e-12:
                    relation = "Metabolomics_leads"
                else:
                    relation = "Synchronous"

                rr = r_ann.loc[r_state]
                mm = m_ann.loc[m_state]
                rc = r_cont.loc[r_state]
                mc = m_cont.loc[m_state]

                records.append({
                    "Condition": condition,
                    "TranscriptState": r_state,
                    "MetaboliteState": m_state,
                    "TranscriptModule": rr["ID"],
                    "MetaboliteModule": mm["ID"],
                    "TranscriptPattern": rr.get("Pattern", ""),
                    "MetabolitePattern": mm.get("Pattern", ""),
                    "TranscriptResponseStatePattern": rr.get("ResponseStatePattern", ""),
                    "MetaboliteResponseStatePattern": mm.get("ResponseStatePattern", ""),
                    "TrajectoryModelRNA": rc.get("TrajectoryModelUsed", ""),
                    "TrajectoryModelMET": mc.get("TrajectoryModelUsed", ""),
                    "ContinuousSimilarity_Lag0": zero[1],
                    "ContinuousRMSE_Lag0": zero[2],
                    "ContinuousCCF_Lag0": zero[3],
                    "BestContinuousLagTime": best_lag,
                    "BestContinuousSimilarity": best_cos,
                    "BestContinuousRMSE": best_rmse,
                    "BestContinuousCrossCorrelation": best_corr,
                    "ContinuousSimilarityGain": (
                        float(best_cos - zero[1]) if np.isfinite(zero[1]) else np.nan
                    ),
                    "NAlignedGridPoints": int(n_align),
                    "ContinuousRelationshipType": relation,
                    "RNA_ContinuousPeakTime": rc.get("ContinuousPeakTime", np.nan),
                    "MET_ContinuousPeakTime": mc.get("ContinuousPeakTime", np.nan),
                    "ContinuousPeakLag": (
                        float(mc.get("ContinuousPeakTime") - rc.get("ContinuousPeakTime"))
                        if np.isfinite(rc.get("ContinuousPeakTime", np.nan))
                        and np.isfinite(mc.get("ContinuousPeakTime", np.nan))
                        else np.nan
                    ),
                    "RNA_ContinuousAbsPeakTime": rc.get("ContinuousAbsPeakTime", np.nan),
                    "MET_ContinuousAbsPeakTime": mc.get("ContinuousAbsPeakTime", np.nan),
                    "ContinuousAbsPeakLag": (
                        float(mc.get("ContinuousAbsPeakTime") - rc.get("ContinuousAbsPeakTime"))
                        if np.isfinite(rc.get("ContinuousAbsPeakTime", np.nan))
                        and np.isfinite(mc.get("ContinuousAbsPeakTime", np.nan))
                        else np.nan
                    ),
                    "RNA_ResponseOnsetTime": rc.get("ContinuousResponseOnsetTime", np.nan),
                    "MET_ResponseOnsetTime": mc.get("ContinuousResponseOnsetTime", np.nan),
                    "ContinuousOnsetLag": (
                        float(mc.get("ContinuousResponseOnsetTime") - rc.get("ContinuousResponseOnsetTime"))
                        if np.isfinite(rc.get("ContinuousResponseOnsetTime", np.nan))
                        and np.isfinite(mc.get("ContinuousResponseOnsetTime", np.nan))
                        else np.nan
                    ),
                    "RNA_OnsetLeftCensored": rc.get("OnsetLeftCensored", False),
                    "MET_OnsetLeftCensored": mc.get("OnsetLeftCensored", False),
                    "RNA_MeanCIWidth": rc.get("MeanCIWidth", np.nan),
                    "MET_MeanCIWidth": mc.get("MeanCIWidth", np.nan),
                    "ModelEstimatedLag": True,
                })

    all_pairs = pd.DataFrame(records)
    all_file = f"{output_prefix}_continuous_lag_all_pairs.csv"
    all_pairs.to_csv(all_file, index=False)
    print(f"Saved continuous-time lag results: {all_file}")

    if all_pairs.empty:
        return all_pairs, pd.DataFrame()

    top = all_pairs[
        all_pairs["BestContinuousSimilarity"] >= float(min_similarity)
    ].copy()
    top["_gain_sort"] = top["ContinuousSimilarityGain"].fillna(-np.inf)
    top = top.sort_values(
        ["BestContinuousSimilarity", "_gain_sort"],
        ascending=[False, False],
    ).drop(columns="_gain_sort").head(max(1, int(top_n)))
    top_file = f"{output_prefix}_continuous_lag_top_pairs.csv"
    top.to_csv(top_file, index=False)
    print(f"Saved top continuous-time lag pairs: {top_file}")
    return all_pairs, top


def prepare_continuous_embedding_table(
    transcript_long: pd.DataFrame,
    metabolite_long: pd.DataFrame,
    transcript_desc: pd.DataFrame,
    metabolite_desc: pd.DataFrame,
) -> pd.DataFrame:
    """Create a wide regular-grid trajectory representation for joint embedding."""
    frames = []
    for long_df, desc, modality in [
        (transcript_long, transcript_desc, "Transcriptomics"),
        (metabolite_long, metabolite_desc, "Metabolomics"),
    ]:
        if long_df.empty:
            continue
        sub = long_df[long_df["RegularGridPoint"] == True].copy()  # noqa: E712
        times = sorted(sub["Time"].dropna().unique().tolist())
        pivot = sub.pivot_table(
            index="StateID", columns="Time", values="PredictedResponse", aggfunc="first"
        )
        pivot = pivot.reindex(columns=times)
        pivot.columns = [f"T{i+1}_{t:g}" for i, t in enumerate(times)]
        pivot = pivot.reset_index()

        annotation_cols = [
            c for c in [
                "StateID", "ID", "Condition", "ReferenceCondition", "Pattern",
                "ResponseStatePattern", "ResponseTiming", "DominantResponse",
                "OverallTrend", "DynamicRange", "MemberConcordanceFraction"
            ] if c in desc.columns
        ]
        ann = desc[annotation_cols].drop_duplicates("StateID")
        wide = pivot.merge(ann, on="StateID", how="left")
        wide["Modality"] = modality
        frames.append(wide)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def plot_continuous_lag_pair_diagnostics(
    top_pairs: pd.DataFrame,
    transcript_long: pd.DataFrame,
    metabolite_long: pd.DataFrame,
    output_prefix: str,
    top_n: int = 12,
    plot_format: str = "png",
) -> None:
    """Plot observed anchors, continuous curves, uncertainty, and best alignment."""
    if top_pairs.empty:
        return

    plot_dir = f"{output_prefix}_continuous_lag_pair_plots"
    os.makedirs(plot_dir, exist_ok=True)

    def get_state(df, state):
        return df[df["StateID"].astype(str) == str(state)].sort_values("Time")

    n_plot = min(int(top_n), top_pairs.shape[0])
    for rank, (_, pair) in enumerate(top_pairs.head(n_plot).iterrows(), start=1):
        rs = str(pair["TranscriptState"])
        ms = str(pair["MetaboliteState"])
        r = get_state(transcript_long, rs)
        m = get_state(metabolite_long, ms)
        if r.empty or m.empty:
            continue

        rr = r[r["RegularGridPoint"] == True].copy()  # noqa: E712
        mm = m[m["RegularGridPoint"] == True].copy()  # noqa: E712
        common_t = np.intersect1d(rr["Time"].to_numpy(float), mm["Time"].to_numpy(float))
        if len(common_t) < 3:
            continue
        rv = np.interp(common_t, rr["Time"], rr["PredictedResponse"])
        mv = np.interp(common_t, mm["Time"], mm["PredictedResponse"])
        lag = float(pair["BestContinuousLagTime"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
        for df, label, marker in [(r, "Transcriptomics", "o"), (m, "Metabolomics", "s")]:
            axes[0].plot(df["Time"], df["PredictedResponse"], linewidth=2, label=f"{label} model")
            finite_ci = np.isfinite(df["CI_Lower"]) & np.isfinite(df["CI_Upper"])
            if finite_ci.any():
                axes[0].fill_between(
                    df.loc[finite_ci, "Time"],
                    df.loc[finite_ci, "CI_Lower"],
                    df.loc[finite_ci, "CI_Upper"],
                    alpha=0.15,
                )
            obs = df[df["ObservedTimepoint"] == True]  # noqa: E712
            axes[0].scatter(obs["Time"], obs["ObservedResponse"], marker=marker, s=38)

        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Trajectory value")
        axes[0].set_title("Observed anchors and modelled trajectories")
        axes[0].grid(True, alpha=0.2)
        axes[0].legend()

        shifted = common_t + lag
        valid = (shifted >= common_t.min()) & (shifted <= common_t.max())
        a = rv[valid]
        b = np.interp(shifted[valid], common_t, mv)
        an = _normalize_profile_for_lag(a)
        bn = _normalize_profile_for_lag(b)
        axes[1].plot(common_t[valid], an, linewidth=2, label="RNA(t)")
        axes[1].plot(common_t[valid], bn, linewidth=2, label=f"MET(t+{lag:g})")
        axes[1].set_xlabel("RNA trajectory time")
        axes[1].set_ylabel("Normalized modelled response")
        axes[1].set_title(
            f"Model-assisted alignment | lag={lag:+g}, "
            f"similarity={pair['BestContinuousSimilarity']:.2f}"
        )
        axes[1].grid(True, alpha=0.2)
        axes[1].legend()

        fig.suptitle(
            f"{pair['TranscriptModule']} ↔ {pair['MetaboliteModule']} | "
            f"condition={pair['Condition']}\n"
            "Continuous lag is model-estimated, not an additional measured time point.",
            fontsize=11,
        )
        fig.tight_layout()
        file_name = os.path.join(
            plot_dir,
            f"{rank:02d}_{safe_output_prefix(pair['TranscriptModule'])}__"
            f"{safe_output_prefix(pair['MetaboliteModule'])}.{plot_format}",
        )
        fig.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {n_plot} continuous-lag diagnostic plots in: {plot_dir}")


def run_granger_directionality(
    candidate_pairs: pd.DataFrame,
    transcript_desc: pd.DataFrame,
    metabolite_desc: pd.DataFrame,
    output_prefix: str,
    min_real_timepoints: int = 12,
    max_lag: int = 1,
    transform: str = "difference",
    scope: str = "top",
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Optional Granger predictive-direction analysis using ORIGINAL measured
    trajectory points only. Modelled/interpolated points are explicitly excluded.

    This tests predictive precedence, not biochemical causality.
    """
    time_cols = get_trajectory_time_columns(transcript_desc)
    labels = trajectory_time_labels(time_cols)
    times = _numeric_times(labels)
    n_real = len(time_cols)

    out_file = f"{output_prefix}_granger_results.csv"
    status_file = f"{output_prefix}_granger_status.txt"

    if n_real < int(min_real_timepoints):
        message = (
            f"Granger analysis skipped: only {n_real} REAL measured time points are "
            f"available; minimum required is {min_real_timepoints}. Interpolated/modelled "
            "grid points are not counted as observations.\n"
        )
        Path(status_file).write_text(message)
        print("WARNING: " + message.strip())
        return pd.DataFrame()

    if times is None or len(times) != n_real:
        message = "Granger analysis skipped: measured time values are not numeric.\n"
        Path(status_file).write_text(message)
        print("WARNING: " + message.strip())
        return pd.DataFrame()

    diffs = np.diff(times)
    if len(diffs) > 1 and not np.allclose(diffs, diffs[0], rtol=1e-3, atol=1e-8):
        message = (
            "Granger analysis skipped: ORIGINAL measured time points are not approximately "
            "equally spaced. MATE will not use interpolated points to manufacture an equally "
            "spaced time series for Granger testing.\n"
        )
        Path(status_file).write_text(message)
        print("WARNING: " + message.strip())
        return pd.DataFrame()

    if grangercausalitytests is None or multipletests is None:
        message = "Granger analysis skipped: statsmodels is unavailable.\n"
        Path(status_file).write_text(message)
        print("WARNING: " + message.strip())
        return pd.DataFrame()

    r_prof, _ = descriptor_profile_matrix(transcript_desc)
    m_prof, _ = descriptor_profile_matrix(metabolite_desc)

    if candidate_pairs is None or candidate_pairs.empty:
        message = "Granger analysis skipped: no candidate RNA-MET pairs were supplied.\n"
        Path(status_file).write_text(message)
        print("WARNING: " + message.strip())
        return pd.DataFrame()

    pairs = candidate_pairs.copy()
    if scope == "top":
        pairs = pairs.head(max(1, int(top_n)))

    records = []
    for _, pair in tqdm(pairs.iterrows(), total=pairs.shape[0], desc="Granger tests"):
        rs = str(pair["TranscriptState"])
        ms = str(pair["MetaboliteState"])
        if rs not in r_prof or ms not in m_prof:
            continue

        r = np.asarray(r_prof[rs], dtype=float)
        m = np.asarray(m_prof[ms], dtype=float)
        good = np.isfinite(r) & np.isfinite(m)
        r = r[good]
        m = m[good]
        if transform == "difference":
            r = np.diff(r)
            m = np.diff(m)
        if len(r) <= (2 * int(max_lag) + 2):
            continue

        tests = [
            ("RNA_to_MET", np.column_stack([m, r])),
            ("MET_to_RNA", np.column_stack([r, m])),
        ]
        for direction, arr in tests:
            try:
                result = grangercausalitytests(arr, maxlag=int(max_lag), verbose=False)
            except TypeError:
                # statsmodels versions where verbose was removed/deprecated.
                result = grangercausalitytests(arr, maxlag=int(max_lag))
            except Exception:
                continue

            for lag_order, lag_result in result.items():
                try:
                    fstat, pval, df_denom, df_num = lag_result[0]["ssr_ftest"]
                except Exception:
                    continue
                records.append({
                    "Condition": pair.get("Condition", ""),
                    "TranscriptState": rs,
                    "MetaboliteState": ms,
                    "TranscriptModule": pair.get("TranscriptModule", ""),
                    "MetaboliteModule": pair.get("MetaboliteModule", ""),
                    "Direction": direction,
                    "LagOrder": int(lag_order),
                    "NRealTimepoints": int(n_real),
                    "NValuesTested": int(len(r)),
                    "Transform": transform,
                    "FStatistic": float(fstat),
                    "PValue": float(pval),
                    "DF_Denom": float(df_denom),
                    "DF_Num": float(df_num),
                    "Interpretation": "predictive_precedence_not_biochemical_causality",
                })

    out = pd.DataFrame(records)
    if not out.empty:
        out["AdjPValue_BH"] = multipletests(out["PValue"].to_numpy(float), method="fdr_bh")[1]
        out = out.sort_values(["AdjPValue_BH", "PValue"]).reset_index(drop=True)
    out.to_csv(out_file, index=False)
    Path(status_file).write_text(
        f"Granger analysis completed on ORIGINAL measured time points only. "
        f"N real time points={n_real}; transform={transform}; max lag={max_lag}. "
        "Results indicate predictive precedence, not biochemical causality.\n"
    )
    print(f"Saved Granger predictive-direction results: {out_file}")
    return out


# -----------------------------------------------------------------------------
# PCA, clustering, and plotting
# -----------------------------------------------------------------------------


def get_fingerprint_numeric_columns(df: pd.DataFrame) -> List[str]:
    meta_cols = {"ID", "Members", "Group", "n_features_found", "n_features_missing", "Cluster", "Cluster_Color"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]


def pca_on_fingerprints(df: pd.DataFrame, output_prefix: str, label: str = "") -> Tuple[pd.DataFrame, np.ndarray]:
    numeric_cols = get_fingerprint_numeric_columns(df)
    if len(numeric_cols) < 2:
        raise ValueError("Need at least two numeric fingerprint columns for PCA.")

    X = df[numeric_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_components = min(5, X.shape[0], X.shape[1])
    if n_components < 1:
        raise ValueError("Not enough data for PCA.")

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df.insert(0, "ID", df["ID"].values)

    for col in ["Group", "n_features_found", "n_features_missing"]:
        if col in df.columns:
            pca_df[col] = df[col].values

    pca_file = f"{output_prefix}{label}_pca.csv"
    pca_df.to_csv(pca_file, index=False)
    print(f"Saved PCA table: {pca_file}")
    return pca_df, explained


def run_trajectory_pca(
    fingerprints: pd.DataFrame,
    descriptors: pd.DataFrame,
    output_prefix: str,
    color_by: str = "Pattern",
    size_by: str = "DynamicRange",
    plot_format: str = "png",
) -> pd.DataFrame:
    """
    PCA of ORIGINAL module metafingerprints, followed by trajectory annotation.

    Pattern, Peak, DynamicRange, slopes, AUC, etc. are NOT PCA input variables.
    They are merged only after PCA so trajectory-colored separation is not circular.
    """
    if fingerprints.empty:
        return pd.DataFrame()

    X_scaled, used_cols = prepare_embedding_matrix(fingerprints)
    n_components = min(2, X_scaled.shape[0], X_scaled.shape[1])
    if n_components < 2:
        raise ValueError("Need at least two dimensions for trajectory-colored PCA.")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    emb = pd.DataFrame({
        "ID": fingerprints["ID"].values,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    })

    for col in ["Members", "n_features_found", "n_features_missing"]:
        if col in fingerprints.columns:
            emb[col] = fingerprints[col].values

    emb = add_embedding_annotations(emb, descriptors)
    emb["EmbeddingMethod"] = "pca"
    emb["n_input_columns"] = len(used_cols)

    table_file = f"{output_prefix}_pca_trajectory_embedding.csv"
    emb.to_csv(table_file, index=False)
    print(f"Saved trajectory-colored PCA table: {table_file}")

    title = (
        f"PCA of module metafingerprints colored by trajectory "
        f"(PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%)"
    )
    plot_embedding(
        emb=emb,
        x_col="PC1",
        y_col="PC2",
        output_file=f"{output_prefix}_pca_by_trajectory.{plot_format}",
        title=title,
        color_by=color_by,
        size_by=size_by,
    )

    return emb


def perform_clustering_and_extract_ids(
    df: pd.DataFrame,
    x: int,
    y: int,
    n_clusters: int,
    method: str = "kmeans",
    **kwargs,
) -> pd.DataFrame:
    pcx = f"PC{x}"
    pcy = f"PC{y}"
    if pcx not in df.columns or pcy not in df.columns:
        print(f"WARNING: {pcx}/{pcy} not available; skipping clustering.")
        return df

    data = df[[pcx, pcy]].to_numpy(dtype=float)
    n_rows = data.shape[0]
    if n_rows < 2:
        print("WARNING: fewer than two rows; skipping clustering.")
        return df

    n_clusters = min(max(1, int(n_clusters)), n_rows)

    if method == "kmeans":
        # n_init='auto' is not supported by older scikit-learn versions; use int for compatibility.
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(data)
    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
    elif method == "agglomerative":
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(data)
    elif method == "gmm":
        labels = GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(data)
    elif method == "spectral":
        labels = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", random_state=42).fit_predict(data)
    elif method == "meanshift":
        bandwidth = estimate_bandwidth(data)
        if bandwidth <= 0 or np.isnan(bandwidth):
            bandwidth = None
        labels = MeanShift(bandwidth=bandwidth).fit_predict(data)
    elif method == "birch":
        labels = Birch(n_clusters=n_clusters).fit_predict(np.ascontiguousarray(data))
    elif method == "affinity":
        damping = kwargs.get("damping", 0.9)
        preference = kwargs.get("preference", -50)
        labels = AffinityPropagation(damping=damping, preference=preference, random_state=42).fit_predict(data)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    out = df.copy()
    out["Cluster"] = labels

    unique_clusters = sorted(pd.unique(labels), key=natural_key)
    palette = sns.color_palette("hsv", len(unique_clusters))
    cluster_colors = dict(zip(unique_clusters, palette))
    out["Cluster_Color"] = out["Cluster"].map(lambda c: to_hex(cluster_colors[c]))
    return out


def plot_pca(pca_df: pd.DataFrame, explained: np.ndarray, output_file: str, title: str, color_by: Optional[str] = None) -> None:
    if "PC1" not in pca_df.columns:
        print("WARNING: PC1 not found; skipping PCA plot.")
        return

    plt.figure(figsize=(8, 6))

    if "PC2" in pca_df.columns:
        x = pca_df["PC1"]
        y = pca_df["PC2"]
        xlabel = f"PC1 ({explained[0]:.2f}% variance)" if len(explained) > 0 else "PC1"
        ylabel = f"PC2 ({explained[1]:.2f}% variance)" if len(explained) > 1 else "PC2"
    else:
        x = pca_df["PC1"]
        y = np.zeros(pca_df.shape[0])
        xlabel = f"PC1 ({explained[0]:.2f}% variance)" if len(explained) > 0 else "PC1"
        ylabel = "0"

    if color_by and color_by in pca_df.columns:
        categories = sorted(pca_df[color_by].astype(str).unique(), key=natural_key)
        for cat in categories:
            mask = pca_df[color_by].astype(str) == cat
            plt.scatter(x[mask], y[mask], label=cat, alpha=0.75, s=35)
        plt.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.scatter(x, y, alpha=0.75, s=35)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved PCA plot: {output_file}")


def combine_groups(fingerprints_by_group: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for group, df in fingerprints_by_group.items():
        tmp = df.copy()
        tmp["Group"] = group
        frames.append(tmp)
    if not frames:
        raise ValueError("No fingerprint groups to combine.")
    return pd.concat(frames, axis=0, ignore_index=True)




# -----------------------------------------------------------------------------
# t-SNE / UMAP embedding of metafingerprints
# -----------------------------------------------------------------------------


def add_embedding_annotations(embedding_df: pd.DataFrame, annotation_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge trajectory descriptor annotations into an embedding table by ID."""
    if annotation_df is None or annotation_df.empty or "ID" not in annotation_df.columns:
        return embedding_df

    annotation_cols = [
        c for c in [
            "ID", "Pattern", "OverallTrend", "NTimepoints", "NTransitions",
            "Peak", "PeakIndex", "AbsPeak", "AbsPeakIndex",
            "Delta_first_last", "DynamicRange", "MaxAbsDelta", "Variance", "AUC",
            "MaximumFC", "n_features_found", "n_features_missing"
        ]
        if c in annotation_df.columns
    ]
    if len(annotation_cols) <= 1:
        return embedding_df

    return embedding_df.merge(annotation_df[annotation_cols], on="ID", how="left")


def prepare_embedding_matrix(fingerprints: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Return scaled fingerprint matrix and numeric columns used for embedding."""
    numeric_cols = get_fingerprint_numeric_columns(fingerprints)
    if len(numeric_cols) < 2:
        raise ValueError("Need at least two numeric fingerprint/sample columns for t-SNE/UMAP embedding.")

    X = fingerprints[numeric_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove columns with zero variance because they do not help distance-based embeddings.
    col_sd = np.nanstd(X, axis=0)
    keep_cols = col_sd > 0
    X = X[:, keep_cols]
    used_cols = [c for c, keep in zip(numeric_cols, keep_cols) if keep]

    if X.shape[1] < 2:
        raise ValueError("Fewer than two non-constant numeric columns remain for embedding.")

    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, used_cols


def run_embedding(
    fingerprints: pd.DataFrame,
    method: str,
    output_prefix: str,
    annotation_df: Optional[pd.DataFrame] = None,
    metric: str = "euclidean",
    tsne_perplexity: float = 30.0,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.10,
    color_by: str = "Pattern",
    size_by: str = "DynamicRange",
    plot_format: str = "png",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run t-SNE or UMAP on all-sample cluster metafingerprints.

    Rows are clusters/modules and columns are sample-level metafingerprint values.
    This is useful for mapping transcript/eigenmetabolite-like cluster fingerprints
    into a shared low-dimensional module-state space.
    """
    method = method.lower()
    if method not in {"tsne", "umap"}:
        raise ValueError("method must be 'tsne' or 'umap'.")

    if fingerprints.empty:
        raise ValueError("Fingerprint table is empty; cannot run embedding.")

    X_scaled, used_cols = prepare_embedding_matrix(fingerprints)
    n_samples = X_scaled.shape[0]

    if n_samples < 3:
        raise ValueError("Need at least three clusters/modules for t-SNE/UMAP embedding.")

    if method == "tsne":
        # scikit-learn requires perplexity < n_samples.
        safe_perplexity = min(float(tsne_perplexity), max(1.0, float(n_samples - 1)))
        # For very small datasets, a smaller perplexity is usually more stable.
        safe_perplexity = min(safe_perplexity, max(1.0, (n_samples - 1) / 3.0))

        reducer = TSNE(
            n_components=2,
            perplexity=safe_perplexity,
            metric=metric,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        coords = reducer.fit_transform(X_scaled)
        x_col, y_col = "tSNE1", "tSNE2"
        title = f"t-SNE embedding of cluster metafingerprints (perplexity={safe_perplexity:.2f})"

    else:
        if umap is None:
            raise ImportError(
                "UMAP requested but umap-learn is not installed. Install with: pip install umap-learn "
                "or conda install -c conda-forge umap-learn"
            )

        safe_neighbors = min(max(2, int(umap_neighbors)), n_samples - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=safe_neighbors,
            min_dist=float(umap_min_dist),
            metric=metric,
            random_state=random_state,
        )
        coords = reducer.fit_transform(X_scaled)
        x_col, y_col = "UMAP1", "UMAP2"
        title = f"UMAP embedding of cluster metafingerprints (n_neighbors={safe_neighbors}, min_dist={umap_min_dist})"

    emb = pd.DataFrame({
        "ID": fingerprints["ID"].values,
        x_col: coords[:, 0],
        y_col: coords[:, 1],
    })

    for col in ["Members", "Group", "n_features_found", "n_features_missing"]:
        if col in fingerprints.columns and col not in emb.columns:
            emb[col] = fingerprints[col].values

    emb = add_embedding_annotations(emb, annotation_df)
    emb["EmbeddingMethod"] = method
    emb["n_input_columns"] = len(used_cols)

    table_file = f"{output_prefix}_{method}_embedding.csv"
    emb.to_csv(table_file, index=False)
    print(f"Saved {method} embedding table: {table_file}")

    plot_embedding(
        emb,
        x_col=x_col,
        y_col=y_col,
        output_file=f"{output_prefix}_{method}_embedding.{plot_format}",
        title=title,
        color_by=color_by,
        size_by=size_by,
    )

    return emb


def plot_embedding(
    emb: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_file: str,
    title: str,
    color_by: str = "Pattern",
    size_by: str = "DynamicRange",
) -> None:
    """
    Publication-friendly embedding scatter plot.

    Color encodes trajectory class (by default Pattern), while point size can
    encode response magnitude (by default DynamicRange).
    """
    plt.figure(figsize=(9, 7))

    hue = color_by if color_by and color_by in emb.columns else None
    size = size_by if size_by and size_by in emb.columns else None

    plot_kwargs = {
        "data": emb,
        "x": x_col,
        "y": y_col,
        "alpha": 0.82,
        "edgecolor": "none",
    }

    if hue:
        plot_kwargs["hue"] = hue
    if size:
        plot_kwargs["size"] = size
        plot_kwargs["sizes"] = (35, 180)
    else:
        plot_kwargs["s"] = 55

    n_unique = emb[hue].astype(str).nunique(dropna=True) if hue else 0

    if hue and n_unique > 25:
        # Too many trajectory classes for a readable categorical legend.
        plot_kwargs.pop("hue", None)
        ax = sns.scatterplot(**plot_kwargs)
        print(
            f"WARNING: '{hue}' has {n_unique} categories; "
            "embedding plotted without trajectory color legend for readability."
        )
    else:
        ax = sns.scatterplot(**plot_kwargs)

    if hue and n_unique <= 25:
        ax.legend(
            title=hue if not size else f"{hue} / size={size}",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
        )
    elif size:
        ax.legend(
            title=size,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
        )

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved embedding plot: {output_file}")


# -----------------------------------------------------------------------------
# Dual-omics trajectory integration and lag-aware analysis
# -----------------------------------------------------------------------------




def common_sample_columns(
    transcript_df: pd.DataFrame,
    metabolite_df: pd.DataFrame,
    metadata: pd.DataFrame,
) -> List[str]:
    """
    Return sample columns present in transcriptomics, metabolomics and metadata,
    preserving metadata order.
    """
    tcols = set(transcript_df.columns) - {"feature"}
    mcols = set(metabolite_df.columns) - {"feature"}
    ordered = metadata["sample"].astype(str).str.strip().tolist()
    common = [s for s in ordered if s in tcols and s in mcols]

    if not common:
        raise ValueError(
            "No common samples were found across transcriptomics, metabolomics and metadata."
        )

    missing_t = [s for s in ordered if s not in tcols]
    missing_m = [s for s in ordered if s not in mcols]
    if missing_t:
        print(f"WARNING: {len(missing_t)} metadata samples are absent from transcriptomics.")
    if missing_m:
        print(f"WARNING: {len(missing_m)} metadata samples are absent from metabolomics.")

    print(f"Dual-omics integration: {len(common)} common samples.")
    return common


def prefix_module_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix module IDs so transcript and metabolite module names cannot collide."""
    out = df.copy()
    out["OriginalID"] = out["ID"].astype(str)
    out["ID"] = prefix + "::" + out["ID"].astype(str)
    return out


def time_sample_groups(
    metadata: pd.DataFrame,
    available_columns: List[str],
    time_column: str = "",
    time_values: str = "",
    n_timepoints: int = 0,
) -> Tuple[str, List[str], Dict[str, List[str]]]:
    """Resolve ordered timepoints and matching sample columns."""
    time_col = resolve_time_column(metadata, time_column)
    ordered_times = resolve_time_values(
        metadata,
        time_col,
        time_values,
        n_timepoints,
        max_timepoints=100,
    )

    md = metadata.copy()
    md[time_col] = md[time_col].astype(str).str.strip()
    md["sample"] = md["sample"].astype(str).str.strip()

    avail = set(available_columns)
    groups = {}
    for t in ordered_times:
        samples = md.loc[md[time_col] == str(t), "sample"].tolist()
        samples = [s for s in samples if s in avail]
        if not samples:
            raise ValueError(f"No common omics samples found at time '{t}'.")
        groups[str(t)] = samples

    return time_col, ordered_times, groups


def module_member_coherence(
    feature_df: pd.DataFrame,
    clusters_frame: pd.DataFrame,
    descriptors: pd.DataFrame,
    metadata: pd.DataFrame,
    time_column: str = "",
    time_values: str = "",
    n_timepoints: int = 0,
    cosine_threshold: float = 0.50,
    stratify_by: str = "condition",
    reference_condition: str = "",
    trajectory_metric: str = "effect_size",
) -> pd.DataFrame:
    """Calculate member-level coherence using matched-reference responses."""
    if descriptors.empty:
        return pd.DataFrame()

    time_col = resolve_time_column(metadata, time_column)
    ordered_times = resolve_time_values(
        metadata, time_col, time_values, n_timepoints, max_timepoints=100
    )
    stratify_col = resolve_stratify_column(metadata, stratify_by)

    sample_cols = [c for c in feature_df.columns if c != "feature"]
    condition_groups = condition_time_sample_groups(
        metadata, sample_cols, ordered_times, time_col, stratify_col
    )

    reference_condition = str(reference_condition).strip()
    if reference_condition not in condition_groups:
        raise ValueError(
            f"Reference condition '{reference_condition}' unavailable for coherence analysis."
        )

    reference_groups = condition_groups[reference_condition]
    time_cols = get_trajectory_time_columns(descriptors)
    lookup = feature_df.set_index("feature", drop=True)
    members_lookup = clusters_frame.set_index("ID")["Members"].to_dict()

    records = []

    for _, row in descriptors.iterrows():
        module_id = row["ID"]
        condition = str(row["Condition"])

        if condition not in condition_groups or condition == reference_condition:
            continue

        treatment_groups = condition_groups[condition]
        members = [
            m for m in parse_members(members_lookup.get(module_id, ""))
            if m in lookup.index
        ]

        module_profile = pd.to_numeric(
            row[time_cols], errors="coerce"
        ).to_numpy(dtype=float)

        cosines = []

        for member in members:
            member_response = []

            for t in ordered_times:
                tr_vals = np.asarray(
                    pd.to_numeric(
                        lookup.loc[member, treatment_groups[str(t)]],
                        errors="coerce"
                    ),
                    dtype=float
                ).reshape(-1)

                ref_vals = np.asarray(
                    pd.to_numeric(
                        lookup.loc[member, reference_groups[str(t)]],
                        errors="coerce"
                    ),
                    dtype=float
                ).reshape(-1)

                tr_vals = tr_vals[np.isfinite(tr_vals)]
                ref_vals = ref_vals[np.isfinite(ref_vals)]

                if trajectory_metric == "effect_size":
                    response = standardized_mean_difference(ref_vals, tr_vals)
                else:
                    response = (
                        float(np.mean(tr_vals) - np.mean(ref_vals))
                        if tr_vals.size and ref_vals.size
                        else np.nan
                    )
                member_response.append(response)

            a = np.nan_to_num(np.asarray(member_response, dtype=float), nan=0.0)
            b = np.nan_to_num(module_profile, nan=0.0)
            denom = np.linalg.norm(a) * np.linalg.norm(b)

            if denom > 0:
                cosines.append(float(np.dot(a, b) / denom))

        records.append({
            "ID": module_id,
            "Condition": condition,
            "StateID": row.get("StateID", f"{module_id}::{condition}"),
            "NMembersEvaluated": int(len(cosines)),
            "MemberMeanCosine": float(np.mean(cosines)) if cosines else np.nan,
            "MemberMedianCosine": float(np.median(cosines)) if cosines else np.nan,
            "MemberConcordanceFraction": (
                float(np.mean(np.asarray(cosines) >= cosine_threshold))
                if cosines else np.nan
            ),
        })

    return pd.DataFrame(records)



def module_member_coherence_pseudotime(
    feature_df: pd.DataFrame,
    clusters_frame: pd.DataFrame,
    descriptors: pd.DataFrame,
    metadata: pd.DataFrame,
    time_column: str = "",
    time_values: str = "",
    n_timepoints: int = 0,
    cosine_threshold: float = 0.50,
    stratify_by: str = "condition",
) -> pd.DataFrame:
    """Member-level support for reference-free pseudotime trajectories."""
    if descriptors.empty:
        return pd.DataFrame()

    time_col = resolve_time_column(metadata, time_column)
    ordered_times = resolve_time_values(
        metadata, time_col, time_values, n_timepoints, max_timepoints=100
    )
    stratify_col = resolve_stratify_column(metadata, stratify_by)

    sample_cols = [c for c in feature_df.columns if c != "feature"]
    condition_groups = condition_time_sample_groups(
        metadata, sample_cols, ordered_times, time_col, stratify_col
    )

    time_cols = get_trajectory_time_columns(descriptors)
    lookup = feature_df.set_index("feature", drop=True)
    members_lookup = clusters_frame.set_index("ID")["Members"].to_dict()
    records = []

    for _, row in descriptors.iterrows():
        module_id = row["ID"]
        condition = str(row["Condition"])

        if condition not in condition_groups:
            continue

        sample_groups = condition_groups[condition]
        members = [
            m for m in parse_members(members_lookup.get(module_id, ""))
            if m in lookup.index
        ]

        module_profile = pd.to_numeric(
            row[time_cols], errors="coerce"
        ).to_numpy(dtype=float)

        cosines = []
        for member in members:
            member_profile = []

            for t in ordered_times:
                vals = np.asarray(
                    pd.to_numeric(
                        lookup.loc[member, sample_groups[str(t)]],
                        errors="coerce",
                    ),
                    dtype=float,
                ).reshape(-1)
                vals = vals[np.isfinite(vals)]
                member_profile.append(
                    float(np.mean(vals)) if vals.size else np.nan
                )

            a = np.nan_to_num(
                np.asarray(member_profile, dtype=float), nan=0.0
            )
            b = np.nan_to_num(module_profile, nan=0.0)
            denom = np.linalg.norm(a) * np.linalg.norm(b)

            if denom > 0:
                cosines.append(float(np.dot(a, b) / denom))

        records.append({
            "ID": module_id,
            "Condition": condition,
            "StateID": row.get("StateID", f"{module_id}::{condition}"),
            "NMembersEvaluated": int(len(cosines)),
            "MemberMeanCosine": float(np.mean(cosines)) if cosines else np.nan,
            "MemberMedianCosine": float(np.median(cosines)) if cosines else np.nan,
            "MemberConcordanceFraction": (
                float(np.mean(np.asarray(cosines) >= cosine_threshold))
                if cosines else np.nan
            ),
        })

    return pd.DataFrame(records)

def merge_coherence_into_descriptors(
    descriptors: pd.DataFrame,
    coherence: pd.DataFrame,
) -> pd.DataFrame:
    if descriptors.empty or coherence.empty:
        return descriptors

    keys = [c for c in ["ID", "Condition", "StateID"] if c in descriptors.columns and c in coherence.columns]
    if not keys:
        keys = ["ID"]

    return descriptors.merge(coherence, on=keys, how="left")


def prepare_joint_fingerprint_table(
    transcript_fp: pd.DataFrame,
    metabolite_fp: pd.DataFrame,
    transcript_desc: pd.DataFrame,
    metabolite_desc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a joint module x condition trajectory-state table.

    For treatment-aware MATE, each row is one module x condition state.
    Continuous timepoint means and transition statistics are retained for
    embedding; Pattern is annotation only.
    """
    t = transcript_desc.copy()
    m = metabolite_desc.copy()
    t["Modality"] = "Transcriptomics"
    m["Modality"] = "Metabolomics"

    return pd.concat([t, m], ignore_index=True, sort=False)


def joint_fingerprint_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Continuous variables used for treatment-aware trajectory embedding.

    Pattern/Condition/Modality are never used as embedding variables.
    """
    time_cols = get_trajectory_time_columns(df)
    transition_cols = [
        c for c in df.columns
        if str(c).startswith("TransitionStat_T")
    ]

    # Primary embedding is the observed trajectory plus continuous transition
    # statistics. This makes the space explicitly temporal.
    cols = time_cols + transition_cols
    return [c for c in cols if c in df.columns]


def prepare_joint_embedding_matrix(
    combined: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    numeric_cols = joint_fingerprint_numeric_columns(combined)
    if len(numeric_cols) < 2:
        raise ValueError(
            "Need at least two common sample-level fingerprint columns for joint embedding."
        )

    X = combined[numeric_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    sd = np.nanstd(X, axis=0)
    keep = sd > 0
    X = X[:, keep]
    used = [c for c, k in zip(numeric_cols, keep) if k]

    if X.shape[1] < 2:
        raise ValueError("Fewer than two non-constant common fingerprint columns remain.")

    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, used


def plot_joint_embedding(
    emb: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_file: str,
    title: str,
    color_by: str = "Pattern",
    size_by: str = "DynamicRange",
) -> None:
    """
    Joint omics embedding:
      color = trajectory Pattern
      marker shape = Modality
      point size = DynamicRange (default)
    """
    plt.figure(figsize=(10, 8))

    kwargs = {
        "data": emb,
        "x": x_col,
        "y": y_col,
        "alpha": 0.82,
        "edgecolor": "none",
    }

    if color_by and color_by in emb.columns:
        kwargs["hue"] = color_by
    if "Modality" in emb.columns:
        kwargs["style"] = "Modality"
    if size_by and size_by in emb.columns:
        kwargs["size"] = size_by
        kwargs["sizes"] = (35, 190)
    else:
        kwargs["s"] = 60

    ax = sns.scatterplot(**kwargs)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.22)
    ax.legend(
        bbox_to_anchor=(1.03, 1),
        loc="upper left",
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved joint embedding plot: {output_file}")


def run_joint_embedding(
    combined: pd.DataFrame,
    method: str,
    output_prefix: str,
    metric: str = "euclidean",
    tsne_perplexity: float = 30.0,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.10,
    color_by: str = "Pattern",
    size_by: str = "DynamicRange",
    plot_format: str = "png",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Embed RNA and metabolite modules together using original module
    metafingerprint sample columns.
    """
    X_scaled, used_cols = prepare_joint_embedding_matrix(combined)
    method = method.lower()
    n = X_scaled.shape[0]

    if method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(X_scaled)
        explained = reducer.explained_variance_ratio_ * 100
        x_col, y_col = "PC1", "PC2"
        title = (
            "Joint MATE PCA: trajectory color, modality shape "
            f"(PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%)"
        )

    elif method == "umap":
        if umap is None:
            raise ImportError(
                "UMAP requested but umap-learn is not installed. "
                "Install with: pip install umap-learn"
            )
        safe_neighbors = min(max(2, int(umap_neighbors)), n - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=safe_neighbors,
            min_dist=float(umap_min_dist),
            metric=metric,
            random_state=random_state,
        )
        coords = reducer.fit_transform(X_scaled)
        x_col, y_col = "UMAP1", "UMAP2"
        title = "Joint MATE UMAP: trajectory color, modality shape"

    elif method == "tsne":
        safe_perplexity = min(float(tsne_perplexity), max(1.0, (n - 1) / 3.0))
        reducer = TSNE(
            n_components=2,
            perplexity=safe_perplexity,
            metric=metric,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        coords = reducer.fit_transform(X_scaled)
        x_col, y_col = "tSNE1", "tSNE2"
        title = "Joint MATE t-SNE: trajectory color, modality shape"
    else:
        raise ValueError("Joint embedding method must be pca, umap or tsne.")

    emb = combined.copy()
    if "StateID" in emb.columns:
        emb["ID"] = emb["StateID"].astype(str)
    emb[x_col] = coords[:, 0]
    emb[y_col] = coords[:, 1]
    emb["EmbeddingMethod"] = method
    emb["n_input_columns"] = len(used_cols)

    table_file = f"{output_prefix}_joint_{method}_embedding.csv"
    emb.to_csv(table_file, index=False)
    print(f"Saved joint {method} embedding table: {table_file}")

    plot_joint_embedding(
        emb,
        x_col=x_col,
        y_col=y_col,
        output_file=f"{output_prefix}_joint_{method}_embedding.{plot_format}",
        title=title,
        color_by=color_by,
        size_by=size_by,
    )
    return emb


def shared_pattern_tables(
    transcript_desc: pd.DataFrame,
    metabolite_desc: pd.DataFrame,
    output_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize patterns shared across omics WITHIN the same condition.
    """
    t = transcript_desc[["ID", "StateID", "Condition", "Pattern"]].copy()
    m = metabolite_desc[["ID", "StateID", "Condition", "Pattern"]].copy()

    t = t[t["Pattern"].notna() & (t["Pattern"].astype(str).str.upper() != "NA")]
    m = m[m["Pattern"].notna() & (m["Pattern"].astype(str).str.upper() != "NA")]

    tc = (
        t.groupby(["Condition", "Pattern"])
        .size()
        .rename("TranscriptModules")
    )
    mc = (
        m.groupby(["Condition", "Pattern"])
        .size()
        .rename("MetaboliteModules")
    )

    summary = (
        pd.concat([tc, mc], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    summary["SharedPattern"] = (
        (summary["TranscriptModules"] > 0) &
        (summary["MetaboliteModules"] > 0)
    )
    summary["PotentialCrossOmicsPairs"] = (
        summary["TranscriptModules"] *
        summary["MetaboliteModules"]
    )

    summary_file = f"{output_prefix}_shared_trajectory_patterns.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved condition-aware shared trajectory summary: {summary_file}")

    pairs = t.merge(
        m,
        on=["Condition", "Pattern"],
        suffixes=("_RNA", "_MET"),
    )
    pairs = pairs.rename(columns={
        "ID_RNA": "TranscriptModule",
        "StateID_RNA": "TranscriptState",
        "ID_MET": "MetaboliteModule",
        "StateID_MET": "MetaboliteState",
    })

    pair_file = f"{output_prefix}_same_pattern_module_pairs.csv"
    pairs.to_csv(pair_file, index=False)
    print(f"Saved condition-aware same-pattern module pairs: {pair_file}")

    return summary, pairs


def descriptor_profile_matrix(
    descriptors: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Return StateID -> time-profile mapping.
    """
    time_cols = get_trajectory_time_columns(descriptors)
    profiles = {}
    for _, row in descriptors.iterrows():
        key = row.get("StateID", row["ID"])
        profiles[str(key)] = pd.to_numeric(
            row[time_cols],
            errors="coerce",
        ).to_numpy(dtype=float)
    return profiles, trajectory_time_labels(time_cols)


def _normalize_profile_for_lag(x: np.ndarray) -> np.ndarray:
    """
    Normalize a complete trajectory before lag alignment without using
    cross-omics correlation.

    Centering removes baseline offset; L2 scaling reduces dominance by
    absolute amplitude while preserving trajectory shape.
    """
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmean(x) if not np.all(np.isnan(x)) else 0.0)
    x = x - np.mean(x)
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    return x


def _aligned_indices(n: int, lag_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positive lag means transcript trajectory precedes metabolite trajectory.
    """
    if lag_steps > 0:
        r_idx = np.arange(0, n - lag_steps)
        m_idx = np.arange(lag_steps, n)
    elif lag_steps < 0:
        k = abs(lag_steps)
        r_idx = np.arange(k, n)
        m_idx = np.arange(0, n - k)
    else:
        r_idx = np.arange(n)
        m_idx = np.arange(n)
    return r_idx, m_idx


def _lag_similarity(
    rna_profile: np.ndarray,
    met_profile: np.ndarray,
    lag_steps: int,
) -> Tuple[float, float, int]:
    """
    Return cosine similarity and RMSE after the requested displacement.

    Profiles are normalized independently before alignment. Pearson correlation
    is deliberately not used as the discovery score.
    """
    r = _normalize_profile_for_lag(rna_profile)
    m = _normalize_profile_for_lag(met_profile)
    n = min(len(r), len(m))
    r = r[:n]
    m = m[:n]

    ri, mi = _aligned_indices(n, lag_steps)
    if len(ri) < 2:
        return np.nan, np.nan, int(len(ri))

    a = r[ri]
    b = m[mi]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / denom) if denom > 0 else np.nan
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return cosine, rmse, int(len(ri))


def _numeric_times(labels: List[str]) -> Optional[np.ndarray]:
    vals = []
    for x in labels:
        match = re.search(r"[-+]?\d*\.?\d+", str(x))
        if not match:
            return None
        try:
            vals.append(float(match.group(0)))
        except Exception:
            return None
    return np.asarray(vals, dtype=float)


def _lag_time_value(
    time_labels: List[str],
    lag_steps: int,
) -> Tuple[float, str]:
    """
    Report the actual time differences represented by an index displacement.
    For irregular sampling, the median difference is reported and all matched
    differences are retained as text.
    """
    times = _numeric_times(time_labels)
    if times is None:
        return np.nan, ""

    ri, mi = _aligned_indices(len(times), lag_steps)
    if len(ri) == 0:
        return np.nan, ""

    diffs = times[mi] - times[ri]
    return float(np.median(diffs)), ";".join(f"{x:g}" for x in diffs)


def cross_omics_lag_analysis(
    transcript_desc: pd.DataFrame,
    metabolite_desc: pd.DataFrame,
    output_prefix: str,
    max_lag_steps: int = 2,
    lag_direction: str = "both",
    top_n: int = 50,
    min_similarity: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate transcript-module x metabolite-module pairs WITHIN each condition.

    Positive BestLagSteps = transcriptomics leads metabolomics.
    Negative BestLagSteps = metabolomics leads transcriptomics.
    """
    r_profiles, time_labels = descriptor_profile_matrix(transcript_desc)
    m_profiles, time_labels_m = descriptor_profile_matrix(metabolite_desc)

    if time_labels != time_labels_m:
        raise ValueError(
            "Transcriptomics and metabolomics descriptors use different time grids."
        )

    n_time = len(time_labels)
    max_allowed = max(0, min(int(max_lag_steps), n_time - 2))

    if lag_direction == "transcript_leads":
        lags = list(range(0, max_allowed + 1))
    else:
        lags = list(range(-max_allowed, max_allowed + 1))

    r_ann = transcript_desc.set_index("StateID")
    m_ann = metabolite_desc.set_index("StateID")

    records = []

    conditions = sorted(
        set(transcript_desc["Condition"]).intersection(
            set(metabolite_desc["Condition"])
        ),
        key=natural_key,
    )

    for condition in conditions:
        r_states = transcript_desc.loc[
            transcript_desc["Condition"] == condition, "StateID"
        ].astype(str).tolist()
        m_states = metabolite_desc.loc[
            metabolite_desc["Condition"] == condition, "StateID"
        ].astype(str).tolist()

        print(
            f"Condition {condition}: evaluating "
            f"{len(r_states) * len(m_states)} RNA x MET pairs."
        )

        for r_state in tqdm(r_states, desc=f"Lag analysis {condition}"):
            r_profile = r_profiles[r_state]

            for m_state in m_states:
                m_profile = m_profiles[m_state]

                lag_results = []
                for lag in lags:
                    cos, rmse, n_aligned = _lag_similarity(
                        r_profile,
                        m_profile,
                        lag,
                    )
                    lag_results.append((lag, cos, rmse, n_aligned))

                valid = [x for x in lag_results if np.isfinite(x[1])]
                if not valid:
                    continue

                best = sorted(
                    valid,
                    key=lambda x: (
                        -x[1],
                        x[2] if np.isfinite(x[2]) else np.inf
                    ),
                )[0]
                zero = next(
                    (x for x in lag_results if x[0] == 0),
                    (0, np.nan, np.nan, 0),
                )

                best_lag, best_cos, best_rmse, n_aligned = best
                median_lag_time, lag_time_values = _lag_time_value(
                    time_labels,
                    best_lag,
                )

                if best_lag > 0:
                    relationship = "Transcriptomics_leads"
                elif best_lag < 0:
                    relationship = "Metabolomics_leads"
                else:
                    relationship = "Synchronous"

                rr = r_ann.loc[r_state]
                mm = m_ann.loc[m_state]

                records.append({
                    "Condition": condition,
                    "TranscriptState": r_state,
                    "MetaboliteState": m_state,
                    "TranscriptModule": rr["ID"],
                    "MetaboliteModule": mm["ID"],
                    "TranscriptPattern": rr.get("Pattern", ""),
                    "MetabolitePattern": mm.get("Pattern", ""),
                    "TranscriptOverallTrend": rr.get("OverallTrend", ""),
                    "MetaboliteOverallTrend": mm.get("OverallTrend", ""),
                    "TranscriptPeak": rr.get("Peak", ""),
                    "MetabolitePeak": mm.get("Peak", ""),
                    "TranscriptDynamicRange": rr.get("DynamicRange", np.nan),
                    "MetaboliteDynamicRange": mm.get("DynamicRange", np.nan),
                    "TranscriptMemberCoherence": rr.get(
                        "MemberConcordanceFraction", np.nan
                    ),
                    "MetaboliteMemberCoherence": mm.get(
                        "MemberConcordanceFraction", np.nan
                    ),
                    "Similarity_Lag0": zero[1],
                    "RMSE_Lag0": zero[2],
                    "BestLagSteps": int(best_lag),
                    "BestLagTimeMedian": median_lag_time,
                    "LagTimeDifferences": lag_time_values,
                    "BestLagSimilarity": best_cos,
                    "BestLagRMSE": best_rmse,
                    "SimilarityGain": (
                        float(best_cos - zero[1])
                        if np.isfinite(zero[1]) else np.nan
                    ),
                    "NAlignedTimepoints": int(n_aligned),
                    "RelationshipType": relationship,
                })

    all_pairs = pd.DataFrame(records)
    all_file = f"{output_prefix}_cross_omics_lag_all_pairs.csv"
    all_pairs.to_csv(all_file, index=False)
    print(f"Saved condition-aware lag results: {all_file}")

    if all_pairs.empty:
        return all_pairs, pd.DataFrame()

    shortlist = all_pairs[
        all_pairs["BestLagSimilarity"] >= float(min_similarity)
    ].copy()

    shortlist["_gain_sort"] = shortlist["SimilarityGain"].fillna(-np.inf)
    shortlist = shortlist.sort_values(
        ["BestLagSimilarity", "_gain_sort"],
        ascending=[False, False],
    ).drop(columns="_gain_sort").head(max(1, int(top_n)))

    shortlist_file = f"{output_prefix}_cross_omics_lag_top_pairs.csv"
    shortlist.to_csv(shortlist_file, index=False)
    print(f"Saved top condition-aware lag pairs: {shortlist_file}")

    return all_pairs, shortlist


def plot_lag_pair_diagnostics(
    top_pairs: pd.DataFrame,
    transcript_desc: pd.DataFrame,
    metabolite_desc: pd.DataFrame,
    output_prefix: str,
    top_n: int = 12,
    plot_format: str = "png",
) -> None:
    """
    For top cross-omics pairs, show:
      A. observed RNA and metabolite trajectories on the original time axis
      B. lag-aligned normalized trajectories

    The second panel is an analytical alignment, not altered observed data.
    """
    if top_pairs.empty:
        return

    r_profiles, time_labels = descriptor_profile_matrix(transcript_desc)
    m_profiles, _ = descriptor_profile_matrix(metabolite_desc)
    numeric_times = _numeric_times(time_labels)
    x_original = numeric_times if numeric_times is not None else np.arange(len(time_labels))

    plot_dir = f"{output_prefix}_lag_pair_plots"
    os.makedirs(plot_dir, exist_ok=True)

    n_plot = min(int(top_n), top_pairs.shape[0])
    for rank, (_, pair) in enumerate(top_pairs.head(n_plot).iterrows(), start=1):
        r_state = pair.get("TranscriptState", pair["TranscriptModule"])
        m_state = pair.get("MetaboliteState", pair["MetaboliteModule"])
        r_id = pair["TranscriptModule"]
        m_id = pair["MetaboliteModule"]
        lag = int(pair["BestLagSteps"])

        r = np.asarray(r_profiles[str(r_state)], dtype=float)
        m = np.asarray(m_profiles[str(m_state)], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        axes[0].plot(x_original, r, marker="o", linewidth=2, label="Transcriptomics")
        axes[0].plot(x_original, m, marker="s", linewidth=2, label="Metabolomics")
        axes[0].set_title("Observed trajectories")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Trajectory value")
        axes[0].set_xticks(x_original)
        axes[0].set_xticklabels(time_labels, rotation=45, ha="right")
        axes[0].grid(True, alpha=0.22)
        axes[0].legend()

        rn = _normalize_profile_for_lag(r)
        mn = _normalize_profile_for_lag(m)
        ri, mi = _aligned_indices(min(len(rn), len(mn)), lag)

        aligned_x = np.arange(len(ri))
        axes[1].plot(aligned_x, rn[ri], marker="o", linewidth=2, label="RNA aligned")
        axes[1].plot(aligned_x, mn[mi], marker="s", linewidth=2, label="MET aligned")
        pair_labels = [
            f"RNA {time_labels[i]} → MET {time_labels[j]}"
            for i, j in zip(ri, mi)
        ]
        axes[1].set_xticks(aligned_x)
        axes[1].set_xticklabels(pair_labels, rotation=35, ha="right")
        axes[1].set_title(
            f"Lag-aligned profiles | lag={lag:+d}, "
            f"similarity={pair['BestLagSimilarity']:.2f}"
        )
        axes[1].set_ylabel("Normalized trajectory")
        axes[1].grid(True, alpha=0.22)
        axes[1].legend()

        fig.suptitle(
            f"{r_id}  ↔  {m_id} | condition={pair.get('Condition', 'ALL')}\n"
            f"{pair['RelationshipType']} | "
            f"lag0={pair['Similarity_Lag0']:.2f}, "
            f"gain={pair['SimilarityGain']:.2f}",
            fontsize=11,
        )
        fig.tight_layout()

        safe_r = safe_output_prefix(r_id)
        safe_m = safe_output_prefix(m_id)
        file_name = os.path.join(
            plot_dir,
            f"{rank:02d}_{safe_r}__{safe_m}.{plot_format}",
        )
        fig.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {n_plot} lag-pair diagnostic plots in: {plot_dir}")


def run_dual_omics_mode(options: argparse.Namespace) -> None:
    """
    Full MATE dual-omics workflow.

    Transcriptomics and metabolomics are processed independently until their
    module metafingerprints and trajectories have been generated. Integration
    then occurs at the module trajectory/fingerprint level.
    """
    output_prefix = safe_output_prefix(options.outfile)

    print("\n" + "=" * 78)
    print("MATE v2.4")
    print("=" * 78)
    print("Transcriptomics and metabolomics modules are calculated independently.")

    if options.trajectory_mode == "reference":
        if not str(options.reference_condition).strip():
            raise ValueError(
                "--reference_condition is required when --trajectory_mode reference."
            )
        print(
            f"Trajectory mode: reference-aware; each treatment is compared with "
            f"matched reference '{options.reference_condition}' at every selected time point."
        )
    else:
        if str(options.reference_condition).strip():
            print(
                "WARNING: --reference_condition is ignored in "
                "--trajectory_mode pseudotime."
            )
        print(
            "Trajectory mode: pseudotime/ordered-time; module trajectories are "
            "followed directly without a mock/reference condition."
        )
        print(
            "NOTE: MATE uses the supplied chronological/pseudotime ordering; "
            "it does not infer pseudotime from the omics matrices."
        )

    print("Integration occurs after modality-specific trajectories are obtained.\n")

    metadata = read_metadata(options.metadata)

    transcript_df = read_feature_table(options.transcriptomics_table)
    metabolite_df = read_feature_table(options.metabolomics_table)

    transcript_df = normalize_feature_table(
        transcript_df,
        do_zscore=not options.no_zscore,
    )
    metabolite_df = normalize_feature_table(
        metabolite_df,
        do_zscore=not options.no_zscore,
    )

    common_samples = common_sample_columns(
        transcript_df,
        metabolite_df,
        metadata,
    )

    transcript_df = transcript_df[["feature"] + common_samples].copy()
    metabolite_df = metabolite_df[["feature"] + common_samples].copy()

    transcript_clusters = load_clusters(
        options.transcriptomics_db,
        options.decay_rate,
    )
    metabolite_clusters = load_clusters(
        options.metabolomics_db,
        options.decay_rate,
    )

    transcript_clusters = prefix_module_ids(transcript_clusters, "RNA")
    metabolite_clusters = prefix_module_ids(metabolite_clusters, "MET")

    # Keep cluster member tables compatible with prefixed IDs.
    transcript_clusters_file = f"{output_prefix}_RNA_modules.csv"
    metabolite_clusters_file = f"{output_prefix}_MET_modules.csv"
    transcript_clusters.to_csv(transcript_clusters_file, index=False)
    metabolite_clusters.to_csv(metabolite_clusters_file, index=False)

    # ------------------------------
    # Independent fingerprints
    # ------------------------------
    transcript_fp, transcript_missing = calculate_fingerprints_for_matrix(
        transcript_df,
        transcript_clusters,
        desc="Transcriptomics module fingerprints",
    )
    metabolite_fp, metabolite_missing = calculate_fingerprints_for_matrix(
        metabolite_df,
        metabolite_clusters,
        desc="Metabolomics module fingerprints",
    )

    transcript_fp.to_csv(
        f"{output_prefix}_RNA_global_fingerprints.csv",
        index=False,
    )
    metabolite_fp.to_csv(
        f"{output_prefix}_MET_global_fingerprints.csv",
        index=False,
    )

    if not transcript_missing.empty:
        transcript_missing.to_csv(
            f"{output_prefix}_RNA_missing_features.csv",
            index=False,
        )
    if not metabolite_missing.empty:
        metabolite_missing.to_csv(
            f"{output_prefix}_MET_missing_features.csv",
            index=False,
        )

    # ------------------------------
    # Independent trajectories
    # ------------------------------
    if options.trajectory_mode == "reference":
        transcript_desc = calculate_trajectory_descriptors(
            transcript_fp,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            flat_threshold=options.trajectory_flat_threshold,
            stratify_by=options.stratify_by,
            trajectory_metric=options.trajectory_metric,
            effect_size_threshold=options.effect_size_threshold,
            reference_condition=options.reference_condition,
        )
        metabolite_desc = calculate_trajectory_descriptors(
            metabolite_fp,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            flat_threshold=options.trajectory_flat_threshold,
            stratify_by=options.stratify_by,
            trajectory_metric=options.trajectory_metric,
            effect_size_threshold=options.effect_size_threshold,
            reference_condition=options.reference_condition,
        )

        transcript_coherence = module_member_coherence(
            transcript_df,
            transcript_clusters,
            transcript_desc,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            cosine_threshold=options.module_coherence_threshold,
            stratify_by=options.stratify_by,
            reference_condition=options.reference_condition,
            trajectory_metric=options.trajectory_metric,
        )
        metabolite_coherence = module_member_coherence(
            metabolite_df,
            metabolite_clusters,
            metabolite_desc,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            cosine_threshold=options.module_coherence_threshold,
            stratify_by=options.stratify_by,
            reference_condition=options.reference_condition,
            trajectory_metric=options.trajectory_metric,
        )
    else:
        transcript_desc = calculate_pseudotime_trajectory_descriptors(
            transcript_fp,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            flat_threshold=options.trajectory_flat_threshold,
            stratify_by=options.stratify_by,
            trajectory_metric=options.trajectory_metric,
            effect_size_threshold=options.effect_size_threshold,
        )
        metabolite_desc = calculate_pseudotime_trajectory_descriptors(
            metabolite_fp,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            flat_threshold=options.trajectory_flat_threshold,
            stratify_by=options.stratify_by,
            trajectory_metric=options.trajectory_metric,
            effect_size_threshold=options.effect_size_threshold,
        )

        transcript_coherence = module_member_coherence_pseudotime(
            transcript_df,
            transcript_clusters,
            transcript_desc,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            cosine_threshold=options.module_coherence_threshold,
            stratify_by=options.stratify_by,
        )
        metabolite_coherence = module_member_coherence_pseudotime(
            metabolite_df,
            metabolite_clusters,
            metabolite_desc,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            cosine_threshold=options.module_coherence_threshold,
            stratify_by=options.stratify_by,
        )

    transcript_desc = merge_coherence_into_descriptors(
        transcript_desc,
        transcript_coherence,
    )
    metabolite_desc = merge_coherence_into_descriptors(
        metabolite_desc,
        metabolite_coherence,
    )

    transcript_desc.to_csv(
        f"{output_prefix}_RNA_trajectory_descriptors.csv",
        index=False,
    )
    metabolite_desc.to_csv(
        f"{output_prefix}_MET_trajectory_descriptors.csv",
        index=False,
    )

    # Separate trajectory summaries are retained for biological inspection.
    plot_trajectory_module_counts(
        transcript_desc,
        f"{output_prefix}_RNA",
        plot_format=options.plot_format,
    )
    plot_trajectory_module_counts(
        metabolite_desc,
        f"{output_prefix}_MET",
        plot_format=options.plot_format,
    )
    plot_trajectory_profiles(
        transcript_desc,
        f"{output_prefix}_RNA",
        plot_format=options.plot_format,
    )
    plot_trajectory_profiles(
        metabolite_desc,
        f"{output_prefix}_MET",
        plot_format=options.plot_format,
    )

    # ------------------------------
    # Same-pattern integration
    # ------------------------------
    shared_pattern_tables(
        transcript_desc,
        metabolite_desc,
        output_prefix,
    )

    # ------------------------------
    # Joint observed embedding
    # ------------------------------
    combined = prepare_joint_fingerprint_table(
        transcript_fp,
        metabolite_fp,
        transcript_desc,
        metabolite_desc,
    )
    combined.to_csv(
        f"{output_prefix}_joint_module_fingerprints.csv",
        index=False,
    )

    # PCA always generated.
    run_joint_embedding(
        combined,
        method="pca",
        output_prefix=output_prefix,
        metric=options.embedding_metric,
        color_by=options.embedding_color_by,
        size_by=options.embedding_size_by,
        plot_format=options.plot_format,
    )

    # UMAP/t-SNE according to the existing embedding switch.
    methods = []
    if options.embedding_method == "both":
        methods = ["umap", "tsne"]
    elif options.embedding_method in {"umap", "tsne"}:
        methods = [options.embedding_method]

    for method in methods:
        try:
            run_joint_embedding(
                combined,
                method=method,
                output_prefix=output_prefix,
                metric=options.embedding_metric,
                tsne_perplexity=options.tsne_perplexity,
                umap_neighbors=options.umap_neighbors,
                umap_min_dist=options.umap_min_dist,
                color_by=options.embedding_color_by,
                size_by=options.embedding_size_by,
                plot_format=options.plot_format,
            )
        except Exception as exc:
            print(f"WARNING: joint {method} embedding failed: {exc}")

    # ------------------------------
    # Optional continuous-time modelling layer (v2.3)
    # ------------------------------
    continuous_top_pairs = pd.DataFrame()
    continuous_enabled = (
        options.continuous_trajectory or options.trajectory_mode == "pseudotime"
    )
    if continuous_enabled:
        print("\n" + "-" * 78)
        print("MATE v2.4 continuous-time / pseudotime trajectory reconstruction")
        print("Modelled grid points are representations of fitted curves, not new observations.")
        if options.trajectory_mode == "pseudotime":
            print(
                "Pseudotime mode requires no mock/reference; response-onset fields "
                "that require a biological reference are intentionally left undefined."
            )
        print("-" * 78)

        transcript_cont_long, transcript_cont_desc, transcript_grid = build_continuous_trajectories(
            transcript_fp,
            transcript_desc,
            metadata,
            modality="Transcriptomics",
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            stratify_by=options.stratify_by,
            reference_condition=options.reference_condition,
            trajectory_mode=options.trajectory_mode,
            trajectory_metric=options.trajectory_metric,
            flat_threshold=options.trajectory_flat_threshold,
            effect_size_threshold=options.effect_size_threshold,
            trajectory_model=options.trajectory_model,
            grid_step=options.continuous_grid_step,
            n_bootstrap=options.continuous_bootstrap,
            ci_level=options.continuous_ci,
            random_seed=options.continuous_seed,
            gam_min_timepoints=options.gam_min_timepoints,
            gam_splines=options.gam_splines,
            gam_alpha=options.gam_alpha,
        )
        metabolite_cont_long, metabolite_cont_desc, metabolite_grid = build_continuous_trajectories(
            metabolite_fp,
            metabolite_desc,
            metadata,
            modality="Metabolomics",
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            stratify_by=options.stratify_by,
            reference_condition=options.reference_condition,
            trajectory_mode=options.trajectory_mode,
            trajectory_metric=options.trajectory_metric,
            flat_threshold=options.trajectory_flat_threshold,
            effect_size_threshold=options.effect_size_threshold,
            trajectory_model=options.trajectory_model,
            grid_step=options.continuous_grid_step,
            n_bootstrap=options.continuous_bootstrap,
            ci_level=options.continuous_ci,
            random_seed=options.continuous_seed + 100003,
            gam_min_timepoints=options.gam_min_timepoints,
            gam_splines=options.gam_splines,
            gam_alpha=options.gam_alpha,
        )

        transcript_cont_long.to_csv(
            f"{output_prefix}_RNA_continuous_trajectories.csv", index=False
        )
        metabolite_cont_long.to_csv(
            f"{output_prefix}_MET_continuous_trajectories.csv", index=False
        )
        transcript_cont_desc.to_csv(
            f"{output_prefix}_RNA_continuous_descriptors.csv", index=False
        )
        metabolite_cont_desc.to_csv(
            f"{output_prefix}_MET_continuous_descriptors.csv", index=False
        )
        print("Saved continuous trajectory and descriptor tables for both modalities.")

        continuous_all_pairs, continuous_top_pairs = continuous_cross_omics_lag_analysis(
            transcript_cont_long,
            metabolite_cont_long,
            transcript_desc,
            metabolite_desc,
            transcript_cont_desc,
            metabolite_cont_desc,
            output_prefix=output_prefix,
            lag_max=options.continuous_lag_max,
            lag_step=options.continuous_lag_step,
            default_grid_step=options.continuous_grid_step,
            lag_direction=options.lag_direction,
            top_n=options.continuous_top_pairs,
            min_similarity=options.min_lag_similarity,
        )

        plot_continuous_lag_pair_diagnostics(
            continuous_top_pairs,
            transcript_cont_long,
            metabolite_cont_long,
            output_prefix=output_prefix,
            top_n=options.continuous_plot_top_n,
            plot_format=options.plot_format,
        )

        if options.continuous_embedding:
            continuous_joint = prepare_continuous_embedding_table(
                transcript_cont_long,
                metabolite_cont_long,
                transcript_desc,
                metabolite_desc,
            )
            continuous_joint.to_csv(
                f"{output_prefix}_continuous_joint_trajectories.csv", index=False
            )
            if not continuous_joint.empty:
                run_joint_embedding(
                    continuous_joint,
                    method="pca",
                    output_prefix=f"{output_prefix}_continuous",
                    metric=options.embedding_metric,
                    color_by=options.embedding_color_by,
                    size_by=options.embedding_size_by,
                    plot_format=options.plot_format,
                )
                for method in methods:
                    try:
                        run_joint_embedding(
                            continuous_joint,
                            method=method,
                            output_prefix=f"{output_prefix}_continuous",
                            metric=options.embedding_metric,
                            tsne_perplexity=options.tsne_perplexity,
                            umap_neighbors=options.umap_neighbors,
                            umap_min_dist=options.umap_min_dist,
                            color_by=options.embedding_color_by,
                            size_by=options.embedding_size_by,
                            plot_format=options.plot_format,
                        )
                    except Exception as exc:
                        print(f"WARNING: continuous {method} embedding failed: {exc}")

    # ------------------------------
    # Lag-aware cross-omics analysis
    # ------------------------------
    all_pairs, top_pairs = cross_omics_lag_analysis(
        transcript_desc,
        metabolite_desc,
        output_prefix=output_prefix,
        max_lag_steps=options.max_lag_steps,
        lag_direction=options.lag_direction,
        top_n=options.top_lag_pairs,
        min_similarity=options.min_lag_similarity,
    )

    plot_lag_pair_diagnostics(
        top_pairs,
        transcript_desc,
        metabolite_desc,
        output_prefix=output_prefix,
        top_n=options.lag_plot_top_n,
        plot_format=options.plot_format,
    )

    # Optional Granger predictive directionality. IMPORTANT: only ORIGINAL
    # measured trajectory points are used; continuous/interpolated grids are excluded.
    if options.granger:
        if options.granger_scope == "all":
            granger_candidates = all_pairs
        elif continuous_enabled and not continuous_top_pairs.empty:
            granger_candidates = continuous_top_pairs.head(options.granger_top_n)
        else:
            granger_candidates = top_pairs.head(options.granger_top_n)

        run_granger_directionality(
            granger_candidates,
            transcript_desc,
            metabolite_desc,
            output_prefix=output_prefix,
            min_real_timepoints=options.granger_min_timepoints,
            max_lag=options.granger_max_lag,
            transform=options.granger_transform,
            scope=options.granger_scope,
            top_n=options.granger_top_n,
        )

    print("\nDual-omics MATE analysis complete.")
    print(
        "Positive BestLagSteps means the transcriptomics module precedes the "
        "metabolomics module."
    )
    print(
        "Lag-aligned similarity is supporting temporal evidence; it is not "
        "treated as proof of biochemical causality."
    )
    if options.continuous_trajectory:
        print(
            "Continuous-time lags are model-estimated temporal displacements. "
            "They do not increase the number of biological observations."
        )
    if options.granger:
        print(
            "Granger results, when enabled, indicate predictive precedence rather "
            "than biochemical causality."
        )



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(options: argparse.Namespace) -> None:
    """Run the MATE v2.4 dual-omics workflow."""
    run_dual_omics_mode(options)


if __name__ == "__main__":
    Options = get_args()
    main(Options)
