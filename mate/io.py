"""Feature tables, sample metadata, and MEANtools-compatible SQLite modules."""

import re
import sqlite3
from contextlib import closing
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import zscore

from .utils import find_column, natural_key


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


def _read_cluster_tables(sqlite_db_name: str) -> Dict[str, pd.DataFrame]:
    """Read tables containing 'clone', matching gizmos.import_from_sql(clone=True).

    Keep SQLite loading independent of the chemistry utilities in gizmos.py.
    """
    with closing(sqlite3.connect(sqlite_db_name)) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        frames = {}
        for (name,) in tables:
            if "clone" in name.lower():
                quoted_name = name.replace('"', '""')
                frames[name] = pd.read_sql_query(
                    f'SELECT * FROM "{quoted_name}";', connection
                )
        return frames


def load_clusters(sqlite_db_name: str, decay_rate: int) -> pd.DataFrame:
    """Import cluster tables from SQLite and keep clusters matching DR_<decay_rate> when available."""
    print(f"Importing clusters from SQLite database: {sqlite_db_name}")
    all_tables = _read_cluster_tables(sqlite_db_name)

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
