"""SVD module fingerprints and modality-specific module identifiers."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io import parse_members
from .utils import orient_vector


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


def combine_groups(fingerprints_by_group: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for group, df in fingerprints_by_group.items():
        tmp = df.copy()
        tmp["Group"] = group
        frames.append(tmp)
    if not frames:
        raise ValueError("No fingerprint groups to combine.")
    return pd.concat(frames, axis=0, ignore_index=True)


def prefix_module_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix module IDs so transcript and metabolite module names cannot collide."""
    out = df.copy()
    out["OriginalID"] = out["ID"].astype(str)
    out["ID"] = prefix + "::" + out["ID"].astype(str)
    return out
