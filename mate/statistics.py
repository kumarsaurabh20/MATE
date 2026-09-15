"""Trajectory separation statistics and measured-time Granger tests."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tsa.stattools import grangercausalitytests
except ImportError:
    grangercausalitytests = None
    multipletests = None

from .embeddings import prepare_embedding_matrix
from .trajectories import (
    _numeric_times,
    descriptor_profile_matrix,
    get_trajectory_time_columns,
    trajectory_time_labels,
)


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
