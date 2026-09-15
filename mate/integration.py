"""Module-member coherence and cross-omics shared-pattern summaries."""

from typing import Tuple

import numpy as np
import pandas as pd

from .io import parse_members
from .trajectories import (
    condition_time_sample_groups,
    get_trajectory_time_columns,
    resolve_stratify_column,
    resolve_time_column,
    resolve_time_values,
    standardized_mean_difference,
)


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
