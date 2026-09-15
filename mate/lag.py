"""Discrete and continuous lag-aware cross-omics comparisons."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .trajectories import _numeric_times, descriptor_profile_matrix
from .utils import natural_key


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
