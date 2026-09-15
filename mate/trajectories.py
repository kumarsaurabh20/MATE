"""Measured-time trajectories in reference-aware and pseudotime modes."""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import find_column, natural_key, parse_comma_list


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
