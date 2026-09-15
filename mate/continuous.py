"""Continuous curve fitting and replicate-bootstrap uncertainty estimates."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

try:
    from statsmodels.gam.api import BSplines, GLMGam
except ImportError:
    GLMGam = None
    BSplines = None

from .trajectories import (
    _numeric_times,
    condition_time_sample_groups,
    fingerprint_sample_columns,
    get_trajectory_time_columns,
    resolve_stratify_column,
    resolve_time_column,
    resolve_time_values,
    standardized_mean_difference,
)


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
                # aliases retained for compatibility with downstream code
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
