"""Trajectory, embedding, and lag-diagnostic figures."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .lag import _aligned_indices, _normalize_profile_for_lag
from .trajectories import (
    _numeric_times,
    descriptor_profile_matrix,
    get_trajectory_time_columns,
    trajectory_time_labels,
)
from .utils import natural_key, safe_output_prefix


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
