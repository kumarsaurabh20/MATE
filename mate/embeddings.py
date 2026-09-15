"""PCA, t-SNE, and optional UMAP representations of module trajectories."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap
except ImportError:
    umap = None

from .plotting import plot_embedding, plot_joint_embedding
from .trajectories import get_trajectory_time_columns


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
