"""Clustering of module PCA coordinates."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex
from sklearn.cluster import (
    DBSCAN,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    SpectralClustering,
    estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture

from .utils import natural_key


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
