#!/usr/bin/env python3
"""Regenerate workflow.png with: python docs/render_workflow.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 12), facecolor="#f8fafc")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ink, muted, line = "#182b42", "#526477", "#8998a9"

    def box(x, y, width, height, title, subtitle, fill, edge, optional=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.3, facecolor=fill, edgecolor=edge,
            linestyle="--" if optional else "-", zorder=2,
        ))
        ax.text(x + width / 2, y + height * 0.72, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color=ink, zorder=3)
        ax.text(x + width / 2, y + height * 0.32, subtitle, ha="center", va="center",
                fontsize=10.5, linespacing=1.5, color=muted, zorder=3)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color=line, zorder=1,
        ))

    def wire(points):
        ax.plot([p[0] for p in points], [p[1] for p in points],
                color=line, linewidth=1.4, zorder=1)

    ax.text(0.5, 0.958, "MATE  /  ANALYSIS WORKFLOW", ha="center",
            fontsize=22, fontweight="bold", color=ink)
    ax.text(0.5, 0.925, "Independent omics modules → shared temporal comparisons",
            ha="center", fontsize=12, color=muted)

    box(0.06, 0.79, 0.39, 0.09, "Transcriptomics inputs",
        "Feature table + transcript modules", "#e6f5f2", "#439a8a")
    box(0.55, 0.79, 0.39, 0.09, "Metabolomics inputs",
        "Feature table + metabolite modules", "#efecfa", "#9280c2")
    box(0.06, 0.655, 0.39, 0.085, "Transcript fingerprints",
        "Normalize features; summarize by SVD", "#e6f5f2", "#439a8a")
    box(0.55, 0.655, 0.39, 0.085, "Metabolite fingerprints",
        "Normalize features; summarize by SVD", "#efecfa", "#9280c2")
    arrow((0.255, 0.782), (0.255, 0.748))
    arrow((0.745, 0.782), (0.745, 0.748))

    box(0.13, 0.505, 0.74, 0.095, "Observed module trajectories",
        "Metadata supplies condition, ordered time, and replicate groups\n"
        "Matched-reference responses or reference-free pseudotime",
        "#eaf0f8", "#819bb9")
    wire([(0.255, 0.647), (0.255, 0.622), (0.745, 0.622), (0.745, 0.647)])
    arrow((0.5, 0.622), (0.5, 0.608))

    box(0.06, 0.35, 0.39, 0.10, "Measured-time descriptors",
        "Patterns, peaks, timing, range\nMember-to-module coherence",
        "#ffffff", "#819bb9")
    box(0.55, 0.35, 0.39, 0.10, "Continuous representation*",
        "Linear / PCHIP / GAM curves\nReplicate-bootstrap uncertainty",
        "#fff5e5", "#c7a468", optional=True)
    wire([(0.5, 0.497), (0.5, 0.477)])
    wire([(0.255, 0.477), (0.745, 0.477)])
    arrow((0.255, 0.477), (0.255, 0.458))
    arrow((0.745, 0.477), (0.745, 0.458))

    box(0.06, 0.195, 0.39, 0.10, "Joint embeddings",
        "PCA; optional UMAP / t-SNE\nObserved or fitted trajectories",
        "#ffffff", "#819bb9")
    box(0.55, 0.195, 0.39, 0.10, "Cross-omics comparisons",
        "Shared patterns; discrete / continuous lags\nOptional measured-time Granger tests",
        "#ffffff", "#819bb9")
    wire([(0.255, 0.342), (0.255, 0.324), (0.745, 0.324), (0.745, 0.342)])
    arrow((0.255, 0.324), (0.255, 0.303))
    arrow((0.745, 0.324), (0.745, 0.303))

    box(0.13, 0.06, 0.74, 0.08, "Tables, figures, and prioritized module pairs",
        "Positive lag: transcriptomics precedes metabolomics",
        "#eaf0f8", "#819bb9")
    wire([(0.255, 0.187), (0.255, 0.165), (0.745, 0.165), (0.745, 0.187)])
    arrow((0.5, 0.165), (0.5, 0.148))
    ax.text(0.5, 0.024,
            "* Optional in reference mode; enabled in pseudotime mode. Fitted grid points are not new observations.",
            ha="center", fontsize=9, color=muted)

    destination = Path(__file__).with_name("workflow.png")
    fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", destination)


if __name__ == "__main__":
    main()
