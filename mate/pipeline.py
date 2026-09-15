"""Orchestration of the MATE dual-omics analysis and output files."""

import argparse

import pandas as pd

from . import __version__
from .continuous import build_continuous_trajectories
from .embeddings import (
    prepare_continuous_embedding_table,
    prepare_joint_fingerprint_table,
    run_joint_embedding,
)
from .fingerprints import calculate_fingerprints_for_matrix, prefix_module_ids
from .integration import (
    merge_coherence_into_descriptors,
    module_member_coherence,
    module_member_coherence_pseudotime,
    shared_pattern_tables,
)
from .io import (
    common_sample_columns,
    load_clusters,
    normalize_feature_table,
    read_feature_table,
    read_metadata,
)
from .lag import continuous_cross_omics_lag_analysis, cross_omics_lag_analysis
from .plotting import (
    plot_continuous_lag_pair_diagnostics,
    plot_lag_pair_diagnostics,
    plot_trajectory_module_counts,
    plot_trajectory_profiles,
)
from .statistics import run_granger_directionality
from .trajectories import (
    calculate_pseudotime_trajectory_descriptors,
    calculate_trajectory_descriptors,
)
from .utils import safe_output_prefix


def run_dual_omics_mode(options: argparse.Namespace) -> None:
    """
    Full MATE dual-omics workflow.

    Transcriptomics and metabolomics are processed independently until their
    module metafingerprints and trajectories have been generated. Integration
    then occurs at the module trajectory/fingerprint level.
    """
    output_prefix = safe_output_prefix(options.outfile)

    print("\n" + "=" * 78)
    print(f"MATE v{__version__}")
    print("=" * 78)
    print("Transcriptomics and metabolomics modules are calculated independently.")

    if options.trajectory_mode == "reference":
        if not str(options.reference_condition).strip():
            raise ValueError(
                "--reference_condition is required when --trajectory_mode reference."
            )
        print(
            f"Trajectory mode: reference-aware; each treatment is compared with "
            f"matched reference '{options.reference_condition}' at every selected time point."
        )
    else:
        if str(options.reference_condition).strip():
            print(
                "WARNING: --reference_condition is ignored in "
                "--trajectory_mode pseudotime."
            )
        print(
            "Trajectory mode: pseudotime/ordered-time; module trajectories are "
            "followed directly without a mock/reference condition."
        )
        print(
            "NOTE: MATE uses the supplied chronological/pseudotime ordering; "
            "it does not infer pseudotime from the omics matrices."
        )

    print("Integration occurs after modality-specific trajectories are obtained.\n")

    metadata = read_metadata(options.metadata)

    transcript_df = read_feature_table(options.transcriptomics_table)
    metabolite_df = read_feature_table(options.metabolomics_table)

    transcript_df = normalize_feature_table(
        transcript_df,
        do_zscore=not options.no_zscore,
    )
    metabolite_df = normalize_feature_table(
        metabolite_df,
        do_zscore=not options.no_zscore,
    )

    common_samples = common_sample_columns(
        transcript_df,
        metabolite_df,
        metadata,
    )

    transcript_df = transcript_df[["feature"] + common_samples].copy()
    metabolite_df = metabolite_df[["feature"] + common_samples].copy()

    transcript_clusters = load_clusters(
        options.transcriptomics_db,
        options.decay_rate,
    )
    metabolite_clusters = load_clusters(
        options.metabolomics_db,
        options.decay_rate,
    )

    transcript_clusters = prefix_module_ids(transcript_clusters, "RNA")
    metabolite_clusters = prefix_module_ids(metabolite_clusters, "MET")

    # Keep cluster member tables compatible with prefixed IDs.
    transcript_clusters_file = f"{output_prefix}_RNA_modules.csv"
    metabolite_clusters_file = f"{output_prefix}_MET_modules.csv"
    transcript_clusters.to_csv(transcript_clusters_file, index=False)
    metabolite_clusters.to_csv(metabolite_clusters_file, index=False)

    # ------------------------------
    # Independent fingerprints
    # ------------------------------
    transcript_fp, transcript_missing = calculate_fingerprints_for_matrix(
        transcript_df,
        transcript_clusters,
        desc="Transcriptomics module fingerprints",
    )
    metabolite_fp, metabolite_missing = calculate_fingerprints_for_matrix(
        metabolite_df,
        metabolite_clusters,
        desc="Metabolomics module fingerprints",
    )

    transcript_fp.to_csv(
        f"{output_prefix}_RNA_global_fingerprints.csv",
        index=False,
    )
    metabolite_fp.to_csv(
        f"{output_prefix}_MET_global_fingerprints.csv",
        index=False,
    )

    if not transcript_missing.empty:
        transcript_missing.to_csv(
            f"{output_prefix}_RNA_missing_features.csv",
            index=False,
        )
    if not metabolite_missing.empty:
        metabolite_missing.to_csv(
            f"{output_prefix}_MET_missing_features.csv",
            index=False,
        )

    # ------------------------------
    # Independent trajectories
    # ------------------------------
    if options.trajectory_mode == "reference":
        transcript_desc = calculate_trajectory_descriptors(
            transcript_fp,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            flat_threshold=options.trajectory_flat_threshold,
            stratify_by=options.stratify_by,
            trajectory_metric=options.trajectory_metric,
            effect_size_threshold=options.effect_size_threshold,
            reference_condition=options.reference_condition,
        )
        metabolite_desc = calculate_trajectory_descriptors(
            metabolite_fp,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            flat_threshold=options.trajectory_flat_threshold,
            stratify_by=options.stratify_by,
            trajectory_metric=options.trajectory_metric,
            effect_size_threshold=options.effect_size_threshold,
            reference_condition=options.reference_condition,
        )

        transcript_coherence = module_member_coherence(
            transcript_df,
            transcript_clusters,
            transcript_desc,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            cosine_threshold=options.module_coherence_threshold,
            stratify_by=options.stratify_by,
            reference_condition=options.reference_condition,
            trajectory_metric=options.trajectory_metric,
        )
        metabolite_coherence = module_member_coherence(
            metabolite_df,
            metabolite_clusters,
            metabolite_desc,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            cosine_threshold=options.module_coherence_threshold,
            stratify_by=options.stratify_by,
            reference_condition=options.reference_condition,
            trajectory_metric=options.trajectory_metric,
        )
    else:
        transcript_desc = calculate_pseudotime_trajectory_descriptors(
            transcript_fp,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            flat_threshold=options.trajectory_flat_threshold,
            stratify_by=options.stratify_by,
            trajectory_metric=options.trajectory_metric,
            effect_size_threshold=options.effect_size_threshold,
        )
        metabolite_desc = calculate_pseudotime_trajectory_descriptors(
            metabolite_fp,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            flat_threshold=options.trajectory_flat_threshold,
            stratify_by=options.stratify_by,
            trajectory_metric=options.trajectory_metric,
            effect_size_threshold=options.effect_size_threshold,
        )

        transcript_coherence = module_member_coherence_pseudotime(
            transcript_df,
            transcript_clusters,
            transcript_desc,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            cosine_threshold=options.module_coherence_threshold,
            stratify_by=options.stratify_by,
        )
        metabolite_coherence = module_member_coherence_pseudotime(
            metabolite_df,
            metabolite_clusters,
            metabolite_desc,
            metadata,
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            cosine_threshold=options.module_coherence_threshold,
            stratify_by=options.stratify_by,
        )

    transcript_desc = merge_coherence_into_descriptors(
        transcript_desc,
        transcript_coherence,
    )
    metabolite_desc = merge_coherence_into_descriptors(
        metabolite_desc,
        metabolite_coherence,
    )

    transcript_desc.to_csv(
        f"{output_prefix}_RNA_trajectory_descriptors.csv",
        index=False,
    )
    metabolite_desc.to_csv(
        f"{output_prefix}_MET_trajectory_descriptors.csv",
        index=False,
    )

    # Separate trajectory summaries are retained for biological inspection.
    plot_trajectory_module_counts(
        transcript_desc,
        f"{output_prefix}_RNA",
        plot_format=options.plot_format,
    )
    plot_trajectory_module_counts(
        metabolite_desc,
        f"{output_prefix}_MET",
        plot_format=options.plot_format,
    )
    plot_trajectory_profiles(
        transcript_desc,
        f"{output_prefix}_RNA",
        plot_format=options.plot_format,
    )
    plot_trajectory_profiles(
        metabolite_desc,
        f"{output_prefix}_MET",
        plot_format=options.plot_format,
    )

    # ------------------------------
    # Same-pattern integration
    # ------------------------------
    shared_pattern_tables(
        transcript_desc,
        metabolite_desc,
        output_prefix,
    )

    # ------------------------------
    # Joint observed embedding
    # ------------------------------
    combined = prepare_joint_fingerprint_table(
        transcript_fp,
        metabolite_fp,
        transcript_desc,
        metabolite_desc,
    )
    combined.to_csv(
        f"{output_prefix}_joint_module_fingerprints.csv",
        index=False,
    )

    # PCA always generated.
    run_joint_embedding(
        combined,
        method="pca",
        output_prefix=output_prefix,
        metric=options.embedding_metric,
        color_by=options.embedding_color_by,
        size_by=options.embedding_size_by,
        plot_format=options.plot_format,
    )

    # UMAP/t-SNE according to the existing embedding switch.
    methods = []
    if options.embedding_method == "both":
        methods = ["umap", "tsne"]
    elif options.embedding_method in {"umap", "tsne"}:
        methods = [options.embedding_method]

    for method in methods:
        try:
            run_joint_embedding(
                combined,
                method=method,
                output_prefix=output_prefix,
                metric=options.embedding_metric,
                tsne_perplexity=options.tsne_perplexity,
                umap_neighbors=options.umap_neighbors,
                umap_min_dist=options.umap_min_dist,
                color_by=options.embedding_color_by,
                size_by=options.embedding_size_by,
                plot_format=options.plot_format,
            )
        except Exception as exc:
            print(f"WARNING: joint {method} embedding failed: {exc}")

    # ------------------------------
    # Optional continuous-time modelling layer
    # ------------------------------
    continuous_top_pairs = pd.DataFrame()
    continuous_enabled = (
        options.continuous_trajectory or options.trajectory_mode == "pseudotime"
    )
    if continuous_enabled:
        print("\n" + "-" * 78)
        print(f"MATE v{__version__} continuous-time / pseudotime trajectory reconstruction")
        print("Modelled grid points are representations of fitted curves, not new observations.")
        if options.trajectory_mode == "pseudotime":
            print(
                "Pseudotime mode requires no mock/reference; response-onset fields "
                "that require a biological reference are intentionally left undefined."
            )
        print("-" * 78)

        transcript_cont_long, transcript_cont_desc, transcript_grid = build_continuous_trajectories(
            transcript_fp,
            transcript_desc,
            metadata,
            modality="Transcriptomics",
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            stratify_by=options.stratify_by,
            reference_condition=options.reference_condition,
            trajectory_mode=options.trajectory_mode,
            trajectory_metric=options.trajectory_metric,
            flat_threshold=options.trajectory_flat_threshold,
            effect_size_threshold=options.effect_size_threshold,
            trajectory_model=options.trajectory_model,
            grid_step=options.continuous_grid_step,
            n_bootstrap=options.continuous_bootstrap,
            ci_level=options.continuous_ci,
            random_seed=options.continuous_seed,
            gam_min_timepoints=options.gam_min_timepoints,
            gam_splines=options.gam_splines,
            gam_alpha=options.gam_alpha,
        )
        metabolite_cont_long, metabolite_cont_desc, metabolite_grid = build_continuous_trajectories(
            metabolite_fp,
            metabolite_desc,
            metadata,
            modality="Metabolomics",
            time_column=options.time_column,
            time_values=options.time_values,
            n_timepoints=options.n_timepoints,
            stratify_by=options.stratify_by,
            reference_condition=options.reference_condition,
            trajectory_mode=options.trajectory_mode,
            trajectory_metric=options.trajectory_metric,
            flat_threshold=options.trajectory_flat_threshold,
            effect_size_threshold=options.effect_size_threshold,
            trajectory_model=options.trajectory_model,
            grid_step=options.continuous_grid_step,
            n_bootstrap=options.continuous_bootstrap,
            ci_level=options.continuous_ci,
            random_seed=options.continuous_seed + 100003,
            gam_min_timepoints=options.gam_min_timepoints,
            gam_splines=options.gam_splines,
            gam_alpha=options.gam_alpha,
        )

        transcript_cont_long.to_csv(
            f"{output_prefix}_RNA_continuous_trajectories.csv", index=False
        )
        metabolite_cont_long.to_csv(
            f"{output_prefix}_MET_continuous_trajectories.csv", index=False
        )
        transcript_cont_desc.to_csv(
            f"{output_prefix}_RNA_continuous_descriptors.csv", index=False
        )
        metabolite_cont_desc.to_csv(
            f"{output_prefix}_MET_continuous_descriptors.csv", index=False
        )
        print("Saved continuous trajectory and descriptor tables for both modalities.")

        continuous_all_pairs, continuous_top_pairs = continuous_cross_omics_lag_analysis(
            transcript_cont_long,
            metabolite_cont_long,
            transcript_desc,
            metabolite_desc,
            transcript_cont_desc,
            metabolite_cont_desc,
            output_prefix=output_prefix,
            lag_max=options.continuous_lag_max,
            lag_step=options.continuous_lag_step,
            default_grid_step=options.continuous_grid_step,
            lag_direction=options.lag_direction,
            top_n=options.continuous_top_pairs,
            min_similarity=options.min_lag_similarity,
        )

        plot_continuous_lag_pair_diagnostics(
            continuous_top_pairs,
            transcript_cont_long,
            metabolite_cont_long,
            output_prefix=output_prefix,
            top_n=options.continuous_plot_top_n,
            plot_format=options.plot_format,
        )

        if options.continuous_embedding:
            continuous_joint = prepare_continuous_embedding_table(
                transcript_cont_long,
                metabolite_cont_long,
                transcript_desc,
                metabolite_desc,
            )
            continuous_joint.to_csv(
                f"{output_prefix}_continuous_joint_trajectories.csv", index=False
            )
            if not continuous_joint.empty:
                run_joint_embedding(
                    continuous_joint,
                    method="pca",
                    output_prefix=f"{output_prefix}_continuous",
                    metric=options.embedding_metric,
                    color_by=options.embedding_color_by,
                    size_by=options.embedding_size_by,
                    plot_format=options.plot_format,
                )
                for method in methods:
                    try:
                        run_joint_embedding(
                            continuous_joint,
                            method=method,
                            output_prefix=f"{output_prefix}_continuous",
                            metric=options.embedding_metric,
                            tsne_perplexity=options.tsne_perplexity,
                            umap_neighbors=options.umap_neighbors,
                            umap_min_dist=options.umap_min_dist,
                            color_by=options.embedding_color_by,
                            size_by=options.embedding_size_by,
                            plot_format=options.plot_format,
                        )
                    except Exception as exc:
                        print(f"WARNING: continuous {method} embedding failed: {exc}")

    # ------------------------------
    # Lag-aware cross-omics analysis
    # ------------------------------
    all_pairs, top_pairs = cross_omics_lag_analysis(
        transcript_desc,
        metabolite_desc,
        output_prefix=output_prefix,
        max_lag_steps=options.max_lag_steps,
        lag_direction=options.lag_direction,
        top_n=options.top_lag_pairs,
        min_similarity=options.min_lag_similarity,
    )

    plot_lag_pair_diagnostics(
        top_pairs,
        transcript_desc,
        metabolite_desc,
        output_prefix=output_prefix,
        top_n=options.lag_plot_top_n,
        plot_format=options.plot_format,
    )

    # Optional Granger predictive directionality. IMPORTANT: only ORIGINAL
    # measured trajectory points are used; continuous/interpolated grids are excluded.
    if options.granger:
        if options.granger_scope == "all":
            granger_candidates = all_pairs
        elif continuous_enabled and not continuous_top_pairs.empty:
            granger_candidates = continuous_top_pairs.head(options.granger_top_n)
        else:
            granger_candidates = top_pairs.head(options.granger_top_n)

        run_granger_directionality(
            granger_candidates,
            transcript_desc,
            metabolite_desc,
            output_prefix=output_prefix,
            min_real_timepoints=options.granger_min_timepoints,
            max_lag=options.granger_max_lag,
            transform=options.granger_transform,
            scope=options.granger_scope,
            top_n=options.granger_top_n,
        )

    print("\nDual-omics MATE analysis complete.")
    print(
        "Positive BestLagSteps means the transcriptomics module precedes the "
        "metabolomics module."
    )
    print(
        "Lag-aligned similarity is supporting temporal evidence; it is not "
        "treated as proof of biochemical causality."
    )
    if options.continuous_trajectory:
        print(
            "Continuous-time lags are model-estimated temporal displacements. "
            "They do not increase the number of biological observations."
        )
    if options.granger:
        print(
            "Granger results, when enabled, indicate predictive precedence rather "
            "than biochemical causality."
        )


def main(options: argparse.Namespace) -> None:
    """Run the MATE dual-omics workflow."""
    run_dual_omics_mode(options)
