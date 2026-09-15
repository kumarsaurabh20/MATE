# MATE v3.0 output columns

This guide describes the current `mate/` implementation. Filenames below use
`<prefix>` for `--outfile` (default `MATE`); `RNA` denotes transcriptomics and
`MET` denotes metabolomics. Outputs are written to the current working directory.
The prefix is sanitized as a filename, so it is not an output-directory option.

Columns depend on trajectory mode, metric, and enabled analysis options. Blank
numeric cells indicate missing or undefined values; strings such as `NA` denote
an undefined classification. An empty candidate set can produce an empty CSV,
and optional outputs are only created when that analysis runs.

## 1. Module definitions and fingerprints

Files: `<prefix>_RNA_modules.csv`, `<prefix>_MET_modules.csv`, and
`<prefix>_{RNA|MET}_global_fingerprints.csv`.

| Column | Meaning |
| --- | --- |
| `ID` | Module identifier prefixed with `RNA::` or `MET::`. |
| `OriginalID` | Identifier before the modality prefix; present in the module tables. |
| `Members` | Feature identifiers from the module database. |
| `n_features_found` | Number of module members found in the feature table. |
| `n_features_missing` | Number of members absent from that table. |
| Sample-name columns | SVD-derived module fingerprint values for individual samples. |

The module fingerprint is oriented toward its mean feature profile. A single
usable feature retains its own sample profile. Missing members are listed in
`<prefix>_{RNA|MET}_missing_features.csv` (`ID`, `missing_feature`), when present.

## 2. Observed trajectory descriptors

Files: `<prefix>_{RNA|MET}_trajectory_descriptors.csv`.
Each row describes one module in one treatment/stratum.

### Identifiers and design

| Column | Meaning |
| --- | --- |
| `ID`, `Members` | Module identifier and member list. |
| `StateID` | Module identifier plus condition, joined with `::`; use this to join condition-specific results. |
| `Condition` | Treatment/stratum being analyzed; `ALL` for pooled trajectories. |
| `ReferenceCondition` | Matched reference label in reference mode; empty in pseudotime mode. |
| `TrajectoryMode` | `reference` or `pseudotime`. |
| `TrajectoryMetric` | `effect_size` or `delta`. |
| `NTimepoints` | Number of selected measured time points. |
| `NTransitions` | Number of adjacent transitions: `NTimepoints - 1`. |

### Time-point and transition columns

`T1_12` means the first selected time point, labeled `12`; the same naming rule
applies to `T2_24`, `T3_48`, and other designs.

| Column pattern | Meaning |
| --- | --- |
| `T<i>_<time>` | Reference mode: treatment-minus-reference mean difference (`delta`) or standardized mean difference (`effect_size`). Pseudotime: mean sample fingerprint at that ordered time. |
| `TreatmentMean_T<i>_<time>`, `ReferenceMean_T<i>_<time>` | Mean sample fingerprints in the two groups; reference mode only. |
| `RawResponse_T<i>_<time>` | Unstandardized difference between treatment and reference fingerprint means. |
| `TreatmentVsReferenceEffectSize_T<i>_<time>` | Standardized response when using reference mode with `effect_size`. |
| `NrepTreatment_T<i>`, `NrepReference_T<i>` | Counts of finite fingerprint values used in the respective groups. |
| `ResponseState_T<i>_<time>` | `UP`, `DOWN`, or `FLAT` relative to the reference at that time. |
| `ObservedMean_T<i>_<time>`, `Nrep_T<i>` | Pseudotime mean fingerprint and count of finite replicate values. |
| `Delta_T<i>_to_T<j>` | Difference between adjacent trajectory values. |
| `TransitionStat_T<i>_to_T<j>` | Statistic used to classify that transition. In reference mode this is the change in response; in pseudotime effect-size mode it is the standardized difference between adjacent replicate groups. |
| `EffectSizeChange_T<i>_to_T<j>` | Change between reference-response effect sizes. |
| `EffectSize_T<i>_to_T<j>` | Adjacent-group effect size in pseudotime mode. |

Standardized mean differences fall back to raw mean differences when replicate
variance is zero or insufficient. Classification uses `--effect_size_threshold`
or `--trajectory_flat_threshold` as appropriate.

### Summary and coherence columns

| Column | Meaning |
| --- | --- |
| `Pattern` | Adjacent transition labels joined by `-`, such as `UP-DOWN`. |
| `ResponseStatePattern` | Reference-relative labels at every measured time; `NA` in pseudotime mode. |
| `Peak`, `PeakIndex` | Time label and 1-based position of the largest signed trajectory value. |
| `AbsPeak`, `AbsPeakIndex` | Time label and 1-based position of the largest absolute trajectory value. |
| `ResponseTiming` | Reference mode: timing of the absolute peak. Pseudotime: alias of `TrajectoryTiming`, which uses the signed peak. Labels include `EARLY`, `MID`, `LATE`, and interior `T<i>` positions. |
| `DominantResponse` | Reference-relative classification at the absolute peak; `NA` in pseudotime mode. |
| `Delta_first_last` | Last trajectory value minus first. |
| `OverallTransitionStat`, `OverallTrend` | First-to-last classification statistic and label, using the selected mode/metric. |
| `Max`, `Min`, `DynamicRange` | Maximum, minimum, and their difference. |
| `MaxAbsDelta` | Largest absolute difference between adjacent trajectory values. |
| `Variance` | Population variance of the time-point trajectory values. |
| `AUC` | Trapezoidal area using **unit spacing between selected time points**, not elapsed time. |
| `NMembersEvaluated` | Number of members with a valid cosine comparison to the module trajectory. |
| `MemberMeanCosine`, `MemberMedianCosine` | Mean/median member-to-module trajectory cosine similarity. |
| `MemberConcordanceFraction` | Fraction of evaluated members at or above `--module_coherence_threshold`. |

`Pattern` describes changes between times; `ResponseStatePattern` describes
response relative to a reference at each time. Read them together in reference
mode. Values summarize module fingerprints, not raw expression fold changes.

`<prefix>_{RNA|MET}_trajectory_pattern_counts.csv` contains `Pattern` and
`ModuleCount`, pooled across the descriptor rows for that modality.

## 3. Shared patterns and embeddings

`<prefix>_shared_trajectory_patterns.csv` reports, for each `Condition`/`Pattern`,
`TranscriptModules`, `MetaboliteModules`, `SharedPattern` (both counts positive),
and `PotentialCrossOmicsPairs` (the product of the counts).
`<prefix>_same_pattern_module_pairs.csv` lists the matching module and state pairs.

`<prefix>_joint_module_fingerprints.csv` combines the observed descriptor tables
and adds `Modality`. Despite its historical filename, the current dual-omics
workflow embeds the observed `T<i>_<time>` trajectory columns when available.
Other descriptors serve as annotations.

| Embedding output columns | Meaning |
| --- | --- |
| `StateID`, `ID` | Condition-specific state key; embedding output `ID` is set to `StateID`. |
| `Modality` | `Transcriptomics` or `Metabolomics`. |
| `PC1`, `PC2` | Joint PCA coordinates. |
| `tSNE1`, `tSNE2` | Joint t-SNE coordinates when requested. |
| `UMAP1`, `UMAP2` | Joint UMAP coordinates when requested and available. |
| `EmbeddingMethod` | `pca`, `tsne`, or `umap`. |
| `n_input_columns` | Number of nonconstant numeric input columns retained. |

Coordinate tables are named `<prefix>_joint_<method>_embedding.csv`; corresponding
figures use the selected plot format. PCA is always generated. Continuous
embedding outputs add `_continuous` to the prefix and use fitted regular-grid
trajectory values (`<prefix>_continuous_joint_trajectories.csv`).

## 4. Discrete lag comparisons

Files: `<prefix>_cross_omics_lag_all_pairs.csv` and
`<prefix>_cross_omics_lag_top_pairs.csv`. Pairs are compared within each condition.

| Column | Meaning |
| --- | --- |
| `TranscriptModule`, `MetaboliteModule` | Paired modality-specific module identifiers. |
| `TranscriptState`, `MetaboliteState`, `Condition` | Condition-specific identities. |
| `TranscriptPattern`, `MetabolitePattern` | Observed trajectory patterns. |
| `TranscriptOverallTrend`, `MetaboliteOverallTrend` | First-to-last trend labels. |
| `TranscriptPeak`, `MetabolitePeak` | Observed signed peak time labels. |
| `TranscriptDynamicRange`, `MetaboliteDynamicRange` | Observed trajectory ranges. |
| `TranscriptMemberCoherence`, `MetaboliteMemberCoherence` | Respective member concordance fractions. |
| `Similarity_Lag0`, `RMSE_Lag0` | Cosine similarity and root-mean-square error with no displacement. |
| `BestLagSteps` | Selected shift in measured-time indices. Positive means transcriptomics leads; negative means metabolomics leads. |
| `BestLagTimeMedian` | Median time displacement for the aligned measured points, when numeric times can be resolved. |
| `LagTimeDifferences` | Semicolon-separated actual time displacements, useful with irregular sampling. |
| `BestLagSimilarity`, `BestLagRMSE` | Cosine similarity and error at the selected lag. |
| `SimilarityGain` | `BestLagSimilarity - Similarity_Lag0`. |
| `NAlignedTimepoints` | Number of measured points retained after shifting; at least two are needed. |
| `RelationshipType` | `Transcriptomics_leads`, `Metabolomics_leads`, or `Synchronous`. |

Profiles are independently centered and L2-normalized before alignment. The best
lag maximizes cosine similarity, with error used to break ties. The top-pairs
table applies `--min_lag_similarity`, sorts by similarity and gain, then retains
`--top_lag_pairs` rows across conditions. Lag similarity is a prioritization score.

## 5. Continuous trajectories and summaries

Enabled by `--continuous_trajectory`, and automatically in pseudotime mode.
Files: `<prefix>_{RNA|MET}_continuous_trajectories.csv` and
`<prefix>_{RNA|MET}_continuous_descriptors.csv`.

| Trajectory column | Meaning |
| --- | --- |
| `Time` | Numeric coordinate on the output grid. |
| `ObservedTimepoint` | Whether this coordinate is an original measured time. |
| `RegularGridPoint` | Whether it belongs to the regular modeling grid; a point can have both flags. |
| `ObservedTrajectoryValue`, `ObservedResponse` | Observed trajectory value and its compatibility alias; absent at unmeasured times. |
| `PredictedTrajectoryValue`, `PredictedResponse` | Fitted value and its compatibility alias. |
| `CI_Lower`, `CI_Upper` | Pointwise replicate-bootstrap interval; undefined when bootstrapping is disabled or unsuccessful. |
| `TrajectoryModel` | Model actually used, including any fallback. |
| `NObservedTimepoints` | Number of original measured time points, independent of grid size. |

| Descriptor column | Meaning |
| --- | --- |
| `TrajectoryModelRequested`, `TrajectoryModelUsed` | Requested model and actual fit/fallback. |
| `ContinuousGridStep` | Regular grid spacing in the metadata time units. |
| `BootstrapRequested`, `BootstrapSuccessful` | Requested resamples and successful fitted bootstrap curves. |
| `ContinuousPeakTime`, `ContinuousPeakResponse` | Signed peak time/value on the regular grid. |
| `ContinuousAbsPeakTime`, `ContinuousAbsPeakResponse` | Absolute peak time and signed value there. |
| `ContinuousDominantResponse` | Classification at the fitted absolute peak; `NA` in pseudotime mode. |
| `ContinuousResponseOnsetTime`, `ContinuousResponseEndTime` | First/last regular-grid times passing the response threshold in the dominant response direction; undefined in pseudotime mode. |
| `ContinuousResponseDuration` | End minus onset; this spans intervening inactive intervals, if any. |
| `OnsetLeftCensored` | Whether the response was already active at the first grid time. |
| `ContinuousMaxSlope`, `ContinuousMinSlope` | Largest/smallest fitted gradient on the regular grid. |
| `ContinuousMaxSlopeTime`, `ContinuousMinSlopeTime` | Times of those gradients. |
| `MeanCIWidth` | Mean width of finite bootstrap intervals on the output grid. |

Grid points represent fitted curves; they do not add biological observations.

## 6. Continuous lag comparisons

Files: `<prefix>_continuous_lag_all_pairs.csv` and
`<prefix>_continuous_lag_top_pairs.csv`. Module/state/condition identifiers and
pattern annotations accompany these fields:

| Column | Meaning |
| --- | --- |
| `TrajectoryModelRNA`, `TrajectoryModelMET` | Models used for each trajectory. |
| `ContinuousSimilarity_Lag0`, `ContinuousRMSE_Lag0`, `ContinuousCCF_Lag0` | No-shift cosine similarity, error, and correlation diagnostic. |
| `BestContinuousLagTime` | Selected model-estimated shift in metadata time units, with the same positive-transcript-leading convention. |
| `BestContinuousSimilarity`, `BestContinuousRMSE` | Cosine similarity and error at that shift. |
| `BestContinuousCrossCorrelation` | Correlation diagnostic at the selected shift; the lag is selected using cosine similarity. |
| `ContinuousSimilarityGain` | Improvement in cosine similarity over lag zero. |
| `NAlignedGridPoints` | Number of aligned fitted points, not biological replicates. |
| `ContinuousRelationshipType` | Direction label for the selected lag. |
| `RNA_ContinuousPeakTime`, `MET_ContinuousPeakTime`, `ContinuousPeakLag` | Signed peak times and metabolite-minus-transcript peak displacement. |
| `RNA_ContinuousAbsPeakTime`, `MET_ContinuousAbsPeakTime`, `ContinuousAbsPeakLag` | Absolute peak times and their displacement. |
| `RNA_ResponseOnsetTime`, `MET_ResponseOnsetTime`, `ContinuousOnsetLag` | Response onset times and their displacement; undefined in pseudotime mode. |
| `RNA_OnsetLeftCensored`, `MET_OnsetLeftCensored` | Flags indicating response already active at the start. |
| `RNA_MeanCIWidth`, `MET_MeanCIWidth` | Mean fitted uncertainty widths. |
| `ModelEstimatedLag` | `True`, distinguishing this estimate from observed-time shifts. |

## 7. Optional Granger results

`--granger` writes `<prefix>_granger_status.txt` describing completion or why the
analysis was skipped. `<prefix>_granger_results.csv` is produced when the tests
reach the result-writing stage. Original measured times must meet the minimum
count (default 12) and spacing checks; interpolated points are excluded.

| Column | Meaning |
| --- | --- |
| `Direction` | `RNA_to_MET` or `MET_to_RNA`. |
| `LagOrder` | Autoregressive lag order tested. |
| `NRealTimepoints`, `NValuesTested` | Original time-point count and values retained after filtering/transformation. |
| `Transform` | `none` or `difference`. |
| `FStatistic`, `PValue` | Test statistic and unadjusted probability. |
| `DF_Denom`, `DF_Num` | Test degrees of freedom. |
| `AdjPValue_BH` | Benjamini–Hochberg correction across the returned tests. |
| `Interpretation` | `predictive_precedence_not_biochemical_causality`. |

For calculations and exact conditional behavior, see [trajectories](../mate/trajectories.py),
[continuous fitting](../mate/continuous.py), [lag comparisons](../mate/lag.py), and
[statistics](../mate/statistics.py).
