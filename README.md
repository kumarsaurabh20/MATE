# MATE

**MATE — Multi-omics Feature Analysis through Trajectory Embeddings**

A treatment-aware, reference-aware and lag-aware framework for integrating **time-resolved transcriptomics and metabolomics** through module-level temporal trajectories rather than relying only on pooled cross-omics correlation.

---

## Overview

Multi-omics time-series experiments often measure transcriptional and metabolic responses at the same nominal time points, but the two molecular layers do not necessarily respond at the same rate. Transcriptional regulation can precede metabolite accumulation, and the same pathway can show an early response under one treatment and a delayed response under another. If samples are pooled across conditions or compared only by contemporaneous correlation, these temporal relationships can be weakened, missed or misinterpreted.

MATE was developed to address this problem.

Instead of immediately correlating every transcript with every metabolite, MATE:

1. builds transcriptomic and metabolomic modules independently;
2. summarizes each module using an SVD-derived **metafingerprint**;
3. represents the response of each module across time;
4. compares treatments with their matched reference/mock samples when such a reference exists;
5. derives trajectory descriptors such as direction, peak timing, dynamic range and response timing;
6. embeds transcript and metabolite trajectories in a common low-dimensional space; and
7. evaluates synchronous and lagged transcript–metabolite relationships.

The central idea is that **temporal behavior is itself an informative multi-omics feature**.

MATE therefore asks:

> Do independently derived transcriptomic and metabolomic modules show compatible temporal responses, and does allowing a biologically plausible delay improve their agreement?

---

## Why MATE?

Many multi-omics integration workflows use correlation as the principal measure of association. Correlation is useful, but it can become problematic in time-resolved experiments.

Consider a biosynthetic pathway in which:

- pathway genes are induced at 12 h,
- enzyme activity increases afterward,
- the corresponding metabolite accumulates at 24–48 h.

A gene and metabolite may be biologically linked even if their values at identical time points are only weakly correlated.

A second problem arises when multiple treatments are pooled. For example:

- Treatment A may induce an early response,
- Treatment B may induce an intermediate response,
- Treatment C may induce a late response.

Pooling these measurements by time can flatten or obscure all three responses.

MATE preserves the **treatment × time** structure and explicitly evaluates temporal displacement between omics layers.

---

# Conceptual workflow

```text
Transcriptomics                         Metabolomics
      |                                      |
      v                                      v
Co-expression modules                 Co-abundance modules
      |                                      |
      v                                      v
SVD metafingerprints                  SVD metafingerprints
      |                                      |
      +------------------+-------------------+
                         |
                         v
          Treatment/reference response
                         |
                         v
                Temporal trajectories
                         |
        +----------------+----------------+
        |                                 |
        v                                 v
Trajectory descriptors              Continuous trajectory
Pattern / peak / AUC /              representation
dynamic range / timing              where appropriate
        |                                 |
        +----------------+----------------+
                         |
                         v
                Joint trajectory space
              PCA / UMAP / t-SNE
                         |
                         v
               Cross-omics comparison
                         |
                         v
          synchronous + lag-aware pairs
```

A key design principle is that the original transcript and metabolite modules are generated **independently**. Cross-omics integration takes place only after module-level temporal representations have been derived.

This reduces circularity: a transcript and metabolite are not placed together simply because they were correlated beforehand.

---

# Core concepts

## 1. Omics-specific modules

MATE operates on modules rather than only individual features.

For transcriptomics, a module may represent a set of co-expressed genes.

For metabolomics, a module may represent a set of co-abundant LC–MS features or annotated metabolites.

The modules can originate from MEANtools-compatible SQLite databases or another upstream network/module workflow, provided that module membership can be mapped back to the quantitative feature matrices.

Keeping transcript and metabolite modules separate at this stage is important. MATE does not use cross-omics similarity to construct the modules.

---

## 2. SVD-based metafingerprints

For each module, MATE extracts the abundance/expression matrix of its members and summarizes the dominant sample-level behavior using singular value decomposition (SVD).

Conceptually, for a module matrix

```text
features × samples
```

MATE obtains the first dominant sample-space singular vector and uses it as a module-level **metafingerprint**.

The metafingerprint acts as a compact representation of the coordinated behavior of the module across samples.

### Why SVD?

SVD provides several useful properties:

- reduces a multi-feature module to one representative profile;
- emphasizes shared variation among module members;
- reduces the influence of individual noisy features;
- makes transcriptomic and metabolomic modules comparable at the module level;
- supports downstream trajectory descriptors and embeddings.

Because singular vectors have arbitrary sign, MATE orients vectors consistently so that equivalent module profiles are not represented with arbitrary opposite directions.

---

## 3. Reference-aware treatment response

For experiments containing matched controls, MATE represents each treatment relative to the appropriate reference at each time point.

For treatment `c` at time `t`:

```text
Response(c,t) = Treatment(c,t) relative to Reference(t)
```

The current treatment-aware workflow supports an effect-size representation in which treatment and reference replicates are contrasted at the same time point.

Example:

```text
Time             12 h          24 h          48 h
---------------------------------------------------
Mock             M12           M24           M48
C. fulvum        C12           C24           C48

Response          C12-M12       C24-M24       C48-M48
```

This prevents normal time-dependent changes in the mock/control from being mistaken for treatment-induced biology.

### Important

A trajectory value and a temporal transition are not the same thing.

Suppose the treatment response is:

```text
12 h     +2.0
24 h     +1.0
48 h     +0.8
```

The module is still positively induced relative to the control at all three time points, but its temporal transitions are:

```text
12 -> 24 : DOWN
24 -> 48 : DOWN
```

Therefore a MATE pattern such as `DOWN-DOWN` means that the **response is decreasing through time**. It does **not** automatically mean that the module is down-regulated relative to the mock.

For this reason MATE distinguishes:

- **response state** at each time point, and
- **trajectory direction** between adjacent time points.

---

## 4. Response-state pattern

With an effect-size threshold, each time point can be classified as:

```text
UP
FLAT
DOWN
```

For example:

```text
Response values:       -0.1     +0.8     +1.4
ResponseStatePattern:  FLAT     UP       UP
```

The response-state pattern answers:

> Is this module induced, unchanged or suppressed relative to its reference at each measured time?

---

## 5. Temporal trajectory pattern

For ordered response values `R1, R2, ..., Rn`, MATE calculates adjacent changes:

```text
Delta12 = R2 - R1
Delta23 = R3 - R2
...
```

Each transition is then classified as:

```text
UP
FLAT
DOWN
```

using a user-defined flat/effect-size threshold.

For three time points:

```text
UP-UP
UP-DOWN
DOWN-UP
DOWN-DOWN
UP-FLAT
FLAT-UP
...
```

For five time points, four transition labels are produced.

The temporal pattern therefore captures **shape**, whereas the response-state pattern captures **direction relative to the reference**.

---

# Trajectory descriptors

MATE can calculate a collection of module-level temporal descriptors. Depending on the script/version and selected trajectory mode, these can include:

| Descriptor | Interpretation |
|---|---|
| `Pattern` | Direction of change between adjacent time points |
| `ResponseStatePattern` | UP/FLAT/DOWN relative to matched reference at each time |
| `Peak` | Time point with the largest positive trajectory value |
| `PeakIndex` | Ordinal index of the peak |
| `AbsPeak` | Time point with the largest absolute response |
| `AbsPeakIndex` | Ordinal index of the absolute peak |
| `Delta_T1_to_T2` | Change between adjacent trajectory states |
| `Delta_first_last` | Final minus initial trajectory value |
| `Max` / `Min` | Maximum and minimum trajectory values |
| `DynamicRange` | Max − Min |
| `MaxAbsDelta` | Largest adjacent change |
| `Variance` | Variability of the trajectory |
| `AUC` | Area under the trajectory |
| `MaximumFC` | Maximum fold-change where mathematically meaningful |
| `ResponseTiming` | Qualitative early / intermediate / late response category |
| coherence metrics | Agreement of module members with the module trajectory |

Not every descriptor should be treated as independent biological evidence. The strongest interpretation usually comes from combining several descriptors.

---

# Continuous and pseudo-time trajectories

MATE v2.3 extends the discrete-time framework toward continuous or pseudo-time representations.

The purpose of interpolation or trajectory modelling is **not to manufacture new biological observations**. Instead, it provides a common continuous representation that can improve:

- comparison of trajectory shape;
- estimation of temporal displacement;
- alignment of measurements collected at unequal time intervals;
- visualization of delayed transcript–metabolite responses.

## Three measured time points

With only three measured time points, complex nonlinear models are weakly constrained.

For datasets such as:

```text
12 h, 24 h, 48 h
```

MATE should treat intermediate values primarily as **interpolation**, not as independently observed biology.

Scientifically conservative options include piecewise-linear or shape-preserving interpolation.

For example:

```text
Measured:       12     24     48
Interpolated:   12 18 24 30 36 42 48
```

The interpolated points provide a denser common grid for trajectory comparison, but they do not increase the experimental degrees of freedom.

With three measured time points, avoid interpreting:

- fine-scale inflection points;
- oscillations;
- complex spline shapes;
- Granger-style temporal causality;
- precise delay estimates smaller than the experimental sampling resolution.

## Five or more measured time points

With approximately five or more well-spaced measured time points, more flexible trajectory models become scientifically more defensible.

Depending on the experimental design, possible approaches include:

- smoothing splines;
- generalized additive models (GAMs);
- other regularized smoothers.

The model should still be constrained to preserve the experimental signal and avoid overfitting.

The purpose is to obtain an interpretable estimate of the underlying trajectory, not merely the curve that maximizes fit to the observed samples.

## Developmental pseudo-time

Pseudo-time experiments require a different interpretation from treatment/mock experiments.

A developmental pseudo-time trajectory may not contain a matched reference condition at every position. In that situation:

- do not manufacture a mock/reference;
- do not interpret unfertilized or baseline samples automatically as matched controls;
- interpret trajectory similarity as temporal co-variation;
- do not infer causal transcription-to-metabolite lags solely from pseudo-time displacement.

Pseudo-time mode is therefore useful for **trajectory alignment and co-dynamics**, but its biological interpretation differs from the treatment-aware lag framework.

---

# Joint transcriptomics–metabolomics embedding

After trajectory features are calculated independently for transcript and metabolite modules, MATE can place them in a shared low-dimensional representation.

Supported or associated embedding approaches include:

- PCA
- UMAP
- t-SNE

The joint embedding can reveal modules with similar temporal properties even when they originate from different omics layers.

A typical plot may encode:

```text
shape  -> omics layer
color  -> trajectory class / response timing
point  -> module × treatment
```

For example:

```text
circle    = transcript module
triangle  = metabolite module

early     = early response
middle    = intermediate response
late      = delayed response
```

Proximity in the embedding is **not proof of biochemical interaction**. It indicates similarity in the trajectory-derived feature space and should be interpreted with biochemical, genetic and pathway evidence.

---

# Lag-aware cross-omics analysis

A major purpose of MATE is to detect cases where a transcriptomic response and a metabolic response are similar but temporally displaced.

For each candidate transcript-module × metabolite-module pair within the same biological condition, MATE can compare:

1. similarity with no temporal shift;
2. similarity after one or more allowed shifts;
3. change in similarity after shifting;
4. residual error;
5. module coherence.

Typical output columns include:

| Column | Meaning |
|---|---|
| `Lag0Similarity` | Similarity without temporal displacement |
| `BestLagSteps` | Shift producing the best permitted alignment |
| `BestLagTimeMedian` | Approximate lag expressed in experimental time units |
| `BestLagSimilarity` | Similarity at the best lag |
| `BestLagRMSE` | Residual error after the best lag alignment |
| `SimilarityGain` | Improvement over lag-0 similarity |
| `RelationshipType` | Synchronous / Transcriptomics_leads / Metabolomics_leads |

In the current MATE convention:

```text
BestLagSteps > 0  -> transcriptomics leads
BestLagSteps = 0  -> synchronous
BestLagSteps < 0  -> metabolomics leads
```

`RelationshipType` should always be considered together with the signed lag.

### Example

```text
Condition:                  C. fulvum
Transcript module:          RNA::203
Metabolite module:          MET::17

RNA response timing:        EARLY
Metabolite response timing: LATE

Lag0Similarity:             0.20
BestLagSteps:               +1
BestLagSimilarity:          0.91
SimilarityGain:             +0.71

RelationshipType:           Transcriptomics_leads
```

This is stronger evidence of delayed temporal coupling than the lag-0 similarity alone.

However, it still does not demonstrate biochemical causality.

---

# How to interpret a strong MATE relationship

No single MATE metric should be used in isolation.

A useful evidence hierarchy is:

1. **Treatment-specific response** — is the module genuinely altered relative to its matched reference?
2. **Trajectory compatibility** — do the two modules show biologically compatible temporal shapes?
3. **Response timing** — does the transcript response occur before or alongside the metabolite response?
4. **Lag improvement** — does a plausible temporal shift substantially improve similarity?
5. **Low alignment error** — is the best-lag RMSE acceptably low?
6. **Module coherence** — is the module trajectory supported by its members rather than driven by one feature?
7. **Biological plausibility** — do annotation, chemistry, genetics or pathway knowledge support the association?

MATE is designed as a **prioritization framework**. It helps identify temporally plausible cross-omics relationships that can then be tested using pathway knowledge and experimental evidence.

---

# Input data

## 1. Transcriptomics quantitative table

Recommended structure:

```text
feature,S01,S02,S03,S04,S05,...
Solyc01g000010,12.1,11.8,13.4,18.5,17.9,...
Solyc01g000020,4.2,4.8,4.1,8.6,8.1,...
...
```

Requirements:

- one row per feature/gene;
- one column per sample;
- feature IDs must match module member IDs in the transcriptomics module database;
- quantitative values must be numeric;
- sample names must match the metadata.

The matrix should already have undergone the normalization appropriate to the upstream transcriptomics analysis.

MATE is not intended to replace RNA-seq count normalization or differential-expression analysis.

---

## 2. Metabolomics quantitative table

Recommended structure:

```text
feature,S01,S02,S03,S04,S05,...
M243T2033,10231,10980,9830,32611,34120,...
M279T2328,2811,3022,2901,7114,7422,...
...
```

Requirements are analogous to the transcriptomics table.

Feature IDs may be:

- annotated metabolite names;
- LC–MS feature IDs;
- m/z-retention-time identifiers;
- other unique identifiers,

provided that they are consistent between the quantitative table and the metabolite module definitions.

MATE does not perform chromatographic peak detection, adduct grouping or metabolite annotation.

---

## 3. Metadata

A minimal metadata table should contain:

```text
sample,condition,time,replicate
CF_12_R1,C_fulvum,12,1
CF_12_R2,C_fulvum,12,2
CF_24_R1,C_fulvum,24,1
CF_24_R2,C_fulvum,24,2
Mock_12_R1,Mock,12,1
Mock_12_R2,Mock,12,2
...
```

Recommended fields:

| Column | Description |
|---|---|
| `sample` | Sample ID matching the quantitative matrices |
| `condition` | Treatment / biological condition |
| `time` | Ordered time value |
| `replicate` | Biological replicate |
| reference indicator | Optional explicit control/reference mapping where needed |

The same experimental sample does not have to exist in both omics matrices if the implementation supports modality-specific sample matching, but the time and condition structure must be comparable.

---

## 4. Module databases

MATE can use SQLite databases containing transcriptomic and metabolomic module definitions.

The original MEANtools-derived database structure uses cluster tables with module membership stored in fields such as:

```text
Cluster
Members
```

and decay-rate-specific tables such as:

```text
DR_25
```

The decay rate can be selected at run time.

Module membership strings should resolve to feature IDs present in the corresponding quantitative table.

---

# Recommended command-line interface

The current MATE refactor uses or is moving toward explicit omics-specific arguments such as:

```bash
python MATE_v2_3.py   --transcriptomics_table fungal_TR_expression.csv   --metabolomics_table fungal_MS_abundance.csv   --transcriptomics_db tomato_fungal_coexp.sqlite   --metabolomics_db tomato_fungal_coabun.sqlite   --metadata new_fungal_pheno.txt   --n_timepoints 3   --time_values "12,24,48"   --embedding_method both   --stratify_by condition   --trajectory_metric effect_size   --effect_size_threshold 0.5   -dr 25   -o MATE
```

The exact option names should be checked against:

```bash
python MATE_v2_3.py --help
```

because older development scripts retained legacy MEANtools flags such as `-ft`, `-qm`, `-dn`, `-g` and `-m`.

---

# Important options

The following options summarize the principal controls used across the current MATE development series.

## Core inputs

```text
--transcriptomics_table
--metabolomics_table
--transcriptomics_db
--metabolomics_db
--metadata
```

## Time structure

```text
--n_timepoints
--time_values
--time_column
```

Example:

```bash
--n_timepoints 3 --time_values "12,24,48"
```

## Treatment-aware trajectories

```text
--stratify_by condition
--trajectory_metric effect_size
--effect_size_threshold 0.5
```

## Module/network selection

```text
-dr 25
```

## Embedding

```text
--embedding_method none
--embedding_method umap
--embedding_method tsne
--embedding_method both
```

Common embedding controls in development versions include:

```text
--embedding_metric
--tsne_perplexity
--umap_neighbors
--umap_min_dist
--embedding_color_by
```

## Output

```text
-o MATE
```

sets the output prefix.

---

# Example: tomato pathogen-response dataset

A representative MATE analysis used transcriptomics and metabolomics from tomato leaves exposed to microbial elicitors at:

```text
12 h
24 h
48 h
```

The analysis contained conditions including:

```text
Cladosporium fulvum
Malassezia restricta
chitin
mock/reference samples
```

A representative command is:

```bash
python MATE_v2_3.py   --transcriptomics_table fungal_TR_expression.csv   --metabolomics_table fungal_MS_abundance.csv   --transcriptomics_db tomato_fungal_coexp.sqlite   --metabolomics_db tomato_fungal_coabun.sqlite   --metadata new_fungal_pheno.txt   --n_timepoints 3   --time_values "12,24,48"   --embedding_method both   --stratify_by condition   --trajectory_metric effect_size   --effect_size_threshold 0.5   -dr 25   -o MATE
```

In the falcarindiol use case, pathway-associated transcript and metabolite behavior provided an interpretable biological example of the central MATE idea: transcriptional responses can precede later metabolic accumulation, and trajectory-aware integration can recover this relationship in a way that same-time-point correlation may not fully represent.

Transcriptomic and metabolomic features/modules associated with the falcarindiol response were observed in compatible regions of the joint trajectory embedding, providing a useful test case for the framework.

This should be interpreted as a validation/case study of trajectory behavior rather than as proof that proximity in UMAP establishes a direct enzymatic relationship.

---

# Output files

Output names vary slightly between development versions. Typical MATE outputs include the following classes of files.

## Module fingerprints

```text
*_global_fingerprints_all_samples.csv
*_transcriptomics_fingerprints.csv
*_metabolomics_fingerprints.csv
```

These contain module-level SVD representations and member statistics.

Typical columns:

```text
ID
Members
n_features_found
n_features_missing
<sample-level fingerprint values>
```

---

## Trajectory descriptors

```text
*_trajectory_descriptors.csv
*_transcriptomics_trajectory_descriptors.csv
*_metabolomics_trajectory_descriptors.csv
```

Typical fields include:

```text
ID
condition
T1_<time>
T2_<time>
T3_<time>
Pattern
ResponseStatePattern
Peak
AbsPeak
DynamicRange
Variance
AUC
ResponseTiming
```

---

## Joint embeddings

Typical outputs include coordinate tables and image files for:

```text
PCA
UMAP
t-SNE
```

Depending on configuration, plots may be colored by:

```text
Pattern
ResponseStatePattern
Peak
AbsPeak
ResponseTiming
condition
modality
```

A combined embedding is generally more informative for MATE than separate transcript-only and metabolite-only plots because the goal is to examine whether the two independently derived omics representations occupy compatible temporal space.

---

## Lag-aware cross-omics pairs

A principal output is typically:

```text
MATE_cross_omics_lag_top_pairs.csv
```

or an equivalent prefix-specific filename.

Typical columns include:

```text
Condition
TranscriptModule
MetaboliteModule
Lag0Similarity
BestLagSteps
BestLagTimeMedian
BestLagSimilarity
BestLagRMSE
SimilarityGain
RelationshipType
```

Additional trajectory and coherence information may also be included.

---

# Interpreting lag output carefully

A large best-lag similarity is not enough by itself.

For example:

```text
Lag0Similarity     = 0.87
BestLagSimilarity  = 0.89
SimilarityGain     = 0.02
```

provides little evidence that the lag itself is important.

In contrast:

```text
Lag0Similarity     = 0.20
BestLagSimilarity  = 0.91
SimilarityGain     = 0.71
```

suggests that temporal displacement is critical to the relationship.

Likewise, a large similarity gain should be considered together with:

- biological sampling resolution;
- number of measured time points;
- RMSE;
- member coherence;
- response magnitude;
- pathway plausibility.

With only three measured time points, a one-step lag can be biologically meaningful, but it is still a coarse temporal statement.

---

# Sampling resolution and lag resolution

MATE cannot resolve biological timing more precisely than supported by the experiment.

For example:

```text
12 h -> 24 h = 12 h
24 h -> 48 h = 24 h
```

A one-step shift is not associated with one constant duration in an unevenly spaced experiment.

For this reason MATE reports or can derive a quantity such as:

```text
BestLagTimeMedian
```

in addition to `BestLagSteps`.

Continuous interpolation can help compare shape on a common temporal grid, but it does not transform a three-time-point experiment into a high-frequency time series.

---

# Missing features

Module databases may contain members that are absent from the quantitative matrix.

MATE reports missing features rather than silently assuming that every module member is present.

Typical output fields include:

```text
n_features_found
n_features_missing
```

and development versions can generate dedicated missing-feature reports.

Modules with no usable quantitative members are skipped.

---

# Normalization

MATE can apply row-wise standardization before SVD in development versions.

The rationale is to prevent high-abundance features from dominating module-level fingerprints purely because of scale.

However, upstream normalization remains essential.

Examples:

### Transcriptomics

Use an appropriate normalized expression representation produced by the RNA-seq workflow.

### Metabolomics

Use a quantitative matrix that has undergone the QC, normalization and missing-value handling appropriate to the LC–MS experiment.

MATE should not be used as a replacement for modality-specific preprocessing.

---

# Experimental design recommendations

MATE performs best when the experimental design contains:

- at least three biologically meaningful time points;
- biological replication;
- matched treatment/reference samples where treatment-response inference is desired;
- comparable time coverage for transcriptomics and metabolomics;
- modules containing enough measured members to estimate a stable fingerprint.

### Three time points

Good for:

- early / intermediate / late response;
- coarse trajectory shape;
- one-step lag hypotheses;
- treatment-aware trajectory embedding.

Use caution with:

- nonlinear curve fitting;
- sub-interval lag estimates;
- causal inference.

### Five or more time points

Better suited to:

- continuous trajectory modelling;
- smoother lag estimation;
- nonlinear response patterns;
- peak/inflexion timing;
- pseudo-time or developmental trajectory analysis.

More time points do not compensate for poor replication or batch-confounded design.

---

# What MATE does not claim

MATE intentionally separates temporal association from causal or biochemical proof.

MATE does **not** by itself establish:

- direct enzyme–substrate relationships;
- reaction direction;
- physical interaction;
- transcriptional causality;
- metabolite identity;
- pathway membership;
- regulatory causation.

A strong MATE pair is a **prioritized hypothesis**.

The hypothesis becomes much stronger when supported by orthogonal evidence such as:

- enzyme annotation;
- genomic clustering;
- knockout/CRISPR phenotypes;
- isotope tracing;
- MS/MS;
- authentic standards;
- spatial co-localization;
- biochemical assays;
- pathway databases.

---

# MATE versus conventional cross-omics correlation

| Conventional correlation | MATE |
|---|---|
| compares feature abundance directly | compares module-level temporal representations |
| often evaluates same-time measurements | explicitly evaluates lagged relationships |
| can pool treatments | preserves treatment-specific responses |
| sensitive to delayed metabolite accumulation | designed to detect temporal displacement |
| feature-centric | module/trajectory-centric |
| association often represented by one coefficient | integrates trajectory shape, timing, lag, error and coherence |
| correlation is the primary integration step | cross-omics integration occurs after independent module summarization |

MATE can still use similarity or correlation-like metrics during trajectory comparison. The distinction is that these are applied **after temporal structure has been represented**, rather than being the starting point for module discovery.

---

# Recommended analysis strategy

For a new dataset:

```text
1. Perform modality-specific QC and normalization.
2. Generate transcript co-expression modules.
3. Generate metabolite co-abundance modules.
4. Prepare metadata with sample, condition and time.
5. Run MATE in treatment-aware mode when matched references exist.
6. Inspect response-state and temporal trajectory patterns.
7. Examine the combined PCA/UMAP.
8. Inspect synchronous and lag-aware transcript–metabolite pairs.
9. Rank relationships using multiple MATE metrics.
10. Add biological evidence: annotation, pathways, MS/MS, genetics, etc.
```

Do not begin by selecting only transcript–metabolite pairs that already show strong direct correlation if the goal is to discover delayed relationships. That would remove one of the main advantages of MATE.

---

# Troubleshooting

## Sample names do not match

**Symptom**

```text
metadata samples not present in feature table
```

**Check**

- whitespace in sample names;
- different separators;
- renamed replicates;
- transcriptomics/metabolomics sample suffixes;
- capitalization.

---

## Modules contain many missing members

Check that the feature identifiers in the SQLite module database use the same identifier system as the quantitative table.

For genes, common problems include:

```text
gene vs transcript identifiers
versioned vs unversioned IDs
old vs new genome annotations
```

For metabolomics, confirm that feature IDs have not changed during peak-table filtering.

---

## UMAP fails or looks unstable

With a small number of modules, reduce:

```text
--umap_neighbors
```

UMAP is stochastic. Use trajectory tables and quantitative lag metrics for interpretation rather than treating exact visual distances as fixed biological quantities.

---

## t-SNE perplexity error

Perplexity must be compatible with the number of observations.

Reduce:

```text
--tsne_perplexity
```

for small module sets.

---

## A biologically expected module has an unexpected `Pattern`

Remember that `Pattern` describes the direction of change **between response values**.

Inspect:

```text
ResponseStatePattern
time-specific response values
Delta_T1_to_T2
Delta_T2_to_T3
```

before interpreting the label biologically.

---

## `Metabolomics_leads` appears inconsistent with the plotted pair

Always verify that:

- the plotted pair corresponds to the same row of the lag table;
- the same condition is shown;
- the same standardized trajectory representation was used;
- the sign convention matches the version of MATE being run.

In the current convention:

```text
positive lag -> transcriptomics leads
negative lag -> metabolomics leads
zero         -> synchronous
```

---

# Reproducibility

For a reproducible MATE analysis, record:

```text
MATE version
Python version
input matrix versions
module database versions
metadata file
decay rate
time points
reference/control definition
trajectory metric
effect-size threshold
embedding parameters
lag search range
random seeds, where applicable
```

Version-controlling the command used for each run is strongly recommended.

---

# Software environment

MATE is implemented in Python.

Development versions use packages including:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
tqdm
umap-learn
```

and MEANtools-compatible helper functions such as `gizmos.py` for importing cluster information from SQLite.

A typical environment can be created with:

```bash
conda create -n mate python=3.11
conda activate mate

pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm umap-learn
```

If the analysis uses MEANtools-derived databases, ensure that the required local MEANtools helper modules are available on `PYTHONPATH` or in the working directory.

---

# Suggested repository structure

```text
MATE/
├── MATE_v2_3.py
├── gizmos.py
├── README.md
├── requirements.txt
├── examples/
│   └── falcarindiol/
│       ├── metadata_example.csv
│       ├── transcriptomics_example.csv
│       └── metabolomics_example.csv
├── docs/
│   ├── workflow.png
│   └── output_columns.md
└── tests/
```

For public release, it is also useful to provide:

```text
LICENSE
CITATION.cff
environment.yml
example command
small test dataset
expected test outputs
```

---

# Relationship to MEANtools

MATE emerged from work on module/metafingerprint analysis in the MEANtools ecosystem but addresses a distinct question.

MEANtools focuses on integration strategies for biosynthetic pathway discovery, including gene–metabolite relationships and pathway hypotheses.

MATE focuses specifically on **time-resolved multi-omics behavior**:

```text
MEANtools-derived modules
        ↓
module metafingerprints
        ↓
temporal trajectories
        ↓
joint embedding
        ↓
lag-aware cross-omics relationships
```

MATE can therefore complement reaction- or correlation-based pathway discovery rather than replacing it.

---

# Biological interpretation: the falcarindiol example

The tomato falcarindiol pathway illustrates why timing matters.

Pathway-associated genes can respond early to pathogen elicitation, while falcarindiol and related metabolites accumulate later. A same-time-point comparison may therefore understate relationships that become clearer after allowing temporal displacement.

The MATE workflow was applied to the falcarindiol dataset as a biological test case. Transcript and metabolite modules with compatible temporal behavior occupied related regions of the shared trajectory embedding, and lag-aware comparisons allowed early transcriptional and later metabolic responses to be evaluated explicitly.

The example illustrates the intended use of MATE:

> identify temporally plausible cross-omics relationships first, then evaluate those relationships using biochemical and genetic evidence.

---

# Current development direction

MATE development is moving from discrete trajectories toward a unified framework supporting both:

### Discrete experimental time series

Examples:

```text
12 h
24 h
48 h
```

with treatment/reference contrasts and coarse lag inference.

### Denser continuous time series

Examples:

```text
0
1
2
4
8
12
24 h
```

with smoother trajectory modelling and more informative temporal alignment.

### Pseudo-time trajectories

Examples:

```text
developmental progression
single-cell pseudo-time
single-organism pseudo-time
```

where trajectory alignment is meaningful but treatment/reference and causal-lag assumptions must be adapted to the biological design.

The long-term goal is to preserve one principle across all modes:

> MATE should represent the experimentally supported trajectory before attempting cross-omics integration.

---

# Citation

MATE is currently under development.

If you use the software before a formal MATE publication is available, please cite the repository/version used and the relevant MEANtools publication or associated project documentation where appropriate.

A formal citation can be added here once available.

---

# Contact

**Integrative Bioinformatics Lab**  
Brightlands Future Farming Institute  
Maastricht University

For questions, bug reports or feature requests, please use the repository issue tracker once the public repository is available.
