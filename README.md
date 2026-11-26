# Athena

**If the Rule Explains the Demonstrations, Why Doesn't It Solve ARC-AGI?**

## Repository map

```text
TinyRecursiveModels/
  config/                        Training and architecture configurations
  dataset/                       ARC dataset preparation
  evaluators/                    ARC evaluation and prediction aggregation
  models/                        TRM, evidence encoding and loss functions
    recursive_reasoning/         Relational maps, core priors and rule execution
  scripts/                       Candidate generation, verification and diagnostics
  tests/                         Task atlas and taxonomy tests
  utils/                         Shared utilities
  datasets_agi.zip               Bundled dataset archive
  pretrain.py                    Model training entry point
  puzzle_dataset.py              Processed puzzle data loader
  requirements.txt               Python dependencies
README.md                        Setup and folder-by-folder run guide
```

Athena separates rule abstraction, transformation verification and neural rule execution. The paper uses Qwen3-VL-8B-Instruct to extract demonstration-side transformations and a fine-tuned Tiny Recursive Model (TRM) to generate output grids conditioned on verified evidence.

| Component | Model / implementation |
|---|---|
| Transformation Extraction Module (TEM) | Qwen3-VL-8B-Instruct |
| Spatial grounding | Relational maps, core priors and transition evidence |
| Rule execution | Tiny Recursive Model (TRM) |
| Matched baseline | TRM, K = 1 |
| High-budget baseline | TRM (Voted), K = 1000 |
| Athena | Matched K = 1; full K = 64 |

The checkout contains training and diagnostic implementations, but does not contain an identifiable standalone TEM launcher, final paper checkpoints, complete paper-run configurations or all experiment ledgers. Commands below distinguish available entry points from steps requiring these additional artifacts. 

## 1. Prepare the Environment

Run commands from the repository root unless otherwise specified. For an anonymous checkout, substitute the reviewer-facing repository URL.

```bash
export ANONYMOUS_REPOSITORY_URL=https://github.com/OWNER/REPOSITORY.git
git clone "$ANONYMOUS_REPOSITORY_URL" athena
cd athena/TinyRecursiveModels
python3 --version
nvidia-smi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
```

Use Python 3.10 or a compatible version and a CUDA-enabled PyTorch installation. The requirements file is not a pinned environment lockfile. The paper reports four NVIDIA A100 80 GB GPUs and mixed-precision inference. Do not install packages globally on a shared server.

For optional Weights & Biases authentication:

```bash
wandb login
```

The TEM model is [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). The released reference TRM checkpoints are available at [arcprize/trm_arc_prize_verification](https://huggingface.co/arcprize/trm_arc_prize_verification). These are reference checkpoints, not the final Athena checkpoints. The dependencies and launch configuration for the paper's TEM run must be supplied separately.

## 2. Install the Dataset Locally

The experiments use [ARC-AGI-1](https://arcprize.org/arc-agi/1), [ARC-AGI-2](https://arcprize.org/arc-agi/2) and ConceptARC. Inspect the bundled archive before extracting it:

```bash
cd /path/to/athena/TinyRecursiveModels
unzip -l datasets_agi.zip
mkdir -p kaggle/combined
```

Arrange the ARC challenge and solution JSON files in the combined-input format expected by `dataset/build_arc_dataset.py`, using the prefix `kaggle/combined/arc-agi`. The archive's internal layout and contents have not been verified; do not assume it contains every paper evaluation split.

### ARC-AGI-1

```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

### ARC-AGI-2

```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

These commands follow the repository's dataset-preparation examples. They do not independently establish the paper's exact split membership. Preserve the following paper protocol when preparing the final experiment data:

| Dataset | Training protocol | Evaluation |
|---|---|---|
| ARC-AGI-1 | 400 training tasks, support demonstrations from 400 evaluation tasks, 160 ConceptARC tasks | 400 evaluation tasks |
| ARC-AGI-2 | 1,000 training tasks and support demonstrations from the public evaluation split | 240 tasks from disjoint semi-private and private splits |

Evaluation query targets must be excluded from optimization and used only for scoring. The ARC-AGI-2 semi-private/private evaluation requires authorized access or an appropriate evaluation service. Do not substitute public tasks and report them as the 240-task evaluation. Do not combine both datasets into an uncontrolled pool; check overlap and preserve checkpoint puzzle-identifier mappings.

## 3. Generate Baseline Answers

The paper compares symbolic, vision-language and recursive neural solvers. This repository provides TRM-related entry points; separate implementations and experiment records are required for the other methods.

### TRM baseline example

Run from `TinyRecursiveModels/`. Supply the local reference configuration, compatible checkpoint, augmented dataset and canonical dataset. The following uses the checked-in candidate-generation script:

```bash
export TRM_CONFIG=/path/to/trm_config.yaml
export TRM_CHECKPOINT=/path/to/trm_checkpoint
export AUG_DATA=data/arc1concept-aug-1000
export CANONICAL_DATA=/path/to/canonical_evaluation_dataset

python scripts/fvr_original_trm_exact_candidate_k.py \
  --config "$TRM_CONFIG" \
  --checkpoint "$TRM_CHECKPOINT" \
  --aug-data-path "$AUG_DATA" \
  --canonical-data "$CANONICAL_DATA" \
  --out-dir runs/trm_matched_k1 \
  --k 1
```

For a high-budget candidate-generation run, use a separate output directory:

```bash
python scripts/fvr_original_trm_exact_candidate_k.py \
  --config "$TRM_CONFIG" \
  --checkpoint "$TRM_CHECKPOINT" \
  --aug-data-path "$AUG_DATA" \
  --canonical-data "$CANONICAL_DATA" \
  --out-dir runs/trm_candidates_k1000 \
  --k 1000
```

The second command generates candidates at the requested budget; reproducing the paper's TRM (Voted) result additionally requires the exact voting/aggregation protocol and checkpoint. Do not equate candidate generation with verified reproduction of the voted baseline.

### Other baseline methods

| Family | Methods in the paper |
|---|---|
| Symbolic and program synthesis | DreamCoder, Icecuber, ARCLE, ARGA |
| Visual and language-based reasoning | Visual ARC, V-ARC, LLM4ARC, GPT-4 ARC, Claude ARC |
| Recursive neural reasoning | HRM, TRM (Matched), TRM (Voted) |



## 4. Metric Settings

Use these definitions and preserve each metric's denominator:

- **TS (Task Solves):** Number of tasks for which every query has at least one exact match among the two permitted attempts. Output dimensions and all cell values must match.
- **ETA (Exact Task Accuracy):** `100 × TS / number of evaluated tasks`. The paper denominators are 400 for ARC-AGI-1 and 240 for ARC-AGI-2.
- **K (Candidate budget):** Number of generated predictions per query before deduplication. It is not the number of submitted attempts or necessarily the number of votes.
- **L (Latency):** Average end-to-end inference latency, including spatial grounding, uncached TEM inference, canonicalization, verification, compilation, candidate generation, ranking and selection.
- **M (Peak GPU memory):** Maximum allocated memory on any one GPU, not total memory across all GPUs.
- **Transformation evidence:** Count unique canonical events before verification and retained events after verification. Each demonstration contributes at most once to an event's support.

Candidate ranking uses consistency-weighted agreement with verified transformations. Ties prefer fewer unsupported edits and then higher TRM confidence; when no verified event survives, use TRM confidence alone. Only the two highest-ranked distinct candidates are submitted. Query targets are not available during selection.

## 5. Train and Evaluate Athena

Athena requires spatially grounded demonstrations, TEM-generated transformation descriptions, canonicalized and verified transformation events, compiled evidence and a compatible TRM executor.

| Setting | Paper configuration |
|---|---|
| TEM | Qwen3-VL-8B-Instruct |
| Verification threshold (gamma) | 0.75 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Weight decay | 0.01 |
| Fine-tuning epochs | 500 |
| Candidate budget | 64 (full), 1 (matched) |
| Hardware | 4 × NVIDIA A100 80 GB |

The transformation extraction stage operates only on demonstration pairs. Canonicalization normalizes operations, source objects, conditions, parameters, targets and order. For an event supported by `s` out of `M` demonstrations, retain it if `s/M >= 0.75`. Compile retained events into structured features and fuse them with query-side relational, core-prior and transition evidence; do not concatenate raw TEM text with grid tokens.

### Training entry points

Inspect the available configurations before starting a run:

```bash
cd /path/to/athena/TinyRecursiveModels
python pretrain.py --config-name cfg_v3_fullsystem_test --cfg job
python scripts/run_stage1_local.py --help
```

`cfg_v3_fullsystem_test.yaml` is a short three-epoch engineering test, not the paper configuration. `new_fvr_full_v2_base.yaml` and `run_stage1_local.py` provide additional local experimental controls. Some configurations contain original machine-specific paths; replace dataset, checkpoint and output paths with local paths before training.

For the final four-GPU run, first supply the actual resolved Athena paper configuration under `config/athena_paper.yaml`, compatible initialization checkpoint, dataset paths and compiled evidence. The following is the intended training entry-point invocation, conditional on those missing inputs:

```bash
torchrun --nproc-per-node=4 pretrain.py --config-name athena_paper
```

The checked-in files do not establish a complete TEM-to-TRM execution command or a final paper-specific Hydra configuration. The table above records paper settings rather than claiming that the short engineering configurations reproduce them.

### Evaluation and candidate diagnostics

```bash
python scripts/eval_arc_agi1.py --help
python scripts/verify_and_select_candidates.py --help
python scripts/fvr_targeted_vote_k.py --help
```

These scripts expose available ARC-AGI-1 evaluation, offline verification probes and candidate-budget diagnostics. The offline harness is not independently verified to implement the paper's full TEM-based ranking pipeline.

Reported Athena results (paper Table 1):

| Method | K | ARC-AGI-1 TS / ETA | ARC-AGI-2 TS / ETA |
|---|---:|---:|---:|
| TRM (Matched) | 1 | 118 / 29.5% | 12 / 5.0% |
| Athena (Matched) | 1 | 164 / 41.0% | 24 / 10.0% |
| Athena (Full) | 64 | 201 / 50.2% | 48 / 20.0% |

These are manuscript-reported results, not fresh runs of this checkout. The TRM (Voted) ARC-AGI-2 result is inconsistent between the manuscript abstract and Table 1 and is omitted pending reconciliation with the original evaluation ledger.

## 6. Reproduce Figures 1–4

### Figures 1 and 2

Figure 1 illustrates task-dependent visual semantics. Figure 2 summarizes cross-demonstration consistency of canonical transformation evidence. Reproduction requires the original demonstration examples and per-task TEM/canonical-event records. The repository includes an ARC task atlas for inspecting processed tasks:

```bash
cd /path/to/athena/TinyRecursiveModels
python scripts/build_arc_task_atlas.py --help
```

The atlas is a diagnostic resource; it is not a dedicated generator for either paper figure. For Figure 2, deduplicate events within each demonstration, count their support across demonstrations, normalize by the number of demonstrations and aggregate the consistency scores using the original plotting protocol.

### Figures 3 and 4

Figure 3 is the architecture diagram for the abstraction–verification–prediction pipeline. Figure 4 illustrates event canonicalization, support computation and threshold-based filtering. These are schematic illustrations rather than benchmark plots. Their editable figure sources and standalone generators are not present in the inspected tree.

## 7. Reproduce Figure 5

Figure 5 uses task-level predictions, failure annotations and candidate-budget measurements. Preserve the original task identifiers, split assignments, checkpoint, candidate ranking and scoring protocol for all methods.

### Panels 5a and 5b

Panel 5a groups tasks by transformation complexity: Easy (approximately 1–2 events), Medium (3–4 events), and Hard (at least 5 events). Panel 5b compares the primary failure categories of TRM (Matched), TRM (Voted) and Athena.

The repository provides atlas, forensic and diagnostic scripts, but the exact paper complexity labels, failure-category ledgers and dedicated plotting scripts were not identified. These inputs are required to reproduce the published panels.

### Panel 5c

Panel 5c studies the relationship between candidate budget, solution quality and computational cost. The paper evaluates `K = 16, 32, 64, 128`. A checked-in candidate-budget diagnostic can be launched after supplying a compatible checkpoint, configuration and reference ledger:

```bash
cd /path/to/athena/TinyRecursiveModels
python scripts/fvr_targeted_vote_k.py \
  --config /path/to/resolved_config.yaml \
  --checkpoint /path/to/checkpoint \
  --reference-ledger /path/to/reference_ledger.json \
  --aug-data-path /path/to/augmented_dataset \
  --out-dir runs/candidate_budget_diagnostic \
  --ks 16,32,64,128
```

This is a repository diagnostic, not a verified replacement for the paper's uncached TEM and consistency-weighted candidate-ranking experiment. Reproducing Panel 5c also requires the end-to-end timing and memory records for each budget.

### Tables 2–4

Table 2 evaluates cross-dataset generalization in both directions. Table 3 removes grounded rule canonicalization, compiled NLP evidence, support-based filtering, consistency weighting and transformation verification, and includes an unverified-transformation variant. Table 4 sweeps the verification threshold (`0.25, 0.50, 0.75, 1.00`) and candidate budget (`16, 32, 64, 128`).

Use separate output directories for each variant and preserve the matched data, checkpoint initialization, optimization and scoring settings. The exact paper ablation configurations, transfer checkpoints and result ledgers are required; the current checkout does not provide a verified one-command reproduction of these tables.
