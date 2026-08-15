# Athena: Verified Transformation Evidence for ARC-AGI

> **If a transformation explains one demonstration, why should it be trusted on an unseen ARC query?**

Athena is a neuro-symbolic framework for the Abstraction and Reasoning Corpus (ARC-AGI). It separates **rule abstraction**, **rule verification**, and **rule execution** instead of asking a neural model to discover and execute a task rule through a single implicit representation.

Given a small set of ARC demonstrations, Athena uses **Qwen3-VL-8B-Instruct** as a Transformation Extraction Module (TEM) to describe observable changes. It grounds and canonicalizes those descriptions, retains transformation evidence supported across demonstrations, compiles the verified evidence into structured features, and conditions a Tiny Recursive Model (TRM) to generate candidate output grids.

The paper associated with this repository is:

> **If the Rule Explains the Demonstrations, Why Doesn't It Solve ARC-AGI?**

## Motivation

An ARC task contains demonstration pairs

$$
\tau = \{(X_m,Y_m)\}_{m=1}^{M}
$$

and an unseen query input $X^*$. The objective is to infer a task-specific transformation $f_\tau$ and predict

$$
\hat{Y}=f_\tau(X^*).
$$

The central difficulty is that an observed transformation can explain one demonstration without expressing the rule shared by the complete task. Athena therefore evaluates transformation evidence across demonstrations before allowing that evidence to guide prediction.

Athena is built around two hypotheses:

- **H1 - Cross-demonstration support:** transformations supported consistently across demonstrations provide a stronger basis for rule abstraction than isolated observations.
- **H2 - Verified execution:** restricting rule execution to verified transformation evidence improves prediction accuracy and reduces reliance on large candidate budgets.

## Method

### 1. Spatially Grounded Visual Evidence

For each demonstration pair $(X_m,Y_m)$, Athena constructs

$$
A^{\mathrm{vis}}_m = \big(R(X_m),R(Y_m),P_m,T_m\big),
$$

where:

- $R(G)$ is a cell-aligned relational map containing grid validity, component geometry, boundary relations, containment, and colour-distance features;
- $P_m$ contains support-derived core-prior evidence; and
- $T_m$ contains aligned colour and object-transition evidence derived from the observable change between $X_m$ and $Y_m$.

The relational map preserves spatial alignment with the ARC grid, allowing symbolic or object-level evidence to be fused with neural cell representations.

### 2. Qwen3-VL Transformation Extraction Module

Athena uses [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) as its Transformation Extraction Module. For demonstration $m$, TEM receives the input grid, output grid, and grounded visual evidence:

$$
F_{\mathrm{TEM}}:(X_m,Y_m,A^{\mathrm{vis}}_m)\mapsto L_m,
$$

where $L_m=\{\ell_{m,1},\ldots,\ell_{m,N_m}\}$ is a set of grounded textual descriptions of observable changes.

TEM is used on the **demonstration side**. It extracts candidate transformation evidence; it does not directly decide the task rule, produce the final grid, or inspect the unavailable query target. Raw TEM text is not concatenated with ARC grid tokens.

### 3. Grounded Rule Canonicalization

Free-form descriptions cannot be compared reliably across demonstrations. Athena therefore maps each extracted description to a typed event:

$$
e_{m,i}=C(\ell_{m,i};A^{\mathrm{vis}}_m)
= (o_{m,i},s_{m,i},\kappa_{m,i},p_{m,i},t_{m,i},q_{m,i}),
$$

with:

| Symbol | Meaning |
|---|---|
| $o$ | transformation operation |
| $s$ | grounded source-object description |
| $\kappa$ | spatial or relational condition |
| $p$ | operation parameters |
| $t$ | target value or location |
| $q$ | transformation order |

Instance-specific identifiers are replaced with grounded relational descriptions and symbolic parameters. Lexical variants are normalized to a fixed operator vocabulary. The event inventory includes recolouring, translation, reflection, rotation, replication, insertion, removal, and counting-based modification.

Malformed events are discarded and exact duplicates are removed within each demonstration:

$$
E_m=\operatorname{Unique}\{e_{m,1},\ldots,e_{m,N_m}\}.
$$

### 4. Cross-Demonstration Verification

The task-level evidence pool is

$$
E_\tau=\bigcup_{m=1}^{M}E_m.
$$

For each canonical event $e$, Athena computes a demonstration-level support score

$$
s(e)=\sum_{m=1}^{M}\mathbb{I}[e\in E_m]
$$

and normalized consistency

$$
c(e)=\frac{s(e)}{M}.
$$

Because support is counted after within-demonstration deduplication, repeated descriptions inside one demonstration cannot inflate an event's score. The verified evidence set is

$$
E_\tau^V=\{e\in E_\tau\mid c(e)\geq\gamma\}.
$$

The paper selects $\gamma=0.75$. In this formulation, *verified* means that an event satisfies the cross-demonstration support criterion. It does not mean that the event has been proven to be the unique underlying ARC rule.

### 5. Evidence Compilation and TRM Execution

Verified events are compiled into structured neural features containing the operation, grounded source and target masks, relational condition, transformation parameters, target value or location, order, and consistency score:

$$
A_\tau^{\mathrm{nlp}}=\operatorname{Compile}(E_\tau^V;R(X^*)).
$$

These features are fused with query-side visual evidence:

$$
A_\tau=\operatorname{Fuse}(A_\tau^{\mathrm{vis}},A_\tau^{\mathrm{nlp}}).
$$

TRM acts as Athena's compact neural rule executor:

$$
\hat{Y}_k=G_\theta(X^*,A_\tau), \qquad k=1,\ldots,K.
$$

Relational, core-prior, and transition evidence condition the recursive state. The factored structure and colour heads remain responsible for writing the output grid.

### 6. Candidate Ranking and Two-Attempt Prediction

Athena generates $K$ candidates from task-preserving augmented views, maps them back to the canonical frame, and removes duplicates. Each distinct candidate is ranked by consistency-weighted agreement with the verified evidence:

$$
S_{\mathrm{VE}}(\hat{Y}_k)
=
\frac{\sum_{e\in E_\tau^V}c(e)\,\mathbb{I}[\hat{Y}_k\models_{X^*}e]}
{\sum_{e\in E_\tau^V}c(e)}.
$$

Ties prefer fewer unsupported edits and then higher TRM confidence. Athena submits the two highest-ranked distinct candidates under the ARC two-attempt protocol.

## Installation

### Requirements

- Python 3.10 or compatible
- PyTorch with CUDA support
- CUDA 12.6 or compatible for GPU training
- A CUDA-capable GPU; the paper's full experiments used four NVIDIA A100 80 GB GPUs
- Access to the Qwen3-VL-8B-Instruct model for TEM inference

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/pushkar1887/trm_c2.git
cd trm_c2/TinyRecursiveModels

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Weights & Biases logging is optional:

```bash
wandb login
```

## ARC Dataset Preparation

Place the ARC-AGI JSON files under `TinyRecursiveModels/kaggle/combined/`, or replace the input prefix below with the location of your ARC files.

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

Do not combine ARC-AGI-1 and ARC-AGI-2 into a single uncontrolled train/evaluation pool: ARC-AGI-2 training data overlaps with ARC-AGI-1 evaluation data.

## Training the Neural Rule Executor

The TRM executor is configured through Hydra. A single-process experiment can be launched with:

```bash
python pretrain.py \
  --config-name cfg_fvr_experiment \
  arch=trm_fvr_c2 \
  data_paths="[data/arc1concept-aug-1000]" \
  epochs=500 \
  global_batch_size=32 \
  lr=1e-4 \
  weight_decay=0.01
```

For four-GPU distributed training:

```bash
torchrun \
  --nproc-per-node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  pretrain.py \
  --config-name cfg_fvr_experiment \
  arch=trm_fvr_c2 \
  data_paths="[data/arc1concept-aug-1000]" \
  epochs=500 \
  global_batch_size=32 \
  lr=1e-4 \
  weight_decay=0.01
```

Hydra overrides should be recorded with the checkpoint and resolved configuration. Checkpoint compatibility depends on the model's evidence schema and puzzle-identifier mapping; do not load a checkpoint against a differently indexed dataset without validating that mapping.

## Acknowledgements

Athena builds on the [Tiny Recursive Models](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) codebase and the [Abstraction and Reasoning Corpus](https://github.com/fchollet/ARC-AGI). The Transformation Extraction Module uses [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
