# Correct-by-Construction

This repository contains compact PyTorch examples for fairness-aware training on tabular data. The core mechanism is to control a sensitive feature either by deterministic response expansion or by randomized response during training and evaluation.

## Contents

- [`census-example.py`](census-example.py): Census Income example.
- [`lsa-example.py`](lsa-example.py): Law School Admissions example.
- [`response.py`](response.py): Batch expansion across sensitive-feature categories.
- [`stochastic.py`](stochastic.py): Randomized-response perturbation utilities.
- [`steps.py`](steps.py): Training, prediction, and fairness-evaluation steps.

## Dependencies

```bash
pip install torch torchiteration numpy pandas scikit-learn pyarrow tensorboard
```

The example scripts read parquet files directly from Hugging Face dataset URLs, so parquet support must be installed locally.

## Running the examples

```bash
python census-example.py
python lsa-example.py
```

Both scripts:

- load a tabular dataset,
- train a `simplecnn` model from `cat-claws/nn`,
- write TensorBoard logs under `runs/`,
- evaluate consistency under sensitive-feature flips.

## TensorBoard

```bash
tensorboard --logdir runs
```

## Notes

- Runtime artifacts such as TensorBoard logs, checkpoints, CSV exports, and cached result folders are intentionally ignored.
- The fairness evaluation in [`steps.py`](steps.py) reports both standard accuracy and consistency under sensitive-feature reversal.
