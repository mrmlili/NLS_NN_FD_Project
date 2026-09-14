# NN-FF-FD-RA: Resolution-Aware Fourier-Feature Neural Correction

This repository contains the source code and selected reproducibility artifacts for the manuscript:

**"Resolution-Aware Fourier-Feature Neural Correction for the One-Dimensional Cubic Focusing Nonlinear Schrödinger Equation"**

## Requirements

* Python 3.8+
* Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

* `src/`: Core numerical solvers and neural-network models.
* `scripts/`: Training, evaluation, benchmarking, and result-generation scripts.
* `tests/`: Numerical validation and conservation/convergence tests.
* `data/`: Canonical reference-solution data.
* `results/`: Final exported tables and summary used for the manuscript.
* `figures/`: Final manuscript figures.

## Reproduction

The repository provides the source code required for the numerical methods, model evaluation, validation tests, and final-result generation.

The canonical final result script is:

```bash
python scripts/paper_results_final.py
```

Training scripts are provided in `scripts/`. Large training datasets and trained model checkpoints are not included in the repository.

## Results

The `results/` directory contains the final exported tables:

* `Table1_main_accuracy.csv`
* `Table2_unseen_512_robustness.csv`
* `Table3_independent_generalization.csv`
* `Table4_operator_cpu_cost.csv`
* `TableS1_error_distribution_metrics.csv`

The corresponding manuscript figures are included in `figures/`.

## Citation

If you use this code or its results, please cite the associated manuscript.

## Repository

https://github.com/mrmlili/NLS_NN_FD_Project
