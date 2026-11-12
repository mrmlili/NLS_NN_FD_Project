# NN-FF-FD-RA: Resolution-Aware Fourier-Feature Neural Correction

This repository provides the numerical implementation, trained model
artifacts, evaluation data, and reproducibility materials associated with
the research work:

**Resolution-Aware Fourier-Feature Neural Correction for the
One-Dimensional Cubic Focusing Nonlinear Schrödinger Equation**

## Repository structure

- `src/`: numerical operators, solvers, data utilities, and neural models.
- `scripts/`: training, evaluation, benchmarking, and result-export scripts.
- `tests/`: numerical validation and conservation/convergence checks.
- `data/nn_fd/`: training dataset, model checkpoints, normalization
  parameters, and training summaries.
- `results/nn_fd/`: raw evaluation artifacts used by the result-export script.
- `results/`: selected manuscript tables and result summaries.
- `figures/`: manuscript figures.
- `REPRODUCIBILITY_MANIFEST.csv`: metadata and SHA256 values for 15
  archived data and model artifacts.

## Requirements

Run from the repository root:

```bash
pip install -r requirements.txt
```

## Evaluation artifacts

| Experiment | File |
|---|---|
| Pointwise holdout (TEST28) | `results/nn_fd/nn_ff_fd_ra_holdout_test28.npz` |
| Unseen resolution (TEST29) | `results/nn_fd/nn_ff_fd_ra_robustness_n512_test29.npz` |
| Independent realizations (TEST37) | `results/nn_fd/independent_realization_generalization_test37.npz` |
| CPU operator benchmark (TEST38S) | `results/nn_fd/operator_cpu_benchmark_test38s.npz` |
| Error-distribution metrics | `results/nn_fd/paper_results_final/test28_error_metrics.npz` |

The training dataset and baseline, V3, and original pointwise RA model
artifacts are stored under `data/nn_fd/`.

The grouped-split RA candidate is a separate lineage and is not substituted
for the original pointwise model in this release.

## Exporting tables and figures

Run from the repository root:

```bash
python scripts/paper_results_final.py
```

The script reads archived evaluation arrays and exports tables, figures,
and a summary under:

`results/nn_fd/paper_results_final_20_8/`

This is a result-export workflow; it does not retrain the neural models.

Training and evaluation scripts are also provided. Scripts using
repository-relative data and output paths should be launched from the
repository root. Training scripts may overwrite checkpoints; preserve
canonical artifacts before conducting a new training run.

## Reproducibility

Archived arrays, checkpoints, normalization parameters, source files,
and exported manuscript results support inspection and reproduction of
the reported operator-level experiments.

No claim is made that every end-to-end experiment was rerun while
preparing this release.

## Citation

Please cite the associated research work and identify the repository
version used. A persistent archive DOI will be added after registration.

## Licenses

- Software source code in `src/`, `scripts/`, and `tests/`: MIT License
  (`LICENSE`).
- Research datasets, model artifacts, numerical results, tables, and
  figures: Creative Commons Attribution 4.0 International (`LICENSE-DATA`).

Third-party dependencies remain subject to their respective licenses.

## Repository

https://github.com/mrmlili/NLS_NN_FD_Project