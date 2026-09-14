# NN-FF-FD-RA: Resolution-Aware Fourier-Feature Neural Correction

This repository contains the source code for the paper:
**"Resolution-Aware Fourier-Feature Neural Correction for the One-Dimensional Cubic Focusing Nonlinear Schrödinger Equation"**

## Requirements
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

## Project Structure
- `src/`: Core numerical solvers and neural network models.
- `scripts/`: Training, evaluation, and figure generation scripts.
- `tests/`: Unit tests for mass conservation, energy, and convergence.
- `data/`: Reference solution data.
- `results/`: Exported CSV files for tables.
- `figures/`: Generated figures for the manuscript.

## How to Reproduce Results
1. Train the models: `python scripts/train_nn_ff_fd_ra.py`
2. Evaluate and generate figures: `python scripts/paper_results_final.py`

## Data and Code Availability
The source code is available at: https://github.com/YOUR_USERNAME/NLS_NN_FD_Project

## Citation
If you use this code, please cite the corresponding paper.
