
"""
=========================================================
TEST #17 — NN-FD INDEPENDENT TEST ACCURACY
=========================================================

Purpose
-------
Independent evaluation of the trained NN-FD baseline model
on the held-out test set.

This test does NOT modify the production FD solver.

Outputs
-------
- MSE
- RMSE
- MAE
- Relative L2 error
- L_inf error
- Separate real/imaginary errors
- Sample prediction comparison

Project
-------
NN-FF-FD for One-Dimensional Cubic Focusing
Nonlinear Schrödinger Equation

=========================================================
"""

import os
import numpy as np
import torch

from nn_model import NNFDBaseline


# =========================================================
# Configuration
# =========================================================

DATA_FILE = "data/nn_fd/nn_fd_training_data.npz"
MODEL_FILE = "data/nn_fd/nn_fd_baseline_best.pt"
NORM_FILE = "data/nn_fd/nn_fd_normalization.npz"

SEED = 12345


# =========================================================
# Reproducibility
# =========================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# Header
# =========================================================

print("=" * 78)
print("TEST #17 — NN-FD INDEPENDENT TEST ACCURACY")
print("=" * 78)

print()
print("PyTorch version:", torch.__version__)
print("Device: cpu")


# =========================================================
# Check files
# =========================================================

for filename in [DATA_FILE, MODEL_FILE, NORM_FILE]:

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"Required file not found:\n{filename}"
        )


# =========================================================
# Load dataset
# =========================================================

print()
print("Loading dataset...")
print("File:", DATA_FILE)

data = np.load(DATA_FILE)

X = data["X"].astype(np.float32)
Y = data["Y"].astype(np.float32)

print()
print("Dataset loaded.")
print("X shape:", X.shape)
print("Y shape:", Y.shape)


# =========================================================
# Reproduce the same split used during training
# =========================================================

n_total = len(X)

n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)

train_end = n_train
val_end = n_train + n_val

X_test = X[val_end:]
Y_test = Y[val_end:]

print()
print("Test set:")
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


# =========================================================
# Load normalization statistics
# =========================================================

print()
print("Loading normalization statistics...")
print("File:", NORM_FILE)

norm = np.load(NORM_FILE)

X_mean = norm["x_mean"].astype(np.float32)
X_std = norm["x_std"].astype(np.float32)

Y_mean = norm["y_mean"].astype(np.float32)
Y_std = norm["y_std"].astype(np.float32)

print("Normalization loaded.")


# =========================================================
# Normalize test data
# =========================================================

X_test_norm = (X_test - X_mean) / X_std
Y_test_norm = (Y_test - Y_mean) / Y_std


# =========================================================
# Load model
# =========================================================

print()
print("Loading trained NN-FD model...")
print("File:", MODEL_FILE)

model = NNFDBaseline()

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)

# Support both possible save formats:
# 1. direct state_dict
# 2. dictionary containing state_dict

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)


model.eval()

print()
print(model)

print()
print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# =========================================================
# Prediction
# =========================================================

print()
print("Running independent test evaluation...")

with torch.no_grad():

    X_tensor = torch.from_numpy(X_test_norm)

    Y_pred_norm = model(X_tensor).cpu().numpy()


# =========================================================
# De-normalize prediction
# =========================================================

Y_pred = (
    Y_pred_norm * Y_std
    + Y_mean
)


# =========================================================
# Error arrays
# =========================================================

error = Y_pred - Y_test

abs_error = np.abs(error)


# =========================================================
# Global metrics
# =========================================================

mse = np.mean(error ** 2)

rmse = np.sqrt(mse)

mae = np.mean(abs_error)

l2_abs = np.linalg.norm(error.ravel())

l2_ref = np.linalg.norm(Y_test.ravel())

relative_l2 = l2_abs / l2_ref

linf = np.max(abs_error)


# =========================================================
# Component-wise metrics
# =========================================================

real_error = error[:, 0]
imag_error = error[:, 1]

real_abs = np.abs(real_error)
imag_abs = np.abs(imag_error)

real_rmse = np.sqrt(np.mean(real_error ** 2))
imag_rmse = np.sqrt(np.mean(imag_error ** 2))

real_mae = np.mean(real_abs)
imag_mae = np.mean(imag_abs)

real_linf = np.max(real_abs)
imag_linf = np.max(imag_abs)

real_relative_l2 = (
    np.linalg.norm(real_error)
    / np.linalg.norm(Y_test[:, 0])
)

imag_relative_l2 = (
    np.linalg.norm(imag_error)
    / np.linalg.norm(Y_test[:, 1])
)


# =========================================================
# Print results
# =========================================================

print()
print("=" * 78)
print("GLOBAL NN-FD TEST ACCURACY")
print("=" * 78)

print()
print(f"MSE            = {mse:.12e}")
print(f"RMSE           = {rmse:.12e}")
print(f"MAE            = {mae:.12e}")
print(f"Absolute L2    = {l2_abs:.12e}")
print(f"Relative L2    = {relative_l2:.12e}")
print(f"Linf           = {linf:.12e}")


# =========================================================
# Component results
# =========================================================

print()
print("=" * 78)
print("COMPONENT-WISE ACCURACY")
print("=" * 78)

print()
print("Real component:")
print(f"  RMSE         = {real_rmse:.12e}")
print(f"  MAE          = {real_mae:.12e}")
print(f"  Relative L2  = {real_relative_l2:.12e}")
print(f"  Linf         = {real_linf:.12e}")

print()
print("Imaginary component:")
print(f"  RMSE         = {imag_rmse:.12e}")
print(f"  MAE          = {imag_mae:.12e}")
print(f"  Relative L2  = {imag_relative_l2:.12e}")
print(f"  Linf         = {imag_linf:.12e}")


# =========================================================
# Sample predictions
# =========================================================

print()
print("=" * 78)
print("SAMPLE PREDICTIONS")
print("=" * 78)

sample_indices = [
    0,
    len(X_test) // 4,
    len(X_test) // 2,
    3 * len(X_test) // 4,
    len(X_test) - 1
]

print()
print(
    "Index        "
    "Target Re        Pred Re         "
    "Target Im        Pred Im"
)

print("-" * 78)

for idx in sample_indices:

    print(
        f"{idx:6d}   "
        f"{Y_test[idx, 0]: .8e}   "
        f"{Y_pred[idx, 0]: .8e}   "
        f"{Y_test[idx, 1]: .8e}   "
        f"{Y_pred[idx, 1]: .8e}"
    )


# =========================================================
# Worst-case sample
# =========================================================

sample_error = np.linalg.norm(
    error,
    axis=1
)

worst_idx = np.argmax(sample_error)

print()
print("=" * 78)
print("WORST TEST SAMPLE")
print("=" * 78)

print()
print("Test index:", worst_idx)

print(
    "Target:",
    Y_test[worst_idx]
)

print(
    "Prediction:",
    Y_pred[worst_idx]
)

print(
    "Absolute error:",
    abs_error[worst_idx]
)

print(
    "Sample L2 error:",
    sample_error[worst_idx]
)


# =========================================================
# Save results
# =========================================================

RESULT_DIR = "results/nn_fd"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RESULT_FILE = (
    f"{RESULT_DIR}/nn_fd_accuracy_test17.npz"
)

np.savez(
    RESULT_FILE,

    mse=mse,
    rmse=rmse,
    mae=mae,

    absolute_l2=l2_abs,
    relative_l2=relative_l2,
    linf=linf,

    real_rmse=real_rmse,
    real_mae=real_mae,
    real_relative_l2=real_relative_l2,
    real_linf=real_linf,

    imag_rmse=imag_rmse,
    imag_mae=imag_mae,
    imag_relative_l2=imag_relative_l2,
    imag_linf=imag_linf
)


# =========================================================
# Final status
# =========================================================

print()
print("=" * 78)
print("TEST #17 COMPLETED")
print("=" * 78)

print()
print("Results saved to:")
print(RESULT_FILE)

print()
print("NN-FD independent test evaluation is complete.")
print("=" * 78)

