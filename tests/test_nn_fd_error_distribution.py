
"""
======================================================================
TEST #19 — NN-FD ERROR DISTRIBUTION BY RESOLUTION
======================================================================

Project:
    NN-FF-FD for One-Dimensional Cubic Focusing NLS

Purpose:
    Diagnose the source of NN-FD accuracy variation across spatial
    resolutions.

This test DOES NOT retrain the model.

It analyzes:
    - Target magnitude
    - Target standard deviation
    - RMSE
    - MAE
    - Median absolute error
    - 95th percentile error
    - 99th percentile error
    - L-infinity error

Both NN-FD and Parent FD are evaluated.

======================================================================
"""

import os
import numpy as np
import torch

from nn_model import NNFDBaseline


# ======================================================================
# Configuration
# ======================================================================

DATA_FILE = "data/nn_fd/nn_fd_training_data.npz"
MODEL_FILE = "data/nn_fd/nn_fd_baseline_best.pt"
NORM_FILE = "data/nn_fd/nn_fd_normalization.npz"

OUTPUT_FILE = (
    "results/nn_fd/"
    "nn_fd_error_distribution_test19.npz"
)

RESOLUTIONS = [32, 64, 128, 256]

L = 20.0


# ======================================================================
# Metrics
# ======================================================================

def error_statistics(pred, ref):

    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    error = pred - ref
    abs_error = np.abs(error)

    return {
        "rmse":
            np.sqrt(np.mean(error ** 2)),

        "mae":
            np.mean(abs_error),

        "median":
            np.median(abs_error),

        "p95":
            np.percentile(abs_error, 95),

        "p99":
            np.percentile(abs_error, 99),

        "linf":
            np.max(abs_error),
    }


# ======================================================================
# Parent FD
# ======================================================================

def parent_fd_operator(X, dx):

    real = X[:, 0:5]
    imag = X[:, 5:10]

    real_xx = (
        real[:, 1]
        - 2.0 * real[:, 2]
        + real[:, 3]
    ) / dx**2

    imag_xx = (
        imag[:, 1]
        - 2.0 * imag[:, 2]
        + imag[:, 3]
    ) / dx**2

    return np.column_stack(
        (real_xx, imag_xx)
    )


# ======================================================================
# Load dataset
# ======================================================================

print("=" * 78)
print("TEST #19 — NN-FD ERROR DISTRIBUTION BY RESOLUTION")
print("=" * 78)

print()
print("PyTorch version:", torch.__version__)
print("Device: cpu")

data = np.load(DATA_FILE)

X = data["X"].astype(np.float32)
Y = data["Y"].astype(np.float32)
N_metadata = data["N"].astype(np.int32)

print()
print("Dataset:")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("N shape:", N_metadata.shape)


# ======================================================================
# Load normalization
# ======================================================================

norm = np.load(NORM_FILE)

X_mean = norm["x_mean"].astype(np.float32)
X_std = norm["x_std"].astype(np.float32)

Y_mean = norm["y_mean"].astype(np.float32)
Y_std = norm["y_std"].astype(np.float32)


# ======================================================================
# Load model
# ======================================================================

print()
print("Loading NN-FD model...")

device = torch.device("cpu")

model = NNFDBaseline()

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded.")


# ======================================================================
# Prediction
# ======================================================================

def nn_predict(X_raw):

    X_norm = (
        X_raw - X_mean
    ) / X_std

    X_tensor = torch.from_numpy(
        X_norm.astype(np.float32)
    )

    with torch.no_grad():

        Y_pred_norm = (
            model(X_tensor)
            .cpu()
            .numpy()
        )

    Y_pred = (
        Y_pred_norm * Y_std
        + Y_mean
    )

    return Y_pred


# ======================================================================
# Resolution-wise analysis
# ======================================================================

results = []

print()
print("=" * 78)
print("RESOLUTION-WISE ERROR DISTRIBUTION")
print("=" * 78)

for N in RESOLUTIONS:

    mask = (
        N_metadata == N
    )

    X_N = X[mask]
    Y_N = Y[mask]

    dx = L / N

    print()
    print("-" * 78)
    print(
        f"N = {N:4d}   "
        f"dx = {dx:.10e}   "
        f"samples = {len(X_N)}"
    )
    print("-" * 78)

    # --------------------------------------------------------------
    # Target statistics
    # --------------------------------------------------------------

    target_abs = np.abs(Y_N)

    target_mean_abs = np.mean(
        target_abs
    )

    target_std = np.std(
        Y_N
    )

    target_rms = np.sqrt(
        np.mean(Y_N ** 2)
    )

    print(
        f"Target mean |Y|   = "
        f"{target_mean_abs:.10e}"
    )

    print(
        f"Target std        = "
        f"{target_std:.10e}"
    )

    print(
        f"Target RMS        = "
        f"{target_rms:.10e}"
    )

    # --------------------------------------------------------------
    # NN-FD
    # --------------------------------------------------------------

    Y_nn = nn_predict(X_N)

    nn_stats = error_statistics(
        Y_nn,
        Y_N
    )

    # --------------------------------------------------------------
    # Parent FD
    # --------------------------------------------------------------

    Y_fd = parent_fd_operator(
        X_N,
        dx
    )

    fd_stats = error_statistics(
        Y_fd,
        Y_N
    )

    # --------------------------------------------------------------
    # Print
    # --------------------------------------------------------------

    print()
    print("NN-FD:")

    print(
        f"  RMSE   = {nn_stats['rmse']:.10e}"
    )

    print(
        f"  MAE    = {nn_stats['mae']:.10e}"
    )

    print(
        f"  Median = {nn_stats['median']:.10e}"
    )

    print(
        f"  P95    = {nn_stats['p95']:.10e}"
    )

    print(
        f"  P99    = {nn_stats['p99']:.10e}"
    )

    print(
        f"  Linf   = {nn_stats['linf']:.10e}"
    )

    print()
    print("Parent FD:")

    print(
        f"  RMSE   = {fd_stats['rmse']:.10e}"
    )

    print(
        f"  MAE    = {fd_stats['mae']:.10e}"
    )

    print(
        f"  Median = {fd_stats['median']:.10e}"
    )

    print(
        f"  P95    = {fd_stats['p95']:.10e}"
    )

    print(
        f"  P99    = {fd_stats['p99']:.10e}"
    )

    print(
        f"  Linf   = {fd_stats['linf']:.10e}"
    )

    # --------------------------------------------------------------
    # Store
    # --------------------------------------------------------------

    results.append(
        {
            "N": N,
            "dx": dx,

            "samples": len(X_N),

            "target_mean_abs":
                target_mean_abs,

            "target_std":
                target_std,

            "target_rms":
                target_rms,

            "nn_rmse":
                nn_stats["rmse"],

            "nn_mae":
                nn_stats["mae"],

            "nn_median":
                nn_stats["median"],

            "nn_p95":
                nn_stats["p95"],

            "nn_p99":
                nn_stats["p99"],

            "nn_linf":
                nn_stats["linf"],

            "fd_rmse":
                fd_stats["rmse"],

            "fd_mae":
                fd_stats["mae"],

            "fd_median":
                fd_stats["median"],

            "fd_p95":
                fd_stats["p95"],

            "fd_p99":
                fd_stats["p99"],

            "fd_linf":
                fd_stats["linf"],
        }
    )


# ======================================================================
# Summary table
# ======================================================================

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)

print()

print(
    "N       Target RMS       "
    "NN RMSE        NN Median      "
    "NN P95         NN P99"
)

print("-" * 90)

for r in results:

    print(
        f"{r['N']:4d}   "
        f"{r['target_rms']:.8e}   "
        f"{r['nn_rmse']:.8e}   "
        f"{r['nn_median']:.8e}   "
        f"{r['nn_p95']:.8e}   "
        f"{r['nn_p99']:.8e}"
    )


# ======================================================================
# Save
# ======================================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

np.savez(
    OUTPUT_FILE,

    N=np.array(
        [r["N"] for r in results]
    ),

    dx=np.array(
        [r["dx"] for r in results]
    ),

    samples=np.array(
        [r["samples"] for r in results]
    ),

    target_mean_abs=np.array(
        [r["target_mean_abs"] for r in results]
    ),

    target_std=np.array(
        [r["target_std"] for r in results]
    ),

    target_rms=np.array(
        [r["target_rms"] for r in results]
    ),

    nn_rmse=np.array(
        [r["nn_rmse"] for r in results]
    ),

    nn_mae=np.array(
        [r["nn_mae"] for r in results]
    ),

    nn_median=np.array(
        [r["nn_median"] for r in results]
    ),

    nn_p95=np.array(
        [r["nn_p95"] for r in results]
    ),

    nn_p99=np.array(
        [r["nn_p99"] for r in results]
    ),

    nn_linf=np.array(
        [r["nn_linf"] for r in results]
    ),

    fd_rmse=np.array(
        [r["fd_rmse"] for r in results]
    ),

    fd_mae=np.array(
        [r["fd_mae"] for r in results]
    ),

    fd_median=np.array(
        [r["fd_median"] for r in results]
    ),

    fd_p95=np.array(
        [r["fd_p95"] for r in results]
    ),

    fd_p99=np.array(
        [r["fd_p99"] for r in results]
    ),

    fd_linf=np.array(
        [r["fd_linf"] for r in results]
    ),
)

print()
print("=" * 78)
print("TEST #19 COMPLETED")
print("=" * 78)

print()
print("Results saved to:")
print(OUTPUT_FILE)