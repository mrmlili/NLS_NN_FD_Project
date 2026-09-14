
"""
======================================================================
TEST #18 — NN-FD vs PARENT FD
Resolution-wise Operator Accuracy Benchmark
======================================================================

Project:
    NN-FF-FD for One-Dimensional Cubic Focusing NLS

Purpose:
    Compare the trained NN-FD local operator against the parent
    second-order finite-difference operator on exactly the same
    dataset samples.

Important:
    The evaluation is performed separately for each spatial
    resolution. This avoids empty subsets caused by a global
    train/test split.

Resolutions:
    N = 32, 64, 128, 256

Metrics:
    RMSE
    MAE
    Relative L2
    Linf
    CPU prediction time

======================================================================
"""

import time
import numpy as np
import torch

from nn_model import NNFDBaseline


# ======================================================================
# Configuration
# ======================================================================

DATA_FILE = "data/nn_fd/nn_fd_training_data.npz"
MODEL_FILE = "data/nn_fd/nn_fd_baseline_best.pt"
NORM_FILE = "data/nn_fd/nn_fd_normalization.npz"

RESOLUTIONS = [32, 64, 128, 256]

L = 20.0

# Number of samples generated per resolution
SAMPLES_PER_RESOLUTION = 200

# Dataset contains four resolutions.
TOTAL_SAMPLES = len(RESOLUTIONS) * SAMPLES_PER_RESOLUTION


# ======================================================================
# Utility functions
# ======================================================================

def compute_metrics(pred, ref):
    """
    Compute global vector-based error metrics.
    """

    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    error = pred - ref
    abs_error = np.abs(error)

    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(abs_error)

    l2 = np.linalg.norm(error.ravel())
    ref_l2 = np.linalg.norm(ref.ravel())

    if ref_l2 > 0.0:
        rel_l2 = l2 / ref_l2
    else:
        rel_l2 = np.nan

    linf = np.max(abs_error)

    return {
        "rmse": rmse,
        "mae": mae,
        "l2": l2,
        "relative_l2": rel_l2,
        "linf": linf,
    }


# ======================================================================
# Parent FD operator
# ======================================================================

def parent_fd_operator(X, dx):
    """
    Standard centered second-order finite-difference approximation
    of the second derivative.

    The NN input contains a radius-2 stencil:

        [u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}]

    for both real and imaginary components.

    X layout:

        X[:, 0:5]  -> real stencil
        X[:, 5:10] -> imaginary stencil

    The Parent FD uses only the central three points:

        u_xx ≈ (u_{i-1} - 2u_i + u_{i+1}) / dx^2

    The output contains:

        [Re(u_xx), Im(u_xx)]
    """

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

    return np.column_stack((real_xx, imag_xx))


# ======================================================================
# Load dataset
# ======================================================================

print("=" * 78)
print("TEST #18 — NN-FD vs PARENT FD")
print("Resolution-wise Operator Accuracy Benchmark")
print("=" * 78)

print()
print("PyTorch version:", torch.__version__)
print("Device: cpu")

device = torch.device("cpu")


print()
print("Loading NN-FD dataset...")
print("File:", DATA_FILE)

data = np.load(DATA_FILE)

X = data["X"].astype(np.float32)
Y = data["Y"].astype(np.float32)
N_metadata = data["N"].astype(np.int32)

print()
print("Dataset loaded.")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("N metadata shape:", N_metadata.shape)


# ======================================================================
# Verify dataset structure
# ======================================================================

print()
print("=" * 78)
print("DATASET STRUCTURE CHECK")
print("=" * 78)

for N in RESOLUTIONS:

    count = np.sum(N_metadata == N)

    print(
        f"N = {N:4d} | "
        f"Samples = {count:5d}"
    )

    if count == 0:
        raise RuntimeError(
            f"No samples found for N = {N}"
        )


# ======================================================================
# Load normalization
# ======================================================================

print()
print("Loading normalization...")
print("File:", NORM_FILE)

norm = np.load(NORM_FILE)

X_mean = norm["x_mean"].astype(np.float32)
X_std = norm["x_std"].astype(np.float32)

Y_mean = norm["y_mean"].astype(np.float32)
Y_std = norm["y_std"].astype(np.float32)

print("Normalization loaded.")


# ======================================================================
# Load trained NN-FD model
# ======================================================================

print()
print("Loading trained NN-FD model...")
print("File:", MODEL_FILE)

model = NNFDBaseline()

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

# Support both plain state_dict and checkpoint dictionaries.
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print()
print(model)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(
    "Trainable parameters:",
    trainable_parameters
)


# ======================================================================
# NN-FD prediction helper
# ======================================================================

def nn_predict(X_raw):
    """
    Predict physical-space Y values using the trained NN.
    """

    X_norm = (
        X_raw - X_mean
    ) / X_std

    X_tensor = torch.from_numpy(
        X_norm.astype(np.float32)
    ).to(device)

    with torch.no_grad():

        Y_pred_norm = model(
            X_tensor
        ).cpu().numpy()

    Y_pred = (
        Y_pred_norm * Y_std
        + Y_mean
    )

    return Y_pred


# ======================================================================
# Global evaluation
# ======================================================================

print()
print("=" * 78)
print("GLOBAL OPERATOR ACCURACY")
print("=" * 78)

start = time.perf_counter()

Y_pred_all = nn_predict(X)

prediction_time = (
    time.perf_counter() - start
)

global_metrics = compute_metrics(
    Y_pred_all,
    Y
)

print(
    f"NN-FD RMSE        = "
    f"{global_metrics['rmse']:.12e}"
)

print(
    f"NN-FD MAE         = "
    f"{global_metrics['mae']:.12e}"
)

print(
    f"NN-FD Relative L2 = "
    f"{global_metrics['relative_l2']:.12e}"
)

print(
    f"NN-FD Linf        = "
    f"{global_metrics['linf']:.12e}"
)

print(
    f"Prediction time   = "
    f"{prediction_time:.6f} s"
)


# ======================================================================
# Resolution-wise evaluation
# ======================================================================

print()
print("=" * 78)
print("RESOLUTION-WISE NN-FD vs PARENT FD")
print("=" * 78)

print()
print(
    "N       dx              "
    "NN RMSE          "
    "FD RMSE           "
    "NN Rel-L2        "
    "FD Rel-L2"
)

print("-" * 100)


results = []


for N in RESOLUTIONS:

    mask = (
        N_metadata == N
    )

    X_N = X[mask]
    Y_N = Y[mask]

    dx = L / N

    # --------------------------------------------------------------
    # NN-FD
    # --------------------------------------------------------------

    start = time.perf_counter()

    Y_nn = nn_predict(X_N)

    nn_time = (
        time.perf_counter() - start
    )

    nn_metrics = compute_metrics(
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

    fd_metrics = compute_metrics(
        Y_fd,
        Y_N
    )

    # --------------------------------------------------------------
    # Print
    # --------------------------------------------------------------

    print(
        f"{N:4d}   "
        f"{dx:.10e}   "
        f"{nn_metrics['rmse']:.10e}   "
        f"{fd_metrics['rmse']:.10e}   "
        f"{nn_metrics['relative_l2']:.10e}   "
        f"{fd_metrics['relative_l2']:.10e}"
    )

    results.append(
        {
            "N": N,
            "dx": dx,

            "nn_rmse":
                nn_metrics["rmse"],

            "nn_mae":
                nn_metrics["mae"],

            "nn_relative_l2":
                nn_metrics["relative_l2"],

            "nn_linf":
                nn_metrics["linf"],

            "fd_rmse":
                fd_metrics["rmse"],

            "fd_mae":
                fd_metrics["mae"],

            "fd_relative_l2":
                fd_metrics["relative_l2"],

            "fd_linf":
                fd_metrics["linf"],

            "nn_time":
                nn_time,
        }
    )


# ======================================================================
# Detailed comparison
# ======================================================================

print()
print("=" * 78)
print("DETAILED COMPARISON")
print("=" * 78)

print()

for r in results:

    print(
        f"N = {r['N']}"
    )

    print(
        f"  dx                = "
        f"{r['dx']:.12e}"
    )

    print(
        f"  NN-FD RMSE        = "
        f"{r['nn_rmse']:.12e}"
    )

    print(
        f"  Parent FD RMSE    = "
        f"{r['fd_rmse']:.12e}"
    )

    print(
        f"  NN-FD Relative L2 = "
        f"{r['nn_relative_l2']:.12e}"
    )

    print(
        f"  Parent FD Rel-L2  = "
        f"{r['fd_relative_l2']:.12e}"
    )

    print(
        f"  NN-FD Linf        = "
        f"{r['nn_linf']:.12e}"
    )

    print(
        f"  Parent FD Linf    = "
        f"{r['fd_linf']:.12e}"
    )

    print(
        f"  NN prediction     = "
        f"{r['nn_time']:.6f} s"
    )

    print()


# ======================================================================
# Save results
# ======================================================================

output_file = (
    "results/nn_fd/"
    "nn_fd_vs_parent_fd_test18.npz"
)

np.savez(
    output_file,
    N=np.array(
        [r["N"] for r in results]
    ),
    dx=np.array(
        [r["dx"] for r in results]
    ),

    nn_rmse=np.array(
        [r["nn_rmse"] for r in results]
    ),
    nn_mae=np.array(
        [r["nn_mae"] for r in results]
    ),
    nn_relative_l2=np.array(
        [r["nn_relative_l2"] for r in results]
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
    fd_relative_l2=np.array(
        [r["fd_relative_l2"] for r in results]
    ),
    fd_linf=np.array(
        [r["fd_linf"] for r in results]
    ),

    global_nn_rmse=
        global_metrics["rmse"],

    global_nn_mae=
        global_metrics["mae"],

    global_nn_relative_l2=
        global_metrics["relative_l2"],

    global_nn_linf=
        global_metrics["linf"],
)


# ======================================================================
# Final message
# ======================================================================

print("=" * 78)
print("TEST #18 COMPLETED")
print("=" * 78)

print()
print("Results saved to:")
print(output_file)

