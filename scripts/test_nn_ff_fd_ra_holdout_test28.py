"""
TEST #28-HOLDOUT — exact pointwise held-out test + independent N=512
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nn_data import (
    generate_function_with_derivative,
    periodic_dxx,
    build_stencil_features,
)
from nn_model import NNFDBaseline
from nn_model_v3 import NNFDV3
from nn_ff_fd_ra_model import NNFFFDRA

L = 20.0

TRAINING_RESOLUTIONS = [32, 64, 128, 256]

SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

UNSEEN_N = 512
UNSEEN_SAMPLES = 200
UNSEEN_SEED = 20260825

DATA_FILE = "data/nn_fd/nn_fd_training_data.npz"

NN_MODEL_FILE = "data/nn_fd/nn_fd_baseline_best.pt"
NN_NORM_FILE = "data/nn_fd/nn_fd_normalization.npz"

V3_MODEL_FILE = "data/nn_fd/nn_fd_v3_best.pt"
V3_NORM_FILE = "data/nn_fd/nn_fd_v3_normalization.npz"

RA_MODEL_FILE = "data/nn_fd/nn_ff_fd_ra_best.pt"
RA_NORM_FILE = "data/nn_fd/nn_ff_fd_ra_normalization.npz"

OUTPUT_FILE = "results/nn_fd/nn_ff_fd_ra_holdout_test28.npz"


def error_statistics(pred, ref):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    err = pred - ref
    a = np.abs(err)
    return {
        "rmse": np.sqrt(np.mean(err ** 2)),
        "mae": np.mean(a),
        "median": np.median(a),
        "p95": np.percentile(a, 95),
        "p99": np.percentile(a, 99),
        "linf": np.max(a),
    }


def load_model(cls, filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found:\n{filename}")
    model = cls()
    ckpt = torch.load(filename, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def load_normalization(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Normalization file not found:\n{filename}"
        )
    z = np.load(filename)
    return (
        z["x_mean"].astype(np.float32),
        z["x_std"].astype(np.float32),
        z["y_mean"].astype(np.float32),
        z["y_std"].astype(np.float32),
    )


def predict_standard(model, X_raw, norm):
    x_mean, x_std, y_mean, y_std = norm
    Xn = (X_raw - x_mean) / x_std
    with torch.no_grad():
        yn = model(torch.from_numpy(Xn.astype(np.float32))).cpu().numpy()
    return yn * y_std + y_mean


def predict_ra(model, X_stencil, dx, norm):
    x_mean, x_std, y_mean, y_std = norm
    dx_col = np.full((len(X_stencil), 1), dx, dtype=np.float32)
    Xin = np.concatenate((X_stencil, dx_col), axis=1)
    Xn = (Xin - x_mean) / x_std
    with torch.no_grad():
        yn = model(torch.from_numpy(Xn.astype(np.float32))).cpu().numpy()
    y_scaled = yn * y_std + y_mean
    return y_scaled * dx**2


def evaluate_dataset(X, Y, N, nn_model, nn_norm, v3_model, v3_norm, ra_model, ra_norm):
    dx = L / N

    parent = error_statistics(np.zeros_like(Y), Y)
    nn = error_statistics(predict_standard(nn_model, X, nn_norm), Y)
    v3 = error_statistics(predict_ra(v3_model, X, dx, v3_norm), Y)
    ra = error_statistics(predict_ra(ra_model, X, dx, ra_norm), Y)

    return parent, nn, v3, ra


def print_stats(label, s):
    print(f"\n{label}")
    print(f"  RMSE   = {s['rmse']:.10e}")
    print(f"  MAE    = {s['mae']:.10e}")
    print(f"  Median = {s['median']:.10e}")
    print(f"  P95    = {s['p95']:.10e}")
    print(f"  P99    = {s['p99']:.10e}")
    print(f"  Linf   = {s['linf']:.10e}")


def print_compare(parent, nn, v3, ra):
    a = parent["rmse"] / nn["rmse"]
    b = parent["rmse"] / v3["rmse"]
    c = parent["rmse"] / ra["rmse"]
    d = nn["rmse"] / v3["rmse"]
    e = nn["rmse"] / ra["rmse"]
    f = v3["rmse"] / ra["rmse"]

    print("\nRMSE improvement over Parent:")
    print(f"  NN-FD       = {a:.6f} x")
    print(f"  NN-FD V3    = {b:.6f} x")
    print(f"  NN-FF-FD-RA = {c:.6f} x")

    print("\nRelative RMSE:")
    print(f"  V3 / NN-FD  = {d:.6f}")
    print(f"  RA / NN-FD  = {e:.6f}")
    print(f"  RA / V3     = {f:.6f}")

    return a, b, c, d, e, f


print("=" * 78)
print("TEST #28-HOLDOUT — HELD-OUT + UNSEEN N=512")
print("=" * 78)

print("\nPyTorch version:", torch.__version__)
print("Device: cpu")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset not found:\n{DATA_FILE}")

data = np.load(DATA_FILE)
X_all = data["X"].astype(np.float32)
Y_all = data["Y"].astype(np.float32)
N_meta = data["N"].astype(np.int32)

print("\nOriginal dataset:")
print("  X shape:", X_all.shape)
print("  Y shape:", Y_all.shape)
print("  N shape:", N_meta.shape)

# Exact reproduction of the training split.
rng = np.random.default_rng(SPLIT_SEED)
indices = rng.permutation(len(X_all))

n_total = len(X_all)
n_train = int(TRAIN_RATIO * n_total)
n_val = int(VAL_RATIO * n_total)

test_indices = indices[n_train + n_val:]

X_test = X_all[test_indices]
Y_test = Y_all[test_indices]
N_test = N_meta[test_indices]

print("\n" + "=" * 78)
print("EXACT HELD-OUT TEST SUBSET")
print("=" * 78)
print("Split seed:", SPLIT_SEED)
print("Train samples:", n_train)
print("Validation samples:", n_val)
print("Held-out test samples:", len(test_indices))

for N in TRAINING_RESOLUTIONS:
    print(
        f"N = {N:4d} | held-out samples = "
        f"{int(np.sum(N_test == N))}"
    )

print("\nLoading frozen models...")
nn_model = load_model(NNFDBaseline, NN_MODEL_FILE)
v3_model = load_model(NNFDV3, V3_MODEL_FILE)
ra_model = load_model(NNFFFDRA, RA_MODEL_FILE)
print("All models loaded.")

nn_norm = load_normalization(NN_NORM_FILE)
v3_norm = load_normalization(V3_NORM_FILE)
ra_norm = load_normalization(RA_NORM_FILE)
print("All training-derived normalization files loaded.")

results = []

print("\n" + "=" * 78)
print("HELD-OUT IN-RANGE RESOLUTION COMPARISON")
print("=" * 78)

for N in TRAINING_RESOLUTIONS:
    mask = N_test == N
    X_N = X_test[mask]
    Y_N = Y_test[mask]

    parent, nn, v3, ra = evaluate_dataset(
        X_N, Y_N, N,
        nn_model, nn_norm,
        v3_model, v3_norm,
        ra_model, ra_norm,
    )

    print("\n" + "-" * 78)
    print(
        f"N = {N:4d} | dx = {L/N:.10e} | "
        f"held-out samples = {len(X_N)}"
    )
    print("-" * 78)

    print_stats("Parent FD:", parent)
    print_stats("NN-FD:", nn)
    print_stats("NN-FD V3:", v3)
    print_stats("NN-FF-FD-RA:", ra)

    cmp = print_compare(parent, nn, v3, ra)

    results.append({
        "N": N,
        "unseen": False,
        "dx": L / N,
        "samples": len(X_N),
        "parent_rmse": parent["rmse"],
        "nn_rmse": nn["rmse"],
        "v3_rmse": v3["rmse"],
        "ra_rmse": ra["rmse"],
        "nn_parent_rmse": cmp[0],
        "v3_parent_rmse": cmp[1],
        "ra_parent_rmse": cmp[2],
        "v3_vs_nn_rmse": cmp[3],
        "ra_vs_nn_rmse": cmp[4],
        "ra_vs_v3_rmse": cmp[5],
    })

print("\n" + "=" * 78)
print("INDEPENDENT UNSEEN RESOLUTION N=512")
print("=" * 78)

N = UNSEEN_N
dx = L / N
rng = np.random.default_rng(UNSEEN_SEED)

X_list = []
Y_list = []

for _ in range(UNSEEN_SAMPLES):
    x = -L / 2.0 + dx * np.arange(N)

    u, u_xx_exact = generate_function_with_derivative(x, rng)
    u_xx_fd = periodic_dxx(u, dx)
    correction = u_xx_exact - u_xx_fd

    X_list.append(build_stencil_features(u))
    Y_list.append(
        np.column_stack(
            (correction.real, correction.imag)
        )
    )

X_512 = np.vstack(X_list).astype(np.float32)
Y_512 = np.vstack(Y_list).astype(np.float32)

print(f"\nN = {N}")
print(f"dx = {dx:.10e}")
print(f"independent realizations = {UNSEEN_SAMPLES}")
print("X shape:", X_512.shape)
print("Y shape:", Y_512.shape)

parent, nn, v3, ra = evaluate_dataset(
    X_512, Y_512, N,
    nn_model, nn_norm,
    v3_model, v3_norm,
    ra_model, ra_norm,
)

print_stats("Parent FD:", parent)
print_stats("NN-FD:", nn)
print_stats("NN-FD V3:", v3)
print_stats("NN-FF-FD-RA:", ra)

cmp = print_compare(parent, nn, v3, ra)

results.append({
    "N": N,
    "unseen": True,
    "dx": dx,
    "samples": UNSEEN_SAMPLES * N,
    "parent_rmse": parent["rmse"],
    "nn_rmse": nn["rmse"],
    "v3_rmse": v3["rmse"],
    "ra_rmse": ra["rmse"],
    "nn_parent_rmse": cmp[0],
    "v3_parent_rmse": cmp[1],
    "ra_parent_rmse": cmp[2],
    "v3_vs_nn_rmse": cmp[3],
    "ra_vs_nn_rmse": cmp[4],
    "ra_vs_v3_rmse": cmp[5],
})

print("\n" + "=" * 78)
print("TEST #28-HOLDOUT SUMMARY")
print("=" * 78)
print("\nN        Parent RMSE       NN-FD RMSE        V3 RMSE           RA RMSE")
print("-" * 115)

for r in results:
    tag = "[UNSEEN]" if r["unseen"] else "[HELD-OUT]"
    print(
        f"{r['N']:4d}   "
        f"{r['parent_rmse']:.8e}   "
        f"{r['nn_rmse']:.8e}   "
        f"{r['v3_rmse']:.8e}   "
        f"{r['ra_rmse']:.8e}   "
        f"{tag}"
    )

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

np.savez(
    OUTPUT_FILE,
    split_seed=np.array(SPLIT_SEED),
    train_ratio=np.array(TRAIN_RATIO),
    val_ratio=np.array(VAL_RATIO),
    test_ratio=np.array(1.0 - TRAIN_RATIO - VAL_RATIO),
    unseen_seed=np.array(UNSEEN_SEED),
    unseen_samples=np.array(UNSEEN_SAMPLES),
    N=np.array([r["N"] for r in results], dtype=np.int32),
    unseen=np.array([r["unseen"] for r in results], dtype=bool),
    dx=np.array([r["dx"] for r in results], dtype=np.float64),
    samples=np.array([r["samples"] for r in results], dtype=np.int32),
    parent_rmse=np.array([r["parent_rmse"] for r in results]),
    nn_rmse=np.array([r["nn_rmse"] for r in results]),
    v3_rmse=np.array([r["v3_rmse"] for r in results]),
    ra_rmse=np.array([r["ra_rmse"] for r in results]),
    nn_parent_rmse=np.array([r["nn_parent_rmse"] for r in results]),
    v3_parent_rmse=np.array([r["v3_parent_rmse"] for r in results]),
    ra_parent_rmse=np.array([r["ra_parent_rmse"] for r in results]),
    v3_vs_nn_rmse=np.array([r["v3_vs_nn_rmse"] for r in results]),
    ra_vs_nn_rmse=np.array([r["ra_vs_nn_rmse"] for r in results]),
    ra_vs_v3_rmse=np.array([r["ra_vs_v3_rmse"] for r in results]),
)

print("\n" + "=" * 78)
print("TEST #28-HOLDOUT COMPLETED")
print("=" * 78)
print("\nResults saved to:")
print(OUTPUT_FILE)
