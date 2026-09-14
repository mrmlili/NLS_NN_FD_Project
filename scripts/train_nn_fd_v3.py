"""
=========================================================
NN-FD V3 TRAINING
=========================================================

Resolution-aware learned FD correction.

Input:
    10 stencil features + dx

Target:
    (u_xx_exact - Dxx_FD) / dx^2

=========================================================
"""

from pathlib import Path
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# Project root
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.dirname(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from nn_model_v3 import (
    NNFDV3,
    count_parameters,
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

DATA_FILE = Path(
    "data"
) / "nn_fd" / "nn_fd_training_data.npz"

MODEL_DIR = Path(
    "data"
) / "nn_fd"

MODEL_FILE = (
    MODEL_DIR
    / "nn_fd_v3_best.pt"
)

STATS_FILE = (
    MODEL_DIR
    / "nn_fd_v3_normalization.npz"
)

RESULT_FILE = (
    MODEL_DIR
    / "nn_fd_v3_training_results.npz"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

BATCH_SIZE = 512
EPOCHS = 300

LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6

PATIENCE = 30

L = 20.0


# =========================================================
# Reproducibility
# =========================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# Normalization
# =========================================================

def normalize_train(
    x_train,
    y_train,
):

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    x_std = np.where(
        x_std < 1.0e-12,
        1.0,
        x_std,
    )

    y_std = np.where(
        y_std < 1.0e-12,
        1.0,
        y_std,
    )

    return (
        x_mean,
        x_std,
        y_mean,
        y_std,
    )


def apply_normalization(
    x,
    y,
    x_mean,
    x_std,
    y_mean,
    y_std,
):

    return (
        (x - x_mean) / x_std,
        (y - y_mean) / y_std,
    )


def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            loss = criterion(
                pred,
                yb,
            )

            bs = xb.shape[0]

            total_loss += (
                loss.item() * bs
            )

            total_samples += bs

    return (
        total_loss / total_samples
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 78)
    print("NN-FD V3 — RESOLUTION-AWARE TRAINING")
    print("=" * 78)

    device = torch.device("cpu")

    print()
    print(
        "PyTorch version:",
        torch.__version__,
    )

    print(
        "Device:",
        device,
    )


    # -------------------------------------------------------
    # Load original corrected dataset
    # -------------------------------------------------------

    data = np.load(
        DATA_FILE
    )

    X = data["X"].astype(
        np.float32
    )

    Y = data["Y"].astype(
        np.float32
    )

    N_metadata = data["N"].astype(
        np.int32
    )

    print()
    print(
        "Original X shape:",
        X.shape,
    )

    print(
        "Original Y shape:",
        Y.shape,
    )


    # -------------------------------------------------------
    # Add dx feature
    # -------------------------------------------------------

    dx = (
        L
        / N_metadata.astype(
            np.float32
        )
    )

    dx_column = dx[:, None]

    X_v3 = np.concatenate(
        (
            X,
            dx_column,
        ),
        axis=1,
    )


    # -------------------------------------------------------
    # Scale target by dx^2
    # -------------------------------------------------------

    dx2 = dx_column ** 2

    Y_scaled = (
        Y / dx2
    )


    print()
    print(
        "V3 X shape:",
        X_v3.shape,
    )

    print(
        "V3 Y shape:",
        Y_scaled.shape,
    )

    print()
    print(
        "Scaled target RMS:",
        np.sqrt(
            np.mean(
                Y_scaled.astype(
                    np.float64
                ) ** 2
            )
        ),
    )


    # -------------------------------------------------------
    # Shuffle
    # -------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    indices = rng.permutation(
        len(X_v3)
    )

    X_v3 = X_v3[indices]
    Y_scaled = Y_scaled[indices]


    # -------------------------------------------------------
    # Split
    # -------------------------------------------------------

    N_total = len(X_v3)

    n_train = int(
        TRAIN_RATIO * N_total
    )

    n_val = int(
        VAL_RATIO * N_total
    )

    X_train = X_v3[:n_train]
    Y_train = Y_scaled[:n_train]

    X_val = X_v3[
        n_train:n_train+n_val
    ]

    Y_val = Y_scaled[
        n_train:n_train+n_val
    ]

    X_test = X_v3[
        n_train+n_val:
    ]

    Y_test = Y_scaled[
        n_train+n_val:
    ]

    print()
    print("Dataset split:")

    print(
        "Train:",
        X_train.shape,
    )

    print(
        "Validation:",
        X_val.shape,
    )

    print(
        "Test:",
        X_test.shape,
    )


    # -------------------------------------------------------
    # Normalization
    # -------------------------------------------------------

    (
        x_mean,
        x_std,
        y_mean,
        y_std,
    ) = normalize_train(
        X_train,
        Y_train,
    )

    X_train, Y_train = (
        apply_normalization(
            X_train,
            Y_train,
            x_mean,
            x_std,
            y_mean,
            y_std,
        )
    )

    X_val, Y_val = (
        apply_normalization(
            X_val,
            Y_val,
            x_mean,
            x_std,
            y_mean,
            y_std,
        )
    )

    X_test, Y_test = (
        apply_normalization(
            X_test,
            Y_test,
            x_mean,
            x_std,
            y_mean,
            y_std,
        )
    )


    # -------------------------------------------------------
    # Tensors
    # -------------------------------------------------------

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Y_train),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(Y_val),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(Y_test),
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------

    model = NNFDV3().to(
        device
    )

    print()
    print(model)

    print()
    print(
        "Trainable parameters:",
        count_parameters(model),
    )


    # -------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1.0e-6,
        )
    )


    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    train_history = []
    val_history = []
    lr_history = []

    print()
    print("=" * 78)
    print("TRAINING")
    print("=" * 78)


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(
                pred,
                yb,
            )

            loss.backward()

            optimizer.step()

            bs = xb.shape[0]

            running_loss += (
                loss.item() * bs
            )

            total_samples += bs

        train_loss = (
            running_loss
            / total_samples
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(
            val_loss
        )

        lr = optimizer.param_groups[0]["lr"]

        train_history.append(
            train_loss
        )

        val_history.append(
            val_loss
        )

        lr_history.append(
            lr
        )


        # ---------------------------------------------------
        # Best model
        # ---------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_dim": 11,

                    "target_scaling":
                        "dx^2",

                    "best_epoch":
                        best_epoch,

                    "best_val_loss":
                        best_val_loss,
                },
                MODEL_FILE,
            )

        else:

            epochs_without_improvement += 1


        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:4d} | "
                f"Train = {train_loss:.6e} | "
                f"Val = {val_loss:.6e} | "
                f"LR = {lr:.3e}"
            )


        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break


    # -------------------------------------------------------
    # Best model
    # -------------------------------------------------------

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    test_loss = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )


    # -------------------------------------------------------
    # Results
    # -------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL RESULTS")
    print("=" * 78)

    print()
    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.10e}",
    )

    print(
        "Test loss:",
        f"{test_loss:.10e}",
    )


    # -------------------------------------------------------
    # Save normalization
    # -------------------------------------------------------

    np.savez(
        STATS_FILE,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )


    # -------------------------------------------------------
    # Save history
    # -------------------------------------------------------

    np.savez(
        RESULT_FILE,
        train_loss=np.asarray(
            train_history
        ),
        val_loss=np.asarray(
            val_history
        ),
        learning_rate=np.asarray(
            lr_history
        ),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        test_loss=test_loss,
    )


    print()
    print(
        "Saved model:",
        MODEL_FILE,
    )

    print(
        "Saved normalization:",
        STATS_FILE,
    )

    print(
        "Saved training history:",
        RESULT_FILE,
    )

    print()
    print("=" * 78)
    print("NN-FD V3 TRAINING COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()