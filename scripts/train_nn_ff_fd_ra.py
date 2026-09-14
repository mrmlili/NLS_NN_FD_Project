"""
=========================================================
NN-FF-FD-RA TRAINING PIPELINE
=========================================================

Project:
    NN-FF-FD for One-Dimensional Cubic Focusing
    Nonlinear Schrödinger Equation

Purpose:
    Train the final Resolution-Aware Fourier-Feature
    NN-FD candidate.

Input:
    10 stencil features + dx

Fourier features:
    8 deterministic local Fourier features

Total model input:
    19

Target:
    C_scaled = C / dx^2

where:

    C = u_xx_exact - Dxx_FD(u)

At inference:

    C_pred = C_scaled_pred * dx^2

Training protocol:
    Same corrected periodic dataset
    Same 70/15/15 split
    Same normalization policy
    Same optimizer
    Same batch size
    Same learning-rate schedule
    Same training budget

No production solver modification.

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


from nn_ff_fd_ra_model import (
    NNFFFDRA,
    count_parameters,
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

L = 20.0

DATA_FILE = (
    Path("data")
    / "nn_fd"
    / "nn_fd_training_data.npz"
)

MODEL_DIR = (
    Path("data")
    / "nn_fd"
)

MODEL_FILE = (
    MODEL_DIR
    / "nn_ff_fd_ra_best.pt"
)

STATS_FILE = (
    MODEL_DIR
    / "nn_ff_fd_ra_normalization.npz"
)

RESULT_FILE = (
    MODEL_DIR
    / "nn_ff_fd_ra_training_results.npz"
)


# Dataset split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# Training
BATCH_SIZE = 512
EPOCHS = 300

LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6

PATIENCE = 30


# =========================================================
# Reproducibility
# =========================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# Utility functions
# =========================================================

def normalize_train(
    x_train,
    y_train,
):
    """
    Compute normalization statistics using training data only.
    """

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
    """
    Apply training-fitted normalization statistics.
    """

    x_norm = (
        x - x_mean
    ) / x_std

    y_norm = (
        y - y_mean
    ) / y_std

    return (
        x_norm,
        y_norm,
    )


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate mean normalized MSE.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            prediction = model(
                xb
            )

            loss = criterion(
                prediction,
                yb,
            )

            batch_size = xb.shape[0]

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

    return (
        total_loss
        / total_samples
    )


# =========================================================
# Main training
# =========================================================

def main():

    print("=" * 78)
    print("NN-FF-FD-RA TRAINING")
    print("=" * 78)

    print()
    print(
        "PyTorch version:",
        torch.__version__,
    )

    device = torch.device(
        "cpu"
    )

    print(
        "Device:",
        device,
    )

    # -----------------------------------------------------
    # Load corrected periodic dataset
    # -----------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    print()
    print(
        "Loading dataset..."
    )

    print(
        "File:",
        DATA_FILE,
    )

    data = np.load(
        DATA_FILE
    )

    X = data["X"].astype(
        np.float32
    )

    Y = data["Y"].astype(
        np.float32
    )

    N_metadata = data[
        "N"
    ].astype(
        np.int32
    )

    print()
    print(
        "Dataset loaded."
    )

    print(
        "X shape:",
        X.shape,
    )

    print(
        "Y shape:",
        Y.shape,
    )

    print(
        "N shape:",
        N_metadata.shape,
    )

    # -----------------------------------------------------
    # Add dx feature
    # -----------------------------------------------------

    dx = (
        L
        / N_metadata.astype(
            np.float32
        )
    )

    dx_column = dx[:, None]

    X_ra = np.concatenate(
        (
            X,
            dx_column,
        ),
        axis=1,
    )

    # -----------------------------------------------------
    # Resolution-aware target scaling
    # -----------------------------------------------------

    dx_squared = (
        dx_column ** 2
    )

    Y_ra = (
        Y
        / dx_squared
    )

    print()
    print(
        "RA input shape:",
        X_ra.shape,
    )

    print(
        "RA target shape:",
        Y_ra.shape,
    )

    print()
    print(
        "Scaled target RMS:",
        np.sqrt(
            np.mean(
                Y_ra.astype(
                    np.float64
                ) ** 2
            )
        ),
    )

    # -----------------------------------------------------
    # Shuffle once
    # -----------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    indices = rng.permutation(
        len(X_ra)
    )

    X_ra = X_ra[
        indices
    ]

    Y_ra = Y_ra[
        indices
    ]

    # -----------------------------------------------------
    # Train / validation / test split
    # -----------------------------------------------------

    N_total = len(
        X_ra
    )

    n_train = int(
        TRAIN_RATIO * N_total
    )

    n_val = int(
        VAL_RATIO * N_total
    )

    n_test = (
        N_total
        - n_train
        - n_val
    )

    X_train = X_ra[
        :n_train
    ]

    Y_train = Y_ra[
        :n_train
    ]

    X_val = X_ra[
        n_train:
        n_train + n_val
    ]

    Y_val = Y_ra[
        n_train:
        n_train + n_val
    ]

    X_test = X_ra[
        n_train + n_val:
    ]

    Y_test = Y_ra[
        n_train + n_val:
    ]

    print()
    print(
        "Dataset split:"
    )

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

    # -----------------------------------------------------
    # Normalization
    # -----------------------------------------------------

    print()
    print(
        "Computing normalization statistics..."
    )

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

    # -----------------------------------------------------
    # Tensors
    # -----------------------------------------------------

    train_dataset = TensorDataset(
        torch.from_numpy(
            X_train
        ),
        torch.from_numpy(
            Y_train
        ),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(
            X_val
        ),
        torch.from_numpy(
            Y_val
        ),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(
            X_test
        ),
        torch.from_numpy(
            Y_test
        ),
    )

    # -----------------------------------------------------
    # Data loaders
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = NNFFFDRA().to(
        device
    )

    print()
    print(
        "Model:"
    )

    print(
        model
    )

    print()
    print(
        "Trainable parameters:",
        count_parameters(model),
    )

    # -----------------------------------------------------
    # Loss / optimizer
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # History
    # -----------------------------------------------------

    train_history = []
    val_history = []
    lr_history = []

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    epochs_without_improvement = 0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print()
    print("=" * 78)
    print(
        "TRAINING"
    )
    print("=" * 78)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:

            xb = xb.to(
                device
            )

            yb = yb.to(
                device
            )

            optimizer.zero_grad()

            prediction = model(
                xb
            )

            loss = criterion(
                prediction,
                yb,
            )

            loss.backward()

            optimizer.step()

            batch_size = xb.shape[
                0
            ]

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

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

        current_lr = (
            optimizer
            .param_groups[0]
            ["lr"]
        )

        train_history.append(
            train_loss
        )

        val_history.append(
            val_loss
        )

        lr_history.append(
            current_lr
        )

        # -------------------------------------------------
        # Best model
        # -------------------------------------------------

        if (
            val_loss
            < best_val_loss
        ):

            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_dim":
                        11,

                    "total_model_input":
                        19,

                    "target_scaling":
                        "dx^2",

                    "fourier_modes":
                        (
                            1,
                            2,
                        ),

                    "stencil_size":
                        5,

                    "best_epoch":
                        best_epoch,

                    "best_val_loss":
                        best_val_loss,
                },
                MODEL_FILE,
            )

        else:

            epochs_without_improvement += 1

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:4d} | "
                f"Train = "
                f"{train_loss:.6e} | "
                f"Val = "
                f"{val_loss:.6e} | "
                f"LR = "
                f"{current_lr:.3e}"
            )

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # -----------------------------------------------------
    # Load best model
    # -----------------------------------------------------

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------

    test_loss = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    # -----------------------------------------------------
    # Final results
    # -----------------------------------------------------

    print()
    print("=" * 78)
    print(
        "FINAL RESULTS"
    )
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

    # -----------------------------------------------------
    # Save normalization
    # -----------------------------------------------------

    np.savez(
        STATS_FILE,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    # -----------------------------------------------------
    # Save history
    # -----------------------------------------------------

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
        "Saved model:"
    )

    print(
        MODEL_FILE
    )

    print()
    print(
        "Saved normalization:"
    )

    print(
        STATS_FILE
    )

    print()
    print(
        "Saved training history:"
    )

    print(
        RESULT_FILE
    )

    print()
    print("=" * 78)
    print(
        "NN-FF-FD-RA TRAINING COMPLETED"
    )
    print("=" * 78)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()