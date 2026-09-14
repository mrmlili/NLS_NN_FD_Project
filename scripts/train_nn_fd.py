
"""
=========================================================
NN-FD TRAINING PIPELINE
=========================================================

Project:
    NN-FF-FD for One-Dimensional Cubic Focusing
    Nonlinear Schrödinger Equation

Purpose:
    Train the baseline NN-FD model on the generated
    finite-difference stencil dataset.

Dataset:
    data/nn_fd/nn_fd_training_data.npz

Model:
    NNFDBaseline

Important:
    - Train / validation / test split
    - Normalization fitted ONLY on training data
    - Best model selected using validation loss
    - Test set used only after training
    - CPU-compatible
    - Reproducible random seed

=========================================================
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from nn_model import NNFDBaseline, count_parameters


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

MODEL_FILE = MODEL_DIR / "nn_fd_baseline_best.pt"

STATS_FILE = MODEL_DIR / "nn_fd_normalization.npz"

RESULT_FILE = MODEL_DIR / "nn_fd_training_results.npz"


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

def normalize_train(x_train, y_train):
    """
    Compute normalization statistics from training data only.
    """

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    # Protect against zero standard deviation.
    x_std = np.where(x_std < 1.0e-12, 1.0, x_std)
    y_std = np.where(y_std < 1.0e-12, 1.0, y_std)

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
    Apply previously computed normalization statistics.
    """

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    return x_norm, y_norm


def evaluate(model, loader, criterion, device):
    """
    Evaluate mean loss on a dataset.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            prediction = model(xb)

            loss = criterion(
                prediction,
                yb,
            )

            batch_size = xb.shape[0]

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


# =========================================================
# Main training
# =========================================================

def main():

    print("=" * 78)
    print("NN-FD BASELINE TRAINING")
    print("=" * 78)

    print()
    print("PyTorch version:", torch.__version__)

    device = torch.device("cpu")

    print("Device:", device)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    print()
    print("Loading dataset...")
    print("File:", DATA_FILE)

    data = np.load(DATA_FILE)

    X = data["X"].astype(
        np.float32
    )

    Y = data["Y"].astype(
        np.float32
    )

    N = X.shape[0]

    print()
    print("Dataset loaded.")

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # -----------------------------------------------------
    # Shuffle once
    # -----------------------------------------------------

    rng = np.random.default_rng(SEED)

    indices = rng.permutation(N)

    X = X[indices]
    Y = Y[indices]

    # -----------------------------------------------------
    # Train / validation / test split
    # -----------------------------------------------------

    n_train = int(
        TRAIN_RATIO * N
    )

    n_val = int(
        VAL_RATIO * N
    )

    n_test = (
        N
        - n_train
        - n_val
    )

    X_train = X[:n_train]
    Y_train = Y[:n_train]

    X_val = X[
        n_train:
        n_train + n_val
    ]

    Y_val = Y[
        n_train:
        n_train + n_val
    ]

    X_test = X[
        n_train + n_val:
    ]

    Y_test = Y[
        n_train + n_val:
    ]

    print()
    print("Dataset split:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # -----------------------------------------------------
    # Normalization
    # -----------------------------------------------------

    print()
    print("Computing normalization statistics...")

    (
        x_mean,
        x_std,
        y_mean,
        y_std,
    ) = normalize_train(
        X_train,
        Y_train,
    )

    X_train, Y_train = apply_normalization(
        X_train,
        Y_train,
        x_mean,
        x_std,
        y_mean,
        y_std,
    )

    X_val, Y_val = apply_normalization(
        X_val,
        Y_val,
        x_mean,
        x_std,
        y_mean,
        y_std,
    )

    X_test, Y_test = apply_normalization(
        X_test,
        Y_test,
        x_mean,
        x_std,
        y_mean,
        y_std,
    )

    # -----------------------------------------------------
    # Convert to tensors
    # -----------------------------------------------------

    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)

    X_val = torch.from_numpy(X_val)
    Y_val = torch.from_numpy(Y_val)

    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test)

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_dataset = TensorDataset(
        X_train,
        Y_train,
    )

    val_dataset = TensorDataset(
        X_val,
        Y_val,
    )

    test_dataset = TensorDataset(
        X_test,
        Y_test,
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

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = NNFDBaseline().to(device)

    print()
    print("Model:")
    print(model)

    print()
    print(
        "Trainable parameters:",
        count_parameters(model),
    )

    # -----------------------------------------------------
    # Loss and optimizer
    # -----------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1.0e-6,
    )

    # -----------------------------------------------------
    # Training history
    # -----------------------------------------------------

    train_history = []
    val_history = []
    lr_history = []

    best_val_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

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

            prediction = model(xb)

            loss = criterion(
                prediction,
                yb,
            )

            loss.backward()

            optimizer.step()

            batch_size = xb.shape[0]

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

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

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

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

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_dim": 10,

                    "output_dim": 2,

                    "hidden1": 64,

                    "hidden2": 64,

                    "hidden3": 32,

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
                f"Train = {train_loss:.6e} | "
                f"Val = {val_loss:.6e} | "
                f"LR = {current_lr:.3e}"
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

    # -----------------------------------------------------
    # Save normalization statistics
    # -----------------------------------------------------

    np.savez(
        STATS_FILE,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    # -----------------------------------------------------
    # Save training history
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

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print()
    print("Saved model:")
    print(MODEL_FILE)

    print()
    print("Saved normalization:")
    print(STATS_FILE)

    print()
    print("Saved training history:")
    print(RESULT_FILE)

    print()
    print("=" * 78)
    print("NN-FD TRAINING COMPLETED")
    print("=" * 78)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()

