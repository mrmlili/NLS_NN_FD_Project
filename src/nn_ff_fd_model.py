"""
=========================================================
NN-FF-FD Fourier-Feature Neural Network Model
=========================================================

Project:
    NN-FF-FD for One-Dimensional Cubic Focusing
    Nonlinear Schrödinger Equation

Purpose:
    Fourier-feature-enhanced neural network for learning
    the finite-difference spatial correction.

Input:
    10 normalized local stencil features.

Fourier mapping:
    gamma(X) = [sin(2*pi*B*X), cos(2*pi*B*X)]

    B is a fixed, deterministic Fourier projection matrix.

The original 10 normalized features are also retained.

Therefore:

    10 original features
          +
     8 Fourier features
          =
          18 features

Architecture:

    18 -> 64 -> 64 -> 32 -> 2

Activation:
    Tanh

Important:
    - B is NOT trainable.
    - The production IMEX2-CNAB solver is not modified.
    - This model is a separate NN-FF-FD ablation model.

=========================================================
"""

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# Configuration
# =====================================================================

FOURIER_SEED = 20260813

NUM_FOURIER_FEATURES = 4

FOURIER_SCALE = 1.0


# =====================================================================
# Fourier Feature Mapping
# =====================================================================

class FourierFeatureMapping(nn.Module):
    """
    Fixed Fourier-feature mapping.

    Input:
        x : (batch_size, input_dim)

    Output:
        mapped_x : (batch_size, 2 * num_fourier_features)

    The projection matrix B is fixed and stored as a buffer.
    """

    def __init__(
        self,
        input_dim=10,
        num_fourier_features=4,
        scale=1.0,
        seed=20260813,
    ):
        super().__init__()

        rng = np.random.default_rng(seed)

        B = rng.normal(
            loc=0.0,
            scale=scale,
            size=(
                num_fourier_features,
                input_dim,
            ),
        ).astype(
            np.float32
        )

        self.register_buffer(
            "B",
            torch.from_numpy(B),
        )

    def forward(self, x):

        projection = (
            x @ self.B.T
        )

        phase = (
            2.0
            * np.pi
            * projection
        )

        return torch.cat(
            (
                torch.sin(phase),
                torch.cos(phase),
            ),
            dim=1,
        )


# =====================================================================
# NN-FF-FD Model
# =====================================================================

class NNFFFD(nn.Module):
    """
    Fourier-feature-enhanced NN-FD model.

    Input:
        10 normalized stencil features.

    Internal representation:
        10 original features
        +
        8 Fourier features
        =
        18 features.

    Output:
        2 correction components:
            real
            imaginary
    """

    def __init__(
        self,
        input_dim=10,
        hidden1=64,
        hidden2=64,
        hidden3=32,
        output_dim=2,
        num_fourier_features=4,
        fourier_scale=1.0,
        fourier_seed=20260813,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.num_fourier_features = (
            num_fourier_features
        )

        self.fourier = FourierFeatureMapping(
            input_dim=input_dim,
            num_fourier_features=num_fourier_features,
            scale=fourier_scale,
            seed=fourier_seed,
        )

        mapped_dim = (
            2
            * num_fourier_features
        )

        total_input_dim = (
            input_dim
            + mapped_dim
        )

        self.network = nn.Sequential(
            nn.Linear(
                total_input_dim,
                hidden1,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden1,
                hidden2,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden2,
                hidden3,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden3,
                output_dim,
            ),
        )

    def forward(self, x):

        fourier_features = (
            self.fourier(x)
        )

        combined = torch.cat(
            (
                x,
                fourier_features,
            ),
            dim=1,
        )

        return self.network(
            combined
        )


# =====================================================================
# Parameter count
# =====================================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# =====================================================================
# Factory
# =====================================================================

def create_model():

    return NNFFFD(
        input_dim=10,
        hidden1=64,
        hidden2=64,
        hidden3=32,
        output_dim=2,
        num_fourier_features=4,
        fourier_scale=FOURIER_SCALE,
        fourier_seed=FOURIER_SEED,
    )


# =====================================================================
# Standalone test
# =====================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NN-FF-FD MODEL TEST")
    print("=" * 70)

    print()
    print("PyTorch version:")
    print(torch.__version__)

    print()
    print("Creating model...")

    model = create_model()

    print()
    print(model)

    print()
    print(
        "Trainable parameters:",
        count_parameters(model),
    )

    print()
    print(
        "Fourier projection shape:",
        tuple(
            model.fourier.B.shape
        ),
    )

    # --------------------------------------------------------------
    # Dummy forward test
    # --------------------------------------------------------------

    x_test = torch.randn(
        8,
        10,
    )

    with torch.no_grad():

        fourier_test = model.fourier(
            x_test
        )

        y_test = model(
            x_test
        )

    print()
    print(
        "Input shape:",
        tuple(
            x_test.shape
        ),
    )

    print(
        "Fourier shape:",
        tuple(
            fourier_test.shape
        ),
    )

    print(
        "Output shape:",
        tuple(
            y_test.shape
        ),
    )

    # --------------------------------------------------------------
    # Assertions
    # --------------------------------------------------------------

    assert (
        x_test.shape
        == (8, 10)
    )

    assert (
        fourier_test.shape
        == (8, 8)
    )

    assert (
        y_test.shape
        == (8, 2)
    )

    print()
    print("=" * 70)
    print("NN-FF-FD MODEL TEST PASSED")
    print("=" * 70)