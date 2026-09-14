"""
======================================================================
NN-FF-FD-RA MODEL
======================================================================

Project:
    NN-FF-FD for One-Dimensional Cubic Focusing NLS

Purpose:
    Resolution-aware Fourier-feature-enhanced neural correction model.

Input:
    10 local stencil features
    + 8 deterministic local Fourier features
    + 1 resolution feature dx

Total input:
    19

Target:
    scaled correction

        C_scaled = C / dx^2

where

        C = u_xx_exact - Dxx_FD(u)

At inference:

        C_pred = C_scaled_pred * dx^2

Architecture:

    19 -> 64 -> 64 -> 32 -> 2

======================================================================
"""

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# Local Fourier Feature Mapping
# ======================================================================

class LocalFourierFeatureMapping(nn.Module):

    def __init__(
        self,
        stencil_size=5,
        modes=(1, 2),
    ):
        super().__init__()

        if stencil_size != 5:
            raise ValueError(
                "This implementation requires a 5-point stencil."
            )

        shifts = np.arange(
            -2,
            3,
            dtype=np.float32,
        )

        basis = []

        for k in modes:

            angle = (
                2.0
                * np.pi
                * k
                * shifts
                / stencil_size
            )

            cos_basis = np.cos(
                angle
            ).astype(
                np.float32
            )

            sin_basis = np.sin(
                angle
            ).astype(
                np.float32
            )

            basis.append(
                cos_basis
            )

            basis.append(
                sin_basis
            )

        basis = np.stack(
            basis,
            axis=0,
        )

        self.register_buffer(
            "basis",
            torch.from_numpy(
                basis
            ),
        )

    def forward(self, x):

        real = x[:, 0:5]
        imag = x[:, 5:10]

        real_fourier = (
            real
            @ self.basis.T
        )

        imag_fourier = (
            imag
            @ self.basis.T
        )

        return torch.cat(
            (
                real_fourier,
                imag_fourier,
            ),
            dim=1,
        )


# ======================================================================
# Resolution-Aware NN-FF-FD
# ======================================================================

class NNFFFDRA(nn.Module):
    """
    Resolution-aware Fourier-feature-enhanced model.

    Input:
        10 stencil features
        1 dx feature

    Fourier features:
        8 features derived from the 5-point stencil

    Total:
        19 features
    """

    def __init__(
        self,
        input_dim=11,
        hidden1=64,
        hidden2=64,
        hidden3=32,
        output_dim=2,
    ):
        super().__init__()

        self.fourier = (
            LocalFourierFeatureMapping()
        )

        # 10 stencil + 8 Fourier + 1 dx
        total_input_dim = 19

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

        if x.shape[1] != 11:
            raise ValueError(
                "NNFFFDRA expects 11 input features: "
                "10 stencil features + dx."
            )

        stencil = x[:, :10]
        dx = x[:, 10:11]

        fourier_features = (
            self.fourier(
                stencil
            )
        )

        combined = torch.cat(
            (
                stencil,
                fourier_features,
                dx,
            ),
            dim=1,
        )

        return self.network(
            combined
        )


# ======================================================================
# Parameter count
# ======================================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ======================================================================
# Factory
# ======================================================================

def create_model():

    return NNFFFDRA()


# ======================================================================
# Standalone test
# ======================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("NN-FF-FD-RA MODEL TEST")
    print("=" * 72)

    print()
    print(
        "PyTorch version:"
    )

    print(
        torch.__version__
    )

    print()
    print(
        "Creating model..."
    )

    model = create_model()

    print()
    print(model)

    print()
    print(
        "Trainable parameters:",
        count_parameters(model),
    )

    # --------------------------------------------------------------
    # Dummy forward pass
    # --------------------------------------------------------------

    x_test = torch.randn(
        8,
        11,
    )

    with torch.no_grad():

        fourier_test = (
            model.fourier(
                x_test[:, :10]
            )
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
    # Verification
    # --------------------------------------------------------------

    assert x_test.shape == (
        8,
        11,
    )

    assert fourier_test.shape == (
        8,
        8,
    )

    assert y_test.shape == (
        8,
        2,
    )

    print()
    print("=" * 72)
    print(
        "NN-FF-FD-RA MODEL TEST PASSED"
    )
    print("=" * 72)