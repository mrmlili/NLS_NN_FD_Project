
"""
=========================================================
NN-FD Neural Network Model
=========================================================

Project:
    NN-FF-FD for One-Dimensional Cubic Focusing
    Nonlinear Schrödinger Equation

Purpose:
    Baseline Neural Network for learning the finite-
    difference spatial derivative operator.

Input:
    10 features
        - 5-point real stencil
        - 5-point imaginary stencil

Output:
    2 values
        - predicted real component
        - predicted imaginary component

Architecture:
    10 -> 64 -> 64 -> 32 -> 2

Activation:
    Tanh

Important:
    This module contains ONLY the neural network.
    The production IMEX2-CNAB solver is not modified.

=========================================================
"""

import torch
import torch.nn as nn


class NNFDBaseline(nn.Module):
    """
    Baseline NN-FD model.

    The network learns the mapping

        X -> Y

    where X contains the local complex-valued stencil
    and Y contains the corresponding finite-difference
    second-derivative target.
    """

    def __init__(
        self,
        input_dim=10,
        hidden1=64,
        hidden2=64,
        hidden3=32,
        output_dim=2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Tanh(),

            nn.Linear(hidden1, hidden2),
            nn.Tanh(),

            nn.Linear(hidden2, hidden3),
            nn.Tanh(),

            nn.Linear(hidden3, output_dim),
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, 10)

        Returns
        -------
        torch.Tensor
            Shape: (batch_size, 2)
        """

        return self.network(x)


def count_parameters(model):
    """
    Return the number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def create_model():
    """
    Factory function for the baseline NN-FD model.
    """

    model = NNFDBaseline()

    return model


if __name__ == "__main__":

    print("=" * 70)
    print("NN-FD BASELINE MODEL TEST")
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
        count_parameters(model)
    )

    # -----------------------------------------------------
    # Dummy forward-pass test
    # -----------------------------------------------------

    x_test = torch.randn(8, 10)

    with torch.no_grad():
        y_test = model(x_test)

    print()
    print("Input shape :", tuple(x_test.shape))
    print("Output shape:", tuple(y_test.shape))

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    assert x_test.shape == (8, 10)
    assert y_test.shape == (8, 2)

    print()
    print("=" * 70)
    print("NN-FD MODEL TEST PASSED")
    print("=" * 70)

