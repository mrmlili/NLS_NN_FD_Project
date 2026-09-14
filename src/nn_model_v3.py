"""
=========================================================
NN-FD V3 — RESOLUTION-AWARE MODEL
=========================================================

Input:
    10 local stencil features + dx

Target:
    scaled correction = correction / dx^2

Final correction:
    predicted_scaled_correction * dx^2

Architecture:
    11 -> 64 -> 64 -> 32 -> 2
=========================================================
"""

import torch
import torch.nn as nn


class NNFDV3(nn.Module):

    def __init__(
        self,
        input_dim=11,
        hidden1=64,
        hidden2=64,
        hidden3=32,
        output_dim=2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
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
        return self.network(x)


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def create_model():
    return NNFDV3()


if __name__ == "__main__":

    print("=" * 70)
    print("NN-FD V3 MODEL TEST")
    print("=" * 70)

    model = create_model()

    print()
    print(model)

    print()
    print(
        "Trainable parameters:",
        count_parameters(model),
    )

    x = torch.randn(
        8,
        11,
    )

    with torch.no_grad():
        y = model(x)

    print()
    print(
        "Input shape :",
        tuple(x.shape),
    )

    print(
        "Output shape:",
        tuple(y.shape),
    )

    assert x.shape == (8, 11)
    assert y.shape == (8, 2)

    print()
    print("=" * 70)
    print("NN-FD V3 MODEL TEST PASSED")
    print("=" * 70)