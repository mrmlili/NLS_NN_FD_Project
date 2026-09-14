"""
=========================================================

Energy conservation test for IMEX2-CNAB solver

Equation

    i*u_t + u_xx + |u|²u = 0

Project

    NN-FF-FD for One-Dimensional
    Cubic Focusing Nonlinear Schrödinger Equation

=========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from imex2_solver import IMEX2Solver
from exact_solution import (
    NLSParameters,
    soliton,
    numerical_energy
)


# ==========================================================
# Energy Test
# ==========================================================

def main():

    params = NLSParameters(
        alpha=1.0,
        beta=0.0,
        L=20.0
    )

    solver = IMEX2Solver(
        L=params.L,
        N=200,
        dt=0.001
    )

    x = solver.grid.x

    u0 = soliton(
        x,
        0.0,
        params
    )

    t, solution = solver.solve(
        u0,
        T=2.0
    )

    energies = []

    for u in solution:

        energies.append(
            numerical_energy(
                u,
                x
            )
        )

    energies = np.array(energies)

    E0 = energies[0]

    relative_error = np.abs(
        energies - E0
    ) / abs(E0)

    print("=" * 65)

    print("IMEX2 Energy Conservation Test")

    print()

    print("Initial Energy :", E0)

    print("Final Energy   :", energies[-1])

    print()

    print(
        "Maximum relative energy error:",
        np.max(relative_error)
    )

    print(
        "Minimum relative energy error:",
        np.min(relative_error)
    )

    print("=" * 65)

    # ------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    plt.figure(
        figsize=(7, 4)
    )

    plt.semilogy(
        t[2:],
        relative_error[2:],
        linewidth=2,
        label="Energy Error"
    )

    plt.xlabel("Time")

    plt.ylabel("Relative Energy Error")

    plt.title("Energy conservation of IMEX2-CNAB")

    plt.grid(True, which="both")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/Figure_3_Energy_Conservation.png",
        dpi=300
    )

    plt.savefig(
        "results/Figure_3_Energy_Conservation.pdf"
    )

    plt.show()


# ==========================================================

if __name__ == "__main__":

    main()