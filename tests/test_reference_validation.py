"""
=========================================================

Reference validation of IMEX2-CNAB solver

Compare numerical solution against a high-accuracy
reference solution.

Project

NN-FF-FD for One-Dimensional
Cubic Focusing Nonlinear Schrödinger Equation

=========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from imex2_solver import IMEX2Solver

from exact_solution import (
    NLSParameters,
    soliton
)


# ==========================================================
# Main
# ==========================================================

def main():

    params = NLSParameters(
        alpha=1.0,
        beta=0.0,
        L=20.0
    )

    # ------------------------------------------------------
    # Load reference solution
    # ------------------------------------------------------

    reference = np.load(
        "reference_solution.npz"
    )

    x_ref = reference["x"]

    u_ref = reference["u"]

    # ------------------------------------------------------
    # Numerical solution
    # ------------------------------------------------------

    N = 200

    dt = 0.001

    solver = IMEX2Solver(
        L=params.L,
        N=N,
        dt=dt
    )

    x = solver.grid.x

    u0 = soliton(
        x,
        0.0,
        params
    )

    u_num = solver.solve_final(
        u0,
        T=2.0
    )

    # ------------------------------------------------------
    # Interpolate reference onto coarse grid
    # ------------------------------------------------------

    interp_real = interp1d(
        x_ref,
        np.real(u_ref),
        kind="cubic",
        fill_value="extrapolate"
    )

    interp_imag = interp1d(
        x_ref,
        np.imag(u_ref),
        kind="cubic",
        fill_value="extrapolate"
    )

    u_reference = (
        interp_real(x)
        +
        1j*interp_imag(x)
    )

    # ------------------------------------------------------
    # Errors
    # ------------------------------------------------------

    error = np.abs(
        u_num-u_reference
    )

    L2 = np.sqrt(
        np.trapezoid(
            error**2,
            x
        )
    )

    Linf = np.max(error)

    print("="*65)

    print("Reference Solution Validation")

    print()

    print("Reference Grid :", len(x_ref))

    print("Solver Grid    :", len(x))

    print()

    print("L2 Error  :", L2)

    print("Linf Error:", Linf)

    print("="*65)

    # ------------------------------------------------------
    # Figures
    # ------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    # ======================================================
    # Figure 4
    # ======================================================

    plt.figure(figsize=(8,4))

    plt.plot(
        x,
        np.abs(u_reference),
        linewidth=2,
        label="Reference"
    )

    plt.plot(
        x,
        np.abs(u_num),
        "--",
        linewidth=2,
        label="IMEX2"
    )

    plt.xlabel("x")

    plt.ylabel("|u|")

    plt.title(
        "Reference vs IMEX2 solution"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/Figure_4_reference_validation.png",
        dpi=300
    )

    plt.savefig(
        "results/Figure_4_reference_validation.pdf"
    )

    # ======================================================
    # Figure 5
    # ======================================================

    plt.figure(figsize=(8,4))

    plt.plot(
        x,
        error,
        linewidth=2
    )

    plt.xlabel("x")

    plt.ylabel("Absolute Error")

    plt.title(
        "Absolute error with respect to reference"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/Figure_5_reference_error.png",
        dpi=300
    )

    plt.savefig(
        "results/Figure_5_reference_error.pdf"
    )

    plt.show()


# ==========================================================

if __name__ == "__main__":

    main()