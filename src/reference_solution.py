"""
======================================================================

HIGH-ACCURACY REFERENCE SOLUTION
for 1D Cubic Focusing Nonlinear Schrödinger Equation

Equation:

    i*u_t + u_xx + |u|^2*u = 0

Purpose
-------
Generate a high-accuracy numerical reference solution for the
NN-FF-FD project.

The reference solution is generated using the production
IMEX2-CNAB solver with a fine spatial grid and a small time step.

This file is used ONLY for reference-data generation.

IMPORTANT
---------
The production solver (imex2_solver.py) is NOT modified here.

Reference configuration
-----------------------
    L      = 20
    N_ref  = 3200
    dt_ref = 1.5625e-4
    T      = 1.0

Output
------
    data/reference/reference_N3200_dt1p5625e-4_T1.npz

Stored quantities
-----------------
    x
    u
    L
    N
    dt
    T

======================================================================
"""

import os
import time
import numpy as np

from imex2_solver import IMEX2Solver

from exact_solution import (
    NLSParameters,
    soliton
)


# =====================================================================
# REFERENCE CONFIGURATION
# =====================================================================

L = 20.0

N_REF = 3200

DT_REF = 1.5625e-4

T_REF = 1.0


# =====================================================================
# OUTPUT PATH
# =====================================================================

REFERENCE_DIR = os.path.join(
    "data",
    "reference"
)


REFERENCE_FILE = os.path.join(
    REFERENCE_DIR,
    "reference_N3200_dt1p5625e-4_T1.npz"
)


# =====================================================================
# COMPUTE REFERENCE SOLUTION
# =====================================================================

def compute_reference_solution():

    """
    Generate the high-accuracy reference solution.

    Returns
    -------
    x : ndarray
        Periodic spatial grid.

    u : ndarray
        Numerical solution at T_REF.
    """

    params = NLSParameters(
        alpha=1.0,
        beta=0.0,
        L=L
    )

    print()
    print("=" * 80)
    print("HIGH-ACCURACY REFERENCE SOLUTION")
    print("=" * 80)

    print()
    print(f"L           = {L}")
    print(f"N_ref       = {N_REF}")
    print(f"dx_ref      = {L / N_REF:.12e}")
    print(f"dt_ref      = {DT_REF:.12e}")
    print(f"T_ref       = {T_REF}")
    print(f"Time steps  = {int(round(T_REF / DT_REF))}")

    print()
    print("Building production IMEX2-CNAB solver...")
    print()

    start_time = time.perf_counter()

    solver = IMEX2Solver(
        L=L,
        N=N_REF,
        dt=DT_REF
    )

    x = solver.grid.x

    # ---------------------------------------------------------------
    # Initial condition
    # ---------------------------------------------------------------

    u0 = soliton(
        x,
        0.0,
        params
    )

    print("Initial condition generated.")

    # ---------------------------------------------------------------
    # Final solution only
    # ---------------------------------------------------------------

    print()
    print("Computing reference solution...")
    print()

    u_ref = solver.solve_final(
        u0,
        T=T_REF
    )

    elapsed = time.perf_counter() - start_time

    print()
    print("Reference computation completed.")

    print()
    print(f"Elapsed time = {elapsed:.6f} seconds")

    print("=" * 80)

    return x, u_ref


# =====================================================================
# SAVE REFERENCE SOLUTION
# =====================================================================

def save_reference():

    """
    Generate and save the reference solution together with
    the numerical parameters required for reproducibility.
    """

    # ---------------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------------

    os.makedirs(
        REFERENCE_DIR,
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # Compute reference
    # ---------------------------------------------------------------

    x, u_ref = compute_reference_solution()

    # ---------------------------------------------------------------
    # Basic consistency checks
    # ---------------------------------------------------------------

    expected_dx = L / N_REF

    actual_N = len(x)

    if actual_N != N_REF:

        raise RuntimeError(
            f"Reference grid size mismatch: "
            f"expected {N_REF}, got {actual_N}"
        )

    if not np.all(np.isfinite(u_ref)):

        raise RuntimeError(
            "Reference solution contains "
            "NaN or Inf values."
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    np.savez(
        REFERENCE_FILE,

        x=x,

        u=u_ref,

        L=L,

        N=N_REF,

        dt=DT_REF,

        T=T_REF,

        dx=expected_dx
    )

    # ---------------------------------------------------------------
    # Verify saved file
    # ---------------------------------------------------------------

    data = np.load(
        REFERENCE_FILE
    )

    x_check = data["x"]

    u_check = data["u"]

    if x_check.shape != x.shape:

        raise RuntimeError(
            "Saved reference grid shape is incorrect."
        )

    if u_check.shape != u_ref.shape:

        raise RuntimeError(
            "Saved reference solution shape is incorrect."
        )

    if not np.allclose(
        x_check,
        x
    ):

        raise RuntimeError(
            "Saved spatial grid does not match "
            "the generated grid."
        )

    if not np.allclose(
        u_check,
        u_ref
    ):

        raise RuntimeError(
            "Saved reference solution does not match "
            "the generated solution."
        )

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("REFERENCE SOLUTION SUCCESSFULLY SAVED")
    print("=" * 80)

    print()
    print(f"File        : {REFERENCE_FILE}")
    print(f"N           : {N_REF}")
    print(f"dx          : {expected_dx:.12e}")
    print(f"dt          : {DT_REF:.12e}")
    print(f"T           : {T_REF}")
    print(f"Array size  : {u_ref.shape}")

    print()
    print("Reference solution is ready for use.")
    print("=" * 80)


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    save_reference()