"""
======================================================================
NN-FD DATA GENERATION — PERIODICALLY CONSISTENT VERSION
======================================================================

Project:
    NN-FF-FD for 1D cubic focusing NLS

Purpose:
    Generate supervised data for learning the spatial finite-difference
    correction.

The neural network learns:

    Correction = u_xx_exact - Dxx_FD(u)

and therefore the final learned operator is:

    Dxx_NNFD(u) = Dxx_FD(u) + Correction_NN(u)

Important:
    The training functions are now explicitly periodic so that they are
    mathematically consistent with the periodic finite-difference
    operator based on np.roll().
======================================================================
"""

import numpy as np
from pathlib import Path


# =====================================================================
# Configuration
# =====================================================================

L = 20.0

OUTPUT_DIR = Path("data") / "nn_fd"

RANDOM_SEED = 20260811

TRAIN_N_VALUES = [32, 64, 128, 256]

SAMPLES_PER_N = 200

STENCIL_RADIUS = 2

# Number of periodic image copies used to periodize the Gaussian.
#
# The exact periodized Gaussian is an infinite image sum. A finite
# truncation with sufficiently many image copies gives machine-level
# periodic consistency for the present parameter ranges.
PERIODIZATION_TERMS = 8


# =====================================================================
# Parameter ranges
# =====================================================================

K_MIN = 1
K_MAX = 8

AMPLITUDE_MIN = 0.5
AMPLITUDE_MAX = 2.0

WIDTH_MIN = 1.0
WIDTH_MAX = 4.0

PHASE_MIN = 0.0
PHASE_MAX = 2.0 * np.pi


# =====================================================================
# Exact periodic function and exact second derivative
# =====================================================================

def generate_function_with_derivative(x, rng):

    amplitude = rng.uniform(
        AMPLITUDE_MIN,
        AMPLITUDE_MAX
    )

    width = rng.uniform(
        WIDTH_MIN,
        WIDTH_MAX
    )

    center = rng.uniform(
        -0.25 * L,
        0.25 * L
    )

    k1 = rng.integers(
        K_MIN,
        K_MAX + 1
    )

    k2 = rng.integers(
        K_MIN,
        K_MAX + 1
    )

    phase0 = rng.uniform(
        PHASE_MIN,
        PHASE_MAX
    )

    # --------------------------------------------------------------
    # Periodized Gaussian envelope
    #
    # E_p(x) = sum_m exp(-((x-center-mL)/width)^2)
    #
    # This is periodic:
    #
    # E_p(x+L) = E_p(x)
    # --------------------------------------------------------------

    m = np.arange(
        -PERIODIZATION_TERMS,
        PERIODIZATION_TERMS + 1
    )

    y = (
        x[:, None]
        - center
        - m[None, :] * L
    )

    E_terms = np.exp(
        -(y / width) ** 2
    )

    E = np.sum(
        E_terms,
        axis=1
    )

    # First derivative of periodized envelope

    E_x = np.sum(
        E_terms
        * (
            -2.0 * y / width**2
        ),
        axis=1
    )

    # Second derivative of periodized envelope

    E_xx = np.sum(
        E_terms
        * (
            -2.0 / width**2
            + 4.0 * y**2 / width**4
        ),
        axis=1
    )

    # --------------------------------------------------------------
    # Periodic phase
    #
    # omega1 = 2*pi*k1/L
    # omega2 = 2*pi*k2/L
    #
    # Hence:
    #
    # phi(x+L) = phi(x) + 2*pi*k1
    #
    # and exp(i*phi) is exactly periodic.
    # --------------------------------------------------------------

    omega1 = (
        2.0 * np.pi * k1 / L
    )

    omega2 = (
        2.0 * np.pi * k2 / L
    )

    phi = (
        omega1 * x
        + 0.25 * np.sin(
            omega2 * x
        )
        + phase0
    )

    phi_x = (
        omega1
        + 0.25
        * omega2
        * np.cos(
            omega2 * x
        )
    )

    phi_xx = (
        -0.25
        * omega2**2
        * np.sin(
            omega2 * x
        )
    )

    # --------------------------------------------------------------
    # Complex function
    # --------------------------------------------------------------

    phase_factor = np.exp(
        1j * phi
    )

    u = (
        amplitude
        * E
        * phase_factor
    )

    # --------------------------------------------------------------
    # Exact second derivative
    #
    # For:
    #
    # u = A E exp(i phi)
    #
    # we have:
    #
    # u_xx =
    # A exp(i phi)
    # [
    #     E_xx
    #     + 2 i E_x phi_x
    #     + i E phi_xx
    #     - E phi_x^2
    # ]
    #
    # This avoids division by E and is numerically safer.
    # --------------------------------------------------------------

    u_xx = (
        amplitude
        * phase_factor
        * (
            E_xx
            + 2.0j * E_x * phi_x
            + 1.0j * E * phi_xx
            - E * phi_x**2
        )
    )

    return u, u_xx


# =====================================================================
# Periodic second-order FD
# =====================================================================

def periodic_dxx(u, dx):

    return (
        np.roll(u, 1)
        - 2.0 * u
        + np.roll(u, -1)
    ) / dx**2


# =====================================================================
# Local stencil
# =====================================================================

def build_stencil_features(u):

    r = STENCIL_RADIUS

    real_features = []
    imag_features = []

    for shift in range(
        -r,
        r + 1
    ):

        real_features.append(
            np.roll(
                u.real,
                shift
            )
        )

        imag_features.append(
            np.roll(
                u.imag,
                shift
            )
        )

    X = np.column_stack(
        real_features
        + imag_features
    )

    return X


# =====================================================================
# Generate dataset
# =====================================================================

def generate_dataset():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    X_all = []
    Y_all = []
    N_all = []

    print("=" * 72)
    print("NN-FD PERIODIC DATASET GENERATION")
    print("=" * 72)

    print()
    print("L =", L)

    print(
        "Grid resolutions =",
        TRAIN_N_VALUES
    )

    print(
        "Samples per resolution =",
        SAMPLES_PER_N
    )

    print(
        "Stencil radius =",
        STENCIL_RADIUS
    )

    print(
        "Periodization terms =",
        PERIODIZATION_TERMS
    )

    print()

    # --------------------------------------------------------------
    # Loop over resolutions
    # --------------------------------------------------------------

    for N in TRAIN_N_VALUES:

        dx = L / N

        print(
            f"Generating N = {N:4d}, "
            f"dx = {dx:.6e}"
        )

        for sample in range(
            SAMPLES_PER_N
        ):

            # Periodic grid
            #
            # Endpoint x=L/2 is excluded, as standard for a periodic
            # grid. The point x=-L/2 is identified periodically with
            # x=L/2.
            x = (
                -L / 2.0
                + dx * np.arange(N)
            )

            # Exact periodic function
            u, u_xx_exact = (
                generate_function_with_derivative(
                    x,
                    rng
                )
            )

            # Parent FD
            u_xx_fd = periodic_dxx(
                u,
                dx
            )

            # ------------------------------------------------------
            # Learned correction
            #
            # exact - FD
            # ------------------------------------------------------

            correction = (
                u_xx_exact
                - u_xx_fd
            )

            # Local input
            X = build_stencil_features(
                u
            )

            # Complex correction -> two real outputs
            Y = np.column_stack(
                (
                    correction.real,
                    correction.imag
                )
            )

            X_all.append(
                X
            )

            Y_all.append(
                Y
            )

            N_all.append(
                np.full(
                    N,
                    N,
                    dtype=np.int32
                )
            )

    # --------------------------------------------------------------
    # Combine
    # --------------------------------------------------------------

    X_all = np.vstack(
        X_all
    )

    Y_all = np.vstack(
        Y_all
    )

    N_all = np.concatenate(
        N_all
    )

    print()
    print("Dataset generation completed.")
    print()
    print("X shape:", X_all.shape)
    print("Y shape:", Y_all.shape)
    print("N metadata shape:", N_all.shape)

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    print()
    print("X min/max:")
    print(
        np.min(X_all),
        np.max(X_all)
    )

    print()
    print("Y min/max:")
    print(
        np.min(Y_all),
        np.max(Y_all)
    )

    print()
    print("Y mean:")
    print(
        np.mean(
            Y_all,
            axis=0
        )
    )

    print()
    print("Y std:")
    print(
        np.std(
            Y_all,
            axis=0
        )
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "nn_fd_training_data.npz"
    )

    np.savez_compressed(
        output_file,
        X=X_all,
        Y=Y_all,
        N=N_all
    )

    print()
    print("Saved:")
    print(output_file)

    print()
    print("=" * 72)

    return X_all, Y_all


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    generate_dataset()