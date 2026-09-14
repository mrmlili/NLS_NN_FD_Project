"""
======================================================================
STEP #34 — PHYSICS-AWARE DATASET AUGMENTATION
======================================================================

Adds soliton-like NLS states to the existing NN-FD dataset.

Original:
    200 samples per N

Augmentation:
    50 soliton-like samples per N

Final:
    250 samples per N

The original dataset is NOT overwritten.

Target:
    C = u_xx_exact - Dxx_FD(u)

Stored:
    X
    Y
    N

======================================================================
"""

import os
import numpy as np


# ======================================================================
# Configuration
# ======================================================================

L = 20.0

RESOLUTIONS = [32, 64, 128, 256]

SOLITON_SAMPLES_PER_N = 50

SEED = 20260831

INPUT_FILE = (
    "data/nn_fd/"
    "nn_fd_training_data.npz"
)

OUTPUT_FILE = (
    "data/nn_fd/"
    "nn_fd_training_data_ra_augmented.npz"
)


# ======================================================================
# Soliton-like parameters
# ======================================================================

AMPLITUDE_MIN = 0.8
AMPLITUDE_MAX = 2.0

WIDTH_MIN = 0.8
WIDTH_MAX = 3.0

CENTER_MIN = -0.25 * L
CENTER_MAX = 0.25 * L

K_MIN = 0.0
K_MAX = 4.0

PHASE_MIN = 0.0
PHASE_MAX = 2.0 * np.pi


# ======================================================================
# Parent FD
# ======================================================================

def periodic_dxx(u, dx):

    return (
        np.roll(u, 1)
        - 2.0 * u
        + np.roll(u, -1)
    ) / dx**2


# ======================================================================
# Stencil
# ======================================================================

def build_stencil_features(u):

    real_features = []
    imag_features = []

    for shift in range(-2, 3):

        real_features.append(
            np.roll(
                u.real,
                shift,
            )
        )

        imag_features.append(
            np.roll(
                u.imag,
                shift,
            )
        )

    return np.column_stack(
        real_features + imag_features
    )


# ======================================================================
# Soliton-like function
# ======================================================================

def generate_soliton_like(
    x,
    rng,
):

    amplitude = rng.uniform(
        AMPLITUDE_MIN,
        AMPLITUDE_MAX,
    )

    width = rng.uniform(
        WIDTH_MIN,
        WIDTH_MAX,
    )

    center = rng.uniform(
        CENTER_MIN,
        CENTER_MAX,
    )

    k = rng.uniform(
        K_MIN,
        K_MAX,
    )

    phase = rng.uniform(
        PHASE_MIN,
        PHASE_MAX,
    )

    y = x - center
    z = y / width

    sech = (
        1.0
        / np.cosh(z)
    )

    tanh = np.tanh(z)

    envelope = (
        amplitude
        * sech
    )

    q_x_over_q = (
        -tanh / width
    )

    q_xx_over_q = (
        2.0 * tanh**2
        - 1.0
    ) / width**2

    phi = (
        k * x
        + phase
    )

    u = (
        envelope
        * np.exp(1j * phi)
    )

    u_xx = u * (
        q_xx_over_q
        + 2j
        * q_x_over_q
        * k
        - k**2
    )

    return u, u_xx


# ======================================================================
# Main
# ======================================================================

print("=" * 78)
print(
    "STEP #34 — PHYSICS-AWARE DATASET AUGMENTATION"
)
print("=" * 78)

print()
print(
    "Original dataset:",
    INPUT_FILE,
)

print(
    "Output dataset:",
    OUTPUT_FILE,
)

print(
    "Soliton samples per N:",
    SOLITON_SAMPLES_PER_N,
)


# ======================================================================
# Load original dataset
# ======================================================================

data = np.load(
    INPUT_FILE
)

X_original = data[
    "X"
].astype(
    np.float32
)

Y_original = data[
    "Y"
].astype(
    np.float32
)

N_original = data[
    "N"
].astype(
    np.int32
)

print()
print(
    "Original shapes:"
)

print(
    "X:",
    X_original.shape,
)

print(
    "Y:",
    Y_original.shape,
)

print(
    "N:",
    N_original.shape,
)


# ======================================================================
# Generate soliton augmentation
# ======================================================================

rng = np.random.default_rng(
    SEED
)

X_augmented = []
Y_augmented = []
N_augmented = []


for N in RESOLUTIONS:

    dx = L / N

    print()
    print(
        f"Generating soliton augmentation "
        f"N={N}, dx={dx:.6e}"
    )

    for _ in range(
        SOLITON_SAMPLES_PER_N
    ):

        x = (
            -L / 2.0
            + dx
            * np.arange(N)
        )

        u, u_xx_exact = (
            generate_soliton_like(
                x,
                rng,
            )
        )

        u_xx_fd = periodic_dxx(
            u,
            dx,
        )

        correction = (
            u_xx_exact
            - u_xx_fd
        )

        X = build_stencil_features(
            u
        )

        Y = np.column_stack(
            (
                correction.real,
                correction.imag,
            )
        )

        X_augmented.append(
            X.astype(
                np.float32
            )
        )

        Y_augmented.append(
            Y.astype(
                np.float32
            )
        )

        N_augmented.append(
            np.full(
                N,
                N,
                dtype=np.int32,
            )
        )


# ======================================================================
# Combine
# ======================================================================

X_augmented = np.vstack(
    X_augmented
)

Y_augmented = np.vstack(
    Y_augmented
)

N_augmented = np.concatenate(
    N_augmented
)


print()
print(
    "Augmentation shapes:"
)

print(
    "X:",
    X_augmented.shape,
)

print(
    "Y:",
    Y_augmented.shape,
)

print(
    "N:",
    N_augmented.shape,
)


# ======================================================================
# Combine original + augmentation
# ======================================================================

X_final = np.vstack(
    (
        X_original,
        X_augmented,
    )
)

Y_final = np.vstack(
    (
        Y_original,
        Y_augmented,
    )
)

N_final = np.concatenate(
    (
        N_original,
        N_augmented,
    )
)


# ======================================================================
# Shuffle complete dataset
# ======================================================================

shuffle_rng = np.random.default_rng(
    SEED + 1
)

indices = shuffle_rng.permutation(
    len(X_final)
)

X_final = X_final[
    indices
]

Y_final = Y_final[
    indices
]

N_final = N_final[
    indices
]


# ======================================================================
# Statistics
# ======================================================================

print()
print(
    "Final augmented dataset:"
)

print(
    "X shape:",
    X_final.shape,
)

print(
    "Y shape:",
    Y_final.shape,
)

print(
    "N shape:",
    N_final.shape,
)

print()
print(
    "X min/max:",
    np.min(X_final),
    np.max(X_final),
)

print()
print(
    "Y mean:",
    np.mean(
        Y_final,
        axis=0,
    ),
)

print()
print(
    "Y std:",
    np.std(
        Y_final,
        axis=0,
    ),
)


# ======================================================================
# Save
# ======================================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True,
)

np.savez_compressed(
    OUTPUT_FILE,

    X=X_final,
    Y=Y_final,
    N=N_final,
)


print()
print("=" * 78)
print(
    "STEP #34 COMPLETED"
)
print("=" * 78)

print()
print(
    "Saved:"
)

print(
    OUTPUT_FILE
)