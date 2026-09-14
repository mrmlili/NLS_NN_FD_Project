import numpy as np
import os


folder = "temporal_self_convergence_results"


# =========================================================
# Load saved solutions
# =========================================================

file_old = os.path.join(
    folder,
    "solution_dt_2.50e-04.npz"
)

file_new = os.path.join(
    folder,
    "solution_dt_1.25e-04.npz"
)


old = np.load(file_old)
new = np.load(file_new)


u_old = old["u"]
u_new = new["u"]


# =========================================================
# Compute difference
# =========================================================

difference = (
    np.linalg.norm(u_old - u_new)
    /
    np.linalg.norm(u_new)
)


# Previous difference from the last refinement
previous_difference = 9.5062442277e-7


rate = np.log2(
    previous_difference / difference
)


# =========================================================
# Print
# =========================================================

print()
print("=" * 70)
print("TEMPORAL SELF-CONVERGENCE — POST PROCESSING")
print("=" * 70)

print()

print(
    "dt       = 2.50e-04"
)

print(
    "dt/2     = 1.25e-04"
)

print()

print(
    f"Difference = {difference:.12e}"
)

print(
    f"Rate       = {rate:.12f}"
)

print()

print("=" * 70)