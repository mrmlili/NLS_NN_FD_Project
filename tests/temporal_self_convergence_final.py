import os
import time
import numpy as np

from imex2_solver import IMEX2Solver
from exact_solution import NLSParameters, soliton


# =========================================================
# Parameters
# =========================================================

params = NLSParameters(
    alpha=1.0,
    beta=0.0,
    L=20.0
)

N = 3200
T = 1.0

dt_old = 2.5e-4
dt_new = 1.25e-4

folder = "temporal_self_convergence_results"


# =========================================================
# Load previously saved solution
# =========================================================

old_file = os.path.join(
    folder,
    "solution_dt_2.50e-04.npz"
)

data = np.load(old_file)

x_old = data["x"]
u_old = data["u"]


print("=" * 70)
print("TEMPORAL SELF-CONVERGENCE FINAL REFINEMENT")
print("=" * 70)

print("N =", N)
print("T =", T)
print("Old dt =", dt_old)
print("New dt =", dt_new)

print()
print("Loaded:")
print(old_file)


# =========================================================
# New computation
# =========================================================

solver = IMEX2Solver(
    L=params.L,
    N=N,
    dt=dt_new
)

x = solver.grid.x

u0 = soliton(
    x,
    0.0,
    params
)


print()
print(
    f"Running dt={dt_new:.2e} ..."
)

start = time.perf_counter()

u_new = solver.solve_final(
    u0,
    T
)

cpu = time.perf_counter() - start


print(
    f"Finished. CPU = {cpu:.3f} s"
)


# =========================================================
# Save new solution
# =========================================================

new_file = os.path.join(
    folder,
    "solution_dt_1.25e-04.npz"
)

np.savez(
    new_file,
    x=x,
    u=u_new,
    dt=dt_new,
    N=N,
    T=T
)


print(
    "Saved:",
    new_file
)


# =========================================================
# Self-convergence difference
# =========================================================

difference = (
    np.linalg.norm(
        u_old - u_new
    )
    /
    np.linalg.norm(u_new)
)


previous_difference = 9.5062442277e-7


rate = np.log2(
    previous_difference
    /
    difference
)


# =========================================================
# Print result
# =========================================================

print()
print("=" * 70)
print("FINAL SELF-CONVERGENCE RESULT")
print("=" * 70)

print()

print(
    f"dt              = {dt_old:.2e}"
)

print(
    f"dt/2            = {dt_new:.2e}"
)

print(
    f"Difference       = {difference:.10e}"
)

print(
    f"Rate             = {rate:.10f}"
)

print(
    f"CPU              = {cpu:.3f} s"
)

print("=" * 70)


# =========================================================
# Save summary
# =========================================================

np.savez(
    os.path.join(
        folder,
        "final_self_convergence_result.npz"
    ),
    dt_old=dt_old,
    dt_new=dt_new,
    difference=difference,
    rate=rate,
    cpu=cpu,
    N=N,
    T=T
)

print()
print("Final self-convergence result saved.")