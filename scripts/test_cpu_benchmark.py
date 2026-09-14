"""
=========================================================

CPU benchmark for IMEX2-CNAB solver

=========================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from imex2_solver import IMEX2Solver
from exact_solution import (
    NLSParameters,
    soliton
)


def main():

    params = NLSParameters(
        alpha=1.0,
        beta=0.0,
        L=20.0
    )

    grid_sizes = [
        100,
        200,
        400,
        800
    ]

    cpu_times = []

    print("="*70)
    print("CPU Benchmark")
    print("="*70)

    for N in grid_sizes:

        solver = IMEX2Solver(
            L=params.L,
            N=N,
            dt=0.001
        )

        x = solver.grid.x

        u0 = soliton(
            x,
            0.0,
            params
        )

        start = time.perf_counter()

        solver.solve_final(
            u0,
            T=2.0
        )

        elapsed = time.perf_counter() - start

        cpu_times.append(elapsed)

        print(
            f"N={N:<5d}   CPU Time = {elapsed:.4f} s"
        )

    os.makedirs(
        "results",
        exist_ok=True
    )

    plt.figure(figsize=(7,4))

    plt.plot(
        grid_sizes,
        cpu_times,
        "o-",
        linewidth=2,
        markersize=7
    )

    plt.xlabel("Grid size N")
    plt.ylabel("CPU Time (s)")
    plt.title("CPU Benchmark of IMEX2-CNAB")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/Figure_6_CPU_Benchmark.png",
        dpi=300
    )

    plt.savefig(
        "results/Figure_6_CPU_Benchmark.pdf"
    )

    plt.show()


if __name__ == "__main__":

    main()