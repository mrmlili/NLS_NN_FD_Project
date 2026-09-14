"""
=========================================================

Mass conservation test for IMEX2-CNAB NLS solver

=========================================================
"""


import numpy as np
import matplotlib.pyplot as plt


from imex2_solver import IMEX2Solver


from exact_solution import (
    NLSParameters,
    soliton
)



# =========================================================
# Mass computation
# =========================================================

def compute_mass(u, x):

    return np.trapezoid(
        np.abs(u)**2,
        x
    )



# =========================================================
# Main test
# =========================================================

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



    # Initial exact soliton

    u0 = soliton(
        x,
        0.0,
        params
    )



    # Numerical solution

    t, u = solver.solve(
        u0,
        T=2.0
    )



    # Compute mass history

    masses = []


    for ui in u:

        masses.append(
            compute_mass(
                ui,
                x
            )
        )


    masses = np.array(masses)



    # Relative mass error

    M0 = masses[0]


    error = np.abs(
        masses - M0
    ) / M0



    # =====================================================
    # Print results
    # =====================================================

    print("="*60)

    print(
        "IMEX2 Mass Conservation Test"
    )

    print()


    print(
        "Initial Mass:",
        M0
    )


    print(
        "Final Mass:",
        masses[-1]
    )


    print(
        "Maximum relative mass error:",
        np.max(error)
    )


    print()


    print(
        "Minimum error:",
        np.min(error)
    )


    print(
        "Maximum error:",
        np.max(error)
    )


    print("="*60)



    # =====================================================
    # Plot
    # =====================================================

    plt.figure(
        figsize=(7,4)
    )


    # Remove t=0 because error=0
    # and logarithmic scale cannot plot zero

    plt.semilogy(
        t[1:],
        error[1:],
        linewidth=2
    )


    plt.xlabel(
        "Time"
    )


    plt.ylabel(
        "Relative Mass Error"
    )


    plt.title(
        "Mass conservation error of IMEX2-CNAB"
    )


    plt.grid(
        True,
        which="both"
    )


    plt.ylim(
        1e-10,
        1e-4
    )


    plt.tight_layout()



    plt.savefig(
        "results/Figure_2_IMEX2_mass_error.png",
        dpi=300
    )



    plt.show()



# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    main()