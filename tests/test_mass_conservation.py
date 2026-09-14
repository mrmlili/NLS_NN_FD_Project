"""
=========================================================

Mass conservation test for IMEX NLS solver


Invariant:

M(t)= integral |u|^2 dx


=========================================================
"""


import numpy as np
import matplotlib.pyplot as plt


from imex_solver import IMEXSolver

from exact_solution import (
    NLSParameters,
    soliton
)



# ======================================================
# Mass function
# ======================================================


def compute_mass(u, x):

    return np.trapezoid(
        np.abs(u)**2,
        x
    )



# ======================================================
# Main test
# ======================================================


def main():


    params = NLSParameters(
        alpha=1.0,
        beta=0.0,
        L=20.0
    )


    solver = IMEXSolver(

        L=params.L,

        N=200,

        dt=0.0002

    )


    x = solver.grid.x



    # Initial exact soliton

    u0 = soliton(
        x,
        0.0,
        params
    )



    times, solution = solver.solve(
        u0,
        T=2.0
    )



    masses=[]



    for u in solution:

        masses.append(
            compute_mass(
                u,
                x
            )
        )



    masses=np.array(masses)



    M0=masses[0]



    relative_error = np.abs(
        masses-M0
    )/M0



    print("="*60)

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
        np.max(relative_error)
    )


    print("="*60)



    # Plot mass evolution


    plt.figure(
        figsize=(7,4)
    )


    plt.plot(
        times,
        masses,
        linewidth=2
    )


    plt.xlabel(
        "Time"
    )


    plt.ylabel(
        "Mass M(t)"
    )


    plt.title(
        "Mass conservation of IMEX NLS solver"
    )


    plt.grid(True)


    plt.tight_layout()



    plt.savefig(
        "results/Figure_2_Mass_conservation.png",
        dpi=300
    )


    plt.show()



if __name__=="__main__":

    main()