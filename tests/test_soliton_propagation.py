"""
=========================================================

Soliton propagation validation for IMEX2-CNAB NLS solver

Comparison:

Numerical IMEX2 solution
        vs
Exact bright soliton solution


Outputs:

1. L2 error
2. Linf error
3. Soliton propagation figure


=========================================================
"""


import numpy as np
import matplotlib.pyplot as plt


from imex2_solver import IMEX2Solver


from exact_solution import (
    NLSParameters,
    soliton,
    l2_error,
    linf_error
)



# =========================================================
# Main validation
# =========================================================


def main():


    params = NLSParameters(

        alpha=1.0,

        beta=0.0,

        L=20.0

    )



    # -----------------------------
    # Numerical solver
    # -----------------------------

    solver = IMEX2Solver(

        L=params.L,

        N=200,

        dt=0.001

    )


    x = solver.grid.x



    # Initial condition

    u0 = soliton(

        x,

        0.0,

        params

    )



    # Time integration

    t, u = solver.solve(

        u0,

        T=2.0

    )



    # Final numerical solution

    u_num = u[-1]



    # Exact solution at final time

    u_exact = soliton(

        x,

        2.0,

        params

    )



    # Errors

    error_L2 = l2_error(

        u_num,

        u_exact

    )


    error_Linf = linf_error(

        u_num,

        u_exact

    )



    print("="*60)

    print(
        "Soliton Propagation Validation"
    )

    print()


    print(
        "Final time:",
        t[-1]
    )


    print(
        "L2 error:",
        error_L2
    )


    print(
        "Linf error:",
        error_Linf
    )


    print("="*60)



    # =====================================================
    # Figure
    # =====================================================


    plt.figure(

        figsize=(8,4)

    )


    plt.plot(

        x,

        np.abs(u_exact),

        "--",

        linewidth=2,

        label="Exact"

    )


    plt.plot(

        x,

        np.abs(u_num),

        linewidth=2,

        label="IMEX2-FD"

    )


    plt.xlabel(

        "x"

    )


    plt.ylabel(

        "|u(x,T)|"

    )


    plt.title(

        "Bright soliton propagation at T=2"

    )


    plt.legend()


    plt.grid(True)


    plt.tight_layout()



    plt.savefig(

        "results/Figure_3_Soliton_Propagation.png",

        dpi=300

    )



    plt.show()



# =========================================================
# Run
# =========================================================


if __name__ == "__main__":

    main()