"""
=========================================================

Spatial convergence test for IMEX2-FD NLS solver


Study:

L2 error versus spatial resolution


Expected:

Second-order convergence

=========================================================
"""


import numpy as np
import matplotlib.pyplot as plt


from imex2_solver import IMEX2Solver


from exact_solution import (
    NLSParameters,
    soliton,
    l2_error
)



# =========================================================
# Single simulation
# =========================================================


def compute_error(N):


    params = NLSParameters(

        alpha=1.0,

        beta=0.0,

        L=20.0

    )



    solver = IMEX2Solver(

        L=params.L,

        N=N,

        dt=0.0005

    )



    x = solver.grid.x



    u0 = soliton(

        x,

        0.0,

        params

    )



    t,u = solver.solve(

        u0,

        T=2.0

    )



    u_num = u[-1]



    u_exact = soliton(

        x,

        2.0,

        params

    )



    error = l2_error(

        u_num,

        u_exact

    )


    return error



# =========================================================
# Main
# =========================================================


def main():


    N_values = [

        100,

        200,

        400,

        800

    ]



    errors=[]



    print("="*70)

    print(
        "IMEX2 Spatial Convergence Study"
    )

    print("="*70)


    previous_error=None



    for N in N_values:


        error = compute_error(N)


        errors.append(error)



        if previous_error is None:

            rate="-"

        else:

            rate=np.log2(
                previous_error/error
            )



        print(

            f"N={N:<6}"
            f"L2 error={error:<15.6e}"
            f"Rate={rate}"

        )


        previous_error=error



    print("="*70)



    # =====================================================
    # Plot
    # =====================================================


    plt.figure(

        figsize=(7,5)

    )



    dx_values = [

        20/N

        for N in N_values

    ]



    plt.loglog(

        dx_values,

        errors,

        "o-",

        linewidth=2,

        label="IMEX2-FD"

    )



    # reference second order

    ref = errors[-1] * (

        np.array(dx_values)

        /

        dx_values[-1]

    )**2



    plt.loglog(

        dx_values,

        ref,

        "--",

        linewidth=2,

        label="Slope 2"

    )



    plt.xlabel(

        r"$\Delta x$"

    )


    plt.ylabel(

        r"$L_2$ error"

    )


    plt.title(

        "Spatial convergence of IMEX2-FD solver"

    )


    plt.grid(

        True,

        which="both"

    )


    plt.legend()


    plt.tight_layout()



    plt.savefig(

        "results/Figure_4_Spatial_convergence.png",

        dpi=300

    )



    plt.show()



# =========================================================
# Run
# =========================================================


if __name__=="__main__":

    main()