"""
=========================================================

Temporal convergence test for IMEX2-CNAB NLS solver


Measure:

L2 error versus exact soliton

Expected:

Second-order temporal convergence


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


def compute_error(dt):


    params = NLSParameters(

        alpha=1.0,

        beta=0.0,

        L=20.0

    )


    solver = IMEX2Solver(

        L=params.L,

        N=800,

        dt=dt

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


    dt_values = [

        0.004,

        0.002,

        0.001,

        0.0005

    ]



    errors = []



    print("="*60)

    print(
        "IMEX2 Temporal Convergence Study"
    )

    print("="*60)



    previous_error = None



    for dt in dt_values:


        error = compute_error(dt)


        errors.append(error)



        if previous_error is None:

            rate = "-"

        else:

            rate = np.log2(
                previous_error/error
            )



        print(

            f"dt={dt:<10}"
            f"L2 error={error:<15.6e}"
            f"Rate={rate}"

        )


        previous_error = error



    print("="*60)



    # =====================================================
    # Plot convergence
    # =====================================================


    plt.figure(

        figsize=(7,5)

    )


    plt.loglog(

        dt_values,

        errors,

        "o-",

        linewidth=2,

        label="IMEX2-FD"

    )


    # reference slope 2

    ref = errors[-1] * (

        np.array(dt_values)

        /

        dt_values[-1]

    )**2



    plt.loglog(

        dt_values,

        ref,

        "--",

        linewidth=2,

        label="Slope 2"

    )



    plt.xlabel(

        r"$\Delta t$"

    )


    plt.ylabel(

        r"$L_2$ error"

    )


    plt.title(

        "Temporal convergence of IMEX2-CNAB"

    )


    plt.grid(True, which="both")


    plt.legend()


    plt.tight_layout()



    plt.savefig(

        "results/Figure_4_Time_convergence.png",

        dpi=300

    )


    plt.show()



# =========================================================
# Run
# =========================================================


if __name__ == "__main__":

    main()