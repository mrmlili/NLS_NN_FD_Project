"""
=========================================================

Plot FD spatial convergence study

Figure 1:
Spatial discretization convergence of
second-order finite difference operators


Project:
NN-FF-FD for Nonlinear Schrödinger Equation

=========================================================
"""


import numpy as np
import matplotlib.pyplot as plt


from fd_operator import (
    PeriodicGrid,
    FiniteDifferenceOperator
)



def compute_errors():

    L = 20.0


    grid_sizes = [
        50,
        100,
        200,
        400,
        800
    ]


    dx_values = []

    errors_dx = []

    errors_dxx = []



    for N in grid_sizes:


        grid = PeriodicGrid(
            L,
            N
        )


        fd = FiniteDifferenceOperator(
            grid
        )


        k = 2*np.pi/L


        u = np.sin(
            k*grid.x
        )


        ux_exact = (
            k*np.cos(k*grid.x)
        )


        uxx_exact = (
            -k*k*np.sin(k*grid.x)
        )


        ux_num = fd.first_derivative(
            u
        )


        uxx_num = fd.second_derivative(
            u
        )


        dx_values.append(
            grid.dx
        )


        errors_dx.append(
            np.linalg.norm(
                ux_num-ux_exact
            )/np.sqrt(N)
        )


        errors_dxx.append(
            np.linalg.norm(
                uxx_num-uxx_exact
            )/np.sqrt(N)
        )


    return (
        dx_values,
        errors_dx,
        errors_dxx
    )




def plot_convergence():


    dx, e1, e2 = compute_errors()



    plt.figure(
        figsize=(7,5)
    )



    plt.loglog(
        dx,
        e2,
        's-',
        linewidth=2,
        markersize=7,
         label=r"$D_{xx}^{FD}$"
    )


    # reference second order line

    ref = e2[0] * (
    np.array(dx)/dx[0]
)**2


    plt.loglog(
        dx,
        ref,
        '--',
        label=r"$O(\Delta x^2)$"
    )



    plt.xlabel(
        r"$\Delta x$"
    )


    plt.ylabel(
        r"$L_2$ Error"
    )


    plt.title(
        "Second-order spatial convergence of FD discretization"
    )


    plt.grid(
        True,
        which="both"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        "results/Figure_1_FD_spatial_convergence.png",
    dpi=300,
    bbox_inches="tight"
    )



    plt.savefig(
    "results/Figure_1_FD_spatial_convergence.pdf",
    bbox_inches="tight"
)
    plt.show()



if __name__=="__main__":

    plot_convergence()