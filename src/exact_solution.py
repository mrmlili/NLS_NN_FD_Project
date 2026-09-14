"""
=========================================================

Exact analytical solutions for the one-dimensional
cubic focusing Nonlinear Schrödinger Equation (NLS)

Equation

        i*u_t + u_xx + |u|^2*u = 0


This module provides:

1. Exact bright soliton solution
2. Analytical mass
3. Analytical energy
4. Numerical validation tools
5. Error measurement utilities


Project:
NN-FF-FD for Nonlinear Schrödinger Equation


=========================================================
"""


from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt



# =========================================================
# Problem Parameters
# =========================================================


@dataclass
class NLSParameters:
    """
    Parameters of the normalized focusing NLS equation.
    """

    alpha: float = 1.0      # eta: soliton parameter
    beta: float = 0.0       # xi: velocity parameter

    L: float = 20.0         # computational domain length
    T: float = 2.0          # final time



# =========================================================
# Mathematical utilities
# =========================================================


def sech(x):
    """
    Hyperbolic secant function.

        sech(x)=1/cosh(x)

    """

    return 1.0 / np.cosh(x)



# =========================================================
# Exact Bright Soliton
# =========================================================


def soliton(x, t, params):
    """
    Exact bright soliton solution of

        i*u_t + u_xx + |u|^2*u = 0


    Mathematical form:

    u(x,t)=sqrt(2)*eta*sech(eta(x-2*xi*t))
            exp(i(xi*x+(eta^2-xi^2)t))

    """

    eta = params.alpha
    xi = params.beta


    envelope = (
        np.sqrt(2)
        *
        eta
        *
        sech(
            eta*(x-2.0*xi*t)
        )
    )


    phase = np.exp(
        1j*
        (
            xi*x
            +
            (eta**2-xi**2)*t
        )
    )


    return envelope*phase



# =========================================================
# Exact Mass
# =========================================================


def exact_mass(params):
    """
    Analytical conserved mass.

    M = integral |u|^2 dx

    For the normalized soliton:

        M = 4 eta

    """

    return 4.0*params.alpha




# =========================================================
# Numerical Mass
# =========================================================


def numerical_mass(u, x):
    """
    Numerical mass using periodic summation.

    M = integral |u|^2 dx

    """

    dx = x[1]-x[0]

    return np.sum(np.abs(u)**2)*dx




# =========================================================
# Exact Energy
# =========================================================


def exact_energy(params):
    """
    Analytical Hamiltonian energy.

    E =
    integral
    ( |u_x|^2 - 0.5|u|^4 ) dx


    For the exact soliton:

        E = -4/3 eta^3

    """

    return -(4.0/3.0)*(params.alpha**3)




# =========================================================
# Numerical Energy
# =========================================================


def numerical_energy(u, x):
    """
    Hamiltonian energy using Fourier spectral derivative.

    """

    N = len(x)

    dx = x[1]-x[0]


    k = (
        2*np.pi*
        np.fft.fftfreq(
            N,
            d=dx
        )
    )


    u_hat = np.fft.fft(u)


    ux = np.fft.ifft(
        1j*k*u_hat
    )


    kinetic = np.abs(ux)**2

    potential = -0.5*np.abs(u)**4


    return np.sum(
        kinetic+potential
    )*dx




# =========================================================
# Error Metrics
# =========================================================


def l2_error(u_num, u_exact):
    """
    Relative L2 error.
    """

    return (
        np.linalg.norm(u_num-u_exact)
        /
        np.linalg.norm(u_exact)
    )



def linf_error(u_num, u_exact):
    """
    Relative Linf error.
    """

    return (
        np.max(np.abs(u_num-u_exact))
        /
        np.max(np.abs(u_exact))
    )




# =========================================================
# Visualization
# =========================================================


def plot_solution(params):

    x = np.linspace(
        -params.L/2,
        params.L/2,
        600,
        endpoint=False
    )


    u = soliton(
        x,
        0.0,
        params
    )


    plt.figure(figsize=(8,4))


    plt.plot(
        x,
        np.abs(u),
        linewidth=2
    )


    plt.xlabel("x")

    plt.ylabel(r"$|u(x,0)|$")

    plt.title(
        "Exact Bright Soliton of NLS"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()




# =========================================================
# Main Test
# =========================================================


def main():


    params = NLSParameters()


    x = np.linspace(
        -params.L/2,
        params.L/2,
        600,
        endpoint=False
    )


    u = soliton(
        x,
        0.0,
        params
    )


    print("="*60)


    print(
        "Exact Mass      :",
        exact_mass(params)
    )


    print(
        "Numerical Mass  :",
        numerical_mass(u,x)
    )


    print()


    print(
        "Exact Energy    :",
        exact_energy(params)
    )


    print(
        "Numerical Energy:",
        numerical_energy(u,x)
    )


    print("="*60)


    plot_solution(params)




# =========================================================
# Run
# =========================================================


if __name__ == "__main__":

    main()