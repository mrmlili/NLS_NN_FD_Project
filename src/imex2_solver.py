"""
=========================================================

IMEX2-CNAB Solver for 1D cubic focusing NLS


Equation:

    i*u_t + u_xx + |u|^2*u = 0


Time discretization:

    Crank-Nicolson
    Adams-Bashforth 2


Space:

    Second-order periodic finite difference


=========================================================
"""


import numpy as np

from scipy.linalg import lu_factor, lu_solve

from fd_operator import (
    PeriodicGrid,
    FiniteDifferenceOperator
)



class IMEX2Solver:


    def __init__(
        self,
        L,
        N,
        dt
    ):


        self.grid = PeriodicGrid(
            L,
            N
        )


        self.fd = FiniteDifferenceOperator(
            self.grid
        )


        self.dt = dt

        self.N = N



        self.I = np.eye(
            N,
            dtype=complex
        )



        # Linear operator

        self.L_operator = (
            1j *
            self.fd.Dxx_matrix
        )



        # CN matrices

        self.A = (
            self.I
            -
            0.5*dt*self.L_operator
        )


        self.B = (
            self.I
            +
            0.5*dt*self.L_operator
        )

        # LU factorization (performed once)
        self.A_lu = lu_factor(self.A)



    # ==================================================
    # Nonlinear operator
    # ==================================================

    def nonlinear_term(self,u):

        return (
            1j *
            np.abs(u)**2 *
            u
        )



    # ==================================================
    # First step
    # ==================================================

    def first_step(self,u):

        """
        First step using Euler explicit
        to generate u^1.
        """

        rhs = (
            self.B @ u
            +
            self.dt *
            self.nonlinear_term(u)
        )


        return lu_solve(
              self.A_lu,
            rhs
        )



    # ==================================================
    # AB2-CN step
    # ==================================================

    def step(
        self,
        u,
        u_old
    ):


        N_current = (
            self.nonlinear_term(u)
        )


        N_old = (
            self.nonlinear_term(u_old)
        )



        nonlinear_AB2 = (

            1.5*N_current
            -
            0.5*N_old

        )



        rhs = (

            self.B @ u
            +
            self.dt *
            nonlinear_AB2

        )



        u_new = lu_solve(
            self.A_lu,
            rhs
        )


        return u_new



    # ==================================================
    # Full solver
    # ==================================================

    def solve(
        self,
        u0,
        T
    ):


        steps=int(
            T/self.dt
        )


        u_old = u0.copy()


        u = self.first_step(
            u0
        )



        solution=[

            u_old.copy(),

            u.copy()

        ]



        times=[

            0.0,

            self.dt

        ]



        for n in range(1,steps):


            u_new = self.step(
                u,
                u_old
            )


            u_old = u

            u = u_new



            solution.append(
                u.copy()
            )


            times.append(
                (n+1)*self.dt
            )



        return (

            np.array(times),

            np.array(solution)

        )

    # ==================================================
    # Final solution only (memory efficient)
    # ==================================================

    def solve_final(self, u0, T):
        """
        Compute only the final solution.

        Unlike solve(), this function does not store the
        entire solution history, making it suitable for
        generating high-accuracy reference solutions.
        """

        steps = int(T / self.dt)

        u_old = u0.copy()

        u = self.first_step(u0)

        for _ in range(1, steps):

            u_new = self.step(
                u,
                u_old
            )

            u_old = u
            u = u_new

        return u



# ======================================================
# Test
# ======================================================


if __name__=="__main__":


    from exact_solution import (
        NLSParameters,
        soliton
    )


    params=NLSParameters(
        alpha=1.0,
        beta=0.0,
        L=20.0
    )


    solver=IMEX2Solver(

        L=params.L,

        N=200,

        dt=0.001

    )



    x=solver.grid.x



    u0=soliton(
        x,
        0.0,
        params
    )



    t,u=solver.solve(
        u0,
        T=1.0
    )



    print(
        "IMEX2 simulation finished"
    )


    print(
        "Number of time steps:",
        len(t)
    )
    