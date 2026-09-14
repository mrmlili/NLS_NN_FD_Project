"""
=========================================================

IMEX Solver for 1D cubic focusing NLS equation


Equation:

    i*u_t + u_xx + |u|^2 u = 0


Time discretization:

    IMEX Crank-Nicolson / Explicit Nonlinear


Spatial:

    Second-order periodic finite difference


Project:

    NN-FF-FD for Nonlinear Schrödinger Equation


=========================================================
"""


import numpy as np

from fd_operator import (
    PeriodicGrid,
    FiniteDifferenceOperator
)



# =========================================================
# IMEX Solver Class
# =========================================================


class IMEXSolver:


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



        # identity matrix

        self.I = np.eye(
            N,
            dtype=complex
        )



        # Linear operator

        self.L_operator = (
            1j *
            self.fd.Dxx_matrix
        )



        # IMEX matrices

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



    # =====================================================
    # Nonlinear term
    # =====================================================


    def nonlinear_term(
        self,
        u
    ):


        return (
            1j *
            np.abs(u)**2 *
            u
        )



    # =====================================================
    # One time step
    # =====================================================


    def step(
        self,
        u
    ):

        rhs = (
            self.B @ u
            +
            self.dt *
            self.nonlinear_term(u)
        )


        u_new = np.linalg.solve(
            self.A,
            rhs
        )


        return u_new
        # =====================================================
    # Full time integration
    # =====================================================

    def solve(
        self,
        u0,
        T
    ):

        """
        Time integration of NLS equation.

        Parameters
        ----------
        u0 :
            Initial condition

        T :
            Final time


        Returns
        -------

        times :
            time array

        solution :
            numerical solution history

        """

        steps = int(
            T / self.dt
        )


        u = u0.copy()


        solution = [
            u.copy()
        ]


        times = [
            0.0
        ]


        for n in range(steps):


            u = self.step(u)


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
    
    # =========================================================
# Test Run
# =========================================================


if __name__=="__main__":


    from exact_solution import (
        NLSParameters,
        soliton
    )


    params = NLSParameters(
        alpha=1.0,
        beta=0.0,
        L=20.0
    )


    solver = IMEXSolver(
        L=params.L,
        N=200,
        dt=0.001
    )


    x = solver.grid.x


    u0 = soliton(
        x,
        0.0,
        params
    )


    t, u = solver.solve(
        u0,
        T=1.0
    )


    print("Simulation finished")


    print(
        "Number of time steps:",
        len(t)
    )