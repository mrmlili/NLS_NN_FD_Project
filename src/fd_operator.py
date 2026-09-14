"""
=========================================================

Finite Difference Operators for 1D Nonlinear
Schrodinger Equation

Equation:

    i*u_t + u_xx + |u|^2*u = 0


This module provides:

1. Periodic grid
2. First derivative FD operator
3. Second derivative FD operator
4. Validation utilities


Project:
NN-FF-FD for NLS


=========================================================
"""


import numpy as np



# =========================================================
# Grid
# =========================================================


class PeriodicGrid:
    """
    Uniform periodic spatial grid.
    """


    def __init__(self, L, N):

        self.L = L

        self.N = N

        self.dx = L/N


        self.x = np.linspace(
            -L/2,
            L/2,
            N,
            endpoint=False
        )



# =========================================================
# Finite Difference Operators
# =========================================================


class FiniteDifferenceOperator:
    """
    Second-order periodic finite difference operators.
    """


    def __init__(self, grid):

        self.grid = grid

        self.dx = grid.dx

        self.N = grid.N

        self.Dx_matrix = self.build_first_derivative_matrix()

        self.Dxx_matrix = self.build_second_derivative_matrix()


    # =====================================================
    # First derivative matrix
    # =====================================================

    def build_first_derivative_matrix(self):

        D = np.zeros(
            (self.N, self.N),
            dtype=complex
        )


        c = 1.0/(2.0*self.dx)


        for i in range(self.N):

            D[i,(i+1)%self.N] = c

            D[i,(i-1)%self.N] = -c


        return D



    # =====================================================
    # Second derivative matrix
    # =====================================================

    def build_second_derivative_matrix(self):

        D2 = np.zeros(
            (self.N,self.N),
            dtype=complex
        )


        c = 1.0/(self.dx**2)


        for i in range(self.N):

            D2[i,i] = -2.0*c

            D2[i,(i+1)%self.N] = c

            D2[i,(i-1)%self.N] = c


        return D2



    # -----------------------------------------------------
    # First derivative
    # -----------------------------------------------------

    def first_derivative(self, u):
        """
        Central difference approximation:

        u_x =
        (u_{j+1}-u_{j-1})/(2dx)

        """


        return (
            np.roll(u,-1)
            -
            np.roll(u,1)
        )/(2*self.dx)




    # -----------------------------------------------------
    # Second derivative
    # -----------------------------------------------------

    def second_derivative(self,u):
        """
        Laplacian in 1D:

        u_xx =
        (u_{j+1}-2u_j+u_{j-1})/dx^2

        """


        return (
            np.roll(u,-1)
            -
            2*u
            +
            np.roll(u,1)
        )/(self.dx**2)



# =========================================================
# Verification
# =========================================================


def test_operator():

    L = 20.0

    N = 200

    grid = PeriodicGrid(
        L,
        N
    )

    fd = FiniteDifferenceOperator(
        grid
    )


    # Periodic test function

    k = 2*np.pi/grid.L


    u = np.sin(
        k*grid.x
    )


    ux_exact = (
        k*np.cos(k*grid.x)
    )


    uxx_exact = (
        -(k**2)*np.sin(k*grid.x)
    )


    ux_num = fd.first_derivative(
        u
    )


    uxx_num = fd.second_derivative(
        u
    )


    error1 = np.linalg.norm(
        ux_num-ux_exact
    )


    error2 = np.linalg.norm(
        uxx_num-uxx_exact
    )


    print("="*50)

    print(
        "First derivative error:",
        error1
    )


    print(
        "Second derivative error:",
        error2
    )


    print("="*50)

# =========================================================
# Convergence Study
# =========================================================


def convergence_test():

    """
    Verify second-order accuracy of FD operators.

    Test function:

        u(x)=sin(2*pi*x/L)

    Expected:

        Error ~ dx^2

    """


    L = 20.0


    grid_sizes = [
        50,
        100,
        200,
        400,
        800
    ]


    errors_first = []

    errors_second = []

    dx_values = []



    print("\nFD Convergence Study")

    print("-"*60)

    print(
        " N        dx          "
        "Error Dx       Rate       "
        "Error Dxx      Rate"
    )



    previous_e1 = None

    previous_e2 = None



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
            -k**2*np.sin(k*grid.x)
        )


        ux_num = fd.first_derivative(
            u
        )


        uxx_num = fd.second_derivative(
            u
        )


        e1 = np.linalg.norm(
            ux_num-ux_exact
        )/np.sqrt(N)


        e2 = np.linalg.norm(
            uxx_num-uxx_exact
        )/np.sqrt(N)



        dx_values.append(
            grid.dx
        )


        errors_first.append(e1)

        errors_second.append(e2)



        if previous_e1 is None:

            r1 = "-"

            r2 = "-"

        else:

            r1 = np.log2(
                previous_e1/e1
            )


            r2 = np.log2(
                previous_e2/e2
            )


        print(
            f"{N:<8}"
            f"{grid.dx:<12.4e}"
            f"{e1:<14.4e}"
            f"{str(r1):<10}"
            f"{e2:<14.4e}"
            f"{str(r2):<10}"
        )


        previous_e1=e1

        previous_e2=e2



    return (
        dx_values,
        errors_first,
        errors_second
    )

# =========================================================
# Run
# =========================================================


if __name__=="__main__":

    test_operator()

    convergence_test()