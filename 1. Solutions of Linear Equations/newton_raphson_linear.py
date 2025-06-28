#!/usr/bin/env python3

import numpy as np
from scipy.optimize import fsolve

def compute_jacobian(F, x, h=1e-5):
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        J[:, i] = (F(x_plus_h) - F(x)) / h
    return J

def newton_raphson(F, x0, tol=1e-10, max_iter=100):
    x = np.array(x0, dtype=float)
    for iteration in range(max_iter):
        J = compute_jacobian(F, x)
        Fx = F(x)
        try:
            delta_x = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            raise ValueError("Jacobian matrix is singular. Newton-Raphson method failed.")
        x += delta_x
        if np.linalg.norm(delta_x) < tol:
            return x.tolist()
    raise ValueError(f"Newton-Raphson method did not converge within {max_iter} iterations.")

def main():
    try:
        n = int(input("Enter the number of equations: "))
        if n <= 0:
            print("Number of equations must be a positive integer.")
            return

        print("Enter the system of equations (one per line, use x0, x1, ..., x{} as variables):".format(n - 1))
        from sympy import symbols, lambdify
        x_symbols = symbols(f'x0:{n}')
        equations = []
        for i in range(n):
            eq = input(f"Equation {i + 1}: ")
            equations.append(lambdify(x_symbols, eq, 'numpy'))

        def F(x):
            return np.array([eq(*x) for eq in equations])

        print("\nEnter initial guess ({} values separated by spaces):".format(n))
        initial_guess = list(map(float, input().strip().split()))
        if len(initial_guess) != n:
            print(f"Initial guess must have exactly {n} elements.")
            return

        tol = float(input("Enter tolerance (e.g., 1e-10): "))
        if tol <= 0:
            print("Tolerance must be a positive number.")
            return

        max_iter = int(input("Enter maximum number of iterations: "))
        if max_iter <= 0:
            print("Maximum iterations must be a positive integer.")
            return

        try:
            solution = newton_raphson(F, initial_guess, tol, max_iter)
            print("\nSolution:")
            for val in solution:
                print(f"{val:.6f}")
        except ValueError as e:
            print(f"\nError: {e}")

    except Exception as e:
        print(f"Invalid input: {e}")

if __name__ == "__main__":
    main()