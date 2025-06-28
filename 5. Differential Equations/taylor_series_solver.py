#!/usr/bin/env python3

import numpy as np
from sympy import symbols, Function, diff, lambdify


def taylor_series_solver(f, x0, y0, h, n, order):
    x = symbols('x')
    y = Function('y')(x)
    derivatives = [f]
    # Compute higher-order derivatives
    for i in range(1, order):
        derivatives.append(diff(derivatives[-1], x).simplify())
    # Convert derivatives to lambdify functions
    derivative_funcs = [lambdify(x, d.subs(y, f), 'numpy') for d in derivatives]
    # Initialize solution array
    solution = [(x0, y0)]
    current_x = x0
    current_y = y0
    for _ in range(n):
        # Compute Taylor series expansion
        delta_y = 0
        for k in range(order):
            delta_y += derivative_funcs[k](current_x) * (h ** (k + 1)) / np.math.factorial(k + 1)
        current_y += delta_y
        current_x += h
        solution.append((current_x, current_y))
    return solution


def main():
    try:
        # Define variables and function
        x = symbols('x')
        y = Function('y')(x)
        print("Enter the ODE in terms of x and y(x):")
        f = input("dy/dx = ")
        f = f.replace('y', 'y(x)')
        from sympy import sympify
        f = sympify(f)

        # Input parameters
        x0 = float(input("Enter initial x value (x0): "))
        y0 = float(input("Enter initial y value (y0): "))
        h = float(input("Enter step size (h): "))
        n = int(input("Enter number of steps: "))
        order = int(input("Enter order of Taylor series: "))

        # Solve ODE
        solution = taylor_series_solver(f, x0, y0, h, n, order)

        # Print solution
        print("\nSolution:")
        for point in solution:
            print(f"x = {point[0]:.4f}, y = {point[1]:.6f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()