#!/usr/bin/env python3

import numpy as np
from sympy import symbols, lambdify


def rk2_method(f, x0, y0, h, n):
    solution = [(x0, y0)]
    current_x = x0
    current_y = y0

    for _ in range(n):
        # Calculate slopes
        k1 = f(current_x, current_y)
        k2 = f(current_x + h / 2, current_y + h / 2 * k1)
        # Update solution
        current_y += h * k2
        current_x += h
        solution.append((current_x, current_y))

    return solution


def main():
    try:
        # Define variables
        x = symbols('x')
        y = symbols('y')

        # Input ODE function
        print("Enter the ODE function dy/dx = f(x, y):")
        f_expr = input()
        f = lambdify((x, y), f_expr, 'numpy')

        # Input parameters
        x0 = float(input("Enter initial x value (x0): "))
        y0 = float(input("Enter initial y value (y0): "))
        h = float(input("Enter step size (h): "))
        n = int(input("Enter number of steps: "))

        # Solve ODE using RK2 method
        solution = rk2_method(f, x0, y0, h, n)

        # Print solution
        print("\nSolution:")
        for point in solution:
            print(f"x = {point[0]:.4f}, y = {point[1]:.6f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()