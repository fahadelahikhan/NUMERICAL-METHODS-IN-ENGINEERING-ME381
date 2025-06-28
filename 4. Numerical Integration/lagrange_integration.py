#!/usr/bin/env python3

import numpy as np
from sympy import symbols, lambdify, integrate


def lagrange_interpolation(x, y, x_val):
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_val - x[j]) / (x[i] - x[j])
        result += term
    return result


def integrate_lagrange(func, a, b, n):
    # Generate interpolation points
    x_vals = np.linspace(a, b, n)
    y_vals = func(x_vals)

    # Define the Lagrange polynomial symbolically
    x_sym = symbols('x')
    lagrange_poly = 0.0
    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if i != j:
                term *= (x_sym - x_vals[j]) / (x_vals[i] - x_vals[j])
        lagrange_poly += term

    # Integrate the Lagrange polynomial
    integral = integrate(lagrange_poly, (x_sym, a, b))
    return integral.evalf()


def main():
    try:
        # Function input
        func_expr = input("Enter the function expression (use 'x' as variable): ")
        x = symbols('x')
        func = lambdify(x, func_expr, 'numpy')

        # Interval and points
        a = float(input("Enter the start of the interval (a): "))
        b = float(input("Enter the end of the interval (b): "))
        if a >= b:
            print("Error: Interval start must be less than end.")
            return

        n = int(input("Enter the number of interpolation points: "))
        if n < 2:
            print("Error: At least 2 interpolation points are required.")
            return

        # Compute integral
        result = integrate_lagrange(func, a, b, n)

        # Output result
        print(
            f"\nThe approximate integral from {a} to {b} using Lagrange interpolation with {n} points is {result:.6f}")

    except Exception as e:
        print(f"Invalid input or computation error: {e}")


if __name__ == "__main__":
    main()