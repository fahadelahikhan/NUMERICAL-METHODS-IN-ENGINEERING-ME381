#!/usr/bin/env python3

import numpy as np
from sympy import symbols, lambdify, diff

def lagrange_interpolation_derivative(x, y, target_x):
    n = len(x)
    h = x[1] - x[0]
    # Using central difference for first derivative
    derivative = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (target_x - x[j]) / (x[i] - x[j])
        derivative += term
    # Calculate derivative of the Lagrange polynomial
    x_sym = symbols('x')
    lagrange_poly = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_sym - x[j]) / (x[i] - x[j])
        lagrange_poly += term
    derivative = diff(lagrange_poly, x_sym).evalf(subs={x_sym: target_x})
    return derivative

def newton_interpolation_derivative(x, y, target_x):
    n = len(x)
    # Compute divided differences
    divided_diff = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        divided_diff[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / (x[i + j] - x[i])
    # Construct Newton polynomial and its derivative
    x_sym = symbols('x')
    newton_poly = divided_diff[0][0]
    derivative = 0.0
    for j in range(1, n):
        term = divided_diff[0][j]
        poly_term = 1.0
        for i in range(j):
            poly_term *= (x_sym - x[i])
        newton_poly += term * poly_term
    derivative = diff(newton_poly, x_sym).evalf(subs={x_sym: target_x})
    return derivative

def main():
    try:
        # Function input
        func_expr = input("Enter the function expression (use 'x' as variable): ")
        x_sym = symbols('x')
        func = lambdify(x_sym, func_expr, 'numpy')

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

        # Generate interpolation points
        x_vals = np.linspace(a, b, n)
        y_vals = func(x_vals)

        # Target point for derivative
        target_x = float(input("Enter the x value where the derivative is to be computed: "))
        if target_x < a or target_x > b:
            print("Error: Target x must be within the interval [a, b].")
            return

        # Compute derivatives using both methods
        lagrange_deriv = lagrange_interpolation_derivative(x_vals, y_vals, target_x)
        newton_deriv = newton_interpolation_derivative(x_vals, y_vals, target_x)

        # Output results
        print(f"\nFirst derivative using Lagrange method at x = {target_x}: {lagrange_deriv:.6f}")
        print(f"First derivative using Newton method at x = {target_x}: {newton_deriv:.6f}")

    except Exception as e:
        print(f"Invalid input or computation error: {e}")

if __name__ == "__main__":
    main()