#!/usr/bin/env python3

import numpy as np

def lagrange_interpolation(x, y, target_x):
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (target_x - x[j]) / (x[i] - x[j])
        result += term
    return result

def compute_derivative(x, y, target_x, order=1):
    h = x[1] - x[0]
    if order == 1:
        # First derivative using central difference
        return (lagrange_interpolation(x, y, target_x + h) - lagrange_interpolation(x, y, target_x - h)) / (2 * h)
    elif order == 2:
        # Second derivative using central difference
        return (lagrange_interpolation(x, y, target_x + h) - 2 * lagrange_interpolation(x, y, target_x) + lagrange_interpolation(x, y, target_x - h)) / (h ** 2)
    else:
        raise ValueError("Only first and second order derivatives are supported.")

def main():
    try:
        # Function input
        func_expr = input("Enter the function expression (use 'x' as variable): ")
        from sympy import symbols, lambdify
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

        # Generate interpolation points
        x_vals = np.linspace(a, b, n)
        y_vals = func(x_vals)

        # Target point for derivative
        target_x = float(input("Enter the x value where the derivative is to be computed: "))
        if target_x < a or target_x > b:
            print("Error: Target x must be within the interval [a, b].")
            return

        # Compute derivatives
        first_derivative = compute_derivative(x_vals, y_vals, target_x, order=1)
        second_derivative = compute_derivative(x_vals, y_vals, target_x, order=2)

        # Output results
        print(f"\nFirst derivative at x = {target_x}: {first_derivative:.6f}")
        print(f"Second derivative at x = {target_x}: {second_derivative:.6f}")

    except Exception as e:
        print(f"Invalid input or computation error: {e}")

if __name__ == "__main__":
    main()