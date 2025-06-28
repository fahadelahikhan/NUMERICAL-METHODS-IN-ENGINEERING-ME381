#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify

def compute_derivative(func, x_vals, h=1e-5):
    return (func(x_vals + h) - func(x_vals - h)) / (2 * h)

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

        n = int(input("Enter the number of points for plotting: "))
        if n < 2:
            print("Error: At least 2 points are required.")
            return

        # Generate points
        x_vals = np.linspace(a, b, n)
        y_vals = func(x_vals)

        # Compute derivative
        y_deriv = compute_derivative(func, x_vals)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label='Function')
        plt.plot(x_vals, y_deriv, label='Numerical Derivative', linestyle='--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Function and Its Numerical Derivative')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Invalid input or computation error: {e}")

if __name__ == "__main__":
    main()