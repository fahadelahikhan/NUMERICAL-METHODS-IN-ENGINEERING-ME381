#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, integrate

def trapezoidal_rule(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    integral = h * (y[0] + y[-1] + 2 * np.sum(y[1:-1])) / 2
    return integral

def simpsons_rule(func, a, b, n):
    if n % 2 != 0:
        raise ValueError("Number of intervals must be even for Simpson's rule.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    integral = h / 3 * (y[0] + 2 * np.sum(y[2:-1:2]) + 4 * np.sum(y[1:-1:2]) + y[-1])
    return integral

def main():
    try:
        # Function input
        func_expr = input("Enter the function expression (use 'x' as variable): ")
        x = symbols('x')
        func = lambdify(x, func_expr, 'numpy')

        # Interval and intervals
        a = float(input("Enter the start of the interval (a): "))
        b = float(input("Enter the end of the interval (b): "))
        if a >= b:
            print("Error: Interval start must be less than end.")
            return

        n = int(input("Enter the number of intervals: "))
        if n <= 0:
            print("Error: Number of intervals must be a positive integer.")
            return

        # Compute integrals
        try:
            trapezoidal_result = trapezoidal_rule(func, a, b, n)
        except Exception as e:
            print(f"Error computing trapezoidal rule: {e}")
            trapezoidal_result = None

        try:
            simpsons_result = simpsons_rule(func, a, b, n)
        except Exception as e:
            print(f"Error computing Simpson's rule: {e}")
            simpsons_result = None

        # Compute actual integral using symbolic integration
        actual_integral = integrate(func_expr, (x, a, b)).evalf()

        # Plotting
        x_vals = np.linspace(a, b, 300)
        y_vals = func(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label='Function')
        plt.fill_between(x_vals, y_vals, alpha=0.3, label='Actual Area')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Function and Actual Area')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Output results
        print(f"\nActual integral from {a} to {b}: {actual_integral:.6f}")
        if trapezoidal_result is not None:
            print(f"Trapezoidal rule approximation: {trapezoidal_result:.6f}")
        if simpsons_result is not None:
            print(f"Simpson's rule approximation: {simpsons_result:.6f}")

    except Exception as e:
        print(f"Invalid input or computation error: {e}")

if __name__ == "__main__":
    main()