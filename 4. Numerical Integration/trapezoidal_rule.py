#!/usr/bin/env python3

import numpy as np

def trapezoidal_rule(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    integral = h * (y[0] + y[-1] + 2 * np.sum(y[1:-1])) / 2
    return integral

def main():
    try:
        # Function input
        func_expr = input("Enter the function expression (use 'x' as variable): ")
        from sympy import symbols, lambdify
        x = symbols('x')
        func = lambdify(x, func_expr, 'numpy')

        # Interval and subintervals
        a = float(input("Enter the start of the interval (a): "))
        b = float(input("Enter the end of the interval (b): "))
        if a >= b:
            print("Error: Interval start must be less than end.")
            return

        n = int(input("Enter the number of subintervals: "))
        if n <= 0:
            print("Error: Number of subintervals must be a positive integer.")
            return

        # Compute integral
        result = trapezoidal_rule(func, a, b, n)

        # Output result
        print(f"\nThe approximate integral from {a} to {b} using {n} subintervals is {result:.6f}")

    except Exception as e:
        print(f"Invalid input or computation error: {e}")

if __name__ == "__main__":
    main()