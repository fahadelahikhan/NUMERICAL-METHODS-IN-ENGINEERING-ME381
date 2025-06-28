#!/usr/bin/env python3

import numpy as np
from sympy import symbols, lambdify

def simpsons_1_3_rule(func, a, b, n):
    if n % 2 != 0:
        raise ValueError("Number of intervals must be even for Simpson's 1/3 rule.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    integral = h / 3 * (y[0] + 2 * np.sum(y[2:-1:2]) + 4 * np.sum(y[1:-1:2]) + y[-1])
    return integral

def simpsons_3_8_rule(func, a, b, n):
    if n % 3 != 0:
        raise ValueError("Number of intervals must be a multiple of 3 for Simpson's 3/8 rule.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    integral = 3 * h / 8 * (y[0] + 3 * np.sum(y[1:-1:3]) + 3 * np.sum(y[2:-1:3]) + 2 * np.sum(y[3:-1:3]) + y[-1])
    return integral

def main():
    try:
        # Function input
        func_expr = input("Enter the function expression (use 'x' as variable): ")
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

        # Compute integrals
        try:
            result_1_3 = simpsons_1_3_rule(func, a, b, n)
            print(f"\nThe approximate integral using Simpson's 1/3 rule is {result_1_3:.6f}")
        except ValueError as e:
            print(f"\nError applying Simpson's 1/3 rule: {e}")

        try:
            result_3_8 = simpsons_3_8_rule(func, a, b, n)
            print(f"The approximate integral using Simpson's 3/8 rule is {result_3_8:.6f}")
        except ValueError as e:
            print(f"Error applying Simpson's 3/8 rule: {e}")

    except Exception as e:
        print(f"Invalid input or computation error: {e}")

if __name__ == "__main__":
    main()