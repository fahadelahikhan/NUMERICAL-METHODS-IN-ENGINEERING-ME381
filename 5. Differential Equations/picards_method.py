#!/usr/bin/env python3

import numpy as np
from sympy import symbols, Function, integrate, lambdify


def picards_method(f, x0, y0, h, n, iterations):
    x = symbols('x')
    y = Function('y')
    current_approx = y0
    approximations = [current_approx]

    for _ in range(iterations):
        integral = integrate(f.subs([(y(x), current_approx), (x, x)]), (x, x0, x))
        next_approx = y0 + integral
        current_approx = next_approx
        approximations.append(current_approx)

    # Convert the final approximation to a lambdify function for evaluation
    final_approx = lambdify(x, approximations[-1], 'numpy')
    solution = []
    current_x = x0
    for _ in range(n):
        solution.append((current_x, final_approx(current_x)))
        current_x += h

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
        iterations = int(input("Enter number of Picard iterations: "))

        # Solve ODE using Picard's method
        solution = picards_method(f, x0, y0, h, n, iterations)

        # Print solution
        print("\nSolution:")
        for point in solution:
            print(f"x = {point[0]:.4f}, y = {point[1]:.6f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()