#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, integrate


# Interpolation Methods
def linear_interpolation(x, y, target_x):
    n = len(x)
    for i in range(n - 1):
        if x[i] <= target_x <= x[i + 1]:
            return y[i] + (y[i + 1] - y[i]) * (target_x - x[i]) / (x[i + 1] - x[i])
    return y[0] if target_x < x[0] else y[-1]


def cubic_interpolation(x, y, target_x):
    n = len(x)
    for i in range(n - 1):
        if x[i] <= target_x <= x[i + 1]:
            h = x[i + 1] - x[i]
            t = (target_x - x[i]) / h
            return (2 * t ** 3 - 3 * t ** 2 + 1) * y[i] + (t ** 3 - 2 * t ** 2 + t) * (y[i + 1] - y[i]) / h + (
                        -2 * t ** 3 + 3 * t ** 2) * y[i + 1] + (t ** 3 - t ** 2) * (y[i + 1] - y[i]) / h


# Integration Methods
def trapezoidal_rule(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return h * (y[0] + y[-1] + 2 * np.sum(y[1:-1])) / 2


def simpsons_rule(func, a, b, n):
    if n % 2 != 0:
        raise ValueError("Number of intervals must be even for Simpson's rule.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return h / 3 * (y[0] + 2 * np.sum(y[2:-1:2]) + 4 * np.sum(y[1:-1:2]) + y[-1])


# Differential Equation Methods
def euler_method(f, x0, y0, h, n):
    solution = [(x0, y0)]
    current_x = x0
    current_y = y0
    for _ in range(n):
        current_y += h * f(current_x, current_y)
        current_x += h
        solution.append((current_x, current_y))
    return solution


def rk4_method(f, x0, y0, h, n):
    solution = [(x0, y0)]
    current_x = x0
    current_y = y0
    for _ in range(n):
        k1 = f(current_x, current_y)
        k2 = f(current_x + h / 2, current_y + h / 2 * k1)
        k3 = f(current_x + h / 2, current_y + h / 2 * k2)
        k4 = f(current_x + h, current_y + h * k3)
        current_y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        current_x += h
        solution.append((current_x, current_y))
    return solution


def main():
    try:
        # Define symbolic variables
        x_sym = symbols('x')
        y_sym = symbols('y')

        # Input function
        func_expr = input(
            "Enter the function expression (use 'x' for integration/interpolation or 'x' and 'y' for ODE): ")
        is_ode = 'y' in func_expr

        if is_ode:
            from sympy import Function
            y = Function('y')
            f = lambdify((x_sym, y_sym), func_expr.replace('y', 'y(x)'), 'numpy')
        else:
            f = lambdify(x_sym, func_expr, 'numpy')

        # Input interval
        a = float(input("Enter the start of the interval (a): "))
        b = float(input("Enter the end of the interval (b): "))

        if is_ode:
            # ODE-specific inputs
            y0 = float(input("Enter initial condition y(a): "))
            h = float(input("Enter step size (h): "))
            n = int(input("Enter number of steps: "))

            # Solve using Euler and RK4 methods
            euler_solution = euler_method(f, a, y0, h, n)
            rk4_solution = rk4_method(f, a, y0, h, n)

            # Prepare data for plotting
            euler_x = [point[0] for point in euler_solution]
            euler_y = [point[1] for point in euler_solution]
            rk4_x = [point[0] for point in rk4_solution]
            rk4_y = [point[1] for point in rk4_solution]

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(euler_x, euler_y, label='Euler Method')
            plt.plot(rk4_x, rk4_y, label='RK4 Method')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Comparison of ODE Solvers')
            plt.legend()
            plt.grid(True)
            plt.show()

        else:
            # Integration/Interpolation-specific inputs
            n = int(input("Enter number of points: "))
            x_vals = np.linspace(a, b, n)
            y_vals = f(x_vals)

            # Input target x for interpolation
            target_x = float(input("Enter x value for interpolation: "))

            # Compute interpolation
            linear_result = linear_interpolation(x_vals, y_vals, target_x)
            cubic_result = cubic_interpolation(x_vals, y_vals, target_x)

            # Compute integration
            trapezoidal_result = trapezoidal_rule(f, a, b, n)
            simpsons_result = simpsons_rule(f, a, b, n)

            # Print results
            print(f"\nInterpolation at x = {target_x}:")
            print(f"Linear: {linear_result:.6f}")
            print(f"Cubic: {cubic_result:.6f}")

            print(f"\nIntegration from {a} to {b}:")
            print(f"Trapezoidal: {trapezoidal_result:.6f}")
            print(f"Simpson's: {simpsons_result:.6f}")

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, 'o', label='Data Points')
            x_fine = np.linspace(a, b, 300)
            plt.plot(x_fine, f(x_fine), label='Actual Function')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Interpolation and Integration Comparison')
            plt.legend()
            plt.grid(True)
            plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()