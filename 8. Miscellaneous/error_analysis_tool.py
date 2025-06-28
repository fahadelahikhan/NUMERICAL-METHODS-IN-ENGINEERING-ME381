#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, integrate

# Numerical Methods
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


# Error Calculation
def calculate_l2_error(exact, approx):
    return np.sqrt(np.sum((exact - approx) ** 2) / len(exact))


def calculate_max_error(exact, approx):
    return np.max(np.abs(exact - approx))


# Main Function
def main():
    try:
        # Define symbolic variables
        x_sym = symbols('x')
        y_sym = symbols('y')

        # Input function
        func_expr = input(
            "Enter the function expression (use 'x' for interpolation/integration or 'x' and 'y' for ODE): ")
        is_ode = 'y' in func_expr

        if is_ode:
            from sympy import Function
            y = Function('y')
            f = lambdify((x_sym, y_sym), func_expr.replace('y', 'y(x)'), 'numpy')
            exact_solution_expr = input("Enter the exact solution expression for the ODE: ")
            exact_solution = lambdify(x_sym, exact_solution_expr, 'numpy')
        else:
            f = lambdify(x_sym, func_expr, 'numpy')
            exact_solution = f

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

            # Prepare data for error calculation
            euler_x = [point[0] for point in euler_solution]
            euler_y = [point[1] for point in euler_solution]
            rk4_x = [point[0] for point in rk4_solution]
            rk4_y = [point[1] for point in rk4_solution]

            # Compute exact solution at Euler's x values
            exact_euler = exact_solution(np.array(euler_x))
            exact_rk4 = exact_solution(np.array(rk4_x))

            # Calculate errors
            euler_l2 = calculate_l2_error(exact_euler, euler_y)
            euler_max = calculate_max_error(exact_euler, euler_y)
            rk4_l2 = calculate_l2_error(exact_rk4, rk4_y)
            rk4_max = calculate_max_error(exact_rk4, rk4_y)

            # Print errors
            print("\nErrors for ODE Solvers:")
            print(f"Euler Method - L2 Error: {euler_l2:.6e}, Max Error: {euler_max:.6e}")
            print(f"RK4 Method - L2 Error: {rk4_l2:.6e}, Max Error: {rk4_max:.6e}")

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(euler_x, euler_y, label='Euler Method')
            plt.plot(rk4_x, rk4_y, label='RK4 Method')
            plt.plot(euler_x, exact_euler, label='Exact Solution', linestyle='--')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Comparison of ODE Solvers with Exact Solution')
            plt.legend()
            plt.grid(True)
            plt.show()

        else:
            # Integration/Interpolation-specific inputs
            n = int(input("Enter number of points: "))
            x_vals = np.linspace(a, b, n)
            y_vals = f(x_vals)

            # Input target x values for interpolation
            target_x = float(input("Enter x value for interpolation: "))

            # Compute interpolation
            linear_result = linear_interpolation(x_vals, y_vals, target_x)
            cubic_result = cubic_interpolation(x_vals, y_vals, target_x)

            # Compute integration
            trapezoidal_result = trapezoidal_rule(f, a, b, n)
            simpsons_result = simpsons_rule(f, a, b, n)

            # Compute exact values for error
            exact_interp = exact_solution(target_x)
            exact_integral = integrate(f, (x_sym, a, b)).evalf()

            # Calculate errors
            linear_interp_error = abs(exact_interp - linear_result)
            cubic_interp_error = abs(exact_interp - cubic_result)
            trapezoidal_error = abs(exact_integral - trapezoidal_result)
            simpsons_error = abs(exact_integral - simpsons_result)

            # Print results
            print(f"\nInterpolation at x = {target_x}:")
            print(f"Linear: {linear_result:.6f} (Error: {linear_interp_error:.6e})")
            print(f"Cubic: {cubic_result:.6f} (Error: {cubic_interp_error:.6e})")

            print(f"\nIntegration from {a} to {b}:")
            print(f"Trapezoidal: {trapezoidal_result:.6f} (Error: {trapezoidal_error:.6e})")
            print(f"Simpson's: {simpsons_result:.6f} (Error: {simpsons_error:.6e})")

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, 'o', label='Data Points')
            x_fine = np.linspace(a, b, 300)
            plt.plot(x_fine, exact_solution(x_fine), label='Exact Function')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Interpolation and Integration Comparison with Exact Solution')
            plt.legend()
            plt.grid(True)
            plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()