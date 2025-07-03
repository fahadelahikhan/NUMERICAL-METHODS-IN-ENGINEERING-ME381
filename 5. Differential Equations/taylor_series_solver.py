#!/usr/bin/env python3

import numpy as np
from sympy import symbols, Function, diff, lambdify, sympify
import math


def compute_derivatives(f_expr, x_sym, y_func, order):
    """
    Compute higher-order derivatives of the ODE using chain rule.

    Args:
        f_expr: The right-hand side of dy/dx = f(x, y)
        x_sym: Symbol for x
        y_func: Function symbol for y(x)
        order: Maximum order of derivatives to compute

    Returns:
        List of derivative expressions
    """
    derivatives = [f_expr]  # f(x, y) = dy/dx

    for i in range(1, order):
        # Apply chain rule: d/dx[f(x, y)] = ∂f/∂x + ∂f/∂y * dy/dx
        prev_deriv = derivatives[-1]

        # Compute partial derivatives
        df_dx = diff(prev_deriv, x_sym)
        df_dy = diff(prev_deriv, y_func)

        # Apply chain rule: next derivative = df/dx + df/dy * f
        next_deriv = df_dx + df_dy * f_expr
        derivatives.append(next_deriv.simplify())

    return derivatives


def taylor_series_solver(f_expr, x0, y0, h, n, order):
    """
    Solve ODE using Taylor series method.

    Args:
        f_expr: Symbolic expression for dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps
        order: Order of Taylor series

    Returns:
        List of (x, y) tuples representing the solution
    """
    x_sym = symbols('x')
    y_func = Function('y')(x_sym)

    # Compute derivatives
    derivatives = compute_derivatives(f_expr, x_sym, y_func, order)

    # Convert to numerical functions
    # Replace y(x) with a second argument for lambdify
    derivative_funcs = []
    for deriv in derivatives:
        # Replace y(x) with a symbol y_val for lambdify
        y_val = symbols('y_val')
        deriv_substituted = deriv.subs(y_func, y_val)
        derivative_funcs.append(lambdify((x_sym, y_val), deriv_substituted, 'numpy'))

    # Initialize solution arrays
    x_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    x_values[0] = x0
    y_values[0] = y0

    # Solve using Taylor series
    for i in range(n):
        current_x = x_values[i]
        current_y = y_values[i]

        # Compute Taylor series terms
        delta_y = 0
        for k in range(order):
            try:
                derivative_val = derivative_funcs[k](current_x, current_y)
                term = derivative_val * (h ** (k + 1)) / math.factorial(k + 1)
                delta_y += term
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                print(f"Warning: Numerical issue at step {i + 1}, term {k + 1}: {e}")
                break

        # Update solution
        y_values[i + 1] = current_y + delta_y
        x_values[i + 1] = current_x + h

    return list(zip(x_values, y_values))


def validate_inputs(x0, y0, h, n, order):
    """
    Validate input parameters.

    Args:
        x0, y0: Initial conditions
        h: Step size
        n: Number of steps
        order: Taylor series order

    Returns:
        bool: True if inputs are valid
    """
    if h <= 0:
        print("Error: Step size must be positive")
        return False
    if n <= 0:
        print("Error: Number of steps must be positive")
        return False
    if order < 1:
        print("Error: Taylor series order must be at least 1")
        return False
    if order > 10:
        print("Warning: High order Taylor series may be unstable")

    return True


def print_solution(solution, precision=6):
    """
    Print the solution in a formatted way.

    Args:
        solution: List of (x, y) tuples
        precision: Number of decimal places to display
    """
    print(f"\n{'Step':<6} {'x':<12} {'y':<15}")
    print("-" * 35)

    for i, (x, y) in enumerate(solution):
        print(f"{i:<6} {x:<12.{precision}f} {y:<15.{precision}f}")


def get_user_input():
    """
    Get user input for ODE and parameters.

    Returns:
        Tuple of (f_expr, x0, y0, h, n, order)
    """
    print("Taylor Series ODE Solver")
    print("=" * 25)
    print("Enter the ODE in the form: dy/dx = f(x, y)")
    print("Use 'y' for the dependent variable")
    print("Example: x + y, x*y, x**2 + y**2, sin(x)*y, etc.")

    while True:
        try:
            f_str = input("\ndy/dx = ")
            if not f_str.strip():
                print("Please enter a valid expression")
                continue

            # Parse the expression
            f_expr = sympify(f_str)

            # Check if expression contains valid symbols
            free_symbols = f_expr.free_symbols
            valid_symbols = {symbols('x'), symbols('y')}

            if not free_symbols.issubset(valid_symbols):
                invalid_symbols = free_symbols - valid_symbols
                print(f"Invalid symbols found: {invalid_symbols}")
                print("Please use only 'x' and 'y'")
                continue

            break

        except Exception as e:
            print(f"Error parsing expression: {e}")
            print("Please try again")

    # Get numerical parameters
    while True:
        try:
            x0 = float(input("Enter initial x value (x0): "))
            y0 = float(input("Enter initial y value (y0): "))
            h = float(input("Enter step size (h): "))
            n = int(input("Enter number of steps: "))
            order = int(input("Enter order of Taylor series (1-10): "))

            if validate_inputs(x0, y0, h, n, order):
                break

        except ValueError:
            print("Please enter valid numerical values")

    return f_expr, x0, y0, h, n, order


def main():
    """
    Main function to run the Taylor series ODE solver.
    """
    try:
        # Get user input
        f_expr, x0, y0, h, n, order = get_user_input()

        print(f"\nSolving: dy/dx = {f_expr}")
        print(f"Initial conditions: x0 = {x0}, y0 = {y0}")
        print(f"Step size: h = {h}")
        print(f"Number of steps: {n}")
        print(f"Taylor series order: {order}")

        # Solve the ODE
        solution = taylor_series_solver(f_expr, x0, y0, h, n, order)

        # Display results
        print_solution(solution)

        # Ask if user wants to save results
        save_option = input("\nSave results to file? (y/n): ").lower()
        if save_option == 'y':
            filename = input("Enter filename (default: taylor_solution.txt): ").strip()
            if not filename:
                filename = "taylor_solution.txt"

            try:
                with open(filename, 'w') as f:
                    f.write(f"Taylor Series ODE Solution\n")
                    f.write(f"ODE: dy/dx = {f_expr}\n")
                    f.write(f"Initial conditions: x0 = {x0}, y0 = {y0}\n")
                    f.write(f"Step size: h = {h}\n")
                    f.write(f"Number of steps: {n}\n")
                    f.write(f"Taylor series order: {order}\n\n")
                    f.write(f"{'Step':<6} {'x':<12} {'y':<15}\n")
                    f.write("-" * 35 + "\n")

                    for i, (x, y) in enumerate(solution):
                        f.write(f"{i:<6} {x:<12.6f} {y:<15.6f}\n")

                print(f"Results saved to {filename}")

            except IOError as e:
                print(f"Error saving file: {e}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()