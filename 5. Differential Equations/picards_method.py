#!/usr/bin/env python3

import numpy as np
from sympy import symbols, Function, integrate, lambdify, sympify, Abs, limit, oo
import math


def compute_picard_approximation(f_expr, x_sym, y_func, y0, x0, current_approx, t_var):
    """
    Compute one iteration of Picard's method.

    Args:
        f_expr: The right-hand side of dy/dx = f(x, y)
        x_sym: Symbol for x
        y_func: Function symbol for y(x)
        y0: Initial value
        x0: Initial x value
        current_approx: Current approximation y_n(x)
        t_var: Integration variable

    Returns:
        Next approximation y_{n+1}(x)
    """
    # Substitute current approximation into f(x, y)
    f_substituted = f_expr.subs(y_func, current_approx)

    # Compute the integral from x0 to x
    try:
        integral = integrate(f_substituted, (t_var, x0, x_sym))
        next_approx = y0 + integral
        return next_approx.simplify()
    except Exception as e:
        print(f"Integration error: {e}")
        return current_approx


def check_convergence(approx_list, tolerance=1e-6, max_check_points=5):
    """
    Check if Picard iterations are converging.

    Args:
        approx_list: List of symbolic approximations
        tolerance: Convergence tolerance
        max_check_points: Number of points to check for convergence

    Returns:
        bool: True if converging, False otherwise
    """
    if len(approx_list) < 2:
        return False

    try:
        x = symbols('x')
        last_approx = lambdify(x, approx_list[-1], 'numpy')
        second_last_approx = lambdify(x, approx_list[-2], 'numpy')

        # Check convergence at several points
        test_points = np.linspace(0, 1, max_check_points)
        max_diff = 0

        for point in test_points:
            try:
                diff = abs(last_approx(point) - second_last_approx(point))
                max_diff = max(max_diff, diff)
            except:
                return False

        return max_diff < tolerance
    except:
        return False


def picards_method(f_expr, x0, y0, h, n, max_iterations, tolerance=1e-6):
    """
    Solve ODE using Picard's iterative method.

    Args:
        f_expr: Symbolic expression for dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size for solution evaluation
        n: Number of solution points
        max_iterations: Maximum number of Picard iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of (solution_points, approximations, converged)
    """
    x_sym = symbols('x')
    y_func = Function('y')(x_sym)
    t_var = symbols('t')  # Integration variable

    # Initialize with y0(x) = y0 (constant initial approximation)
    current_approx = y0
    approximations = [current_approx]

    print(f"Starting Picard iterations...")
    print(f"Initial approximation: y0(x) = {current_approx}")

    converged = False
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}:")

        # Compute next approximation
        next_approx = compute_picard_approximation(
            f_expr, x_sym, y_func, y0, x0, current_approx, t_var
        )

        approximations.append(next_approx)
        current_approx = next_approx

        print(f"y{iteration + 1}(x) = {current_approx}")

        # Check convergence
        if iteration > 0 and check_convergence(approximations, tolerance):
            converged = True
            print(f"Converged after {iteration + 1} iterations!")
            break

    if not converged:
        print(f"Did not converge after {max_iterations} iterations")

    # Evaluate the final approximation at specified points
    try:
        final_approx_func = lambdify(x_sym, approximations[-1], 'numpy')
        solution_points = []

        for i in range(n + 1):
            x_val = x0 + i * h
            try:
                y_val = float(final_approx_func(x_val))
                solution_points.append((x_val, y_val))
            except (ValueError, TypeError, OverflowError):
                # Handle cases where evaluation fails
                solution_points.append((x_val, float('nan')))

        return solution_points, approximations, converged

    except Exception as e:
        print(f"Error evaluating final approximation: {e}")
        return [], approximations, converged


def validate_inputs(x0, y0, h, n, max_iterations):
    """
    Validate input parameters.

    Args:
        x0, y0: Initial conditions
        h: Step size
        n: Number of steps
        max_iterations: Maximum iterations

    Returns:
        bool: True if inputs are valid
    """
    if h <= 0:
        print("Error: Step size must be positive")
        return False
    if n <= 0:
        print("Error: Number of steps must be positive")
        return False
    if max_iterations <= 0:
        print("Error: Maximum iterations must be positive")
        return False
    if max_iterations > 20:
        print("Warning: High number of iterations may lead to complex expressions")

    return True


def print_solution(solution_points, precision=6):
    """
    Print the solution in a formatted way.

    Args:
        solution_points: List of (x, y) tuples
        precision: Number of decimal places to display
    """
    print(f"\n{'Step':<6} {'x':<12} {'y':<15}")
    print("-" * 35)

    for i, (x, y) in enumerate(solution_points):
        if math.isnan(y):
            print(f"{i:<6} {x:<12.{precision}f} {'NaN':<15}")
        else:
            print(f"{i:<6} {x:<12.{precision}f} {y:<15.{precision}f}")


def print_approximations(approximations):
    """
    Print all Picard approximations.

    Args:
        approximations: List of symbolic approximations
    """
    print("\nPicard Approximations:")
    print("=" * 50)

    for i, approx in enumerate(approximations):
        print(f"y{i}(x) = {approx}")


def get_user_input():
    """
    Get user input for ODE and parameters.

    Returns:
        Tuple of (f_expr, x0, y0, h, n, max_iterations, tolerance)
    """
    print("Picard's Method ODE Solver")
    print("=" * 27)
    print("Enter the ODE in the form: dy/dx = f(x, y)")
    print("Use 'y' for the dependent variable")
    print("Example: x + y, x*y, 2*x + 3*y, x**2 + y, etc.")
    print("Note: Picard's method works best for simple ODEs")

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
            max_iterations = int(input("Enter maximum Picard iterations (1-20): "))

            # Optional tolerance
            tolerance_input = input("Enter convergence tolerance (default 1e-6): ").strip()
            if tolerance_input:
                tolerance = float(tolerance_input)
            else:
                tolerance = 1e-6

            if validate_inputs(x0, y0, h, n, max_iterations):
                break

        except ValueError:
            print("Please enter valid numerical values")

    return f_expr, x0, y0, h, n, max_iterations, tolerance


def main():
    """
    Main function to run the Picard's method ODE solver.
    """
    try:
        # Get user input
        f_expr, x0, y0, h, n, max_iterations, tolerance = get_user_input()

        print(f"\nSolving: dy/dx = {f_expr}")
        print(f"Initial conditions: x0 = {x0}, y0 = {y0}")
        print(f"Step size: h = {h}")
        print(f"Number of steps: {n}")
        print(f"Maximum iterations: {max_iterations}")
        print(f"Convergence tolerance: {tolerance}")

        # Solve the ODE
        solution_points, approximations, converged = picards_method(
            f_expr, x0, y0, h, n, max_iterations, tolerance
        )

        # Display approximations
        print_approximations(approximations)

        # Display solution
        if solution_points:
            print_solution(solution_points)
        else:
            print("No solution points could be computed")

        # Convergence status
        if converged:
            print("\n✓ Method converged successfully")
        else:
            print("\n⚠ Method did not converge - results may be inaccurate")

        # Ask if user wants to save results
        save_option = input("\nSave results to file? (y/n): ").lower()
        if save_option == 'y':
            filename = input("Enter filename (default: picard_solution.txt): ").strip()
            if not filename:
                filename = "picard_solution.txt"

            try:
                with open(filename, 'w') as f:
                    f.write(f"Picard's Method ODE Solution\n")
                    f.write(f"ODE: dy/dx = {f_expr}\n")
                    f.write(f"Initial conditions: x0 = {x0}, y0 = {y0}\n")
                    f.write(f"Step size: h = {h}\n")
                    f.write(f"Number of steps: {n}\n")
                    f.write(f"Maximum iterations: {max_iterations}\n")
                    f.write(f"Convergence tolerance: {tolerance}\n")
                    f.write(f"Converged: {converged}\n\n")

                    f.write("Picard Approximations:\n")
                    f.write("=" * 50 + "\n")
                    for i, approx in enumerate(approximations):
                        f.write(f"y{i}(x) = {approx}\n")

                    f.write(f"\n{'Step':<6} {'x':<12} {'y':<15}\n")
                    f.write("-" * 35 + "\n")

                    for i, (x, y) in enumerate(solution_points):
                        if math.isnan(y):
                            f.write(f"{i:<6} {x:<12.6f} {'NaN':<15}\n")
                        else:
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