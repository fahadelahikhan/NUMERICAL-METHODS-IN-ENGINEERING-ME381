#!/usr/bin/env python3

import numpy as np
from sympy import symbols, lambdify, sympify
import sys


def modified_euler_method(f, x0, y0, h, n, method='heun'):
    """
    Solve ODE using Modified Euler method

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps
        method: 'heun' for Heun's method, 'midpoint' for RK2 midpoint method

    Returns:
        List of tuples (x, y) representing the solution
    """
    # Pre-allocate arrays for better performance and easier translation
    x_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # Set initial conditions
    x_values[0] = x0
    y_values[0] = y0

    # Modified Euler method iterations
    for i in range(n):
        x_i = x_values[i]
        y_i = y_values[i]

        if method == 'heun':
            # Heun's method (true Modified Euler)
            # Step 1: Calculate predictor using Euler's method
            k1 = f(x_i, y_i)
            y_predictor = y_i + h * k1
            x_next = x_i + h

            # Step 2: Calculate corrector using trapezoidal rule
            k2 = f(x_next, y_predictor)
            y_values[i + 1] = y_i + (h / 2) * (k1 + k2)

        elif method == 'midpoint':
            # RK2 Midpoint method
            # Step 1: Calculate slope at current point
            k1 = f(x_i, y_i)

            # Step 2: Calculate slope at midpoint
            x_mid = x_i + h / 2
            y_mid = y_i + (h / 2) * k1
            k2 = f(x_mid, y_mid)

            # Step 3: Use midpoint slope for full step
            y_values[i + 1] = y_i + h * k2

        else:
            raise ValueError(f"Unknown method: {method}")

        # Update x
        x_values[i + 1] = x_i + h

    # Return as list of tuples for compatibility
    solution = [(x_values[i], y_values[i]) for i in range(n + 1)]
    return solution


def euler_method(f, x0, y0, h, n):
    """
    Basic Euler method for comparison

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps

    Returns:
        List of tuples (x, y) representing the solution
    """
    x_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    x_values[0] = x0
    y_values[0] = y0

    for i in range(n):
        derivative = f(x_values[i], y_values[i])
        y_values[i + 1] = y_values[i] + h * derivative
        x_values[i + 1] = x_values[i] + h

    solution = [(x_values[i], y_values[i]) for i in range(n + 1)]
    return solution


def parse_function(func_str):
    """
    Parse and validate the input function string

    Args:
        func_str: String representation of the function

    Returns:
        Compiled function ready for numerical computation
    """
    try:
        # Define symbols
        x, y = symbols('x y')

        # Parse the expression
        expr = sympify(func_str)

        # Convert to numerical function
        f = lambdify((x, y), expr, 'numpy')

        # Test the function with dummy values
        test_result = f(1.0, 1.0)
        if not np.isfinite(test_result):
            raise ValueError("Function produces non-finite values")

        return f

    except Exception as e:
        raise ValueError(f"Invalid function: {e}")


def get_numerical_input(prompt, input_type=float, validation_func=None):
    """
    Get and validate numerical input from user

    Args:
        prompt: Input prompt string
        input_type: Type to convert input to (float, int)
        validation_func: Optional validation function

    Returns:
        Validated input value
    """
    while True:
        try:
            value = input_type(input(prompt))
            if validation_func and not validation_func(value):
                print("Invalid input. Please try again.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)


def get_method_choice():
    """
    Get user's choice of Modified Euler method variant

    Returns:
        Method string ('heun' or 'midpoint')
    """
    print("\nChoose Modified Euler method variant:")
    print("1. Heun's method (true Modified Euler)")
    print("2. RK2 Midpoint method")

    while True:
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == '1':
                return 'heun'
            elif choice == '2':
                return 'midpoint'
            else:
                print("Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)


def print_solution(solution, method_name, precision=6):
    """
    Print the solution in a formatted way

    Args:
        solution: List of (x, y) tuples
        method_name: Name of the method used
        precision: Number of decimal places for y values
    """
    print(f"\nSolution using {method_name}:")
    print("-" * 35)
    print(f"{'Step':<6} {'x':<10} {'y':<15}")
    print("-" * 35)

    for i, (x, y) in enumerate(solution):
        print(f"{i:<6} {x:<10.4f} {y:<15.{precision}f}")


def compare_methods(f, x0, y0, h, n, func_str):
    """
    Compare different methods for solving the ODE

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps
        func_str: String representation of the function
    """
    print("\n" + "=" * 60)
    print("COMPARISON OF METHODS")
    print("=" * 60)

    # Solve using different methods
    euler_sol = euler_method(f, x0, y0, h, n)
    heun_sol = modified_euler_method(f, x0, y0, h, n, 'heun')
    midpoint_sol = modified_euler_method(f, x0, y0, h, n, 'midpoint')

    # Print comparison table
    print(f"\nComparison for dy/dx = {func_str}")
    print(f"Initial condition: y({x0}) = {y0}, Step size: {h}")
    print("-" * 70)
    print(f"{'x':<10} {'Euler':<15} {'Heun':<15} {'Midpoint':<15}")
    print("-" * 70)

    for i in range(min(11, len(euler_sol))):  # Show first 10 steps + initial
        x_val = euler_sol[i][0]
        euler_y = euler_sol[i][1]
        heun_y = heun_sol[i][1]
        midpoint_y = midpoint_sol[i][1]

        print(f"{x_val:<10.4f} {euler_y:<15.6f} {heun_y:<15.6f} {midpoint_y:<15.6f}")


def validate_inputs(h, n):
    """
    Validate input parameters

    Args:
        h: Step size
        n: Number of steps

    Returns:
        True if inputs are valid, False otherwise
    """
    if h <= 0:
        print("Error: Step size must be positive")
        return False

    if n <= 0:
        print("Error: Number of steps must be positive")
        return False

    if h > 1.0:
        print("Warning: Large step size may lead to numerical instability")

    if n > 10000:
        print("Warning: Large number of steps may take significant time")

    return True


def main():
    """
    Main program function
    """
    print("=" * 60)
    print("MODIFIED EULER METHOD FOR SOLVING ODEs")
    print("=" * 60)
    print("Solve ODEs of the form: dy/dx = f(x, y)")
    print("Available methods:")
    print("  - Heun's method (Modified Euler)")
    print("  - RK2 Midpoint method")
    print("\nExamples of valid functions:")
    print("  - x + y")
    print("  - x**2 + y**2")
    print("  - sin(x) + cos(y)")
    print("  - exp(-x) * y")
    print("  - -2*x*y")
    print("-" * 60)

    try:
        # Get function input
        print("\nStep 1: Define the ODE function")
        while True:
            func_str = input("Enter dy/dx = f(x, y): ").strip()
            if func_str:
                try:
                    f = parse_function(func_str)
                    print(f"Function parsed successfully: dy/dx = {func_str}")
                    break
                except ValueError as e:
                    print(f"Error: {e}")
                    print("Please enter a valid function.")
            else:
                print("Please enter a non-empty function.")

        # Get initial conditions
        print("\nStep 2: Set initial conditions")
        x0 = get_numerical_input("Enter initial x value (x0): ", float)
        y0 = get_numerical_input("Enter initial y value (y0): ", float)

        # Get numerical parameters
        print("\nStep 3: Set numerical parameters")
        h = get_numerical_input("Enter step size (h): ", float, lambda x: x > 0)
        n = get_numerical_input("Enter number of steps: ", int, lambda x: x > 0)

        # Validate inputs
        if not validate_inputs(h, n):
            return

        # Get method choice
        method = get_method_choice()
        method_name = "Heun's method" if method == 'heun' else "RK2 Midpoint method"

        # Display problem summary
        print("\nProblem Summary:")
        print(f"ODE: dy/dx = {func_str}")
        print(f"Initial condition: y({x0}) = {y0}")
        print(f"Method: {method_name}")
        print(f"Step size: {h}")
        print(f"Number of steps: {n}")
        print(f"Final x value: {x0 + n * h}")

        # Solve the ODE
        print("\nSolving...")
        solution = modified_euler_method(f, x0, y0, h, n, method)

        # Display results
        print_solution(solution, method_name)

        # Display final result
        final_x, final_y = solution[-1]
        print(f"\nFinal result: y({final_x:.4f}) ≈ {final_y:.6f}")

        # Ask if user wants to compare methods
        if n <= 20:  # Only offer comparison for small problems
            compare_choice = input("\nWould you like to compare with other methods? (y/n): ").strip().lower()
            if compare_choice == 'y':
                compare_methods(f, x0, y0, h, n, func_str)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())