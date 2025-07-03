#!/usr/bin/env python3

import numpy as np
from sympy import symbols, lambdify, sympify
import sys


def euler_method(f, x0, y0, h, n):
    """
    Solve ODE using Euler's method

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps

    Returns:
        List of tuples (x, y) representing the solution
    """
    # Pre-allocate arrays for better performance and easier translation
    x_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # Set initial conditions
    x_values[0] = x0
    y_values[0] = y0

    # Euler method iterations
    for i in range(n):
        # Calculate derivative at current point
        derivative = f(x_values[i], y_values[i])

        # Update y using Euler's formula: y_{n+1} = y_n + h * f(x_n, y_n)
        y_values[i + 1] = y_values[i] + h * derivative

        # Update x
        x_values[i + 1] = x_values[i] + h

    # Return as list of tuples for compatibility
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


def print_solution(solution, precision=6):
    """
    Print the solution in a formatted way

    Args:
        solution: List of (x, y) tuples
        precision: Number of decimal places for y values
    """
    print("\nSolution:")
    print("-" * 30)
    print(f"{'x':<10} {'y':<15}")
    print("-" * 30)

    for i, (x, y) in enumerate(solution):
        print(f"{x:<10.4f} {y:<15.{precision}f}")


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
    print("=" * 50)
    print("EULER METHOD FOR SOLVING ODEs")
    print("=" * 50)
    print("Solve ODEs of the form: dy/dx = f(x, y)")
    print("Examples of valid functions:")
    print("  - x + y")
    print("  - x**2 + y**2")
    print("  - sin(x) + cos(y)")
    print("  - exp(-x) * y")
    print("-" * 50)

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

        # Display problem summary
        print("\nProblem Summary:")
        print(f"ODE: dy/dx = {func_str}")
        print(f"Initial condition: y({x0}) = {y0}")
        print(f"Step size: {h}")
        print(f"Number of steps: {n}")
        print(f"Final x value: {x0 + n * h}")

        # Solve the ODE
        print("\nSolving...")
        solution = euler_method(f, x0, y0, h, n)

        # Display results
        print_solution(solution)

        # Display final result
        final_x, final_y = solution[-1]
        print(f"\nFinal result: y({final_x:.4f}) ≈ {final_y:.6f}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())