#!/usr/bin/env python3

import numpy as np
import math


def trapezoidal_rule(func, a, b, n):
    """
    Compute numerical integration using the Trapezoidal Rule.

    Args:
        func: Function to integrate (callable)
        a: Lower bound of integration
        b: Upper bound of integration
        n: Number of subintervals

    Returns:
        Approximate value of the definite integral
    """
    if n <= 0:
        raise ValueError("Number of subintervals must be positive")
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")

    h = (b - a) / n

    # Calculate function values at all points
    integral_sum = 0.0

    # First point (coefficient = 1)
    integral_sum += func(a)

    # Middle points (coefficient = 2)
    for i in range(1, n):
        x_i = a + i * h
        integral_sum += 2.0 * func(x_i)

    # Last point (coefficient = 1)
    integral_sum += func(b)

    # Apply trapezoidal rule formula
    integral = (h / 2.0) * integral_sum

    return integral


def validate_inputs(a, b, n):
    """
    Validate numerical inputs for integration bounds and subintervals.

    Args:
        a: Lower bound
        b: Upper bound
        n: Number of subintervals

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False, "Integration bounds must be numeric"

    if not isinstance(n, int):
        return False, "Number of subintervals must be an integer"

    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return False, "Integration bounds cannot be NaN or infinite"

    if a >= b:
        return False, "Lower bound must be less than upper bound"

    if n <= 0:
        return False, "Number of subintervals must be positive"

    return True, ""


def create_function_from_expression(expr_str):
    """
    Create a callable function from a string expression.

    Args:
        expr_str: String representation of mathematical function

    Returns:
        Callable function that takes a single numeric argument
    """
    try:
        from sympy import symbols, lambdify, sympify
        x = symbols('x')

        # Parse and validate the expression
        expr = sympify(expr_str)

        # Create numpy-compatible function
        func = lambdify(x, expr, ['numpy', 'math'])

        # Test the function with a simple value
        test_val = func(1.0)
        if math.isnan(test_val) or math.isinf(test_val):
            raise ValueError("Function produces invalid values")

        return func

    except Exception as e:
        raise ValueError(f"Invalid function expression: {e}")


def get_user_inputs():
    """
    Get and validate user inputs for integration parameters.

    Returns:
        tuple: (function, a, b, n) or None if invalid input
    """
    try:
        # Get function expression
        print("Enter a mathematical function to integrate.")
        print("Examples: x**2, sin(x), exp(x), x*cos(x)")
        func_expr = input("Function f(x) = ").strip()

        if not func_expr:
            print("Error: Function expression cannot be empty")
            return None

        func = create_function_from_expression(func_expr)

        # Get integration bounds
        a = float(input("Enter lower bound (a): "))
        b = float(input("Enter upper bound (b): "))

        # Get number of subintervals
        n = int(input("Enter number of subintervals (n): "))

        # Validate inputs
        is_valid, error_msg = validate_inputs(a, b, n)
        if not is_valid:
            print(f"Error: {error_msg}")
            return None

        return func, a, b, n

    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        return None
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return None


def display_results(func_expr, a, b, n, result):
    """
    Display integration results in a formatted manner.

    Args:
        func_expr: Original function expression string
        a: Lower bound
        b: Upper bound
        n: Number of subintervals
        result: Computed integral value
    """
    print("\n" + "=" * 50)
    print("NUMERICAL INTEGRATION RESULTS")
    print("=" * 50)
    print(f"Function: f(x) = {func_expr}")
    print(f"Integration bounds: [{a}, {b}]")
    print(f"Number of subintervals: {n}")
    print(f"Step size (h): {(b - a) / n:.6f}")
    print(f"Approximate integral: {result:.8f}")
    print("=" * 50)


def run_integration():
    """
    Main integration workflow function.

    Returns:
        bool: True if successful, False otherwise
    """
    print("TRAPEZOIDAL RULE NUMERICAL INTEGRATION")
    print("-" * 40)

    # Get user inputs
    inputs = get_user_inputs()
    if inputs is None:
        return False

    func, a, b, n = inputs

    try:
        # Compute integral using trapezoidal rule
        result = trapezoidal_rule(func, a, b, n)

        # Display results
        display_results("user_function", a, b, n, result)

        return True

    except Exception as e:
        print(f"Error during computation: {e}")
        return False


def main():
    """
    Main program entry point with error handling and user interaction.
    """
    try:
        success = run_integration()

        if success:
            print("\nIntegration completed successfully!")
        else:
            print("\nIntegration failed. Please check your inputs and try again.")

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


# Example functions for testing (not part of main execution)
def test_function_1(x):
    """Test function: f(x) = x^2"""
    return x ** 2


def test_function_2(x):
    """Test function: f(x) = sin(x)"""
    return np.sin(x)


def test_function_3(x):
    """Test function: f(x) = e^x"""
    return np.exp(x)


if __name__ == "__main__":
    main()