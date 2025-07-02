#!/usr/bin/env python3

import math


def simpsons_1_3_rule(func, a, b, n):
    """
    Compute numerical integration using Simpson's 1/3 Rule.

    Args:
        func: Function to integrate (callable)
        a: Lower bound of integration
        b: Upper bound of integration
        n: Number of subintervals (must be even)

    Returns:
        Approximate value of the definite integral
    """
    if n <= 0:
        raise ValueError("Number of subintervals must be positive")
    if n % 2 != 0:
        raise ValueError("Number of subintervals must be even for Simpson's 1/3 rule")
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")

    h = (b - a) / n
    integral_sum = 0.0

    # First point: coefficient = 1
    integral_sum += func(a)

    # Odd indexed points: coefficient = 4
    for i in range(1, n, 2):
        x_i = a + i * h
        integral_sum += 4.0 * func(x_i)

    # Even indexed points (except first and last): coefficient = 2
    for i in range(2, n, 2):
        x_i = a + i * h
        integral_sum += 2.0 * func(x_i)

    # Last point: coefficient = 1
    integral_sum += func(b)

    # Apply Simpson's 1/3 rule formula
    integral = (h / 3.0) * integral_sum

    return integral


def simpsons_3_8_rule(func, a, b, n):
    """
    Compute numerical integration using Simpson's 3/8 Rule.

    Args:
        func: Function to integrate (callable)
        a: Lower bound of integration
        b: Upper bound of integration
        n: Number of subintervals (must be multiple of 3)

    Returns:
        Approximate value of the definite integral
    """
    if n <= 0:
        raise ValueError("Number of subintervals must be positive")
    if n % 3 != 0:
        raise ValueError("Number of subintervals must be multiple of 3 for Simpson's 3/8 rule")
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")

    h = (b - a) / n
    integral_sum = 0.0

    # First point: coefficient = 1
    integral_sum += func(a)

    # Process points in groups of 3
    for i in range(1, n):
        x_i = a + i * h

        if i % 3 == 0:
            # Points at positions 3, 6, 9, ... (multiples of 3): coefficient = 2
            integral_sum += 2.0 * func(x_i)
        else:
            # Points at positions 1, 2, 4, 5, 7, 8, ... (non-multiples of 3): coefficient = 3
            integral_sum += 3.0 * func(x_i)

    # Last point: coefficient = 1
    integral_sum += func(b)

    # Apply Simpson's 3/8 rule formula
    integral = (3.0 * h / 8.0) * integral_sum

    return integral


def validate_inputs(a, b, n):
    """
    Validate numerical inputs for integration parameters.

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

        # Create function compatible with both numpy and math
        func = lambdify(x, expr, ['math', 'numpy'])

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
        tuple: (function, a, b, n_1_3, n_3_8) or None if invalid input
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

        # Validate basic inputs
        is_valid, error_msg = validate_inputs(a, b, 1)
        if not is_valid:
            print(f"Error: {error_msg}")
            return None

        # Get number of subintervals for 1/3 rule
        print("\nFor Simpson's 1/3 rule:")
        n_1_3 = int(input("Enter number of subintervals (must be even): "))

        # Get number of subintervals for 3/8 rule
        print("\nFor Simpson's 3/8 rule:")
        n_3_8 = int(input("Enter number of subintervals (must be multiple of 3): "))

        return func, func_expr, a, b, n_1_3, n_3_8

    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        return None
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return None


def display_results(func_expr, a, b, n_1_3, result_1_3, n_3_8, result_3_8):
    """
    Display integration results in a formatted manner.

    Args:
        func_expr: Original function expression string
        a: Lower bound
        b: Upper bound
        n_1_3: Number of subintervals for 1/3 rule
        result_1_3: Result from 1/3 rule (or None if failed)
        n_3_8: Number of subintervals for 3/8 rule
        result_3_8: Result from 3/8 rule (or None if failed)
    """
    print("\n" + "=" * 60)
    print("SIMPSON'S RULE NUMERICAL INTEGRATION RESULTS")
    print("=" * 60)
    print(f"Function: f(x) = {func_expr}")
    print(f"Integration bounds: [{a}, {b}]")
    print("-" * 60)

    if result_1_3 is not None:
        print(f"Simpson's 1/3 Rule:")
        print(f"  Subintervals: {n_1_3}")
        print(f"  Step size (h): {(b - a) / n_1_3:.6f}")
        print(f"  Approximate integral: {result_1_3:.8f}")
    else:
        print(f"Simpson's 1/3 Rule: Failed (check that n={n_1_3} is even)")

    print("-" * 60)

    if result_3_8 is not None:
        print(f"Simpson's 3/8 Rule:")
        print(f"  Subintervals: {n_3_8}")
        print(f"  Step size (h): {(b - a) / n_3_8:.6f}")
        print(f"  Approximate integral: {result_3_8:.8f}")
    else:
        print(f"Simpson's 3/8 Rule: Failed (check that n={n_3_8} is multiple of 3)")

    print("=" * 60)


def run_integration():
    """
    Main integration workflow function.

    Returns:
        bool: True if at least one method succeeded, False otherwise
    """
    print("SIMPSON'S RULE NUMERICAL INTEGRATION")
    print("(Both 1/3 and 3/8 rules)")
    print("-" * 40)

    # Get user inputs
    inputs = get_user_inputs()
    if inputs is None:
        return False

    func, func_expr, a, b, n_1_3, n_3_8 = inputs

    # Attempt Simpson's 1/3 rule
    result_1_3 = None
    try:
        result_1_3 = simpsons_1_3_rule(func, a, b, n_1_3)
    except ValueError as e:
        print(f"\nError with Simpson's 1/3 rule: {e}")
    except Exception as e:
        print(f"\nUnexpected error with Simpson's 1/3 rule: {e}")

    # Attempt Simpson's 3/8 rule
    result_3_8 = None
    try:
        result_3_8 = simpsons_3_8_rule(func, a, b, n_3_8)
    except ValueError as e:
        print(f"\nError with Simpson's 3/8 rule: {e}")
    except Exception as e:
        print(f"\nUnexpected error with Simpson's 3/8 rule: {e}")

    # Display results
    display_results(func_expr, a, b, n_1_3, result_1_3, n_3_8, result_3_8)

    # Return success if at least one method worked
    return result_1_3 is not None or result_3_8 is not None


def main():
    """
    Main program entry point with error handling and user interaction.
    """
    try:
        success = run_integration()

        if success:
            print("\nIntegration completed!")
        else:
            print("\nBoth integration methods failed. Please check your inputs.")

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


# Test functions for verification (not part of main execution)
def test_function_1(x):
    """Test function: f(x) = x^2"""
    return x ** 2


def test_function_2(x):
    """Test function: f(x) = sin(x)"""
    return math.sin(x)


def test_function_3(x):
    """Test function: f(x) = e^x"""
    return math.exp(x)


if __name__ == "__main__":
    main()