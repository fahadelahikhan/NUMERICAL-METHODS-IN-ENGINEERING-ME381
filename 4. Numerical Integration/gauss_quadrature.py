#!/usr/bin/env python3

import math


def gauss_legendre_nodes_weights(n):
    """
    Compute Gauss-Legendre quadrature nodes and weights for interval [-1, 1].

    Args:
        n: Number of quadrature points

    Returns:
        tuple: (nodes, weights) arrays
    """
    if n <= 0:
        raise ValueError("Number of quadrature points must be positive")

    nodes = []
    weights = []

    # Pre-computed values for common orders (easy to translate to other languages)
    if n == 1:
        nodes = [0.0]
        weights = [2.0]
    elif n == 2:
        nodes = [-0.5773502691896257, 0.5773502691896257]
        weights = [1.0, 1.0]
    elif n == 3:
        nodes = [-0.7745966692414834, 0.0, 0.7745966692414834]
        weights = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]
    elif n == 4:
        nodes = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
        weights = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
    elif n == 5:
        nodes = [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
        weights = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]
    else:
        # For higher orders, use iterative method (Newton-Raphson)
        nodes, weights = compute_legendre_nodes_weights_iterative(n)

    return nodes, weights


def compute_legendre_nodes_weights_iterative(n):
    """
    Compute Gauss-Legendre nodes and weights using iterative method.

    Args:
        n: Number of quadrature points

    Returns:
        tuple: (nodes, weights) arrays
    """
    nodes = []
    weights = []

    # Initial guess for roots
    for i in range(n):
        # Initial approximation for the i-th root
        x = math.cos(math.pi * (i + 0.75) / (n + 0.5))

        # Newton-Raphson iteration
        for _ in range(10):  # Usually converges quickly
            p1 = 1.0
            p2 = 0.0

            # Compute Legendre polynomial and its derivative
            for j in range(n):
                p3 = p2
                p2 = p1
                p1 = ((2 * j + 1) * x * p2 - j * p3) / (j + 1)

            # Derivative of Legendre polynomial
            pp = n * (x * p1 - p2) / (x * x - 1)

            # Newton-Raphson update
            x1 = x
            x = x1 - p1 / pp

            if abs(x - x1) < 1e-14:
                break

        nodes.append(x)
        weights.append(2.0 / ((1 - x * x) * pp * pp))

    return nodes, weights


def gauss_legendre_quadrature(func, a, b, n):
    """
    Compute integral using Gauss-Legendre quadrature.

    Args:
        func: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of quadrature points

    Returns:
        Approximate integral value
    """
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")

    # Get nodes and weights for [-1, 1]
    nodes, weights = gauss_legendre_nodes_weights(n)

    # Transform from [-1, 1] to [a, b]
    integral_sum = 0.0
    transformation_factor = (b - a) / 2.0

    for i in range(len(nodes)):
        # Transform node from [-1, 1] to [a, b]
        x_transformed = a + (b - a) * (nodes[i] + 1) / 2.0

        # Add weighted function value
        integral_sum += weights[i] * func(x_transformed)

    # Apply transformation factor
    integral = transformation_factor * integral_sum

    return integral


def adaptive_gauss_legendre(func, a, b, n, tolerance=1e-6, max_subdivisions=100):
    """
    Adaptive Gauss-Legendre quadrature with error control.

    Args:
        func: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of quadrature points per interval
        tolerance: Relative error tolerance
        max_subdivisions: Maximum number of subdivisions

    Returns:
        tuple: (integral_value, estimated_error, subdivisions_used)
    """
    # Initial estimate
    I1 = gauss_legendre_quadrature(func, a, b, n)

    # Estimate with doubled points
    I2 = gauss_legendre_quadrature(func, a, b, min(n * 2, 10))

    # Estimate error
    error = abs(I2 - I1)

    if error < tolerance * abs(I2) or max_subdivisions <= 0:
        return I2, error, 1

    # Subdivide interval
    mid = (a + b) / 2.0

    # Recursively integrate both halves
    I_left, error_left, sub_left = adaptive_gauss_legendre(
        func, a, mid, n, tolerance / 2, max_subdivisions // 2
    )

    I_right, error_right, sub_right = adaptive_gauss_legendre(
        func, mid, b, n, tolerance / 2, max_subdivisions // 2
    )

    total_integral = I_left + I_right
    total_error = error_left + error_right
    total_subdivisions = sub_left + sub_right

    return total_integral, total_error, total_subdivisions


def validate_inputs(a, b, n):
    """
    Validate numerical inputs for integration parameters.

    Args:
        a: Lower bound
        b: Upper bound
        n: Number of quadrature points

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False, "Integration bounds must be numeric"

    if not isinstance(n, int):
        return False, "Number of quadrature points must be an integer"

    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return False, "Integration bounds cannot be NaN or infinite"

    if a >= b:
        return False, "Lower bound must be less than upper bound"

    if n <= 0:
        return False, "Number of quadrature points must be positive"

    if n > 10:
        return False, "Maximum supported order is 10 for stability"

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

        # Create function compatible with basic math operations
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
        tuple: (function, func_expr, a, b, n, use_adaptive) or None if invalid
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

        # Get number of quadrature points
        n = int(input("Enter number of quadrature points (1-10): "))

        # Ask about adaptive integration
        adaptive_choice = input("Use adaptive integration? (y/n): ").lower().strip()
        use_adaptive = adaptive_choice in ['y', 'yes']

        # Validate inputs
        is_valid, error_msg = validate_inputs(a, b, n)
        if not is_valid:
            print(f"Error: {error_msg}")
            return None

        return func, func_expr, a, b, n, use_adaptive

    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        return None
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return None


def display_results(func_expr, a, b, n, result, error=None, subdivisions=None, use_adaptive=False):
    """
    Display integration results in a formatted manner.

    Args:
        func_expr: Original function expression
        a: Lower bound
        b: Upper bound
        n: Number of quadrature points
        result: Integration result
        error: Estimated error (for adaptive method)
        subdivisions: Number of subdivisions used (for adaptive method)
        use_adaptive: Whether adaptive method was used
    """
    print("\n" + "=" * 60)
    print("GAUSS-LEGENDRE QUADRATURE INTEGRATION RESULTS")
    print("=" * 60)
    print(f"Function: f(x) = {func_expr}")
    print(f"Integration bounds: [{a}, {b}]")
    print(f"Quadrature points: {n}")

    if use_adaptive:
        print(f"Method: Adaptive Gauss-Legendre")
        print(f"Subdivisions used: {subdivisions}")
        print(f"Estimated error: {error:.2e}")
    else:
        print(f"Method: Standard Gauss-Legendre")

    print(f"Approximate integral: {result:.10f}")
    print("=" * 60)


def run_integration():
    """
    Main integration workflow function.

    Returns:
        bool: True if successful, False otherwise
    """
    print("GAUSS-LEGENDRE QUADRATURE NUMERICAL INTEGRATION")
    print("-" * 50)

    # Get user inputs
    inputs = get_user_inputs()
    if inputs is None:
        return False

    func, func_expr, a, b, n, use_adaptive = inputs

    try:
        if use_adaptive:
            # Use adaptive integration
            result, error, subdivisions = adaptive_gauss_legendre(func, a, b, n)
            display_results(func_expr, a, b, n, result, error, subdivisions, True)
        else:
            # Use standard integration
            result = gauss_legendre_quadrature(func, a, b, n)
            display_results(func_expr, a, b, n, result, use_adaptive=False)

        return True

    except Exception as e:
        print(f"Error during computation: {e}")
        return False


def main():
    """
    Main program entry point with error handling.
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


# Test functions for verification
def test_function_1(x):
    """Test function: f(x) = x^2"""
    return x ** 2


def test_function_2(x):
    """Test function: f(x) = sin(x)"""
    return math.sin(x)


def test_function_3(x):
    """Test function: f(x) = e^(-x^2)"""
    return math.exp(-x ** 2)


if __name__ == "__main__":
    main()