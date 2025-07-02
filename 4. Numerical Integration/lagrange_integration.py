#!/usr/bin/env python3

import math


def lagrange_interpolation(x_points, y_points, x_eval):
    """
    Evaluate Lagrange interpolating polynomial at a given point.

    Args:
        x_points: List of x-coordinates of interpolation points
        y_points: List of y-coordinates of interpolation points
        x_eval: Point at which to evaluate the polynomial

    Returns:
        Value of interpolating polynomial at x_eval
    """
    n = len(x_points)
    if n != len(y_points):
        raise ValueError("x_points and y_points must have the same length")

    if n == 0:
        raise ValueError("At least one interpolation point is required")

    result = 0.0

    for i in range(n):
        # Compute Lagrange basis polynomial L_i(x_eval)
        basis_term = 1.0

        for j in range(n):
            if i != j:
                denominator = x_points[i] - x_points[j]
                if abs(denominator) < 1e-15:
                    raise ValueError(f"Duplicate x-coordinates at indices {i} and {j}")

                basis_term *= (x_eval - x_points[j]) / denominator

        # Add contribution from this basis function
        result += y_points[i] * basis_term

    return result


def composite_lagrange_integration(func, a, b, n_intervals, points_per_interval=3):
    """
    Numerical integration using composite Lagrange interpolation.

    Args:
        func: Function to integrate
        a: Lower bound
        b: Upper bound
        n_intervals: Number of subintervals
        points_per_interval: Number of interpolation points per subinterval

    Returns:
        Approximate integral value
    """
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")

    if n_intervals <= 0:
        raise ValueError("Number of intervals must be positive")

    if points_per_interval < 2:
        raise ValueError("At least 2 points per interval required")

    if points_per_interval > 5:
        raise ValueError("Maximum 5 points per interval for numerical stability")

    interval_width = (b - a) / n_intervals
    total_integral = 0.0

    for i in range(n_intervals):
        # Define subinterval bounds
        xi_start = a + i * interval_width
        xi_end = a + (i + 1) * interval_width

        # Generate interpolation points within this subinterval
        x_points = []
        y_points = []

        for j in range(points_per_interval):
            if points_per_interval == 2:
                # Linear interpolation - use endpoints
                if j == 0:
                    x_point = xi_start
                else:
                    x_point = xi_end
            else:
                # Distribute points evenly within the subinterval
                x_point = xi_start + j * interval_width / (points_per_interval - 1)

            x_points.append(x_point)
            y_points.append(func(x_point))

        # Integrate the Lagrange polynomial over this subinterval
        subinterval_integral = integrate_lagrange_polynomial(
            x_points, y_points, xi_start, xi_end
        )

        total_integral += subinterval_integral

    return total_integral


def integrate_lagrange_polynomial(x_points, y_points, a, b, n_quad_points=100):
    """
    Integrate a Lagrange polynomial over [a, b] using numerical quadrature.

    Args:
        x_points: X-coordinates of interpolation points
        y_points: Y-coordinates of interpolation points
        a: Lower integration bound
        b: Upper integration bound
        n_quad_points: Number of quadrature points for integration

    Returns:
        Integral of the Lagrange polynomial
    """
    if a >= b:
        return 0.0

    # Use trapezoidal rule for numerical integration of the polynomial
    h = (b - a) / n_quad_points
    integral_sum = 0.0

    # First point
    x_eval = a
    y_eval = lagrange_interpolation(x_points, y_points, x_eval)
    integral_sum += y_eval

    # Middle points
    for i in range(1, n_quad_points):
        x_eval = a + i * h
        y_eval = lagrange_interpolation(x_points, y_points, x_eval)
        integral_sum += 2.0 * y_eval

    # Last point
    x_eval = b
    y_eval = lagrange_interpolation(x_points, y_points, x_eval)
    integral_sum += y_eval

    # Apply trapezoidal rule
    integral = (h / 2.0) * integral_sum

    return integral


def simple_lagrange_integration(func, a, b, n_points):
    """
    Simple Lagrange interpolation integration over entire interval.

    Args:
        func: Function to integrate
        a: Lower bound
        b: Upper bound
        n_points: Number of interpolation points

    Returns:
        Approximate integral value
    """
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")

    if n_points < 2:
        raise ValueError("At least 2 interpolation points required")

    if n_points > 10:
        raise ValueError("Maximum 10 points recommended for numerical stability")

    # Generate interpolation points
    x_points = []
    y_points = []

    for i in range(n_points):
        if n_points == 2:
            # Use endpoints for linear interpolation
            x_point = a if i == 0 else b
        else:
            # Distribute points evenly
            x_point = a + i * (b - a) / (n_points - 1)

        x_points.append(x_point)
        y_points.append(func(x_point))

    # Integrate the Lagrange polynomial
    integral = integrate_lagrange_polynomial(x_points, y_points, a, b)

    return integral


def validate_inputs(a, b, n_points, n_intervals=None):
    """
    Validate numerical inputs for integration parameters.

    Args:
        a: Lower bound
        b: Upper bound
        n_points: Number of interpolation points
        n_intervals: Number of intervals (optional)

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False, "Integration bounds must be numeric"

    if not isinstance(n_points, int):
        return False, "Number of interpolation points must be an integer"

    if n_intervals is not None and not isinstance(n_intervals, int):
        return False, "Number of intervals must be an integer"

    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return False, "Integration bounds cannot be NaN or infinite"

    if a >= b:
        return False, "Lower bound must be less than upper bound"

    if n_points < 2:
        return False, "At least 2 interpolation points required"

    if n_intervals is not None and n_intervals <= 0:
        return False, "Number of intervals must be positive"

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
        tuple: (function, func_expr, a, b, method_choice, parameters) or None
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

        # Choose integration method
        print("\nChoose integration method:")
        print("1. Simple Lagrange interpolation (single polynomial)")
        print("2. Composite Lagrange interpolation (multiple subintervals)")

        method_choice = input("Enter choice (1 or 2): ").strip()

        if method_choice == "1":
            n_points = int(input("Enter number of interpolation points (2-10): "))

            # Validate inputs
            is_valid, error_msg = validate_inputs(a, b, n_points)
            if not is_valid:
                print(f"Error: {error_msg}")
                return None

            if n_points > 10:
                print("Warning: High-degree polynomials may be numerically unstable")
                proceed = input("Continue anyway? (y/n): ").lower()
                if proceed not in ['y', 'yes']:
                    return None

            return func, func_expr, a, b, "simple", (n_points,)

        elif method_choice == "2":
            n_intervals = int(input("Enter number of subintervals: "))
            points_per_interval = int(input("Enter points per subinterval (2-5): "))

            # Validate inputs
            is_valid, error_msg = validate_inputs(a, b, points_per_interval, n_intervals)
            if not is_valid:
                print(f"Error: {error_msg}")
                return None

            return func, func_expr, a, b, "composite", (n_intervals, points_per_interval)

        else:
            print("Error: Invalid method choice")
            return None

    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        return None
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return None


def display_results(func_expr, a, b, method, parameters, result):
    """
    Display integration results in a formatted manner.

    Args:
        func_expr: Original function expression
        a: Lower bound
        b: Upper bound
        method: Integration method used
        parameters: Method parameters
        result: Integration result
    """
    print("\n" + "=" * 60)
    print("LAGRANGE INTERPOLATION INTEGRATION RESULTS")
    print("=" * 60)
    print(f"Function: f(x) = {func_expr}")
    print(f"Integration bounds: [{a}, {b}]")

    if method == "simple":
        n_points = parameters[0]
        print(f"Method: Simple Lagrange interpolation")
        print(f"Interpolation points: {n_points}")
        print(f"Polynomial degree: {n_points - 1}")

    elif method == "composite":
        n_intervals, points_per_interval = parameters
        print(f"Method: Composite Lagrange interpolation")
        print(f"Number of subintervals: {n_intervals}")
        print(f"Points per subinterval: {points_per_interval}")
        print(f"Total interpolation points: {n_intervals * points_per_interval}")

    print(f"Approximate integral: {result:.10f}")
    print("=" * 60)


def run_integration():
    """
    Main integration workflow function.

    Returns:
        bool: True if successful, False otherwise
    """
    print("LAGRANGE INTERPOLATION NUMERICAL INTEGRATION")
    print("-" * 50)

    # Get user inputs
    inputs = get_user_inputs()
    if inputs is None:
        return False

    func, func_expr, a, b, method, parameters = inputs

    try:
        if method == "simple":
            n_points = parameters[0]
            result = simple_lagrange_integration(func, a, b, n_points)

        elif method == "composite":
            n_intervals, points_per_interval = parameters
            result = composite_lagrange_integration(func, a, b, n_intervals, points_per_interval)

        else:
            print("Error: Unknown integration method")
            return False

        display_results(func_expr, a, b, method, parameters, result)
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
    """Test function: f(x) = 1/(1+x^2)"""
    return 1.0 / (1.0 + x ** 2)


if __name__ == "__main__":
    main()