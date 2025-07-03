#!/usr/bin/env python3

import numpy as np
import math


def lagrange_interpolation(x, y, target_x):
    """
    Compute Lagrange interpolation at target_x using given data points.

    Args:
        x: array of x-coordinates
        y: array of y-coordinates
        target_x: point where interpolation is needed

    Returns:
        Interpolated value at target_x
    """
    n = len(x)
    result = 0.0

    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (target_x - x[j]) / (x[i] - x[j])
        result += term

    return result


def compute_derivative_finite_difference(x, y, target_x, order=1, h=None):
    """
    Compute numerical derivative using finite difference method.

    Args:
        x: array of x-coordinates
        y: array of y-coordinates
        target_x: point where derivative is needed
        order: derivative order (1 or 2)
        h: step size (if None, uses average spacing)

    Returns:
        Numerical derivative at target_x
    """
    if h is None:
        h = (x[-1] - x[0]) / (len(x) - 1)  # Average spacing

    if order == 1:
        # First derivative using central difference
        y_plus = lagrange_interpolation(x, y, target_x + h)
        y_minus = lagrange_interpolation(x, y, target_x - h)
        return (y_plus - y_minus) / (2 * h)
    elif order == 2:
        # Second derivative using central difference
        y_plus = lagrange_interpolation(x, y, target_x + h)
        y_center = lagrange_interpolation(x, y, target_x)
        y_minus = lagrange_interpolation(x, y, target_x - h)
        return (y_plus - 2 * y_center + y_minus) / (h ** 2)
    else:
        raise ValueError("Only first and second order derivatives are supported.")


def compute_derivative_direct(x, y, target_x, order=1):
    """
    Compute numerical derivative by differentiating Lagrange polynomial directly.

    Args:
        x: array of x-coordinates
        y: array of y-coordinates
        target_x: point where derivative is needed
        order: derivative order (1 or 2)

    Returns:
        Numerical derivative at target_x
    """
    n = len(x)

    if order == 1:
        # First derivative of Lagrange polynomial
        result = 0.0
        for i in range(n):
            # Derivative of i-th Lagrange basis polynomial
            basis_derivative = 0.0
            for k in range(n):
                if k != i:
                    product = 1.0
                    for j in range(n):
                        if j != i and j != k:
                            product *= (target_x - x[j]) / (x[i] - x[j])
                    basis_derivative += product / (x[i] - x[k])
            result += y[i] * basis_derivative
        return result

    elif order == 2:
        # Second derivative of Lagrange polynomial
        result = 0.0
        for i in range(n):
            # Second derivative of i-th Lagrange basis polynomial
            basis_second_derivative = 0.0
            for k in range(n):
                if k != i:
                    for m in range(n):
                        if m != i and m != k:
                            product = 2.0
                            for j in range(n):
                                if j != i and j != k and j != m:
                                    product *= (target_x - x[j]) / (x[i] - x[j])
                            basis_second_derivative += product / ((x[i] - x[k]) * (x[i] - x[m]))
            result += y[i] * basis_second_derivative
        return result

    else:
        raise ValueError("Only first and second order derivatives are supported.")


def evaluate_function(func_expr, x_vals):
    """
    Evaluate a mathematical function at given x values.

    Args:
        func_expr: function expression as string
        x_vals: array of x values

    Returns:
        Array of function values
    """
    try:
        # Try using sympy for symbolic evaluation
        from sympy import symbols, lambdify, sympify
        x = symbols('x')
        expr = sympify(func_expr)
        func = lambdify(x, expr, 'numpy')
        return func(x_vals)
    except ImportError:
        # Fallback to basic evaluation if sympy not available
        print("Warning: sympy not available. Using basic evaluation.")
        # Simple evaluation for basic functions
        y_vals = []
        for x_val in x_vals:
            # Replace 'x' with actual value and evaluate
            expr = func_expr.replace('x', str(x_val))
            # Basic safety check
            if any(dangerous in expr for dangerous in ['import', 'exec', 'eval', '__']):
                raise ValueError("Invalid function expression")
            y_vals.append(eval(expr))
        return np.array(y_vals)


def get_user_input():
    """
    Get user input for the numerical derivative calculation.

    Returns:
        Dictionary containing all input parameters
    """
    print("=== Numerical Derivative Calculator ===\n")

    # Function input
    func_expr = input("Enter the function expression (use 'x' as variable, e.g., 'x**2 + 3*x + 1'): ")

    # Interval and points
    a = float(input("Enter the start of the interval (a): "))
    b = float(input("Enter the end of the interval (b): "))

    if a >= b:
        raise ValueError("Interval start must be less than end.")

    n = int(input("Enter the number of interpolation points: "))
    if n < 2:
        raise ValueError("At least 2 interpolation points are required.")

    # Target point for derivative
    target_x = float(input("Enter the x value where the derivative is to be computed: "))
    if target_x < a or target_x > b:
        raise ValueError("Target x must be within the interval [a, b].")

    # Method selection
    print("\nSelect derivative computation method:")
    print("1. Finite Difference (using interpolation)")
    print("2. Direct Differentiation (of Lagrange polynomial)")
    method = int(input("Enter method choice (1 or 2): "))

    if method not in [1, 2]:
        raise ValueError("Invalid method choice. Use 1 or 2.")

    return {
        'func_expr': func_expr,
        'a': a,
        'b': b,
        'n': n,
        'target_x': target_x,
        'method': method
    }


def compute_and_display_results(params):
    """
    Compute derivatives and display results.

    Args:
        params: dictionary containing input parameters
    """
    # Generate interpolation points
    x_vals = np.linspace(params['a'], params['b'], params['n'])
    y_vals = evaluate_function(params['func_expr'], x_vals)

    # Compute derivatives based on selected method
    if params['method'] == 1:
        first_derivative = compute_derivative_finite_difference(x_vals, y_vals, params['target_x'], order=1)
        second_derivative = compute_derivative_finite_difference(x_vals, y_vals, params['target_x'], order=2)
        method_name = "Finite Difference"
    else:
        first_derivative = compute_derivative_direct(x_vals, y_vals, params['target_x'], order=1)
        second_derivative = compute_derivative_direct(x_vals, y_vals, params['target_x'], order=2)
        method_name = "Direct Differentiation"

    # Display results
    print(f"\n=== Results using {method_name} Method ===")
    print(f"Function: f(x) = {params['func_expr']}")
    print(f"Interval: [{params['a']}, {params['b']}]")
    print(f"Interpolation points: {params['n']}")
    print(f"Target point: x = {params['target_x']}")
    print(f"\nFirst derivative at x = {params['target_x']}: {first_derivative:.8f}")
    print(f"Second derivative at x = {params['target_x']}: {second_derivative:.8f}")

    # Display interpolation points for reference
    print(f"\nInterpolation points used:")
    for i in range(min(5, len(x_vals))):  # Show first 5 points
        print(f"  x[{i}] = {x_vals[i]:.4f}, f(x[{i}]) = {y_vals[i]:.6f}")
    if len(x_vals) > 5:
        print(f"  ... and {len(x_vals) - 5} more points")


def main():
    """
    Main function to run the numerical derivative calculator.
    """
    try:
        # Get user input
        params = get_user_input()

        # Compute and display results
        compute_and_display_results(params)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your input and try again.")


if __name__ == "__main__":
    main()