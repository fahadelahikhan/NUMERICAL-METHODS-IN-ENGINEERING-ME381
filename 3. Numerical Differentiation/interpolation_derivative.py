#!/usr/bin/env python3

import numpy as np
import math


def lagrange_interpolation_derivative(x, y, target_x, order=1):
    """
    Compute derivative of Lagrange interpolating polynomial at target_x.

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
            # Derivative of i-th Lagrange basis polynomial L_i'(x)
            basis_derivative = 0.0
            for k in range(n):
                if k != i:
                    # Product rule for derivative of product
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
            # Second derivative of i-th Lagrange basis polynomial L_i''(x)
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


def newton_interpolation_derivative(x, y, target_x, order=1):
    """
    Compute derivative of Newton interpolating polynomial at target_x.

    Args:
        x: array of x-coordinates
        y: array of y-coordinates
        target_x: point where derivative is needed
        order: derivative order (1 or 2)

    Returns:
        Numerical derivative at target_x
    """
    n = len(x)

    # Compute divided differences table
    divided_diff = [[0.0 for _ in range(n)] for _ in range(n)]

    # Initialize first column with y values
    for i in range(n):
        divided_diff[i][0] = y[i]

    # Fill the divided differences table
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / (x[i + j] - x[i])

    if order == 1:
        # First derivative of Newton polynomial
        derivative = 0.0

        for k in range(1, n):
            # Coefficient for this term
            coeff = divided_diff[0][k]

            # Derivative of product (x - x[0])(x - x[1])...(x - x[k-1])
            term_derivative = 0.0
            for i in range(k):
                # Sum of products where we differentiate the i-th factor
                product = 1.0
                for j in range(k):
                    if j != i:
                        product *= (target_x - x[j])
                term_derivative += product

            derivative += coeff * term_derivative

        return derivative

    elif order == 2:
        # Second derivative of Newton polynomial
        derivative = 0.0

        for k in range(2, n):
            # Coefficient for this term
            coeff = divided_diff[0][k]

            # Second derivative of product (x - x[0])(x - x[1])...(x - x[k-1])
            term_second_derivative = 0.0
            for i in range(k):
                for j in range(i + 1, k):
                    # Sum of products where we differentiate the i-th and j-th factors
                    product = 2.0
                    for m in range(k):
                        if m != i and m != j:
                            product *= (target_x - x[m])
                    term_second_derivative += product

            derivative += coeff * term_second_derivative

        return derivative

    else:
        raise ValueError("Only first and second order derivatives are supported.")


def spline_interpolation_derivative(x, y, target_x, order=1):
    """
    Compute derivative using cubic spline interpolation.

    Args:
        x: array of x-coordinates
        y: array of y-coordinates
        target_x: point where derivative is needed
        order: derivative order (1 or 2)

    Returns:
        Numerical derivative at target_x
    """
    n = len(x)
    if n < 3:
        raise ValueError("At least 3 points required for cubic spline.")

    # Find the interval containing target_x
    if target_x <= x[0]:
        interval_idx = 0
    elif target_x >= x[-1]:
        interval_idx = n - 2
    else:
        interval_idx = 0
        for i in range(n - 1):
            if x[i] <= target_x <= x[i + 1]:
                interval_idx = i
                break

    # Simple cubic spline construction for the local interval
    # Using the four closest points for better accuracy
    start_idx = max(0, interval_idx - 1)
    end_idx = min(n, start_idx + 4)
    if end_idx - start_idx < 4:
        start_idx = max(0, end_idx - 4)

    # Extract local points
    x_local = x[start_idx:end_idx]
    y_local = y[start_idx:end_idx]

    # Use Lagrange method on local points for simplicity
    return lagrange_interpolation_derivative(x_local, y_local, target_x, order)


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
        y_vals = []
        for x_val in x_vals:
            # Replace 'x' with actual value and evaluate
            expr = func_expr.replace('x', str(x_val))
            # Basic safety check
            if any(dangerous in expr for dangerous in ['import', 'exec', 'eval', '__']):
                raise ValueError("Invalid function expression")
            try:
                y_vals.append(eval(expr))
            except:
                # Handle math functions
                import math
                expr = expr.replace('sin', 'math.sin')
                expr = expr.replace('cos', 'math.cos')
                expr = expr.replace('tan', 'math.tan')
                expr = expr.replace('exp', 'math.exp')
                expr = expr.replace('log', 'math.log')
                expr = expr.replace('sqrt', 'math.sqrt')
                y_vals.append(eval(expr))
        return np.array(y_vals)


def get_user_input():
    """
    Get user input for the interpolation derivative calculation.

    Returns:
        Dictionary containing all input parameters
    """
    print("=== Interpolation Derivative Calculator ===\n")

    # Function input
    func_expr = input("Enter the function expression (use 'x' as variable, e.g., 'x**3 - 2*x + 1'): ")

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
    print("\nSelect interpolation method:")
    print("1. Lagrange Interpolation")
    print("2. Newton Interpolation")
    print("3. Cubic Spline Interpolation")
    print("4. All Methods (Comparison)")
    method = int(input("Enter method choice (1-4): "))

    if method not in [1, 2, 3, 4]:
        raise ValueError("Invalid method choice. Use 1, 2, 3, or 4.")

    # Derivative order
    print("\nSelect derivative order:")
    print("1. First derivative")
    print("2. Second derivative")
    print("3. Both")
    order_choice = int(input("Enter order choice (1-3): "))

    if order_choice not in [1, 2, 3]:
        raise ValueError("Invalid order choice. Use 1, 2, or 3.")

    return {
        'func_expr': func_expr,
        'a': a,
        'b': b,
        'n': n,
        'target_x': target_x,
        'method': method,
        'order_choice': order_choice
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

    # Determine which orders to compute
    orders = []
    if params['order_choice'] == 1:
        orders = [1]
    elif params['order_choice'] == 2:
        orders = [2]
    else:
        orders = [1, 2]

    # Determine which methods to use
    methods = []
    if params['method'] == 1:
        methods = [('Lagrange', lagrange_interpolation_derivative)]
    elif params['method'] == 2:
        methods = [('Newton', newton_interpolation_derivative)]
    elif params['method'] == 3:
        methods = [('Cubic Spline', spline_interpolation_derivative)]
    else:
        methods = [
            ('Lagrange', lagrange_interpolation_derivative),
            ('Newton', newton_interpolation_derivative),
            ('Cubic Spline', spline_interpolation_derivative)
        ]

    # Display basic information
    print(f"\n=== Results ===")
    print(f"Function: f(x) = {params['func_expr']}")
    print(f"Interval: [{params['a']}, {params['b']}]")
    print(f"Interpolation points: {params['n']}")
    print(f"Target point: x = {params['target_x']}")

    # Compute and display derivatives
    for order in orders:
        print(f"\n--- {'First' if order == 1 else 'Second'} Derivative ---")

        for method_name, method_func in methods:
            try:
                if method_name == 'Cubic Spline' and params['n'] < 3:
                    print(f"{method_name:15}: N/A (requires at least 3 points)")
                    continue

                derivative = method_func(x_vals, y_vals, params['target_x'], order)
                print(f"{method_name:15}: {derivative:.8f}")

            except Exception as e:
                print(f"{method_name:15}: Error - {e}")

    # Display interpolation points for reference
    print(f"\nInterpolation points used:")
    for i in range(min(5, len(x_vals))):
        print(f"  x[{i}] = {x_vals[i]:.4f}, f(x[{i}]) = {y_vals[i]:.6f}")
    if len(x_vals) > 5:
        print(f"  ... and {len(x_vals) - 5} more points")


def main():
    """
    Main function to run the interpolation derivative calculator.
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