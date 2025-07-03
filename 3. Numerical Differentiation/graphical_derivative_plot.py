#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math


def compute_derivative_central(func, x_vals, h=None):
    """
    Compute numerical derivative using central difference method.

    Args:
        func: function to differentiate
        x_vals: array of x values
        h: step size (if None, uses adaptive step size)

    Returns:
        Array of derivative values
    """
    if h is None:
        # Adaptive step size based on spacing
        h = np.minimum(1e-5, (x_vals[-1] - x_vals[0]) / (10 * len(x_vals)))

    # Handle boundary conditions
    y_deriv = np.zeros_like(x_vals)

    # Central difference for interior points
    for i in range(len(x_vals)):
        try:
            y_deriv[i] = (func(x_vals[i] + h) - func(x_vals[i] - h)) / (2 * h)
        except:
            # Fallback for edge cases
            if i == 0:
                y_deriv[i] = (func(x_vals[i] + h) - func(x_vals[i])) / h
            elif i == len(x_vals) - 1:
                y_deriv[i] = (func(x_vals[i]) - func(x_vals[i] - h)) / h
            else:
                y_deriv[i] = (func(x_vals[i] + h) - func(x_vals[i] - h)) / (2 * h)

    return y_deriv


def compute_derivative_forward(func, x_vals, h=None):
    """
    Compute numerical derivative using forward difference method.

    Args:
        func: function to differentiate
        x_vals: array of x values
        h: step size

    Returns:
        Array of derivative values
    """
    if h is None:
        h = np.minimum(1e-5, (x_vals[-1] - x_vals[0]) / (10 * len(x_vals)))

    y_deriv = np.zeros_like(x_vals)

    for i in range(len(x_vals)):
        y_deriv[i] = (func(x_vals[i] + h) - func(x_vals[i])) / h

    return y_deriv


def compute_derivative_backward(func, x_vals, h=None):
    """
    Compute numerical derivative using backward difference method.

    Args:
        func: function to differentiate
        x_vals: array of x values
        h: step size

    Returns:
        Array of derivative values
    """
    if h is None:
        h = np.minimum(1e-5, (x_vals[-1] - x_vals[0]) / (10 * len(x_vals)))

    y_deriv = np.zeros_like(x_vals)

    for i in range(len(x_vals)):
        y_deriv[i] = (func(x_vals[i]) - func(x_vals[i] - h)) / h

    return y_deriv


def compute_derivative_five_point(func, x_vals, h=None):
    """
    Compute numerical derivative using five-point stencil method.

    Args:
        func: function to differentiate
        x_vals: array of x values
        h: step size

    Returns:
        Array of derivative values
    """
    if h is None:
        h = np.minimum(1e-5, (x_vals[-1] - x_vals[0]) / (10 * len(x_vals)))

    y_deriv = np.zeros_like(x_vals)

    for i in range(len(x_vals)):
        try:
            # Five-point stencil: f'(x) ≈ [f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)] / (12h)
            y_deriv[i] = (func(x_vals[i] - 2 * h) - 8 * func(x_vals[i] - h) +
                          8 * func(x_vals[i] + h) - func(x_vals[i] + 2 * h)) / (12 * h)
        except:
            # Fallback to central difference
            y_deriv[i] = (func(x_vals[i] + h) - func(x_vals[i] - h)) / (2 * h)

    return y_deriv


def compute_second_derivative(func, x_vals, h=None):
    """
    Compute second derivative using central difference method.

    Args:
        func: function to differentiate
        x_vals: array of x values
        h: step size

    Returns:
        Array of second derivative values
    """
    if h is None:
        h = np.minimum(1e-5, (x_vals[-1] - x_vals[0]) / (10 * len(x_vals)))

    y_second_deriv = np.zeros_like(x_vals)

    for i in range(len(x_vals)):
        try:
            y_second_deriv[i] = (func(x_vals[i] + h) - 2 * func(x_vals[i]) + func(x_vals[i] - h)) / (h ** 2)
        except:
            y_second_deriv[i] = 0.0

    return y_second_deriv


def evaluate_function(func_expr, x_vals):
    """
    Evaluate a mathematical function at given x values.

    Args:
        func_expr: function expression as string
        x_vals: array of x values

    Returns:
        Function object and array of function values
    """
    try:
        # Try using sympy for symbolic evaluation
        from sympy import symbols, lambdify, sympify
        x = symbols('x')
        expr = sympify(func_expr)
        func = lambdify(x, expr, 'numpy')
        return func, func(x_vals)
    except ImportError:
        # Fallback to basic evaluation if sympy not available
        print("Warning: sympy not available. Using basic evaluation.")

        def func(x_val):
            if isinstance(x_val, np.ndarray):
                return np.array([evaluate_single(val) for val in x_val])
            else:
                return evaluate_single(x_val)

        def evaluate_single(x_val):
            expr = func_expr.replace('x', str(x_val))
            # Basic safety check
            if any(dangerous in expr for dangerous in ['import', 'exec', 'eval', '__']):
                raise ValueError("Invalid function expression")
            try:
                return eval(expr)
            except:
                # Handle math functions
                expr = expr.replace('sin', 'math.sin')
                expr = expr.replace('cos', 'math.cos')
                expr = expr.replace('tan', 'math.tan')
                expr = expr.replace('exp', 'math.exp')
                expr = expr.replace('log', 'math.log')
                expr = expr.replace('sqrt', 'math.sqrt')
                expr = expr.replace('pi', 'math.pi')
                return eval(expr)

        return func, func(x_vals)


def get_user_input():
    """
    Get user input for the graphical derivative calculation.

    Returns:
        Dictionary containing all input parameters
    """
    print("=== Graphical Derivative Calculator ===\n")

    # Function input
    func_expr = input("Enter the function expression (use 'x' as variable, e.g., 'x**2 + sin(x)'): ")

    # Interval and points
    a = float(input("Enter the start of the interval (a): "))
    b = float(input("Enter the end of the interval (b): "))

    if a >= b:
        raise ValueError("Interval start must be less than end.")

    n = int(input("Enter the number of points for plotting: "))
    if n < 2:
        raise ValueError("At least 2 points are required.")

    # Step size option
    use_custom_h = input("Use custom step size? (y/n): ").lower().strip()
    h = None
    if use_custom_h == 'y':
        h = float(input("Enter step size (h): "))
        if h <= 0:
            raise ValueError("Step size must be positive.")

    # Method selection
    print("\nSelect derivative computation method:")
    print("1. Central Difference (most accurate)")
    print("2. Forward Difference")
    print("3. Backward Difference")
    print("4. Five-Point Stencil (high accuracy)")
    print("5. All Methods (Comparison)")
    method = int(input("Enter method choice (1-5): "))

    if method not in [1, 2, 3, 4, 5]:
        raise ValueError("Invalid method choice. Use 1-5.")

    # Derivative order
    print("\nSelect what to plot:")
    print("1. Function and First Derivative")
    print("2. Function and Second Derivative")
    print("3. Function, First and Second Derivatives")
    plot_choice = int(input("Enter plot choice (1-3): "))

    if plot_choice not in [1, 2, 3]:
        raise ValueError("Invalid plot choice. Use 1-3.")

    # Additional plotting options
    show_analytical = input("Show analytical derivative if available? (y/n): ").lower().strip() == 'y'

    return {
        'func_expr': func_expr,
        'a': a,
        'b': b,
        'n': n,
        'h': h,
        'method': method,
        'plot_choice': plot_choice,
        'show_analytical': show_analytical
    }


def compute_analytical_derivative(func_expr):
    """
    Compute analytical derivative if possible.

    Args:
        func_expr: function expression as string

    Returns:
        Analytical derivative function or None
    """
    try:
        from sympy import symbols, lambdify, sympify, diff
        x = symbols('x')
        expr = sympify(func_expr)

        # First derivative
        first_deriv_expr = diff(expr, x)
        first_deriv_func = lambdify(x, first_deriv_expr, 'numpy')

        # Second derivative
        second_deriv_expr = diff(expr, x, 2)
        second_deriv_func = lambdify(x, second_deriv_expr, 'numpy')

        return first_deriv_func, second_deriv_func
    except:
        return None, None


def create_plots(params, x_vals, y_vals, func, derivatives_data):
    """
    Create and display plots based on user preferences.

    Args:
        params: user input parameters
        x_vals: array of x values
        y_vals: array of function values
        func: function object
        derivatives_data: computed derivative data
    """
    # Determine subplot configuration
    if params['plot_choice'] == 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        axes = [ax1, ax2]
    elif params['plot_choice'] == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        axes = [ax1, ax2]
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
        axes = [ax1, ax2, ax3]

    # Plot function
    axes[0].plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title(f'Function: f(x) = {params["func_expr"]}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot derivatives based on choice
    if params['plot_choice'] in [1, 3]:
        # First derivative
        ax_idx = 1
        for method_name, first_deriv in derivatives_data['first']:
            axes[ax_idx].plot(x_vals, first_deriv, linewidth=2, label=f"{method_name}", linestyle='--')

        # Add analytical derivative if available
        if params['show_analytical'] and derivatives_data['analytical'][0] is not None:
            analytical_first = derivatives_data['analytical'][0](x_vals)
            axes[ax_idx].plot(x_vals, analytical_first, 'k-', linewidth=2, label='Analytical', alpha=0.7)

        axes[ax_idx].set_xlabel('x')
        axes[ax_idx].set_ylabel("f'(x)")
        axes[ax_idx].set_title("First Derivative")
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].legend()

    if params['plot_choice'] in [2, 3]:
        # Second derivative
        ax_idx = 2 if params['plot_choice'] == 3 else 1
        for method_name, second_deriv in derivatives_data['second']:
            axes[ax_idx].plot(x_vals, second_deriv, linewidth=2, label=f"{method_name}", linestyle=':')

        # Add analytical second derivative if available
        if params['show_analytical'] and derivatives_data['analytical'][1] is not None:
            analytical_second = derivatives_data['analytical'][1](x_vals)
            axes[ax_idx].plot(x_vals, analytical_second, 'k-', linewidth=2, label='Analytical', alpha=0.7)

        axes[ax_idx].set_xlabel('x')
        axes[ax_idx].set_ylabel("f''(x)")
        axes[ax_idx].set_title("Second Derivative")
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].legend()

    plt.tight_layout()
    plt.show()


def compute_and_display_results(params):
    """
    Compute derivatives and display results.

    Args:
        params: dictionary containing input parameters
    """
    # Generate points
    x_vals = np.linspace(params['a'], params['b'], params['n'])

    # Evaluate function
    func, y_vals = evaluate_function(params['func_expr'], x_vals)

    # Compute analytical derivatives if requested
    analytical_derivs = (None, None)
    if params['show_analytical']:
        analytical_derivs = compute_analytical_derivative(params['func_expr'])

    # Method mapping
    methods = {
        1: ('Central Difference', compute_derivative_central),
        2: ('Forward Difference', compute_derivative_forward),
        3: ('Backward Difference', compute_derivative_backward),
        4: ('Five-Point Stencil', compute_derivative_five_point)
    }

    # Compute derivatives
    derivatives_data = {
        'first': [],
        'second': [],
        'analytical': analytical_derivs
    }

    if params['method'] == 5:
        # All methods
        for method_num, (method_name, method_func) in methods.items():
            try:
                first_deriv = method_func(func, x_vals, params['h'])
                second_deriv = compute_second_derivative(func, x_vals, params['h'])
                derivatives_data['first'].append((method_name, first_deriv))
                derivatives_data['second'].append((method_name, second_deriv))
            except Exception as e:
                print(f"Error computing {method_name}: {e}")
    else:
        # Single method
        method_name, method_func = methods[params['method']]
        try:
            first_deriv = method_func(func, x_vals, params['h'])
            second_deriv = compute_second_derivative(func, x_vals, params['h'])
            derivatives_data['first'].append((method_name, first_deriv))
            derivatives_data['second'].append((method_name, second_deriv))
        except Exception as e:
            print(f"Error computing {method_name}: {e}")
            return

    # Display numerical results
    print(f"\n=== Numerical Results ===")
    print(f"Function: f(x) = {params['func_expr']}")
    print(f"Interval: [{params['a']}, {params['b']}]")
    print(f"Number of points: {params['n']}")
    print(f"Step size: {params['h'] if params['h'] else 'Auto'}")

    # Show sample values
    mid_idx = len(x_vals) // 2
    x_mid = x_vals[mid_idx]
    print(f"\nSample values at x = {x_mid:.4f}:")
    print(f"f(x) = {y_vals[mid_idx]:.6f}")

    for method_name, first_deriv in derivatives_data['first']:
        print(f"f'(x) [{method_name}] = {first_deriv[mid_idx]:.6f}")

    for method_name, second_deriv in derivatives_data['second']:
        print(f"f''(x) [{method_name}] = {second_deriv[mid_idx]:.6f}")

    if params['show_analytical']:
        if analytical_derivs[0] is not None:
            print(f"f'(x) [Analytical] = {analytical_derivs[0](x_mid):.6f}")
        if analytical_derivs[1] is not None:
            print(f"f''(x) [Analytical] = {analytical_derivs[1](x_mid):.6f}")

    # Create plots
    create_plots(params, x_vals, y_vals, func, derivatives_data)


def main():
    """
    Main function to run the graphical derivative calculator.
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