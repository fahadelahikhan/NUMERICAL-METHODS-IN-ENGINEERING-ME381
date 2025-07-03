#!/usr/bin/env python3

import numpy as np
from sympy import symbols, lambdify, sympify
import sys


def rk2_method(f, x0, y0, h, n, variant='midpoint'):
    """
    Solve ODE using RK2 method with different variants

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps
        variant: 'midpoint', 'heun', or 'ralston'

    Returns:
        List of tuples (x, y) representing the solution
    """
    # Pre-allocate arrays for better performance and easier translation
    x_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # Set initial conditions
    x_values[0] = x0
    y_values[0] = y0

    # RK2 method iterations
    for i in range(n):
        x_i = x_values[i]
        y_i = y_values[i]

        if variant == 'midpoint':
            # RK2 Midpoint method (your original implementation)
            k1 = f(x_i, y_i)
            k2 = f(x_i + h / 2, y_i + h / 2 * k1)
            y_values[i + 1] = y_i + h * k2

        elif variant == 'heun':
            # RK2 Heun's method (Modified Euler)
            k1 = f(x_i, y_i)
            k2 = f(x_i + h, y_i + h * k1)
            y_values[i + 1] = y_i + h / 2 * (k1 + k2)

        elif variant == 'ralston':
            # RK2 Ralston's method
            k1 = f(x_i, y_i)
            k2 = f(x_i + 3 * h / 4, y_i + 3 * h / 4 * k1)
            y_values[i + 1] = y_i + h * (k1 / 3 + 2 * k2 / 3)

        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Update x
        x_values[i + 1] = x_i + h

    # Return as list of tuples for compatibility
    solution = [(x_values[i], y_values[i]) for i in range(n + 1)]
    return solution


def rk4_method(f, x0, y0, h, n):
    """
    Fourth-order Runge-Kutta method for comparison

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
        x_i = x_values[i]
        y_i = y_values[i]

        # Calculate the four slopes
        k1 = f(x_i, y_i)
        k2 = f(x_i + h / 2, y_i + h / 2 * k1)
        k3 = f(x_i + h / 2, y_i + h / 2 * k2)
        k4 = f(x_i + h, y_i + h * k3)

        # Weighted average
        y_values[i + 1] = y_i + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_values[i + 1] = x_i + h

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


def get_rk2_variant():
    """
    Get user's choice of RK2 variant

    Returns:
        Variant string ('midpoint', 'heun', or 'ralston')
    """
    print("\nChoose RK2 method variant:")
    print("1. Midpoint method (evaluates at h/2)")
    print("2. Heun's method (Modified Euler)")
    print("3. Ralston's method (evaluates at 3h/4)")

    while True:
        try:
            choice = input("Enter choice (1, 2, or 3): ").strip()
            if choice == '1':
                return 'midpoint'
            elif choice == '2':
                return 'heun'
            elif choice == '3':
                return 'ralston'
            else:
                print("Please enter 1, 2, or 3.")
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
    print("-" * 40)
    print(f"{'Step':<6} {'x':<10} {'y':<15}")
    print("-" * 40)

    for i, (x, y) in enumerate(solution):
        print(f"{i:<6} {x:<10.4f} {y:<15.{precision}f}")


def compare_methods(f, x0, y0, h, n, func_str):
    """
    Compare different RK2 variants and other methods

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps
        func_str: String representation of the function
    """
    print("\n" + "=" * 80)
    print("COMPARISON OF NUMERICAL METHODS")
    print("=" * 80)

    # Solve using different methods
    euler_sol = euler_method(f, x0, y0, h, n)
    rk2_midpoint_sol = rk2_method(f, x0, y0, h, n, 'midpoint')
    rk2_heun_sol = rk2_method(f, x0, y0, h, n, 'heun')
    rk2_ralston_sol = rk2_method(f, x0, y0, h, n, 'ralston')

    # Include RK4 for comparison if n is small
    if n <= 20:
        rk4_sol = rk4_method(f, x0, y0, h, n)
        include_rk4 = True
    else:
        include_rk4 = False

    # Print comparison table
    print(f"\nComparison for dy/dx = {func_str}")
    print(f"Initial condition: y({x0}) = {y0}, Step size: {h}")
    print("-" * 80)

    if include_rk4:
        print(f"{'x':<8} {'Euler':<12} {'RK2-Mid':<12} {'RK2-Heun':<12} {'RK2-Ral':<12} {'RK4':<12}")
    else:
        print(f"{'x':<8} {'Euler':<12} {'RK2-Mid':<12} {'RK2-Heun':<12} {'RK2-Ral':<12}")

    print("-" * 80)

    num_display = min(11, len(euler_sol))  # Show first 10 steps + initial
    for i in range(num_display):
        x_val = euler_sol[i][0]
        euler_y = euler_sol[i][1]
        mid_y = rk2_midpoint_sol[i][1]
        heun_y = rk2_heun_sol[i][1]
        ralston_y = rk2_ralston_sol[i][1]

        if include_rk4:
            rk4_y = rk4_sol[i][1]
            print(f"{x_val:<8.3f} {euler_y:<12.6f} {mid_y:<12.6f} {heun_y:<12.6f} {ralston_y:<12.6f} {rk4_y:<12.6f}")
        else:
            print(f"{x_val:<8.3f} {euler_y:<12.6f} {mid_y:<12.6f} {heun_y:<12.6f} {ralston_y:<12.6f}")


def calculate_error_analysis(f, x0, y0, h, n, analytical_func=None):
    """
    Perform error analysis if analytical solution is known

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps
        analytical_func: Analytical solution function (optional)
    """
    if analytical_func is None:
        return

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    # Calculate numerical solutions
    rk2_sol = rk2_method(f, x0, y0, h, n, 'midpoint')

    # Calculate errors
    print(f"{'x':<10} {'Numerical':<15} {'Analytical':<15} {'Error':<15}")
    print("-" * 60)

    for i in range(0, len(rk2_sol), max(1, len(rk2_sol) // 10)):
        x_val, y_num = rk2_sol[i]
        y_exact = analytical_func(x_val)
        error = abs(y_num - y_exact)
        print(f"{x_val:<10.4f} {y_num:<15.6f} {y_exact:<15.6f} {error:<15.2e}")


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
    print("=" * 70)
    print("RK2 (SECOND-ORDER RUNGE-KUTTA) METHOD FOR SOLVING ODEs")
    print("=" * 70)
    print("Solve ODEs of the form: dy/dx = f(x, y)")
    print("Available RK2 variants:")
    print("  - Midpoint method")
    print("  - Heun's method (Modified Euler)")
    print("  - Ralston's method")
    print("\nExamples of valid functions:")
    print("  - x + y")
    print("  - x**2 - y")
    print("  - sin(x) * cos(y)")
    print("  - exp(-x) * y")
    print("  - -2*x*y")
    print("  - y - x**2")
    print("-" * 70)

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

        # Get RK2 variant choice
        variant = get_rk2_variant()
        variant_names = {
            'midpoint': 'RK2 Midpoint Method',
            'heun': 'RK2 Heun\'s Method',
            'ralston': 'RK2 Ralston\'s Method'
        }
        method_name = variant_names[variant]

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
        solution = rk2_method(f, x0, y0, h, n, variant)

        # Display results
        print_solution(solution, method_name)

        # Display final result
        final_x, final_y = solution[-1]
        print(f"\nFinal result: y({final_x:.4f}) ≈ {final_y:.6f}")

        # Ask if user wants to compare methods
        if n <= 25:  # Only offer comparison for reasonably sized problems
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