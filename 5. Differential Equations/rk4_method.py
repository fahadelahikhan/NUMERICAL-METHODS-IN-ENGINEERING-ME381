#!/usr/bin/env python3

import numpy as np
from sympy import symbols, lambdify, sympify
import matplotlib.pyplot as plt
import sys


def rk4_method(f, x0, y0, h, n):
    """
    Fourth-order Runge-Kutta method for solving ODEs

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps

    Returns:
        Tuple of (x_values, y_values) as NumPy arrays
    """
    # Pre-allocate arrays for better performance and easier translation
    x_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    # Set initial conditions
    x_values[0] = x0
    y_values[0] = y0

    # RK4 method iterations
    for i in range(n):
        x_i = x_values[i]
        y_i = y_values[i]

        # Calculate the four slopes
        k1 = f(x_i, y_i)
        k2 = f(x_i + h / 2, y_i + h / 2 * k1)
        k3 = f(x_i + h / 2, y_i + h / 2 * k2)
        k4 = f(x_i + h, y_i + h * k3)

        # Weighted average (Simpson's rule)
        y_values[i + 1] = y_i + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_values[i + 1] = x_i + h

    return x_values, y_values


def rk2_method(f, x0, y0, h, n):
    """
    Second-order Runge-Kutta method for comparison

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps

    Returns:
        Tuple of (x_values, y_values) as NumPy arrays
    """
    x_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    x_values[0] = x0
    y_values[0] = y0

    for i in range(n):
        x_i = x_values[i]
        y_i = y_values[i]

        k1 = f(x_i, y_i)
        k2 = f(x_i + h / 2, y_i + h / 2 * k1)

        y_values[i + 1] = y_i + h * k2
        x_values[i + 1] = x_i + h

    return x_values, y_values


def euler_method(f, x0, y0, h, n):
    """
    Euler method for comparison

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps

    Returns:
        Tuple of (x_values, y_values) as NumPy arrays
    """
    x_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    x_values[0] = x0
    y_values[0] = y0

    for i in range(n):
        derivative = f(x_values[i], y_values[i])
        y_values[i + 1] = y_values[i] + h * derivative
        x_values[i + 1] = x_values[i] + h

    return x_values, y_values


def adaptive_rk4_method(f, x0, y0, h_initial, x_final, tolerance=1e-6):
    """
    Adaptive step-size RK4 method

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h_initial: Initial step size
        x_final: Final x value
        tolerance: Error tolerance for adaptive stepping

    Returns:
        Tuple of (x_values, y_values) as NumPy arrays
    """
    x_values = [x0]
    y_values = [y0]

    x_current = x0
    y_current = y0
    h = h_initial

    while x_current < x_final:
        # Adjust step size if we're close to the end
        if x_current + h > x_final:
            h = x_final - x_current

        # Single step with step size h
        x_temp, y_temp = rk4_method(f, x_current, y_current, h, 1)
        y_single = y_temp[1]

        # Two steps with step size h/2
        x_temp, y_temp = rk4_method(f, x_current, y_current, h / 2, 2)
        y_double = y_temp[2]

        # Estimate error
        error = abs(y_double - y_single) / 15  # Richardson extrapolation

        if error <= tolerance:
            # Accept the step
            x_current += h
            y_current = y_double  # Use more accurate value
            x_values.append(x_current)
            y_values.append(y_current)

            # Increase step size for next iteration
            if error < tolerance / 10:
                h *= 1.5
        else:
            # Reject the step and reduce step size
            h *= 0.5

        # Prevent step size from becoming too small
        if h < 1e-10:
            print("Warning: Step size became very small. Stopping.")
            break

    return np.array(x_values), np.array(y_values)


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
    Get user's choice of solution method

    Returns:
        Method string ('rk4', 'adaptive', 'comparison')
    """
    print("\nChoose solution method:")
    print("1. Standard RK4 method")
    print("2. Adaptive RK4 method")
    print("3. Compare multiple methods")

    while True:
        try:
            choice = input("Enter choice (1, 2, or 3): ").strip()
            if choice == '1':
                return 'rk4'
            elif choice == '2':
                return 'adaptive'
            elif choice == '3':
                return 'comparison'
            else:
                print("Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)


def print_solution(x_values, y_values, method_name, max_display=20):
    """
    Print the solution in a formatted way

    Args:
        x_values: Array of x values
        y_values: Array of y values
        method_name: Name of the method used
        max_display: Maximum number of points to display
    """
    print(f"\nSolution using {method_name}:")
    print("-" * 45)
    print(f"{'Step':<6} {'x':<12} {'y':<15}")
    print("-" * 45)

    n_points = len(x_values)
    if n_points <= max_display:
        # Display all points
        for i in range(n_points):
            print(f"{i:<6} {x_values[i]:<12.6f} {y_values[i]:<15.8f}")
    else:
        # Display first few, some middle, and last few points
        display_indices = []
        display_indices.extend(range(min(5, n_points)))

        if n_points > 10:
            # Add some middle points
            mid_start = n_points // 2 - 2
            mid_end = n_points // 2 + 3
            display_indices.extend(range(max(5, mid_start), min(mid_end, n_points - 5)))

            # Add last few points
            display_indices.extend(range(max(n_points - 5, 0), n_points))

        # Remove duplicates and sort
        display_indices = sorted(list(set(display_indices)))

        for i in display_indices:
            if i > 0 and display_indices[display_indices.index(i) - 1] < i - 1:
                print("   ...")
            print(f"{i:<6} {x_values[i]:<12.6f} {y_values[i]:<15.8f}")


def plot_solution(x_values, y_values, method_name, func_str, save_plot=False):
    """
    Plot the solution

    Args:
        x_values: Array of x values
        y_values: Array of y values
        method_name: Name of the method used
        func_str: String representation of the function
        save_plot: Whether to save the plot to file
    """
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, y_values, 'b-', linewidth=2, label=f'{method_name} Solution')
    plt.plot(x_values, y_values, 'ro', markersize=4, alpha=0.7, label='Solution Points')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Solution of dy/dx = {func_str}\nusing {method_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add some styling
    plt.tight_layout()

    if save_plot:
        plt.savefig('ode_solution.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'ode_solution.png'")

    plt.show()


def compare_methods(f, x0, y0, h, n, func_str):
    """
    Compare different numerical methods

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        h: Step size
        n: Number of steps
        func_str: String representation of the function
    """
    print("\n" + "=" * 70)
    print("COMPARISON OF NUMERICAL METHODS")
    print("=" * 70)

    # Solve using different methods
    x_euler, y_euler = euler_method(f, x0, y0, h, n)
    x_rk2, y_rk2 = rk2_method(f, x0, y0, h, n)
    x_rk4, y_rk4 = rk4_method(f, x0, y0, h, n)

    # Print comparison table
    print(f"\nNumerical comparison for dy/dx = {func_str}")
    print(f"Initial condition: y({x0}) = {y0}, Step size: {h}")
    print("-" * 70)
    print(f"{'x':<10} {'Euler':<15} {'RK2':<15} {'RK4':<15}")
    print("-" * 70)

    # Display every few points for readability
    step_size = max(1, n // 10)
    for i in range(0, n + 1, step_size):
        if i < len(x_euler):
            print(f"{x_euler[i]:<10.4f} {y_euler[i]:<15.8f} {y_rk2[i]:<15.8f} {y_rk4[i]:<15.8f}")

    # Plot comparison
    plt.figure(figsize=(14, 8))
    plt.plot(x_euler, y_euler, 'r--', linewidth=2, label='Euler Method', alpha=0.8)
    plt.plot(x_rk2, y_rk2, 'g--', linewidth=2, label='RK2 Method', alpha=0.8)
    plt.plot(x_rk4, y_rk4, 'b-', linewidth=2, label='RK4 Method')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Comparison of Methods for dy/dx = {func_str}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_step_size_analysis(f, x0, y0, x_final, func_str):
    """
    Analyze the effect of different step sizes on accuracy

    Args:
        f: Function representing dy/dx = f(x, y)
        x0: Initial x value
        y0: Initial y value
        x_final: Final x value
        func_str: String representation of the function
    """
    print("\n" + "=" * 60)
    print("STEP SIZE ANALYSIS")
    print("=" * 60)

    step_sizes = [0.1, 0.05, 0.025, 0.01, 0.005]
    final_values = []

    print(f"{'Step Size':<12} {'Steps':<8} {'Final y':<15} {'Time Points':<12}")
    print("-" * 60)

    for h in step_sizes:
        n = int((x_final - x0) / h)
        x_vals, y_vals = rk4_method(f, x0, y0, h, n)
        final_y = y_vals[-1]
        final_values.append(final_y)

        print(f"{h:<12.3f} {n:<8} {final_y:<15.8f} {len(x_vals):<12}")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogx(step_sizes, final_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Step Size', fontsize=12)
    plt.ylabel(f'Final Value y({x_final})', fontsize=12)
    plt.title(f'Step Size Convergence Analysis\nfor dy/dx = {func_str}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def validate_inputs(h, n, x0, x_final=None):
    """
    Validate input parameters

    Args:
        h: Step size
        n: Number of steps (can be None for adaptive method)
        x0: Initial x value
        x_final: Final x value (for adaptive method)

    Returns:
        True if inputs are valid, False otherwise
    """
    if h <= 0:
        print("Error: Step size must be positive")
        return False

    if n is not None and n <= 0:
        print("Error: Number of steps must be positive")
        return False

    if x_final is not None and x_final <= x0:
        print("Error: Final x value must be greater than initial x value")
        return False

    if h > 2.0:
        print("Warning: Large step size may lead to numerical instability")

    if n is not None and n > 50000:
        print("Warning: Large number of steps may take significant time")

    return True


def main():
    """
    Main program function
    """
    print("=" * 75)
    print("RK4 (FOURTH-ORDER RUNGE-KUTTA) METHOD FOR SOLVING ODEs")
    print("=" * 75)
    print("Solve ODEs of the form: dy/dx = f(x, y)")
    print("Features:")
    print("  - Standard RK4 method")
    print("  - Adaptive step-size RK4")
    print("  - Method comparison")
    print("  - Visualization and analysis")
    print("\nExamples of valid functions:")
    print("  - x + y                 (exponential growth)")
    print("  - -y + sin(x)           (oscillatory with forcing)")
    print("  - x**2 - y              (polynomial)")
    print("  - y*(1 - y)             (logistic equation)")
    print("  - -2*x*y                (exponential decay)")
    print("  - cos(x) + sin(y)       (trigonometric)")
    print("-" * 75)

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

        # Get method choice
        method = get_method_choice()

        if method == 'rk4':
            # Standard RK4 method
            print("\nStep 3: Set numerical parameters")
            h = get_numerical_input("Enter step size (h): ", float, lambda x: x > 0)
            n = get_numerical_input("Enter number of steps: ", int, lambda x: x > 0)

            if not validate_inputs(h, n, x0):
                return

            # Display problem summary
            print("\nProblem Summary:")
            print(f"ODE: dy/dx = {func_str}")
            print(f"Initial condition: y({x0}) = {y0}")
            print(f"Method: Standard RK4")
            print(f"Step size: {h}")
            print(f"Number of steps: {n}")
            print(f"Final x value: {x0 + n * h}")

            # Solve the ODE
            print("\nSolving...")
            x_values, y_values = rk4_method(f, x0, y0, h, n)

            # Display results
            print_solution(x_values, y_values, "RK4 Method")

            # Plot solution
            plot_solution(x_values, y_values, "RK4 Method", func_str)

            # Final result
            print(f"\nFinal result: y({x_values[-1]:.6f}) ≈ {y_values[-1]:.8f}")

        elif method == 'adaptive':
            # Adaptive RK4 method
            print("\nStep 3: Set adaptive parameters")
            h_initial = get_numerical_input("Enter initial step size: ", float, lambda x: x > 0)
            x_final = get_numerical_input("Enter final x value: ", float, lambda x: x > x0)
            tolerance = get_numerical_input("Enter tolerance (default 1e-6): ", float, lambda x: x > 0) or 1e-6

            if not validate_inputs(h_initial, None, x0, x_final):
                return

            # Display problem summary
            print("\nProblem Summary:")
            print(f"ODE: dy/dx = {func_str}")
            print(f"Initial condition: y({x0}) = {y0}")
            print(f"Method: Adaptive RK4")
            print(f"Initial step size: {h_initial}")
            print(f"Final x value: {x_final}")
            print(f"Tolerance: {tolerance}")

            # Solve the ODE
            print("\nSolving...")
            x_values, y_values = adaptive_rk4_method(f, x0, y0, h_initial, x_final, tolerance)

            # Display results
            print_solution(x_values, y_values, "Adaptive RK4 Method")

            # Plot solution
            plot_solution(x_values, y_values, "Adaptive RK4 Method", func_str)

            # Final result
            print(f"\nFinal result: y({x_values[-1]:.6f}) ≈ {y_values[-1]:.8f}")
            print(f"Total steps taken: {len(x_values) - 1}")

        elif method == 'comparison':
            # Method comparison
            print("\nStep 3: Set comparison parameters")
            h = get_numerical_input("Enter step size for comparison: ", float, lambda x: x > 0)
            n = get_numerical_input("Enter number of steps: ", int, lambda x: x > 0)

            if not validate_inputs(h, n, x0):
                return

            # Compare methods
            compare_methods(f, x0, y0, h, n, func_str)

            # Optional: Step size analysis
            analysis_choice = input("\nPerform step size analysis? (y/n): ").strip().lower()
            if analysis_choice == 'y':
                x_final = x0 + n * h
                calculate_step_size_analysis(f, x0, y0, x_final, func_str)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())