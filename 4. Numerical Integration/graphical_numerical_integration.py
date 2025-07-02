#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, integrate, sympify
import sys
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class NumericalIntegrator:
    """
    A class for performing numerical integration using various methods.
    Designed for easy translation to other programming languages.
    """

    def __init__(self):
        self.x_symbol = symbols('x')

    def trapezoidal_rule(self, func, a, b, n):
        """
        Compute integral using trapezoidal rule.

        Args:
            func: Function to integrate (numpy-compatible)
            a: Lower bound
            b: Upper bound
            n: Number of intervals

        Returns:
            float: Approximated integral value
        """
        if n <= 0:
            raise ValueError("Number of intervals must be positive")

        h = (b - a) / n
        x_values = np.linspace(a, b, n + 1)

        try:
            y_values = func(x_values)
        except Exception as e:
            raise ValueError(f"Error evaluating function: {e}")

        # Trapezoidal formula: h/2 * [f(a) + 2*sum(f(xi)) + f(b)]
        integral = (h / 2) * (y_values[0] + 2 * np.sum(y_values[1:-1]) + y_values[-1])
        return float(integral)

    def simpsons_rule(self, func, a, b, n):
        """
        Compute integral using Simpson's 1/3 rule.

        Args:
            func: Function to integrate (numpy-compatible)
            a: Lower bound
            b: Upper bound
            n: Number of intervals (must be even)

        Returns:
            float: Approximated integral value
        """
        if n <= 0:
            raise ValueError("Number of intervals must be positive")
        if n % 2 != 0:
            raise ValueError("Number of intervals must be even for Simpson's rule")

        h = (b - a) / n
        x_values = np.linspace(a, b, n + 1)

        try:
            y_values = func(x_values)
        except Exception as e:
            raise ValueError(f"Error evaluating function: {e}")

        # Simpson's formula: h/3 * [f(a) + 4*sum(odd) + 2*sum(even) + f(b)]
        integral = (h / 3) * (
                y_values[0] +
                4 * np.sum(y_values[1::2]) +
                2 * np.sum(y_values[2:-1:2]) +
                y_values[-1]
        )
        return float(integral)

    def simpsons_3_8_rule(self, func, a, b, n):
        """
        Compute integral using Simpson's 3/8 rule.

        Args:
            func: Function to integrate (numpy-compatible)
            a: Lower bound
            b: Upper bound
            n: Number of intervals (must be multiple of 3)

        Returns:
            float: Approximated integral value
        """
        if n <= 0:
            raise ValueError("Number of intervals must be positive")
        if n % 3 != 0:
            raise ValueError("Number of intervals must be multiple of 3 for Simpson's 3/8 rule")

        h = (b - a) / n
        x_values = np.linspace(a, b, n + 1)

        try:
            y_values = func(x_values)
        except Exception as e:
            raise ValueError(f"Error evaluating function: {e}")

        integral = 0
        for i in range(0, n, 3):
            integral += (3 * h / 8) * (
                    y_values[i] + 3 * y_values[i + 1] +
                    3 * y_values[i + 2] + y_values[i + 3]
            )
        return float(integral)

    def adaptive_method_selection(self, n, user_choice):
        """
        Automatically adjust interval count for different methods when needed.

        Args:
            n: Original number of intervals
            user_choice: User's method choice

        Returns:
            dict: Adjusted intervals for each method
        """
        intervals = {}

        if user_choice in ['trapezoidal', 'all']:
            intervals['trapezoidal'] = n

        if user_choice in ['simpson_1_3', 'all']:
            # Ensure even number for Simpson's 1/3
            intervals['simpson_1_3'] = n if n % 2 == 0 else n + 1

        if user_choice in ['simpson_3_8', 'all']:
            # Ensure multiple of 3 for Simpson's 3/8
            intervals['simpson_3_8'] = n if n % 3 == 0 else n + (3 - n % 3)

        return intervals

    def parse_function(self, func_expr):
        """
        Parse string expression into a callable function with enhanced error handling.

        Args:
            func_expr: String representation of function

        Returns:
            tuple: (sympy_expr, numpy_func)
        """
        try:
            # Clean up common user input issues
            func_expr = func_expr.replace('^', '**')  # Replace ^ with **
            func_expr = func_expr.replace('ln', 'log')  # Replace ln with log

            # Parse the expression using sympify for better error handling
            sympy_expr = sympify(func_expr)
            numpy_func = lambdify(self.x_symbol, sympy_expr, 'numpy')

            # Test the function with a simple value to catch early errors
            test_val = numpy_func(1.0)
            if not np.isfinite(test_val):
                raise ValueError("Function produces non-finite values")

            return sympy_expr, numpy_func
        except Exception as e:
            raise ValueError(f"Invalid function expression '{func_expr}': {e}")

    def compute_exact_integral(self, sympy_expr, a, b):
        """
        Compute exact integral using symbolic integration with better error handling.

        Args:
            sympy_expr: Sympy expression
            a: Lower bound
            b: Upper bound

        Returns:
            float or None: Exact integral value or None if computation fails
        """
        try:
            result = integrate(sympy_expr, (self.x_symbol, a, b))
            exact_value = float(result.evalf())

            # Check if the result is finite
            if not np.isfinite(exact_value):
                return None

            return exact_value
        except Exception:
            return None

    def plot_function_and_area(self, numpy_func, a, b, title="Function and Integration Area"):
        """
        Plot the function and highlight the integration area with enhanced error handling.

        Args:
            numpy_func: Numpy-compatible function
            a: Lower bound
            b: Upper bound
            title: Plot title
        """
        try:
            # Create more points for smoother curve
            x_vals = np.linspace(a, b, 1000)
            y_vals = numpy_func(x_vals)

            # Check for problematic values
            if not np.all(np.isfinite(y_vals)):
                print("Warning: Function contains non-finite values. Plot may be incomplete.")
                # Filter out non-finite values
                finite_mask = np.isfinite(y_vals)
                x_vals = x_vals[finite_mask]
                y_vals = y_vals[finite_mask]

            plt.figure(figsize=(12, 8))
            plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
            plt.fill_between(x_vals, y_vals, alpha=0.3, color='lightblue', label='Integration Area')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=a, color='r', linestyle='--', alpha=0.7, label=f'x = {a}')
            plt.axvline(x=b, color='r', linestyle='--', alpha=0.7, label=f'x = {b}')

            plt.xlabel('x', fontsize=12)
            plt.ylabel('f(x)', fontsize=12)
            plt.title(title, fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting function: {e}")

    def convergence_analysis(self, numpy_func, a, b, method='trapezoidal', max_intervals=1000):
        """
        Perform convergence analysis by doubling intervals.

        Args:
            numpy_func: Function to integrate
            a, b: Integration bounds
            method: Integration method to use
            max_intervals: Maximum number of intervals to test
        """
        intervals = [10, 20, 40, 80, 160, 320, 640]
        intervals = [n for n in intervals if n <= max_intervals]
        results = []

        for n in intervals:
            try:
                if method == 'trapezoidal':
                    result = self.trapezoidal_rule(numpy_func, a, b, n)
                elif method == 'simpson_1_3':
                    n_adj = n if n % 2 == 0 else n + 1
                    result = self.simpsons_rule(numpy_func, a, b, n_adj)
                elif method == 'simpson_3_8':
                    n_adj = n if n % 3 == 0 else n + (3 - n % 3)
                    result = self.simpsons_3_8_rule(numpy_func, a, b, n_adj)
                else:
                    continue

                results.append((n, result))
            except Exception:
                continue

        return results


class InputOutput:
    """
    Handle all input/output operations.
    Separated for easy translation to other languages.
    """

    @staticmethod
    def get_function_input():
        """Get function expression from user with examples."""
        print("\nFunction Input Examples:")
        print("  - Polynomial: x**2 + 3*x + 1")
        print("  - Trigonometric: sin(x) + cos(x)")
        print("  - Exponential: exp(x) or e**x")
        print("  - Logarithmic: log(x)")
        print("  - Complex: x**3 - 2*x**2 + 5*x - 1")
        print("  - Note: Use ** for powers, * for multiplication")

        while True:
            try:
                func_expr = input("\nEnter the function expression (use 'x' as variable): ").strip()
                if not func_expr:
                    print("Error: Function expression cannot be empty.")
                    continue
                return func_expr
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                sys.exit(0)
            except Exception as e:
                print(f"Input error: {e}")

    @staticmethod
    def get_interval_input():
        """Get integration interval from user with validation."""
        while True:
            try:
                print("\nIntegration Interval:")
                a = float(input("Enter the start of the interval (a): "))
                b = float(input("Enter the end of the interval (b): "))

                if a >= b:
                    print("Error: Interval start must be less than end.")
                    continue

                if not (np.isfinite(a) and np.isfinite(b)):
                    print("Error: Interval bounds must be finite numbers.")
                    continue

                return a, b
            except ValueError:
                print("Error: Please enter valid numbers for the interval.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                sys.exit(0)

    @staticmethod
    def get_intervals_input():
        """Get number of intervals from user with suggestions."""
        print("\nRecommended intervals:")
        print("  - Quick test: 10-50 intervals")
        print("  - Normal accuracy: 100-500 intervals")
        print("  - High accuracy: 1000+ intervals")

        while True:
            try:
                n = int(input("Enter the number of intervals: "))
                if n <= 0:
                    print("Error: Number of intervals must be a positive integer.")
                    continue
                if n > 10000:
                    confirm = input(f"Warning: {n} intervals may be slow. Continue? (y/n): ")
                    if confirm.lower() != 'y':
                        continue
                return n
            except ValueError:
                print("Error: Please enter a valid positive integer.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                sys.exit(0)

    @staticmethod
    def get_method_choice():
        """Get integration method choice from user with descriptions."""
        methods = {
            '1': 'trapezoidal',
            '2': 'simpson_1_3',
            '3': 'simpson_3_8',
            '4': 'all'
        }

        while True:
            try:
                print("\nSelect integration method:")
                print("1. Trapezoidal Rule (Linear approximation)")
                print("2. Simpson's 1/3 Rule (Quadratic approximation, requires even intervals)")
                print("3. Simpson's 3/8 Rule (Cubic approximation, requires intervals divisible by 3)")
                print("4. All methods (with automatic interval adjustment)")

                choice = input("Enter your choice (1-4): ").strip()

                if choice in methods:
                    return methods[choice]
                else:
                    print("Error: Please enter a valid choice (1-4).")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                sys.exit(0)

    @staticmethod
    def display_results(results_dict, intervals_used, a, b):
        """Display integration results with interval information."""
        print(f"\n{'=' * 60}")
        print(f"INTEGRATION RESULTS FROM {a} TO {b}")
        print(f"{'=' * 60}")

        for method, result in results_dict.items():
            if result is not None and method != 'exact':
                intervals_info = f"(n={intervals_used.get(method, 'N/A')})"
                print(f"{method:25} {intervals_info:>10}: {result:.10f}")

        if 'exact' in results_dict and results_dict['exact'] is not None:
            print(f"{'Exact integral':25} {'':>10}: {results_dict['exact']:.10f}")

        # Calculate and display errors if exact solution exists
        if 'exact' in results_dict and results_dict['exact'] is not None:
            exact = results_dict['exact']
            print(f"\n{'Error Analysis':^60}")
            print("-" * 60)
            print(f"{'Method':25} {'Intervals':>10} {'Absolute Error':>15} {'Relative Error (%)':>15}")
            print("-" * 60)

            for method, result in results_dict.items():
                if method != 'exact' and result is not None:
                    error = abs(result - exact)
                    relative_error = (error / abs(exact)) * 100 if exact != 0 else 0
                    intervals_info = intervals_used.get(method, 'N/A')
                    print(f"{method:25} {intervals_info:>10} {error:>15.2e} {relative_error:>14.4f}")


def main():
    """Main program execution with enhanced error handling."""
    integrator = NumericalIntegrator()
    io_handler = InputOutput()

    print("=" * 60)
    print("NUMERICAL INTEGRATION CALCULATOR")
    print("=" * 60)
    print("This program computes definite integrals using numerical methods.")

    try:
        # Get inputs
        func_expr = io_handler.get_function_input()
        sympy_expr, numpy_func = integrator.parse_function(func_expr)
        print(f"✓ Function parsed successfully: {func_expr}")

        a, b = io_handler.get_interval_input()
        print(f"✓ Integration interval: [{a}, {b}]")

        n = io_handler.get_intervals_input()
        method_choice = io_handler.get_method_choice()

        # Get adjusted intervals for different methods
        intervals_dict = integrator.adaptive_method_selection(n, method_choice)

        # Store results
        results = {}
        intervals_used = {}

        # Compute exact integral
        print("\n" + "=" * 40)
        print("COMPUTING INTEGRALS...")
        print("=" * 40)

        exact_result = integrator.compute_exact_integral(sympy_expr, a, b)
        if exact_result is not None:
            results['exact'] = exact_result
            print("✓ Exact integral computed successfully")
        else:
            print("! Exact integral could not be computed symbolically")

        # Compute numerical approximations based on user choice
        if method_choice in ['trapezoidal', 'all']:
            try:
                n_trap = intervals_dict['trapezoidal']
                trap_result = integrator.trapezoidal_rule(numpy_func, a, b, n_trap)
                results['Trapezoidal Rule'] = trap_result
                intervals_used['Trapezoidal Rule'] = n_trap
                print(f"✓ Trapezoidal rule completed with {n_trap} intervals")
            except Exception as e:
                print(f"✗ Error with Trapezoidal Rule: {e}")

        if method_choice in ['simpson_1_3', 'all']:
            try:
                n_simp = intervals_dict['simpson_1_3']
                simp_result = integrator.simpsons_rule(numpy_func, a, b, n_simp)
                results["Simpson's 1/3 Rule"] = simp_result
                intervals_used["Simpson's 1/3 Rule"] = n_simp
                print(f"✓ Simpson's 1/3 rule completed with {n_simp} intervals")
                if n_simp != n:
                    print(f"  (Adjusted from {n} to {n_simp} for even requirement)")
            except Exception as e:
                print(f"✗ Error with Simpson's 1/3 Rule: {e}")

        if method_choice in ['simpson_3_8', 'all']:
            try:
                n_simp38 = intervals_dict['simpson_3_8']
                simp38_result = integrator.simpsons_3_8_rule(numpy_func, a, b, n_simp38)
                results["Simpson's 3/8 Rule"] = simp38_result
                intervals_used["Simpson's 3/8 Rule"] = n_simp38
                print(f"✓ Simpson's 3/8 rule completed with {n_simp38} intervals")
                if n_simp38 != n:
                    print(f"  (Adjusted from {n} to {n_simp38} for multiple of 3 requirement)")
            except Exception as e:
                print(f"✗ Error with Simpson's 3/8 Rule: {e}")

        # Display results
        io_handler.display_results(results, intervals_used, a, b)

        # Additional analysis options
        print(f"\n{'=' * 60}")
        print("ADDITIONAL OPTIONS")
        print("=" * 60)

        # Plot function
        plot_choice = input("Do you want to plot the function? (y/n): ").lower().strip()
        if plot_choice == 'y':
            integrator.plot_function_and_area(numpy_func, a, b,
                                              f"Integration of {func_expr} from {a} to {b}")

        # Convergence analysis
        if len(results) > 1:  # Only if we have numerical results
            conv_choice = input("Do you want to perform convergence analysis? (y/n): ").lower().strip()
            if conv_choice == 'y':
                method = list(results.keys())[0] if 'exact' not in results else list(results.keys())[1]
                method_key = method.lower().replace("'", "").replace(" ", "_")
                conv_results = integrator.convergence_analysis(numpy_func, a, b, method_key)

                if conv_results:
                    print(f"\nConvergence Analysis for {method}:")
                    print("-" * 40)
                    print(f"{'Intervals':>10} {'Result':>15} {'Change':>15}")
                    print("-" * 40)
                    for i, (n_val, result_val) in enumerate(conv_results):
                        if i == 0:
                            print(f"{n_val:>10} {result_val:>15.8f} {'':>15}")
                        else:
                            change = abs(result_val - conv_results[i - 1][1])
                            print(f"{n_val:>10} {result_val:>15.8f} {change:>15.2e}")

    except Exception as e:
        print(f"\nProgram error: {e}")
        return 1

    print(f"\n{'=' * 60}")
    print("Program completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)