#!/usr/bin/env python3
"""
Lagrange Interpolation Implementation
Author: Enhanced for cross-language compatibility and robustness
"""

import math


def lagrange_basis_polynomial(x, i, target_x):
    """
    Calculate the i-th Lagrange basis polynomial L_i(target_x).

    Args:
        x: List of x values (data points)
        i: Index of the current basis polynomial
        target_x: Target x value for evaluation

    Returns:
        Value of L_i(target_x)
    """
    n = len(x)
    result = 1.0

    for j in range(n):
        if i != j:
            # Check for division by zero
            denominator = x[i] - x[j]
            if abs(denominator) < 1e-15:
                raise ValueError(f"Duplicate x values found at indices {i} and {j}: x[{i}] = x[{j}] = {x[i]}")

            result *= (target_x - x[j]) / denominator

    return result


def lagrange_interpolation(x, y, target_x):
    """
    Perform Lagrange interpolation to find value at target_x.

    Args:
        x: List of x values (data points)
        y: List of y values (data points)
        target_x: Target x value for interpolation

    Returns:
        Interpolated y value at target_x
    """
    n = len(x)
    result = 0.0

    for i in range(n):
        # Calculate the i-th Lagrange basis polynomial
        basis_value = lagrange_basis_polynomial(x, i, target_x)

        # Add the contribution of this term to the result
        result += y[i] * basis_value

    return result


def validate_input_data(x, y):
    """
    Validate input data for Lagrange interpolation.

    Args:
        x: List of x values
        y: List of y values

    Returns:
        Tuple (is_valid, error_message)
    """
    # Check if arrays have the same length
    if len(x) != len(y):
        return False, "x and y arrays must have the same length"

    # Check minimum number of points
    if len(x) < 1:
        return False, "At least 1 data point is required"

    # Check for duplicate x values
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if abs(x[i] - x[j]) < 1e-15:
                return False, f"Duplicate x values found: x[{i}] = x[{j}] = {x[i]}"

    # Check for invalid values (NaN, infinity)
    for i in range(len(x)):
        if math.isnan(x[i]) or math.isinf(x[i]):
            return False, f"Invalid x value at index {i}: {x[i]}"
        if math.isnan(y[i]) or math.isinf(y[i]):
            return False, f"Invalid y value at index {i}: {y[i]}"

    return True, ""


def get_input_data():
    """
    Get input data from user with enhanced error handling.

    Returns:
        Tuple (x, y, target_x) or None if input is invalid
    """
    try:
        n = int(input("Enter the number of data points: "))
        if n <= 0:
            print("Number of data points must be a positive integer.")
            return None

        x = []
        y = []

        print("Enter the x values:")
        for i in range(n):
            while True:
                try:
                    val = float(input(f"x[{i}]: "))
                    if math.isnan(val) or math.isinf(val):
                        print("Please enter a valid finite number.")
                        continue
                    x.append(val)
                    break
                except ValueError:
                    print("Please enter a valid number.")

        print("Enter the y values:")
        for i in range(n):
            while True:
                try:
                    val = float(input(f"y[{i}]: "))
                    if math.isnan(val) or math.isinf(val):
                        print("Please enter a valid finite number.")
                        continue
                    y.append(val)
                    break
                except ValueError:
                    print("Please enter a valid number.")

        while True:
            try:
                target_x = float(input("Enter the x value to interpolate: "))
                if math.isnan(target_x) or math.isinf(target_x):
                    print("Please enter a valid finite number.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")

        return x, y, target_x

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return None
    except Exception as e:
        print(f"Unexpected error during input: {e}")
        return None


def display_results(x, y, target_x, result):
    """
    Display interpolation results in a formatted manner.

    Args:
        x: List of x values
        y: List of y values
        target_x: Target x value
        result: Interpolated result
    """
    print(f"\n{'=' * 50}")
    print("LAGRANGE INTERPOLATION RESULTS")
    print(f"{'=' * 50}")

    print(f"Number of data points: {len(x)}")
    print(f"Target x value: {target_x}")
    print(f"Interpolated y value: {result:.10f}")

    # Show if target is within or outside the data range
    x_min, x_max = min(x), max(x)
    if x_min <= target_x <= x_max:
        print(f"Status: Interpolation (target within data range [{x_min:.6f}, {x_max:.6f}])")
    else:
        print(f"Status: Extrapolation (target outside data range [{x_min:.6f}, {x_max:.6f}])")
        print("Warning: Extrapolation may be less accurate than interpolation.")

    print(f"{'=' * 50}")


def display_data_table(x, y):
    """
    Display the input data in a formatted table.

    Args:
        x: List of x values
        y: List of y values
    """
    print(f"\n{'=' * 30}")
    print("INPUT DATA TABLE")
    print(f"{'=' * 30}")
    print(f"{'i':<3} {'x':<12} {'y':<12}")
    print(f"{'-' * 30}")

    for i in range(len(x)):
        print(f"{i:<3} {x[i]:<12.6f} {y[i]:<12.6f}")

    print(f"{'=' * 30}")


def calculate_polynomial_degree_warning(n):
    """
    Warn user about potential issues with high-degree polynomials.

    Args:
        n: Number of data points
    """
    if n > 10:
        print(f"\nWarning: Using {n} data points results in a polynomial of degree {n - 1}.")
        print("High-degree polynomials may exhibit oscillatory behavior (Runge's phenomenon).")
        print("Consider using fewer points or alternative interpolation methods for better stability.")


def main():
    """
    Main function to run the Lagrange interpolation program.
    """
    print("=" * 60)
    print("LAGRANGE INTERPOLATION PROGRAM")
    print("=" * 60)
    print("This program performs Lagrange interpolation on given data points.")
    print("Note: All x values must be distinct (no duplicates allowed).")
    print()

    # Get input data
    input_data = get_input_data()
    if input_data is None:
        return

    x, y, target_x = input_data

    # Validate input data
    is_valid, error_msg = validate_input_data(x, y)
    if not is_valid:
        print(f"Error: {error_msg}")
        return

    # Display input data
    display_data_table(x, y)

    # Warn about high-degree polynomials
    calculate_polynomial_degree_warning(len(x))

    # Perform interpolation
    try:
        result = lagrange_interpolation(x, y, target_x)
        display_results(x, y, target_x, result)

    except ValueError as e:
        print(f"Interpolation error: {e}")
    except Exception as e:
        print(f"Unexpected error during interpolation: {e}")


# Test function for development/debugging
def run_test_case(x, y, target_x, expected=None, test_name="", tolerance=1e-10):
    """
    Run a test case for Lagrange interpolation.

    Args:
        x: List of x values
        y: List of y values
        target_x: Target x value
        expected: Expected result (optional)
        test_name: Name of the test case
        tolerance: Tolerance for comparison with expected result
    """
    print(f"\n{'=' * 50}")
    print(f"TEST CASE: {test_name}")
    print(f"{'=' * 50}")
    print(f"x values: {x}")
    print(f"y values: {y}")
    print(f"Target x: {target_x}")

    # Validate input
    is_valid, error_msg = validate_input_data(x, y)
    if not is_valid:
        print(f"Validation Error: {error_msg}")
        return False

    try:
        result = lagrange_interpolation(x, y, target_x)
        print(f"Interpolated value: {result:.10f}")

        if expected is not None:
            error = abs(result - expected)
            print(f"Expected value: {expected:.10f}")
            print(f"Absolute error: {error:.2e}")

            if error <= tolerance:
                print("✓ Test PASSED")
                return True
            else:
                print("✗ Test FAILED")
                return False
        else:
            print("✓ Test completed (no expected value provided)")
            return True

    except Exception as e:
        print(f"Error during interpolation: {e}")
        print("✗ Test FAILED")
        return False


def run_all_tests():
    """
    Run all test cases to verify the implementation.
    """
    print("RUNNING ALL TEST CASES...")

    # Test Case 1: Linear function
    test1_passed = run_test_case(
        x=[0, 1, 2],
        y=[1, 3, 5],
        target_x=1.5,
        expected=4.0,
        test_name="Linear Function (y = 2x + 1)"
    )

    # Test Case 2: Quadratic function
    test2_passed = run_test_case(
        x=[0, 1, 2, 3],
        y=[0, 1, 4, 9],
        target_x=2.5,
        expected=6.25,
        test_name="Quadratic Function (y = x²)"
    )

    # Test Case 3: Exact match
    test3_passed = run_test_case(
        x=[1, 2, 3, 4],
        y=[2, 4, 6, 8],
        target_x=3,
        expected=6.0,
        test_name="Exact Match Test"
    )

    print(f"\n{'=' * 50}")
    print("TEST SUMMARY")
    print(f"{'=' * 50}")
    print(f"Test 1 (Linear): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Quadratic): {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Test 3 (Exact Match): {'PASSED' if test3_passed else 'FAILED'}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    # Uncomment the line below to run test cases instead of interactive mode
    # run_all_tests()
    main()