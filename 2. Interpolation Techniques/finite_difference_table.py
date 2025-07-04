#!/usr/bin/env python3
"""
Finite Difference Table Implementation with Newton's Forward/Backward Interpolation
Author: Enhanced for cross-language compatibility and comprehensive functionality
"""

import math


def generate_forward_difference_table(y):
    """
    Generate forward difference table for given y values.

    Args:
        y: List of y values

    Returns:
        2D list containing forward differences
    """
    n = len(y)
    diff = [[0.0 for _ in range(n)] for _ in range(n)]

    # First column contains the original y values
    for i in range(n):
        diff[i][0] = y[i]

    # Calculate forward differences
    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    return diff


def generate_backward_difference_table(y):
    """
    Generate backward difference table for given y values.

    Args:
        y: List of y values

    Returns:
        2D list containing backward differences
    """
    n = len(y)
    diff = [[0.0 for _ in range(n)] for _ in range(n)]

    # First column contains the original y values
    for i in range(n):
        diff[i][0] = y[i]

    # Calculate backward differences
    for j in range(1, n):
        for i in range(j, n):
            diff[i][j] = diff[i][j - 1] - diff[i - 1][j - 1]

    return diff


def factorial(n):
    """
    Calculate factorial of n.

    Args:
        n: Integer value

    Returns:
        Factorial of n
    """
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def newton_forward_interpolation(x, y, target_x):
    """
    Perform Newton's forward interpolation.

    Args:
        x: List of x values (equally spaced)
        y: List of y values
        target_x: Target x value for interpolation

    Returns:
        Interpolated y value at target_x
    """
    n = len(x)
    h = x[1] - x[0]  # Step size

    # Generate forward difference table
    diff = generate_forward_difference_table(y)

    # Calculate u = (target_x - x[0]) / h
    u = (target_x - x[0]) / h

    # Newton's forward interpolation formula
    result = diff[0][0]  # y[0]

    for i in range(1, n):
        # Calculate binomial coefficient-like term
        term = diff[0][i]
        for j in range(i):
            term *= (u - j)
        term /= factorial(i)
        result += term

    return result


def newton_backward_interpolation(x, y, target_x):
    """
    Perform Newton's backward interpolation.

    Args:
        x: List of x values (equally spaced)
        y: List of y values
        target_x: Target x value for interpolation

    Returns:
        Interpolated y value at target_x
    """
    n = len(x)
    h = x[1] - x[0]  # Step size

    # Generate backward difference table
    diff = generate_backward_difference_table(y)

    # Calculate u = (target_x - x[n-1]) / h
    u = (target_x - x[n - 1]) / h

    # Newton's backward interpolation formula
    result = diff[n - 1][0]  # y[n-1]

    for i in range(1, n):
        # Calculate binomial coefficient-like term
        term = diff[n - 1][i]
        for j in range(i):
            term *= (u + j)
        term /= factorial(i)
        result += term

    return result


def validate_input_data(x, y):
    """
    Validate input data for finite difference interpolation.

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
    if len(x) < 2:
        return False, "At least 2 data points are required"

    # Check if x values are equally spaced
    if len(x) > 1:
        h = x[1] - x[0]
        tolerance = 1e-10
        for i in range(len(x) - 1):
            if abs((x[i + 1] - x[i]) - h) > tolerance:
                return False, "x values must be equally spaced for finite difference interpolation"

    # Check for invalid values (NaN, infinity)
    for i in range(len(x)):
        if math.isnan(x[i]) or math.isinf(x[i]):
            return False, f"Invalid x value at index {i}: {x[i]}"
        if math.isnan(y[i]) or math.isinf(y[i]):
            return False, f"Invalid y value at index {i}: {y[i]}"

    return True, ""


def print_forward_difference_table(x, diff):
    """
    Print forward difference table in a formatted manner.

    Args:
        x: List of x values
        diff: 2D list containing forward differences
    """
    n = len(x)
    print(f"\n{'=' * 80}")
    print("FORWARD DIFFERENCE TABLE")
    print(f"{'=' * 80}")

    # Print header
    header = f"{'i':<3} {'x':<10} {'y':<12}"
    for j in range(1, n):
        header += f"{'Δ^' + str(j) + 'y':<12}"
    print(header)
    print("-" * 80)

    # Print data rows
    for i in range(n):
        row = f"{i:<3} {x[i]:<10.4f} {diff[i][0]:<12.6f}"
        for j in range(1, n - i):
            row += f"{diff[i][j]:<12.6f}"
        print(row)

    print(f"{'=' * 80}")


def print_backward_difference_table(x, diff):
    """
    Print backward difference table in a formatted manner.

    Args:
        x: List of x values
        diff: 2D list containing backward differences
    """
    n = len(x)
    print(f"\n{'=' * 80}")
    print("BACKWARD DIFFERENCE TABLE")
    print(f"{'=' * 80}")

    # Print header
    header = f"{'i':<3} {'x':<10} {'y':<12}"
    for j in range(1, n):
        header += f"{'∇^' + str(j) + 'y':<12}"
    print(header)
    print("-" * 80)

    # Print data rows
    for i in range(n):
        row = f"{i:<3} {x[i]:<10.4f} {diff[i][0]:<12.6f}"
        for j in range(1, min(i + 2, n)):
            row += f"{diff[i][j]:<12.6f}"
        print(row)

    print(f"{'=' * 80}")


def get_input_data():
    """
    Get input data from user with enhanced error handling.

    Returns:
        Tuple (x, y) or None if input is invalid
    """
    try:
        n = int(input("Enter the number of data points: "))
        if n <= 0:
            print("Number of data points must be a positive integer.")
            return None

        x = []
        y = []

        print("Enter the x values (must be equally spaced):")
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

        return x, y

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return None
    except Exception as e:
        print(f"Unexpected error during input: {e}")
        return None


def get_interpolation_target():
    """
    Get target x value for interpolation.

    Returns:
        Target x value or None if invalid
    """
    while True:
        try:
            target_x = float(input("\nEnter the x value to interpolate (or press Enter to skip): "))
            if math.isnan(target_x) or math.isinf(target_x):
                print("Please enter a valid finite number.")
                continue
            return target_x
        except ValueError:
            # If user presses Enter without input, return None
            return None
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return None


def choose_interpolation_method(x, target_x):
    """
    Choose appropriate interpolation method based on target position.

    Args:
        x: List of x values
        target_x: Target x value

    Returns:
        String indicating recommended method
    """
    x_min, x_max = x[0], x[-1]
    mid_point = (x_min + x_max) / 2

    if target_x <= mid_point:
        return "forward"
    else:
        return "backward"


def display_interpolation_results(x, y, target_x, forward_result, backward_result, recommended_method):
    """
    Display interpolation results in a formatted manner.

    Args:
        x: List of x values
        y: List of y values
        target_x: Target x value
        forward_result: Result from forward interpolation
        backward_result: Result from backward interpolation
        recommended_method: Recommended interpolation method
    """
    print(f"\n{'=' * 60}")
    print("INTERPOLATION RESULTS")
    print(f"{'=' * 60}")

    print(f"Target x value: {target_x}")
    print(f"Data range: [{x[0]:.6f}, {x[-1]:.6f}]")

    # Determine if interpolation or extrapolation
    if x[0] <= target_x <= x[-1]:
        print("Status: Interpolation (target within data range)")
    else:
        print("Status: Extrapolation (target outside data range)")
        print("Warning: Extrapolation may be less accurate.")

    print(f"\nForward interpolation result: {forward_result:.8f}")
    print(f"Backward interpolation result: {backward_result:.8f}")
    print(f"Recommended method: {recommended_method.capitalize()}")

    # Calculate difference between methods
    difference = abs(forward_result - backward_result)
    print(f"Difference between methods: {difference:.2e}")

    if difference > 1e-10:
        print("Note: Significant difference between methods may indicate numerical instability.")

    print(f"{'=' * 60}")


def main():
    """
    Main function to run the finite difference table program.
    """
    print("=" * 80)
    print("FINITE DIFFERENCE TABLE AND INTERPOLATION PROGRAM")
    print("=" * 80)
    print("This program generates finite difference tables and performs interpolation.")
    print("Note: x values must be equally spaced for accurate interpolation.")
    print()

    # Get input data
    input_data = get_input_data()
    if input_data is None:
        return

    x, y = input_data

    # Validate input data
    is_valid, error_msg = validate_input_data(x, y)
    if not is_valid:
        print(f"Error: {error_msg}")
        return

    # Generate and display difference tables
    print("\n" + "=" * 80)
    print("GENERATING DIFFERENCE TABLES")
    print("=" * 80)

    # Forward difference table
    forward_diff = generate_forward_difference_table(y)
    print_forward_difference_table(x, forward_diff)

    # Backward difference table
    backward_diff = generate_backward_difference_table(y)
    print_backward_difference_table(x, backward_diff)

    # Optional interpolation
    target_x = get_interpolation_target()
    if target_x is not None:
        try:
            # Perform both interpolations
            forward_result = newton_forward_interpolation(x, y, target_x)
            backward_result = newton_backward_interpolation(x, y, target_x)

            # Choose recommended method
            recommended_method = choose_interpolation_method(x, target_x)

            # Display results
            display_interpolation_results(x, y, target_x, forward_result, backward_result, recommended_method)

        except Exception as e:
            print(f"Error during interpolation: {e}")

    print("\nProgram completed successfully!")


# Test function for development/debugging
def run_test_case(x, y, target_x, expected_forward=None, expected_backward=None, test_name=""):
    """
    Run a test case for finite difference interpolation.

    Args:
        x: List of x values
        y: List of y values
        target_x: Target x value
        expected_forward: Expected forward interpolation result
        expected_backward: Expected backward interpolation result
        test_name: Name of the test case
    """
    print(f"\n{'=' * 60}")
    print(f"TEST CASE: {test_name}")
    print(f"{'=' * 60}")
    print(f"x values: {x}")
    print(f"y values: {y}")
    print(f"Target x: {target_x}")

    # Validate input
    is_valid, error_msg = validate_input_data(x, y)
    if not is_valid:
        print(f"Validation Error: {error_msg}")
        return False

    try:
        # Generate difference tables
        forward_diff = generate_forward_difference_table(y)
        backward_diff = generate_backward_difference_table(y)

        # Perform interpolations
        forward_result = newton_forward_interpolation(x, y, target_x)
        backward_result = newton_backward_interpolation(x, y, target_x)

        print(f"Forward interpolation result: {forward_result:.8f}")
        print(f"Backward interpolation result: {backward_result:.8f}")

        # Check against expected results
        tolerance = 1e-8
        forward_ok = True
        backward_ok = True

        if expected_forward is not None:
            forward_error = abs(forward_result - expected_forward)
            print(f"Expected forward result: {expected_forward:.8f}")
            print(f"Forward error: {forward_error:.2e}")
            forward_ok = forward_error <= tolerance

        if expected_backward is not None:
            backward_error = abs(backward_result - expected_backward)
            print(f"Expected backward result: {expected_backward:.8f}")
            print(f"Backward error: {backward_error:.2e}")
            backward_ok = backward_error <= tolerance

        if forward_ok and backward_ok:
            print("✓ Test PASSED")
            return True
        else:
            print("✗ Test FAILED")
            return False

    except Exception as e:
        print(f"Error during test: {e}")
        print("✗ Test FAILED")
        return False


def run_all_tests():
    """
    Run all test cases to verify the implementation.
    """
    print("RUNNING ALL TEST CASES...")

    # Test Case 1: Linear function
    test1_passed = run_test_case(
        x=[0, 1, 2, 3, 4],
        y=[1, 3, 5, 7, 9],
        target_x=2.5,
        expected_forward=6.0,
        expected_backward=6.0,
        test_name="Linear Function (y = 2x + 1)"
    )

    # Test Case 2: Quadratic function
    test2_passed = run_test_case(
        x=[0, 1, 2, 3, 4],
        y=[0, 1, 4, 9, 16],
        target_x=1.5,
        expected_forward=2.25,
        expected_backward=2.25,
        test_name="Quadratic Function (y = x²)"
    )

    # Test Case 3: Cubic function
    test3_passed = run_test_case(
        x=[0, 1, 2, 3],
        y=[0, 1, 8, 27],
        target_x=1.5,
        expected_forward=3.375,
        expected_backward=3.375,
        test_name="Cubic Function (y = x³)"
    )

    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Test 1 (Linear): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Quadratic): {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Test 3 (Cubic): {'PASSED' if test3_passed else 'FAILED'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Uncomment the line below to run test cases instead of interactive mode
    # run_all_tests()
    main()