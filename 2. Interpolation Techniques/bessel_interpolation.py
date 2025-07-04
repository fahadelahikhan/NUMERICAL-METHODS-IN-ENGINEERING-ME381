#!/usr/bin/env python3
"""
Bessel Interpolation Implementation
Author: Enhanced for cross-language compatibility
"""

import math


def calculate_forward_differences(y):
    """
    Calculate forward differences table for given y values.

    Args:
        y: List of y values

    Returns:
        2D list containing forward differences
    """
    n = len(y)
    diff = [[0.0 for _ in range(n)] for _ in range(n)]

    # First column is the original y values
    for i in range(n):
        diff[i][0] = y[i]

    # Calculate forward differences
    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

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


def bessel_interpolation(x, y, target_x):
    """
    Perform Bessel interpolation to find value at target_x.

    Args:
        x: List of x values (equally spaced)
        y: List of y values
        target_x: Target x value for interpolation

    Returns:
        Interpolated y value at target_x
    """
    n = len(x)
    h = x[1] - x[0]

    # Calculate forward differences
    diff = calculate_forward_differences(y)

    # Find the position and calculate u
    # For Bessel interpolation, we need to find the appropriate starting point
    position = (target_x - x[0]) / h
    k = int(position)  # Integer part
    u = position - k  # Fractional part

    # Adjust k to ensure we have enough points for interpolation
    # Bessel works best when the target is between x[k] and x[k+1]
    if k >= n - 1:
        k = n - 2
    if k < 0:
        k = 0

    # Recalculate u based on adjusted k
    u = (target_x - x[k]) / h

    # Bessel interpolation formula
    # Start with the average of y[k] and y[k+1]
    if k + 1 < n:
        result = (y[k] + y[k + 1]) / 2.0
    else:
        result = y[k]

    # Add correction terms
    max_terms = min(n - k - 1, k + 1)  # Ensure we don't go out of bounds

    for r in range(1, max_terms):
        if k - r >= 0 and k + r + 1 < n:
            # Calculate the binomial coefficient-like term
            term_coeff = 1.0
            for i in range(r):
                term_coeff *= (u - 0.5 - i) * (u - 0.5 + i + 1)
            term_coeff /= factorial(2 * r)

            # Add the difference term
            if r < n - k:
                avg_diff = (diff[k - r][2 * r] + diff[k - r + 1][2 * r]) / 2.0
                result += term_coeff * avg_diff

        # Odd order terms
        if r < max_terms - 1 and k - r >= 0 and k + r + 1 < n:
            term_coeff = 1.0
            for i in range(r):
                term_coeff *= (u - 0.5 - i) * (u - 0.5 + i + 1)
            term_coeff *= (u - 0.5)
            term_coeff /= factorial(2 * r + 1)

            if 2 * r + 1 < n - k + r:
                result += term_coeff * diff[k - r][2 * r + 1]

    return result


def validate_input_data(x, y):
    """
    Validate input data for Bessel interpolation.

    Args:
        x: List of x values
        y: List of y values

    Returns:
        Tuple (is_valid, error_message)
    """
    if len(x) != len(y):
        return False, "x and y arrays must have the same length"

    if len(x) < 2:
        return False, "At least 2 data points are required"

    # Check if x values are equally spaced
    h = x[1] - x[0]
    tolerance = 1e-10
    for i in range(len(x) - 1):
        if abs((x[i + 1] - x[i]) - h) > tolerance:
            return False, "x values must be equally spaced"

    return True, ""


def get_input_data():
    """
    Get input data from user.

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
            val = float(input(f"x[{i}]: "))
            x.append(val)

        print("Enter the y values:")
        for i in range(n):
            val = float(input(f"y[{i}]: "))
            y.append(val)

        target_x = float(input("Enter the x value to interpolate: "))

        return x, y, target_x

    except ValueError as e:
        print(f"Invalid input: {e}")
        return None


def display_results(target_x, result):
    """
    Display interpolation results.

    Args:
        target_x: Target x value
        result: Interpolated result
    """
    print(f"\nInterpolation Results:")
    print(f"Target x value: {target_x}")
    print(f"Interpolated y value: {result:.8f}")


def main():
    """
    Main function to run the Bessel interpolation program.
    """
    print("=== Bessel Interpolation Program ===")
    print("This program performs Bessel interpolation on equally spaced data points.\n")

    # Get input data
    input_data = get_input_data()
    if input_data is None:
        return

    x, y, target_x = input_data

    # Validate input
    is_valid, error_msg = validate_input_data(x, y)
    if not is_valid:
        print(f"Error: {error_msg}")
        return

    # Check if target_x is within reasonable bounds
    if target_x < x[0] or target_x > x[-1]:
        print("Warning: Target x value is outside the given data range.")
        print("Extrapolation may be less accurate than interpolation.")

    # Perform interpolation
    try:
        result = bessel_interpolation(x, y, target_x)
        display_results(target_x, result)
    except Exception as e:
        print(f"Error during interpolation: {e}")


# Test function for development/debugging
def run_test_case(x, y, target_x, expected=None, test_name=""):
    """
    Run a test case for Bessel interpolation.

    Args:
        x: List of x values
        y: List of y values
        target_x: Target x value
        expected: Expected result (optional)
        test_name: Name of the test case
    """
    print(f"\n--- Test Case: {test_name} ---")
    print(f"x values: {x}")
    print(f"y values: {y}")
    print(f"Target x: {target_x}")

    is_valid, error_msg = validate_input_data(x, y)
    if not is_valid:
        print(f"Validation Error: {error_msg}")
        return

    try:
        result = bessel_interpolation(x, y, target_x)
        print(f"Interpolated value: {result:.8f}")
        if expected is not None:
            error = abs(result - expected)
            print(f"Expected: {expected:.8f}")
            print(f"Absolute error: {error:.8f}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Uncomment the line below to run test cases instead of interactive mode
    # run_test_cases()
    main()