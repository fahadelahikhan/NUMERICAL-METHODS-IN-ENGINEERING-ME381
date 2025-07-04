#!/usr/bin/env python3

import math


def gauss_central_interpolation(x, y, target_x, show_table=False):
    """
    Perform Gauss central interpolation.

    Args:
        x: List of x values (must be equally spaced)
        y: List of y values
        target_x: Point to interpolate
        show_table: Whether to display the difference table

    Returns:
        Interpolated value at target_x
    """
    n = len(x)

    # Validate inputs
    if n < 3:
        raise ValueError("At least 3 data points are required for central interpolation")

    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")

    # Check if x values are equally spaced
    h = x[1] - x[0]
    tolerance = 1e-10
    for i in range(n - 1):
        if abs((x[i + 1] - x[i]) - h) > tolerance:
            raise ValueError("x values must be equally spaced")

    # Calculate the central difference table
    diff = [[0.0 for _ in range(n)] for _ in range(n)]

    # First column is the y values
    for i in range(n):
        diff[i][0] = y[i]

    # Calculate forward differences (will be used to construct central differences)
    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    # Display difference table if requested
    if show_table:
        print("\nForward Difference Table (for Central Interpolation):")
        print("i\tx[i]\ty[i]", end="")
        for j in range(1, n):
            print(f"\tΔ^{j}y[i]", end="")
        print()

        for i in range(n):
            print(f"{i}\t{x[i]:.2f}\t{diff[i][0]:.6f}", end="")
            for j in range(1, n):
                if i <= n - j - 1:
                    print(f"\t{diff[i][j]:.6f}", end="")
                else:
                    print("\t-", end="")
            print()

    # Find the central position
    # For central interpolation, we need to find the middle point
    mid_index = n // 2
    x_mid = x[mid_index] if n % 2 == 1 else (x[mid_index - 1] + x[mid_index]) / 2

    # Calculate u parameter for central interpolation
    u = (target_x - x_mid) / h

    # Gauss Central Interpolation Formula
    # For odd number of points: uses central differences around the middle point
    # For even number of points: uses average of central differences

    if n % 2 == 1:  # Odd number of points
        # Use Gauss forward central difference formula
        result = diff[mid_index][0]  # y_0 (middle value)

        # Add terms alternately forward and backward
        for j in range(1, n):
            if j % 2 == 1:  # Odd order differences
                # Use forward differences
                term = diff[mid_index - j // 2][j]
                coeff = u
                for k in range(1, j):
                    if k % 2 == 1:
                        coeff *= (u + k // 2 + 1)
                    else:
                        coeff *= (u - k // 2)
                term = term * coeff / math.factorial(j)
                result += term
            else:  # Even order differences
                # Use central differences (average of forward differences)
                if mid_index - j // 2 >= 0:
                    term = diff[mid_index - j // 2][j]
                    coeff = u
                    for k in range(1, j):
                        if k % 2 == 1:
                            coeff *= (u + k // 2 + 1)
                        else:
                            coeff *= (u - k // 2)
                    term = term * coeff / math.factorial(j)
                    result += term
    else:  # Even number of points
        # Use Gauss backward central difference formula
        mid_index = n // 2 - 1
        result = diff[mid_index][0]  # y_0

        for j in range(1, n):
            if j % 2 == 1:  # Odd order differences
                term = diff[mid_index - j // 2][j]
                coeff = u + 0.5
                for k in range(1, j):
                    if k % 2 == 1:
                        coeff *= (u + 0.5 + k // 2)
                    else:
                        coeff *= (u + 0.5 - k // 2 - 1)
                term = term * coeff / math.factorial(j)
                result += term
            else:  # Even order differences
                if mid_index - j // 2 + 1 >= 0:
                    term = diff[mid_index - j // 2 + 1][j]
                    coeff = u + 0.5
                    for k in range(1, j):
                        if k % 2 == 1:
                            coeff *= (u + 0.5 + k // 2)
                        else:
                            coeff *= (u + 0.5 - k // 2 - 1)
                    term = term * coeff / math.factorial(j)
                    result += term

    return result


def gauss_central_interpolation_simplified(x, y, target_x):
    """
    Simplified version of Gauss central interpolation using Newton's approach
    with central positioning - more suitable for conversion to other languages.
    """
    n = len(x)
    h = x[1] - x[0]

    # Calculate forward differences
    diff = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        diff[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    # Find the best central position
    # Use the middle point as reference
    mid_index = n // 2
    if n % 2 == 0:
        mid_index = mid_index - 1

    # Calculate u parameter
    u = (target_x - x[mid_index]) / h

    # Apply Newton's formula centered around the middle point
    result = diff[mid_index][0]

    # Add positive and negative terms alternately
    for j in range(1, min(n, mid_index + 1)):
        if mid_index - j >= 0:
            # Forward term
            term = diff[mid_index - j][j]
            coeff = u
            for k in range(1, j):
                coeff *= (u + k)
            term = term * coeff / math.factorial(j)
            result += term

        if mid_index + j < n and j < n - mid_index:
            # Backward term (if we have enough points)
            term = diff[mid_index][j]
            coeff = u
            for k in range(1, j):
                coeff *= (u - k)
            term = term * coeff / math.factorial(j)
            result += term

    return result


def validate_numeric_input(prompt, input_type=float):
    """Helper function to validate numeric input"""
    while True:
        try:
            value = input_type(input(prompt))
            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")


def get_data_points():
    """Get data points from user input"""
    while True:
        n = validate_numeric_input("Enter the number of data points: ", int)
        if n <= 2:
            print("Number of data points must be at least 3 for central interpolation.")
            continue
        break

    x = []
    y = []

    print("Enter the x values:")
    for i in range(n):
        val = validate_numeric_input(f"x[{i}]: ")
        x.append(val)

    print("Enter the y values:")
    for i in range(n):
        val = validate_numeric_input(f"y[{i}]: ")
        y.append(val)

    return x, y


def main():
    """Main function to handle user interaction"""
    print("Gauss Central Interpolation")
    print("=" * 27)

    try:
        # Get data points
        x, y = get_data_points()

        # Get target point
        target_x = validate_numeric_input("Enter the x value to interpolate: ")

        # Ask if user wants to see the difference table
        show_table = input("Show difference table? (y/n): ").lower().startswith('y')

        # Ask which method to use
        print("\nChoose interpolation method:")
        print("1. Full Gauss Central Interpolation")
        print("2. Simplified Central Interpolation (recommended for conversion)")
        method = validate_numeric_input("Enter choice (1 or 2): ", int)

        # Perform interpolation
        if method == 1:
            result = gauss_central_interpolation(x, y, target_x, show_table)
        else:
            result = gauss_central_interpolation_simplified(x, y, target_x)
            if show_table:
                # Show table for simplified version
                gauss_central_interpolation(x, y, target_x, True)

        print(f"\nResult:")
        print(f"The interpolated value at x = {target_x} is {result:.8f}")

        # Additional information
        print(f"\nAdditional Info:")
        print(f"Number of data points: {len(x)}")
        print(f"Step size (h): {x[1] - x[0]:.6f}")
        print(f"Interpolation range: [{x[0]:.6f}, {x[-1]:.6f}]")
        print(f"Central point: {x[len(x) // 2]:.6f}")

        if target_x < x[0] or target_x > x[-1]:
            print("Warning: Target point is outside the data range (extrapolation).")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()