#!/usr/bin/env python3

import math


def newton_backward_interpolation(x, y, target_x, show_table=False):
    """
    Perform Newton's backward interpolation.

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
    if n < 2:
        raise ValueError("At least 2 data points are required")

    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")

    # Check if x values are equally spaced
    h = x[1] - x[0]
    tolerance = 1e-10
    for i in range(n - 1):
        if abs((x[i + 1] - x[i]) - h) > tolerance:
            raise ValueError("x values must be equally spaced")

    # Calculate the backward difference table
    diff = [[0.0 for _ in range(n)] for _ in range(n)]

    # First column is the y values
    for i in range(n):
        diff[i][0] = y[i]

    # Calculate backward differences
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            diff[i][j] = diff[i][j - 1] - diff[i - 1][j - 1]

    # Display difference table if requested
    if show_table:
        print("\nBackward Difference Table:")
        print("i\tx[i]\ty[i]", end="")
        for j in range(1, n):
            print(f"\t∇^{j}y[i]", end="")
        print()

        for i in range(n):
            print(f"{i}\t{x[i]:.2f}\t{diff[i][0]:.6f}", end="")
            for j in range(1, n):
                if i >= j:
                    print(f"\t{diff[i][j]:.6f}", end="")
                else:
                    print("\t-", end="")
            print()

    # Calculate interpolation using Newton's backward formula
    # p = (target_x - x_n) / h where x_n is the last point
    p = (target_x - x[-1]) / h

    # Start with the last y value
    result = diff[-1][0]

    # Add terms: p*∇y_n/1! + p(p+1)*∇²y_n/2! + ...
    for j in range(1, n):
        term = diff[-1][j]

        # Calculate p(p+1)(p+2)...(p+j-1)
        for i in range(j):
            term *= (p + i)

        # Divide by j!
        term /= math.factorial(j)

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
        if n <= 1:
            print("Number of data points must be at least 2.")
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
    print("Newton's Backward Interpolation")
    print("=" * 35)

    try:
        # Get data points
        x, y = get_data_points()

        # Get target point
        target_x = validate_numeric_input("Enter the x value to interpolate: ")

        # Ask if user wants to see the difference table
        show_table = input("Show difference table? (y/n): ").lower().startswith('y')

        # Perform interpolation
        result = newton_backward_interpolation(x, y, target_x, show_table)

        print(f"\nResult:")
        print(f"The interpolated value at x = {target_x} is {result:.8f}")

        # Additional information
        print(f"\nAdditional Info:")
        print(f"Number of data points: {len(x)}")
        print(f"Step size (h): {x[1] - x[0]:.6f}")
        print(f"Interpolation range: [{x[0]:.6f}, {x[-1]:.6f}]")

        if target_x < x[0] or target_x > x[-1]:
            print("Warning: Target point is outside the data range (extrapolation).")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()