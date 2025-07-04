#!/usr/bin/env python3

import math


def stirling_interpolation(x, y, target_x, show_table=False):
    """
    Perform Stirling's interpolation.

    Stirling's formula is a central difference interpolation method that provides
    high accuracy for interpolation near the center of the data range.

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
        raise ValueError("At least 3 data points are required for Stirling interpolation")

    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")

    # Check if x values are equally spaced
    h = x[1] - x[0]
    tolerance = 1e-10
    for i in range(n - 1):
        if abs((x[i + 1] - x[i]) - h) > tolerance:
            raise ValueError("x values must be equally spaced")

    # Calculate forward differences (will be used to construct central differences)
    diff = [[0.0 for _ in range(n)] for _ in range(n)]

    # First column is the y values
    for i in range(n):
        diff[i][0] = y[i]

    # Calculate forward differences
    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    # Display difference table if requested
    if show_table:
        print("\nForward Difference Table:")
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

    # Find the central position for Stirling interpolation
    # Stirling works best when the target point is near the center
    mid_index = n // 2

    # For even number of points, use the middle-left point as reference
    if n % 2 == 0:
        mid_index = mid_index - 1

    # Calculate u parameter relative to the central point
    u = (target_x - x[mid_index]) / h

    # Apply Stirling's interpolation formula
    # f(x) = y₀ + u(Δy₋₁ + Δy₀)/2 + u²Δ²y₋₁/2! + u(u²-1)(Δ³y₋₂ + Δ³y₋₁)/3!×2 + ...

    result = diff[mid_index][0]  # Start with y₀ (central value)

    # Add Stirling terms
    for j in range(1, min(n, mid_index + 1)):
        if j % 2 == 1:  # Odd order differences
            # Use average of central differences for odd orders
            if mid_index - j // 2 - 1 >= 0:
                # Average of two central differences
                diff1 = diff[mid_index - j // 2 - 1][j] if mid_index - j // 2 - 1 >= 0 else 0
                diff2 = diff[mid_index - j // 2][j] if mid_index - j // 2 >= 0 else 0
                avg_diff = (diff1 + diff2) / 2

                # Calculate u factor for odd terms: u, u(u²-1), u(u²-1)(u²-4), ...
                u_factor = u
                for k in range(1, (j + 1) // 2):
                    u_factor *= (u * u - k * k)

                term = avg_diff * u_factor / math.factorial(j)
                result += term
        else:  # Even order differences
            # Use central differences for even orders
            if mid_index - j // 2 >= 0:
                central_diff = diff[mid_index - j // 2][j]

                # Calculate u factor for even terms: u², u²(u²-1), u²(u²-1)(u²-4), ...
                u_factor = u * u
                for k in range(1, j // 2):
                    u_factor *= (u * u - k * k)

                term = central_diff * u_factor / math.factorial(j)
                result += term

    return result


def stirling_interpolation_simplified(x, y, target_x):
    """
    Simplified version of Stirling interpolation - more suitable for conversion.
    Uses a cleaner implementation of the Stirling formula.
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

    # Find central position
    mid_index = n // 2
    if n % 2 == 0:
        mid_index = mid_index - 1

    # Calculate u parameter
    u = (target_x - x[mid_index]) / h

    # Apply Stirling's formula using a simplified approach
    result = diff[mid_index][0]

    # Add terms systematically
    for order in range(1, min(n, mid_index + 1)):
        if order % 2 == 1:  # Odd order terms
            # First order: u * (Δy₋₁ + Δy₀)/2
            # Third order: u(u²-1) * (Δ³y₋₂ + Δ³y₋₁)/(3! * 2)
            if mid_index - order // 2 - 1 >= 0 and mid_index - order // 2 >= 0:
                diff_sum = diff[mid_index - order // 2 - 1][order] + diff[mid_index - order // 2][order]

                # Calculate u(u²-1)(u²-4)...
                u_product = u
                for k in range(1, (order + 1) // 2):
                    u_product *= (u * u - k * k)

                term = diff_sum * u_product / (2 * math.factorial(order))
                result += term
        else:  # Even order terms
            # Second order: u² * Δ²y₋₁/2!
            # Fourth order: u²(u²-1) * Δ⁴y₋₂/4!
            if mid_index - order // 2 >= 0:
                central_diff = diff[mid_index - order // 2][order]

                # Calculate u²(u²-1)(u²-4)...
                u_product = u * u
                for k in range(1, order // 2):
                    u_product *= (u * u - k * k)

                term = central_diff * u_product / math.factorial(order)
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
            print("Number of data points must be at least 3 for Stirling interpolation.")
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
    print("Stirling's Interpolation")
    print("=" * 23)

    try:
        # Get data points
        x, y = get_data_points()

        # Get target point
        target_x = validate_numeric_input("Enter the x value to interpolate: ")

        # Ask if user wants to see the difference table
        show_table = input("Show difference table? (y/n): ").lower().startswith('y')

        # Ask which method to use
        print("\nChoose interpolation method:")
        print("1. Full Stirling Interpolation")
        print("2. Simplified Stirling Interpolation (recommended for conversion)")
        method = validate_numeric_input("Enter choice (1 or 2): ", int)

        # Perform interpolation
        if method == 1:
            result = stirling_interpolation(x, y, target_x, show_table)
        else:
            result = stirling_interpolation_simplified(x, y, target_x)
            if show_table:
                # Show table for simplified version
                stirling_interpolation(x, y, target_x, True)

        print(f"\nResult:")
        print(f"The interpolated value at x = {target_x} is {result:.8f}")

        # Additional information
        print(f"\nAdditional Info:")
        print(f"Number of data points: {len(x)}")
        print(f"Step size (h): {x[1] - x[0]:.6f}")
        print(f"Interpolation range: [{x[0]:.6f}, {x[-1]:.6f}]")
        print(f"Central reference point: {x[len(x) // 2 if len(x) % 2 == 1 else len(x) // 2 - 1]:.6f}")

        # Check if target is in central region (recommended for Stirling)
        central_region_start = x[len(x) // 4]
        central_region_end = x[3 * len(x) // 4]
        if central_region_start <= target_x <= central_region_end:
            print("✓ Target point is in the central region (optimal for Stirling interpolation)")
        else:
            print("⚠ Target point is outside the central region (Stirling works best in the center)")

        if target_x < x[0] or target_x > x[-1]:
            print("Warning: Target point is outside the data range (extrapolation).")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()