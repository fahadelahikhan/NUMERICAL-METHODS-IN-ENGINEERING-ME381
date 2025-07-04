#!/usr/bin/env python3

def validate_input_data(x, y):
    """
    Validate input data for interpolation
    Returns: (is_valid, error_message)
    """
    if len(x) != len(y):
        return False, "X and Y arrays must have the same length"

    if len(x) < 2:
        return False, "At least 2 data points are required"

    # Check if x values are in ascending order
    for i in range(len(x) - 1):
        if x[i] >= x[i + 1]:
            return False, "X values must be in strictly ascending order"

    return True, ""


def linear_interpolation(x, y, target_x):
    """
    Perform linear interpolation
    Args:
        x: array of x values (independent variable)
        y: array of y values (dependent variable)
        target_x: x value to interpolate at
    Returns:
        interpolated y value
    """
    n = len(x)

    # Handle edge cases
    if target_x <= x[0]:
        return y[0]  # Extrapolation: return first point
    if target_x >= x[n - 1]:
        return y[n - 1]  # Extrapolation: return last point

    # Find the interval containing target_x
    for i in range(n - 1):
        if x[i] <= target_x <= x[i + 1]:
            # Linear interpolation formula
            slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            interpolated_value = y[i] + slope * (target_x - x[i])
            return interpolated_value

    # This should never be reached due to edge case handling
    return y[0]


def lagrange_interpolation(x, y, target_x):
    """
    Perform Lagrange interpolation (polynomial interpolation)
    Args:
        x: array of x values (independent variable)
        y: array of y values (dependent variable)
        target_x: x value to interpolate at
    Returns:
        interpolated y value
    """
    n = len(x)
    result = 0.0

    for i in range(n):
        # Calculate the Lagrange basis polynomial L_i(target_x)
        basis = 1.0
        for j in range(n):
            if i != j:
                basis *= (target_x - x[j]) / (x[i] - x[j])

        result += y[i] * basis

    return result


def newton_divided_difference(x, y, target_x):
    """
    Perform Newton's divided difference interpolation
    Args:
        x: array of x values (independent variable)
        y: array of y values (dependent variable)
        target_x: x value to interpolate at
    Returns:
        interpolated y value
    """
    n = len(x)

    # Create divided difference table
    dd_table = [[0.0 for _ in range(n)] for _ in range(n)]

    # Fill first column with y values
    for i in range(n):
        dd_table[i][0] = y[i]

    # Fill the divided difference table
    for j in range(1, n):
        for i in range(n - j):
            dd_table[i][j] = (dd_table[i + 1][j - 1] - dd_table[i][j - 1]) / (x[i + j] - x[i])

    # Calculate interpolated value using Newton's formula
    result = dd_table[0][0]
    product = 1.0

    for i in range(1, n):
        product *= (target_x - x[i - 1])
        result += dd_table[0][i] * product

    return result


def read_data_points():
    """
    Read data points from user input
    Returns: (x_values, y_values, success)
    """
    try:
        n = int(input("Enter the number of data points: "))
        if n <= 1:
            print("Error: Number of data points must be at least 2.")
            return [], [], False

        x = []
        y = []

        print("Enter the data points (x y pairs):")
        print("Format: x_value y_value (one pair per line)")

        for i in range(n):
            line = input(f"Point {i + 1}: ").strip()
            parts = line.split()

            if len(parts) != 2:
                print("Error: Please enter exactly two values per line (x y)")
                return [], [], False

            x_val = float(parts[0])
            y_val = float(parts[1])

            x.append(x_val)
            y.append(y_val)

        return x, y, True

    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        return [], [], False


def display_results(target_x, methods_results):
    """
    Display interpolation results
    Args:
        target_x: the x value that was interpolated
        methods_results: dictionary of method names and their results
    """
    print(f"\nInterpolation Results for x = {target_x}:")
    print("-" * 50)

    for method_name, result in methods_results.items():
        print(f"{method_name:25}: {result:.6f}")


def main():
    """
    Main function to orchestrate the interpolation program
    """
    print("=== Real World Interpolation Program ===")
    print("This program supports multiple interpolation methods:")
    print("1. Linear Interpolation")
    print("2. Lagrange Interpolation")
    print("3. Newton's Divided Difference")
    print()

    # Read input data
    x, y, success = read_data_points()
    if not success:
        return

    # Validate input data
    is_valid, error_msg = validate_input_data(x, y)
    if not is_valid:
        print(f"Error: {error_msg}")
        return

    # Get target x value
    try:
        target_x = float(input("Enter the x value to interpolate at: "))
    except ValueError:
        print("Error: Invalid target x value")
        return

    # Perform interpolations
    methods_results = {}

    try:
        # Linear interpolation
        linear_result = linear_interpolation(x, y, target_x)
        methods_results["Linear Interpolation"] = linear_result

        # Lagrange interpolation
        lagrange_result = lagrange_interpolation(x, y, target_x)
        methods_results["Lagrange Interpolation"] = lagrange_result

        # Newton's divided difference
        newton_result = newton_divided_difference(x, y, target_x)
        methods_results["Newton's Method"] = newton_result

        # Display results
        display_results(target_x, methods_results)

        # Show data range info
        print(f"\nData Range: x ∈ [{min(x):.2f}, {max(x):.2f}]")
        if target_x < min(x) or target_x > max(x):
            print("Note: Target x is outside the data range (extrapolation)")

    except Exception as e:
        print(f"Error during interpolation: {e}")


if __name__ == "__main__":
    main()