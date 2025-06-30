#!/usr/bin/env python3

def create_augmented_matrix(coefficients, constants):
    """
    Create augmented matrix from coefficient matrix and constants vector.
    This function makes the data structure explicit for easier translation.
    """
    n = len(coefficients)
    augmented = []
    for i in range(n):
        row = coefficients[i][:] + [constants[i]]  # Deep copy + append constant
        augmented.append(row)
    return augmented


def partial_pivoting(matrix, current_row):
    """
    Find the row with maximum absolute value in current column for pivoting.
    Returns the index of the best pivot row.
    """
    n = len(matrix)
    max_row = current_row
    max_val = abs(matrix[current_row][current_row])

    for i in range(current_row + 1, n):
        if abs(matrix[i][current_row]) > max_val:
            max_val = abs(matrix[i][current_row])
            max_row = i

    return max_row


def swap_rows(matrix, row1, row2):
    """
    Swap two rows in the matrix.
    """
    if row1 != row2:
        matrix[row1], matrix[row2] = matrix[row2], matrix[row1]


def forward_elimination(matrix):
    """
    Perform forward elimination with partial pivoting.
    Modifies the matrix in-place to upper triangular form.
    """
    n = len(matrix)

    for i in range(n):
        # Find best pivot and swap if necessary
        pivot_row = partial_pivoting(matrix, i)
        swap_rows(matrix, i, pivot_row)

        # Check for zero pivot after pivoting
        if abs(matrix[i][i]) < 1e-12:  # Use small epsilon instead of exact zero
            raise ValueError(f"Matrix is singular or near-singular at row {i + 1}")

        # Eliminate column entries below pivot
        for j in range(i + 1, n):
            if matrix[j][i] != 0:  # Skip if already zero
                factor = matrix[j][i] / matrix[i][i]
                # Update entire row
                for k in range(i, n + 1):
                    matrix[j][k] -= factor * matrix[i][k]


def back_substitution(matrix):
    """
    Perform back substitution on upper triangular matrix.
    Returns the solution vector.
    """
    n = len(matrix)
    solution = [0.0] * n

    for i in range(n - 1, -1, -1):
        # Check for zero diagonal element
        if abs(matrix[i][i]) < 1e-12:
            raise ValueError(f"Matrix is singular at diagonal element {i + 1}")

        # Calculate solution for current variable
        sum_ax = 0.0
        for j in range(i + 1, n):
            sum_ax += matrix[i][j] * solution[j]

        solution[i] = (matrix[i][n] - sum_ax) / matrix[i][i]

    return solution


def gauss_elimination_solver(coefficients, constants):
    """
    Solve system of linear equations using Gauss elimination with partial pivoting.

    Args:
        coefficients: 2D list representing coefficient matrix
        constants: 1D list representing constants vector

    Returns:
        solution: 1D list representing solution vector
    """
    # Input validation
    n = len(coefficients)
    if n == 0:
        raise ValueError("Empty coefficient matrix")

    if len(constants) != n:
        raise ValueError("Number of constants must equal number of equations")

    for i, row in enumerate(coefficients):
        if len(row) != n:
            raise ValueError(f"Row {i + 1} has incorrect number of coefficients")

    # Create augmented matrix (deep copy to avoid modifying input)
    matrix = create_augmented_matrix(coefficients, constants)

    # Solve using Gauss elimination
    forward_elimination(matrix)
    solution = back_substitution(matrix)

    return solution


def print_matrix(matrix, title="Matrix"):
    """
    Utility function to print matrix in readable format.
    """
    print(f"\n{title}:")
    for row in matrix:
        print("  " + "  ".join(f"{val:8.4f}" for val in row))


def print_solution(solution, precision=6):
    """
    Print solution vector in readable format.
    """
    print(f"\nSolution:")
    for i, val in enumerate(solution):
        print(f"x{i + 1} = {val:.{precision}f}")


def validate_solution(coefficients, constants, solution, tolerance=1e-10):
    """
    Verify that the solution satisfies the original equations.
    """
    n = len(solution)
    print(f"\nSolution Verification (tolerance: {tolerance}):")

    max_error = 0.0
    for i in range(n):
        calculated = sum(coefficients[i][j] * solution[j] for j in range(n))
        error = abs(calculated - constants[i])
        max_error = max(max_error, error)
        status = "✓" if error < tolerance else "✗"
        print(f"Equation {i + 1}: {calculated:.10f} = {constants[i]:.10f} {status} (error: {error:.2e})")

    print(f"Maximum error: {max_error:.2e}")
    return max_error < tolerance


def read_system_from_input():
    """
    Read system of equations from user input.
    Returns coefficients matrix and constants vector.
    """
    try:
        n = int(input("Enter the number of equations: "))
        if n <= 0:
            raise ValueError("Number of equations must be positive")

        coefficients = []
        constants = []

        print(f"Enter the coefficients for each equation:")
        print(f"(Enter {n} coefficients separated by spaces)")

        for i in range(n):
            while True:
                try:
                    row_input = input(f"Equation {i + 1} coefficients: ").strip()
                    row = list(map(float, row_input.split()))
                    if len(row) != n:
                        print(f"Please enter exactly {n} coefficients.")
                        continue
                    coefficients.append(row)
                    break
                except ValueError:
                    print("Invalid input. Please enter numeric values separated by spaces.")

        print(f"Enter the constants for each equation:")
        for i in range(n):
            while True:
                try:
                    constant = float(input(f"Constant for equation {i + 1}: "))
                    constants.append(constant)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")

        return coefficients, constants

    except ValueError as e:
        raise ValueError(f"Input error: {e}")


def main():
    """
    Main function demonstrating the Gauss elimination solver.
    """
    print("Gauss Elimination Solver with Partial Pivoting")
    print("=" * 50)

    try:
        # Read input
        coefficients, constants = read_system_from_input()

        # Display input system
        print(f"\nInput System:")
        for i in range(len(coefficients)):
            equation = " + ".join(f"{coefficients[i][j]:.3f}*x{j + 1}" for j in range(len(coefficients[i])))
            print(f"  {equation} = {constants[i]:.3f}")

        # Solve system
        solution = gauss_elimination_solver(coefficients, constants)

        # Display results
        print_solution(solution)

        # Verify solution
        is_valid = validate_solution(coefficients, constants, solution)

        if is_valid:
            print("\n✓ Solution verified successfully!")
        else:
            print("\n✗ Solution verification failed - check for numerical errors.")

    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()