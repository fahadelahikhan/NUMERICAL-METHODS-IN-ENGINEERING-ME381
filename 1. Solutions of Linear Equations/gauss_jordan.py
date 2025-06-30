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


def find_pivot_row(matrix, col, start_row):
    """
    Find the row with maximum absolute value in specified column.
    Used for partial pivoting to improve numerical stability.
    """
    n = len(matrix)
    max_row = start_row
    max_val = abs(matrix[start_row][col])

    for i in range(start_row + 1, n):
        if abs(matrix[i][col]) > max_val:
            max_val = abs(matrix[i][col])
            max_row = i

    return max_row


def swap_rows(matrix, row1, row2):
    """
    Swap two rows in the matrix.
    """
    if row1 != row2:
        matrix[row1], matrix[row2] = matrix[row2], matrix[row1]


def scale_row_to_pivot_one(matrix, row, pivot_col):
    """
    Scale the specified row so that the pivot element becomes 1.
    This is the normalization step in Gauss-Jordan elimination.
    """
    n_cols = len(matrix[row])
    pivot_val = matrix[row][pivot_col]

    if abs(pivot_val) < 1e-12:
        raise ValueError(f"Pivot element is too small (near zero) at row {row + 1}, column {pivot_col + 1}")

    # Scale entire row by 1/pivot_val
    for j in range(pivot_col, n_cols):
        matrix[row][j] /= pivot_val


def eliminate_column(matrix, pivot_row, pivot_col):
    """
    Eliminate all elements in the pivot column except the pivot element.
    This makes all other elements in the column equal to zero.
    """
    n = len(matrix)
    n_cols = len(matrix[0])

    for i in range(n):
        if i != pivot_row:  # Skip the pivot row itself
            factor = matrix[i][pivot_col]
            if abs(factor) > 1e-12:  # Only eliminate if factor is significant
                # Subtract factor * pivot_row from current row
                for j in range(pivot_col, n_cols):
                    matrix[i][j] -= factor * matrix[pivot_row][j]


def gauss_jordan_elimination(matrix):
    """
    Perform Gauss-Jordan elimination with partial pivoting.
    Transforms the augmented matrix to reduced row echelon form (RREF).
    Returns the solution vector.
    """
    n = len(matrix)

    # Forward elimination with pivoting
    for i in range(n):
        # Find best pivot for current column
        pivot_row = find_pivot_row(matrix, i, i)

        # Swap rows if necessary
        swap_rows(matrix, i, pivot_row)

        # Check if pivot is acceptable
        if abs(matrix[i][i]) < 1e-12:
            raise ValueError(f"Matrix is singular or near-singular at step {i + 1}")

        # Scale pivot row to make diagonal element = 1
        scale_row_to_pivot_one(matrix, i, i)

        # Eliminate all other elements in current column
        eliminate_column(matrix, i, i)

    # Extract solution from the last column
    solution = []
    for i in range(n):
        solution.append(matrix[i][n])

    return solution


def gauss_jordan_solver(coefficients, constants):
    """
    Solve system of linear equations using Gauss-Jordan elimination.

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

    # Solve using Gauss-Jordan elimination
    solution = gauss_jordan_elimination(matrix)

    return solution


def print_matrix(matrix, title="Matrix", precision=4):
    """
    Utility function to print matrix in readable format.
    """
    print(f"\n{title}:")
    for i, row in enumerate(matrix):
        row_str = "  ".join(f"{val:8.{precision}f}" for val in row)
        print(f"Row {i + 1}: [{row_str}]")


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
    all_valid = True

    for i in range(n):
        # Calculate Ax for equation i
        calculated = 0.0
        for j in range(n):
            calculated += coefficients[i][j] * solution[j]

        error = abs(calculated - constants[i])
        max_error = max(max_error, error)

        status = "✓" if error < tolerance else "✗"
        if error >= tolerance:
            all_valid = False

        print(f"Equation {i + 1}: {calculated:.10f} = {constants[i]:.10f} {status} (error: {error:.2e})")

    print(f"Maximum error: {max_error:.2e}")
    return all_valid


def display_system(coefficients, constants):
    """
    Display the system of equations in readable mathematical format.
    """
    n = len(coefficients)
    print(f"\nInput System of Equations:")

    for i in range(n):
        terms = []
        for j in range(n):
            coeff = coefficients[i][j]
            if j == 0:
                terms.append(f"{coeff:+.3f}*x{j + 1}")
            else:
                terms.append(f"{coeff:+.3f}*x{j + 1}")

        equation = " ".join(terms).replace("+-", "- ")
        print(f"  {equation} = {constants[i]:.3f}")


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

        print(f"\nEnter the augmented matrix (coefficients + constant for each row):")
        print(f"Each row should have {n + 1} values separated by spaces")

        for i in range(n):
            while True:
                try:
                    row_input = input(f"Row {i + 1}: ").strip()
                    values = list(map(float, row_input.split()))

                    if len(values) != n + 1:
                        print(f"Please enter exactly {n + 1} values ({n} coefficients + 1 constant).")
                        continue

                    # Split into coefficients and constant
                    coefficients.append(values[:-1])
                    constants.append(values[-1])
                    break

                except ValueError:
                    print("Invalid input. Please enter numeric values separated by spaces.")

        return coefficients, constants

    except ValueError as e:
        raise ValueError(f"Input error: {e}")


def demonstrate_with_example(example_name, coefficients, constants):
    """
    Demonstrate the solver with a predefined example.
    """
    print(f"\n{'=' * 60}")
    print(f"Example: {example_name}")
    print(f"{'=' * 60}")

    try:
        # Display the system
        display_system(coefficients, constants)

        # Solve the system
        solution = gauss_jordan_solver(coefficients, constants)

        # Display results
        print_solution(solution)

        # Verify solution
        is_valid = validate_solution(coefficients, constants, solution)

        if is_valid:
            print("✓ Solution verified successfully!")
        else:
            print("✗ Solution verification failed!")

        return solution

    except Exception as e:
        print(f"Error solving system: {e}")
        return None


def main():
    """
    Main function demonstrating the Gauss-Jordan elimination solver.
    """
    print("Gauss-Jordan Elimination Solver with Partial Pivoting")
    print("=" * 60)

    choice = input("Choose mode: (1) Interactive input (2) Run examples [1/2]: ").strip()

    if choice == "2":
        # Run predefined examples
        run_examples()
    else:
        # Interactive mode
        try:
            # Read input
            coefficients, constants = read_system_from_input()

            # Solve and display
            demonstrate_with_example("User Input System", coefficients, constants)

        except ValueError as e:
            print(f"\nError: {e}")
        except Exception as e:
            print(f"\nUnexpected error: {e}")


def run_examples():
    """
    Run predefined test examples.
    """
    # Example 1: Simple 3x3 system
    coefficients1 = [
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ]
    constants1 = [8, -11, -3]
    demonstrate_with_example("Simple 3x3 System", coefficients1, constants1)

    # Example 2: 2x2 system requiring pivoting
    coefficients2 = [
        [0, 2],
        [1, 1]
    ]
    constants2 = [4, 3]
    demonstrate_with_example("2x2 System Requiring Pivoting", coefficients2, constants2)

    # Example 3: 4x4 identity-like system
    coefficients3 = [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    constants3 = [5, 3, 2, 4]
    demonstrate_with_example("4x4 Mixed System", coefficients3, constants3)


if __name__ == "__main__":
    main()