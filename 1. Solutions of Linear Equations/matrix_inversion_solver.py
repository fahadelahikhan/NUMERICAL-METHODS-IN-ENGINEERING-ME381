#!/usr/bin/env python3

def create_identity_matrix(n):
    """Create an n×n identity matrix."""
    identity = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)
        identity.append(row)
    return identity


def copy_matrix(matrix):
    """Create a deep copy of a matrix."""
    n = len(matrix)
    m = len(matrix[0])
    copy = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(matrix[i][j])
        copy.append(row)
    return copy


def matrix_determinant(A):
    """
    Calculate determinant using LU decomposition.
    Returns the determinant of matrix A.
    """
    n = len(A)
    # Create a copy to avoid modifying original matrix
    matrix = copy_matrix(A)

    det = 1.0

    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                max_row = k

        # Swap rows if needed
        if max_row != i:
            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
            det = -det  # Row swap changes sign of determinant

        # Check for singular matrix
        if abs(matrix[i][i]) < 1e-12:
            return 0.0

        det *= matrix[i][i]

        # Eliminate column
        for k in range(i + 1, n):
            factor = matrix[k][i] / matrix[i][i]
            for j in range(i, n):
                matrix[k][j] -= factor * matrix[i][j]

    return det


def gauss_jordan_inversion(A):
    """
    Compute matrix inverse using Gauss-Jordan elimination.
    Returns the inverse matrix or raises ValueError if singular.
    """
    n = len(A)

    # Check if matrix is square
    for row in A:
        if len(row) != n:
            raise ValueError("Matrix must be square for inversion")

    # Check determinant first (optional but good practice)
    det = matrix_determinant(A)
    if abs(det) < 1e-12:
        raise ValueError("Matrix is singular (determinant ≈ 0) and cannot be inverted")

    # Create augmented matrix [A|I]
    augmented = []
    identity = create_identity_matrix(n)

    for i in range(n):
        row = []
        # Add original matrix elements
        for j in range(n):
            row.append(float(A[i][j]))
        # Add identity matrix elements
        for j in range(n):
            row.append(identity[i][j])
        augmented.append(row)

    # Gauss-Jordan elimination
    for i in range(n):
        # Find pivot row
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k

        # Swap rows if needed
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

        # Check for zero pivot
        if abs(augmented[i][i]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted")

        # Scale pivot row
        pivot = augmented[i][i]
        for j in range(2 * n):
            augmented[i][j] /= pivot

        # Eliminate other rows
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]

    # Extract inverse matrix from right side of augmented matrix
    inverse = []
    for i in range(n):
        row = []
        for j in range(n, 2 * n):
            row.append(augmented[i][j])
        inverse.append(row)

    return inverse


def matrix_vector_multiply(A, b):
    """
    Multiply matrix A by vector b.
    Returns the result vector A * b.
    """
    n = len(A)
    result = []

    for i in range(n):
        sum_val = 0.0
        for j in range(n):
            sum_val += A[i][j] * b[j]
        result.append(sum_val)

    return result


def matrix_condition_number(A):
    """
    Estimate condition number using simple method.
    Higher condition numbers indicate ill-conditioned matrices.
    """
    try:
        A_inv = gauss_jordan_inversion(A)
        # Simple estimation: ||A|| * ||A^-1|| using max row sum norm
        norm_A = 0.0
        norm_A_inv = 0.0

        n = len(A)
        for i in range(n):
            row_sum_A = sum(abs(A[i][j]) for j in range(n))
            row_sum_A_inv = sum(abs(A_inv[i][j]) for j in range(n))
            norm_A = max(norm_A, row_sum_A)
            norm_A_inv = max(norm_A_inv, row_sum_A_inv)

        return norm_A * norm_A_inv
    except:
        return float('inf')


def matrix_inversion_method(A, b):
    """
    Solve linear system Ax = b using matrix inversion method.

    Parameters:
    A: coefficient matrix (list of lists)
    b: constant vector (list)

    Returns:
    solution vector (list)
    """
    n = len(b)

    # Validate inputs
    if len(A) != n:
        raise ValueError("Matrix A must be square and match the size of vector b")

    for i, row in enumerate(A):
        if len(row) != n:
            raise ValueError(f"Row {i} of matrix A has incorrect length")

    # Check condition number
    cond_num = matrix_condition_number(A)
    if cond_num > 1e12:
        print(f"Warning: Matrix is ill-conditioned (condition number ≈ {cond_num:.2e})")
        print("Results may be inaccurate. Consider using iterative methods instead.")

    try:
        # Compute matrix inverse
        A_inv = gauss_jordan_inversion(A)

        # Multiply A^(-1) * b to get solution
        x = matrix_vector_multiply(A_inv, b)

        return x

    except ValueError as e:
        raise ValueError(f"Matrix inversion failed: {e}")


def print_matrix(A, b=None):
    """Print matrix in a readable format."""
    n = len(A)
    if b is not None:
        print("\nAugmented Matrix [A|b]:")
        for i in range(n):
            row_str = "["
            for j in range(n):
                row_str += f"{A[i][j]:8.3f} "
            row_str += f"| {b[i]:8.3f}]"
            print(row_str)
    else:
        print("\nMatrix A:")
        for i in range(n):
            row_str = "["
            for j in range(n):
                row_str += f"{A[i][j]:8.3f} "
            row_str += "]"
            print(row_str)


def print_solution(x):
    """Print the solution vector."""
    print("\nSolution:")
    for i, val in enumerate(x):
        print(f"x[{i}] = {val:.8f}")


def verify_solution(A, b, x):
    """Verify the solution by computing residual."""
    n = len(A)
    residual = []
    max_residual = 0.0

    for i in range(n):
        sum_ax = 0.0
        for j in range(n):
            sum_ax += A[i][j] * x[j]
        res = abs(b[i] - sum_ax)
        residual.append(res)
        if res > max_residual:
            max_residual = res

    print(f"\nVerification - Maximum residual: {max_residual:.2e}")
    if max_residual < 1e-10:
        print("✓ Solution verification passed")
    else:
        print("⚠ Solution verification: residual is larger than expected")


def get_matrix_input():
    """Get matrix input from user."""
    try:
        n = int(input("Enter the number of equations: "))
        if n <= 0:
            print("Number of equations must be a positive integer.")
            return None, None

        A = []
        b = []
        print(f"Enter each equation's coefficients followed by the constant term.")
        print(f"Each line should have {n + 1} numbers separated by spaces:")

        for i in range(n):
            while True:
                try:
                    row_input = input(f"Equation {i + 1}: ").strip()
                    row = list(map(float, row_input.split()))
                    if len(row) != n + 1:
                        print(f"Please enter exactly {n + 1} numbers.")
                        continue
                    A.append(row[:-1])
                    b.append(row[-1])
                    break
                except ValueError:
                    print("Please enter only numeric values.")

        return A, b

    except ValueError:
        print("Invalid input for number of equations.")
        return None, None


def main():
    """Main function to run the Matrix Inversion solver."""
    print("=== Matrix Inversion Method for Linear Systems ===")
    print("Note: This method is suitable for small to medium-sized systems.")
    print("For large systems, consider iterative methods like Gauss-Seidel.\n")

    # Get matrix input
    A, b = get_matrix_input()
    if A is None:
        return

    # Display the system
    print_matrix(A, b)

    # Display matrix properties
    det = matrix_determinant(A)
    print(f"\nMatrix determinant: {det:.6e}")

    cond_num = matrix_condition_number(A)
    print(f"Matrix condition number: {cond_num:.2e}")

    if cond_num > 1e6:
        print("Warning: Matrix appears to be ill-conditioned!")

    # Solve the system
    try:
        solution = matrix_inversion_method(A, b)
        print_solution(solution)
        verify_solution(A, b, solution)

    except ValueError as e:
        print(f"\nError: {e}")


# Test function for automated testing
def test_matrix_inversion(A, b, test_name=""):
    """Test function for automated testing."""
    print(f"\n{'=' * 50}")
    print(f"{test_name}")
    print('=' * 50)

    print_matrix(A, b)

    det = matrix_determinant(A)
    print(f"\nMatrix determinant: {det:.6e}")

    cond_num = matrix_condition_number(A)
    print(f"Matrix condition number: {cond_num:.2e}")

    try:
        solution = matrix_inversion_method(A, b)
        print_solution(solution)
        verify_solution(A, b, solution)
        return True
    except ValueError as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    main()