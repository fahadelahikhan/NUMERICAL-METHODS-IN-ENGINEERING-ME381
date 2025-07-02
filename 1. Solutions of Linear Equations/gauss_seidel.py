#!/usr/bin/env python3

def check_diagonal_dominance(A):
    """
    Check if the matrix is diagonally dominant (sufficient condition for convergence).
    Returns True if diagonally dominant, False otherwise.
    """
    n = len(A)
    for i in range(n):
        diagonal_element = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diagonal_element <= row_sum:
            return False
    return True


def gauss_seidel_method(A, b, x0, tol=1e-10, max_iter=100):
    """
    Solve linear system Ax = b using Gauss-Seidel iterative method.

    Parameters:
    A: coefficient matrix (list of lists)
    b: constant vector (list)
    x0: initial guess (list)
    tol: tolerance for convergence
    max_iter: maximum number of iterations

    Returns:
    solution vector (list), number of iterations used
    """
    n = len(b)

    # Validate inputs
    if len(A) != n or any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square and match the size of vector b")
    if len(x0) != n:
        raise ValueError("Initial guess x0 must have the same size as vector b")

    # Check for zero diagonal elements
    for i in range(n):
        if abs(A[i][i]) < 1e-15:  # Using small epsilon instead of exact zero
            raise ValueError(f"Diagonal element A[{i}][{i}] is effectively zero. "
                             "Gauss-Seidel method requires non-zero diagonal elements.")

    # Check diagonal dominance (optional warning)
    if not check_diagonal_dominance(A):
        print("Warning: Matrix is not diagonally dominant. Convergence is not guaranteed.")

    # Initialize solution vector
    x = [0.0] * n
    for i in range(n):
        x[i] = x0[i]

    # Gauss-Seidel iterations
    for iteration in range(max_iter):
        x_prev = [0.0] * n
        for i in range(n):
            x_prev[i] = x[i]

        # Update each variable
        for i in range(n):
            # Sum of a_ij * x_j for j < i (already updated values)
            sum1 = 0.0
            for j in range(i):
                sum1 += A[i][j] * x[j]

            # Sum of a_ij * x_j for j > i (previous iteration values)
            sum2 = 0.0
            for j in range(i + 1, n):
                sum2 += A[i][j] * x_prev[j]

            # Update x[i]
            x[i] = (b[i] - sum1 - sum2) / A[i][i]

        # Check convergence using relative error
        max_error = 0.0
        for i in range(n):
            if abs(x[i]) > 1e-15:  # Avoid division by zero
                error = abs((x[i] - x_prev[i]) / x[i])
            else:
                error = abs(x[i] - x_prev[i])
            if error > max_error:
                max_error = error

        if max_error < tol:
            return x, iteration + 1

    raise ValueError(f"Gauss-Seidel method did not converge within {max_iter} iterations. "
                     f"Final error: {max_error:.2e}")


def print_matrix(A, b):
    """Print the augmented matrix in a readable format."""
    n = len(A)
    print("\nAugmented Matrix [A|b]:")
    for i in range(n):
        row_str = "["
        for j in range(n):
            row_str += f"{A[i][j]:8.3f} "
        row_str += f"| {b[i]:8.3f}]"
        print(row_str)


def print_solution(x, iterations):
    """Print the solution vector and iteration count."""
    print(f"\nSolution converged in {iterations} iterations:")
    for i, val in enumerate(x):
        print(f"x[{i}] = {val:.8f}")


def verify_solution(A, b, x):
    """Verify the solution by computing residual."""
    n = len(A)
    residual = [0.0] * n
    max_residual = 0.0

    for i in range(n):
        sum_ax = 0.0
        for j in range(n):
            sum_ax += A[i][j] * x[j]
        residual[i] = abs(b[i] - sum_ax)
        if residual[i] > max_residual:
            max_residual = residual[i]

    print(f"\nVerification - Maximum residual: {max_residual:.2e}")
    if max_residual < 1e-6:
        print("✓ Solution verification passed")
    else:
        print("⚠ Solution verification failed - residual too large")


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


def get_parameters():
    """Get iteration parameters from user."""
    try:
        print(f"\nEnter initial guess (press Enter for all zeros):")
        initial_guess_input = input().strip()
        if initial_guess_input:
            x0 = list(map(float, initial_guess_input.split()))
        else:
            x0 = None

        tol_input = input("Enter tolerance (press Enter for 1e-10): ").strip()
        tol = float(tol_input) if tol_input else 1e-10
        if tol <= 0:
            print("Tolerance must be positive. Using default 1e-10.")
            tol = 1e-10

        max_iter_input = input("Enter maximum iterations (press Enter for 100): ").strip()
        max_iter = int(max_iter_input) if max_iter_input else 100
        if max_iter <= 0:
            print("Maximum iterations must be positive. Using default 100.")
            max_iter = 100

        return x0, tol, max_iter

    except ValueError:
        print("Invalid input for parameters. Using defaults.")
        return None, 1e-10, 100


def main():
    """Main function to run the Gauss-Seidel solver."""
    print("=== Gauss-Seidel Method for Linear Systems ===")

    # Get matrix input
    A, b = get_matrix_input()
    if A is None:
        return

    n = len(b)

    # Get parameters
    x0, tol, max_iter = get_parameters()
    if x0 is None:
        x0 = [0.0] * n
    elif len(x0) != n:
        print(f"Initial guess must have {n} elements. Using zeros.")
        x0 = [0.0] * n

    # Display the system
    print_matrix(A, b)

    # Solve the system
    try:
        solution, iterations = gauss_seidel_method(A, b, x0, tol, max_iter)
        print_solution(solution, iterations)
        verify_solution(A, b, solution)

    except ValueError as e:
        print(f"\nError: {e}")


# Test function for automated testing
def test_gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=100, test_name=""):
    """Test function for automated testing."""
    print(f"\n=== {test_name} ===")
    n = len(b)
    if x0 is None:
        x0 = [0.0] * n

    print_matrix(A, b)

    try:
        solution, iterations = gauss_seidel_method(A, b, x0, tol, max_iter)
        print_solution(solution, iterations)
        verify_solution(A, b, solution)
        return True
    except ValueError as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    main()