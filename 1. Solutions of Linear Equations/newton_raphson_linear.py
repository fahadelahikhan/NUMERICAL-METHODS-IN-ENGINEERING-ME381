#!/usr/bin/env python3

import math


def create_zero_matrix(n, m):
    """Create an n×m matrix filled with zeros."""
    matrix = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(0.0)
        matrix.append(row)
    return matrix


def create_zero_vector(n):
    """Create a vector of size n filled with zeros."""
    return [0.0] * n


def copy_vector(v):
    """Create a copy of vector v."""
    return [x for x in v]


def vector_norm(v):
    """Compute the Euclidean norm of vector v."""
    sum_squares = 0.0
    for x in v:
        sum_squares += x * x
    return math.sqrt(sum_squares)


def matrix_vector_multiply(A, b):
    """Multiply matrix A by vector b."""
    n = len(A)
    result = create_zero_vector(n)
    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * b[j]
    return result


def gauss_elimination_solve(A, b):
    """
    Solve linear system Ax = b using Gauss elimination with partial pivoting.
    Returns the solution vector x.
    """
    n = len(b)

    # Create augmented matrix
    augmented = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(A[i][j])
        row.append(b[i])
        augmented.append(row)

    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k

        # Swap rows
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

        # Check for singular matrix
        if abs(augmented[i][i]) < 1e-12:
            raise ValueError("Matrix is singular - no unique solution exists")

        # Eliminate column
        for k in range(i + 1, n):
            factor = augmented[k][i] / augmented[i][i]
            for j in range(i, n + 1):
                augmented[k][j] -= factor * augmented[i][j]

    # Back substitution
    x = create_zero_vector(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]
        x[i] /= augmented[i][i]

    return x


def compute_numerical_jacobian(F_func, x, h=1e-7):
    """
    Compute Jacobian matrix using finite differences.
    F_func: function that takes x and returns F(x)
    x: current point (list)
    h: step size for finite differences
    """
    n = len(x)
    F_x = F_func(x)
    J = create_zero_matrix(n, n)

    for j in range(n):
        x_plus_h = copy_vector(x)
        x_plus_h[j] += h
        F_x_plus_h = F_func(x_plus_h)

        for i in range(n):
            J[i][j] = (F_x_plus_h[i] - F_x[i]) / h

    return J


def newton_raphson_nonlinear(F_func, x0, tol=1e-10, max_iter=100):
    """
    Solve nonlinear system F(x) = 0 using Newton-Raphson method.

    Parameters:
    F_func: function that takes x (list) and returns F(x) (list)
    x0: initial guess (list)
    tol: tolerance for convergence
    max_iter: maximum number of iterations

    Returns:
    solution vector (list), number of iterations used
    """
    n = len(x0)
    x = copy_vector(x0)

    print(f"Starting Newton-Raphson iteration with initial guess: {x}")

    for iteration in range(max_iter):
        # Evaluate function at current point
        F_x = F_func(x)

        # Check if we're already at the solution
        F_norm = vector_norm(F_x)
        print(f"Iteration {iteration + 1}: ||F(x)|| = {F_norm:.2e}")

        if F_norm < tol:
            print(f"Converged: ||F(x)|| < {tol}")
            return x, iteration + 1

        # Compute Jacobian matrix
        J = compute_numerical_jacobian(F_func, x)

        # Solve J * delta_x = -F(x)
        try:
            delta_x = gauss_elimination_solve(J, [-f for f in F_x])
        except ValueError as e:
            raise ValueError(f"Newton-Raphson failed at iteration {iteration + 1}: {e}")

        # Update solution
        for i in range(n):
            x[i] += delta_x[i]

        # Check convergence based on step size
        delta_norm = vector_norm(delta_x)
        if delta_norm < tol:
            print(f"Converged: step size ||Δx|| = {delta_norm:.2e} < {tol}")
            return x, iteration + 1

        print(f"Step size: ||Δx|| = {delta_norm:.2e}")

    raise ValueError(f"Newton-Raphson method did not converge within {max_iter} iterations")


def linear_system_as_nonlinear(A, b):
    """
    Convert linear system Ax - b = 0 to nonlinear form F(x) = 0.
    This demonstrates how Newton-Raphson can solve linear systems.
    """

    def F_func(x):
        # F(x) = Ax - b
        result = matrix_vector_multiply(A, x)
        for i in range(len(result)):
            result[i] -= b[i]
        return result

    return F_func


# Predefined nonlinear system examples
def example_system_1(x):
    """
    Example 1: Simple 2x2 nonlinear system
    F1(x,y) = x^2 + y^2 - 4 = 0
    F2(x,y) = x*y - 1 = 0
    """
    if len(x) != 2:
        raise ValueError("This system requires exactly 2 variables")

    x_val, y_val = x[0], x[1]
    f1 = x_val * x_val + y_val * y_val - 4.0
    f2 = x_val * y_val - 1.0
    return [f1, f2]


def example_system_2(x):
    """
    Example 2: 2x2 system with exponential
    F1(x,y) = x + y - 3 = 0
    F2(x,y) = x^2 + y^2 - 5 = 0
    """
    if len(x) != 2:
        raise ValueError("This system requires exactly 2 variables")

    x_val, y_val = x[0], x[1]
    f1 = x_val + y_val - 3.0
    f2 = x_val * x_val + y_val * y_val - 5.0
    return [f1, f2]


def example_system_3(x):
    """
    Example 3: 3x3 nonlinear system
    F1(x,y,z) = x^2 + y + z - 6 = 0
    F2(x,y,z) = x + y^2 + z - 6 = 0
    F3(x,y,z) = x + y + z^2 - 6 = 0
    """
    if len(x) != 3:
        raise ValueError("This system requires exactly 3 variables")

    x_val, y_val, z_val = x[0], x[1], x[2]
    f1 = x_val * x_val + y_val + z_val - 6.0
    f2 = x_val + y_val * y_val + z_val - 6.0
    f3 = x_val + y_val + z_val * z_val - 6.0
    return [f1, f2, f3]


def print_vector(v, label="Vector"):
    """Print vector in a readable format."""
    print(f"\n{label}:")
    for i, val in enumerate(v):
        print(f"  x[{i}] = {val:.8f}")


def print_matrix(A, label="Matrix"):
    """Print matrix in a readable format."""
    print(f"\n{label}:")
    for i, row in enumerate(A):
        row_str = "  ["
        for val in row:
            row_str += f"{val:8.4f} "
        row_str += "]"
        print(row_str)


def verify_solution(F_func, x, tol=1e-6):
    """Verify the solution by evaluating F(x)."""
    F_x = F_func(x)
    residual_norm = vector_norm(F_x)
    print(f"\nSolution Verification:")
    print(f"||F(x)|| = {residual_norm:.2e}")

    if residual_norm < tol:
        print("✓ Solution verification passed")
        return True
    else:
        print("⚠ Solution verification failed - residual too large")
        return False


def get_user_input():
    """Get user input for the problem type."""
    print("=== Newton-Raphson Method for Nonlinear Systems ===")
    print("\nChoose problem type:")
    print("1. Predefined Example 1 (2x2 system: x² + y² = 4, xy = 1)")
    print("2. Predefined Example 2 (2x2 system: x + y = 3, x² + y² = 5)")
    print("3. Predefined Example 3 (3x3 nonlinear system)")
    print("4. Linear system (converted to nonlinear form)")

    try:
        choice = int(input("\nEnter your choice (1-4): "))
        return choice
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None


def handle_predefined_example(example_num):
    """Handle predefined examples."""
    if example_num == 1:
        print("\nSolving: x² + y² = 4, xy = 1")
        print("Expected solutions: approximately (±1.932, ±0.518)")
        F_func = example_system_1
        n = 2
    elif example_num == 2:
        print("\nSolving: x + y = 3, x² + y² = 5")
        print("Expected solutions: (1, 2) or (2, 1)")
        F_func = example_system_2
        n = 2
    elif example_num == 3:
        print("\nSolving 3x3 system: x² + y + z = 6, x + y² + z = 6, x + y + z² = 6")
        print("Expected solution: approximately (2, 2, 2)")
        F_func = example_system_3
        n = 3
    else:
        return None, None

    # Get initial guess
    print(f"\nEnter initial guess ({n} values separated by spaces):")
    try:
        x0 = list(map(float, input().strip().split()))
        if len(x0) != n:
            print(f"Initial guess must have exactly {n} elements.")
            return None, None
        return F_func, x0
    except ValueError:
        print("Invalid input for initial guess.")
        return None, None


def handle_linear_system():
    """Handle linear system input."""
    print("\nLinear System Solver (using Newton-Raphson principles)")
    try:
        n = int(input("Enter the number of equations: "))
        if n <= 0:
            print("Number of equations must be positive.")
            return None, None

        print(f"Enter coefficient matrix A and vector b:")
        A = []
        b = []

        for i in range(n):
            row_input = input(f"Row {i + 1} (coefficients + constant): ").strip()
            row = list(map(float, row_input.split()))
            if len(row) != n + 1:
                print(f"Each row must have {n + 1} elements.")
                return None, None
            A.append(row[:-1])
            b.append(row[-1])

        F_func = linear_system_as_nonlinear(A, b)

        print(f"Enter initial guess ({n} values):")
        x0 = list(map(float, input().strip().split()))
        if len(x0) != n:
            print(f"Initial guess must have {n} elements.")
            return None, None

        return F_func, x0

    except ValueError:
        print("Invalid input.")
        return None, None


def main():
    """Main function."""
    choice = get_user_input()
    if choice is None:
        return

    if choice in [1, 2, 3]:
        F_func, x0 = handle_predefined_example(choice)
    elif choice == 4:
        F_func, x0 = handle_linear_system()
    else:
        print("Invalid choice.")
        return

    if F_func is None:
        return

    # Get solver parameters
    try:
        tol_input = input("Enter tolerance (press Enter for 1e-10): ").strip()
        tol = float(tol_input) if tol_input else 1e-10

        max_iter_input = input("Enter max iterations (press Enter for 50): ").strip()
        max_iter = int(max_iter_input) if max_iter_input else 50

    except ValueError:
        print("Using default parameters: tol=1e-10, max_iter=50")
        tol = 1e-10
        max_iter = 50

    # Solve the system
    try:
        print(f"\n{'=' * 60}")
        solution, iterations = newton_raphson_nonlinear(F_func, x0, tol, max_iter)
        print(f"{'=' * 60}")

        print_vector(solution, "Final Solution")
        print(f"\nConverged in {iterations} iterations")
        verify_solution(F_func, solution)

    except ValueError as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()