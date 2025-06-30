#!/usr/bin/env python3
import math


def validate_input_matrices(coefficients, constants):
    """
    Validate input coefficient matrix and constants vector.
    Ensures proper dimensions and non-zero diagonal elements.
    """
    n = len(coefficients)
    if n == 0:
        raise ValueError("Empty coefficient matrix")

    if len(constants) != n:
        raise ValueError("Number of constants must equal number of equations")

    for i, row in enumerate(coefficients):
        if len(row) != n:
            raise ValueError(f"Row {i + 1} has incorrect number of coefficients")

        # Check for zero diagonal elements
        if abs(coefficients[i][i]) < 1e-14:
            raise ValueError(f"Diagonal element A[{i + 1}][{i + 1}] is zero or too small. "
                             "Jacobi method requires non-zero diagonal elements.")

    return True


def check_diagonal_dominance(coefficients, strict=True):
    """
    Check if the coefficient matrix is diagonally dominant.
    This is a sufficient condition for Jacobi method convergence.

    Args:
        coefficients: 2D coefficient matrix
        strict: If True, checks for strict diagonal dominance

    Returns:
        (is_dominant, dominance_info): Boolean and diagnostic information
    """
    n = len(coefficients)
    dominance_info = []
    is_dominant = True

    for i in range(n):
        diagonal = abs(coefficients[i][i])
        off_diagonal_sum = sum(abs(coefficients[i][j]) for j in range(n) if j != i)

        if strict:
            row_dominant = diagonal > off_diagonal_sum
        else:
            row_dominant = diagonal >= off_diagonal_sum

        dominance_info.append({
            'row': i + 1,
            'diagonal': diagonal,
            'off_diagonal_sum': off_diagonal_sum,
            'ratio': diagonal / off_diagonal_sum if off_diagonal_sum > 0 else float('inf'),
            'dominant': row_dominant
        })

        if not row_dominant:
            is_dominant = False

    return is_dominant, dominance_info


def compute_jacobi_iteration(coefficients, constants, x_current):
    """
    Perform one iteration of the Jacobi method.
    This function computes x^(k+1) from x^(k).

    Args:
        coefficients: 2D coefficient matrix A
        constants: 1D constants vector b
        x_current: Current solution estimate

    Returns:
        x_new: New solution estimate
    """
    n = len(constants)
    x_new = [0.0] * n

    for i in range(n):
        # Compute sum of off-diagonal terms
        off_diagonal_sum = 0.0
        for j in range(n):
            if j != i:
                off_diagonal_sum += coefficients[i][j] * x_current[j]

        # Compute new x_i
        x_new[i] = (constants[i] - off_diagonal_sum) / coefficients[i][i]

    return x_new


def compute_solution_error(x_new, x_old, error_type='absolute'):
    """
    Compute the error between consecutive iterations.

    Args:
        x_new: New solution vector
        x_old: Previous solution vector
        error_type: 'absolute', 'relative', or 'max_norm'

    Returns:
        error_value: Computed error measure
    """
    n = len(x_new)

    if error_type == 'absolute':
        # L2 norm of difference
        error_sum = sum((x_new[i] - x_old[i]) ** 2 for i in range(n))
        return math.sqrt(error_sum)

    elif error_type == 'relative':
        # Relative error with L2 norm
        diff_norm = math.sqrt(sum((x_new[i] - x_old[i]) ** 2 for i in range(n)))
        x_norm = math.sqrt(sum(x_new[i] ** 2 for i in range(n)))
        return diff_norm / max(x_norm, 1e-14)  # Avoid division by zero

    elif error_type == 'max_norm':
        # Maximum absolute difference (infinity norm)
        return max(abs(x_new[i] - x_old[i]) for i in range(n))

    else:
        raise ValueError(f"Unknown error type: {error_type}")


def compute_residual(coefficients, constants, solution):
    """
    Compute the residual vector r = b - Ax for solution verification.

    Args:
        coefficients: 2D coefficient matrix A
        constants: 1D constants vector b
        solution: Current solution vector x

    Returns:
        (residual_vector, residual_norm): Residual and its L2 norm
    """
    n = len(constants)
    residual = [0.0] * n

    for i in range(n):
        # Compute (Ax)_i
        ax_i = sum(coefficients[i][j] * solution[j] for j in range(n))
        residual[i] = constants[i] - ax_i

    # Compute L2 norm of residual
    residual_norm = math.sqrt(sum(r ** 2 for r in residual))

    return residual, residual_norm


def jacobi_iterative_solver(coefficients, constants, initial_guess=None,
                            tolerance=1e-10, max_iterations=1000,
                            error_type='absolute', verbose=False):
    """
    Solve system of linear equations using Jacobi iterative method.

    Args:
        coefficients: 2D list representing coefficient matrix A
        constants: 1D list representing constants vector b
        initial_guess: Initial solution estimate (default: zero vector)
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        error_type: Type of error measure ('absolute', 'relative', 'max_norm')
        verbose: Print iteration details

    Returns:
        result: Dictionary containing solution and convergence information
    """
    # Validate inputs
    validate_input_matrices(coefficients, constants)

    n = len(constants)

    # Set initial guess
    if initial_guess is None:
        x_current = [0.0] * n
    else:
        if len(initial_guess) != n:
            raise ValueError(f"Initial guess must have {n} elements")
        x_current = initial_guess[:]  # Deep copy

    # Check diagonal dominance
    is_dominant, dominance_info = check_diagonal_dominance(coefficients)

    # Initialize iteration tracking
    iteration_history = []
    converged = False

    if verbose:
        print(f"Starting Jacobi iteration with {error_type} error measure")
        print(f"Tolerance: {tolerance}, Max iterations: {max_iterations}")
        if not is_dominant:
            print("Warning: Matrix is not diagonally dominant. Convergence not guaranteed.")

    # Main iteration loop
    for iteration in range(max_iterations):
        # Perform one Jacobi iteration
        x_new = compute_jacobi_iteration(coefficients, constants, x_current)

        # Compute error
        error = compute_solution_error(x_new, x_current, error_type)

        # Compute residual for this iteration
        residual_vector, residual_norm = compute_residual(coefficients, constants, x_new)

        # Store iteration information
        iteration_info = {
            'iteration': iteration + 1,
            'solution': x_new[:],  # Copy for history
            'error': error,
            'residual_norm': residual_norm
        }
        iteration_history.append(iteration_info)

        if verbose and (iteration < 5 or iteration % 10 == 0 or error < tolerance):
            print(f"Iteration {iteration + 1:3d}: Error = {error:.2e}, Residual = {residual_norm:.2e}")

        # Check convergence
        if error < tolerance:
            converged = True
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            break

        # Update for next iteration
        x_current = x_new

    # Prepare result
    result = {
        'solution': x_new if converged else None,
        'converged': converged,
        'iterations': iteration + 1,
        'final_error': error,
        'final_residual_norm': residual_norm,
        'diagonal_dominant': is_dominant,
        'dominance_info': dominance_info,
        'iteration_history': iteration_history,
        'tolerance': tolerance,
        'error_type': error_type
    }

    if not converged:
        result['message'] = f"Did not converge within {max_iterations} iterations (final error: {error:.2e})"

    return result


def print_dominance_analysis(dominance_info, is_dominant):
    """
    Print diagonal dominance analysis in readable format.
    """
    print(f"\nDiagonal Dominance Analysis:")
    print(f"Matrix is {'DIAGONALLY DOMINANT' if is_dominant else 'NOT diagonally dominant'}")
    print("Row | Diagonal | Off-diag Sum | Ratio    | Status")
    print("-" * 50)

    for info in dominance_info:
        ratio_str = f"{info['ratio']:.3f}" if info['ratio'] != float('inf') else "∞"
        status = "✓" if info['dominant'] else "✗"
        print(f"{info['row']:3d} | {info['diagonal']:8.3f} | {info['off_diagonal_sum']:11.3f} | "
              f"{ratio_str:8s} | {status}")


def print_solution_summary(result):
    """
    Print comprehensive solution summary.
    """
    print(f"\n{'=' * 60}")
    print("JACOBI METHOD SOLUTION SUMMARY")
    print(f"{'=' * 60}")

    if result['converged']:
        print("✓ CONVERGED")
        print(f"Solution found after {result['iterations']} iterations")
        print(f"Final error: {result['final_error']:.2e} (tolerance: {result['tolerance']:.2e})")
        print(f"Final residual norm: {result['final_residual_norm']:.2e}")

        print(f"\nSolution vector:")
        for i, val in enumerate(result['solution']):
            print(f"x{i + 1} = {val:12.8f}")
    else:
        print("✗ DID NOT CONVERGE")
        print(result['message'])

    dominance_status = "Yes" if result['diagonal_dominant'] else "No"
    print(f"Diagonally dominant: {dominance_status}")


def validate_solution(coefficients, constants, solution, tolerance=1e-10):
    """
    Validate the solution by computing residual and checking equations.
    """
    n = len(solution)
    print(f"\nSolution Validation:")
    print("Equation | Computed | Expected | Error    | Status")
    print("-" * 50)

    max_error = 0.0
    all_valid = True

    for i in range(n):
        # Compute left-hand side of equation i
        lhs = sum(coefficients[i][j] * solution[j] for j in range(n))
        rhs = constants[i]
        error = abs(lhs - rhs)
        max_error = max(max_error, error)

        status = "✓" if error < tolerance else "✗"
        if error >= tolerance:
            all_valid = False

        print(f"{i + 1:8d} | {lhs:8.4f} | {rhs:8.4f} | {error:.2e} | {status}")

    print(f"Maximum equation error: {max_error:.2e}")
    return all_valid


def read_system_from_input():
    """
    Read system of equations from user input with enhanced validation.
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
                        print(f"Please enter exactly {n + 1} values.")
                        continue

                    coefficients.append(values[:-1])
                    constants.append(values[-1])
                    break

                except ValueError:
                    print("Invalid input. Please enter numeric values.")

        # Get iteration parameters
        print(f"\nIteration Parameters:")

        # Initial guess
        initial_input = input(f"Initial guess ({n} values, or press Enter for zeros): ").strip()
        if initial_input:
            initial_guess = list(map(float, initial_input.split()))
            if len(initial_guess) != n:
                raise ValueError(f"Initial guess must have {n} values")
        else:
            initial_guess = None

        # Tolerance
        tol_input = input("Tolerance (default 1e-10): ").strip()
        tolerance = float(tol_input) if tol_input else 1e-10

        # Max iterations
        iter_input = input("Max iterations (default 1000): ").strip()
        max_iter = int(iter_input) if iter_input else 1000

        # Error type
        print("Error types: (1) absolute (2) relative (3) max_norm")
        error_choice = input("Choose error type (default 1): ").strip()
        error_types = {'1': 'absolute', '2': 'relative', '3': 'max_norm'}
        error_type = error_types.get(error_choice, 'absolute')

        return coefficients, constants, initial_guess, tolerance, max_iter, error_type

    except ValueError as e:
        raise ValueError(f"Input error: {e}")


def demonstrate_example(name, coefficients, constants, initial_guess=None,
                        tolerance=1e-10, max_iter=1000, error_type='absolute'):
    """
    Demonstrate the solver with a given example.
    """
    print(f"\n{'=' * 70}")
    print(f"EXAMPLE: {name}")
    print(f"{'=' * 70}")

    # Display system
    n = len(coefficients)
    print("System of equations:")
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

    try:
        # Solve system
        result = jacobi_iterative_solver(
            coefficients, constants, initial_guess,
            tolerance, max_iter, error_type, verbose=True
        )

        # Print results
        print_dominance_analysis(result['dominance_info'], result['diagonal_dominant'])
        print_solution_summary(result)

        # Validate if converged
        if result['converged']:
            validate_solution(coefficients, constants, result['solution'])

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """
    Main function for Jacobi iterative method demonstration.
    """
    print("Jacobi Iterative Method Solver")
    print("=" * 60)

    choice = input("Choose mode: (1) Interactive input (2) Run examples [1/2]: ").strip()

    if choice == "2":
        run_examples()
    else:
        try:
            coefficients, constants, initial_guess, tolerance, max_iter, error_type = read_system_from_input()

            result = demonstrate_example(
                "User Input System", coefficients, constants,
                initial_guess, tolerance, max_iter, error_type
            )

        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


def run_examples():
    """
    Run predefined test examples covering different scenarios.
    """
    # Example 1: Diagonally dominant system (guaranteed convergence)
    coefficients1 = [
        [10, -1, 2],
        [-1, 11, -1],
        [2, -1, 10]
    ]
    constants1 = [6, 25, -11]
    demonstrate_example("Diagonally Dominant 3x3", coefficients1, constants1)

    # Example 2: System requiring many iterations
    coefficients2 = [
        [4, -1],
        [-1, 4]
    ]
    constants2 = [3, 7]
    demonstrate_example("Slower Convergence 2x2", coefficients2, constants2,
                        tolerance=1e-8, max_iter=100)

    # Example 3: Larger system with specific initial guess
    coefficients3 = [
        [5, 1, 1, 1],
        [1, 5, 1, 1],
        [1, 1, 5, 1],
        [1, 1, 1, 5]
    ]
    constants3 = [8, 8, 8, 8]
    initial_guess3 = [1.0, 1.0, 1.0, 1.0]
    demonstrate_example("4x4 Symmetric System", coefficients3, constants3,
                        initial_guess3, tolerance=1e-6)


if __name__ == "__main__":
    main()