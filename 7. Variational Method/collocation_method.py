#!/usr/bin/env python3

import numpy as np
from sympy import symbols, diff, lambdify, solve, Eq


def create_basis_functions(basis_type='polynomial', order=2):
    """
    Create basis functions based on specified type and order.

    Args:
        basis_type: Type of basis ('polynomial', 'legendre', 'chebyshev')
        order: Order of basis functions

    Returns:
        List of symbolic basis functions
    """
    x = symbols('x')

    if basis_type == 'polynomial':
        return [x ** i for i in range(order + 1)]
    elif basis_type == 'legendre':
        # First few Legendre polynomials
        legendre_polys = [1, x, (3 * x ** 2 - 1) / 2, (5 * x ** 3 - 3 * x) / 2, (35 * x ** 4 - 30 * x ** 2 + 3) / 8]
        return legendre_polys[:order + 1]
    elif basis_type == 'chebyshev':
        # First few Chebyshev polynomials
        from sympy import cos, acos
        chebyshev_polys = [1, x, 2 * x ** 2 - 1, 4 * x ** 3 - 3 * x, 8 * x ** 4 - 8 * x ** 2 + 1]
        return chebyshev_polys[:order + 1]
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


def setup_collocation_points(point_type='uniform', n_points=3, domain=(0, 1)):
    """
    Generate collocation points based on specified type.

    Args:
        point_type: Type of points ('uniform', 'chebyshev', 'custom')
        n_points: Number of collocation points
        domain: Domain bounds (start, end)

    Returns:
        List of collocation points
    """
    a, b = domain

    if point_type == 'uniform':
        return [a + (b - a) * (i + 1) / (n_points + 1) for i in range(n_points)]
    elif point_type == 'chebyshev':
        # Chebyshev points mapped to [a, b]
        import math
        points = []
        for i in range(n_points):
            xi = math.cos((2 * i + 1) * math.pi / (2 * n_points))
            # Map from [-1, 1] to [a, b]
            xi_mapped = (b - a) * (xi + 1) / 2 + a
            points.append(xi_mapped)
        return sorted(points)
    elif point_type == 'gauss':
        # Gauss-Legendre points (simplified for common cases)
        if n_points == 1:
            return [(a + b) / 2]
        elif n_points == 2:
            h = (b - a) / 2
            c = (a + b) / 2
            return [c - h / np.sqrt(3), c + h / np.sqrt(3)]
        elif n_points == 3:
            h = (b - a) / 2
            c = (a + b) / 2
            return [c - h * np.sqrt(3 / 5), c, c + h * np.sqrt(3 / 5)]
        else:
            # Fall back to uniform for higher orders
            return [a + (b - a) * (i + 1) / (n_points + 1) for i in range(n_points)]
    else:
        raise ValueError(f"Unknown point type: {point_type}")


def assemble_collocation_system(basis_funcs, differential_operator, collocation_points, boundary_conditions,
                                domain=(0, 1)):
    """
    Assemble the collocation system matrix and vector.

    Args:
        basis_funcs: List of basis functions
        differential_operator: Function that applies differential operator
        collocation_points: List of collocation points
        boundary_conditions: Dictionary of boundary conditions
        domain: Solution domain

    Returns:
        System matrix and right-hand side vector
    """
    x = symbols('x')
    n = len(basis_funcs)
    n_eq = len(collocation_points) + len(boundary_conditions)

    # Initialize system matrix and RHS vector
    system_matrix = np.zeros((n_eq, n))
    rhs_vector = np.zeros(n_eq)

    equation_idx = 0

    # Add boundary condition equations
    for bc_point, bc_value in boundary_conditions.items():
        for j, basis_func in enumerate(basis_funcs):
            # Evaluate basis function at boundary point
            basis_at_point = float(basis_func.subs(x, bc_point))
            system_matrix[equation_idx, j] = basis_at_point
        rhs_vector[equation_idx] = bc_value
        equation_idx += 1

    # Add collocation equations
    for point in collocation_points:
        for j, basis_func in enumerate(basis_funcs):
            # Apply differential operator to basis function
            operator_result = differential_operator(basis_func)
            # Evaluate at collocation point
            operator_at_point = float(operator_result.subs(x, point))
            system_matrix[equation_idx, j] = operator_at_point
        rhs_vector[equation_idx] = 0.0  # Residual should be zero at collocation points
        equation_idx += 1

    return system_matrix, rhs_vector


def solve_collocation_system(system_matrix, rhs_vector):
    """
    Solve the collocation system.

    Args:
        system_matrix: System matrix
        rhs_vector: Right-hand side vector

    Returns:
        Solution coefficients
    """
    try:
        # Try direct solution
        solution = np.linalg.solve(system_matrix, rhs_vector)
        return solution
    except np.linalg.LinAlgError:
        # Try least squares solution for over/under-determined systems
        solution = np.linalg.lstsq(system_matrix, rhs_vector, rcond=None)[0]
        return solution


def collocation_method(basis_funcs, differential_operator, forcing_function,
                       collocation_points, boundary_conditions, domain=(0, 1)):
    """
    Solve BVP using the collocation method.

    Args:
        basis_funcs: List of basis functions
        differential_operator: Function that applies differential operator
        forcing_function: Forcing function (RHS of differential equation)
        collocation_points: List of collocation points
        boundary_conditions: Dictionary of boundary conditions
        domain: Solution domain

    Returns:
        Solution coefficients and approximate solution function
    """
    x = symbols('x')

    # Create modified differential operator that includes forcing function
    def modified_operator(u):
        return differential_operator(u) - forcing_function

    # Assemble system
    system_matrix, rhs_vector = assemble_collocation_system(
        basis_funcs, modified_operator, collocation_points, boundary_conditions, domain
    )

    # Solve system
    coefficients = solve_collocation_system(system_matrix, rhs_vector)

    # Create approximate solution function
    def approximate_solution(x_val):
        result = 0
        for i, (coeff, basis_func) in enumerate(zip(coefficients, basis_funcs)):
            result += coeff * float(basis_func.subs(x, x_val))
        return result

    # Create symbolic approximate solution
    symbolic_solution = sum(coeff * basis_func for coeff, basis_func in zip(coefficients, basis_funcs))

    return coefficients, approximate_solution, symbolic_solution


def validate_solution(coefficients, basis_funcs, differential_operator, forcing_function,
                      collocation_points, boundary_conditions):
    """
    Validate the solution by checking residuals.

    Args:
        coefficients: Solution coefficients
        basis_funcs: List of basis functions
        differential_operator: Differential operator
        forcing_function: Forcing function
        collocation_points: Collocation points
        boundary_conditions: Boundary conditions

    Returns:
        Dictionary with validation results
    """
    x = symbols('x')

    # Construct approximate solution
    approximate_solution = sum(coeff * basis_func for coeff, basis_func in zip(coefficients, basis_funcs))

    # Check boundary conditions
    bc_errors = {}
    for point, value in boundary_conditions.items():
        computed_value = float(approximate_solution.subs(x, point))
        bc_errors[point] = abs(computed_value - value)

    # Check residuals at collocation points
    residuals = []
    for point in collocation_points:
        operator_result = differential_operator(approximate_solution)
        residual = float((operator_result - forcing_function).subs(x, point))
        residuals.append(residual)

    return {
        'boundary_errors': bc_errors,
        'collocation_residuals': residuals,
        'max_boundary_error': max(bc_errors.values()) if bc_errors else 0.0,
        'max_residual': max(abs(r) for r in residuals) if residuals else 0.0
    }


def solve_ode_collocation(differential_eq, forcing_func, domain, boundary_conditions,
                          basis_type='polynomial', order=3, point_type='uniform', n_points=None):
    """
    High-level function to solve ODE using collocation method.

    Args:
        differential_eq: Function that applies differential operator
        forcing_func: Forcing function
        domain: Solution domain
        boundary_conditions: Boundary conditions
        basis_type: Type of basis functions
        order: Order of approximation
        point_type: Type of collocation points
        n_points: Number of collocation points (default: order - len(boundary_conditions))

    Returns:
        Solution coefficients, approximate solution function, and validation results
    """
    # Create basis functions
    basis_funcs = create_basis_functions(basis_type, order)

    # Determine number of collocation points
    if n_points is None:
        n_points = max(1, order + 1 - len(boundary_conditions))

    # Create collocation points
    collocation_points = setup_collocation_points(point_type, n_points, domain)

    # Solve using collocation method
    coefficients, approx_sol_func, symbolic_sol = collocation_method(
        basis_funcs, differential_eq, forcing_func, collocation_points, boundary_conditions, domain
    )

    # Validate solution
    validation = validate_solution(
        coefficients, basis_funcs, differential_eq, forcing_func,
        collocation_points, boundary_conditions
    )

    return coefficients, approx_sol_func, symbolic_sol, validation


def main():
    """Main function with example usage."""
    try:
        x = symbols('x')

        print("=== Collocation Method Examples ===\n")

        # Example 1: Second-order ODE: u'' + u = 1, u(0) = 0, u(1) = 0
        print("Example 1: u'' + u = 1 with u(0) = 0, u(1) = 0")
        print("-" * 50)

        def diff_operator_ex1(u):
            return diff(u, x, 2) + u

        forcing_func_ex1 = 1
        domain_ex1 = (0, 1)
        bc_ex1 = {0: 0, 1: 0}

        coeffs1, approx_sol1, symbolic_sol1, validation1 = solve_ode_collocation(
            diff_operator_ex1, forcing_func_ex1, domain_ex1, bc_ex1,
            basis_type='polynomial', order=4, point_type='uniform', n_points=3
        )

        print("Coefficients:")
        for i, coeff in enumerate(coeffs1):
            print(f"c{i} = {coeff:.6f}")

        print(f"\nSymbolic solution: {symbolic_sol1}")
        print(f"Max boundary error: {validation1['max_boundary_error']:.2e}")
        print(f"Max residual: {validation1['max_residual']:.2e}")

        # Test at some points
        test_points = [0.2, 0.5, 0.8]
        print("\nSolution at test points:")
        for pt in test_points:
            print(f"u({pt}) = {approx_sol1(pt):.6f}")

        print("\n" + "=" * 70 + "\n")

        # Example 2: First-order ODE: u' - 2u = x, u(0) = 1
        print("Example 2: u' - 2u = x with u(0) = 1")
        print("-" * 50)

        def diff_operator_ex2(u):
            return diff(u, x) - 2 * u

        forcing_func_ex2 = x
        domain_ex2 = (0, 1)
        bc_ex2 = {0: 1}

        coeffs2, approx_sol2, symbolic_sol2, validation2 = solve_ode_collocation(
            diff_operator_ex2, forcing_func_ex2, domain_ex2, bc_ex2,
            basis_type='polynomial', order=3, point_type='chebyshev', n_points=3
        )

        print("Coefficients:")
        for i, coeff in enumerate(coeffs2):
            print(f"c{i} = {coeff:.6f}")

        print(f"\nSymbolic solution: {symbolic_sol2}")
        print(f"Max boundary error: {validation2['max_boundary_error']:.2e}")
        print(f"Max residual: {validation2['max_residual']:.2e}")

        # Test at some points
        print("\nSolution at test points:")
        for pt in test_points:
            print(f"u({pt}) = {approx_sol2(pt):.6f}")

        print("\n" + "=" * 70 + "\n")

        # Example 3: Poisson equation: u'' = -sin(πx), u(0) = 0, u(1) = 0
        print("Example 3: u'' = -sin(πx) with u(0) = 0, u(1) = 0")
        print("-" * 50)

        from sympy import sin, pi

        def diff_operator_ex3(u):
            return diff(u, x, 2)

        forcing_func_ex3 = -sin(pi * x)
        domain_ex3 = (0, 1)
        bc_ex3 = {0: 0, 1: 0}

        coeffs3, approx_sol3, symbolic_sol3, validation3 = solve_ode_collocation(
            diff_operator_ex3, forcing_func_ex3, domain_ex3, bc_ex3,
            basis_type='polynomial', order=5, point_type='gauss', n_points=4
        )

        print("Coefficients:")
        for i, coeff in enumerate(coeffs3):
            print(f"c{i} = {coeff:.6f}")

        print(f"\nSymbolic solution: {symbolic_sol3}")
        print(f"Max boundary error: {validation3['max_boundary_error']:.2e}")
        print(f"Max residual: {validation3['max_residual']:.2e}")

        # Compare with analytical solution: u = sin(πx)/π²
        analytical_sol = sin(pi * x) / (pi ** 2)
        print(f"\nAnalytical solution: {analytical_sol}")

        print("\nComparison at test points:")
        for pt in test_points:
            numerical = approx_sol3(pt)
            analytical = float(analytical_sol.subs(x, pt))
            error = abs(numerical - analytical)
            print(f"u({pt}): Numerical = {numerical:.6f}, Analytical = {analytical:.6f}, Error = {error:.2e}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()