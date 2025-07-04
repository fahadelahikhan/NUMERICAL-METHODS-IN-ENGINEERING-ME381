#!/usr/bin/env python3

import numpy as np
from sympy import symbols, integrate, diff, lambdify


def create_basis_functions(basis_type='polynomial', order=2):
    """
    Create basis functions for trial space.

    Args:
        basis_type: Type of basis ('polynomial', 'legendre', 'trigonometric')
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
    elif basis_type == 'trigonometric':
        from sympy import sin, cos, pi
        funcs = [1]  # Constant term
        for i in range(1, order + 1):
            funcs.extend([sin(i * pi * x), cos(i * pi * x)])
        return funcs[:order + 1]
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


def create_test_functions(test_type='polynomial', order=2, trial_funcs=None):
    """
    Create test functions for test space (can be different from trial space).

    Args:
        test_type: Type of test functions
        order: Order of test functions
        trial_funcs: Trial functions (for Galerkin method, test = trial)

    Returns:
        List of symbolic test functions
    """
    x = symbols('x')

    if test_type == 'galerkin' and trial_funcs is not None:
        # Standard Galerkin: test functions = trial functions
        return trial_funcs
    elif test_type == 'polynomial':
        return [x ** i for i in range(order + 1)]
    elif test_type == 'shifted_polynomial':
        # Shifted polynomials for better conditioning
        return [(x - 0.5) ** i for i in range(order + 1)]
    elif test_type == 'derivative':
        # Use derivatives of polynomials as test functions
        polys = [x ** i for i in range(1, order + 2)]
        return [diff(p, x) for p in polys]
    elif test_type == 'weighted':
        # Weighted polynomials
        weight = (1 - x ** 2) if order > 0 else 1
        return [weight * x ** i for i in range(order + 1)]
    else:
        raise ValueError(f"Unknown test function type: {test_type}")


def compute_bilinear_form(test_func, trial_func, differential_operator, domain):
    """
    Compute the bilinear form a(u,v) = ∫ v * L[u] dx where L is the differential operator.

    Args:
        test_func: Test function
        trial_func: Trial function
        differential_operator: Function that applies differential operator
        domain: Integration domain

    Returns:
        Value of the bilinear form
    """
    x = symbols('x')

    # Apply differential operator to trial function
    Lu = differential_operator(trial_func)

    # Compute the bilinear form: (test_func, L[trial_func])
    integrand = test_func * Lu

    try:
        result = integrate(integrand, (x, domain[0], domain[1]))
        return float(result)
    except:
        # Fallback to numerical integration
        integrand_func = lambdify(x, integrand, 'numpy')
        from scipy.integrate import quad
        result, _ = quad(integrand_func, domain[0], domain[1])
        return result


def compute_linear_form(test_func, forcing_func, domain):
    """
    Compute the linear form L(v) = ∫ v * f dx.

    Args:
        test_func: Test function
        forcing_func: Forcing function
        domain: Integration domain

    Returns:
        Value of the linear form
    """
    x = symbols('x')

    integrand = test_func * forcing_func

    try:
        result = integrate(integrand, (x, domain[0], domain[1]))
        return float(result)
    except:
        # Fallback to numerical integration
        integrand_func = lambdify(x, integrand, 'numpy')
        from scipy.integrate import quad
        result, _ = quad(integrand_func, domain[0], domain[1])
        return result


def assemble_petrov_galerkin_system(trial_funcs, test_funcs, differential_operator,
                                    forcing_func, domain, element_domains=None):
    """
    Assemble the Petrov-Galerkin system matrix and vector.

    Args:
        trial_funcs: List of trial functions
        test_funcs: List of test functions
        differential_operator: Differential operator function
        forcing_func: Forcing function
        domain: Global domain
        element_domains: List of element domains (for multi-element problems)

    Returns:
        System matrix and RHS vector
    """
    n_trial = len(trial_funcs)
    n_test = len(test_funcs)

    # For well-posed problems, we need n_trial = n_test
    if n_trial != n_test:
        print(f"Warning: Number of trial functions ({n_trial}) != number of test functions ({n_test})")

    system_matrix = np.zeros((n_test, n_trial))
    rhs_vector = np.zeros(n_test)

    # Use element domains if provided, otherwise use global domain
    if element_domains is None:
        element_domains = [domain]

    # Assemble over all elements
    for elem_domain in element_domains:
        for i, test_func in enumerate(test_funcs):
            for j, trial_func in enumerate(trial_funcs):
                # Compute bilinear form a(trial_func, test_func)
                system_matrix[i, j] += compute_bilinear_form(
                    test_func, trial_func, differential_operator, elem_domain
                )

            # Compute linear form L(test_func)
            rhs_vector[i] += compute_linear_form(test_func, forcing_func, elem_domain)

    return system_matrix, rhs_vector


def apply_boundary_conditions_petrov_galerkin(system_matrix, rhs_vector, trial_funcs, boundary_conditions, domain):
    """
    Apply boundary conditions to the Petrov-Galerkin system.

    Args:
        system_matrix: System matrix
        rhs_vector: RHS vector
        trial_funcs: Trial functions
        boundary_conditions: Dictionary of boundary conditions
        domain: Solution domain

    Returns:
        Modified system matrix and RHS vector
    """
    x = symbols('x')
    matrix = system_matrix.copy()
    vector = rhs_vector.copy()

    # Handle boundary conditions by modifying the system
    for bc_point, bc_value in boundary_conditions.items():
        # Find which equation to modify (typically use penalty method or modify rows)
        # For simplicity, we'll use a penalty method approach

        # Add penalty terms to enforce boundary conditions
        penalty_factor = 1e12

        # Create a row for the boundary condition
        bc_row = np.zeros(len(trial_funcs))
        for j, trial_func in enumerate(trial_funcs):
            bc_row[j] = float(trial_func.subs(x, bc_point))

        # Add penalty terms to the first equation (can be made more sophisticated)
        if len(matrix) > 0:
            matrix[0, :] += penalty_factor * bc_row
            vector[0] += penalty_factor * bc_value

    return matrix, vector


def solve_petrov_galerkin_system(system_matrix, rhs_vector):
    """
    Solve the Petrov-Galerkin system.

    Args:
        system_matrix: System matrix
        rhs_vector: RHS vector

    Returns:
        Solution coefficients
    """
    try:
        return np.linalg.solve(system_matrix, rhs_vector)
    except np.linalg.LinAlgError:
        # Use least squares for over/under-determined systems
        return np.linalg.lstsq(system_matrix, rhs_vector, rcond=None)[0]


def petrov_galerkin_method(differential_operator, forcing_func, domain, boundary_conditions,
                           trial_basis='polynomial', test_basis='polynomial',
                           trial_order=3, test_order=3, n_elements=1):
    """
    Solve differential equation using Petrov-Galerkin method.

    Args:
        differential_operator: Function that applies differential operator
        forcing_func: Forcing function
        domain: Solution domain
        boundary_conditions: Boundary conditions
        trial_basis: Type of trial basis functions
        test_basis: Type of test basis functions
        trial_order: Order of trial functions
        test_order: Order of test functions
        n_elements: Number of elements

    Returns:
        Solution coefficients, approximate solution, and basis functions
    """
    x = symbols('x')

    # Create basis functions
    trial_funcs = create_basis_functions(trial_basis, trial_order)
    test_funcs = create_test_functions(test_basis, test_order, trial_funcs)

    # Create element domains
    if n_elements > 1:
        element_length = (domain[1] - domain[0]) / n_elements
        element_domains = [(domain[0] + i * element_length, domain[0] + (i + 1) * element_length)
                           for i in range(n_elements)]
    else:
        element_domains = [domain]

    # Assemble system
    system_matrix, rhs_vector = assemble_petrov_galerkin_system(
        trial_funcs, test_funcs, differential_operator, forcing_func, domain, element_domains
    )

    # Apply boundary conditions
    system_matrix, rhs_vector = apply_boundary_conditions_petrov_galerkin(
        system_matrix, rhs_vector, trial_funcs, boundary_conditions, domain
    )

    # Solve system
    coefficients = solve_petrov_galerkin_system(system_matrix, rhs_vector)

    # Create approximate solution
    approximate_solution = sum(coeff * trial_func for coeff, trial_func in zip(coefficients, trial_funcs))

    # Create numerical evaluation function
    def evaluate_solution(x_vals):
        if hasattr(x_vals, '__iter__'):
            return [float(approximate_solution.subs(x, val)) for val in x_vals]
        else:
            return float(approximate_solution.subs(x, x_vals))

    return coefficients, approximate_solution, evaluate_solution, trial_funcs, test_funcs


def validate_petrov_galerkin_solution(coefficients, trial_funcs, test_funcs,
                                      differential_operator, forcing_func, domain):
    """
    Validate the Petrov-Galerkin solution.

    Args:
        coefficients: Solution coefficients
        trial_funcs: Trial functions
        test_funcs: Test functions
        differential_operator: Differential operator
        forcing_func: Forcing function
        domain: Solution domain

    Returns:
        Validation metrics
    """
    x = symbols('x')

    # Construct approximate solution
    approximate_solution = sum(coeff * trial_func for coeff, trial_func in zip(coefficients, trial_funcs))

    # Compute residual
    residual = differential_operator(approximate_solution) - forcing_func

    # Compute weighted residuals (should be zero for all test functions)
    weighted_residuals = []
    for test_func in test_funcs:
        weighted_residual = compute_linear_form(test_func, residual, domain)
        weighted_residuals.append(weighted_residual)

    return {
        'weighted_residuals': weighted_residuals,
        'max_weighted_residual': max(abs(r) for r in weighted_residuals),
        'residual_norm': sum(r ** 2 for r in weighted_residuals) ** 0.5
    }


def main():
    """Main function with example usage."""
    try:
        x = symbols('x')

        print("=== Petrov-Galerkin Method Examples ===\n")

        # Example 1: Second-order ODE: u'' + u = x with u(0) = 0, u(1) = 0
        print("Example 1: u'' + u = x with u(0) = 0, u(1) = 0")
        print("Using polynomial trial functions and polynomial test functions")
        print("-" * 60)

        def diff_operator_ex1(u):
            return diff(u, x, 2) + u

        forcing_func_ex1 = x
        domain_ex1 = (0, 1)
        bc_ex1 = {0: 0, 1: 0}

        coeffs1, approx_sol1, eval_func1, trial_funcs1, test_funcs1 = petrov_galerkin_method(
            diff_operator_ex1, forcing_func_ex1, domain_ex1, bc_ex1,
            trial_basis='polynomial', test_basis='polynomial',
            trial_order=4, test_order=4
        )

        print("Coefficients:")
        for i, coeff in enumerate(coeffs1):
            print(f"c{i} = {coeff:.6f}")

        print(f"\nApproximate solution: {approx_sol1}")

        # Validate solution
        validation1 = validate_petrov_galerkin_solution(
            coeffs1, trial_funcs1, test_funcs1, diff_operator_ex1, forcing_func_ex1, domain_ex1
        )
        print(f"Max weighted residual: {validation1['max_weighted_residual']:.2e}")

        # Test at some points
        test_points = [0.2, 0.5, 0.8]
        print("\nSolution at test points:")
        for pt in test_points:
            print(f"u({pt}) = {eval_func1(pt):.6f}")

        print("\n" + "=" * 70 + "\n")

        # Example 2: First-order ODE: u' - u = 1 with u(0) = 0
        print("Example 2: u' - u = 1 with u(0) = 0")
        print("Using polynomial trial and derivative test functions (Petrov-Galerkin)")
        print("-" * 60)

        def diff_operator_ex2(u):
            return diff(u, x) - u

        forcing_func_ex2 = 1
        domain_ex2 = (0, 1)
        bc_ex2 = {0: 0}

        coeffs2, approx_sol2, eval_func2, trial_funcs2, test_funcs2 = petrov_galerkin_method(
            diff_operator_ex2, forcing_func_ex2, domain_ex2, bc_ex2,
            trial_basis='polynomial', test_basis='derivative',
            trial_order=3, test_order=3
        )

        print("Coefficients:")
        for i, coeff in enumerate(coeffs2):
            print(f"c{i} = {coeff:.6f}")

        print(f"\nApproximate solution: {approx_sol2}")

        # Validate solution
        validation2 = validate_petrov_galerkin_solution(
            coeffs2, trial_funcs2, test_funcs2, diff_operator_ex2, forcing_func_ex2, domain_ex2
        )
        print(f"Max weighted residual: {validation2['max_weighted_residual']:.2e}")

        # Analytical solution for comparison: u = e^x - 1
        from sympy import exp
        analytical_sol2 = exp(x) - 1
        print(f"Analytical solution: {analytical_sol2}")

        print("\nComparison at test points:")
        for pt in test_points:
            numerical = eval_func2(pt)
            analytical = float(analytical_sol2.subs(x, pt))
            error = abs(numerical - analytical)
            print(f"u({pt}): Numerical = {numerical:.6f}, Analytical = {analytical:.6f}, Error = {error:.2e}")

        print("\n" + "=" * 70 + "\n")

        # Example 3: Advection-diffusion equation: u'' - u' = 1 with u(0) = 0, u(1) = 0
        print("Example 3: u'' - u' = 1 with u(0) = 0, u(1) = 0")
        print("Using polynomial trial and weighted test functions")
        print("-" * 60)

        def diff_operator_ex3(u):
            return diff(u, x, 2) - diff(u, x)

        forcing_func_ex3 = 1
        domain_ex3 = (0, 1)
        bc_ex3 = {0: 0, 1: 0}

        coeffs3, approx_sol3, eval_func3, trial_funcs3, test_funcs3 = petrov_galerkin_method(
            diff_operator_ex3, forcing_func_ex3, domain_ex3, bc_ex3,
            trial_basis='polynomial', test_basis='weighted',
            trial_order=4, test_order=4
        )

        print("Coefficients:")
        for i, coeff in enumerate(coeffs3):
            print(f"c{i} = {coeff:.6f}")

        print(f"\nApproximate solution: {approx_sol3}")

        # Validate solution
        validation3 = validate_petrov_galerkin_solution(
            coeffs3, trial_funcs3, test_funcs3, diff_operator_ex3, forcing_func_ex3, domain_ex3
        )
        print(f"Max weighted residual: {validation3['max_weighted_residual']:.2e}")

        print("\nSolution at test points:")
        for pt in test_points:
            print(f"u({pt}) = {eval_func3(pt):.6f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()