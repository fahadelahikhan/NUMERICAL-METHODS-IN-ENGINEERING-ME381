#!/usr/bin/env python3

import numpy as np
from sympy import symbols, integrate, lambdify, diff, Function


def compute_element_matrix(basis_funcs, residual_func, domain, differential_operator=None):
    """
    Compute the element matrix for the least squares method.

    Args:
        basis_funcs: List of basis functions
        residual_func: Function that computes residual given trial function
        domain: Integration domain (start, end)
        differential_operator: Optional differential operator function

    Returns:
        Element matrix as numpy array
    """
    x = symbols('x')
    n = len(basis_funcs)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Create trial function as linear combination of basis functions
            # For least squares: (L[phi_i], L[phi_j]) where L is the differential operator
            if differential_operator:
                L_phi_i = differential_operator(basis_funcs[i])
                L_phi_j = differential_operator(basis_funcs[j])
            else:
                L_phi_i = basis_funcs[i]
                L_phi_j = basis_funcs[j]

            integrand = L_phi_i * L_phi_j
            try:
                integral_result = integrate(integrand, (x, domain[0], domain[1]))
                matrix[i, j] = float(integral_result)
            except:
                # Fallback to numerical integration if symbolic fails
                integrand_func = lambdify(x, integrand, 'numpy')
                from scipy.integrate import quad
                result, _ = quad(integrand_func, domain[0], domain[1])
                matrix[i, j] = result

    return matrix


def compute_element_vector(basis_funcs, residual_func, domain, differential_operator=None):
    """
    Compute the element vector for the least squares method.

    Args:
        basis_funcs: List of basis functions
        residual_func: Function that computes residual
        domain: Integration domain (start, end)
        differential_operator: Optional differential operator function

    Returns:
        Element vector as numpy array
    """
    x = symbols('x')
    n = len(basis_funcs)
    vector = np.zeros(n)

    for i in range(n):
        # For least squares: (f, L[phi_i]) where f is the forcing function
        if differential_operator:
            L_phi_i = differential_operator(basis_funcs[i])
        else:
            L_phi_i = basis_funcs[i]

        integrand = residual_func * L_phi_i
        try:
            integral_result = integrate(integrand, (x, domain[0], domain[1]))
            vector[i] = float(integral_result)
        except:
            # Fallback to numerical integration if symbolic fails
            integrand_func = lambdify(x, integrand, 'numpy')
            from scipy.integrate import quad
            result, _ = quad(integrand_func, domain[0], domain[1])
            vector[i] = result

    return vector


def assemble_system(basis_funcs, residual_func, elements, differential_operator=None):
    """
    Assemble the global system matrices and vectors.

    Args:
        basis_funcs: List of basis functions
        residual_func: Function representing the residual/forcing term
        elements: List of element dictionaries
        differential_operator: Optional differential operator function

    Returns:
        Global matrix and vector
    """
    if not elements:
        raise ValueError("No elements provided")

    num_elements = len(elements)
    total_dofs = sum(len(element['nodes']) for element in elements)

    # For simple case, assume global DOFs equal total DOFs
    max_node = max(max(element['nodes']) for element in elements)
    num_dofs = max_node + 1

    global_matrix = np.zeros((num_dofs, num_dofs))
    global_vector = np.zeros(num_dofs)

    for element in elements:
        element_domain = element['domain']
        local_matrix = compute_element_matrix(basis_funcs, residual_func, element_domain, differential_operator)
        local_vector = compute_element_vector(basis_funcs, residual_func, element_domain, differential_operator)

        nodes = element['nodes']
        for j in range(len(nodes)):
            for k in range(len(nodes)):
                global_matrix[nodes[j], nodes[k]] += local_matrix[j, k]
            global_vector[nodes[j]] += local_vector[j]

    return global_matrix, global_vector


def apply_boundary_conditions(global_matrix, global_vector, boundary_conditions):
    """
    Apply the specified boundary conditions using penalty method.

    Args:
        global_matrix: Global system matrix
        global_vector: Global system vector
        boundary_conditions: Dictionary of {dof: value} pairs

    Returns:
        Modified global matrix and vector
    """
    matrix = global_matrix.copy()
    vector = global_vector.copy()

    for dof, value in boundary_conditions.items():
        if dof >= len(vector):
            continue
        # Set row to enforce boundary condition
        matrix[dof, :] = 0.0
        matrix[dof, dof] = 1.0
        vector[dof] = value

    return matrix, vector


def solve_system(global_matrix, global_vector):
    """
    Solve the system of equations.

    Args:
        global_matrix: Global system matrix
        global_vector: Global system vector

    Returns:
        Solution vector
    """
    try:
        return np.linalg.solve(global_matrix, global_vector)
    except np.linalg.LinAlgError:
        # Try pseudo-inverse for ill-conditioned systems
        return np.linalg.pinv(global_matrix) @ global_vector


def create_basis_functions(basis_type='polynomial', order=2):
    """
    Create basis functions based on specified type and order.

    Args:
        basis_type: Type of basis ('polynomial', 'trigonometric')
        order: Order of basis functions

    Returns:
        List of basis functions
    """
    x = symbols('x')

    if basis_type == 'polynomial':
        return [x ** i for i in range(order + 1)]
    elif basis_type == 'trigonometric':
        from sympy import sin, cos, pi
        funcs = [1]  # Constant term
        for i in range(1, order + 1):
            funcs.extend([sin(i * pi * x), cos(i * pi * x)])
        return funcs[:order + 1]
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


def solve_ode_problem(differential_eq, forcing_func, domain, boundary_conditions,
                      basis_type='polynomial', order=2):
    """
    Solve an ODE problem using least squares variational method.

    Args:
        differential_eq: Function that applies differential operator to input
        forcing_func: Forcing function (RHS of ODE)
        domain: Solution domain (start, end)
        boundary_conditions: Dictionary of boundary conditions
        basis_type: Type of basis functions
        order: Order of approximation

    Returns:
        Solution coefficients and basis functions
    """
    # Create basis functions
    basis_funcs = create_basis_functions(basis_type, order)

    # Create elements (single element for now)
    elements = [{'domain': domain, 'nodes': list(range(len(basis_funcs)))}]

    # Assemble system
    global_matrix, global_vector = assemble_system(
        basis_funcs, forcing_func, elements, differential_eq
    )

    # Apply boundary conditions
    global_matrix, global_vector = apply_boundary_conditions(
        global_matrix, global_vector, boundary_conditions
    )

    # Solve system
    solution = solve_system(global_matrix, global_vector)

    return solution, basis_funcs


def main():
    """Main function with example usage."""
    try:
        x = symbols('x')

        # Example 1: Simple second-order ODE: u'' + u = 1
        print("=== Example 1: u'' + u = 1 ===")

        def diff_operator_ex1(u):
            return diff(u, x, 2) + u

        forcing_func_ex1 = 1
        domain_ex1 = (0, 1)
        bc_ex1 = {0: 0, 1: 0}  # u(0) = 0, u(1) = 0

        solution1, basis1 = solve_ode_problem(
            diff_operator_ex1, forcing_func_ex1, domain_ex1, bc_ex1,
            basis_type='polynomial', order=3
        )

        print("Coefficients:")
        for i, coeff in enumerate(solution1):
            print(f"c{i} = {coeff:.6f}")

        # Construct approximate solution
        approx_sol1 = sum(coeff * basis for coeff, basis in zip(solution1, basis1))
        print(f"Approximate solution: {approx_sol1}")

        print("\n" + "=" * 50 + "\n")

        # Example 2: First-order ODE: u' - u = x
        print("=== Example 2: u' - u = x ===")

        def diff_operator_ex2(u):
            return diff(u, x) - u

        forcing_func_ex2 = x
        domain_ex2 = (0, 1)
        bc_ex2 = {0: 1}  # u(0) = 1

        solution2, basis2 = solve_ode_problem(
            diff_operator_ex2, forcing_func_ex2, domain_ex2, bc_ex2,
            basis_type='polynomial', order=2
        )

        print("Coefficients:")
        for i, coeff in enumerate(solution2):
            print(f"c{i} = {coeff:.6f}")

        # Construct approximate solution
        approx_sol2 = sum(coeff * basis for coeff, basis in zip(solution2, basis2))
        print(f"Approximate solution: {approx_sol2}")

        print("\n" + "=" * 50 + "\n")

        # Example 3: Poisson equation: u'' = -sin(πx)
        print("=== Example 3: u'' = -sin(πx) ===")

        from sympy import sin, pi

        def diff_operator_ex3(u):
            return diff(u, x, 2)

        forcing_func_ex3 = -sin(pi * x)
        domain_ex3 = (0, 1)
        bc_ex3 = {0: 0, 1: 0}  # u(0) = 0, u(1) = 0

        solution3, basis3 = solve_ode_problem(
            diff_operator_ex3, forcing_func_ex3, domain_ex3, bc_ex3,
            basis_type='polynomial', order=4
        )

        print("Coefficients:")
        for i, coeff in enumerate(solution3):
            print(f"c{i} = {coeff:.6f}")

        # Construct approximate solution
        approx_sol3 = sum(coeff * basis for coeff, basis in zip(solution3, basis3))
        print(f"Approximate solution: {approx_sol3}")

        # Analytical solution for comparison: u = sin(πx)/π²
        analytical = sin(pi * x) / (pi ** 2)
        print(f"Analytical solution: {analytical}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()