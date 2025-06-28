#!/usr/bin/env python3

import numpy as np
from sympy import symbols, integrate, lambdify, diff


def compute_element_matrix(basis_funcs, residual, domain, args):
    """Compute the element matrix for the least squares method."""
    x = symbols('x')
    n = len(basis_funcs)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            phi_i = basis_funcs[i](x, *args)
            phi_j = basis_funcs[j](x, *args)
            integrand = phi_i * phi_j
            matrix[i, j] = integrate(integrand, (x, domain[0], domain[1]))
    return matrix


def compute_element_vector(basis_funcs, residual, domain, args):
    """Compute the element vector for the least squares method."""
    x = symbols('x')
    n = len(basis_funcs)
    vector = np.zeros(n)
    for i in range(n):
        phi_i = basis_funcs[i](x, *args)
        integrand = phi_i * residual
        vector[i] = integrate(integrand, (x, domain[0], domain[1]))
    return vector


def assemble_system(basis_funcs, residual, elements, domain):
    """Assemble the global system matrices and vectors."""
    num_elements = len(elements)
    num_dofs = sum(len(element['nodes']) for element in elements)
    global_matrix = np.zeros((num_dofs, num_dofs))
    global_vector = np.zeros(num_dofs)

    for i, element in enumerate(elements):
        element_domain = element['domain']
        args = element['args']
        local_matrix = compute_element_matrix(basis_funcs, residual, element_domain, args)
        local_vector = compute_element_vector(basis_funcs, residual, element_domain, args)
        nodes = element['nodes']
        for j in range(len(nodes)):
            for k in range(len(nodes)):
                global_matrix[nodes[j], nodes[k]] += local_matrix[j, k]
            global_vector[nodes[j]] += local_vector[j]

    return global_matrix, global_vector


def apply_boundary_conditions(global_matrix, global_vector, boundary_conditions):
    """Apply the specified boundary conditions."""
    for dof, value in boundary_conditions.items():
        global_matrix[dof, :] = 0.0
        global_matrix[dof, dof] = 1.0
        global_vector[dof] = value
    return global_matrix, global_vector


def solve_system(global_matrix, global_vector):
    """Solve the system of equations."""
    return np.linalg.solve(global_matrix, global_vector)


def main():
    try:
        # Define symbolic variables
        x = symbols('x')
        c1, c2 = symbols('c1 c2')

        # Define basis functions (example: simple linear functions)
        def basis_func1(x, *args):
            return 1

        def basis_func2(x, *args):
            return x

        basis_funcs = [basis_func1, basis_func2]

        # Define the differential equation residual (example: simple function)
        def approximate_solution(x, c1, c2):
            return c1 + c2 * x

        residual = diff(approximate_solution(x, c1, c2), x, 2) + 1

        # Input elements and boundary conditions
        elements = [
            {'domain': (0, 1), 'nodes': [0, 1], 'args': (1, 2)},
        ]
        boundary_conditions = {0: 0, 1: 0}

        # Assemble the system
        global_matrix, global_vector = assemble_system(basis_funcs, residual, elements, (0, 1))

        # Apply boundary conditions
        global_matrix, global_vector = apply_boundary_conditions(global_matrix, global_vector, boundary_conditions)

        # Solve the system
        solution = solve_system(global_matrix, global_vector)

        # Output the results
        print("Coefficients of the basis functions:")
        for i, coeff in enumerate(solution):
            print(f"c{i + 1} = {coeff:.6f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()