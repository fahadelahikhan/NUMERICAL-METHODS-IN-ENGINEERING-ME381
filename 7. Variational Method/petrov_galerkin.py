#!/usr/bin/env python3

import numpy as np
from sympy import symbols, integrate, lambdify


def compute_element_matrix(test_func, trial_func, domain, test_args, trial_args):
    """Compute the element matrix for the given test and trial functions."""
    x = symbols('x')
    phi = trial_func(x, *trial_args)
    psi = test_func(x, *test_args)
    integrand = phi * psi
    return integrate(integrand, (x, domain[0], domain[1]))


def compute_element_vector(test_func, source_func, domain, test_args):
    """Compute the element vector for the given test function and source term."""
    x = symbols('x')
    psi = test_func(x, *test_args)
    integrand = psi * source_func(x)
    return integrate(integrand, (x, domain[0], domain[1]))


def assemble_system(test_func, trial_func, source_func, elements, domain):
    """Assemble the global system matrices and vectors."""
    num_elements = len(elements)
    num_dofs = sum(len(element['nodes']) for element in elements)
    global_matrix = np.zeros((num_dofs, num_dofs))
    global_vector = np.zeros(num_dofs)

    for i, element in enumerate(elements):
        element_domain = element['domain']
        test_args = element['test_args']
        trial_args = element['trial_args']
        local_matrix = compute_element_matrix(test_func, trial_func, element_domain, test_args, trial_args)
        local_vector = compute_element_vector(test_func, source_func, element_domain, test_args)
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

        # Define test and trial functions (example: simple linear functions)
        def test_func(x, *args):
            return x

        def trial_func(x, *args):
            return x

        # Define source term (example: simple function)
        def source_func(x):
            return 1

        # Input elements and boundary conditions
        elements = [
            {'domain': (0, 1), 'nodes': [0, 1], 'test_args': (1,), 'trial_args': (1,)},
            {'domain': (1, 2), 'nodes': [1, 2], 'test_args': (2,), 'trial_args': (2,)}
        ]
        boundary_conditions = {0: 0, 2: 0}

        # Assemble the system
        global_matrix, global_vector = assemble_system(test_func, trial_func, source_func, elements, (0, 2))

        # Apply boundary conditions
        global_matrix, global_vector = apply_boundary_conditions(global_matrix, global_vector, boundary_conditions)

        # Solve the system
        solution = solve_system(global_matrix, global_vector)

        # Output the results
        print("Nodal values:")
        for i, value in enumerate(solution):
            print(f"Node {i}: {value:.6f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()