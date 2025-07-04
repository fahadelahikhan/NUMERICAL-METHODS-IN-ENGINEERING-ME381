#!/usr/bin/env python3

import numpy as np
from sympy import symbols, integrate, lambdify, diff
import warnings


class GalerkinSolver:
    """
    Galerkin Method solver for 1D boundary value problems.
    Designed for easy translation to other programming languages.
    """

    def __init__(self):
        self.x = symbols('x')
        self.tolerance = 1e-12

    def linear_basis_function(self, x, node_coords, local_node_id):
        """
        Linear basis function for 1D elements.

        Args:
            x: symbolic variable
            node_coords: list of node coordinates [x0, x1]
            local_node_id: 0 or 1 for linear element

        Returns:
            basis function expression
        """
        x0, x1 = node_coords[0], node_coords[1]
        h = x1 - x0

        if local_node_id == 0:
            return (x1 - x) / h
        elif local_node_id == 1:
            return (x - x0) / h
        else:
            raise ValueError("Invalid local node ID for linear element")

    def quadratic_basis_function(self, x, node_coords, local_node_id):
        """
        Quadratic basis function for 1D elements.

        Args:
            x: symbolic variable
            node_coords: list of node coordinates [x0, x1, x2]
            local_node_id: 0, 1, or 2 for quadratic element

        Returns:
            basis function expression
        """
        x0, x1, x2 = node_coords[0], node_coords[1], node_coords[2]
        h = x2 - x0

        if local_node_id == 0:
            return 2 * (x - x1) * (x - x2) / (h ** 2)
        elif local_node_id == 1:
            return -4 * (x - x0) * (x - x2) / (h ** 2)
        elif local_node_id == 2:
            return 2 * (x - x0) * (x - x1) / (h ** 2)
        else:
            raise ValueError("Invalid local node ID for quadratic element")

    def compute_element_stiffness_matrix(self, element_info, diffusion_coeff=1.0):
        """
        Compute element stiffness matrix for -d/dx(k*du/dx) term.

        Args:
            element_info: dictionary containing element information
            diffusion_coeff: diffusion coefficient (default: 1.0)

        Returns:
            local stiffness matrix
        """
        basis_type = element_info['basis_type']
        node_coords = element_info['node_coords']
        domain = element_info['domain']

        if basis_type == 'linear':
            n_nodes = 2
            basis_func = self.linear_basis_function
        elif basis_type == 'quadratic':
            n_nodes = 3
            basis_func = self.quadratic_basis_function
        else:
            raise ValueError(f"Unsupported basis type: {basis_type}")

        stiffness_matrix = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(n_nodes):
                phi_i = basis_func(self.x, node_coords, i)
                phi_j = basis_func(self.x, node_coords, j)

                # Compute ∫(k * dφ_i/dx * dφ_j/dx) dx
                dphi_i_dx = diff(phi_i, self.x)
                dphi_j_dx = diff(phi_j, self.x)

                integrand = diffusion_coeff * dphi_i_dx * dphi_j_dx
                integral_result = integrate(integrand, (self.x, domain[0], domain[1]))

                stiffness_matrix[i, j] = float(integral_result)

        return stiffness_matrix

    def compute_element_mass_matrix(self, element_info, mass_coeff=1.0):
        """
        Compute element mass matrix for reaction term.

        Args:
            element_info: dictionary containing element information
            mass_coeff: mass coefficient (default: 1.0)

        Returns:
            local mass matrix
        """
        basis_type = element_info['basis_type']
        node_coords = element_info['node_coords']
        domain = element_info['domain']

        if basis_type == 'linear':
            n_nodes = 2
            basis_func = self.linear_basis_function
        elif basis_type == 'quadratic':
            n_nodes = 3
            basis_func = self.quadratic_basis_function
        else:
            raise ValueError(f"Unsupported basis type: {basis_type}")

        mass_matrix = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(n_nodes):
                phi_i = basis_func(self.x, node_coords, i)
                phi_j = basis_func(self.x, node_coords, j)

                # Compute ∫(c * φ_i * φ_j) dx
                integrand = mass_coeff * phi_i * phi_j
                integral_result = integrate(integrand, (self.x, domain[0], domain[1]))

                mass_matrix[i, j] = float(integral_result)

        return mass_matrix

    def compute_element_load_vector(self, element_info, source_func):
        """
        Compute element load vector.

        Args:
            element_info: dictionary containing element information
            source_func: source function f(x)

        Returns:
            local load vector
        """
        basis_type = element_info['basis_type']
        node_coords = element_info['node_coords']
        domain = element_info['domain']

        if basis_type == 'linear':
            n_nodes = 2
            basis_func = self.linear_basis_function
        elif basis_type == 'quadratic':
            n_nodes = 3
            basis_func = self.quadratic_basis_function
        else:
            raise ValueError(f"Unsupported basis type: {basis_type}")

        load_vector = np.zeros(n_nodes)

        for i in range(n_nodes):
            phi_i = basis_func(self.x, node_coords, i)

            # Compute ∫(f * φ_i) dx
            integrand = source_func(self.x) * phi_i
            integral_result = integrate(integrand, (self.x, domain[0], domain[1]))

            load_vector[i] = float(integral_result)

        return load_vector

    def assemble_global_system(self, elements, source_func, diffusion_coeff=1.0, mass_coeff=0.0):
        """
        Assemble global system matrices and vectors.

        Args:
            elements: list of element dictionaries
            source_func: source function f(x)
            diffusion_coeff: diffusion coefficient
            mass_coeff: mass coefficient

        Returns:
            global_matrix, global_vector
        """
        # Determine total number of degrees of freedom
        all_nodes = set()
        for element in elements:
            all_nodes.update(element['global_nodes'])

        n_dofs = len(all_nodes)
        node_map = {node: i for i, node in enumerate(sorted(all_nodes))}

        global_matrix = np.zeros((n_dofs, n_dofs))
        global_vector = np.zeros(n_dofs)

        for element in elements:
            # Compute element matrices
            stiffness_matrix = self.compute_element_stiffness_matrix(element, diffusion_coeff)
            mass_matrix = self.compute_element_mass_matrix(element, mass_coeff)
            load_vector = self.compute_element_load_vector(element, source_func)

            # Combine stiffness and mass matrices
            element_matrix = stiffness_matrix + mass_matrix

            # Map local nodes to global nodes
            global_nodes = element['global_nodes']
            local_to_global = [node_map[node] for node in global_nodes]

            # Assemble into global system
            for i, global_i in enumerate(local_to_global):
                for j, global_j in enumerate(local_to_global):
                    global_matrix[global_i, global_j] += element_matrix[i, j]
                global_vector[global_i] += load_vector[i]

        return global_matrix, global_vector, node_map

    def apply_boundary_conditions(self, global_matrix, global_vector, node_map, boundary_conditions):
        """
        Apply Dirichlet boundary conditions.

        Args:
            global_matrix: global system matrix
            global_vector: global load vector
            node_map: mapping from physical nodes to matrix indices
            boundary_conditions: dict of {node_id: value}

        Returns:
            modified global_matrix, global_vector
        """
        for node_id, value in boundary_conditions.items():
            if node_id in node_map:
                dof_index = node_map[node_id]

                # Set row to identity
                global_matrix[dof_index, :] = 0.0
                global_matrix[dof_index, dof_index] = 1.0
                global_vector[dof_index] = value

        return global_matrix, global_vector

    def solve_system(self, global_matrix, global_vector):
        """
        Solve the linear system.

        Args:
            global_matrix: global system matrix
            global_vector: global load vector

        Returns:
            solution vector
        """
        # Check for singular matrix
        if np.linalg.det(global_matrix) < self.tolerance:
            warnings.warn("Matrix is nearly singular. Solution may be unreliable.")

        try:
            solution = np.linalg.solve(global_matrix, global_vector)
            return solution
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Failed to solve system: {e}")

    def solve_bvp(self, elements, source_func, boundary_conditions,
                  diffusion_coeff=1.0, mass_coeff=0.0):
        """
        Solve boundary value problem using Galerkin method.

        Args:
            elements: list of element dictionaries
            source_func: source function f(x)
            boundary_conditions: dict of {node_id: value}
            diffusion_coeff: diffusion coefficient
            mass_coeff: mass coefficient

        Returns:
            solution vector, node mapping
        """
        # Assemble global system
        global_matrix, global_vector, node_map = self.assemble_global_system(
            elements, source_func, diffusion_coeff, mass_coeff)

        # Apply boundary conditions
        global_matrix, global_vector = self.apply_boundary_conditions(
            global_matrix, global_vector, node_map, boundary_conditions)

        # Solve system
        solution = self.solve_system(global_matrix, global_vector)

        return solution, node_map


def main():
    """
    Main function demonstrating the Galerkin solver.
    """
    try:
        # Create solver instance
        solver = GalerkinSolver()

        # Define source function
        def source_func(x):
            return 1.0  # f(x) = 1

        # Example 1: Simple linear elements
        print("=" * 50)
        print("Example 1: Linear elements, uniform mesh")
        print("=" * 50)

        elements = [
            {
                'basis_type': 'linear',
                'node_coords': [0.0, 0.5],
                'domain': (0.0, 0.5),
                'global_nodes': [0, 1]
            },
            {
                'basis_type': 'linear',
                'node_coords': [0.5, 1.0],
                'domain': (0.5, 1.0),
                'global_nodes': [1, 2]
            }
        ]

        boundary_conditions = {0: 0.0, 2: 0.0}

        solution, node_map = solver.solve_bvp(
            elements, source_func, boundary_conditions, diffusion_coeff=1.0)

        print("Solution:")
        for node_id in sorted(node_map.keys()):
            dof_index = node_map[node_id]
            print(f"Node {node_id} (x={node_id * 0.5:.1f}): u = {solution[dof_index]:.6f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()