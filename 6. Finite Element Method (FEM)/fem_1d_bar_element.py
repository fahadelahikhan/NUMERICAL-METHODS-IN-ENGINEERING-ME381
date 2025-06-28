#!/usr/bin/env python3

import numpy as np


def compute_element_stiffness(E, A, L):
    """Compute the stiffness matrix for a 1D bar element."""
    k = E * A / L
    return np.array([[k, -k], [-k, k]])


def assemble_global_stiffness(E, A, L, num_elements):
    """Assemble the global stiffness matrix."""
    global_stiffness = np.zeros((num_elements + 1, num_elements + 1))
    for i in range(num_elements):
        element_stiffness = compute_element_stiffness(E, A, L / num_elements)
        global_stiffness[i:i + 2, i:i + 2] += element_stiffness
    return global_stiffness


def apply_boundary_conditions(global_stiffness, force_vector, fixed_nodes):
    """Modify the global stiffness matrix and force vector to apply boundary conditions."""
    for node in fixed_nodes:
        global_stiffness[node, :] = 0.0
        global_stiffness[:, node] = 0.0
        global_stiffness[node, node] = 1.0
        force_vector[node] = 0.0
    return global_stiffness, force_vector


def solve_fem(E, A, L, num_elements, force_vector, fixed_nodes):
    """Solve the FEM system for displacements."""
    global_stiffness = assemble_global_stiffness(E, A, L, num_elements)
    force_vector = np.array(force_vector)
    global_stiffness, force_vector = apply_boundary_conditions(global_stiffness, force_vector, fixed_nodes)
    displacements = np.linalg.solve(global_stiffness, force_vector)
    return displacements


def compute_reactions(E, A, L, num_elements, displacements, fixed_nodes):
    """Compute the reactions at the fixed nodes."""
    global_stiffness = assemble_global_stiffness(E, A, L, num_elements)
    reactions = np.dot(global_stiffness, displacements)
    return reactions[fixed_nodes]


def main():
    try:
        # Input parameters
        E = float(input("Enter the modulus of elasticity (E): "))
        A = float(input("Enter the cross-sectional area (A): "))
        L = float(input("Enter the length of the bar (L): "))
        num_elements = int(input("Enter the number of elements: "))
        force_vector = list(map(float, input("Enter the force vector (space-separated, e.g., 0 100 200): ").split()))
        fixed_nodes = list(map(int, input("Enter the fixed nodes (space-separated, e.g., 0): ").split()))

        # Validate inputs
        if len(force_vector) != num_elements + 1:
            print("Error: Force vector length must be equal to the number of nodes (num_elements + 1).")
            return
        if not all(0 <= node <= num_elements for node in fixed_nodes):
            print("Error: Fixed nodes must be valid node indices (0 to num_elements).")
            return

        # Solve FEM
        displacements = solve_fem(E, A, L, num_elements, force_vector, fixed_nodes)
        reactions = compute_reactions(E, A, L, num_elements, displacements, fixed_nodes)

        # Output results
        print("\nNodal Displacements:")
        for i, disp in enumerate(displacements):
            print(f"Node {i}: {disp:.6f} units")

        print("\nReactions at Fixed Nodes:")
        for i, reaction in enumerate(reactions):
            print(f"Node {fixed_nodes[i]}: {reaction:.6f} units")

    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()