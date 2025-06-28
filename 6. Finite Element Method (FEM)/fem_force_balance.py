#!/usr/bin/env python3

import numpy as np


def compute_element_stiffness(E, A, L):
    """Compute the stiffness matrix for a 1D bar element."""
    k = E * A / L
    return np.array([[k, -k], [-k, k]])


def assemble_global_stiffness(elements):
    """Assemble the global stiffness matrix from individual element stiffness matrices."""
    max_node = max(max(element['nodes']) for element in elements)
    global_stiffness = np.zeros((max_node + 1, max_node + 1))

    for element in elements:
        E = element['E']
        A = element['A']
        L = element['L']
        nodes = element['nodes']

        element_stiffness = compute_element_stiffness(E, A, L)
        for i in range(2):
            for j in range(2):
                global_stiffness[nodes[i], nodes[j]] += element_stiffness[i, j]

    return global_stiffness


def apply_boundary_conditions(global_stiffness, force_vector, fixed_nodes):
    """Apply displacement boundary conditions."""
    for node in fixed_nodes:
        global_stiffness[node, :] = 0.0
        global_stiffness[node, node] = 1.0
        force_vector[node] = 0.0
    return global_stiffness, force_vector


def solve_fem(global_stiffness, force_vector):
    """Solve the FEM system for displacements."""
    return np.linalg.solve(global_stiffness, force_vector)


def compute_reactions(global_stiffness, displacements, fixed_nodes):
    """Compute reactions at fixed nodes."""
    reactions = np.dot(global_stiffness, displacements)
    return reactions[fixed_nodes]


def main():
    try:
        # Input: Number of elements
        num_elements = int(input("Enter the number of elements: "))

        # Input: Element properties
        elements = []
        for i in range(num_elements):
            E = float(input(f"Enter modulus of elasticity for element {i}: "))
            A = float(input(f"Enter cross-sectional area for element {i}: "))
            L = float(input(f"Enter length for element {i}: "))
            node_i = int(input(f"Enter start node for element {i}: "))
            node_j = int(input(f"Enter end node for element {i}: "))
            elements.append({'E': E, 'A': A, 'L': L, 'nodes': [node_i, node_j]})

        # Input: Force vector
        max_node = max(max(element['nodes']) for element in elements)
        force_vector = np.zeros(max_node + 1)
        num_forces = int(input("Enter the number of applied forces: "))
        for _ in range(num_forces):
            node = int(input("Enter node number: "))
            force = float(input("Enter force value: "))
            force_vector[node] = force

        # Input: Boundary conditions
        fixed_nodes = list(map(int, input("Enter fixed nodes (space-separated): ").split()))

        # Assemble global stiffness matrix
        global_stiffness = assemble_global_stiffness(elements)

        # Apply boundary conditions
        global_stiffness, force_vector = apply_boundary_conditions(global_stiffness, force_vector, fixed_nodes)

        # Solve for displacements
        displacements = solve_fem(global_stiffness, force_vector)

        # Compute reactions
        reactions = compute_reactions(global_stiffness, displacements, fixed_nodes)

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