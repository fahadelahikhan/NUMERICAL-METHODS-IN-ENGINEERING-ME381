#!/usr/bin/env python3

import numpy as np


def compute_element_stiffness(E, A, L):
    """Compute the stiffness matrix for a 1D bar element."""
    k = E * A / L
    return np.array([[k, -k], [-k, k]])


def assemble_global_stiffness(elements, nodes):
    """Assemble the global stiffness matrix from individual element stiffness matrices."""
    num_dofs = len(nodes)
    global_stiffness = np.zeros((num_dofs, num_dofs))

    for element in elements:
        # Extract element properties
        E = element['E']
        A = element['A']
        L = element['L']
        node_i = element['node_i']
        node_j = element['node_j']

        # Compute element stiffness matrix
        element_stiffness = compute_element_stiffness(E, A, L)

        # Add element stiffness to global stiffness matrix
        global_stiffness[node_i, node_i] += element_stiffness[0, 0]
        global_stiffness[node_i, node_j] += element_stiffness[0, 1]
        global_stiffness[node_j, node_i] += element_stiffness[1, 0]
        global_stiffness[node_j, node_j] += element_stiffness[1, 1]

    return global_stiffness


def main():
    try:
        # Input: Number of nodes and elements
        num_nodes = int(input("Enter the number of nodes: "))
        num_elements = int(input("Enter the number of elements: "))

        # Input: Node positions
        nodes = []
        for i in range(num_nodes):
            pos = float(input(f"Enter position for node {i}: "))
            nodes.append(pos)

        # Input: Element properties
        elements = []
        for i in range(num_elements):
            E = float(input(f"Enter modulus of elasticity for element {i}: "))
            A = float(input(f"Enter cross-sectional area for element {i}: "))
            L = float(input(f"Enter length for element {i}: "))
            node_i = int(input(f"Enter start node for element {i}: "))
            node_j = int(input(f"Enter end node for element {i}: "))
            elements.append({'E': E, 'A': A, 'L': L, 'node_i': node_i, 'node_j': node_j})

        # Assemble global stiffness matrix
        global_stiffness = assemble_global_stiffness(elements, nodes)

        # Output the global stiffness matrix
        print("\nGlobal Stiffness Matrix:")
        print(global_stiffness)

    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()