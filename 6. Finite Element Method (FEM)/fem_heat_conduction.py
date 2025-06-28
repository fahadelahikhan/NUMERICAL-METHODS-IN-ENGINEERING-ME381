#!/usr/bin/env python3

import numpy as np


def compute_element_conductivity(k, length):
    """Compute the thermal conductivity matrix for a 1D element."""
    return np.array([[k / length, -k / length],
                     [-k / length, k / length]])


def assemble_global_conductivity(elements):
    """Assemble the global thermal conductivity matrix."""
    num_nodes = elements[-1]['end_node'] + 1
    global_cond = np.zeros((num_nodes, num_nodes))

    for elem in elements:
        elem_matrix = compute_element_conductivity(elem['k'], elem['length'])
        global_cond[elem['start_node'], elem['start_node']] += elem_matrix[0, 0]
        global_cond[elem['start_node'], elem['end_node']] += elem_matrix[0, 1]
        global_cond[elem['end_node'], elem['start_node']] += elem_matrix[1, 0]
        global_cond[elem['end_node'], elem['end_node']] += elem_matrix[1, 1]

    return global_cond


def apply_boundary_conditions(global_cond, temp_BC):
    """Apply temperature boundary conditions."""
    for node, temp in temp_BC.items():
        global_cond[node, :] = 0.0
        global_cond[node, node] = 1.0
    return global_cond


def solve_fem_heat_conduction(elements, temp_BC, heat_load=None):
    """Solve the FEM system for nodal temperatures."""
    num_nodes = elements[-1]['end_node'] + 1
    global_cond = assemble_global_conductivity(elements)
    global_cond = apply_boundary_conditions(global_cond, temp_BC)

    # Create load vector (assuming uniform heat generation if provided)
    load = np.zeros(num_nodes)
    if heat_load is not None:
        for elem in elements:
            load[elem['start_node']] += heat_load * elem['length'] / 2
            load[elem['end_node']] += heat_load * elem['length'] / 2

    # Solve the system
    temp = np.linalg.solve(global_cond, load)
    return temp


def main():
    try:
        # Input: Number of elements
        num_elements = int(input("Enter the number of elements: "))

        # Input: Element properties
        elements = []
        for i in range(num_elements):
            k = float(input(f"Enter thermal conductivity for element {i}: "))
            length = float(input(f"Enter length for element {i}: "))
            start_node = i * 2
            end_node = start_node + 1
            elements.append({'k': k, 'length': length, 'start_node': start_node, 'end_node': end_node})

        # Input: Boundary conditions
        temp_BC = {}
        num_BC = int(input("Enter the number of temperature boundary conditions: "))
        for _ in range(num_BC):
            node = int(input("Enter node number: "))
            temp = float(input("Enter temperature value: "))
            temp_BC[node] = temp

        # Input: Heat load (optional)
        heat_load = None
        has_load = input("Is there a uniform heat generation? (y/n): ")
        if has_load.lower() == 'y':
            heat_load = float(input("Enter heat generation per unit volume: "))

        # Solve FEM
        temperatures = solve_fem_heat_conduction(elements, temp_BC, heat_load)

        # Output results
        print("\nNodal Temperatures:")
        for i, temp in enumerate(temperatures):
            print(f"Node {i}: {temp:.6f} K")

    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()