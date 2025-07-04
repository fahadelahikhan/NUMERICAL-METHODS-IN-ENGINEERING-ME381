#!/usr/bin/env python3

import numpy as np
import sys


def compute_element_conductivity_matrix(k, length):
    """
    Compute the thermal conductivity matrix for a 1D element.

    Args:
        k: Thermal conductivity coefficient
        length: Element length

    Returns:
        2x2 numpy array representing element conductivity matrix
    """
    coeff = k / length
    return np.array([[coeff, -coeff],
                     [-coeff, coeff]])


def assemble_global_conductivity_matrix(elements, num_nodes):
    """
    Assemble the global thermal conductivity matrix.

    Args:
        elements: List of element dictionaries
        num_nodes: Total number of nodes

    Returns:
        Global conductivity matrix
    """
    global_K = np.zeros((num_nodes, num_nodes))

    for elem in elements:
        elem_matrix = compute_element_conductivity_matrix(elem['k'], elem['length'])
        start_node = elem['start_node']
        end_node = elem['end_node']

        # Add element matrix to global matrix
        global_K[start_node, start_node] += elem_matrix[0, 0]
        global_K[start_node, end_node] += elem_matrix[0, 1]
        global_K[end_node, start_node] += elem_matrix[1, 0]
        global_K[end_node, end_node] += elem_matrix[1, 1]

    return global_K


def create_load_vector(elements, num_nodes, heat_generation=None, point_loads=None):
    """
    Create the global load vector.

    Args:
        elements: List of element dictionaries
        num_nodes: Total number of nodes
        heat_generation: Uniform heat generation per unit volume (optional)
        point_loads: Dictionary of point loads {node: load_value} (optional)

    Returns:
        Global load vector
    """
    load_vector = np.zeros(num_nodes)

    # Add distributed loads from heat generation
    if heat_generation is not None:
        for elem in elements:
            # For uniform heat generation, distribute equally to nodes
            distributed_load = heat_generation * elem['length'] / 2.0
            load_vector[elem['start_node']] += distributed_load
            load_vector[elem['end_node']] += distributed_load

    # Add point loads
    if point_loads is not None:
        for node, load in point_loads.items():
            if 0 <= node < num_nodes:
                load_vector[node] += load

    return load_vector


def apply_temperature_boundary_conditions(global_K, load_vector, temp_BC):
    """
    Apply temperature boundary conditions using the penalty method.

    Args:
        global_K: Global conductivity matrix
        load_vector: Global load vector
        temp_BC: Dictionary of temperature boundary conditions {node: temperature}

    Returns:
        Modified global_K and load_vector
    """
    # Create copies to avoid modifying originals
    K_modified = global_K.copy()
    F_modified = load_vector.copy()

    # Apply temperature boundary conditions
    for node, temperature in temp_BC.items():
        if 0 <= node < len(load_vector):
            # Set the row to zero except diagonal
            K_modified[node, :] = 0.0
            K_modified[node, node] = 1.0
            F_modified[node] = temperature

    return K_modified, F_modified


def solve_fem_system(global_K, load_vector):
    """
    Solve the FEM system of equations.

    Args:
        global_K: Global conductivity matrix
        load_vector: Global load vector

    Returns:
        Solution vector (temperatures)
    """
    try:
        # Check if matrix is singular
        if np.linalg.det(global_K) == 0:
            raise ValueError("Singular matrix - check boundary conditions")

        temperatures = np.linalg.solve(global_K, load_vector)
        return temperatures

    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to solve system: {e}")


def solve_fem_heat_conduction(elements, temp_BC, heat_generation=None, point_loads=None):
    """
    Main solver function for FEM heat conduction.

    Args:
        elements: List of element dictionaries
        temp_BC: Dictionary of temperature boundary conditions
        heat_generation: Uniform heat generation (optional)
        point_loads: Dictionary of point loads (optional)

    Returns:
        Array of nodal temperatures
    """
    # Determine number of nodes
    max_node = max(max(elem['start_node'], elem['end_node']) for elem in elements)
    num_nodes = max_node + 1

    # Assemble global matrices
    global_K = assemble_global_conductivity_matrix(elements, num_nodes)
    load_vector = create_load_vector(elements, num_nodes, heat_generation, point_loads)

    # Apply boundary conditions
    K_modified, F_modified = apply_temperature_boundary_conditions(global_K, load_vector, temp_BC)

    # Solve system
    temperatures = solve_fem_system(K_modified, F_modified)

    return temperatures


def validate_input_data(elements, temp_BC):
    """
    Validate input data for consistency.

    Args:
        elements: List of element dictionaries
        temp_BC: Dictionary of temperature boundary conditions

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not elements:
        raise ValueError("No elements provided")

    # Check element data
    for i, elem in enumerate(elements):
        if elem['k'] <= 0:
            raise ValueError(f"Element {i}: thermal conductivity must be positive")
        if elem['length'] <= 0:
            raise ValueError(f"Element {i}: length must be positive")
        if elem['start_node'] < 0 or elem['end_node'] < 0:
            raise ValueError(f"Element {i}: node numbers must be non-negative")
        if elem['start_node'] == elem['end_node']:
            raise ValueError(f"Element {i}: start and end nodes must be different")

    # Check if we have at least one boundary condition
    if not temp_BC:
        raise ValueError("At least one temperature boundary condition is required")

    return True


def print_results(temperatures, elements=None):
    """
    Print the results in a formatted way.

    Args:
        temperatures: Array of nodal temperatures
        elements: List of elements (optional, for additional output)
    """
    print("\n" + "=" * 50)
    print("FEM HEAT CONDUCTION RESULTS")
    print("=" * 50)
    print("\nNodal Temperatures:")
    print("-" * 30)
    for i, temp in enumerate(temperatures):
        print(f"Node {i:2d}: {temp:10.6f} K")

    if elements:
        print(f"\nNumber of elements: {len(elements)}")
        print(f"Number of nodes: {len(temperatures)}")


def get_input_data():
    """
    Get input data from user with validation.

    Returns:
        Tuple of (elements, temp_BC, heat_generation, point_loads)
    """
    try:
        # Get number of elements
        num_elements = int(input("Enter the number of elements: "))
        if num_elements <= 0:
            raise ValueError("Number of elements must be positive")

        # Get element properties
        elements = []
        print("\nEnter element properties:")
        for i in range(num_elements):
            print(f"\nElement {i}:")
            k = float(input(f"  Thermal conductivity: "))
            length = float(input(f"  Length: "))
            start_node = int(input(f"  Start node: "))
            end_node = int(input(f"  End node: "))

            elements.append({
                'k': k,
                'length': length,
                'start_node': start_node,
                'end_node': end_node
            })

        # Get temperature boundary conditions
        temp_BC = {}
        num_temp_BC = int(input("\nEnter the number of temperature boundary conditions: "))
        if num_temp_BC <= 0:
            raise ValueError("At least one temperature boundary condition is required")

        print("Enter temperature boundary conditions:")
        for i in range(num_temp_BC):
            node = int(input(f"  Node {i}: "))
            temp = float(input(f"  Temperature: "))
            temp_BC[node] = temp

        # Get heat generation (optional)
        heat_generation = None
        has_heat_gen = input("\nIs there uniform heat generation? (y/n): ").lower()
        if has_heat_gen == 'y':
            heat_generation = float(input("Enter heat generation per unit volume: "))

        # Get point loads (optional)
        point_loads = None
        has_point_loads = input("Are there any point loads? (y/n): ").lower()
        if has_point_loads == 'y':
            point_loads = {}
            num_point_loads = int(input("Enter number of point loads: "))
            for i in range(num_point_loads):
                node = int(input(f"  Point load {i} - Node: "))
                load = float(input(f"  Point load {i} - Load value: "))
                point_loads[node] = load

        return elements, temp_BC, heat_generation, point_loads

    except ValueError as e:
        print(f"Input error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(1)


def main():
    """Main function to run the FEM heat conduction solver."""
    print("FEM Heat Conduction Solver")
    print("=" * 50)

    try:
        # Get input data
        elements, temp_BC, heat_generation, point_loads = get_input_data()

        # Validate input data
        validate_input_data(elements, temp_BC)

        # Solve FEM problem
        temperatures = solve_fem_heat_conduction(elements, temp_BC, heat_generation, point_loads)

        # Print results
        print_results(temperatures, elements)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()