#!/usr/bin/env python3

import numpy as np
import sys


def compute_element_stiffness_matrix(E, A, L):
    """
    Compute the stiffness matrix for a 1D bar element.

    Args:
        E: Young's modulus (Pa)
        A: Cross-sectional area (m²)
        L: Element length (m)

    Returns:
        2x2 numpy array representing element stiffness matrix
    """
    if E <= 0:
        raise ValueError("Young's modulus must be positive")
    if A <= 0:
        raise ValueError("Cross-sectional area must be positive")
    if L <= 0:
        raise ValueError("Element length must be positive")

    k = E * A / L
    return np.array([[k, -k],
                     [-k, k]])


def compute_element_length(node_coords, node_i, node_j):
    """
    Compute the length of an element from node coordinates.

    Args:
        node_coords: List of node coordinates
        node_i: Start node index
        node_j: End node index

    Returns:
        Element length
    """
    return abs(node_coords[node_j] - node_coords[node_i])


def assemble_global_stiffness_matrix(elements, num_nodes):
    """
    Assemble the global stiffness matrix from individual element stiffness matrices.

    Args:
        elements: List of element dictionaries
        num_nodes: Total number of nodes

    Returns:
        Global stiffness matrix
    """
    global_K = np.zeros((num_nodes, num_nodes))

    for elem_id, element in enumerate(elements):
        try:
            # Extract element properties
            E = element['E']
            A = element['A']
            L = element['L']
            node_i = element['node_i']
            node_j = element['node_j']

            # Validate node indices
            if node_i < 0 or node_i >= num_nodes:
                raise ValueError(f"Element {elem_id}: Invalid start node {node_i}")
            if node_j < 0 or node_j >= num_nodes:
                raise ValueError(f"Element {elem_id}: Invalid end node {node_j}")
            if node_i == node_j:
                raise ValueError(f"Element {elem_id}: Start and end nodes must be different")

            # Compute element stiffness matrix
            element_stiffness = compute_element_stiffness_matrix(E, A, L)

            # Add element stiffness to global stiffness matrix
            global_K[node_i, node_i] += element_stiffness[0, 0]
            global_K[node_i, node_j] += element_stiffness[0, 1]
            global_K[node_j, node_i] += element_stiffness[1, 0]
            global_K[node_j, node_j] += element_stiffness[1, 1]

        except KeyError as e:
            raise ValueError(f"Element {elem_id}: Missing property {e}")

    return global_K


def solve_static_analysis(global_K, forces, boundary_conditions):
    """
    Solve the static equilibrium equation K*u = F for displacements.

    Args:
        global_K: Global stiffness matrix
        forces: Force vector
        boundary_conditions: Dictionary of fixed displacements {node: displacement}

    Returns:
        Displacement vector
    """
    num_dofs = len(forces)
    K_modified = global_K.copy()
    F_modified = forces.copy()

    # Apply boundary conditions (fixed displacements)
    for node, displacement in boundary_conditions.items():
        if 0 <= node < num_dofs:
            # Set the row to zero except diagonal
            K_modified[node, :] = 0.0
            K_modified[node, node] = 1.0
            F_modified[node] = displacement

    try:
        # Check if matrix is singular
        if np.linalg.det(K_modified) == 0:
            raise ValueError("Singular stiffness matrix - check boundary conditions")

        displacements = np.linalg.solve(K_modified, F_modified)
        return displacements

    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to solve system: {e}")


def compute_element_forces(elements, displacements):
    """
    Compute internal forces in each element.

    Args:
        elements: List of element dictionaries
        displacements: Nodal displacement vector

    Returns:
        List of element forces
    """
    element_forces = []

    for elem_id, element in enumerate(elements):
        E = element['E']
        A = element['A']
        L = element['L']
        node_i = element['node_i']
        node_j = element['node_j']

        # Element displacement vector
        u_i = displacements[node_i]
        u_j = displacements[node_j]

        # Axial force in element (tension positive)
        force = (E * A / L) * (u_j - u_i)
        element_forces.append(force)

    return element_forces


def validate_input_data(elements, node_coords):
    """
    Validate input data for consistency.

    Args:
        elements: List of element dictionaries
        node_coords: List of node coordinates

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not elements:
        raise ValueError("No elements provided")

    if not node_coords:
        raise ValueError("No node coordinates provided")

    # Check if node coordinates are sorted (for 1D problems)
    if len(node_coords) > 1:
        for i in range(1, len(node_coords)):
            if node_coords[i] <= node_coords[i - 1]:
                print("Warning: Node coordinates are not in ascending order")
                break

    # Check element data
    for i, elem in enumerate(elements):
        required_keys = ['E', 'A', 'L', 'node_i', 'node_j']
        for key in required_keys:
            if key not in elem:
                raise ValueError(f"Element {i}: Missing property '{key}'")

    return True


def print_stiffness_matrix(global_K, title="Global Stiffness Matrix"):
    """
    Print the stiffness matrix in a formatted way.

    Args:
        global_K: Global stiffness matrix
        title: Title for the matrix
    """
    print(f"\n{title}:")
    print("-" * (len(title) + 1))

    # Print matrix with proper formatting
    num_nodes = global_K.shape[0]

    # Print header
    print("Node", end="")
    for j in range(num_nodes):
        print(f"{j:12d}", end="")
    print()

    # Print matrix rows
    for i in range(num_nodes):
        print(f"{i:4d}", end="")
        for j in range(num_nodes):
            print(f"{global_K[i, j]:12.3e}", end="")
        print()


def print_detailed_results(global_K, elements, node_coords, displacements=None, forces=None, element_forces=None):
    """
    Print detailed analysis results.

    Args:
        global_K: Global stiffness matrix
        elements: List of elements
        node_coords: Node coordinates
        displacements: Nodal displacements (optional)
        forces: Applied forces (optional)
        element_forces: Element forces (optional)
    """
    print("\n" + "=" * 60)
    print("FEM STIFFNESS MATRIX ANALYSIS RESULTS")
    print("=" * 60)

    # Print model information
    print(f"\nModel Information:")
    print(f"Number of nodes: {len(node_coords)}")
    print(f"Number of elements: {len(elements)}")

    # Print node coordinates
    print(f"\nNode Coordinates:")
    print("-" * 20)
    for i, coord in enumerate(node_coords):
        print(f"Node {i}: {coord:10.6f}")

    # Print element information
    print(f"\nElement Properties:")
    print("-" * 50)
    print("Elem   E(Pa)      A(m²)     L(m)    Nodes")
    for i, elem in enumerate(elements):
        print(f"{i:4d}  {elem['E']:8.1e}  {elem['A']:8.1e}  {elem['L']:6.3f}  {elem['node_i']}-{elem['node_j']}")

    # Print stiffness matrix
    print_stiffness_matrix(global_K)

    # Print displacements if provided
    if displacements is not None:
        print(f"\nNodal Displacements:")
        print("-" * 25)
        for i, disp in enumerate(displacements):
            print(f"Node {i}: {disp:12.6e} m")

    # Print applied forces if provided
    if forces is not None:
        print(f"\nApplied Forces:")
        print("-" * 20)
        for i, force in enumerate(forces):
            if abs(force) > 1e-10:  # Only print non-zero forces
                print(f"Node {i}: {force:12.6e} N")

    # Print element forces if provided
    if element_forces is not None:
        print(f"\nElement Forces (Tension +ve):")
        print("-" * 30)
        for i, force in enumerate(element_forces):
            print(f"Element {i}: {force:12.6e} N")


def get_input_data():
    """
    Get input data from user with validation.

    Returns:
        Tuple of (elements, node_coords)
    """
    try:
        # Get number of nodes
        num_nodes = int(input("Enter the number of nodes: "))
        if num_nodes <= 0:
            raise ValueError("Number of nodes must be positive")

        # Get node coordinates
        node_coords = []
        print(f"\nEnter node coordinates:")
        for i in range(num_nodes):
            coord = float(input(f"Node {i} position: "))
            node_coords.append(coord)

        # Get number of elements
        num_elements = int(input(f"\nEnter the number of elements: "))
        if num_elements <= 0:
            raise ValueError("Number of elements must be positive")

        # Get element properties
        elements = []
        print(f"\nEnter element properties:")
        for i in range(num_elements):
            print(f"\nElement {i}:")
            E = float(input(f"  Young's modulus (Pa): "))
            A = float(input(f"  Cross-sectional area (m²): "))
            node_i = int(input(f"  Start node: "))
            node_j = int(input(f"  End node: "))

            # Calculate length from node coordinates
            if 0 <= node_i < num_nodes and 0 <= node_j < num_nodes:
                L = compute_element_length(node_coords, node_i, node_j)
                print(f"  Calculated length: {L:.6f} m")
            else:
                raise ValueError(f"Invalid node indices for element {i}")

            elements.append({
                'E': E,
                'A': A,
                'L': L,
                'node_i': node_i,
                'node_j': node_j
            })

        return elements, node_coords

    except ValueError as e:
        print(f"Input error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(1)


def get_analysis_options():
    """
    Get additional analysis options from user.

    Returns:
        Dictionary containing analysis options
    """
    options = {'solve_displacements': False, 'forces': None, 'boundary_conditions': None}

    # Ask if user wants to solve for displacements
    solve_option = input("\nDo you want to solve for displacements? (y/n): ").lower()
    if solve_option == 'y':
        options['solve_displacements'] = True

        # Get applied forces
        num_nodes = int(input("Enter number of nodes with applied forces: "))
        forces_dict = {}
        for i in range(num_nodes):
            node = int(input(f"  Node {i}: "))
            force = float(input(f"  Force value (N): "))
            forces_dict[node] = force

        # Get boundary conditions
        num_bc = int(input("Enter number of fixed displacement boundary conditions: "))
        bc_dict = {}
        for i in range(num_bc):
            node = int(input(f"  Fixed node {i}: "))
            displacement = float(input(f"  Displacement value (m): "))
            bc_dict[node] = displacement

        options['forces'] = forces_dict
        options['boundary_conditions'] = bc_dict

    return options


def main():
    """Main function to run the FEM stiffness matrix analysis."""
    print("FEM Stiffness Matrix Analysis")
    print("=" * 50)

    try:
        # Get input data
        elements, node_coords = get_input_data()

        # Validate input data
        validate_input_data(elements, node_coords)

        # Assemble global stiffness matrix
        num_nodes = len(node_coords)
        global_K = assemble_global_stiffness_matrix(elements, num_nodes)

        # Get analysis options
        options = get_analysis_options()

        # Solve for displacements if requested
        displacements = None
        element_forces = None
        applied_forces = None

        if options['solve_displacements']:
            # Create force vector
            applied_forces = np.zeros(num_nodes)
            if options['forces']:
                for node, force in options['forces'].items():
                    if 0 <= node < num_nodes:
                        applied_forces[node] = force

            # Solve for displacements
            displacements = solve_static_analysis(global_K, applied_forces, options['boundary_conditions'])

            # Compute element forces
            element_forces = compute_element_forces(elements, displacements)

        # Print results
        print_detailed_results(global_K, elements, node_coords, displacements, applied_forces, element_forces)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()