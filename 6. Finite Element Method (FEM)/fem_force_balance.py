#!/usr/bin/env python3

import numpy as np


def validate_element_data(elements):
    """
    Validate element data for consistency and correctness
    Returns: (is_valid, error_message)
    """
    if len(elements) == 0:
        return False, "No elements defined"

    # Check each element
    for i, element in enumerate(elements):
        if element['E'] <= 0:
            return False, f"Element {i}: Young's modulus must be positive"
        if element['A'] <= 0:
            return False, f"Element {i}: Cross-sectional area must be positive"
        if element['L'] <= 0:
            return False, f"Element {i}: Length must be positive"
        if element['nodes'][0] == element['nodes'][1]:
            return False, f"Element {i}: Start and end nodes must be different"
        if element['nodes'][0] < 0 or element['nodes'][1] < 0:
            return False, f"Element {i}: Node numbers must be non-negative"

    return True, ""


def validate_force_data(force_vector, max_node):
    """
    Validate force vector data
    Returns: (is_valid, error_message)
    """
    if len(force_vector) != max_node + 1:
        return False, f"Force vector size mismatch: expected {max_node + 1}, got {len(force_vector)}"

    return True, ""


def validate_boundary_conditions(fixed_nodes, max_node):
    """
    Validate boundary conditions
    Returns: (is_valid, error_message)
    """
    if len(fixed_nodes) == 0:
        return False, "At least one node must be fixed for structural stability"

    # Check for valid node numbers
    for node in fixed_nodes:
        if node < 0 or node > max_node:
            return False, f"Fixed node {node} is out of range [0, {max_node}]"

    # Check for duplicates
    if len(fixed_nodes) != len(set(fixed_nodes)):
        return False, "Duplicate nodes found in fixed nodes list"

    return True, ""


def compute_element_stiffness_matrix(E, A, L):
    """
    Compute the local stiffness matrix for a 1D bar element
    Args:
        E: Young's modulus
        A: Cross-sectional area
        L: Element length
    Returns:
        2x2 element stiffness matrix
    """
    k = E * A / L
    element_stiffness = np.array([[k, -k], [-k, k]], dtype=np.float64)
    return element_stiffness


def assemble_global_stiffness_matrix(elements):
    """
    Assemble the global stiffness matrix from element stiffness matrices
    Args:
        elements: List of element dictionaries
    Returns:
        Global stiffness matrix
    """
    # Find maximum node number to determine matrix size
    max_node = 0
    for element in elements:
        for node in element['nodes']:
            if node > max_node:
                max_node = node

    num_nodes = max_node + 1
    global_stiffness = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    # Loop through each element
    for element in elements:
        E = element['E']
        A = element['A']
        L = element['L']
        nodes = element['nodes']

        # Compute element stiffness matrix
        element_stiffness = compute_element_stiffness_matrix(E, A, L)

        # Assemble into global matrix
        for i in range(2):
            for j in range(2):
                global_node_i = nodes[i]
                global_node_j = nodes[j]
                global_stiffness[global_node_i, global_node_j] += element_stiffness[i, j]

    return global_stiffness


def apply_displacement_boundary_conditions(global_stiffness, force_vector, fixed_nodes):
    """
    Apply displacement boundary conditions using penalty method
    Args:
        global_stiffness: Global stiffness matrix
        force_vector: Global force vector
        fixed_nodes: List of nodes with zero displacement
    Returns:
        Modified stiffness matrix and force vector
    """
    # Create copies to avoid modifying original arrays
    K_modified = global_stiffness.copy()
    F_modified = force_vector.copy()

    # Apply boundary conditions
    for node in fixed_nodes:
        # Zero out the row and column, set diagonal to 1
        K_modified[node, :] = 0.0
        K_modified[:, node] = 0.0
        K_modified[node, node] = 1.0
        F_modified[node] = 0.0

    return K_modified, F_modified


def solve_displacement_system(global_stiffness, force_vector, fixed_nodes):
    """
    Solve the FEM system for nodal displacements
    Args:
        global_stiffness: Global stiffness matrix
        force_vector: Global force vector
        fixed_nodes: List of fixed nodes
    Returns:
        Displacement vector
    """
    # Apply boundary conditions
    K_modified, F_modified = apply_displacement_boundary_conditions(
        global_stiffness, force_vector, fixed_nodes)

    # Solve the system
    try:
        displacements = np.linalg.solve(K_modified, F_modified)
    except np.linalg.LinAlgError:
        raise ValueError("System is singular - check boundary conditions and element connectivity")

    return displacements


def compute_element_forces(elements, displacements):
    """
    Compute internal forces in each element
    Args:
        elements: List of element dictionaries
        displacements: Nodal displacement vector
    Returns:
        Element internal forces
    """
    element_forces = []

    for i, element in enumerate(elements):
        E = element['E']
        A = element['A']
        L = element['L']
        nodes = element['nodes']

        # Get nodal displacements for this element
        u1 = displacements[nodes[0]]
        u2 = displacements[nodes[1]]

        # Compute element internal force
        strain = (u2 - u1) / L
        stress = E * strain
        internal_force = stress * A

        element_forces.append({
            'element': i,
            'nodes': nodes,
            'strain': strain,
            'stress': stress,
            'internal_force': internal_force
        })

    return element_forces


def compute_reaction_forces(original_global_stiffness, displacements, fixed_nodes):
    """
    Compute reaction forces at fixed nodes
    Args:
        original_global_stiffness: Original (unmodified) global stiffness matrix
        displacements: Computed nodal displacements
        fixed_nodes: List of fixed nodes
    Returns:
        Reaction forces at fixed nodes
    """
    # Compute all nodal forces: F = K * u
    all_forces = np.dot(original_global_stiffness, displacements)

    # Extract reactions at fixed nodes
    reactions = []
    for node in fixed_nodes:
        reactions.append(all_forces[node])

    return np.array(reactions)


def check_force_equilibrium(applied_forces, reaction_forces, tolerance=1e-10):
    """
    Check global force equilibrium
    Args:
        applied_forces: Applied force vector
        reaction_forces: Reaction force vector
        tolerance: Numerical tolerance
    Returns:
        (is_balanced, force_sum, percentage_error)
    """
    total_applied = np.sum(applied_forces)
    total_reaction = np.sum(reaction_forces)
    force_sum = total_applied + total_reaction

    # Calculate percentage error
    if abs(total_applied) > tolerance:
        percentage_error = abs(force_sum / total_applied) * 100
    else:
        percentage_error = 0.0

    is_balanced = abs(force_sum) < tolerance

    return is_balanced, force_sum, percentage_error


def read_element_data():
    """
    Read element data from user input
    Returns: (elements, success)
    """
    try:
        print("=== FEM Force Balance Analysis ===\n")

        num_elements = int(input("Enter the number of elements: "))
        if num_elements <= 0:
            print("Error: Number of elements must be positive")
            return [], False

        elements = []
        print(f"\nEnter data for {num_elements} elements:")

        for i in range(num_elements):
            print(f"\nElement {i + 1}:")
            E = float(input("  Young's Modulus (E) [Pa]: "))
            A = float(input("  Cross-sectional Area (A) [m²]: "))
            L = float(input("  Length (L) [m]: "))
            node_i = int(input("  Start Node: "))
            node_j = int(input("  End Node: "))

            elements.append({
                'E': E,
                'A': A,
                'L': L,
                'nodes': [node_i, node_j]
            })

        return elements, True

    except ValueError as e:
        print(f"Input Error: {e}")
        return [], False


def read_force_data(max_node):
    """
    Read force data from user input
    Returns: (force_vector, success)
    """
    try:
        num_nodes = max_node + 1
        force_vector = np.zeros(num_nodes, dtype=np.float64)

        print(f"\nEnter Applied Forces (for {num_nodes} nodes, 0 to {max_node}):")
        num_forces = int(input("Number of applied forces: "))

        for i in range(num_forces):
            print(f"Force {i + 1}:")
            node = int(input("  Node number: "))
            force = float(input("  Force value [N]: "))

            if node < 0 or node > max_node:
                print(f"Warning: Node {node} is out of range")
                continue

            force_vector[node] += force  # Allow multiple forces on same node

        return force_vector, True

    except ValueError as e:
        print(f"Input Error: {e}")
        return np.array([]), False


def read_boundary_conditions(max_node):
    """
    Read boundary conditions from user input
    Returns: (fixed_nodes, success)
    """
    try:
        print(f"\nEnter Boundary Conditions (nodes 0 to {max_node}):")
        fixed_input = input("Fixed nodes (space-separated): ").strip()

        if not fixed_input:
            print("Error: At least one node must be fixed")
            return [], False

        fixed_nodes = list(map(int, fixed_input.split()))
        return fixed_nodes, True

    except ValueError as e:
        print(f"Input Error: {e}")
        return [], False


def display_comprehensive_results(displacements, reactions, fixed_nodes,
                                  element_forces, applied_forces, equilibrium_check):
    """
    Display comprehensive analysis results
    """
    print("\n" + "=" * 70)
    print("FEM FORCE BALANCE ANALYSIS RESULTS")
    print("=" * 70)

    # Nodal displacements
    print("\nNODAL DISPLACEMENTS:")
    print("-" * 40)
    for i, disp in enumerate(displacements):
        print(f"Node {i:2d}: {disp:12.6e} m")

    # Reaction forces
    print("\nREACTION FORCES:")
    print("-" * 40)
    total_reaction = 0
    for i, reaction in enumerate(reactions):
        print(f"Node {fixed_nodes[i]:2d}: {reaction:12.6e} N")
        total_reaction += reaction

    # Element forces
    print("\nELEMENT INTERNAL FORCES:")
    print("-" * 60)
    for elem_force in element_forces:
        elem_num = elem_force['element']
        nodes = elem_force['nodes']
        force = elem_force['internal_force']
        stress = elem_force['stress']
        strain = elem_force['strain']

        print(f"Element {elem_num + 1:2d} (Nodes {nodes[0]}-{nodes[1]}):")
        print(f"  Internal Force: {force:12.6e} N")
        print(f"  Stress:         {stress:12.6e} Pa")
        print(f"  Strain:         {strain:12.6e}")

    # Force equilibrium check
    print("\nFORCE EQUILIBRIUM CHECK:")
    print("-" * 40)
    total_applied = np.sum(applied_forces)
    is_balanced, force_sum, percentage_error = equilibrium_check

    print(f"Total Applied Forces:  {total_applied:12.6e} N")
    print(f"Total Reaction Forces: {total_reaction:12.6e} N")
    print(f"Force Sum (should be 0): {force_sum:12.6e} N")
    print(f"Percentage Error:      {percentage_error:8.4f} %")

    if is_balanced:
        print("✓ EQUILIBRIUM SATISFIED")
    else:
        print("✗ EQUILIBRIUM NOT SATISFIED - Check input data")

    # Summary
    print("\nSUMMARY:")
    print("-" * 40)
    print(f"Maximum Displacement: {np.max(np.abs(displacements)):12.6e} m")
    print(f"Maximum Internal Force: {np.max([abs(ef['internal_force']) for ef in element_forces]):12.6e} N")
    print(f"Maximum Stress:       {np.max([abs(ef['stress']) for ef in element_forces]):12.6e} Pa")


def main():
    """
    Main function to orchestrate the FEM force balance analysis
    """
    # Step 1: Read element data
    elements, success = read_element_data()
    if not success:
        return

    # Validate element data
    is_valid, error_msg = validate_element_data(elements)
    if not is_valid:
        print(f"Element Validation Error: {error_msg}")
        return

    # Find maximum node number
    max_node = 0
    for element in elements:
        for node in element['nodes']:
            if node > max_node:
                max_node = node

    # Step 2: Read force data
    force_vector, success = read_force_data(max_node)
    if not success:
        return

    # Step 3: Read boundary conditions
    fixed_nodes, success = read_boundary_conditions(max_node)
    if not success:
        return

    # Validate boundary conditions
    is_valid, error_msg = validate_boundary_conditions(fixed_nodes, max_node)
    if not is_valid:
        print(f"Boundary Condition Error: {error_msg}")
        return

    try:
        # Step 4: Assemble global stiffness matrix
        print("\nStep 1: Assembling global stiffness matrix...")
        global_stiffness = assemble_global_stiffness_matrix(elements)

        # Step 5: Solve for displacements
        print("Step 2: Solving for nodal displacements...")
        displacements = solve_displacement_system(global_stiffness, force_vector, fixed_nodes)

        # Step 6: Compute reaction forces
        print("Step 3: Computing reaction forces...")
        reactions = compute_reaction_forces(global_stiffness, displacements, fixed_nodes)

        # Step 7: Compute element forces
        print("Step 4: Computing element internal forces...")
        element_forces = compute_element_forces(elements, displacements)

        # Step 8: Check force equilibrium
        print("Step 5: Checking force equilibrium...")
        equilibrium_check = check_force_equilibrium(force_vector, reactions)

        # Step 9: Display results
        display_comprehensive_results(displacements, reactions, fixed_nodes,
                                      element_forces, force_vector, equilibrium_check)

    except Exception as e:
        print(f"Analysis Error: {e}")


if __name__ == "__main__":
    main()