#!/usr/bin/env python3

import numpy as np


def validate_input_parameters(E, A, L, num_elements, force_vector, fixed_nodes):
    """
    Validate all input parameters for FEM analysis
    Returns: (is_valid, error_message)
    """
    # Check material properties
    if E <= 0:
        return False, "Modulus of elasticity (E) must be positive"
    if A <= 0:
        return False, "Cross-sectional area (A) must be positive"
    if L <= 0:
        return False, "Length (L) must be positive"

    # Check mesh parameters
    if num_elements <= 0:
        return False, "Number of elements must be positive"

    # Check force vector
    expected_nodes = num_elements + 1
    if len(force_vector) != expected_nodes:
        return False, f"Force vector length must be {expected_nodes} (num_elements + 1)"

    # Check fixed nodes
    if len(fixed_nodes) == 0:
        return False, "At least one node must be fixed (boundary condition required)"

    for node in fixed_nodes:
        if node < 0 or node >= expected_nodes:
            return False, f"Fixed node {node} is out of range [0, {expected_nodes - 1}]"

    # Check for duplicate fixed nodes
    if len(fixed_nodes) != len(set(fixed_nodes)):
        return False, "Duplicate nodes found in fixed nodes list"

    return True, ""


def compute_element_stiffness_matrix(E, A, L_element):
    """
    Compute the local stiffness matrix for a 1D bar element
    Args:
        E: Young's modulus
        A: Cross-sectional area
        L_element: Length of the element
    Returns:
        2x2 element stiffness matrix
    """
    k = E * A / L_element
    element_stiffness = np.array([[k, -k], [-k, k]], dtype=np.float64)
    return element_stiffness


def assemble_global_stiffness_matrix(E, A, L, num_elements):
    """
    Assemble the global stiffness matrix from element matrices
    Args:
        E: Young's modulus
        A: Cross-sectional area
        L: Total length of the bar
        num_elements: Number of elements
    Returns:
        Global stiffness matrix
    """
    num_nodes = num_elements + 1
    global_stiffness = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    # Element length (assuming uniform mesh)
    L_element = L / num_elements

    # Loop through each element
    for elem in range(num_elements):
        # Get element stiffness matrix
        element_stiffness = compute_element_stiffness_matrix(E, A, L_element)

        # Global node numbers for this element
        node1 = elem
        node2 = elem + 1

        # Assemble into global matrix
        global_stiffness[node1, node1] += element_stiffness[0, 0]
        global_stiffness[node1, node2] += element_stiffness[0, 1]
        global_stiffness[node2, node1] += element_stiffness[1, 0]
        global_stiffness[node2, node2] += element_stiffness[1, 1]

    return global_stiffness


def apply_essential_boundary_conditions(global_stiffness, force_vector, fixed_nodes):
    """
    Apply essential (displacement) boundary conditions using elimination method
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

    # Apply boundary conditions (set displacement = 0 at fixed nodes)
    for node in fixed_nodes:
        # Set row to zero except diagonal
        K_modified[node, :] = 0.0
        K_modified[node, node] = 1.0
        F_modified[node] = 0.0

        # Set column to zero to maintain symmetry
        K_modified[:, node] = 0.0
        K_modified[node, node] = 1.0

    return K_modified, F_modified


def solve_displacement_system(global_stiffness, force_vector, fixed_nodes):
    """
    Solve the FEM system for nodal displacements
    Args:
        global_stiffness: Global stiffness matrix
        force_vector: Global force vector
        fixed_nodes: List of fixed nodes
    Returns:
        Nodal displacement vector
    """
    # Apply boundary conditions
    K_modified, F_modified = apply_essential_boundary_conditions(
        global_stiffness, force_vector, fixed_nodes)

    # Solve the system K*u = F
    try:
        displacements = np.linalg.solve(K_modified, F_modified)
    except np.linalg.LinAlgError:
        raise ValueError("System is singular - check boundary conditions")

    return displacements


def compute_element_stress_strain(E, A, L, num_elements, displacements):
    """
    Compute stress and strain in each element
    Args:
        E: Young's modulus
        A: Cross-sectional area
        L: Total length
        num_elements: Number of elements
        displacements: Nodal displacements
    Returns:
        Element stresses and strains
    """
    L_element = L / num_elements
    stresses = np.zeros(num_elements)
    strains = np.zeros(num_elements)

    for elem in range(num_elements):
        # Nodes for this element
        node1 = elem
        node2 = elem + 1

        # Calculate strain (du/dx)
        strain = (displacements[node2] - displacements[node1]) / L_element

        # Calculate stress (σ = E * ε)
        stress = E * strain

        strains[elem] = strain
        stresses[elem] = stress

    return stresses, strains


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
    # Calculate all forces: F = K * u
    all_forces = np.dot(original_global_stiffness, displacements)

    # Extract reactions at fixed nodes
    reactions = np.zeros(len(fixed_nodes))
    for i, node in enumerate(fixed_nodes):
        reactions[i] = all_forces[node]

    return reactions


def read_problem_data():
    """
    Read all input data from user
    Returns: (E, A, L, num_elements, force_vector, fixed_nodes, success)
    """
    try:
        print("=== FEM 1D Bar Element Analysis ===\n")

        # Material properties
        print("Enter Material Properties:")
        E = float(input("Young's Modulus (E) [Pa]: "))
        A = float(input("Cross-sectional Area (A) [m²]: "))

        # Geometry
        print("\nEnter Geometry:")
        L = float(input("Total Length (L) [m]: "))
        num_elements = int(input("Number of Elements: "))

        # Loading
        print(f"\nEnter Loading (for {num_elements + 1} nodes):")
        print("Format: F0 F1 F2 ... (space-separated forces in Newtons)")
        force_input = input("Force Vector: ").strip()
        force_vector = list(map(float, force_input.split()))

        # Boundary conditions
        print("\nEnter Boundary Conditions:")
        print("Format: node_indices (space-separated, 0-based indexing)")
        fixed_input = input("Fixed Nodes (zero displacement): ").strip()
        fixed_nodes = list(map(int, fixed_input.split()))

        return E, A, L, num_elements, force_vector, fixed_nodes, True

    except ValueError as e:
        print(f"Input Error: {e}")
        return 0, 0, 0, 0, [], [], False


def display_results(displacements, reactions, fixed_nodes, stresses, strains, L, num_elements):
    """
    Display comprehensive analysis results
    """
    print("\n" + "=" * 60)
    print("FEM ANALYSIS RESULTS")
    print("=" * 60)

    # Nodal displacements
    print("\nNODAL DISPLACEMENTS:")
    print("-" * 30)
    for i, disp in enumerate(displacements):
        print(f"Node {i:2d}: {disp:12.6e} m")

    # Reaction forces
    print("\nREACTION FORCES:")
    print("-" * 30)
    total_reaction = 0
    for i, reaction in enumerate(reactions):
        print(f"Node {fixed_nodes[i]:2d}: {reaction:12.6e} N")
        total_reaction += reaction
    print(f"Total Reaction: {total_reaction:12.6e} N")

    # Element results
    print("\nELEMENT RESULTS:")
    print("-" * 50)
    L_element = L / num_elements
    for elem in range(num_elements):
        x_center = (elem + 0.5) * L_element
        print(f"Element {elem + 1:2d} (x = {x_center:6.3f} m):")
        print(f"  Stress:  {stresses[elem]:12.6e} Pa")
        print(f"  Strain:  {strains[elem]:12.6e}")

    # Summary
    print("\nSUMMARY:")
    print("-" * 30)
    print(f"Maximum Displacement: {np.max(np.abs(displacements)):12.6e} m")
    print(f"Maximum Stress:       {np.max(np.abs(stresses)):12.6e} Pa")
    print(f"Maximum Strain:       {np.max(np.abs(strains)):12.6e}")


def main():
    """
    Main function to orchestrate the FEM analysis
    """
    # Read input data
    E, A, L, num_elements, force_vector, fixed_nodes, success = read_problem_data()
    if not success:
        return

    # Validate inputs
    is_valid, error_msg = validate_input_parameters(E, A, L, num_elements, force_vector, fixed_nodes)
    if not is_valid:
        print(f"Validation Error: {error_msg}")
        return

    try:
        # Step 1: Assemble global stiffness matrix
        print("\nStep 1: Assembling global stiffness matrix...")
        global_stiffness = assemble_global_stiffness_matrix(E, A, L, num_elements)

        # Step 2: Solve for displacements
        print("Step 2: Solving for nodal displacements...")
        force_vector_np = np.array(force_vector, dtype=np.float64)
        displacements = solve_displacement_system(global_stiffness, force_vector_np, fixed_nodes)

        # Step 3: Compute reactions
        print("Step 3: Computing reaction forces...")
        reactions = compute_reaction_forces(global_stiffness, displacements, fixed_nodes)

        # Step 4: Compute element stresses and strains
        print("Step 4: Computing element stresses and strains...")
        stresses, strains = compute_element_stress_strain(E, A, L, num_elements, displacements)

        # Step 5: Display results
        display_results(displacements, reactions, fixed_nodes, stresses, strains, L, num_elements)

        # Verify equilibrium
        total_applied_force = np.sum(force_vector)
        total_reaction_force = np.sum(reactions)
        print(f"\nEQUILIBRIUM CHECK:")
        print(f"Applied Forces:  {total_applied_force:12.6e} N")
        print(f"Reaction Forces: {total_reaction_force:12.6e} N")
        print(f"Difference:      {abs(total_applied_force + total_reaction_force):12.6e} N")

    except Exception as e:
        print(f"Analysis Error: {e}")


if __name__ == "__main__":
    main()