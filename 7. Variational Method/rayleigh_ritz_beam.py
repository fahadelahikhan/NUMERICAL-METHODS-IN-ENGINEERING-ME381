#!/usr/bin/env python3

import numpy as np
from sympy import symbols, diff, integrate, lambdify, solve, Matrix, zeros


def calculate_strain_energy(deflection, x, E, I, L):
    """Calculate the strain energy for the beam."""
    d2y_dx2 = diff(deflection, x, 2)
    integrand = 0.5 * E * I * d2y_dx2 ** 2
    strain_energy = integrate(integrand, (x, 0, L))
    return strain_energy


def calculate_work_by_load(deflection, x, load_type, load_value, L, load_position=None):
    """Calculate the work done by external loads."""
    if load_type == 'distributed':
        # Work done by distributed load q
        integrand = load_value * deflection
        work = integrate(integrand, (x, 0, L))
    elif load_type == 'point':
        # Work done by point load P at position a
        if load_position is None:
            load_position = L / 2  # Default to midspan
        work = load_value * deflection.subs(x, load_position)
    else:
        raise ValueError("load_type must be 'distributed' or 'point'")
    return work


def create_admissible_function(boundary_type, n_terms, x, L):
    """Create admissible function that satisfies geometric boundary conditions."""
    if boundary_type == 'simply_supported':
        # For simply supported beam: y(0) = 0, y(L) = 0
        # Use sine series: y = sum(c_i * sin(i*pi*x/L))
        params = symbols([f'c{i}' for i in range(1, n_terms + 1)])
        deflection = sum(params[i] * np.sin((i + 1) * np.pi * x / L) for i in range(n_terms))
    elif boundary_type == 'cantilever':
        # For cantilever beam: y(0) = 0, y'(0) = 0
        # Use polynomial series: y = sum(c_i * x^(i+2))
        params = symbols([f'c{i}' for i in range(1, n_terms + 1)])
        deflection = sum(params[i] * x ** (i + 2) for i in range(n_terms))
    elif boundary_type == 'fixed_fixed':
        # For fixed-fixed beam: y(0) = 0, y'(0) = 0, y(L) = 0, y'(L) = 0
        # Use polynomial series satisfying all BCs
        params = symbols([f'c{i}' for i in range(1, n_terms + 1)])
        deflection = sum(params[i] * x ** 2 * (x - L) ** 2 * x ** (i - 1) for i in range(n_terms))
    else:
        raise ValueError("boundary_type must be 'simply_supported', 'cantilever', or 'fixed_fixed'")

    return deflection, params


def assemble_stiffness_matrix(strain_energy, params):
    """Assemble the stiffness matrix by taking second derivatives of strain energy."""
    n = len(params)
    K = zeros(n, n)

    for i in range(n):
        for j in range(n):
            K[i, j] = diff(strain_energy, params[i], params[j])

    return K


def assemble_load_vector(work_by_load, params):
    """Assemble the load vector by taking derivatives of work done by loads."""
    n = len(params)
    F = zeros(n, 1)

    for i in range(n):
        F[i] = diff(work_by_load, params[i])

    return F


def solve_system(K, F):
    """Solve the system K * c = F for coefficients c."""
    try:
        solution = K.LUsolve(F)
        return solution
    except Exception as e:
        print(f"Error solving system: {e}")
        return None


def evaluate_deflection(deflection, params, coefficients, x_vals):
    """Evaluate deflection at given x values using solved coefficients."""
    # Substitute coefficients into deflection function
    deflection_with_coeffs = deflection
    for i, param in enumerate(params):
        deflection_with_coeffs = deflection_with_coeffs.subs(param, coefficients[i])

    # Convert to numerical function
    deflection_func = lambdify(symbols('x'), deflection_with_coeffs, 'numpy')

    # Handle case where deflection might be a constant
    try:
        y_vals = deflection_func(x_vals)
    except TypeError:
        # If deflection is constant, create array of same value
        y_vals = np.full_like(x_vals, float(deflection_with_coeffs))

    return y_vals, deflection_func


def get_user_input():
    """Get input parameters from user."""
    print("=== Rayleigh-Ritz Beam Analysis ===\n")

    # Beam properties
    L = float(input("Enter the length of the beam (L): "))
    E = float(input("Enter the modulus of elasticity (E): "))
    I = float(input("Enter the moment of inertia (I): "))

    # Boundary conditions
    print("\nBoundary condition types:")
    print("1. Simply supported")
    print("2. Cantilever")
    print("3. Fixed-fixed")
    bc_choice = int(input("Select boundary condition (1-3): "))

    boundary_types = {1: 'simply_supported', 2: 'cantilever', 3: 'fixed_fixed'}
    boundary_type = boundary_types[bc_choice]

    # Number of terms in series
    n_terms = int(input("Enter number of terms in approximation series: "))

    # Load type
    print("\nLoad types:")
    print("1. Distributed load")
    print("2. Point load")
    load_choice = int(input("Select load type (1-2): "))

    if load_choice == 1:
        load_type = 'distributed'
        load_value = float(input("Enter the distributed load (q): "))
        load_position = None
    else:
        load_type = 'point'
        load_value = float(input("Enter the point load (P): "))
        load_position = float(input("Enter the position of point load (a): "))

    return L, E, I, boundary_type, n_terms, load_type, load_value, load_position


def print_results(params, coefficients, deflection_func, L):
    """Print analysis results."""
    print("\n=== RESULTS ===")
    print("\nCoefficients:")
    for i, param in enumerate(params):
        print(f"{param} = {float(coefficients[i]):.6e}")

    print("\nDeflection at key points:")
    x_points = np.linspace(0, L, 11)
    for xi in x_points:
        try:
            yi = deflection_func(xi)
            print(f"x = {xi:.2f}: y = {yi:.6e}")
        except:
            print(f"x = {xi:.2f}: y = 0.000000e+00")

    # Find maximum deflection
    x_dense = np.linspace(0, L, 1000)
    try:
        y_dense = deflection_func(x_dense)
        max_deflection = np.max(np.abs(y_dense))
        max_location = x_dense[np.argmax(np.abs(y_dense))]
        print(f"\nMaximum deflection: {max_deflection:.6e} at x = {max_location:.2f}")
    except:
        print("\nMaximum deflection: 0.000000e+00")


def main():
    """Main function to run the Rayleigh-Ritz analysis."""
    try:
        # Get input parameters
        L, E, I, boundary_type, n_terms, load_type, load_value, load_position = get_user_input()

        # Define symbolic variables
        x = symbols('x')

        # Create admissible function
        deflection, params = create_admissible_function(boundary_type, n_terms, x, L)

        # Calculate strain energy
        strain_energy = calculate_strain_energy(deflection, x, E, I, L)

        # Calculate work done by external loads
        work_by_load = calculate_work_by_load(deflection, x, load_type, load_value, L, load_position)

        # Assemble system matrices
        K = assemble_stiffness_matrix(strain_energy, params)
        F = assemble_load_vector(work_by_load, params)

        print(f"\nStiffness matrix shape: {K.shape}")
        print(f"Load vector shape: {F.shape}")

        # Solve system
        coefficients = solve_system(K, F)

        if coefficients is None:
            print("Failed to solve the system!")
            return

        # Evaluate deflection
        x_vals = np.linspace(0, L, 100)
        y_vals, deflection_func = evaluate_deflection(deflection, params, coefficients, x_vals)

        # Print results
        print_results(params, coefficients, deflection_func, L)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()