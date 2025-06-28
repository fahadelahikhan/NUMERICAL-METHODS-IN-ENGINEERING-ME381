#!/usr/bin/env python3

import numpy as np
from sympy import symbols, diff, integrate, lambdify


def calculate_strain_energy(deflection, x, E, I, L):
    """Calculate the strain energy for the beam."""
    d2y_dx2 = diff(deflection, x, 2)
    integrand = 0.5 * E * I * d2y_dx2 ** 2
    strain_energy = integrate(integrand, (x, 0, L))
    return strain_energy


def calculate_potential_energy(deflection, x, q, L):
    """Calculate the potential energy due to the distributed load."""
    integrand = 0.5 * q * deflection ** 2
    potential_energy = integrate(integrand, (x, 0, L))
    return potential_energy


def assemble_system(strain_energy, potential_energy, params):
    """Assemble the system of equations by differentiating the total potential energy."""
    total_potential = strain_energy + potential_energy
    n = len(params)
    equations = []
    for param in params:
        equations.append(diff(total_potential, param))
    return equations


def solve_coefficients(equations, params):
    """Solve the system of equations for the coefficients."""
    from sympy import linsolve
    solution = linsolve(equations, params)
    return solution


def main():
    try:
        # Input parameters
        L = float(input("Enter the length of the beam (L): "))
        E = float(input("Enter the modulus of elasticity (E): "))
        I = float(input("Enter the moment of inertia (I): "))
        q = float(input("Enter the distributed load (q): "))

        # Define symbolic variables
        x = symbols('x')
        c1, c2 = symbols('c1 c2')
        params = (c1, c2)

        # Define admissible function (example: quadratic function for simply supported beam)
        deflection = c1 * x + c2 * x ** 2

        # Apply boundary conditions (example: simply supported beam with y(0)=0 and y(L)=0)
        deflection = deflection.subs(c1, 0)  # y(0) = 0
        deflection = deflection.subs(x, L).subs(c2, 0)  # y(L) = 0
        deflection = c1 * x + c2 * x ** 2 + c3 * x ** 3
        # Re-apply boundary conditions after modification
        bc1 = deflection.subs(x, 0)  # y(0) = 0
        bc2 = deflection.subs(x, L)  # y(L) = 0
        c3 = symbols('c3')
        params = (c1, c2, c3)
        solution_bc = solve_coefficients([bc1, bc2], params[:2])
        c1_val = list(solution_bc)[0][0]
        c2_val = list(solution_bc)[0][1]
        deflection = deflection.subs(c1, c1_val).subs(c2, c2_val)
        params = (c3,)

        # Calculate strain energy and potential energy
        strain_energy = calculate_strain_energy(deflection, x, E, I, L)
        potential_energy = calculate_potential_energy(deflection, x, q, L)

        # Assemble and solve the system
        equations = assemble_system(strain_energy, potential_energy, params)
        solution = solve_coefficients(equations, params)

        # Output results
        print("\nSolution for coefficients:")
        for param, value in zip(params, list(solution)[0]):
            print(f"{param} = {value}")

        # Convert deflection to a numerical function
        deflection_func = lambdify(x, deflection.subs(c3, list(solution)[0][0]), 'numpy')
        x_vals = np.linspace(0, L, 100)
        y_vals = deflection_func(x_vals)

        # Print deflection at several points
        print("\nDeflection at several points:")
        for xi in np.linspace(0, L, 5):
            print(f"x = {xi:.2f}: y = {deflection_func(xi):.6f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()