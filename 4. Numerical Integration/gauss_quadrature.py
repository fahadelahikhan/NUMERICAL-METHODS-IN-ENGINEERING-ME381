#!/usr/bin/env python3

import numpy as np
from scipy.special import roots_legendre, roots_laguerre, roots_hermite

def gaussian_quadrature(func, a, b, n, quad_type='legendre'):
    if quad_type == 'legendre':
        x, w = roots_legendre(n)
        # Transform interval from [-1, 1] to [a, b]
        t = 0.5 * (x + 1) * (b - a) + a
        integral = 0.5 * (b - a) * np.sum(w * func(t))
    elif quad_type == 'laguerre':
        x, w = roots_laguerre(n)
        integral = np.sum(w * func(x))
    elif quad_type == 'hermite':
        x, w = roots_hermite(n)
        integral = np.sum(w * func(x))
    else:
        raise ValueError("Unsupported quadrature type. Choose from 'legendre', 'laguerre', or 'hermite'.")
    return integral

def main():
    try:
        # Function input
        func_expr = input("Enter the function expression (use 'x' as variable): ")
        from sympy import symbols, lambdify
        x = symbols('x')
        func = lambdify(x, func_expr, 'numpy')

        # Interval
        a = float(input("Enter the start of the interval (a): "))
        b = float(input("Enter the end of the interval (b): "))
        if a >= b:
            print("Error: Interval start must be less than end.")
            return

        # Quadrature parameters
        n = int(input("Enter the order of Gaussian quadrature: "))
        if n <= 0:
            print("Error: Order must be a positive integer.")
            return

        quad_type = input("Enter the quadrature type (legendre/laguerre/hermite): ").lower()
        if quad_type not in ['legendre', 'laguerre', 'hermite']:
            print("Error: Unsupported quadrature type. Choose from 'legendre', 'laguerre', or 'hermite'.")
            return

        # Compute integral
        result = gaussian_quadrature(func, a, b, n, quad_type)

        # Output result
        print(f"\nThe approximate integral using {quad_type} Gaussian quadrature of order {n} is {result:.6f}")

    except Exception as e:
        print(f"Invalid input or computation error: {e}")

if __name__ == "__main__":
    main()