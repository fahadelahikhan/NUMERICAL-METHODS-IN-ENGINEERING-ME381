#!/usr/bin/env python3

import numpy as np

def matrix_inversion_method(A, b):
    try:
        A_inv = np.linalg.inv(A)
        x = np.dot(A_inv, b)
        return x.tolist()
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Matrix inversion failed: {e}")

def main():
    try:
        n = int(input("Enter the number of equations: "))
        if n <= 0:
            print("Number of equations must be a positive integer.")
            return

        A = []
        b = []
        print("Enter each equation's coefficients followed by the constant term, separated by spaces:")
        for _ in range(n):
            row = list(map(float, input().strip().split()))
            if len(row) != n + 1:
                print(f"Each row must have exactly {n + 1} elements (coefficients + constant).")
                return
            A.append(row[:-1])
            b.append(row[-1])

        try:
            solution = matrix_inversion_method(A, b)
            print("\nSolution:")
            for val in solution:
                print(f"{val:.6f}")
        except ValueError as e:
            print(f"\nError: {e}")

    except ValueError as e:
        print(f"Invalid input: {e}")

if __name__ == "__main__":
    main()