#!/usr/bin/env python3

def forward_elimination(matrix):
    n = len(matrix)
    for i in range(n):
        if matrix[i][i] == 0:
            raise ValueError("Pivot element is zero. Try using partial pivoting.")
        for j in range(i + 1, n):
            factor = matrix[j][i] / matrix[i][i]
            for k in range(i, n + 1):
                matrix[j][k] -= factor * matrix[i][k]

def back_substitution(matrix):
    n = len(matrix)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if matrix[i][i] == 0:
            raise ValueError("Diagonal element is zero. System may be singular.")
        x[i] = matrix[i][n]
        for j in range(i + 1, n):
            x[i] -= matrix[i][j] * x[j]
        x[i] /= matrix[i][i]
    return x

def gauss_elimination(matrix):
    forward_elimination(matrix)
    return back_substitution(matrix)

def main():
    matrix = []
    try:
        n = int(input("Enter the number of equations: "))
        print("Enter each equation's coefficients followed by the constant term, separated by spaces:")
        for _ in range(n):
            row = list(map(float, input().split()))
            if len(row) != n + 1:
                print(f"Each row must have exactly {n + 1} elements.")
                return
            matrix.append(row)
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    try:
        solution = gauss_elimination(matrix)
        print("\nSolution:")
        for val in solution:
            print(f"{val:.6f}")
    except ValueError as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()