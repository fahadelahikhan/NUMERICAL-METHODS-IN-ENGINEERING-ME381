#!/usr/bin/env python3

def gauss_jordan_elimination(matrix):
    n = len(matrix)
    for i in range(n):
        # Make the diagonal element 1
        pivot = matrix[i][i]
        if pivot == 0:
            raise ValueError("Pivot element is zero. System may be singular.")
        for j in range(i, n + 1):
            matrix[i][j] /= pivot

        # Eliminate all other elements in the current column
        for k in range(n):
            if k != i:
                factor = matrix[k][i]
                for j in range(i, n + 1):
                    matrix[k][j] -= factor * matrix[i][j]
    return [row[-1] for row in matrix]


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
        solution = gauss_jordan_elimination(matrix)
        print("\nSolution:")
        for val in solution:
            print(f"{val:.6f}")
    except ValueError as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()