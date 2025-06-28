#!/usr/bin/env python3

def gauss_seidel_method(A, b, x0, tol=1e-10, max_iter=100):
    n = len(b)
    for i in range(n):
        if A[i][i] == 0:
            raise ValueError(f"Diagonal element A[{i}][{i}] is zero. Gauss-Seidel method requires non-zero diagonal elements.")
    x = x0.copy()
    for iteration in range(max_iter):
        x_prev = x.copy()
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(i))
            s += sum(A[i][j] * x_prev[j] for j in range(i + 1, n))
            x[i] = (b[i] - s) / A[i][i]
        # Check convergence
        converged = True
        for i in range(n):
            if abs(x[i] - x_prev[i]) >= tol:
                converged = False
                break
        if converged:
            return x
    raise ValueError(f"Gauss-Seidel method did not converge within {max_iter} iterations.")

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

        print("\nEnter initial guess ({} values separated by spaces) or press enter for zeros:".format(n))
        initial_guess_input = input().strip()
        if initial_guess_input:
            x0 = list(map(float, initial_guess_input.split()))
            if len(x0) != n:
                print(f"Initial guess must have exactly {n} elements.")
                return
        else:
            x0 = [0.0] * n

        tol = float(input("Enter tolerance (e.g., 1e-10): "))
        if tol <= 0:
            print("Tolerance must be a positive number.")
            return

        max_iter = int(input("Enter maximum number of iterations: "))
        if max_iter <= 0:
            print("Maximum iterations must be a positive integer.")
            return

        try:
            solution = gauss_seidel_method(A, b, x0, tol, max_iter)
            print("\nSolution:")
            for val in solution:
                print(f"{val:.6f}")
        except ValueError as e:
            print(f"\nError: {e}")

    except ValueError as e:
        print(f"Invalid input: {e}")

if __name__ == "__main__":
    main()