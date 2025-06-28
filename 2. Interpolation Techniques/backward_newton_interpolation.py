#!/usr/bin/env python3

def newton_backward_interpolation(x, y, target_x):
    n = len(x)
    # Calculate the differences
    diff = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        diff[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            diff[i][j] = diff[i][j - 1] - diff[i - 1][j - 1]

    # Calculate the result using the backward difference formula
    h = x[1] - x[0]
    p = (target_x - x[-1]) / h
    result = diff[-1][0]
    for j in range(1, n):
        term = diff[-1][j]
        for i in range(j):
            term *= (p + i)
        term /= (j * (j + 1) // 2)
        result += term
    return result


def main():
    try:
        n = int(input("Enter the number of data points: "))
        if n <= 0:
            print("Number of data points must be a positive integer.")
            return

        x = []
        y = []
        print("Enter the x values:")
        for _ in range(n):
            val = float(input())
            x.append(val)
        print("Enter the y values:")
        for _ in range(n):
            val = float(input())
            y.append(val)

        print("Enter the x value to interpolate:")
        target_x = float(input())

        # Check if x values are equally spaced
        if not all(x[i + 1] - x[i] == x[1] - x[0] for i in range(n - 1)):
            print("Error: x values must be equally spaced.")
            return

        result = newton_backward_interpolation(x, y, target_x)
        print(f"The interpolated value at x = {target_x} is {result:.6f}")

    except ValueError as e:
        print(f"Invalid input: {e}")


if __name__ == "__main__":
    main()