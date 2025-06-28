#!/usr/bin/env python3

def stirling_interpolation(x, y, target_x):
    n = len(x)
    h = x[1] - x[0]
    # Calculate the central differences
    diff = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        diff[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    # Find the position of target_x in the table
    position = (target_x - x[0]) / h
    u = position - 0.5  # Adjusted position for central differences

    # Calculate the result using Stirling's formula
    result = diff[0][0]
    for j in range(1, n):
        factor = 1.0
        for k in range(j):
            factor *= (u + k) if k % 2 == 0 else (-u + k + 1)
        term = (diff[(j + 1) // 2][j] + diff[j // 2][j]) / 2 * factor / (j * (j + 1))
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

        result = stirling_interpolation(x, y, target_x)
        print(f"The interpolated value at x = {target_x} is {result:.6f}")

    except ValueError as e:
        print(f"Invalid input: {e}")


if __name__ == "__main__":
    main()