#!/usr/bin/env python3

def lagrange_interpolation(x, y, target_x):
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (target_x - x[j]) / (x[i] - x[j])
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

        result = lagrange_interpolation(x, y, target_x)
        print(f"The interpolated value at x = {target_x} is {result:.6f}")

    except ValueError as e:
        print(f"Invalid input: {e}")


if __name__ == "__main__":
    main()