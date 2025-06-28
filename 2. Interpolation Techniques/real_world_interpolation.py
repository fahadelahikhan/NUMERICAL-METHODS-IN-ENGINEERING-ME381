#!/usr/bin/env python3

def linear_interpolation(x, y, target_x):
    # Find the interval
    for i in range(len(x) - 1):
        if x[i] <= target_x <= x[i + 1]:
            # Calculate the interpolation
            slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            return y[i] + slope * (target_x - x[i])
    # If target_x is outside the range, return the nearest endpoint value
    if target_x < x[0]:
        return y[0]
    else:
        return y[-1]


def main():
    try:
        n = int(input("Enter the number of data points: "))
        if n <= 1:
            print("Number of data points must be at least 2.")
            return

        x = []
        y = []
        print("Enter the time values (e.g., hours):")
        for _ in range(n):
            val = float(input())
            x.append(val)
        print("Enter the corresponding values (e.g., temperature/sales):")
        for _ in range(n):
            val = float(input())
            y.append(val)

        print("Enter the time to interpolate:")
        target_x = float(input())

        # Check if x values are in ascending order
        if not all(x[i] < x[i + 1] for i in range(n - 1)):
            print("Error: Time values must be in ascending order.")
            return

        result = linear_interpolation(x, y, target_x)
        print(f"The interpolated value at time {target_x} is {result:.2f}")

    except ValueError as e:
        print(f"Invalid input: {e}")


if __name__ == "__main__":
    main()