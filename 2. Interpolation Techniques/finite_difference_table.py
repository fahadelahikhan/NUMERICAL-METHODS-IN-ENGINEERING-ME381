#!/usr/bin/env python3

def generate_difference_table(y):
    n = len(y)
    diff = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        diff[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    return diff


def print_difference_table(x, diff):
    from tabulate import tabulate
    table = []
    for i in range(len(x)):
        row = [x[i]] + diff[i]
        table.append(row)
    headers = ["x"] + [f"Δ^{j}y" for j in range(len(x))]
    print(tabulate(table, headers=headers, floatfmt=".6f"))


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

        diff = generate_difference_table(y)
        print_difference_table(x, diff)

    except ValueError as e:
        print(f"Invalid input: {e}")


if __name__ == "__main__":
    main()