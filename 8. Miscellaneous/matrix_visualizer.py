#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def generate_matrix(rows, cols):
    """Generate a sample coefficient matrix."""
    return np.random.rand(rows, cols)


def visualize_matrix(matrix):
    """Visualize the matrix using a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Coefficient Value')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.title('Matrix Coefficient Heatmap')
    plt.show()


def main():
    try:
        # Generate or input matrix
        rows = int(input("Enter the number of rows: "))
        cols = int(input("Enter the number of columns: "))

        # For demonstration, generate a random matrix
        matrix = generate_matrix(rows, cols)

        # Visualize the matrix
        visualize_matrix(matrix)

    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()