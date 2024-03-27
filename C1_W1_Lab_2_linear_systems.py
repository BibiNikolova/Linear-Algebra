import numpy as np
import matplotlib.pyplot as plt
#from utils import plot_lines

A = np.array([
        [-1, 3],
        [3, 2]
    ], dtype=np.dtype(float))

b = np.array([7, 1], dtype=np.dtype(float))

print(f"Matrix A:\n{A}")
print(f"\nArray b:\n{b}")

print(f"Shape of A: {A.shape}")
print(f"Shape of b: {b.shape}")

x = np.linalg.solve(A, b)
print(f"Solution: {x}")

d = np.linalg.det(A)
print(f"Determinant of matrix A: {d:.2f}")

A_system = np.hstack((A, b.reshape((2, 1))))
print(A_system)

print(A_system[1])

#plot_lines(A_system)

A_2 = np.array([
        [-1, 3],
        [3, -9]
    ], dtype=np.dtype(float))

b_2 = np.array([7, 1], dtype=np.dtype(float))

d_2 = np.linalg.det(A_2)

print(f"Determinant of matrix A_2: {d_2:.2f}")

try:
    x_2 = np.linalg.solve(A_2, b_2)
except np.linalg.LinAlgError as err:
    print(err)