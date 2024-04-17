import numpy as np
import matplotlib.pyplot as plt
from utils import plot_lines

x = np.arange(0, 5, 0.1);
y = np.sin(x)
plt.plot(x, y)

# A linear system is **singular** if it has no unique solution, and otherwise it is said to be **non-singular**.

# 1. System of Linear Equations as Matrices
A = np.array([
        [-1, 3],
        [3, 2]
    ], dtype=np.dtype(float)) # represent the coefficients of the system as its own matrix

b = np.array([7, 1], dtype=np.dtype(float)) # represent the outputs of the system as a vector

print(f"Matrix A:\n{A}")
print(f"Array b:\n{b}")

print(f"Shape of A: {A.shape}") # the dimensions for matrix
print(f"Shape of b: {b.shape}") # the dimensions for vector

x = np.linalg.solve(A, b) # find the solution of the system
print(f"Solution: {x}")

# 2. Evaluating Determinant of a Matrix
d = np.linalg.det(A)
print(f"Determinant of matrix A: {d:.2f}") # Linear system containing two (or more) equations with the same number of unknown variables will have one solution if and only if matrix has non-zero determinant.

A_system = np.hstack((A, b.reshape((2, 1))))
print(f"Linear system:\n{A_system}")

print(f"Extract a row of a matrix A: {A_system[1]}\n")

# 3. Graphical Representation of the Solution
plot_lines(A_system)

A_2 = np.array([
        [-1, 3],
        [3, -9]
    ], dtype=np.dtype(float))

b_2 = np.array([7, 1], dtype=np.dtype(float))

print(f"Matrix A_2:\n{A_2}")
print(f"Array b_2:\n{b_2}")

d_2 = np.linalg.det(A_2)
print(f"Determinant of matrix A_2: {d_2:.2f}") # System of Linear Equations with No Solutions(either infinitely many solutions or none)

try:
    x_2 = np.linalg.solve(A_2, b_2)
except np.linalg.LinAlgError as err:
    print(err)

A_2_system = np.hstack((A_2, b_2.reshape((2, 1))))
print(f"Linear system_2:\n{A_2_system}\n")

plot_lines(A_2_system)

b_3 = np.array([7, -21], dtype=np.dtype(float))
A_3_system = np.hstack((A_2, b_3.reshape((2, 1))))
print(f"Matrix A_3:\n{A_2}")
print(f"Array b_3:\n{b_3}")

d_3 = np.linalg.det(A_2)
print(f"Determinant of matrix A_3: {d_3:.2f}")

print(f"Linear system_3:\n{A_3_system}")

plot_lines(A_3_system)