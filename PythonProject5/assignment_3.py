import numpy as np
from scipy.linalg import lu

# Question 1

A1 = np.array([
    [2, -1, 1, 6],
    [1, 3, 1, 0],
    [-1, 5, 4, -3]
], dtype=float)

n1 = A1.shape[0]

for i in range(n1):
    A1[i] = A1[i] / A1[i][i]
    for j in range(i + 1, n1):
        factor = A1[j][i]
        A1[j] = A1[j] - factor * A1[i]

x = np.zeros(n1)
for i in range(n1 - 1, -1, -1):
    x[i] = A1[i][-1] - np.sum(A1[i][i+1:n1] * x[i+1:n1])


print(x)
# Question 2
A2 = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
], dtype=float)

P, L, U = lu(A2)
det_A2 = np.linalg.det(A2)

print(f"\n{det_A2:.16f}\n")

np.set_printoptions(precision=2, suppress=True)
print(L)
print()
print(U)

#Question 3
A3 = np.array([
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
], dtype=float)

ratios = []
for i in range(A3.shape[0]):
    diag = abs(A3[i, i])
    off_diag_sum = np.sum(np.abs(A3[i])) - diag
    ratio = diag / off_diag_sum if off_diag_sum != 0 else float('inf')
    ratios.append(ratio)

min_ratio = min(ratios)
print(f"\n{min_ratio:.16f}")
# Question 4
A4 = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2]
], dtype=float)

eigenvalues = np.linalg.eigvals(A4)
min_eigenvalue = np.min(eigenvalues)

print(f"\n{min_eigenvalue:.16f}")
