import numpy as np

A = np.array([
    [56.0, 0.0, 4.4, 68.0],
    [1.2, 104.0, 52.0, 8.0],
    [1.8, 135.0, 99.0, 0.9],
])

print(A)

cal = A.sum(axis=0)
print(cal)

percentage = 100 * A / cal
print(percentage)

B = np.array([[1, 2, 3], [4, 5, 6]])
C = np.array([100, 200, 300])

print(B + C)

D = np.array([[100], [200]])

print(B+D)
