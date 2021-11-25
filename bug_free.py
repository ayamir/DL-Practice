import numpy as np

# Don't use it
a = np.random.randn(5)
print(a)
print(a.shape)
print(a.T)
print(np.dot(a, a.T))

a = np.random.randn(5, 1)
print(a)
print(a.shape)
print(a.T)
print(np.dot(a, a.T))
