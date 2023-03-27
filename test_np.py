
import numpy as np


a = np.array([10, 20, 30])
# print(np.multiply(a, a.transpose()))
# print(np.matmul(a, a.transpose()))
# print(np.matmul(a.transpose(), a))

aa = a.reshape([3, 1])
print(aa)
print(np.matmul(a.reshape([3, 1]), a.reshape([1, 3])))

# np.matmul()

print(np.power(a, 2))
print(a / np.array([1,4,9]))
print(np.sum(a))