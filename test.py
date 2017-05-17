import numpy as np
from utils import *
from NPCANDECOMP import *

#boring accuracy test examples
X = NormalTensorComposition((10, 10, 10), 5)
for r in range(1, 10):
    (s, A, B, C, error) = NPGDCANDECOMP(X, r)
    print("Approximating rank 5 with R = %d error: %g" % (r, accuracy(X, r, s, A, B, C)))

#fun sanity example
A = np.array([[1.0, 0.0], [2.0, 1.0], [1.0, 2.0]]).T
B = np.array([[0.0, 1.0], [2.0, 2.0], [1.0, 2.0]]).T
C = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]).T
X = np.einsum('ir,jr,kr->ijk', A, B, C)
(s_out, A_out, B_out, C_out, error) = NPGDCANDECOMP(X, 3)
Y = np.einsum('r,ir,jr,kr->ijk', s_out, A_out, B_out, C_out)
error = accuracy(X, 3, s_out, A_out, B_out, C_out)
print("error: ", error)

print("s")
a_nrm = np.linalg.norm(A, ord = 2, axis = 0)
A /= a_nrm
b_nrm = np.linalg.norm(B, ord = 2, axis = 0)
B /= b_nrm
c_nrm = np.linalg.norm(C, ord = 2, axis = 0)
C /= c_nrm
print(a_nrm*b_nrm*c_nrm)
print("s_out")
print(s_out)

print("A")
print(A)
print("A_out")
print(A_out)
print("B")
print(B)
print("B_out")
print(B_out)
print("C")
print(C)
print("C_out")
print(C_out)
print("X")
print(X)
print("Y")
print(Y)
