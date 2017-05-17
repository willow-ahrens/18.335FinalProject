import numpy as np
from scipy.linalg import pinv
from utils import *
import os
import pdb

def F(T, A, B, C, mode="A"):
    if mode == "A":
        return (NPIR(KR(B, C)).dot(Unfold(T, mode="J"))).T
    elif mode == "B":
        return (NPIR(KR(C, A)).dot(Unfold(T, mode="K"))).T
    elif mode == "C":
        return (NPIR(KR(A, B)).dot(Unfold(T, mode="I"))).T


def NPALSCANDECOMP(X, R, maxsteps=2000, tol=0.000001, true = None):
    I, J, K = X.shape
    A = np.random.randn(I, R)
    B = np.random.randn(J, R)
    C = np.random.randn(K, R)
    Y = np.einsum('ir,jr,kr->ijk',A,B,C)
    step = 0
    M = np.vstack([true[0], true[1], true[2]])
    M_hat = np.vstack([A, B, C])
    error = np.zeros(maxsteps + 1)
    error[0] = np.linalg.norm(X-Y)**2
    fac_error = np.zeros(maxsteps + 1)
    fac_error[0],_ = exact_factor_acc(M_hat, M)
    while step < maxsteps:
        step += 1
        B = F(X,A,B,C, mode="B")
        C = F(X,A,B,C, mode="C")
        A = F(X,A,B,C, mode="A")
        Y = np.einsum('ir,jr,kr->ijk',A,B,C)
        e = np.linalg.norm(X-Y)**2
        error[step] = e
        M_hat = np.vstack([A, B, C])
        fac_error[step], _ = exact_factor_acc(M_hat, M)
        if error[step - 1] - error[step] < tol:
            print "Tolerance limit reached, stopping!"
            break
        #print(type(e))
    # a_nrm = np.linalg.norm(A, ord = 2, axis = 0)
    # A /= a_nrm
    # b_nrm = np.linalg.norm(B, ord = 2, axis = 0)
    # B /= b_nrm
    # c_nrm = np.linalg.norm(C, ord = 2, axis = 0)
    # C /= c_nrm

    errors = error[0: step + 1]
    fac_errors = fac_error[0: step + 1]
    # return (a_nrm * b_nrm * c_nrm, A, B, C, error)
    return A, B, C, errors, fac_errors

#boring accuracy test examples
def main():
    X = NormalTensorComposition((10, 10, 10), 5)
    for r in range(1, 10):
        (s, A, B, C, error) = NPALSCANDECOMP(X, r)
    print("Approximating rank 5 with R = %d error: %g" % (r, accuracy(X, r, s, A, B, C)))

    #fun sanity example
    A = np.array([[1.0, 0.0], [2.0, 1.0], [1.0, 2.0]]).T
    B = np.array([[0.0, 1.0], [2.0, 2.0], [1.0, 2.0]]).T
    C = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]).T
    X = np.einsum('ir,jr,kr->ijk', A, B, C)
    (s, A_out, B_out, C_out, errors) = NPALSCANDECOMP(X, 4)
    Y = np.einsum('r,ir,jr,kr->ijk', s, A_out, B_out, C_out)
    error = accuracy(X, 3, s, A_out, B_out, C_out)
    print("error: ", error)
    print("s")
    print(s)
    print("A")
    print(A_out)
    print("B")
    print(B_out)
    print("C")
    print(C_out)
    print("X")
    print(X)
    print("Y")
    print(Y)
    print("Errors")
    print(errors)


if __name__=="__main__":
    main()
