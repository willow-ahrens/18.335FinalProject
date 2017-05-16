import numpy as np
from scipy.linalg import pinv
from utils import *
import os
import pdb
def PI(M): #Pseudoinverse
    return np.linalg.inv(np.dot(M.T, M)).dot(M.T)

def PIR(M, eta = 0.000001):  #implements PseudoInverse with a eta
    MM = np.dot(M.T, M)
    d = np.dot(M.T, M) + eta*np.eye(MM.shape[0], MM.shape[1])
    return np.linalg.inv(d).dot(M.T)

def SPIR(M):
    return pinv(M)

def NPIR(M):
    return np.linalg.pinv(M)

def KR(A, H):  #implements Khatri-Rao product
    assert A.shape[1] == H.shape[1], "Can't take KR product,\
    uneven sizes"
    I,J = A.shape
    K,J = H.shape
    M = np.zeros((I*K, J), dtype=np.float64)
    for j in range(J):
        M[:,j] = np.kron(A[:,j],H[:,j])
    return M


def Unfold(T, mode="I"):
    assert T.ndim == 3, "Not unfolding a tensor"
    I, J, K = T.shape
    a = []
    if mode == "K":  # represents $T_{KI \times J}$
        for k in range(K):
            a.append(T[:, :, k])
        return np.vstack(a)
    elif mode == "J":  # represents $T_{JK \times I}$
        for j in range(J):
            a.append(T[:, j, :].T)
        return np.vstack(a)
    elif mode == "I":  # represents $T_{KI \times J}$
        for i in range(I):
            a.append(T[i, :, :])
        return np.vstack(a)

def F(T, A, B, C, mode="A"):
    if mode == "A":
        return (NPIR(KR(B, C)).dot(Unfold(T, mode="J"))).T
    elif mode == "B":
        return (NPIR(KR(C, A)).dot(Unfold(T, mode="K"))).T
    elif mode == "C":
        return (NPIR(KR(A, B)).dot(Unfold(T, mode="I"))).T


def NPALSCANDECOMP(X, R, maxsteps=2000, tol=0.000001):
    I, J, K = X.shape
    A = np.random.randn(I, R)
    B = np.random.randn(J, R)
    C = np.random.randn(K, R)
    Y = np.einsum('ir,jr,kr->ijk',A,B,C)
    X_sq = np.sum(X**2)
    step = 0
    error = np.zeros(maxsteps + 1)
    error[0] = np.linalg.norm(X-Y)**2
    while step < maxsteps:
        step += 1
        B = F(X,A,B,C, mode="B")
        C = F(X,A,B,C, mode="C")
        A = F(X,A,B,C, mode="A")
        Y = np.einsum('ir,jr,kr->ijk',A,B,C)
        e = np.linalg.norm(X-Y)**2
        error[step] = e
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

    # return (a_nrm * b_nrm * c_nrm, A, B, C, error)
    print A.shape
    print B.shape
    print C.shape
    print errors.shape
    return A, B, C, errors
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
