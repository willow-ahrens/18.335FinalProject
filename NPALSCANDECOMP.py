import copy
import numpy as np
from scipy.linalg import pinv
from utils import *
import pdb
import time
import test

def F(T, A, B, C, mode="A"):
    if mode == "A":
        return (NPIR(KR(B, C)).dot(Unfold(T, mode="J"))).T
    elif mode == "B":
        return (NPIR(KR(C, A)).dot(Unfold(T, mode="K"))).T
    elif mode == "C":
        return (NPIR(KR(A, B)).dot(Unfold(T, mode="I"))).T

def NPALSCANDECOMP(X, R, maxtime = 0, maxsteps=2000, tol=0.000001):
    I, J, K = X.shape
    A = np.random.randn(I, R)
    B = np.random.randn(J, R)
    C = np.random.randn(K, R)
    Y = np.einsum('ir,jr,kr->ijk', A, B, C)
    X_sq = np.linalg.norm(X)**2

    A_history = []
    B_history = []
    C_history = []
    A_history.append(copy.deepcopy(A))
    B_history.append(copy.deepcopy(B))
    C_history.append(copy.deepcopy(C))

    step = 0
    error = np.zeros(maxsteps + 1)
    error[0] = np.linalg.norm(X-Y)**2/X_sq
    elapsed = 0
    while maxsteps == 0 or step < maxsteps:
        tic = time.clock()
        step += 1
        B = F(X,A,B,C, mode="B")
        C = F(X,A,B,C, mode="C")
        A = F(X,A,B,C, mode="A")
        Y = np.einsum('ir,jr,kr->ijk',A,B,C)
        error[step] = np.linalg.norm(X-Y)**2/X_sq
        toc = time.clock()

        elapsed += toc - tic
        A_history.append(copy.deepcopy(A))
        B_history.append(copy.deepcopy(B))
        C_history.append(copy.deepcopy(C))

        if tol > 0 and error[step - 1] - error[step] < tol:
            break
        if maxtime > 0 and elapsed > maxtime:
            break

    a_nrm = np.linalg.norm(A, ord = 2, axis = 0)
    b_nrm = np.linalg.norm(B, ord = 2, axis = 0)
    c_nrm = np.linalg.norm(C, ord = 2, axis = 0)

    results = {}
    results["s"] = a_nrm * b_nrm * c_nrm
    results["A"] = A/a_nrm
    results["B"] = B/b_nrm
    results["C"] = C/c_nrm
    results["error_history"] = error[0: step + 1]
    results["A_history"] = A_history
    results["B_history"] = B_history
    results["C_history"] = C_history
    results["time"] = elapsed
    return results

if __name__=="__main__":
    test.test(NPALSCANDECOMP)
