import numpy as np
from scipy.linalg import pinv
from utils import *
import pdb
import time

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
    X_sq = np.linalg.norm(X)

    A_record = []
    B_record = []
    C_record = []
    A_record.append(A)
    B_record.append(B)
    C_record.append(C)

    step = 0
    error = np.zeros(maxsteps + 1)
    error[0] = np.linalg.norm(X-Y)**2
    while maxsteps == 0 or step < maxsteps:
        step += 1
        B = F(X,A,B,C, mode="B")
        C = F(X,A,B,C, mode="C")
        A = F(X,A,B,C, mode="A")
        Y = np.einsum('ir,jr,kr->ijk',A,B,C)
        error[step] = np.linalg.norm(X-Y)**2/X_sq

        A_record.append(A)
        B_record.append(B)
        C_record.append(C)

        if tol > 0 and error[step - 1] - error[step] < tol:
            break
        if maxtime > 0 and time.clock() - tic > maxtime:
            break
    a_nrm = np.linalg.norm(A, ord = 2, axis = 0)
    A /= a_nrm
    b_nrm = np.linalg.norm(B, ord = 2, axis = 0)
    B /= b_nrm
    c_nrm = np.linalg.norm(C, ord = 2, axis = 0)
    C /= c_nrm
    total_time = np.sum(np.asarray(times))
    errors = error[0: step + 1]
    r = {}
    r["s"] = a_nrm * b_nrm * c_nrm
    r["A"] = A
    r["B"] = B
    r["C"] = C
    r["errors"] = errors
    r["A_record"] = A_record
    r["B_record"] = B_record
    r["C_record"] = C_record
    return r

if __name__=="__main__":
    test(NPALSCANDECOMP)
