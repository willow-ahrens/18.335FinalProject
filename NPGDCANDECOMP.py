import copy
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from scipy.optimize import fmin
import time
import test

def NPGDCANDECOMP(X, R, maxtime = 0, maxsteps=5000, tol=0.0001):
    A = np.random.randn(X.shape[0], R)
    B = np.random.randn(X.shape[1], R)
    C = np.random.randn(X.shape[2], R)
    Y = np.einsum('ir,jr,kr->ijk', A, B, C)
    X_sq = np.linalg.norm(X)**2

    A_history = []
    B_history = []
    C_history = []
    A_history.append(copy.deepcopy(A))
    B_history.append(copy.deepcopy(B))
    C_history.append(copy.deepcopy(C))

    stepsize = 0.01 #initial stepsize guess

    step = 0
    error = [np.linalg.norm(X - Y)**2/X_sq]
    elapsed = 0
    while maxsteps == 0 or step < maxsteps:
        tic = time.clock()
        step += 1
        gA = 2 * (np.einsum('lt,jt,kt,js,ks->ls',A,B,C,B,C) - np.einsum('ljk,js,ks->ls',X,B,C))
        gB = 2 * (np.einsum('it,lt,kt,is,ks->ls',A,B,C,A,C) - np.einsum('ilk,is,ks->ls',X,A,C))
        gC = 2 * (np.einsum('it,jt,lt,is,js->ls',A,B,C,A,B) - np.einsum('ijl,is,js->ls',X,A,B))
        stepsize = fmin(lambda stepsize: np.sum((X - np.einsum('ir,jr,kr->ijk', A - stepsize*gA, B - stepsize*gB, C - stepsize*gC))**2), stepsize, disp=False)
        A -= gA * stepsize
        B -= gB * stepsize
        C -= gC * stepsize
        Y = np.einsum('ir,jr,kr->ijk', A, B, C)
        error.append(np.linalg.norm(X - Y)**2/X_sq)
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
    results["error_history"] = error
    results["A_history"] = A_history
    results["B_history"] = B_history
    results["C_history"] = C_history
    results["time"] = elapsed
    print(A_history[0] == A_history[-1])
    return results

if __name__=="__main__":
    test.test(NPGDCANDECOMP)
