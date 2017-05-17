import numpy as np
from utils import *
from scipy.optimize import fmin

def r1(gM, M):
    return (gM.reshape(M.shape[1],M.shape[0])).T

def r2(gM, M):
    return gM.reshape(M.shape[0],M.shape[1])

def obj(lr, X, A, B, C, gA, gB, gC):
    Anew = A - lr * gA
    Bnew = B - lr * gB
    Cnew = C - lr * gC
    Y = np.einsum('ir,jr,kr->ijk',A,B,C)
    return np.linalg.norm(X-Y)**2

def NPGDCANDECOMP(X, R, maxsteps=5000, lr=0.0001, tol=np.finfo(float).eps, ELS=False,true=None):
    I, J, K = X.shape
    A = np.random.randn(I, R)
    B = np.random.randn(J, R)
    C = np.random.randn(K, R)
    Y = np.einsum('ir,jr,kr->ijk', A, B, C)
    step = 0
    M = np.vstack([true[0], true[1], true[2]])
    M_hat = np.vstack([A, B, C])
    error = np.zeros(maxsteps + 1)
    error[0] = np.linalg.norm(X-Y)**2
    fac_error = np.zeros(maxsteps + 1)
    fac_error[0], _ = exact_factor_acc(M_hat, M)
    while step < maxsteps:
        step += 1
        gA, gB, gC = agradient(X, A, B, C)
        gAr, gBr, gCr = r1(gA, A), r1(gB, B), r1(gC, C)
        if ELS:
            lrg = 0.001
            lr = fmin(func=obj, x0=[lrg],
                      args=(X, A, B, C, gAr, gBr, gCr))
            A, B, C = A - lr*gAr, B - lr*gBr, C - lr*gCr
        else:
            A, B, C = A - lr*gAr, B - lr*gBr, C - lr*gCr
        Y = np.einsum('ir,jr,kr->ijk', A, B, C)
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
