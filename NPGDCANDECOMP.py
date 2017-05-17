import os
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

stepsize = 0.0001

def NPGDCANDECOMP(X, R, maxsteps=5000, tol=0.0001):
    T = X.dtype
    A = np.random_normal((X.shape[0], R), dtype=T)
    B = np.random_normal((X.shape[1], R), dtype=T)
    C = np.random_normal((X.shape[2], R), dtype=T)
    Y = np.einsum('ir,jr,kr->ijk', A, B, C)
    loss = np.sum((X - Y)**2)

    X_sq = np.sum(X**2)

    error = np.zeros(maxsteps + 1)
    step = 0
    error[0] = loss/X_sq
    while step < maxsteps:
        step += 1
        gA = 2 * (np.einsum('lt,jt,kt,js,ks->ls',A,B,C,B,C) - np.einsum('ljk,js,ks->ls',X,B,C))
        gB = 2 * (np.einsum('it,lt,kt,is,ks->ls',A,B,C,A,C) - np.einsum('ilk,is,ks->ls',X,A,C))
        gC = 2 * (np.einsum('it,jt,lt,is,js->ls',A,B,C,A,B) - np.einsum('ijl,is,js->ls',X,A,B))
        A -= gA * stepsize
        B -= gB * stepsize
        C -= gC * stepsize
        Y = np.einsum('ir,jr,kr->ijk', A, B, C)
        loss = np.sum((X - Y)**2)
        error[step] = loss/X_sq
        if error[step - 1] - error[step] < tol:
            break

    (A_out, B_out, C_out) = sess.run([A, B, C], {})
    a_nrm = np.linalg.norm(A_out, ord = 2, axis = 0)
    A_out /= a_nrm
    b_nrm = np.linalg.norm(B_out, ord = 2, axis = 0)
    B_out /= b_nrm
    c_nrm = np.linalg.norm(C_out, ord = 2, axis = 0)
    C_out /= c_nrm

    error = error[0: step + 1]

    return (a_nrm * b_nrm * c_nrm, A_out, B_out, C_out, error)
