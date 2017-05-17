import numpy as np
import itertools
from scipy.linalg import pinv

def PI(M):  # Manual Pseudoinverse
    return np.linalg.inv(np.dot(M.T, M)).dot(M.T)


def PIR(M, eta=0.000001):  # implements PseudoInverse with a eta
    MM = np.dot(M.T, M)
    d = np.dot(M.T, M) + eta*np.eye(MM.shape[0], MM.shape[1])
    return np.linalg.inv(d).dot(M.T)


def SPIR(M):
    return pinv(M)  # QR based pseudoinverse


def NPIR(M):
    return np.linalg.pinv(M)  # SVD based pseudoinverse


def KR(A, H):  #implements Khatri-Rao product
    assert A.shape[1] == H.shape[1], "Can't take KR product,\
    uneven sizes"
    I, J = A.shape
    K, J = H.shape
    M = np.zeros((I*K, J), dtype=np.float64)
    for j in range(J):
        M[:, j] = np.kron(A[:, j],H[:, j])
    return M


def Unfold(T, mode="I"):  # implements unfolding
    # as described in eqn 7 (page 13) of Comon
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


def agradient(T, A, B, C):  # calculates analytic gradient
    # Equations derived from page 12 of Comon
    gA1 = np.dot(np.kron(np.eye(A.shape[0]),
                         np.multiply(np.dot(C.T, C), np.dot(B.T, B))),
                 A.T.flatten())
    gA2 = np.dot(np.kron(np.eye(A.shape[0]), KR(C, B)).T,
                 Unfold(T, mode="J").flatten())
    gA = gA1 + gA2

    gB1 = np.dot(np.kron(np.eye(B.shape[0]),
                         np.multiply(np.dot(A.T, A), np.dot(C.T, C))),
                 B.T.flatten())
    gB2 = np.dot(np.kron(np.eye(B.shape[0]), KR(A, C)).T,
                 Unfold(T, mode="K").flatten())
    gB = gB1 + gB2

    gC1 = np.dot(np.kron(np.eye(C.shape[0]),
                         np.multiply(np.dot(B.T, B), np.dot(A.T, A))),
                 C.T.flatten())
    gC2 = np.dot(np.kron(np.eye(C.shape[0]), KR(B, A)).T,
                 Unfold(T, mode="I").flatten())
    gC = gC1 + gC2
    return gA, gB, gC


def delta(A, B):
    s = 0
    assert A.shape == B.shape, "Factor matrix \
    shapes don't match!"
    for j in range(A.shape[1]):
        u, v = A[:, j], B[:, j]
        s += np.linalg.norm(u - (np.dot(v.T, u)/np.dot(v.T, v))*(v))
    return s


def min_delta(u, v):
    return np.linalg.norm(u - (np.dot(v.T, u)/np.dot(v.T, v))*(v))


def largest_norm(M):
    col_norms = np.linalg.norm(M, axis=0)
    return np.argmax(col_norms)


def exact_factor_acc(M_hat, M):
    assert M_hat.shape == M.shape, "Factor matrix \
    shapes don't match!"
    ncols = M.shape[1]
    perms = list(itertools.permutations(range(ncols)))
    #print perms
    errors = np.zeros(len(perms))
    for i, perm in enumerate(perms):
        perm = np.asarray(list(perm))
        errors[i] = delta(M[:, perm], M_hat)
    #print errors
    min_idx = np.argmin(errors)
    return errors[min_idx]


def factor_acc_history(results, ground_truth):
    A_hat_history, B_hat_history, C_hat_history = results["A_history"], results["B_history"], results["C_history"]
    A,B,C = ground_truth[0], ground_truth[1], ground_truth[2]
    M = np.vstack([A,B,C])
    factor_acc = []
    assert len(A_hat_history) == len(B_hat_history) == len(C_hat_history)
    for i in range(len(A_hat_history)):
        A_hat = A_hat_history[i]
        B_hat = B_hat_history[i]
        C_hat = C_hat_history[i]
        M_hat = np.vstack([A_hat, B_hat, C_hat])
        factor_acc.append(exact_factor_acc(M_hat, M))
    return np.asarray(factor_acc)

def greedy_factor_acc(M_hat, M):
    # TODO: IMPLEMENT the greedy factor matrix distance
    # described at end of page 17 and start of 18 in Comon
    pass


def accuracy(X, R, s, A, B, C):
    Y = np.einsum('r,ir,jr,kr->ijk', s, A, B, C)
    error = np.sum((X - Y)**2)/np.sum(X**2)
    return error


def NormalTensor(shape):
    return np.random.randn(shape)


def NormalTensorComposition(shape, R):
    A = np.random.randn(shape[0], R)
    B = np.random.randn(shape[1], R)
    C = np.random.randn(shape[2], R)
    X = np.einsum('ir,jr,kr->ijk', A, B, C)
    return X
