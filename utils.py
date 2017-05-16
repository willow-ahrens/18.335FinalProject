import numpy as np

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

def 


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
