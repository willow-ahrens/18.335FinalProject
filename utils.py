import numpy as np

def accuracy(X, R, CANDECOMP):
  (s, A, B, C) = CANDECOMP(X, R)
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
