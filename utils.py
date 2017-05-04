import tensorflow as tf
import numpy as np

def TFCANDECOMP(X, R):
  optimizer = tf.train.GradientDescentOptimizer(0.001)
  steps = 1000

  T = tf.as_dtype(X.dtype)
  A = tf.Variable(tf.random_normal((X.shape[0], R), dtype=T), dtype = T)
  B = tf.Variable(tf.random_normal((X.shape[1], R), dtype=T), dtype = T)
  C = tf.Variable(tf.random_normal((X.shape[2], R), dtype=T), dtype = T)
  Y = tf.einsum('ir,jr,kr->ijk', A, B, C)
  loss = tf.reduce_sum(tf.square(X - Y))

  train = optimizer.minimize(loss)
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init, {})
  for i in range(steps):
    sess.run(train, {})
  (A_out, B_out, C_out, Y_out, loss_out) = sess.run([A, B, C, Y, loss], {})
  na = np.linalg.norm(A_out, ord = 2, axis = 0)
  A_out /= na
  nb = np.linalg.norm(B_out, ord = 2, axis = 0)
  B_out /= nb
  nc = np.linalg.norm(C_out, ord = 2, axis = 0)
  C_out /= nc

  return (na * nb * nc, A_out, B_out, C_out)

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

#boring accuracy test examples
X = NormalTensorComposition((10, 10, 10), 5)
for r in range(1, 10):
  print("Approximating rank 5 with R = %d error: %g" % (r, accuracy(X, r, TFCANDECOMP)))

#fun sanity example
A = np.array([[1.0, 0.0], [2.0, 1.0], [1.0, 2.0]]).T
B = np.array([[0.0, 1.0], [2.0, 2.0], [1.0, 2.0]]).T
C = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]).T
X = np.einsum('ir,jr,kr->ijk', A, B, C)
(s, A_out, B_out, C_out) = TFCANDECOMP(X, 4)
Y = np.einsum('r,ir,jr,kr->ijk', s, A_out, B_out, C_out)
error = accuracy(X, 3, TFCANDECOMP)
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
