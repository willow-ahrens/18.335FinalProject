import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import time
import test

def TFGDCANDECOMP(X, R, maxtime = 0, maxsteps=5000, tol=0.0001):
    T = tf.as_dtype(X.dtype)
    A = tf.Variable(tf.random_normal((X.shape[0], R), dtype=T), dtype = T)
    B = tf.Variable(tf.random_normal((X.shape[1], R), dtype=T), dtype = T)
    C = tf.Variable(tf.random_normal((X.shape[2], R), dtype=T), dtype = T)
    Y = tf.einsum('ir,jr,kr->ijk', A, B, C)
    loss = tf.norm(X-Y)**2
    X_sq = np.linalg.norm(X)**2

    A_history = []
    B_history = []
    C_history = []
    A_history.append(A)
    B_history.append(B)
    C_history.append(C)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init, {})

    error = np.zeros(maxsteps + 1)/X_sq
    step = 0
    error[0] = sess.run(loss, {})/X_sq
    elapsed = 0
    while maxsteps == 0 or step < maxsteps:
        tic = time.clock()
        step += 1
        sess.run(train, {})
        error[step] = sess.run(loss, {})/X_sq
        toc = time.clock()

        elapsed += toc - tic
        (A_out, B_out, C_out) = sess.run([A, B, C], {})
        A_history.append(A_out)
        B_history.append(B_out)
        C_history.append(C_out)

        if tol > 0 and error[step - 1] - error[step] < tol:
            break
        if maxtime > 0 and elapsed > maxtime:
            break

    (A_out, B_out, C_out) = sess.run([A, B, C], {})

    a_nrm = np.linalg.norm(A_out, ord = 2, axis = 0)
    b_nrm = np.linalg.norm(B_out, ord = 2, axis = 0)
    c_nrm = np.linalg.norm(C_out, ord = 2, axis = 0)

    results = {}
    results["s"] = a_nrm * b_nrm * c_nrm
    results["A"] = A_out/a_nrm
    results["B"] = B_out/b_nrm
    results["C"] = C_out/c_nrm
    results["error_history"] = error[0: step + 1]
    results["A_history"] = A_history
    results["B_history"] = B_history
    results["C_history"] = C_history
    results["time"] = elapsed
    return results

if __name__=="__main__":
    test.test(TFGDCANDECOMP)
