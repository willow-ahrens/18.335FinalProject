import numpy as np
import itertools
import matplotlib.pyplot as plt
from NPALSCANDECOMP import NPALSCANDECOMP
from NPGDCANDECOMP import NPGDCANDECOMP
from TFGDCANDECOMP import TFGDCANDECOMP

def delta(u, v):
    return np.linalg.norm(u-v)

def exact_factor_acc(M_hat, M, return_perm=False):
    assert M_hat.shape == M.shape, "Factor matrix \
    shapes don't match!"
    ncols = M.shape[1]
    perms = list(itertools.permutations(range(ncols)))
    # print perms
    errors = np.zeros(len(perms))
    for i, perm in enumerate(perms):
        perm = np.asarray(list(perm))
        # errors[i] = delta(M[:, perm], M_hat)
        errors[i] = delta(M[:, perm], M_hat)
    # print errors
    min_idx = np.argmin(errors)
    if return_perm:
        return errors[min_idx], perms[min_idx]
    else:
        return errors[min_idx]

def ABC(result,A,B,C):
    M = np.vstack([A,B,C])
    M_hat = np.vstack([result["A"], result["B"], result["C"]])


def rc(result):
    A, B, C = result["A"], result["B"], result["C"]
    s = result["s"]
    Y = np.einsum('r,ir,jr,kr->ijk', s, A, B, C)
    return Y

def tensor_heatmaps(X,A,B,C,R,maxtime=1,ncol=2):
    result_als = NPALSCANDECOMP(X, R, maxtime = maxtime,tol=0)
    result_gd = NPGDCANDECOMP(X, R, maxtime = maxtime,tol=0, els=False)
    result_gd_els = NPGDCANDECOMP(X,R, maxtime = maxtime, tol=0, els=True)
    result_tf = NPGDCANDECOMP(X,R, maxtime = maxtime, tol=0)
    tensors = [X, rc(result_als), rc(result_gd_els), rc(result_gd), rc(result_tf)]
    titles = ["Ground truth", "ALS", "GD-ELS", "GD", "TF-GD"]
    fig, axes = plt.subplots(figsize=(12,12), nrows=ncol, ncols=5)
    for i, t in enumerate(tensors):
        for j in range(ncol):
            axes[j,i].pcolor(t[j,:,:])
            axes[j,i].set(title="%s slice %i"%(titles[i], j))
    for ax in axes.flatten():
        ax.axis('tight')
    plt.suptitle("Reconstruction of a (20,20,20) Rank 5 Tensor")
    plt.tight_layout(pad=4.0, w_pad=3.5, h_pad=4.5)
    plt.savefig("Reconstruction.png", dpi=350)
    plt.show()

def ABC_heatmaps(X,A,B,C,R,maxtime=1,ncol=2):
    a_nrm = np.linalg.norm(A, ord = 2, axis = 0)
    b_nrm = np.linalg.norm(B, ord = 2, axis = 0)
    c_nrm = np.linalg.norm(C, ord = 2, axis = 0)
    A = A/a_nrm
    B = B/b_nrm
    C = C/c_nrm
    
    

