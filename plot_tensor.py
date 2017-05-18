import numpy as np
import matplotlib.pyplot as plt

from NPALSCANDECOMP import NPALSCANDECOMP
from NPGDCANDECOMP import NPGDCANDECOMP
from TFGDCANDECOMP import TFGDCANDECOMP


def rc(result):
    A, B, C = result["A"], result["B"], result["C"]
    s = result["s"]
    Y = np.einsum('r,ir,jr,kr->ijk', s, A, B, C)
    return Y

def tensor_heatmaps(X,A,B,C,R,maxtime=5, ncol=3):
    result_als = NPALSCANDECOMP(X, R, maxtime = maxtime,tol=0)
    result_gd = NPGDCANDECOMP(X, R, maxtime = maxtime,tol=0, els=False)
    result_gd_els = NPGDCANDECOMP(X,R, maxtime = maxtime, tol=0, els=True)
    result_tf = NPGDCANDECOMP(X,R, maxtime = maxtime, tol=0)
    tensors = [X, rc(result_als), rc(result_gd), rc(result_gd_els), rc(result_tf)]
    titles = ["Ground truth", "ALS", "GD", "GD-ELS", "TF-GD"]
    fig, axes = plt.subplots(figsize=(16,14), nrows=5, ncols=ncol)
    for i, t in enumerate(tensors):
        for j in range(ncol):
            axes[i,j].pcolor(t[j,:,:])
            axes.set(title=titles[i])
    for ax in axes.flatten():
        ax.axis('tight')
    plt.tight_layout()
    plt.show()
