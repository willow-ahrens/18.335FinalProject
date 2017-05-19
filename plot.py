import numpy as np
import matplotlib.pyplot as plt
import time
from TFGDCANDECOMP import TFGDCANDECOMP
from TFADAMCANDECOMP import TFADAMCANDECOMP
from NPGDCANDECOMP import NPGDCANDECOMP
from NPALSCANDECOMP import NPALSCANDECOMP
from utils import *

#Important plotting parameters:
trials = 1 #Number of trials when benchmarking

n_1 = 20
n_2 = 5
R = 4

def NormieTensor(I, J, K, R):
    A = np.random.randn(I, R)
    B = np.random.randn(J, R)
    C = np.random.randn(K, R)
    X = np.einsum('ir,jr,kr->ijk', A, B, C)
    return {"X":X, "A":A, "B":B, "C":C}

tensors = [{"name" : "BigCube",
            "rank" : R,
            "ranktype" : "true",
            "data" : lambda : NormieTensor(n_1, n_1, n_1, R),
            "time" : 45}]

decomps = [
           {"name"  : "Alternating Least Squares",
            "color" : "blue",
            "kwargs": {},
            "bound" : lambda n: 3*n*n*(7*n*n + n) + 3*n*n*n*n + 3*n*(n*n + n) + 11*n*n*n,
            "func"  : NPALSCANDECOMP},
           {"name"  : "Gradient Descent",
            "color" : "green",
            "bound" : lambda n: 12*n*n*n*n + 6*n*n*n,
            "kwargs": {"stepsize" : 0.0001, "els" : False},
            "func"  : NPGDCANDECOMP},
           {"name"  : "Gradient Descent ELS",
            "color" : "orange",
            "kwargs": {"stepsize" : 0.0001, "els" : True},
            "func"  : NPGDCANDECOMP},
           {"name"  : "TensorFlow Gradient Descent",
            "color" : "red",
            "kwargs": {"stepsize" : 0.0001},
            "func"  : TFGDCANDECOMP}]

#plot 0

def rc(result):
    A, B, C = result["A"], result["B"], result["C"]
    s = result["s"]
    Y = np.einsum('r,ir,jr,kr->ijk', s, A, B, C)
    return Y

def tensor_heatmaps(maxtime=0.05,ncol=2):
    R = 7
    A = np.random.randn(20,5)
    B = np.random.randn(20,5)
    C = np.random.randn(20,5)
    X = np.einsum('ir,jr,kr->ijk', A, B, C)
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
    plt.savefig("plot0_reconstruction.pdf", dpi=350)
    plt.clf()

print("Creating plot0_reconstruction...")
tensor_heatmaps()

#Plot 1

for tensor in tensors:
    plot_name = "plot1_%s.pdf" % tensor["name"]
    plot_name = plot_name.replace(" ", "_")
    print("Creating %s..." % plot_name)
    x_max = 3.154e+7 #number of seconds in a year
    for decomp in decomps:
        print("Running %s..." % decomp["name"])
        data = [tensor["data"]() for trial in range(trials)]
        A = [data[trial]["A"] for trial in range(trials)]
        B = [data[trial]["B"] for trial in range(trials)]
        C = [data[trial]["C"] for trial in range(trials)]
        X = [data[trial]["X"] for trial in range(trials)]
        elapsed = 0
        error_histories = []
        for trial in range(trials):
            results = decomp["func"](X[trial], tensor["rank"], tol = 0, maxtime = tensor["time"], maxsteps = 0, **decomp["kwargs"])
            error_histories.append(results["error_history"])
            elapsed += results["time"]
        timestep = elapsed/sum([len(error_history) for error_history in error_histories])
        num_steps = min([len(error_history) for error_history in error_histories])
        errors = np.array([error_history[0:num_steps] for error_history in error_histories])
        errors = np.mean(errors, axis = 0)
        times = np.array(range(len(errors))) * timestep
        plt.plot(times, errors, color = decomp["color"], label = decomp["name"])
        x_max = min(x_max, len(errors) * timestep)

    plt.xlabel('Time Spent Computing CANDECOMP (s)')
    plt.ylabel('Relative Sum Of Squared Residual Error')
    plt.title('Sum Of Squared Error vs. Time To Factorize Rank %d Tensor %s' % (tensor["rank"], tensor["name"]))
    plt.legend(loc='best')

    plt.yscale("log")
    plt.xscale("log")
    plt.xlim([0, x_max])
    plt.ylim([0, 2.2])
    plt.savefig(plot_name)
    plt.clf()

#plot 2


for tensor in tensors:
    plot_name = "plot2_%s.pdf" % tensor["name"]
    plot_name = plot_name.replace(" ", "_")
    print("Creating %s..." % plot_name)
    x_max = 3.154e+7 #number of seconds in a year
    for decomp in decomps:
        print("Running %s..." % decomp["name"])
        data = [tensor["data"]() for trial in range(trials)]
        A = [data[trial]["A"] for trial in range(trials)]
        B = [data[trial]["B"] for trial in range(trials)]
        C = [data[trial]["C"] for trial in range(trials)]
        X = [data[trial]["X"] for trial in range(trials)]
        elapsed = 0
        error_histories = []
        for trial in range(trials):
            results = decomp["func"](X[trial], tensor["rank"], tol = 0, maxtime = tensor["time"], maxsteps = 0, **decomp["kwargs"])
            factor_error_history(results, A[trial],B[trial],C[trial])
            error_histories.append(results["factor_error_history"])
            elapsed += results["time"]
        timestep = elapsed/sum([len(error_history) for error_history in error_histories])
        num_steps = min([len(error_history) for error_history in error_histories])
        errors = np.array([error_history[0:num_steps] for error_history in error_histories])
        errors = np.mean(errors, axis = 0)
        times = np.array(range(len(errors))) * timestep
        plt.plot(times, errors, color = decomp["color"], label = decomp["name"])
        x_max = min(x_max, len(errors) * timestep)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([0, x_max])
    plt.xlabel('Time Spent Computing CANDECOMP (s)')
    plt.ylabel('Factor Error')
    plt.title('Factor Error vs. Time To Factorize Rank %d Tensor %s' % (tensor["rank"], tensor["name"]))
    plt.legend(loc='best')
    plt.savefig(plot_name)
    plt.clf()

#Plot 3

for decomp in decomps[0:2]:
    plot_name = "plot3_%s.pdf" % decomp["name"]
    plot_name = plot_name.replace(" ", "_")
    print("Creating %s..." % plot_name)
    x_max = 3.154e+7 #number of seconds in a year

    print("Running %s..." % decomp["name"])

    times = []
    bounds = []
    for n in range(1, 50):
        X = NormalTensorComposition((n, n, n), n)
        results = decomp["func"](X, n, tol = 0, maxtime = 0.1, maxsteps = 0, **decomp["kwargs"])
        times.append(results["time"]/len(results["error_history"]))
        bounds.append(decomp["bound"](n) /3.3e9)

    plt.plot(times, color = decomp["color"], label = decomp["name"])
    plt.plot(bounds, color = "purple", label = "Theoretical Bound")

    plt.xscale("log")
    plt.yscale("log")
    plt.title('Time Per Iteration To Decompose n x n x n Tensor Of Rank n')
    plt.ylabel('Time Per Iteration (s)')
    plt.xlabel('n')
    plt.legend(loc='best')
    plt.savefig(plot_name)
    plt.clf()
