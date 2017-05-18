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
nbars = 10 #Approximately how many bars to use

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

"""
{"name" : "Pancake",
"rank" : R,
"ranktype" : "true",
"data" : lambda : NormalTensorComposition((n_1, n_1, n_2), R),
"time" : 20},
{"name" : "Burrito",
"rank" : R,
"ranktype" : "true",
"data" : lambda : NormalTensorComposition((n_1, n_2, n_2), R),
"time" : 20},
{"name" : "SmallCube",
"rank" : R,
"ranktype" : "true",
"data" : lambda : NormalTensorComposition((n_2, n_2, n_2), R),
"time" : 10}
"""

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


#Plot 1

for tensor in tensors:
    plot_name = "plot1_%s.pdf" % tensor["name"]
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
        hi_bars = np.std(errors, axis = 0)
        lo_bars = np.std(errors, axis = 0)
        errors = np.mean(errors, axis = 0)
        times = np.array(range(len(errors))) * timestep
        plt.plot(times, errors, color = decomp["color"], label = decomp["name"])
        nbarsp = len(times) // nbars
        #plt.errorbar(times[0::nbarsp], errors[0::nbarsp], yerr = [lo_bars[0::nbarsp], hi_bars[0::nbarsp]], color = decomp["color"], linestyle="")
        x_max = min(x_max, len(errors) * timestep)

    plt.xlabel('Time Spent Computing CANDECOMP (s)')
    plt.ylabel('Relative Sum Of Squared Residual Error')
    plt.title('Sum Of Squared Error vs. Time To Factorize Rank %d Tensor %s' % (tensor["rank"], tensor["name"]))
    plt.legend(loc='best')

    plt.xlim([0, x_max])
    plt.ylim([0, 2])
    plt.savefig(plot_name)

    plt.yscale("log")
    plt.xlim([0, x_max])
    plt.ylim([0, 2])
    plt.savefig("log"+plot_name)

    plt.xscale("log")
    plt.xlim([0, x_max])
    plt.ylim([0, 2])
    plt.savefig("loglog"+plot_name)
    plt.clf()
"""

#plot 2


for tensor in tensors:
    plot_name = "plot2_%s.pdf" % tensor["name"]
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
            print(A[trial].shape)
            print(B[trial].shape)
            print(C[trial].shape)
            print(X[trial].shape)
            print(results["A_history"][0].shape)
            print(results["B_history"][0].shape)
            print(results["C_history"][0].shape)
            factor_acc_history(results, A[trial],B[trial],C[trial])
            error_histories.append(results["factor_error"])
            elapsed += results["time"]
        timestep = elapsed/sum([len(error_history) for error_history in error_histories])
        num_steps = min([len(error_history) for error_history in error_histories])
        errors = np.array([error_history[0:num_steps] for error_history in error_histories])
        hi_bars = np.std(errors, axis = 0)
        lo_bars = np.std(errors, axis = 0)
        errors = np.mean(errors, axis = 0)
        times = np.array(range(len(errors))) * timestep
        plt.plot(times, errors, color = decomp["color"], label = decomp["name"])
        nbarsp = len(times) // nbars
        #plt.errorbar(times[0::nbarsp], errors[0::nbarsp], yerr = [lo_bars[0::nbarsp], hi_bars[0::nbarsp]], color = decomp["color"], linestyle="")
        x_max = min(x_max, len(errors) * timestep)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([0, x_max])
    plt.ylim([0, 2])
    plt.xlabel('Time Spent Computing CANDECOMP (s)')
    plt.ylabel('Relative Factor Error')
    plt.title('Factor Error vs. Time To Factorize Rank %d Tensor %s' % (tensor["rank"], tensor["name"]))
    plt.legend(loc='best')
    plt.savefig(plot_name)
    plt.clf()
"""

#Plot 3

for decomp in decomps[0:2]:
    plot_name = "plot3_%s.pdf" % decomp["name"]
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
