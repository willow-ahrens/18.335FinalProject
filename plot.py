import numpy as np
import matplotlib.pyplot as plt
import time
from TFGDCANDECOMP import TFGDCANDECOMP
from TFADAMCANDECOMP import TFADAMCANDECOMP
from NPGDCANDECOMP import NPGDCANDECOMP
from NPALSCANDECOMP import NPALSCANDECOMP
from utils import *

#Important plotting parameters:
trials = 5 #Number of trials when benchmarking
nbars = 10 #Approximately how many bars to use

n_1 = 20
n_2 = 5
R = 4

tensors = [{"name" : "BigCube",
            "rank" : R,
            "ranktype" : "true",
            "data" : lambda : NormalTensorComposition((n_1, n_1, n_1), R),
            "time" : 20},
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
            "time" : 20}]

decomps = [
           {"name"  : "Gradient Descent",
            "color" : "green",
            "kwargs": {"stepsize" : 0.001, "els" : False},
            "func"  : NPGDCANDECOMP},
           {"name"  : "Gradient Descent ELS",
            "color" : "orange",
            "kwargs": {"stepsize" : 0.1, "els" : True},
            "func"  : NPGDCANDECOMP},
           {"name"  : "Alternating Least Squares",
            "color" : "blue",
            "kwargs": {},
            "func"  : NPALSCANDECOMP},
           {"name"  : "TensorFlow Gradient Descent",
            "color" : "red",
            "kwargs": {"stepsize" : 0.001},
            "func"  : TFGDCANDECOMP}]


#Plot 1

for tensor in tensors:
    plot_name = "plot1_%s.pdf" % tensor["name"]
    print("Creating %s..." % plot_name)
    x_max = 3.154e+7 #number of seconds in a year
    for decomp in decomps:
        print("Running %s..." % decomp["name"])
        X = [tensor["data"]() for trial in range(trials)]
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

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([0, x_max])
    plt.ylim([0, 2])
    plt.xlabel('Time Spent Computing CANDECOMP (s)')
    plt.ylabel('Relative Sum Of Squared Residual Error')
    plt.title('Sum Of Squared Error vs. Time To Factorize Rank %d Tensor %s' % (tensor["rank"], tensor["name"]))
    plt.legend(loc='best')
    #plt.savefig(plot_name)
    plt.show()
    plt.clf()

#Plot 2

"""
for tensor in tensors:
    NormalTensorComposition((n_1, n_1, n_1), R),
    plot_name = "plot1_%s.pdf" % tensor["name"]
    print("Creating %s..." % plot_name)
    x_max = 3.154e+7 #number of seconds in a year
    for decomp in decomps:
        print("Running %s..." % decomp["name"])
        X = [tensor["data"]() for trial in range(trials)]
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

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([0, x_max])
    plt.ylim([0, 2])
    plt.xlabel('Time Spent Computing CANDECOMP (s)')
    plt.ylabel('Relative Sum Of Squared Residual Error')
    plt.title('Sum Of Squared Error vs. Time To Factorize Rank %d Tensor %s' % (tensor["rank"], tensor["name"]))
    plt.legend(loc='best')
    #plt.savefig(plot_name)
    plt.show()
    plt.clf()
"""
