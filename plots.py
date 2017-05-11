import numpy as np
import matplotlib.pyplot as plt
import time
from TFGDCANDECOMP import TFGDCANDECOMP
from TFADAMCANDECOMP import TFADAMCANDECOMP
from utils import *

#Important plotting parameters:
trials = 4 #Number of trials when benchmarking
tol = 0.001 #How long to run these things for
nbars = 10 #Approximately how many bars to use

n_1 = 20
n_2 = 5
R = 4

tensors = [{"name" : "BigCube",
            "rank" : R,
            "ranktype" : "true",
            "data" : lambda : NormalTensorComposition((n_1, n_1, n_1), R)},
           {"name" : "Pancake",
            "rank" : R,
            "ranktype" : "true",
            "data" : lambda : NormalTensorComposition((n_1, n_1, n_2), R)},
           {"name" : "Burrito",
            "rank" : R,
            "ranktype" : "true",
            "data" : lambda : NormalTensorComposition((n_1, n_2, n_2), R)},
           {"name" : "SmallCube",
            "rank" : R,
            "ranktype" : "true",
            "data" : lambda : NormalTensorComposition((n_2, n_2, n_2), R)}]

decomps = [{"name"  : "TensorFlow Gradient Descent",
            "color" : "red",
            "func"  : TFGDCANDECOMP},
           {"name"  : "TensorFlow ADAM",
            "color" : "blue",
            "func"  : TFADAMCANDECOMP}]

#Sanity Checks

#Plot 1

for tensor in tensors:
  plot_name = "plot1_%s.pdf" % tensor["name"]
  print("Creating %s..." % plot_name)
  for decomp in decomps:
    print("Running %s..." % decomp["name"])
    X = [tensor["data"]() for trial in range(trials)]
    total = -time.clock()
    errors_collection = []
    for trial in range(trials):
      errors_collection.append(decomp["func"](X[trial], tensor["rank"], tol = tol)[4])
    total += time.clock()
    timestep = total/sum([len(errors) for errors in errors_collection])
    errors = np.zeros((trials, max([len(errors) for errors in errors_collection])))
    for i in range(trials):
      for j in range(errors.shape[1]):
        if j >= len(errors_collection[i]):
          errors[i][j] = errors_collection[i][- 1]
        else:
          errors[i][j] = errors_collection[i][j]
    hi_bars = np.max(errors, axis = 0) - np.mean(errors, axis = 0)
    lo_bars = np.mean(errors, axis = 0) - np.min(errors, axis = 0)
    errors = np.mean(errors, axis = 0)
    times = np.array(range(len(errors))) * timestep
    plt.plot(times, errors, color = decomp["color"], label = decomp["name"])
    nbarsp = len(times) // nbars
    plt.errorbar(times[0::nbarsp], errors[0::nbarsp], yerr = [lo_bars[0::nbarsp], hi_bars[0::nbarsp]], color = decomp["color"], linestyle="")

  plt.xlabel('Time Spent Computing CANDECOMP (s)')
  plt.ylabel('Relative Sum Of Squared Residual Error')
  plt.title('Sum Of Squared Error vs. Time To Factorize %s Tensor'%tensor["name"])
  plt.legend(loc='best')
  plt.savefig(plot_name)
  plt.clf()
