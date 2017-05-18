import numpy as np
import matplotlib.pyplot as plt

from NPALSCANDECOMP import NPALSCANDECOMP
from NPGDCANDECOMP import NPGDCANDECOMP
from TFGDCANDECOMP import TFGDCANDECOMP


def heatmaps(X,A,B,C,R):
    result_als = NPALSCANDECOMP(X,R)
    result_gd = NPGDCANDECOMP(X,R)


