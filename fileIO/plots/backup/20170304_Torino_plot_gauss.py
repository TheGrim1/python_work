import sys, os
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import math
from simplecalc.gauss_fitting import gauss_func


def main():

    fig, (ax1)= plt.subplots()

    x = np.arange(100.0)
    y = gauss_func([100,50,5],x)

    plt.plot(x,y,color = 'black', linewidth = 3)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])

    
    plt.show()
    
if __name__ == "__main__":
    main()
