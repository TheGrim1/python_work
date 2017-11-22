from builtins import range
import sys, os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.figsize': [4.0,6.0]})

import scipy.ndimage as nd
import matplotlib.colors as colors
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.datafiles.open_data import open_data



fname = '/tmp_14_days/johannes1/lincom/spectra/xanes_lin_com.dat'

concentrations, header = open_data(fname,delimiter = '\t')

colors = ['g','r','blue','darkblue']

fig = plt.figure()
ax1 = plt.gca()

# toplot = np.zeros(shape = (concentrations.shape[0],concentrations.shape[1]-1))
# toplot[:,1] = concentrations[:,1]
# toplot[:,2] = concentrations[:,2]
# toplot[:,3] = concentrations[:,3] + concentrations[:,4]
toplot = concentrations
toplot[:,0] = 0

ycoord = list(range(len(toplot[:,1])))

btm = toplot[:,0]
wdth = [0.8] * len(ycoord)
for i in range(1,len(toplot[0,:])):
    ax1.barh(ycoord,toplot[:,i], height=wdth, left= btm, color = colors[i-1])
    btm += toplot[:,i]

ax1.set_yticklabels([])
ax1.set_title('linear combination')
ax1.set_xlim(0,1.0)

ax1.set_xlabel('relative content')

plt.tight_layout()


savename = '/tmp_14_days/johannes1/lincom/spectra/lincomplot'
plt.savefig(savename + '.svg', transparent=True)
plt.savefig(savename + '.png', transparent=True)
plt.savefig(savename + '.eps', transparent=True)
plt.savefig(savename + '.pdf', transparent=True)
plt.show()


