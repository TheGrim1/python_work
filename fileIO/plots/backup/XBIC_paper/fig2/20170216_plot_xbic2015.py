from __future__ import division
from builtins import range
from past.utils import old_div
import sys, os
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from simplecalc.calc import define_a_line_as_mask, normalize_self, get_fwhm
from fileIO.datafiles.save_data import save_data

import math
### side project ###
### making a alpha channel colormap:
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

# Choose colormap
cmap = pl.cm.gray_r

# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))

# Set alpha
my_cmap[:,-1] = np.linspace(1,0, cmap.N)

# Create new colormap
my_cmap = ListedColormap(my_cmap)



### data
fname2015 = '/tmp_14_days/johannes1/results/mg01_5_4_3/results/mg01_5_4_3/mg01_5_4_3.replace.h5'
h5f2015 = h5py.File(fname2015,'r')

xbic2015 = np.asarray(h5f2015['counters/zap_p201_IC/data'])[::-1,::-1,:]
ga2015 = np.asarray(h5f2015['detectorsum/Ga-K/data'])[::-1,::-1,:]
as2015 = np.asarray(h5f2015['detectorsum/As-K/data'])[::-1,::-1,:]
ni2015 = np.asarray(h5f2015['detectorsum/Ni-K/data'])[::-1,::-1,:]

### positions

samp2015range  = [np.asarray(h5f2015['/detectorsum/Ga-K/slow']),np.asarray(h5f2015['/detectorsum/Ga-K/fast'])]
samp2015range[0] = np.atleast_1d(np.asarray((samp2015range[0] - min(samp2015range[0]))))
samp2015range[1] = np.atleast_1d(np.asarray((samp2015range[1] - min(samp2015range[1]))))

posx, posy = np.meshgrid(samp2015range[1], samp2015range[0] , sparse=False)
pos = np.asarray(np.transpose([posy,posx]))
h5f2015.close()

# plotting

## masking the wire:
mask2015 = np.atleast_2d(define_a_line_as_mask(xbic2015[:,:,1].shape, inclination=old_div(-82.0,54), yintersect=105, width=4))

bias2015 = [-1, -0.5, -0.7, -0.6, -0.3, 0, 0.3, 0.6, 1.0, 2.0]


# for i, scanno in enumerate([0,2,3,4,5,6,9,10,11,12]):
#     fig, ax1 = plt.subplots()
    
#     print ('scan {}, with bias {}'.format(scanno, bias2015[i]))
#     ax1.matshow(np.where(mask2015,ga2015[:,:,scanno],0))
#     ax1.matshow(mask2015,cmap=my_cmap)
    
#     plt.show()
    
### integrate along line
maskedga   = np.zeros(shape = ga2015.shape)
maskedxbic = np.zeros(shape = ga2015.shape)
maskedas   = np.zeros(shape = ga2015.shape)




for i in range(ga2015.shape[2]):
    maskedga[:,:,i]   = (np.where(mask2015,ga2015[:,:,i],0))
    maskedxbic[:,:,i] = (np.where(mask2015,xbic2015[:,:,i],0))
    maskedas[:,:,i]   = (np.where(mask2015,as2015[:,:,i],0))

rotatedmask = nd.interpolation.rotate(mask2015,math.atan(old_div(-82.0,54))*180/math.pi)
rotatedga   = nd.interpolation.rotate(ga2015,math.atan(old_div(-82.0,54))*180/math.pi)
rotatedas   = nd.interpolation.rotate(as2015,math.atan(old_div(-82.0,54))*180/math.pi)
rotatedxbic = nd.interpolation.rotate(xbic2015,math.atan(old_div(-82.0,54))*180/math.pi)
rotatedpos  = nd.interpolation.rotate(pos,math.atan(old_div(-82.0,54))*180/math.pi)
rotatedpos[:,:,0] += -rotatedpos[58,10,0]
rotatedpos[:,:,1] += -rotatedpos[58,10,1]

pos = np.power(np.power(rotatedpos[:,:,1],2) + np.power(rotatedpos[:,:,1],2),0.5)

posline = old_div((np.sum(pos[55:62,:],axis = 0)),7)
galine = (np.sum(rotatedga[55:62,:,:],axis = 0))
asline = (np.sum(rotatedas[55:62,:,:],axis = 0))
xbicline = (np.sum(rotatedxbic[55:62,:,:],axis = 0))

posline  = (posline[20:80]) * 0.020
galine   = galine[20:80,:]
asline   = asline[20:80,:]
xbicline = xbicline[20:80,:]

garatioline = (old_div(galine,(galine+asline)))
gaasline = np.zeros(shape=(galine.shape))
xbic_norm = np.zeros(shape=(galine.shape))
for i in range(len(galine[1,:])):
    gaasline[:,i] = normalize_self((galine[:,i] + asline[:,i]))
    xbicline[:,i] = normalize_self(xbicline[:,i])

    
# fwhm = []
# for i, scanno in enumerate([0,2,3,4,5,6,9,10,11,12]):
#     fig, ax1 = plt.subplots()

#     plt.plot(posline, garatioline[:,scanno], color = 'red')
#     plt.plot(posline, gaasline[:,scanno], color = 'green')
#     plt.plot(posline, xbicline[:,scanno], color = 'blue')
#     currentfwhm = get_fwhm(np.asarray(np.transpose([posline,xbicline[:,scanno]])))
#     print bias2015[i],currentfwhm
#     fwhm.append(currentfwhm)
    
#     plt.title('scan {}, with bias {} and FWHM {}'.format(scanno, bias2015[i], currentfwhm))


#     #plt.show()

fig, ax1 = plt.subplots()
posline = posline - 0.7

plt.rcParams.update({'font.size': 30})
ax1.plot(posline[::-1], garatioline[:,4], color = 'red', linewidth = 2)
ax1.plot(posline[::-1], gaasline[:,4], color = 'green', linewidth = 2)

ax1.set_xlabel('position along nanowire')
ax1.set_ylabel('signal [norm.]')
ax1.plot(posline[::-1], xbicline[:,4], color = 'black', linewidth = 2)
plt.legend(['Ga/(Ga+As) ratio', '(Ga+As)','XBIC current at \n -0.7 V bias'],loc = 2)

ax2 = ax1.twinx()

ax2.set_ylabel('XBIC current [nA]', color = 'blue')
ax2.yaxis.label.set_color('blue')
ax2.tick_params(axis='y', colors='blue')
ax2.plot(posline[::-1], xbicline[:,4]*2.5, color = 'black', linewidth = 2)
yticklabels = []
for tick in ax2.get_yticks():
    yticklabels.append("${}$".format(tick)+ '$ \cdot 10^{-9}$')  
ax2.set_yticklabels(yticklabels)
ax1.set_xlim(0.0,0.8)



plt.show()


### save the data
data = np.zeros(shape = (len(posline[::-1]),4))
header = []

data[:,0] = posline[::-1]
header.append('energy [keV]')

data[:,1] = garatioline[:,4]
header.append('Ga/(Ga+As) ratio')

data[:,2] = gaasline[:,4]
header.append('(Ga+As) [norm]')

data[:,3] = xbicline[:,4]*2.5
header.append('XBIC curren at -0.7 V bias [nA]')


savepath = '/tmp_14_days/johannes1/results/mg01_5_4_3/singleXRF/'
save_data(savepath + 'figure2.dat', data, header, delimiter='\t')


