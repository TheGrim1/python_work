import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40})
plt.rcParams.update({'figure.figsize': [4.0,6.0]})

import sys, os

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

from fileIO.datafiles.open_data import open_data
from fileIO.datafiles.save_data import save_data
from simplecalc.calc import normalize_xanes, combine_datasets
from simplecalc.linear_combination import do_component_analysis
import numpy as np

energyrange = [10360,10400]

def crop_energyrange(data,energyrange):
    energystart = np.searchsorted(data[:,0], energyrange[0], 'left')
    energyend   = np.searchsorted(data[:,0], energyrange[1], 'right')
    return data[energystart:energyend,:]

savepath = '/tmp_14_days/johannes1/lincom/plots'
specpath = '/tmp_14_days/johannes1/lincom/spectra/'


def derivative_edge(signal, axis, positive = True):
    ''' return value on axis for which the derivative of signal with respect to axis is maximum positive (or negative)'''

    incl = np.diff(signal)/np.diff(axis)

    maxincl = axis[np.argmax(incl)]
    
    return maxincl


#getting data

### from Gemas graph

gema, gemaheader = open_data(specpath + 'gema.dat', delimiter = '\t')

gema = crop_energyrange(gema,energyrange)

#e0gaas_gema = np.interp(0.5, gema[:,4], gema[:,0])
#e0ga2o3_gema = np.interp(0.5, gema[:,2], gema[:,0])
e0gaas_gema = derivative_edge(gema[:,4], gema[:,0])
e0ga2o3_gema = derivative_edge(gema[:,2], gema[:,0])

# XANES, XBIC

xanes, xanesheader  = open_data(specpath + 'redo_xanes.dat', delimiter = '\t')
xbic, xbicheader    = open_data(specpath + 'redo_xbic_nonstan.dat', delimiter = '\t')
#xanes200, xanes200header = open_data(specpath + 'redo_ga_nonstan.dat', delimiter = '\t')

xbic[:,0]  *= 1000
xanes[:,0] *= 1000
#xanes200[:,0] *= 1000

# ax1 = plt.gca()
# for i in range(1,len(xbic[0,:])):
#     ax1.plot(xbic[:,0],xbic[:,i])
# ax1.legend(range(4))
# plt.show()
### measured
gaasmeasfname = '/tmp_14_days/johannes1/lincom/gaas/GaAs_wafer_and_I0.dat'
gaasmeas, gaasmeasheader = open_data(gaasmeasfname, '\t')

gaasmeas_norm, gaasmeas_edge, step = normalize_xanes(np.asarray([gaasmeas[:,0],gaasmeas[:,1]]).transpose(), e0 = 10.367, postedge = (10.392,11.000), preedge = (0,10.360), verbose = False)
gaasmeas_norm[:,0] *= 1000
#e0gaas_gaasmeas = np.interp(0.5, gaasmeas_norm[:,1], gaasmeas_norm[:,0])
e0gaas_gaasmeas =  derivative_edge(gaasmeas_norm[:,1], gaasmeas_norm[:,0])
measured_correction = e0gaas_gema - e0gaas_gaasmeas

print measured_correction

xanes[:,0] += measured_correction
#xanes200[:,0] += measured_correction
xbic[:,0]  += measured_correction
gaasmeas_norm[:,0] += measured_correction

xbic        = crop_energyrange(xbic,energyrange)
xanes       = crop_energyrange(xanes,energyrange)
#xanes200    = crop_energyrange(xanes200,energyrange)



### GaAs literature
gaaslitfname = '/tmp_14_days/johannes1/lincom/gaas/gaaslit_norm.dat'
gaaslit, gaaslitheader = open_data(gaaslitfname, ' ')
gaaslit_norm, gaaslit_edge, step = normalize_xanes(np.asarray([gaaslit[:,0],gaaslit[:,1]]).transpose(), e0 = 10367, postedge = (10392,11000), preedge = (0,10360), verbose = False)
#e0gaas_gaaslit = np.interp(0.5, gaaslit_norm[:,1], gaaslit_norm[:,0])
e0gaas_gaaslit = derivative_edge( gaaslit_norm[:,1], gaaslit_norm[:,0])

gaaslit_norm[:,0] += e0gaas_gema - e0gaas_gaaslit
gaaslit_norm      = crop_energyrange(gaaslit_norm, energyrange)

### JPCB data

jpcb, jpcbheader  = open_data(specpath + 'JPCB102_10190.dat', delimiter = '\t')
#e0ga2o3_jpcb = np.interp(0.5, jpcb[:,11], jpcb[:,0])
e0ga2o3_jpcb = derivative_edge( jpcb[:-10,11], jpcb[:-10,0])

jpcb[:,0] += e0ga2o3_gema - e0ga2o3_jpcb
jpcb     = crop_energyrange(jpcb, energyrange)
jpcb[:,11] = jpcb[:,11]/jpcb[-10,11]

### PSS data

pss, pssheader = open_data(specpath + 'PSS_RRL_9_652.dat', delimiter = '\t')
#e0ga2o3_pss = np.interp(0.5, pss[:,1], pss[:,0])
e0ga2o3_pss = derivative_edge( pss[:,1], pss[:,0])
pss[:,0] += e0ga2o3_gema - e0ga2o3_pss
pss      = crop_energyrange(pss, energyrange)
pss[:,1] = pss[:,1]/pss[-1,1]

### gaga2o0

gaga2o3_orig, gaga2o3header = open_data(specpath + 'gaga2o3.dat', delimiter = '\t')
gaga2o3_1, dummy, dummy2    = normalize_xanes(np.asarray([gaga2o3_orig[:,0],gaga2o3_orig[:,1]]).transpose(), e0 = 10367, postedge = (10392,11000), preedge = (0,10360), verbose = False)
gaga2o3_2, dummy, dummy2    = normalize_xanes(np.asarray([gaga2o3_orig[:,0],gaga2o3_orig[:,2]]).transpose(), e0 = 10367, postedge = (10392,11000), preedge = (0,10360), verbose = False)
gaga2o3         = np.asarray([gaga2o3_1[:,0],gaga2o3_1[:,1],gaga2o3_2[:,1]]).transpose()
gaga2o3         = crop_energyrange(gaga2o3, energyrange)

#e0ga2o3_gaga2o3 = np.interp(0.5, gaga2o3[:,1], gaga2o3[:,0])
e0ga2o3_gaga2o3 = derivative_edge( gaga2o3[:,1], gaga2o3[:,0])
gaga2o3[:,0]   += e0ga2o3_gema - e0ga2o3_gaga2o3
#e0ga_gaga2o3    = np.interp(0.5, gaga2o3[:,2], gaga2o3[:,0])
e0ga_gaga2o3    = derivative_edge( gaga2o3[:,2], gaga2o3[:,0])
gaga2o3[:,2]   = gaga2o3[:,2]/gaga2o3[-1,2]
gaga2o3[:,2]   = gaga2o3[:,1]/gaga2o3[-1,1]

### Ga_metal.dat data

ga, gaheader = open_data(specpath + 'ga_metal.dat', delimiter = '\t')
ga           = crop_energyrange(ga, energyrange)
#e0ga_ga      = np.interp(0.5, ga[:,4], ga[:,0]) # where the ga is liquid
e0ga_ga      = derivative_edge( ga[:,4], ga[:,0]) # where the ga is liquid
ga[:,0]     += e0ga_gaga2o3 - e0ga_ga

### Ga_zeolite.dat data

gazeo, gazeoheader = open_data(specpath + 'ga_zeolite.dat', delimiter = '\t')
#e0ga2o3_zeo = np.interp(0.5, gazeo[:,2], gazeo[:,0])
e0ga2o3_zeo = derivative_edge( gazeo[:,2], gazeo[:,0])
gazeo[:,0] += e0ga2o3_gema - e0ga2o3_zeo
gazeo       = crop_energyrange(gazeo, energyrange)



### plotting


# # GaAs
# ax1 = plt.gca()
# ax1.plot(gema[:,0], gema[:,4])
# ax1.plot(gaasmeas_norm[:,0], gaasmeas_norm[:,1])
# ax1.plot(gaaslit_norm[:,0], gaaslit_norm[:,1])
# ax1.legend(['From Gemas graph','Measured data','Owens et al. RPC'])
# plt.title('GaAs')
# plt.show()

#  # Ga2O3, aligned to gema:
ax1 = plt.gca()
ax1.plot(gema[:,0], gema[:,2],linewidth =4)
ax1.plot(jpcb[:,0], jpcb[:,11],linewidth =2)
ax1.plot(pss[:,0], pss[:,1],linewidth =2)
ax1.plot(gazeo[:,0], gazeo[:,2],linewidth =2)
ax1.plot(gaga2o3[:,0], gaga2o3[:,1],linewidth =2)
ax1.hlines([0.5],ax1.get_xlim()[0],ax1.get_xlim()[1])
ax1.vlines([e0ga2o3_gema],ax1.get_ylim()[0],ax1.get_ylim()[1])
ax1.legend(['Martinez-Criado et al. [  ]','Nishi et al. [  ]','Revenant et al. [  ]','Wei et al. [  ]','Armbruester et al. [  ]'])
ax1.set_xticklabels(['{:d}'.format(int(x)) for x in ax1.get_xticks()])
ax1.set_xlabel('aligned energy [eV]')
ax1.set_ylabel('standardized signal [norm.]')
plt.title('beta-Ga2O3, Ga-K edge XANES literature')
plt.show()

### Ga metal

# ax1 = plt.gca()
# # for i in range(1,len(ga[0,:])):
# #     lw = 1
# #     if i == 3 or i == 4:
# #         lw = 1
# #     ax1.plot(ga[:,0], ga[:,i], linewidth = lw)
# ax1.plot(ga[:,0], ga[:,-1], linewidth = 3)

# ax1.legend([x for x in gaheader[1::]])    
# ax1.plot(gaga2o3[:,0], gaga2o3[:,2], linewidth = 3, color = 'black')
# ax1.vlines([e0gaas_gema],0,1.5) 

# plt.title('Ga_metal')
# plt.show()

# # find lincom for measuered XANES

# ax1 = plt.gca()
# for i in range(1,2):#,len(xanes[0,:])):
#     lw = 1
#     if i == 4:
#         lw = 3
#     ax1.plot(xanes[:,0], xanes[:,i], linewidth = lw)

# ax1.set_title('XRF all')
# # ax1.vlines([e0gaas_gema],0,1.5) 


# #ax1 = plt.gca()
# for i in range(1,2):#,len(xbic[0,:])):
#     ax1.plot(xbic[:,0], xbic[:,i])

# ax1.set_title('XBIC all')
# #ax1.vlines([e0gaas_gema],0,1.5)
# plt.show()


# ### XBIC and XANES
## made XANES_XBIC.png
ax1 = plt.gca()
#fig=plt.figure()
#fig.set_size_inches((3,3))
xanesavg = np.zeros(shape = (len(xanes[:,0]),2))
xbicavg = np.zeros(shape = (len(xbic[:,0]),2))

xanesavg[:,0] = xanes[:,0]
xbicavg[:,0] = xbic[:,0]

xanesavg[:,1] = np.sum(xanes[:,1::], axis = -1) /len(xanes[0,1::])
xbicavg[:,1] = np.sum(xbic[:,1::], axis = -1)      /len(xbic[0,1::])

xanesavg[:,1]=xanesavg[:,1]/np.max(xanesavg[:,1])*1.5
xbicavg[:,1]=xbicavg[:,1]/np.max(xbicavg[:,1])*1.5

ax1.plot(xanesavg[:,0], xanesavg[:,1], color = 'red', linewidth = 2)
ax1.plot(xbicavg[:,0], xbicavg[:,1], color = 'black', linewidth = 2)

ax1.set_xlim((10355,10400))
ax1.set_ylim((0,1.7))
ax1.legend(['Ga-K XRF','XBIC'])
ax1.set_xticks(ax1.get_xticks()[::2])
ax1.set_xticklabels(['{:d}'.format(int(x)) for x in ax1.get_xticks()])
ax1.set_xlabel('energy [eV]')
ax1.set_ylabel('normalized linear units [arb.]')

plt.show()
# ax1.vlines([e0gaas_gema],0,1.5) 


# ax1 = plt.gca()
# for i in range(1,len(xbic[0,:])):
#     ax1.plot(xbic[:,0], xbic[:,i])
#     ax1.plot(xanes200[:,0], xanes200[:,i])
# ax1.set_title('XBIC and XANES ROI')
# #ax1.vlines([e0gaas_gema],0,1.5)
# plt.show()

# # # lincoms:

# ax1.vlines([e0gaas_gema],0,2)


# ax1.legend(['measured','literature','gema gaas'])

# plt.ylabel('signal [norm.]')
# plt.xlabel('energy [eV]')
# plt.show()

# ### ## real combinations

# lincomplots = {}


# lincomplots.update({'GaAs':np.transpose(np.asarray([gema[:,0], gema[:,4]]))})
# lincomplots.update({'beta_Ga2O3':np.transpose(np.asarray([gema[:,0], gema[:,3]]))})
# lincomplots.update({'alpha_Ga2O3':np.transpose(np.asarray([gema[:,0], gema[:,2]]))})
# #lincomplots.update({'Ga:HO2':np.transpose(np.asarray([pss[:,0], pss[:,2]]))})
# #lincomplots.update({'ingazo200':np.transpose(np.asarray([pss[:,0], pss[:,3]]))})
# lincomplots.update({'Ga_metal':np.transpose(np.asarray([ga[:,0], ga[:,5]]))})
# #lincomplots.update({'Ga_3':np.transpose(np.asarray([ga[:,0], ga[:,3]]))})
# compos, compoheader = combine_datasets(lincomplots)

# savepath = '/tmp_14_days/johannes1/lincom/spectra/'
# ### save the lincom plots
# save_data(savepath + 'lin_components.dat', compos, compoheader, delimiter='\t')


# header  = ['scanno'] + compoheader[1::]
# results = []

# colors = ['b','g','r','c','m','y','k']

# for i in [2,6,8]:#range(1,len(xanes[0,:])):
#     dataxanes   = np.transpose(np.asarray([xanes[:,0], xanes[:,i]]))

#     beta,residual = do_component_analysis(dataxanes, compos, verbose = True)
#     results.append([i]+list(beta))
# results = np.asarray(results)

# save_data(savepath + 'xanes_lin_com.dat', results, header, delimiter='\t')



# results = []

# for i in range(1,len(xbic[0,:])):
#     dataxbic   = np.transpose(np.asarray([xbic[:,0], xbic[:,i]]))

#     beta,residual = do_component_analysis(dataxbic, compos, verbose = False)
#     results.append([i]+list(beta))

# dataxbic   = np.transpose(np.asarray([xbic[:,0], np.sum(xbic[:,1::],axis = -1)/4]))
# beta,residual = do_component_analysis(dataxbic, compos, verbose = True)
# results.append([9]+list(beta))

# results = np.asarray(results)
# save_data(savepath + 'xbic_lin_com.dat', results, header, delimiter='\t')



# # ax1 = plt.gca()

# # for i in range(1,len(results[0,:])):
# #     ax1.plot(results[:,0], results[:,i]+1, color = colors[i])


# # ax1.legend(header[1::])

# plt.show()
