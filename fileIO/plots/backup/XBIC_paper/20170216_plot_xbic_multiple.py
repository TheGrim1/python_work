from __future__ import division

from past.utils import old_div
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd


def main():
    e0 = 10.367
    fname200 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_200ms_Ga/MG154_fluoXAS_1.replace.h5'
    fname400 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_400ms_Ga/MG154_fluoXAS_1.replace.h5'

    h5f200 = h5py.File(fname200,'r')

    ### data
    stepg200 = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/xanes_step'])
    edgeg200 = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/xanes_edge'])
    edgegdiff200 = edgeg200 - 10.367
    ga200    = np.asarray(h5f200['/detectorsum/Ga-K_norm/data'])
    ni200    = np.asarray(h5f200['/detectorsum/Ni-K/data'])
    xbic200  = np.asarray(h5f200['/counters/zap_p201_Xbic_norm/data'])
    mask200  = np.where(ga200[:,:,50] > 5e-9,1,0)

    ### positions
    samp200  = [np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/sampz']),np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/sampy'])]
    samp200[0] = samp200[0] - min(samp200[0])
    samp200[1] = samp200[1] - min(samp200[1])


    stepx200 = np.asarray(h5f200['/counters/zap_p201_Xbic_norm_stan/xanes_step'])
    edgex200 = np.asarray(h5f200['/counters/zap_p201_Xbic_norm_stan/xanes_edge'])
    edgexdiff200 = edgex200 - 10.367

    h5f200.close()

    h5f400 = h5py.File(fname400,'r')

    ### data
    stepg400 = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/xanes_step'])
    edgeg400 = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/xanes_edge'])
    edgegdiff400 = edgeg400 - 10.367
    ga400    = np.asarray(h5f400['/detectorsum/Ga-K_norm/data'])
    ni400    = np.asarray(h5f400['/detectorsum/Ni-K/data'])
    mask400  = np.where(ga400[:,:,50] > 5e-9,1,0)

    ## position
    samp400  = [np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/sampz']),np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/sampy'])]
    samp400[0] = samp400[0] - min(samp400[0])
    samp400[1] = samp400[1] - min(samp400[1])

    h5f400.close()

    ### data
    fname2015 = '/tmp_14_days/johannes1/results/mg01_5_4_3/mg01_5_4_3.replace.h5'
    h5f2015 = h5py.File(fname2015,'r')

    xbic2015 = np.asarray(h5f2015['counters/zap_p201_IC/data'])[::-1,::-1,:]
    ga2015 = np.asarray(h5f2015['detectorsum/Ga-K/data'])[::-1,::-1,:]
    ni2015 = np.asarray(h5f2015['detectorsum/Ni-K/data'])[::-1,::-1,:]
    as2015 = np.asarray(h5f2015['detectorsum/As-K/data'])[::-1,::-1,:]
    maskga2015 = np.where(np.sum(ga2015, axis=-1) > 200, 1, 0)
    ### positions

    samp2015  = [np.asarray(h5f2015['/detectorsum/Ga-K/slow']),np.asarray(h5f2015['/detectorsum/Ga-K/fast'])]
    samp2015[0] = samp2015[0] - min(samp2015[0])
    samp2015[1] = samp2015[1] - min(samp2015[1])

    h5f2015.close()

    ### masking

    from matplotlib.colors import ListedColormap
    import matplotlib.pylab as pl
    cmap = pl.cm.hot_r
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    ## plotting

    data1 = (np.sum(ga400, axis =-1))
    data1 = old_div(data1,np.max(data1))
    
    #    data1 = np.where(edgexdiff200 > 0, np.where(mask200,edgexdiff200*1000,0),0)
    data2 = np.where(edgegdiff200 > 0, np.where(mask200,edgegdiff200*1000,0),0)
    data3 = np.where(edgegdiff400 > 0, np.where(mask400,edgegdiff400*1000,0),0)


    data4 = (np.sum(ga2015, axis =-1))
    data5 = (np.sum(as2015, axis =-1))
    
    sumgaas = data4 + data5

    data5 = np.where(sumgaas > 0, old_div(data4,sumgaas), 0) 
    data4 = old_div(data4,np.max(data4))
    data5 = old_div(data5,np.max(data5))

    data6 = xbic2015[:,:,4]
    data6 = old_div(data6,np.max(data6))

    ### scale images so that 1 pxl = 2nm:
    data1 = nd.zoom(data1, 15, order = 0)
    data2 = nd.zoom(data2, 25, order = 0)
    data3 = nd.zoom(data3, 15, order = 0)
    data4 = nd.zoom(data4, 10, order = 0)
    data5 = nd.zoom(data5, 10, order = 0)
    data6 = nd.zoom(data6, 10, order = 0)

    shift2015 = [-150,160]

    #### select what to plot
    cropslice = []
    cropslice.append(min(data1.shape[0], data2.shape[0], data3.shape[0], data4.shape[0], data5.shape[0], data6.shape[0]))
    cropslice.append(min(data1.shape[1], data2.shape[1], data3.shape[1], data4.shape[1], data5.shape[1], data6.shape[1]))
    
    image1 = data1[400:cropslice[0],0:cropslice[1]]
    image2 = data2[400:cropslice[0],0:cropslice[1]]
    image3 = data3[400:cropslice[0],0:cropslice[1]]
    
    image4 = data4[shift2015[0]+400:cropslice[0]+shift2015[0],shift2015[1]:cropslice[1]+shift2015[1]]
    image5 = data5[shift2015[0]+400:cropslice[0]+shift2015[0],shift2015[1]:cropslice[1]+shift2015[1]]
    image6 = data6[shift2015[0]+400:cropslice[0]+shift2015[0],shift2015[1]:cropslice[1]+shift2015[1]]


    cmax1 = max(np.max(image1),np.max(image2), np.max(image3))
    cmin1 = min(np.min(image2),np.min(image1), np.min(image3))


    ## setup figure


    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(nrows=2,
                                                          ncols=3
                                                       #gridspec_kw={'height_ratios':[1,1,1,1,1,1]})
                                                       )
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)


    ### image1 ######################

    im1 = ax1.matshow(image1, vmin = 0, vmax = 1)
    ax1.set_title('a) XRF Ga (2017)', loc='left')

    ### ticklabels1
    x1ticklabels = []
    ax1.locator_params(nbins=5, axis='x')
    for tick in ax1.get_xticks():
        x1ticklabels.append("{0:.2f}".format(tick *0.002))
    ax1.set_xticklabels(x1ticklabels)

    yticklabels = []
    for tick in ax1.get_yticks():
        yticklabels.append("{0:.2f}".format(tick *0.002))    
    ax1.set_yticklabels(yticklabels[::-1])
   
    ax1.set_ylabel('y position [um]')
    ax1.tick_params(labelleft = 'on', labelbottom = 'off', labeltop = 'off')

    # colorbar1
    divider1 = make_axes_locatable(ax1)
    cax1     = divider1.append_axes("right", size="10%", pad = 0.05)
    cbar1    = plt.colorbar(im1, cax=cax1, orientation = 'vertical')
    cbar1.set_label('XBIC signale [norm.]')



    ### image2 #########################

    im2 = ax2.matshow(image2, vmin = cmin1, vmax = cmax1)
    ax2.set_title('b) Ga XRF XANES 200ms', loc='left')

    ### colorbar2

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="10%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2, orientation = 'vertical')
    cbar2.set_label('absorption edge shift [eV]')

    ### ticklabels2
    x2ticklabels = []
    ax2.locator_params(nbins=5, axis='x')
    for tick in ax2.get_xticks():
        x2ticklabels.append("{0:.2f}".format(tick *0.002))
    ax2.set_xticklabels(x2ticklabels)


    #ax2.set_yticklabels(ax1.get_yticks() * 0.05 * (scalingfactor1[1]*10))
#    ax2.set_xlabel('x position [um]')
    # ax2.set_ylabel('y position [um]')
    ax2.tick_params(labelleft = 'off', labelbottom = 'off', labeltop = 'off')


    ### image3 ######################

    im3 = ax3.matshow(image3, vmin = cmin1, vmax = cmax1)
    ax3.set_title('c) Ga XRF XANES 400ms', loc='left')

    ### ticklabels3
    x3ticklabels = []
    ax3.locator_params(nbins=5, axis='x')
    for tick in ax3.get_xticks():
        x3ticklabels.append("{0:.2f}".format(tick *0.002))
    ax3.set_xticklabels(x3ticklabels)

    #ax3.set_yticklabels(ax3.get_yticks() * 0.03 / 10)
#    ax3.set_xlabel('x position [um]')
    # ax3.set_ylabel('y position [um]')
    ax3.tick_params(labelleft = 'off', labelbottom = 'off', labeltop = 'off')

    # colorbar3
    divider3 = make_axes_locatable(ax3)
    cax3     = divider3.append_axes("right", size="10%", pad = 0.05)
    cbar3    = plt.colorbar(im3, cax=cax3, orientation = 'vertical')
    cbar3.set_label('absorption edge shift [eV]')


    ### image4 ######################

    im4 = ax4.matshow(image4, vmin = 0, vmax = 1)
    ax4.set_title('d) XRF Ga (2015)', loc='left')

    ### ticklabels4
    x4ticklabels = []
    ax4.locator_params(nbins=5, axis='x')
    for tick in ax4.get_xticks():
        x4ticklabels.append("{0:.2f}".format(tick *0.002))
    ax4.set_xticklabels(x4ticklabels)


   
    yticklabels = []
    for tick in ax4.get_yticks():
        yticklabels.append("{0:.2f}".format(tick *0.002))    
    ax4.set_yticklabels(yticklabels[::-1])

    ax4.set_xlabel('x position [um]')
    ax4.set_ylabel('y position [um]')
    ax4.tick_params(labelleft = 'on',labelbottom = 'on', labeltop = 'off')
    
    # colorbar4
    divider4 = make_axes_locatable(ax4)
    cax4     = divider4.append_axes("right", size="10%", pad = 0.05)
    cbar4    = plt.colorbar(im4, cax=cax4, orientation = 'vertical')
    cbar4.set_label('Ga XRF signal [norm.]')

    ### image5 ######################

    im5 = ax5.matshow(image5, vmin = 0, vmax = 1)
    ax5.set_title('e) Ga/(Ga + As) (2015)', loc='left')

    ### ticklabels5
    x5ticklabels = []
    ax5.locator_params(nbins=5, axis='x')
    for tick in ax5.get_xticks():
        x5ticklabels.append("{0:.2f}".format(tick *0.002))
    ax5.set_xticklabels(x5ticklabels)

    #ax5.set_yticklabels(ax5.get_yticks() * 0.03 / 10)
    ax5.set_xlabel('x position [um]')
    # ax5.set_ylabel('y position [um]')
    ax5.tick_params(labelleft = 'off', labelbottom = 'on', labeltop = 'off')

    # colorbar5
    divider5 = make_axes_locatable(ax5)
    cax5     = divider5.append_axes("right", size="10%", pad = 0.05)
    cbar5    = plt.colorbar(im5, cax=cax5, orientation = 'vertical')
    cbar5.set_label('Ga/(Ga + As) ratio')

    ### image6 ######################

    im6 = ax6.matshow(image6, vmin = 0, vmax = 1)
    ax6.set_title('f) XBIC (2015)', loc='left')

    ### ticklabels6
    x6ticklabels = []
    ax6.locator_params(nbins=5, axis='x')
    for tick in ax6.get_xticks():
        x6ticklabels.append("{0:.2f}".format(tick *0.002))
    ax6.set_xticklabels(x6ticklabels)

    #ax6.set_yticklabels(ax6.get_yticks() * 0.03 / 10)
    ax6.set_xlabel('x position [um]')
    # ax6.set_ylabel('y position [um]')
    ax6.tick_params(labelleft = 'off', labelbottom = 'on', labeltop = 'off')

    # colorbar6
    divider6 = make_axes_locatable(ax6)
    cax6     = divider6.append_axes("right", size="10%", pad = 0.05)
    cbar6    = plt.colorbar(im6, cax=cax6, orientation = 'vertical')
    cbar6.set_label('XBIC signal [norm.]')


    fig.set_figheight(8)
    fig.set_figwidth(16)
    plt.tight_layout()
    plt.savefig('/tmp_14_days/johannes1/maps1.svg', transparent=True)
    plt.savefig('/tmp_14_days/johannes1/maps1.png', transparent=True)
    plt.savefig('/tmp_14_days/johannes1/maps1.eps', transparent=True)
    plt.savefig('/tmp_14_days/johannes1/maps1.pdf', transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
