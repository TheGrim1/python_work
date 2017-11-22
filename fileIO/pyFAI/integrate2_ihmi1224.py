from __future__ import print_function
import sys
import matplotlib.pyplot as plt
import fabio
import pyFAI

def plot_h5(data,
            index=0, 
            title = "Title"):
    dimension = len(data.shape)
    print("dimension of data to be plotted is %s" % dimension)
    if dimension == 3:
        plt.imshow(data[index,:,:], interpolation = 'none')
    elif dimension == 2:
        plt.imshow(data[:,:], interpolation = 'none')
    elif dimension not in (2,3):
        print("invalid data for plotting \ntitle  : %s\n%s" % (title, dimension))
    plt.clim(0,10)

    plt.show()

    plt.title(title)

def main(file_list):
    ai = pyFAI.AzimuthalIntegrator()
    ai.load('/data/id13/inhouse5/THEDATA_I5_1/d_2016-06-09_inh_ihmi1224/PROCESS/ANALYSIS/log/calib3_test.poni')
#    ai.maskfile = '../PROCESS/SESSION3/calibration/total_mask.edf'
    
    for f in file_list:
        data = fabio.open(f).data

        regrouped = ai.integrate2d(data, npt_rad=500, npt_azim=500, unit='2th_deg', radial_range=(0,45))        
        
        plot_h5(regrouped[0])

        print("regrouped[1].shape %s" % regrouped[1].shape)
   
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python integrate.py <your_files.edf>')
        sys.exit(0)
    else:
        main(sys.argv[1:])
    
    
