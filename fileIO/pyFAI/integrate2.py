import sys
import matplotlib.pyplot as plt
import fabio
import pyFAI

def main(file_list):
    ai = pyFAI.AzimuthalIntegrator()
    ai.load('/data/id13/inhouse5/THEDATA_I5_1/d_2016-06-09_inh_ihmi1224/PROCESS/ANALYSIS/log')
    ai.maskfile = '../PROCESS/SESSION3/calibration/total_mask.edf'
    
    for f in file_list:
        data = fabio.open(f).data
        q, I = ai.integrate1d(data, 2000, unit='q_nm^-1')
        plt.plot(q, I)
        
    plt.show()
  
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: python integrate.py <your_files.edf>'
        sys.exit(0)
    else:
        main(sys.argv[1:])
    
    
