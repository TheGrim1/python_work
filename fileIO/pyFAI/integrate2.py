from __future__ import print_function
import sys
import matplotlib.pyplot as plt
import fabio
import pyFAI

def main(file_list):
    ai = pyFAI.AzimuthalIntegrator()
    ai.load('/data/id13/inhouse6/THEDATA_I6_1/d_2016-12-09_user_sc4415_smith/PROCESS/SESSION25/OUT/tr5_As40_40p_180um_1315_avg_00000.edf')
    ai.maskfile = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-12-09_user_sc4415_smith/PROCESS/post_exp/tr5_As40_40p_180um_1315_avg_00000-mask.edf'
    
    for f in file_list:
        data = fabio.open(f).data
        q, I = ai.integrate1d(data, 2000, unit='q_nm^-1')
        plt.plot(q, I)
        
    plt.show()
  
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python integrate.py <your_files.edf>')
        sys.exit(0)
    else:
        main(sys.argv[1:])
    
    
