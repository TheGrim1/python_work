import sys, os
sys.path.append('/data/id13/inhouse2/AJ/skript')
from fileIO.datafiles.save_data import open_data, save_data

def main(lut_fname):

    data, header =open_data(lut_fname)
    print('lookuptable header:')
    print(header)

    data[:,1:] *= 1000

    i = 0
    for i,mot in enumerate(header):
        if mot =='y':
            posy=i
            
    data[:,posy] *= -1

    save_path = os.path.dirname(lut_fname)
    savename_list = os.path.basename(lut_fname).split('.')
    savename_list[0]+='_piezo'
    save_fname = os.path.sep.join([save_path,'.'.join([savename_list[0],savename_list[1]])])

    save_data(save_fname, data,header = header)

    print('lookuptable values multiplied by 1000 and saved in')
    print(save_fname)
    
if __name__=='__main__':
    usage = 'python multiply_lut_by_1000.py <your lut filename>'

    if len(sys.argv)!= 1:
        print(usage)
    lut_fname = os.path.realpath(sys.argv[1])
    if not os.path.exists(lut_fname):
        print('cant find lookupfile '.format(lutfname))
        print(usage)

    main(lut_fname)
