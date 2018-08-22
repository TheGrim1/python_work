
import numpy as np

import sys
sys.path.append('/data/id13/inhouse2/AJ/skript/')
from fileIO.datafiles import open_data, save_data

from pythonmisc.string_format import ListToFormattedString as list2string

SOURCE_FNAME = '/tmp_14_days/johanne/edf_edna/hkl.txt'
DEST_FNAME = '/tmp_14_days/johanne/edf_edna/hkl_reduced.txt'

def add_hkl_to_ax(ax, hkl_fname=DEST_FNAME):
    hkl_data, hkl_header = open_data(hklred_fname)
    for row in hkl_data:
        h,k,l,x,y,d,I = [x for x in row]
        ax.plot(x,y,'ro',mfc='none')
        ax.text(x+15,y+5,list2string([str(int(h)),str(int(k)),str(int(l))],1),fontdict={'color':'w'})

def main():
    f = file(SOURCE_FNAME,'r')
    f2 = file(DEST_FNAME,'w')
    all_lines = f.readlines()
    header = all_lines[0]
    print('header')
    print(header)
    for line in all_lines[1:]:
        print(line)
        f2.writelines(line[0:line.find('i')-1]+'\n')
    f2.flush()
    f2.close()
    f.close()
    hkl_data, hkl_header = open_data.open_data(DEST_FNAME)
    
    max_hkl = [0,0,0]
    
    for row in hkl_data:
        # print(row)
        No,h,k,l,x,y,d,I = [x for x in row]
        max_hkl = np.maximum(max_hkl,[int(abs(h)),int(abs(k)),int(abs(l))])
        # print([int(abs(h)),int(abs(k)),int(abs(l))])
        # print(max_hkl)

    print('max_hkl')
    print(max_hkl)
    hkl_cube = np.zeros(shape=tuple([2*x+1 for x in max_hkl]+[7]))

    for row in hkl_data:
        No,h,k,l,x,y,d,I = [x for x in row]
        old_val = hkl_cube[int(h)+max_hkl[0],int(k)+max_hkl[1],int(l)+max_hkl[2]]
        weights = [old_val[6],I]
        old_val[0] = int(h)
        old_val[1] = int(k)
        old_val[2] = int(l)
        old_val[3] = np.average([old_val[3],x],weights=weights)
        old_val[4] = np.average([old_val[4],y],weights=weights)
        old_val[5] = np.average([old_val[5],d],weights=weights)
        old_val[6] = np.sum(weights)
        print(No,weights)
        hkl_cube[int(h)+max_hkl[0],int(k)+max_hkl[1],int(l)+max_hkl[2]] = old_val

    print(hkl_cube)
    hkl_flat = hkl_cube.reshape(len(hkl_cube.flatten())/7,7)

    hkl_reduce = np.zeros(shape=(np.sum(np.where(hkl_cube[:,:,:,6]==0,0,1)),7))
    i=0
    for val in hkl_flat:
        if val[6]!=0:
            hkl_reduce[i]+=val
            i+=1
                
    save_data.save_data(DEST_FNAME,hkl_reduce,header=header[0:header.find('I')+1])


if  __name__ == '__main__':
    main()

