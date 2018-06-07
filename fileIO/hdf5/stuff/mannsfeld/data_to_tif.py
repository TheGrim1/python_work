import sys, os
import h5py
import numpy as np
import PIL.Image as Image

SAVE_PATH = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-03_user_ch5315_mannsfeld/PROCESS/SESSION_sum/test/'

##### helper functions
def save_tif(data,savename,verbose=False):
    """
    gets a 2D numpy array of shape 
    (width, height)
    and saves it into imagefname 
    """
    img = Image.fromarray(data)
    if verbose:
        print("saving {}".format(os.path.realpath(savename)))
    img.save(savename)

def parse_master_fname(data_fname):
    master_path = os.path.dirname(data_fname)
    master_fname = os.path.basename(data_fname)[:os.path.basename(data_fname).find("data")]+'master.h5'
    return master_path + os.path.sep + master_fname

##### this does the work:
def save_frames_as_tif(args):

    master_fname = args[0]
    save_fname_tpl =args[1]
    verbose = args[2]
    
    if not os.path.exists(master_fname):
        raise ValueError('file not found '.format(master_fname))
        
    # open the h5 file:
    with h5py.File(master_fname) as master_h5:
        data_group = master_h5['entry/data']
        no_frames = 0
        #loop over the datasets in the h5 file:
        for key in data_group.keys():
            if verbose:
                print('reading {}'.format(key))
            # loop over the frames in each dataset:
            for i,frame in enumerate(data_group[key]):
                if verbose:
                    print('reading frame {}'.format(i))

                # threshhold:
                frame = np.where(frame>65000,0,frame)
                data = np.asarray(frame,dtype=np.int16)
                save_tif(data, save_fname_tpl.format(no_frames),verbose)
                no_frames +=1


##### this organises the work:
def main(data_fname):
    
    master_fname = parse_master_fname(data_fname)
    verbose = True    
    save_fname_tpl = SAVE_PATH + os.path.basename(master_fname)[0:os.path.basename(master_fname).find('_master')] +  '_{:06d}.tif'
    
    save_frames_as_tif([master_fname, save_fname_tpl, verbose])
    
if __name__=='__main__':
    if len(sys.argv) > 2:
        print('usage: python data_to_tif.py <datafilename>')
    main(sys.argv[1])
