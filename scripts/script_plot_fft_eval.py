import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from multiprocessing import Pool
sys.path.append('/data/id13/inhouse2/AJ/skript')
from pythonmisc.worker_suicide import worker_init

 
def clean_angles(data):
    return calc.clean_outliers(data,0.3,3)

def do_plot(scanname,eval_h5, fft_h5, index):
    save_fname = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/{}/plots/'.format(scanname)+'{}_pos_{:04d}_{:04d}.png'.format(scanname,index[0],index[1])

    save_path = os.path.dirname(save_fname)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig, axes = plt.subplots(2,2)

    fig.set_tight_layout(True)
    fig.set_figheight(5.5)
    fig.set_figwidth(8)

    for axis in axes.flatten():
        axis.tick_params(axis='both',direction='in')
    
    ## the real image
    axes[0,0].set_title('camera image')
    axes[0,0].matshow(fft_h5['entry/real/frames'][index], vmin=10, vmax=120)
    axes[0,0].set_xlabel('x_pxl')
    axes[0,0].set_ylabel('y_pxl')
    
    ## the fft amp_spec
    axes[0,1].set_title('amplitude spectrum of image')
    axes[0,1].matshow(fft_h5['entry/fft/amplitude_spec'][index][220:512-220,:62])# vmin=0,vmax=1e6)

    axes[0,1].set_xlabel('horz_freq')
    axes[0,1].set_ylabel('vert_freq [offset]')


    ## the anaysed angle positions and phases
    axes[1,0].set_title('position')

    axes[1,0].set_ylabel('phi [deg]')
    axes[1,0].set_xlabel('kappa [deg]')

    axes[1,1].set_title('phase')
    axes[1,1].set_ylabel('period [deg]')
    axes[1,1].set_xlabel('frame')

    if scanname == 'dmesh_4':
        axes[1,0].set_xlim(0.23991-0.025,0.23991+0.025)
        axes[1,0].set_ylim(1.93491-0.025,1.93491+0.025)
    else:
        phi_mean = np.mean(eval_h5['entry/fft_eval/phi'])
        kappa_mean = np.mean(eval_h5['entry/fft_eval/kappa'])
        axes[1,0].set_xlim(kappa_mean-0.005,kappa_mean+0.005)
        axes[1,0].set_ylim(phi_mean-0.005,phi_mean+0.005)
        

    axes[1,1].set_xlim(0,10000)
    axes[1,1].set_ylim(-180,180)
    
    for i in range(index[0]):
        for j in range(100):

            phi = eval_h5['entry/fft_eval/phi'][i,j]
            kappa = eval_h5['entry/fft_eval/kappa'][i,j]

            axes[1,0].plot(kappa,phi,'bx')
            axes[1,1].plot(i*100+j,eval_h5['entry/fft_eval/phase'][i,j],'bo')

    i = index[0]
    for j in range(max(0,index[1]-1000),index[1]):
        phi = eval_h5['entry/fft_eval/phi'][i,j]
        kappa = eval_h5['entry/fft_eval/kappa'][i,j]
        
        axes[1,0].plot(kappa,phi,'bx')
        axes[1,1].plot(i*100+j,eval_h5['entry/fft_eval/phase'][i,j],'bo')
        
    j = index[1]
    phi = eval_h5['entry/fft_eval/phi'][i,j]
    kappa = eval_h5['entry/fft_eval/kappa'][i,j]

    axes[1,0].plot(kappa,phi,'rx')
    axes[1,1].plot(i*100+j,eval_h5['entry/fft_eval/phase'][i,j],'ro')

    print('saving {}'.format(save_fname))
    plt.savefig(save_fname,transparent=True)
    plt.close('all')

    
def do_last_plot(args):
    scanname=args[0]

    eval_fname= '/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/{}/{}_fft_eval.h5'.format(scanname,scanname)
    fft_fname= '/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/{}/{}_fft.h5'.format(scanname,scanname)
    
    with h5py.File(eval_fname,'r') as eval_h5:

        mapshape = eval_h5['entry/fft_eval/kappa'].shape

        indexes = np.meshgrid(*[range(x) for x in mapshape],indexing='ij')
        index_list = zip(*[x.flatten() for x in indexes])
        no_frames= len(index_list)

        
        with h5py.File(fft_fname,'r') as fft_h5:
            
            
            do_plot(scanname, eval_h5, fft_h5, index_list[-1])
      
def do_full_plot(args):
    scanname=args[0]

    eval_fname= '/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/{}/{}_fft_eval.h5'.format(scanname,scanname)
    fft_fname= '/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/{}/{}_fft.h5'.format(scanname,scanname)
    
    with h5py.File(eval_fname,'r') as eval_h5:

        mapshape = eval_h5['entry/fft_eval/kappa'].shape

        indexes = np.meshgrid(*[range(x) for x in mapshape],indexing='ij')
        index_list = zip(*[x.flatten() for x in indexes])
        no_frames= len(index_list)

        
        with h5py.File(fft_fname,'r') as fft_h5:
            
            for index in index_list:
                do_plot(scanname, eval_h5, fft_h5, index)
    
    
if __name__== '__main__':

    # scanname= sys.argv[1]

    scanname_list = []
    scanname_list += ['loopscan_{}'.format(x) for x in range(6,11)]
    scanname_list += ['dmesh_4']
    scanname_list += ['kmap_{}'.format(x) for x in [12]]

    # do_plot(scanname_list[0])

    todo_list = []
    for scanname in scanname_list:

        todo_list.append([scanname])
        
    # do_full_plot(todo_list[0])
    print(todo_list)
    pool=Pool(len(todo_list),worker_init(os.getpid()))
    pool.map(do_last_plot,todo_list)
    pool.close()
    pool.join()
                  
