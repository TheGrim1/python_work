import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from multiprocessing import Pool
sys.path.append('/data/id13/inhouse2/AJ/skript')
from pythonmisc.worker_suicide import worker_init
import scipy.ndimage.filters as fil


def do_plot_loopscan(scanname, source_h5, index):
    save_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/vlm/{}/plots/'.format(scanname)+'{}_pos_{:04d}.png'.format(scanname,index)

    save_path = os.path.dirname(save_fname)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig, axes = plt.subplots(1,3)

    fig.set_tight_layout(True)
    fig.set_figheight(3.5)
    fig.set_figwidth(12)

    for axis in axes.flatten():
        axis.tick_params(axis='both',direction='in')
    ## the real image
    axes[0].set_title('camera image')
    axes[0].matshow(source_h5['entry/image/frames'][index][::-1], vmin=10, vmax=110)
    axes[0].set_xlabel('y_pxl')
    axes[0].set_ylabel('z_pxl')
    
    ## the position
    axes[1].set_title('position y - z')
    axes[1].set_xlabel('y position [um]')
    axes[1].set_ylabel('z position [um]')
    axes[1].set_xlim(-0.4,0.4)
    axes[1].set_ylim(-0.4,0.4)

    # for i in range(max(0,index-1000),index):
    for i in range(index):
        y_pos = source_h5['entry/positions/del_x'][i]
        z_pos = source_h5['entry/positions/del_z'][i]
        axes[1].plot(y_pos,z_pos,'bo')

    y_pos = source_h5['entry/positions/del_x'][index]
    z_pos = source_h5['entry/positions/del_z'][index]
    axes[1].plot(y_pos,z_pos,'rx')
    
    ## the analysed angle positions and phases
    axes[2].set_title('y position vs time')
    axes[2].set_xlabel('time [s]')
    axes[2].set_ylabel('y position [um]')
    axes[2].set_ylim(-0.4,0.4)
    time_arr = np.asarray(source_h5['entry/positions/time_of_frame'])
    end_time = time_arr.flatten()[-1]
    start_time = time_arr.flatten()[0]
    axes[2].set_xlim(0,end_time-start_time)

    del_x = np.asarray(source_h5['entry/positions/del_x']).flatten()
    axes[2].plot(time_arr.flatten()[:index]-start_time,del_x[:index],'b-')

    print('saving {}'.format(save_fname))
    plt.savefig(save_fname,transparent=True)
    plt.close('all')

    

def do_plot_kmap(scanname,source_h5, index):
    save_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/vlm/{}/plots/'.format(scanname)+'{}_pos_{:04d}_{:04d}.png'.format(scanname,index[0],index[1])

    save_path = os.path.dirname(save_fname)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig, axes = plt.subplots(1,4)

    fig.set_tight_layout(True)
    fig.set_figheight(5)
    fig.set_figwidth(12)

    for axis in axes.flatten():
        axis.tick_params(axis='both',direction='in')
    
    ## the real image
    axes[0].set_title('camera image')
    axes[0].matshow(source_h5['entry/image/frames'][index][::-1], vmin=10, vmax=120)
    axes[0].set_xlabel('y_pxl')
    axes[0].set_ylabel('z_pxl')
    
    ## the position
    axes[1].set_title('delta y')
    axes[1].set_xlabel('scan position y [um]')
    axes[1].set_ylabel('scan position z [um]')
    axes[1].set_xticks([range(0,101,20)])
    axes[1].set_yticks([range(0,101,20)])
    axes[1].set_xticklabels(['.1f'.format(x/20) for x in axes[1].get_xticks()])
    axes[1].set_yticklabels(['.1f'.format(x/20) for x in axes[1].get_yticks()])
    
    ## the position
    axes[2].set_title('delta z')
    axes[2].set_xlabel('scan position y [um]')
    axes[2].set_ylabel('scan position z [um]')
    axes[2].set_xticks([range(0,101,20)])
    axes[2].set_yticks([range(0,101,20)])
    axes[2].set_xticklabels(['.1f'.format(x/20) for x in axes[2].get_xticks()])
    axes[2].set_yticklabels(['.1f'.format(x/20) for x in axes[2].get_yticks()])
    

    ## the analysed angle positions and phases
    axes[3].set_title('y position vs time')
    axes[3].set_xlabel('time [s]')
    axes[3].set_ylabel('y position [um]')
    axes[3].set_ylim(-0.8,0.8)
    time_arr = np.asarray(source_h5['entry/positions/time_of_frame'])
    end_time = time_arr.flatten()[-1]
    start_time = time_arr.flatten()[0]
    axes[3].set_xlim(0,end_time-start_time)

    del_x = np.asarray(source_h5['entry/positions/del_x']).flatten()
    axes[3].plot(time_arr.flatten()[:index[0]*100+index[1]]-start_time,del_x.flatten()[:index[0]*100+index[1]],'b-')

    
    full_x = np.asarray(source_h5['entry/positions/del_x'])
    full_z = np.asarray(source_h5['entry/positions/del_z'])
    mask = np.zeros_like(full_x,dtype=bool)

    for i in range(index[0]):
        mask[:i] = 1
    i = index[0]
    mask[i,:index[1]] = 1

    axes[1].matshow(full_x*mask,vmin=-0.7,vmax=0.7)
    axes[2].matshow(full_z*mask,vmin=-0.7,vmax=0.7)

    print('saving {}'.format(save_fname))
    plt.savefig(save_fname,transparent=True)
    plt.close('all')

    
    
def do_full_plot(args):
    scanname=args[0]

    source_fname= '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/vlm/{}/{}_vlm1.h5'.format(scanname,scanname)
    
    with h5py.File(source_fname,'r') as source_h5:

        mapshape = (100,100)

        indexes = np.meshgrid(*[range(x) for x in mapshape],indexing='ij')
        index_list = zip(*[x.flatten() for x in indexes])
        no_frames= len(index_list)

        for i, index in enumerate(index_list):
            if scanname.find('opscan')>0:
                do_plot_loopscan(scanname, source_h5, i)
            else:
                do_plot_kmap(scanname, source_h5, index)
       
def do_last_plot(args):
    scanname=args[0]

    source_fname= '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/vlm/{}/{}_vlm1.h5'.format(scanname,scanname)
    
    with h5py.File(source_fname,'r') as source_h5:

        mapshape = (100,100)

        indexes = np.meshgrid(*[range(x) for x in mapshape],indexing='ij')
        index_list = zip(*[x.flatten() for x in indexes])
        no_frames= len(index_list)

        if scanname.find('opscan')>0:
            do_plot_loopscan(scanname, source_h5, len(index_list)-1)
        else:
            do_plot_kmap(scanname, source_h5, index_list[-1])

def do_noise_plot(args):
     scanname_list=args[0]
     labels = args[1] 
     x_del_list=[]
     timestep_list = []
     fig,ax = plt.subplots(1)
     fft_list=[]
     fig.set_tight_layout(True)
     fig.set_figheight(5)
     fig.set_figwidth(6)

     
     
     for i,scanname in enumerate(scanname_list):
         
         source_fname= '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/vlm/{}/{}_vlm1.h5'.format(scanname,scanname)
         with h5py.File(source_fname) as sh5:
             x_del = np.asarray(sh5['entry/positions/del_x'][5:][5:]).flatten()
             x_del_list.append(x_del)
             time_ds = np.asarray(sh5['entry/positions/time_of_frame']).flatten()
             timestep = (time_ds[199]-time_ds[100])/100
             timestep_list.append(timestep)
             power_spec=np.abs((np.fft.rfft(x_del)))**2
             fft_list.append(power_spec)
             f_ax = np.linspace(0.0,len(power_spec)-1,len(power_spec))/len(x_del)/timestep
             power_plot = np.log(fil.gaussian_filter(power_spec,1))
             power_plot = np.where(power_plot<0,0,power_plot)
             ax.plot(f_ax,power_plot,label=labels[i])
             
             ax.set_ylim(0,10)
             ax.set_xlim(0,25)
     ax.legend(ncol=2)
     ax.set_xlabel('freq [Hz]')
     ax.set_ylabel('log power spectrum [shifted]')
     save_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/vlm/noise_20ms_spec.png'
     fig.savefig(save_fname,transparent=True)
     print('saved {}'.format(save_fname))
     plt.close('all')
         
if __name__== '__main__':

    # scanname= sys.argv[1]
    # do_full_plot([scanname])
    scanname_list = []
    # scanname_list += ['loopscan_{}'.format(x) for x in range(14,45)]
    scanname_list.append('loopscan_19')
    # scanname_list += ['kmap_{}'.format(x) for x in [25,26,27,28,29,40,41,42]]
    scanname_list += ['kmap_{}'.format(x) for x in [27, 42]]
    # labels = ['loopscan','MARS 10 ms','MARS 11 ms', 'MARS 20 ms', 'HERA 10 ms', 'HERA 11 ms', 'HERA 20 ms']
    # labels = ['loopscan','MARS 10 ms','MARS 11 ms', 'MARS 20 ms', 'MARS 110 ms', 'MARS 210 ms']#, 'HERA 10 ms', 'HERA 11 ms', 'HERA 20 ms']
    labels = ['loopscan 20 ms', 'MARS 20 ms', 'HERA 20 ms']
    todo_list = []
    for i,scanname in enumerate(scanname_list):
        
        todo_list.append(scanname)
        # do_last_plot([scanname])

    do_noise_plot([todo_list,labels])
    # do_full_plot(todo_list[0])
    # print(todo_list)
    # pool=Pool(len(todo_list),worker_init(os.getpid()))
    # pool.map(do_last_plot,todo_list)
    # pool.close()
    # pool.join()
                  
