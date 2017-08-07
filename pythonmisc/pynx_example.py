
from pynx.ptycho.runner.id13 import PtychoRunnerScanID13, params
print('Import OK')

from pylab import rcParams

rcParams['figure.figsize'] = (12, 8)  # Figure size for inline display

import os

## this only works in an ipython environment for some reason?

def main():
    params['saveprefix']='vincent/ResultsScan%04d/Run%04d'

    params['specfile']='/gz/data/id13/inhouse7/THEDATA_I7_1/d_2017-06-28_user_ma3324_godard/DATA/siemens_ptycho/siemens_ptycho.dat'
    params['scan']=9
    params['h5file']='/gz/data/id13/inhouse7/THEDATA_I7_1/d_2017-06-28_user_ma3324_godard/DATA/AUTO-TRANSFER/eiger1/siemens_ptycho_2104_master.h5'
    params['probe']='40e-6x40e-6,0.01'  # Starting from a simulated probe, here described as gaussian
    params['defocus']=-1e-6        #Defocus the calculated initial probe by 100 microns
    params['detectordistance']=1.92
    params['ptychomotors']='nnp6,nnp5,-y,x'   # id13 motors
    params['algorithm'] = '20DM'    # Begin with difference map, object only
    params['object'] = 'random,0.95,1,0,0.2'   # High energy => start from an almost-flat object
    params['verbose'] = 10          # Print every 10 cycles
    params['liveplot'] = True       # Liveplot updated at the end of each cell.
    #params['moduloframe'] =2,0      # Take only half frames (faster, less memory used)
    params['maxsize'] = 256         # Use only 256 pixels (faster, less memory used)
    beamcenter = (580,1378)
    halfsize   = int(params['maxsize'] / 2)
    params['roi'] =  beamcenter[0] - halfsize , beamcenter[0] + halfsize, beamcenter[1] - halfsize, beamcenter[1]+ halfsize     # xmin, xmax, ymin, ymax. Data is partially corrupted so use manual ROI

    ws = PtychoRunnerScanID13(params, params['scan'])

    ws.load_data()
    ws.center_crop_data()  # Crop raw data
    ws.prepare()  # This will prepare the initial object and probe

    ws.run()
    ws.run_algorithm('probe=1,40DM')


    ws.run_algorithm('50AP')
    ws.run_algorithm('50ML')
    ws.run_algorithm('Analyze')  # Analyze the optimized probe (modes, propagation)

    params['loadprobe'] = os.path.join(os.path.dirname(params['saveprefix']%(params['scan'], 0)),'latest.npz')
    ws.prepare()
    ws.run()
    ws.run_algorithm('probe=1,80DM')
    ws.run_algorithm('100AP')
    ws.run_algorithm('100ML')

    ws.run_algorithm('Analyse')
    ws.run_algorithm('nbprobe=3,100AP')

    ws.run_algorithm('100ML')

    ws.save_plot(ws._run, display_plot=True) # This makes a more elaborate plot than 'liveplot'
    ws.run_algorithm('Analyze')  # Analyze the optimized probe (modes, propagation)

    ws.plot_llk_history()


if __name__=='__main__':
    main()


## from vincents email:

def do_this_in_terminal:
    """
1) on lid13gpu1, activate the PyNX virtual environment (now installed
serf-wide on /sware):

 source /sware/exp/pynx/devel/bin/activate

[note that I would not normally recommend using the ‘devel’ version,
rather the ‘stable’, but there are a few new features and
corrections]

2) run a  ptychography analysis script, which for id13 would look like
(a bit long) - I give here the example from a few months old data, I
hope the format did not change :

 cd /users/opid13/favre/UM2017/

    pynx-id13pty.py h5file=/gz/data/id13/inhouse7/THEDATA_I7_1/d_2017-06-28_user_ma3324_godard/DATA/AUTO-TRANSFER/eiger1/siemens_ptycho_2105_master.h5 specfile=/gz/data/id13/inhouse7/THEDATA_I7_1/d_2017-06-28_user_ma3324_godard/DATA/siemens_ptycho/siemens_ptycho.dat scan=13 detectordistance=1.90 ptychomotors=nnp5,nnp6,-x,y probe=40e-6x40e-6,0.01 defocus=-100e-6 object=random,.9,1,-.2,.2 algorithm=20DM,probe=1,100DM,100AP,100ML,analyze maxsize=256 verbose=10 save=all saveplot liveplot


 You can also use “
algorithm=20DM,probe=1,80DM,100AP,100ML,nbprobe=3,100AP,100ML,analyze”
if you want to analyse with 3 modes, but the aperture on id13 is such that I do not expect you need it.

    """
    
