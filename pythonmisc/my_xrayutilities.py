import numpy as np
import xrayutilities as xu
import sys,os
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

from simplecalc.slicing import rebin, troi_to_slice

def get_id01_experiment(par_dict):    
    qconv=xu.experiment.QConversion(['y+','z+'],['y+'],[1,0,0])
    hxrd = xu.HXRD([-1,0,0],[0,0,1],qconv=qconv,sampleor='sam')

    
    hxrd._set_energy(par_dict['energy_keV'] * 1000.0)
    
    hxrd._A2QConversion.init_area('z-',
                                  'y+',
                                  cch1=(par_dict['cch1']-par_dict['troi'][0][0])/par_dict['bin_size'],
                                  cch2=(par_dict['cch2']-par_dict['troi'][0][1])/par_dict['bin_size'],
                                  distance=par_dict['distance'],
                                  pwidth1=par_dict['pixel_width']*par_dict['bin_size'],
                                  pwidth2=par_dict['pixel_width']*par_dict['bin_size'],
                                  Nch1=par_dict['Nch1'],
                                  Nch2=par_dict['Nch2'])
    
    return hxrd


def get_id13_experiment(troi, troi_poni):
    
    qconv=xu.experiment.QConversion(['z+','x+','z+'],[],[1,0,0])
    hxrd = xu.HXRD([-1,0,0],[0,1,0],qconv=qconv,sampleor='sam')

    hxrd._set_wavelength(troi_poni['Wavelength']*1e10)
    
    hxrd._A2QConversion.init_area('z-',
                                  'y+',
                                  cch1=troi_poni['Poni1']/troi_poni['PixelSize1'],
                                  cch2=troi_poni['Poni2']/troi_poni['PixelSize2'],
                                  distance=troi_poni['Distance'],
                                  pwidth1=troi_poni['PixelSize1'],
                                  pwidth2=troi_poni['PixelSize2'],
                                  Nch1=troi[1][0],
                                  Nch2=troi[1][1])
    
    return hxrd

def get_maxmin_Q(Theta,kappa,phi,expriment=None,troi=None,troi_poni=None):
    '''
    needs experiment or (troi, troi_poni)
    Theta, kappa, phi can be type array, must have same shape or len(1).
    '''
    if type(experiment)==type(None):
        experiment = get_id13_experiment(troi, troi_poni)

    qx,qy,qz = experiment.Ang2Q.area(Theta,kappa,phi)
    
    return [[qx.min(), qx.max()], [qy.min(), qy.max()], [qz.min(), qz.max()]]


def get_maxmin_Q_id01(Theta,kappa,phi,expriment=None,troi=None,troi_poni=None):
    '''
    needs experiment or (troi, troi_poni)
    eta, phi can be type array, must have same shape or len(1).
    '''
    if type(experiment)==type(None):
        experiment = get_id01_experiment(troi, troi_poni)

    qx,qy,qz = experiment.Ang2Q.area(eta,phi)
    
    return [[qx.min(), qx.max()], [qy.min(), qy.max()], [qz.min(), qz.max()]]


