import numpy as np
import xrayutilities as xu
import sys,os


def get_id13_experiment(troi, troi_poni):
    
    qconv=xu.experiment.QConversion(['z+','x+','z+'],[],[1,0,0])
    hxrd = xu.HXRD([-1,0,0],[0,1,0],qconv=qconv,sampleor='y+')

    hxrd.wavelength = troi_poni['Wavelength']*1e10
    
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



