import sys
sys.path.append('Y:\inhouse2\AJ\skript')
import cameraIO.CamView_stages as cvs
import numpy as np
import time
from pythonmisc.string_format import ListToFormattedString as list2string
import os

class move_permutation():
    def __init__(self, stage, save_fname=None, motors= ['phi','kappa']):
        time.sleep(10)
        self.move_no = 0
        self.stage = stage
        self.motors = motors
        self.motor_permutation = [[x,1] for x in self.motors]
        self.motor_permutation += [[x,-1] for x in self.motors]
        
        if type(save_fname)!=type(None):
            self.fname = 'sms_test_dmesh_0p01.dat'
            print('writing positions to {}'.format(os.path.realpath(self.fname)))
            with file(self.fname,'a') as f:
                f.write('#'+list2string(motors,10)+'\n')
        else:
            self.fname = None
            
        self.print_pos()

    def print_pos(self):
        pos_dc = self.stage.get_motor_dict()
        motors = self.motors
        print('move no {:6}'.format(self.move_no))
        print(list2string(motors,10))
        print(list2string([str(pos_dc[x]) for x in motors],10))
        if type(self.fname) != type(None):
            with file(self.fname,'a') as f:
                f.write(list2string([str(pos_dc[x]) for x in motors],10)+'\n')

    def move_rel(self, stepsize):
        move_no = self.move_no
        motors = self.motors

        permutation = move_no % (len(motors)*2)
        motor = self.motor_permutation[permutation][0]
        sign = self.motor_permutation[permutation][1]
        move = sign*stepsize
        print('mvr {} by {}'.format(motor,move))
        self.stage.mvr(motor,move)
        self.move_no += 1
        self.print_pos()
        time.sleep(1)

                
    def move_abs(self,phi_pos,kapp_pos,stepsize):
        move_no = self.move_no
        motors = self.motors

        permutation = move_no % (len(motors)*2)
        motor = self.motor_permutation[permutation][0]
        sign = self.motor_permutation[permutation][1]
        if motor == 'phi':
            move = phi_pos
        elif motor == 'kappa':
            move = kappa_pos
            
        move += sign*stepsize/2.0
        
        print('mv {} by {}'.format(motor,move))
        self.stage.mv(motor,move)
        self.move_no += 1
        self.print_pos()
        time.sleep(1)

    def dmesh(self, mot1, mot1_start, mot1_stop, mot1_points, mot2, mot2_start, mot2_stop, mot2_points, ct):

        mot2_pos = self.stage.wm(mot2)
        mot2_start = mot2_start + mot2_pos
        mot2_stop = mot2_stop + mot2_pos

        mot1_pos = self.stage.wm(mot1)
        mot1_start = mot1_start + mot1_pos
        mot1_stop = mot1_stop + mot1_pos

        mot2_positions = np.linspace(mot2_start,mot2_stop,mot2_points)
        for mot2_pos in mot2_positions:
            self.stage.mv(mot2, mot2_pos)
            self.ascan(mot1, mot1_start, mot1_stop, mot1_points, ct)  
        
    def ascan(self, mot, start, stop, points, ct):
        mot_positions = np.linspace(start,stop,points)
        for pos in mot_positions:
            self.stage.mv(mot, pos)
            self.print_pos()
            time.sleep(ct)       
            

def do_test_dmesh(save_fname=None):
    stage = cvs.phi_kappa_gonio()
    mover = move_permutation(stage, save_fname)
    mover.dmesh('phi',0,0.1,10,'kappa',0,0.1,10,0.1)

def do_test_random(save_fname=None):
    stage = cvs.phi_kappa_gonio()
    mover = move_permutation(stage, save_fname)
    phi_pos = stage.wm('phi')
    kappa_pos = stage.wm('kappa')
    random_stepsize = np.random.random_sample()*0.2
    
    for i in range(100):
        mover.move_rel(random_stepsize)
        time.sleep(1)
        stage.mv('phi',phi_pos)
        stage.mv('kappa',kappa_pos)

def do_test_abs(save_fname=None):
    stage = cvs.phi_kappa_gonio()
    mover = move_permutation(stage, save_fname)
    phi_pos = stage.wm('phi')
    kappa_pos = stage.wm('kappa')
    for i in range(100):
        mover.move_abs(phi_pos, kappa_pos, random_stepsize)

def do_test_rel(save_fname=None):
    stage = cvs.phi_kappa_gonio()
    mover = move_permutation(stage, save_fname)
    for i in range(100):
        mover.move_rel(0.1)
    
if __name__=='__main__':
    usage = 'python test_smphi_smkappa <dmesh/random/abs/rel> <optional savename for positions>' 
    
    
    arg = sys.argv[1:]
    if len(arg) not in [1,2]:
        print(usage)
        sys.exit(0)
        
    if len(arg) == 2:
        save_fname = arg[1]
    else:
        save_fname = None

    if str(arg[0]).lower == 'random':
        do_test_random(save_fname)
    elif str(arg[0]).lower == 'dmesh':
        do_test_dmesh(save_fname)
    elif str(arg[0]).lower == 'abs':
        do_test_abs(save_fname)
    elif str(arg[0]).lower == 'rel':
        do_test_rel(save_fname)
    else:
        print(usage)
        sys.exit(0)


