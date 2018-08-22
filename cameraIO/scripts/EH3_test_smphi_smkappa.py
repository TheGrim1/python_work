import sys
sys.path.append('/data/id13/inhouse2/AJ/skript/')
import cameraIO.CamView_stages as cvs
import numpy as np
import time
from pythonmisc.string_format import ListToFormattedString as list2string
import os
import scipy.ndimage as nd
import fileIO.images.image_tools as it

class move_permutation():
    def __init__(self, stage, save_fname=None, motors= ['move_no','time','phi','kappa','phi_camera','kappa_camera']):
        self.sleeptime=0.01
        self.move_no = 0
        self.stage = stage
        self.motors = motors
        self.motor_permutation = [[x,1] for x in ['phi','kappa']]
        self.motor_permutation += [[x,-1] for x in ['phi','kappa']]
        
        if type(save_fname)!=type(None):
            self.fname = os.path.realpath(save_fname)
            print('writing positions to {}'.format(os.path.realpath(self.fname)))
            with file(self.fname,'w') as f:
                f.write('#'+list2string(motors,15)+'\n')
        else:
            self.fname = None
            
        self.camera_position = {}
        self.calibrate_camera_position()
        self.print_pos()

    def _get_camera_com(self):
        image = self.stage._get_view('wall')
        image = self.stage._get_view('wall') # making sure its new
        image = it.optimize_imagestack_contrast(imagestack=image,cutcontrast=0.5)
        return nd.measurements.center_of_mass(image)        
        
    def calibrate_camera_position(self):
        com = self._get_camera_com()
        calibration = self.camera_calibration = self.stage.calibration['wall']['phi']
        pos_dc = self.stage.get_motor_dict()
        phi_real = pos_dc['phi']
        kappa_real = pos_dc['kappa']

        self.camera_position['kappa'] = kappa_real*calibration - com[0]
        self.camera_position['phi'] = phi_real*calibration - com[1]

        print('found kappa = {:.4f} at horz_pixel = {}'.format(kappa_real, com[0]))
        print('found phi   = {:.4f} at vert_pixel = {}'.format(phi_real, com[1]))
        
    def add_camera_positions(self, pos_dc):
        com = self._get_camera_com()
        pos_phi = (self.camera_position['phi'] + com[1]) / self.camera_calibration
        pos_kappa = (self.camera_position['kappa'] + com[0]) / self.camera_calibration
        pos_dc['phi_camera']=pos_phi
        pos_dc['kappa_camera']=pos_kappa
        return pos_dc

        
    def print_pos(self):
        pos_dc = self.stage.get_motor_dict()
        motors = self.motors
        pos_dc = self.add_camera_positions(pos_dc)
        pos_dc['time'] = time.time()
        pos_dc['move_no'] = self.move_no
        print('move no {:6}'.format(self.move_no))
        print(list2string(motors,15))
        print(list2string([str(pos_dc[x]) for x in motors],15))
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
        time.sleep(self.sleeptime)

                
    def move_abs(self,phi_pos,kappa_pos,stepsize):
        move_no = self.move_no
        motors = self.motors

        permutation = move_no % (len(motors)*2)
        motor = self.motor_permutation[permutation][0]
        sign = self.motor_permutation[permutation][1]
        if motor == 'phi':
            move = phi_pos
        elif motor == 'kappa':
            move = kappa_pos
        else:
            return None
            
        move += sign*stepsize/2.0
        print('mv {} by {}'.format(motor,move))
        self.stage.mv(motor,move)
        self.move_no += 1
        self.print_pos()
        time.sleep(self.sleeptime)
        

    def log_move_abs(self,phi_pos,kappa_pos, move_using_lookup=False):
        self.move_no += 1
        for i in range(10):
            self.stage.mv('phi',phi_pos, move_using_lookup=move_using_lookup)
            self.stage.mv('kappa',phi_pos, move_using_lookup=move_using_lookup)
            self.print_pos()
        
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
    stage = cvs.EH3_hex_phikappa_gonio()
    mover = move_permutation(stage, save_fname)
    mover.dmesh('phi',0,0.1,10,'kappa',0,0.1,10,0.1)

def do_test_random(save_fname=None,no_steps = 1000):
    stage = cvs.EH3_hex_phikappa_gonio()
    mover = move_permutation(stage, save_fname)
    phi_pos = stage.wm('phi')
    kappa_pos = stage.wm('kappa')

    
    random_stepsizes = np.random.random_sample(2*no_steps)*0.3 - 0.15

    for i in range(no_steps):
        stage.mvr('phi', random_stepsizes[i])
        stage.mvr('kappa', random_stepsizes[i+1])
        mover.log_move_abs(phi_pos,kappa_pos)

def do_test_abs(save_fname=None,no_steps = 1000):
    stage = cvs.EH3_hex_phikappa_gonio()
    mover = move_permutation(stage, save_fname)
    phi_pos = stage.wm('phi')
    kappa_pos = stage.wm('kappa')
    
    for i in range(no_steps):
        stage.mvr('phi', 0.1)
        stage.mvr('kappa', 0.1)
        mover.log_move_abs(phi_pos, kappa_pos)


def do_test_camera_res(save_fname=None,no_steps = 1000):
    stage = cvs.EH3_hex_phikappa_gonio()
    mover = move_permutation(stage, save_fname)
    phi_pos = stage.wm('phi')
    kappa_pos = stage.wm('kappa')

    for i in range(no_steps):
        mover.log_move_abs(phi_pos+i*0.001, kappa_pos+i*0.001)




        
def do_test_rel(save_fname=None):
    stage = cvs.EH3_hex_phikappa_gonio()
    mover = move_permutation(stage, save_fname)
    for i in range(200):
        mover.move_rel(0.1)
    
if __name__=='__main__':
    usage = 'usage:\npython test_smphi_smkappa <dmesh/random/abs/rel/time> <optional savename for positions>' 
    
    arg = sys.argv[1:]
    if len(arg) not in [1,2]:
        print('wrong number of arguements')
        print(usage)
        sys.exit(0)
        
    if len(arg) == 2:
        save_fname = arg[1]
    else:
        save_fname = None

    print(arg)
    if str(arg[0]).lower() == 'random':
        do_test_random(save_fname)
    elif str(arg[0]).lower() == 'dmesh':
        do_test_dmesh(save_fname)
    elif str(arg[0]).lower() == 'abs':
        do_test_abs(save_fname)
    elif str(arg[0]).lower() == 'rel':
        do_test_rel(save_fname)
    elif str(arg[0]).lower() == 'time':
        do_test_time(save_fname)
    else:
        print('did not understand {}'.format(str(arg[0]).lower()))
        print(usage)
        sys.exit(0)


