# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:34:22 2017

@author: leake

At id01 we would use the microscope and then kmap to define the marker positions
i.e. need diffraction contrast.

This script was written in relation to a NFFA project for nanopositioning 
        (Thomas Keller, Manuel Abuin and co @ NFFA)

# 180-on the rotations why?

DISCLAIMER:
Please execute window by window i.e after manually modifying a quantity 
you must execute calcuate to update the relevant parameters

"""

import sys, os
import numpy as np
import math as m
from PyQt4 import QtCore, QtGui,uic
import id01lib.ID01_NFFApython as NFFA
import id01lib.SpecClientWrapper as id01SpecClientWrapper
#import id01lib as id01

form_class = uic.loadUiType("bin/NanosuitcaseID01.ui")[0]

class Window(QtGui.QMainWindow,form_class):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self,parent)
        self.setupUi(self)
        
        # buttons
        self.loadfile.clicked.connect(self.loadfile_clicked)        
        self.calculate_newpos.clicked.connect(self.calculate_newpos_clicked) 
        self.import_pos_spec.clicked.connect(self.import_pos_spec_clicked)
        self.send_group_spec.clicked.connect(self.send_group_spec_clicked)        
        self.send_group_spec_all.clicked.connect(self.send_group_spec_all_clicked)        
        
        # radiobuttons
        self.rB_manualinput.toggled.connect(self.disconnect_all_pos)        
        
        self.rB_invert_inst1_x.toggled.connect(self.rB_invert_inst1_x_clicked)
        self.rB_invert_inst1_y.toggled.connect(self.rB_invert_inst1_y_clicked)
        self.rB_invert_inst1_z.toggled.connect(self.rB_invert_inst1_z_clicked)
        self.rB_invert_inst2_x.toggled.connect(self.rB_invert_inst2_x_clicked)
        self.rB_invert_inst2_y.toggled.connect(self.rB_invert_inst2_y_clicked)
        self.rB_invert_inst2_z.toggled.connect(self.rB_invert_inst2_z_clicked)
        
        self.instrument1geometry = np.identity(3)
        self.instrument2geometry = np.identity(3)
        
        self.offset_h1m1_inst1=np.array([0,0,0])
        self.offset_h1m1_inst2=np.array([0,0,0])
        self.U=np.identity(3)
        
        self.group_id01={}

        self.path2file.setText('SampleonSTMholder_test.txt')
        self.path2file.setText('hc3201_sample6.txt')
        self.id01_spec_session_name.setText('nano2:psic_nano')
        self.piezo_center = 100
        self.scale_inst1=1000.
        #self.rB_invert_inst2_x.setChecked(True)
        
        self.debug=False
        self.show()   
        
    def loadfile_clicked(self):
        # clear Qcomboboxes  
        self.h1m1_name_in.clear()
        self.h1m2_name_in.clear()
        self.h2m1_name_in.clear()
        self.h2m2_name_in.clear()
        self.target_name_in.clear()
        self.target_name_delta.clear()
        self.target_name_abs.clear()
 
        if self.rB_manualinput.isChecked():
            self.rB_manualinput.setChecked(False)
        if str(self.path2file.text()).endswith('.cfg')==False:            
            NFFA.parseTXT2config(os.path.join(os.getcwd(),str(self.path2file.text())))
            self.path2file.setText(str(self.path2file.text()).split('.')[0]+'.cfg')
        #  populate positions
        print "Loaded: ", str(self.path2file.text())
        self._config=NFFA.get_config(str(self.path2file.text()))
        for pos_prefix in ['h1m1','h1m2','h2m1','h2m2','target']: 
            self.generate_pos(pos_prefix,'_in')
        # QComboBox
        #self.h1m1_name_in.currentIndexChanged.connect(self.calling_function)
        self.h1m1_name_in.currentIndexChanged.connect(lambda: self.change_pos('h1m1','_in'))
        self.h1m2_name_in.currentIndexChanged.connect(lambda: self.change_pos('h1m2','_in'))
        self.h2m1_name_in.currentIndexChanged.connect(lambda: self.change_pos('h2m1','_in'))
        self.h2m2_name_in.currentIndexChanged.connect(lambda: self.change_pos('h2m2','_in'))
        self.target_name_in.currentIndexChanged.connect(lambda: self.change_pos('target','_in'))
        
        for pos_suffix in ['_delta','_abs']: 
            self.generate_pos_output('target',pos_suffix)        
        
        self.target_name_delta.currentIndexChanged.connect(lambda: self.change_pos_delta('target','_delta'))
        self.target_name_abs.currentIndexChanged.connect(lambda: self.change_pos_abs('target','_abs'))
        
        # set defaults 
        index = self.h1m1_name_in.findText("h1m1",QtCore.Qt.MatchFixedString)
        if index>=0:
            self.h1m1_name_in.setCurrentIndex(index)
        index = self.h1m2_name_in.findText("h1m2",QtCore.Qt.MatchFixedString)
        if index>=0:
            self.h1m2_name_in.setCurrentIndex(index)
        index = self.h2m1_name_in.findText("h2m1",QtCore.Qt.MatchFixedString)
        if index>=0:
            self.h2m1_name_in.setCurrentIndex(index)
        index = self.h2m2_name_in.findText("h2m2",QtCore.Qt.MatchFixedString)
        if index>=0:
            self.h2m2_name_in.setCurrentIndex(index)
        index = self.target_name_in.findText("target",QtCore.Qt.MatchFixedString)
        if index>=0:
            self.target_name_in.setCurrentIndex(index)
            
    def generate_pos(self,pos_prefix,pos_suffix):
        exec("self.%s_name%s.addItems(self._config.sections())"%(pos_prefix,pos_suffix))
        self.change_pos(pos_prefix,pos_suffix)

    def change_pos(self,pos_prefix,pos_suffix,):
        #if exec("self.%s_name%s.currentText()"%(pos_prefix,pos_suffix))!='':
        exec("self.tmp_array=NFFA.get_pars_pos_config(str(self.path2file.text()),str(self.%s_name%s.currentText()),scale=self.scale_inst1)"%(pos_prefix,pos_suffix))
        #self.tmp_array*=self.scale
        for axis in ['x','y','z']:
            exec("self.%s%s%s.setText('%.6f')"%(pos_prefix,axis,pos_suffix,self.tmp_array[axis+'coordinate']))
    
    def disconnect_all_pos(self):
        for pos_prefix in ['h1m1','h1m2','h2m1','h2m2','target']: 
            exec("self.%s_nam100e%s.clear()"%(pos_prefix,'_in'))

    def import_pos_spec_clicked(self):
        # clear the Qcomboboxes first        
        self.h1m1_name_in_inst2.clear()
        self.h1m2_name_in_inst2.clear()

        # use specclient to read the group positions
        # specclient not available in python3.... moving back to python2
        self.SC_psic_nano = id01SpecClientWrapper.SpecClientSession(sv_limaroi = 'LIMA_ROI', sv_limadev = 'LIMA_DEV',specname=str(self.id01_spec_session_name.text()))
        self.group_id01_import = self.SC_psic_nano.get_sv('GROUP_LIST') 
        print(self.group_id01_import)
        self.group_id01=NFFA.get_group_from_spec(str(self.id01_spec_groupname.text()),self.group_id01_import) #      
        print(self.group_id01)
        for pos_prefix in ['h1m1','h1m2']:#group.keys(): 
            self.generate_pos_spec(pos_prefix,'_in_inst2',self.group_id01.keys())
            
        # QComboBox
        self.h1m1_name_in_inst2.currentIndexChanged.connect(lambda: self.change_pos_inst2('h1m1','_in_inst2'))
        self.h1m2_name_in_inst2.currentIndexChanged.connect(lambda: self.change_pos_inst2('h1m2','_in_inst2'))

        # set defaults 
        index = self.h1m1_name_in_inst2.findText("h1m1")
        if index>=0:
            self.h1m1_name_in_inst2.setCurrentIndex(index)
        index = self.h1m2_name_in_inst2.findText("h1m2")
        if index>=0:
            self.h1m2_name_in_inst2.setCurrentIndex(index)
            
    def generate_pos_spec(self,pos_prefix,pos_suffix,list_keys):
        #print pos_prefix,pos_suffix,list_keys
        exec("self.%s_name%s.addItems(%s)"%(pos_prefix,pos_suffix,list_keys))
        self.change_pos_inst2(pos_prefix,pos_suffix)

    def change_pos_inst2(self,pos_prefix,pos_suffix):
        #if exec("self.%s_name%s.currentText()"%(pos_prefix,pos_suffix))!='':
        exec("self.tmp_array_inst2=self.group_id01[str(self.%s_name%s.currentText())]"%(pos_prefix,pos_suffix))
        
        if self.debug:
            print "inst2: ", self.tmp_array_inst2
        # populate motor pos
        for axis in ['thx','thy','thz']:
            exec("self.%s%s_ID01.setText('%.6f')"%(pos_prefix,axis,self.tmp_array_inst2[axis]))
        for axis in ['pix','piy','piz']:
            exec("self.%s%s_ID01_piezo.setText('%.6f')"%(pos_prefix,axis,self.tmp_array_inst2[axis]-self.piezo_center))
        for hexa,piezo in [['thx','pix'],['thy','piy'],['thz','piz']]:
            exec("self.%s%s_ID01_abs.setText('%.6f')"%(pos_prefix,hexa[-1],self.tmp_array_inst2[hexa]+(self.tmp_array_inst2[piezo]-self.piezo_center)/1000))


    def calculate_newpos_clicked(self):
        # instrument_2 - instrument_1 - the origin is therefore h1m1 + offset
        #self.offset_h1m1_inst1=np.array([float(self.h1m1x_ID01_abs.text())-float(self.h1m1x_in.text()),float(self.h1m1y_ID01_abs.text())-float(self.h1m1y_in.text()),float(self.h1m1z_ID01_abs.text())-float(self.h1m1z_in.text())])
        #self.offset_h1m1_inst2=np.array([float(self.h1m1x_ID01_abs.text())-float(self.h1m1x_in.text()),float(self.h1m1y_ID01_abs.text())-float(self.h1m1y_in.text()),float(self.h1m1z_ID01_abs.text())-float(self.h1m1z_in.text())])
        self.offset_h1m1_inst1=np.array([float(self.h1m1x_in.text()),float(self.h1m1y_in.text()),float(self.h1m1z_in.text())])
        self.offset_h1m1_inst2=np.array([float(self.h1m1x_ID01_abs.text()),float(self.h1m1y_ID01_abs.text()),float(self.h1m1z_ID01_abs.text())])

        # Rz first
        v1a=np.array([float(self.h1m1x_in.text()),float(self.h1m1y_in.text())])
        v1b=np.array([float(self.h1m2x_in.text()),float(self.h1m2y_in.text())])
        v2a=np.array([float(self.h1m1x_ID01_abs.text()),float(self.h1m1y_ID01_abs.text())])
        v2b=np.array([float(self.h1m2x_ID01_abs.text()),float(self.h1m2y_ID01_abs.text())])

        v1=v1b-v1a
        v2=v2b-v2a

        if self.debug:
            print v1a,v1b
            print v2a,v2b
            print v1,v2
            
        yaw=np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
        self.calc_rot.setText(str(180-np.rad2deg(yaw)))
        self.calc_yaw.setText(str(180-np.rad2deg(yaw)))
        
        # Ry second,scale=self.scale_inst1
        v1a=np.array([float(self.h1m1x_in.text()),float(self.h1m1z_in.text())])
        v1b=np.array([float(self.h1m2x_in.text()),float(self.h1m2z_in.text())])
        v2a=np.array([float(self.h1m1x_ID01_abs.text()),float(self.h1m1z_ID01_abs.text())])
        v2b=np.array([float(self.h1m2x_ID01_abs.text()),float(self.h1m2z_ID01_abs.text())])

        v1=v1b-v1a
        v2=v2b-v2a
        pit=np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
        
        if m.isnan(pit):
            pit=0.0
        self.calc_pit.setText(str(180-np.rad2deg(pit)))

        # Rx third
        v1a=np.array([float(self.h1m1y_in.text()),float(self.h1m1z_in.text())])
        v1b=np.array([float(self.h1m2y_in.text()),float(self.h1m2z_in.text())])
        v2a=np.array([float(self.h1m1y_ID01_abs.text()),float(self.h1m1z_ID01_abs.text())])
        v2b=np.array([float(self.h1m2y_ID01_abs.text()),float(self.h1m2z_ID01_abs.text())])

        v1=v1b-v1a
        v2=v2b-v2a
        rol=np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
        
        if m.isnan(rol):
            rol=0.0
        self.calc_rol.setText(str(180-np.rad2deg(rol)))
        
        self.U = np.dot(NFFA.Rz(np.rad2deg(yaw)),NFFA.Ry(np.rad2deg(pit)),NFFA.Rx(np.rad2deg(rol)))

        if self.debug:        
            print "Rz,Ry,Rx \n",NFFA.Rz(np.rad2deg(yaw)),NFFA.Ry(np.rad2deg(pit)),NFFA.Rx(np.rad2deg(rol))
            print "instrument 1 \n",self.instrument1geometry
            print "instrument 2 \n",self.instrument2geometry
            print "UB matrix: \n",self.U
        
        #update coordinate positions h2m1/h2m2/target delta
        for pos_prefix in ['h2m1','h2m2']:
            #in the instrument1 frame:
            exec("_pos=np.array([float(self.%sx_in.text()),float(self.%sy_in.text()),float(self.%sz_in.text())])"%(pos_prefix,pos_prefix,pos_prefix))
            # subtract h1m1 postion for relative, full geometry considered
            _delta=np.dot(_pos-self.offset_h1m1_inst1,self.instrument1geometry)
            #print "delta: \n"
            #print _delta 
            _rotdelta = np.dot(_delta,self.U)
            #print "rotdelta: \n"
            #print _rotdelta
            exec("self.%sx_delta.setText(str(%.6f))"%(pos_prefix,_rotdelta[0]))
            exec("self.%sy_delta.setText(str(%.6f))"%(pos_prefix,_rotdelta[1]))
            exec("self.%sz_delta.setText(str(%.6f))"%(pos_prefix,_rotdelta[2]))

        for pos_prefix in ['h2m1','h2m2']:
            #in the instrument2 frame
            exec("_rotdelta=np.array([float(self.%sx_delta.text()),float(self.%sy_delta.text()),float(self.%sz_delta.text())])"%(pos_prefix,pos_prefix,pos_prefix))
            #delta rotated by UB matrix 
            #print _delta
            #_rotdelta=np.dot(_delta,self.U)
            #delta+offset h1m1 of inst2 
            _abs=np.dot(_rotdelta,self.instrument2geometry)+self.offset_h1m1_inst2
            #print pos,_abs
            exec("self.%sx_abs.setText(str(%.6f))"%(pos_prefix,_abs[0]))
            exec("self.%sy_abs.setText(str(%.6f))"%(pos_prefix,_abs[1]))
            exec("self.%sz_abs.setText(str(%.6f))"%(pos_prefix,_abs[2]))
            
        # must  be a better way to do this - I am tired...       
  
    def generate_pos_output(self,pos_prefix,pos_suffix):
        exec("self.%s_name%s.addItems(self._config.sections())"%(pos_prefix,pos_suffix))
        #self.change_pos(pos_prefix,pos_suffix)
  
    def change_pos_delta(self,pos_prefix,pos_suffix):
        exec("self.tmp_array=NFFA.get_pars_pos_config(str(self.path2file.text()),str(self.%s_name%s.currentText()),scale=self.scale_inst1)"%(pos_prefix,pos_suffix))
        _pos=np.array([self.tmp_array['xcoordinate'],self.tmp_array['ycoordinate'],self.tmp_array['zcoordinate']])#-self.offset_h1m1_inst1
        _pos-= self.offset_h1m1_inst1      
        #print _pos,self.offset_h1m1_inst1
        # subtract h1m1 postion for relative, full geometry considered
        _delta=np.dot(_pos,self.instrument1geometry)
        _rotdelta = np.dot(_delta,self.U)
        #print pos,delta
        exec("self.%sx_delta.setText(str(%.6f))"%(pos_prefix,_rotdelta[0]))
        exec("self.%sy_delta.setText(str(%.6f))"%(pos_prefix,_rotdelta[1]))
        exec("self.%sz_delta.setText(str(%.6f))"%(pos_prefix,_rotdelta[2]))
    
    def change_pos_abs(self,pos_prefix,pos_suffix):
        # explicitly calculate the abs of a specific point      
        exec("self.tmp_array=NFFA.get_pars_pos_config(str(self.path2file.text()),str(self.%s_name%s.currentText()),scale=self.scale_inst1)"%(pos_prefix,pos_suffix))        
        _pos=np.array([self.tmp_array['xcoordinate'],self.tmp_array['ycoordinate'],self.tmp_array['zcoordinate']])#-self.offset_h1m1_inst1
        _pos-=self.offset_h1m1_inst1        
        #print _pos,self.offset_h1m1_inst1      
        # consider geometry changes
        _delta=np.dot(_pos,self.instrument1geometry)
        ### apply all rotations here ###
        _rotdelta=np.dot(_delta,self.U)
        #delta+offset h1m1 of inst2 
        _abs=np.dot(_rotdelta,self.instrument2geometry)+self.offset_h1m1_inst2
        #print pos,_abs
        exec("self.%sx_abs.setText(str(%.6f))"%(pos_prefix,_abs[0]))
        exec("self.%sy_abs.setText(str(%.6f))"%(pos_prefix,_abs[1]))
        exec("self.%sz_abs.setText(str(%.6f))"%(pos_prefix,_abs[2]))
    
    def send_group_spec_clicked(self):
        # update groups in spec with additional positions
        send_str = ["groupadd %s  pix piy piz thx thy thz"%str(self.id01_spec_groupname_output.text()),\
                    "groupaccuracy %s  0.005 0.005 0.005 0.005 0.005 0.005"%str(self.id01_spec_groupname_output.text()),\
                    "groupoffsets %s  0 0 0 0 0 0"%str(self.id01_spec_groupname_output.text())]
        # add all positions in new reference frame
        for x in send_str:
            print x

        # load into specr
        [self.SC_psic_nano.send_sc(x) for x in send_str]
        # target position puts piezo in the center of its stroke.
        command="groupaddpos %s %s %.6f %.6f %.6f %.6f %.6f %.6f"%(str(self.id01_spec_groupname_output.text()),str(self.target_name_abs.currentText()).replace(" ","_"),\
                 self.piezo_center,self.piezo_center,self.piezo_center,float(self.targetx_abs.text())-self.piezo_center/1000,float(self.targety_abs.text())-self.piezo_center/1000,float(self.targetz_abs.text())-self.piezo_center/1000)         
        self.SC_psic_nano.send_sc(command)
        
    def send_group_spec_all_clicked(self):
        # update groups in spec with additional positions
        send_str = ["groupadd %s  pix piy piz thx thy thz"%str(self.id01_spec_groupname_output.text()),\
                    "groupaccuracy %s  0.005 0.005 0.005 0.005 0.005 0.005"%str(self.id01_spec_groupname_output.text()),\
                    "groupoffsets %s  0 0 0 0 0 0"%str(self.id01_spec_groupname_output.text())]
        # add all positions in new reference frame
        for x in send_str:
            print x

        # load into specr
        #[self.SC_psic_nano.send_sc(x) for x in send_str]
        # target position puts piezo in the center of its stroke.
        
        all_pos_names = [str(self.target_name_abs.itemText(i)) for i in range(self.target_name_abs.count())]
        
        for item in all_pos_names:
            # explicitly calculate the abs of a specific point      
            self.tmp_array=NFFA.get_pars_pos_config(str(self.path2file.text()),item,scale=self.scale_inst1)        
            _pos=np.array([self.tmp_array['xcoordinate'],self.tmp_array['ycoordinate'],self.tmp_array['zcoordinate']])#-self.offset_h1m1_inst1
            _pos-=self.offset_h1m1_inst1        
            #print _pos,self.offset_h1m1_inst1      
            # consider geometry changes
            _delta=np.dot(_pos,self.instrument1geometry)
            ### apply all rotations here ###
            _rotdelta=np.dot(_delta,self.U)
            #delta+offset h1m1 of inst2 
            _abs=np.dot(_rotdelta,self.instrument2geometry)+self.offset_h1m1_inst2            
            x,y,z = _abs
            command="groupaddpos %s %s %.6f %.6f %.6f %.6f %.6f %.6f"%(str(self.id01_spec_groupname_output.text()),item.replace(" ","_"),\
                 self.piezo_center,self.piezo_center,self.piezo_center,x-self.piezo_center/1000,y-self.piezo_center/1000,z-self.piezo_center/1000)         
            #print command
            self.SC_psic_nano.send_sc(command)
        

    
    def rB_invert_inst1_x_clicked(self):
        self.instrument1geometry[0,0]*=-1
        print "inst1_geometry: ",self.instrument1geometry

    def rB_invert_inst1_y_clicked(self):
        self.instrument1geometry[1,1]*=-1
        print "inst1_geometry: ",self.instrument1geometry
        
    def rB_invert_inst1_z_clicked(self):
        self.instrument1geometry[2,2]*=-1
        print "inst1_geometry: ",self.instrument1geometry

    def rB_invert_inst2_x_clicked(self):
        self.instrument2geometry[0,0]*=-1
        print "inst2_geometry: ",self.instrument1geometry

    def rB_invert_inst2_y_clicked(self):
        self.instrument2geometry[1,1]*=-1
        print "inst2_geometry: ",self.instrument1geometry

    def rB_invert_inst2_z_clicked(self):
        self.instrument2geometry[2,2]*=-1
        print "inst2_geometry: ",self.instrument1geometry
        
def run():
    app = QtGui.QApplication(sys.argv)
    myWindow = Window(None)
    sys.exit(app.exec_())

if __name__=="__main__":
    run()