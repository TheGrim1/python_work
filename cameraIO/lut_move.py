from __future__ import print_function
import collections
import matplotlib.pyplot as plt

import numpy as np
import sys
import copy
sys.path.append('/data/id13/inhouse2/AJ/skript')
from fileIO.datafiles import open_data, save_data
from cameraIO.CamView_lookup import LookupDict, LookupDict_Phi_XZKappa, LookupDict_Phi_hexXZKappa
from cameraIO.CamView_lookup import LookupDict_smhi_nnp4nnp6_smkappa
import os

class LUT_Anyberg(object):
    '''
    collects all methods that are a priory independent of the respective setup geometry
    The store for all positions is in a LookupDict class
    see LUT_Navitar for the phi-kappa gonio geometry example
    use LUT_Generic for stages without dynamic lookups
    '''
    MDC = dict(
        kappa = dict(
            is_rotation = True,
            invert      = False,
            COR_motors  = ['x','z'],
            default_pos = 0,
        ),
        x = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),
        y = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),
    )
    
    def __init__(self):
        self.lookup_fnames ={}
        self.lookup = {}
        self.dynamic_lookup = {}
        self.motors = self.MDC
        # initial positions
        pos_list = [(k,0) for k in list(self.motors.keys())]
        self.pos = dict(pos_list)

        [self.init_current_pos(mot) for mot in []]
        
        # external to internal motorname translation
        self.mto_lookup = mto_lookup = dict(
            phi = "phi",
            kappa = "kappa",
            fine_x = "x",
            fine_y = "y",
            fine_z = "z"
        )
        tems = list(mto_lookup.items())
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)
        self.show()

    def init_current_pos(self, motor):
        self.lookup[motor]=LookupDict(self.pos)
        self.dynamic_lookup[motor]=LookupDict(self.pos)
        
        
    def show(self):
        print('\nLookuptable function available from:\n{}\nclass: {}'.format(os.path.realpath(__file__),self.__class__.__name__))

        print('Motor table:')
        
        def p_nice_line(str1,str2):
            print(str1 + ' '*(20-len(str1)) + str2)
            
        p_nice_line('external name','lut name')
        print('-'*25)
        for external, internal in self.mto_lookup.items():
            p_nice_line(external,internal)


    def mto_eig_dict(self, lookup_dc):
        '''
        return the translated dict of <{mot1_internal_name:pos, etc. }>  into <{mot1_external_name:pos, etc. }> 
        '''
        tems = list(lookup_dc.items())
        kdc = self.mto_eig
        stem = [(kdc[k],v) for (k,v) in tems]
        res = dict(stem)
        return res

    def mto_lookup_dict(self, eig_dc):
        '''
        return the translated dict of <{mot1_external_name:pos, etc. }>  into <{mot1_internal_name:pos, etc. }> 
        filters for names listed in MDC
        '''
        print('in mto_lookup_dict, eig_dc')
        print(eig_dc)
        tems = list(eig_dc.items())
        kdc = self.mto_lookup
        
        print('in mto_lookup_dict, self.kdc')
        print(kdc)
        print('in mto_lookup_dict, self.motors')
        print(self.motors)
        
        stem = [(kdc[k],v) for (k,v) in tems if kdc[k] in list(self.motors.keys())]
        res = dict(stem)
        print('in mto_lookup_dict, res')
        print(res)
        return res

    def get_lookup_correction(self, function, startpos_dc, end_pos, dynamic=True):
        '''
        returns a dict with {mot1_name:mot1_correction .. etc} to correct the movement of <function> from <start_pos> to <end_pos>
        <dynamic>[bool] selects whether the dynamic lookuptable is used.
        '''
        self.sync_pos(startpos_dc)
        corrdc = dict()

        start_pos = startpos_dc[function]
        
        if dynamic:
            lookup = self.dynamic_lookup
        else:
            lookup = self.lookup

        if self.motors[function]['is_rotation']:
            start_pos = start_pos % 360.0
            end_pos   = end_pos   % 360.0

        if function in list(lookup.keys()):
            for mot in list(lookup[function].keys()):
                if mot != function:
                    start_correction = np.interp(start_pos, lookup[function][function], lookup[function][mot])
                    end_correction   = np.interp(end_pos, lookup[function][function], lookup[function][mot])
                    correction = end_correction-start_correction
                    corrdc[mot] = correction
            return corrdc
        else:
            print('no lookuptable found for ' , function)
            return None

    def sync_pos(self, pos_dc):
        '''
        syncs the current mockup position to pos_dc in lookup table nomencature
        accepst for example self.sync_pos(self.mto_lookup_dict(ne.read_all_motic_pos()))
        '''
        self.pos.update(pos_dc)
        for mot in list(self.dynamic_lookup.keys()):
            self.dynamic_lookup[mot].mockup_currpos(pos_dc)
        # print('updated the currentposition to the lookuptabe interface')
        # print(res)

    def link_dynamic(self):
        '''
        generic undynamic "dynamic" dict
        '''
        for key, lookup in self.lookup.items():
            motor_dict = self.motors
            dynamic_dict = self.dynamic_lookup[key] = LookupDict(motor_dict)
            dynamic_dict.update(lookup)  
        
### methods migrated here from the CamView stage class:
        
    def load_lookup(self, fname):
        data, header = open_data.open_data(fname)
        lookupmotor = header[0]
        print("found lookuptable for motor: ", lookupmotor)
        print('using (unsorted) motors ', header[1:])
        self.lookup[header[0]] = LookupDict(self.motors)
        for i, mot in enumerate(header):
            self.lookup[header[0]][mot] = data[:,i]
        self.lookup_fnames.update({lookupmotor:fname})
        self.link_dynamic()
            
    def save_lookup(self, function, savename=None):
        data   = np.zeros(shape = (len(self.lookup[function][function]),len(list(self.lookup[function].keys()))))

        if not type(savename) == type('asfd'):
            savename = (self.lookup_fnames[function])
                        
        unsorted_header = list(self.lookup[function].keys())
        header    = []
        header.append(unsorted_header.pop(unsorted_header.index(function)))
        header   += unsorted_header
        for i, mot in enumerate(header):
            data[:,i] = self.lookup[function][mot]
        
        save_data.save_data(savename, data, header = header)
        print('saved lookup as: {}'.format(os.path.realpath(savename)))
                
    def plot_lookup(self, motor='phi',plot_temp=False):
        if plot_temp:
            lookup = self.tmp_lookup[motor]
        else:
            lookup = self.lookup[motor]
        for mot in list(lookup.keys()):
            if not mot==motor:
                dummy, ax1 = plt.subplots(1) 
                ax1.set_title('%s vs %s'%(mot,motor))                              
                ax1.plot(lookup[motor],lookup[mot])

                
    def update_lookup(self, motor, shift_lookup, overwrite=False, lookupmotors=None, shift_relative=False):
        '''
        if overwrite == False, shift_lookup is added to the old lookup 
        if <shift_relative>, as a relative change i.e. lookup_new = merge(lookup_old and (shift_lookup + np.interp(lookup_old)))
        if not <shift_relative>, as absolute positions i.e. lookup_new = merge(lookup_old and <shift_lookup>) 
        if lookupmotors == None: # assume the same motors as for COR
            lookupmotors = self.stagegeometry['COR_motors'][motor]['motors']
        '''
        # we have to add or update the values to the old lookup


        if motor not in list(self.lookup.keys()):
            self.lookup.update({motor:{}})
        positions = shift_lookup.pop(motor)
        lookupmotors = list(shift_lookup.keys())
        
        if self.motors[motor]['is_rotation']:
            positions=np.asarray(positions)
            positions=positions % 360.0

        # after mod the list is no longer sorted but it needs to be for the np.interp and my following update of the old lookup
        # so now sort:
        positions_list=list(positions)
        together_array = np.zeros(shape = (len(positions), len(lookupmotors)+1))
        together_array[:,0]=positions
        
        for i, mot in enumerate(lookupmotors):
            together_array[:,i+1] = list(shift_lookup[mot])
        
        #print 'together_array'
        #print together_array
        together_list=[]
        for i in range(len(positions)):
            together_list.append([x for x in together_array[i,:]])
        #together = zip(positions_list, shift_0, shift_1)

        sorted_together =  sorted(together_list)

        # to make the lookuptable look continuos for np.interp, we add the value for 360.0 and 0.0
        if self.motors[motor]['is_rotation']:
            if not sorted_together[0][0]==0.0:
                if sorted_together[0][0] < 360.0-sorted_together[-1][0]:
                    sorted_together.insert(0, [0.0] +  list([x for x in sorted_together[0][1:]]))
                else:
                    sorted_together.insert(0, [0.0] +  list([x for x in sorted_together[-1][1:]]))

            if not sorted_together[-1][0]==360.0:
                if sorted_together[0][0] < 360.0-sorted_together[-1][0]:
                    sorted_together.append([360.0] + list([x for x in sorted_together[0][1:]]))
                else:
                    sorted_together.append([360.0] +  list([x for x in sorted_together[-1][1:]]))

        #print 'sorted_together_list'
        #print sorted_together
        positions = [x[0] for x in sorted_together]
        shift = []
        for i in  range(len(lookupmotors)):
            shift.append(np.asarray([x[i+1] for x in sorted_together]))

        # print('positions')
        # print(positions)
        # print('shift')
        # print(shift)

        ## The following commented code adds shift = (shift with previous lookup) + (new shift_lookup)
        old_mots=[]
        if shift_relative:
            old_positions = self.lookup[motor][motor]
            for i,mot in enumerate(lookupmotors):
                if mot in self.lookup[motor].keys():
                    old_mots.append(list(self.lookup[motor][mot]))
                    d_i = np.asarray((np.interp(positions, self.lookup[motor][motor], self.lookup[motor][mot])))
                else:
                    d_i = np.asarray([0.0]*len(positions))
                    old_mots.append([0.0]*len(old_positions))

                s_i = np.asarray(shift[i])
                shift[i] = list(d_i + s_i)
        if not overwrite and not shift_relative:
            old_positions = self.lookup[motor][motor]
            for i,mot in enumerate(lookupmotors):
                if mot in self.lookup[motor].keys():
                    old_mots.append(list(self.lookup[motor][mot]))
                else:
                    old_mots.append([0.0]*len(old_positions))                    


        if not overwrite:
            print('updating old lookuptable')
            
            for i, new_theta in enumerate(positions):
                j = 0
                old_theta = old_positions[j]
                while new_theta > old_theta:
                    j+=1
                    if j==len(old_positions):
                        old_positions.append(new_theta)

                        for k in range(len(old_mots)):
                            old_mots[k].append(shift[k][i])
                        #old_mot1.append(new_mot1[i])
                        break
                    else:
                        old_theta = old_positions[j]
                else:
                    if new_theta == old_theta:
                    
                        for k in range(len(old_mots)):
                            old_mots[k][i] = shift[k][i]

                        #old_mot1[j]=new_mot1[i]
                    else:
                        old_positions.insert(j,new_theta)
                        for k in range(len(old_mots)):
                            old_mots[k].insert(j,shift[k][i])
                        #old_mot1.insert(j,new_mot1[i])
        

            new_positions = old_positions
            new_mots=[]
            for k in range(len(old_mots)):
                new_mots.append(old_mots[k])
            #new_mot0 = old_mot0
        else:
            # just overwrite the old lookup
            print('writing new lookuptable')
            new_positions = positions

            print('new_positions')
            print(new_positions)

            new_mots=[]
            for k in range(len(shift)):
                new_mots.append(shift[k])
                #new_mot1   = shift_1
            
        # enter the new positions in the lookuptable:

        # other moters may be allready in the lookuptable, but not updated this time:
        other_motors = [x for x in self.lookup[motor].keys() if x not in lookupmotors+[motor]]
        print('other_motors')
        print(other_motors)
        for mot in other_motors:
            mot_positions = np.interp(new_positions, self.lookup[motor][motor], self.lookup[motor][mot])
            self.lookup[motor].update({mot: np.asarray(mot_positions)})
            print('updating '+ mot +' with')
            print(mot_positions)

        self.lookup[motor].update({motor: np.asarray(new_positions)})
        for i, mot in enumerate(lookupmotors):
            print('updating '+ mot +' with')
            print(new_mots[i])
            self.lookup[motor].update({mot: np.asarray(new_mots[i])})
            
        self.link_dynamic()
        #self.lookup[motor].update({mot1: np.asarray(new_mot1)})

        
    def tmp_to_lookup(self, lookupmotor, overwrite=True, shift_relative=False):
        '''
        updates the old lookup for <lookupmotor> with the values from tmp_lookup
        if you don't <overwrite> be sure that this tmp lookup was made at the exact same position as the orignal (eg. same microscope magnification)!
        essentially a wrapper for self.update_lookup)
        '''
        if overwrite:
            print('overwriting lookup for %s with:' % lookupmotor)
        else:
            print('adding positions to lookup for %s with:' % lookupmotor)
        lookupmotors = []
        for mot,values in list(self.tmp_lookup[lookupmotor].items()):
            lookupmotors.append(mot)
            print(mot)
            print(values)
            
        self.update_lookup(motor=lookupmotor, shift_lookup=copy.deepcopy(self.tmp_lookup[lookupmotor]), overwrite=overwrite, lookupmotors=lookupmotors, shift_relative=shift_relative)

    def lookup_to_tmp(self, lookupmotor):
        '''
        the tmp lookup can be modified using other methods
        '''
        
        print('Original lookup copied into temp position:')

        save_motor_list = [mot for mot in self.lookup[lookupmotor].keys() if mot!= lookupmotor]
        self.initialize_tmp_lookup(lookupmotor, save_motor_list=save_motor_list)
        
        for key, value in self.lookup[lookupmotor].items():
            print(key,value)
            self.tmp_lookup[lookupmotor].update({key:value})
          
    def initialize_tmp_lookup(self, lookupmotor = 'phi', save_motor_list=['x','y','z']):
        '''
        initialize the creation of a new lookuptable with add_pos_to_tmp_lookup.
        only the positions of motors in save_motor_list will be stored in the new lookuptable
        If you have interdependencies between motors, make sure you are in the respective default position!
        '''

        self.tmp_lookup = {}
        self.tmp_lookup.update({lookupmotor:{}})
        
        for mot in save_motor_list:
            self.tmp_lookup[lookupmotor].update({mot:[]})
        self.tmp_lookup[lookupmotor].update({lookupmotor:[]})

        print('ready to save lookup positions for {} and '.format(lookupmotor), save_motor_list)
        print('If you have interdependencies between lookupmotors, make sure you are in the respective default position!')
        print('eg, kappa = 0')


    def add_pos_to_tmp_lookup(self, lookupmotor, motor_dc):
        '''
        collect positions that will form a new lookuptable in the tmp_lookup
        does not need to be sorted
        can use old lookuptable to get to the new positions
        can be used outside of the default positions, but then the old lookuptable HAS to be overwritten
        '''
        tmp_lookup = self.tmp_lookup[lookupmotor]

        # this adds a correction for motors not in the default position:
        for mk in self.tmp_lookup.keys():
            if not mk == lookupmotor:
                target_pos = self.motors[mk]['default_pos']
                start_pos = motor_dc[mk]
                corr = self.get_lookup_correction(mk, start_pos, target_pos, dynamic=True)
                for (k,v) in list(corr.items()):
                    motor_dc[k] += v

                

        # actual lookup dict:
        for mot in self.tmp_lookup[lookupmotor].keys():
            tmp_lookup[mot].append(motor_dc[mot]) 

    def shift_COR_of_tmp_lookup(self,
                               rotmotor='rot',
                               COR_shift = [0.1,0.1]):
        '''
        shift the COR of <rotmotor> by adding the corresponding shifts to the COR_motors
        COR_motors=self.motors[rotmotor]['COR_motors']
        '''
        COR_motors=self.motors[rotmotor]['COR_motors']

        if not self.motors[rotmotor]['is_rotation']:
            print('does it make sense to shift a COR if self.motors[rotmotor]["is_rotation"] == False ?')
            
        COR_shift = [float(x) for x in COR_shift]
        print(('shifting lookuptable for ',rotmotor,' with ',COR_motors,' COR by ',COR_shift))
        lookup = self.tmp_lookup[rotmotor]

        if self.motors[rotmotor]['invert']:
            rot_rad = -lookup[rotmotor]/180.0*np.pi
        else:
            rot_rad = lookup[rotmotor]/180.0*np.pi
        lookup[COR_motors[0]] += COR_shift[0]*np.cos(rot_rad) - COR_shift[1]*np.sin(rot_rad)
        lookup[COR_motors[1]] += COR_shift[0]*np.sin(rot_rad) + COR_shift[1]*np.cos(rot_rad)


### debugging/fake functions
            
    def show_lookup(self):
        lc = self.lookup
        for (k,lookup) in lc.items():
            print(k, lookup)

    def show_single_lookup(self, lookupmotor, temp=False):
        if temp:
            lc = self.temp_lookup[lookupmotor]
        else:
            lc = self.lookup[lookupmotor]
        for (k,lookup) in lc.items():
            print(k, lookup)

    def dummy_mv(self, mk, pos, pos_dc=None):
        if type(pos_dc) == dict:
            self.sync_pos(pos_dc) 
        self.pos[mk] = pos
        self.dynamic_lookup.mockup_currpos({mk:pos})
        return pos

    def dummy_mvr(self, mk, distance, pos_dc=None):
        if type(pos_dc) == dict:
            self.sync_pos(pos_dc) 
        start_pos = self.pos[mk]
        target_pos = start_pos+distance
        self.dummy_mv(mk, target_pos)

    def corrected_move(self, mk, target_pos, dynamic=False, pos_dc=None):
        if type(pos_dc) == dict:
            self.sync_pos(pos_dc) 
        startpos_dc = self.pos
        if mk in list(self.lookup.keys()):
            corr = self.get_lookup_correction(mk, startpos_dc, target_pos, dynamic=dynamic)
            self.dummy_mv(mk, target_pos)
            self.multiple_mvr(corr)
        else:
            self.dummy_mv(mk, target_pos)

    def multiple_mvr(self, target_pos, pos_dc=None):
        if type(pos_dc) == dict:
            self.sync_pos(pos_dc) 
        for (k,v) in target_pos.items():
            self.dummy_mvr(k,v)
            
    def multiple_mv(self, target_pos, pos_dc=None):
        if type(pos_dc) == dict:
            self.sync_pos(pos_dc) 
        for (k,v) in target_pos.items():
            self.dummy_mv(k,v)

class LUT_TOMO_Navitar(LUT_Anyberg):
    ''' 
    specific lookuptable interface for the phi-kappa gonio based on the smaract motors on strx,y,z in 
    as used by the CamView stage class in the nanolab optical setup
    '''
    MDC = dict(
        kappa = dict(
            is_rotation = True,
            invert      = False,
            COR_motors  = ['x','z'],
            default_pos = 0, # at this position lookups do not interfere, i.e. this is the default position to make lookuptables for any other motors
        ),
        phi = dict(
            is_rotation = True,
            invert      = True,
            COR_motors  = ['y','x'],
            default_pos = 0,
        ),
        x = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),
        y = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),
        z = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),

    )

    def __init__(self):
        self.lookup_fnames ={}
        self.lookup = {}
        self.dynamic_lookup = {}
        
        self.motors = self.MDC
        # initial positions
        pos_list = [(k,0) for k in list(self.motors.keys())]
        self.pos = dict(pos_list)

        [self.init_current_pos(mot) for mot in ['phi','kappa']]
        
        # external to internal motorname translation
        self.mto_lookup = mto_lookup = dict(
            phi = "phi",
            kappa = "kappa",
            x = "x",
            y = "y",
            z = "z")
        
        tems = list(mto_lookup.items())
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)
        self.show()

        
    def link_dynamic(self, load=False):
        '''
        relink the dynamic lookups
        '''
        
        self.phi_dynamic_lookup = dyl = LookupDict_Phi_XZKappa(self.motors, self.lookup)
        dyl.mockup_currpos(self.pos)

        self.dynamic_lookup['phi'] = dyl
        self.dynamic_lookup['kappa'] = self.lookup['kappa']

        print("dynam lookup done.")
 
class LUT_Navitar(LUT_Anyberg):
    ''' 
    specific lookuptable interface for the phi-kappa gonio based on the smaract motors on strx,y,z in 
    as used by the CamView stage class in the nanolab optical setup
    '''
    MDC = dict(
        kappa = dict(
            is_rotation = True,
            invert      = True,
            COR_motors  = ['x','z'],
            default_pos = 0, # at this position lookups do not interfere, i.e. this is the default position to make lookuptables for any other motors
        ),
        phi = dict(
            is_rotation = True,
            invert      = True,
            COR_motors  = ['x','y'],
            default_pos = 0,
        ),
        x = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),
        y = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),
        z = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),

    )

    def __init__(self):
        self.lookup_fnames ={}
        self.lookup = {}
        self.dynamic_lookup = {}
        
        self.motors = self.MDC
        # initial positions
        pos_list = [(k,0) for k in list(self.motors.keys())]
        self.pos = dict(pos_list)

        [self.init_current_pos(mot) for mot in ['phi','kappa']]
        
        # external to internal motorname translation
        self.mto_lookup = mto_lookup = dict(
            phi = "phi",
            kappa = "kappa",
            x = "x",
            y = "y",
            z = "z")
        
        tems = list(mto_lookup.items())
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)
        self.show()

        
    def link_dynamic(self, load=False):
        '''
        relink the dynamic lookups
        '''
        
        self.phi_dynamic_lookup = dyl = LookupDict_Phi_XZKappa(self.motors, self.lookup)
        dyl.mockup_currpos(self.pos)

        self.dynamic_lookup['phi'] = dyl
        self.dynamic_lookup['kappa'] = self.lookup['kappa']

        print("dynam lookup done.")
       
class LUT_VO2(LUT_Anyberg):
    ''' 
    specific lookuptable interface for the phi-kappa gonio based on the smaract motors on strx,y,z in 
    as used by the CamView stage class in the nanolab optical setup
    '''
    MDC = dict(
        smkappa = dict(
            is_rotation = True,
            invert      = True,
            COR_motors  = ['nnp4','nnp6'],
            default_pos = 0, # at this position lookups do not interfere, i.e. this is the default position to make lookuptables for any other motors
        ),
        smphi = dict(
            is_rotation = True,
            invert      = True,
            COR_motors  = ['nnp4','nnp5'],
            default_pos = 0,
        ),
        nnp4 = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),
        nnp5 = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),
        nnp6 = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),
        smx = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),
        smy = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
        ),      
    )

    def __init__(self):
        self.lookup_fnames ={}
        self.lookup = {}
        self.dynamic_lookup = {}
        
        self.motors = self.MDC
        # initial positions
        pos_list = [(k,0) for k in list(self.motors.keys())]
        self.pos = dict(pos_list)

        [self.init_current_pos(mot) for mot in ['smphi','smkappa']]
        
        # external to internal motorname translation
        self.mto_lookup = mto_lookup = dict(
            smphi = "smphi",
            smkappa = "smkappa",
            nnp4 = "nnp4",
            nnp5 = "nnp5",
            nnp6 = "nnp6",
            smx = "smx",
            smy = "smy")
        
        tems = list(mto_lookup.items())
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)
        self.show()

        
    def link_dynamic(self, load=False):
        '''
        relink the dynamic lookups
        not dynamic!
        '''
        
        self.phi_dynamic_lookup = dyl = LookupDict(self.motors)
        dyl.mockup_currpos(self.pos)

        self.dynamic_lookup['smphi'] = self.lookup['smphi']
        self.dynamic_lookup['smkappa'] = self.lookup['smkappa']

        print("dynam lookup done.")

class LUT_Generic(LUT_Anyberg):
    '''
    if there are no dynamic dependecies of lookuptables, this class can handle your stage application
    '''
    def __init__(self, mot_dict, stagegeometry_dict):
        self.lookup_fnames ={}
        [self.lookup_fnames.update({lookupmotor:''}) for lookupmotor in stagegeometry_dict['COR_motors'].keys()]
        self.lookup = {}
        self.dynamic_lookup = {}

        LUT_mot_dict={}
        for motor in mot_dict:
            if motor in stagegeometry_dict['COR_motors']:
                LUT_mot_dict.update({motor:{'is_rotation':True,
                                   'invert':stagegeometry_dict['COR_motors'][motor]['invert'],
                                   'COR_motors':stagegeometry_dict['COR_motors'][motor]['motors'],
                                   'default_pos':0}})
            else:
                LUT_mot_dict.update({motor:{'is_rotation':False,
                                   'invert':False,
                                   'default_pos':0}})
        print('LUT_mot_dict =')
        print(LUT_mot_dict)
        self.motors = LUT_mot_dict
        # initial positions
        pos_list = [(k,LUT_mot_dict[k]['default_pos']) for k in list(LUT_mot_dict.keys())]
        self.pos = dict(pos_list)
        print(pos_list)

        [self.init_current_pos(mot) for mot in []]
        
        # external to internal motername translation
        mto_lookup={}
        [mto_lookup.update({motor:motor}) for motor in self.motors]
        self.mto_lookup = mto_lookup
        
        tems = list(mto_lookup.items())
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)

        self.show()

class LUT_EH3_smrhex(LUT_Anyberg):
    '''
    adapted from LUT_Generic
    '''
    # definitions for the lookuptable functions
    MDC = dict(rotz = dict(
            is_rotation = True,
            invert      = False,
            COR_motors  = ['x','y'],
            default_pos = 0,
            ),
                          x = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
            ),
                          y = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
            ),
                          z = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,      
            ),
               )

    # external to internal motorname translation
    MOTORNAMES_DICT={'smrot':'rotz',
                    'coarse_x':'x',
                    'coarse_y':'y',
                    'coarse_z':'z'}

    def __init__(self, mot_dict=MOTORNAMES_DICT, stagegeometry_dict=MDC):
        self.lookup_fnames ={}
        self.lookup = {}
        self.dynamic_lookup = {}

        self.motors = stagegeometry_dict
        # initial positions
        pos_list = [(k,self.motors[k]['default_pos']) for k in list(self.motors.keys())]
        self.pos = dict(pos_list)

        [self.init_current_pos(mot) for mot in ['rotz']]
        
        self.mto_lookup = mot_dict
        
        tems = list(self.mto_lookup.items())
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)
        self.show()


class LUT_EH3_smrpiezo(LUT_Anyberg):
    '''
    adapted from LUT_Generic
    '''
    MDC = dict(rotz = dict(
            is_rotation = True,
            invert      = False,
            COR_motors  = ['x','y'],
            default_pos = 0,
            ),
                          x = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
            ),
                          y = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
            ),
                          z = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,      
            ),
               )

    # external to internal motorname translation
    MOTORNAMES_DICT={'smrot':'rotz',
                     'fine_x':'x',
                     'fine_y':'y',
                     'fine_z':'z'}


    def __init__(self, mot_dict=MOTORNAMES_DICT, stagegeometry_dict=MDC):
        self.lookup_fnames ={}
        self.lookup = {}
        self.dynamic_lookup = {}

        self.motors = stagegeometry_dict
        # initial positions
        pos_list = [(k,self.motors[k]['default_pos']) for k in list(self.motors.keys())]
        self.pos = dict(pos_list)

        [self.init_current_pos(mot) for mot in ['rotz']]
        
        self.mto_lookup = mot_dict
        
        tems = list(self.mto_lookup.items())
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)
        self.show()
    

class LUT_EH3_smrhexpiezo(LUT_Anyberg):
    '''
    adapted from LUT_Generic
    '''
    MDC = dict(rotz = dict(
            is_rotation = True,
            invert      = False,
            COR_motors  = ['x','y'],
            default_pos = 0,
            ),
               x = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
            ),
               y = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
            ),
               z = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,      
            ),
               hex_x = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,      
            ),
               hex_y = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,      
            ),
               hex_z = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,      
            ),
               )

    # external to internal motorname translation
    MOTORNAMES_DICT={'smrot':'rotz',
                     'fine_x':'x',
                     'fine_y':'y',
                     'fine_z':'z',
                     'coarse_x':'hex_x',
                     'coarse_y':'hex_y', 
                     'coarse_z':'hex_z'}

    def __init__(self, mot_dict=MOTORNAMES_DICT, stagegeometry_dict=MDC):
        
        self.lookup_fnames ={}
        self.lookup = {}
        self.dynamic_lookup = {}

        self.motors = stagegeometry_dict
        # initial positions
        pos_list = [(k,self.motors[k]['default_pos']) for k in list(self.motors.keys())]
        self.pos = dict(pos_list)

        [self.init_current_pos(mot) for mot in ['rotz']]

        self.mto_lookup = mot_dict
        
        tems = list(self.mto_lookup.items())
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)
        self.show()

class LUT_EH3_phikappamars(LUT_Anyberg):
    '''
    adapted from LUT_Generic
    '''
    MDC = dict(phi = dict(
            is_rotation = True,
            invert      = False,
            COR_motors  = ['hex_x','hex_y'],
            default_pos = 0,
            ),
               kappa = dict(
            is_rotation = True,
            invert      = True,
            COR_motors  = ['hex_x','hex_z'],
            default_pos = 0,
            ),  
            x = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
            ),
               y = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,
            ),
               z = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,      
            ),
               hex_x = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,      
            ),
               hex_y = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,      
            ),
               hex_z = dict(
            is_rotation = False,
            invert      = False,
            default_pos = 0,      
            ),
               )

    # external to internal motorname translation
    MOTORNAMES_DICT={'smphi'  :'phi',
                     'smkappa':'kappa',
                     'fine_x2':'x',
                     'fine_y2':'y',
                     'fine_z2':'z',
                     'coarse_x':'hex_x',
                     'coarse_y':'hex_y', 
                     'coarse_z':'hex_z'}

    def __init__(self, mot_dict=MOTORNAMES_DICT, stagegeometry_dict=MDC):
        
        self.lookup_fnames ={}
        self.lookup = {}
        self.dynamic_lookup = {}

        self.motors = stagegeometry_dict
        # initial positions
        pos_list = [(k,self.motors[k]['default_pos']) for k in list(self.motors.keys())]
        self.pos = dict(pos_list)

        [self.init_current_pos(mot) for mot in ['phi','kappa']]

        self.mto_lookup = mot_dict
        
        tems = list(self.mto_lookup.items())
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)
        self.show()
          
    def link_dynamic(self, load=False):
        '''
        relink the dynamic lookups
        '''
        
        self.phi_dynamic_lookup = dyl = LookupDict_Phi_hexXZKappa(self.motors, self.lookup)
        dyl.mockup_currpos(self.pos)

        self.dynamic_lookup['phi'] = dyl
        self.dynamic_lookup['kappa'] = self.lookup['kappa']

        print("dynam lookup done.")

class PosPlot(object):

    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.phi = []
        self.kappa = []

    def logit(self, zf):
        p = zf.pos
        self.x.append(p['x'])
        self.y.append(p['y'])
        self.z.append(p['z'])
        self.phi.append(p['phi'])
        self.kappa.append(p['kappa'])

    def plot(self):
        plt.plot(self.x, self.y)

    def plotrx(self):
        plt.plot(self.phi, self.x)

    def plotry(self):
        plt.plot(self.phi, self.y)

        
def _test2():
    pl_s = PosPlot()
    pl_d = PosPlot()
    zf = LUT_Feldberg(lookup1_fname='lookuptable_phi_dense.dat', lookup2_fname='kappa_dense.dat')
    zf.show_lookup()
    zf.compile()

    #

    zf.correctedmove('kappa', -70.0, dynamic=True)

    for x in range(0, 360):
        zf.corrected_move('phi', float(x), dynamic=False)
        pl_s.logit(zf)

    for x in range(0, 360):
        zf.corrected_move('phi', float(x), dynamic=True)
        pl_d.logit(zf)

    plt.ion()
    if 1:
        pl_s.plot()
        pl_d.plot()
        pl_s.plot()

    if 0:
        pl_s.plotrx()
        pl_s.plotry()
        pl_d.plotrx()
        pl_d.plotry()

    input('...')


def _test1():
    zf = LUT_Feldberg()
    zf.load_lookup('lookuptable_phi_dense.dat')
    zf.show_lookup()
    for phipos in range(0,180,30):
        print(phipos)
        print(zf.get_lookup_correction("phi", phipos*1.0, phipos+30.0))

_test = _test2

if __name__ == '__main__':
    _test()
