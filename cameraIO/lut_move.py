import collections
from matplotlib import pyplot

import numpy as np

from fileIO.datafiles import open_data, save_data
from cameraIO.CamView_lookup import LookupDict, LookupDict_Phi_XZKappa



class LUT_Anyberg(object):
    '''
    collects all methods that are a priory independent of the respective setup geometry
    see LUT_Feldberg for the phi-kappa gonio geometry example
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
        self.lut_fnames ={}
        self.lookup = {}
        self.dynamiclut = {}
        self.motors = self.MDC
        # initial positions
        pos_list = [(k,0) for k in self.MDC.keys()]
        self.pos = dict(pos_list)
        
        # external to internal motorname translation
        self.mto_lut = mto_lut = dict(
            phi = "phi",
            kappa = "kappa",
            fine_x = "x",
            fine_y = "y",
            fine_z = "z"
        )
        tems = mto_lut.items()
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)

        pass

    def mto_eig_dict(self, lut_dc):
        '''
        return the translated dict of <{mot1_internal_name:pos, etc. }>  into <{mot1_external_name:pos, etc. }> 
        '''
        tems = lut_dc.items()
        kdc = self.mto_eig
        stem = [(kdc[k],v) for (k,v) in tems]
        res = dict(stem)
        return res

    def mto_lut_dict(self, eig_dc):
        '''
        return the translated dict of <{mot1_external_name:pos, etc. }>  into <{mot1_internal_name:pos, etc. }> 
        filters for names listed in MDC
        '''
        tems = eig_dc.items()
        kdc = self.mto_lut
        stem = [(kdc[k],v) for (k,v) in tems if k in self.MDC.keys()]
        res = dict(stem)
        return res


    def get_lut_correction(self, function, startpos_dc, end_pos, dynamic=True):
        '''
        returns a dict with {mot1_name:mot1_correction .. etc} to correct the movement of <function> from <start_pos> to <end_pos>
        <dynamic>[bool] selects whether the dynamic lookuptable is used.
        '''
        sync_pos(self, startpos_dc)
        corrdc = dict()

        start_pos = startpos_dc[function]
        
        if dynamic:
            lookup = self.dynamiclut
        else:
            lookup = self.lookup

        if self.motors[function]['is_rotation']:
            start_pos = start_pos % 360.0
            end_pos   = end_pos   % 360.0
        if function in lookup.keys():
            for mot in lookup[function].keys():
                if mot != function:
                    start_correction = np.interp(start_pos, lookup[function][function], lookup[function][mot])
                    end_correction   = np.interp(end_pos, lookup[function][function], lookup[function][mot])
                    correction = end_correction-start_correction
                    corrdc[mot] = correction
            return corrdc
        else:
            print 'no lookuptable found for ' , function
            return None

    def sync_pos(self, pos_dc):
        '''
        get outside update of the current positions
        TODO test: should accept ne.read_all_motic_pos() and select for all motors in self.MCD.keys()
        '''
        tems = pos_dc.items()
        kdc = self.mto_lut
        stem = [(kdc[k],float(v)) for (k,v) in tems if k in self.MDC.keys()]
        res = dict(stem)
        self.pos.update(res)
        for mot in self.dynamiclut.keys():
            self.dynamiclut[mot].mockup_currpos.update(res)
        print('updated the currentposition to the lookuptabe interface')
        print res

### methods migrated here from the CamView stage class:
        
    def load_lut(self, fname):
        data, header           =  open_data.open_data(fname)
        lut_motor = header[0]
        print "found lookuptable for motor: ", lut_motor
        print 'using (unsorted) motors ', header[1:]
        self.lookup[header[0]] = LookupDict(self, self.motors)
        for i, mot in enumerate(header):
            self.lookup[header[0]][mot] = data[:,i]
        self.lut_fnames.update({lut_motor:fname})
            
    def save_lut(self, function, savename=None):
        data   = np.zeros(shape = (len(self.lookup[function][function]),len(self.lookup[function].keys())))

        if not type(savename) == type('asfd'):
            savename = (self.lut_fnames[function])
                        
        unsorted_header = self.lookup[function].keys()
        header    = []
        header.append(unsorted_header.pop(unsorted_header.index(function)))
        header   += unsorted_header
        for i, mot in enumerate(header):
            data[:,i] = self.lookup[function][mot]
        
        save_data.save_data(savename, data, header = header)
                
    def plot_lut(self, motor='phi', plot_motors=None):
        lut = self.lookup[motor]
        for mot in lut.keys():
            if not mot==motor:
                dummy, ax1 = plt.subplots(1) 
                ax1.set_title('%s vs %s'%(mot,motor))                              
                ax1.plot(lut[motor],lut[mot])
                
    def update_lut(self, motor, shift_lookup, overwrite=False, lookup_motors=None):
        '''
        if overwrite == False, shift_lookup is added to the old lookup as a relative change as if shift lookup was measured using the coorections of self.lookup as a base correction.
        if lookup_motors == None: # assume the same motors as for COR
            lookup_motors = self.stagegeometry['COR_motors'][motor]['motors']
        '''
        # we have to add or update the values to the old lookup


        if motor not in self.lookup.keys():
            self.lookup.update({motor:{}})
        positions = shift_lookup.pop(motor)
        lookup_motors = shift_lookup.keys()
        
        if self.motors[motor]['is_rotation']:
            positions=np.asarray(positions)
            positions=positions % 360.0

        # after mod the list is no longer sorted but it needs to be for the np.interp and my following update of the old lookup
        # so now sort:
        positions_list=list(positions)
        together_array = np.zeros(shape = (len(positions), len(lookup_motors)+1))
        together_array[:,0]=positions
        
        for i,mot in enumerate(lookup_motors):
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
        for i in  range(len(lookup_motors)):
            shift.append(np.asarray([x[i+1] for x in sorted_together]))

        #print('positions')
        #print(positions)
        #print('shift')
        #print(shift)
            
        if not overwrite:
            print('updating old lookuptable')
            old_positions = list(self.lookup[motor][motor])
            old_mots = []
            for i,mot in enumerate(lookup_motors):
                old_mots.append(list(self.lookup[motor][mot]))
                d_i = np.asarray((np.interp(positions, self.lookup[motor][motor], self.lookup[motor][mot])))
                s_i = np.asarray(shift[i])
                shift[i] = list(d_i + s_i)


            for i, new_theta in enumerate(new_positions):
                j = 0
                old_theta = old_positions[j]
                while new_theta > old_theta:
                    j+=1
                    if j==len(old_positions):
                        old_positions.append(new_theta)

                        for k in range(len(old_mots)):
                            old_mots[k].append(shift[i][k])
                        #old_mot1.append(new_mot1[i])
                        break
                    else:
                        old_theta = old_positions[j]
                else:
                    if new_theta == old_theta:
                        for k in range(len(old_mots)):
                            old_mots[k] = shift[i][k]

                        #old_mot1[j]=new_mot1[i]
                    else:
                        old_positions.insert(j,new_theta)
                        for k in range(len(old_mots)):
                            old_mots[k].insert(j,shift[i][k])
                        #old_mot1.insert(j,new_mot1[i])

            new_positions = old_positions
            new_mots=[]
            for k in range(len(old_mots)):
                new_mots.append(old_mot[k])
            #new_mot0 = old_mot0
        else:
            # just overwrite the old lookup
            print('writing new lookuptable')
            new_positions = positions
            print new_positions
            print 'new_positions'
            new_mots=[]
            for k in range(len(shift)):
                new_mots.append(shift[k])
                #new_mot1   = shift_1
            
        self.lookup[motor].update({motor: np.asarray(new_positions)})
        for i,mot in enumerate(lookup_motors):
            self.lookup[motor].update({mot: np.asarray(new_mots[i])})
        self.lookup[motor].update({motor: np.asarray(new_positions)})

        #self.lookup[motor].update({mot1: np.asarray(new_mot1)})

    def tmp_to_lut(self, lookupmotor):
        '''
        overwrites the old lookup for <lookupmotor> with the values from tmp_lookup
        '''
        print 'overwriting lookup for %s with:' % lookupmotor
        lookup_motors = []
        for mot,values in self.tmp_lookup[lookupmotor].items():
            lookup_motors.append(mot)
            print mot
            print values
        self.update_lut(motor=lookupmotor, shift_lookup=self.tmp_lookup[lookupmotor], overwrite=True, lookup_motors=lookup_motors)      
          
    def initialize_temp_lut(self, lookup_motor = 'phi', save_motor_list=['x','y','z']):
        '''
        initialize the creation of a new lookuptable with add_pos_to_temp_lut.
        only the positions of motors in save_motor_list will be stored in the new lookuptable
        '''

        self.tmp_lookup = {}
        self.tmp_lookup.update({lookup_motor:{}})
        
        for mot in save_motor_list:
            self.tmp_lookup[lookup_motor].update({mot:[]})
        self.tmp_lookup[lookup_motor].update({lookup_motor:[]})

        print 'ready to save lookup positions for ', save_motor_list

    def add_pos_to_temp_lut(self, lookupmotor, motor_dc):
        '''
        collect positions that will form a new lookuptable in the tmp_lookup
        does not need to be sorted
        can use old lookuptable to get to the new positions
        can be used outside of the default positions (untested)
        '''
        tmp_lut = self.tmp_lookup[lookupmotor]


        for mk in self.lookup.keys():
            if not mk == lookup_motor:
                target_pos = self.motors[mk]['default_pos']
                start_pos = self.wm(mk)
                corr = self.get_lut_correction(mk, start_pos, target_pos, dynamic=True)
                for (k,v) in corr.items():
                    motor_dc[k] += v
        
        for mot in self.tmp_lookup[lookupmotor].keys():
            tmp_lut[mot].append(motor_dc[mot]) 

    def shift_COR_of_lut(self,
                     rotmotor='rot',
                     COR_shift = [0.1,0.1]):
        '''
        shift the COR of <rotmotor> by adding the corresponding shifts to the COR_motors
        COR_motors=self.motors[rotmotor]['COR_motors']
        '''
        COR_motors=self.motors[rotmotor]['COR_motors']

        if not self.motors[rotmotor]['is_rotation']:
            print 'does it make sense to shift a COR if self.motors[rotmotor]["is_rotation"] == False ?'
            
        COR_shift = [float(x) for x in COR_shift]
        print('shifting lookuptable for ',rotmotor,' with ',COR_motors,' COR by ',COR_shift)
        lookup = self.lookup[rotmotor]

        if self.motors[rotmotor]['invert']:
            rot_rad = -lookup[rotmotor]/180.0*np.pi
        else:
            rot_rad = lookup[rotmotor]/180.0*np.pi
        lookup[COR_motors[0]] += COR_shift[0]*np.cos(rot_rad) - COR_shift[1]*np.sin(rot_rad)
        lookup[COR_motors[1]] += COR_shift[0]*np.sin(rot_rad) + COR_shift[1]*np.cos(rot_rad)
        


### debugging/fake functions
            
    def show_lut(self):
        lc = self.lookup
        for (k,lut) in lc.iteritems():
            print k, lut

    def dummy_mv(self, mk, pos):
        self.sync_pos()
        self.pos[mk] = pos
        self.dynamiclut['phi'].mockup_currpos({mk:pos})
        return pos

    def dummy_mvr(self, mk, distance):
        self.sync_pos()
        start_pos = self.pos[mk]
        target_pos = start_pos+distance
        self.dummy_mv(mk, target_pos)

    def corrected_move(self, mk, target_pos, dynamic=False):
        self.sync_pos()
        start_pos = self.pos[mk]
        if mk in self.lookup.keys():
            self.dummy_mv(mk, target_pos)
            corr = self.get_lut_correcton(mk, start_pos, target_pos, dynamic=dynamic)
            self.multiple_mvr(corr)
        else:
            self.dummy_mv(mk, target_pos)

    def multiple_mvr(self, target_pos):
        for (k,v) in target_pos.iteritems():
            self.dummy_mvr(k,v)
            
    def multiple_mv(self, target_pos):
        for (k,v) in target_pos.iteritems():
            self.dummy_mv(k,v)



class AJLutMoveUserInfra(object):
    pass

#class LUT_Feldberg(Neron1):


class LUT_Navitar(LUT_Anyberg):
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
        self.lut_fnames ={}
        self.lut_fnames.update({'phi':''})
        self.lut_fnames.update({'kappa': ''})
        self.lookup = {}
        self.dynamiclut = dict()
        self.motors = self.MDC
        # initial positions
        pos_list = [(k,0) for k in self.MDC.keys()]
        self.pos = dict(pos_list)
        
        # external to internal motername translation
        self.mto_lut = mto_lut = dict(
            phi = "phi",
            kappa = "kappa",
            x = "x",
            y = "y",
            z = "z"
        )
        tems = mto_lut.items()
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)

    def compile(self):
        '''
        load all lookuptables and link the dynamic lookups
        '''
        for function in self.lut_fnames.keys():
            self.load_lut(self.lut_fnames[function])

        print "luts loaded"

        self.phi_dynamiclut = dyl = LookupDict_Phi_XZKappa(self.motors, self.lookup)
        dyl.mockup_currpos(self.pos)

        self.dynamiclut['phi'] = dyl
        self.dynamiclut['kappa'] = self.lookup['kappa']

        print "dynam lut done."
    
class LUT_Feldberg(LUT_Anyberg):
    ''' 
    specific lookuptable interface for the phi-kappa gonio based on the smaract motors on strx,y,z in EH2
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

    def __init__(self, lut1_fname, lut2_fname):
        self.lut_fnames ={}
        self.lut_fnames.update({'phi': lut1_fname})
        self.lut_fnames.update({'kappa': lut2_fname})
        self.lookup = {}
        self.dynamiclut = dict()
        self.motors = self.MDC
        # initial positions
        pos_list = [(k,0) for k in self.MDC.keys()]
        self.pos = dict(pos_list)
        
        # external to internal motername translation
        self.mto_lut = mto_lut = dict(
            phi = "phi",
            kappa = "kappa",
            fine_x = "x",
            fine_y = "y",
            fine_z = "z"
        )
        tems = mto_lut.items()
        semt = [(v,k) for (k,v) in tems]
        self.mto_eig = dict(semt)

    def compile(self):
        '''
        load all lookuptables and link the dynamic lookups
        '''
        for function in self.lut_fnames.keys():
            self.load_lut(self.lut_fnames[function])

        print "luts loaded"

        self.phi_dynamiclut = dyl = LookupDict_Phi_XZKappa(self.motors, self.lookup)
        dyl.mockup_currpos(self.pos)

        self.dynamiclut['phi'] = dyl
        self.dynamiclut['kappa'] = self.lookup['kappa']

        print "dynam lut done."


            
# develop fix point tracking

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
        pyplot.plot(self.x, self.y)

    def plotrx(self):
        pyplot.plot(self.phi, self.x)

    def plotry(self):
        pyplot.plot(self.phi, self.y)

        
def _test2():
    pl_s = PosPlot()
    pl_d = PosPlot()
    zf = LUT_Feldberg(lut1_fname='lookuptable_phi_dense.dat', lut2_fname='kappa_dense.dat')
    zf.show_lut()
    zf.compile()

    #

    zf.corrected_move('kappa', -70.0, dynamic=True)

    for x in range(0, 360):
        zf.corrected_move('phi', float(x), dynamic=False)
        pl_s.logit(zf)

    for x in range(0, 360):
        zf.corrected_move('phi', float(x), dynamic=True)
        pl_d.logit(zf)

    pyplot.ion()
    if 1:
        pl_s.plot()
        pl_d.plot()
        pl_s.plot()

    if 0:
        pl_s.plotrx()
        pl_s.plotry()
        pl_d.plotrx()
        pl_d.plotry()

    raw_input('...')


def _test1():
    zf = LUT_Feldberg()
    zf.load_lut('lookuptable_phi_dense.dat')
    zf.show_lut()
    for phipos in range(0,180,30):
        print phipos
        print zf.get_lut_correction("phi", phipos*1.0, phipos+30.0)

_test = _test2

if __name__ == '__main__':
    _test()
