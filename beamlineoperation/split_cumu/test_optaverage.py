""" Lego-like azimuthal integration only compo flow.

"""
from __future__ import print_function
from builtins import str
import numpy as np
from o8qq.qqudo1.api import utilapi as uti
from o8qq.qqudo1.api import ptiapi


from o8x3.compomatrix import compoflow as cmf
from o8x3.compomatrix import elements_c1 as elc_1
from o8x3.compomatrix import compopatterns as compa
from o8x3.compomatrix import compomgr as mgr
from o8x3.compomatrix import elements_c3 as elc_3
from o8x3.compomatrix import elements_c4 as elc_4
from o8x3.compomatrix import elements_c9 as elc_9
from o8x3.compute     import compute7 as compu

#from britta.compoflow import elements_c6 as elc_6
#import hack_17jun2016_elc6 as elc_6
import hack_09jul2016_elc6 as elc_6
from o8x3.adapt       import MyTools_inteface
from o8x3.util.utils  import dstring_to_numbers


def generate_local(srckey, keys):
    keys = keys.split()
    l = []
    for k in keys:
        l.append("%s     = %s.%s" % (k, srckey, k))
    l.append("")
    return '\n'.join(l)

class error(Exception): pass


class NewCompo(mgr.CompoMgr1):

    _PARS = dict()
    _PARS.update(mgr.CompoMgr1._PARS)
    _PARS.update(
        dict(
            tipe   = "compo_mgr",
            name   = "compo_mgr",
            spt    = None
        )
    )

    def __init__(self, *po, **params):
        mgr.CompoMgr1.__init__(self, **params)

    def _init(self):
        rp = self.spt.get_rootnode()
        
        self.program = program = rp.project.script.program

        self.flows = []

        ____________________________________EX = generate_local("program",
            """ outindex
            """)
        exec(____________________________________EX)

        self.outcfg_el = outcfg_el = elc_1.Config()
        outcfg_el.set_repo('outindex', outindex)

        self.flow1 = flow1 = self._init_flow1()
        self.flows.append(flow1)

    def post_flow(self):
        pass

    def run(self):
        _tmer=uti.make_timer()
        _tmer.reset()
        fl1 = self.flow1

        spt = self.spt

        ____________________________________EX = generate_local("self", """

            outcfg_el program

        """); exec(____________________________________EX)
        
        self.flow1.run_through()
        self.post_flow()

        print('-'*40)
        for (i,l) in enumerate(fl1.layers):
            for (k,v) in l.dc.items():
                print("%10s = %s" % (k,str(v)))
        print("elapsed time:", _tmer.time())


    def _init_flow1(self):

        # ##########################################################
        # flow adapted from BoneCompisite (Britta Weinhausen)
        #

        spt = self.spt

        ____________________________________EX = generate_local("self", """

            outcfg_el program

        """); exec(____________________________________EX)


        ____________________________________EX = generate_local("program", """

            infiles infi_mode outfiles

            binning threshold skip use_numbers
        """); exec(____________________________________EX)

        if 'auto' == threshold:
            pass
        elif '<pytohn-none>' == threshold:
            threshold = None
        else:
            threshold = float(threshold)  # this hs to be fixed !!!

        # prepare input
        infiles.mode = infi_mode
        use_numbers = dstring_to_numbers(use_numbers)

        # start initializing composite flow
        cf = cmf.CompoFlow()
        base_layer   = cf.add_layer()
        head_layer   = cf.add_layer()
        main_layer   = cf.add_layer()
        write_layer  = cf.add_layer()
    
        # this is now fully compatible with CompoStdFlowHead1
        # and it can read eiger data in eigerdata3 fashion    

        head_compa = compa.CompoStdFlowHead7(   
            binning = None,
            threshold = None,
            infiles=infiles,
            indices = use_numbers,
            base_layer = base_layer,
            head_layer = head_layer,
        )


        # main layers
        idx_el = head_compa.index_el
        src_el = head_compa.src_el

        idx_log_el = elc_1.IndexLogger()
        idx_log_el.link( 'r_index', idx_el.repo, 'index')
        base_layer.add_elem(idx_log_el, k='idxlogger')

        avg_el = elc_3.OptAverageArr(threshold=threshold)
        avg_el.link( 'r_index', idx_el.repo, 'index')
        avg_el.link( 'r_leng', idx_el.repo, 'leng')
        avg_el.link( 'r_arr', src_el.repo, 'arr')
        avg_el.link( 'r_prep_arr', src_el.repo, 'prep_arr')
        main_layer.add_elem(avg_el, k='average')

        ouch_el = elc_1.OuchSparseWriter(outfiles = outfiles, skip = skip)
        ouch_el.link( 'r_index', idx_el.repo, 'index')
        ouch_el.link( 'r_outindex', outcfg_el.repo, 'outindex')
        ouch_el.link( 'r_out_arr', avg_el.repo, 'avg_arr')
        write_layer.add_elem(ouch_el, k='ouch')


        # ##########################################################
        # add further processing in this flow here ...
        #


        # ##########################################################
        # end of flow
        #

        return cf



def run_1(spt, *p):
    themgr = NewCompo(spt=spt)
    themgr.run()
