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
#import hack_09jul2016_elc6 as elc_6
from o8x3.adapt       import MyTools_inteface
#local import
from limited_c9_aj import Hit_and_more

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

        print '-'*40
        for (i,l) in enumerate(fl1.layers):
            for (k,v) in l.dc.iteritems():
                print "%10s = %s" % (k,str(v))
        print "elapsed time:", _tmer.time()


    def _init_flow1(self):

        # ##########################################################
        # flow adapted from BoneCompisite (Britta Weinhausen)
        #

        spt = self.spt

        ____________________________________EX = generate_local("self", """

            outcfg_el program

        """); exec(____________________________________EX)


        ____________________________________EX = generate_local("program", """

            scan

            infiles outfiles output outoffset skip

            azim_mode infi_mode


            binning threshold hit_threshold
        """); exec(____________________________________EX)

        ____________________________________EX = generate_local("scan", """

            specfile scannmb meshshape

        """); exec(____________________________________EX)

        if 'auto' == threshold:
            pass
        elif '<pytohn-none>' == threshold:
            threshold = None
        else:
            threshold = float(threshold)  # this hs to be fixed !!!

        # prepare input
        infiles.mode = infi_mode

        # start initializing composite flow
        cf = cmf.CompoFlow()
        base_layer   = cf.add_layer()
        head_layer   = cf.add_layer()
        main_layer   = cf.add_layer()
        compo_layer  = cf.add_layer()
        write_layer  = cf.add_layer()
    
        # this is now fully compatible with CompoStdFlowHead1
        # and it can read eiger data in eigerdata3 fashion    

        head_compa = compa.CompoStdFlowHead7(   
            binning = binning,
            threshold = None,
            infiles=infiles,
            base_layer = base_layer,
            head_layer = head_layer,
        )


        
        compo_compa = compa.CompoWritePattern1(
            mesh_shape = meshshape,
            outfiles = outfiles,
            ooff = outoffset,
            outcfg = outcfg_el,
            n_items = 5,
            skip = skip,
            idx_el = head_compa.index_el,
            compo_layer = compo_layer,
            write_layer = write_layer,
        )

 
     
        # main layers
        idx_el = head_compa.index_el
        src_el = head_compa.src_el

        idx_log_el = elc_1.IndexLogger()
        idx_log_el.link( 'r_index', idx_el.repo, 'index')
        base_layer.add_elem(idx_log_el, k='idxlogger')

        hp_a1000 = Hit_and_more(
            thd_low=hit_threshold, thd_high=threshold)
        hp_a1000.link( 'r_index', idx_el.repo, 'index')
        hp_a1000.link( 'r_arr', src_el.repo, 'arr')
        hp_a1000.link( 'r_prep_arr', src_el.repo, 'prep_arr')
        main_layer.add_elem(hp_a1000, k='hitproc')

        compo_arr_keys = ['nph','sumh','cogrow', 'cogcol', 'var']
        comp_make_elems = [hp_a1000, hp_a1000, hp_a1000, hp_a1000, hp_a1000 ]
        
        # ##########################################################
        # add further processing in this flow here ...
        #

        for i in range(len(compo_arr_keys)):
            compo_compa.hook_up_source(compo_compa.compos[i], comp_make_elems[i], arrkey=compo_arr_keys[i])


        # ##########################################################
        # end of flow
        #

        return cf



def run_1(spt, *p):
    themgr = NewCompo(spt=spt)
    themgr.run()
