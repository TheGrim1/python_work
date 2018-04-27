from __future__ import print_function
from __future__ import absolute_import
import sys,os, ast
# for Manfreds functions, doesn seem to work... have to setd20 before runnning
# sys.path.append('/users/opid13/.muc_local/.MCB_DVP/PYTHON_LOCAL_INSTALL_20')
# sys.path.append('/users/opid13/.muc_local/.MCB_DVP/PYTHON_LOCAL_INSTALL_20/lib/python')

from o8qq.qqudo1.api import ptiapi
from o8qq.qqudo1.cumulative import cumulative1 as cumu1


# local import
from .dvp_new_compo2_aj import run_1

def run_meta(spt, *p):
    '''runs a meta pti to create a set of composite images. Input parameters are taken from local backup folder and logfile\n params = 
\n 
"indname"           : "/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/DATA/AUTO-TRANSFER/eiger1"}
{"outdname"          : "/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/compos/r1_w3_xzth"}
{"fileprefix"        : "r1_w3_xzth_"}
{"troi"              : ((1104, 42), (558, 633))}
{"outfileidentifier" : "ROI2"}
{"meshshape"         :(166, 61)}
{"firstinternalno"   : 167}
{"firsteigerno"      : 5}
{"ncompos"           : 26}
'''

# take parameters from log:
    logfilename = '/data/id13/inhouse2/AJ/skript/compos/compobatch/backup/ihhc2997/r1_w3_xzth_ROI2_.txt'
    params = {}
    f = open(logfilename) 
    for line in f:
        params.update(ast.literal_eval(line.rstrip()))
    f.close

# set protocol
    rp          = spt.get_rootnode()
    program     = rp.project.script.program


# translating input into pti or templates as required:

    program.infiles.dname    = params['indname']
    filenametpl              = params['fileprefix'] + '_%s_%s_%s_000000.%s'
    program.infiles.troi     = params['troi']
    program.scan.meshshape   = params['meshshape']
    infiletpl                = filenametpl %('%s', '%s', 'data','h5')
    outfiletpl               = filenametpl %('%s', '%s', params['outfileidentifier'],'edf')
    program.outfiles.dname   = params['outdname']
    dname                    = params['outdname']
    firstinternalno          = params['firstinternalno']
    firsteigerno             = params['firsteigerno']
    ncompos                  = params['ncompos']



    # Test whether dir exists:

    try:
        os.mkdir(dname)
    except:
        try:
            print(dname, ":", path.exists(dname))
        except:
            print("fundamental error:", dname)


    for i in range(ncompos):
        k = i + firstinternalno
        l = i + firsteigerno 

        print("\n===============================\n")
        print("doing composite %s of %s on %s"  % (i, ncompos, infiletpl  % (k,l)))
        print("composite outfile = %s" %outfiletpl % (k,l))
        print("in folder %s" % program.outfiles.dname)
        print("\n===============================\n")

        program.infiles.fname  = infiletpl   % (k,l)
        program.outfiles.fname = outfiletpl  % (k,l)
#        program.outfiles.fname = "batchtest_00000.edf"

        run_1(spt)
        
        
