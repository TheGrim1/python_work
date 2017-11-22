import ast,os,sys

from o8qq.qqudo1.api import ptiapi
from o8qq.qqudo1.cumulative import cumulative1 as cumu1
from o8qq.compute.compute import average
from cumulative import run_cumulative_imgs

# import from Manfred:
sys.path.append('/data/id13/inhouse3/Manfred/SW/BRICO/new_composite_azim_project/new_compo2')
import optaverage

def run_meta(spt, *p):
    '''runs a meta pti to create a set of composite images. Input parameters are taken from backup folder params = 
\n 
{"indname"             : "/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/DATA/AUTO-TRANSFER/eiger1"}
{"outdname"          : "/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/compos/r1_w3_xzth"}
{"fileprefix"        : "r1_w3_xzth_"}
{"mode"              : "optavg"} # TODO or max 
{"firstinternalno"   : 167}
{"firsteigerno"      : 5}
{"nframes            : 25000"}
{"nruns"             : 27}
{"skip"              : 200}
'''
   
# protocol
    rp          = spt.get_rootnode()
    program     = rp.project.script.program


# take parameters from log:
    logfilename = '/data/id13/inhouse2/AJ/skript/beamlineoperation/split_cumu/backup/r1_w3_xzth.txt'
    params = {}
    f = open(logfilename) 
    for line in f:
        params.update(ast.literal_eval(line.rstrip()))
    f.close

# translating input into pti or templates as required:

    program.infiles.dname    = os.path.realpath(params['indname'])
    mode                     = params['mode']
    filenametpl              = params['fileprefix'] + '_%s_%s_%s_000000.%s'
    infiletpl                = filenametpl %('%s', '%s', 'data','h5')
    outfiletpl               = filenametpl %('%s', '%s', mode,'edf')
    program.outfiles.dname   = os.path.realpath(params['outdname'])
    dname                    = os.path.realpath(params['outdname'])
    firstinternalno          = params['firstinternalno']
    firsteigerno             = params['firsteigerno']
    nruns                    = params['nruns']   
    nframes                  = params['nframes'] 
    program.skip             = params['skip']
  
     # Test whether dirs exists:

    try:
        os.mkdir(dname)
    except:
        try:
            print(dname, "\nexists: ", os.path.exists(dname))
        except:
            print("fundamental error on path:", dname)
            sys.exit()
    

    numtpl = '%1d-%1d'


    for i in range(nruns):
        k            = i + firstinternalno
        l            = i + firsteigerno 

        print "\n===============================\n"
        print "doing %s projection %s of %s on %s"  % (mode,i, nruns, infiletpl  % (k,l))
        print "composite outfile = %s" %outfiletpl % (k,l)
        print "in folder %s" % program.outfiles.dname
        print "\n===============================\n"

        program.use_numbers     = numtpl      % (0,nframes-1)
        program.infiles.fname   = infiletpl   % (k,l)
        program.outfiles.fname  = outfiletpl  % (k,l)

        
  
        optaverage.run_1(spt)
        
        

