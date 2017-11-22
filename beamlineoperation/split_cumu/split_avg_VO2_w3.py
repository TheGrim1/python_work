from __future__ import print_function
from builtins import range
from o8qq.qqudo1.api import ptiapi
from o8qq.qqudo1.cumulative import cumulative1 as cumu1
from cumulative import run_cumulative_imgs

def run_meta(spt, *p):

    #protocol
    rp          = spt.get_rootnode()
    program     = rp.project.script.program

    infiles   = program.infiles
    outfiles  = program.outfiles
    outindex  = program.outindex
    modetype = program.mode.modetype
    
    # Test whether dir exists:
    dname = program.infiles.dname
    try:
        os.mkdir(dname)
    except:
        try:
            print(dname, ":", path.exists(dname))
        except:
            print("fundamental error:", dname)
    k = 0
    l = 0
    numtpl = '%1d-%1d'
    # Average over this number of images:
    noimages    = 21
    # This many times:
    noaverages  = 7000
    for i in range(noaverages):
        k = i * noimages
        l = (i + 1) * noimages - 1
        infiles.numbers = numtpl % (k,l)
        print("new infile template:", infiles.numbers)
        program.outindex = i
        run_cumulative_imgs(spt)
        
        
