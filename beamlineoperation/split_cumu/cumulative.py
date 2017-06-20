from o8qq.qqudo1.api import ptiapi
from o8qq.qqudo1.cumulative import cumulative1 as cumu1

def run_cumulative_imgs(spt, *p):

    #protocol
    rp          = spt.get_rootnode()
    program     = rp.project.script.program

    infiles   = program.infiles
    outfiles  = program.outfiles
    outindex  = program.outindex
    modetype = program.mode.modetype
    cumfuncs  = cumu1.CumulativeChannelOperatons()

    # setup
    inch = ptiapi.make_img_inch(infiles)
    ouch = ptiapi.make_img_ouch(outfiles)
    idx = outindex
    cumulative_func = cumfuncs.get_cumulfunc(modetype)
    brr = cumulative_func(inch)
    ouch[idx] = brr
