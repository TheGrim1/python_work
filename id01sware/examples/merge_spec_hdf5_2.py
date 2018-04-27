### See also ../tests/test_id01h5.py !
from id01lib import id01h5
import glob


sample = 'E17089' # name of top level h5 entry
h5file = 'hc3211.h5' # output file
imgdir = None#'/data/visitor/hc3211/id01/mpx/e17089/'

#speclist = glob.glob(specdir+'e16014.spec')
specfile = '/data/visitor/hc3211/id01/spec/e17089.spec' # source file
scanno = ("37.1", "327.1") # None for all, must be tuple i.e. ("1.1",) for single scanno

# code

with id01h5.ID01File(h5file,'a') as h5f: # the safe way
    s = h5f.addSample(sample)
    s.importSpecFile(specfile,
                     numbers=scanno,
                     verbose=True,
                     imgroot=imgdir,
                     overwrite=False, # skip if scan exists
                     compr_lvl=6)



