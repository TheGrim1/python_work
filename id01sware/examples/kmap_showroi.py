#!/usr/bin/env python
import os
from silx.io import specfile
import pylab as pl
import collections

#pl.figure()

plotcol = [
            "mpx4int",
#            "mpx4ro1",
#            "mpx4ro2",
#            "roi1",
            "roi2",
            "roi3",
            "roi4",
            "roi5",
#            "roi6",
          ]
numrows = 1


Dir = "/data/visitor/ma3319/id01/BFO/KMAP_0401"
#plotfunc = pl.log
plotfunc = lambda x: x**.1

#flist = "/data/id01/inhouse/tobias/2017mar/SiC/ihhc3107/id01/SiCgriffith_bridge18phi90/KMAP_0349/spec/default_fast_00001.spec"


#flist = [open("/data/id01/inhouse/tobias/2017mar/SiC/ihhc3107/id01/SiCgriffith_bridge18phi90/KMAP_0346/spec/default_fast_00001.spec")]
#scansize = 40
#arraysizex = 35
#arraysizey = 32


#flist = [open("/data/id01/inhouse/tobias/2017mar/SiC/ihhc3107/id01/SiCgriffith_bridge18/KMAP_0345/spec/default_fast_00001.spec")]
#scansize = 40
#arraysizex = 30
#arraysizey = 80


#flist = [open("/data/visitor/in906/id01/G14_D15_004/KMAP_0336/spec/default_fast_00001.spec")]
#scansize = 20
#arraysizex = 60
#arraysizey = 60


#flist = [open("/data/visitor/in906/id01/G14_D15_531/KMAP_0330/spec/default_fast_00001.spec")]
#scansize = 40
#arraysizex = 120
#arraysizey = 120

#flist = [open("/data/visitor/in906/id01/G5_D15_531_phi0/KMAP_0327/spec/default_fast_00001.spec")]
#scansize = 33
#arraysizex = 50
#arraysizey = 48


#flist = [open("/data/visitor/in906/id01/G5_D15_531_phi90/KMAP_0325/spec/default_fast_00001.spec"),
#         open("/data/visitor/in906/id01/G5_D15_531_phi90/KMAP_0326/spec/default_fast_00001.spec")]
#scansize = 40
#arraysizex = 48
#arraysizey = 58


#flist = [open ("/users/opid01/visitor/in906/id01/G14_D15_531_phi90/KMAP_0322/spec/default_fast_00001.spec"),
#         open ("/users/opid01/visitor/in906/id01/G14_D15_531_phi90/KMAP_0323/spec/default_fast_00001.spec")]; # stitching
#scansize = 40
#arraysizex = 40
#arraysizey = 56

#flist = open ("/users/opid01/visitor/in906/id01/G14_D15_531_disc2/KMAP_0320/spec/default_fast_00001.spec"); scansize = 40
#flist = open ("/users/opid01/visitor/in906/id01/G14_D15_531/KMAP_0319/spec/default_fast_00001.spec")
#flist = [open ("/users/opid01/visitor/in906/id01/G14_D15_531/KMAP_03%02i/spec/default_fast_00001.spec"%i) for i in (19,17,18)]; scansize = 31
#arraysizex = 60
#arraysizey = 40

#flist = open ("/users/opid01/visitor/in906/id01/G14_D15_004/KMAP_0316/spec/default_fast_00001.spec")
#flist = open ("/users/opid01/visitor/in906/id01/G14_D15_004/KMAP_0316/spec/default_fast_00001.spec")

#flag = False

if not isinstance(Dir, list):
    Dir = [Dir]

flist = [os.path.join(D, "spec/default_fast_00001.spec") for D in Dir]

if not isinstance(flist, list):
    flist = [flist]

DATA = collections.defaultdict(list)
MOT = collections.defaultdict(list)


for f in flist:
    sf = specfile.SpecFile(f)
    cols = sf.labels(0)
    for sc in sf:
        for col in cols:
            DATA[col].append(sc.data_column_by_name(col))
        for mot in sc.motor_names:
            MOT[mot].append(sc.motor_position_by_name(mot))

h = sc.scan_header_dict["S"].split()
assert h[1] == "pscando"
arraysizey = int(h[5])
arraysizex = int(h[9])
scansize = len(DATA[col])


skip = 0
for d in DATA:
    print d
    if len(DATA[d])>1:
        l1 = len(DATA[d][-1])
        l2 = len(DATA[d][-2])
        if l1 != l2:
            DATA[d].pop(-1)
            skip = 1
    DATA[d] = pl.vstack(DATA[d])
    DATA[d] = DATA[d].reshape(scansize-skip,arraysizex, arraysizey)
    if d in plotcol:
        DATA[d] = plotfunc(DATA[d])
    print DATA[d].shape,

scansize = scansize - skip
#print list(DATA)

for col in plotcol:
    if col not in DATA:
        raise ValueError("Column not in data: %s"%col)
    x = DATA["adcY"]
    y = DATA["adcX"]
    d = DATA[col]
    Imax = d.max()
    Imin = d.min()
    print("Column: %s, Imax: %g"%(col, Imax))
    #nrow = int(pl.ceil(pl.sqrt(scansize)*9./16))
    numcols = int(pl.ceil(float(scansize)/numrows))
    fig, ax = pl.subplots(numrows, numcols, sharex=True, sharey=True, figsize=(20,9), num=col, squeeze=False)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    ax = ax.ravel()
    for i in range(scansize):
        #ax[i].imshow(d[i].T, cmap='jet', interpolation='none', origin='lower', vmax=Imax)
        ax[i].pcolor(x[i], y[i], d[i], cmap='jet', vmin=Imin, vmax=Imax)
        ax[i].set_xlim(x[i].min(),x[i].max())
        ax[i].set_ylim(y[i].min(),y[i].max())
        ax[i].set_aspect('equal', adjustable="box-forced")
        ax[i].set_title("Eta = %.3f"%MOT["eta"][i])
    #ax[i].axis("equal")

    #fig.suptitle(col)
    #fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95)
    fig.savefig(os.path.join(Dir[0].rstrip("/")+"_%s.png"%col))
    fig.tight_layout()


pl.show()

