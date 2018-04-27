import pylab as pl
from id01lib.process import rebin



N = 50001

x1 = pl.rand(N) - 0.5
x2 = (pl.rand(N) - 0.5) * 2

y = pl.exp(-(x1**2 + x2**2))


_x1, _y1 = rebin.rebin1d(x1, y, bins=500) # projection

pl.plot(_x1, _y1)
pl.show()


_x1, _x2, _y = rebin.rebin2d(x1, x2, y, bins=100, edges=True) # rebinned 

#pl.pcolormesh(_x1, _x2, _y, vmin=pl.nanmin(y), vmax=pl.nanmax(y))
pl.imshow(_y, vmin=pl.nanmin(y), vmax=pl.nanmax(y),
          extent=[_x2[0], _x2[-1], _x1[0], _x1[-1]],
          interpolation="nearest")
pl.show()
