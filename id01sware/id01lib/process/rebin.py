import numpy as np

def rebin1d(x, y, weights=None, bins=None, xmin=None, xmax=None,
          discard_empty=False, edges=False, return_all=False):
    """

        Function that averages unsorted data via histogramming. The data
        does not have to be present for repeatingly the same values of the
        independent variable and does not have to be ordered. Therefore, a
        rebinning to a new equi-distant x-axis takes place. No subpixel 
        splitting is done.

        Typical usage:


            #q_random -- array of randomly spaced and unsorted q values
            #I -- Intensity at q_random
            #monitor -- monitor readings at q_random

            q_equidist, I_normalized = rebin1d(q_random, I, monitor, bins=400)


        If there are empty bins, they will carry the value `np.nan`.
        These can be discarded using the `discard_empty` key word argument.

        If `edges` is True: Return edges of the new bins instead of the
                            centers (results in +1 increased length)
    """
    #x = np.hstack(x)
    if np.ndim(x) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    ind = (x>=xmin) * (x<=xmax)
    x = x[ind]
    y = y[ind]
    if bins is None:
        bins = (x.max()-x.min())/np.diff(np.sort(x)).max()
        bins = int(np.floor(bins))
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = np.ravel(weights)[ind]
    y, newx = np.histogram(x, bins=bins, weights=y, range=(xmin, xmax))
    num, newx = np.histogram(x, bins=bins, weights=weights, range=(xmin, xmax))
    dx = newx[1] - newx[0]
    x = newx[1:] - dx/2.
    y /= num
    ind = num > 0
    if not ind.all() and discard_empty:
        y = y[ind]
        x = x[ind]
    if edges:
        x = np.append(x[0] - dx/2., x + dx/2.)
    if return_all:
        return dict(x=x,
                    y=y,
                    num=num,
                    bins=bins)
    return x, y





def rebin2d(x1, x2, y, weights=None, bins=50,
            x1min=None, x1max=None, x2min=None, x2max=None,
            edges=False):
    """
        (Similar to Gridder2d)

        Function that averages unsorted data via histogramming.
        The input data can be given on an irregular grid and the order is
        not relevant.
        All input arrays should have the same .size and will be flattened.
        A rebinning to a new equi-distant x-axis takes place. No subpixel 
        splitting is done.

        Typical usage:


            #q_x, q_y -- arrays of randomly spaced and unsorted q values
            #I -- Intensity at (q_x, q_y)
            #monitor -- monitor readings at (q_x, q_y)

            q_x_regular,
            q_y_regular,
            I_normalized = rebin1d(q_x, q_y, I, monitor, bins=100)


        If there are empty bins, they will carry the value `np.nan`.

        If `edges` is True: Return edges of the new bins instead of the
                            centers (results in +1 increased length)

    """
    x1 = np.ravel(x1)
    x2 = np.ravel(x2)
    y = np.ravel(y)

    if x1min is None:
        x1min = x1.min()
    if x1max is None:
        x1max = x1.max()
    if x2min is None:
        x2min = x2.min()
    if x2max is None:
        x2max = x2.max()

    ind = (x1>=x1min) * (x1<=x1max) \
         *(x2>=x2min) * (x2<=x2max)
    if not ind.all():
        x1 = x1[ind]
        x2 = x2[ind]
        y = y[ind]

    if isinstance(bins, (long, int)):
        bins = [bins]*2
    if len(bins) != 2:
        raise ValueError("Invalid input for `bins`")
    if weights is None:
        weights = np.ones_like(y)
    else:
        weights = np.ravel(weights)[ind]

    rng = ([x1min, x1max], [x2min, x2max])

    num,  _,  _ = np.histogram2d(x1, x2, weights=weights, bins=bins, range=rng)
    y,   x1, x2 = np.histogram2d(x1, x2, weights=y      , bins=bins, range=rng)

    y /= num
    if edges:
        return x1, x2, y

    dx1 = x1[1] - x1[0]
    dx2 = x2[1] - x2[0]

    x1 = x1[1:] - dx1/2.
    x2 = x2[1:] - dx2/2.
    return x1, x2, y





