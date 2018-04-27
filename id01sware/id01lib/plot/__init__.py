"""
    Some functions on top of `matplotlib`, most notably a
    draggable colorbar which allows rescaling the color range interactively.
"""

import pylab as pl

def colorwheel(text_col='black', pixels=100,
                                 fs=16,
                                 ax=None,
                                 textleft='$\\leftarrow$',
                                 textright='$\\rightarrow$'):
    """
    PyNX-like color wheel for azimuthal hue information in hsv colormap.

    Args:
        pixels: number of pixels in the wheel diameter
        text_col: colour of text
        fs: fontsize in points
        ax: a matplotlib AxesSubplot instance

    Returns:
        Nothing. Displays a colorwheel in the current or supplied figure.
    """
    x = pl.linspace(-1, 1, pixels)
    y = pl.linspace(-1, 1, pixels)[:, pl.newaxis]
    r = pl.sqrt(x**2 + y**2)
    phi = pl.arctan2(y, x)  # range [-pi, pi].
    hue = phi/(2*pl.pi) + 0.5
    saturation = r
    value = pl.ones_like(r)
    hsv = pl.array([hue, saturation, value]).transpose(1,2,0)
    rgb = pl.matplotlib.colors.hsv_to_rgb(hsv)
    rgba = pl.dstack((rgb, r<1))
    if ax is None:
        ax = pl.gca()
    ax.set_axis_off()
    bg = pl.ones_like(rgba)
    bg[:,:,3] *= (r<1)
    ax.imshow(bg, aspect='equal')
    ax.imshow(rgba, aspect='equal')
    text_kw = dict(fontsize=fs,
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   color=text_col)
    if textright:
        ax.text(1.05, 0.55, textright, **text_kw)
    if textleft:
        ax.text(-.15, 0.55, textleft, **text_kw)
    return ax

