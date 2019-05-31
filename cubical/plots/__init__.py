import numpy as np
from builtins import range

# these control the layout of saved plots
DPI = 150.  # resolution: determines size of text relative to plots
ZOOM = 10  # image zoom. Roughly determines size of "pixels" when rendering images

def make_antenna_xaxis(antnames):
    import pylab
    """Helper function. Sets up a plot X axis that shows antenna numbers nicely"""
    nant = len(antnames)
    # make a tick at every antenna, but label only the originally labelled ones
    locs = np.arange(nant)
    labels = [''] * nant
    pylab.xlim(-1, nant)
    for x, lbl in zip(*pylab.xticks()):
        x = int(x)
        if x >= 0 and x < nant:
            labels[x] = antnames[x]
    pylab.xticks(locs, labels)
    # draw vertical guides for antennas -- thicker line every 5 antennas
    for x in range(0, nant):
        pylab.axvline(x, c="grey", lw=0.5 if x%5 else 1, ls=':' if x%5 else '-')
    pylab.xlabel("antenna")


def make_summary_plots(st, ms, GD, basename):
    """

    Args:
        st:         statistics.SolverStats object
        GD:         dict of global options
        basename:   base filename for plots
    Returns:

    """
    stats.make_stats_plots(st, GD, basename, ms.metadata)

from . import ifrgains, stats

