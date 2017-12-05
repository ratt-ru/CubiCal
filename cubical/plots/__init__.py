import warnings
import numpy as np

# these control the layout of saved plots
DPI = 150.  # resolution: determines size of text relative to plots
ZOOM = 10  # image zoom. Roughly determines size of "pixels" when rendering images


import ifrgains
import stats


def make_summary_plots(st, ms, GD, basename):
    """

    Args:
        st:         statistics.SolverStats object
        GD:         dict of global options
        basename:   base filename for plots
    Returns:

    """
    from cubical import solver

    if solver.ifrgain_machine and solver.ifrgain_machine.is_computing():
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.ComplexWarning)
            ifrgains.make_ifrgain_plots(solver.ifrgain_machine.reload(), ms, GD, basename)

    stats.make_stats_plots(st, GD, basename)


