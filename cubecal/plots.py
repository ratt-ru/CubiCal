import matplotlib
matplotlib.use("Agg")

import numpy as np
from cubecal.tools import logger
log = logger.getLogger("plots")

# these control the layout of saved plots
DPI = 150.  # resolution: determines size of text relative to plots
ZOOM = 10  # image zoom. Roughly determines size of "pixels" when rendering images


def make_summary_plots(st, GD, basename):
    """

    Args:
        st:         statistics.SolverStats object
        GD:         dict of global options
    Returns:

    """
    import pylab

    def save_figure(name, width, height):
        pylab.gcf().set_size_inches(min(width, 10000 / DPI), min(height, 10000 / DPI))
        filename = "{}.{}.png".format(basename, name)
        pylab.savefig(filename, dpi=DPI)
        print>> log, "saved plot " + filename
        if GD["out"]["plots-show"]:
            pylab.show()
        pylab.figure()

    # plot noise per time/channel
    pylab.subplot(121)
    pylab.title("Noise on input data")
    pylab.xlabel("channel")
    pylab.ylabel("timeslot")
    st.timechan.dv2[st.timechan.dv2 == 0] = np.inf
    pylab.imshow(np.sqrt(st.timechan.dv2))
    pylab.colorbar()
    pylab.subplot(122)
    pylab.title("Noise on residuals")
    pylab.xlabel("channel")
    pylab.ylabel("timeslot")
    st.timechan.dr2[st.timechan.dr2 == 0] = np.inf
    pylab.imshow(np.sqrt(st.timechan.dr2))
    pylab.colorbar()
    nt, nf = st.timechan.dv2.shape
    save_figure("noise.tf", nf * ZOOM / DPI * 2.5, nt * ZOOM / DPI * 1.1)

    # plot chi-sq per time/channel
    pylab.subplot(121)
    pylab.title("Chi-sq on input data")
    pylab.xlabel("channel")
    pylab.ylabel("timeslot")
    st.timechan.initchi2[st.timechan.initchi2 == 0] = np.nan
    st.timechan.chi2[st.timechan.chi2 == 0] = np.nan
    pylab.imshow(st.timechan.initchi2)
    pylab.colorbar()
    pylab.subplot(122)
    pylab.title("Chi-sq on residuals")
    pylab.xlabel("channel")
    pylab.ylabel("timeslot")
    pylab.imshow(st.timechan.chi2)
    pylab.colorbar()
    nt, nf = st.timechan.chi2.shape
    save_figure("chi2.tf", nf * ZOOM / DPI * 2.5, nt * ZOOM / DPI * 1.1)

    # plot noise per antenna/channel
    pylab.subplot(121)
    noise = np.sqrt(st.chanant.dr2)
    noise[noise == 0] = np.inf
    nf, nant = noise.shape
    for ant in xrange(nant):
        pylab.plot(noise[:, ant], 'o-')
    pylab.title("Noise by antenna")
    pylab.xlabel("channel")
    pylab.ylabel("noise")
    pylab.subplot(122)
    for chan in xrange(nf):
        pylab.plot(noise[chan, :], 'o-')
    pylab.title("Noise by channel")
    pylab.xlabel("antenna")
    pylab.ylabel("noise")
    save_figure("noise.antchan", 10, 5)

    # plot chi2 per antenna/channel
    pylab.subplot(121)
    chi2 = st.chanant.chi2
    chi2[chi2 == 0] = np.inf
    nf, nant = chi2.shape
    for ant in xrange(nant):
        pylab.plot(chi2[:, ant], 'o-')
    pylab.title("Chi-sq by antenna")
    pylab.xlabel("channel")
    pylab.ylabel("$\chi^2$")
    pylab.subplot(122)
    for chan in xrange(nf):
        pylab.plot(chi2[chan, :], 'o-')
    pylab.title("Chi-sq by channel")
    pylab.xlabel("antenna")
    pylab.ylabel("$\chi^2$")
    save_figure("chi2.antchan", 10, 5)

