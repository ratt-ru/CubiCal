# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Creates summary plots using the solver stats.
"""
from __future__ import print_function
from builtins import range
import numpy as np
from cubical.tools import logger
log = logger.getLogger("plots")

from cubical.plots import DPI, ZOOM

def make_stats_plots(st, GD, basename, metadata):
    from cubical.plots import make_antenna_xaxis

    def save_figure(name, width, height):
        import pylab
        pylab.gcf().set_size_inches(min(width, 10000 / DPI), min(height, 10000 / DPI))
        filename = "{}.{}.png".format(basename, name)
        pylab.savefig(filename, dpi=DPI)
        print("saved plot " + filename, file=log)
        if GD["out"]["plots"] == "show":
            pylab.show()
        pylab.figure()

    import pylab

    # plot noise per time/channel
    pylab.subplot(121)
    pylab.title("Noise on input data")
    pylab.xlabel("channel")
    pylab.ylabel("timeslot")
    st.timechan.dv2[st.timechan.dv2 == 0] = np.inf
    pylab.imshow(np.sqrt(st.timechan.dv2), aspect='auto')
    pylab.colorbar()
    pylab.subplot(122)
    pylab.title("Noise on residuals")
    pylab.xlabel("channel")
    pylab.ylabel("timeslot")
    st.timechan.dr2[st.timechan.dr2 == 0] = np.inf
    pylab.imshow(np.sqrt(st.timechan.dr2), aspect='auto')
    pylab.colorbar()
    nt, nf = st.timechan.dv2.shape
    save_figure("noise.tf", max(nf * ZOOM / DPI * 2.5, 10), max(nt * ZOOM / DPI * 1.1,8))

    # plot chi-sq per time/channel
    pylab.subplot(121)
    pylab.title("Chi-sq on input data")
    pylab.xlabel("channel")
    pylab.ylabel("timeslot")
    st.timechan.initchi2[st.timechan.initchi2 == 0] = np.nan
    st.timechan.chi2[st.timechan.chi2 == 0] = np.nan
    pylab.imshow(st.timechan.initchi2, aspect='auto')
    pylab.colorbar()
    pylab.subplot(122)
    pylab.title("Chi-sq on residuals")
    pylab.xlabel("channel")
    pylab.ylabel("timeslot")
    pylab.imshow(st.timechan.chi2, aspect='auto')
    pylab.colorbar()
    nt, nf = st.timechan.chi2.shape
    save_figure("chi2.tf", max(nf * ZOOM / DPI * 2.5, 10), max(nt * ZOOM / DPI * 1.1, 8))

    # plot noise per antenna/channel
    pylab.subplot(121)
    noise = np.sqrt(st.chanant.dr2)
    noise[noise == 0] = np.inf
    nf, nant = noise.shape
    for ant in range(nant):
        pylab.plot(noise[:, ant], 'o-')
    for x in pylab.xticks()[0]:
        pylab.axvline(x, c="grey", lw=.5, ls=':', zorder=999)
    pylab.title("Noise (colour: antenna)")
    pylab.xlabel("channel")
    pylab.ylabel("noise")
    pylab.subplot(122)
    make_antenna_xaxis(metadata.antenna_name)
    for chan in range(nf):
        pylab.plot(noise[chan, :], 'o-')
    pylab.title("Noise (colour: channel)")
    pylab.ylabel("noise")
    save_figure("noise.antchan", 10, 5)

    # plot chi2 per antenna/channel
    pylab.subplot(121)
    chi2 = st.chanant.chi2
    chi2[chi2 == 0] = np.inf
    nf, nant = chi2.shape
    for ant in range(nant):
        pylab.plot(chi2[:, ant], 'o-')
    for x in pylab.xticks()[0]:
        pylab.axvline(x, c="grey", lw=.5, ls=':', zorder=999)
    pylab.title("Chi-sq (colour: antenna)")
    pylab.xlabel("channel")
    pylab.ylabel("$\chi^2$")
    pylab.subplot(122)
    make_antenna_xaxis(metadata.antenna_name)
    for chan in range(nf):
        pylab.plot(chi2[chan, :], 'o-')
    pylab.title("Chi-sq (colour: channel)")
    pylab.ylabel("$\chi^2$")
    save_figure("chi2.antchan", 10, 5)

