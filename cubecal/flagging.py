import numpy as np
from cubecal.tools import logger
log = logger.getLogger("flagging")
import plots

def flag_chisq (st, GD, basename, nddid):
    """
    Flags timeslots and channels based on accumulated statistics

    """
    chi2 = np.ma.masked_array(st.timechan.chi2, st.timechan.chi2==0)
    total = (~chi2.mask).sum()
    chi2n = st.timechan.chi2n
    chi2n = np.ma.masked_array(chi2n, chi2n == 0)

    median = np.ma.median(chi2)
    median_np = np.ma.median(chi2n)
    print>>log, "median chi2 value is {:.3} from {} valid t/f slots".format(median, total)
    print>>log, "median count per slot is {}".format(median_np)

    chi_median_thresh = GD["flags"]["tf-chisq-median"]
    np_median_thresh  = GD["flags"]["tf-np-median"]
    time_density      = GD["flags"]["time-density"]
    chan_density      = GD["flags"]["chan-density"]
    ddid_density      = GD["flags"]["ddid-density"]

    make_plots = GD["out"]["plots"]
    show_plots = GD["out"]["plots-show"]


    if make_plots:
        import pylab
        pylab.figure(figsize=(32, 10))
        pylab.subplot(161)
        pylab.imshow(chi2, vmin=0, vmax=5 * median)
        pylab.title("$\chi^2$")
        pylab.colorbar()
        pylab.subplot(162)
        pylab.imshow(chi2n)
        pylab.title("counts")
        pylab.colorbar()

    flag = (chi2 > chi_median_thresh * median)
    chi2[flag] = np.ma.masked
    nflag = flag.sum()
    print>>log, "{} slots ({:.2%}) flagged on chi2 > {}*median".format(nflag, nflag/float(total), chi_median_thresh)

    if make_plots:
        pylab.subplot(163)
        pylab.imshow(chi2)
        pylab.title("$\chi^2$ median flagged")
        pylab.colorbar()

    flag2 = (chi2n < np_median_thresh * median_np)
    n_new = (flag2&~flag).sum()
    print>>log, "{} more slots ({:.2%}) flagged on counts < {}*median".format(n_new, n_new/float(total), np_median_thresh)
    flag |= flag2

    chi2[flag] = np.ma.masked
    if make_plots:
        pylab.subplot(164)
        pylab.imshow(chi2)
        pylab.title("counts flagged")
        pylab.colorbar()

    nt, nf = flag.shape

    # flag channels with overdense flagging
    freqcount = flag.sum(axis=0)
    freqflags = freqcount > nt * chan_density
    n_new = (freqflags&~(freqcount==nt)).sum()
    print>>log, "{} more channels flagged on density > {}".format(n_new, chan_density)

    # flag timeslots with overdense flagging
    timecount = flag.sum(axis=1)
    timeflags = timecount > nf * time_density
    n_new = (timeflags&~(timecount==nf)).sum()
    print>>log, "{} more timeslots flagged on density > {}".format(n_new, time_density)

    flag = flag | freqflags[np.newaxis,:] | timeflags[:,np.newaxis]
    chi2[flag] = np.ma.masked
    if make_plots:
        pylab.subplot(165)
        pylab.imshow(chi2)
        pylab.title("overdense flagged")
        pylab.colorbar()

    # reshape flag array into time, ddid, channel
    flag3 = flag.reshape((nt, nddid, nf / nddid))

    # flag entire DDIDs with overdense flagging
    maxcount = nt*nf/nddid
    ddidcounts = flag3.sum(axis=(0, 2))
    ddidflags = ddidcounts > maxcount * ddid_density
    n_new = (ddidflags&~(ddidcounts==maxcount)).sum()
    print>>log, "{} more ddids flagged on density > {}".format(n_new, ddid_density)

    flag3 |= ddidflags[np.newaxis, :, np.newaxis]
    chi2[flag] = np.ma.masked
    if make_plots:
        pylab.subplot(166)
        pylab.imshow(chi2)
        pylab.title("overdense DDID")
        pylab.colorbar()
        filename = basename+".chiflag.png"
        pylab.savefig(filename, DPI=plots.DPI)
        print>> log, "saved chi-sq flagging plot to "+filename
        if show_plots:
            pylab.show()

    return flag3