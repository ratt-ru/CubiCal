import math,cmath
import numpy as np
from cubical.tools import logger
log = logger.getLogger("plots")

from cubical.plots import DPI, ZOOM

def _cmp_antenna (sa, sb):
    """Helper function to sort antenna names. Try numeric compare first, fall back to text compare if failed""";
    try:
        return cmp(int(sa), int(sb));
    except:
        return cmp(sa, sb);


def _normifrgain(rr):
    """Converts gains to mean offsets with std"""
    if type(rr) in (float, complex):
        return abs(rr), 0
    else:
        offset = abs(rr[rr != 1])
        if offset.count():
            return float(offset.mean()), float(offset.std())
        else:
            return 1, 0


def _complexifrgain(rr):
    """Converts gains to complex offsets with std"""
    if type(rr) in (float, complex):
        return rr, 0
    else:
        vals = rr[rr != 1]
        if vals.count():
            offset = float(abs(vals).mean())
            mean = vals.mean().ravel()[0]
            mean = cmath.rect(offset, cmath.phase(mean))
            return mean, abs(vals - mean).std()
        else:
            return 1,0


def _is_unity(rr, ll):
    return (not rr.count() or (rr==1).all()) and \
           (not ll.count() or (ll==1).all())


#def make_ifrgain_plots(filename="$STEFCAL_DIFFGAIN_SAVE", prefix="IG", feed="$IFRGAIN_PLOT_FEED", msname="$MS"):
def make_ifrgain_plots(ig, ms, GD, basename):
    """Makes a set of ifrgain plots from the specified saved file."""
    print>>log(0),"generating plots for suggested baseline-based corrections (BBCs)"

    import pylab
    def save_figure(name, width, height):
        import pylab
        pylab.gcf().set_size_inches(min(width, 10000 / DPI), min(height, 10000 / DPI))
        filename = "{}.{}.png".format(basename, name)
        pylab.savefig(filename, dpi=DPI)
        print>> log, "saved plot " + filename
        if GD["out"]["plots"] == "show":
            pylab.show()
        pylab.figure()

    # load baseline info, if MS is available
    antpos = zip(ms.antnames, ms.antpos)
    # make dictionary of IFR name: baseline length
    baseline = {
        "%s-%s"%(p,q): math.sqrt(((ppos - qpos) ** 2).sum()) for p, ppos in antpos for q, qpos in antpos
    }
    feeds = [ (x+y).upper() for x in ms.feeds for y in ms.feeds ]

    def plot_xy(content, title):
        """Plots x vs y"""
        pylab.errorbar(
            [x for l, (x, xe), (y, ye) in content], [y for l, (x, xe), (y, ye) in content],
            [ye for l, (x, xe), (y, ye) in content], [xe for l, (x, xe), (y, ye) in content],
            fmt="none", ecolor="lightgrey"
        )
        # plot labels
        for label, (x, xe), (y, ye) in content:
            pylab.text(x, y, label, horizontalalignment='center', verticalalignment='center', size=8)
        pylab.title(title)

    def plot_baseline(content, baseline, title, feeds):
        """Plots offset versus baseline"""
        bl = []
        xx = []
        xxe = []
        lab = []
        col = []
        for l, (x, xe), (y, ye) in content:
            b = baseline.get(l, None)
            if b is None:
                print>>log(0, "red"),"baseline '{}' not found in MS ANTENNA table".format(l)
            else:
                lab += ["%s:%s" % (l, feeds[0]), "%s:%s" % (l, feeds[1])]
                col += ["blue", "red"]
                xx += [x, y]
                xxe += [xe, ye]
                bl += [b, b]
        pylab.axhline(1, color='lightgrey')
        pylab.errorbar(bl, xx, yerr=xxe, fmt="none", ecolor="lightgrey")
        # plot labels
        for l, x, y, c in zip(lab, bl, xx, col):
            pylab.text(x, y, l, horizontalalignment='center', verticalalignment='center', size=8, color=c)
        pylab.xlabel("Baseline, m.")
        pylab.title(title)

    def plot_hist(content, title):
        """Plots histogram"""
        values = [x for l, (x, xe), (y, ye) in content] + [y for l, (x, xe), (y, ye) in content]
        x0, x1 = min(values), max(values)
        if (x1-x0) > 1e-5:
            hist = pylab.hist(values)
            pylab.xlim(x0, x1)
        pylab.title(title)

    def plot_complex(content, title):
        """Plots x vs y"""
        # plot errors bars, if available
        pylab.axhline(0, color='lightgrey')
        pylab.axvline(1, color='lightgrey')
        pylab.errorbar(
            [x.real for l1, l2, (x, xe), (y, ye) in content], [x.imag for l1, l2, (x, xe), (y, ye) in content],
            [xe for l1, l2, (x, xe), (y, ye) in content], [xe for l1, l2, (x, xe), (y, ye) in content],
            fmt="none", ecolor="lightgrey"
        )
        pylab.errorbar(
            [y.real for l1, l2, (x, xe), (y, ye) in content], [y.imag for l1, l2, (x, xe), (y, ye) in content],
            [ye for l1, l2, (x, xe), (y, ye) in content], [ye for l1, l2, (x, xe), (y, ye) in content],
            fmt="none", ecolor="lightgrey"
        )
        # max plot amplitude -- max point plus 1/4th of the error bar
        maxa = max([max(abs(x), abs(y)) for l1, l2, (x, xe), (y, ye) in content])
        # plotlim = max([ abs(np.array([
        #                  getattr(v,attr)+sign*e/4 for v,e in (x,xe),(y,ye) for attr in 'real','imag' for sign in 1,-1
        #                ])).max()
        #   for l1,l2,(x,xe),(y,ye) in content ])
        minre, maxre, minim, maxim = 2, -2, 2, -2
        for l1, l2, (x, xe), (y, ye) in content:
            offs = np.array([getattr(v, attr) + sign * e / 4 for v, e in (x, xe), (y, ye)
                                for attr in 'real', 'imag' for sign in 1, -1])
            minre, maxre = min(x.real - xe / 4, y.real - ye / 4, minre), max(x.real + xe / 4, y.real + ye / 4, maxre)
            minim, maxim = min(x.imag - xe / 4, y.imag - ye / 4, minim), max(x.imag + xe / 4, y.imag + ye / 4, maxim)
        # plot labels
        for l1, l2, (x, xe), (y, ye) in content:
            pylab.text(x.real, x.imag, l1, horizontalalignment='center', verticalalignment='center', color='blue',
                       size=8)
            pylab.text(y.real, y.imag, l2, horizontalalignment='center', verticalalignment='center', color='red',
                       size=8)
        # pylab.xlim(-plotlim,plotlim)
        # pylab.ylim(-plotlim,plotlim)
        pylab.xlim(minre, maxre)
        pylab.ylim(minim, maxim)
        pylab.title(title + " (max %.5g)" % maxa)

    def plot_ants(content, title):
        """Plots x vs y"""
        # plot labels
        for i, (p, gainlist) in enumerate(content):
            for label, color, (value, std) in gainlist:
                if value:
                    pylab.plot(i, value, 'w.')
                    pylab.text(i, value, label, horizontalalignment='center', verticalalignment='center', size=8,
                               color=color)
        pylab.xlim(-1, len(content))
        pylab.xticks(range(len(content)), [p for p, gainlist in content])
        pylab.title(title)

    nfreq, nant, _, _, _ = ig.shape
    # collect a list of valid RR/LL and RL/LR pairs (i.e. ones not all unity)

    # plot size and layout
    width, height = 8, 6  # size per sublot

    # loop over diag and offdiag elements
    components = [ (0, 0, 1, 1, "diag") ]
    if GD["bbc"]["compute-2x2"]:
        components.append((0, 1, 1, 0, "offdiag"))

    for i1,j1,i2,j2,label in components:
        # this is a 3x2 plot
        NR, NC = 3, 2
        # collect a list of valid RR/LL and RL/LR pairs (i.e. ones not all unity)
        valid_igs = []
        ifr_pairs = {}
        for p in xrange(nant):
            for q in xrange(p, nant):
                ifrname = "%s-%s" % (ms.antnames[p], ms.antnames[q])
                rr = ig[:, p, q, i1, j1]
                ll = ig[:, p, q, i2, j2]
                if not _is_unity(rr, ll):
                    valid_igs.append((ifrname, rr, ll))
                    ifr_pairs[ifrname] = ms.antnames[p], ms.antnames[q]
        if not valid_igs:
            continue
        FEEDS = (ms.feeds[i1]+ms.feeds[j1]).upper(), (ms.feeds[i2]+ms.feeds[j2]).upper()

        norm_igs = [ (pq, _normifrgain(rr),_normifrgain(ll)) for pq, rr, ll in valid_igs ]
        complex_igs = [("{}:{}".format(pq, FEEDS[0]), "{}:{}".format(pq, FEEDS[1]),
                        _complexifrgain(rr), _complexifrgain(ll)) for pq, rr, ll in valid_igs]
        pylab.subplot(NR, NC, 3)
        plot_xy(norm_igs, "IFR gain amplitude (%s vs. %s)" % FEEDS)
        pylab.subplot(NR, NC, 4)
        plot_hist(norm_igs, "IFR gain histogram for %s and %s" % FEEDS)
        pylab.subplot(NR, NC, 1)
        plot_complex(complex_igs, "IFR complex %s %s gains" % FEEDS)
        igpa = {}
        igpa0 = {}
        igpa0_means = []
        for pq, rr, ll in valid_igs:
            p,q = ifr_pairs[pq]
            rr0 = np.ma.masked_array(abs(rr - 1).data, rr.mask)
            ll0 = np.ma.masked_array(abs(ll - 1).data, ll.mask)
            rr0.mask |= (rr0 == 0)
            ll0.mask |= (ll0 == 0)
            if not rr0.mask.all():
                igpa0_means += [rr0.mean()]
            if not ll0.mask.all():
                igpa0_means += [ll0.mean()]
            if not rr0.mask.all() or not ll0.mask.all():
                igpa0.setdefault(p, {})[q] = rr0, ll0
                igpa0.setdefault(q, {})[p] = rr0, ll0
            rr, ll = _normifrgain(rr), _normifrgain(ll)
            igpa.setdefault(p, []).append(("%s:%s" % (q, FEEDS[0]), 'blue', rr))
            igpa.setdefault(q, []).append(("%s:%s" % (p, FEEDS[0]), 'blue', rr))
            igpa.setdefault(p, []).append(("%s:%s" % (q, FEEDS[1]), 'red', ll))
            igpa.setdefault(q, []).append(("%s:%s" % (p, FEEDS[1]), 'red', ll))
        content = [(p, igpa[p]) for p in sorted(igpa.keys(), cmp=_cmp_antenna)]
        pylab.subplot(NR, NC, 2)
        plot_ants(content, "IFR %s %s gain amplitudes per antenna" % FEEDS)
        if baseline:
            pylab.subplot(NR, NC/2, 3)
            plot_baseline(norm_igs, baseline, "IFR gain amplitude vs. baseline length", FEEDS)
        save_figure("bbc-{}".format(label), width*NC, height*NR)

        # make per-antenna figure
        antennas = sorted(igpa0.keys(), _cmp_antenna)
        NC = 4
        NR = int(math.ceil(len(antennas) / float(NC)))
        offset = np.median(igpa0_means)
        for iant, pant in enumerate(antennas):
            pylab.subplot(NR, NC, iant + 1)
            igains = igpa0[pant]
            ants1 = sorted(igains.keys(), _cmp_antenna)
            for i, qant in enumerate(ants1):
                rr, ll = igains[qant]
                if rr.count() > 1:
                    a1, a2 = np.ma.flatnotmasked_edges(rr)
                    line, = pylab.plot(rr + i * offset, '-')
                    pylab.text(a1, rr[a1] + i * offset, "%s:%s" % (qant, feeds[0]), horizontalalignment='left',
                               verticalalignment='center', size=8,
                               color=line.get_color())
                if ll.count() > 1:
                    a1, a2 = np.ma.flatnotmasked_edges(ll)
                    line, = pylab.plot(ll + i * offset, '-')
                    pylab.text(a2, ll[a2] + i * offset, "%s:%s" % (qant, feeds[3]), horizontalalignment='right',
                               verticalalignment='center', size=8,
                               color=line.get_color())
            pylab.title("antenna %s" % pant)
        save_figure("bbc-{}-ant".format(label), 8 * NC, 6 * NR)

