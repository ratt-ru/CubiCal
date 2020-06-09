from __future__ import print_function
from builtins import range
import numpy as np
import traceback

from cubical.tools import BREAK  # useful: can set static breakpoints by putting BREAK() in the code

from cubical.tools import logger
log = logger.getLogger("madmax")

def make_dual_absres_plot(absres, fl_prior, fl_new, p, q, metadata, subplot_titles={}):
    import pylab
    feeds = metadata.feeds
    # inv: data that was flagged prior to this mad max step
    fl_prior = fl_prior[:, :, p, q] & ~fl_new[:, :, p, q]
    figure = pylab.figure(figsize=(16, 10))
    resmask = np.zeros_like(absres[0, :, :, p, q], dtype=bool)
    resmask[:] = fl_prior[..., np.newaxis, np.newaxis] #| absres[0, :, :, p, q]==0
    res = np.ma.masked_array(absres[0, :, :, p, q], resmask)
    resmin, resmax = res.min(), res.max()
    vmin = float(max(resmin if resmin is not np.ma.masked else 0, 1e-9))
    vmax = float(resmax if resmax is not np.ma.masked else 1)
    if vmin != vmax:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin, vmax)
    else:
        norm = None
    for c1, x1 in enumerate(feeds.upper()):
        for c2, x2 in enumerate(feeds.upper()):
            pylab.subplot(2, 4, 1 + c1 * 2 + c2)
            if (res[..., c1, c2] > 0).any():
                pylab.imshow(res[..., c1, c2], norm=norm, aspect='auto')
                try:
                    pylab.colorbar()
                except ZeroDivisionError:
                    pass
            pylab.title(subplot_titles.get((c1,c2)))
    for c1, x1 in enumerate(feeds.upper()):
        for c2, x2 in enumerate(feeds.upper()):
            pylab.subplot(2, 4, 5 + c1 * 2 + c2)
            res1 = np.ma.masked_array(absres[0, :, :, p, q, c1, c2], fl_prior | fl_new[:, :, p, q])
            if (res1 > 0).any():
                pylab.imshow(res1, norm=norm, aspect='auto')
                try:
                    pylab.colorbar()
                except ZeroDivisionError:
                    pass
            pylab.title("{}{} flagged".format(x1, x2))
    return figure



def make_baseline_mad_plot(mad, medmad, med_thr, metadata, max_label="", chunk_label="", antenna_mad_threshold=0):
    import pylab
    colors = [["black", "red"], ["green", "blue"]]
    n_mod = mad.shape[0]
    n_ant = mad.shape[1]
    per_corr = medmad.ndim == 3
    n_cor = medmad.shape[1]

    outflags = np.zeros(mad.shape[1:],bool)

    # sort baselines by length and form up index list
    baselines = [ (p,q) for p in range(n_ant) for q in range(p+1, n_ant) ]
    # sort arrays by baseline length
    indices_pq = sorted([(metadata.baseline_length[p, q], p, q) for p, q in baselines])

    from cubical.madmax.flagger import SIGMA_MAD

    def make_antenna_mads(mad_threshold):
        print("make_baseline_mad_plot: plotting antennas", file=log(3))
        # compute per-antenna MAD
        if per_corr:
            ## again, wanted to do this
            # medant = np.ma.median(mad[0,...], axis=(1,2,3))
            ## but not all numpys like it, so being defensive:
            shape1 = mad[0,...].shape
            shape1 = [shape1[0], shape1[1]*shape1[2]*shape1[3]] + list(shape1[4:])
            medant = np.ma.median(mad[0,...].reshape(shape1), axis=1)
        else:
            medant = np.ma.median(mad[0,...], axis=1)
        antnum = np.ma.masked_array(range(n_ant), medant.mask)
        if not medant.mask.all():
            medmed = np.ma.median(medant)
            madmed = np.ma.median(abs(medant-medmed))

            # make plot
            try:
                pylab.plot(antnum, medant, ls='-', color='white')
                pylab.axhline(medmed, color='0.5')
                thresholds = [(1,'green'), (2, 'blue'), (3,'orange')]
                if mad_threshold:
                    thresholds.append((mad_threshold, "red"))
                for thr,color in thresholds:
                    pylab.axhline(medmed+thr*SIGMA_MAD*madmed, color=color, ls=':')
                for p in range(n_ant):
                    if antnum.mask is np.ma.nomask or not antnum.mask[p]:
                        pylab.axvline(antnum[p], color="0.9")
                        color = "black"
                        for thr, col in thresholds:
                            if medant[p] > medmed + thr*SIGMA_MAD*madmed:
                                color = col
                        pylab.text(antnum[p], medant[p], metadata.antenna_name[p], color=color, ha='center', va='center')
                pylab.xticks([])
                pylab.xlabel("antenna")
                pylab.ylabel("MAD residual over baselines")
            except Exception as exc:
                traceback.print_exc()
                print("WARNING: {}: exception {} raised while generating Mad Max antenna-MAD plot".format(
                                        chunk_label, exc), file=log(1,"red"))
                print("Although harmless, this may indicate a problem with the data, or a bug in CubiCal.", file=log(1))
                print("Please see stack trace above, and report if you think this is a bug.", file=log(1))


            if mad_threshold:
                antmask = medant > medmed + mad_threshold*SIGMA_MAD*madmed
                if antmask.any():
                    print("{}: antennas {} have mad residuals, refer to Mad Max plots".format(max_label,
                                                ",".join([metadata.antenna_name[p] for p,fl in enumerate(antmask) if fl])), file=log(0,"red"))
                else:
                    print("{}: no antennas with mad residuals".format(max_label), file=log(1))
                outflags[antmask,:] = True
                outflags[:,antmask] = True
                if per_corr:
                    mad.mask |= antmask[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
                    mad.mask |= antmask[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
                else:
                    mad.mask |= antmask[np.newaxis, :, np.newaxis]
                    mad.mask |= antmask[np.newaxis, np.newaxis, :]
                return antmask
        return None

    def make_baseline_mads():
        # remake indices, since flagged (masked) baselines may have changed
        if per_corr:
            indices = [(bl,p,q,c1,c2) for bl,p,q in indices_pq for c1 in range(n_cor) for c2 in range(n_cor)]
            mask = [mad.mask[0,p,q,c1,c2] for _,p,q,c1,c2 in indices]
            blmad = np.ma.masked_array([(mad[0,p,q,c1,c2] or 0) for _,p,q,c1,c2 in indices],mask)
        else:
            indices = [(bl,p,q,None,None) for bl,p,q in indices_pq]
            mask = [mad.mask[0,p,q] for _,p,q in indices_pq]
            blmad = np.ma.masked_array([(mad[0,p,q] or 0) for _,p,q in indices_pq],mask)

        bllen = np.ma.masked_array([bl for bl,_,_,_,_ in indices],mask)
        # N,2 array of pq indices for each entry
        blpq  = np.array([[p,q] for bl,p,q,_,_ in indices])

        # for every baseline, compute local MMAD
        from cubical.madmax.flagger import SIGMA_MAD
        print("make_baseline_mad_plot: computing LMMAD", file=log(3))
        lmmad = {}
        for i,(_,p,q,_,_) in enumerate(indices):
            if (p,q) not in lmmad:
                bllen0 = metadata.baseline_length[p,q]
                selection = (bllen>0.8*bllen0)&(bllen<1.2*bllen0)&((p!=blpq[:,0])|(q!=blpq[:,1]))
                selection = selection&~blmad.mask
                if selection.any():
                    med = np.ma.median(blmad[selection&~blmad.mask])
                    if med is not np.ma.masked:
                        lmmad[p,q] = med

        lmmad_threshold = 3
        lmmad_mask = [((p,q) not in lmmad) for _,p,q in indices_pq]
        lmmad_bllen = np.ma.masked_array([bl for bl,_,_ in indices_pq], lmmad_mask)
        lmmad_mad   = np.ma.masked_array([(lmmad.get((p,q), 0) or 0) for _,p,q in indices_pq], lmmad_mask)

        lmmad_ad = np.ma.masked_array([abs((blmad1 or 0) - lmmad.get((p,q), 0)) for (_,p,q,_,_),blmad1 in zip(indices,blmad)], blmad.mask)
        lmmad_madmad = np.ma.median(lmmad_ad)
        print("make_baseline_mad_plot: plotting baselines", file=log(3))


        xlim = [0, 0]
        ylim = [0, 0]
        try:
            wh = ~lmmad_bllen.mask
            pylab.fill_between(lmmad_bllen[wh], lmmad_mad[wh], (lmmad_mad+lmmad_madmad*lmmad_threshold)[wh], color='0.8', step='mid')
            if med_thr is not None:
                if per_corr:
                    for ic1, c1 in enumerate(metadata.feeds.upper()):
                        for ic2, c2 in enumerate(metadata.feeds.upper()):
                            if not medmad.mask[0, ic1, ic2]: 
                                pylab.axhline(medmad[0, ic1, ic2], ls="-", color=colors[ic1][ic2])
                                pylab.text(0, medmad[0, ic1, ic2], "MMAD", color=colors[ic1][ic2],
                                       ha='right', va='center', size='x-small')
                                if med_thr is not None:
                                    pylab.axhline(med_thr[0, ic1, ic2], ls=":", color=colors[ic1][ic2])
                                    pylab.text(0, med_thr[0, ic1, ic2], "threshold", color=colors[ic1][ic2],
                                           ha='right', va='center', size='x-small')
                else:
                    pylab.axhline(medmad[0], ls="-", color="blue")
                    pylab.text(0, medmad[0], "MMAD", color="blue",
                               ha='right', va='center', size='x-small')
                    if med_thr is not None:
                        pylab.axhline(med_thr[0], ls=":", color="black")
                        pylab.text(0, med_thr[0], "threshold", color="black",
                                   ha='right', va='center', size='x-small')

            for p in range(n_ant):
                for q in range(p + 1, n_ant):
                    if not mad.mask[0, p, q].all():
                        uvdist = metadata.baseline_length[p, q]
                        xlim[1] = max(xlim[1], uvdist)
                        if per_corr:
                            for ic1, c1 in enumerate(metadata.feeds.upper()):
                                for ic2, c2 in enumerate(metadata.feeds.upper()):
                                    if not mad.mask[0, p, q, ic1, ic2]:
                                        y = mad[0, p, q, ic1, ic2]
                                        pylab.text(uvdist, y, metadata.baseline_name[p, q],
                                                   color=colors[ic1][ic2],
                                                   horizontalalignment='center', verticalalignment='center')
                                        ylim[0] = min(ylim[0], y)
                                        ylim[1] = max(ylim[1], y)
                        else:
                            y = mad[0, p, q]
                            pylab.text(uvdist, y, metadata.baseline_name[p, q],
                                       horizontalalignment='center', verticalalignment='center')
                            ylim[0] = min(ylim[0], y)
                            ylim[1] = max(ylim[1], y)

            import matplotlib.lines as mlines
            if per_corr:
                handles = []
                for ic1, c1 in enumerate(metadata.feeds):
                    for ic2, c2 in enumerate(metadata.feeds):
                        handles.append(mlines.Line2D([], [], color=colors[ic1][ic2], label="{}{}".format(c1, c2).upper()))
            else:
                handles = [ mlines.Line2D([], [], color="blue", ls="_", label="MMAD"),
                            mlines.Line2D([], [], color="black", label="threshold") ]
            handles.append(mlines.Line2D([], [], color="0.8", label="LMMAD*{:.2f}".format(lmmad_threshold)))
            pylab.legend(handles=handles)
            pylab.xlim(*xlim)
            pylab.ylim(*ylim)
            pylab.xlabel("Baseline, m.")
            pylab.ylabel("MAD residuals")
            pylab.title("{}: MAD residuals".format(max_label))
        except Exception as exc:
            traceback.print_exc()
            print("WARNING: {}: exception {} raised while generating Mad Max baseline-MAD plot".format(
                chunk_label, exc), file=log(1, "red"))
            print("Although harmless, this may indicate a problem with the data, or a bug in CubiCal.", file=log(1))
            print("Please see stack trace above, and report if you think this is a bug.", file=log(1))


    figure = pylab.figure(figsize=(16, 10))
    pylab.subplot(2,2,2)
    make_baseline_mads()

    pylab.subplot(2,2,1)
    make_antenna_mads(antenna_mad_threshold)

    pylab.subplot(2,2,4)
    make_baseline_mads()

    pylab.subplot(2,2,3)
    make_antenna_mads(antenna_mad_threshold)

    print("make_baseline_mad_plot: done", file=log(3))

    return outflags, figure

