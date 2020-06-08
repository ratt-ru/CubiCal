from __future__ import print_function
from builtins import range
import numpy as np
import os, os.path
import traceback
from cubical.tools import logger, ModColor
from cubical.flagging import FL
from cubical.statistics import SolverStats
from cubical.tools import BREAK  # useful: can set static breakpoints by putting BREAK() in the code
from cubical.madmax import plots

log = logger.getLogger("madmax")


# Conversion factor for sigma = SIGMA_MAD*mad
SIGMA_MAD = 1.4826

import builtins
try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile 
    

class Flagger(object):
    def __init__(self, GD, chunk_label, metadata, stats):
        self.GD = GD
        self.metadata = metadata
        self.stats = stats
        self.chunk_label = chunk_label

        self.flag_warning_threshold = GD['flags']["warn-thr"]

        self.mad_threshold = GD['madmax']['threshold']
        self.medmad_threshold = GD['madmax']['global-threshold']
        
        # make copies of theses lists, since we manipuate them in get_thresholds()
        if not isinstance(self.mad_threshold, list):
            self.mad_threshold = [self.mad_threshold]
        else:
            self.mad_threshold = list(self.mad_threshold)
        if not isinstance(self.medmad_threshold, list):
            self.medmad_threshold = [self.medmad_threshold]
        else:
            self.medmad_threshold = list(self.medmad_threshold)
            
        self.mad_diag = GD['madmax']['diag']
        self.mad_offdiag = self.metadata.num_corrs == 4 and GD['madmax']['offdiag']
        if not self.mad_diag and not self.mad_offdiag:
            self._mode = False

        # setup MAD estimation settings
        self.mad_per_corr = False
        if GD['madmax']['estimate'] == 'corr':
            self.mad_per_corr = True
            self.mad_estimate_diag, self.mad_estimate_offdiag = self.mad_diag, self.mad_offdiag
        elif GD['madmax']['estimate'] == 'all':
            self.mad_estimate_diag = True
            self.mad_estimate_offdiag = self.metadata.num_corrs == 4
        elif GD['madmax']['estimate'] == 'diag':
            self.mad_estimate_diag, self.mad_estimate_offdiag = True, False
        elif GD['madmax']['estimate'] == 'offdiag':
            if self.metadata.num_corrs == 4:
                self.mad_estimate_diag, self.mad_estimate_offdiag = False, True
            else:
                self.mad_estimate_diag, self.mad_estimate_offdiag = True, False
        else:
            raise RuntimeError("invalid --madmax-estimate {} setting".format(GD['madmax']['estimate']))

        if GD['madmax']['plot-bl']:
            self._plot_baselines = set(GD['madmax']['plot-bl'].split(","))
        else:
            self._plot_baselines = set()

        self._plotnum = 0

    def set_mode(self, mode):
        """Sets operational mode for flagger"""
        self._mode = mode
        self._pretend = self._mode == "pretend"
        self.trial_mode = self._trial = self._mode == "trial"
        if self._pretend:
            self.desc_mode = "Pretend-Mad Max"
        elif self._mode == "trial":
            self.desc_mode = "Trial-Mad Max"
        elif self._mode:
            self.desc_mode = "Mad Max"
        else:
            self.desc_mode = "No Max"
        self.flagbit = 0 if self._pretend else FL.MAD

    def get_mad_thresholds(self):
        """MAD thresholds above are either a list, or empty. Each time we access the list, we pop the first element,
        until the list is down to one element."""
        if not self._mode:
            return 0, 0
        return self.mad_threshold.pop(0) if len(self.mad_threshold) > 1 else \
                   (self.mad_threshold[0] if self.mad_threshold else 0), \
               self.medmad_threshold.pop(0) if len(self.medmad_threshold) > 1 else \
                   (self.medmad_threshold[0] if self.medmad_threshold else 0)

    def get_plot_filename(self, kind=''):
        plotdir = '{}-madmax.plots'.format(self.GD['out']['name'])
        if not os.path.exists(plotdir):
            try:
                os.mkdir(plotdir)
            # allow a failure -- perhaps two workers got unlucky and both are trying to make the
            # same directory. Let savefig() below fail instead
            except OSError:
                pass
        if kind:
            filename = '{}/{}.{}.{}.png'.format(plotdir, self.chunk_label, self._plotnum, kind)
        else:
            filename = '{}/{}.{}.png'.format(plotdir, self.chunk_label, self._plotnum)
        self._plotnum += 1
        return filename

    @builtins.profile
    def report_carnage(self, absres, mad, baddies, flags_arr, method, max_label):
        made_plots = False
        n_tim, n_fre, n_ant, n_ant = baddies.shape
        nbad = int(baddies.sum())
        self.stats.chunk.num_mad_flagged += nbad

        if nbad or self._plot_baselines:
            if nbad < flags_arr.size * self.flag_warning_threshold:
                warning, color = "", "blue"
            else:
                warning, color = "WARNING: ", "red"
            frac = nbad / float(baddies.size)
            mode = "trial-" if self._trial else ("pretend-" if self._pretend else "")
            print("{warning}{max_label} {method} {mode}flags {nbad} ({frac:.2%}) visibilities".format(**locals()), file=log(1, color))
            if log.verbosity() > 2 or self.GD['madmax']['plot']:
                per_bl = []
                total_elements = float(n_tim * n_fre)
                interesting_fraction = self.GD['madmax']['plot-frac-above']*total_elements
                plot_explicit_baselines = []
                for p in range(n_ant):
                    for q in range(p + 1, n_ant):
                        n_flagged = baddies[:, :, p, q].sum()
                        if n_flagged and n_flagged >= interesting_fraction:
                            per_bl.append((n_flagged, p, q))
                        if self.metadata.baseline_name[p,q] in self._plot_baselines:
                            plot_explicit_baselines.append(((n_flagged, p, q), "--madmax-plot-bl"))
                per_bl = sorted(per_bl, reverse=True)
                # print
                per_bl_str = ["{} ({}m): {} ({:.2%})".format(self.metadata.baseline_name[p,q],
                                int(self.metadata.baseline_length[p,q]), n_flagged, n_flagged/total_elements)
                              for n_flagged, p, q in per_bl]
                print("{} of which per baseline: {}".format(max_label, ", ".join(per_bl_str)), file=log(3))
                # plot, if asked to
                if self.GD['madmax']['plot']:
                    baselines_to_plot = []
                    if len(per_bl):
                        baselines_to_plot.append((per_bl[0], "worst baseline"))
                    if len(per_bl)>2:
                        baselines_to_plot.append((per_bl[len(per_bl)//2], "median baseline"))
                    baselines_to_plot += plot_explicit_baselines
                    import pylab
                    for (n_flagged, p, q), baseline_label in baselines_to_plot:
                        blname = self.metadata.baseline_name[p, q] 
                        # make subplots
                        subplot_titles = {}
                        for c1,x1 in enumerate(self.metadata.feeds.upper()):
                            for c2,x2 in enumerate(self.metadata.feeds.upper()):
                                mm = mad[0,p,q,c1,c2] if self.mad_per_corr else mad[0,p,q]
                                subplot_titles[c1,c2] = "{}{} residuals (MAD {:.3g})".format(x1, x2, 0 if mm is np.ma.masked else mm) 
                        try:
                            figure = plots.make_dual_absres_plot(absres, flags_arr!=0, baddies, p, q, self.metadata, subplot_titles)
                            # make plot title with some info
                            fraction = n_flagged / total_elements
                            blname = self.metadata.baseline_name[p,q]
                            bllen  = int(self.metadata.baseline_length[p,q])
                            pylab.suptitle("{} {}: baseline {} ({}m), {} ({:.2%}) visibilities killed ({})".format(max_label,
                                            method, blname, bllen, n_flagged, fraction, baseline_label))
                            # save or show plot
                            if self.GD['madmax']['plot'] == 'show':
                                pylab.show()
                            else:
                                filename = self.get_plot_filename()
                                figure.savefig(filename, dpi=300)
                                print("{}: saving Mad Max flagging plot to {}".format(self.chunk_label,filename), file=log(1))
                            pylab.close(figure)
                            del figure
                            made_plots = True
                        except Exception as exc:
                            if self.GD['debug']['escalate-warnings']:
                                raise
                            traceback.print_exc()
                            print("WARNING: {}: exception {} raised while generating Mad Max waterfall plot for baseline {} ({})".format(
                                            self.chunk_label, exc, blname, baseline_label), file=log(0, "red"))
                            print("Although harmless, this may indicate a problem with the data, or a bug in CubiCal.", file=log(0))
                            print("Please see stack trace above, and report if you think this is a bug.", file=log(0))
        else:
            print("{} {} abides".format(max_label, method), file=log(2))

        return made_plots, nbad>0


    @builtins.profile
    def beyond_thunderdome(self, resid_arr, data_arr, model_arr, flags_arr, threshold, med_threshold, max_label):
        """This function implements MAD-based flagging on residuals"""
        if not threshold and not med_threshold:
            return False
        # residuals can pe per-model (when invoked from solver) or not, when invoked on final residuals.
        # cope with both cases
        if resid_arr.ndim == 7:
            n_mod, _, _, n_ant, n_ant, n_cor, n_cor = resid_arr.shape
        else:
            _, _, n_ant, n_ant, n_cor, n_cor = resid_arr.shape
            n_mod = 1
            resid_arr = resid_arr.reshape([1]+list(resid_arr.shape))

        import cubical.kernels
        madmax = cubical.kernels.import_kernel("madmax")
        # estimate MAD of off-diagonal elements
        absres = np.empty_like(resid_arr, dtype=np.float32)
        np.abs(resid_arr, out=absres)
        if self.mad_per_corr:
            mad, goodies = madmax.compute_mad_per_corr(absres, flags_arr, diag=self.mad_estimate_diag, offdiag=self.mad_estimate_offdiag)
        else:
            mad, goodies = madmax.compute_mad(absres, flags_arr, diag=self.mad_estimate_diag, offdiag=self.mad_estimate_offdiag)
        # any of it non-zero?
        if mad.mask.all():
            return
        # estimate median MAD
        ### want to do this:
        ## medmad = np.ma.median(mad, axis=(1,2))
        ### but this seems to thrown an error on earlier numpys (1.11?), so let's be defensive and reshape into one axis:
        shape1 = [mad.shape[0], mad.shape[1]*mad.shape[2]] + list(mad.shape[3:])
        medmad = np.ma.median(mad.reshape(shape1), axis=1)
        # all this was worth it, just so I could type "mad.max()" as legit code
        print("{} per-baseline MAD min {:.3g}, max {:.3g}, median {:.3g} to {:.3g}".format(max_label, mad.min(), mad.max(), medmad.min(), medmad.max()), file=log(2))
        if log.verbosity() > 4:
            for imod in range(n_mod):
                if self.mad_per_corr:
                    for ic1,c1 in enumerate(self.metadata.feeds):
                        for ic2,c2 in enumerate(self.metadata.feeds):
                            per_bl = [(mad[imod,p,q,ic1,ic2], p, q) for p in range(n_ant)
                                      for q in range(p+1, n_ant) if not mad.mask[imod,p,q,ic1,ic2]]
                            per_bl = ["{} ({}m): {:.3g}".format(self.metadata.baseline_name[p,q], int(self.metadata.baseline_length[p,q]), x)
                                      for x, p, q in sorted(per_bl)[::-1]]
                            print("{} model {} {}{} MADs are {}".format(max_label, imod,
                                                                                c1.upper(), c2.upper(), ", ".join(per_bl)), file=log(4))
                else:
                    per_bl = [(mad[imod,p,q,], p, q) for p in range(n_ant)
                              for q in range(p+1, n_ant) if not mad.mask[imod,p,q]]
                    per_bl = ["{} ({}m) {:.3g}".format(self.metadata.baseline_name[p,q], int(self.metadata.baseline_length[p,q]), x)
                              for x, p, q in sorted(per_bl)[::-1]]
                    print("{} model {} MADs are {}".format(max_label, imod, ", ".join(per_bl)), file=log(4))


        made_plots = flagged_something = False

        thr = np.zeros((n_mod, n_ant, n_ant, n_cor, n_cor), dtype=np.float32)
        # apply per-baseline MAD threshold
        if threshold:
            if self.mad_per_corr:
                thr[:] = threshold * mad / SIGMA_MAD
            else:
                thr[:] = threshold * mad[...,np.newaxis,np.newaxis] / SIGMA_MAD
            baddies = madmax.threshold_mad(absres, thr, flags_arr, self.flagbit, goodies,
                                             diag=self.mad_diag, offdiag=self.mad_offdiag)
            made_plots, flagged_something  = self.report_carnage(absres, mad, baddies, flags_arr,
                                                "baseline-based Mad Max ({} sigma)".format(threshold), max_label)
            if not self._pretend:
                baddies = baddies.astype(bool)
                if model_arr is not None:
                    model_arr[:,:,baddies,:,:] = 0
                if data_arr is not None:
                    data_arr[:,baddies,:,:] = 0

        # apply global median MAD threshold
        if med_threshold:
            med_thr = med_threshold * medmad / SIGMA_MAD
            if self.mad_per_corr:
                thr[:] = med_thr[:,np.newaxis,np.newaxis,:,:]
            else:
                thr[:] = med_thr[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
            baddies = madmax.threshold_mad(absres, thr, flags_arr, self.flagbit, goodies,
                                             diag=self.mad_diag, offdiag=self.mad_offdiag)

            made, flagged = \
                self.report_carnage(absres, mad, baddies, flags_arr,
                                       "global Mad Max ({} sigma)".format(med_threshold), max_label)

            made_plots = made_plots or made
            flagged_something = flagged_something or flagged

            if not self._pretend:
                baddies = baddies.astype(bool)
                if model_arr is not None:
                    model_arr[:, :, baddies, :, :] = 0
                if data_arr is not None:
                    data_arr[:, baddies, :, :] = 0
        else:
            med_thr = None

        # generate overview plot
        if made_plots:
            import pylab
            outflags, figure = plots.make_baseline_mad_plot(mad, medmad, med_thr, metadata=self.metadata,
                                max_label=max_label, chunk_label=self.chunk_label,
                                antenna_mad_threshold=self.GD['madmax']['flag-ant-thr'])
            if outflags.any():
                flagged_something = True
                if self.mad_per_corr:
                    outflags = outflags.any(axis=(-1,-2))
                if self.GD['madmax']['flag-ant'] and not self._pretend:
                    print("{} baselines {}flagged on mad residuals (--madmax-flag-ant 1)".format(
                                            outflags.sum()/2, "trial-" if self._trial else ""), file=log(0, "red"))
                    flags_arr[:,:,outflags] |= self.flagbit
                    if model_arr is not None:
                        model_arr[:,:,:,:,outflags,:,:] = 0
                    if data_arr is not None:
                        data_arr[:,:,:,outflags,:,:] = 0
                else:
                    print("{} baselines would have been flagged due to mad residuals (use --madmax-flag-ant to enable this)".format(outflags.sum()/2), file=log(0, "red"))

            try:
                if self.GD['madmax']['plot'] == 'show':
                    pylab.show()
                else:
                    filename = self.get_plot_filename('mads')
                    print("{}: saving MAD distribution plot to {}".format(self.chunk_label,filename), file=log(1))
                    figure.savefig(filename, dpi=300)
                    from future.moves import pickle
                    pickle_file = filename+".cp"
                    with open(pickle_file, "wb") as pf:
                        pickle.dump((mad, medmad, med_thr, self.metadata, max_label), pf, 2)
                    print("{}: pickled MAD distribution to {}".format(self.chunk_label, pickle_file), file=log(1))
                pylab.close(figure)
                del figure
            except Exception as exc:
                if self.GD['debug']['escalate-warnings']:
                    raise
                traceback.print_exc()
                print("WARNING: {}: exception {} raised while rendering Mad Max summary plot".format(
                                        self.chunk_label, exc), file=log(0,"red"))
                print("Although harmless, this may indicate a problem with the data, or a bug in CubiCal.", file=log(0))
                print("Please see stack trace above, and report if you think this is a bug.", file=log(0))

        return flagged_something and not self._pretend