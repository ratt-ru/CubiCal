# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from __future__ import print_function
from builtins import range
import numpy as np
from cubical.flagging import FL
from cubical.machines.abstract_machine import MasterMachine
import cubical.kernels
from cubical.solver import log
import logging
import re
from numpy.ma import masked_array

def copy_or_identity(array, time_ind=0, out=None):
    if out is None:
        return array
    np.copyto(out, array)
    return out


class PerIntervalGains(MasterMachine):
    """
    This is a base class for all gain solution machines that use solutions intervals.
    """

    def __init__(self, label, data_arr, ndir, nmod, times, frequencies, chunk_label, options):
        """
        Initialises a gain machine which supports solution intervals.
        
        Args:
            label (str):
                Label identifying the Jones term.
            data_arr (np.ndarray): 
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing observed 
                visibilities. 
            ndir (int):
                Number of directions.
            nmod (nmod):
                Number of models.
            times (np.ndarray):
                Times for the data being processed.
            freqs (np.ndarray):
                Frequencies for the data being processsed.
            options (dict): 
                Dictionary of options.
            diag_gains (bool):
                If True, gains are diagonal-only. Else gains are full 2x2.
        """

        MasterMachine.__init__(self, label, data_arr, ndir, nmod, times, frequencies,
                               chunk_label, options)

        # === Kernel selection
        # We have three types of (progressively faster) kernels: 2x2, Diag (diag gains), and DiagDiag (diag gains and data).
        # We need to choose a kernel for the solve process, and a kernel for computing full residuals.
        # The choice is based on three binary options that control kernel selection:
        #
        #  1. diag-data   (True if data is null on the off-diagonals)
        #  2. diag-only   (True if only the diagonal terms should be used for the solution)
        #  3. is_diagonal (True if the gain matrix is diagonal)
        #
        # Here's the selection matrix:
        #
        #  diag-data  diag-only  is_diagonal    solve kernel    full kernel
        #  ---------  ---------  -----------    ------------    -----------
        #      0          0           0             2x2            2x2
        #      0          0           1            diag           diag
        #      0          1           0             2x2            2x2
        #      0          1           1          diag-diag        diag
        #      1          0           0             2x2            2x2
        #      1          0           1          diag-diag      diag-diag
        #      1          1           0             2x2            2x2
        #      1          1           1          diag-diag      diag-diag

        # select which kernels to use for computing full data
        self.kernel = self.get_full_kernel(options, self.is_diagonal)

        # as per matrix above, the solve kernel is the same as the full kernel, except if
        # solving for diagonal terms only, with a diagonal gain, in which case we can use diag-diag.
        if not options.get('diag-data') and self.diag_only and self.is_diagonal:
            self.kernel_solve = cubical.kernels.import_kernel('diagdiag_complex')
        else:
            self.kernel_solve = self.kernel

        log(2).print("{} kernels are {} {}".format(label, self.kernel, self.kernel_solve))

        self.t_int = options["time-int"] or self.n_tim
        self.f_int = options["freq-int"] or self.n_fre
        self.eps = 1e-6

        # Initialise attributes used for computing values over intervals.
        # n_tim and n_fre are the time and frequency dimensions of the data arrays.
        # n_timint and n_freint are the time and frequency dimensions of the gains.

        self.t_bins = list(range(0, self.n_tim, self.t_int))
        self.f_bins = list(range(0, self.n_fre, self.f_int))

        self.n_timint = len(self.t_bins)
        self.n_freint = len(self.f_bins)
        self.n_tf_ints = self.n_timint * self.n_freint

        # number of valid solutions
        self.n_valid_sols = self.n_dir * self.n_tf_ints

        # split grids into intervals, and find the centre of gravity of each
        timebins = np.split(times, self.t_bins[1:])
        freqbins = np.split(frequencies, self.f_bins[1:])
        timegrid = np.array([float(x.mean()) for x in timebins])
        freqgrid = np.array([float(x.mean()) for x in freqbins])

        # interval_grid determines the per-interval grid poins
        self.interval_grid = dict(time=timegrid, freq=freqgrid)
        # data_grid determines the full resolution grid
        self.data_grid = dict(time=times, freq=frequencies)

        # compute index from each data point to interval number
        t_ind = np.arange(self.n_tim)//self.t_int
        f_ind = np.arange(self.n_fre)//self.f_int

        self.t_mapping, self.f_mapping = np.meshgrid(t_ind, f_ind, indexing="ij")

        # Initialise attributes used in convergence testing. n_cnvgd is the number
        # of solutions which have converged.

        self._has_stalled = False
        self.n_cnvgd = 0
        self._frac_cnvgd = 0
        self.iters = 0
        self.min_quorum = options["conv-quorum"]
        self.update_type = set(options["update-type"].split("-"))
        self.ref_ant = options["ref-ant"]
        if self.ref_ant is not None:
            if type(self.ref_ant) is not str:
                self.ref_ant = str(self.ref_ant)
            if self.ref_ant[0] == "#":
                self.ref_ant = int(self.ref_ant[1:])
            else:
                from cubical.solver import metadata
                self.ref_ant = metadata.antenna_index[self.ref_ant]

        self.fix_directions = options["fix-dirs"] if options["fix-dirs"] is not None and \
                options["fix-dirs"] != "" else []

        if type(self.fix_directions) is int:
            self.fix_directions = [self.fix_directions]
        if type(self.fix_directions) is str and re.match(r"^\W*\d{1,}(\W*,\W*\d{1,})*\W*$", self.fix_directions):
            self.fix_directions = map(int, map(str.strip, ",".split(self.fix_directions)))

        if not (type(self.fix_directions) is list and
                all(map(lambda x: type(x) is int, self.fix_directions))):
            raise ValueError("fix-dir must be number or list of numbers")

        # True if gains are loaded from a DB
        self._gains_loaded = False

        # Construct flag array and populate flagging attributes.
        self.max_gain_error = options["max-prior-error"]
        self.max_post_error = options["max-post-error"]
        self.low_snr_warn = options["low-snr-warn"]
        self.high_gain_var_warn = options["high-gain-var-warn"]
        self.clip_lower = options["clip-low"]
        self.clip_upper = options["clip-high"]
        self.clip_after = options["clip-after"]

        self.init_gains()
        self.old_gains = self.gains.copy()

        # mask of intervals/antennas for which we have no data (i.e. all flagged to begin with). Shape TFA.
        self.missing_intervals_ant = None
        # mask of intervals/antennas for which we have valid solutions. Note that this is not necessarily
        # a full complement to missing_intervals_ant -- some intervals/antennas may become flagged due to low SNR,
        # ill-conditioning, etc., in which case they're neither valid, nor missing. Shape DTFA.
        self.valid_intervals_ant = None

        # number of gains flagged on various conditions
        self.num_flagged_on = {}

        # Gain error estimates. Populated by subclasses, if available
        # Should be array of same shape as the gains
        self.prior_gain_error = None
        self.posterior_gain_error = None

        # buffers for arrays used in internal updates
        self._jh = self._jhr = self._jhj = self._gh = self._r = self._ginv = self._ghinv = None
        self._update = None

        # flag: have gains been updated
        self._gh_update = self._ghinv_update = True

    @classmethod
    def determine_allocators(cls, options):
        kernel = cls.get_full_kernel(options, diag_gains=cls.determine_diagonality(options))
        return kernel.allocate_vis_array, kernel.allocate_flag_array, kernel.allocate_gain_array

    @classmethod
    def get_full_kernel(cls, options, diag_gains):
        """
        Class method. Given a machine class and a set of options, returns numerical kernel
        for "full 2x2" operations on visibilities.
        """
        # return cubical.kernels.import_kernel('full_complex')
        # select which kernels to use for full correction/residuals -- see selection matrix above
        if diag_gains:
            if options.get('diag-data'):
                return cubical.kernels.import_kernel('diagdiag_complex')
            else:
                return cubical.kernels.import_kernel('diag_complex')
        else:
            return cubical.kernels.import_kernel('full_complex')

    def get_conj_gains(self):
        if self._gh is None:
            self._gh = np.empty_like(self.gains)
        if self._gh_update:
            np.conj(self.gains.transpose(0, 1, 2, 3, 5, 4), out=self._gh)
            self._gh_update = False
        return self._gh

    def get_inverse_gains(self):
        if self._ginv is None:
            self._ginv = np.empty_like(self.gains)
            self._ghinv = np.empty_like(self.gains)
        if self._ghinv_update:
            self._ghinv_flag_count = self.kernel_solve.invert_gains(
                self.gains, self._ginv, self.gflags, self.eps, FL.ILLCOND)
            np.conj(self._ginv.transpose(0, 1, 2, 3, 5, 4), out=self._ghinv)
            self._ghinv_update = False
        return self._ginv, self._ghinv, self._ghinv_flag_count

    def get_new_jh(self, model_arr):
        if self._jh is None:
            self._jh = np.empty_like(model_arr)
        # else:
        #     self._jh.fill(0)
        return self._jh

    def get_new_jhr(self):
        if self._jhr is None:
            self._jhr = np.zeros_like(self.gains)
        else:
            self._jhr.fill(0)
        return self._jhr

    def get_new_jhj(self):
        if self._jhj is None:
            self._jhj = np.zeros_like(self.gains)
            self._jhjinv = np.empty_like(self.gains)
        else:
            self._jhj.fill(0)
        return self._jhj, self._jhjinv

    def init_update(self, jhr):
        if self._update is None:
            self._update = np.zeros_like(jhr)
        else:
            self._update.fill(0)
        return self._update

    def get_obs_or_res(self, obser_arr, model_arr):
        if self.n_dir > 1:
            if self._r is None:
                self._r = np.empty_like(obser_arr)
            self.compute_residual(obser_arr, model_arr, self._r, require_full=False)
            return self._r
        else:
            return obser_arr

    def get_gain_mask(self):
        """Helper method: make a gain array mask from gain flags by broadcasting the corr1/2 axes."""
        mask = np.zeros_like(self.gains, bool)
        mask[:] = (self.gflags!=0)[...,np.newaxis,np.newaxis]
        return mask

    def init_gains(self):
        """
        Construct gain and flag arrays. Normally we have one gain/one flag per interval, but 
        subclasses may redefine this, if they deal with e.g. full-resolution gains.
        
        Sets up the following attributes: 
            gain_shape (list):
                shape of the gains array, i.e. (n_dir, NT, NF, n_ant, n_cor, n_cor)
                Default version has NT=n_timint, NF=n_freint, i.e. one gain per interval.
                 
            gain_grid (dict):
                grid on which gains are defined, as a dict of {'time': times, 'freq': frequencies},
                where times is a vector of NT points and frequencies is a vector of NF points.
                
            gains (np.ndarray):
                gains array of the specified shape
                
            gflags (np.ndarray):
                gain flags array of the same shape, minus the two correlation axes
                
            gain_intervals (list):
                intervals on which gains are defined. Default version has [t_int, f_int].
                
            _gainres_to_fullres (callable):
                function to go from an array of gain shape to full time/freq resolution
                (default uses unpack_intervals)
            
            _interval_to_gainres (callable):
                function to go from an array of interval shape to gain shape (default is identity)
        """
        # Construct the appropriate shape for the gains.
        self.gain_intervals = self.t_int, self.f_int
        self.gain_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, self.n_cor, self.n_cor]
        self.gain_grid = self.interval_grid

        self.gains = self.kernel_solve.allocate_gain_array(self.gain_shape, self.dtype)

        self.gains[:] = np.eye(self.n_cor)
        self.gflags = np.zeros(self.gain_shape[:-2], FL.dtype)

        # function used to unpack the gains or flags into full time/freq resolution
        self._gainres_to_fullres  = self.unpack_intervals
        # function used to unpack interval resolution to gain resolution
        self._interval_to_gainres = self.copy_or_identity

    def compute_residual(self, obser_arr, model_arr, resid_arr, require_full=True):
        gains_h = self.get_conj_gains()

        np.copyto(resid_arr, obser_arr)

        (self.kernel if require_full else self.kernel_solve).compute_residual(model_arr,
                                                                self.gains, gains_h, resid_arr, *self.gain_intervals)

        return resid_arr

    def apply_gains(self, model_arr, full2x2=True, dd_only=False):
        gains_h = self.get_conj_gains()

        (self.kernel if full2x2 else self.kernel_solve).apply_gains(model_arr,
                                        self.gains, gains_h, *self.gain_intervals)

        return model_arr

    def apply_inv_gains(self, obser_arr, corr_vis=None, full2x2=True, direction=None):
        if corr_vis is None:
            corr_vis = np.empty_like(obser_arr)
    
        # parset uses -1 for None, so may as well support it here        
        if direction is not None and direction < 0:
            direction = None

        if self.dd_term and direction is None:
            corr_vis[:] = obser_arr
            return corr_vis, 0

        g_inv, gh_inv, flag_count = self.get_inverse_gains()

        dirslice = slice(0,1) if not self.dd_term else slice(direction, direction+1)
        g_inv = g_inv[dirslice]
        gh_inv = gh_inv[dirslice]

        (self.kernel if full2x2 else self.kernel_solve).compute_corrected(obser_arr,
                                                            g_inv, gh_inv, corr_vis, *self.gain_intervals)

        return corr_vis, flag_count

    @staticmethod
    def exportable_solutions():
        """ Returns a dictionary of exportable solutions for this machine type. """

        return { "gain": (1+0j, ("dir", "time", "freq", "ant", "corr1", "corr2")),
                 "gain.err": (0., ("dir", "time", "freq", "ant", "corr1", "corr2")) }

    def importable_solutions(self, grid0):
        """ Returns a dictionary of importable solutions for this machine type. """
        return { "gain": dict(dir=grid0['dir'] if self.dd_term else [0], ant=grid0['ant'], corr1=grid0['corr'], corr2=grid0['corr'],
                              **self.interval_grid) }

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """
        
        mask = self.get_gain_mask()

        sols = { "gain": (masked_array(self.gains, mask), self.gain_grid) }

        if self.posterior_gain_error is not None:
            sols["gain.err"] = (masked_array(self.posterior_gain_error, mask), self.gain_grid)
        else:
            sols["gain.err"] = (masked_array(np.zeros(self.gain_shape, float), np.ones(self.gain_shape, bool)),
                                self.gain_grid)

        return sols

    def import_solutions(self, soldict):
        """ 
        Loads solutions from a dict. 
        
        Args:
            soldict (dict):
                Contains gains solutions which must be loaded.
        """

        sol = soldict.get("gain")
        if sol is not None:
            self.gains[:] = sol.data
            # gain of all-1 are actually missing from the DB -- replace by unity
            all_ones = (self.gains == 1.+0j).all(axis=(-1, -2)) 
            sol.mask[all_ones,:,:] = True
            self.gains[all_ones, 0, 1] = self.gains[all_ones, 1, 0] = 0
            # flag them as NOSOL if we're not solvable
            if not self.solvable:
                self.gflags[sol.mask.any(axis=(-1,-2))] |= FL.NOSOL
            self._gains_loaded = True

        self.restrict_solution(self.gains)

    def precompute_attributes(self, data_arr, model_arr, flags_arr, inv_var_chan):
        """Precomputes various attributes of the machine before starting a solution"""
        unflagged = MasterMachine.precompute_attributes(self, data_arr, model_arr, flags_arr, inv_var_chan)

        self.missing_gain_fraction = self.missing_intervals_ant.sum() / float(self.missing_intervals_ant.size)

        # number of valid records per time/frequency/antenna (summed over baseline)
        numvis_tfa = unflagged.sum(axis=-1)

        missing_slots = numvis_tfa == 0

        missing_intervals = self.missing_intervals_ant.all(axis=-1)

        if self.solvable and missing_intervals.any() and log.verbosity() > 1:
            self.raise_userwarning(
                logging.WARNING,    # OMS: was CRITICAL but I disagree. Chunks of band/scans are often flagged, why cry wolf about it?
                "{0:s} {1:s}: {2:d}/{3:d} solution intervals are fully flagged".format(
                    self.chunk_label, self.jones_label,
                    missing_intervals.sum(), missing_intervals.size
                ),
                90, raise_once="prior_fully_flagged_dirs", verbosity=2)

        # compute error estimates per direction, antenna, and interval
        if inv_var_chan is not None:
            # collapse direction axis, if not directional
            if not self.dd_term:
                model_arr = model_arr.sum(axis=0, keepdims=True)
            # mean |model|^2 per direction+TFA, normalized by the number of equations/2
            # (a full 2x2 coherency has 8 equations, therefore 4 model components.)
            # TODO: think how this plays out in diag-only scenarios
            with np.errstate(invalid='ignore', divide='ignore'):
                modelsq = (model_arr*np.conj(model_arr)).real.sum(axis=(1,-1,-2,-3)) / \
                      (self.eqs_per_record/2 * numvis_tfa)
                sigmasq = 1.0/inv_var_chan                        # squared noise per channel. Could be infinite if no data
            modelsq[:, missing_slots] = 0
            # take the sigma (in quadrature) over each interval
            # divided by quadrature unflagged contributing interferometers per interval
            # this yields var<g>
            # (numeq_tfa becomes number of unflagged points per interval, antenna)
            numvis_int_ant = self.interval_sum(numvis_tfa)
            sigmasq[np.isnan(sigmasq)|np.isinf(sigmasq)] = 0.0
            modelsq[np.isnan(modelsq)|np.isinf(modelsq)] = 0.0

            # sum-square-model over each interval
            modelsq_sum = self.interval_sum(modelsq, 1)
            # mask of intervals where the model sum is zero or invalid
            invalid_models = (modelsq_sum == 0) | np.isnan(modelsq_sum) | np.isinf(modelsq_sum)

            # if interval was missing (fully flagged), it's not invalid (these will be masked out separately)
            invalid_models[:, self.missing_intervals_ant] = False
            if invalid_models.any():
                self.raise_userwarning(
                    logging.CRITICAL,
                    "{0:s} {1:s}: {2:d}/{3:d} directions/intervals has an invalid or null model. " +
                    "This can be caused by bad data, but should have been handled gracefully, so please report this issue!".format(
                        self.chunk_label, self.jones_label, invalid_models.sum(), invalid_models.size
                    ),
                    90, raise_once="invalid_models", verbosity=2, color="red")

            with np.errstate(invalid='ignore', divide='ignore'):
                NSR_int = self.interval_sum(np.ones_like(modelsq) * (sigmasq)[None, None, :, None], 1) / \
                              (modelsq_sum * numvis_int_ant)
                self.prior_gain_error = np.sqrt(NSR_int)

            # convert that into a gain error per direction,interval,antenna
            # zero the PGEs corresponding to fully-flagged intervals, since we don't want to shout about them
            self.prior_gain_error[:, self.missing_intervals_ant] = 0
            # zero the PGEs corresponding to invalid models, we've already warned about those
            self.prior_gain_error[invalid_models] = 0
            # zero the PGEs for fixed directions
            if self.dd_term:
                self.prior_gain_error[self.fix_directions, ...] = 0

            # if we did everything right above (ignoring flagged data etc.), PGE can't be inf/nan,
            # so this is just a sanity check
            pge_flag_invalid = np.isnan(self.prior_gain_error) | np.isinf(self.prior_gain_error)
            if pge_flag_invalid.any():
                self.raise_userwarning(
                    logging.CRITICAL,
                    "{0:s} {1:s}: prior gain error estimate invalid for {2:d}/{3:d} directions/intervals. " +
                    "This can be caused by bad data, but should have been handled gracefully, so please report this issue!".format(
                        self.chunk_label, self.jones_label, pge_flag_invalid.sum(), pge_flag_invalid.size
                    ),
                    90, raise_once="invalid_models", verbosity=2, color="red")
                # flag these as invalid
                self.gflags[self._interval_to_gainres(pge_flag_invalid, 1)] |= FL.INVALID

            # log(0).print("{}: max gain error {}, invalid PGEs {}".format(self.chunk_label, self.max_gain_error, bad_gain_intervals.sum()))

            if self.max_gain_error:
                low_snr = self.prior_gain_error > self.max_gain_error  # shape DTFA
                self.gflags[self._interval_to_gainres(low_snr, 1)] |= FL.LOWSNR
                self._report_gain_flags(low_snr, "on low SNR", "your max-prior-error settings", self.low_snr_warn)

        # if INVALID or LOWSNR raised above, propagate and recompute equation counts
        # This also propagates the NOSOL flag from missing solutions
        if self._update_gain_flags("INVALID", flags_arr) + \
                self._update_gain_flags("LOWSNR", flags_arr) + \
                self._update_gain_flags("NOSOL", flags_arr):
            unflagged = flags_arr == 0
            self.update_equation_counts(unflagged)

        # if ILLCOND may have been raised due to low equation counts
        if self._update_gain_flags("ILLCOND", flags_arr):
            unflagged = flags_arr == 0

        # gains flagged for any reason
        self.flagged = self.gflags != 0
        self.n_flagged = self.flagged.sum()

        return unflagged

        # # get error estimate model
        # model = np.sqrt(self.interval_sum(modelsq,1))
        # self.model_error = model*self.gain_error*(2+self.gain_error)

    def _report_gain_flags(self, whats_flagged, why_flagged="on low SNR",
                           check_settings="your max-prior-error settings",
                           threshold=0):
        def issue_warning(msg, sort_index=70):
            msg = "{0:s} {1:s}: {2:s}. " \
                  "You may need to check {3:s} and/or solution intervals. " \
                  "New flags will be raised for this chunk of data".format(
                self.chunk_label, self.jones_label, msg, check_settings)
            self.raise_userwarning(logging.WARNING, msg, sort_index, verbosity=log.verbosity())

        if whats_flagged.all():
            issue_warning("all solutions flagged {}".format(why_flagged))

        else:
            # whats_flagged has shape dir, time, freq, ant
            # check for various slices that are all-flagged and issue warnings
            if whats_flagged.all(axis=(0, 1, 3)).any():
                issue_warning("some frequency intervals flagged {}".format(why_flagged))
            if whats_flagged.all(axis=(0, 2, 3)).any():
                issue_warning("some time intervals flagged {}".format(why_flagged))
            # bad stations
            bad_stations = np.argwhere(whats_flagged.all(axis=(0, 1, 2))).flatten()
            if len(bad_stations):
                issue_warning("(faulty?) stations {0:s} ({1:d}/{2:d}) flagged {3:s}".format(
                    ", ".join(map(str, bad_stations)),
                    len(bad_stations), self.n_ant, why_flagged))
            # percentage flagged per direction
            percflagged = whats_flagged.sum(axis=(1, 2, 3)) * 100.0 / whats_flagged[0].size
            bad_dirs = np.argwhere(percflagged > threshold)

            if len(bad_dirs):
                if log.verbosity() > 1:
                    def __extract_dirs(d):
                        if (type(d) is list or type(d) is np.ndarray) and len(d) == 1:
                            return str(d[0])
                        elif (type(d) is list or type(d) is np.ndarray):
                            return ", ".join(d)
                        else:
                            return str(d)

                    def __extract_dirs_perc(d):
                        if (type(d) is list or type(d) is np.ndarray) and len(d) == 1:
                            return "{0:.3f}%".format(percflagged[d[0]])
                        elif (type(d) is list or type(d) is np.ndarray):
                            return ", ".join(["{0:.3f}%".format(percflagged[dd]) for dd in d])
                        else:
                            return "UNDETERMINED"

                    msg = "{} directions ({}) flagged {}".format(
                        len(bad_dirs),
                        ", ".join(["dir {0:s}: {1:s} gains affected".format(
                                  __extract_dirs(d), __extract_dirs_perc(d)) 
                            for d in bad_dirs]),
                        why_flagged)
                else:
                    msg = "{} directions flagged {}. Verbosity level 2 gives a full list of these directions".format(
                        len(bad_dirs), why_flagged)
                issue_warning(msg, 50)

    def _update_gain_flags(self, flagtype, flags_arr):
        """Helper function. Propagates flags given by mask onto data flags (if this is enabled),
        and updates counters. Returns number of gains flagged by this mask."""
        flagmask = getattr(FL, flagtype)
        flagged_gains = ((self.gflags & flagmask) != 0)

        self.num_flagged_on[flagtype] = num_flagged = flagged_gains.sum()

        if num_flagged and self.propagates_flags:
            bad_intervals = flagged_gains.any(axis=0) if self.propagates_anydir_flags else flagged_gains.all(axis=0)
            if bad_intervals.any():
                bad_slots = self.unpack_intervals(bad_intervals)
                bad_slots = bad_slots[:, :, :, np.newaxis]|bad_slots[:, :, np.newaxis, :]
                flags_arr[bad_slots] |= flagmask

        return num_flagged


    def update_equation_counts(self, unflagged):
        """Sets up equation counters based on flagging information. Overrides base version to compute
        additional stuff"""
        MasterMachine.update_equation_counts(self, unflagged)

        # The above computes:
        # Equations per antenna (summed over all time/freq slots)
        #    self.eqs_per_antenna = np.sum(unflagged, axis=(0, 1, 2)) * self.eqs_per_record
        # Total equations per time/freq slot (summed over all baselines)
        #    self.eqs_per_tf_slot = np.sum(unflagged, axis=(-1, -2)) * self.eqs_per_record

        # Equations per antenna and time/freq slot (summed over baselines)
        self.eqs_per_tf_slot_ant  = unflagged.sum(axis=-1) * self.eqs_per_record
        # Equations per antenna and solution interval
        self.eqs_per_interval_ant = self.interval_sum(self.eqs_per_tf_slot_ant)

        # First time 'round, set up missing intervals
        if self.missing_intervals_ant is None:
            # missing interval/antennas -- we can't solve for these, flag them as MISSING
            self.missing_intervals_ant = self.eqs_per_interval_ant == 0
            if self.solvable:
                self.gflags[:, self._interval_to_gainres(self.missing_intervals_ant)] = FL.MISSING

        # Mask of valid time/frequency intervals/antennas (i.e. ones for which we have enough equations)
        # This may be a subset of ~missing_intervals_ant, if some intervals have more than zero but insufficient equations

        self.valid_intervals_ant = self.eqs_per_interval_ant >= self.dof_per_antenna
        self.num_valid_intervals_ant = self.valid_intervals_ant.sum()

        # any interval/antennas that weren't missing (0 equations) to begin with, but don't have equations
        # now are marked ILLCOND
        illcond = ~(self.valid_intervals_ant | (self.eqs_per_interval_ant==0))
        self.gflags[:, self._interval_to_gainres(illcond)] = FL.ILLCOND

        # Now work out the chi-sq normalization factors

        # Total equations per interval
        eqs_per_interval = self.interval_sum(self.eqs_per_tf_slot)
        ndir = self.n_dir - len(self.fix_directions) if self.dd_term else 1
        # Number of unknowns per interval
        unknowns_per_interval = self.dof_per_antenna*self.n_ant*ndir

        valid_intervals = eqs_per_interval > unknowns_per_interval

        if valid_intervals.any():
            # Adjust chi-sq normalisation based on DoF count: MasterMachine computes chi-sq normalization
            # as 1/N_eq, we want to compute it as the reduced chi-square statistic, 1/(N_eq-N_dof)
            # This results in a per-interval correction factor

            with np.errstate(invalid='ignore', divide='ignore'):
                corrfact = eqs_per_interval.astype(float)/(eqs_per_interval - unknowns_per_interval)
            corrfact[~valid_intervals] = 0

            self._chisq_tf_norm_factor = self._chisq_tf_norm_factor_0 * self.unpack_intervals(corrfact)
            self._chisq_norm_factor = self._chisq_norm_factor * corrfact.sum() / valid_intervals.sum()


    def next_iteration(self):
        np.copyto(self.old_gains, self.gains)
        return MasterMachine.next_iteration(self)

    def flag_solutions(self, flags_arr, final=0):
        """ Flags gain solutions based on certain criteria, e.g. out-of-bounds, null, etc. """

        # Anything previously flagged for another reason will not be reflagged.

        flagged = self.flagged
        nfl0 = self.n_flagged

        # flag on OOB and such
        gain_mags = np.abs(self.gains)

        # Check for inf/nan solutions. One bad correlation will trigger flagging for all correlations.

        boom = (~np.isfinite(self.gains)).any(axis=(-1,-2))
        self.gflags[boom&~flagged] |= FL.BOOM
        flagged |= boom
        gain_mags[boom] = 0
        self._update_gain_flags("BOOM", flags_arr)

        # Check for gain solutions for which diagonal terms have gone to 0.

        gnull = (self.gains[..., 0, 0] == 0) | (self.gains[..., 1, 1] == 0)
        self.gflags[gnull&~flagged] |= FL.GNULL
        flagged |= gnull
        self._update_gain_flags("GNULL", flags_arr)

        # Check for gain solutions which are out of bounds (based on clip thresholds).
        if self.clip_after <= self.iters and (self.clip_upper or self.clip_lower):
            goob = np.zeros(gain_mags.shape, bool)
            if self.clip_upper:
                goob = gain_mags.max(axis=(-1, -2)) > self.clip_upper
            if self.clip_lower:
                goob |= (gain_mags[...,0,0]<self.clip_lower) | (gain_mags[...,1,1,]<self.clip_lower)
            self.gflags[goob&~flagged] |= FL.GOOB
            flagged |= goob
            self._update_gain_flags("GOOB", flags_arr)

        # in final (post-solution) flagging, check the posterior error estimate
        if final>0:
            if self.posterior_gain_error is not None and self.max_post_error:
                # reset to 0 for fixed directions
                if self.dd_term:
                    self.posterior_gain_error[self.fix_directions, ...] = 0
                # flag gains on max error
                # the last axis is correlation -- we flag if any correlation is flagged
                bad_gain_intervals = (self.posterior_gain_error > self.max_post_error).any(axis=(-1,-2))  # dir,time,freq,ant

                # mask high-variance gains that are not already otherwise flagged
                pge_flags = mask = self._interval_to_gainres(bad_gain_intervals, 1)&~flagged
                # raise FL.GVAR flag on these gains (and clear on all others!)
                self.gflags &= ~FL.GVAR
                self.gflags[mask] |= FL.GVAR
                flagged[mask] = True

                self._n_flagged_on_max_posterior_error = mask.sum(axis=(1, 2, 3)) if mask.any() else None

                if pge_flags.all(axis=0).all():
                    msg = "'{0:s}' {1:s} All directions flagged by posterior gain variance. This probably indicates significant RFI / outliers"\
                          "You need to check your max-post-error setting and data for selected intervals. "\
                          "New flags will be raised for this chunk of data".format(
                                self.jones_label, self.chunk_label)
                    self.raise_userwarning(logging.CRITICAL, msg, 70, verbosity=log.verbosity(), color="red")

                else:
                    dir_snr = {}
                    for d in range(pge_flags.shape[0]):
                        percflagged = np.sum(pge_flags[d]) * 100.0 / pge_flags[d].size
                        if percflagged > self.high_gain_var_warn and d not in self.fix_directions: dir_snr[d] = percflagged
                    if len(dir_snr) > 0:
                        if log.verbosity() > 2:
                            msg = "Signficiant gain variance in one or more directions of gain '{0:s}' chunk '{1:s}':".format(
                                    self.jones_label, self.chunk_label) +\
                                  "\n{0:s}\n".format("\n".join(["\t direction {0:s}: {1:.3f}% gains affected".format(
                                                        str(d), dir_snr[d]) for d in sorted(dir_snr)])) +\
                                  "Check your setting for max-post-error and your data for this interval. "\
                                  "New flags will be raised for this chunk. "
                        else:
                            msg = "'{0:s}' {1:s} Significant gain variance in directions {2:s}. "\
                                  "Check your data for this interval or raise max-post-error! "\
                                  "New flags will be raised for this chunk.".format(
                                self.jones_label, self.chunk_label, ", ".join(map(str, sorted(dir_snr))))
                        self.raise_userwarning(logging.CRITICAL, msg, 50, verbosity=log.verbosity(), color="red")

                    if pge_flags.all(axis=0).all(axis=0).all(axis=-1).any():
                        msg = "'{0:s}' {1:s} All time of one or more frequency intervals flagged due to gain variance. "\
                              "You need to check your max-post-error and data for this interval. "\
                              "New flags will be raised for this chunk of data".format(
                                    self.jones_label, self.chunk_label)
                        self.raise_userwarning(logging.WARNING, msg, 70, verbosity=log.verbosity())

                    if pge_flags.all(axis=0).all(axis=1).all(axis=-1).any():
                        msg = "'{0:s}' {1:s} All channels of one or more time intervals flagged due to gain variance. "\
                              "You need to check your max-post-error and data for this interval. "\
                              "New flags will be raised for this chunk of data".format(
                                    self.jones_label, self.chunk_label)
                        self.raise_userwarning(logging.WARNING, msg, 70, verbosity=log.verbosity())
                    stationflags = np.argwhere(pge_flags.all(axis=0).all(axis=0).all(axis=0)).flatten()
                    if stationflags.size > 0:
                        msg = "'{0:s}' {1:s} Stations {2:s} ({3:d}/{4:d}) fully flagged due to gain variation. "\
                              "These stations may be faulty or your variation requirements (max-post-error) are not met. "\
                              "New flags will be raised for this chunk of data".format(
                                    self.jones_label, self.chunk_label, ", ".join(map(str, stationflags)),
                                    np.sum(pge_flags.all(axis=0).all(axis=0).all(axis=0)), pge_flags.shape[3])
                        self.raise_userwarning(logging.WARNING, msg, 70, verbosity=log.verbosity())


                # if bad_gain_intervals.any():
                #     # (n_dir,) array showing how many were flagged per direction
                #     self._n_flagged_on_max_error = bad_gain_intervals.sum(axis=(1, 2, 3))
                #     # raised corresponding gain flags
                #     mask = self._interval_to_gainres(bad_gain_intervals, 1)
                #     self.gflags[mask] |= FL.GVAR
                #     flagged[mask] = True
                # else:
                #     self._n_flagged_on_max_posterior_error = None

        # Count the gain flags, excluding those set a priori due to missing data.

        self.flagged = self.gflags != 0
        self.n_flagged = flagged.sum()

        # keep flagged gains at their previous values
        self.gains[self.flagged] = self.old_gains[self.flagged]

        # propagate flags out to data if something new has been flagged, or if final<0 (which means
        # we're applying solutions, so better propagate everything)
        if (self.n_flagged > nfl0 or final<0) and self.propagates_flags:
            # convert gain flags to full time/freq resolution, and add directions together
            nodir_flags = self._gainres_to_fullres(np.bitwise_or.reduce(self.gflags, axis=0))

            # We remove the FL.MISSING bit when propagating as this bit is pre-set for data flagged
            # as PRIOR|MISSING. This prevents every PRIOR but not MISSING flag from becoming MISSING.

            flags_arr |= nodir_flags[:,:,:,np.newaxis]&~FL.MISSING
            flags_arr |= nodir_flags[:,:,np.newaxis,:]&~FL.MISSING

        if self.n_flagged != nfl0:
            self.update_equation_counts(flags_arr == 0)
            return True

        return False


    def num_gain_flags(self, mask=None, final=False):
        return int((self.gflags&(mask or ~FL.MISSING) != 0).sum()), self.gflags.size

    @property
    def dof_per_antenna(self):
        """This property returns the number of real degrees of freedom per antenna, per solution interval"""
        if "diag" in self.update_type:
            dofs = 2
        elif "scalar" in self.update_type:
            dofs = 1
        else:
            dofs = 4
        if "amp" not in self.update_type and "phase" not in self.update_type:
            dofs *= 2

        return dofs

    @property
    def has_valid_solutions(self):
        return (self.gflags==0).any()

    @property
    def num_solutions(self):
        return self.n_tf_ints

    @property
    def num_converged_solutions(self):
        return self.n_cnvgd

    def check_convergence(self, min_delta_g):
        """
        Updates the convergence parameters of the current time-frequency chunk. 

        Args:
            min_delta_g (float):
                Threshold for the minimum change in the gains - convergence criterion.
        """
        
        diff_g = np.square(np.abs(self.old_gains - self.gains))
        diff_g[self.flagged] = 0
        diff_g = diff_g.sum(axis=(-1,-2,-3))
        
        norm_g = np.square(np.abs(self.gains))
        norm_g[self.flagged] = 1
        norm_g = norm_g.sum(axis=(-1,-2,-3))

        norm_diff_g = diff_g/norm_g

        self.max_update = np.sqrt(np.max(diff_g))
        self.n_cnvgd = (norm_diff_g <= min_delta_g**2).sum()
        self._frac_cnvgd = self.n_cnvgd / float(norm_diff_g.size)

    def restrict_solution(self, gains):
        """
        Restricts the solutions by, for example, selecting a reference antenna or taking only the 
        amplitude. 
        """
        # raise flag so updates of G^H and G^-1 are computed
        self._gh_update = self._ghinv_update = True

        if "diag" in self.update_type or "scalar" in self.update_type:
            gains[...,(0,1),(1,0)] = 0

        if "scalar" in self.update_type:
            gains[...,(0,1),(0,1)] = gains[...,(0,1),(0,1)].mean(axis=-1)[...,np.newaxis]

        if "phase" in self.update_type:
            gabs = abs(gains)
            gnull = gabs==0
            with np.errstate(invalid='ignore'):
                gains /= gabs
            gains[gnull] = 0

        if "amp" in self.update_type:
            np.abs(gains, out=gains.real)
            gains.imag[:] = 0
        
        ## explicitly roll back invalid gains to previously known good values
        #self.gains[self.gflags != 0] = self.old_gains[self.gflags != 0]
        
        # explicitly roll back gains to previously known good values for fixed directions
        for idir in self.fix_directions:
            gains[idir, ...] = self.old_gains[idir, ...]
            # This might not be initialized in the load_from case.
            if self.posterior_gain_error is not None:
                self.posterior_gain_error[idir, ...] = 0

    @staticmethod
    def copy_or_identity(array, time_ind=0, out=None):
        """Helper conversion method. Returns array itself, or copies it to out"""
        if out is None:
            return array
        np.copyto(out, array)
        return out

    def unpack_intervals(self, arr, tdim_ind=0, out=None):
        """
        Helper conversion method. Unpacks an array that has time/freq axes in terms of intervals into a shape
        that has time/freq axes at full resolution.

        Args:
            arr (np.ndarray):
                Array with adjacent time and frequency axes.
            tdim_ind (int, optional):
                Position of time axis in array axes. 

        Returns:
            np.ndarray:
                Array at full resolution.
        """
        if out is not None:
            out[:] = arr[tuple([slice(None)] * tdim_ind + [self.t_mapping, self.f_mapping])]
            return out
        else:
            return arr[tuple([slice(None)] * tdim_ind + [self.t_mapping, self.f_mapping])]

    def interval_sum(self, arr, tdim_ind=0, out=None):
        """
        Helper conversion method. Sums an array with full resolution time/freq axes into time/freq intervals.

        Args:
            arr (np.ndarray):
                Array with adjacent time and frequency axes.
            tdim_ind (int, optional):
                Position of time axis in array axes. 

        Returns:
            np.ndarray:
                Array with interval resolution.
        """

        return np.add.reduceat(np.add.reduceat(arr, self.t_bins, tdim_ind), self.f_bins, tdim_ind+1, out=out)

    def interval_and(self, arr, tdim_ind=0, out=None):
        """
        Helper method. Logical-ands an array with full resolution time/freq axes into time/freq 
        intervals.

        Args:
            arr (np.ndarray):
                Array with adjacent time and frequency axes.
            tdim_ind (int, optional):
                Position of time axis in array axes. 

        Returns:
            np.ndarray:
                Array with interval resolution.
        """

        return np.logical_and.reduceat(np.logical_and.reduceat(arr, self.t_bins, tdim_ind), 
                                                                    self.f_bins, tdim_ind+1, out=out)
    @property
    def converged_fraction(self):
        return self._frac_cnvgd

    @property
    def has_converged(self):
        """ Returns convergence status. """
        return self.converged_fraction >= self.min_quorum or self.iters >= self.maxiter

    @has_converged.setter
    def has_converged(self, value):
        if not value:
            self._frac_cnvgd = self.n_cnvgd = 0

    @property
    def conditioning_status_string(self):
        """Returns conditioning status string"""
        if self.solvable:
            mineqs = self.eqs_per_interval_ant[self.valid_intervals_ant].min() if self.num_valid_intervals_ant else 0
            maxeqs = self.eqs_per_interval_ant.max()
            anteqs = (self.eqs_per_antenna!=0).sum()
            string = "{}: {}/{} ints".format(self.jones_label, self.num_valid_intervals_ant, self.n_tf_ints*self.n_ant)
            if self.num_valid_intervals_ant:
                string += " ({}-{} EPA)".format(mineqs, maxeqs)
                if self.dd_term:
                    string += " {} dirs".format(self.n_dir)
                string += " {}/{} ants, MGE {}".format(anteqs, self.n_ant,
                    " ".join(["{:.3}".format(self.prior_gain_error[idir, :].max()) for idir in range(self.n_dir)]))
                flagged = ["{}:{}".format(flagtype, count) for flagtype, count in self.num_flagged_on.items() if count]
                if flagged:
                    string += " (flagged {})".format(" ".join(flagged))

            return string
        else:
            return "{}: n/s".format(self.jones_label)


    @property
    def flagging_stats_string(self):
        """Returns a string describing per-flagset statistics"""
        fstats = []

        for flag, mask in FL.categories().items():
            n_flag = ((self.gflags & mask) != 0).sum()
            if n_flag:
                fstats.append("{}:{}({:.2%})".format(flag, n_flag, n_flag/float(self.gflags.size)))

        return " ".join(fstats)



    @property
    def current_convergence_status_string(self):
        """
        This property must return a status string for the gain machine, e.g.
            "G: 20 iters, conv 60.02%, g/fl 15.00%"
        """
        if self.solvable:
            string = "{}: {} iters, conv {:.2%}".format(self.jones_label, self.iters, self.converged_fraction)
            nfl, ntot = self.num_gain_flags()
            if nfl:
                string += ", g/fl {:.2%} [{}]".format(nfl/float(ntot), self.flagging_stats_string)
            if self.missing_gain_fraction:
                string += ", d/fl {:.2%}".format(self.missing_gain_fraction)
            string += ", max update {:.4}".format(self.max_update)
            if self.posterior_gain_error is not None:
                string += ", PGE " + " ".join(["{:.3}".format(self.posterior_gain_error[idir, :].max())
                                               for idir in range(self.n_dir)])
            return string
        else:
            return "{}: n/s{}".format(self.jones_label, ", loaded" if self._gains_loaded else "")

    @property
    def final_convergence_status_string(self):
        """
        This property must return a status string for the gain machine, e.g.
            "G: 20 iters, conv 60.02%, g/fl 15.00%"
        """
        if self.solvable:
            string = "{}: {} iters, conv {:.2%}".format(self.jones_label, self.iters, self.converged_fraction)
            nfl, ntot = self.num_gain_flags(final=True)
            if nfl:
                string += ", g/fl {:.2%} [{}]".format(nfl/float(ntot), self.flagging_stats_string)
            if self.missing_gain_fraction:
                string += ", d/fl {:.2%}".format(self.missing_gain_fraction)
            if self.posterior_gain_error is not None:
                string += ", PGE " + " ".join(["{:.3}".format(self.posterior_gain_error[idir, :].max())
                                               for idir in range(self.n_dir)])
            flagged = ["{}:{}".format(flagtype, count) for flagtype, count in self.num_flagged_on.items() if count]
            if flagged:
                string += " (flagged {})".format(" ".join(flagged))
            return string
        else:
            return "{}: n/s{}".format(self.jones_label, ", loaded" if self._gains_loaded else "")

    @property
    def has_stalled(self):
        """ Returns stalled status. """

        return self._has_stalled

    @has_stalled.setter
    def has_stalled(self, value):
        """ Sets stalled status. """

        self._has_stalled = value
