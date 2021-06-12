# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from __future__ import print_function
from cubical.machines.parameterised_machine import ParameterisedGains
import numpy as np
from numpy.ma import masked_array
import cubical.kernels
from cubical.tools import logger

log = logger.getLogger("slopes")

def _normalize(x, dtype):
    """
    Helper function: normalizes array to [0,1] interval.
    """

    if len(x) > 1:
        return ((x - x[0]) / (x[-1] - x[0])).astype(dtype)
    elif len(x) == 1:
        return np.zeros(1, dtype)
    else:
        return x


import builtins
try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile


class PhaseSlopeGains(ParameterisedGains):
    """
    This class implements the diagonal phase-only parameterised slope gain machine.
    """

    def __init__(self, label, data_arr, ndir, nmod, chunk_ts, chunk_fs, chunk_label, options):
        """
        Initialises a diagonal phase-slope gain machine.

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
            chunk_ts (np.ndarray):
                Times for the data being processed.
            chunk_fs (np.ndarray):
                Frequencies for the data being processsed.
            options (dict):
                Dictionary of options.
        """
        ParameterisedGains.__init__(self, label, data_arr, ndir, nmod,
                                    chunk_ts, chunk_fs, chunk_label, options)

        self.slope_type = options["type"]
        self.n_param = 3 if self.slope_type == "tf-plane" else 2
        self._estimate_delays = options["estimate-delays"]
        self._pin_slope_iters = options["pin-slope-iters"]

        if self._pin_slope_iters > self.maxiter:
            raise ValueError("Slope pinning iterations is greater than the maximum "
                             "number of iterations. Please check term-iters, max-iters "
                             "and pin-slope-iters.")

        if (self.slope_type != "f-slope") and (self._pin_slope_iters != 0):
            raise ValueError("Slope pinning is not supported for modes other than "
                             "f-slope. Please ensure that pin-slope-iters is zero.")

        self.param_shape = [self.n_dir, self.n_timint, self.n_freint,
                            self.n_ant, self.n_param, self.n_cor, self.n_cor]

        # This needs to change - we want to initialise these slope params
        # from the fourier transform along the relevant axis.

        self.slope_params = np.zeros(self.param_shape, dtype=self.ftype)
        self.posterior_slope_error = None

        self.chunk_ts = _normalize(chunk_ts, self.ftype)
        self.chunk_fs = _normalize(chunk_fs, self.ftype)

        if self.slope_type == "tf-plane":
            self.slope = cubical.kernels.import_kernel("tf_plane")
            self._labels = dict(phase=0, delay=1, rate=2)
        elif self.slope_type == "f-slope":
            self.slope = cubical.kernels.import_kernel("f_slope")
            self._labels = dict(phase=0, delay=1)
            if self._estimate_delays:
                # Select all baselines containing the reference antenna.

                ref_ant_data = data_arr[:, :, :, self.ref_ant]

                # Average over time solution intervals. This should improve SNR,
                # but is not always necesssary.

                interval_data = np.add.reduceat(ref_ant_data, self.t_bins, 1)
                interval_smpl = np.add.reduceat(ref_ant_data != 0, self.t_bins, 1)
                selection = np.where(interval_smpl)
                interval_data[selection] /= interval_smpl[selection]

                bad_bins = np.add.reduceat(interval_data, self.f_bins, 2)

                # FFT the data along the frequency axis. TODO: Need to consider
                # what happens if we solve for few delays across the band. As there
                # is no guarantee that the band will be perfectly split, this may
                # need to be a loop over frequency solution intervals.

                pad_factor = options["delay-estimate-pad-factor"]
                
                for i in range(self.n_freint):
                    edges = self.f_bins + [None]
                    slice_fs = self.chunk_fs[edges[i]:edges[i+1]]

                    slice_data = interval_data[:, :, edges[i]:edges[i+1]]

                    slice_nchan = slice_data.shape[2]

                    fft_data = np.abs(np.fft.fft(slice_data, n=slice_nchan*pad_factor, axis=2))
                    fft_data = np.fft.fftshift(fft_data, axes=2)

                    # Convert the normalised frequency values into delay values.

                    delta_freq = slice_fs[1] - slice_fs[0]
                    fft_freq = np.fft.fftfreq(slice_nchan*pad_factor, delta_freq)
                    fft_freq = np.fft.fftshift(fft_freq)

                    # Find the delay value at which the FFT of the data is
                    # maximised. As we do not pad the values, this only a rough
                    # approximation of the delay. We also reintroduce the
                    # frequency axis for consistency.

                    delay_est_ind = np.argmax(fft_data, axis=2)
                    delay_est_ind = np.expand_dims(delay_est_ind, axis=2)

                    delay_est = fft_freq[delay_est_ind]
                    delay_est[...,(0,1),(1,0)] = 0
                    
                    # Check for bad data points (bls missing across all channels)
                    # and set their estimates to zero.

                    selection = np.where(bad_bins[:, :, i:i+1] == 0)
                    delay_est[selection] = 0

                    # Zero the off diagonals and take the negative delay values -
                    # this is necessary as we technically get the delay
                    # corresponding to the conjugate term.

                    self.slope_params[..., i:i+1, :, 1, :, :] = -1*delay_est
                    self.slope_params[..., (1,0), (0,1)] = 0
                    
                    log(1).print("{}: slope estimates are {}, {}".format(chunk_label,
                                        " ".join(map(str, self.slope_params[..., i:i+1, :, 1, 0, 0])),
                                        " ".join(map(str, self.slope_params[..., i:i+1, :, 1, 1, 1]))
                                         ))


        elif self.slope_type == "t-slope":
            self.slope = cubical.kernels.import_kernel("t_slope")
            self._labels = dict(phase=0, rate=1)
        else:
            raise RuntimeError("unknown machine type '{}'".format(self.slope_type))

        self.slope.construct_gains(self.slope_params, self.gains,
                                   self.chunk_ts, self.chunk_fs, self.t_int,
                                   self.f_int)

        # kernel used in solver is diag-diag in diag mode, else uses full kernel version
        if options.get('diag-data') or options.get('diag-only'):
            self.kernel_solve = cubical.kernels.import_kernel('diag_phase_only')
        else:
            self.kernel_solve = cubical.kernels.import_kernel('phase_only')

        self._jhr0 = self._gerr = None

    @property
    def has_converged(self):
        """ Returns convergence status. """

        condition_1 = self.converged_fraction >= self.min_quorum
        condition_2 = self.iters >= self.maxiter
        condition_3 = self.iters > self._pin_slope_iters

        return (condition_1 or condition_2) and condition_3

    @classmethod
    def determine_diagonality(cls, options):
        return True

    @classmethod
    def get_full_kernel(cls, options, diag_gains):
        from .phase_diag_machine import PhaseDiagGains
        return PhaseDiagGains.get_full_kernel(options, diag_gains)

    @staticmethod
    def exportable_solutions():
        """ Returns a dictionary of exportable solutions for this machine type. """

        exportables = ParameterisedGains.exportable_solutions()

        exportables.update({
            "phase": (0., ("dir", "time", "freq", "ant", "corr")),
            "delay":  (0., ("dir", "time", "freq", "ant", "corr")),
            "rate":   (0., ("dir", "time", "freq", "ant", "corr")),
            "phase.err": (0., ("dir", "time", "freq", "ant", "corr")),
            "delay.err": (0., ("dir", "time", "freq", "ant", "corr")),
            "rate.err": (0., ("dir", "time", "freq", "ant", "corr")),
        })

        return exportables

    def get_inverse_gains(self):
        """Returns inverse gains and inverse conjugate gains. For phase-only, conjugation is inverse"""
        gh = self.get_conj_gains()
        return gh, self.gains, 0

    def importable_solutions(self, grid0):
        """ Returns a dictionary of importable solutions for this machine type. """
        # Note that complex gain (as a derived parameter) is exported, but not imported
        # defines solutions we can import from
        return { label: dict(dir=grid0['dir'] if self.dd_term else [0], ant=grid0['ant'], corr=grid0['corr'], **self.interval_grid)
                 for label in self._labels.keys() }

    def export_solutions(self):
        """ Saves the solutions to a dict of {label: solutions,grids} items. """

        solutions = ParameterisedGains.export_solutions(self)

        for label, num in self._labels.items():
            solutions[label] = masked_array(self.slope_params[...,num,(0,1),(0,1)]), self.interval_grid
            if self.posterior_slope_error is not None:
                solutions[label+".err"] = masked_array(self.posterior_slope_error[..., num, :]), self.interval_grid

        
        with np.printoptions(precision=3, linewidth=100000, threshold=10000): 
            log(1).print("{}: slope solutions are {}, {}".format(self.chunk_label,
                            self.slope_params[..., :, :, 1, 0, 0],
                            self.slope_params[..., :, :, 1, 1, 1]))

        return solutions

    def import_solutions(self, soldict):
        """
        Loads solutions from a dict.

        Args:
            soldict (dict):
                Contains gains solutions which must be loaded.
        """

        # Note that this is inherently very flexible. For example, we can init from a solutions
        # table which only has a "phase" entry, e.g. one generated by a phase_only solver (and the
        # delay will then be left at zero).

        loaded = False
        for label, num in self._labels.items():
            value = soldict.get(label)
            if value is not None:
                self.slope_params[...,num,(0,1),(0,1)] = value
                loaded = True

        if loaded:
            self.restrict_solution(self.slope_params)
            self.slope.construct_gains(self.slope_params, self.gains,
                                           self.chunk_ts, self.chunk_fs, self.t_int, self.f_int)
            self._gains_loaded = True

    def compute_js(self, obser_arr, model_arr):
        """
        This function computes the J\ :sup:`H`\R term of the GN/LM method.

        Args:
            obser_arr (np.ndarray):
                Shape (n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the
                observed visibilities.
            model_arr (np.ndrray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing the
                model visibilities.

        Returns:
            np.ndarray:
                J\ :sup:`H`\R
        """

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = self.gains.shape

        gh = self.get_conj_gains()
        jh = self.get_new_jh(model_arr)

        self.slope.compute_jh(model_arr, self.gains, jh, 1, 1)

        jhr1 = self.get_new_jhr()

        r = self.get_obs_or_res(obser_arr, model_arr)

        # use appropriate phase-only kernel (with 1,1 intervals) to compute inner JHR
        self.kernel_solve.compute_jhr(gh, jh, r, jhr1, 1, 1)

        jhr1 = jhr1.imag

        jhr_shape = [n_dir, self.n_timint, self.n_freint, n_ant, self.n_param, n_cor, n_cor]

        if self._jhr0 is None:
            self._jhr0 = self.slope.allocate_param_array(jhr_shape, dtype=jhr1.dtype, zeros=True)
        else:
            self._jhr0.fill(0)

        self.slope.compute_jhr(jhr1, self._jhr0, self.chunk_ts, self.chunk_fs, self.t_int, self.f_int)

        return self._jhr0, self.jhjinv, 0

    @property
    def dof_per_antenna(self):
        """This property returns the number of real degrees of freedom per antenna, per solution interval"""
        if self.slope_type=="tf-plane":
            return 6
        elif self.slope_type=="f-slope":
            return 4
        elif self.slope_type=="t-slope":
            return 4

    def implement_update(self, jhr, jhjinv):

        # variance of slope parms is diagonal of jhjinv
        diag = (0,2) if self.n_param == 2 else (0,3,5)   # weird numbering to get diagonal elements
        var_slope = jhjinv[...,(0,1),(0,1)].real[...,diag,:]
        if self.posterior_slope_error is None:
            self.posterior_slope_error = np.sqrt(var_slope)
        else:
            np.sqrt(var_slope, out=self.posterior_slope_error)
            
        #import ipdb; ipdb.set_trace()
        with np.printoptions(precision=3, linewidth=100000, threshold=10000): 
            log(1).print("Iteration {}".format(self.iters))
            log(1).print("slope variance X: ", var_slope[0,:,0,:,1,0].max(axis=0))
            log(1).print("slope variance Y: ", var_slope[0,:,0,:,1,1].max(axis=0))
            
        # variance of gain is sum of slope parameter variances
        if self._gerr is None:
            self._gerr = var_slope.sum(axis=-2)
        else:
            var_slope.sum(axis=-2, out=self._gerr)

        np.sqrt(self._gerr, out=self._gerr)
        if self.posterior_gain_error is None:
            self.posterior_gain_error = np.zeros_like(self.gains, dtype=float)
        self._interval_to_gainres(self._gerr[...,0], 1, out=self.posterior_gain_error[...,0,0])
        self._interval_to_gainres(self._gerr[...,0], 1, out=self.posterior_gain_error[...,1,1])

        update = self.init_update(jhr)

        self.slope.compute_update(jhr, jhjinv, update)
        
        with np.printoptions(precision=3, linewidth=100000, threshold=10000): 
            log(1).print("proposed slope update X: ", update[0,:,0,:,1,0,0].max(axis=0))
            log(1).print("proposed slope update Y: ", update[0,:,0,:,1,1,1].max(axis=0))
            log(1).print("proposed phase update X (deg): ", update[0,:,0,:,0,0,0].max(axis=0)*180/np.pi)
            log(1).print("proposed phase update Y (deg): ", update[0,:,0,:,0,1,1].max(axis=0)*180/np.pi)

        # Pin the delay component if self._pin_slope_iters is set. Note that
        # this is likely incorrect for delay+rate calibration.
        if self.iters < self._pin_slope_iters:
            update[..., 1, :, :] = 0

        # It is safer to average every iteration when using a phase only solver.
        if self.iters%1 == 0:
            update *= 0.5
        
        # suppress slope updates for the first 10 iterations, to let phase settle    
        #if self.iters < 10:
        #    update *= 0.01
        #    log(1).print("update damped")

        self.slope_params += update

        self.restrict_solution(self.slope_params)

        # Need to turn updated parameters into gains.

        self.slope.construct_gains(self.slope_params, self.gains, self.chunk_ts, self.chunk_fs, self.t_int, self.f_int)

        # raise flag so updates of G^H and G^-1 are computed
        self._gh_update = self._ghinv_update = True


    def restrict_solution(self, slope_params):
        """
        Restricts the solution by invoking the inherited restrict_solution method and applying
        any machine specific restrictions.
        """

        # ParameterisedGains.restrict_solution(self)
        # These need to be set here as we do not call the interval restrict solutions -
        # out solutions are the parameters and not the gains themselves.
        self._gh_update = self._ghinv_update = True

        if self.ref_ant is not None:
            # complicated slice :) we take the 0,0 phase offset of the reference antenna,
            # and subtract that from the phases of all other antennas and elements
            ref_slice = slice(self.ref_ant, self.ref_ant+1, 1)
            ref_params = slope_params[:,:,:,ref_slice,:,0,0]
            slope_params[:,:,:,:,:,(0,1),(0,1)] -= ref_params[..., None]

        for idir in self.fix_directions:
            slope_params[idir, ...] = 0

    def precompute_attributes(self, data_arr, model_arr, flags_arr, inv_var_chan):
        """
        Precompute (J\ :sup:`H`\J)\ :sup:`-1`, which does not vary with iteration.

        Args:
            model_arr (np.ndarray):
                Shape (n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) array containing
                model visibilities.
        """
        ParameterisedGains.precompute_attributes(self, data_arr, model_arr, flags_arr, inv_var_chan)

        jhj1_shape = [self.n_dir, self.n_tim, self.n_fre, self.n_ant, 2, 2]

        jhj1 = self.slope.allocate_gain_array(jhj1_shape, dtype=self.dtype, zeros=True)

        # use appropriate phase-only kernel (with 1,1 intervals) to compute inner JHJ
        self.kernel_solve.compute_jhj(model_arr, jhj1, 1, 1)

        blocks_per_inverse = 6 if self.slope_type=="tf-plane" else 3

        jhj_shape = [self.n_dir, self.n_timint, self.n_freint, self.n_ant, blocks_per_inverse, 2, 2]

        jhj = self.slope.allocate_param_array(jhj_shape, dtype=self.ftype, zeros=True)

        self.slope.compute_jhj(jhj1.real, jhj, self.chunk_ts, self.chunk_fs, self.t_int, self.f_int)

        self.jhjinv = np.zeros_like(jhj)

        self.slope.compute_jhjinv(jhj, self.jhjinv, self.eps)

