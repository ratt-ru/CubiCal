# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from __future__ import print_function
import numpy as np
from numpy.ma import masked_array

from cubical import param_db

from cubical.tools import logger, ModColor
log = logger.getLogger("gain_machine")


class IfrGainMachine(object):
    """
    Interferometer-based gain machine.
    
    Note that this class is outside of the normal gain machine hierarchy. It wraps all the 
    functionality required to manage IFR-based gain (aka baseline-based corrections) 
    calculations.
    """

    def __init__(self, gmfactory, ifrgain_opts, compute=True):
        """
        Initializes the IFR-based gains machinery.
        
        Args:
            gmfactory:      a GainMachine Factory is used to manage the solution databases
            ifrgain_opts:   dict of options
            compute:        if False, gains are not computed even if options ask them to
        """
        from cubical.main import expand_templated_name
        self.gmfactory = gmfactory
        load_from = expand_templated_name(ifrgain_opts['load-from'])
        save_to = expand_templated_name(ifrgain_opts['save-to'])
        self._ifrgains_per_chan = ifrgain_opts['per-chan']
        self._ifrgain = None
        self._nfreq = gmfactory.grid["freq"]
        nfreq, nant, ncorr = [len(gmfactory.grid[axis]) for axis in ("freq", "ant", "corr")]
        if load_from:
            filename = load_from
            print(ModColor.Str("applying baseline-based corrections (BBCs) from {}".format(filename),
                                         col="green"), file=log(0))
            if "//" in filename:
                filename, prefix = filename.rsplit("//", 1)
            else:
                filename, prefix = filename, "BBC"
            parm = param_db.load(filename).get(prefix)
            if parm is None:
                print(ModColor.Str("  no solutions for '{}' in {}".format(prefix, filename)), file=log(0))
            else:
                self._ifrgain = parm.reinterpolate(freq=gmfactory.grid["freq"]).filled()
                if tuple(self._ifrgain.shape) != (nfreq, nant, nant, ncorr, ncorr):
                    print(ModColor.Str("  invalid BBC shape {}, will ignore".format(self._ifrgain.shape)), file=log(0))
                    self._ifrgain = None
                else:
                    print("  loaded per-channel BBCs of shape {}".format(filename, self._ifrgain.shape), file=log(0))
                    if not self._ifrgains_per_chan:
                        print("  using one mean value across band", file=log(0))
                        self._ifrgain[np.newaxis,...] = self._ifrgain.mean(axis=0)
                    # reset off-diagonal values, if needed
                    if ifrgain_opts["apply-2x2"]:
                        print(ModColor.Str(
                            "  using full 2x2 BBCs. You'd better know what you're doing!",col="green"), file=log(0))
                    else:
                        self._ifrgain[..., (0, 1), (1, 0)] = 1
                        print("  using parallel-hand BBCs only", file=log(0))
        if save_to and compute:
            self._compute_2x2 = ifrgain_opts["compute-2x2"]
            # setup axes for IFR-based gains
            axes = ["freq", "ant1", "ant2", "corr1", "corr2"]
            # define the ifrgain parameter
            self._save_filename = save_to
            parm = gmfactory.define_param(self._save_filename, "BBC", 1+0j, axes, interpolation_axes=["freq"])
            self._ifrgains_grid = {axis: parm.grid[i] for i, axis in enumerate(axes)}
            # initialize accumulators for M.D^H and D.D^H terms
            self._mdh_sum = np.ma.zeros(parm.shape, gmfactory.ctype, fill_value=0)
            self._ddh_sum = np.ma.zeros(parm.shape, gmfactory.ctype, fill_value=0)
            #
            print("will compute & save suggested baseline-based corrections (BBCs) to {}".format(
                            self._save_filename), file=log(0))
            print("  (these can optionally be applied in a subsequent CubiCal run)", file=log(0))
        else:
            self._ifrgains_grid = None

    def is_applying(self):
        """Returns True if we have valid IFR gains to apply"""
        return self._ifrgain is not None

    def is_computing(self):
        """Returns True if we're computing output IFR gains"""
        return self._ifrgains_grid is not None

    def apply(self, obser_arr, freq_slice):
        """
        Applies IFR-based gains (if any) to input data
        
        Args:
            obser_arr:   data array (shape ntime,nfreq,nant,nant,ncorr,ncorr)
            freq_slice: frequency slice in data array
        """
        # IFR gains are freq,ant,ant,corr,corr, so broadcast the time axes, and slice the freq axis
        if self._ifrgain is not None:
            obser_arr *= self._ifrgain[np.newaxis, freq_slice, ...]

    def update(self, obser_arr, corrupt_model, flags, freq_slice, soldict):
        """
        Computes update to interferometer gains based on data and corrupt model. Saves this to 
        SharedDict of solutions.

        Args:
            obser_arr:      data array (nmod,ntime,nfreq,nant,nant,ncorr,ncorr) 
            corrupt_model:  corrupt model array (nmod,ntime,nfreq,nant,nant,ncorr,ncorr)
            flags:          flags array, same shape as data
            freq_slice:     frequency slice which is being updated
            soldict:        shared dict to save gain update to
        """
        if not self.is_computing():
            return
        # shape of BB arrays is nfreq,nant,nant,ncorr,ncorr or nant,nant,ncorr,ncorr
        bbshape = obser_arr.shape[-5:]
        DDH = soldict.addSharedArray('ifrgains:DDH__', bbshape, dtype=obser_arr.dtype)
        MDH = soldict.addSharedArray('ifrgains:MDH__', bbshape, dtype=obser_arr.dtype)
        fl = soldict.addSharedArray('ifrgains:flags__', bbshape[:-2], dtype=bool)
        soldict["ifrgains:freqslice__"] = freq_slice

        D, M = obser_arr, corrupt_model
        DH = np.conj(D)
        DDH[:] = (D * DH).sum(axis=(0,1))
        MDH[:] = (M * DH).sum(axis=(0,1))
        fl[:]  = flags.all(axis=0)

    def accumulate(self, subdict):
        """
        Gets solutions from SharedDict populated by update(), and updates internal accumulators
        based on this. 
        Args:
            subdict: dict populated by update() above

        Returns:

        """
        # update ifr gains
        if not self.is_computing():
            return
        ddh = subdict['ifrgains:DDH__']
        # promote corr1,corr2 axes in flags, to make them into a proper mask
        fl = np.empty(ddh.shape, bool)
        fl[:] = subdict['ifrgains:flags__'][...,np.newaxis,np.newaxis]
        ddh = masked_array(ddh, fl, fill_value=0)
        mdh = masked_array(subdict['ifrgains:MDH__'], fl, fill_value=0)
        self._ddh_sum[subdict['ifrgains:freqslice__'],...] += ddh
        self._mdh_sum[subdict['ifrgains:freqslice__'],...] += mdh

    def save(self):
        """
        Finalizes the IFR gain solutions, and saves them to the database  
        """
        if not self.is_computing():
            return
        ## if not per-channel, then sum over channels
        if not self._ifrgains_per_chan:
            self._ddh_sum = self._ddh_sum.sum(0)
            self._mdh_sum = self._mdh_sum.sum(0)

        # mask out null values
        self._ddh_sum.mask |= (self._ddh_sum==0)
        with np.errstate(under='ignore'): # work around https://github.com/numpy/numpy/issues/4895
            ifrgain = self._mdh_sum/self._ddh_sum
        ifrgain.data[self._ddh_sum.mask] = 1
        ifrgain.fill_value = 1

        # promote freq axis back out
        if not self._ifrgains_per_chan:
            ifrgain = np.ma.vstack([ifrgain]*len(self._nfreq))

        # multiply by previous IFR gain, if that was set. This will promote a frequency axis
        if self._ifrgain is not None:
            ifrgain *= self._ifrgain

        # if not 2x2, then reset off-diagonals
        if not self._compute_2x2:
            ifrgain[...,0,1] = ifrgain[...,1,0] = 1

        self.gmfactory.get_solution_db("BBC").add_chunk("BBC", ifrgain, self._ifrgains_grid)

    def reload(self):
        """Reloads saved IFR gain solutions from database"""
        return param_db.load(self._save_filename)["BBC"].get_cube()