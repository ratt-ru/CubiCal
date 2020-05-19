# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Handles solver statistics.
"""
from __future__ import print_function
from builtins import range
import math
import numpy as np
from future.moves import pickle

from cubical.tools import logger
from cubical.tools import ModColor
from collections import OrderedDict

log = logger.getLogger("stats")

class SolverStats (object):
    """
    SolverStats is a container for various per-chunk statistics collected during the solving 
    process.
    """
    def __init__ (self, obj):
        """
        Initialisation for the SolverStats object.

        Args:
            obj (dict or np.ndarray or file):
                Object from which to initialise the stats object.

        Raises:
            TypeError:
                If obj is not an understood type.
        """

        if type(obj) is np.ndarray:
            self._init_for_chunk(obj)
        elif type(obj) is dict:
            self._concatenate(obj)
        elif type(obj) is file:
            self.load(obj)
        else:
            raise TypeError("can't init SolverStats from object of type %s" % type(obj))

    def _init_for_chunk (self, data):
        """
        Initializes stats for a chunk of data.

        Args:
            data (np.ndarray):
                A CubiCal data block.
        """

        n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = data.shape
        # summary record arrays (per channel-antenna, time-antenna, time-channel)
        dtype = [ ('dv2', 'f8'), ('dr2', 'f8'), ('dv2n', 'i4'), ('dr2n', 'i4'),
                  ('chi2', 'f8'), ('chi2n', 'i4'),
                  ('initchi2', 'f8'), ('initchi2n', 'i4') ]
        self.chanant  = np.rec.array(np.zeros((n_fre, n_ant), dtype))
        self.timeant  = np.rec.array(np.zeros((n_tim, n_ant), dtype))
        self.timechan = np.rec.array(np.zeros((n_tim, n_fre), dtype))

        # other stats: per chunk

        # these are truly per chunk, corresponding to starting and final values
        dtype = [ ('label', 'S32'), ('num_prior_flagged', 'i8'), ('num_data_points', 'i8') ]

        # these have intermediate values as well (e.g. when solving for a chain)
        self._chunk_stats_intermediate_fields = [
                    ('chi2u', 'f8'), ('noise', 'f8'), ('chi2', 'f8'),
                    ('iters', 'i4'),
                    ('num_solutions', 'i4'), ('num_converged', 'i4'), ('num_stalled', 'i4'),
                    ('num_sol_flagged', 'i4'), ('num_mad_flagged', 'i4'),
                    ('frac_converged', 'f8'), ('frac_stalled', 'f8'),
                    ('end_chi2', 'f8') ]

        dtype += self._chunk_stats_intermediate_fields

        self._max_intermediate_fields = 100
        for i in range(self._max_intermediate_fields):
            dtype += [("{}_{}".format(field, i), dt) for field, dt in self._chunk_stats_intermediate_fields]

        self.chunk = np.rec.array(np.zeros((), dtype))

    def save_chunk_stats(self, step):
        for key, _ in self._chunk_stats_intermediate_fields:
            self.chunk["{}_{}".format(key, step)] = self.chunk[key]

    def save(self, filename):
        """
        Pickles contents to file. Better than pickling whole object, as the pickle then only 
        contains standard classes (i.e. don't need CubiCal to read it).

        Args:
            filename (str):
                Name for pickled file.
        """
        with open(filename, 'wb') as pf:
            pickle.dump((self.chanant, self.timeant, self.timechan, self.chunk), pf, 2)

    def load(self, fileobj):
        """
        Loads contents from file object
        """

        self.chanant, self.timeant, self.timechan, self.chunk = pickle.load(fileobj)

    def estimate_noise (self, data, flags, residuals=False):
        """
        Given a data cube with dimensions (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) and a flag cube
        with dimensions (n_tim, n_fre, n_ant, n_ant), this function estimates the noise in the data 
        by taking the differences between adjacent channels.

        Sum of delta-visibilities squared, and sum of valid terms, is stored in the 'dv2', 'dv2n' or
        'dr2', 'dr2n' summary fields (the latter if residuals=True).

        Args:
            data (np.ndarray):
                A CubiCal data block.
            flags (np.ndarray):
                A the flag block assosciated with data.
            residuals (bool, optional):
                If True, results are stored in the 'dr2' and 'dr2n' fields.

        Returns:
            tuple:
                Noise, inverse_noise_per_antenna_channel_squared, inverse_noise_per_antenna_squared 
                and inverse_noise_per_channel_squared. 
        """

        n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = data.shape

        # For only one frequency channel, we can't estimate noise -- return 1.

        if n_fre == 1:
            return 1., np.ones((n_fre, n_ant), np.float32), np.ones(n_ant, np.float32), np.ones(n_fre, np.float32)

        deltaflags = flags!=0
        deltaflags[:, 1:, ...] = deltaflags[:, 1:, ...] | deltaflags[:, :-1, ...]
        deltaflags[:, 0 , ...] = deltaflags[:,   1, ...]

        # Create array for the squared difference between channel-adjacent visibilities.

        deltavis2 = np.zeros((n_tim, n_fre, n_ant, n_ant), np.float32)

        # Square the absolute value of the difference between channel-adjacent visibilities and sum
        # over correlations. Normalize the result by n_cor*n_cor*4. The factor of 4 arises because
        # Var<c1-c2> = Var<c1>+Var<c2> and Var<c>=Var<r>+Var<i>. Thus, the square of the abs 
        # difference between two complex visibilities has contributions from _four_ noise terms.

        # TODO: When fewer than 4 correlations are provided, the normalisation needs to be 
        # different.

        # TODO: Something smart with multiple-model calibration. Currently only the first dataset
        # is used.

        deltavis2[:, 1:, ...]  = np.square(abs(data[0, :, 1:, ...] - data[0, :, :-1, ...])).sum(axis=(-2,-1))
        deltavis2[:, 1:, ...] /= n_cor*n_cor*4
        deltavis2[:, 0 , ...]  = deltavis2[:, 1, ...]

        # The flagged elements are zeroed; we don't have an adequate noise estimate for those
        # channels.

        deltavis2[deltaflags] = 0

        # This flag inversion gives a count of the valid estimates in deltavis2.

        deltaflags = ~deltaflags

        # Sum into the various stats arrays (use dvis2 or dvis2p field, depending on whether 
        # pre- or post-solver noise is being estimated)

        dv2attr = 'dr2'   if residuals else 'dv2'
        dn2attr = 'dr2n'  if residuals else 'dv2n'

        deltavis2_chan_ant = getattr(self.chanant,  dv2attr)[...] = deltavis2.sum(axis=(0, 2))  # sum, per chan, ant
        getattr(self.timeant,  dv2attr)[...] = deltavis2.sum(axis=(1, 2))  # sum, per chan, time
        getattr(self.timechan, dv2attr)[...] = deltavis2.sum(axis=(2, 3))  # sum, per time, chan
        deltanum_chan_ant = getattr(self.chanant,  dn2attr)[...] = deltaflags.sum(axis=(0, 2))  # sum, per chan, ant
        getattr(self.timeant,  dn2attr)[...] = deltaflags.sum(axis=(1, 2))  # sum, per chan, time
        getattr(self.timechan, dn2attr)[...] = deltaflags.sum(axis=(2, 3))  # sum, per time, chan

        # Compute the variance overall, and per antenna, and per channel.

        with np.errstate(divide='ignore', invalid='ignore'):  # ignore division by 0
            inv_var = deltanum_chan_ant.sum() / deltavis2_chan_ant.sum()
            noise_est = math.sqrt(1/inv_var)
            inv_var_antchan =  deltavis2_chan_ant / deltanum_chan_ant
            inv_var_ant  = deltanum_chan_ant.sum(axis=0) / deltavis2_chan_ant.sum(axis=0)
            inv_var_chan = deltanum_chan_ant.sum(axis=1) / deltavis2_chan_ant.sum(axis=1)

        # Isolated but valid channels may end up with no noise estimate at all. Fill one in from noise_est
        validchans = (flags==0).sum(axis=(0,2,3)) != 0
        inv_var_chan[validchans&~np.isfinite(inv_var_chan)] = inv_var

        # Antennas/channels with no data end up with NaNs here, so replace them with 0.
        
        inv_var_antchan[~np.isfinite(inv_var_antchan)] = 0
        inv_var_ant[~np.isfinite(inv_var_ant)] = 0
        inv_var_chan[~np.isfinite(inv_var_chan)] = 0
        
        return noise_est, inv_var_antchan, inv_var_ant, inv_var_chan

    @staticmethod
    def add_records(recarray, recarray2):
        """ Adds two record-type arrays together. """
        
        for field in recarray.dtype.fields.keys():
            recarray[field] += recarray2[field]

    @staticmethod
    def normalize_records(recarray):
        """ Normalizes record-type arrays by dividing each field 'X' by the field 'Xn'. """

        for field in recarray.dtype.fields.keys():
            if field[-1] != 'n':
                nval = recarray[field+'n']
                mask = nval!=0
                recarray[field][mask] /= nval[mask]

        return np.rec.array(recarray)

    def _concatenate(self, stats):
        """
        Concatenates stats from a dictionary (indexed by time_index,freq_index) into a single 
        object.

        Args:
            stats (dict):
                A dictionary of useful solver statistics.
        """
        
        # Get lists of unique time and channel indices occurring in the dict.
        
        times = sorted(set([time for time, _ in stats.keys()]))
        chans = sorted(set([chan for _, chan in stats.keys()]))

        # Concatenate and add up cumulative stats.
        
        self.chanant = np.concatenate([stats[times[0], chan].chanant for chan in chans], axis=0)

        for time in times[1:]:
            self.add_records(
                self.chanant, np.concatenate([stats[time, chan].chanant for chan in chans], axis=0))

        self.timeant = np.concatenate([stats[time, chans[0]].timeant for time in times], axis=0)

        for chan in chans[1:]:
            self.add_records(self.timeant, 
                np.concatenate([stats[time, chan].timeant for time in times], axis=0))

        self.timechan = \
            np.concatenate([np.concatenate([stats[time, chan].timechan for time in times], axis=0)
                for chan in chans], axis=1)

        # Note that for some reason np.concatenate of record arrays produces structured arrays 
        # instead of nd.recarrays - normalize_records() will convert them back.

        # Normalize by number of values.
        
        self.chanant = self.normalize_records(self.chanant)
        self.timeant = self.normalize_records(self.timeant)
        self.timechan = self.normalize_records(self.timechan)

        # Make 2D array of per-chunk values.

        self.chunk = np.rec.array([[stats[time, chan].chunk for chan in chans] for time in times],
                                  dtype=stats[times[0], chans[0]].chunk.dtype)

    def get_notrivial_chunk_statfields(self):
        """Returns list of interesting (i.e. non-0) stat fields"""
        return [field for field in self.chunk.dtype.names if field != "label" and (self.chunk[field]!=0).any()]

    def format_chunk_stats(self, format_string, ncol=8, threshold=None):
        """
        :param format: format string applied to each record
        :param maxcol: maximum number of columns to allocate
        :return:
        """
        nt, nf = self.chunk.shape
        nt_per_col = 1
        nf_per_col = None
        if nf < ncol:
            nt_per_col = ncol//nf
        else:
            nf_per_col = ncol
        # convert stats to list of columns
        output_rows  = [[("", False)]]
        for itime in range(nt):
            # start new line every NT_PER_COL-th time chunk
            if itime%nt_per_col == 0:
                output_rows.append([])
            for ifreq in range(nf):
                # start new line every NF_PER_COL-th freq chunk, if frequencies span lines
                if nf_per_col is not None and output_rows[-1] and ifreq%nf_per_col == 0:
                    output_rows.append([])
                statrec = self.chunk[itime, ifreq]
                statrec_dict = {field:statrec[field] for field in self.chunk.dtype.fields}
                # new line: prepend chunk label
                label = statrec.label.decode() if hasattr(statrec.label, 'decode') else statrec.label
                if not output_rows[-1]:
                    output_rows[-1].append((label, False))
                # put it in header as well
                if len(output_rows) == 2:
                    output_rows[0].append((label, False))
                # check for threshold
                warn = False
                if threshold is not None:
                    for field, value in threshold:
                        if statrec[field] > value:
                            warn = True
                text = format_string.format(**statrec_dict)
                output_rows[-1].append((text, warn))

        # now work out column widths and format
        ncol = max([len(row) for row in output_rows])
        colwidths = [max([len(row[icol][0]) for row in output_rows if icol<len(row)]) for icol in range(ncol)]
        colformat = ["{{:{}}}  ".format(w) for w in colwidths]

        output_rows = [[(colformat[icol].format(col), warn) for icol, (col, warn) in enumerate(row)] for row in output_rows]

        return ["".join([(ModColor.Str(col, 'red') if warn else col) for col, warn in row]) for row in output_rows]

    def apply_flagcube(self, flag3):
        """
        Applies additional flag cube to statistics. Basically, if something is flagged in the 
        output based on chi-sq or other criteria, we want to remove it from the stats.

        Args:
            flag3 (np.ndarray):
                An (n_times, n_ddid, n_chan) block of flags.
        """

        # Out stats are n_tim, n_fre -- reform cube.

        n_tim, n_ddid, n_fre = flag3.shape
        flag3 = flag3.reshape((n_tim, n_ddid*n_fre))

        FIELDS = list(self.timeant.dtype.fields.keys())

        flagged_times = flag3.all(axis=1)
        flagged_chans = flag3.all(axis=0)

        print("adjusting statistics based on output flags", file=log)
        print("  {:.2%} of all timeslots are flagged".format(
                                        flagged_times.sum()/float(flagged_times.size)), file=log)
        print("  {:.2%} of all channels are flagged".format(
                                        flagged_chans.sum()/float(flagged_chans.size)), file=log)

        for field in FIELDS:
            self.chanant[field][flagged_chans, :] = 0
            self.timeant[field][flagged_times, :] = 0
            self.timechan[field][flagged_times, :] = 0
            self.timechan[field][:, flagged_chans] = 0




