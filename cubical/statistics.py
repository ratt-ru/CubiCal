import math
import numpy as np
import cPickle

from cubical.tools import logger

log = logger.getLogger("stats")

class SolverStats (object):
    """SolverStats is a container for various per-chunk statistics collected during the solving process.
    """
    def __init__ (self, obj):
        if type(obj) is np.ndarray:
            self._init_for_chunk(obj)
        elif type(obj) is dict:
            self._concatenate(obj)
        else:
            raise TypeError("can't init SolverStats from object of type %s" % type(obj))

    def _init_for_chunk (self, data):
        """
        Initializes stats for a chunk of data.
        """
        n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = data.shape
        # summary record arrays (per channel-antenna, time-antenna, time-channel)
        dtype = [ ('dv2', 'f8'), ('dr2', 'f8'), ('dv2n', 'i4'), ('dr2n', 'i4'),
                  ('chi2', 'f8'), ('chi2n', 'i4'),
                  ('initchi2', 'f8'), ('initchi2n', 'i4') ]
        self.chanant  = np.rec.array(np.zeros((n_fre, n_ant), dtype))
        self.timeant  = np.rec.array(np.zeros((n_tim, n_ant), dtype))
        self.timechan = np.rec.array(np.zeros((n_tim, n_fre), dtype))
        # other stats: per chunk
        dtype = [ ('label', 'S32'), ('iters', 'i4'),
                  ('num_intervals', 'i4'), ('num_converged', 'i4'), ('num_stalled', 'i4'),
                  ('num_sol_flagged', 'i4'),
                  ('init_chi2', 'f8'), ('init_noise', 'f8'), ('chi2', 'f8'), ('noise', 'f8') ]
        self.chunk = np.rec.array(np.zeros((), dtype))

    def save(self, filename):
        """Pickles contents to file. Better than pickling whole object, as the pickle then only contains
        standard classes (i.e. don't need cubical to read it)"""
        cPickle.dump((self.chanant, self.timeant, self.timechan, self.chunk), open(filename, 'w'), 2)

    def estimate_noise (self, data, flags, residuals=False):
        """
        Given a data cube with dimensions (n_tim, n_fre, n_ant, n_ant, n_cor, n_cor) and a flag cube
        (n_tim, n_fre, n_ant, n_ant), this function estimates the noise in the data by taking the
        differences between adjacent channels.

        Sum of delta-visibilities squared, and sum of valid terms, is stored in the 'dv2', 'dv2n' or
        'dr2', 'dr2n' summary fields (the latter if residuals=True)

        Returns tuple of noise, inverse_noise_per_antenna_channel_squared, inverse_noise_per_antenna_squared and inverse_noise_per_channel_squared.
        """

        n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor = data.shape

        # if only one frequency channel, can't estimate noise -- return 1

        if n_fre == 1:
            return 1., np.ones((n_fre, n_ant), np.float32), np.ones(n_ant, np.float32), np.ones(n_fre, np.float32)


        deltaflags = (flags!=0)
        deltaflags[:, 1:, ...] = deltaflags[:, 1:, ...] | deltaflags[:, :-1, ...]
        deltaflags[:, 0 , ...] = deltaflags[:,   1, ...]

        # Create array for the squared difference between channel-adjacent visibilities

        deltavis2 = np.zeros((n_tim, n_fre, n_ant, n_ant), np.float32)

        # Square the absolute value of the difference between channel-adjacent visibilities and sum
        # over correlations. Normalize the result by n_cor*n_cor*4. The factor of 4 arises because
        # Var<c1-c2> = Var<c1>+Var<c2> and Var<c>=Var<r>+Var<i>. Thus, the square of the abs difference
        # between two complex visibilities has contributions from _four_ noise terms.

        # TODO: When fewer than 4 correlations are provided, the normalisation needs to be different.

        # TODO: something smart with multiple-model cal. Currently only the first dataset is used.

        deltavis2[:, 1:, ...]  = np.square(abs(data[0, :, 1:, ...] - data[0, :, :-1, ...])).sum(axis=(-2,-1))
        deltavis2[:, 1:, ...] /= n_cor*n_cor*4
        deltavis2[:, 0 , ...]  = deltavis2[:, 1, ...]

        # The flagged elements are zeroed; we don't have an adequate noise estimate for those channels.

        deltavis2[deltaflags] = 0

        # This flag inversion gives a count of the valid estimates in deltavis2.

        deltaflags = ~deltaflags

        # sum into the various stats arrays (use dvis2 or dvis2p field, depending on whether pre- or post-solver noise
        # is being estimated)
        dv2attr = 'dr2'   if residuals else 'dv2'
        dn2attr = 'dr2n'  if residuals else 'dv2n'

        deltavis2_chan_ant = getattr(self.chanant,  dv2attr)[...] = deltavis2.sum(axis=(0, 2))  # sum, per chan, ant
        getattr(self.timeant,  dv2attr)[...] = deltavis2.sum(axis=(1, 2))  # sum, per chan, time
        getattr(self.timechan, dv2attr)[...] = deltavis2.sum(axis=(2, 3))  # sum, per time, chan
        deltanum_chan_ant = getattr(self.chanant,  dn2attr)[...] = deltaflags.sum(axis=(0, 2))  # sum, per chan, ant
        getattr(self.timeant,  dn2attr)[...] = deltaflags.sum(axis=(1, 2))  # sum, per chan, time
        getattr(self.timechan, dn2attr)[...] = deltaflags.sum(axis=(2, 3))  # sum, per time, chan

        # now compute the variance overall, and per antenna, and per channel
        with np.errstate(divide='ignore', invalid='ignore'):  # ignore division by 0
            noise_est = math.sqrt(deltavis2_chan_ant.sum() / deltanum_chan_ant.sum())
            inv_var_antchan =  deltavis2_chan_ant / deltanum_chan_ant
            inv_var_ant  = deltanum_chan_ant.sum(axis=0) / deltavis2_chan_ant.sum(axis=0)
            inv_var_chan = deltanum_chan_ant.sum(axis=1) / deltavis2_chan_ant.sum(axis=1)
        # antennas/channels with no data end up with NaNs here, so replace them with 0
        inv_var_antchan[~np.isfinite(inv_var_antchan)] = 0
        inv_var_ant[~np.isfinite(inv_var_ant)] = 0
        inv_var_chan[~np.isfinite(inv_var_chan)] = 0
        return noise_est, inv_var_antchan, inv_var_ant, inv_var_chan


    @staticmethod
    def add_records(recarray, recarray2):
        """Adds two record-type arrays together"""
        for field in recarray.dtype.fields.iterkeys():
            recarray[field] += recarray2[field]

    @staticmethod
    def normalize_records(recarray):
        """Normalizes record-type arrays by dividing each field 'X' by the field 'Xn'"""
        for field in recarray.dtype.fields.iterkeys():
            if field[-1] != 'n':
                nval = recarray[field+'n']
                mask = nval!=0
                recarray[field][mask] /= nval[mask]
        return np.rec.array(recarray)

    def _concatenate(self, stats):
        """
        Concatenates stats from a dictionary (indexed by time_index,freq_index) into a single object
        """
        # get lists of unique time and channel indices occurring in the dict
        times = sorted(set([key[0] for key in stats.iterkeys()]))
        chans = sorted(set([key[1] for key in stats.iterkeys()]))

        # now concatenate and add up cumulative stats
        self.chanant = np.concatenate([stats[times[0], chan].chanant for chan in chans], axis=0)
        for time in times[1:]:
            self.add_records(self.chanant, np.concatenate([stats[time, chan].chanant for chan in chans], axis=0))

        self.timeant = np.concatenate([stats[time, chans[0]].timeant for time in times], axis=0)
        for chan in chans[1:]:
            self.add_records(self.timeant, np.concatenate([stats[time, chan].timeant for time in times], axis=0))

        self.timechan = np.concatenate([np.concatenate([stats[time, chan].timechan for time in times], axis=0)
                                       for chan in chans], axis=1)

        # note that for some reason np.concatenate of record arrays produces structured arrays instead of nd.recarrays:
        # normalize_records() will convert them back

        # normalize by number of values
        self.chanant = self.normalize_records(self.chanant)
        self.timeant = self.normalize_records(self.timeant)
        self.timechan = self.normalize_records(self.timechan)

        # make 2D array of per-chunk values
        self.chunk = np.rec.array([[stats[time, chan].chunk for chan in chans] for time in times])

    def apply_flagcube(self, flag3):
        """
        Applies additional flag cube of shape n_times,n_ddid,n_chan to statistics. Basically if something
        is flagged in the output based on chi-sq or other means, we want to take it out of the stats.
        """

        # out stats are n_tim, n_fre -- reform cube
        n_tim, n_ddid, n_fre = flag3.shape
        flag3 = flag3.reshape((n_tim, n_ddid*n_fre))

        FIELDS = self.timeant.dtype.fields.keys()

        flagged_times = flag3.all(axis=1)
        flagged_chans = flag3.all(axis=0)

        print>>log,"adjusting statistics based on output flags"
        print>>log,"  {:.2%} of all timeslots are flagged".format(flagged_times.sum()/float(flagged_times.size))
        print>>log,"  {:.2%} of all channels are flagged".format(flagged_chans.sum()/float(flagged_chans.size))

        for field in FIELDS:
            self.chanant[field][flagged_chans, :] = 0
            self.timeant[field][flagged_times, :] = 0
            self.timechan[field][flagged_times, :] = 0
            self.timechan[field][:, flagged_chans] = 0




