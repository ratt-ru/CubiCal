import numpy as np
from collections import Counter, OrderedDict
import pyrap.tables as pt
import cPickle

try:
    import MBTiggerSim as mbt
    import TiggerSourceProvider as tsp
except:
    print "Montblanc is not installed - simulation mode will crash."

from time import time

class ReadModelHandler:

    def __init__(self, ms_name, sm_name, taql=None, fid=None, ddid=None,
                 precision="32", ddes=False, simulate=False):

        self.ms_name = ms_name
        self.sm_name = sm_name
        self.fid = fid if fid is not None else 0

        self.taql = self.build_taql(taql, fid, ddid)

        self.ms = pt.table(self.ms_name, readonly=False)

        if self.taql:
            print "Applying TAQL command: " , self.taql
            self.data = self.ms.query(self.taql)
        else:
            self.data = self.ms

        self._anttab = pt.table(self.ms_name + "::ANTENNA")
        self._fldtab = pt.table(self.ms_name + "::FIELD")
        self._spwtab = pt.table(self.ms_name + "::SPECTRAL_WINDOW")
        self._poltab = pt.table(self.ms_name + "::POLARIZATION")

        self.ctype = np.complex128 if precision=="64" else np.complex64
        self.ftype = np.float64 if precision=="64" else np.float32
        self.nrows = self.data.nrows()
        self.ntime = len(np.unique(self.fetch("TIME")))
        self.nfreq = self._spwtab.getcol("NUM_CHAN")[0]
        self.ncorr = self._poltab.getcol("NUM_CORR")[0]
        self.nants = pt.table(self.ms_name + "::ANTENNA").nrows()

        self._nchans = self._spwtab.getcol("NUM_CHAN")
        self._rfreqs = self._spwtab.getcol("REF_FREQUENCY")
        self._chanfr = self._spwtab.getcol("CHAN_FREQ")
        self._antpos = self._anttab.getcol("POSITION")
        self._phadir = self._fldtab.getcol("PHASE_DIR", startrow=self.fid,
                                           nrow=1)[0][0]

        self.obvis = None
        self.movis = None
        self.covis = None
        self.antea = None
        self.anteb = None
        self.rtime = None
        self.times = None
        self.tdict = None
        self.flags = None
        self.bflag = None
        self.weigh = None
        self.uvwco = None

        self.chunk_tdim = None
        self.chunk_fdim = None
        self.chunk_tind = None
        self.chunk_find = None
        self.chunk_tkey = None
        self._first_t = None
        self._last_t = None
        self._first_f = None
        self._last_f = None

        self.simulate = simulate
        self.use_ddes = ddes
        self.apply_flags = False
        self.bitmask = None
        self.gain_dict = {}

    def build_taql(self, taql=None, fid=None, ddid=None):

        if taql is not None:
            taqls = [ "(" + taql +")" ]
        else:
            taqls = []

        if fid is not None:
            taqls.append("FIELD_ID == %d" % fid)

        if ddid is not None:
            if isinstance(ddid,(tuple,list)) and len(ddid) == 2:
                taqls.append("DATA_DESC_ID IN %d:%d" % (ddid[0], ddid[1]))
            else:
                taqls.append("DATA_DESC_ID == %d" % ddid)

        return " && ".join(taqls)

    def fetch(self, *args, **kwargs):
        """
        Convenience function which mimics pyrap.tables.table.getcol().

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            data.getcol(*args, **kwargs)
        """

        return self.data.getcol(*args, **kwargs)

    def mass_fetch(self, *args, **kwargs):
        """
        Convenience function for grabbing all the necessary data from the MS.
        Assigns values to initialised attributes.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        self.obvis = self.fetch("DATA", *args, **kwargs).astype(self.ctype)
        self.movis = self.fetch("MODEL_DATA", *args, **kwargs).astype(self.ctype)
        self.antea = self.fetch("ANTENNA1", *args, **kwargs)
        self.anteb = self.fetch("ANTENNA2", *args, **kwargs)
        self.rtime = self.fetch("TIME", *args, **kwargs)
        self.times = self.t_to_ind(self.rtime)
        self.tdict = OrderedDict(sorted(Counter(self.times).items()))
        self.covis = np.empty_like(self.obvis)
        self.flags = self.fetch("FLAG", *args, **kwargs)
        self.uvwco = self.fetch("UVW", *args, **kwargs)

        if "WEIGHT_SPECTRUM" in self.ms.colnames():
            self.weigh = self.fetch("WEIGHT_SPECTRUM", *args, **kwargs)
        else:
            self.weigh = self.fetch("WEIGHT", *args, **kwargs)
            self.weigh = self.weigh[:,np.newaxis,:].repeat(self.nfreq, 1)

        if "BITFLAG" in self.ms.colnames():
            self.bflag = self.fetch("BITFLAG", *args, **kwargs)

    def t_to_ind(self, times):
        """
        Converts times into indices.

        Args:
            times (np.array): Time data.

        Returns:
            times (np.array): Time data as indices.
        """

        indices = np.empty_like(times, dtype=np.int32)

        for i, j in enumerate(np.unique(times)):
            indices[times == j] = i

        return indices

    def define_chunk(self, tdim=1, fdim=1):
        """
        Defines the chunk dimensions for the data.

        Args:
            tdim (int): Timeslots per chunk.
            fdim (int): Frequencies per chunk.
        """

        self.chunk_tdim = tdim
        self.chunk_fdim = fdim

        self.chunk_tind = [0]
        self.chunk_tind.extend(self.tdict.values())
        self.chunk_tind = list(np.cumsum(self.chunk_tind)[::self.chunk_tdim])

        break_i, break_t = self.check_contig()

        if self.chunk_tind[-1] != self.nrows:
            self.chunk_tind.append(self.nrows)

        self.chunk_tkey = self.tdict.keys()[::self.chunk_tdim]
        self.chunk_tkey.append(self.ntime)

        # self.chunk_tkey.extend(break_t)
        # self.chunk_tind.extend(break_i)

        self.chunk_tkey = sorted(np.unique(self.chunk_tkey))
        self.chunk_tind = sorted(np.unique(self.chunk_tind))

        self.chunk_find = range(0, self.nfreq, self.chunk_fdim)
        self.chunk_find.append(self.nfreq)

    def check_contig(self):

        ddid = self.fetch("SCAN_NUMBER")

        break_i, break_t = [], []

        for i in xrange(len(ddid) - 1):
            if ddid[i] != ddid[i + 1]:
                break_i.append(i + 1)
                break_t.append(self.times[i + 1])

        return break_i, break_t

    def __iter__(self):
        return next(self)

    def next(self):
        """
        Generator for the ReadModeHandler object.

        Yields:
            np.array: The next N-dimensional measurement matrix to be processed.
        """

        for i in xrange(len(self.chunk_tind[:-1])):
            for j in xrange(len(self.chunk_find[:-1])):

                self._first_t = self.chunk_tind[i]
                self._last_t = self.chunk_tind[i + 1]

                self._first_f = self.chunk_find[j]
                self._last_f = self.chunk_find[j + 1]

                t_dim = self.chunk_tkey[i + 1] - self.chunk_tkey[i]
                f_dim = self._last_f - self._first_f

                obs_arr = self.col_to_arr("obser", t_dim, f_dim)
                mod_arr = self.col_to_arr("model", t_dim, f_dim)

                if self.simulate is True:
                    mssrc = mbt.MSSourceProvider(self, t_dim, f_dim)
                    tgsrc = tsp.TiggerSourceProvider(self._phadir, self.sm_name,
                                                         use_ddes=self.use_ddes)
                    arsnk = mbt.ArraySinkProvider(self, t_dim, f_dim, tgsrc._nclus)

                    srcprov = [mssrc, tgsrc]
                    snkprov = [arsnk]

                    for clus in xrange(tgsrc._nclus):
                        mbt.simulate(srcprov, snkprov)
                        tgsrc.update_target()
                        arsnk._dir += 1

                    mod_shape = list(arsnk._sim_array.shape)[:-1] + [2,2]
                    mod_arr = arsnk._sim_array.reshape(mod_shape)

                yield obs_arr, mod_arr


    def col_to_arr(self, target, chunk_tdim, chunk_fdim):
        """
        Converts input data into N-dimensional measurement matrices.

        Args:
            chunk_tdim (int):  Timeslots per chunk.
            chunk_fdim (int): Frequencies per chunk.
            f_t_row (int): First time row to be accessed in data.
            l_t_row (int): Last time row to be accessed in data.
            f_f_col (int): First frequency column to be accessed in data.
            l_f_col (int): Last frequency column to be accessed in data.

        Returns:
            flags_arr (np.array): Array containing flags for the active data.
        """

        opts = {"model": self.movis, 
                "obser": self.obvis,
                "weigh": self.weigh}

        column = opts[target]

        # Creates empty 5D array into which the column data can be packed.

        out_arr = np.empty([chunk_tdim, chunk_fdim, self.nants,
                            self.nants, self.ncorr], dtype=self.ctype)

        # Grabs the relevant time and antenna info.

        achunk = self.antea[self._first_t:self._last_t]
        bchunk = self.anteb[self._first_t:self._last_t]
        tchunk = self.times[self._first_t:self._last_t]
        tchunk -= np.min(tchunk)

        # Creates a tuple of slice objects to make subsequent slicing easier.

        selection = (slice(self._first_t, self._last_t),
                     slice(self._first_f, self._last_f),
                     slice(None))

        # This handles flagging.TODO: Make using basic flags optional.

        if self.apply_flags:

            flags_arr = self.flags[selection]

            if self.bitmask is not None:
                flags_arr |= ((self.bflag[selection] & self.bitmask) != False)

            column[selection][flags_arr] = 0
            
        # The following takes the arbitrarily ordered data from the MS and
        # places it into tho 5D data structure (correlation matrix).

        out_arr[tchunk, :, achunk, bchunk, :] = column[selection]
        out_arr[tchunk, :, bchunk, achunk, :] = column[selection].conj()[...,(0,2,1,3)]

        # This zeros the diagonal elements in the "baseline" plane. This is
        # purely a precaution - we do not want autocorrelations on the
        # diagonal.

        out_arr[:,:,range(self.nants),range(self.nants),:] = 0

        # The final step here reshapes the output to ensure compatability
        # with code elsewhere. 

        if target is "obser":
            out_arr = out_arr.reshape([chunk_tdim, chunk_fdim,
                                       self.nants, self.nants, 2, 2])
        elif target is "model":
            out_arr = out_arr.reshape([1, chunk_tdim, chunk_fdim,
                                       self.nants, self.nants, 2, 2])
        elif target is "weigh":
            out_arr = np.average(out_arr.astype(self.ftype), axis=-1)
            out_arr = out_arr.reshape([chunk_tdim, chunk_fdim,
                                       self.nants, self.nants])

        return out_arr


    def arr_to_col(self, in_arr, bounds):
        """
        Converts the calibrated measurement matrix back into the MS style.

        Args:
            in_arr (np.array): Input array which is to be made MS friendly.
            f_t_row (int): First time row in MS to which the data belongs.
            l_t_row (int): Last time row in MS to which the data belongs.
            f_f_col (int): First frequency in MS to which the data belongs.
            l_f_col (int): Last frequency in MS to which the data belongs.

        """

        f_t_row, l_t_row, f_f_col, l_f_col = bounds

        new_shape = [l_t_row - f_t_row, l_f_col - f_f_col, 4]

        tchunk = self.times[f_t_row:l_t_row]
        achunk = self.antea[f_t_row:l_t_row]
        bchunk = self.anteb[f_t_row:l_t_row]

        self.covis[f_t_row:l_t_row, f_f_col:l_f_col, :] = \
            in_arr[tchunk, :, achunk, bchunk, :].reshape(new_shape)

    def add_to_gain_dict(self, gains, bounds, t_int=1, f_int=1):

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gains.shape

        first_t, last_t, first_f, last_f = bounds

        times = np.unique(self.rtime[first_t: last_t])
        time_indices = [[] for i in xrange(n_tim)]

        for t, time in enumerate(times):
            time_indices[t//t_int].append(time)

        freqs = range(first_f,last_f)
        freq_indices = [[] for i in xrange(n_fre)]

        for f, freq in enumerate(freqs):
            freq_indices[f//f_int].append(freq)

        for d in xrange(n_dir):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    comp_idx = (d,tuple(time_indices[t]),tuple(freq_indices[f]))
                    self.gain_dict[comp_idx] = gains[d,t,f,:]

    def write_gain_dict(self, output_name=None):

        if output_name is None:
            output_name = self.ms_name + "/gains.p"

        cPickle.dump(self.gain_dict, open(output_name, "wb"), protocol=2)


    def save(self, values, col_name):
        """
        Saves values to column in MS.

        Args
        values (np.array): Values to be written to column.
        col_name (str): Name of target column.
        """

        self.data.putcol(col_name, values)
