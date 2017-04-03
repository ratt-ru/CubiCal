import numpy as np
from collections import Counter, OrderedDict
import pyrap.tables as pt
import cPickle
import better_exceptions

from Tools import logger, ModColor
log = logger.getLogger("ReadModelHandler")

try:
    import MBTiggerSim as mbt
    import TiggerSourceProvider as tsp
except:
    print>>log, ModColor.Str("Montblanc is not installed - simulation mode will crash.")

from time import time

class ReadModelHandler:

    def __init__(self, ms_name, sm_name, taql=None, fid=None, ddid=None,
                 precision="32", ddes=False, simulate=False, apply_weights=False):

        self.ms_name = ms_name
        self.sm_name = sm_name
        self.fid = fid if fid is not None else 0

        self.taql = self.build_taql(taql, fid, ddid)

        self.ms = pt.table(self.ms_name, readonly=False, ack=False)

        if self.taql:
            print>>log, "Applying TAQL query: %s"%self.taql
            self.data = self.ms.query(self.taql)
        else:
            self.data = self.ms

        print>>log, ModColor.Str("reading MS %s"%self.ms_name, col="green")

        self._anttab = pt.table(self.ms_name + "::ANTENNA", ack=False)
        self._fldtab = pt.table(self.ms_name + "::FIELD", ack=False)
        self._spwtab = pt.table(self.ms_name + "::SPECTRAL_WINDOW", ack=False)
        self._poltab = pt.table(self.ms_name + "::POLARIZATION", ack=False)
        self._ddesctab = pt.table(self.ms_name + "::DATA_DESCRIPTION", ack=False)

        self.ctype = np.complex128 if precision=="64" else np.complex64
        self.ftype = np.float64 if precision=="64" else np.float32
        self.nrows = self.data.nrows()
        self.ntime = len(np.unique(self.fetch("TIME")))
        self.nfreq = self._spwtab.getcol("NUM_CHAN")[0]
        self.ncorr = self._poltab.getcol("NUM_CORR")[0]
        self.nants = self._anttab.nrows()

        self._nchans = self._spwtab.getcol("NUM_CHAN")
        self._rfreqs = self._spwtab.getcol("REF_FREQUENCY")
        self._chanfr = self._spwtab.getcol("CHAN_FREQ")
        self._antpos = self._anttab.getcol("POSITION")
        self._phadir = self._fldtab.getcol("PHASE_DIR", startrow=self.fid,
                                           nrow=1)[0][0]

        # figure out DDID range
        if ddid is not None:
            if isinstance(ddid, (tuple, list)) and len(ddid) == 2:
                self._ddids = range(*ddid)
            else:
                self._ddids = [ddid]
        else:
            self._ddids = range(self._ddesctab.nrows())

        print>>log,"%d antennas, %d rows, %d DDIDs, %d timeslots, %d channels, %d corrs" % (self.nants,
                    self.nrows, len(self._ddids), self.ntime, self.nfreq, self.ncorr)

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
        self._first_f = None
        self._last_f = None

        self.simulate = simulate
        self.use_ddes = ddes
        self.apply_weights = apply_weights
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
        # time & DDID columns
        self.time_col = self.fetch("TIME", *args, **kwargs)
        self.ddid_col = self.fetch("DATA_DESC_ID", *args, **kwargs)
        # list of unique times
        self.uniq_times = np.unique(self.time_col)
        # timeslot index (per row, each element gives index of timeslot)
        self.times = np.empty_like(self.time_col, dtype=np.int32)
        for i, t in enumerate(self.uniq_times):
            self.times[self.time_col == t] = i
        # not that timeslot does not need to be monotonic, but must be monotonic within each DDID
        # # check that times increase monotonically
        # if min(self.times[1:]-self.times[:-1]) < 0:
        #     print>>log, ModColor.Str("TIME column is not in increasing order. Cannot deal with this MS!")
        #     raise RuntimeError
        self.covis = np.empty_like(self.obvis)
        self.flags = self.fetch("FLAG", *args, **kwargs)
        self.uvwco = self.fetch("UVW", *args, **kwargs)

        if self.apply_weights:
            if "WEIGHT_SPECTRUM" in self.ms.colnames():
                self.weigh = self.fetch("WEIGHT_SPECTRUM", *args, **kwargs)
            else:
                self.weigh = self.fetch("WEIGHT", *args, **kwargs)
                self.weigh = self.weigh[:,np.newaxis,:].repeat(self.nfreq, 1)

        if "BITFLAG" in self.ms.colnames():
            self.bflag = self.fetch("BITFLAG", *args, **kwargs)

    def define_chunk(self, tdim=1, fdim=1):
        """
        Defines the chunk dimensions for the data.

        Args:
            tdim (int): Timeslots per chunk.
            fdim (int): Frequencies per chunk.
        """

        self.chunk_tdim = tdim
        self.chunk_fdim = fdim

        # make list of timeslots at which we cut our time chunks
        # JK: when you start dealing with SCAN_NUMBER, you'll need to massage this list
        timechunks = range(self.times[0], self.times[-1]+1, self.chunk_tdim)
        timechunks.append(self.times[-1]+1)
        print>>log,"using %d time chunks: %s"%(len(timechunks)-1, " ".join(map(str, timechunks)))

        # number of timeslots per each time chunk
        self.chunk_ntimes = []
        # unique timestamps per each time chunk
        self.chunk_timestamps = []
        # for each time chunk, make mask of rows associated with that chunk
        timechunk_mask = {}
        for tchunk in range(len(timechunks) - 1):
            ts0, ts1 = timechunks[tchunk:tchunk + 2]
            timechunk_mask[tchunk] = (self.times>=ts0) & (self.times<ts1)
            self.chunk_ntimes.append(ts1-ts0)
            self.chunk_timestamps.append(np.unique(self.times[timechunk_mask[tchunk]]))

        # now, chunk_rind (chunk row index) will be an ordered dict, keyed by ddid,timechunk tuple.
        # Per each such tuple, it gives a _list of row indices_ corresponding to that chunk
        self.chunk_rind = OrderedDict()
        for ddid in self._ddids:
            ddid_rowmask = self.ddid_col==ddid
            for tchunk in range(len(timechunks)-1):
                self.chunk_rind[ddid,tchunk] = np.where(ddid_rowmask & timechunk_mask[tchunk])[0]

        # retiring this for now
        # break_i, break_t = self.check_contig()

        # self.chunk_tkey = self.tdict.keys()[::self.chunk_tdim]
        # self.chunk_tkey.append(self.ntime)

        # self.chunk_tkey.extend(break_t)
        # self.chunk_tind.extend(break_i)

        # self.chunk_tkey = sorted(np.unique(self.chunk_tkey))
        # self.chunk_tind = sorted(np.unique(self.chunk_tind))
        print>>log,"using %d row chunks"%(len(self.chunk_rind),)

        self.chunk_find = range(0, self.nfreq, self.chunk_fdim)
        self.chunk_find.append(self.nfreq)
        print>>log,"using %d freq chunks: %s"%(len(self.chunk_find)-1, " ".join(map(str, self.chunk_find)))

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

        obs_arr = None
        mod_arr = None
        wgt_arr = None 

        for (ddid, tchunk), rows  in self.chunk_rind.iteritems():
            nrows = len(rows)
            if not nrows:
                continue

            for j in xrange(len(self.chunk_find[:-1])):

                self._chunk_label = "D%dT%dF%d" % (ddid, tchunk, j)

                self._chunk_ddid = ddid
                self._chunk_tchunk = tchunk
                self._first_f = self.chunk_find[j]
                self._last_f = self.chunk_find[j + 1]

                t_dim = self.chunk_ntimes[tchunk]
                f_dim = self._last_f - self._first_f

                obs_arr = self.col_to_arr("obser", t_dim, f_dim, rows)
                mod_arr = self.col_to_arr("model", t_dim, f_dim, rows)

                if self.apply_weights:
                    wgt_arr = self.col_to_arr("weigh", t_dim, f_dim, rows)

                if self.simulate:
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

                yield obs_arr, mod_arr, wgt_arr, self._chunk_label


    def col_to_arr(self, target, chunk_tdim, chunk_fdim, rows):
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
                            self.nants, 4], dtype=self.ctype)

        # Grabs the relevant time and antenna info.

        achunk = self.antea[rows]
        bchunk = self.anteb[rows]
        tchunk = self.times[rows]
        tchunk -= np.min(tchunk)

        # Creates a tuple of slice objects to make subsequent slicing easier.

        selection = (rows,
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

        if self.ncorr==4:
            out_arr[tchunk, :, achunk, bchunk, :] = column[selection]
            out_arr[tchunk, :, bchunk, achunk, :] = column[selection].conj()[...,(0,2,1,3)]
        elif self.ncorr==2:
            out_arr[tchunk, :, achunk, bchunk, ::3] = column[selection]
            out_arr[tchunk, :, bchunk, achunk, ::3] = column[selection].conj()
        elif self.ncorr==1:
            out_arr[tchunk, :, achunk, bchunk, ::3] = column[selection][:,:,(0,0)]
            out_arr[tchunk, :, bchunk, achunk, ::3] = column[selection][:,:,(0,0)].conj()

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

        ddid, timechunk, f_f_col, l_f_col = bounds

        rows = self.chunk_rind[ddid,timechunk]
        new_shape = [len(rows), l_f_col - f_f_col, 4]

        tchunk = self.times[rows]
        tchunk -= tchunk[0]        # is this correct -- does in_array start from beginning of chunk?
        achunk = self.antea[rows]
        bchunk = self.anteb[rows]

        if self.ncorr==4:
            self.covis[rows, f_f_col:l_f_col, :] = \
                in_arr[tchunk, :, achunk, bchunk, :].reshape(new_shape)
        elif self.ncorr==2:
            self.covis[rows, f_f_col:l_f_col, :] = \
                in_arr[tchunk, :, achunk, bchunk, :].reshape(new_shape)[:,:,::3]
        elif self.ncorr==1:
            self.covis[rows, f_f_col:l_f_col, :] = \
                in_arr[tchunk, :, achunk, bchunk, :].reshape(new_shape)[:,:,::4]


    def add_to_gain_dict(self, gains, bounds, t_int=1, f_int=1):

        n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gains.shape

        ddid, timechunk, first_f, last_f = bounds

        timestamps = self.chunk_timestamps[timechunk]

        freqs = range(first_f,last_f)
        freq_indices = [[] for i in xrange(n_fre)]

        for f, freq in enumerate(freqs):
            freq_indices[f//f_int].append(freq)

        for d in xrange(n_dir):
            for t in xrange(n_tim):
                for f in xrange(n_fre):
                    comp_idx = (d,tuple(timestamps),tuple(freq_indices[f]))
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
