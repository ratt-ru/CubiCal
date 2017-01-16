import numpy as np
from collections import Counter, OrderedDict
import pyrap.tables as pt
import MBTiggerSim as mbt
import TiggerSourceProvider as tsp

from time import time

class ReadModelHandler:

    def __init__(self, ms_name, taql=None, fid=None, ddid=None,
                 precision="32"):

        self.ms_name = ms_name
        self.sm_name = "3C147-GdB-spw0+pybdsm.lsm.html"
        self.fid = fid if fid is not None else 0

        self.taql = self.build_taql(taql, fid, ddid)

        self.ms = pt.table(self.ms_name, readonly=False)

        if self.taql:
            print "Applying TAQL command: " , self.taql
            self.data = self.ms.query(self.taql)
        else:
            self.data = self.ms

        self.ctype = np.complex128 if precision=="64" else np.complex64
        self.ftype = np.float64 if precision=="64" else np.float32
        self.nrows = self.data.nrows()
        self.ntime = len(np.unique(self.fetch("TIME")))
        self.nfreq, self.ncorr = self.data.getcoldesc("DATA")["shape"]
        self.nants = pt.table(self.ms_name + "::ANTENNA").nrows()

        self._anttab = pt.table(self.ms_name + "::ANTENNA")
        self._fldtab = pt.table(self.ms_name + "::FIELD")
        self._spwtab = pt.table(self.ms_name + "::SPECTRAL_WINDOW")

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

        self.apply_flags = False
        self.bitmask = None


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
        self.weigh = self.fetch("WEIGHT", *args, **kwargs)
        self.uvwco = self.fetch("UVW", *args, **kwargs)

        try:
            self.bflag = self.fetch("BITFLAG", *args, **kwargs)
        except:
            print "No BITFLAG column in MS."


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

                mssrc = mbt.MSSourceProvider(self, t_dim, f_dim)
                tgsrc = tsp.TiggerSourceProvider(self._phadir, self.sm_name,
                                                    use_ddes=Truegit )
                arsnk = mbt.ArraySinkProvider(self, t_dim, f_dim, tgsrc._nclus)

                srcprov = [mssrc, tgsrc]
                snkprov = [arsnk]

                for i in xrange(tgsrc._nclus):
                    mbt.simulate(srcprov, snkprov)
                    tgsrc.update_target()
                    arsnk._dir += 1

                print arsnk._sim_array.shape


                yield self.vis_to_array(t_dim, f_dim, self._first_t,
                                        self._last_t, self._first_f,
                                        self._last_f)

    def vis_to_array(self, chunk_tdim, chunk_fdim, f_t_row, l_t_row, f_f_col,
                     l_f_col):
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

        # Creates empty 5D arrays into which the model and observed data can
        # be packed. TODO: 6D? dtype?

        obser_arr = np.empty([chunk_tdim, chunk_fdim, self.nants,
                              self.nants, self.ncorr], dtype=self.ctype)

        model_arr = np.empty([chunk_tdim, chunk_fdim, self.nants,
                              self.nants, self.ncorr], dtype=self.ctype)

        # Grabs the relevant time and antenna info.

        tchunk = self.times[f_t_row:l_t_row]
        tchunk -= np.min(tchunk)
        achunk = self.antea[f_t_row:l_t_row]
        bchunk = self.anteb[f_t_row:l_t_row]

        # The following takes the arbitrarily ordered data from the MS and
        # places it into a 5D data structure (measurement matrix).

        if self.apply_flags:
            flags_arr = self.make_flag_array(f_t_row, l_t_row, f_f_col, l_f_col)
            self.obvis[f_t_row:l_t_row, f_f_col:l_f_col, :][flags_arr] = 0
            self.movis[f_t_row:l_t_row, f_f_col:l_f_col, :][flags_arr] = 0

        obser_arr[tchunk, :, achunk, bchunk, :] \
            = self.obvis[f_t_row:l_t_row, f_f_col:l_f_col, :]
        obser_arr[tchunk, :, bchunk, achunk, :] \
            = self.obvis[f_t_row:l_t_row, f_f_col:l_f_col, :].conj()[...,(0,2,1,3)]

        model_arr[tchunk, :, achunk, bchunk, :] \
            = self.movis[f_t_row:l_t_row, f_f_col:l_f_col, :]
        model_arr[tchunk, :, bchunk, achunk, :] \
            = self.movis[f_t_row:l_t_row, f_f_col:l_f_col, :].conj()[...,(0,2,1,3)]

        # This zeros the diagonal elements in the "baseline" plane. This is
        # purely a precaution - we do not want autocorrelations on the
        # diagonal. TODO: For loop with fill_diagonal?

        obser_arr[:,:,range(self.nants),range(self.nants),:] = 0
        model_arr[:,:,range(self.nants),range(self.nants),:] = 0

        obser_arr = obser_arr.reshape([chunk_tdim, chunk_fdim,
                                       self.nants, self.nants, 2, 2])
        model_arr = model_arr.reshape([chunk_tdim, chunk_fdim,
                                       self.nants, self.nants, 2, 2])

        return obser_arr, model_arr

    def make_flag_array(self, f_t_row, l_t_row, f_f_col, l_f_col):
        """
        Combines flags into a flag array which can be applied to the data.

        Args:
            f_t_row (int): First time row to be accessed in data.
            l_t_row (int): Last time row to be accessed in data.
            f_f_col (int): First frequency column to be accessed in data.
            l_f_col (int): Last frequency column to be accessed in data.

        Returns:
            flags_arr (np.array): Array containing flags for the active data.
        """

        flags_arr = self.flags[f_t_row:l_t_row, f_f_col:l_f_col, :]

        #MAY BE WRONG NOW!!!

        if self.bitmask != 0:
            flags_arr &= ((self.bflag[f_t_row:l_t_row, f_f_col:l_f_col,
                           :] & self.bitmask) == True)

        return flags_arr


    def array_to_vis(self, in_arr, f_t_row, l_t_row, f_f_col, l_f_col):
        """
        Converts the calibrated measurement matrix back into the MS style.

        Args:
            in_arr (np.array): Input array which is to be made MS friendly.
            f_t_row (int): First time row in MS to which the data belongs.
            l_t_row (int): Last time row in MS to which the data belongs.
            f_f_col (int): First frequency in MS to which the data belongs.
            l_f_col (int): Last frequency in MS to which the data belongs.

        """

        new_shape = [l_t_row - f_t_row, l_f_col - f_f_col, 4]

        tchunk = self.times[f_t_row:l_t_row]
        achunk = self.antea[f_t_row:l_t_row]
        bchunk = self.anteb[f_t_row:l_t_row]

        self.covis[f_t_row:l_t_row, f_f_col:l_f_col, :] = \
            in_arr[tchunk, :, achunk, bchunk, :].reshape(new_shape)

    def save(self, values, col_name):
        """
        Saves values to column in MS.

        Args
        values (np.array): Values to be written to column.
        col_name (str): Name of target column.
        """

        self.data.putcol(col_name, values)



# ms = ReadModelHandler("~/measurements/WESTERBORK_POL_2.MS")
# ms.mass_fetch()
# ms.define_chunk(100,64)
#
# for obs, mod in ms:
#     pass