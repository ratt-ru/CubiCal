from pyrap.tables import table
import numpy as np
from collections import Counter, OrderedDict
# from pprint import pprint

class DataHandler:
    """
    Class for handling measurement sets.

    CURRENT FEATURES:

    PLANNED FEATURES:
    Streaming I/O
    """
    def __init__(self, ms_name):
        self.ms_name = ms_name
        self.data = table(self.ms_name)

        self.nrows = self.data.nrows()
        self.ntime = len(np.unique(self.fetch("TIME")))
        self.nfreq = self.data.getcoldesc("DATA")["shape"][0]
        self.ncorr = self.data.getcoldesc("DATA")["shape"][1]
        self.nants = table(self.ms_name + "/ANTENNA").nrows()

        self.obvis = None
        self.movis = None
        self.antea = None
        self.anteb = None
        self.times = None

        self.tdict = None

    def fetch(self, *args, **kwargs):
        """
        Convenience function which mimics pyrap.tables.table.getcol().
        """

        return self.data.getcol(*args, **kwargs)

    def index_t(self, times):
        """
        Converts time values into an array of indices.
        """

        for i, j in enumerate(np.unique(times)):
            times[times == j] = i

        return times.astype(np.int64)

    def fetch_all(self, *args, **kwargs):

        self.obvis = self.fetch("DATA", *args, **kwargs)
        self.movis = self.fetch("MODEL_DATA", *args, **kwargs)
        self.antea = self.fetch("ANTENNA1", *args, **kwargs)
        self.anteb = self.fetch("ANTENNA2", *args, **kwargs)
        self.times = self.index_t(self.fetch("TIME", *args, **kwargs))
        self.tdict = OrderedDict(sorted(Counter(self.times).items()))

    def define_chunk(self, tdim=1, fdim=1):
        """
        Defines the chunk dimensions for the data.
        """

        self.chunk_tdim = tdim
        self.chunk_fdim = fdim

        # self.chunk_nrow = self.tdict[self.chunk_tdim - 1]

        # self.curtdim = tdim
        # self.curfdim = fdim
        #
        # self.lind = tdim
        # self.lrow = np.sum(self.tdict.values()[:self.lind])
        #
        # self.lfre = 0

    def vis_to_array(self, f_t_row, l_t_row, f_f_col, l_f_col):


        # Creates empty 5D arrays into which the model and observed data can
        # be packed. TODO: 6D? dtype?

        obser_arr = np.empty([self.chunk_tdim, self.chunk_fdim, self.nants,
                              self.nants, self.ncorr], dtype=np.complex128)

        model_arr = np.empty([self.chunk_tdim, self.chunk_fdim, self.nants,
                              self.nants, self.ncorr], dtype=np.complex128)

        # Grabs the relevant time and antenna info.

        tchunk = self.times[f_t_row:l_t_row]
        achunk = self.antea[f_t_row:l_t_row]
        bchunk = self.anteb[f_t_row:l_t_row]

        # The following takes the arbitrarily ordered data from the MS and
        # places it into a 5D data structure (measurement matrix).

        self.obvis[5,0,1] = 100
        self.movis[5,0,1] = 123

        obser_arr[tchunk, :, achunk, bchunk, :] \
            = self.obvis[f_t_row:l_t_row, f_f_col:l_f_col, :]
        obser_arr[tchunk, :, bchunk, achunk, :] \
            = self.obvis[f_t_row:l_t_row, f_f_col:l_f_col, :].conj()

        model_arr[tchunk, :, achunk, bchunk, :] \
            = self.movis[f_t_row:l_t_row, f_f_col:l_f_col, :]
        model_arr[tchunk, :, bchunk, achunk, :] \
            = self.movis[f_t_row:l_t_row, f_f_col:l_f_col, :].conj()

        # This zeros the diagonal elements in the "baseline" plane. This is
        # purely a precaution - we do not want autocorrelations on the
        # diagonal. TODO: For loop with fill_diagonal?

        obser_arr.reshape([-1, self.chunk_fdim, self.nants**2, self.ncorr]) \
            [:, :, ::self.nants + 1, :] = 0

        model_arr.reshape([-1, self.chunk_fdim, self.nants**2, self.ncorr]) \
            [:, :, ::self.nants + 1, :] = 0

        obser_arr = obser_arr.reshape([self.chunk_tdim, self.chunk_fdim,
                                       self.nants, self.nants, 2, 2])
        model_arr = model_arr.reshape([self.chunk_tdim, self.chunk_fdim,
                                       self.nants, self.nants, 2, 2])

        return obser_arr, model_arr
