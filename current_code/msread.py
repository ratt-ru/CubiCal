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

        self.chunk_tdim = None
        self.chunk_fdim = None
        self.chunk_tind = None
        self.chunk_find = None
        self.active_t = 0
        self.active_f = 0

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

        self.chunk_tind = [0]
        self.chunk_tind.extend(self.tdict.values())
        self.chunk_tind = list(np.cumsum(self.chunk_tind)[::self.chunk_tdim])
        self.chunk_tind.append(self.nrows)

        self.chunk_tkeys = self.tdict.keys()[::self.chunk_tdim]
        self.chunk_tkeys.append(self.ntime)

        self.chunk_find = range(0, self.nfreq, self.chunk_fdim)
        self.chunk_find.append(self.nfreq)

        print self.chunk_tind
        print self.chunk_find

    def vis_to_array(self,chunk_tdim, chunk_fdim, f_t_row, l_t_row, f_f_col,
                     l_f_col):


        # Creates empty 5D arrays into which the model and observed data can
        # be packed. TODO: 6D? dtype?

        obser_arr = np.empty([chunk_tdim, chunk_fdim, self.nants,
                              self.nants, self.ncorr], dtype=np.complex128)

        model_arr = np.empty([chunk_tdim, chunk_fdim, self.nants,
                              self.nants, self.ncorr], dtype=np.complex128)

        # Grabs the relevant time and antenna info.

        tchunk = self.times[f_t_row:l_t_row]
        tchunk = tchunk - np.min(tchunk)
        achunk = self.antea[f_t_row:l_t_row]
        bchunk = self.anteb[f_t_row:l_t_row]

        # The following takes the arbitrarily ordered data from the MS and
        # places it into a 5D data structure (measurement matrix).

        # self.obvis[5,0,1] = 100
        # self.movis[5,0,1] = 100

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

        obser_arr.reshape([-1, chunk_fdim, self.nants**2, self.ncorr]) \
            [:, :, ::self.nants + 1, :] = 0

        model_arr.reshape([-1, chunk_fdim, self.nants**2, self.ncorr]) \
            [:, :, ::self.nants + 1, :] = 0

        obser_arr = obser_arr.reshape([chunk_tdim, chunk_fdim,
                                       self.nants, self.nants, 2, 2])
        model_arr = model_arr.reshape([chunk_tdim, chunk_fdim,
                                       self.nants, self.nants, 2, 2])

        return obser_arr, model_arr

    def __iter__(self):
        return next(self)

    def next(self):

        for i in xrange(len(self.chunk_tind[:-1]) - 1):
            for j in self.chunk_find[:-1]:

                first_t = self.chunk_tind[i]
                last_t = self.chunk_tind[i + 1]

                first_f = self.chunk_find[j]
                last_f = self.chunk_find[j + 1]

                print i, j

                t_dim = self.chunk_tkeys[i + 1] - \
                        self.chunk_tkeys[i]
                f_dim = last_f - first_f

                yield self.vis_to_array(t_dim, f_dim, first_t, last_t, first_f,
                                        last_f)



        # first_t = self.chunk_tind[self.active_t]
        # last_t = self.chunk_tind[self.active_t + 1]
        # first_f = self.chunk_find[self.active_f]
        # last_f = self.chunk_find[self.active_f + 1]
        # t_dim = self.chunk_tkeys[self.active_t + 1] - \
        #         self.chunk_tkeys[self.active_t]
        # f_dim = last_f - first_f
        #
        # self.active_f += 1
        #
        # if self.active_f == len(self.chunk_find) - 1:
        #     self.active_f = 0
        #     self.active_t += 1
        #     if self.active_t ==  len(self.chunk_tind) - 1:
        #         raise StopIteration
        #
        # return self.vis_to_array(t_dim, f_dim, first_t, last_t, first_f, last_f)
        #




