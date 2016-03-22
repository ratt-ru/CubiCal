import numpy as np
import pyrap.tables as pt
from collections import Counter, OrderedDict


class DataHandler:
    """
    Data handler object which interfaces with the measurement set.
    """

    def __init__(self, ms_name):
        """
        Initialisation routine for the data handler object. Assosciates the
        handler with an MS and extracts crucial data regarding size and shape.
        """
        self.ms_name = ms_name
        self.ms = pt.table(self.ms_name)
        self.nrows = self.ms.nrows()
        self.nfreq = self.ms.getcoldesc("DATA")["shape"][0]
        self.nfeed = self.ms.getcoldesc("DATA")["shape"][1]
        self.shape = [self.nrows, self.nfreq, self.nfeed]
        self.nante = pt.table(self.ms_name+"/ANTENNA").nrows()
        self.times = self.t2ind()
        self.tdict = self.get_tdict()
        self.ntime = len(self.tdict.keys())

        self.obvis = None
        self.movis = None
        self.antea = None
        self.anteb = None

        self.find = 0
        self.lind = 0
        self.frow = 0
        self.lrow = 0
        self.ffre = 0
        self.lfre = 0
        self.curfdim = 0
        self.curtdim = 0


    def get_data(self, blc=(-1, -1), trc=(-1, -1), startrow=0, nrow=-1):
        """
        Loads data necessary for solution into memory. Default is to load all
        the data, but optional arguments may be used to reduce the amount of
        data loaded at once.
        """

        self.obvis = self.ms.getcolslice("DATA", blc, trc, startrow=startrow,
                                         nrow=nrow)
        self.movis = self.ms.getcolslice("MODEL_DATA", blc, trc,
                                         startrow=startrow, nrow=nrow)
        self.antea = self.ms.getcol("ANTENNA1", startrow=startrow, nrow=nrow)
        self.anteb = self.ms.getcol("ANTENNA2", startrow=startrow, nrow=nrow)

    def t2ind(self):
        """
        Converts time values into an array of indices.
        """

        times = self.ms.getcol("TIME")

        for i,j in enumerate(np.unique(times)):
            times[times==j] = i

        return times.astype(np.int64)

    def get_tdict(self):
        """
        Converts time values into a dictionary containing the unique times as
        keys and the number of occurrences as the associated value.
        """

        tcount = Counter(self.times)
        tdict = OrderedDict(sorted(tcount.items()))

        return tdict

    def define_chunk(self, tdim=1, fdim=1):
        """
        Defines the chunk dimensions for the data.
        """

        self.ctdim = tdim
        self.cfdim = fdim

        self.curtdim = tdim
        self.curfdim = fdim

        self.lind = tdim
        self.lrow = np.sum(self.tdict.values()[:self.lind])

        self.lfre = 0

    def vis2mat(self):
        """
        Stacks visibilities into a matrix.
        """

        obsmat = np.zeros([self.curfdim, self.curtdim, self.nante, self.nante],
                                dtype=np.complex128)

        modmat = np.zeros([self.curfdim, self.curtdim, self.nante, self.nante],
                                dtype=np.complex128)

        tchunk = self.times[self.frow:self.lrow, np.newaxis] - self.find
        achunk = self.antea[self.frow:self.lrow, np.newaxis]
        bchunk = self.anteb[self.frow:self.lrow, np.newaxis]

        vchunk = self.obvis[self.frow:self.lrow, self.ffre:self.lfre, 0]
        vchunk = np.rollaxis(vchunk,1)[..., np.newaxis]

        obsmat[:, tchunk, achunk, bchunk] = vchunk
        obsmat[:, tchunk, bchunk, achunk] = vchunk.conj()

        vchunk = self.movis[self.frow:self.lrow, self.ffre:self.lfre, 0]
        vchunk = np.rollaxis(vchunk,1)[..., np.newaxis]

        modmat[:, tchunk, achunk, bchunk] = vchunk
        modmat[:, tchunk, bchunk, achunk] = vchunk.conj()

        for i in range(self.curfdim):
            for j in range(self.curtdim):
                np.fill_diagonal(obsmat[i,j,:,:], 0)
                np.fill_diagonal(modmat[i,j,:,:], 0)

        return obsmat, modmat

    def __iter__(self):
        return self

    def next(self):

        if self.lfre == self.nfreq:
            if self.lrow == self.nrows:
                raise StopIteration

            self.find = self.lind

            if (self.lind + self.ctdim) > self.ntime:
                self.curtdim = self.ntime - self.lind
                self.lind = self.ntime
            else:
                self.lind += self.ctdim

            self.frow = self.lrow
            self.lrow = np.sum(self.tdict.values()[:self.lind])

            self.lfre = 0

        self.ffre = self.lfre

        if (self.lfre + self.cfdim) > self.nfreq:
            self.curfdim = self.nfreq - self.lfre
            self.lfre = self.nfreq
        else:
            self.lfre += self.cfdim
            self.curfdim = self.cfdim

        return self.vis2mat()
