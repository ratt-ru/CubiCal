import numpy as np
from collections import Counter, OrderedDict
import pyrap.tables as pt
import cPickle
import re
import traceback
import sys
from cubical.tools import shared_dict
import cubical.flagging as flagging
from cubical.flagging import FL
from pdb import set_trace as BREAK  # useful: can set static breakpoints by putting BREAK() in the code
from cubical.MBTiggerSim import simulate, MSSourceProvider, ColumnSinkProvider
from cubical.TiggerSourceProvider import TiggerSourceProvider
from montblanc.impl.rime.tensorflow.sources import CachedSourceProvider, FitsBeamSourceProvider
#import better_exceptions

from cubical.tools import logger, ModColor
log = logger.getLogger("data_handler")

def _parse_range(arg, nmax):
    """Helper function. Parses an argument into a list of numbers. Nmax is max number.
    Supports e.g. 5, "5", "5~7" (inclusive range), "5:8" (pythonic range), "5,6,7" (list)
    """
    fullrange = range(nmax)
    if arg is None:
        return fullrange
    elif type(arg) is int:
        return [arg]
    elif type(arg) is tuple:
        return list(arg)
    elif type(arg) is list:
        return arg
    elif type(arg) is not str:
        raise TypeError("can't parse range of type '%s'"%type(arg))
    arg = arg.strip()
    if re.match("\d+$", arg):
        return [ int(arg) ]
    elif "," in arg:
        return map(int,','.split(arg))
    if re.match("(\d*)~(\d*)$", arg):
        i0, i1 = arg.split("~", 1)
        i0 = int(i0) if i0 else None
        i1 = int(i1)+1 if i1 else None
    elif re.match("(\d*):(\d*)$", arg):
        i0, i1 = arg.split(":", 1)
        i0 = int(i0) if i0 else None
        i1 = int(i1) if i1 else None
    else:
        raise ValueError("can't parse range '%s'"%arg)
    return fullrange[slice(i0,i1)]


## TERMINOLOGY:
## A "chunk" is data for one DDID, a range of timeslots (thus, a subset of the MS rows), and a slice of channels.
## Chunks are the basic parallelization unit. Solver deal with a chunk of data.
##
## A "row chunk" is data for one DDID, a range of timeslots, and *all* channels. One can imagine a row chunk
## as a "horizontal" vector of chunks across frequency.
##
## A "tile" is a collection of row chunks that are adjacent in time and/or DDID. One can imagine a tile as
## a vertical stack of row chunks


class RowChunk(object):
    """Very basic helper class, encapsulates a row chunk"""
    def __init__(self, ddid, tchunk, rows):
        self.ddid, self.tchunk, self.rows = ddid, tchunk, rows


class Tile(object):
    """Helper class, encapsulates a tile. A tile is a sequence of row chunks that's read and written as a unit.
    """
    # the tile list is effectively global. This is needed because worker subprocesses need to access the tiles.
    tile_list = None

    def __init__(self, handler, chunk):
        """Creates a tile, sets the first row chunk"""
        self.handler = handler
        self.rowchunks = [chunk]
        self.first_row = chunk.rows[0]
        self.last_row = chunk.rows[-1]
        self._rows_adjusted = False
        self._updated = False
        self.data = None


    def append(self, chunk):
        """Appends a row chunk to a tile"""
        self.rowchunks.append(chunk)
        self.first_row = min(self.first_row, chunk.rows[0])
        self.last_row = max(self.last_row, chunk.rows[-1])

    def merge(self, other):
        """Merges another tile into this one"""
        self.rowchunks += other.rowchunks
        self.first_row = min(self.first_row, other.first_row)
        self.last_row = max(self.last_row, other.last_row)

    def finalize(self):
        """
        Creates a list of chunks within the tile that can be iterated over, returns list of chunk labels.

        This also adjusts the row indices of all row chunks so that they become relative to the start of the tile.
        """
        self._data_dict_name = "DATA:{}:{}".format(self.first_row, self.last_row)

        # adjust row indices so they become relative to the first row of the tile

        if not self._rows_adjusted:
            for rowchunk in self.rowchunks:
                rowchunk.rows -= self.first_row
            self._rows_adjusted = True

        # create dict of { chunk_label: rows, chan0, chan1 } for all chunks in this tile

        self._chunk_dict = OrderedDict()
        num_freq_chunks = len(self.handler.chunk_find)-1
        for rowchunk in self.rowchunks:
            for ifreq in range(num_freq_chunks):
                label = "D{}T{}F{}".format(rowchunk.ddid, rowchunk.tchunk, ifreq)
                key = rowchunk.tchunk, rowchunk.ddid*num_freq_chunks + ifreq
                chan0, chan1 = self.handler.chunk_find[ifreq:ifreq + 2]
                self._chunk_dict[key] = label, rowchunk, chan0, chan1

        # copy various useful info from handler and make a simple list of unique ddids.

        self.ddids = np.unique([i[1].ddid for i in self._chunk_dict.values()])
        self.ddid_col = self.handler.ddid_col[self.first_row:self.last_row+1]
        self.time_col = self.handler.time_col[self.first_row:self.last_row+1]
        self.antea = self.handler.antea[self.first_row:self.last_row+1]
        self.anteb = self.handler.anteb[self.first_row:self.last_row+1]
        self.times = self.handler.times[self.first_row:self.last_row+1]
        self.ctype = self.handler.ctype
        self.nants = self.handler.nants
        self.ncorr = self.handler.ncorr
        self.nchan = self.handler._nchans[0]

    def get_chunk_keys(self):
        return self._chunk_dict.iterkeys()

    def get_chunk_label(self, key):
        return self._chunk_dict[key][0]

    def get_chunk_tfs(self, key):

        _, rowchunk, chan0, chan1 = self._chunk_dict[key]

        return self.time_col[rowchunk.rows], self.handler._chanfr[rowchunk.ddid, chan0:chan1]

    def load(self):
        """
        Fetches data from MS into tile data shared dict. Returns dict.
        This is meant to be called in the main or I/O process.
        """
        
        # Create a shared dict for the data arrays.
        
        data = shared_dict.create(self._data_dict_name)

        # These flags indicate if the (corrected) data or flags have been updated
        # Gotcha for shared_dict users! The only truly shared objects are arrays.
        # Thus, we create an array for the flags.
        
        data['updated'] = np.array([False, False])

        print>>log,"reading tile for MS rows {}~{}".format(self.first_row, self.last_row)
        
        nrows = self.last_row - self.first_row + 1
        
        data['obvis'] = self.handler.fetch(self.handler.data_column, self.first_row, nrows).astype(self.handler.ctype)
        print>> log(2), "  read " + self.handler.data_column

        data['uvwco'] = self.handler.fetch("UVW", self.first_row, nrows)
        print>> log(2), "  read UVW coordinates"

        if self.handler.sm_name!="":

            print>>log, "simulating model visibilities"

            expected_nrows, sort_ind, row_identifiers = self.prep_data(data)

            srcs, snks = [], []

            measet_src = MSSourceProvider(self, data, sort_ind)
            tigger_src = TiggerSourceProvider(self)
            cached_src = CachedSourceProvider(tigger_src, clear_start=True, clear_stop=True)

            srcs.append(measet_src)
            srcs.append(cached_src)

            if self.handler.beam_pattern:
                arbeam_src = FitsBeamSourceProvider(self.handler.beam_pattern,
                                                    self.handler.beam_l_axis,       
                                                    self.handler.beam_m_axis)
                srcs.append(arbeam_src)

            ndirs = tigger_src._nclus
            model_shape = np.array([ndirs, 1, expected_nrows, self.nchan, self.ncorr])

            data.addSharedArray('movis', model_shape, self.handler.ctype)

            column_snk = ColumnSinkProvider(self, data, sort_ind)

            snks.append(column_snk)

            for direction in xrange(ndirs):
                simulate(srcs, snks)
                tigger_src.update_target()
                column_snk._dir += 1

            self.unprep_data(data, nrows)

            data['movis'] = data['movis'][:,:,row_identifiers,:,:]
        
        else:

            print>>log, "sky model path unspecified or incorrect - no simulation."

            model_shape = [1,1] + list(data['obvis'].shape)
            data.addSharedArray('movis', model_shape, self.handler.ctype)
            
        if self.handler.model_column:
            
            data['movis'][0,0,...] += self.handler.fetch(self.handler.model_column, self.first_row, 
                                                            nrows).astype(self.handler.ctype)
            
            print>> log(2), "  read " + self.handler.model_column

        data.addSharedArray('covis', data['obvis'].shape, self.handler.ctype)

        if self.handler.weight_column:
            weight = self.handler.fetch(self.handler.weight_column, self.first_row, nrows)
            # If weight_column is WEIGHT, expand along the freq axis (looks like WEIGHT SPECTRUM).
            if self.handler.weight_column == "WEIGHT":
                data['weigh'] = weight[:, np.newaxis, :].repeat(self.handler.nfreq, 1)
            else:
                data['weigh'] = weight
            print>> log(2), "  read weights from column {}".format(self.handler.weight_column)

        # The following block of code deals with the various flagging operations and columns. The
        # aim is to correctly populate flag_arr from the various flag sources.

        # Make a flag array. This will contain FL.PRIOR for any points flagged in the MS.
        flag_arr = data.addSharedArray("flags", data['obvis'].shape, dtype=FL.dtype)

        # FLAG/FLAG_ROW only needed if applying them, or auto-filling BITLAG from them
        flagcol = flagrow = None
        if self.handler._apply_flags or self.handler._auto_fill_bitflag:
            flagcol = self.handler.fetch("FLAG", self.first_row, nrows)
            flagrow = self.handler.fetch("FLAG_ROW", self.first_row, nrows)
            print>> log(2), "  read FLAG/FLAG_ROW"

        if self.handler._apply_flags:
            flag_arr[flagcol] = FL.PRIOR
            flag_arr[flagrow, :, :] = FL.PRIOR

        # Form up bitflag array, if needed.
        if self.handler._apply_bitflags or self.handler._save_bitflag or self.handler._auto_fill_bitflag:
            read_bitflags = False
            # If not explicitly re-initializing, try to read column.
            if not self.handler._reinit_bitflags:
                self.bflagrow = self.handler.fetch("BITFLAG_ROW", self.first_row, nrows)
                # If there's an error reading BITFLAG, it must be unfilled. This is a common 
                # occurrence so we may as well deal with it. In this case, if auto-fill is set, 
                # fill BITFLAG from FLAG/FLAG_ROW.
                try:
                    self.bflagcol = self.handler.fetch("BITFLAG", self.first_row, nrows)
                    print>> log(2), "  read BITFLAG/BITFLAG_ROW"
                    read_bitflags = True
                except Exception:
                    if not self.handler._auto_fill_bitflag:
                        print>> log, ModColor.Str(traceback.format_exc().strip())
                        print>> log, ModColor.Str("Error reading BITFLAG column, and --flags-auto-init is not set.")
                        raise
                    print>>log,"  error reading BITFLAG column: not fatal, since we'll auto-fill it from FLAG"
                    for line in traceback.format_exc().strip().split("\n"):
                        print>> log, "    "+line
            # If column wasn't read, create arrays.
            if not read_bitflags:
                self.bflagcol = np.zeros(flagcol.shape, np.int32)
                self.bflagrow = np.zeros(flagrow.shape, np.int32)
                if self.handler._auto_fill_bitflag:
                    self.bflagcol[flagcol] = self.handler._auto_fill_bitflag
                    self.bflagrow[flagrow] = self.handler._auto_fill_bitflag
                    # mark flags as updated: they will be saved below
                    data['updated'][1] = True
                    print>> log, "  auto-filled BITFLAG/BITFLAG_ROW of shape %s"%str(self.bflagcol.shape)
            if self.handler._apply_bitflags:
                flag_arr[(self.bflagcol & self.handler._apply_bitflags) != 0] = FL.PRIOR
                flag_arr[(self.bflagrow & self.handler._apply_bitflags) != 0, :, :] = FL.PRIOR

        # Create a placeholder for the gains.
        data.addSubdict("gains")

        return data

    def prep_data(self, data):

        # Given data, we need to make sure that it looks the way MB wants it to.
        # First step - check the number of rows.

        n_bl = (self.nants*(self.nants - 1))/2
        ntime = len(np.unique(self.time_col))

        nrows = self.last_row - self.first_row + 1
        expected_nrows = n_bl*ntime*len(self.ddids)

        # The row identifiers determine which rows in the SORTED/ALL ROWS are required for the data
        # that is present in the MS. Essentially, they allow for the selection of an array of a size
        # matching that of the observed data. First term determines the offset by ddid, the second
        # is the offset by time, and the last turns antea and anteb into a unique offset per 
        # baseline.

        ddid_ind = self.ddid_col.copy()

        for ind, ddid in enumerate(self.ddids):
            ddid_ind[ddid_ind==ddid] = ind

        row_identifiers = ddid_ind*n_bl*ntime + (self.times - np.min(self.times))*n_bl + \
                          (-0.5*self.antea**2 + (self.nants - 1.5)*self.antea + self.anteb - 1).astype(np.int32)

        if nrows == expected_nrows:
            logstr = (nrows, ntime, n_bl, len(self.ddids))
            print>> log, "  {} rows consistent with {} timeslots and {} baselines across {} bands".format(*logstr)
            
            sorted_ind = np.lexsort((self.anteb, self.antea, self.time_col, self.ddid_col))

        elif nrows < expected_nrows:
            logstr = (nrows, ntime, n_bl, len(self.ddids))
            print>> log, "  {} rows inconsistent with {} timeslots and {} baselines across {} bands".format(*logstr)
            print>> log, "  {} fewer rows than expected".format(expected_nrows - nrows)

            nmiss = expected_nrows - nrows

            baselines = [(a,b) for a in xrange(self.nants) for b in xrange(self.nants) if b>a]

            missing_bl = []
            missing_t = []
            missing_ddids = []

            for ddid in self.ddids:
                for t in np.unique(self.time_col):
                    t_sel = np.where((self.time_col==t)&(self.ddid_col==ddid))
                
                    missing_bl.extend(set(baselines) - set(zip(self.antea[t_sel], self.anteb[t_sel])))
                    missing_t.extend([t]*(n_bl - t_sel[0].size))
                    missing_ddids.extend([ddid]*(n_bl - t_sel[0].size))

            missing_uvw = [[0,0,0]]*nmiss 
            missing_antea = np.array([bl[0] for bl in missing_bl])
            missing_anteb = np.array([bl[1] for bl in missing_bl])
            missing_t = np.array(missing_t)
            missing_ddids = np.array(missing_ddids)

            data['uvwco'] = np.concatenate((data['uvwco'], missing_uvw))
            self.antea = np.concatenate((self.antea, missing_antea))
            self.anteb = np.concatenate((self.anteb, missing_anteb))
            self.time_col = np.concatenate((self.time_col, missing_t))
            self.ddid_col = np.concatenate((self.ddid_col, missing_ddids))

            sorted_ind = np.lexsort((self.anteb, self.antea, self.time_col, self.ddid_col))

        elif nrows > expected_nrows:
            logstr = (nrows, ntime, n_bl, len(self.ddids))
            print>> log, "  {} rows inconsistent with {} timeslots and {} baselines across {} bands".format(*logstr)
            print>> log, "  {} more rows than expected".format(nrows - expected_nrows)
            print>> log, "  assuming additional rows are auto-correlations - ignoring"

            sorted_ind = np.lexsort((self.anteb, self.antea, self.time_col, self.ddid_col))            
            sorted_ind = sorted_ind[np.where(self.antea!=self.anteb)]

            if np.shape(sorted_ind) != expected_nrows:
                raise ValueError("Number of rows inconsistent after removing auto-correlations.")

        return expected_nrows, sorted_ind, row_identifiers

    def unprep_data(self, data, nrows):

            data['uvwco'] = data['uvwco'][:nrows,...]
            self.antea = self.antea[:nrows]
            self.anteb = self.anteb[:nrows]
            self.time_col = self.time_col[:nrows]
            self.ddid_col = self.ddid_col[:nrows]

    def get_chunk_cubes(self, key):
        """
        Returns label, data, model, flags, weights cubes for the given chunk key.

        Shapes are as follows:
            data:       [Nmod, Ntime, Nfreq, Nant, Nant, 2, 2]
            model:      [Ndir, Nmod, Ntime, Nfreq, Nant, Nant, 2, 2]
            flags:      [Ntime, Nfreq, Nant, Nant]
            weights:    [Nmod, Ntime, Nfreq, Nant, Nant] or None for no weighting

        Nmod refers to number of models simultaneously fitted.

        """
        data = shared_dict.attach(self._data_dict_name)

        label, rowchunk, freq0, freq1 = self._chunk_dict[key]

        t_dim = self.handler.chunk_ntimes[rowchunk.tchunk]
        f_dim = freq1 - freq0
        freq_slice = slice(freq0, freq1)
        rows = rowchunk.rows
        nants = self.handler.nants

        flags = self._column_to_cube(data['flags'], t_dim, f_dim, rows, freq_slice, FL.dtype, FL.MISSING)
        flags = np.bitwise_or.reduce(flags, axis=-1)
        obs_arr = self._column_to_cube(data['obvis'], t_dim, f_dim, rows, freq_slice, self.handler.ctype, reqdims=6)
        obs_arr = obs_arr.reshape(list(obs_arr.shape[:-1]) + [2, 2])
        mod_arr = self._column_to_cube(data['movis'], t_dim, f_dim, rows, freq_slice, self.handler.ctype, reqdims=7)
        mod_arr = mod_arr.reshape(list(mod_arr.shape[:-1]) + [2, 2])

        # flag invalid data or model
        flags[(~np.isfinite(obs_arr[0, ])).any(axis=(-2, -1))] |= FL.INVALID
        flags[(~np.isfinite(mod_arr[0, 0, ...])).any(axis=(-2, -1))] |= FL.INVALID

        flagged = flags != 0

        if 'weigh' in data:
            wgt_arr = self._column_to_cube(data['weigh'], t_dim, f_dim, rows, freq_slice, self.handler.ftype)
            wgt_arr = np.sqrt(np.sum(wgt_arr, axis=-1))  # take the square root of sum over correlations
            wgt_arr[flagged] = 0
            wgt_arr = wgt_arr.reshape([1, t_dim, f_dim, nants, nants])
        else:
            wgt_arr = None

        # zero flagged entries in data and model
        obs_arr[0, flagged, :, :] = 0
        mod_arr[0, 0, flagged, :, :] = 0

        return obs_arr, mod_arr, flags, wgt_arr

    def set_chunk_cubes(self, cube, flag_cube, key, column='covis'):
        """Copies a visibility cube, and an optional flag cube, back to tile column"""
        data = shared_dict.attach(self._data_dict_name)
        label, rowchunk, freq0, freq1 = self._chunk_dict[key]
        rows = rowchunk.rows
        freq_slice = slice(freq0, freq1)
        if cube is not None:
            data['updated'][0] = True
            self._cube_to_column(data[column], cube, rows, freq_slice)
        if flag_cube is not None:
            data['updated'][1] = True
            self._cube_to_column(data['flags'], flag_cube, rows, freq_slice, flags=True)

    def set_chunk_gains(self, gains, key):
        """Copies chunk gains to tile"""
        data = shared_dict.attach(self._data_dict_name)
        label = self.get_chunk_label(key)
        sd = data['gains'].addSubdict(label)
        sd['gains'] = gains
        sd['key'] = key

    def save(self, unlock=False):
        """
        Saves 'corrected' column, and any updated flags, back to MS.
        """
        nrows = self.last_row - self.first_row + 1
        data = shared_dict.attach(self._data_dict_name)
        if self.handler.output_column and data['updated'][0]:
            print>> log, "saving {} for MS rows {}~{}".format(self.handler.output_column, self.first_row, self.last_row)
            if self.handler._add_column(self.handler.output_column):
                self.handler._reopen()
            self.handler.data.putcol(self.handler.output_column, data['covis'], self.first_row, nrows)

        if self.handler._save_bitflag and data['updated'][1]:
            print>> log, "saving flags for MS rows {}~{}".format(self.first_row, self.last_row)
            # add bitflag to points where data wasn't flagged for prior reasons
            self.bflagcol[(data['flags']&~FL.PRIOR) != 0] |= self.handler._save_bitflag
            self.handler.data.putcol("BITFLAG", self.bflagcol, self.first_row, nrows)
            print>> log, "  updated BITFLAG column"
            self.bflagrow = np.bitwise_and.reduce(self.bflagcol,axis=(-1,-2))
            self.handler.data.putcol("BITFLAG_ROW", self.bflagrow, self.first_row, nrows)
            flag_col = self.bflagcol != 0
            self.handler.data.putcol("FLAG", flag_col, self.first_row, nrows)
            print>> log, "  updated FLAG column ({:.2%} visibilities flagged)".format(
                flag_col.sum() / float(flag_col.size))
            flag_row = flag_col.all(axis=(-1, -2))
            self.handler.data.putcol("FLAG_ROW", flag_row, self.first_row, nrows)
            print>> log, "  updated FLAG_ROW column ({:.2%} rows flagged)".format(
                flag_row.sum() / float(flag_row.size))

        if unlock:
            self.handler.unlock()

    def release(self):
        """
        Releases the data dict
        """
        data = shared_dict.attach(self._data_dict_name)
        data.delete()

    def _column_to_cube(self, column, chunk_tdim, chunk_fdim, rows, freqs, dtype, zeroval=0, reqdims=5):
        """
        Converts input data into N-dimensional measurement matrices.

        Args:
            column:      column array from which this will be filled
            chunk_tdim (int):  Timeslots per chunk.
            chunk_fdim (int): Frequencies per chunk.
            rows:        row slice (or set of indices)
            freqs:       frequency slice
            dtype:       data type
            zeroval:     null value to fill missing elements with

        Returns:
            Output cube of shape [chunk_tdim, chunk_fdim, self.nants, self.nants, 4]
        """

        # Start by establishing the possible dimensions and those actually present. Dimensions which
        # are not present are set to one, for flexibility reasons. Output shape is determined by
        # reqdims, which selects dimensions in reverse order from (ndir, nmod, nt, nf, na, na, nc). 
        # NOTE: The final dimension will be reshaped into 2x2 blocks outside this function.

        col_ndim = column.ndim

        possible_dims = ["dirs", "mods", "rows", "freqs", "cors"]

        dims = {possible_dims[-i] : column.shape[-i] for i in xrange(1, col_ndim + 1)}

        dims.setdefault("mods", 1)
        dims.setdefault("dirs", 1)

        out_shape = [dims["dirs"], dims["mods"], chunk_tdim, chunk_fdim, self.nants, self.nants, 4]
        out_shape = out_shape[-reqdims:]

        # Creates empty N-D array into which the column data can be packed.
        out_arr = np.full(out_shape, zeroval, dtype)

        # Grabs the relevant time and antenna info.

        achunk = self.antea[rows]
        bchunk = self.anteb[rows]
        tchunk = self.times[rows]
        tchunk -= np.min(tchunk)

        # Creates lists of selections to make subsequent selection from column and oout_arr easier.

        corr_slice = slice(None) if self.ncorr==4 else slice(None, None, 3)

        col_selections = [[dirs, mods, rows, freqs, slice(None)][-col_ndim:] 
                            for dirs in xrange(dims["dirs"]) for mods in xrange(dims["mods"])]

        cub_selections = [[dirs, mods, tchunk, slice(None), achunk, bchunk, corr_slice][-reqdims:]
                            for dirs in xrange(dims["dirs"]) for mods in xrange(dims["mods"])]

        n_sel = len(col_selections)

        # The following takes the arbitrarily ordered data from the MS and places it into a N-D 
        # data structure (correlation matrix).

        for col_selection, cub_selection in zip(col_selections, cub_selections):

            if self.ncorr == 4:
                out_arr[cub_selection] = colsel = column[col_selection]
                cub_selection[-3], cub_selection[-2] = cub_selection[-2], cub_selection[-3]
                if dtype == self.ctype:
                    out_arr[cub_selection] = colsel.conj()[..., (0, 2, 1, 3)]
                else:
                    out_arr[cub_selection] = colsel[..., (0, 2, 1, 3)]
            
            elif self.ncorr == 2:
                out_arr[cub_selection] = colsel = column[col_selection]
                cub_selection[-3], cub_selection[-2] = cub_selection[-2], cub_selection[-3]
                if dtype == self.ctype:
                    out_arr[cub_selection] = colsel.conj()
                else:
                    out_arr[cub_selection] = colsel
            
            elif self.ncorr == 1:
                out_arr[cub_selection] = colsel = column[col_selection][..., (0,0)]
                cub_selection[-3], cub_selection[-2] = cub_selection[-2], cub_selection[-3]
                if dtype == self.ctype:
                    out_arr[cub_selection] = colsel.conj()
                else:
                    out_arr[cub_selection] = colsel

        # This zeros the diagonal elements in the "baseline" plane. This is purely a precaution - 
        # we do not want autocorrelations on the diagonal.
        
        out_arr[..., range(self.nants), range(self.nants), :] = zeroval

        return out_arr


    def _cube_to_column(self, column, in_arr, rows, freqs, flags=False):
        """
        Converts the calibrated measurement matrix back into the MS style.

        Args:
            in_arr (np.array): Input array which is to be made MS friendly.
            rows: row indices or slice
            freqs: freq indices or slice
            flags: if True, input array is a flag cube (i.e. no correlation axes)
        """
        tchunk = self.times[rows]
        tchunk -= tchunk[0]  # is this correct -- does in_array start from beginning of chunk?
        achunk = self.antea[rows]
        bchunk = self.anteb[rows]
        # flag cube has no correlation axis, so copy it into coutput column
        if flags:
            column[rows, freqs, :] = in_arr[tchunk, :, achunk, bchunk, np.newaxis]
        # for other cubes, reform the 2,2 axes at end into 4
        else:
            chunk = in_arr[tchunk, :, achunk, bchunk, :]
            newshape = list(chunk.shape[:-2]) + [chunk.shape[-2]*chunk.shape[-1]]
            chunk = chunk.reshape(newshape)
            if self.ncorr == 4:
                column[rows, freqs, :] = chunk
            elif self.ncorr == 2:
                column[rows, freqs, :] = chunk[..., ::3]  # 2 corrs -- take elements 0,3
            elif self.ncorr == 1:                         # 1 corr -- take element 0
                column[rows, freqs, :] = chunk[..., :1]


class ReadModelHandler:

    def __init__(self, ms_name, data_column, sm_name, model_column, output_column=None,
                 taql=None, fid=None, ddid=None, flagopts={}, precision="32", ddes=False, 
                 weight_column=None, beam_pattern=None, beam_l_axis=None, beam_m_axis=None):

        self.ms_name = ms_name
        self.sm_name = sm_name
        self.beam_pattern = beam_pattern
        self.beam_l_axis = beam_l_axis
        self.beam_m_axis = beam_m_axis

        self.fid = fid if fid is not None else 0

        self.ms = pt.table(self.ms_name, readonly=False, ack=False)

        print>>log, ModColor.Str("reading MS %s"%self.ms_name, col="green")

        self._anttab = pt.table(self.ms_name + "::ANTENNA", ack=False)
        self._fldtab = pt.table(self.ms_name + "::FIELD", ack=False)
        self._spwtab = pt.table(self.ms_name + "::SPECTRAL_WINDOW", ack=False)
        self._poltab = pt.table(self.ms_name + "::POLARIZATION", ack=False)
        self._ddesctab = pt.table(self.ms_name + "::DATA_DESCRIPTION", ack=False)
        self._feedtab = pt.table(self.ms_name + "::FEED", ack=False)

        self.ctype = np.complex128 if precision=="64" else np.complex64
        self.ftype = np.float64 if precision=="64" else np.float32
        self.nfreq = self._spwtab.getcol("NUM_CHAN")[0]
        self.ncorr = self._poltab.getcol("NUM_CORR")[0]
        self.nants = self._anttab.nrows()

        self._nchans = self._spwtab.getcol("NUM_CHAN")
        self._rfreqs = self._spwtab.getcol("REF_FREQUENCY")
        self._chanfr = self._spwtab.getcol("CHAN_FREQ")
        self._antpos = self._anttab.getcol("POSITION")
        self._phadir = self._fldtab.getcol("PHASE_DIR", startrow=self.fid, nrow=1)[0][0]
        self._poltype = np.unique(self._feedtab.getcol('POLARIZATION_TYPE')['array'])
        
        if np.any([pol in self._poltype for pol in ['L','l','R','r']]):
            self._poltype = "circular"
        elif np.any([pol in self._poltype for pol in ['X','x','Y','y']]):
            self._poltype = "linear"
        else:
            print>>log,"  unsupported feed type. Terminating."
            sys.exit()

        # print some info on MS layout
        print>>log,"  detected {} feeds".format(self._poltype)
        print>>log,"  fields are "+", ".join(["{}{}: {}".format('*' if i==fid else "",i,name) for i, name in enumerate(self._fldtab.getcol("NAME"))])
        self._spw_chanfreqs = self._spwtab.getcol("CHAN_FREQ")  # nspw x nfreq array of frequencies
        print>>log,"  {} spectral windows of {} channels each ".format(*self._spw_chanfreqs.shape)

        # figure out DDID range
        self._ddids = _parse_range(ddid, self._ddesctab.nrows())

        # use TaQL to select subset
        self.taql = self.build_taql(taql, fid, self._ddids)

        if self.taql:
            print>> log, "Applying TAQL query: %s" % self.taql
            self.data = self.ms.query(self.taql)
        else:
            self.data = self.ms

        self.nrows = self.data.nrows()

        self._datashape = (self.nrows, self.nfreq, self.ncorr)

        if not self.nrows:
            raise ValueError("MS selection returns no rows")

        self.ntime = len(np.unique(self.fetch("TIME")))

        self._ddid_spw = self._ddesctab.getcol("SPECTRAL_WINDOW_ID")
        # select frequencies corresponding to DDID range
        self._ddid_chanfreqs = np.array([self._spw_chanfreqs[self._ddid_spw[ddid]] for ddid in self._ddids ])

        print>>log,"%d antennas, %d rows, %d/%d DDIDs, %d timeslots, %d channels, %d corrs" % (self.nants,
                    self.nrows, len(self._ddids), self._ddesctab.nrows(), self.ntime, self.nfreq, self.ncorr)
        print>>log,"DDID central frequencies are at {} GHz".format(
                    " ".join(["%.2f"%(self._ddid_chanfreqs[i][self.nfreq/2]*1e-9) for i in range(len(self._ddids))]))
        self.nddid = len(self._ddids)


        self.data_column = data_column
        self.model_column = model_column
        self.weight_column = weight_column
        self.output_column = output_column
        self.simulate = bool(self.sm_name)
        self.use_ddes = ddes

        # figure out flagging situation
        if "BITFLAG" in self.ms.colnames():
            if flagopts["reinit-bitflags"]:
                self.ms.removecols("BITFLAG")
                if "BITFLAG_ROW" in self.ms.colnames():
                    self.ms.removecols("BITFLAG_ROW")
                print>> log, ModColor.Str("Removing BITFLAG column, since --flags-reinit-bitflags is set.")
                bitflags = None
            else:
                bitflags = flagging.Flagsets(self.ms)
        else:
            bitflags = None
        apply_flags  = flagopts.get("apply")
        save_bitflag = flagopts.get("save")
        auto_init    = flagopts.get("auto-init")

        self._reinit_bitflags = flagopts["reinit-bitflags"]
        self._apply_flags = self._apply_bitflags = self._save_bitflag = self._auto_fill_bitflag = None

        # no BITFLAG. Should we auto-init it?

        if auto_init:
            if not bitflags:
                self._add_column("BITFLAG", like_type='int')
                if "BITFLAG_ROW" not in self.ms.colnames():
                    self._add_column("BITFLAG_ROW", like_col="FLAG_ROW", like_type='int')
                self._reopen()
                bitflags = flagging.Flagsets(self.ms)
                self._auto_fill_bitflag = bitflags.flagmask(auto_init, create=True)
                print>> log, ModColor.Str("Will auto-fill new BITFLAG '{}' ({}) from FLAG/FLAG_ROW".format(auto_init, self._auto_fill_bitflag), col="green")
            else:
                self._auto_fill_bitflag = bitflags.flagmask(auto_init, create=True)
                print>> log, "BITFLAG column found. Will auto-fill with '{}' ({}) from FLAG/FLAG_ROW if not filled".format(auto_init, self._auto_fill_bitflag)

        # OK, we have BITFLAG somehow -- use these

        if bitflags:
            self._apply_flags = None
            self._apply_bitflags = 0
            if apply_flags:
                # --flags-apply specified as a bitmask, or a string, or a list of strings
                if type(apply_flags) is int:
                    self._apply_bitflags = apply_flags
                else:
                    if type(apply_flags) is str:
                        apply_flags = apply_flags.split(",")
                    for fset in apply_flags:
                        self._apply_bitflags |= bitflags.flagmask(fset)
            if self._apply_bitflags:
                print>> log, ModColor.Str("Applying BITFLAG {} ({}) to input data".format(apply_flags, self._apply_bitflags), col="green")
            else:
                print>> log, ModColor.Str("No flags will be read, since --flags-apply was not set.")
            if save_bitflag:
                self._save_bitflag = bitflags.flagmask(save_bitflag, create=True)
                print>> log, ModColor.Str("Will save new flags into BITFLAG '{}' ({}), and into FLAG/FLAG_ROW".format(save_bitflag, self._save_bitflag), col="green")

        # else no BITFLAG -- fall back to using FLAG/FLAG_ROW if asked, but definitely can'tr save

        else:
            if save_bitflag:
                raise RuntimeError("No BITFLAG column in this MS. Either use --flags-auto-init to insert one, or disable --flags-save.")
            self._apply_flags = bool(apply_flags)
            self._apply_bitflags = 0
            if self._apply_flags:
                print>> log, ModColor.Str("No BITFLAG column in this MS. Using FLAG/FLAG_ROW.")
            else:
                print>> log, ModColor.Str("No flags will be read, since --flags-apply was not set.")

        self.gain_dict = {}

    def build_taql(self, taql=None, fid=None, ddid=None):

        if taql:
            taqls = [ "(" + taql +")" ]
        else:
            taqls = []

        if fid is not None:
            taqls.append("FIELD_ID == %d" % fid)

        if ddid is not None:
            if isinstance(ddid,(tuple,list)):
                taqls.append("DATA_DESC_ID IN [%s]" % ",".join(map(str,ddid)))
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

    def define_chunk(self, tdim=1, fdim=1, chunk_by_scan=True, min_chunks_per_tile=4):
        """
        Fetches indexing columns (TIME, DDID, ANTENNA1/2) and defines the chunk dimensions for the data.

        Args:
            tdim (int): Timeslots per chunk.
            fdim (int): Frequencies per chunk.
            single_chunk_id:   If set, iterator will yield only the one specified chunk. Useful for debugging.
            chunk_by_scan:  If True, chunks will have boundaries by SCAN_NUMBER
        """
        self.antea = self.fetch("ANTENNA1")
        self.anteb = self.fetch("ANTENNA2")
        # read TIME and DDID columns, because those determine our chunking strategy
        self.time_col = self.fetch("TIME")
        self.ddid_col = self.fetch("DATA_DESC_ID")
        print>> log, "  read indexing columns"
        # list of unique times
        self.uniq_times = np.unique(self.time_col)
        # timeslot index (per row, each element gives index of timeslot)
        self.times = np.empty_like(self.time_col, dtype=np.int32)
        for i, t in enumerate(self.uniq_times):
            self.times[self.time_col == t] = i
        print>> log, "  built timeslot index"

        self.chunk_tdim = tdim
        self.chunk_fdim = fdim

        # TODO: this assumes each DDID has the same number of channels. I don't know of cases where it is not true,
        # but, technically, this is not precluded by the MS standard. Need to handle this one day
        self.chunk_find = range(0, self.nfreq, self.chunk_fdim)
        self.chunk_find.append(self.nfreq)
        num_freq_chunks = len(self.chunk_find) - 1

        print>> log, "  using %d freq chunks: %s" % (num_freq_chunks, " ".join(map(str, self.chunk_find)))

        # Constructs a list of timeslots at which we cut our time chunks. Use scans if specified, else
        # simply break up all timeslots

        if chunk_by_scan:
            scan_chunks = self.check_contig()
            timechunks = []
            for scan_num in xrange(len(scan_chunks) - 1):
                timechunks.extend(range(scan_chunks[scan_num], scan_chunks[scan_num+1], self.chunk_tdim))
        else:
            timechunks = range(0, self.times[-1], self.chunk_tdim)
        timechunks.append(self.times[-1]+1)        
        
        print>>log,"  found %d time chunks: %s"%(len(timechunks)-1, " ".join(map(str, timechunks)))

        # Number of timeslots per time chunk
        self.chunk_ntimes = []
        
        # Unique timestamps per time chunk
        self.chunk_timestamps = []
        
        # For each time chunk, create a mask for associated rows.
        
        timechunk_mask = {}
        
        for tchunk in range(len(timechunks) - 1):
            ts0, ts1 = timechunks[tchunk:tchunk + 2]
            timechunk_mask[tchunk] = (self.times>=ts0) & (self.times<ts1)
            self.chunk_ntimes.append(ts1-ts0)
            self.chunk_timestamps.append(np.unique(self.times[timechunk_mask[tchunk]]))


        # now make list of "row chunks": each element will be a tuple of (ddid, time_chunk_number, rowlist)

        chunklist = []

        for ddid in self._ddids:
            ddid_rowmask = self.ddid_col==ddid

            for tchunk in range(len(timechunks)-1):
                rows = np.where(ddid_rowmask & timechunk_mask[tchunk])[0]
                if rows.size:
                    chunklist.append(RowChunk(ddid, tchunk, rows))

        print>>log,"  generated {} row chunks based on time and DDID".format(len(chunklist))

        # init this, for compatibility with the chunk iterator below
        self.chunk_rind = OrderedDict([ ((chunk.ddid, chunk.tchunk), chunk.rows) for chunk in chunklist])

        # re-sort these row chunks into naturally increasing order (by first row of each chunk)
        def _compare_chunks(a, b):
            return cmp(a.rows[0], b.rows[0])
        chunklist.sort(cmp=_compare_chunks)

        # now, break the row chunks into tiles. Tiles are an "atom" of I/O. First, we try to define each tile as a
        # sequence of overlapping row chunks (i.e. chunks such that the first row of a subsequent chunk comes before
        # the last row of the next chunk). Effectively, if DDIDs are interleaved with timeslots, then all per-DDIDs
        # chunks will be grouped into a single tile.
        # It is also possible that we end up with one chunk = one tile (i.e. no chunks overlap).
        tile_list = []
        for chunk in chunklist:
            # if rows do not overlap, start new tile with this chunk
            if not tile_list or chunk.rows[0] > tile_list[-1].last_row:
                tile_list.append(Tile(self,chunk))
            # else extend previous tile
            else:
                tile_list[-1].append(chunk)

        print>> log, "  row chunks yield {} potential tiles".format(len(tile_list))

        # now, for effective I/O and parallelisation, we need to have a minimum amount of chunks per tile.
        # Coarsen our tiles to achieve this
        coarser_tile_list = []
        for tile in tile_list:
            # start new "coarse tile" if previous coarse tile already has the min number of chunks
            if not coarser_tile_list or len(coarser_tile_list[-1].rowchunks)*num_freq_chunks >= min_chunks_per_tile:
                coarser_tile_list.append(tile)
            else:
                coarser_tile_list[-1].merge(tile)

        Tile.tile_list = coarser_tile_list
        for tile in Tile.tile_list:
            tile.finalize()

        print>> log, "  coarsening this to {} tiles (min {} chunks per tile)".format(len(Tile.tile_list), min_chunks_per_tile)

    def check_contig(self):

        scan = self.fetch("SCAN_NUMBER")

        if np.all(scan==scan[0]):
            scan_t = [0, self.ntime]
        else:
            scan_i = np.where(np.roll(scan,1)!=scan)[0]
            scan_t = list(self.times[scan_i])
            scan_t.append(self.ntime)

        return scan_t

    def flag3_to_col(self, flag3):
        """
        Converts a 3D flag cube (ntime, nddid, nchan) back into the MS style.

        Args:
            flag3 (np.array): Input array which is to be made MS friendly.

        Returns:
            bool array, same shape as self.obvis
        """

        ntime, nddid, nchan = flag3.shape

        flagout = np.zeros(self._datashape, bool)

        for ddid in xrange(nddid):
            ddid_rows = self.ddid_col == ddid
            for ts in xrange(ntime):
                # find all rows associated with this DDID and timeslot
                rows = ddid_rows & (self.times == ts)
                if rows.any():
                    flagout[rows, :, :] = flag3[ts, ddid, :, np.newaxis]

        return flagout

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

    def _add_column (self, col_name, like_col="DATA", like_type=None):
        """
        Inserts new column ionto MS.
        col_name (str): Name of target column.
        like_col (str): Column will be patterned on the named column.
        like_type (str or None): if set, column type will be changed

        Returns True if new column was inserted
        """
        if col_name not in self.ms.colnames():
            # new column needs to be inserted -- get column description from column 'like_col'
            print>> log, "  inserting new column %s" % (col_name)
            desc = self.ms.getcoldesc(like_col)
            desc["name"] = col_name
            desc['comment'] = desc['comment'].replace(" ", "_")  # got this from Cyril, not sure why
            # if a different type is specified, insert that
            if like_type:
                desc['valueType'] = like_type
            self.ms.addcols(desc)
            return True
        return False

    def unlock(self):
        if self.taql:
            self.data.unlock()
        self.ms.unlock()

    def lock(self):
        self.ms.lock()
        if self.taql:
            self.data.lock()

    def close(self):
        if self.taql:
            self.data.close()
        self.ms.close()

    def _reopen(self):
        """Reopens the MS. Unfortunately, this is needed when new columns are added"""
        self.close()
        self.ms = self.data = pt.table(self.ms_name, readonly=False, ack=False)
        if self.taql:
            self.data = self.ms.query(self.taql)

    def save_flags(self, flags):
        """
        Saves flags to column in MS.

        Args
        flags (np.array): Values to be written to column.
        bitflag (str or int): Bitflag to save to.
        """
        print>>log,"Writing out new flags"
        bflag_col = self.fetch("BITFLAG")
        # raise specified bitflag
        print>> log, "  updating BITFLAG column with flagbit %d"%self._save_bitflag
        bflag_col[flags] |= self._save_bitflag
        self.data.putcol("BITFLAG", bflag_col)
        print>>log, "  updating BITFLAG_ROW column"
        self.data.putcol("BITFLAG_ROW", np.bitwise_and.reduce(bflag_col, axis=(-1,-2)))
        flag_col = bflag_col != 0
        print>> log, "  updating FLAG column ({:.2%} visibilities flagged)".format(flag_col.sum()/float(flag_col.size))
        self.data.putcol("FLAG", flag_col)
        flag_row = flag_col.all(axis=(-1,-2))
        print>> log, "  updating FLAG_ROW column ({:.2%} rows flagged)".format(flag_row.sum()/float(flag_row.size))
        self.data.putcol("FLAG_ROW", flag_row)

