# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
import numpy as np
from collections import OrderedDict
import traceback
import itertools

from cubical.tools import shared_dict
from cubical.flagging import FL
from cubical import data_handler

from cubical.tools import logger, ModColor
log = logger.getLogger("data_handler")

## TERMINOLOGY:
## A "chunk" is data for one DDID, a range of timeslots (thus, a subset of the MS rows), and a
## slice of channels. Chunks are the basic parallelization unit. Solver deals with a chunk of data.
##
## A "row chunk" is data for one DDID, a range of timeslots, and *all* channels. One can imagine a
## row chunk as a "horizontal" vector of chunks across frequency.
##
## A "tile" is a collection of row chunks that are adjacent in time and/or DDID. One can imagine a
## tile as a vertical stack of row chunks

class RowChunk(object):
    """ Very basic helper class. Encapsulates a row chunk -- a set of rows representing a time interval, and a single
    DDID, split into a number of frequency chunks.

    A row chunk also knows about row rebinning.
    """

    def __init__(self, ddid, tchunk, timeslice, rows, rows0):
        """
        Initialises a RowChunk.

        Args:
            ddid (int):
                DDID index for the RowChunk.
            tchunk (int):
                Time index for the RowChunk.
            rows (np.ndarray):
                An (nrows_in_chunk) size array of rebinned row indices.
            rows0 (np.ndarray):
                An (nrows_in_chunk) size array of original row indices.
        """

        self.ddid, self.tchunk, self.timeslice, self.rows, self.rows0 = ddid, tchunk, timeslice, rows, rows0


class MSTile(object):
    """
    Helper class which encapsulates a tile. A tile is a sequence of row chunks that's read and
    written as a unit.
    """
    class Subset(object):
        """
        A Subset represents a part of the tile (one or more DDIDs) with the same channel structure.

        Subsets are read from the MS in one go, and predicted by Montblanc in one go.
        """
        def __init__(self, tile, label, datadict, ms_rows, rows0):
            self.tile = tile
            self.datadict = datadict
            self.rows0 = rows0              # row numbers in original (sorted) MS
            self.label = label
            self.nants = tile.nants
            # subsets of rebinned rows
            self.ddid_col = self.tile.dh.ddid_col[ms_rows]
            self.time_col = self.tile.dh.time_col[ms_rows]
            self.antea = self.tile.dh.antea[ms_rows]
            self.anteb = self.tile.dh.anteb[ms_rows]
            self.times = self.tile.dh.times[ms_rows]
            self.nrows = len(self.times)
            # first DDID in subset
            self.first_ddid = self.ddid_col[0]
            self.nfreq = len(tile.dh.chanfreqs[self.first_ddid])
            self.ncorr = tile.dh.ncorr

            # row map for rebinning -- label of None means subset is full tile
            if label is None:
                self.rebin_row_map = tile.rebin_row_map
            else:
                self.rebin_row_map = tile.rebin_row_map[rows0]
                # since the rebinned map is now sparse in the output because we've got a subset of input rows,
                # uniquify it so it becomes contiguous in the output.
                # Note that '-' signs need to be preserved
                sign = np.sign(self.rebin_row_map)
                self.rebin_row_map, _, _ = data_handler.uniquify(abs(self.rebin_row_map))
                self.rebin_row_map *= sign

            # channel map for rebinning
            self.rebin_chan_map = tile.dh.rebin_chan_maps[self.first_ddid]
            if self.rebin_chan_map is None:
                self.rebin_chan_map = np.arange(0, len(tile.dh.chanfreqs[self.first_ddid]), dtype=np.int64)

            # filled in by self.load_montblanc_models below
            self._mb_measet_src = None

        def upsample(self, data):
            """Helper method. Upsamples an array back to full resolution"""
            if not self.tile.dh.do_freq_rebin and not self.tile.dh.do_time_rebin:
                return data
            shape = len(self.rebin_row_map), len(self.rebin_chan_map), data.shape[2]
            print>>log(1),"upsampling to {} rows and {} channels".format(shape[0], shape[1])
            return data[self.rebin_row_map[:, np.newaxis],
                        self.rebin_chan_map[np.newaxis, :], None].reshape(shape)

        def load_montblanc_models(self, uvwco, loaded_models, model_source, cluster, imod, idir):
            """
            Invoke Montblanc to compute model visibilities

            Args:
                uvwco:
                loaded_models:
                model_source:
                cluster:
                imod:
                idir:

            Returns:

            """

            import MBTiggerSim
            from montblanc.impl.rime.tensorflow.sources import CachedSourceProvider, FitsBeamSourceProvider

            # setup montblanc machinery once per subset (may be called multiple times for different models)
            if not self._mb_measet_src:
                # massage data columns into Montblanc-friendly shapes
                # Given data, we need to make sure that it looks the way MB wants it to.
                # First step - check the number of rows.

                n_bl = (self.nants * (self.nants - 1)) / 2
                uniq_times = np.unique(self.times)
                ntime = len(uniq_times)
                uniq_time_col = np.unique(self.time_col)
                t_offset = uniq_times[0]

                # The row identifiers determine which rows in the SORTED/ALL ROWS are required for the data
                # that is present in the MS. Essentially, they allow for the selection of an array of a size
                # matching that of the observed data. First term determines the offset by ddid, the second
                # is the offset by time, and the last turns antea and anteb into a unique offset per
                # baseline.

                ddid_index, uniq_ddids, _ = data_handler.uniquify(self.ddid_col)

                self._freqs = np.array([self.tile.dh.chanfreqs[ddid] for ddid in uniq_ddids])

                self._row_identifiers = ddid_index * n_bl * ntime + (self.times - self.times[0]) * n_bl + \
                                  (-0.5 * self.antea ** 2 + (self.nants - 1.5) * self.antea + self.anteb - 1).astype(
                                      np.int32)

                # make full list of row indices in Montblanc-compliant order (ddid-time-ant1-ant2)
                full_index = [(p, q, t, d) for d in xrange(len(uniq_ddids)) for t in uniq_times
                              for p in xrange(self.nants) for q in xrange(self.nants)
                              if p < q]

                self._expected_nrows = len(full_index)

                # and corresponding full set of indices
                full_row_set = set(full_index)

                print>> log(1), "  {} rows ({} expected for {} timeslots, {} baselines and {} DDIDs)".format(
                    self.nrows, self._expected_nrows, ntime, n_bl, len(uniq_ddids))

                # make mapping from existing indices -> row numbers, omitting autocorrelations
                current_row_index = {(p, q, t, d): row for row, (p, q, t, d) in
                                     enumerate(zip(self.antea, self.anteb, self.times, ddid_index)) if p != q}

                # do we need to add fake rows for missing data?
                missing = full_row_set.difference(current_row_index.iterkeys())
                nmiss = len(missing)

                if nmiss:
                    print>> log(1), "  {} rows will be padded in for Montblanc".format(nmiss)
                    # pad up columns
                    uvwco = np.concatenate((uvwco, [[0, 0, 0]] * nmiss))
                    antea = np.concatenate((self.antea,
                                            np.array([p for (p, q, t, d) in missing])))
                    anteb = np.concatenate((self.anteb,
                                            np.array([q for (p, q, t, d) in missing])))
                    time_col = np.concatenate((self.time_col,
                                               np.array([uniq_time_col[t - t_offset]
                                                         for (p, q, t, d) in missing])))
                    ddid_index = np.concatenate((ddid_index,
                                               np.array([d for (p, q, t, d) in missing])))
                    # extend row index
                    current_row_index.update({idx: (row + self.nrows) for row, idx in enumerate(missing)})
                else:
                    antea, anteb, time_col = self.antea, self.anteb, self.time_col

                # lookup each index in Montblanc order, convert it to a row number
                self._mb_sorted_ind = np.array([current_row_index[idx] for idx in full_index])

                self._mb_measet_src = MBTiggerSim.MSSourceProvider(self.tile, time_col, antea, anteb, ddid_index, uvwco,
                                                       self._freqs, self._mb_sorted_ind, len(self.time_col))

                self._mb_cached_ms_src = CachedSourceProvider(self._mb_measet_src,
                                                 cache_data_sources=["parallactic_angles"],
                                                 clear_start=False, clear_stop=False)
                if self.tile.dh.beam_pattern:
                    self._mb_arbeam_src = FitsBeamSourceProvider(self.tile.dh.beam_pattern,
                                                        self.tile.dh.beam_l_axis,
                                                        self.tile.dh.beam_m_axis)
                else:
                    self._mb_arbeam_src = None

            print>> log(0), "  computing visibilities for {}".format(model_source)
            # setup Montblanc computation for this LSM
            tigger_source = model_source
            cached_src = CachedSourceProvider(tigger_source, clear_start=True, clear_stop=True)
            srcs = [self._mb_cached_ms_src, cached_src]
            if self._mb_arbeam_src:
                srcs.append(self._mb_arbeam_src)

            # make a sink with an array to receive visibilities
            ndirs = model_source._nclus
            model_shape = (ndirs, 1, self._expected_nrows, self.nfreq, self.tile.dh.ncorr)
            full_model = np.zeros(model_shape, self.tile.dh.ctype)
            column_snk = MBTiggerSim.ColumnSinkProvider(self.tile.dh, self._freqs.shape, full_model, self._mb_sorted_ind)
            snks = [column_snk]

            for direction in xrange(ndirs):
                tigger_source.set_direction(direction)
                column_snk.set_direction(direction)
                MBTiggerSim.simulate(srcs, snks, self.tile.dh.mb_opts)

            # now associate each cluster in the LSM with an entry in the loaded_models cache
            loaded_models[model_source] = { clus: full_model[i, 0, self._row_identifiers, :, :]
                                                for i, clus in enumerate(tigger_source._cluster_keys)}

            model = loaded_models[model_source][cluster]
            print>> log(1), "  using {}{} for model {} direction {}".format(model_source,
                                "" if not cluster else ("()" if cluster == 'die' else "({})".format(cluster)),
                                imod, idir)

            # release memory asap
            del column_snk, snks

            return model

        def _column_to_cube(self, column, chunk_tdim, chunk_fdim, rows, freqs, dtype, zeroval=0, reqdims=6,
                            allocator=np.empty):
            """
            Converts input data into N-dimensional measurement matrices.

            Args:
                column (np.ndarray):
                    column array from which this will be filled
                chunk_tdim (int):
                    Timeslots per chunk.
                chunk_fdim (int):
                    Frequencies per chunk.
                rows (np.ndarray):
                    Row slice (or set of indices).
                freqs (slice):
                    Frequency slice.
                dtype (various):
                    Data type of the resulting measurement matrix.
                zeroval (various, optional):
                    Null value with which to fill missing array elements.
                reqdims (int):
                    Required number of output dimensions.
                allocator (callable):
                    Function to call to allocate empty array. Must have signature (shape,dtype).
                    Default is np.empty.

            Returns:
                np.ndarray:
                    Output cube of with reqdims axes.
            """

            # Start by establishing the possible dimensions and those actually present. Dimensions which
            # are not present are set to one, for flexibility reasons. Output shape is determined by
            # reqdims, which selects dimensions in reverse order from (ndir, nmod, nt, nf, na, na, nc).
            # NOTE: The final dimension will be reshaped into 2x2 blocks outside this function.

            col_ndim = column.ndim

            possible_dims = ["dirs", "mods", "rows", "freqs", "cors"]

            dims = {possible_dims[-i]: column.shape[-i] for i in xrange(1, col_ndim + 1)}

            dims.setdefault("mods", 1)
            dims.setdefault("dirs", 1)

            out_shape = [dims["dirs"], dims["mods"], chunk_tdim, chunk_fdim, self.nants, self.nants, 2, 2]
            out_shape = out_shape[-reqdims:]

            # this shape has "4" at the end instead of 2,2
            out_shape_4 = out_shape[:-2] + [4]

            # Creates empty N-D array into which the column data can be packed.
            # Place the antenna axes first, for better performance in the kernels

            out_arr0 = allocator(out_shape, dtype)
            out_arr0.fill(zeroval)

            # view onto output array with 4 corr axis rather than 2,2
            out_arr = out_arr0.reshape(out_shape_4)

            # Grabs the relevant time and antenna info.

            achunk = self.antea[rows]
            bchunk = self.anteb[rows]
            tchunk = self.times[rows]
            tchunk -= np.min(tchunk)

            # Creates lists of selections to make subsequent selection from column and out_arr easier.

            corr_slice = slice(None) if self.ncorr == 4 else slice(None, None, 3)

            col_selections = [[dirs, mods, rows, freqs, slice(None)][-col_ndim:]
                              for dirs in xrange(dims["dirs"]) for mods in xrange(dims["mods"])]

            cub_selections = [[dirs, mods, tchunk, slice(None), achunk, bchunk, corr_slice][-(reqdims - 1):]
                              for dirs in xrange(dims["dirs"]) for mods in xrange(dims["mods"])]

            # The following takes the arbitrarily ordered data from the MS and places it into a N-D
            # data structure (correlation matrix).

            for col_selection, cub_selection in zip(col_selections, cub_selections):

                if self.ncorr == 4:
                    out_arr[cub_selection] = colsel = column[col_selection]
                    cub_selection[-3], cub_selection[-2] = cub_selection[-2], cub_selection[-3]
                    if np.iscomplexobj(out_arr):
                        out_arr[cub_selection] = colsel.conj()[..., (0, 2, 1, 3)]
                    else:
                        out_arr[cub_selection] = colsel[..., (0, 2, 1, 3)]

                elif self.ncorr == 2:
                    out_arr[cub_selection] = colsel = column[col_selection]
                    cub_selection[-3], cub_selection[-2] = cub_selection[-2], cub_selection[-3]
                    if np.iscomplexobj(out_arr):
                        out_arr[cub_selection] = colsel.conj()
                    else:
                        out_arr[cub_selection] = colsel

                elif self.ncorr == 1:
                    out_arr[cub_selection] = colsel = column[col_selection][..., (0, 0)]
                    cub_selection[-3], cub_selection[-2] = cub_selection[-2], cub_selection[-3]
                    if np.iscomplexobj(out_arr):
                        out_arr[cub_selection] = colsel.conj()
                    else:
                        out_arr[cub_selection] = colsel

            # This zeros the diagonal elements in the "baseline" plane. This is purely a precaution -
            # we do not want autocorrelations on the diagonal.

            out_arr[..., range(self.nants), range(self.nants), :] = zeroval

            return out_arr0

        def _cube_to_column(self, column, in_arr, rows, freqs, flags=False):
            """
            Converts a measurement matrix back into an MS style column.

            Args:
                in_arr (np.ndarray):
                    Input array which is to be made MS friendly.
                rows (np.ndarray):
                    Row indices or slice.
                freqs (slice):
                    Frequency slice.
                flags (bool, optional):
                    If True, input array is a flag cube (i.e. no correlation axes).
            """

            tchunk = self.times[rows]
            tchunk -= tchunk[0]  # is this correct -- does in_array start from beginning of chunk?
            achunk = self.antea[rows]
            bchunk = self.anteb[rows]

            # Flag cube has no correlation axis, so copy it into output column.

            if flags:
                column[rows, freqs, :] = in_arr[tchunk, :, achunk, bchunk, np.newaxis]

            # For other cubes, rehape the last two axes into one (faltten 2x2 correlation block).
            else:
                chunk = in_arr[tchunk, :, achunk, bchunk, :]
                newshape = list(chunk.shape[:-2]) + [chunk.shape[-2] * chunk.shape[-1]]
                chunk = chunk.reshape(newshape)
                if self.ncorr == 4:
                    column[rows, freqs, :] = chunk
                elif self.ncorr == 2:  # 2 corr -- take elements 0,3
                    column[rows, freqs, :] = chunk[..., ::3]
                elif self.ncorr == 1:  # 1 corr -- take element 0
                    column[rows, freqs, :] = chunk[..., :1]


    def __init__(self, handler, chunk):
        """
        Initialises a tile and sets the first row chunk.

        Args:
            handler (:obj:`~cubical.data_handler.ReadModelHandler`):
                Data hander object.
            chunk (:obj:`~cubical.data_handler.RowChunk`):
                Row chunk which is used to initialise the tile.
        """

        self.dh = handler
        self.rowchunks = [chunk]
        self.first_row0 = chunk.rows0[0]
        self.last_row0  = chunk.rows0[-1]
        self._rows_adjusted = False
        self._updated = False
        self.data = None
        self.label = "tile"

    def append(self, chunk):
        """
        Appends a row chunk to a tile.

        Args:
            chunk (:obj:`~cubical.data_handler.RowChunk`):
                Row chunk which will be appended to the assosciated tile.
        """

        self.rowchunks.append(chunk)
        self.first_row0 = min(self.first_row0, chunk.rows0[0])
        self.last_row0  = max(self.last_row0, chunk.rows0[-1])

    def total_tf_chunks(self):
        """Returns total number of time/frequency chunks in the tile, counting the freq chunks in each DDID"""
        return sum([len(self.dh.freqchunks[chunk.ddid]) for chunk in self.rowchunks])

    def merge(self, other):
        """
        Merges another tile into this one.

        Args:
            other (:obj:`~cubical.data_handler.Tile`):
                Tile which will be merged.
        """

        self.rowchunks += other.rowchunks
        self.first_row0 = min(self.first_row0, other.first_row0)
        self.last_row0  = max(self.last_row0, other.last_row0)

    def finalize(self, label=None):
        """
        Creates a list of chunks within the tile that can be iterated over and creates a list of
        chunk labels.

        This also adjusts the row indices of all row chunks so that they become relative to the
        start of the tile.

        Note that this is called in the main process, so all structures set up here will be inherited
        by the I/O and solver workers.
        """
        if label is not None:
            self.label = label

        # row rebin map, relative to start (row0) of tile
        self.rebin_row_map = self.dh.rebin_row_map[self.first_row0:self.last_row0+1]
        first_row, last_row = abs(self.rebin_row_map[0]), abs(self.rebin_row_map[-1])
        # first rebinned row is also 0
        self.rebin_row_map = np.sign(self.rebin_row_map)*(abs(self.rebin_row_map) - first_row)

        self._data_dict_name = "DATA:{}:{}".format(self.first_row0, self.last_row0)

        # Adjust rebinned row indices so that first rebinned row of tile is 0
        if not self._rows_adjusted:
            for rowchunk in self.rowchunks:
                rowchunk.rows -= first_row
            self._rows_adjusted = True

        # Note that first_row0, last_row0 amd rowchunk.rows0 continues to refer to MS rows

        # list of DDIDs in this tile
        self.ddids = np.unique([rowchunk.ddid for rowchunk in self.rowchunks])
        # reverse index: from DDID to its number in the self.ddids list
        self._ddid_reverse = { ddid:num for num, ddid in enumerate(self.ddids) }

        # Create a dict of { chunk_label: rows, chan0, chan1 } for all chunks in this tile.

        self._chunk_dict = OrderedDict()
        self._chunk_indices = {}


        # collect row chunks and freqs, and also row numbers per each DDID
        ddid_rows0 = {}
        for rowchunk in self.rowchunks:
            ddid_rows0.setdefault(rowchunk.ddid, set()).update(rowchunk.rows0)
            freqchunks = self.dh.freqchunks[rowchunk.ddid]
            for ifreq,chan0 in enumerate(freqchunks):
                key = "D{}T{}F{}".format(rowchunk.ddid, rowchunk.tchunk, ifreq)
                chan1 = freqchunks[ifreq+1] if ifreq<len(freqchunks)-1 else len(self.dh.chanfreqs[rowchunk.ddid])
                self._chunk_dict[key] = rowchunk, chan0, chan1
                self._chunk_indices[key] = rowchunk.tchunk, (rowchunk.ddid, chan0, chan1)

        self.nants = self.dh.nants
        self.ncorr = self.dh.ncorr

        # set up per-DDID subsets. These will be inherited by workers.

        # gives list of subsets to be read in, as a tuple of shared dict name, MS row subset
        self._subsets = []
        self._ddid_data_dict = {}

        if self.dh._ddids_unequal:
            for ddid in self.ddids:
                rows = np.where(self.dh.ddid_col==ddid)[0]
                rows0 = np.array(sorted(ddid_rows0[ddid]))
                datadict = "{}:D{}".format(self._data_dict_name, ddid)
                subset = MSTile.Subset(self, "DDID {}".format(ddid), datadict, rows, rows0)
                self._subsets.append(subset)
                self._ddid_data_dict[ddid] = subset, slice(None)
        else:
            rows = slice(first_row, last_row + 1)
            rows0 = np.arange(self.first_row0, self.last_row0 + 1)
            subset = MSTile.Subset(self, None, self._data_dict_name, rows, rows0)
            self._subsets.append(subset)
            for ddid in self.ddids:
                self._ddid_data_dict[ddid] = subset, np.where(subset.ddid_col == ddid)[0]


    def get_chunk_indices(self, key):
        """ Returns chunk indices based on the key value. """

        return self._chunk_indices[key]

    def get_chunk_keys(self):
        """ Returns all chunk keys. """

        return self._chunk_dict.iterkeys()

    def get_chunk_tfs(self, key):
        """
        Returns timestamps and freqs for the given chunk associated with key, as well as two slice
        objects describing its position in the global time/freq space.

        Args:
            key (str):
                The label corresponding to the chunk of interest.

        Returns:
            tuple:
                Unique times, channel frequencies, time axis slice, frequency axis slice (in overall subset of freqs)
        """

        rowchunk, chan0, chan1 = self._chunk_dict[key]
        # lookup ordinal number of this DDID, and convert this to offset in frequencies
        chan_offset = self.dh.ddid_first_chan[rowchunk.ddid]
        return self.dh.uniq_times[rowchunk.timeslice], \
               self.dh.chanfreqs[rowchunk.ddid][chan0:chan1], \
               rowchunk.timeslice, \
               slice(chan_offset + chan0, chan_offset + chan1)

    def load(self, load_model=True):
        """
        Fetches data from MS into tile data shared dict. This is meant to be called in the main
        or I/O process.

        Args:
            load_model (bool, optional):
                If False, omits weights and model visibilities.

        Returns:
            :obj:`~cubical.tools.shared_dict.SharedDict`:
                Shared dictionary containing the MS data relevant to the tile.

        Raises:
            RuntimeError:
                If neither --model-lsm nor --model-column set (see [model] section in
                DefaultParset.cfg).
        """

        # Create a shared dict for the data arrays.

        data0 = shared_dict.create(self._data_dict_name)

        # These two variables indicate if the (corrected) data or flags have been updated
        # (Gotcha for shared_dict users! The only truly shared objects are arrays.
        # Thus, we create an array for the two "updated" variables.)

        data0['updated'] = np.array([False, False])
        self._auto_filled_bitflag = False

        # now run over the subsets of the tile set up above. Each subset is a chunk of rows with the same
        # channel shape. If all DDIDs have the same shape, this will be just the one

        for subset in self._subsets:
            if subset.label is None:
                print>> log(0, "blue"), "{}: reading MS rows {}~{}".format(self.label, self.first_row0, self.last_row0)
                data = data0
            else:
                print>> log(0, "blue"), "{}: reading MS rows {}~{}, {} ({} rows)".format(self.label, self.first_row0,
                                                                                          self.last_row0, subset.label,
                                                                                          len(subset.rows0))
                data = shared_dict.create(subset.datadict)

            table_subset = self.dh.data.selectrows(subset.rows0)
            nrows0 = table_subset.nrows()

            original_row_numbers = table_subset.rownumbers(self.data)

            obvis0 = self.dh.fetchslice(self.dh.data_column, subset=table_subset).astype(self.dh.ctype)
            print>> log(2), "  read " + self.dh.data_column

            uvw0 = table_subset.getcol("UVW")
            print>> log(2), "  read UVW coordinates"

            # read weight columns, if a model is to be read

            if self.dh.has_weights and load_model:
                weights0 = np.zeros([len(self.dh.models)] + list(obvis0.shape), self.dh.wtype)
                wcol_cache = {}
                for i, (_, weight_col) in enumerate(self.dh.models):
                    if weight_col not in wcol_cache:
                        print>> log(1), "  reading weights from {}".format(weight_col)
                        wcol = table_subset.getcol(weight_col)
                        # If weight_column is WEIGHT, expand along the freq axis (looks like WEIGHT SPECTRUM).
                        if weight_col == "WEIGHT":
                            wcol_cache[weight_col] = np.empty_like(obvis0, self.dh.wtype)
                            wcol_cache[weight_col][:] = wcol[:, np.newaxis, self.dh._corr_slice]
                        else:
                            wcol_cache[weight_col] = wcol[:, self.dh._channel_slice, self.dh._corr_slice]
                    weights0[i, ...] = wcol_cache[weight_col]
                del wcol_cache
                num_weights = len(self.dh.models)
            else:
                weights0 = np.zeros((len(self.dh.models), 1, 1, 1), self.dh.wtype)
                num_weights = 0

            # The following block of code deals with the various flagging operations and columns. The
            # aim is to correctly populate flag_arr from the various flag sources.

            # Make a flag array. This will contain FL.PRIOR for any points flagged in the MS.

            flag_arr0 = np.zeros(obvis0.shape, dtype=FL.dtype)

            # FLAG/FLAG_ROW only needed if applying them, or auto-filling BITLAG from them.

            flagcol = flagrow = None
            self.bflagcol = None
            self._flagcol_sum = 0
            self.dh.flagcounts["TOTAL"] += flag_arr0.size

            if self.dh._apply_flags or self.dh._auto_fill_bitflag:
                flagcol = self.dh.fetchslice("FLAG", subset=table_subset)
                flagrow = table_subset.getcol("FLAG_ROW")
                flagcol[flagrow, :, :] = True
                print>> log(2), "  read FLAG/FLAG_ROW"
                # compute stats
                self._flagcol_sum = flagcol.sum()
                self.dh.flagcounts["FLAG"] += self._flagcol_sum

                if self.dh._apply_flags:
                    flag_arr0[flagcol] = FL.PRIOR

            # if an active row subset is specified, flag non-active rows as priors. Start as all flagged,
            # the clear the flags
            if self.dh.inactive_rows is not None:
                inactive = self.dh.inactive_rows[original_row_numbers]
            else:
                inactive = np.zeros(nrows0, bool)
            num_inactive = inactive.sum()
            if num_inactive:
                print>> log(0), "  applying a solvable subset deselects {} rows".format(num_inactive)
            # apply baseline selection
            if self.dh.min_baseline or self.dh.max_baseline:
                uv2 = (uvw0[:, 0:2] ** 2).sum(1)
                inactive[uv2 < self.dh.min_baseline ** 2] = True
                if self.dh.max_baseline:
                    inactive[uv2 > self.dh.max_baseline ** 2] = True
                print>> log(0), "  applying solvable baseline cutoff deselects {} rows".format(
                    inactive.sum() - num_inactive)
                num_inactive = inactive.sum()
            if num_inactive:
                print>> log(0), "  {:.2%} visibilities have been deselected".format(num_inactive / float(inactive.size))
                flag_arr0[inactive] |= FL.SKIPSOL

            # Form up bitflag array, if needed.
            if self.dh._apply_bitflags or self.dh._save_bitflag or self.dh._auto_fill_bitflag:
                read_bitflags = False
                # If not explicitly re-initializing, try to read column.
                if not self.dh._reinit_bitflags:
                    self.bflagrow = table_subset.getcol("BITFLAG_ROW")
                    # If there's an error reading BITFLAG, it must be unfilled. This is a common
                    # occurrence so we may as well deal with it. In this case, if auto-fill is set,
                    # fill BITFLAG from FLAG/FLAG_ROW.
                    try:
                        self.bflagcol = self.dh.fetchslice("BITFLAG", subset=table_subset)
                        print>> log(2), "  read BITFLAG/BITFLAG_ROW"
                        read_bitflags = True
                    except Exception:
                        if not self.dh._auto_fill_bitflag:
                            print>> log, ModColor.Str(traceback.format_exc().strip())
                            print>> log, ModColor.Str("Error reading BITFLAG column, and --flags-auto-init is not set.")
                            raise
                        print>> log, "  error reading BITFLAG column: not fatal, since we'll auto-fill it from FLAG"
                        for line in traceback.format_exc().strip().split("\n"):
                            print>> log, "    " + line
                # If column wasn't read, create arrays.
                if not read_bitflags:
                    self.bflagcol = np.zeros(flagcol.shape, np.int32)
                    self.bflagrow = np.zeros(flagrow.shape, np.int32)
                    if self.dh._auto_fill_bitflag:
                        self.bflagcol[flagcol] = self.dh._auto_fill_bitflag
                        self.bflagrow[flagrow] = self.dh._auto_fill_bitflag
                        print>> log, "  auto-filling BITFLAG/BITFLAG_ROW of shape %s" % str(self.bflagcol.shape)
                        self._auto_filled_bitflag = True
                # compute stats
                for flagset, bitmask in self.dh.bitflags.iteritems():
                    flagged = self.bflagcol & bitmask != 0
                    flagged[self.bflagrow & bitmask != 0, :, :] = True
                    self.dh.flagcounts[flagset] += flagged.sum()

                # apply
                if self.dh._apply_bitflags:
                    flag_arr0[(self.bflagcol & self.dh._apply_bitflags) != 0] = FL.PRIOR
                    flag_arr0[(self.bflagrow & self.dh._apply_bitflags) != 0, :, :] = FL.PRIOR

            # if bitflag column is not kept, yet we need to save flags, keep the flag column
            if self.bflagcol is None and self.dh._save_flags:
                self._flagcol = flagcol

            flagged = flag_arr0 != 0
            nfl = flagged.sum()
            self.dh.flagcounts["IN"] += nfl
            print>> log, "  {:.2%} input visibilities flagged and/or deselected".format(nfl / float(flagged.size))

            # now rebin arrays if appropriate
            if self.dh.do_freq_rebin or self.dh.do_time_rebin:
                nrows = abs(subset.rebin_row_map[-1])+1
                nchan = subset.rebin_chan_map[-1]+1
                print>> log(0), "  rebinning into {} rows and {} channels".format(nrows, nchan)

                import cubical.kernels
                rebinning = cubical.kernels.import_kernel("rebinning")

                obvis = data.addSharedArray('obvis', [nrows, nchan, self.dh.ncorr], obvis0.dtype)
                flag_arr = data.addSharedArray('flags', obvis.shape, FL.dtype)
                flag_arr.fill(-1)
                uvwco = data.addSharedArray('uvwco', [nrows, 3], float)
                # make dummy weight array if not 0
                if num_weights:
                    weights = data.addSharedArray('weigh', [num_weights] + list(obvis.shape), self.dh.wtype)
                else:
                    weights = np.zeros((1, 1, 1, 1), self.dh.wtype)

                rebinning.rebin_vis(obvis, obvis0, uvwco, uvw0,
                                      flag_arr, flag_arr0,
                                      weights, weights0, num_weights,
                                      subset.rebin_row_map, subset.rebin_chan_map)

                del obvis0, uvw0
                # we'll need flag_arr0 and weights0 for load_models below so don't delete
            # else copy arrays to shm directly
            else:
                # still need to adjust conjugate rows
                obvis0[subset.rebin_row_map<0] = obvis0[subset.rebin_row_map<0].conjugate()
                nrows = nrows0
                obvis = data['obvis'] = obvis0
                data['flags'] = flag_arr0
                uvwco = data['uvwco'] = uvw0
                if num_weights:
                    data['weigh'] = weights0
                del obvis0, flag_arr0, uvw0, weights0

            # The following either reads model visibilities from the measurement set, or uses an lsm
            # and Montblanc to simulate them. Data may need to be massaged to be compatible with
            # Montblanc's strict requirements.

            if load_model:
                model_shape = [len(self.dh.model_directions), len(self.dh.models)] + list(obvis.shape)
                loaded_models = {}
                movis = data.addSharedArray('movis', model_shape, self.dh.ctype)

                for imod, (dirmodels, _) in enumerate(self.dh.models):
                    # populate directions of this model
                    for idir, dirname in enumerate(self.dh.model_directions):
                        if dirname in dirmodels:
                            # loop over additive components
                            for model_source, cluster in dirmodels[dirname]:
                                # see if data for this model is already loaded
                                if model_source in loaded_models:
                                    print>> log(1), "  reusing {}{} for model {} direction {}".format(model_source,
                                                                                                      "" if not cluster else (
                                                                                                          "()" if cluster == 'die' else "({})".format(
                                                                                                              cluster)),
                                                                                                      imod, idir)
                                    model = loaded_models[model_source][cluster]
                                # cluster of None signifies that this is a visibility column
                                elif cluster is None:
                                    if model_source is 1:
                                        print>> log(0), "  using 1.+0j for model {} direction {}".format(model_source,
                                                                                                         imod, idir)
                                        model = np.ones_like(obvis)
                                    else:
                                        print>> log(0), "  reading {} for model {} direction {}".format(model_source, imod,
                                                                                                        idir)
                                        model0 = self.dh.fetchslice(model_source, subset=table_subset)
                                        if self.dh.do_freq_rebin or self.dh.do_time_rebin:
                                            model = np.empty_like(obvis)
                                            rebinning.rebin_model(model, model0, flag_arr0,
                                                                  weights0[imod], num_weights > 0,
                                                                  subset.rebin_row_map, subset.rebin_chan_map)
                                        else:
                                            model0[subset.rebin_row_map < 0] = model0[subset.rebin_row_map < 0].conjugate()
                                            model = model0
                                        model0 = None
                                    loaded_models.setdefault(model_source, {})[None] = model
                                # else evaluate a Tigger model with Montblanc
                                else:
                                    model = subset.load_montblanc_models(uvwco, loaded_models, model_source, cluster, imod, idir)

                                # finally, add model in at correct slot
                                movis[idir, imod, ...] += model
                                del model

                # release memory (gc.collect() particularly important), as model visibilities are *THE* major user (especially
                # in the DD case)
                del loaded_models
                import gc
                gc.collect()

            data.addSharedArray('covis', data['obvis'].shape, self.dh.ctype)

        # Create a placeholder for the gain solutions
        data.addSubdict("solutions")

        return data


    def get_chunk_cubes(self, key, ctype=np.complex128, allocator=np.empty, flag_allocator=np.empty):
        """
        Produces the CubiCal data cubes corresponding to the specified key.

        Args:
            key (str):
                The label corresponding to the chunk of interest.
            ctype (type):
                Data type of complex arrays.
            allocator (callable):
                Function called to allocate array. Signature (shape,dtype)
            flag_allocator (callable):
                Function called to allocate flag-like arrays. Signature (shape,dtype)


        Returns:
            tuple:
                The data, model, flags and weights cubes for the given chunk key.
                Shapes are as follows:

                - data (np.ndarray):    [n_tim, n_fre, n_ant, n_ant, 2, 2]
                - model (np.ndarray):   [n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, 2, 2]
                - flags (np.ndarray):   [n_tim, n_fre, n_ant, n_ant]
                - weights (np.ndarray): [n_mod, n_tim, n_fre, n_ant, n_ant] or None for no weighting

                n_mod refers to number of models simultaneously fitted.
        """
        rowchunk, freq0, freq1 = self._chunk_dict[key]

        subset, _ = self._ddid_data_dict[rowchunk.ddid]

        data = shared_dict.attach(subset.datadict)

        t_dim = self.dh.chunk_ntimes[rowchunk.tchunk]
        f_dim = freq1 - freq0
        freq_slice = slice(freq0, freq1)
        rows = rowchunk.rows
        nants = self.dh.nants

        flags_2x2 = subset._column_to_cube(data['flags'], t_dim, f_dim, rows, freq_slice,
                                         FL.dtype, FL.MISSING, allocator=allocator)
        flags = flag_allocator(flags_2x2.shape[:-2], flags_2x2.dtype)
        if self.ncorr == 4:
            np.bitwise_or.reduce(flags_2x2, axis=(-1, -2), out=flags)
        else:
            flags[:] = flags_2x2[..., 0, 0]
            flags |= flags_2x2[..., 1, 1]

        obs_arr = subset._column_to_cube(data['obvis'], t_dim, f_dim, rows, freq_slice, ctype,
                                       reqdims=6, allocator=allocator)
        if 'movis' in data:
            mod_arr = subset._column_to_cube(data['movis'], t_dim, f_dim, rows, freq_slice, ctype,
                                           reqdims=8, allocator=allocator)
            # flag invalid model visibilities
            flags[(~np.isfinite(mod_arr[0, 0, ...])).any(axis=(-2, -1))] |= FL.INVALID
        else:
            mod_arr = None

        # flag invalid data
        flags[(~np.isfinite(obs_arr)).any(axis=(-2, -1))] |= FL.INVALID
        flagged = flags != 0

        if 'weigh' in data:
            wgt_2x2 = subset._column_to_cube(data['weigh'], t_dim, f_dim, rows, freq_slice, self.dh.wtype,
                                           allocator=allocator)
            wgt_arr = flag_allocator(wgt_2x2.shape[:-2], wgt_2x2.dtype)
            np.mean(wgt_2x2, axis=(-1, -2), out=wgt_arr)
            #            wgt_arr = np.sqrt(wgt_2x2.sum(axis=(-1,-2)))    # this is wrong
            wgt_arr[flagged] = 0
            wgt_arr = wgt_arr.reshape([1, t_dim, f_dim, nants, nants])
        else:
            wgt_arr = None

        # # zero flagged entries in data and model. NB: this is now done in the solver instead
        # obs_arr[flagged, :, :] = 0
        # if mod_arr is not None:
        #     mod_arr[0, 0, flagged, :, :] = 0

        return obs_arr, mod_arr, flags, wgt_arr

    def set_chunk_cubes(self, cube, flag_cube, key, column='covis'):
        """
        Copies a visibility cube, and an optional flag cube, back to tile column.

        Args:
            cube (np.ndarray):
                Cube containing visibilities.
            flag_cube (np.ndarray):
                Cube containing flags.
            key (str):
                The label corresponding to the chunk of interest.
            column (str, optional):
                The column to which the cube must be copied.
        """
        rowchunk, freq0, freq1 = self._chunk_dict[key]
        subset, _ = self._ddid_data_dict[rowchunk.ddid]
        data = shared_dict.attach(subset.datadict)
        rows = rowchunk.rows
        freq_slice = slice(freq0, freq1)
        if cube is not None:
            data['updated'][0] = True
            subset._cube_to_column(data[column], cube, rows, freq_slice)
        if flag_cube is not None:
            data['updated'][1] = True
            subset._cube_to_column(data['flags'], flag_cube, rows, freq_slice, flags=True)

    def create_solutions_chunk_dict(self, key):
        """
        Creates a shared dict for the given chunk in which to store gain solutions.

        Args:
            key (str):
                The label corresponding to the chunk of interest.

        Returns:
            :obj:`~cubical.tools.shared_dict.SharedDict`:
                Shared dictionary containing gain solutions.
        """

        data = shared_dict.attach(self._data_dict_name)
        sd = data['solutions'].addSubdict(key)

        return sd

    def iterate_solution_chunks(self):
        """
        Iterates over per-chunk solution dictionaries.

        Yields:
            tuple:
                A gain subdictionary and the time and frequency slices to which it corresponds:

                - Subdictionary (:obj:`~cubical.tools.shared_dict.SharedDict`))
                - Time slice (slice)
                - Frequency slice (slice)
        """

        data = shared_dict.attach(self._data_dict_name)
        soldict = data['solutions']
        for key in soldict.iterkeys():
            yield soldict[key]

    def save(self, unlock=False):
        """
        Saves 'corrected' column, and any updated flags, back to MS.

        Args:
            unlock (bool, optional):
                If True, calls the unlock method on the handler.
        """
        data0 = shared_dict.attach(self._data_dict_name)

        # insert columns first, if needed, and reopen MS
        added = False

        for subset in self._subsets:
            if subset.label is None:
                print>> log(0, "blue"), "{}: saving MS rows {}~{}".format(self.label, self.first_row0, self.last_row0)
                data = data0
            else:
                print>> log(0, "blue"), "{}: saving MS rows {}~{}, {} ({} rows)".format(self.label, self.first_row0,
                                                                                          self.last_row0, subset.label,
                                                                                          len(subset.rows0))
                data = shared_dict.attach(subset.datadict)

            # insert output columns, if needed, and reopen MS if they were actually added
            if not added:
                if self.dh.output_column and data0['updated'][0]:
                    added = self.dh._add_column(self.dh.output_column)
                if self.dh.output_model_column and 'movis' in data:
                    added = added or self.dh._add_column(self.dh.output_model_column)
                if added:
                    self.dh.reopen()
                added = True

            table_subset = self.dh.data.selectrows(subset.rows0)

            if self.dh.output_column and data0['updated'][0]:
                covis = subset.upsample(data['covis'])
                print>> log, "  writing {} column".format(self.dh.output_column)
                self.dh.putslice(self.dh.output_column, covis, subset=table_subset)

            if self.dh.output_model_column and 'movis' in data:
                # take first mode, and sum over directions if needed
                model = data['movis'][:, 0]
                if model.shape[0] == 1:
                    model = model.reshape(model.shape[1:])
                else:
                    model = model.sum(axis=0)
                model = subset.upsample(model)
                print>> log, "  writing {} column".format(self.dh.output_model_column)
                self.dh.putslice(self.dh.output_model_column, model, subset=table_subset)

            # write flags if (a) solver has generated flags, and we're saving them, (b) always, if auto-filling BITFLAG column
            #
            flag_col = None     # if not None, FLAG/FLAG_ROW will be saved
            bflag_col = None    # if not None, BITFLAG/BITFLAG_ROW will be saved

            if data0['updated'][1]:
                # count new flags
                newflags = subset.upsample(data['flags'] & ~(FL.PRIOR | FL.SKIPSOL) != 0)
                nfl = newflags.sum()
                # add to stats
                self.dh.flagcounts['NEW'] += nfl
                ratio = nfl / float(newflags.size)

                # write to BITFLAG, if asked to
                if self.dh._save_bitflag:
                    # clear bitflag column first
                    self.bflagcol &= ~self.dh._save_bitflag
                    self.bflagcol[newflags] |= self.dh._save_bitflag
                    bflag_col = True
                    if self.dh._save_flags:
                        print>> log, "  {:.2%} visibilities flagged by solver: saving to BITFLAG and FLAG columns".format(ratio)
                        flag_col = self.bflagcol != 0
                    else:
                        print>> log, "  {:.2%} visibilities flagged by solver: saving to BITFLAG column only".format(ratio)

                # else write to FLAG/FLAG_ROW only, if asked to
                elif self.dh._save_flags:
                    print>> log, "  {:.2%} visibilities flagged by solver: saving to FLAG column".format(ratio)
                    self._flagcol[newflags] = True
                    flag_col = self._flagcol

                # else just message
                else:
                    print>> log, "  {:.2%} visibilities flagged by solver, but we're not saving flags".format(ratio)
            else:
                print>> log, "  no new flags were generated"
                if self._auto_filled_bitflag:
                    bflag_col = True

            # now figure out what to write
            # this is set if BITFLAG/BITFLAG_ROW is to be written out
            if bflag_col is not None:
                self.dh.putslice("BITFLAG", self.bflagcol, subset=table_subset)
                totflags = (self.bflagcol != 0).sum()
                self.dh.flagcounts['OUT'] += totflags
                print>> log, "  updated BITFLAG column ({:.2%} visibilities flagged)".format(totflags / float(self.bflagcol.size))
                self.bflagrow = np.bitwise_and.reduce(self.bflagcol, axis=(-1, -2))
                table_subset.putcol("BITFLAG_ROW", self.bflagrow)
                print>> log, "  updated BITFLAG_ROW column ({:.2%} rows flagged)".format(
                    (self.bflagrow!=0).sum()/float(self.bflagrow.size))

            # this is set if FLAG/FLAG_ROW is to be written out
            if flag_col is not None:
                self.dh.putslice("FLAG", flag_col, subset=table_subset)
                totflags = flag_col.sum()
                if bflag_col is None:                  # only count if not counted above
                    self.dh.flagcounts['OUT'] += totflags
                print>> log, "  updated FLAG column ({:.2%} visibilities flagged)".format(totflags / float(flag_col.size))
                flag_row = flag_col.all(axis=(-1, -2))
                table_subset.putcol("FLAG_ROW", flag_row)
                print>> log, "  updated FLAG_ROW column ({:.2%} rows flagged)".format(flag_row.sum() / float(flag_row.size))

            if unlock:
                self.dh.unlock()

    def release(self):
        """ Releases the shared memory data dicts. """

        data = shared_dict.attach(self._data_dict_name)
        data.delete()
        for subset in self._subsets:
            if subset.label is not None:
                data = shared_dict.attach(subset.datadict)
                data.delete()




