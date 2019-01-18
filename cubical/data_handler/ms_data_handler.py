# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
import numpy as np
from collections import OrderedDict
import pyrap.tables as pt
import cPickle
import re
import traceback
import math
import os.path
import itertools
import logging

import cubical.flagging as flagging

from cubical import data_handler
from cubical.data_handler import Metadata

from cubical.data_handler.ms_tile import RowChunk, MSTile

from cubical.tools import logger, ModColor
from cubical.machines.parallactic_machine import parallactic_machine
log = logger.getLogger("data_handler")


def _divide_up(n, k):
    """For two integers n and k, returns ceil(n/k)"""
    return n//k + (1 if n%k else 0)

def _parse_slice(arg, what="slice"):
    """
    Helper function. Parses an string argument into a slice.  
    Supports e.g. "5~7" (inclusive range), "5:8" (pythonic range). An optional ":STEP" may be added

    Args:
        arg (str):
            Raw range expression.
        what (str):
            How to refer to the slice in error messages. Default is "slice"

    Returns:
        slice:
            Slice object.

    Raises:
        TypeError:
            If the type of arg is not understood. 
        ValueError:
            If the slice cannot be parsed.
    """
    if not arg:
        return slice(None)
    elif type(arg) is not str:
        raise TypeError("can't parse argument of type '{}' as a {}".format(type(arg), what))
    arg = arg.strip()
    m1 = re.match("(\d*)~(\d*)(:(\d+))?$", arg)
    m2 = re.match("(\d*):(\d*)(:(\d+))?$", arg)
    if m1:
        i0, i1, i2 = [ int(x) if x else None for x in m1.group(1),m1.group(2),m1.group(4) ]
        if i1 is not None:
            i1 += 1
    elif m2:
        i0, i1, i2 = [ int(x) if x else None for x in m2.group(1),m2.group(2),m2.group(4) ]
    else:
        raise ValueError("can't parse '{}' as a {}".format(arg, what))
    return slice(i0,i1,i2)


def _parse_range(arg, nmax):
    """
    Helper function. Parses an argument into a list of numbers. Nmax is max number.
    Supports e.g. 5, "5", "5~7" (inclusive range), "5:8" (pythonic range), "5,6,7" (list).

    Args:
        arg (int or tuple or list or str):
            Raw range expression.
        nmax (int):
            Maximum possible range.

    Returns:
        list:
            Range of numbers.

    Raises:
        TypeError:
            If the type of arg is not understood. 
        ValueError:
            If the range cannot be parsed.
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
        raise TypeError("can't parse argument of type '%s' as a range or slice"%type(arg))
    arg = arg.strip()

    if re.match("\d+$", arg):
        return [ int(arg) ]
    elif "," in arg:
        return map(int,','.split(arg))

    return fullrange[_parse_slice(arg, "range or slice")]

_prefixes = dict(m=1e-3, k=1e+3, M=1e+6, G=1e+9, T=1e+12)

def _parse_bin(binspec, units, default_int=None, default_float=None, kind='bin'):
    """
    Parses a bin specification of the form N or "Xunit"
    Returns tuple of N,X where either N or X is set from
    the specification, and the other one is default.
    """
    if not binspec:
        return default_int, default_float
    elif type(binspec) is int:
        return binspec, default_float
    elif type(binspec) is str:
        for unit, multiplier in units.items():
            if binspec.endswith(unit) and len(binspec) > len(unit):
                xval = binspec[:-len(unit)]
                if xval[-1] in _prefixes:
                    multiplier *= _prefixes[xval[-1]]
                    xval = xval[:-1]
                try:
                    return default_int, float(xval)*multiplier
                except:
                    raise ValueError("invalid {} specification '{}'".format(kind, binspec))
    raise ValueError("invalid {} specification '{}'".format(kind, binspec))


def _parse_timespec(timespec, default_int=None, default_float=None):
    return _parse_bin(timespec, dict(s=1, m=60, h=3600),
                      default_int, default_float, "time")

def _parse_freqspec(freqspec, default_int=None, default_float=None):
    return _parse_bin(freqspec, dict(Hz=1),
                      default_int, default_float, "frequency")


class MSDataHandler:
    """ Main data handler. Interfaces with the measurement set. """

    def __init__(self, ms_name, data_column, output_column=None, output_model_column=None,
                 output_weight_column=None, reinit_output_column=False,
                 taql=None, fid=None, ddid=None, channels=None,
                 diag=False,
                 beam_pattern=None, beam_l_axis=None, beam_m_axis=None,
                 active_subset=None, min_baseline=0, max_baseline=0,
                 chunk_freq=None, rebin_freq=None,
                 do_load_CASA_kwtables=True,
                 enable_solve_parallactic_rotation=True,
                 enable_apply_parallactic_rotation=True):
        """
        Initialises a DataHandler object.

        Args:
            ms_name (str):
                Name of measeurement set.
            data_colum (str):
                Name of the input observed data column.
            sm_name (str):
                Name of sky model.
            model_column (str):
                Name of input model column.
            output_column (str or None, optional):
                Name of output column if specified, else None.
            output_column (str or None, optional):
                Name of output model column if specified, else None.
            taql (str):
                Additional TAQL query for data selection.
            fid (int or None, optional):
                Field identifier if specified, else None.
            ddid (int, list or None, optional):
                Data descriptor identifer/s if specified, else None.
            flagopts (dict, optional):
                Flagging options.
            diag (bool)
                If True, only the diagonal correlations are read in
            ddes (bool, optional):
                If True, use direction dependent simulation.
            weight_column (str or None, optional):
                Name of input weight column if specified, else None.
            beam_pattern (str or None, optional):
                Pattern for reading beam files if specified, else None.
            beam_l_axis (str or None, optional):
                Corresponding axis in fits beam, else None.
            beam_m_axis (str or None, optional):
                Corresponding axis in fits beam, else None.
            mb_opts (dict or None):
                Dictionary of Montblanc options if specified, else None.
            rebin_time (int):
                Average specified number of timeslots together on-the-fly
            rebin_freq (int):
                Average specified number of channels together on-the-fly
            do_load_CASA_kwtables
                Should load CASA MS MEMO 229 keyword tables (optional). If not loaded
                no CASA-style gaintables can be produced.
            enable_solve_parallactic_rotation
                Should rotate sky model (either lsm or MODEL_DATA) around observer's third axis 
            enable_apply_parallactic_rotation
                Should derotate corrected data after calibration
        Raises:
            RuntimeError:
                If Montblanc cannot be imported but simulation is required.
            ValueError:
                If selection from MS returns no rows.
        """

        self.ms_name = ms_name
        self.beam_pattern = beam_pattern
        self.beam_l_axis = beam_l_axis
        self.beam_m_axis = beam_m_axis

        self.fid = fid if fid is not None else 0

        print>>log, ModColor.Str("reading MS %s"%self.ms_name, col="green")

        self.ms = pt.table(self.ms_name, readonly=False, ack=False)
        self.data = None

        #print>>log, "  sorting MS by TIME column"
        #self.ms = self.ms.sort("TIME")

        _anttab = pt.table(self.ms_name + "::ANTENNA", ack=False)
        _fldtab = pt.table(self.ms_name + "::FIELD", ack=False)
        _spwtab = pt.table(self.ms_name + "::SPECTRAL_WINDOW", ack=False)
        _poltab = pt.table(self.ms_name + "::POLARIZATION", ack=False)
        _ddesctab = pt.table(self.ms_name + "::DATA_DESCRIPTION", ack=False)
        _obstab = pt.table(self.ms_name + "::OBSERVATION", ack=False)
        _feedtab = pt.table(self.ms_name + "::FEED", ack=False)

        self.metadata = Metadata()

        self.ctype = np.complex64   # MS complex data type
        self.wtype = np.float32     # MS weights type
        self.nmscorrs = _poltab.getcol("NUM_CORR")[0]
        if self.nmscorrs == 4 and diag:
            self._corr_4to2 = True
            self.ncorr = 2
            self._corr_slice = (0,3)
        elif self.nmscorrs in (2,4):
            self.ncorr = self.nmscorrs
            self._corr_4to2 = False
            self._corr_slice = slice(None)
        else:
            raise RuntimeError("MS with {} correlations not (yet) supported".format(self.nmscorrs))
        self.diag = diag
        self.nants = self.metadata.num_antennas = _anttab.nrows()
        self.metadata.num_baselines = self.nants*(self.nants-1)/2
        self.metadata.num_corrs  = self.ncorr

        antnames = _anttab.getcol("NAME")
        antpos = _anttab.getcol("POSITION")
        # strip common prefix from antenna names to make short antenna names
        minlength = min([len(name) for name in antnames])
        prefix_length = 0
        while prefix_length < minlength and len(set([name[:prefix_length+1] for name in antnames])) == 1:
            prefix_length += 1

        self.metadata.antenna_name = antnames
        self.metadata.antenna_name_short = antnames_short = [name[prefix_length:] for name in antnames]
        self.metadata.antenna_name_prefix = antnames[0][:prefix_length]

        self.metadata.baseline_name = { (p,q): "{}-{}".format(antnames[p], antnames_short[q])
                                        for p in xrange(self.nants) for q in xrange(p+1, self.nants)}
        self.metadata.baseline_length = { (p,q): math.sqrt(((antpos[p]-antpos[q])**2).sum())
                                        for p in xrange(self.nants) for q in xrange(p+1, self.nants)}

        if do_load_CASA_kwtables:
            # antenna fields to be used when writing gain tables
            anttabcols = ["OFFSET", "POSITION", "TYPE",
                          "DISH_DIAMETER", "FLAG_ROW", "MOUNT", "NAME",
                          "STATION"]
            assert set(anttabcols) <= set(
                _anttab.colnames()), "Measurement set conformance error - keyword table ANTENNA incomplete. Perhaps disable --out-casa-gaintables or check your MS!"
            self._anttabcols = {t: _anttab.getcol(t) if _anttab.iscelldefined(t, 0) else np.array([]) for t in
                                anttabcols}

            # field information to be used when writing gain tables
            fldtabcols = ["DELAY_DIR", "PHASE_DIR", "REFERENCE_DIR",
                          "CODE", "FLAG_ROW", "NAME", "NUM_POLY",
                          "SOURCE_ID", "TIME"]
            assert set(fldtabcols) <= set(
                _fldtab.colnames()), "Measurement set conformance error - keyword table FIELD incomplete. Perhaps disable --out-casa-gaintables or check your MS!"
            self._fldtabcols = {t: _fldtab.getcol(t) if _fldtab.iscelldefined(t, 0) else np.array([]) for t in
                                fldtabcols}

            # spw information to be used when writing gain tables
            spwtabcols = ["MEAS_FREQ_REF", "CHAN_FREQ", "REF_FREQUENCY",
                          "CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION",
                          "FLAG_ROW", "FREQ_GROUP", "FREQ_GROUP_NAME",
                          "IF_CONV_CHAIN", "NAME", "NET_SIDEBAND",
                          "NUM_CHAN", "TOTAL_BANDWIDTH"]

            assert set(spwtabcols) <= set(
                _spwtab.colnames()), "Measurement set conformance error - keyword table SPECTRAL_WINDOW incomplete. Perhaps disable --out-casa-gaintables or check your MS!"
            self._spwtabcols = {t: _spwtab.getcol(t) for t in spwtabcols}

            # read observation details
            obstabcols = ["TIME_RANGE", "LOG", "SCHEDULE", "FLAG_ROW",
                          "OBSERVER", "PROJECT", "RELEASE_DATE", "SCHEDULE_TYPE",
                          "TELESCOPE_NAME"]
            assert set(obstabcols) <= set(
                _obstab.colnames()), "Measurement set conformance error - keyword table OBSERVATION incomplete. Perhaps disable --out-casa-gaintables or check your MS!"
            self._obstabcols = {t: _obstab.getcol(t) if _obstab.iscelldefined(t, 0) else np.array([]) for t in
                                obstabcols}

        self.antpos   = _anttab.getcol("POSITION")
        self.antnames = _anttab.getcol("NAME")
        self.phadir  = _fldtab.getcol("PHASE_DIR", startrow=self.fid, nrow=1)[0][0]
        self.metadata.ra0, self.metadata.dec0 = self.phadir
        self._poltype = np.unique(_feedtab.getcol('POLARIZATION_TYPE')['array'])
        
        if np.any([pol in self._poltype for pol in ['L','l','R','r']]):
            self._poltype = "circular"
            self.feeds = self.metadata.feeds = "rl"
        elif np.any([pol in self._poltype for pol in ['X','x','Y','y']]):
            self._poltype = "linear"
            self.feeds = self.metadata.feeds = "xy"
        else:
            raise TypeError("unsupported POLARIZATION_TYPE {}. Terminating.".format(self._poltype))

        # print some info on MS layout
        print>>log,"  detected {} ({}) feeds".format(self._poltype, self.feeds)
        print>>log,"  fields are "+", ".join(["{}{}: {}".format('*' if i==fid else "",i,name) for i, name in enumerate(_fldtab.getcol("NAME"))])

        # get list of channel frequencies (this may have varying sizes)
        self._spw_chanfreqs = [ _spwtab.getcell("CHAN_FREQ", i) for i in xrange(_spwtab.nrows()) ]
        self._spw_chanwidth = [ _spwtab.getcell("CHAN_WIDTH", i) for i in xrange(_spwtab.nrows()) ]
        print>>log,"  MS contains {} spectral windows".format(len(self._spw_chanfreqs))

        # figure out DDID range
        self._num_total_ddids = _ddesctab.nrows()
        self._ddids = _parse_range(ddid, self._num_total_ddids)
        if not self._ddids:
            raise ValueError("'ddid' did not select any valid DDIDs".format(ddid))

        # figure out channel slices per DDID
        self._channel_slice = _parse_slice(channels)
        # form up blc/trc/incr arguments for getcolslice() and putcolslice()
        if self._channel_slice != slice(None):
            print>> log, "  applying a channel selection of {}".format(channels)
            chan0 = self._channel_slice.start if self._channel_slice.start is not None else 0
            chan1 = self._channel_slice.stop - 1 if self._channel_slice.stop is not None else -1
            self._ms_blc = (chan0, 0)
            self._ms_trc = (chan1, self.ncorr - 1)
            self._ms_incr = (1, 3) if self._corr_4to2 else (1, 1)
        elif self._corr_4to2:
            self._ms_blc = (0, 0)
            self._ms_trc = (-1, 3)
            self._ms_incr = (1, 3)
        else:
            self._ms_trc = self._ms_blc = self._ms_incr = None  # tells fetchslice that no slicing is done

        # get channel information per SPW
        self._ddid_spw = _ddesctab.getcol("SPECTRAL_WINDOW_ID")

        # now compute binning/chunking for each DDID
        chunk_chans, chunk_hz = _parse_freqspec(chunk_freq, 1<<31, 1e+99)
        print>>log,"  max freq chunk size is {} channels and/or {} MHz".format(
            '--' if chunk_chans == 1<<31 else chunk_chans,
            '--' if chunk_hz == 1e+99 else chunk_hz*1e-6)
        rebin_chans, rebin_hz = _parse_freqspec(rebin_freq, 1, None)
        if rebin_hz is not None:
            print>>log, "  rebinning into {} MHz channels".format(rebin_hz*1e-6)
        elif rebin_chans > 1:
            print>>log,"  rebinning by {} channels".format(rebin_chans)

        # per DDID:
        self.rebin_chan_maps = {}   # map from raw channel to rebinned channel
        self.chanfreqs = {}         # vector of rebinned channel centres
        self.chanwidth = {}         # vector of rebinned channel widths
        self.freqchunks = {}        # vector of first (rebinned) channel in each chunk
        self._nchan0_orig = {}       # map from DDID to original channel size -- needed to init new columns

        for ddid in self._ddids:
            chanfreqs0 = self._spw_chanfreqs[self._ddid_spw[ddid]]
            nchan0_orig = self._nchan0_orig[ddid] = len(chanfreqs0)
            chanfreqs0 = chanfreqs0[self._channel_slice]
            chanwidth0 = self._spw_chanwidth[self._ddid_spw[ddid]][self._channel_slice]
            nchan0 = len(chanfreqs0)

            rebin_chan_map = self.rebin_chan_maps[ddid] = np.zeros(nchan0, np.int64)

            # this is a list of channels at which freq chunks start
            freqchunk_chan0 = []

            # start of current chunk
            chunk_start_chan0, chunk_start_freq0 = -(1<<31), 0
            # rebinned channel edges
            chan_edges = {}
            rebinning = False
            # current output (rebinned) channel
            chan = bin_start_chan0 = -1
            bin_start_freq0 = 0

            for chan0, (freq0, width0) in enumerate(zip(chanfreqs0, chanwidth0)):
                # start new chunk if hit boundary
                if (chan0-chunk_start_chan0) >= chunk_chans or (chunk_hz and abs(freq0-chunk_start_freq0) >= chunk_hz):
                    freqchunk_chan0.append(chan0)
                    chunk_start_chan0 = chan0
                    chunk_start_freq0 = freq0
                    # start new bin
                    chan += 1
                    bin_start_chan0, bin_start_freq0 = chan0, freq0
                # work out output channel number based on rebinning factor
                if rebin_hz is None:
                    if chan0 - bin_start_chan0 >= rebin_chans:
                        chan += 1
                        bin_start_chan0, bin_start_freq0 = chan0, freq0
                else:
                    if abs(freq0 - bin_start_freq0) >= rebin_hz:
                        chan += 1
                        bin_start_chan0, bin_start_freq0 = chan0, freq0
                rebin_chan_map[chan0] = chan
                if chan0 != chan:
                    rebinning = True
                # update rebinned edges
                fmin, fmax = chan_edges.get(chan, (1e+99,-1e+99))
                chan_edges[chan] = min(fmin, freq0 - width0/2), max(fmax, freq0 + width0/2)

            if rebinning:
                # make lists of rebinned channel centres and widths
                nchan = chan+1
                chanfreqs = self.chanfreqs[ddid] = np.empty(nchan, float)
                chanwidth = self.chanwidth[ddid] = np.empty(nchan, float)
                for chan in xrange(nchan):
                    fmin, fmax = chan_edges.get(chan)
                    chanfreqs[chan] = (fmin+fmax)/2
                    chanwidth[chan] = (fmax-fmin)
                self.freqchunks[ddid] = freqchunks = [rebin_chan_map[chan0] for chan0 in freqchunk_chan0]
                print>>log(0),"  DDID {}: {}/{} selected channels will be rebinned into {} channels".format(
                    ddid, nchan0, nchan0_orig, nchan)
                print>>log(1),"    rebinned channel freqs (MHz): {}".format(
                    " ".join([str(x*1e-6) for x in chanfreqs]))
                print>>log(1),"    rebinned channel widths (MHz): {}".format(
                    " ".join([str(x*1e-6) for x in chanwidth]))
            else:
                nchan = nchan0
                self.chanfreqs[ddid] = chanfreqs0
                self.chanwidth[ddid] = chanwidth0
                self.freqchunks[ddid] = freqchunks = freqchunk_chan0
                self.rebin_chan_maps[ddid] = None
                print>>log(0),"  DDID {}: {}/{} channels selected".format(ddid, nchan0, nchan0_orig)

            print>>log(0),"    found {} frequency chunks: {}".format(len(freqchunks),
                            " ".join([str(ch) for ch in freqchunks + [nchan]]))

        # now accumulate list of all frequencies, and also see if selected DDIDs have a uniform rebinning and chunking map
        all_freqs = set(self.chanfreqs[self._ddids[0]])
        self.do_freq_rebin = any([m is not None for m in self.rebin_chan_maps.values()])
        self._ddids_unequal = False
        ddid0_map = self.rebin_chan_maps[self._ddids[0]]
        for ddid in self._ddids[1:]:
            map1 = self.rebin_chan_maps[ddid]
            if ddid0_map is None and map1 is None:
                continue
            if (ddid0_map is None and map1 is not None) or (ddid0_map is not None and map1 is None) or \
                    len(ddid0_map) != len(map1) or (ddid0_map!=map1).any():
                self._ddids_unequal = True
            all_freqs.update(self.chanfreqs[ddid])

        if self._ddids_unequal:
            print>>log(0,"red"),"Selected DDIDs have differing channel structure. Processing may be less efficient."


        # TODO: this assumes DDIDs are ordered in frequency. Exotic cases where this is not?
        self.all_freqs = np.fromiter(itertools.chain(*[list(self.chanfreqs[ddid]) for ddid in self._ddids]), float)

        deltafreq = self.all_freqs[1:] - self.all_freqs[:-1]
        if not (deltafreq<0).all() and not (deltafreq>0).all():
            raise RuntimeError("The selected DDID/frequencies in this MS are not laid out monotonically. This is currently not supported.")

        # make index of DDID -> ordinal number within DDID selection
        self._ddid_index = { d: num for num,d in enumerate(self._ddids) }

        # make index of DDID -> first channel representing that DDID in allfreqs
        first_chan = np.cumsum([[len(self.chanfreqs[ddid]) for ddid in self._ddids]])
        first_chan -= len(self.chanfreqs[self._ddids[0]])
        self.ddid_first_chan = {ddid:first_chan[num] for num,ddid in enumerate(self._ddids)}

        print>>log(1),"   overall frequency space (MHz): {}".format(" ".join([str(f*1e-6) for f in self.all_freqs]))
        print>>log(1),"   DDIDs start at channels: {}".format(" ".join([str(ch) for ch in self.ddid_first_chan]))


        # use TaQL to select subset
        self.taql = self.build_taql(taql, fid, self._ddids)

        # reopen ms, sort, apply TaQL query
        self.reopen()

        if self.taql:
            print>> log, "  applying TAQL query '%s' (%d/%d rows selected)" % (self.taql,
                                                                             self.data.nrows(), self.ms.nrows())

        if active_subset:
            subset = self.data.query(active_subset)
            self.active_row_numbers = np.array(subset.rownumbers(self.data))
            self.inactive_rows = np.zeros(self.data.nrow(), True)
            self.inactive_rows[self.active_row_numbers] = False
            print>> log, "  applying TAQL query '%s' for solvable subset (%d/%d rows)" % (active_subset,
                                                            subset.nrows(), self.data.nrows())
        else:
            self.active_row_numbers = self.inactive_rows = None
        self.min_baseline, self.max_baseline = min_baseline, max_baseline

        self.nrows = self.data.nrows()

        self._datashape = {ddid: (self.nrows, len(freqs), self.ncorr) for ddid, freqs in self.chanfreqs.items()}

        if not self.nrows:
            raise ValueError("MS selection returns no rows")

        self.time_col = self.fetch("TIME")
        self.uniq_times = np.unique(self.time_col)
        self.ntime = len(self.uniq_times)

        print>>log,"  %d antennas, %d rows, %d/%d DDIDs, %d timeslots, %d corrs %s" % (self.nants,
                    self.nrows, len(self._ddids), self._num_total_ddids, self.ntime,
                    self.nmscorrs, "(using diag only)" if self._corr_4to2 else "")
        print>>log,"  DDID central frequencies are at {} GHz".format(
                " ".join(["%.2f"%(self.chanfreqs[d][len(self.chanfreqs[d])//2]*1e-9) for d in self._ddids]))
        if self.do_freq_rebin and (output_column or output_model_column):
            print>>log(0, "red"),"WARNING: output columns will be upsampled from frequency-binned data!"
        self.nddid = len(self._ddids)

        self.data_column = data_column
        self.output_column = output_column
        self.output_model_column = output_model_column
        self.output_weight_column = output_weight_column
        if reinit_output_column:
            reinit_columns = [col for col in [output_column, output_model_column]
                               if col and col in self.ms.colnames()]
            if reinit_columns:
                print>>log(0),"reinitializing output column(s) {}".format(" ".join(reinit_columns))
                self.ms.removecols(reinit_columns)
                for col in reinit_columns:
                    self._add_column(col)
                if output_weight_column is not None:
                    print>>log(0),"reinitializing output weight column {}".format(output_weight_column)
                    try:
                        self.ms.removecols(output_weight_column) #Just remove column will be added later
                    except:
                        print>>log(0),"No output weight column {}, will just proceed".format(output_weight_column)

                self.reopen()

        self.gain_dict = {}

        self.enable_solve_parallactic_rotation = enable_solve_parallactic_rotation
        self.enable_apply_parallactic_rotation = enable_apply_parallactic_rotation
        if enable_solve_parallactic_rotation or enable_apply_parallactic_rotation:
            self.parallactic_machine = parallactic_machine(antnames,
                                                           antpos,
                                                           feed_basis=self._poltype,
                                                           enable_rotation=enable_solve_parallactic_rotation,
                                                           enable_derotation=enable_apply_parallactic_rotation,
                                                           field_centre=tuple(np.rad2deg(self.phadir)))
        else:
            self.parallactic_machine = None
        pass

    def init_models(self, models, weights, mb_opts={}, use_ddes=False):
        """Parses the model list and initializes internal structures"""

        # ensure we have as many weights as models
        self.has_weights = weights is not None
        if weights is None:
            weights = [None] * len(models)
        elif len(weights) == 1:
            weights = weights*len(models)
        elif len(weights) != len(models):
            raise ValueError,"need as many sets of weights as there are models"

        self.use_montblanc = False    # will be set to true if Montblanc is invoked
        self.models = []
        self.model_directions = set() # keeps track of directions in Tigger models
        global montblanc
        montblanc = None

        for imodel, (model, weight_col) in enumerate(zip(models, weights)):
            # list of per-direction models
            dirmodels = {}
            self.models.append((dirmodels, weight_col))
            for idir, dirmodel in enumerate(model.split(":")):
                if not dirmodel:
                    continue
                idirtag = " dir{}".format(idir if use_ddes else 0)
                for component in dirmodel.split("+"):
                    # special case: "1" means unity visibilities
                    if component == "1":
                        dirmodels.setdefault(idirtag, []).append((1, None))
                    # else check for an LSM component
                    elif component.startswith("./") or component not in self.ms.colnames():
                        # check if LSM ends with @tag specification
                        if "@" in component:
                            component, tag = component.rsplit("@",1)
                        else:
                            tag = None
                        if os.path.exists(component):
                            if montblanc is None:
                                montblanc, exc = data_handler.import_montblanc()
                                if montblanc is None:
                                    print>> log, ModColor.Str("Error importing Montblanc: ")
                                    for line in traceback.format_exception(*exc):
                                        print>> log, "  " + ModColor.Str(line)
                                    print>> log, ModColor.Str("Without Montblanc, LSM functionality is not available.")
                                    raise RuntimeError("Error importing Montblanc")
                            self.use_montblanc = True
                            import TiggerSourceProvider
                            component = TiggerSourceProvider.TiggerSourceProvider(component, self.phadir,
                                                                                  dde_tag=use_ddes and tag)
                            for key in component._cluster_keys:
                                dirname = idirtag if key == 'die' else key
                                dirmodels.setdefault(dirname, []).append((component, key))
                        else:
                            raise ValueError,"model component {} is neither a valid LSM nor an MS column".format(component)
                    # else it is a visibility column component
                    else:
                        dirmodels.setdefault(idirtag, []).append((component, None))
            self.model_directions.update(dirmodels.iterkeys())
        # Now, each model is a dict of dirmodels, keyed by direction name (unnamed directions are _dir0, _dir1, etc.)
        # Get all possible direction names
        self.model_directions = sorted(self.model_directions)

        # print out the results
        print>>log(0),ModColor.Str("Using {} model(s) for {} directions(s){}".format(
                                        len(self.models),
                                        len(self.model_directions),
                                        " (DDEs explicitly disabled)" if not use_ddes else""),
                                   col="green")
        for imod, (dirmodels, weight_col) in enumerate(self.models):
            print>>log(0),"  model {} (weight {}):".format(imod, weight_col)
            for idir, dirname in enumerate(self.model_directions):
                if dirname in dirmodels:
                    comps = []
                    for comp, tag in dirmodels[dirname]:
                        if not tag or tag == 'die':
                            comps.append("{}".format(comp))
                        else:
                            comps.append("{}({})".format(tag, comp))
                    print>>log(0),"    direction {}: {}".format(idir, " + ".join(comps))
                else:
                    print>>log(0),"    direction {}: empty".format(idir)

        self.use_ddes = len(self.model_directions) > 1

        if montblanc is not None:
            self.mb_opts = mb_opts
            mblogger = logging.getLogger("montblanc")
            mblogger.propagate = False
            # NB: this assume that the first handler of the Montblanc logger is the console logger
            mblogger.handlers[0].setLevel(getattr(logging, mb_opts["verbosity"]))



    def build_taql(self, taql=None, fid=None, ddid=None):
        """
        Generate a combined TAQL query using possible options.

        Args:
            taql (str or None, optional):
                Additional TAQL query for data selection.
            fid (int or None, optional):
                Field identifier if specified, else None.
            ddid (int, list or None, optional):
                Data descriptor identifer/s if specified, else None.

        Returns:
            str:
                A TAQL query string. 
        """

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

    def fetch(self, colname, first_row=0, nrows=-1, subset=None):
        """
        Convenience function which mimics pyrap.tables.table.getcol().

        Args:
            colname (str):
                column name
            first_row (int):
                starting row
            nrows (int):
                number of rows to fetch
            subset:
                table to fetch from, else uses self.data

        Returns:
            np.ndarray:
                Result of getcol(\*args, \*\*kwargs).
        """

        return (subset or self.data).getcol(colname, first_row, nrows)

    def fetchslice(self, column, startrow=0, nrows=-1, subset=None):
        """
        Convenience function similar to fetch(), but assumes a column of NFREQxNCORR shape,
        and calls pyrap.tables.table.getcolslice() if there's a channel slice to be applied,
        else just uses getcol().
        
        Args:
            startrow (int):
                Starting row to read.
            nrows (int):
                Number of rows to read.

        Returns:
            np.ndarray:
                Result of getcolslice()
        """
        subset = subset or self.data
        if self._ms_blc == None:
            return subset.getcol(column, startrow, nrows)
        return subset.getcolslice(column, self._ms_blc, self._ms_trc, self._ms_incr, startrow, nrows)

    def fetchslicenp(self, column, data, startrow=0, nrows=-1, subset=None):
        """
        Convenience function similar to fetch(), but assumes a column of NFREQxNCORR shape,
        and calls pyrap.tables.table.getcolslice() if there's a channel slice to be applied,
        else just uses getcol(). This version reads dirctly into an array in memory given by 'data'.

        Args:
            startrow (int):
                Starting row to read.
            nrows (int):
                Number of rows to read.

        Returns:
            np.ndarray:
                Result of getcolslice()
        """
        subset = subset or self.data
        if self._ms_blc == None:
            return subset.getcolnp(column, data, startrow, nrows)
        return subset.getcolslicenp(column, data, self._ms_blc, self._ms_trc, self._ms_incr, startrow, nrows)

    def putslice(self, column, value, startrow=0, nrows=-1, subset=None):
        """
        The opposite of fetchslice(). Assumes a column of NFREQxNCORR shape,
        and calls pyrap.tables.table.putcolslice() if there's a channel slice to be applied,
        else just uses putcol().
        If column is variable-shaped and the cell at startrow is not initialized, attempts to
        initialize an entire section of the column before writing the slice.

        Args:
            startrow (int):
                Starting row to write.
            nrows (int):
                Number of rows to write.

        Returns:
            np.ndarray:
                Result of putcolslice()
        """
        subset = subset or self.data
        # if no slicing, just use putcol to put the whole thing. This always works,
        # unless the MS is screwed up
        if self._ms_blc == None:
            return subset.putcol(column, value, startrow, nrows)
        # A variable-shape column may be uninitialized, in which case putcolslice will not work.
        # But we try it first anyway, especially if the first row of the block looks initialized
        if self.data.iscelldefined(column, startrow):
            try:
                return subset.putcolslice(column, value, self._ms_blc, self._ms_trc, [], startrow, nrows)
            except Exception, exc:
                pass
        if nrows<0:
            nrows = subset.nrows()
        print>>log(0),"  attempting to initialize column {} rows {}:{}".format(column, startrow, startrow+nrows)
        ddid = subset.getcol("DATA_DESC_ID", 0, 1)[0]
        value0 = np.zeros((nrows, self._nchan0_orig[ddid], self.nmscorrs), value.dtype)
        value0[:, self._channel_slice, self._corr_slice] = value
        return subset.putcol(column, value0, startrow, nrows)

    def define_chunk(self, chunk_time, rebin_time, fdim=1, chunk_by=None, chunk_by_jump=0, chunks_per_tile=4, max_chunks_per_tile=0):
        """
        Fetches indexing columns (TIME, DDID, ANTENNA1/2) and defines the chunk dimensions for 
        the data.

        Args:
            tdim (int): 
                Timeslots per chunk.
            fdim (int): 
                Frequencies per chunk.
            chunk_by (str or None, optional):   
                If set, chunks will have boundaries imposed by jumps in the listed columns
            chunk_by_jump (int, optional): 
                The magnitude of a jump has to be over this value to force a chunk boundary.
            chunks_per_tile (int, optional): 
                The minimum number of chunks to be placed in a single tile.
            max_chunks_per_tile (int, optional)
                The maximum number of chunks to be placed in a single tile.
            
        Attributes:
            antea (np.ndarray): ANTENNA1 column of MS subset.
            anteb (np.ndarray): 
                ANTENNA2 column of MS subset.
            ddid_col (np.ndarray): 
                DDID column of MS subset.
            time_col (np.ndarray): 
                TIME column of MS subset.
            times (np.ndarray):    
                Timeslot index number with same size as self.time_col.
            uniq_times (np.ndarray): 
                Unique timestamps in time_col.
                
        Returns:
            max_chunks, tile_list:
                - max number of chunks per tile
                - list of tiles
        """

        self.antea = antea = self.fetch("ANTENNA1").astype(np.int64)
        self.anteb = anteb = self.fetch("ANTENNA2").astype(np.int64)
        self.time_col = time_col = self.fetch("TIME")
        self.ddid_col = ddid_col = ddid_col0 = self.fetch("DATA_DESC_ID").astype(np.int64)
        print>> log, "  read indexing columns ({} total rows)".format(len(self.time_col))
        self.do_time_rebin = False

        self.utc_timestamps = time_col
        self.times, self.uniq_times,_ = data_handler.uniquify(time_col)
        print>> log, "  built timeslot index ({} unique timestamps)".format(len(self.uniq_times))

        chunk_timeslots, chunk_seconds = _parse_timespec(chunk_time, 1<<31, 1e+99)
        print>>log,"  max chunk size is {} timeslots and/or {} seconds".format(
            '--' if chunk_timeslots == 1<<31 else chunk_timeslots,
            '--' if chunk_seconds == 1e+99 else chunk_seconds)
        rebin_timeslots, rebin_seconds = _parse_timespec(rebin_time, 1, None)
        if rebin_seconds is not None:
            print>>log, "  computing time rebinning into {} seconds".format(rebin_seconds)
        elif rebin_timeslots > 1:
            print>>log,"  computing time rebinning by {} timeslots".format(rebin_timeslots)

        import cubical.kernels
        rebinning = cubical.kernels.import_kernel("rebinning")

        nrows0 = len(time_col)

        # this is a map from output timeslot and ddid and baseline to the output row allocated to it.
        output_row = {}

        # this is a list of rows at which rowchunks (i.e. in time) start
        timechunk_row0 = []
        # this is a map from input row0 to rebinned row (uses -row if conjugation is required, i.e. ant2<ant1)
        self.rebin_row_map = np.zeros(nrows0, np.int64)

        # count of rows in output
        nrow_out = 0                                     # number of rows allocated in output
        chunk_end_ts = chunk_end_time = None             # current end-of-chunk boundary

        # set chunk-by boundaries, if specified
        boundaries = np.zeros_like(time_col, bool)
        if chunk_by:
            for column in chunk_by:
                value = self.fetch(column)
                boundaries |= abs(np.roll(value, 1) - value) > chunk_by_jump

        for row0, (a1,a2,ts,time,ddid,boundary) in enumerate(
                zip(antea, anteb, self.times, time_col, ddid_col, boundaries)):
            # start new chunk if hit boundary
            newchunk = boundary or ts >= chunk_end_ts or time >= chunk_end_time
            if newchunk:
                timechunk_row0.append(row0)
                chunk_start_ts = ts
                chunk_start_time = time
                chunk_end_ts = ts + chunk_timeslots
                chunk_end_time = time + chunk_seconds
                # this is a map from output timeslot and ddid and baseline to the output row allocated to it.
                output_row = {}
            # work out output timeslot number (within chunk) based on rebinning factor
            if rebin_seconds is None:
                ts_out = (ts - chunk_start_ts)//rebin_timeslots
            else:
                ts_out = int((time - chunk_start_time)/rebin_seconds)

            # have we already allocated an output row for this timeslot of the rebinned a1, a2 data?
            row_key = (ts_out, a1, a2, ddid)
            row = output_row.get(row_key)
            # no, allocate one
            if row is None:
                output_row[row_key] = row = nrow_out
                nrow_out += 1
            self.rebin_row_map[row0] = row if a1<a2 else -row

        print>>log,"  found {} time chunks: {} {}".format(len(timechunk_row0),
                        " ".join([str(self.times[r]) for r in timechunk_row0]),
                        str(self.times[-1]+1))

        # at the end of this, we have a list of timechunk_row0: i.e. a list of starting rows for
        # each time chunk (which may composed of multiple DDIDs), plus rebin_row_map: a vector giving
        # the rebinned row number for the original row number (row0). Use this to rebin the time vector

        if nrow_out < nrows0:
            # rebin indexing columns
            self.time_col = np.zeros(nrow_out, time_col.dtype)
            self.antea    = np.zeros(nrow_out, antea.dtype)
            self.anteb    = np.zeros(nrow_out, anteb.dtype)
            self.ddid_col = np.zeros(nrow_out, ddid_col.dtype)
            rebinning.rebin_index_columns(self.time_col, time_col,
                               self.antea, antea, self.anteb, anteb, self.ddid_col, ddid_col,
                               self.rebin_row_map)

            self.times, self.uniq_times, _ = data_handler.uniquify(self.time_col)
            self.do_time_rebin = True
            print>> log, "  will rebin into {} rows ({} rebinned timeslots)".format(nrow_out, len(self.uniq_times))
            if self.output_column or self.output_model_column:
                print>> log(0, "red"), "WARNING: output columns will be upsampled from time-binned data!"
        else:
            self.rebin_row_map = np.arange(nrows0, dtype=int)
            # swap conjugate baselines
            conj = self.antea > self.anteb
            if conj.any():
                aa = self.antea[conj].copy()
                self.antea[conj] = self.anteb[conj]
                self.anteb[conj] = aa
                self.rebin_row_map[conj] *= -1

            self.do_time_rebin = False

        ## at the end of this, we have rebinned versions of
        ##      self.time_col, self.antea, self.anteb, self.ddid_col
        ## and self.rebin_row_map, giving the rebinning map (row0->row)
        ## and timechunk_row0, giving the (rebinned) starting row of each chunk


        # Number of timeslots per time chunk
        self.chunk_ntimes = []
        
        # Unique timestamps per time chunk
        self.chunk_timestamps = []
        
        # For each time chunk, create a mask for associated binned and unbinned rows
        timechunk_masks = []
        timechunk_masks0 = []

        timechunk_row = [abs(self.rebin_row_map[row0]) for row0 in timechunk_row0] + [nrow_out]
        timechunk_row0.append(nrows0)

        for tchunk in range(len(timechunk_row0) - 1):
            r0a, r0b = timechunk_row0[tchunk:tchunk + 2]
            ra, rb   = timechunk_row[tchunk:tchunk + 2]
            mask0 = np.zeros(nrows0, bool)
            mask = np.zeros(nrow_out, bool)
            mask0[r0a:r0b] = True
            mask[ra:rb] = True
            timechunk_masks0.append(mask0)
            timechunk_masks.append(mask)
            uniq_ts = np.unique(self.times[ra:rb])
            self.chunk_ntimes.append(len(uniq_ts))
            self.chunk_timestamps.append(np.unique(self.times[ra:rb]))

        # now make list of "row chunks": each element will be a tuple of (ddid, time_chunk_number, rowlist)

        chunklist = []

        self._actual_ddids = []
        for ddid in self._ddids:
            ddid_rowmask0 = ddid_col0==ddid
            ddid_rowmask = self.ddid_col==ddid
            if ddid_rowmask.any():
                self._actual_ddids.append(ddid)
                for tchunk, (mask0, mask) in enumerate(zip(timechunk_masks0, timechunk_masks)):
                    rows = np.where(ddid_rowmask & mask)[0]
                    if rows.size:
                        rows0 = np.where(ddid_rowmask0 & mask0)[0]
                        timeslice = slice(self.times[rows[0]], self.times[rows[-1]]+1)
                        chunklist.append(RowChunk(ddid, tchunk, timeslice, rows, rows0))
        self.nddid_actual = len(self._actual_ddids)

        print>>log,"  generated {} row chunks based on time and DDID".format(len(chunklist))

        # re-sort these row chunks into naturally increasing order (by first row of each chunk)
        def _compare_chunks(a, b):
            return cmp(a.rows[0], b.rows[0])
        chunklist.sort(cmp=_compare_chunks)

        # now, break the row chunks into tiles. Tiles are an "atom" of I/O. First, we try to define each tile as a
        # sequence of overlapping row chunks (i.e. chunks such that the first row of a subsequent chunk comes before
        # the last row of the next chunk).
        # Effectively, whether DDIDs are interleaved with timeslots or not, all per-DDIDs chunks will be grouped into a
        # single tile.
        # (If DDIDs are unequal in shape, we'll use tab.selectrows() to read them in individually.)
        # It is also possible that we end up with one chunk = one tile (i.e. no chunks overlap).


        tile_list = []
        for chunk in chunklist:
            # if rows do not overlap, start new tile with this chunk
            if not tile_list or chunk.rows0[0] > tile_list[-1].last_row0:
                tile_list.append(MSTile(self,chunk))
            # else extend previous tile
            else:
                tile_list[-1].append(chunk)

        print>> log, "  row chunks yield {} potential tiles".format(len(tile_list))

        # now, for effective I/O and parallelisation, we need to have a minimum amount of chunks per tile.
        # Coarsen our tiles to achieve this
        coarser_tile_list = [tile_list[0]]
        for tile in tile_list[1:]:
            cur_chunks = coarser_tile_list[-1].total_tf_chunks()
            new_chunks = cur_chunks + tile.total_tf_chunks()
            # start new "coarse tile" if previous coarse tile already has the min number of chunks, else
            # merge tiles together
            if cur_chunks > chunks_per_tile or new_chunks > (max_chunks_per_tile or 1e+999):
                coarser_tile_list.append(tile)
            else:
                coarser_tile_list[-1].merge(tile)

        tile_list = coarser_tile_list
        for i, tile in enumerate(tile_list):
            tile.finalize("tile #{}/{}".format(i+1, len(tile_list)))

        max_chunks = max([tile.total_tf_chunks() for tile in tile_list])

        print>> log, "  coarsening this to {} tiles (max {} chunks per tile, based on {}/{} requested)".format(
            len(tile_list), max_chunks, chunks_per_tile, max_chunks_per_tile)

        return max_chunks, tile_list

    def define_flags(self, tile_list, flagopts):

        reinit_bitflags = flagopts.get("reinit-bitflags")
        apply_flags  = flagopts.get("apply")
        save_bitflag = flagopts.get("save")
        save_flags   = flagopts.get("save-legacy")
        auto_init    = flagopts.get("auto-init") or reinit_bitflags

        # Do we have a proper bitflag column?
        bitflags = None
        if "BITFLAG" in self.ms.colnames():
            print>> log(1), "checking MS BITFLAG column"
            # asked to re-initialize: blow it away
            if reinit_bitflags:
                print>> log(0, "red"), "will re-initialize BITFLAG column, since --flags-reinit-bitflags is set."
                print>> log(0, "red"), "WARNING: current state of FLAG column will be used to init bitflags!"
            # check for consistency: BITFLAG_ROW must be present too
            elif "BITFLAG_ROW" not in self.ms.colnames():
                print>> log(0, "red"), "WARNING: the BITFLAG_ROW column does not appear to be properly initialized. " \
                                       "This is perhaps due to a previous CubiCal run being interrupted while it was filling the column. "
            # auto-fill keyword must be cleared (otherwise a filling loop was interrupted)
            elif "AUTOINIT_IN_PROGRESS" in self.ms.colkeywordnames("BITFLAG"):
                print>> log(0, "red"), "WARNING: the BITFLAG column does not appear to be properly initialized. " \
                                       "This is perhaps due to a previous CubiCal run being interrupted while it was filling the column. "
            # all cells must be defined
            elif not all([self.data.iscelldefined("BITFLAG", i) for i in xrange(self.data.nrows())]):
                print>> log(0, "red"), "WARNING: the BITFLAG column appears to have missing cells. " \
                                       "This is perhaps due to a previous CubiCal run being interrupted while it was filling the column. "
            # OK, it's valid as best as we can tell
            else:
                print>> log(0), "the MS appears to have a properly formed BITFLAG column"
                bitflags = flagging.Flagsets(self.ms)

            # If no bitflags at this stage (though the column exists), then blow it away if auto_init is enabled.
            # Note that this arises only if (a) the column is malformed, or (b) --flags-reinit-bitflags was
            # explicitly set (which implies auto_init is set)
            if not bitflags and auto_init:
                for kw in self.ms.colkeywordnames("BITFLAG"):
                    self.ms.removecolkeyword("BITFLAG", kw)
                self.ms.removecols("BITFLAG")
                if "BITFLAG_ROW" in self.ms.colnames():
                    self.ms.removecols("BITFLAG_ROW")
                print>> log(0, "red"), "removing current BITFLAG/BITFLAG_ROW columns"
                self.reopen()

        self._apply_flags = self._apply_bitflags = self._save_bitflag = self._auto_fill_bitflag = None
        self._save_flags = bool(save_bitflag) if save_flags == "auto" else save_flags
        self._save_flags_apply = save_flags == 'apply'

        # Insert BITFLAG column, if so specified (note it may have been blown away above)
        if not bitflags and auto_init:
            self._add_column("BITFLAG", like_type='int')
            if "BITFLAG_ROW" not in self.ms.colnames():
                self._add_column("BITFLAG_ROW", like_col="FLAG_ROW", like_type='int')
            self.reopen()
            bitflags = flagging.Flagsets(self.ms)

        if auto_init:
            if type(auto_init) is not str:
                raise ValueError("Illegal --flags-auto-init setting -- a flagset name such as 'legacy' must be specified")
            if auto_init in bitflags.names():
                print>>log(0), "  bitflag '{}' already exists, will not auto-fill".format(auto_init)
            else:
                print>>log(0, "blue"), "  auto-filling bitflag '{}' from FLAG/FLAG_ROW column. Please do not interrupt this process!".format(auto_init)
                print>>log(0), "    note that all other bitflags will be cleared by this"
                self.ms.putcolkeyword("BITFLAG", "AUTOINIT_IN_PROGRESS", True)
                self._auto_fill_bitflag = bitflags.flagmask(auto_init, create=True)

                for itile, tile in enumerate(tile_list):
                    tile.fill_bitflags(self._auto_fill_bitflag)

                self.ms.removecolkeyword("BITFLAG", "AUTOINIT_IN_PROGRESS")
                print>>log(0, "blue"), "  auto-fill complete"


        # init flagcounts dict
        self.flagcounts = OrderedDict(TOTAL=0, FLAG=0)

        if bitflags:
            self._apply_flags = None
            self._apply_bitflags = 0
            if apply_flags:
                if type(apply_flags) is list:
                    apply_flags = ",".join(apply_flags)
                # --flags-apply specified as a bitmask, or a single string, or a single negated string, or a list of strings
                if type(apply_flags) is int:
                    self._apply_bitflags = apply_flags
                elif type(apply_flags) is not str:
                    raise ValueError("Illegal --flags-apply setting -- string or bitmask values expected")
                else:
                    print>>log,"  BITFLAG column defines the following flagsets: {}".format(
                        " ".join(['{}:{}'.format(name, bitflags.bits[name]) for name in bitflags.names()]))
                    if apply_flags == "FLAG":
                        self._apply_flags = True
                    elif apply_flags[0] == '-':
                        flagset = apply_flags[1:]
                        print>> log(0), "  will exclude flagset {}".format(flagset)
                        if flagset not in bitflags.bits:
                            print>>log(0,"red"),"    flagset '{}' not found -- ignoring".format(flagset)
                        self._apply_bitflags = sum([bitmask for fset, bitmask in bitflags.bits.iteritems() if fset != flagset])
                    else:
                        print>> log(0), "  will apply flagset(s) {}".format(apply_flags)
                        apply_flags = apply_flags.split(",")
                        for flagset in apply_flags:
                            if flagset not in bitflags.bits:
                                print>>log(0,"red"),"    flagset '{}' not found -- ignoring".format(flagset)
                            else:
                                self._apply_bitflags |= bitflags.bits[flagset]
            if self._apply_flags:
                print>> log, "  using flags from FLAG/FLAG_ROW columns"
            if self._apply_bitflags:
                print>> log(0, "blue"), "  applying BITFLAG mask {} to input data".format(self._apply_bitflags)
            elif not self._apply_flags:
                print>> log(0, "red"), "  no input flags will be applied!"
            if save_bitflag:
                self._save_bitflag = bitflags.flagmask(save_bitflag, create=True)
                if self._save_flags:
                    if self._save_flags_apply:
                        print>> log(0,"blue"), "  will save output flags into BITFLAG '{}' ({}), and all flags (including input) FLAG/FLAG_ROW".format(
                            save_bitflag, self._save_bitflag)
                    else:
                        print>> log(0,"blue"), "  will save output flags into BITFLAG '{}' ({}), and into FLAG/FLAG_ROW".format(save_bitflag, self._save_bitflag)
                else:
                    print>> log(0,"red"), "  will save output flags into BITFLAG '{}' ({}), but not into FLAG/FLAG_ROW".format(save_bitflag, self._save_bitflag)
            else:
                if self._save_flags:
                    if self._save_flags_apply:
                        print>> log(0, "blue"), "  will save all flags (including input) into FLAG/FLAG_ROW"
                    else:
                        print>> log(0, "blue"), "  will save output flags into FLAG/FLAG_ROW"

            for flagset in bitflags.names():
                self.flagcounts[flagset] = 0
            self.bitflags = bitflags.bits

        # else no BITFLAG -- fall back to using FLAG/FLAG_ROW if asked, but definitely can't save

        else:
            if save_bitflag:
                raise RuntimeError("No BITFLAG column in this MS. Either use --flags-auto-init to insert one, or disable --flags-save.")
            self._apply_flags = bool(apply_flags)
            self._apply_bitflags = 0
            if self._apply_flags:
                print>> log, ModColor.Str("  no BITFLAG column in this MS. Using flags from FLAG/FLAG_ROW columns")
            else:
                print>> log, ModColor.Str("  no flags will be read, since --flags-apply was not set")
            if self._save_flags:
                print>> log(0, "blue"), "  will save output flags into into FLAG/FLAG_ROW"
            self._save_flags_apply = False   # no point in saving input flags, as nothing would change
            self.bitflags = {}

        self.flagcounts['DESEL'] = 0
        self.flagcounts['IN'] = 0
        self.flagcounts['NEW'] = 0
        self.flagcounts['OUT'] = 0


    def update_flag_counts(self, counts):
        self.flagcounts.update(counts)

    def get_flag_counts(self):
        total = float(self.flagcounts['TOTAL'])
        result = []
        for name, count in self.flagcounts.iteritems():
            if name != 'TOTAL':
                result.append("{}:{:.2%}".format(name, count/total))
        return result

    def flag3_to_col(self, flag3):
        """
        Converts a 3D flag cube (ntime, nddid, nchan) back into the MS style.

        Args:
            flag3 (np.ndarray): 
                Input array which is to be made MS friendly.

        Returns:
            np.ndarray:
                Boolean array with same shape as self.obvis.
        """
        flagout = np.zeros(self._datashape, bool)

        flagout[:] = flag3[self.times, self.ddid_col, :, np.newaxis]

        return flagout

    def add_to_gain_dict(self, gains, bounds, t_int=1, f_int=1):
        """
        Adds a gain array to the gain dictionary.

        Args:
            gains (np.ndarray):
                Gains for the current chunk.
            bounds (tuple):
                Tuple of (ddid, timechunk, first_f, last_f).
            t_int (int, optional):
                Number of timeslots per solution interval.
            f_int (int, optional):
                Number of frequencies per soultion interval.
        """

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
        """
        Writes out a gain dictionary to disk.

        Args:
            output_name (str or None, optional):
                Name of output pickle file.
        """

        if output_name is None:
            output_name = self.ms_name + "/gains.p"

        cPickle.dump(self.gain_dict, open(output_name, "wb"), protocol=2)

    def _add_column (self, col_name, like_col="DATA", like_type=None):
        """
        Inserts a new column into the measurement set.

        Args:
            col_name (str): 
                Name of target column.
            like_col (str, optional): 
                Column will be patterned on the named column.
            like_type (str or None, optional): 
                If set, column type will be changed.

        Returns:
            bool:
                True if a new column was inserted, else False.
        """

        if col_name not in self.ms.colnames():
            # new column needs to be inserted -- get column description from column 'like_col'
            print>> log, "  inserting new column %s" % (col_name)
            desc = self.ms.getcoldesc(like_col)
            desc['name'] = col_name
            desc['comment'] = desc['comment'].replace(" ", "_")  # got this from Cyril, not sure why
            dminfo = self.ms.getdminfo(like_col)
            dminfo["NAME"] =  "{}-{}".format(dminfo["NAME"], col_name)
            # if a different type is specified, insert that
            if like_type:
                desc['valueType'] = like_type
            self.ms.addcols(desc, dminfo)
            return True
        return False

    def finalize(self):
        self.unlock()

    def unlock(self):
        """ Unlocks the measurement set. """

        if self.data is not None:
            self.data.unlock()
        self.ms.unlock()

    def lock(self):
        """ Locks the measurement set. """

        self.ms.lock()
        if self.data is not None:
            self.data.lock()

    def close(self):
        """ Closes the measurement set. """

        if self.data is not None:
            self.data.close()
        if self.ms is not None:
            self.ms.close()
        self.data = self.ms = None

    def flush(self):
        """ Flushes the measurement set. """

        if self.data is not None:
            self.data.flush()
        self.ms.flush()

    def reopen(self):
        """ Reopens the MS. Unfortunately, this is needed when new columns are added. """
        self.close()
        self.ms = pt.table(self.ms_name, readonly=False, ack=False)
        sortlist = "TIME"
        if self.taql:
            self.data = self.ms.query(self.taql, sortlist=sortlist)
        else:
            self.data = self.ms.sort(sortlist)

    def save_flags(self, flags):
        """
        Saves flags to column in MS.

        Args:
            flags (np.ndarray): 
                Flag values to be written to column.
        """
        
        print>>log,"Writing out new flags"
        try:
            bflag_col = self.fetch("BITFLAG")
        except Exception:
            if not self._auto_fill_bitflag:
                print>> log, ModColor.Str(traceback.format_exc().strip())
                print>> log, ModColor.Str("Error reading BITFLAG column, and --flags-auto-init is not set.")
                raise
            print>> log(0,"red"), "Error reading BITFLAG column: not fatal, since we'll auto-fill it from FLAG"
            print>> log(0,"red"), "However, it really should have been filled above, so this may be a bug."
            print>> log(0,"red"), "Please save your logfile and contact the developers."
            for line in traceback.format_exc().strip().split("\n"):
                print>> log, "    " + line
            flag_col = self.fetch("FLAG")
            bflag_col = np.zeros(flag_col.shape, np.int32)
            bflag_col[flag_col] = self._auto_fill_bitflag
        # raise specified bitflag
        print>> log, "  updating BITFLAG column flagbit %d"%self._save_bitflag
        #bflag_col[:, self._channel_slice, :] &= ~self._save_bitflag         # clear the flagbit first
        bflag_col[:, self._channel_slice, :][flags] |= self._save_bitflag
        self.data.putcol("BITFLAG", bflag_col)
        print>>log, "  updating BITFLAG_ROW column"
        self.data.putcol("BITFLAG_ROW", np.bitwise_and.reduce(bflag_col, axis=(-1,-2)))
        flag_col = bflag_col != 0
        print>> log, "  updating FLAG column ({:.2%} visibilities flagged)".format(
                                                                flag_col.sum()/float(flag_col.size))
        self.data.putcol("FLAG", flag_col)
        flag_row = flag_col.all(axis=(-1,-2))
        print>> log, "  updating FLAG_ROW column ({:.2%} rows flagged)".format(
                                                                flag_row.sum()/float(flag_row.size))
        self.data.putcol("FLAG_ROW", flag_row)
        self.data.flush()

