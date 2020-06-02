# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
from builtins import range
from cubical.database.pickled_db import PickledDatabase
from cubical.data_handler.ms_data_handler import MSDataHandler
from cubical.tools import logger
from pyrap.tables import table as tbl
import os
import shutil
import numpy as np
import subprocess
import six

log = logger.getLogger("casa_db_adaptor")

# to the tune of KATDAL :)
BLANK_TABLE_NAME = 'blankcaltable.CASA'
BLANK_TABLE_TARBALL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "{}.tgz".format(BLANK_TABLE_NAME))

class casa_caltable_factory(object):
        """
            Gaintable Factory
            
            Useful methods for creating and writing CASA-style gaintables
        """
        
        @classmethod
        def init_empty(cls, db, filename, solfreqs, solants, field_ndir=1, is_complex=True, viscal_label="B Jones"):
            """
                Initialize empty calibration table with metadata
                
                Args:
                    db: a cubical pickled_db instance
                    filename: filename of CASA gaintable to be written
                    solfreqs: centre frequencies of solutions to be stored in table
                    solants: Names from Cubical database antenna axis, used to order solutions
                    field_ndir: Number of solvable directions in field
                    is_complex: Solutions are complex or real-valued
                    viscal_label: Sets viscal property of CASA table - used as identifier in CASA
            """
            if six.PY3:
                log.error("Gaintables cannot be written in Python 3 mode due to current casacore implementation issues")
                return
            if os.path.exists(filename):
                if os.path.isfile(filename):
                    log.error("CASA calibration table destination already exists but is not a directory. Will not remove.")    
                    return
                else:
                    log.info("Destination CASA gain table '%s' exists. Will overwrite." % filename)
                    shutil.rmtree(filename) # CASA convention is to overwrite
                    
            basedir = os.path.dirname(filename) or '.'
            subprocess.check_output(["tar", "zxvf", BLANK_TABLE_TARBALL, "-C", basedir])
            os.rename(os.path.join(basedir, BLANK_TABLE_NAME), filename)

            antorder = [db.antnames.index(an) for an in solants]
            with tbl("%s::ANTENNA" % str(filename), ack=False, readonly=False) as t:
                t.addrows(nrows=len(db.anttype))
                t.putcol("OFFSET", db.antoffset[antorder])
                t.putcol("POSITION", db.antpos[antorder])
                t.putcol("TYPE", np.array(db.anttype)[antorder])
                t.putcol("DISH_DIAMETER", db.antdishdiam[antorder])
                t.putcol("FLAG_ROW", db.antflagrow[antorder])
                t.putcol("MOUNT", np.array(db.antmount)[antorder])
                t.putcol("NAME", np.array(db.antnames)[antorder])
                t.putcol("STATION", np.array(db.antstation)[antorder])
                
            assert "field" in db.metadata, "Solver field not passed in metadata. This is a bug"
            assert type(db.metadata["field"]) is int, "Currently only supports single field"
            selfield = np.arange(len(db.fieldname)) == db.metadata["field"]
            with tbl("%s::FIELD" % str(filename), ack=False, readonly=False) as t:
                t.addrows(nrows=field_ndir)
                t.putcol("DELAY_DIR", np.tile(db.fielddelaydirs[selfield], (field_ndir, 1)))
                t.putcol("PHASE_DIR", np.tile(db.fieldphasedirs[selfield], (field_ndir, 1)))
                t.putcol("REFERENCE_DIR", np.tile(db.fieldrefdir[selfield], (field_ndir, 1)))
                t.putcol("CODE", np.tile(np.array(db.fieldcode)[selfield], (field_ndir, 1)))
                t.putcol("FLAG_ROW", np.tile(db.fieldflagrow[selfield], (field_ndir, 1)))
                t.putcol("NAME", np.array(map(str, ["%s_DIR_%d" % (f, fdi) for fdi, f in enumerate([db.fieldname[np.where(selfield)[0][0]]] * field_ndir)])).T)
                t.putcol("SOURCE_ID", np.tile(db.fieldsrcid[selfield], (field_ndir, 1)) + np.arange(field_ndir).T)
                t.putcol("TIME", np.tile(db.fieldtime[selfield], (field_ndir, 1)))
    
            with tbl("%s::OBSERVATION" % str(filename), ack=False, readonly=False) as t:
                t.addrows(nrows=len(db.obsobserver))
                (len(db.obstimerange) != 0) and t.putcol("TIME_RANGE", db.obstimerange)
                (len(db.obslog) != 0) and t.putcol("LOG", db.obslog)
                (len(db.obsschedule) != 0) and t.putcol("SCHEDULE", db.obsschedule)
                (len(db.obsflagrow) != 0) and t.putcol("FLAG_ROW", db.obsflagrow)
                (len(db.obsobserver) != 0) and t.putcol("OBSERVER", db.obsobserver)
                (len(db.obsproject) != 0) and t.putcol("PROJECT", db.obsproject)
                (len(db.obsreleasedate) != 0) and t.putcol("RELEASE_DATE", db.obsreleasedate)
                (len(db.obstelescopename) != 0) and t.putcol("TELESCOPE_NAME", db.obstelescopename)
            
            with tbl("%s::SPECTRAL_WINDOW" % str(filename), ack=False, readonly=False) as t:
                t.addrows(nrows=len(db.sel_ddids))
                # Per DDID determine solution spacing in frequency
                for iddid, ddid in enumerate(db.sel_ddids):
                    spwid = db.ddid_spw_map[ddid]
                    minfreq = np.min(db.spwchanfreq[spwid] - 0.5 * db.spwchanwidth[spwid])
                    maxfreq = np.max(db.spwchanfreq[spwid] + 0.5 * db.spwchanwidth[spwid]) 
                    ddsolfreqs = solfreqs[np.logical_and(solfreqs >= minfreq,
                                                         solfreqs <= maxfreq)]
                    # assume linearly-spaced solutions for N - 1 solutions along grid and last solution 
                    # may have different spacing if solution interval does not exactly divide number of channels
                    if ddsolfreqs.size > 1:
                        ch0 = np.min(db.spwchanfreq)
                        chN = np.max(db.spwchanfreq)
                        ddsolwidthmax = 2 * (ddsolfreqs[0] - ch0)
                        ddsolwidth = np.ones(ddsolfreqs.size) * ddsolwidthmax
                        ddsolwidth[-1] = ddsolwidthmax + (chN - (ddsolfreqs.size * ddsolwidthmax + ch0))
                    else:
                        ddsolwidth = db.spwtotalbandwidth
                        
                    t.putcell("MEAS_FREQ_REF", iddid, db.spwmeasfreq[spwid])
                    t.putcell("CHAN_FREQ", iddid, ddsolfreqs)
                    t.putcell("REF_FREQUENCY", iddid, db.spwreffreq[spwid])
                    t.putcell("CHAN_WIDTH", iddid, ddsolwidth)
                    t.putcell("EFFECTIVE_BW", iddid, ddsolwidth) # TODO: this may not be true
                    t.putcell("RESOLUTION", iddid, db.spwresolution[spwid])
                    t.putcell("FLAG_ROW", iddid, db.spwflagrow[spwid])
                    t.putcell("FREQ_GROUP", iddid, db.spwfreqgroup[spwid])
                    t.putcell("FREQ_GROUP_NAME", iddid, db.spwfreqgroupname[spwid])
                    t.putcell("IF_CONV_CHAIN", iddid, db.spwifconvchain[spwid])
                    t.putcell("NAME", iddid, str(db.spwname[spwid]))
                    t.putcell("NET_SIDEBAND", iddid, db.spwnetsideband[spwid])
                    t.putcell("NUM_CHAN", iddid, ddsolfreqs.size)
                    t.putcell("TOTAL_BANDWIDTH", iddid, maxfreq - minfreq)
                    
            with tbl(str(filename), ack=False, readonly=False) as t:
                t.putkeyword("ParType", "Complex" if is_complex else "Float")
                t.putkeyword("VisCal", viscal_label)
                
                if not is_complex:
                    cdesc = t.getcoldesc("CPARAM")
                    cdesc["valueType"] = "float"
                    t.addcols({"FPARAM": cdesc})
                    t.removecols("CPARAM")
                        
        @classmethod
        def create_G_table(cls, db, gname, outname = "Gphase"):
            """
                Write diagonal G-phase caltable
                
                Args:
                    db: a cubical pickled_db instance
                    gname: name of pickled_db solutions to export
                    outname: suffix of exported CASA gaintable
            """
            if six.PY3:
                log.error("Gaintables cannot be written in Python 3 mode due to current casacore implementation issues")
                return
            if np.prod(db[gname].shape) == 0:
                log.warn("No %s solutions. Will not write CASA table" % gname)
                return
            paramerrs = cls.__check_param_err(db, gname)
            assert db[gname].axis_labels == ('dir', 'time', 'freq', 'ant', 'corr'), "DB table in unrecognized format"
            
            ddids = db.sel_ddids
            ndir = len(db[gname].grid[db[gname].ax.dir])
            ntime = len(db[gname].grid[db[gname].ax.time])
            nant = len(db[gname].grid[db[gname].ax.ant])
            ncorr = len(db[gname].grid[db[gname].ax.corr])
            nddids = len(db.sel_ddids)
            nrow = ndir * ntime * \
                    nant * len(ddids)
            assert ncorr == 2, "Expected diagnonal Jones matrix"
            
            cls.init_empty(db, 
                           db.filename + ".%s.casa" % outname, 
                           db[gname].grid[db[gname].ax.freq],
                           db[gname].grid[db[gname].ax.ant],
                           field_ndir=ndir,
                           viscal_label="G Jones")
            
            with tbl(str(db.filename + ".%s.casa" % outname), ack=False, readonly=False) as t:
                t.addrows(nrows=nrow)
                for iddid, ddid in enumerate(db.sel_ddids):
                    spwid = db.ddid_spw_map[ddid]
                    minfreq = np.min(db.spwchanfreq[spwid] - 0.5 * db.spwchanwidth[spwid])
                    maxfreq = np.max(db.spwchanfreq[spwid] + 0.5 * db.spwchanwidth[spwid]) 
                    ddsolfreqindx = np.argwhere(np.logical_and(db[gname].grid[db[gname].ax.freq] >= minfreq,
                                                               db[gname].grid[db[gname].ax.freq] <= maxfreq))
                    params = np.swapaxes(db[gname].get_cube()[:, :, ddsolfreqindx, :, :],
                                         2, 3).reshape(ndir * ntime * nant, len(ddsolfreqindx), ncorr) 
                    paramerrs = np.swapaxes(paramerrs[:, :, ddsolfreqindx, :, :],
                                            2, 3).reshape(ndir * ntime * nant, len(ddsolfreqindx), ncorr) 
                    flags = np.ma.getmaskarray(params)
                    fieldid = np.repeat(np.arange(ndir), ntime * nant) # dir (marked as field) is slowest varying
                    time = np.repeat(np.tile(db[gname].grid[db[gname].ax.time], ndir), nant)
                    ant1 = np.tile(np.arange(nant), ndir * ntime) # FK ndir * ntime blocks
                    #ant2 can be the same - it is not used unless specifying mueller matricies
                    nrowsdd = ntime *  nant * ndir
                    t.putcol("TIME", time, startrow=nrowsdd * iddid)
                    t.putcol("FIELD_ID", fieldid, startrow=nrowsdd * iddid)
                    t.putcol("SPECTRAL_WINDOW_ID", np.ones(ant1.shape) * iddid, startrow=nrowsdd * iddid) # new spectral window with freq range for these sols
                    t.putcol("ANTENNA1", ant1, startrow=nrowsdd * iddid)
                    t.putcol("ANTENNA2", np.ones(ant1.shape) * -1, startrow=nrowsdd * iddid) 
                    t.putcol("INTERVAL", np.zeros(ant1.shape), startrow=nrowsdd * iddid) #TODO: unclear from MEMO 229
                    t.putcol("SCAN_NUMBER", np.ones(ant1.shape) * -1, startrow=nrowsdd * iddid) #TODO this FK info is not available yet @oms
                    t.putcol("OBSERVATION_ID", np.ones(ant1.shape) * -1, startrow=nrowsdd * iddid) #TODO this FK info is not available yet @oms
                    t.putcol("CPARAM", np.ma.getdata(params), startrow=nrowsdd * iddid)
                    t.putcol("PARAMERR", np.ma.getdata(paramerrs), startrow=nrowsdd * iddid)
                    t.putcol("FLAG", flags, startrow=nrowsdd * iddid)
                    t.putcol("SNR", np.ones(params.shape) * np.inf, startrow=nrowsdd * iddid) #TODO this is not available @oms
                    t.putcol("WEIGHT", np.ones(params.shape), startrow=nrowsdd * iddid) #TODO this is not available @oms
        @classmethod
        def __check_param_err(cls, db, gname):
            if db[gname].shape != db[gname + ".err"].shape:
                log.warn("PARAM err shape %s does not match PARAM shape %s. This is a solver bug so we will assume infinite errors on all solutions." %
                         (",".join(map(str, db[gname].shape)),
                          ",".join(map(str, db[gname + ".err"].shape))))
                return np.zeros_like(db[gname].get_cube(), dtype=db[gname + ".err"].get_cube().dtype) * np.inf
            return db[gname + ".err"].get_cube()
        @classmethod
        def create_B_table(cls, db, gname, outname = "B", diag=True):
            """
                Write diagonal B-Jones caltable
                
                Args:
                    db: a cubical pickled_db instance
                    gname: name of pickled_db solutions to export
                    outname: suffix of exported CASA gaintable
                    diag: Write out diagonal of Jones matrix if true, off-diagonal (leakage) terms otherwise.
            """
            if six.PY3:
                log.error("Gaintables cannot be written in Python 3 mode due to current casacore implementation issues")
                return
            if np.prod(db[gname].shape) == 0:
                log.warn("No %s solutions. Will not write CASA table" % gname)
                return
            paramerrs = cls.__check_param_err(db, gname)
            assert db[gname].axis_labels == ('dir', 'time', 'freq', 'ant', 'corr1', 'corr2'), "DB table in unrecognized format"
            
            ddids = db.sel_ddids
            ndir = len(db[gname].grid[db[gname].ax.dir])
            ntime = len(db[gname].grid[db[gname].ax.time])
            nant = len(db[gname].grid[db[gname].ax.ant])
            ncorr1 = len(db[gname].grid[db[gname].ax.corr1])
            ncorr2 = len(db[gname].grid[db[gname].ax.corr2])
            nddids = len(db.sel_ddids)
            nrow = ndir * ntime * \
                    nant * len(ddids)
            assert ncorr1 == ncorr2 and ncorr1 == 2, "Expected 2x2 solution, this is a bug"
            
            cls.init_empty(db, 
                           db.filename + ".%s.casa" % outname, 
                           db[gname].grid[db[gname].ax.freq],
                           db[gname].grid[db[gname].ax.ant],
                           field_ndir=ndir,
                           viscal_label="B Jones" if diag else "D Jones")
            
            with tbl(str(db.filename) + ".%s.casa" % outname, ack=False, readonly=False) as t:
                t.addrows(nrows=nrow)
                
                for iddid, ddid in enumerate(db.sel_ddids):
                    spwid = db.ddid_spw_map[ddid]
                    minfreq = np.min(db.spwchanfreq[spwid] - 0.5 * db.spwchanwidth[spwid])
                    maxfreq = np.max(db.spwchanfreq[spwid] + 0.5 * db.spwchanwidth[spwid]) 
                    ddsolfreqindx = np.argwhere(np.logical_and(db[gname].grid[db[gname].ax.freq] >= minfreq,
                                                               db[gname].grid[db[gname].ax.freq] <= maxfreq))
                    jones_entries = [0, 3] if diag else [1, 2] # diagonal or crosshands
                    params = np.swapaxes(db[gname].get_cube()[:, :, ddsolfreqindx, :, :], 
                                         2, 3).reshape(ndir * ntime * nant, len(ddsolfreqindx), ncorr1 * ncorr2)[:, :, jones_entries]
                    paramerrs = np.swapaxes(paramerrs[:, :, ddsolfreqindx, :, :],
                                            2, 3).reshape(ndir * ntime * nant, len(ddsolfreqindx), ncorr1 * ncorr2)[:, :, jones_entries]
                    flags = np.ma.getmaskarray(params)
                    fieldid = np.repeat(np.arange(ndir), ntime * nant) # dir (marked as field) is slowest varying
                    time = np.repeat(np.tile(db[gname].grid[db[gname].ax.time], ndir), nant)
                    ant1 = np.tile(np.arange(nant), ndir * ntime) # FK ndir * ntime blocks
                    #ant2 can be the same - it is not used unless specifying mueller matricies
                    nrowsdd = ntime *  nant * ndir
                    t.putcol("TIME", time, startrow=nrowsdd * iddid)
                    t.putcol("FIELD_ID", fieldid, startrow=nrowsdd * iddid)
                    t.putcol("SPECTRAL_WINDOW_ID", np.ones(ant1.shape) * iddid, startrow=nrowsdd * iddid) # new spectral window with freq range for these sols
                    t.putcol("ANTENNA1", ant1, startrow=nrowsdd * iddid)
                    t.putcol("ANTENNA2", np.ones(ant1.shape) * -1, startrow=nrowsdd * iddid) 
                    t.putcol("INTERVAL", np.zeros(ant1.shape), startrow=nrowsdd * iddid) #TODO: unclear from MEMO 229
                    t.putcol("SCAN_NUMBER", np.ones(ant1.shape) * -1, startrow=nrowsdd * iddid) #TODO this FK info is not available yet @oms
                    t.putcol("OBSERVATION_ID", np.ones(ant1.shape) * -1, startrow=nrowsdd * iddid) #TODO this FK info is not available yet @oms
                    t.putcol("CPARAM", np.ma.getdata(params), startrow=nrowsdd * iddid)
                    t.putcol("PARAMERR", np.ma.getdata(paramerrs), startrow=nrowsdd * iddid)
                    t.putcol("FLAG", flags, startrow=nrowsdd * iddid)
                    t.putcol("SNR", np.ones(params.shape) * np.inf, startrow=nrowsdd * iddid) #TODO this is not available @oms
                    t.putcol("WEIGHT", np.ones(params.shape), startrow=nrowsdd * iddid) #TODO this is not available @oms
                    
        @classmethod
        def create_D_table(cls, db, gname, outname = "D"):
            """
                Wrapper for off-diagonal B-Jones (leakage / "D") Jones table
                
                Args:
                    db: a cubical pickled_db instance
                    gname: name of pickled_db solutions to export
                    outname: suffix of exported CASA gaintable
            """
            if six.PY3:
                log.error("Gaintables cannot be written in Python 3 mode due to current casacore implementation issues")
                return
            cls.create_B_table(db, gname, outname, diag=False)
        
        @classmethod
        def create_K_table(cls, db, gname, outname = "K"):
            """
                Write real-valued K-Jones table
                
                Args:
                    db: a cubical pickled_db instance
                    gname: name of pickled_db solutions to export
                    outname: suffix of exported CASA gaintable
            """
            if six.PY3:
                log.error("Gaintables cannot be written in Python 3 mode due to current casacore implementation issues")
                return
            if np.prod(db[gname].shape) == 0:
                log.warn("No %s solutions. Will not write CASA table" % gname)
                return
            paramerrs = cls.__check_param_err(db, gname)
            assert db[gname].axis_labels == ('dir', 'time', 'freq', 'ant', 'corr'), "DB table in unrecognized format"
            
            ddids = db.sel_ddids
            ndir = len(db[gname].grid[db[gname].ax.dir])
            ntime = len(db[gname].grid[db[gname].ax.time])
            nant = len(db[gname].grid[db[gname].ax.ant])
            ncorr = len(db[gname].grid[db[gname].ax.corr])
            nddids = len(db.sel_ddids)
            nrow = ndir * ntime * \
                    nant * len(ddids)
            assert ncorr == 2, "Expected diagnonal Jones matrix"
            
            cls.init_empty(db, 
                           db.filename + ".%s.casa" % outname, 
                           db[gname].grid[db[gname].ax.freq],
                           db[gname].grid[db[gname].ax.ant],
                           field_ndir=ndir,
                           is_complex=False,
                           viscal_label="K Jones")
            
            with tbl(str(db.filename + ".%s.casa" % outname), ack=False, readonly=False) as t:
                t.addrows(nrows=nrow)
                for iddid, ddid in enumerate(db.sel_ddids):
                    spwid = db.ddid_spw_map[ddid]
                    minfreq = np.min(db.spwchanfreq[spwid] - 0.5 * db.spwchanwidth[spwid])
                    maxfreq = np.max(db.spwchanfreq[spwid] + 0.5 * db.spwchanwidth[spwid]) 
                    ddsolfreqindx = np.argwhere(np.logical_and(db[gname].grid[db[gname].ax.freq] >= minfreq,
                                                               db[gname].grid[db[gname].ax.freq] <= maxfreq))
                    # note -- CASA K table delays are in nanoseconds. This presumes delays in the cubical tables are already denormalized into seconds
                    params = np.swapaxes(db[gname].get_cube()[:, :, ddsolfreqindx, :, :],
                                         2, 3).reshape(ndir * ntime * nant, len(ddsolfreqindx), ncorr) * 1.0e9
                    paramerrs = np.swapaxes(paramerrs[:, :, ddsolfreqindx, :, :],
                                            2, 3).reshape(ndir * ntime * nant, len(ddsolfreqindx), ncorr) * 1.0e9
                    flags = np.ma.getmaskarray(params)
                    fieldid = np.repeat(np.arange(ndir), ntime * nant) # dir (marked as field) is slowest varying
                    time = np.repeat(np.tile(db[gname].grid[db[gname].ax.time], ndir), nant)
                    ant1 = np.tile(np.arange(nant), ndir * ntime) # FK ndir * ntime blocks
                    #ant2 can be the same - it is not used unless specifying mueller matricies
                    nrowsdd = ntime *  nant * ndir
                    t.putcol("TIME", time, startrow=nrowsdd * iddid)
                    t.putcol("FIELD_ID", fieldid, startrow=nrowsdd * iddid)
                    t.putcol("SPECTRAL_WINDOW_ID", np.ones(ant1.shape) * iddid, startrow=nrowsdd * iddid) # new spectral window with freq range for these sols
                    t.putcol("ANTENNA1", ant1, startrow=nrowsdd * iddid)
                    t.putcol("ANTENNA2", np.ones(ant1.shape) * -1, startrow=nrowsdd * iddid) 
                    t.putcol("INTERVAL", np.zeros(ant1.shape), startrow=nrowsdd * iddid) #TODO: unclear from MEMO 229
                    t.putcol("SCAN_NUMBER", np.ones(ant1.shape) * -1, startrow=nrowsdd * iddid) #TODO this FK info is not available yet @oms
                    t.putcol("OBSERVATION_ID", np.ones(ant1.shape) * -1, startrow=nrowsdd * iddid) #TODO this FK info is not available yet @oms
                    t.putcol("FPARAM", np.ma.getdata(params), startrow=nrowsdd * iddid)
                    t.putcol("PARAMERR", np.ma.getdata(paramerrs), startrow=nrowsdd * iddid)
                    t.putcol("FLAG", flags, startrow=nrowsdd * iddid)
                    t.putcol("SNR", np.ones(params.shape) * np.inf, startrow=nrowsdd * iddid) #TODO this is not available @oms
                    t.putcol("WEIGHT", np.ones(params.shape), startrow=nrowsdd * iddid) #TODO this is not available @oms
        
class casa_db_adaptor(PickledDatabase):
    
    def __init__(self):
        """
            CASA-style gaintable adaptor for Cubical pickled_db database writer
            
            This adaptor adds functionality to export databases to CASA gaintables
            which can be read and plotted in CASA plotms or similar diagnostic utilities.
        """
        PickledDatabase.__init__(self)
        self.meta_avail = False
        self.export_enabled = False
    
    @property
    def export_CASA_gaintable(self):
        return self.export_enabled
    
    @export_CASA_gaintable.setter
    def export_CASA_gaintable(self, value):
        self.export_enabled = value
    
    def set_metadata(self, src):
        """
            Sets up metadata taken from data provider
            
            Args:
                src: a cubical.data_handler instance
        """
        if six.PY3 and self.export_enabled:
            log.error("Gaintables cannot be written in Python 3 mode due to current casacore implementation issues")
            self.export_enabled = False

        if not isinstance(src, MSDataHandler):
            raise TypeError("src must be of type Cubical DataHandler")

        if self.export_enabled:
            self.antoffset = src._anttabcols["OFFSET"]
            self.antpos = src._anttabcols["POSITION"]
            self.anttype = src._anttabcols["TYPE"]
            self.antdishdiam = src._anttabcols["DISH_DIAMETER"]
            self.antflagrow = src._anttabcols["FLAG_ROW"]
            self.antmount = src._anttabcols["MOUNT"]
            self.antnames = src._anttabcols["NAME"]
            self.antstation = src._anttabcols["STATION"]
            self.fielddelaydirs = src._fldtabcols["DELAY_DIR"]
            self.fieldphasedirs = src._fldtabcols["PHASE_DIR"]
            self.fieldrefdir = src._fldtabcols["REFERENCE_DIR"]
            self.fieldcode = src._fldtabcols["CODE"]
            self.fieldflagrow = src._fldtabcols["FLAG_ROW"]
            self.fieldname = src._fldtabcols["NAME"]
            self.fieldnumpoly = src._fldtabcols["NUM_POLY"]
            self.fieldsrcid = src._fldtabcols["SOURCE_ID"]
            self.fieldtime = src._fldtabcols["TIME"]
            self.obstimerange = src._obstabcols["TIME_RANGE"]
            self.obslog = src._obstabcols["LOG"]
            self.obsschedule = src._obstabcols["SCHEDULE"]
            self.obsflagrow = src._obstabcols["FLAG_ROW"]
            self.obsobserver = src._obstabcols["OBSERVER"]
            self.obsproject = src._obstabcols["PROJECT"]
            self.obsreleasedate = src._obstabcols["RELEASE_DATE"]
            self.obsscheduletype = src._obstabcols["SCHEDULE_TYPE"]
            self.obstelescopename = src._obstabcols["TELESCOPE_NAME"]
            self.spwmeasfreq = src._spwtabcols["MEAS_FREQ_REF"]
            self.spwchanfreq = src._spwtabcols["CHAN_FREQ"]
            self.spwreffreq = src._spwtabcols["REF_FREQUENCY"]
            self.spwchanwidth = src._spwtabcols["CHAN_WIDTH"]
            self.spweffbw = src._spwtabcols["EFFECTIVE_BW"]
            self.spwresolution = src._spwtabcols["RESOLUTION"]
            self.spwflagrow = src._spwtabcols["FLAG_ROW"]
            self.spwfreqgroup = src._spwtabcols["FREQ_GROUP"]
            self.spwfreqgroupname = src._spwtabcols["FREQ_GROUP_NAME"]
            self.spwifconvchain = src._spwtabcols["IF_CONV_CHAIN"]
            self.spwname = src._spwtabcols["NAME"]
            self.spwnetsideband = src._spwtabcols["NET_SIDEBAND"]
            self.spwnumchan = src._spwtabcols["NUM_CHAN"]
            self.spwtotalbandwidth = src._spwtabcols["TOTAL_BANDWIDTH"]
            self.ddid_spw_map = src._ddid_spw
            self.sel_ddids = src._ddids
            self.do_write_casatbl = True
            self.meta_avail = True
        else:
            self.meta_avail = False
            

    def __export(self):
        """ exports the database to CASA gaintables """
        if six.PY3:
            log.error("Gaintables cannot be written in Python 3 mode due to current casacore implementation issues")
            return
        self._load(self.filename)
        
        if not self.meta_avail:
            log.error("Measurement metadata not set. Cannot write CASA-style gaintable.")
        elif not self.export_enabled:
            log.warning("By user request not exporting CASA-style gaintables.")
        else:
            log.info("Exporting to CASA gaintables")
            if "G:phase" in self.names():
                assert "G:phase.err" in self.names(), "Gain phase error not present in solutions db? This is a bug"
                casa_caltable_factory.create_G_table(self, "G:phase")
            if "G:gain" in self.names():
                assert "G:gain.err" in self.names(), "Gain error not present in solutions db? This is a bug"
                casa_caltable_factory.create_B_table(self, "G:gain")
                casa_caltable_factory.create_D_table(self, "G:gain")
            if "G:delay" in self.names():
                assert "G:delay.err" in self.names(), "Delay error not present in solutions db? This is a bug"
                casa_caltable_factory.create_K_table(self, "G:delay")
                
        self.close()
            
    def close(self):
        """ see iface_database.close() for details """
        # move to closed state before exporting and loading back and sorting data
        do_export = (self.mode is "create")
        PickledDatabase.close(self) 
        if do_export and self.export_enabled:
            self.__export()
    
