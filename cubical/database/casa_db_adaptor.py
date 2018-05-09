# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

from cubical.database.pickled_db import PickledDatabase
from cubical.data_handler import DataHandler
from cubical.tools import logger
from pyrap.tables import table as tbl
import os
import shutil
import numpy as np

log = logger.getLogger("casa_db_adaptor")

# to the tune of KATDAL :)
BLANK_TABLE_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "blankcaltable.CASA")

class casa_caltable_factory(object):
        """
            Gaintable Factory
            
            Useful methods for creating and writing CASA-style gaintables
        """
        
        @classmethod
        def init_empty(cls, db, filename, solfreqs, solants, is_complex=True, viscal_label="B Jones"):
            """
                Initialize empty calibration table with metadata
                
                Args:
                    db: a cubical pickled_db instance
                    filename: filename of CASA gaintable to be written
                    solfreqs: centre frequencies of solutions to be stored in table
                    solants: Names from Cubical database antenna axis, used to order solutions
                    is_complex: Solutions are complex or real-valued
                    viscal_label: Sets viscal property of CASA table - used as identifier in CASA
            """
            if os.path.exists(filename):
                if os.path.isfile(filename):
                    log.error("CASA calibration table destination already exists but is not a directory. Will not remove.")    
                    return
                else:
                    log.info("Destination CASA gain table exists. Will overwrite.")
                    shutil.rmtree(filename) # CASA convention is to overwrite
                    
            shutil.copytree(BLANK_TABLE_TEMPLATE, filename)
            antorder = [db.antnames.index(an) for an in solants]
            with tbl("%s::ANTENNA" % filename, ack=False, readonly=False) as t:
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
            with tbl("%s::FIELD" % filename, ack=False, readonly=False) as t:
                t.addrows(nrows=1)
                t.putcol("DELAY_DIR", db.fielddelaydirs[selfield])
                t.putcol("PHASE_DIR", db.fieldphasedirs[selfield])
                t.putcol("REFERENCE_DIR", db.fieldrefdir[selfield])
                t.putcol("CODE", np.array(db.fieldcode)[selfield])
                t.putcol("FLAG_ROW", db.fieldflagrow[selfield])
                t.putcol("NAME", np.array(db.fieldname)[selfield])
                t.putcol("SOURCE_ID", db.fieldsrcid[selfield])
                t.putcol("TIME", db.fieldtime[selfield])
    
            with tbl("%s::OBSERVATION" % filename, ack=False, readonly=False) as t:
                t.addrows(nrows=len(db.obsobserver))
                t.putcol("TIME_RANGE", db.obstimerange)
                t.putcol("LOG", db.obslog)
                t.putcol("SCHEDULE", db.obsschedule)
                t.putcol("FLAG_ROW", db.obsflagrow)
                t.putcol("OBSERVER", db.obsobserver)
                t.putcol("PROJECT", db.obsproject)
                t.putcol("RELEASE_DATE", db.obsreleasedate)
                t.putcol("TELESCOPE_NAME", db.obstelescopename)
                
            assert (type(db.metadata["ddids"]) is list) \
                    and all([type(dd) is int for dd in db.metadata["ddids"]]), "Expect list of selected ddids"
            
            with tbl("%s::SPECTRAL_WINDOW" % filename, ack=False, readonly=False) as t:
                t.addrows(nrows=len(db.metadata["ddids"]))
                for iddid, ddid in enumerate(db.metadata["ddids"]):
                    spwid = db.ddid_spw_map[ddid]
                    minfreq = np.min(db.spwchanfreq[spwid] - 0.5 * db.spwchanwidth[spwid])
                    maxfreq = np.max(db.spwchanfreq[spwid] + 0.5 * db.spwchanwidth[spwid]) 
                    ddsolfreqs = solfreqs[np.logical_and(solfreqs >= minfreq,
                                                         solfreqs <= maxfreq)]
                    # assume linearly-spaced solutions
                    if ddsolfreqs.size > 1:
                        ddsolwidth = ddsolfreqs[1:] - ddsolfreqs[:-1]
                        assert np.all(ddsolwidth == ddsolwidth[0]), "Solutions should be equidistant, bug"
                    else:
                        ddsolwidth = db.spwtotalbandwidth
                        
                    ddsolwidth = np.ones(ddsolfreqs.shape) * ddsolwidth[0]
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
                    t.putcell("NAME", iddid, db.spwname[spwid])
                    t.putcell("NET_SIDEBAND", iddid, db.spwnetsideband[spwid])
                    t.putcell("NUM_CHAN", iddid, ddsolfreqs.size)
                    t.putcell("TOTAL_BANDWIDTH", iddid, maxfreq - minfreq)
                    
            with tbl(filename, ack=False, readonly=False) as t:
                t.putkeyword("ParType", "Complex" if is_complex else "Float")
                t.putkeyword("VisCal", viscal_label)
                
                if not is_complex:
                    cdesc = t.getcoldesc("CPARAM")
                    cdesc["valueType"] = "float"
                    t.addcols({"FPARAM", cdesc})
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
            if np.prod(db[gname].shape) == 0:
                log.warn("No %s solutions. Will not write CASA table" % gname)
                return
            assert db[gname].shape == db[gname + ".err"].shape, "PARAM err shape does not match PARAM shape, this is a bug"
            assert db[gname].axis_labels == ('dir', 'time', 'freq', 'ant', 'corr'), "DB table in unrecognized format"
            
            ddids = db.metadata["ddids"]
            ndir = len(db[gname].grid[db[gname].ax.dir])
            ntime = len(db[gname].grid[db[gname].ax.time])
            nant = len(db[gname].grid[db[gname].ax.ant])
            ncorr = len(db[gname].grid[db[gname].ax.corr])
            nddids = len(db.metadata["ddids"])
            nrow = ndir * ntime * \
                    nant * len(ddids)
            assert ncorr == 2, "Expected diagnonal Jones matrix"
            
            cls.init_empty(db, 
                           db.filename + ".%s.casa" % outname, 
                           db[gname].grid[db[gname].ax.freq],
                           db[gname].grid[db[gname].ax.ant],
                           viscal_label="G Jones")
            
            with tbl(db.filename + ".%s.casa" % outname, ack=False, readonly=False) as t:
                t.addrows(nrows=nrow)
                for iddid, ddid in enumerate(db.metadata["ddids"]):
                    spwid = db.ddid_spw_map[ddid]
                    minfreq = np.min(db.spwchanfreq[spwid] - 0.5 * db.spwchanwidth[spwid])
                    maxfreq = np.max(db.spwchanfreq[spwid] + 0.5 * db.spwchanwidth[spwid]) 
                    ddsolfreqindx = np.argwhere(np.logical_and(db[gname].grid[db[gname].ax.freq] >= minfreq,
                                                               db[gname].grid[db[gname].ax.freq] <= maxfreq))
                    params = np.swapaxes(db[gname].get_cube()[:, :, ddsolfreqindx, :, :],
                                         2, 3).reshape(ndir * ntime * nant, len(ddsolfreqindx), ncorr) 
                    paramerrs = np.swapaxes(db[gname + ".err"].get_cube()[:, :, ddsolfreqindx, :, :],
                                            2, 3).reshape(ndir * ntime * nant, len(ddsolfreqindx), ncorr) 
                    flags = np.ma.getmaskarray(params)
                    fieldid = np.zeros(ndir * ntime * nant) #temp need to specify directions properly
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
        def create_B_table(cls, db, gname, outname = "B", diag=True):
            """
                Write diagonal B-Jones caltable
                
                Args:
                    db: a cubical pickled_db instance
                    gname: name of pickled_db solutions to export
                    outname: suffix of exported CASA gaintable
                    diag: Write out diagonal of Jones matrix if true, off-diagonal (leakage) terms otherwise.
            """
            if np.prod(db[gname].shape) == 0:
                log.warn("No %s solutions. Will not write CASA table" % gname)
                return
            assert db[gname].shape == db[gname + ".err"].shape, "PARAM err shape does not match PARAM shape, this is a bug"
            assert db[gname].axis_labels == ('dir', 'time', 'freq', 'ant', 'corr1', 'corr2'), "DB table in unrecognized format"
            
            ddids = db.metadata["ddids"]
            ndir = len(db[gname].grid[db[gname].ax.dir])
            ntime = len(db[gname].grid[db[gname].ax.time])
            nant = len(db[gname].grid[db[gname].ax.ant])
            ncorr1 = len(db[gname].grid[db[gname].ax.corr1])
            ncorr2 = len(db[gname].grid[db[gname].ax.corr2])
            nddids = len(db.metadata["ddids"])
            nrow = ndir * ntime * \
                    nant * len(ddids)
            assert ncorr1 == ncorr2 and ncorr1 == 2, "Expected 2x2 solution, this is a bug"
            
            cls.init_empty(db, 
                           db.filename + ".%s.casa" % outname, 
                           db[gname].grid[db[gname].ax.freq],
                           db[gname].grid[db[gname].ax.ant],
                           viscal_label="B Jones" if diag else "D Jones")
            
            with tbl(db.filename + ".%s.casa" % outname, ack=False, readonly=False) as t:
                t.addrows(nrows=nrow)
                
                for iddid, ddid in enumerate(db.metadata["ddids"]):
                    spwid = db.ddid_spw_map[ddid]
                    minfreq = np.min(db.spwchanfreq[spwid] - 0.5 * db.spwchanwidth[spwid])
                    maxfreq = np.max(db.spwchanfreq[spwid] + 0.5 * db.spwchanwidth[spwid]) 
                    ddsolfreqindx = np.argwhere(np.logical_and(db[gname].grid[db[gname].ax.freq] >= minfreq,
                                                               db[gname].grid[db[gname].ax.freq] <= maxfreq))
                    jones_entries = [0, 3] if diag else [1, 2] # diagonal or crosshands
                    params = np.swapaxes(db[gname].get_cube()[:, :, ddsolfreqindx, :, :], 
                                         2, 3).reshape(ndir * ntime * nant, len(ddsolfreqindx), ncorr1 * ncorr2)[:, :, jones_entries]
                    paramerrs = np.swapaxes(db[gname + ".err"].get_cube()[:, :, ddsolfreqindx, :, :],
                                            2, 3).reshape(ndir * ntime * nant, len(ddsolfreqindx), ncorr1 * ncorr2)[:, :, jones_entries]
                    flags = np.ma.getmaskarray(params)
                    fieldid = np.zeros(ndir * ntime * nant) #temp need to specify directions properly
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
            cls.create_B_table(db, gname, outname, diag=False)

class casa_db_adaptor(PickledDatabase):
    
    def __init__(self):
        """
            CASA-style gaintable adaptor for Cubical pickled_db database writer
            
            This adaptor adds functionality to export databases to CASA gaintables
            which can be read and plotted in CASA plotms or similar diagnostic utilities.
        """
        PickledDatabase.__init__(self)
        self.meta_avail = False
        self.export_enabled = True
    
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
        if not isinstance(src, DataHandler):
            raise TypeError("src must be of type Cubical DataHandler")
        
        self.antoffset = src.antoffset
        self.antpos = src.antpos
        self.anttype = src.anttype
        self.antdishdiam = src.antdishdiam
        self.antflagrow = src.antflagrow
        self.antmount = src.antmount
        self.antnames = src.antnames
        self.antstation = src.antstation
        self.fielddelaydirs = src.fielddelaydirs
        self.fieldphasedirs = src.fieldphasedirs
        self.fieldrefdir = src.fieldrefdir
        self.fieldcode = src.fieldcode
        self.fieldflagrow = src.fieldflagrow
        self.fieldname = src.fieldname
        self.fieldnumpoly = src.fieldnumpoly
        self.fieldsrcid = src.fieldsrcid
        self.fieldtime = src.fieldtime
        self.obstimerange = src.obstimerange
        self.obslog = src.obslog
        self.obsschedule = src.obsschedule
        self.obsflagrow = src.obsflagrow
        self.obsobserver = src.obsobserver
        self.obsproject = src.obsproject
        self.obsreleasedate = src.obsreleasedate
        self.obsscheduletype = src.obsscheduletype
        self.obstelescopename = src.obstelescopename
        self.spwmeasfreq = src.spwmeasfreq
        self.spwchanfreq = src.spwchanfreq
        self.spwreffreq = src.spwreffreq
        self.spwchanwidth = src.spwchanwidth
        self.spweffbw = src.spweffbw
        self.spwresolution = src.spwresolution
        self.spwflagrow = src.spwflagrow
        self.spwfreqgroup = src.spwfreqgroup
        self.spwfreqgroupname = src.spwfreqgroupname
        self.spwifconvchain = src.spwifconvchain
        self.spwname = src.spwname
        self.spwnetsideband = src.spwnetsideband
        self.spwnumchan = src.spwnumchan
        self.spwtotalbandwidth = src.spwtotalbandwidth
        self.ddid_spw_map = src._ddid_spw
        self.do_write_casatbl = True
        self.meta_avail = True
    
    def __export(self):
        """ exports the database to CASA gaintables """
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
                pass
        self.close()
            
    def close(self):
        """ see iface_database.close() for details """
        
        # move to closed state before exporting and loading back and sorting data
        do_export = (self.mode is "create")
        PickledDatabase.close(self) 
        if do_export:
            self.__export()
    
