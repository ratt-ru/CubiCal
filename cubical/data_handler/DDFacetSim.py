try:
    from DDFacet.Imager import ClassDDEGridMachine
    from DDFacet.cbuild.Gridder import _pyGridderSmearPolsClassic
    from DDFacet.ToolsDir.ModToolBox import EstimateNpix
    from DDFacet.Array import shared_dict
    from DDFacet.ToolsDir import ModFFTW
except ImportError:
    raise ImportError("Could not import DDFacet")

from . import DicoSourceProvider
from cubical.tools import logger, ModColor
log = logger.getLogger("DDFacetSim")
import numpy as np
import math

class DDFacetSim(object):
    __degridding_semaphores = None
    __initted_CF_directions = []
    __CF_dict = shared_dict.SharedDict("convfilters.cubidegrid")
    __should_init_sems = True

    def __init__(self):
        """ 
            Initializes a DDFacet model predictor
        """
        self.__direction = None
        self.__model = None
        self.init_sems()

    def set_direction(self, val):
        """ sets the direction in the cubical model cube to pack model data into """
        self.__direction = val
    
    def set_model_provider(self, model):
        """ sets the model provider """
        if not isinstance(model, DicoSourceProvider.DicoSourceProvider):
            raise TypeError("Model provider must be a DicoSourceProvider")
        log.info("Model predictor is switching to model '{0:s}'".format(str(model)))
        self.__model = model

    @classmethod
    def init_sems(cls, NSemaphores = 3373):
        """ Init semaphores """
        if not cls.__should_init_sems:
            return
        cls.__should_init_sems = False
        cls.__degridding_semaphores = ["Semaphore.cubidegrid{0:d}".format(i) for i in
                                        range(NSemaphores)]
        _pyGridderSmearPolsClassic.pySetSemaphores(cls.__degridding_semaphores)
    
    @classmethod
    def del_sems(cls):
        """ Deinit semaphores """
        _pyGridderSmearPolsClassic.pyDeleteSemaphore()
        cls.__degridding_semaphores = None

    def bandmapping(self, vis_freqs, nbands):
        """ 
            Gives the frequency mapping for visibility to degrid band 
            For now we assume linear regular spacing, so we may end up
            with a completely empty band somewhere in the middle
            if we have spws that are separated in frequency
        """
        band_frequencies = np.linspace(np.min(vis_freqs), np.max(vis_freqs), nbands)
        def find_nearest(array, value):
            idx = np.searchsorted(array, value, side="left")
            if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
                return idx - 1
            else:
                return idx
        freq_mapping = [find_nearest(band_frequencies, v) for v in vis_freqs]
        return band_frequencies, freq_mapping

    def __init_grid_machine(self, src, dh, tile, poltype, freqs):
        """ initializes a grid machine for this direction """
        if self.__degridding_semaphores is None:
            raise RuntimeError("Need to initialize degridding semaphores first. Call init_sems")
        if poltype == "linear":
            DataCorrelationFormat = [9, 10, 11, 12] # Defined in casacore Stokes.h
        elif poltype == "circular":
            DataCorrelationFormat = [5, 6, 7, 8]
        else:
            raise ValueError("Only supports linear or circular for now")

        should_init_cf = "dir_{0:s}_{1:s}".format(str(self.__model), str(self.__direction)) not in self.__initted_CF_directions
        
        # Kludge up a DDF GD
        GD = dict(RIME={}, Facets={}, CF={}, Image={}, DDESolutions={}, Comp={})
        GD["RIME"]["Precision"] = "S"
        GD["Facets"]["Padding"] = dh.degrid_opts["Padding"]
        GD["CF"]["OverS"] = dh.degrid_opts["OverS"]
        GD["CF"]["Support"] = dh.degrid_opts["Support"]
        GD["CF"]["Nw"] = dh.degrid_opts["Nw"]
        GD["Image"]["Cell"] = src.pixel_scale
        GD["DDESolutions"]["JonesMode"] = "Full" 	  # #options:Scalar|Diag|Full
        GD["DDESolutions"]["Type"] = "Nearest" # Deprecated? #options:Krigging|Nearest
        GD["DDESolutions"]["Scale"] = 1.      # Deprecated? #metavar:DEG
        GD["DDESolutions"]["gamma"] = 4.	  # Deprecated? 
        GD["RIME"]["FFTMachine"] = "FFTW" 
        GD["Comp"]["BDAJones"] = 0         # If disabled, gridders and degridders will apply a Jones terms per visibility. 
                                           # If 'grid', gridder will apply them per BDA block, if 'both' so will the degridder. This is faster but possibly less 
                                           # accurate, if you have rapidly evolving Jones terms.
        GD["DDESolutions"]["DDModeGrid"] = "AP"	  # In the gridding step, apply Jones matrices Amplitude (A) or Phase (P) or Amplitude&Phase (AP)
        GD["DDESolutions"]["DDModeDeGrid"] = "AP"	  # In the degridding step, apply Jones matrices Amplitude (A) or Phase (P) or Amplitude&Phase (AP)
        GD["DDESolutions"]["ScaleAmpGrid"] = 0 # Deprecated?
        GD["DDESolutions"]["ScaleAmpDeGrid"] = 0	  # Deprecated?
        GD["DDESolutions"]["CalibErr"] = 10.	  # Deprecated?
        GD["DDESolutions"]["ReWeightSNR"] = 0.	  # Deprecated? 
        GD["RIME"]["BackwardMode"] = "BDA"
        GD["RIME"]["ForwardMode"] = "Classic" # Only classic for now... why would you smear unnecessarily in a model predict??
        
        # INIT degridder machine from DDF
        band_frequencies, freq_mapping = self.bandmapping(freqs, dh.degrid_opts["NDegridBand"])
        src.set_frequency(band_frequencies)
        wmax = np.max([dh.metadata.baseline_length[k] for k in dh.metadata.baseline_length])

        if should_init_cf:
            log.info("This is the first time predicting for '{0:s}' direction '{1:s}'. "
                     "Initializing degridder - this may take a wee bit of time.".format(str(self.__model), str(self.__direction)))
            self.__initted_CF_directions.append("dir_{0:s}_{1:s}".format(str(self.__model), str(self.__direction)))
            cf_dict = self.__CF_dict
        else:
            cf_dict = None
        gmach = ClassDDEGridMachine.ClassDDEGridMachine(GD,
                                    ChanFreq = freqs,
                                    Npix = src.get_direction_npix(),
                                    lmShift = np.deg2rad(src.get_direction_pxoffset() * src.pixel_scale / 3600),
                                    IDFacet = src.direction,
                                    SpheNorm = True, # Depricated, set ImToGrid True in .get!!
                                    NFreqBands = dh.degrid_opts["NDegridBand"],
                                    DataCorrelationFormat=DataCorrelationFormat,
                                    ExpectedOutputStokes=[1], # Stokes I
                                    ListSemaphores=self.__degridding_semaphores,
                                    cf_dict=cf_dict, compute_cf=should_init_cf,
                                    wmax=wmax,
                                    bda_grid=None, bda_degrid=None)
        #gmach.FT = ModFFTW.FFTW_2Donly_np(src.degrid_cube_shape, np.complex64, ncores = 1).fft
        return gmach

    def simulate(self, dh, tile, tile_subset, poltype, uvwco, freqs):
        """ Predicts model data for the set direction of the dico source provider 
            returns a ndarray model of shape nrow x nchan x 4
        """
        if self.__direction is None:
            raise RuntimeError("Direction has not been set. Please set direction before simulating")
        if self.__model is None:
            raise RuntimeError("Model has not been set. Please set model before simulating")
        freqs = freqs.ravel()
        src = self.__model
        src.set_direction(self.__direction)
        band_frequencies, freq_mapping = self.bandmapping(freqs, dh.degrid_opts["NDegridBand"])
        gm = self.__init_grid_machine(src, dh, tile, poltype, freqs)
        nrow = uvwco.shape[0]
        nfreq = len(freqs)
        ncorr = 4 # assume cubical expects 2x2 models
        model = np.zeros((nrow, nfreq, 4), np.complex64)
        flagged = np.zeros_like(model, dtype=np.bool)
        # now we predict for this direction
        log.info("Predicting direction '{0:s}'...".format(str(self.__direction)))
        model = gm.get(
            times=tile_subset.time_col, 
            uvw=uvwco, 
            visIn=model, 
            flag=flagged, 
            A0A1=[tile_subset.antea, tile_subset.anteb], 
            ModelImage=src.get_degrid_model(), 
            PointingID=src.direction,
            Row0Row1=(0, -1),
            DicoJonesMatrices=None, 
            freqs=freqs, 
            ImToGrid=False,
            TranformModelInput="FT", 
            ChanMapping=freq_mapping, 
            sparsification=None)
        
        return model

import atexit

def _cleanup_degridder_semaphores():
    DDFacetSim.del_sems()

atexit.register(_cleanup_degridder_semaphores)
