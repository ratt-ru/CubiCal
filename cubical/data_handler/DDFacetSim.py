# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

from __future__ import print_function
try:
    from DDFacet.Imager import ClassDDEGridMachine
    from DDFacet.cbuild.Gridder import _pyGridderSmearPolsClassic
    from DDFacet.ToolsDir.ModToolBox import EstimateNpix
    from DDFacet.Array import shared_dict
    from DDFacet.ToolsDir import ModFFTW
    import DDFacet.ToolsDir.ModTaper as MT
except ImportError:
    raise ImportError("Could not import DDFacet")

from . import DicoSourceProvider
from cubical.tools import logger, ModColor
log = logger.getLogger("DDFacetSim")
import numpy as np
import math
import os
from concurrent.futures import (ProcessPoolExecutor as PPE,
                                ThreadPoolExecutor as TPE)
import hashlib

def init_degridders(dir_CFs, 
                    dir_DCs,
                    CFs_dict, 
                    freqs, 
                    subregion_index, 
                    lmShift, 
                    dname, 
                    nfreqbands, 
                    DataCorrelationFormat,
                    sems,
                    npix,
                    should_init_cf,
                    wmax,
                    GD,
                    gmachines=None):
    """ Dummy initialization method to be executed on process pool """
    dir_CFs = dir_CFs.instantiate()
    dir_DCs = dir_DCs.instantiate()
    CFs_dict = CFs_dict.instantiate()
    gmach = ClassDDEGridMachine.ClassDDEGridMachine(GD,
                ChanFreq = freqs,
                Npix = npix,
                lmShift = lmShift,
                IDFacet = dir_CFs[dname][subregion_index],
                SpheNorm = False, # Depricated, set ImToGrid True in .get!!
                NFreqBands = nfreqbands,
                DataCorrelationFormat=DataCorrelationFormat,
                ExpectedOutputStokes=[1], # Stokes I
                ListSemaphores=sems,
                cf_dict=CFs_dict,
                compute_cf=should_init_cf,
                wmax=wmax)
    (gmachines is not None) and gmachines.append(gmach)
    if should_init_cf:
        wnd_detaper = MT.Sphe2D(npix)
        wnd_detaper[wnd_detaper != 0] = 1.0 / wnd_detaper[wnd_detaper != 0]
        dir_DCs["facet_{}".format(dir_CFs[dname][subregion_index])] = wnd_detaper

global UNIQ_ID
try:
    UNIQ_ID
except NameError:
    UNIQ_ID = os.getpid()

class DDFacetSim(object):
    __degridding_semaphores = None
    __initted_CF_directions = []
    __direction_CFs = shared_dict.SharedDict("facetids.cubidegrid.{}".format(UNIQ_ID))
    __ifacet = 0
    __CF_dict = {}
    __should_init_sems = True
    __detaper_cache = shared_dict.SharedDict("tapers.cubidegrid.{}".format(UNIQ_ID))
    __exec_pool = None
    __IN_PARALLEL_INIT = False
    def __init__(self, num_processes=0):
        """ 
            Initializes a DDFacet model predictor
        """
        self.__direction = None
        self.__model = None
        self.init_sems()
    
    @classmethod
    def initialize_pool(cls, num_processes=0):
        if num_processes > 1:
            DDFacetSim.__exec_pool = PPE(max_workers=num_processes)
            DDFacetSim.__IN_PARALLEL_INIT = True

    @classmethod
    def shutdown_pool(cls):
        if DDFacetSim.__IN_PARALLEL_INIT:
            DDFacetSim.__exec_pool.shutdown()

    def set_direction(self, val):
        """ sets the direction in the cubical model cube to pack model data into """
        self.__direction = val
    
    def set_model_provider(self, model):
        """ sets the model provider """
        if not isinstance(model, DicoSourceProvider.DicoSourceProvider):
            raise TypeError("Model provider must be a DicoSourceProvider")
        log(2).print("Model predictor is switching to model '{0:s}'".format(str(model)))
        self.__model = model

    @classmethod
    def dealloc_degridders(cls):
        """ resets CF allocation state of all degridders """
        cls.__initted_CF_directions = []
        cls.__direction_CFs.delete() # cleans out /dev/shm allocated memory
        cls.__ifacet = 0
        for cf in cls.__CF_dict:
            cls.__CF_dict[cf].delete() # cleans out /dev/shm allocated memory
        cls.__CF_dict = {}
        cls.__should_init_sems = True
        cls.__detaper_cache.delete() # cleans out /dev/shm allocated memory

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

    def __cachename_compute(self, src):
        reg_props = str(src.subregion_count)
        for subregion_index in range(src.subregion_count):
            reg_props += " npx:" + str(src.get_direction_npix(subregion_index=subregion_index)) + \
                    " scale:" + "x".join(map(str, list(np.deg2rad(src.get_direction_pxoffset(subregion_index=subregion_index)) *
                                                       src.pixel_scale / 3600.0)))
        res = "dir_{0:s}_{1:s}_{2:s}".format(str(self.__model), str(self.__direction), str(reg_props))
        return hashlib.md5(res).hexdigest()

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

        should_init_cf = self.__cachename_compute(src) not in self.__initted_CF_directions

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
        gmachines = []
        dname = self.__cachename_compute(src)
        if should_init_cf:
            log.info("This is the first time predicting for '{0:s}' direction '{1:s}'. "
                     "Initializing degridder for {2:d} facets - this may take a wee bit of time.".format(
                         str(self.__model), str(self.__direction), src.subregion_count))
            DDFacetSim.__initted_CF_directions.append(dname)    
            DDFacetSim.__direction_CFs[dname] = DDFacetSim.__direction_CFs.get(dname, []) + list(DDFacetSim.__ifacet + np.arange(src.subregion_count)) #unique facet index for this subregion
            DDFacetSim.__ifacet += src.subregion_count
            for sri in range(src.subregion_count):
                DDFacetSim.__CF_dict["{}.{}".format(dname, sri)] = \
                    shared_dict.SharedDict("convfilters.cubidegrid.{}.{}.{}".format(UNIQ_ID, dname, sri))

        # init in parallel
        futures = []
        for subregion_index in range(src.subregion_count):
            init_args = (DDFacetSim.__direction_CFs.readwrite(), 
                        DDFacetSim.__detaper_cache.readwrite(),
                        DDFacetSim.__CF_dict["{}.{}".format(dname, subregion_index)].readwrite(),
                        freqs,
                        subregion_index,
                        np.deg2rad(src.get_direction_pxoffset(subregion_index=subregion_index) * 
                                    src.pixel_scale / 3600.0),
                        dname,
                        dh.degrid_opts["NDegridBand"],
                        DataCorrelationFormat,
                        self.__degridding_semaphores,
                        src.get_direction_npix(subregion_index=subregion_index),
                        should_init_cf,
                        wmax,
                        GD)
            if DDFacetSim.__IN_PARALLEL_INIT:
                futures.append(DDFacetSim.__exec_pool.submit(init_degridders, *init_args))
            else:
                init_degridders(*init_args)

        if DDFacetSim.__IN_PARALLEL_INIT:
            for f in futures:
                expt = f.exception()
                if expt is not None:
                    raise expt
        
        # construct handles after the initialization has been completed
        for subregion_index in range(src.subregion_count):
            init_degridders(DDFacetSim.__direction_CFs.readwrite(), 
                            DDFacetSim.__detaper_cache.readwrite(),
                            DDFacetSim.__CF_dict["{}.{}".format(dname, subregion_index)].readwrite(),
                            freqs,
                            subregion_index,
                            np.deg2rad(src.get_direction_pxoffset(subregion_index=subregion_index) * 
                                       src.pixel_scale / 3600.0),
                            dname,
                            dh.degrid_opts["NDegridBand"],
                            DataCorrelationFormat,
                            self.__degridding_semaphores,
                            src.get_direction_npix(subregion_index=subregion_index),
                            False, #already initialized
                            wmax,
                            GD,
                            gmachines
                            )
        return gmachines

    @classmethod
    def __detaper_model(cls, gm, model_image, idfacet):
        """ Detapers model image by the fourier inverse of the convolution kernel 
            Assertions that the tapering function is larger than the model facet,
            both are square and odd sized
        """
        assert model_image.shape[2] == model_image.shape[3]
        wnd_detaper = cls.__detaper_cache.readonly().instantiate()["facet_{}".format(idfacet)]
        assert wnd_detaper.shape[0] == model_image.shape[2] and wnd_detaper.shape[1] == model_image.shape[3]
        model_image.real *= wnd_detaper[None, None, :, :]
        return model_image

    def simulate(self, dh, tile, tile_subset, poltype, uvwco, freqs, model_type):
        """ Predicts model data for the set direction of the dico source provider 
            returns a ndarray model of shape nrow x nchan x 4
        """
        if self.__direction is None:
            raise RuntimeError("Direction has not been set. Please set direction before simulating")
        if self.__model is None:
            raise RuntimeError("Model has not been set. Please set model before simulating")
        DDFacetSim.initialize_pool(dh.degrid_opts["NProcess"])
        freqs = freqs.ravel()
        src = self.__model
        src.set_direction(self.__direction)
        band_frequencies, freq_mapping = self.bandmapping(freqs, dh.degrid_opts["NDegridBand"])
        gmacs = self.__init_grid_machine(src, dh, tile, poltype, freqs)
        nrow = uvwco.shape[0]
        nfreq = len(freqs)
        if model_type not in ["cplx2x2", "cplxdiag", "cplxscalar"]:
            raise ValueError("Only supports cplx2x2 or cplxdiag or cplxscalar models at the moment")
        ncorr = 4
        region_model = np.zeros((nrow, nfreq, 4), dtype=np.complex64)
        flagged = np.zeros_like(region_model, dtype=np.bool)
        # now we predict for this direction
        log.info("Computing visibilities in {1:d} facets for direction '{0:s}' for model '{2:s}'...".format(
            str(self.__direction), src.subregion_count, str(self.__model)))

        for gm, subregion_index in zip(gmacs, range(src.subregion_count)):
            model = np.zeros((nrow, nfreq, 4), dtype=np.complex64).copy()
            model_image = src.get_degrid_model(subregion_index=subregion_index).astype(dtype=np.complex64).copy() #degridder needs transposes, dont mod the data globally

            if not np.any(model_image):
                log(2).print("Facet {0:d} is empty. Skipping".format(subregion_index))
                continue
            dname = self.__cachename_compute(src)
            model_image = DDFacetSim.__detaper_model(gm, model_image.view(), self.__direction_CFs[dname][subregion_index]).copy() # degridder don't respect strides must be contiguous

            # apply normalization factors for FFT
            model_image[...] *= (model_image.shape[3] ** 2) * (gm.WTerm.OverS ** 2) 

            region_model += -1 * gm.get( #default of the degridder is to subtract from the previous model
                times=tile_subset.time_col, 
                uvw=uvwco.astype(dtype=np.float64).copy(), 
                visIn=model, 
                flag=flagged, 
                A0A1=[tile_subset.antea.astype(dtype=np.int32).copy(), tile_subset.anteb.astype(dtype=np.int32).copy()], 
                ModelImage=model_image, 
                PointingID=src.direction,
                Row0Row1=(0, -1),
                DicoJonesMatrices=None, 
                freqs=freqs.astype(dtype=np.float64).copy(), 
                ImToGrid=False,
                TranformModelInput="FT", 
                ChanMapping=np.array(freq_mapping, dtype=np.int32), 
                sparsification=None)

        model_corr_slice = np.s_[0] if model_type == "cplxscalar" else \
                           np.s_[0::3] if model_type == "cplxdiag" else \
                           np.s_[:] # if model_type == "cplx2x2"
        if not np.any(region_model[:, :, model_corr_slice]):
            log.critical("Model in region '{0:s}' is completely empty. This may indicate user "
                         "error and lead to non-correcting direction dependent gains!".format(self.__direction))

        return region_model[:, :, model_corr_slice]

import atexit

def _cleanup_degridders():
    DDFacetSim.del_sems()
    DDFacetSim.dealloc_degridders()
    DDFacetSim.shutdown_pool()
        
atexit.register(_cleanup_degridders)
