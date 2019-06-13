try:
    from DDFacet.Imager import ClassDDEGridMachine
    from DDFacet.cbuild.Gridder import _pyGridderSmearPolsClassic
    from DDFacet.ToolsDir.ModToolBox import EstimateNpix

except ImportError:
    raise ImportError("Could not import DDFacet")

import numpy as np

class DDFacetSim(object):
    __degridding_semaphores = None
    __initted_CF_directions = []
    __CF_dict = {}
    
    def __init__(self):
        self.__direction = 0

    def set_direction(self, val):
        """ sets the direction in the cubical model cube to pack model data into """
        self.__direction = val
    
    @classmethod
    def init_sems(cls, NSemaphores = 3373):
        """ Init semaphores """
        cls.__degridding_semaphores = ["Semaphore.cubidegrid{0:d}".format(i) for i in
                                        range(NSemaphores)]
        _pyGridderSmearPolsClassic.pySetSemaphores(cls.__degridding_semaphores)
    
    @classmethod
    def del_sems(cls):
        """ Deinit semaphores """
        _pyGridderSmearPolsClassic.pyDeleteSemaphore()
        cls.__degridding_semaphores = None

    def __init_grid_machine(self, src, dh, tile, poltype):
        """ initializes a grid machine for this direction """
        if self.__degridding_semaphores is None:
            raise RuntimeError("Need to initialize degridding semaphores first. Call init_sems")
        if poltype == "linear":
            DataCorrelationFormat = [9, 10, 11, 12] # Defined in casacore Stokes.h
        elif poltype == "circular":
            DataCorrelationFormat = [5, 6, 7, 8]
        else:
            raise ValueError("Only supports linear or circular for now")

        should_init_cf = self.__direction not in self.__initted_CF_directions
        
        # Kludge up a DDF GD
        GD = {}
        GD["RIME"]["Precision"] = "S"
        GD["Facets"]["Padding"] = dh.degrid_opts["Padding"]
        GD["CF"]["OverS"] = dh.degrid_opts["OverS"]
        GD["CF"]["Support"] = dh.degrid_opts["Support"]
        GD["CF"]["Nw"] = dh.degrid_opts["Nw"]
        GD["Image"]["Cell"] = dh.degrid_opts["Cell"]
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
        data_chan_freqs = tile._freqs
        band_frequencies = np.linspace(np.min(data_chan_freqs), np.max(data_chan_freqs), dh.degrid_opts["NFreqBands"])
        src.set_frequency(band_frequencies)
        wmax = np.max(dh.metadata.baseline_length)

        if should_init_cf:
            self.__initted_CF_directions.append(self.__direction)
            cf_dict = self.__CF_dict
        else:
            cf_dict = None

        gmach = ClassDDEGridMachine(GD,
                                    ChanFreq = data_chan_freqs,
                                    Npix = self.__direction,
                                    lmshift = np.deg2rad(src.get_direction_pxoffset * src.pixel_scale / 3600),
                                    IDFacet = src.direction,
                                    SpheNorm = True, # Depricated, set ImToGrid True in .get!!
                                    NFreqBands = dh.degrid_opts["NFreqBands"],
                                    DataCorrelationFormat=DataCorrelationFormat,
                                    ExpectedOutputStokes=[1], # Stokes I
                                    ListSemaphores=self.__degridding_semaphores,
                                    cf_dict=cf_dict, compute_cf=should_init_cf,
                                    wmax=wmax,
                                    bda_grid=None, bda_degrid=None)

    def simulate(self, src, dh, tile, poltype):
        """ Predicts model data for the set direction of the dico source provider """
        src.pad_clusters(dh.degrid_opts["Padding"])
        gm = self.__init_grid_machine(src, dh, tile, poltype)
        
        