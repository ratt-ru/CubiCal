from __future__ import print_function
try:
    from DDFacet.Imager import ModModelMachine as MMM
    from DDFacet.ToolsDir.ModToolBox import EstimateNpix
except ImportError:
    raise ImportError("Cannot import DDFacet")

from regions import DS9Parser
import numpy as np
from .geometry import BoundingConvexHull, BoundingBox, BoundingBoxFactory
from cubical.tools import logger, ModColor
log = logger.getLogger("DDFacetSim")

class DicoSourceProvider(object):
    __cached_model_machines = {}
    __cached_providers = {}

    @classmethod
    def __cachename_compute(cls, diconame, regname):
        return "src_model_{0:s}_@_{1:s}".format(diconame,
                                                regname if regname is not None else "die")

    def __init__(self,
                 fn,
                 phasedir,
                 facet_padding_factor,
                 max_facet_size,
                 min_nfacet_per_axis,
                 clustercat=None):
        self.__dicofn = fn

        if fn not in DicoSourceProvider.__cached_model_machines:
            self.__dicomodel = MMM.ClassModModelMachine().GiveInitialisedMMFromFile(fn)
            DicoSourceProvider.__cached_model_machines[fn] = self.__dicomodel
        else:
            log(2).print("Reusing previously initialized DDFacet model machine '{0:s}'".format(fn))
            self.__dicomodel = DicoSourceProvider.__cached_model_machines[fn]

        # assume gd is stored in the dico... this is only true for ddf versions 0.4.1 and beyond
        self.__pxscale = self.__dicomodel.GD["Image"]["Cell"] # in arcseconds

        self.__nchan, self.__npol, self.__ny, self.__nx = self.__dicomodel.ModelShape
        assert self.__ny == self.__nx # check cyril's model is square
        assert self.__nx % 2 == 1 # must be odd for the logic downstairs to work

        # regions relative to the unpadded images
        req_npx_img = self.__dicomodel.GD["Image"]["NPix"]
        nx_image, nx_image_padded = EstimateNpix(req_npx_img,
                                                 Padding=self.__dicomodel.GD["Facets"]["Padding"])
        assert nx_image == self.__nx # the predicted image size better correspond to dicomodels shape
        self.__padded_nx = self.__padded_ny = nx_image_padded
        self.__phasedir = phasedir

        cachename = DicoSourceProvider.__cachename_compute(fn, clustercat)
        if cachename not in DicoSourceProvider.__cached_providers:
            log.info("Initializing new source provider for DDFacet model '{0:s}' into regions specified by '{1:s}'.".format(
                fn, clustercat if clustercat is not None else 'die'))
            self.__clustercat = self.__read_regions_file(clustercat, 
                                                         facet_padding_factor, 
                                                         max_facet_size,
                                                         min_nfacet_per_axis)
            DicoSourceProvider.__cached_providers[cachename] = self.__clustercat
            log(2).print("initialization sequence of source provider '{0:s}' (regions '{1:s}') completed".format(
                    fn, clustercat if clustercat is not None else 'die'))
        else:
            self.__clustercat = DicoSourceProvider.__cached_providers[cachename]
            log(2).print("reused previous initialization of source provider '{0:s}' (regions '{1:s}')".format(
                fn, clustercat if clustercat is not None else 'die'))

        self.__current_direction = 0
        self.__degridcube = None
        self.__degridfreqs = None
        self.__padding_factor = facet_padding_factor

    def __str__(self):
        return self.__dicofn

    @property
    def subregion_count(self):
        return len(self.__clustercat[self.__current_direction])

    @property
    def pixel_scale(self):
        return self.__pxscale

    @property
    def degrid_cube_shape(self):
        if self.__degridcube is None:
            raise RuntimeError("Degrid frequencies not set yet. Please set frequencies before enquiring about the degrid cube")
        return self.__degridcube.shape

    def __read_regions_file(self, fn, padding_factor, max_size, min_nfacet_per_axis):
        """ Reads a ds9 region file and sets up a grid of sub clusters (aka. facets) for each hull read out of
            the file. Each of the sub clusters are padded to a padding factor as determined by
            DDFacet's EstimateNpix. The number of sub clusters is determined from the maximum
            size of such sub cluster.

            If the fn is None it is assumed that the user intends to split the full sky up into a single
            direction with each sub cluster contributing to this single direction (i.e. regular
            tesselation of the sky into square facets).

            Subclusters are guaranteed to be odd sized and square

            return dictionary of directions, each entry containing a list of sub regions
        """
        clusters = []
        if fn is not None: # dde case
            with open(fn) as f:
                parser = DS9Parser(f.read())
                for regi, reg in enumerate(parser.shapes):
                    coords = list(map(int, [c.value for c in reg.coord]))
                    assert len(coords) % 2 == 0, "Number of region coords must be multiple of 2-tuple"
                    coords = np.array(coords).reshape([len(coords) // 2, 2])
                    clusters.append(BoundingConvexHull(coords,
                                                       name="DDE_REG{0:d}".format(regi + 1)))
        else: # die case
            clusters = [BoundingBox(0,
                                    self.__nx - 1,
                                    0,
                                    self.__ny - 1,
                                    name="die",
                                    check_mask_outofbounds=True)]
        log(2).print("\tInitialized bounding boxes for regions. There are {0:d} region(s)".format(len(clusters)))
        # now create axis aligned bounding boxes for each of the regions
        # and further split them to the maximum permitted facet size
        clusters = [BoundingBoxFactory.AxisAlignedBoundingBox(c, check_mask_outofbounds=False) for c in clusters]
        def __split_regular_region(reg, max_size):
            if max_size < 0:
                raise ValueError("Expected positive value for min_size")
            reg_size_deg = np.max(np.array(reg.box_npx) * self.pixel_scale / 3600.0)
            nsplit = max(1, max(min_nfacet_per_axis, int(np.ceil(reg_size_deg / max_size))))
            return BoundingBoxFactory.SplitBox(reg, nsubboxes=nsplit, check_mask_outofbounds=True)
        log(2).print("\tSplitting regions into facetted regions, with maximum unpadded size of {0:.2f} degrees per facet".format(max_size))
        clusters = [aasubreg for aareg in map(lambda reg: __split_regular_region(reg, max_size), clusters) 
                    for aasubreg in aareg]
        clusters = list(map(lambda reg: BoundingBoxFactory.AxisAlignedBoundingBox(reg, square=True, check_mask_outofbounds=False), clusters))
        
        def __pad_cluster(c, padding_factor):
            npx,_ = c.box_npx # square facet at this point
            # this returns an odd npix:
            npixunpadded, npixpadded = EstimateNpix(npx, Padding=padding_factor)
            return BoundingBoxFactory.PadBox(c, npixpadded, npixpadded, check_mask_outofbounds=False)
        log(2).print("\tPadding all facets by a minimum factor of {0:.2f}x".format(padding_factor))
        clusters = list(map(lambda c: __pad_cluster(c, padding_factor), clusters))
        log.debug("\tNormalizing regional weights")
        BoundingConvexHull.normalize_masks(clusters)
        log(2).print("\tCaching regional weight maps for future predicts")
        list(map(lambda x: x.mask, clusters)) # cache mask
        dirs = {} 
        for c in clusters:
            dirs[c.name] = dirs.get(c.name, []) + [c]
        return dirs

    @property
    def _cluster_keys(self):
        """ returns list of cluster name keys if there are any clusters
            otherwise ["die"] if no dE clusters were specified upon construction
        """
        return ["die"] if self.__clustercat is None else \
            self.__clustercat.keys()
    
    def set_direction(self, v):
        """ sets the cluster (direction) index to predict from """
        if v not in self._cluster_keys:
            raise KeyError("Unknown direction '{0:s}' for model '{1:s}'".format(str(v), str(self)))
        self.__current_direction = v
    
    @property
    def _nclus(self):
        return len(self.__clustercat.keys())

    @property
    def direction(self):
        return self.__current_direction

    @direction.setter
    def direction(self, v):
        self.set_direction(v)

    @classmethod
    def crosspattern(cls, model_image, nterms = 7, first_position_from_centre=0.05, last_position_from_edge=0.8):
        if model_image.shape[2] != model_image.shape[3]:
            raise ValueError("Expected square grid")
        model_image[...] = 0
        for v in np.linspace(model_image.shape[2]//2 * first_position_from_centre, 
                             model_image.shape[2]//2 * last_position_from_edge, nterms):
            model_image[:, :, model_image.shape[2]//2 - int(v), model_image.shape[3]//2 - int(v)] = 1.0
            model_image[:, :, model_image.shape[2]//2 + int(v), model_image.shape[3]//2 + int(v)] = 1.0
            model_image[:, :, model_image.shape[2]//2 + int(v), model_image.shape[3]//2 - int(v)] = 1.0
            model_image[:, :, model_image.shape[2]//2 - int(v), model_image.shape[3]//2 + int(v)] = 1.0
        return model_image


    def set_frequency(self, freqs):
        """ Sets the model prediction frequencies. This initializes the model degridding bands """
        self.__degridfreqs = freqs.copy()
        self.__degridcube = self.__dicomodel.GiveModelImage(FreqIn=self.__degridfreqs)
        #self.__degridcube = DicoSourceProvider.crosspattern(self.__degridcube) #widefield debug

    def get_direction_npix(self, subregion_index=0):
        if subregion_index >= len(self.__clustercat[self.__current_direction]):
            raise IndexError("Index {0:d} is out of bounds for region {1:s} which contains {2:d} subregions".format(
                subregion_index, self.__current_direction, len(self.__clustercat[self.__current_direction])))
        return self.__clustercat[self.__current_direction][subregion_index].box_npx[0]

    def get_direction_pxoffset(self, subregion_index=0):
        """ get the pixel offset to the centre of the facet """
        if subregion_index >= len(self.__clustercat[self.__current_direction]):
            raise IndexError("Index {0:d} is out of bounds for region {1:s} which contains {2:d} subregions".format(
                subregion_index, self.__current_direction, len(self.__clustercat[self.__current_direction])))

        cluster = self.__clustercat[self.__current_direction][subregion_index]
        ctr = cluster.centre
        offset = (ctr[::-1] - np.array([self.__ny//2, self.__nx//2]))[::-1]
        offset[0] = -offset[0] #RA increases to the left
        return offset

    def get_degrid_model(self, subregion_index=0):
        """ get a facet from model cube for the direction previously set by set_direction """
        if self.__degridcube is None:
            raise ValueError("Set predict frequencies before running this method")
        if subregion_index >= len(self.__clustercat[self.__current_direction]):
            raise IndexError("Index {0:d} is out of bounds for region {1:s} which contains {2:d} subregions".format(
                subregion_index, self.__current_direction, len(self.__clustercat[self.__current_direction])))
        reg_data, reg_extents = BoundingConvexHull.regional_data(self.__clustercat[self.__current_direction][subregion_index],
                                                                 np.swapaxes(self.__degridcube, 2, 3)[:, :, :, ::-1])
        # cyril's model is transposed and flipped
        reg_data[...] = reg_data[:, :, ::-1, ::-1]
        return reg_data
