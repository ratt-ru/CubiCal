try:
    from DDFacet.Imager import ModModelMachine as MMM
    from DDFacet.ToolsDir.ModToolBox import EstimateNpix
except ImportError:
    raise ImportError("Cannot import DDFacet")

from regions import DS9Parser
import numpy as np
from geometry import BoundingConvexHull, BoundingBox, BoundingBoxFactory
from cubical.tools import logger, ModColor
log = logger.getLogger("DDFacetSim")

class DicoSourceProvider(object):
    def __init__(self,
                 fn,
                 phasedir,
                 facet_padding_factor,
                 max_facet_size,
                 min_nfacet_per_axis,
                 clustercat=None):
        log.info("Initializing new source provider for DDFacet model '{0:s}' into regions specified by '{1:s}'.".format(
            fn, clustercat if clustercat is not None else 'die'))
        self.__dicofn = fn
        self.__dicomodel = MMM.ClassModModelMachine().GiveInitialisedMMFromFile(fn)
        # assume GD is stored in the Dico... this is only true for DDF versions 0.4.1 and beyond
        self.__pxscale = self.__dicomodel.GD["Image"]["Cell"] # in arcseconds
        nchan, npol, ny, nx = self.__dicomodel.ModelShape
        self.__phasedir = phasedir
        self.__clustercat = self.__read_regions_file(clustercat, 
                                                     nx//2, 
                                                     ny//2, 
                                                     facet_padding_factor, 
                                                     max_facet_size,
                                                     min_nfacet_per_axis)
        self.__current_direction = 0
        self.__degridcube = None
        self.__degridfreqs = None
        self.__padding_factor = facet_padding_factor
        log.info("Initialization sequence of source provider '{0:s}' completed".format(fn))

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

    def __read_regions_file(self, fn, offsetx, offsety, padding_factor, max_size, min_nfacet_per_axis):
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
        nchan, npol, ny, nx = self.__dicomodel.ModelShape
    
        clusters = []
        if fn is not None: # dde case
            with open(fn) as f:
                parser = DS9Parser(f.read())
                for regi, reg in enumerate(parser.shapes):
                    coords = [c.value for c in reg.coord]
                    assert coords % 2 == 0, "Number of region coords must be multiple of 2-tuple"
                    coords = coords.reshape([coords // 2, 2])
                    coords += np.array([offsetx, offsety])[None, 2]
                    clusters += BoundingConvexHull(coords,
                                                   name=regi)
        else: # die case
            clusters = [BoundingBox(0, nx - 1, 0, ny - 1, name="die", check_mask_outofbounds=False)]
        log.info("\tInitialized bounding boxes for regions. There are {0:d} region(s)".format(len(clusters)))
        # now create axis aligned bounding boxes for each of the regions
        # and further split them to the maximum permitted facet size
        clusters = [BoundingBoxFactory.AxisAlignedBoundingBox(c, check_mask_outofbounds=False) for c in clusters]
        def __split_regular_region(reg, max_size):
            if max_size < 0:
                raise ValueError("Expected positive value for min_size")
            reg_size_deg = np.max(np.array(reg.box_npx) * self.pixel_scale / 3600.0)
            nsplit = max(1, max(min_nfacet_per_axis, int(np.ceil(reg_size_deg / max_size))))
            return BoundingBoxFactory.SplitBox(reg, nsubboxes=nsplit, check_mask_outofbounds=False)
        log.info("\tSplitting regions into facetted regions, with maximum unpadded size of {0:.2f} degrees per facet".format(max_size))
        clusters = [aasubreg for aareg in map(lambda reg: __split_regular_region(reg, max_size), clusters) 
                    for aasubreg in aareg]
        clusters = map(lambda reg: BoundingBoxFactory.AxisAlignedBoundingBox(reg, square=True, check_mask_outofbounds=False), clusters)
        
        def __pad_cluster(c, padding_factor):
            npx,_ = c.box_npx # square facet at this point
            # this returns an odd npix:
            npixunpadded, npixpadded = EstimateNpix(npx, Padding=padding_factor)
            return BoundingBoxFactory.PadBox(c, npixpadded, npixpadded, check_mask_outofbounds=False)
        log.info("\tPadding all facets by a minimum factor of {0:.2f}x".format(padding_factor))
        clusters = map(lambda c: __pad_cluster(c, padding_factor), clusters)
        #TODO: need to rethink this: 
        #log.info("\tNormalizing regional weights")
        #BoundingConvexHull.normalize_masks(clusters)
        log.info("\tCaching regional weight maps for future predicts")
        map(lambda x: x.mask, clusters) # cache mask
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

        nchan, npol, nx, ny = self.__dicomodel.ModelShape
        cluster = self.__clustercat[self.__current_direction][subregion_index]
        ctr = cluster.centre
        offset = ctr[::-1] - np.array([ny//2, nx//2])
        return offset

    def get_degrid_model(self, subregion_index=0):
        """ get a facet from model cube for the direction previously set by set_direction """
        if self.__degridcube is None:
            raise ValueError("Set predict frequencies before running this method")
        if subregion_index >= len(self.__clustercat[self.__current_direction]):
            raise IndexError("Index {0:d} is out of bounds for region {1:s} which contains {2:d} subregions".format(
                subregion_index, self.__current_direction, len(self.__clustercat[self.__current_direction])))
        self.__clustercat[self.__current_direction][subregion_index].globaldata = self.__degridcube
        reg_data, reg_extents = BoundingConvexHull.regional_data(self.__clustercat[self.__current_direction][subregion_index],
                                                                 self.__degridcube)
        return reg_data
