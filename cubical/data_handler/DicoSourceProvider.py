try:
    from DDFacet.Imager import ModModelMachine as MMM
    from DDFacet.ToolsDir.ModToolBox import EstimateNpix
except ImportError:
    raise ImportError("Cannot import DDFacet")

from regions import DS9Parser
import numpy as np
from geometry import BoundingConvexHull, BoundingBox

class DicoSourceProvider(object):
    def __init__(self,
                 fn,
                 phasedir,
                 facet_padding_factor,
                 clustercat=None):
        self.__dicofn = fn
        self.__dicomodel = MMM.ClassModModelMachine().GiveInitialisedMMFromFile(fn)
        # assume GD is stored in the Dico... this is only true for DDF versions 0.4.1 and beyond
        self.__pxscale = self.__dicomodel.GD["Image"]["Cell"] # in arcseconds
        nchan, npol, ny, nx = self.__dicomodel.ModelShape
        self.__phasedir = phasedir
        if clustercat:
            self.__clustercat = self.__read_regions_file(clustercat, nx//2, ny//2, facet_padding_factor)
        else:
            self.__clustercat = {"die": self.pad_cluster(BoundingBox(0, nx + 1, 0, ny + 1, name="die"),
                                                         facet_padding_factor)}
        self.__current_direction = 0
        self.__degridcube = None
        self.__degridfreqs = None
        self.__padding_factor = facet_padding_factor

    def __str__(self):
        return self.__dicofn

    @property
    def pixel_scale(self):
        return self.__pxscale
    
    @property
    def degrid_cube_shape(self):
        if self.__degridcube is None:
            raise RuntimeError("Degrid frequencies not set yet. Please set frequencies before enquiring about the degrid cube")
        return self.__degridcube.shape

    @classmethod
    def pad_cluster(cls, c, padding_factor):
        npx,_ = c.box_npx # square facet at this point
        # this returns an odd npix:
        npixunpadded, npixpadded = EstimateNpix(npx, Padding=padding_factor)
        xc, yc = c.centre
        xl = xc - npixpadded // 2
        xu = xc + npixpadded // 2
        yl = yc - npixpadded // 2 
        yu = yc + npixpadded // 2
        return BoundingBox(xl, xu + 1, yl, yu + 1,
                           c.name)

    @classmethod
    def __read_regions_file(cls, fn, offsetx, offsety, padding_factor):
        clusters = []
        with open(fn) as f:
            parser = DS9Parser(f.read())
            for regi, reg in enumerate(parser.shapes):
                coords = [c.value for c in reg.coord]
                assert coords % 2 == 0, "Number of region coords must be multiple of 2-tuple"
                coords = coords.reshape([coords // 2, 2])
                coords += np.array([offsetx, offsety])[None, 2]
                clusters += BoundingConvexHull(coords,
                                               name=regi)
        
        # now create axis aligned bounding boxes for each of the regions
        # and further split them to the maximum permitted facet size
        clusters = [BoundingBox.AxisAlignedBoundingBox(c) for c in clusters]
        clusters = [BoundingBox.AxisAlignedBoundingBox(aasubbox, square=True) 
                    for aasubbox in BoundingBox.SplitBox(c) for c in clusters]
        
        clusters = map(lambda c: cls.pad_cluster(c, padding_factor), clusters)
        
        #finally rename them to be unique
        for ci, c in enumerate(clusters):
            c.name = ci
    
        return dict(zip([c.name for c in clusters], clusters))

    @property
    def _cluster_keys(self):
        """ returns list of cluster name keys if there are any clusters
            otherwise ["die"] if no dE clusters were specified upon construction
        """
        return ["die"] if self.__clustercat is None else \
            [self.__clustercat[c].name for c in self.__clustercat]
    
    def set_direction(self, v):
        """ sets the cluster (direction) index to predict from """
        if v not in self._cluster_keys:
            raise KeyError("Unknown direction '{0:s}' for model '{1:s}'".format(str(v), str(self)))
        self.__current_direction = v
    
    @property
    def _nclus(self):
        return len(self.__clustercat) if self.__clustercat is not None \
            else 1 #die

    @property
    def direction(self):
        return self.__current_direction

    @direction.setter
    def direction(self, v):
        self.set_direction(v)

    def set_frequency(self, freqs):
        self.__degridfreqs = freqs.copy()
        self.__degridcube = self.__dicomodel.GiveModelImage(FreqIn=self.__degridfreqs)
        for c in self.__clustercat:
            cluster = self.__clustercat[c]
            cluster.globaldata = self.__degridcube

    def get_direction_npix(self):
       return self.__clustercat[self.__current_direction].box_npx[0]

    def get_direction_pxoffset(self):
        """ get the pixel offset to the centre of the facet """
        nchan, npol, nx, ny = self.__dicomodel.ModelShape
        ctr = self.__clustercat[self.__current_direction].centre
        offset = ctr - np.array([ny//2, nx//2])
        return offset

    def get_degrid_model(self):
        """ get a facet from model cube for the direction previously set by set_direction """
        if self.__degridcube is None:
            raise ValueError("Set predict frequencies before running this method")

        if self.__clustercat is None: #DIE - single facet
            return self.__degridcube
        else: #DDE - facet for current direction
            self.__clustercat[self.__current_direction].globaldata = self.__degridcube
            reg_data = self.__clustercat[self.__current_direction].regional_data
            return reg_data
