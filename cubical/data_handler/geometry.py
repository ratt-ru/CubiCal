import numpy as np
import scipy.stats as sstats
import scipy.signal as ssig
import scipy.spatial as spat
import Tigger
from astropy import wcs

DEBUG = False

class BoundingConvexHull():
    def __init__(self, list_hulls, name, imdata = None):
        self._data = imdata
        self._name = name
        self._vertices = points = np.vstack([b.corners
            if hasattr(b, "corners") else [b[0], b[1]] for b in list_hulls])
        self._hull = spat.ConvexHull(points)

    def __str__(self):
        return ",".join(["({0:d},{1:d})".format(x,y) for (x,y) in self.corners])

    @property
    def regional_data(self):
        """ 2D array containing all values within convex hull """
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2]); maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2]); maxy = np.max(lines[:, 1:4:2])
        x = np.arange(minx, maxx, 1)
        y = np.arange(miny, maxy, 1)
        bounding_mesh = np.meshgrid(x, y)
        # inverted mask
        region_selected = np.zeros((y.shape[0], x.shape[0]), dtype=np.bool)
        for x1, y1, x2, y2 in lines:
            sel_indx = np.logical_and(np.logical_and(bounding_mesh[0]>=min(x1, x2),
                                                     bounding_mesh[0]<=max(x1, x2)),
                                      np.logical_and(bounding_mesh[1]>=min(y1, y2),
                                                     bounding_mesh[1]<=max(y1, y2))) 
            region_selected[sel_indx] = True
        self._data[: , :, miny:maxy, minx:maxx][region_selected] = np.nan
        if DEBUG:
            from matplotlib import pyplot as plt
            plt.figure
            plt.imshow(self._data[miny:maxy, minx:maxx])
            plt.show()
        selected_data = self._data[:, :, miny:maxy, minx:maxx][np.logical_not(np.isnan(
            self._data[:, :, miny:maxy, minx:maxx]))]
        return selected_data
        

    @property
    def area(self):
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        return 0.5 * np.abs(np.sum([x1*y2-x2*y1 for x1,y1,x2,y2 in lines]))

    @property
    def name(self):
        return self._name

    @property
    def global_data(self):
        return self._data.view()

    @global_data.setter
    def global_data(self, v):
        self._data = v

    @property
    def corners(self):
        """ Returns vertices and guarentees clockwise winding """
        return self._vertices[self._hull.vertices]

    def normals(self, left = True):
        """ return a list of left normals to the hull """
        normals = []
        for i in xrange(self.corners.shape[0]):
            # assuming clockwise winding
            j = (i + 1) % self.corners.shape[0]
            edge = self.corners[j, :] - self.corners[i, :]
            if left:
                normals.append((-edge[1], edge[0]))
            else:
                normals.append((edge[1], -edge[0]))
        return np.asarray(normals, dtype=np.double)

    @property
    def lnormals(self):
        return self.normals(left = True)

    @property
    def rnormals(self):
        return self.normals(left=False)
    
    def is_neighbour(self, other, min_sep_dist=1.0e-4):
        """ 
            Implements the separating lines collision detection theorem 
        """
        if not isinstance(other, BoundingConvexHull):
            raise TypeError("rhs must be a BoundingConvexHull")

        # get the projection axes
        normals = np.vstack([self.lnormals, other.lnormals])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[None, 2]

        # compute vectors to corners from origin
        vecs_reg1 = self.corners
        vecs_reg2 = other.corners

        # compute projections onto normals
        for ni, n in enumerate(normals):
            projs = np.dot(vecs_reg1, n.T)
            minproj_reg1 = np.min(projs)
            maxproj_reg1 = np.max(projs)
            projs = np.dot(vecs_reg2, n.T)
            minproj_reg2 = np.min(projs)
            maxproj_reg2 = np.max(projs)
            if minproj_reg2 - maxproj_reg1 > 1.0e-4 or minproj_reg1 - maxproj_reg2 > 1.0e-4:
                return False
        return True

    @property
    def centre(self):
        # Barycentre of polygon
        return np.mean(self._vertices, axis=0)

    def __contains__(self, s):
        if not isinstance(s, Tigger.Models.SkyModel.Source):
            raise TypeError("Source must be a Tigger lsm source")
        ra = np.rad2deg(s.pos.ra)
        dec = np.rad2deg(s.pos.dec)
        x, y, _, _ = self._wcs.all_world2pix([[ra, dec, 0, 0]], 1)[0]

        dot = 0
        for i in range(len(self.corners)):
            j = (i + 1) % len(self.corners)
            v1 = self.corners[i] - np.array([x, y])
            v2 = self.corners[j] - np.array([x, y])
            dot += np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, +1))
        return np.abs(360 - np.rad2deg(dot)) < 1.0e-6

class BoundingBox(BoundingConvexHull):
    def __init__(self, xl, xu, yl, yu, name, imdata):
        BoundingConvexHull.__init__(self,
                                    [[xl,yl],[xl,yu],[xu,yu],[xu,yl]],
                                    name)
        self.__xnpx = abs(xu - xl) + 1
        self.__ynpx = abs(xu - xl) + 1
    
    @property
    def box_npx(self):
        return (self.__xnpx, self.__ynpx)

    @classmethod                                
    def AxisAlignedBoundingBox(self, convex_hull_object):
        """ Constructs an axis aligned square bounding box around convex hull """
        if not isinstance(convex_hull_object, BoundingConvexHull):
            raise TypeError("Convex hull object passed in constructor is not of type BoundingConvexHull")
        boxdiam = np.max(np.sum((convex_hull_object.corners - 
                                 convex_hull_object.centre[None, :])**2, axis=1))
        cx, cy = convex_hull_object.centre
        xl = cx - boxdiam
        xu = cx + boxdiam
        yl = cy - boxdiam
        yu = cy + boxdiam
        BoundingBox.__init__(self, xl, xu, yl, yu,
                             convex_hull_object.name)