import numpy as np
import scipy.stats as sstats
import scipy.signal as ssig
import scipy.spatial as spat
import Tigger
from astropy import wcs

DEBUG = False

class BoundingConvexHull(object):
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

        pad_left = max(0, 0 - minx)
        pad_bottom = max(0, 0 - miny)
        pad_right = max(0, maxx - self._data.shape[3])
        pad_top = max(0, maxy - self._data.shape[2])
        cube_selected = np.tile(np.tile(region_selected[pad_bottom:region_selected.shape[0]-pad_top,
                                                pad_left:region_selected.shape[1]-pad_right],
                                        (self._data.shape[1], 1, 1)),
                                (self._data.shape[0], 1, 1, 1))
        self._data[: , :, pad_bottom+miny:maxy-pad_top, pad_left+minx:maxx-pad_right][cube_selected] = np.nan
        if DEBUG:
            from matplotlib import pyplot as plt
            plt.figure
            plt.imshow(self._data[pad_bottom+miny:maxy-pad_top, pad_left+minx:maxx-pad_right])
            plt.show()
        selected_data = self._data[:, :, pad_bottom+miny:maxy-pad_top, pad_left+minx:maxx-pad_right][np.logical_not(np.isnan(
            self._data[:, :, pad_bottom+miny:maxy-pad_top, pad_left+minx:maxx-pad_right]))].reshape(
            (self._data.shape[0], self._data.shape[1], maxy-pad_top - (pad_bottom+miny), maxx-pad_right - (pad_left+minx)))
        if any(np.array([pad_left, pad_bottom, pad_right, pad_top]) > 0):
            dcp_selected_data = np.zeros((self._data.shape[0],
                                          self._data.shape[1],
                                          maxy - miny,
                                          maxx - minx), dtype=selected_data.dtype)
            dcp_selected_data[:, :, 
                              pad_bottom:pad_bottom+selected_data.shape[2], 
                              pad_left:pad_left+selected_data.shape[3]] = \
                selected_data
        else:
            dcp_selected_data = selected_data.copy()
        return dcp_selected_data


    @property
    def area(self):
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        return 0.5 * np.abs(np.sum([x1*y2-x2*y1 for x1,y1,x2,y2 in lines]))

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, v):
        self._name = v

    @property
    def globaldata(self):
        return self._data.view()

    @globaldata.setter
    def globaldata(self, v):
        if not isinstance(v, np.ndarray):
            raise TypeError("data cube must be ndarray")
        if not v.ndim == 4:
            raise ValueError("data cube must be 4 dimensional")
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
        return map(lambda x: int(x), np.mean(self._vertices, axis=0))

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
    def __init__(self, xl, xu, yl, yu, name, imdata=None):
        BoundingConvexHull.__init__(self,
                                    [[xl,yl],[xl,yu],[xu,yu],[xu,yl]],
                                    name,
                                    imdata=imdata)
        self.__xnpx = abs(xu - xl)
        self.__ynpx = abs(xu - xl)
    
    @property
    def box_npx(self):
        return (self.__xnpx, self.__ynpx)

    @classmethod                                
    def AxisAlignedBoundingBox(cls, convex_hull_object, square=False, enforce_odd=True):
        """ Constructs an axis aligned bounding box around convex hull """
        if not isinstance(convex_hull_object, BoundingConvexHull):
            raise TypeError("Convex hull object passed in constructor is not of type BoundingConvexHull")
        if square:
            boxdiam = np.max(np.sum((convex_hull_object.corners - 
                                    convex_hull_object.centre[None, :])**2, axis=1))
            cx, cy = convex_hull_object.centre
            xl = cx - boxdiam
            xu = cx + boxdiam
            yl = cy - boxdiam
            yu = cy + boxdiam
        else:
            xl = np.min(convex_hull_object.corners[:, 0])
            xu = np.max(convex_hull_object.corners[:, 0])
            yl = np.min(convex_hull_object.corners[:, 1])
            yu = np.max(convex_hull_object.corners[:, 1])

        xu += (xu - xl + 1) % 2 if enforce_odd else 0
        yu += (yu - yl + 1) % 2 if enforce_odd else 0

        return BoundingBox(xl, xu, yl, yu,
                           convex_hull_object.name,
                           imdata=convex_hull_object.globaldata)

    @classmethod
    def SplitBox(cls, bounding_box_object, nsubboxes=1):
        """ Split a axis-aligned bounding box into smaller boxes """
        if not isinstance(bounding_box_object, BoundingBox):
            raise TypeError("Expected bounding box object")
        xl = np.min(bounding_box_object.corners[:, 0])
        xu = np.max(bounding_box_object.corners[:, 0])
        yl = np.min(bounding_box_object.corners[:, 1])
        yu = np.max(bounding_box_object.corners[:, 1])
        
        # construct a nonregular meshgrid bound to xu and yu
        x = xl + np.arange(0, nsubboxes) * int(np.ceil((xu - xl + 1) / float(nsubboxes)))
        x[-1] = min(xu, x[-1])
        y = yl + np.arange(0, nsubboxes) * int(np.ceil((yu - yl + 1) / float(nsubboxes)))
        y[-1] = min(yu, y[-1])
        xx, yy = np.meshgrid(np.arange(xl, xl + int(np.ceil((xu+1 - xl) / float(nsubboxes))) * nsubboxes),
                             np.arange(yl, yl + int(np.ceil((yu+1 - yl) / float(nsubboxes))) * nsubboxes), 
                             sparse=False)
        xls = xx[0:-1:2]
        xus = xx[1::2]
        yls = yy[0:-1:2]
        yus = yy[1::2]
        
        #coordinates for all the contained boxes, anti-clockwise wound
        bl = zip(xls, yls)
        br = zip(xus, yls)
        ur = zip(xus, yus)
        ul = zip(xls, yus)
        contained_boxes = zip(bl, br, ur, ul)

        #finally create bbs for each of the contained boxes
        new_regions = [BoundingBox(bl, br, ur, ul,
                                   bounding_box_object.name,
                                   imdata=bounding_box_object.globaldata)
                       for bl, br, ur, ul in contained_boxes]
        return new_regions

