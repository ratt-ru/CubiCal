import numpy as np
import scipy.stats as sstats
import scipy.signal as ssig
import scipy.spatial as spat

DEBUG = False

class BoundingConvexHull(object):
    def __init__(self, list_hulls, name="unnamed", imdata = None, mask = None):
        if mask is not None:
            if not isinstance(mask, list):
                raise TypeError("Mask must be list")
            if not (hasattr(mask, "__len__") and (len(mask) == 0 or (hasattr(mask[0], "__len__") and len(mask[0]) == 2))):
                raise TypeError("Mask must be a sparse mask of 2 element values")
        self._data = imdata
        self._name = name
        self._vertices = points = np.vstack([b.corners
            if hasattr(b, "corners") else [b[0], b[1]] for b in list_hulls])
        self._hull = spat.ConvexHull(points)
        if mask is None:
            self._mask = self.init_mask()
        else: 
            self._mask = filter(lambda c: c in self, mask)

    def __str__(self):
        return ",".join(["({0:d},{1:d})".format(x,y) for (x,y) in self.corners])
    
    def init_mask(self):
        """ creates a sparse mask of the convex hull """
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2]); maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2]); maxy = np.max(lines[:, 1:4:2])
        x = np.arange(minx, maxx, 1)
        y = np.arange(miny, maxy, 1)
        meshgrid = np.meshgrid(x, y)
        bounding_mesh = zip(*map(lambda x: np.ravel(x), np.meshgrid(x, y)))
        sparse_mask = filter(lambda c: c in self, bounding_mesh)
        return sparse_mask

    @property
    def sparse_mask(self):
        return self._mask
    
    @property
    def mask(self):
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2]); maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2]); maxy = np.max(lines[:, 1:4:2])
        nx = maxx - minx + 1 # inclusive
        ny = maxy - miny + 1
        mesh = np.zeros(nx*ny, dtype=np.bool)
        if nx==0 or ny==0 or len(self.sparse_mask) == 0: 
            return mesh
        else:
            sparse_mask = np.array(self.sparse_mask)
            sel = np.logical_and(np.logical_and(sparse_mask[:, 0] >= minx,
                                                sparse_mask[:, 0] <= maxx),
                                 np.logical_and(sparse_mask[:, 1] >= miny,
                                                sparse_mask[:, 1] <= maxy))

            sparse_mask[sel]
            flat_index = (sparse_mask[sel][:, 1] - miny) + (sparse_mask[sel][:, 0] - minx)*ny
            mesh[flat_index] = 1
            return mesh.reshape((nx, ny)).T

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
        """ area contained in hull """
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
        return self._data

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
        return self._vertices[self._hull.vertices][::-1]

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
    def edges(self):
        """ return edge segments of the hull (clockwise wound) """
        edges = []
        for i in xrange(self.corners.shape[0]):
            # assuming clockwise winding
            j = (i + 1) % self.corners.shape[0]
            edge = tuple([self.corners[i, :], self.corners[j, :]])
            edges.append(edge)
        return np.asarray(edges, dtype=np.double)

    @property
    def edge_midpoints(self):
        """ return edge midpoints of the hull (clockwise wound) """
        edges = self.edges
        return np.mean(edges, axis=1)

    @property
    def lnormals(self):
        """ left normals to the edges of the hull """
        return self.normals(left = True)

    @property
    def rnormals(self):
        """ right normals to the edges of the hull """
        return self.normals(left=False)
    
    def overlaps_with(self, other, min_sep_dist=1.0e-4):
        """ 
            Implements the separating lines collision detection theorem 
            to test whether the hull intersects with 'other' hull
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
            if minproj_reg2 - maxproj_reg1 > min_sep_dist or minproj_reg1 - maxproj_reg2 > min_sep_dist:
                return False
        return True

    @property
    def centre(self, integral=True):
        """ Barycentre of hull """
        if integral:
            return map(lambda x: int(x), np.mean(self._vertices, axis=0))
        else:
            return np.mean(self._vertices, axis=0)

    def __contains__(self, s):
        """ tests whether a point s(x,y) is in the convex hull """
        x, y = s
        isin = True
        normals = self.rnormals 
        xyvec = np.array([x, y])[None, :] - np.array(self.corners)
        # scalar projection  between all inner pointing right normals (clockwise winding) 
        # and the point must be positive if the point were to lie inside
        # the region         
        dot = np.einsum("ij,ij->i", normals, xyvec)
        return np.all(dot >= 0)

class BoundingBox(BoundingConvexHull):
    def __init__(self, xl, xu, yl, yu, name="unnamed", imdata=None, mask=None):
        if not all(map(lambda x: isinstance(x, int), [xl, xu, yl, yu])):
            raise ValueError("Box limits must be integers")
        self.__xnpx = abs(xu - xl)
        self.__ynpx = abs(yu - yl)
        BoundingConvexHull.__init__(self,
                                    [[xl,yl],[xl,yu],[xu,yu],[xu,yl]],
                                    name,
                                    imdata=imdata,
                                    mask=mask)

    def __contains__(self, s):
        """ tests whether a point s(x,y) is in the box"""
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2]); maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2]); maxy = np.max(lines[:, 1:4:2])
        return s[0] >= minx and s[0] <= maxx and s[1] >= miny and s[1] <= maxy

    def init_mask(self):
        """ creates a sparse mask of the box 
            since this is a box we can cheat a bit, making masks for general hulls is a lot
            more computationally expensive. We do however store the coordinates of the mask
            for self consistency
        """
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2]); maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2]); maxy = np.max(lines[:, 1:4:2])
        x = np.arange(minx, maxx, 1)
        y = np.arange(miny, maxy, 1)
        bounding_mesh = zip(*map(lambda x: np.ravel(x), np.meshgrid(x, y)))
        return bounding_mesh

    @property
    def box_npx(self):
        return (self.__xnpx, self.__ynpx)

    @classmethod                                
    def AxisAlignedBoundingBox(cls, convex_hull_object, square=False, enforce_odd=True):
        """ Constructs an axis aligned bounding box around convex hull """
        if not isinstance(convex_hull_object, BoundingConvexHull):
            raise TypeError("Convex hull object passed in constructor is not of type BoundingConvexHull")
        if square:
            boxdiam = np.sqrt(np.max(np.sum((convex_hull_object.corners - 
                                             np.array(convex_hull_object.centre)[None, :])**2, 
                                    axis=1)))
            boxdiam = int(np.ceil(boxdiam))
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
                           imdata=convex_hull_object.globaldata,
                           mask=convex_hull_object.sparse_mask)

    @classmethod
    def SplitBox(cls, bounding_box_object, nsubboxes=1):
        """ Split a axis-aligned bounding box into smaller boxes """
        if not isinstance(bounding_box_object, BoundingBox):
            raise TypeError("Expected bounding box object")
        if not (isinstance(nsubboxes, int) and nsubboxes >= 1):
            raise ValueError("nsubboxes must be integral type and be 1 or more")
        xl = np.min(bounding_box_object.corners[:, 0])
        xu = np.max(bounding_box_object.corners[:, 0])
        yl = np.min(bounding_box_object.corners[:, 1])
        yu = np.max(bounding_box_object.corners[:, 1])
        
        # construct a nonregular meshgrid bound to xu and yu
        x = xl + np.arange(0, nsubboxes + 1) * int(np.ceil((xu - xl + 1) / float(nsubboxes)))
        y = yl + np.arange(0, nsubboxes + 1) * int(np.ceil((yu - yl + 1) / float(nsubboxes)))
        xx, yy = np.meshgrid(x, y)

        # split into boxes
        xls = xx[0:-1, 0:-1]
        xus = xx[1:, 1:]
        yls = yy[0:-1, 0:-1]
        yus = yy[1:, 1:]

        # make sure no boxes overlap
        xus[:, :-1] -= 1
        yus[:-1, :] -= 1

        # clamp the final coordinate to the upper end (may result in rectanglular box at the end)
        xls[:, 0] = max(xl, min(xls[-1, 0], xl)) 
        yls[0, :] = max(yl, min(yls[0, -1], yl))
        xus[:, -1] = max(xu, min(xus[0, -1], xu)) 
        yus[-1, :] = max(yu, min(yus[-1, 0], yu))
        
        #coordinates for all the contained boxes, anti-clockwise wound
        xls = xls.ravel()
        yls = yls.ravel()
        xus = xus.ravel()
        yus = yus.ravel()
        bl = zip(xls, yls)
        br = zip(xus, yls)
        ur = zip(xus, yus)
        ul = zip(xls, yus)
        contained_boxes = zip(bl, br, ur, ul)

        #finally create bbs for each of the contained boxes with the mask
        #chopped up between the boxes by the convex hull initializer
        new_regions = [BoundingBox(bl[0], br[0], ur[0], ul[0],
                                   bounding_box_object.name,
                                   imdata=bounding_box_object.globaldata,
                                   mask=bounding_box_object.sparse_mask)
                       for bl, br, ur, ul in contained_boxes]

        return new_regions

if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    
    # # test case 1
    # vals = np.array([[50, 60], [20, 40], [-74, 50], [-95, +10], [20, 60]])
    # bh = BoundingConvexHull(vals)
    # mask = bh.mask 
    # assert np.abs(mask.sum() - bh.area) / bh.area < 0.01 # integral mask area needs to be close to true area
    # normalized_normals = bh.rnormals / np.linalg.norm(bh.rnormals, axis=1)[:, None]

    # # test case 2
    # for e, n in zip(bh.edges, normalized_normals):
    #     edge_vec = e[1] - e[0]
    #     assert np.all(np.abs(np.dot(edge_vec, n)) < 1.0e-8)

    # # test case 3
    # vals2 = np.array([[-20, 40], [-40, 0], [20, 30]])
    # bh2 = BoundingConvexHull(vals2)

    # vals3 = np.array([[-20, 58], [-40, 80], [20, 100]])
    # bh3 = BoundingConvexHull(vals3)
    # assert bh.overlaps_with(bh2)
    # assert not bh.overlaps_with(bh3)
    # assert not bh2.overlaps_with(bh3)

    # # test case 4
    # assert (-1000, -1000) not in bh
    # assert (30, 0) not in bh
    # assert (0, 0) not in bh
    # assert (-40, 30) in bh

    # # visual inspection
    # plt.figure(figsize=(7, 2.5))
    # for h in [bh, bh2, bh3]:
    #     for ei, e in enumerate(h.edges):
    #         plt.plot(e[:, 0], e[:, 1], "r--")
    #         plt.text(e[0, 0], e[0, 1], str(ei))
    
    # plt.plot(bh.edge_midpoints[:, 0], bh.edge_midpoints[:, 1], "ko")
    # for e, n in zip(bh.edge_midpoints, normalized_normals):
    #     p0 = e 
    #     p = e + n*6
    #     plt.plot([p0[0], p[0]], [p0[1], p[1]], "b--", lw=2)
    
    # plt.scatter(vals[:, 0], vals[:, 1])    
    # plt.imshow(mask, extent=[np.min(vals[:, 0]), np.max(vals[:, 0]), np.max(vals[:, 1]), np.min(vals[:, 1])])

    # plt.grid(True)
    # plt.show(False)

    # test case 5
    bb = BoundingBox(-14, 21, 30, 50)
    assert bb.centre == [3, 40]
    assert bb.box_npx == (35, 20)
    assert bb.area == 35 * 20
    assert np.sum(bb.mask) == bb.area
    assert (-15, 35) not in bb
    assert (0, 35) in bb
    # bb2 = BoundingBox.AxisAlignedBoundingBox(bb) #enforce odd
    # assert bb2.centre == [3, 40]
    # assert bb2.box_npx == (35, 21)
    # assert bb2.area == 35 * 21
    # assert bb.sparse_mask == bb2.sparse_mask
    # assert (-15, 35) not in bb2
    # assert (0, 35) in bb2
    # bb3 = BoundingBox.AxisAlignedBoundingBox(bb, square=True) #enforce odd
    # assert bb3.centre == [3, 40]
    # assert bb3.box_npx[0] == bb3.box_npx[1]
    # assert bb3.box_npx[0] % 2 == 1 #enforce odd
    # assert bb3.area == bb3.box_npx[0]**2
    # assert bb.sparse_mask == bb2.sparse_mask
    # assert (-15, 35) not in bb2
    # assert (0, 35) in bb2

    # test case 6
    bb4s = BoundingBox.SplitBox(bb, nsubboxes=3)
    assert np.sum([b.area for b in bb4s]) == bb.area

    for bb4 in bb4s:
        assert bb4.area == np.sum(bb4.mask)


