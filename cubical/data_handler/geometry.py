import numpy as np
import scipy.stats as sstats
import scipy.signal as ssig
import scipy.spatial as spat
import copy

DEBUG = True

class BoundingConvexHull(object):
    def __init__(self, list_hulls, name="unnamed", mask = None):
        if mask is not None:
            if not isinstance(mask, list):
                raise TypeError("Mask must be list")
            if not (hasattr(mask, "__len__") and (len(mask) == 0 or (hasattr(mask[0], "__len__") and len(mask[0]) == 2))):
                raise TypeError("Mask must be a sparse mask of 2 element values")
        self._name = name
        self._cached_filled_mask = None
        self._vertices = points = np.vstack([b.corners
            if hasattr(b, "corners") else [b[0], b[1]] for b in list_hulls])
        self._hull = spat.ConvexHull(points)
        if mask is None:
            self._mask = self.init_mask()
        else: 
            self._mask = copy.deepcopy(filter(lambda c: (c[1], c[0]) in self, mask))

    def invalidate_cached_masks(self):
        """ Invalidates the cached masks (sparse or regular) """
        self._cached_filled_mask = None
        self.init_mask()

    def __str__(self):
        return ",".join(["({0:d},{1:d})".format(x,y) for (x,y) in self.corners])
    
    def init_mask(self):
        """ creates a sparse mask of the convex hull of the form (y, x) tuples """
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2]); maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2]); maxy = np.max(lines[:, 1:4:2])
        x = np.arange(minx, maxx + 1, 1) #upper limit inclusive
        y = np.arange(miny, maxy + 1, 1)
        meshgrid = np.meshgrid(y, x)
        bounding_mesh = zip(*map(lambda x: np.ravel(x), np.meshgrid(y, x)))
        sparse_mask = filter(lambda c: c[::-1] in self, bounding_mesh)
        return sparse_mask

    @property
    def sparse_mask(self):
        """ returns a sparse mask (y, x) values of all points in the masked region """
        return self._mask
    
    @property
    def mask(self):
        """ Creates a filled rectangular mask grid of size y, x """
        if self._cached_filled_mask is not None:
            return self._cached_filled_mask

        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2]); maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2]); maxy = np.max(lines[:, 1:4:2])
        nx = maxx - minx + 1 # inclusive
        ny = maxy - miny + 1
        mesh = np.zeros(nx*ny, dtype=np.bool)
        if nx==0 or ny==0 or len(self.sparse_mask) == 0: 
            self._cached_filled_mask = mesh.reshape((ny, nx))
        else:
            sparse_mask = np.array(self.sparse_mask)
            sel = np.logical_and(np.logical_and(sparse_mask[:, 1] >= minx,
                                                sparse_mask[:, 1] <= maxx),
                                 np.logical_and(sparse_mask[:, 0] >= miny,
                                                sparse_mask[:, 0] <= maxy))

            flat_index = (sparse_mask[sel][:, 0] - miny)*nx + (sparse_mask[sel][:, 1] - minx)
            mesh[flat_index] = 1
            self._cached_filled_mask = mesh.reshape((ny, nx))
        return self._cached_filled_mask

    @classmethod
    def regional_data(cls, sel_region, data_cube, axes=(2, 3), oob_value=0):
        """ 2D array containing all values within convex hull 
            sliced out along axes provided as argument. Portions of sel_region
            that are outside of the data_cube is set to oob_value

            assumes the last value of axes is the fastest varying axis
        """
        if not isinstance(sel_region, BoundingConvexHull):
            raise TypeError("Object passed in is not of type BoundingConvexHull")
        if not (hasattr(axes, "__len__") and len(axes) == 2):
            raise ValueError("Expected a tupple of axes along which to slice out a region")
        axes = sorted(axes)

        lines = np.hstack([sel_region.corners, np.roll(sel_region.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2]); maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2]); maxy = np.max(lines[:, 1:4:2])
        x = np.arange(minx, maxx + 1, 1)
        y = np.arange(miny, maxy + 1, 1)

        pad_left = max(0, 0 - minx)
        pad_bottom = max(0, 0 - miny)
        pad_right = max(0, maxx - data_cube.shape[axes[1]] + 1) #inclusive of upper limit
        pad_top = max(0, maxy - data_cube.shape[axes[0]] + 1)

        # extract data, pad if necessary
        slc_data = [slice(None)] * len(data_cube.shape)
        for (start, end), axis in zip([(miny + pad_bottom, maxy - pad_top + 1), 
                                       (minx + pad_left, maxx - pad_right + 1)], axes):
            slc_data[axis] = slice(start, end)
        slc_padded = [slice(None)] * len(data_cube.shape)
        for (start, end), axis in zip([(pad_bottom, -miny + maxy + 1 - pad_top), 
                                       (pad_left, -minx + maxx + 1 - pad_right)], axes):
            slc_padded[axis] = slice(start, end)

        selected_data = data_cube[tuple(slc_data)]
        new_shape = list(data_cube.shape)
        new_shape[axes[0]] = pad_bottom + (maxy - miny + 1) + pad_top
        new_shape[axes[1]] = pad_left + (maxx - minx + 1) + pad_right

        if any(np.array([pad_left, pad_bottom, pad_right, pad_top]) > 0):
            padded_data = np.zeros(tuple(new_shape), dtype=selected_data.dtype) * oob_value
            padded_data[tuple(slc_padded)] = selected_data.copy()
        else:
            padded_data = selected_data.copy()

        # finally apply mask
        slc_mask = [slice(None)] * len(padded_data.shape)
        for (start, end), axis in zip([(0, maxy - miny + 1),  #mask starts at origin in the padded image
                                       (0, maxx - minx + 1)], axes):
            slc_mask[axis] = slice(start, end)
        assert padded_data[tuple(slc_mask)].size == sel_region.mask.size
        mask = sel_region.mask.astype(np.float32)
        mask[mask == 0] = oob_value
        padded_data[tuple(slc_mask)] *= mask
        window_extents = [- pad_left, maxx + pad_right + pad_left - 1, 
                          - pad_bottom, maxy + pad_top + pad_bottom - 1]
        return padded_data, window_extents

    @property
    def circumference(self):
        """ area contained in hull """
        lines = self.edges
        return np.sum(np.linalg.norm(lines[:, 1, :] - lines[:, 0, :], axis=1) + 1)
    @property
    def area(self):
        """ area contained in hull """
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        return 0.5 * np.abs(np.sum([x1*(y2)-(x2)*y1 for x1,y1,x2,y2 in lines])) + 0.5 * self.circumference - 1

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, v):
        self._name = v

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
    
    def overlaps_with(self, other, min_sep_dist=0.5): #less than half a pixel away
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

    def __contains__(self, s, tolerance=0.5): #less than half a pixel away
        """ tests whether a point s(x,y) is in the convex hull """
        # there are three cases to consider
        # CASE 1:
        # scalar projection  between all inner pointing right normals (clockwise winding) 
        # and the point must be positive if the point were to lie inside
        # the region (true)
        # CASE 2:
        # point is on an edge - the scalar projection onto the axis is 0 for that edge
        # and greater than 0 for the other edges (true)
        # CASE 3:
        # it is outside (false)
        x, y = s
        isin = True
        normals = self.rnormals 
        xyvec = np.array([x, y])[None, :] - np.array(self.corners)
        
        dot = np.einsum("ij,ij->i", normals, xyvec)
        return np.all(dot > -tolerance)

class BoundingBox(BoundingConvexHull):
    def __init__(self, xl, xu, yl, yu, name="unnamed", mask=None):
        if not all(map(lambda x: isinstance(x, int), [xl, xu, yl, yu])):
            raise ValueError("Box limits must be integers")
        self.__xnpx = abs(xu - xl + 1) #inclusive of the upper pixel
        self.__ynpx = abs(yu - yl + 1)
        BoundingConvexHull.__init__(self,
                                    [[xl,yl],[xl,yu],[xu,yu],[xu,yl]],
                                    name,
                                    mask=mask)

    def __contains__(self, s):
        """ tests whether a point s(x,y) is in the box"""
        lines = np.hstack([self.corners, np.roll(self.corners, -1, axis=0)])
        minx = np.min(lines[:, 0:4:2]); maxx = np.max(lines[:, 0:4:2])
        miny = np.min(lines[:, 1:4:2]); maxy = np.max(lines[:, 1:4:2])
        return s[0] >= minx and s[0] <= maxx and s[1] >= miny and s[1] <= maxy

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

        xu += (xu - xl) % 2 if enforce_odd else 0
        yu += (yu - yl) % 2 if enforce_odd else 0

        return BoundingBox(xl, xu, yl, yu,
                           convex_hull_object.name,
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
        xls = xx[0:-1, 0:-1].copy()
        xus = xx[1:, 1:].copy()
        yls = yy[0:-1, 0:-1].copy()
        yus = yy[1:, 1:].copy()

        # make sure no boxes overlap
        xus = xus - 1
        yus = yus - 1

        # clamp the final coordinate to the upper end (may result in rectanglular box at the end)
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
        new_regions = [BoundingBox(bl[0], br[0], bl[1], ul[1],
                                   bounding_box_object.name,
                                   mask=bounding_box_object.sparse_mask)
                       for bl, br, ur, ul in contained_boxes]

        return new_regions

    @classmethod
    def PadBox(cls, bounding_box_object, desired_nx, desired_ny):
        """ Creates a box with a padded border around a axis-aligned bounding box """
        if not isinstance(bounding_box_object, BoundingBox):
            raise TypeError("Expected bounding box object")
        nx, ny = bounding_box_object.box_npx
        if desired_nx - nx < 0 or desired_ny - ny < 0:
            raise ValueError("Padded box must be bigger than original box")
        pad_left = desired_nx // 2
        pad_right = desired_nx - pad_left - 1
        pad_bottom = desired_ny // 2
        pad_top = desired_ny - pad_bottom - 1
        cx, cy = bounding_box_object.centre
        xl = cx - pad_left
        xu = cx + pad_right
        yl = cy - pad_bottom
        yu = cy + pad_top
        return BoundingBox(xl, xu, yl, yu,
                           c.name, 
                           mask=bounding_box_object.sparse_mask) #mask unchanged in the new shape, border frame discarded
if __name__ == "__main__":    
    # test case 1
    vals = np.array([[50, 60], [20, 40], [-74, 50], [-95, +10], [20, 60]])
    bh = BoundingConvexHull(vals)
    mask = bh.mask 
    assert mask.shape == (np.max(vals[:, 1]) - np.min(vals[:, 1]) + 1, np.max(vals[:, 0]) - np.min(vals[:, 0]) + 1)
    assert np.abs(mask.sum() - bh.area) / bh.area < 0.05 # integral mask area needs to be close to true area
    normalized_normals = bh.rnormals / np.linalg.norm(bh.rnormals, axis=1)[:, None]
    # test case 2
    for e, n in zip(bh.edges, normalized_normals):
        edge_vec = e[1] - e[0]
        assert np.all(np.abs(np.dot(edge_vec, n)) < 1.0e-8)

    # test case 3
    valsextract = np.array([[-10, 120], [90, 268], [293, 110],[40, -30]])
    bh_extract = BoundingConvexHull(valsextract)
    sinc_npx = 255
    sinc = np.sinc(np.linspace(-10, 10, sinc_npx))
    sinc2d = np.outer(sinc, sinc).reshape((1, 1, sinc_npx, sinc_npx))
    extracted_data, extracted_window_extents = BoundingConvexHull.regional_data(bh_extract, sinc2d, oob_value=np.nan)
    assert extracted_window_extents == [-10, 341, -30, 311]
    sparse_mask = np.array(bh_extract.sparse_mask)
    lines = np.hstack([bh_extract.corners, np.roll(bh_extract.corners, -1, axis=0)])
    minx = np.min(lines[:, 0:4:2]); maxx = np.max(lines[:, 0:4:2])
    miny = np.min(lines[:, 1:4:2]); maxy = np.max(lines[:, 1:4:2])
    sel = np.logical_and(np.logical_and(sparse_mask[:, 1] >= 0,
                                        sparse_mask[:, 1] < 255),
                         np.logical_and(sparse_mask[:, 0] >= 0,
                                        sparse_mask[:, 0] < 255))
    
    flat_index = (sparse_mask[sel][:, 0])*sinc_npx + (sparse_mask[sel][:, 1])
    assert np.abs(np.sum(sinc2d.ravel()[flat_index]) - np.nansum(extracted_data.ravel())) < 1.0e-8

    # test case 4
    vals2 = np.array([[-20, -120], [0, 60], [40, -60]])
    vals3 = np.array([[-20, 58], [-40, 80], [20, 100]])
    bh2 = BoundingConvexHull(vals2)
    bh3 = BoundingConvexHull(vals3)
    assert bh.overlaps_with(bh2)
    assert not bh.overlaps_with(bh3)
    assert not bh2.overlaps_with(bh3)

    # test case 5
    assert (-1000, -1000) not in bh
    assert (30, 0) not in bh
    assert (0, 0) not in bh
    assert (-40, 30) in bh

    # test case 6
    bb = BoundingBox(-14, 20, 30, 49)
    assert bb.centre == [3, 39]
    assert bb.box_npx == (35, 20)
    assert bb.mask.shape == bb.box_npx[::-1]
    assert bb.area == 35 * 20
    
    assert np.sum(bb.mask) == bb.area
    assert (-15, 35) not in bb
    assert (0, 35) in bb

    bb2 = BoundingBox.AxisAlignedBoundingBox(bb) #enforce odd
    assert bb2.box_npx == (35, 21)
    assert bb2.area == 35 * 21
    assert bb.sparse_mask == bb2.sparse_mask
    assert (-15, 35) not in bb2
    assert (0, 35) in bb2

    bb3 = BoundingBox.AxisAlignedBoundingBox(bb, square=True) #enforce odd
    assert bb3.box_npx[0] == bb3.box_npx[1]
    assert bb3.box_npx[0] % 2 == 1 #enforce odd
    assert bb3.area == bb3.box_npx[0]**2
    assert bb.sparse_mask == bb3.sparse_mask
    assert (-15, 35) not in bb2
    assert (0, 35) in bb2

    # test case 7
    bb4s = BoundingBox.SplitBox(bb, nsubboxes=3)
    assert len(bb4s) == 9
    xlims = [(np.min(c.corners[:, 0]), np.max(c.corners[:, 0])) for c in bb4s][0:3]
    ylims = [(np.min(c.corners[:, 1]), np.max(c.corners[:, 1])) for c in bb4s][0::3]
    assert np.all(xlims == np.array([(-14, -3), (-2, 9), (10, 20)]))
    assert np.all(ylims == np.array([(30, 36), (37, 43), (44, 49)]))
    assert np.sum([b.area for b in bb4s]) == bb.area

    for bb4 in bb4s:
        assert bb4.area == np.sum(bb4.mask)

    # test case 8
    bb5 = BoundingBox(-14, 20, 30, 50)
    assert bb5.box_npx == (35, 21)
    bb6 = BoundingBox.PadBox(bb5, 41, 27)
    assert bb6.box_npx == (41, 27)
    assert bb5.centre == bb6.centre
    assert np.sum(bb5.mask) == np.sum(bb6.mask)
    bb7s = map(lambda x: BoundingBox.PadBox(x, 17, 11), bb4s)
    assert all([b.box_npx == (17, 11) for b in bb7s])
    assert np.sum([np.sum(b.mask) for b in bb7s]) == np.sum([np.sum(b.mask) for b in bb4s])

    # visual inspection
    if DEBUG:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(7, 2.5))
        for h in [bh, bh2, bh3]:
            for ei, e in enumerate(h.edges):
                plt.plot(e[:, 0], e[:, 1], "r--")
                plt.text(e[0, 0], e[0, 1], str(ei))
        
        plt.plot(bh.edge_midpoints[:, 0], bh.edge_midpoints[:, 1], "ko")
        for e, n in zip(bh.edge_midpoints, normalized_normals):
            p0 = e 
            p = e + n*6
            plt.plot([p0[0], p[0]], [p0[1], p[1]], "b--", lw=2)
        
        plt.scatter(vals[:, 0], vals[:, 1])    
        plt.imshow(mask, extent=[np.min(vals[:, 0]), np.max(vals[:, 0]), np.max(vals[:, 1]), np.min(vals[:, 1])])

        plt.grid(True)
        plt.figure(figsize=(7, 2.5))
        for h in [bh_extract]:
            for ei, e in enumerate(h.edges):
                plt.plot(e[:, 0], e[:, 1], "r--")
        plt.imshow(sinc2d[0, 0, :, :], extent=[0, sinc_npx, sinc_npx, 0])
        plt.grid(True)

        plt.figure(figsize=(7, 2.5))
        for h in [bh_extract]:
            for ei, e in enumerate(h.edges):
                plt.plot(e[:, 0], e[:, 1], "r--")
        plt.imshow(extracted_data[0, 0, :, :], 
            extent=[extracted_window_extents[0], extracted_window_extents[1],
                    extracted_window_extents[3], extracted_window_extents[2]])
        plt.show(True)


