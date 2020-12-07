# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Handles parameter databases which can contain solutions and other relevant values. 
"""
from __future__ import print_function
from builtins import range
import numpy as np
from numpy.ma import masked_array
from cubical.tools import logger

log = logger.getLogger("param_db", 0)
import scipy.interpolate, scipy.spatial
import itertools
from collections import OrderedDict


class _Record(object):
    """
    Helper class: a record is initialized from a dict, and has attributes corresponding to the 
    dict keys.
    """

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


class Parameter(object):
    """
    Defines a parameter object. A parameter represents an N-dimensional set of values (and flags),
    along with axis information.

    A parameter has two types of axes: continuous and interpolatable (e.g. time, frequency), 
    and discrete (e.g. antenna, direction, correlation). Internally, a parameter is stored as 
    a set of masked arrays (e.g. 2D time/frequency arrays), one such "slice" per each discrete 
    point (e.g. per each antenna, direction, correlation, etc.)

    Discrete grids may use sets of string labels (e.g. antenna names) instead of indices.

    Each such slice may be accessed directly with get_slice() (e.g. get_slice(ant='antX',corr1=0,corr2=1))
    The returned array is a reference to the underlying slice, and may be modified. Note that each
    slice can (in principle) be defined on a different subset of the overall continuous grid.

    A parameter also supports reinterpolation onto a different continuous grid. This is slower 
    (and returns a new array). This is done via the reinterpolate() method.
    """

    def __init__(self, name, dtype, axes, interpolation_axes=[], empty=0, metadata=None, grid={}):
        """
        Initialises a Parameter object.

        Args:
            name (str):
                Name of object, e.g. "G".
            dtype (type):
                A numpy data type.
            axes (list):
                Axis names (str).
            interpolation_axes (list, optional): 
                Axes over which interpolation will be supported (1 or 2 axes).
            empty (various, optional):
                Empty value for undefined parameters, usually 0.
            metadata (str or None, optional):
                Parameter metadata.
            grid (dict, optional):
                Dict of grid coordinates, {axis: coordinates}, if known.
                Any particular axis can also be populated by add_chunk() later.
        """
        interpolation_axes = interpolation_axes or []
        assert (len(interpolation_axes) in [0, 1, 2])
        print("defining parameter '{}' over {}".format(name, ",".join(axes)), file=log(1))

        self.name, self.dtype, self.axis_labels = name, dtype, axes
        self.empty, self.metadata = empty, metadata
        # axis index: dict from axis name to axis number
        self.axis_index = {label: i for i, label in enumerate(axes)}
        # convenience member: makes axis numbers available as e.g. "self.ax.time"
        self.ax = _Record(**self.axis_index)
        # list of axis numbers which can be interpolated over (e.g. time/freq)
        self.interpolation_axes = [self.axis_index[axis] for axis in interpolation_axes]
        # list of grid values for each axis, or None if not yet defined
        self.grid = [grid.get(axis) for axis in axes]
        # index of grid values to ordinal numbers (for sparse or labelled axes)
        self.grid_index = {axis: OrderedDict([(x, i) for i, x in enumerate(values)])
                           for axis, values in grid.items() if axis not in interpolation_axes}
        # shape
        self.shape = [0 if g is None else len(g) for g in self.grid]
        # list of sets of grid values actually defined, maintained during prototype->skeleton state
        self._grid_set = [set() for _ in axes]
        # default set of grid values or labels, applicable to sparse axes
        self._default_grid_set = [set(grid.get(axis, [])) for axis in axes]

        # becomes true if parameter is populated
        # A prototype is initially unpopulated; once _update_shape has been called
        # at least once, it becomes populated
        self._populated = False

        # A Parameter object can be in one of three states, or can be transitioning between them.
        # The first two states are internal to the database; the third state is exposed to the
        # user once a database is loaded.
        #
        # * Prototype state (only __init__ has been called) describing the parameter. The grids
        #   can be completely or partially undefined at this stage.
        #
        # * Skeleton state (_update_shape/_finalize_shape has been called). The grids over the
        #   solution space are fully defined at this stage, but no values are loaded.
        #
        # * Fully populated state (_init_arrays/_paste_slice/_finalize_arrays has been called)
        #   containing a complete set of values. This is now a "user-friendly" object on which
        #   get_slice() and reinterpolate() may be called.
        #
        # Once a database is loaded, the user only sees fully populated Parameter objects. The other
        # two states only occur internally, while the database is being populated or loaded.
        #
        # The lifecycle of a Parameter in the context of a database is as follows. Keep in mind
        # that the database is simply a flat file.
        #
        # 1. A new empty database file is created. Parameters are defined (see define_param() below)
        #    one by one, and written to the file in their prototype state.
        #
        # 2. A solver runs, and spits out chunks of the solution space (see add_chunk() below).
        #    This causes _update_shape() to be called. The chunks are written to the file.
        #
        # 3. The solver finishes. _finalize_shape() is called: the Parameter is now a full
        #    skeleton. The skeletons are written to <database>.skel, and the main database file
        #    is closed. It is now what's called a "fragmented" database.
        #
        # To load a fragmented database (see PickledDatabase._load())
        #
        # 1. Skeleton Parameters are read from from <database>.skel. If this does not exist, it can
        #    be re-generated by scanning through the prototypes and slices in the main database file.
        #    _init_arrays() is called on the skeleton Parameteres.
        #
        # 2. The main database is read, and each chunk is passed to the appropriate Parameter's _paste_slice().
        #
        # 3. _finalize_arrays() is called: each Parameter is now fully populated.
        #
        # If any parameter values are changed, PickledDatabase.save() can be called to write the database in
        # "consolidated" mode. In this mode, it's simply a pickle of all fully-populated parameters.

    def _update_shape(self, shape, grid):
        """
        Called repeatedly during the prototype -> skeleton phase, as the 
        solver generates solutions for subsets of the overall parameter space.
        Updates shape of each axis based on the supplied shape and grid. Grid is a dict of 
        {axis: coordinates}, and need only be supplied for shapes that are a partial 
        slice along an axis.

        Args:
            shape ():

            grid ():

        Raises:
            ValueError:
                If grid of axis is inconsistent with previous definition.
            ValueError:
                If axis length is inconsistent with previously defined length.

        """

        self._populated = True

        for i, axis in enumerate(self.axis_labels):
            # if a grid along an axis is supplied, it is potentially a slice
            if axis in grid:
                assert (len(grid[axis]) == shape[i])
                # build up grid as we go along
                self._grid_set[i].update(grid[axis])
                # if full axis grid was predefined by define_param, and it's not
                # an interpolatable one, then supplied grid must be a subset
                if axis not in self.interpolation_axes and self.grid[i] is not None:
                    if set(grid[axis]) - self._grid_set[i]:
                        raise ValueError("grid of axis {} does not match previous definition".format(axis))
            # else it's a full axis -- check that shape conforms.
            else:
                if not self.shape[i]:
                    self.shape[i] = shape[i]
                elif self.shape[i] != shape[i]:
                    raise ValueError("axis {} of length {} does not match previously defined length {}".format(
                        axis, shape[i], self.shape[i]))
                # init grid if not already set
                self._grid_set[i] = self._default_grid_set[i]

    def _finalize_shape(self):
        """
        Finalizes shapes and axes based on accumulated _update_shape() calls.
        This turns the object into a fully fledged skeleton.

        Returns:
            bool:
                True if successful.
        """

        if not self._populated:
            return False

        self.grid_index = {}
        for iaxis, (axis, grid) in enumerate(zip(self.axis_labels, self.grid)):
            # if an actual grid was accumulated via update_shape, and either (a) axis
            # is interpolatable, or (b) no grid was predefined, then we use the accumulated grid
            if self._grid_set[iaxis] and (iaxis in self.interpolation_axes or grid is None):
                self.grid[iaxis] = grid = np.array(sorted(self._grid_set[iaxis]))
                self.shape[iaxis] = len(grid)
            # else use predefined grid, or 0...n-1 if not predefined
            elif grid is None:
                self.grid[iaxis] = grid = np.arange(self.shape[iaxis])
            # build grid index
            self.grid_index[axis] = OrderedDict([(x, i) for i, x in enumerate(grid)])
        # build interpolation grids normalized to [0,1]
        self._norm_grid = {}
        self._norm_grid_map = {}
        self._gminmax = {}
        for iaxis in self.interpolation_axes:
            grid = self.grid[iaxis]
            gmin = grid.min()
            g1 = grid - gmin
            gmax = float(g1.max()) or 1
            self._norm_grid[iaxis] = g1 = g1 / gmax
            self._gminmax[iaxis] = gmin, gmax
        print("dimensions of {} are {}".format(self.name, ','.join(map(str, self.shape))), file=log(0))
        if log.verbosity() > 0:
            for iaxis, axis in enumerate(self.axis_labels):
                if iaxis in self.interpolation_axes:
                    print("  axis {}: {} to {}".format(axis, *self._gminmax[iaxis]), file=log(1))
                else:
                    print("  axis {}: {}".format(axis, " ".join(map(str, self.grid_index[axis].keys()))), file=log(1))
        return True

    def _to_norm(self, iaxis, g):
        """ 
        Converts grid of given axis to normalized grid. 

        Args:
            iaxis ():

            g ():

        Returns:

        """

        gmin, gmax = self._gminmax[iaxis]

        return (g - gmin) / gmax

    def _from_norm(self, iaxis, g):
        """ Converts grid of given axis to unnormalized grid. 

        Args:
            iaxis ():

            g ():

        Returns:

        """

        gmin, gmax = self._gminmax[iaxis]

        return (g * gmax) + gmin

    def _init_arrays(self):
        """
        Initializes internal arrays based on skeleton information. This begins the skeleton -> 
        populated transition.
        """

        # initialize arrays -- all flagged initially, unflagged as slices are pasted in
        self._full_array = masked_array(np.full(self.shape, self.empty, self.dtype),
                                        np.ones(self.shape, bool),
                                        fill_value=self.empty)
        self._array_slices = {}
        print("  loading {}, shape {}".format(self.name, 'x'.join(map(str, self.shape))), file=log(0))

    def _paste_slice(self, item):
        """
        "Pastes" a subset of values into the internal arrays. Called repeatedly during the skeleton 
        -> populated transition.

        Args:
            item ():

        """

        # form up slice operator to "paste" slice into array
        array_slice = []
        for iaxis, axis in enumerate(self.axis_labels):
            if axis in item.grid:
                grid_index = self.grid_index[axis]
                array_slice.append(np.array([grid_index[g] for g in item.grid[axis]]))
            else:
                array_slice.append(np.arange(self.shape[iaxis]))
        self._full_array[np.ix_(*array_slice)] = item.array

    def _finalize_arrays(self):
        """
        Finalizes internal arrays by breaking them into slices. This completes the skeleton -> 
        populated transition.
        """

        interpol_axes = self.interpolation_axes  # list of axis numbers over which we interpolate
        interpol_shape = []  # shape of interpolatable slice
        slicer_axes = []  # list of axis numbers over which we slice
        slicers = []  # list of iterators for each sliced axis
        for i, shape in enumerate(self.shape):
            if i in interpol_axes:
                interpol_shape.append(shape)
                slicers.append((None,))
            else:
                slicer_axes.append(i)
                slicers.append(range(shape))

        self._interpolators = {}

        # get grid over interpolatable axes
        print("decomposing {} into slices".format(self.name), file=log(2))
        # loop over all not-interpolatable slices (e.g. direction, antenna, correlation)
        for slicer in itertools.product(*slicers):
            array_slicer = tuple([slice(None) if sl is None else sl for sl in slicer])
            array = self._full_array[array_slicer]
            flags = array.mask
            # this is a list, per axis, of which subset of the full axis grid the slice is associated with.
            # slice(None) is the full grid (i.e. nothing masked in the array)
            grids = ([self.grid[axis] for axis in interpol_axes],
                     [self._norm_grid[axis] for axis in interpol_axes])
            subset = [slice(None) for _ in interpol_axes]
            if flags is not np.ma.nomask:
                # now, for every axis in the slice, cut out fully flagged points
                allaxis = set(range(array.ndim))
                for iaxis in range(array.ndim):
                    # find points on this axis which are fully flagged along other axes
                    if array.ndim == 1:
                        allflag = flags
                    else:
                        allflag = flags.all(axis=tuple(allaxis - {iaxis}))
                    # all flagged? Indicate this by array=None
                    if allflag.all():
                        print("  slice {} fully flagged".format(slicer), file=log(2))
                        array = None
                        break
                    # if such points exist, extract subset of array and grid
                    elif allflag.any():
                        print("  slice {} flagged at {} {} points".format(slicer, allflag.sum(),
                                                                                    self.axis_labels[
                                                                                        interpol_axes[iaxis]]), file=log(2))
                        # make corresponding slice
                        array_slice = [slice(None)] * array.ndim
                        # also set subset to the mask of the valid points
                        array_slice[iaxis] = subset[iaxis] = ~allflag
                        for gr in grids:
                            gr[iaxis] = gr[iaxis][subset[iaxis]]
                        # apply it to array, flags
                        array = array[tuple(array_slice)]
            # store resulting slice. E.g. for 2 interpolation axes of shape N and M, the
            # attributes are:
            #   array:      shape NxM
            #   grid:       [N-vector, M-vector] grid points
            #   norm_grid:  [N-vector, M-vector] normalized grid points
            #   subset:     [N-vector bool or slice(None), M-vector bool or slice(None)]
            #   gridmap:    [{x: i}, {y: j}] two dicts giving reverse mapping from grid values (of interpolatable axes)
            #               to rows/columns of array
            self._array_slices[tuple(slicer)] = _Record(array=array,
                        grid=grids[0], norm_grid=grids[1], subset=subset,
                        gridmap=[{x: i for i, x in enumerate(grid)} for grid in grids[0]])
        self._full_array = None

    def _get_slicer(self, **axes):
        """  Builds up an index tuple corresponding to keyword arguments that specify a slice. """

        slicer = []
        for iaxis, (axis, n) in enumerate(zip(self.axis_labels, self.shape)):
            if axis in axes:
                value = value0 = axes[axis]
                # if value of axis is in the grid index, look it up
                if axis in self.grid_index and value in self.grid_index[axis]:
                    value = self.grid_index[axis][value]
                # better be an int now
                if not isinstance(value, int):
                    raise ValueError("invalid axis {}={}".format(axis, value0))
                slicer.append(value)
            elif iaxis in self.interpolation_axes:
                slicer.append(None)
            elif n == 1:
                slicer.append(0)
            else:
                raise TypeError("axis {} not specified".format(axis))

        return tuple(slicer)

    def get_slice(self, **axes):
        """
        Returns array and grids associated with given slice, as given by keyword arguments.
        Note that a single index must be specified for all discrete axes with a size of greater 
        than 1. Array may be None, to indicate no solutions for the given slice.
        """

        rec = self._array_slices[self._get_slicer(**axes)]

        return rec.array, rec.grid

    def is_slice_valid(self, **axes):
        """
        Returns True if there are valid solutions for a given slice, as given by keyword arguments.
        Note that a single index must be specified for all discrete axes with a size of greater 
        than 1.
        """

        rec = self._array_slices[self._get_slicer(**axes)]

        return rec.array is not None

    def get_cube(self):
        """
        Returns full cube of solutions (dimensions of all axes), interpolated onto the superset of 
        all slice grids.
        """
        return self.lookup(grid={self.axis_labels[iaxis]: self.grid[iaxis]
                                 for iaxis in self.interpolation_axes})


    def _prepare_interpolation(self, **grid):
        """
        Helper function, to interpolate a named parameter onto the specified grid.

        Args:
            grid (dict): 
                Axes to be returned. For interpolatable axes, grid should be a vector of coordinates
                (the superset of all slice coordinates will be used if an axis is not supplied). 
                Discrete axes may be specified as a single value, or a list of values, or a
                slice object. If any axis is missing, the full axis is returned.

        Returns:
            A tuple of lists:
                - output_shape: shape of output array (before output_reduction is applied)
                - input_slicers: per each axis, an iterable giving the points to be sampled along that axis, or
                       [None] for the interpolatable axes. The net result is that
                       itertools.product(*input_slicers) iterates over all relevant slices
                - output_slicers: corresponding list of iterables giving the index in the output array to which
                       the resampled input slice is to be assigned
                - output_slice_grid: per each interpolatable axis with size>1, grid onto which interpolation is done,
                                     or None if size==1
                - output_reduction:  index applied to output array, to eliminate axes for which only a single element
                        was requested
        """

        output_shape = []
        input_slicers = []
        output_slicers = []
        output_slice_grid = []
        output_reduction = []

        for iaxis, (axis, size) in enumerate(zip(self.axis_labels, self.shape)):
            output_reduction.append(0 if axis in grid and np.isscalar(grid[axis]) else slice(None))
            # interpolatable axis
            if iaxis in self.interpolation_axes:
                if axis in grid:
                    g = np.array(grid[axis]) if np.isscalar(grid[axis]) else grid[axis]
                else:
                    g = self.grid[iaxis]
                output_slice_grid.append(g)
                output_shape.append(len(g))
                input_slicers.append([None])
                output_slicers.append([slice(None)])
            # discrete axis, so return shape is determined by index in **grid, else full axis returned
            else:
                sl = np.arange(size)
                if axis in grid:
                    value = grid[axis]
                    # list of values: looked up in grid index
                    if isinstance(value,(list, tuple, np.ndarray)):
                        sl = [self.grid_index[axis][x] for x in value]
                    # slice: direct index
                    elif type(value) is slice:
                        sl = sl[grid[axis]]
                    # other: look up in grid index
                    else:
                        sl = self.grid_index[axis].get(value, None)
                        if sl is None:
                            raise ValueError("invalid axis value {}={}".format(axis, value))

                if np.isscalar(sl):
                    output_shape.append(1)
                    input_slicers.append([sl])
                    output_slicers.append([0])
                else:
                    sl = np.array(sl)
                    output_shape.append(len(sl))
                    input_slicers.append(sl)
                    output_slicers.append(sl - sl[0])

        return output_shape,\
                input_slicers,\
                output_slicers,\
                output_slice_grid,\
                output_reduction


    def reinterpolate(self, **grid):
        """
        Interpolates named parameter onto the specified grid.

        Args:
            grid (dict): 
                Axes to be returned. For interpolatable axes, grid should be a vector of coordinates
                (the superset of all slice coordinates will be used if an axis is not supplied). 
                Discrete axes may be specified as a single index, or a vector of indices, or a 
                slice object. If any axis is missing, the full axis is returned.

        Returns:
            :obj:`~numpy.ma.core.MaskedArray`:
                Masked array of interpolated values. Mask will indicate values that could not be 
                interpolated. Shape of masked array will correspond to the order axes defined by 
                the parameter, omitting the axes in \*\*grid for which only a single index has been
                specified.
        """

        output_shape,\
        input_slicers,\
        output_slicers,\
        output_slice_grid,\
        output_reduction    = self._prepare_interpolation(**grid)

        # create output array of corresponding shape
        output_array = np.full(output_shape, self.empty, self.dtype)

        print("will interpolate {} solutions onto {} grid".format(self.name,
                                "x".join(map(str, output_shape))), file=log(1))

        # now loop over all slices
        for slicer, out_slicer in zip(itertools.product(*input_slicers), itertools.product(*output_slicers)):
            # arse is the current array slice we work with
            arse = self._array_slices[slicer]
            if arse.array is None:
                print("  slice {} fully flagged".format(slicer), file=log(2))
                output_array[out_slicer] = np.nan
            else:
                # Check which subset of the slice needs to be interpolated
                # We build up the following lists describing the interpolation process
                # (assuming N interpolatable axes, of which M<=N are of size>1, so interpolation
                # is done in M dimensions. For now, M=0,1,2)
                #
                # segment_grid (M): float array of normalized coordinates corresponding
                #                   to (input) array segment being interpolated
                # output_coord (M): float array of normalized (output) interpolation coordinates
                # array_segment_slice (N): index object to extract segment from array. Note that this
                #     will reduce dimensions to M
                # interp_shape (M): shape of interpolation result (M-dim)
                # interp_broadcast (N): index object to broadcast result of interpolation (M-dim)
                #     back to N-dim shape
                # input_grid_segment (M): used to identify the interpolator object in the cache
                segment_grid = []
                output_coord = []
                array_segment_slice = []
                input_grid_segment = []
                interp_shape = []
                interp_broadcast = []
                # build up the two objects above
                for iaxis, outgr, agr, angr in zip(self.interpolation_axes,
                                                   output_slice_grid,
                                                   arse.grid, arse.norm_grid):
                    # interpolatable axis of size >1: process by interpolator
                    if self.grid[iaxis].size > 1:
                        output_coord.append(self._to_norm(iaxis, outgr))
                        # find [i0,i1]: index of envelope of output grid points in the array grid
                        i0, i1 = np.searchsorted(agr, [outgr[0], outgr[-1]])
                        i0, i1 = max(0, i0 - 1), min(len(agr), i1 + 1)
                        segment_grid.append(angr[i0:i1])
                        input_grid_segment.append((iaxis, i0, i1))
                        # extract segment on input to interpolator, do not broadcast back out
                        array_segment_slice.append(slice(i0, i1))
                        interp_shape.append(len(outgr))
                        interp_broadcast.append(slice(None))
                    # size=1: reduce it in the input to interpolator, broadcast back out in the output with newaxis
                    else:
                        array_segment_slice.append(0)
                        interp_broadcast.append(np.newaxis)

                # see if we can reuse an interpolator
                interpolator, input_grid_segment0 = self._interpolators.get(slicer, (None, None))

                # check if the grid segment  of the cached interpolator is a strict superset of the current one: if so,
                # we can reuse the interpolator. Otherwise, create a new one
                if not interpolator or len(input_grid_segment0) != len(input_grid_segment) or \
                        not all([ia == ja and i0 <= j0 and i1 >= j1
                                 for (ia, i0, i1), (ja, j0, j1) in zip(input_grid_segment0, input_grid_segment)]):
                    print("  slice {} preparing {}D interpolator for {}".format(slicer,
                        len(segment_grid), ",".join(["{}:{}".format(*seg[1:]) for seg in input_grid_segment])), file=log(2))
                    # arseg: relevant segment of array slice
                    arseg = arse.array[tuple(array_segment_slice)]
                    # arav: linear array of all values, adata: all unflagged values
                    arav = arseg.ravel()
                    adata = arav.data[~arav.mask] if arav.mask is not np.ma.nomask else arav.data

                    # this is used by the 2D and >2D cases, so make a function
                    def makeLinearNDInterpolator(segment_grid, arav, adata):
                        meshgrids_full = np.array([g.ravel() for g in np.meshgrid(*segment_grid, indexing='ij')]).T
                        meshgrids = meshgrids_full[~arav.mask, :] if arav.mask is not np.ma.nomask else meshgrids_full
                        qhull_options = "Qbb Qc Qz Q12"
                        # edge case of <4 valid points. Delaunay falls over, so artificially duplicate points,
                        # and allow Qhull juggling with the QJ option
                        if len(adata) < 4:
                            adata = np.resize(adata, 4)
                            meshgrids = np.resize(meshgrids, (4, 2))
                            qhull_options += " QJ"
                        # edge case of all points along an axis being on the same line. Allow juggling then,
                        # else Delaunay also falls over
                        elif len(set(meshgrids[:, 0])) < 2 or len(set(meshgrids[:, 1])) < 2:
                            qhull_options += " QJ"
                        triang = scipy.spatial.Delaunay(meshgrids, qhull_options=qhull_options)
                        return scipy.interpolate.LinearNDInterpolator(triang, adata, fill_value=np.nan), meshgrids_full

                    # edge case: no valid data. Make fake interpolator
                    if not len(adata):
                        interpolator = lambda coords: np.full(coords.shape[:-1], np.nan, adata.dtype)
                    # for ndim=0, just return the 0,0 element of array
                    elif not len(segment_grid):
                        interpolator = lambda coords: arse.array
                    # for ndim=1, use 1D interpolator
                    elif len(segment_grid) == 1:
                        agrid = segment_grid[0]
                        if arav.mask is not np.ma.nomask:
                            agrid = agrid[~arav.mask]
                        # handle edge case of 1 valid point, since interp1d() falls over on this
                        if len(adata) == 1:
                            adata = np.array([adata[0], adata[0]])
                            agrid = np.array([agrid[0] - 1e-6, agrid[0] + 1e-6])
                        # make normal interpolator
                        interpolator = scipy.interpolate.interp1d(agrid, adata, bounds_error=False,
                                                                  fill_value="extrapolate")
                    # for 2D, use fill, then interp2d (which extrapolates)
                    elif len(segment_grid) == 2:
                        # if any points inside the slice are masked, inpaint them with LinearNDInterpolator first,
                        # so that we have a complete 2D filled_array
                        if arseg.mask is not np.ma.nomask and arseg.mask.any():
                            ndinp, meshgrids = makeLinearNDInterpolator(segment_grid, arav, adata)
                            # list of coordinates where we're missing values
                            missing_grids = meshgrids[arav.mask, :]
                            # fill missing values using interpolator
                            filled_array = arav.copy()
                            filled_array[arav.mask] = ndinp(missing_grids)
                            filled_array = filled_array.reshape(arseg.shape)
                        else:
                            filled_array = arseg.data
                        # make 2D interpolator
                        interpolator1 = scipy.interpolate.interp2d(segment_grid[0], segment_grid[1],
                                                                  filled_array.T, bounds_error=False)
                        interpolator = lambda *output_coord: interpolator1(*output_coord).T

                    # for >2 dims, use LinearNDInterpolator. Note that this does not extrapolate.
                    elif len(segment_grid) > 2:
                        interpolator1, _ = makeLinearNDInterpolator(segment_grid, arav, adata)
                        interpolator = lambda *output_coord: \
                            interpolator1(np.array([x.ravel() for x in np.meshgrid(*output_coord, indexing='ij')]).T).\
                                reshape(interp_shape)

                    # cache the interpolator
                    self._interpolators[slicer] = interpolator, input_grid_segment

                result = interpolator(*output_coord)

                # # make a meshgrid of output coordinates, and massage into correct shape for interpolator
                # coords = np.array([x.ravel() for x in np.meshgrid(*output_coord, indexing='ij')])
                # # call interpolator. Reshape into output slice shape
                # result = interpolator(coords.T).reshape(interp_shape)
                print("  interpolated onto {} grid".format("x".join(map(str, interp_shape))), file=log(2))
                output_array[out_slicer] = result[tuple(interp_broadcast)]
        # return array, throwing out unneeded axes
        output_array = output_array[tuple(output_reduction)]
        # also, mask missing values from the interpolator with the fill value
        missing = np.isnan(output_array)
        output_array[missing] = self.empty
        print("{} solutions: interpolation results in {}/{} missing values".format(self.name,
                            missing.sum(), missing.size), file=log(1))
        return masked_array(output_array, missing, fill_value=self.empty)


    def lookup(self, **grid):
        """
        Looks up values on the specified grid. The difference with reinterpolate() is that
        the grid values must match exactly. Any missing values will be masked out.

        Args:
            grid (dict): 
                Axes to be returned. For interpolatable axes, grid should be a vector of coordinates
                (the superset of all slice coordinates will be used if an axis is not supplied). 
                Discrete axes may be specified as a single index, or a vector of indices, or a 
                slice object. If any axis is missing, the full axis is returned.

        Returns:
            :obj:`~numpy.ma.core.MaskedArray`:
                Masked array of values. Mask will indicate grid points that didn't match.
        """
        output_shape,\
        input_slicers,\
        output_slicers,\
        output_slice_grid,\
        output_reduction    = self._prepare_interpolation(**grid)

        # create output array of corresponding shape
        output_array = np.full(output_shape, self.empty, self.dtype)
        output_mask  = np.ones(output_shape, bool)

        print("will lookup {} solutions on {} grid".format(self.name,
                                "x".join(map(str, output_shape))), file=log(1))


        # now loop over all slices
        for slicer, out_slicer in zip(itertools.product(*input_slicers), itertools.product(*output_slicers)):
            # arse is the current array slice we work with
            arse = self._array_slices[slicer]
            if arse.array is None:
                print("  slice {} fully flagged".format(slicer), file=log(2))
            else:
                # segment_grid: float array of normalized coordinates corresponding
                #               to segment being interpolated over
                # output_coord: float array of normalized output coordinates
                # array_segment_slice: index object to extract segment from array. Note that this
                #     may also reduce dimensions, if size=1 interpolatable axes are present.
                # input_grid_segment: used to cache the interpolator.
                input_indices = []
                output_indices = []
                # build up the two objects above
                for outgr, gmap in zip(output_slice_grid, arse.gridmap):
                    # for each output grid point, lookup corresponding grid point in the array slice
                    ij = [ (i, gmap.get(x)) for i,x in enumerate(outgr) ]
                    input_indices.append([ j for i,j in ij if j is not None])
                    output_indices.append([ i for i,j in ij if j is not None])
                print("  slice {}: looking up {} valid points".format(slicer,
                        "x".join([str(len(idx)) for idx in input_indices])), file=log(2))

                out = output_array[out_slicer]
                outmask = output_mask[out_slicer]
                ox, ix = np.ix_(*output_indices), np.ix_(*input_indices)
                out[ox] = arse.array[ix]
                if arse.array.mask is np.ma.nomask:
                    out[ox] = False
                else:
                    outmask[ox] = arse.array.mask[ix]

        # return array, throwing out unneeded axes
        output_array = output_array[tuple(output_reduction)]
        output_mask  = output_mask[tuple(output_reduction)]
        output_array[output_mask] = self.empty

        print("{} solutions: no interpolation done on {}/{} missing values".format(self.name,
                            output_mask.sum(), output_mask.size), file=log(1))

        return masked_array(output_array, output_mask, fill_value=self.empty)

    def find_mismatched_grids(self, **grid):
        """
        Looks up the specified grid values and returns list of mismatches.
        Empty list indicates that lookup() can be done on the grids
        (if not empty, then reinterpolate() should be used).

        Args:
            grid (dict): 
                Grid coordinates to look up

        Returns:
            list of (axis, list_of_mismatching_values) tuples
        """
        mismatches = []
        for axis, gridvalues in grid.items():
            iaxis = self.axis_index[axis]
            grid0values = self._grid_set[iaxis] or set(range(self.shape[iaxis]))
            mismatch = set(gridvalues) - grid0values
            if mismatch:
                mismatches.append((axis, sorted(mismatch), iaxis in self.interpolation_axes))

        return mismatches

    def release_cache(self):
        """ 
        Scrubs the object in preparation for saving to a file (e.g. removes cache data etc.)
        """
        self._interpolators = {}
