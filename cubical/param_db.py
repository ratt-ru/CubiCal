"""
Handles parameter databases which can contain solutions and other relevant values. 
"""

import cPickle, os, os.path
import numpy as np
from numpy.ma import masked_array
import traceback
from cubical.tools import logger, ModColor
log = logger.getLogger("param_db", verbose=1)
import scipy.interpolate
import itertools
import time
from collections import OrderedDict, Iterator

class _Record(object):
    """
    Helper class: a record is initialized from a dict, and has attributes corresponding to the 
    dict keys.
    """

    def __init__(self, **kw):

        for key, value in kw.iteritems():
            setattr(self, key, value)

class Parameter(object):
    """
    Defines a parameter object. A parameter represents an N-dimensional set of values (and flags),
    along with axis information.
    
    A parameter has two types of axes: continuous and interpolatable (e.g. time, frequency), 
    and discrete (e.g. antenna, direction, correlation). Internally, a parameter is stored as 
    a set of masked arrays (e.g. 2D time/frequency arrays), one such "slice" per each discrete 
    point (e.g. per each antenna, direction, correlation, etc.) 
    
    Each such slice may be accessed directly with get_slice() (e.g. get_slice(ant=N,corr1=0,corr2=1)) 
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

        assert (len(interpolation_axes) in [1, 2])
        print>> log(1), "defining parameter '{}' over {}".format(name, ",".join(axes))

        self.name, self.dtype, self.axis_labels = name, dtype, axes
        self.empty, self.metadata = empty, metadata
        # axis index: dict from axis name to axis number
        self.axis_index = {label: i for i, label in enumerate(axes)}
        # convenience member: makes axis numbers available as e.g. "self.ax.time"
        self.ax = _Record(**self.axis_index)
        # list of axis numbers for flags
        self.interpolation_axes = [self.axis_index[axis] for axis in interpolation_axes]
        # list of grid values for each axis, or None if not yet defined
        self.grid = [grid.get(axis) for axis in axes]
        # shape
        self.shape = [0 if g is None else len(g) for g in self.grid]
        # list of sets of grid values, maintained during prorotype->skeleton state
        self._grid_set = [set(grid.get(axis, [])) for axis in axes]
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
                assert(len(grid[axis]) == shape[i])
                # if axis grid was predefined by define_param, then supplied grid must be a subset
                if self.grid[i] is not None:
                    if set(grid[axis]) - self._grid_set[i]:
                        raise ValueError("grid of axis {} does not match previous definition".format(axis))
                # else build up grid as we go along
                else:
                    self._grid_set[i].update(grid[axis])
            # else it's a full axis -- check that shape conforms.
            elif not self.shape[i]:
                self.shape[i] = shape[i]
            elif self.shape[i] != shape[i]:
                raise ValueError,"axis {} of length {} does not match previously defined length {}".format(
                    axis, shape[i], self.shape[i])

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

        self.grid_index = []
        for iaxis, (axis, grid) in enumerate(zip(self.axis_labels, self.grid)):
            # if grid wasn't set by define_param, go and reprocess it based on the accumulated set
            if grid is None:
                # if slices were accumulated, set grid from union of all slices' grids
                if self._grid_set[iaxis]:
                    self.grid[iaxis] = grid = np.array(sorted(self._grid_set[iaxis]))
                    self.shape[iaxis] = len(grid)
                # else no slices: simply set default grid based on accumulated shape
                else:
                    self.grid[iaxis] = grid = np.arange(self.shape[iaxis])
            # build grid index
            self.grid_index.append({x: i for i, x in enumerate(grid)})
        # build interpolation grids normalized to [0,1]
        self._norm_grid = {}
        self._gminmax   = {}
        for iaxis in self.interpolation_axes:
            grid = self.grid[iaxis]
            gmin = grid.min()
            g1 = grid - gmin
            gmax = float(g1.max()) or 1
            self._norm_grid[iaxis] = g1/gmax
            self._gminmax[iaxis] = gmin, gmax
        self._grid_set = None
        print>>log(0), "dimensions of {} are {}".format(self.name, ','.join(map(str, self.shape)))
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
        
        return (g-gmin)/gmax

    def _from_norm(self, iaxis, g):
        """ Converts grid of given axis to unnormalized grid. 

        Args:
            iaxis ():
                
            g ():

        Returns:
            
        """

        gmin, gmax = self._gminmax[iaxis]

        return (g*gmax) + gmin

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
        print>> log(0), "  loading {}, shape {}".format(self.name, 'x'.join(map(str, self.shape)))

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
                grid_index = self.grid_index[iaxis]
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
                slicers.append(xrange(shape))

        # list of grids for interpolatable axes
        interpol_grid0 = [self._norm_grid[axis] for axis in interpol_axes]
        self._interpolators = {}

        # get grid over interpolatable axes
        print>> log(2), "decomposing {} into slices".format(self.name)
        # loop over all not-interpolatable slices (e.g. direction, antenna, correlation)
        for slicer in itertools.product(*slicers):
            array_slicer = [slice(None) if sl is None else sl for sl in slicer]
            array = self._full_array[array_slicer]
            flags = array.mask
            interpol_grid = list(interpol_grid0)
            if flags is not np.ma.nomask:
                # now, for every axis in the slice, cut out fully flagged points
                allaxis = set(xrange(array.ndim))
                for iaxis in xrange(array.ndim):
                    # find points on this axis which are fully flagged along other axes
                    if array.ndim == 1:
                        allflag = flags
                    else:
                        allflag = flags.all(axis=tuple(allaxis - {iaxis}))
                    # all flagged? Indicate this by array=None
                    if allflag.all():
                        print>> log(2), "  slice {} fully flagged".format(slicer)
                        array = None
                        break
                    # if such points exist, extract subset of array and grid
                    elif allflag.any():
                        print>> log(2), "  slice {} flagged at {} {} points".format(slicer, allflag.sum(),
                                                                                    self.axis_labels[interpol_axes[iaxis]])
                        # make corresponding slice
                        array_slice = [slice(None)] * array.ndim
                        array_slice[iaxis] = ~allflag
                        # apply it to array, flags and grids
                        array = array[tuple(array_slice)]
                        interpol_grid[iaxis] = interpol_grid[iaxis][~allflag]
            # store resulting slice
            self._array_slices[tuple(slicer)] = array, interpol_grid
        self._full_array = None

    def _get_slicer(self, **axes):
        """  Builds up an index tuple corresponding to keyword arguments that specify a slice. """
        
        slicer = []
        for iaxis, (axis, n) in enumerate(zip(self.axis_labels, self.shape)):
            if axis in axes:
                if not isinstance(axes[axis], int):
                    raise TypeError("invalid axis {}={}".format(axis, axes[axis]))
                slicer.append(axes[axis])
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

        array, grids = self._array_slices[self._get_slicer(**axes)]

        return array, [self._from_norm(self.interpolation_axes[i], g) for i, g in enumerate(grids)]

    def is_slice_valid(self, **axes):
        """
        Returns True if there are valid solutions for a given slice, as given by keyword arguments.
        Note that a single index must be specified for all discrete axes with a size of greater 
        than 1.
        """

        array, _ = self._array_slices[self._get_slicer(**axes)]

        return array is not None

    def get_cube(self):
        """
        Returns full cube of solutions (dimensions of all axes), interpolated onto the superset of 
        all slice grids.
        """

        return self.reinterpolate(grid={self.axis_labels[iaxis]: self.grid[iaxis] for iaxis in 
                                                                        self.interpolation_axes})

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

        # make output grid and output array
        # ok this is hugely complicated so as to cover all cases, so let's document carefully
        output_shape = []            # shape of output array (before output_reduction is applied)
        input_slicers = []           # per each axis, an iterable giving the points to be sampled along that axis, or
                                     #     [None] for the interpolatable axes. The net result is that
                                     #     itertools.product(*input_slicers) iterates over all relevant slices
        output_slicers = []          # corresponding list of iterables giving the index in the output array to which
                                     #     the resampled input slice is assigned
        input_grid_segment = []      # per each interpolatable axis, (i0,i1) gives the indices of the relevant segment
                                     #     (i.e. the envelope of the requested grid)
        input_slice_reduction = []   # applied to each input slice to reduce size=1 axes
        input_slice_broadcast = []   # applied to each interpolation result to broadcast back size=1 axes
        output_slice_grid = []       # per each interpolatablae axis with size>1, grid onto which interpolation is done
        output_reduction = []        # index applied to output array, to eliminate axes for which only a single
                                     # element was requested
        # build up list of (normalized) output slice grids
        for iaxis, (axis, size) in enumerate(zip(self.axis_labels, self.shape)):
            output_reduction.append(0 if axis in grid and np.isscalar(grid[axis]) else slice(None))
            if iaxis in self.interpolation_axes:
                g = g0 = self._norm_grid[iaxis]  # full normalized axis
                # if a different grid for the axis is specified, find segment of full axis that needs to be interpolated
                if axis in grid:
                    if np.isscalar(grid[axis]):
                        g = self._to_norm(iaxis, np.array(grid[axis]))
                    else:
                        g = self._to_norm(iaxis, grid[axis])
                    i0, i1 = np.searchsorted(g0, [g[0], g[-1]])
                    # this slice of the full grid incorporates the requested interpolation points
                    i0, i1 = max(0, i0 - 1), min(len(g0), i1 + 1)
                else:
                    i0, i1 = 0, len(g0)
                output_shape.append(len(g))
                input_slicers.append([None])
                output_slicers.append([slice(None)])
                # case A: interpolatable axis with >1 points: prepare for proper interpolation
                if size > 1:
                    output_slice_grid.append(g)
                    input_grid_segment.append((i0, i1))
                    input_slice_reduction.append(slice(None))
                    input_slice_broadcast.append(slice(None))
                # case B: interpolatable axis with 1 point: will need to be collapsed in input to interpolator,
                # and broadcast back out
                else:
                    input_grid_segment.append((0,1))
                    input_slice_reduction.append(0)
                    input_slice_broadcast.append(np.newaxis)
            # case C: discrete axis, so return shape is determined by index in **grid, else full axis returned
            else:
                sl = np.arange(size)
                if axis in grid:
                    sl = sl[grid[axis]]
                if np.isscalar(sl):
                    output_shape.append(1)
                    input_slicers.append([sl])
                    output_slicers.append([0])
                else:
                    output_shape.append(len(sl))
                    input_slicers.append(sl)
                    output_slicers.append(sl - sl[0])
        # create output array of corresponding shape
        output_array = np.full(output_shape, self.empty, self.dtype)

        print>> log(1), "will interpolate {} solutions onto {} grid".format(self.name,
                                                                            "x".join(map(str, output_shape)))

        # now loop over all slices
        for slicer, out_slicer in zip(itertools.product(*input_slicers), itertools.product(*output_slicers)):
            array, slice_grid = self._array_slices[slicer]
            if array is None:
                print>> log(2), "  slice {} fully flagged".format(slicer)
            else:
                # see if we can reuse an interpolator
                interpolator, input_grid_segment0 = self._interpolators.get(slicer, (None, None))
                # check if the grid segment  of the cached interpolator is a strict superset of the current one: if so, reuse
                if not interpolator or not all(
                        [i0 <= j0 and i1 >= j1 for (i0, i1), (j0, j1) in zip(input_grid_segment0, input_grid_segment)]):
                    # segment_grid: coordinates corresponding to segment being interpolated over
                    # array_segment_slice: index object to extract segment from array. Note that this
                    #  may also reduce dimensions, if size=1 interpolatable axes are present
                    segment_grid = []
                    array_segment_slice = []
                    # build up the two objects above
                    for sg, (i0, i1), isr in zip(slice_grid, input_grid_segment, input_slice_reduction):
                        if isr is not 0:
                            segment_grid.append(sg[i0:i1])
                            array_segment_slice.append(slice(i0, i1))
                        else:
                            array_segment_slice.append(0)
                    print>> log(2), "  slice {} preparing {}D interpolator for {}".format(slicer,
                                        len(segment_grid),
                                        ",".join(["{}:{}".format(*seg) for seg in input_grid_segment]))
                    # make a meshgrid of all points
                    arav = array[array_segment_slice].ravel()
                    # for ndim=0, just return the 0,0 element of array
                    if not len(segment_grid):
                        interpolator = lambda coords:array[tuple(input_slice_reduction)]
                    # for ndim=1, use interp1d...
                    elif len(segment_grid) == 1:
                        if arav.mask is np.ma.nomask:
                            interpolator = scipy.interpolate.interp1d(segment_grid[0], arav.data,
                                                            bounds_error=False, fill_value=np.nan)
                        else:
                            interpolator = scipy.interpolate.interp1d(segment_grid[0][~arav.mask], arav.data[~arav.mask],
                                                            bounds_error=False, fill_value=np.nan)
                    # ...because LinearNDInterpolator works for ndim>1 only
                    else:
                        meshgrids = np.array([g.ravel() for g in np.meshgrid(*segment_grid, 
                                                                                indexing='ij')]).T
                        if arav.mask is np.ma.nomask:
                            interpolator = scipy.interpolate.LinearNDInterpolator(meshgrids, 
                                                                    arav.data, fill_value=np.nan)
                        else:
                            interpolator = scipy.interpolate.LinearNDInterpolator(
                                meshgrids[~arav.mask, :], arav.data[~arav.mask], fill_value=np.nan)
                        self._interpolators[slicer] = interpolator, input_grid_segment
                # make a meshgrid of output and massage into correct shape for interpolator
                coords = np.array([x.ravel() for x in np.meshgrid(*output_slice_grid, indexing='ij')])
                result = interpolator(coords.T).reshape([len(x) for x in output_slice_grid])
                print>> log(2), "  interpolated onto {} grid".format(
                    "x".join([str(len(x)) for x in output_slice_grid]))
                output_array[out_slicer] = result[input_slice_broadcast]
        # return array, throwing out unneeded axes
        output_array = output_array[output_reduction]

        return masked_array(output_array, np.isnan(output_array), fill_value=self.empty)

    def _scrub(self):
        """ 
        Scrubs the object in preparation for saving to a file (e.g. removes cache data etc.)
        """

        self._interpolators = {}

class _ParmSegment(object):
    """ A subset of the solution space of a parameter. """

    def __init__(self, name, array, grid):
        self.name, self.array, self.grid = name, array, grid

def create(filename, metadata={}, backup=True):
    """
    Creates a new parameter database.
    
    Args:
        filename (str): 
            Name of file to save DB to.
        metadata (dict, optional): 
            Optional dictionary of metadata.
        backup (bool, optional):
            If True, and an old database with the same filename exists, make a backup.

    Returns:
        :obj:`~cubical.param_db.PickledDatabase`:
            A resulting parameter database.
    """

    db = PickledDatabase()
    db._create(filename, metadata, backup)
    
    return db

def load(filename):
    """
    Loads a parameter database

    Args:
        filename (str): 
            Name of file to load DB from.

    Returns:
        :obj:`~cubical.param_db.PickledDatabase`:
            A resulting parameter database.
    """

    db = PickledDatabase()
    db._load(filename)
    
    return db

class PickledDatabase(object):
    """
    This class implements a simple parameter database saved to a pickle.
    """

    def __init__(self):
        self._fobj = None

    def _create(self, filename, metadata={}, backup=True):
        """
        Creates a parameter database given by the filename and opens it in "create" mode.
        
        Args:
            filename (str): 
                Name of database.
            metadata (dict, optional): 
                Optional metadata to be stored in DB.
            backup (bool, optional):
                If True, and an old database with the same filename exists, make a backup.
        """

        self.mode = "create"
        self.filename = filename
        self.do_backup = backup
        self.metadata = OrderedDict(mode=self.MODE_FRAGMENTED, time=time.time(), **metadata)
        # we'll write to a temp file, and do a backup on successful closure
        self._fobj = open(filename+".tmp", 'w')
        cPickle.dump(self.metadata, self._fobj)
        self._fobj.flush()
        self._parameters = {}
        self._parm_written = set()
        print>>log(0),"creating {} in {} mode".format(self.filename, self.metadata['mode'])

    def define_param(self, *args, **kw):
        """
        Defines a parameter. Only valid in "create" mode.
        
        Args:
            args (tuple):
                Positional arguments.
            kw (dict):
                Keyword arguments.
        """

        assert(self.mode is "create")
        parm = Parameter(*args,**kw)
        self._parameters[parm.name] = parm
        # we don't write it to DB yet -- write it in add_chunk() rather
        # this makes it easier to deal with I/O workers (all IO is done by one process)

    def add_chunk(self, name, array, grid={}):
        """
        Adds a slice of values for a parameter.

        Args:
            name (str):
                The parameter name e.g. "G".
            array (:obj:`~numpy.ma.core.MaskedArray`):
                The values which are to be added.
            grid (dict, optional):
                Grid coordinates for each sliced parameter axis. 

        """

        assert(self.mode is "create")
        parm = self._parameters.get(name)
        assert(parm is not None)
        # dump parm to DB the first time a slice shows up
        if name not in self._parm_written:
            cPickle.dump(parm, self._fobj, 2)
            self._parm_written.add(name)
        # update axis shapes and grids based on slice
        parm._update_shape(array.shape, grid)
        # dump slice to DB
        item = _ParmSegment(name=name, array=np.ma.asarray(array), grid=grid)
        cPickle.dump(item, self._fobj, 2)

    def close(self):
        """ Closes the database. """

        if self._fobj:
            self._fobj.close()
            self._fobj = None
        # in create mode, update the descriptions file
        if self.mode is "create":
            self._save_desc()
            self._backup_and_rename(self.do_backup)
        self.mode = "closed"

    def _save_desc(self):
        """ Helper function. Writes accumulated parameter descriptions to filename.desc. """
        
        for desc in self._parameters.itervalues():
            desc._finalize_shape()
        for key in self._parameters.keys():
            if not self._parameters[key]._populated:
                del self._parameters[key]
        cPickle.dump(self._parameters, open(self.filename+".skel", 'w'), 2)
        print>>log(0),"saved updated parameter skeletons to {}".format(self.filename+".skel")

    def _backup_and_rename (self, backup):
        """
        May create a backup and renames the temporary DB.

        Args:
            backup (bool):
                If True, creates a backup, otherwise just renames.
        """
        if os.path.exists(self.filename):
            if backup:
                backup_filename = os.path.join(os.path.dirname(self.filename),
                                               "~" + os.path.basename(self.filename))
                print>> log(0), "previous DB will be backed up as " + backup_filename
                if os.path.exists(backup_filename):
                    print>> log(0), "  removing old backup " + backup_filename
                    os.unlink(backup_filename)
                os.rename(self.filename, backup_filename)
            else:
                os.unlink(self.filename)
        os.rename(self.filename + ".tmp", self.filename)
        print>>log(0),"wrote {} in {} mode".format(self.filename, self.metadata['mode'])

    def save(self, filename=None, backup=True):
        """
        Save the database.

        Args:
            filename (str, optional):
                Name of output file.
            backup (bool, optional):
                If True, create a backup.
        """
        assert(self.mode is "load")
        self.metadata['mode'] = self.MODE_CONSOLIDATED
        filename = filename or self.filename
        with open(filename+".tmp", 'w') as fobj:
            cPickle.dump(self.metadata, fobj, 2)
            for parm in self._parameters.itervalues():
                parm._scrub()
            cPickle.dump(self._parameters, fobj, 2)
        # successfully written? Backup and rename
        self.filename = filename
        self._backup_and_rename(backup)


    MODE_FRAGMENTED = "fragmented"
    MODE_CONSOLIDATED = "consolidated"

    class _Unpickler(Iterator):
        def __init__(self, filename):
            self.fobj = open(filename)
            self.metadata = cPickle.load(self.fobj)
            if type(self.metadata) is not OrderedDict or not "mode" in self.metadata:
                raise IOError("{}: invalid metadata entry".format(filename))
            self.mode = self.metadata['mode']

        def next(self):
            try:
                return cPickle.load(self.fobj)
            except EOFError:
                raise StopIteration

    def _load(self, filename):
        """
        Loads database from file. This will create arrays corresponding to the stored parameter
        shapes.

        Args:
            filename (str):
                Name of file to load.
        """

        self.mode = "load"
        self.filename = filename

        db = self._Unpickler(filename)
        print>>log(0),"reading {} in {} mode".format(self.filename, db.mode)
        self.metadata = db.metadata
        for key,value in self.metadata.iteritems():
            if key != "mode":
                print>>log(1),"  metadata '{}': {}".format(key,value)

        # now load differently depending on mode
        # in consolidated mode, just unpickle the parameter objects
        if db.mode == PickledDatabase.MODE_CONSOLIDATED:
            self._parameters = db.next()
            for parm in self._parameters.itervalues():
                print>>log(1),"  read {} of shape {}".format(parm.name,
                                    'x'.join(map(str, parm.shape)))
            return

        # otherwise we're in fragmented mode
        if db.mode != PickledDatabase.MODE_FRAGMENTED:
            raise IOError("{}: invalid mode".format(self.filename, self.metadata.mode))

        # in fragmented mode? try to read the desc file
        descfile = filename + '.skel'
        self._parameters = None
        if not os.path.exists(descfile):
            print>>log(0),ModColor.Str("{} does not exist, will try to rebuild".format(descfile))
        elif os.path.getmtime(descfile) < os.path.getmtime(self.filename):
            print>>log(0),ModColor.Str("{} older than database: will try to rebuild".format(descfile))
        elif os.path.getmtime(descfile) < os.path.getmtime(__file__):
            print>>log(0),ModColor.Str("{} older than this code: will try to rebuild".format(descfile))
        else:
            try:
                self._parameters = cPickle.load(open(descfile, 'r'))
            except:
                traceback.print_exc()
                print>> log(0), ModColor.Str("error loading {}, will try to rebuild".format(descfile))
        # rebuild the skeletons, if they weren't loaded
        if self._parameters is None:
            self._parameters = {}
            for item in db:
                if isinstance(item, Parameter):
                    self._parameters[item.name] = item
                elif isinstance(item, _ParmSegment):
                    self._parameters[item.name]._update_shape(item.array.shape, item.grid)
                else:
                    raise IOError("{}: unexpected entry of type '{}'".format(self.filename, type(item)))
            self._save_desc()

        # initialize arrays
        for parm in self._parameters.itervalues():
            parm._init_arrays()

        # go over all slices to paste them into the arrays
        db = self._Unpickler(filename)
        for item in db:
            if type(item) is Parameter:
                pass
            elif type(item) is _ParmSegment:
                parm = self._parameters.get(item.name)
                if parm is None:
                    raise IOError, "{}: no parm found for {}'".format(filename, item.name)
                parm._paste_slice(item)
            else:
                raise IOError("{}: unknown item type '{}'".format(filename, type(item)))

        # ok, now arrays and flags each contain a full-sized array. Break it up into slices.
        for parm in self._parameters.itervalues():
            parm._finalize_arrays()

    def names(self):
        """ Returns names of all defined parameters. """
        
        return self._parameters.keys()

    def __contains__(self, name):
        return name in self._parameters

    def __getitem__(self, name):
        return self.get(name)

    def get(self, name):
        """
        Gets Parameter object associated with the named parameter.

        Args:
            name (str):
                Name of parameter.

        Returns:
            :obj:`~cubical.param_db.Parameter`:
                The requested Parameter object.
        """

        return self._parameters[name]


if __name__ == "__main__":
    log.verbosity(2)
    print "Creating test DB"
    db = create("test.db")
    db.define_param("G", np.float64,
                    ["ant", "time", "freq", "corr"], interpolation_axes=["time", "freq"])
    db.define_param("B", np.float64,
                    ["ant", "time", "freq", "corr"], interpolation_axes=["time", "freq"])
    for i0,i1 in (0,2),(4,6),(7,9):
        arr = np.full((3,i1-i0,1,2), i0, float)
        db.add_chunk("G", arr, grid=dict(time=np.arange(i0,i1)))
        arr = np.full((3,1,i1-i0,2), i0, float)
        db.add_chunk("B", arr, grid=dict(freq=np.arange(i0,i1)))
    db.close()

    print "Loading test DB"
    db = load("test.db")
    print db.names()
    G = db['G']
    B = db['B']
    print "G", db["G"].axis_labels, db["G"].shape
    print "B", db["B"].axis_labels, db["B"].shape
    print "G", G.get_slice(ant=0,corr=0)
    print "B", G.get_slice(ant=0,corr=0)
    print "Gint", G.reinterpolate(time=np.arange(0,10,.5),freq=np.arange(0,10,1.5))

