import cPickle, os, os.path
import numpy as np
import traceback
from cubical.tools import logger, ModColor
log = logger.getLogger("param_db", verbose=2)
from collections import OrderedDict
import scipy.interpolate

class SimpleParameterDB(object):
    """
    This class implements a simple parameter database
    """
    @staticmethod
    def create(filename, metadata=None, backup=True):
        db = SimpleParameterDB()
        db._create(filename, metadata, backup)
        return db

    @staticmethod
    def load(filename):
        db = SimpleParameterDB()
        db._load(filename)
        return db

    def _create(self, filename, metadata=None, backup=True):
        """
        Creates a parameter database given by the filename, opens it in "create" mode
        
        Args:
            filename: name of database
            metadata: optional metadata to be stored in DB
        """
        self.mode = "create"
        self.filename = filename
        self.do_backup = backup
        # we'll write to a temp file, and do a backup on successful closure
        self._fobj = open(filename+".tmp", 'w')
        self._parmdescs = {}
        self._parmdims = {}

    def define_param(self, name, dtype, axis_labels, empty=0, metadata=None, grids=None):
        """
        Defines a parameter. Only valid in "create" mode.
        
        Args:
            name:           name, e.g. "G"
            dtype:          numpy dtype
            axis_labels:    axis names
            empty:          empty value for undefined parameters, usually 0
            metadata:       optional parameter metadata
            grids:          optional dict of grid coordinates. Can also be populated by add_slice() later.
        """
        assert(self.mode is "create")
        # axis index: dict from axis name to axis number
        axis_index = {label: i for i, label in enumerate(axis_labels)}
        print>>log(1), "defining parameter '{}' over {}".format(name,",".join(axis_labels))
        parmdesc = dict(entry="parmdesc", name=name, dtype=dtype,
                        grids=grids or {}, axis_labels=axis_labels,
                        shape=[0]*len(axis_labels),  # shape starts empty and gets accumulated
                        axis_index=axis_index, empty=empty, metadata=metadata)
        self._parmdescs[name] = parmdesc
        # we don't write it to DB yet -- write it in add_slice() rather
        # this makes it easier to deal with I/O workers (all IO is done by one process)

    def _update_parm_shape(self, parmdesc, shape, grids):
        """Internal method. Updates shape of each parameter axis based on the supplied shape and slice info"""
        dims = parmdesc['shape']
        for i, axis in enumerate(parmdesc['axis_labels']):
            if grids and axis in grids:
                parmdesc['grids'].setdefault(axis, set()).update(grids[axis])
            else:
                if dims[i] == 0:
                    dims[i] = shape[i]
                elif dims[i] != shape[i]:
                    raise ValueError,"axis {[i]}({}) of length {} does not match previously defined shape {}".format(
                        i, axis, shape[i], dims[i])

    def add_slice(self, name, array, grids):
        """
        Adds a slice of values for a parameter
        
        Args:
            name:       parameter name e.g. "G"
            array:      array
            slices:     dict of slices into each axis, e.g. {'time':slice(0,100)} defines
                        an array corresponding to the first 100 timeslots
            grids:      dict of grid coordinates for each parameter axis 
        """
        assert(self.mode is "create")
        parmdesc = self._parmdescs.get(name)
        assert(parmdesc is not None)
        self._update_parm_shape(parmdesc, array.shape, grids)
        # dump parmdesc to DB the first time a slice shows up
        if not parmdesc.get('written'):
            cPickle.dump(parmdesc, self._fobj, 2)
            parmdesc['written'] = True
        # dump to DB
        item = dict(entry="slice", name=name, array=array, grids=grids)
        cPickle.dump(item, self._fobj, 2)

    def close(self):
        """
        Closes the database
        """
        if self._fobj:
            self._fobj.close()
            self._fobj = None
        # in create mode, update the descriptions file
        if self.mode is "create":
            self._save_desc()
            if os.path.exists(self.filename):
                if self.do_backup:
                    backup_filename = os.path.join(os.path.dirname(self.filename),
                                                  "~" + os.path.basename(self.filename))
                    print>>log(0),"previous DB will be backed up as "+backup_filename
                    if os.path.exists(backup_filename):
                        print>> log(0), "  removing old backup " + backup_filename
                        os.unlink(backup_filename)
                    os.rename(self.filename, backup_filename)
                else:
                    os.unlink(self.filename)
            os.rename(self.filename+".tmp", self.filename)
        self.mode = "closed"

    def _save_desc(self):
        """Helper function. Writes accumulated parameter descriptions to filename.desc"""
        for name, desc in self._parmdescs.iteritems():
            desc['grid_index'] = {}
            for iaxis, axis in enumerate(desc['axis_labels']):
                if axis in desc['grids']:
                    desc['grids'][axis] = grid = sorted(desc['grids'][axis])
                    desc['grid_index'][axis] = { x: i for i, x in enumerate(grid)}
                    desc['shape'][desc['axis_index'][axis]] = len(grid)
                else:
                    desc['grids'][axis] = np.arange(desc['shape'][iaxis])
            print>>log(0), "dimensions of {} are {}".format(name, ','.join(map(str, desc['shape'])))
        cPickle.dump(self._parmdescs, open(self.filename+".desc", 'w'), 2)
        print>>log(0),"saved updated parameter descriptions to {}".format(self.filename+".desc")

    def reload(self):
        """
        Closes and reloads the database
        """
        self.close()
        self.load(self.filename)

    def _load(self, filename):
        """
        Loads database from file. This will create arrays corresponding to the stored parameter shapes.
        """
        self.mode = "load"
        self.filename = filename
        self._fobj = None
        self._arrays = {}
        self._interpolators = {}
        # try to read the desc file
        descfile = filename + '.desc'
        try:
            self._parmdescs = cPickle.load(open(descfile, 'r'))
        except:
            traceback.print_exc()
            print>> log(0), ModColor.Str("error loading dimensions from {} (see above), will try to re-generate".format(descfile))
            self._parmdescs = {}
            with open(filename) as fobj:
                while True:
                    try:
                        item = cPickle.load(fobj)
                    except EOFError:
                        break
                    itemtype = item['entry']
                    name = item['name']
                    if itemtype == "parmdesc":
                        self._parmdescs[name] = item
                    elif itemtype == "slice":
                        self._update_parm_shape(self._parmdescs[name], item['array'].shape, item['grids'])
            self._save_desc()

        # initialize arrays
        for name, desc in self._parmdescs.iteritems():
            self._arrays[name] = np.full(desc['shape'], desc['empty'], desc['dtype'])
        print>> log(0), "initializing '{}' of shape {}".format(name, ','.join(map(str, desc['shape'])))

        with open(filename) as fobj:
            while True:
                try:
                    item = cPickle.load(fobj)
                except EOFError:
                    break
                itemtype = item['entry']
                name = item['name']
                if itemtype == "slice":
                    array = self._arrays.get(name)
                    desc = self._parmdescs.get(name)
                    if name is None or desc is None:
                        raise IOError, "{}: no parmdesc found for {}'".format(filename, name)
                    # form up slice operator to "paste" slice into array
                    total_slice = []
                    for iaxis, axis in enumerate(desc['axis_labels']):
                        if axis in item['grids']:
                            grid_index = desc['grid_index'][axis]
                            total_slice.append(np.array([grid_index[g] for g in item['grids'][axis]]))
                        else:
                            total_slice.append(np.arange(desc['shape'][iaxis]))
                    array[np.ix_(*total_slice)] = item['array']
                elif itemtype != "parmdesc":
                    raise IOError("{}: unknown item type '{}'".format(filename, itemtype))

    def names(self):
        """
        Returns names of all defined parameters
        """
        return self._parmdescs.keys()

    def __contains__(self, name):
        return name in self._parmdescs

    def get(self, name):
        """
        Returns array associated with the named parameter
        """
        assert(self.mode == "load")
        return self._arrays[name]

    def get_desc(self, name):
        """
        Returns description associated with the named parameter
        """
        return self._parmdescs[name]

    def reinterpolate (self, name, grids, fill_value=1):
        """
        Interpolates named parameter onto the specified grid
        
        Args:
            name: 
            grids: dict of axes to be interpolated 

        Returns:
            Array of interpolated values

        """
        array = self._arrays[name]
        desc = self._parmdescs[name]
        axes = desc['axis_labels']
        input_grids = desc['grids']
        interpolator = self._interpolators.get(name)
        if interpolator is None:
            # make list of grid coordinates per axis
            input_coords = [ input_grids[ax] for ax in axes ]
            interpolator = self._interpolators[name] = \
                scipy.interpolate.RegularGridInterpolator(input_coords, array,
                         method='linear', bounds_error=False, fill_value=fill_value)
            print>>log(0),"preparing to interpolate {} from {} grid".format(name,
                            "x".join([ str(len(x)) for x in input_coords]))
        # make list of output coordinates
        output_coords = [ grids[ax] if ax in grids else input_grids[ax] for ax in axes ]
        # make a meshgrid, massage into correct shape for interpolator
        coords = np.array([x.ravel() for x in np.meshgrid(*output_coords)])
        result = interpolator(coords.T).reshape([len(x) for x in output_coords])
        print>>log(0),"  interpolated onto {} grid".format(
                        "x".join([ str(len(x)) for x in output_coords]))
        return result


if __name__ == "__main__":
    print "Creating test DB"
    db = SimpleParameterDB.create("test.db")
    db.define_param("G", np.int32,
                    ["ant", "time", "freq", "corr"])
    db.define_param("B", np.int32,
                    ["ant", "time", "freq", "corr"])
    for i0,i1 in (0,2),(4,6),(7,9):
        arr = np.full((3,i1-i0,1,2), i0, np.int32)
        db.add_slice("G", arr, dict(time=slice(i0, i1)), dict(time=np.arange(0,10)*.1))
        arr = np.full((3,1,i1-i0,2), i0, np.int32)
        db.add_slice("B", arr, dict(freq=slice(i0, i1)), dict(freq=np.arange(0,10)*.1))
    db.close()

    print "Loading test DB"
    db = SimpleParameterDB.load("test.db")
    print db.names()
    print "G", db.get("G")
    print "B", db.get("B")
    print "G", db.get_desc("G")
    print "B", db.get_desc("B")
