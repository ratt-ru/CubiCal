import cPickle
import numpy as np
import traceback
from cubical.tools import logger, ModColor
log = logger.getLogger("flagging")
from collections import OrderedDict


class SimpleParameterDB(object):
    """
    This class implements a simple parameter database
    """
    @staticmethod
    def create(filename, metadata=None):
        db = SimpleParameterDB()
        db._create(filename, metadata)
        return db

    @staticmethod
    def load(filename):
        db = SimpleParameterDB()
        db._load(filename)
        return db


    def _create(self, filename, metadata=None):
        """
        Creates a parameter database given by the filename, opens it in "create" mode
        
        Args:
            filename: name of database
            metadata: optional metadata to be stored in DB
        """
        self.mode = "create"
        self.filename = filename
        self._fobj = open(filename, 'w')
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
        cPickle.dump(parmdesc, self._fobj, 2)

    def _update_parm_shape(self, parmdesc, shape, slices, grids):
        """Internal method. Updates shape of each parameter axis based on the supplied shape and slice info"""
        dims = parmdesc['shape']
        for i, axis in enumerate(parmdesc['axis_labels']):
            if axis in slices:
                dims[i] = max(dims[i], slices[axis].stop)
            elif dims[i] == 0:
                dims[i] = shape[i]
            elif dims[i] != shape[i]:
                raise ValueError,"axis {[i]}({}) of length {} does not match previously defined shape {}".format(
                    i, axis, shape[i], dims[i])
            # update grids, if supplied
            if grids and axis in grids:
                parmdesc['grids'].setdefault(axis, grids[axis])

    def add_slice(self, name, array, slices, grids):
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
        assert(all([axis in parmdesc['axis_index'] for axis in slices]))
        self._update_parm_shape(parmdesc, array.shape, slices, grids)
        # dump to DB
        item = dict(entry="slice", name=name, array=array, slices=slices, grids=grids)
        cPickle.dump(item, self._fobj, 2)

    def close(self):
        """
        Closes the database
        """
        self._fobj.close()
        self._fobj = None
        # in create mode, update the descriptions file
        if self.mode is "create":
            self._save_desc()
        self.mode = "closed"

    def _save_desc(self):
        """Helper function. Writes accumulated parameter descriptions to filename.desc"""
        for name, desc in self._parmdescs.iteritems():
            print>> log(0), "dimensions of {} are {}".format(name, ','.join(map(str, desc['shape'])))
        cPickle.dump(self._parmdescs, open(self.filename+".desc", 'w'), 2)
        print>>log,"saved updated parameter descriptions to {}".format(self.filename+".desc")

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
        self._arrays = {}
        # try to read the desc file
        try:
            self._parmdescs = cPickle.load(open(filename + '.desc', 'r'))
        except:
            traceback.print_exc()
            print>> log(0), ModColor.Str("error loading dimensions from {} (see above), will try to re-generate".format(dimsfile))
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
                        self._update_parm_shape(self._parmdescs[name], item['array'].shape, item['slices'], item['grids'])
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
                    total_slice = [slice(None)]*len(desc['shape'])
                    for axis, axis_slice in item['slices'].iteritems():
                        total_slice[desc['axis_index'][axis]] = axis_slice
                    array[total_slice] = item['array']
                elif itemtype != "parmdesc":
                    raise IOError("{}: unknown item type '{}'".format(filename, itemtype))

    def names(self):
        """
        Returns names of all defined parameters
        """
        return self._parmdescs.keys()

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
