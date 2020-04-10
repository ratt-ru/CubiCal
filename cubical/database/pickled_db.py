# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Handles parameter databases which can contain solutions and other relevant values. 
"""
from __future__ import print_function
from future.moves import pickle
import os, os.path
import numpy as np
import traceback
from cubical.tools import logger, ModColor
log = logger.getLogger("param_db")
import time
from collections import OrderedDict, Iterator

from cubical.database.parameter import Parameter, _Record
from .iface_database import iface_database
class _ParmSegment(_Record):
    """
    A ParmSegment is just a Record -- we just want it to be a special type so that it
    can be identified when unpickled.
    """
    pass

class PickledDatabase(iface_database):
    """
    This class implements a simple parameter database saved to a pickle.
    """

    def __init__(self):
        self._fobj = None

    def _create(self, filename, metadata={}, backup=True, **kw):
        """
        Creates a parameter database given by the filename and opens it in "create" mode.

        Args:
            filename (str): 
                Name of database.
            metadata (dict, optional): 
                Optional metadata to be stored in DB.
            backup (bool, optional):
                If True, and an old database with the same filename exists, make a backup.
            kw (dict):
                Keyword arguments.
        """

        self.mode = "create"
        self.filename = filename
        self.do_backup = backup
        self.metadata = OrderedDict(mode=self.MODE_FRAGMENTED, time=time.time(), **metadata)
        # we'll write to a temp file, and do a backup on successful closure
        self._fobj = open(filename + ".tmp", 'wb')
        pickle.dump(self.metadata, self._fobj)
        self._fobj.flush()
        self._parameters = {}
        self._parm_written = set()
        print("creating {} in {} mode".format(self.filename, self.metadata['mode']), file=log(0))

    def define_param(self, *args, **kw):
        """
        Defines a parameter. Only valid in "create" mode.

        Args:
            args (tuple):
                Positional arguments.
            kw (dict):
                Keyword arguments.
        """

        assert (self.mode is "create")
        parm = Parameter(*args, **kw)
        self._parameters[parm.name] = parm
        # we don't write it to DB yet -- write it in add_chunk() rather
        # this makes it easier to deal with I/O workers (all IO is done by one process)
        return parm

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

        assert (self.mode is "create")
        parm = self._parameters.get(name)
        assert (parm is not None)
        # dump parm to DB the first time a slice shows up
        if name not in self._parm_written:
            pickle.dump(parm, self._fobj, 2)
            self._parm_written.add(name)
        # update axis shapes and grids based on slice
        parm._update_shape(array.shape, grid)
        # dump slice to DB
        item = _ParmSegment(name=name, array=np.ma.asarray(array), grid=grid)
        pickle.dump(item, self._fobj, 2)

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

        for desc in self._parameters.values():
            desc._finalize_shape()
        for key in list(self._parameters.keys()):
            if not self._parameters[key]._populated:
                del self._parameters[key]
        with open(self.filename + ".skel", 'wb') as pf:
            pickle.dump(self._parameters, pf, 2)
        print("saved updated parameter skeletons to {}".format(self.filename + ".skel"), file=log(0))

    def _backup_and_rename(self, backup):
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
                print("previous DB will be backed up as " + backup_filename, file=log(0))
                if os.path.exists(backup_filename):
                    print("  removing old backup " + backup_filename, file=log(0))
                    os.unlink(backup_filename)
                os.rename(self.filename, backup_filename)
            else:
                os.unlink(self.filename)
        os.rename(self.filename + ".tmp", self.filename)
        print("wrote {} in {} mode".format(self.filename, self.metadata['mode']), file=log(0))

    def save(self, filename=None, backup=True):
        """
        Save the database.

        Args:
            filename (str, optional):
                Name of output file.
            backup (bool, optional):
                If True, create a backup.
        """
        assert (self.mode is "load")
        self.metadata['mode'] = self.MODE_CONSOLIDATED
        filename = filename or self.filename
        with open(filename + ".tmp", 'w') as fobj:
            pickle.dump(self.metadata, fobj, 2)
            for parm in self._parameters.values():
                parm.release_cache()
            pickle.dump(self._parameters, fobj, 2)
        # successfully written? Backup and rename
        self.filename = filename
        self._backup_and_rename(backup)

    MODE_FRAGMENTED = "fragmented"
    MODE_CONSOLIDATED = "consolidated"

    class _Unpickler(Iterator):
        def __init__(self, filename):
            self.fobj = open(filename, 'rb')
            self.metadata = pickle.load(self.fobj)
            if type(self.metadata) is not OrderedDict or not "mode" in self.metadata:
                raise IOError("{}: invalid metadata entry".format(filename))
            self.mode = self.metadata['mode']
        
        def __del__(self):
            self.fobj.close()

        def __next__(self):
            try:
                return pickle.load(self.fobj)
            except EOFError:
                raise StopIteration

        def next(self):
            return self.__next__()

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
        print("reading {} in {} mode".format(self.filename, db.mode), file=log(0))
        self.metadata = db.metadata
        for key, value in self.metadata.items():
            if key != "mode":
                print("  metadata '{}': {}".format(key, value), file=log(1))

        # now load differently depending on mode
        # in consolidated mode, just unpickle the parameter objects
        if db.mode == PickledDatabase.MODE_CONSOLIDATED:
            self._parameters = next(db)
            for parm in self._parameters.values():
                print("  read {} of shape {}".format(parm.name,
                                                               'x'.join(map(str, parm.shape))), file=log(1))
            return

        # otherwise we're in fragmented mode
        if db.mode != PickledDatabase.MODE_FRAGMENTED:
            raise IOError("{}: invalid mode".format(self.filename, self.metadata.mode))

        # in fragmented mode? try to read the desc file
        descfile = filename + '.skel'
        self._parameters = None
        if not os.path.exists(descfile):
            print(ModColor.Str("{} does not exist, will try to rebuild".format(descfile)), file=log(0))
        elif os.path.getmtime(descfile) < os.path.getmtime(self.filename):
            print(ModColor.Str("{} older than database: will try to rebuild".format(descfile)), file=log(0))
        elif os.path.getmtime(descfile) < os.path.getmtime(__file__):
            print(ModColor.Str("{} older than this code: will try to rebuild".format(descfile)), file=log(0))
        else:
            try:
                with open(descfile, 'rb') as pf:
                    self._parameters = pickle.load(pf)
            except:
                traceback.print_exc()
                print(ModColor.Str("error loading {}, will try to rebuild".format(descfile)), file=log(0))
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
        for parm in self._parameters.values():
            parm._init_arrays()

        # go over all slices to paste them into the arrays
        db = self._Unpickler(filename)
        for item in db:
            if type(item) is Parameter:
                pass
            elif type(item) is _ParmSegment:
                parm = self._parameters.get(item.name)
                if parm is None:
                    raise IOError("{}: no parm found for {}'".format(filename, item.name))
                parm._paste_slice(item)
            else:
                raise IOError("{}: unknown item type '{}'".format(filename, type(item)))

        # ok, now arrays and flags each contain a full-sized array. Break it up into slices.
        for parm in self._parameters.values():
            parm._finalize_arrays()

    def names(self):
        """ Returns names of all defined parameters. """

        return list(self._parameters.keys())

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