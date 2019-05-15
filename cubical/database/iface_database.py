# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Defines database interface
"""
from six import add_metaclass
import abc


@add_metaclass(abc.ABCMeta)
class iface_database:
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError("To be defined")
    
    @abc.abstractmethod
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
        raise NotImplementedError("To be defined")
    
    @abc.abstractmethod
    def define_param(self, *args, **kw):
        """
        Defines a parameter. Only valid in "create" mode.

        Args:
            args (tuple):
                Positional arguments.
            kw (dict):
                Keyword arguments.
        """
        raise NotImplementedError("To be defined")
    
    @abc.abstractmethod
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
        raise NotImplementedError("To be defined")
    
    @abc.abstractmethod
    def close(self):
        """ Closes the database. """
        raise NotImplementedError("To be defined")
    
    @abc.abstractmethod
    def _save_desc(self):
        """ Helper function. Writes accumulated parameter descriptions to filename.desc. """
        raise NotImplementedError("To be defined")
    
    @abc.abstractmethod
    def _backup_and_rename(self, backup):
        """
        May create a backup and renames the temporary DB.

        Args:
            backup (bool):
                If True, creates a backup, otherwise just renames.
        """
        raise NotImplementedError("To be defined")
        
    @abc.abstractmethod
    def save(self, filename=None, backup=True):
        """
        Save the database.

        Args:
            filename (str, optional):
                Name of output file.
            backup (bool, optional):
                If True, create a backup.
        """
        raise NotImplementedError("To be defined")
    
    @abc.abstractmethod
    def _load(self, filename):
        """
        Loads database from file. This will create arrays corresponding to the stored parameter
        shapes.

        Args:
            filename (str):
                Name of file to load.
        """
        raise NotImplementedError("To be defined")
    
    @abc.abstractmethod
    def names(self):
        """ Returns names of all defined parameters. """
        raise NotImplementedError("To be defined")

    @abc.abstractmethod
    def __contains__(self, name):
        raise NotImplementedError("To be defined")

    @abc.abstractmethod
    def __getitem__(self, name):
        raise NotImplementedError("To be defined")

    @abc.abstractmethod
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
        raise NotImplementedError("To be defined")