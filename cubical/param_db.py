# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details
"""
Handles parameter databases which can contain solutions and other relevant values. 
"""
import numpy as np
from cubical.tools import logger
log = logger.getLogger("param_db")

#from database.pickled_db import PickledDatabase
from .database.casa_db_adaptor import casa_db_adaptor
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

    db = casa_db_adaptor()
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

    db = casa_db_adaptor()
    db._load(filename)
    
    return db



if __name__ == "__main__":
    log.verbosity(2)
    print("Creating test DB")
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

    print("Loading test DB")
    db = load("test.db")
    print(db.names())
    G = db['G']
    B = db['B']
    print("G", db["G"].axis_labels, db["G"].shape)
    print("B", db["B"].axis_labels, db["B"].shape)
    print("G", G.get_slice(ant=0,corr=0))
    print("B", G.get_slice(ant=0,corr=0))
    print("Gint", G.reinterpolate(time=np.arange(0,10,.5),freq=np.arange(0,10,1.5)))

