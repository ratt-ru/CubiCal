from itertools import count
import numpy as np
import pyrap.tables as pt
from collections import Counter, OrderedDict
import time
from data_handler_obj import *

def phocal_update():
    return

class Solver:

    solver_list = {"PHOCAL": phocal_update}

    def __init__(self, solver_type, data_handler):
        self.solver_type = solver_type
        self.dh = data_handler




# a = Solver("PHOCAL")
# print a.solver_type

ms = DataHandler("~/MeasurementSets/WESTERBORK_GAP.MS")

ms.get_data()
ms.define_chunk()
# ms.vis2mat()

for i in ms:
    print i[0].shape
# print ms.movis.shape, ms.obvis.shape, ms.antea.shape, ms.anteb\
#     .shape
