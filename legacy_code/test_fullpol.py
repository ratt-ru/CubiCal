from data_handler_obj import *

ms = DataHandler("~/MeasurementSets/WESTERBORK_POLAR_PHASE.MS")
ms.get_data()

d = ms.obvis[:105, 0, :]
print d.shape
ms.define_chunk()

for i in ms:
    print i

