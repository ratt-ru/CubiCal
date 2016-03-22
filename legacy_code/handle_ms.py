from pyrap.tables import *

def get_cols(ms_name):

    ms = table(ms_name)
    data = ms.getcol("DATA")
    data = (data[:,0,0] + data[:,0,3])/2

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

get_cols("~/MeasurementSets/WESTERBORK.MS")