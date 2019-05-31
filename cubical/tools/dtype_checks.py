import six
import numpy as np

if six.PY3:
    long = int

def assert_isint(v):
    if isinstance(v, (int, long, bool)):
        return True
    elif isinstance(v, list):
        return all(map(lambda x: isinstance(x, (int, long, bool)), v))
    elif isinstance(v, np.ndarray):
        return v.dtype in [np.int, np.int16, np.int32, np.int64, np.int8,
                           np.int_, np.intc, np.integer, np.bool, int, long]
    else:
        return False

def assert_isfp(v):
    if isinstance(v, float):
        return True
    elif isinstance(v, list):
        return all(map(lambda x: isinstance(x, float), v))
    elif isinstance(v, np.ndarray):
        return v.dtype in [np.float, np.float_, np.float64, np.float16, np.float32, np.double]
    else:
        return False