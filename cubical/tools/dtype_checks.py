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
        return np.issubdtype(v.dtype, np.integer) or v.dtype == np.bool_
    else:
        return False

def assert_isfp(v):
    if isinstance(v, float):
        return True
    elif isinstance(v, list):
        return all(map(lambda x: isinstance(x, float), v))
    elif isinstance(v, np.ndarray):
        return np.issubdtype(v.dtype, np.floating)
    else:
        return False