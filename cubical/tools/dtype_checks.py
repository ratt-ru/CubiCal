#   Copyright 2020 Jonathan Simon Kenyon
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
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