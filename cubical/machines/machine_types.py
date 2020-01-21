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
from . import complex_2x2_machine
from . import complex_W_2x2_machine
from . import phase_diag_machine
from . import slope_machine


# this provides a map from string "Jones type" identifiers to specific GainMachine classes

GAIN_MACHINE_TYPES = {
    'complex-2x2': complex_2x2_machine.Complex2x2Gains,
    'complex-diag': complex_2x2_machine.Complex2x2Gains,
    'phase-diag': phase_diag_machine.PhaseDiagGains,
    'robust-2x2': complex_W_2x2_machine.ComplexW2x2Gains,
    'f-slope': slope_machine.PhaseSlopeGains,
    't-slope': slope_machine.PhaseSlopeGains,
    'tf-plane': slope_machine.PhaseSlopeGains
}

def get_machine_class(typestr):
    """
    Returns gain machine class object corresponding to the type string

    Args:
        type (str): GM type, e.g. "complex-2x2"

    Returns:
        gain machine class object, or None if not found
    """
    return GAIN_MACHINE_TYPES.get(typestr)
