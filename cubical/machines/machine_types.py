from . import complex_2x2_machine
from . import complex_W_2x2_machine
from . import pol_gain_machine
from . import phase_diag_machine
from . import slope_machine


# this provides a map from string "Jones type" identifiers to specific GainMachine classes

GAIN_MACHINE_TYPES = {
    'complex-2x2': complex_2x2_machine.Complex2x2Gains,
    'complex-diag': complex_2x2_machine.Complex2x2Gains,
    'complex-pol': pol_gain_machine.PolarizationGains,
    'phase-diag': phase_diag_machine.PhaseDiagGains,
    'robust-2x2': complex_W_2x2_machine.ComplexW2x2Gains,
    'robust-diag': complex_W_2x2_machine.ComplexW2x2Gains, 
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
