import numpy as np
from physics.constants import GAS_CONSTANT_AIR, STD_PRESSURE_PA

def air_density_from_temp(air_temp_c):
    """rho = P / (R * T), pressure held at standard sea-level since FastF1 gives no pressure column."""
    T_kelvin = air_temp_c + 273.15
    return STD_PRESSURE_PA / (GAS_CONSTANT_AIR * T_kelvin)