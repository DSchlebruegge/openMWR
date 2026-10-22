"""
Physical constants used in the retrievals.
"""

#: Gravitational acceleration [m s-2].
g = 9.81

#: Specific gas constant of dry air [J kg-1 K-1].
R_d = 287.058

#: Specific gas constant of water vapor [J kg-1 K-1].
R_v = 461.5

#: Ratio ``R_d / R_v`` [1].
eps = R_d / R_v

#: Specific heat at constant pressure of dry air [J kg-1 K-1].
c_pd = 1005.

#: Specific heat capacity of liquid water [J kg-1 K-1].
c_w = 4187.

#: Specific heat at constant pressure of water vapor [J kg-1 K-1].
c_pv = 1870.

#: Reference temperature [K].
T_0 = 273.15

#: Latent heat of vaporization at ``T_0`` [J kg-1].
L_0 = 2.50 * 10**6
