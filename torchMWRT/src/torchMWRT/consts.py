"""
Physical constants used by ``torchMWRT``.

The values for radiative-transfer constants are aligned with the original
``pyrtlib.utils.constants`` definitions. Thermodynamic constants are kept
as class attributes for direct access across the package.

References
----------
- :cite:alp:`Mohr-Taylor-Newell-2015-CODATA2014-NIST`
- :cite:alp:`Janssen-1993-AtmosphericRemoteSensing`, p. 12.
"""

#: Reference temperature [K].
T_0 = 273.15

#: Specific gas constant of dry air [J kg-1 K-1].
R_d = 287.058

#: Specific gas constant of water vapor [J kg-1 K-1].
R_v = 461.52

#: Ratio ``R_d / R_v`` [1].
eps = R_d / R_v

#: Earth radius [m].
EarthRadius = 6370949.0  # [m]

#: Cosmic background temperature [K].
Tcosmicbkg = 2.728  # [K]

#: Planck constant [J Hz-1].
planck = 6.626075499999999e-34  # [J Hz-1]

#: Boltzmann constant [J K-1].
boltzmann = 1.3806579999999998e-23  # [J K-1]

#: Speed of light in vacuum [m s-1].
light = 299792458.0  # [m s-1]
