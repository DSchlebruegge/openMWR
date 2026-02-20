"""
Adiabatic cloud model for liquid water content (LWC) computation and cloud detection.
Provides the ``CloudColumn`` class for representing vertical atmospheric columns
and calculating cloud properties based on temperature and humidity profiles.

Example usage:

.. code-block:: python

    from openMWR.cloud import CloudColumn, CloudModelConfig
    import numpy as np
    # Sample atmospheric profile data
    z = np.array([0, 100, 200, 300, 400, 500])  # Altitude in meters
    p = np.array([100000, 98000, 96000, 94000, 92000, 90000])  # Pressure in Pa
    T = np.array([293.15, 290.15, 287.15, 284.15, 281.15, 278.15])  # Temperature in K
    rh = np.array([0.8, 0.85, 0.9, 0.95, 0.98, 1.0])  # Relative Humidity (0–1)

    config = CloudModelConfig(rh_thres=0.95)
    cloud_column = CloudColumn(z, p, T, rh, config)
    lwc = cloud_column.calculate_lwc()
    lwp = cloud_column.calculate_lwp()
    cloud_column.plot_cloud()

Or just use the default configuration:

.. code-block:: python

    cloud_column = CloudColumn(z, p, T, rh)
    lwc = cloud_column.calculate_lwc()
    lwp = cloud_column.calculate_lwp()
    cloud_column.plot_cloud()
"""


import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

from openMWR import consts
from openMWR.atm import saturation_vapor_pressure


@dataclass
class CloudModelConfig:
    """
    Configuration object for cloud microphysics and detection parameters.

    Parameters
    ----------
    formel : int, optional
        Formula index for saturation mixing ratio gradient computation.
        Options are {0, 1, 2}. Default is 0. Should all give same result. 
        Just for educational purposes.
    R_a_aprox : bool, optional
        If True, uses dry air gas constant approximation. Default is False.
    c_pd_aprox : bool, optional
        If True, uses dry specific heat approximation. Default is False.
    use_f_ad : bool, optional
        If True, uses the adiabatic fraction to scale LWC. Default is True.
    L_aprox : bool, optional
        If True, uses temperature-independent latent heat. Default is False.
    a : float, optional
        Parameter for non-Karstens adiabatic fraction profile. Default is 0.7.
        Formula: f_ad = a * (1 - (h/h_max) ** b) (h is height above cloud base)
    b : float, optional
        Parameter for non-Karstens adiabatic fraction profile. Default is 2.3.
        Formula: f_ad = a * (1 - (h/h_max) ** b) (h is height above cloud base)
    karstens : bool, optional
        If True, uses Karstens et al. (1994) formulation for adiabatic fraction.
        In this case, parameters `a` and `b` must not be manually modified.
        Default is False.
    rh_thres : float, optional
        Relative humidity threshold (0–1) for cloud detection. Default is 0.95.

    """

    formel: int = 0 
    R_a_aprox: bool = False
    c_pd_aprox: bool = False
    L_aprox: bool = False
    use_f_ad: bool = True
    a: float = 0.7
    b: float = 2.3
    karstens: bool = False
    rh_thres: float = 0.95

class CloudColumn:
    """
    Represents a vertical atmospheric column (z, p, T, rh) and provides
    cloud detection, liquid water content (LWC) computation, and plotting tools.

    Parameters
    ----------
    z : np.ndarray
        Altitude array in meters.
    p : np.ndarray
        Pressure array in pascals.
    T : np.ndarray
        Temperature array in Kelvin.
    rh : np.ndarray
        Relative humidity array (0–1).
    config : CloudModelConfig, optional
        Model configuration object. If None, default configuration is used.

    Notes
    -----
    The class stores all results internally after calling `calculate_lwc()`.
    """

    def __init__(
        self,
        z: np.ndarray,
        p: np.ndarray,
        T: np.ndarray,
        rh: np.ndarray,
        config: Optional[CloudModelConfig] = None,
    ):
        self.z = z
        self.p = p
        self.T = T
        self.rh = rh
        self.config = CloudModelConfig() if config is None else config

        self.i_top: Optional[np.ndarray] = None
        self.i_base: Optional[np.ndarray] = None
        self.lwc: Optional[np.ndarray] = None
        self.f_ad: Optional[np.ndarray] = None
        self.T_cl: Optional[np.ndarray] = None
        self.lwp: Optional[float] = None

    def calculate_lwc(self):
        """
        Detects liquid cloud layers and computes liquid water content (LWC) profile
        using a moist-adiabatic model.

        Returns
        -------
        lwc : np.ndarray
            Liquid water content profile [g/m³].
        """
        self._detect_liq_cloud()
        self._adiabatic_model()

        if self.config.use_f_ad:
            self._scale_with_adiabatic_fraction()
        else:
            self.lwc = self.lwc_ad
        
        return self.lwc
    
    def calculate_lwp(self):
        """
        Computes liquid water path (LWP) from the current LWC profile.

        Returns
        -------
        lwp : float
            Liquid water path (vertically integrated LWC) [g/m²].

        Raises
        ------
        RuntimeError
            If LWC has not been computed yet.
        """
        if self.lwc is None:
            raise RuntimeError("LWC must be computed before calculating LWP.")
        
        dz = np.append(np.diff(self.z), np.diff(self.z)[-1])
        self.lwp = float((self.lwc * dz).sum())
        
        return self.lwp

    def _detect_liq_cloud(self):
        """
        Detects liquid cloud layers based on humidity and temperature.

        Returns
        -------
        i_top : np.ndarray
            Indices where cloud layers end.
        i_base : np.ndarray
            Indices where cloud layers begin.

        Raises
        ------
        ValueError
            If a cloud extends to the end of the profile.
        """
        t_thres = 253.15  # Liquid threshold temperature [K]
        rh_thres = self.config.rh_thres

        cloud = (self.rh > rh_thres) & (self.T > t_thres)

        self.i_base = np.where(np.diff(cloud.astype(int), prepend=0) == 1)[0]
        self.i_top = np.where(np.diff(cloud.astype(int)) == -1)[0] + 1

        if cloud[-1]:
            raise ValueError(
                "Detected cloud reaches final array element; cloud top cannot be determined."
            )

    def _adiabatic_model(self):
        """
        Computes liquid water content (LWC) and moist-adiabatic cloud temperature along detected cloud layers.

        Returns
        -------
        lwc : np.ndarray
            Liquid water content [g/m³].
        T_cl : np.ndarray
            Cloud temperature profile [K].

        Raises
        ------
        RuntimeError
            If cloud indices are not initialized.
        """
        if self.i_base is None or self.i_top is None:
            raise RuntimeError("Cloud detection must be run before computing LWC.")
        
        z = self.z
        p = self.p
        T = self.T
        i_base = self.i_base
        i_top = self.i_top
        cfg = self.config

        self.lwc_ad = np.zeros(z.shape)

        # Cloud temperature
        T_cl = T.copy()

        # Liquid-water mixing ratio
        my_l = np.zeros(z.shape)
        
        for i_cloud in range(len(i_base)):
            # Cloud-base temperature is the measured temperature
            T_cl[i_base[i_cloud]] = T[i_base[i_cloud]]
            

            for i in range(i_base[i_cloud], i_top[i_cloud] + 1):
                delta_z = z[i + 1] - z[i]
                
                e_s = saturation_vapor_pressure(T_cl[i])

                # Specific gas constant for moist air
                if cfg.R_a_aprox:
                    R_m = consts.R_d
                else:
                    R_m = p[i] * consts.R_v * consts.R_d / (consts.R_d * e_s + consts.R_v*(p[i] - e_s))


                # Saturated specific humidity
                my_s = R_m/consts.R_v * e_s / p[i]

                #Specific heat at constant pressure of moist air 
                if cfg.c_pd_aprox:
                    c_pm = consts.c_pd
                else:
                    c_pm = ((consts.R_d * e_s * consts.c_pv + consts.R_v * (p[i] - e_s) * consts.c_pd) / 
                            (consts.R_d * e_s + consts.R_v * (p[i] - e_s)))

                # Calculate latent heat of vaporization
                if cfg.L_aprox:
                    L = consts.L_0
                else:
                    L = consts.L_0 - (consts.c_w - consts.c_pv)*(T_cl[i] - consts.T_0)
                
                # Moist-adiabatic lapse rate
                gamma_s = (
                    (consts.g / c_pm)
                    * (1 + (L * my_s / (R_m * T_cl[i])))
                    / (1 + (my_s * L**2 / (c_pm * consts.R_v * T_cl[i]**2)))
                )
                
                # Compute cloud temperature for the next layer
                if i != i_top[i_cloud]:
                    T_cl[i + 1] = T_cl[i] + (-gamma_s) * delta_z

                # Gradient of the saturation mixing ratio
                if cfg.formel == 0:
                    my_s_gradient = - c_pm/L * ((-gamma_s) + consts.g/c_pm)
                elif cfg.formel == 1:
                    my_s_gradient = my_s * (L/(consts.R_v * T_cl[i]**2) * (-gamma_s) + consts.g/(R_m * T_cl[i]))
                elif cfg.formel == 2:
                    my_s_gradient = my_s * consts.g * ((consts.R_v / R_m * c_pm * T_cl[i] - L ) / 
                                                (consts.R_v * c_pm * T_cl[i]**2 + my_s * L**2))
                
                # Liquid-water mixing ratio
                if i != i_top[i_cloud]:
                    my_l[i + 1] = my_l[i] - my_s_gradient * delta_z
                
                # Density of moist air
                rho_m = p[i]/(R_m * T_cl[i])

                self.lwc_ad[i] = my_l[i] * rho_m 

        # In g/m³
        self.lwc_ad *= 1000
        self.T_cl = T_cl
    
    def _scale_with_adiabatic_fraction(self):
        """
        Scales the liquid water content (LWC) profile by the adiabatic fraction.

        Raises
        ------
        RuntimeError
            If LWC has not been computed yet.
        """
        if self.lwc_ad is None:
            raise RuntimeError("LWC must be computed before scaling.")


        z = self.z
        i_base = self.i_base
        i_top = self.i_top
        cfg = self.config

        #Adiabatic fraction 
        self.f_ad = np.ones(z.shape)
        if cfg.karstens:
            self.f_ad *= 1.239
        else:
            self.f_ad *= cfg.a
        
        for i_cloud in range(len(i_base)):

            h = z[i_base[i_cloud]: i_top[i_cloud] + 1] - z[i_base[i_cloud]]

            if cfg.karstens:
                f_ad_func = lambda h: 1.239 - 0.145 * np.log(h + 1)
                #Source: Karstens, U., Simmer, C. & Ruprecht, E. Remote sensing of cloud liquid water. 
                #Meteorl. Atmos. Phys. 54, 157-171 (1994). https://doi.org/10.1007/BF01030057
            else:
                #h_max = z[i_top[i_cloud] + 1] - z[i_base[i_cloud]]
                h_max = h[-1]
                f_ad_func = lambda h: cfg.a * (1 - (h/h_max) ** cfg.b)

            self.f_ad[i_base[i_cloud]: i_top[i_cloud] + 1] = f_ad_func(h)


        #Adiabatic fraction 
        if cfg.karstens:
            self.f_ad[self.f_ad < 0] = 0
            
        self.lwc = self.lwc_ad * self.f_ad

    def plot_cloud(self):
        """
        Plots cloud layers, thermodynamic profiles, LWC, and the adiabatic fraction.

        Raises
        ------
        RuntimeError
            If results are not computed yet (`calculate_cloud()` not called).
        """
        if self.lwc is None or self.f_ad is None:
            raise RuntimeError("Cloud must be computed before plotting.")

        z_km = self.z / 1000.0
        rh_thres = self.config.rh_thres

        plt.figure(figsize=(14, 7))

        # --- Subplot 1: RH & LWC ---
        plt.subplot(1, 2, 1)

        for idx, (ib, it) in enumerate(zip(self.i_base, self.i_top)):
            if idx == 0:
                plt.axhspan(z_km[ib], z_km[it], color="grey", alpha=0.5, ec=None, label="Cloud")
            else:
                plt.axhspan(z_km[ib], z_km[it], color="grey", alpha=0.5, ec=None)

        plt.axvline(x=rh_thres, label="rh_thres")
        plt.plot(self.rh, z_km, label="RH")
        plt.plot(self.lwc, z_km, label="LWC [g/m³]")
        if self.config.use_f_ad and self.f_ad is not None:
            plt.plot(self.f_ad, z_km, label="f_ad")

        plt.ylabel("Height [km]")
        plt.legend()
        plt.grid()

        # --- Subplot 2: T & T_cloud ---
        plt.subplot(1, 2, 2)

        for idx, (ib, it) in enumerate(zip(self.i_base, self.i_top)):
            if idx == 0:
                plt.axhspan(z_km[ib], z_km[it], color="grey", alpha=0.5, ec=None, label="Cloud")
            else:
                plt.axhspan(z_km[ib], z_km[it], color="grey", alpha=0.5, ec=None)

        plt.plot(self.T_cl, z_km, label="T_cloud")
        plt.plot(self.T, z_km, label="T")
        plt.ylabel("Height [km]")
        plt.legend()
        plt.grid()
        plt.show()
