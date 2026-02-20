import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd
import logging
from suntime import Sun
import io
import math
import xarray as xr
from datetime import timedelta

import metpy.calc as mpcalc
from metpy.plots import SkewT as metpy_SkewT
from metpy.units import units

from openMWR.utils import round_down, round_up
from openMWR.paths import radiosonde_station_dir

logger = logging.getLogger(__name__)

class SkewT(metpy_SkewT):
    """
    SkewT plotting helper based on a metpy SkewT subclass.
    Creates and manages a skew-T / log-p diagram for plotting microwave radiometer
    retrievals and radiosonde profiles. The class encapsulates figure creation,
    axis scaling (rotation and aspect adjustments), plotting convenience methods for
    radiosonde and retrieval profiles, and finalization routines to annotate and
    export the plot.

    Parameters
    ----------
    date : datetime-like
        Reference datetime for the plot. Used to align retrieval and radiosonde
        times when selecting data from xarray datasets.
    data_dir : str
        Base directory containing all openMWR-managed data.
    min_pressure : int, optional
        Minimum pressure (hPa) to include on the plot (top of the plot). Default is
        100 hPa.
    max_pressure : int, optional
        Maximum pressure (hPa) to include on the plot (bottom of the plot).
        Default is 1000 hPa.
    T_range : float, optional
        Temperature range in degrees Celsius for the x-axis span. Default is 70 °C.
    station_alt : float, optional
        Station altitude in meters. Used to compute and label height tick marks on
        the y-axis. Default is 550 m.
    
    Examples
    --------
    Typical usage:

    .. code-block:: python

        skew = SkewT(date=date, data_dir='data')
        skew.plot_radiosonde('03715')
        skew.plot_retrieval(pred, label='NN Retrieval', color='red')
        skew.finalize()
        skew.save('skewt_profile.svg')
    """

    def __init__(self, date, data_dir, min_pressure=100, max_pressure=1000, T_range=70, station_alt=550):

        self.date = date
        self.data_dir = data_dir
        self.min_pressure = min_pressure
        self.max_pressure = max_pressure
        self.T_range = T_range
        self.station_alt = station_alt

        fig = plt.figure(figsize=(10, 9), constrained_layout=True)

        #rotation, aspect
        scale_fac = np.log10(1000/100) / np.log10(max_pressure/min_pressure) * T_range / 70
        aspect_0 = 65#80.5
        self.aspect_new = aspect_0 * scale_fac

        rotation_0 = 45
        self.rotation_new = 90 - np.rad2deg (np.arctan( np.tan(np.deg2rad(90-rotation_0)) * scale_fac))

        super().__init__(fig, rotation=self.rotation_new, aspect=self.aspect_new)

        self.retrieval_time = self.date
        self.synop_time = self.date

        self.retrievals = []

        self.radiosonde_plotted = False

    def _get_radiosonde(self, station_id):
        """Load radiosonde data for station_id and select the synop nearest to self.date.

        Loads the file
        ``{data_dir}/radiosondes/station_{station_id}/raw_data_{year}.nc``,
        restricts synops to the same calendar date as self.date, selects the nearest
        synop, drops NaNs along "meastime", swaps dims to "height", and removes
        duplicate heights. Raises ValueError if no synops are found for the date.
        """
        

        station_dir = radiosonde_station_dir(self.data_dir, station_id)
        raso_data = xr.load_dataset(station_dir / f'raw_data_{self.date.year}.nc')

        synops_on_date = raso_data.synop.where(raso_data.synop.dt.date == self.date.date(), drop=True)

        if synops_on_date.size == 0:
            raise ValueError(f"No matching radiosondes found for date: {self.date}")
        
        selected_synop = synops_on_date.sel(synop=self.date, method="nearest")

        selected_raso_data = raso_data.sel(synop=selected_synop).dropna(dim="meastime") 

        selected_raso_data = selected_raso_data.swap_dims({'meastime': 'height'})

        # Remove duplicate heights
        selected_raso_data = selected_raso_data.sel(height=~selected_raso_data.height.to_series().duplicated().values)

        return selected_raso_data
    
    def _extract_variables(self, raso_data):
        """Extract and convert variables from a rasosonde dataset into a units-aware dict.

        Parameters
        ----------
        raso_data : Mapping-like (e.g., xarray.Dataset or dict)
            Input sounding dataset containing the keys 'height', 'p', 'T', 'TD',
            'wind_speed', and 'wind_dir'. Each entry is expected to be array-like
            (e.g., numpy.ndarray) accessible via indexing (e.g., raso_data["height"].values).
        
        Returns
        -------
        dict
            Dictionary of variables with attached MetPy/pint units:
            - 'height' : array (units.meter)
            - 'p'      : array (units.hPa)
            - 'T'      : array (units.kelvin)
            - 'TD'     : array (units.kelvin)
            - 'wind_speed' : array (units.m/s)
            - 'wind_dir'   : array (units.degrees)
            - 'u'      : array (zonal wind component, same units as wind_speed)
            - 'v'      : array (meridional wind component, same units as wind_speed)
        """
        
        raso_variables = {}
        raso_variables["height"] = raso_data["height"].values * units.meter
        raso_variables["p"] = raso_data["p"].values * units.hPa
        raso_variables["T"] = raso_data["T"].values * units.kelvin
        raso_variables["TD"] = raso_data["TD"].values * units.kelvin
        raso_variables["wind_speed"] = raso_data["wind_speed"].values * units("m/s")
        raso_variables["wind_dir"] = raso_data["wind_dir"].values * units.degrees
        raso_variables["u"], raso_variables["v"] = mpcalc.wind_components(
            raso_variables["wind_speed"], raso_variables["wind_dir"]
        )
        return raso_variables
    
    def _calculate_parameters(self, raso_variables):
        #Not used currently
        
        # Surface parcel
        # Calculate the parcel profile.
        parcel_prof = mpcalc.parcel_profile(
            raso_variables["p"], raso_variables["T"][0], raso_variables["TD"][0]
        )

        # Calculate the LCL
        lcl_pressure, lcl_temperature = mpcalc.lcl(
            raso_variables["p"][0], raso_variables["T"][0], raso_variables["TD"][0]
        )

        # Calculate the CCL
        ccl_pressure, ccl_temperature, convective_temperature = mpcalc.ccl(
            raso_variables["p"], raso_variables["T"], raso_variables["TD"]
        )

        # Calculate LFC
        lfc_pressure, _ = mpcalc.lfc(raso_variables["p"], raso_variables["T"], raso_variables["TD"])

        # Calculate EL
        el_pressure, _ = mpcalc.el(
            raso_variables["p"], raso_variables["T"], raso_variables["TD"], parcel_prof
        )

        # Calculate CAPE and CIN
        cape, cin = mpcalc.cape_cin(
            raso_variables["p"], raso_variables["T"], raso_variables["TD"], parcel_prof
        )

        # Calculate lifted index
        lift_index = mpcalc.lifted_index(raso_variables["p"], raso_variables["T"], parcel_prof)

        # mixed layer parcel
        # Calculate the mixed CAPE and CIN
        # ML_cape, ML_cin = mpcalc.mixed_layer_cape_cin(p, T, TD)
        # print(ML_cape, ML_cin)

        # # Calculate the MU CAPE and CIN
        # MU_cape, MU_cin = mpcalc.most_unstable_cape_cin(p, T, TD)
        # print(MU_cape, MU_cin)

        # # Calculate precipitable water
        # precipitable_water = mpcalc.precipitable_water(p, TD)
        # print(precipitable_water)

        # Calculate wet bulb temperature
        wet_bulb_T = mpcalc.wet_bulb_temperature(
            raso_variables["p"], raso_variables["T"], raso_variables["TD"]
        )

        calculated_params = {
            "parcel_prof": parcel_prof,
            "lcl_pressure": lcl_pressure,
            "lcl_temperature": lcl_temperature,
            "ccl_pressure": ccl_pressure,
            "ccl_temperature": ccl_temperature,
            "convective_temperature": convective_temperature,
            "lfc_pressure": lfc_pressure,
            "el_pressure": el_pressure,
            "cape": cape,
            "cin": cin,
            "lift_index": lift_index,
            "wet_bulb_T": wet_bulb_T,
        }
        return calculated_params

    def plot_radiosonde(self, raso_station_id = '03715'):
        """
        Plot the radiosonde profile for the specified station ID on the SkewT.

        Parameters
        ----------
        raso_station_id : str, optional
            The station ID of the radiosonde to plot. Default is '03715'.
        """
        try:
            raso_data = self._get_radiosonde(raso_station_id)

            self.synop_time = pd.to_datetime(raso_data.synop.values)
            self.retrieval_time = pd.to_datetime(raso_data.starttime.values)

        except (FileNotFoundError, ValueError) as e:
            logger.error(e)
            logger.debug(e, exc_info=True)

            return

        raso_variables = self._extract_variables(raso_data)
        raso_label = f"Radiosonde\n{raso_data.starttime.dt.strftime('%H:%M').values} - {raso_data.endtime.dt.strftime('%H:%M UTC').values}"

        self.plot(raso_variables["p"], raso_variables["T"], "black", label="Temperature\n"+raso_label)
        self.plot(raso_variables["p"], raso_variables["TD"], "black", linestyle='--', label="Dewpoint\n"+raso_label)

        # Set spacing interval--Every 50 mb from 1000 to 100 mb
        my_interval = np.arange(100, 1000, 50) * units("mbar")

        # Get indexes of values closest to defined interval
        ix = mpcalc.resample_nn_1d(raso_variables["p"], my_interval)

        # Plot only values nearest to defined interval values
        self.plot_barbs(
            raso_variables["p"][ix],
            raso_variables["u"][ix],
            raso_variables["v"][ix],
            plot_units=units.knot,
            length=6
        )

        self.raso_station_id = raso_station_id
        self.radiosonde_plotted = True
        self.raso_data = raso_data
        self.retrievals.append(raso_data)

    def plot_retrieval(self, pred, label='', color="red", plot_TD=True):
        """
        Plot the microwave radiometer retrieval profile on the SkewT.

        Parameters
        ----------
        pred : xarray.Dataset
            The retrieval dataset containing 'p', 'T', and optionally 'TD' variables.
        label : str, optional
            Label for the retrieval profile in the legend. Default is an empty string.
        color : str, optional
            Color for the retrieval profile lines. Default is "red".
        plot_TD : bool, optional
            Whether to plot the dewpoint temperature profile. Default is True.
        """

        retrieval_data = pred.sel(time = self.retrieval_time, method="nearest")

        label += f"\n{retrieval_data.time.dt.strftime('%H:%M UTC').values}"

        self.plot(retrieval_data["p"].values * units.hPa, retrieval_data["T"].values * units.kelvin, color, label="Temperature MWR\n"+label)
        if plot_TD:
            self.plot(retrieval_data["p"].values * units.hPa, retrieval_data["TD"].values * units.kelvin, color, linestyle='--', label="Dewpoint MWR\n"+label)

        self.retrievals.append(retrieval_data)

    def finalize(self):
        """
        Finalize the SkewT plot by adjusting axes, adding special lines,
        and annotating with titles and labels.
        """
        data_on_plot = [data.where((data["p"] > self.min_pressure) & (data["p"] < self.max_pressure)) for data in self.retrievals]

        #T_range / np.log(max_pressure/min_pressure) * np.log(max_pressure/data['p']) * aspect_new/100 / np.tan(np.deg2rad(90-rotation_new))

        Tmax_list = [(data["T"] + self.aspect_new * np.log10(self.max_pressure/data['p']) / np.tan(np.deg2rad(90-self.rotation_new)) ).max().item() for data in data_on_plot]
        Tmax = max(Tmax_list) - 273.15 + 1

        Tmax = round_up(Tmax, 10) #+ 10
        Tmin = Tmax - self.T_range

        # Plot a zero degree isotherm
        self.ax.axvline(0, color="gray", linestyle="-", linewidth=1)

        # Add the relevant special lines
        self.plot_dry_adiabats(pressure=np.array([self.max_pressure, self.min_pressure]) * units.hPa, linewidth=0.5, linestyle="-")
        self.plot_moist_adiabats(pressure=np.array([self.max_pressure, self.min_pressure]) * units.hPa, linewidth=0.5, linestyle="-")

        mixing_ratio = np.array([0.0001, 0.0002, 0.0004, 0.001, 0.002, 0.004, 0.007, 0.01, 0.016, 0.024, 0.032])
        lc = self.plot_mixing_lines(pressure=np.array([self.max_pressure, self.min_pressure]) * units.hPa, linewidth=0.5, linestyle="-", mixing_ratio=mixing_ratio)

        for line, ratio in zip(lc.get_segments(), mixing_ratio):
            x, y = line[0]  # Startpunkt der Linie
            if x > Tmin and x <= Tmax:
                self.ax.text(x, y, f'{ratio*1000:.1f}', fontsize=7, color='green', ha='right', va='bottom')

        self.ax.grid(linewidth=0.5)

        #Min, Max
        self.ax.set_ylim(self.max_pressure, self.min_pressure)

        self.ax.set_xlim(Tmin, Tmax)

        # Achsenbeschriftung
        #skew.ax.set_xlabel("Temperature [°C]")
        self.ax.set_xlabel("")
        self.ax.text(0.5, -0.06, 'Temperature [°C], ', transform=self.ax.transAxes, color='black', ha='right')
        self.ax.text(0.5, -0.06, 'Mixing Ratio [g/kg]', transform=self.ax.transAxes, color='green', ha='left')

        #skew.ax.set_ylabel("Pressure [hPa]")
        self.ax.set_ylabel("")
        self.ax.text(
            -0.06, 0.5, "Pressure [hPa], ", transform=self.ax.transAxes, #-0.12
            fontsize=10, color="black", va="top", ha="center", rotation='vertical'
        )
        self.ax.text(
            -0.06, 0.5, "Height [km]", transform=self.ax.transAxes,
            fontsize=10, color="gray", va="bottom", ha="center", rotation='vertical'
        )

        # height_array = np.array([1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000]) #raso_data.station_alt

        # if min_pressure == 500:
        #     height_array = np.array([1000, 2000, 3000, 4000, 5000])

        height_min = round_up(self.station_alt, 1000)

        height_array = np.arange(height_min, 17000, 1000)

        retrieval_data = self.retrievals[0]

        if self.radiosonde_plotted:
            p_array = self.raso_data.sel(height=height_array, method="nearest").p.values
        else:
            p_array = retrieval_data.sel(height=height_array-self.station_alt, method="nearest").p.values

        indices = np.where((p_array > self.min_pressure) & (p_array < self.max_pressure))[0]
        p_array = p_array[indices]
        height_array = height_array[indices]

        ylabels = [f"{self.station_alt/1000:5.2f}"] + [f"{height/1000:<5.1f}" for height in height_array]
        p_array = np.insert(p_array, 0, retrieval_data.p.values[0])


        self.ax.set_yticks(ticks=p_array, minor=True)
        self.ax.set_yticklabels(labels=ylabels, minor=True, color="gray")
        self.ax.tick_params(direction='in', axis="y", which="minor", color="gray", pad=-22, length=3, labelsize=7) #pad=25, length=5,

        self.ax.tick_params(labelsize=9)

        self.ax.legend(fontsize="medium", loc="upper left", bbox_to_anchor=(1, 1), frameon=False) #, loc=3

        raso_locations = {'03715': 'Oberschleißheim', '02290': 'Hohenpeißenberg'}

        if self.radiosonde_plotted:
            loc = raso_locations.get(self.raso_station_id, 'Unknown Location')
            title = f"Microwave Radiometer Profiles and Radiosonde at {loc} - {self.synop_time.strftime('%d.%m.%Y')}"
        else:
            title = f"Microwave Radiometer Profiles - {self.synop_time.strftime('%d.%m.%Y')}"

        self.ax.set_title(title, pad=15, fontsize='large')

    def save(self, filename):
        """
        Save the SkewT plot to a file.

        Parameters
        ----------
        filename : str
            The path to the file where the plot will be saved.
        """

        self._fig.savefig(filename, bbox_inches="tight") #, dpi=150
        plt.close(self._fig)
        logger.info(f"{filename} plotted")



class Axis:
    """
    Helper class to create and manage additional y-axes for time series plots.
    Each Axis instance corresponds to one y-axis, allowing for multiple y-axes
    to be added to a single time series figure. The class handles axis creation,
    spine coloring, label placement, and provides a convenient plot method.

    Parameters
    ----------
    zeitreihe : Zeitreihe
        The Zeitreihe instance to which this axis belongs.
    axis_num : int
        The index of the axis (0 for the first y-axis, 1 for the second, etc.).
    ylabel : str, optional
        The label for the y-axis. Default is None.
    spine_color : str, optional
        The color for the axis spines and ticks. Default is None, which uses the
        Zeitreihe's default axis color.
    lower_ylabel : bool, optional
        If True, places the y-axis label below the axis instead of beside it.
        Default is False.
    """

    def __init__(self, zeitreihe, axis_num, ylabel=None, spine_color=None, lower_ylabel=False):
        self.zeitreihe = zeitreihe
        self.ax_position = [0.027, 0.08, 0.8748, 0.883] #[left, bottom, width, height]
        self.max_x = None

        self.custom_labels = []
        self.custom_handles = []

        if axis_num == 0:
            self.ax = self.zeitreihe.fig.add_axes(self.ax_position) #0.027, 0.08, 0.9018-0.027, 0.963-0.08

            self.ax.set_title(self.zeitreihe.title, pad=3, fontsize=self.zeitreihe.titlesize)

            #x-Achse
            self._create_x_axis()

            # Draw sunrise and sunset
            self._add_sunset_sunrise()

            #self.ax.grid(linewidth=0.5,linestyle='--')
            self.ax.grid(linewidth=0.3,linestyle='-', color='gray')
        else:
            self.ax = self.zeitreihe.axes[0].ax.twinx()

        self.ax.set_rasterization_zorder(None)  # Deaktiviert Rasterization basierend auf Z-Werten

        # Define spine color
        if spine_color is None:
            spine_color = self.zeitreihe.color_axes

        # Color the spines
        if axis_num == 0:
            self.ax.spines['left'].set_color(spine_color)
        else:
            self.ax.spines['right'].set_color(spine_color)
            self.ax.spines['left'].set_visible(False)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['bottom'].set_visible(False)
            self.zeitreihe.axes[0].ax.spines['right'].set_visible(False)
            self.ax.xaxis.set_visible(False)

        # Shift the 3rd and 4th axes
        if axis_num >= 2:
            self.ax.spines['right'].set_position(("axes", 1 + (491/4374)*(axis_num -1)/3))

        # Change spine thickness
        for spine in self.ax.spines.values():
            spine.set_linewidth(self.zeitreihe.spinewidth)

        # Format y-ticks
        self.ax.tick_params(axis='y', colors=spine_color, length=0, pad=2, labelsize=self.zeitreihe.fs)

        # Y labels
        if ylabel is not None:
            if lower_ylabel:
                self.ax.text(-0.03 + axis_num*1.035, -0.08, ylabel, color=spine_color, transform=self.ax.transAxes, fontsize=self.zeitreihe.fs)
            else:
                if axis_num == 0:
                    self.ax.set_ylabel(ylabel, color=spine_color, labelpad=-3, fontsize=self.zeitreihe.fs)
                else:
                    self.ax.set_ylabel(ylabel, color=spine_color, labelpad=3.0, fontsize=self.zeitreihe.fs)

    def _create_x_axis(self):
        self.ax.set_xlim(self.zeitreihe.date, self.zeitreihe.date + timedelta(days=1))
        self.ax.set_xticks(pd.date_range(start=self.zeitreihe.date, end=self.zeitreihe.date + timedelta(days=1), periods=13))
        self.ax.set_xticklabels(['   00','02','04','06','08','10','12','14','16','18','20','22','24   '], fontsize=self.zeitreihe.fs)
        self.ax.set_xlabel('Time [UTC]', labelpad=1.5, fontsize=self.zeitreihe.fs)
        self.ax.tick_params(axis='x', length=0)

    def _add_sunset_sunrise(self):
        sun = Sun(48.148, 11.573)
        sunrise = sun.get_sunrise_time(self.zeitreihe.date)
        sunset = sun.get_sunset_time(self.zeitreihe.date)

        if sunset < sunrise: # Bug in package suntime
            sunset += timedelta(days=1)

        trans = transforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)
        self.ax.axvline(x=sunrise, color=self.zeitreihe.color_axes, linestyle='--', linewidth=1)
        self.ax.text(sunrise,-0.08,'sunrise', horizontalalignment='center', transform=trans, fontsize=self.zeitreihe.fs)
        self.ax.axvline(x=sunset, color=self.zeitreihe.color_axes, linestyle='--',linewidth=1)
        self.ax.text(sunset,-0.08,'sunset', horizontalalignment='center', transform=trans, fontsize=self.zeitreihe.fs)

    def plot(self, *args, **kwargs):
        """
        Plot data on this axis.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to ax.plot().
        **kwargs : dict
            Keyword arguments to pass to ax.plot().
        """

        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = 1.5

        self.ax.plot(*args, **kwargs) 

    def plot_wind_dir(self, x: np.array, y: np.array, **line_settings):
        """
        Plot wind direction data, handling the circular nature of wind direction.

        Parameters
        ----------
        x : np.array
            The x-coordinates (e.g., time).
        y : np.array
            The wind direction values in degrees.
        line_settings : dict
            Additional keyword arguments for line styling (e.g., color, linestyle).
        """

        if 'linewidth' not in line_settings:
            line_settings['linewidth'] = 1.5

        if np.isnan(y).all():
            self.ax.plot(x, y, **line_settings) 
            return None

        offset = 0
        for i in range(1, y.size):
            offsets = [offset, offset + 360, offset - 360]
            offset = min(offsets, key=lambda o: abs(y[i] + o - y[i-1]))  # Sucht die kleinste Winddrehung
            y[i] += offset

        for i, offset in enumerate(range(int(round_down(np.nanmin(y), 360) ), int(round_up(np.nanmax(y), 360) ), 360)):
            if i == 1:
                line_settings['label'] = None
            self.ax.plot(x, y - offset, **line_settings) 

    def add_colorbar(self, c, label, yticks=None, pad=0.03, size=0.02):
        """
        Adds a colorbar to the figure

        Parameters
        ----------
        c : mappable
            The mappable object (e.g., from pcolormesh or contourf) to which the colorbar applies.
        label : str
            The label for the colorbar.
        yticks : list, optional
            Custom y-ticks for the colorbar. Default is None.
        pad : float, optional
            Padding between the axis and the colorbar in figure fraction. Default is 0.03.
        size : float, optional
            Width of the colorbar in figure fraction. Default is 0.02.
        """
        self.cax_pos = [
            self.ax_position[0] + self.ax_position[2] + pad, #left
            self.ax_position[1], #bottom
            size, #width
            self.ax_position[3] #height
        ]

        self.cax = self.zeitreihe.fig.add_axes(self.cax_pos)
        self.cbar = self.zeitreihe.fig.colorbar(c, cax=self.cax)
        self.cax.set_ylabel(label)
        if yticks is not None:
            self.cax.set_yticks(yticks)

    def plot_colormap(self, x, y, data, colorbar_label, cmap, vmin=None, vmax=None):
        """
        Plot a colormap using ax.pcolormesh.

        Parameters
        ----------
        x, y : Coordinates for the data.
        data : 2D array of data values.
        colorbar_label : str
            Label for the colorbar.
        cmap : Colormap to use.
        vmin, vmax : float, optional
            Minimum and maximum values for the colorbar. Default is None.
        """
        if np.all(data == 0) and vmax is None:
            vmax = 1

        c = self.ax.pcolormesh(x, y, data, cmap=cmap, shading='auto', rasterized=True, vmin=vmin, vmax=vmax)  

        self.max_x = x.max().values

        self.add_colorbar(c, colorbar_label)
        
        self.ax.set_ylim(y.min(),y.max()) 

    def plot_filled_contours(self, x, y, data, colorbar_label, cmap, contours=True, vmin=None, vmax=None, center=None, extend=None, n_bins_goal = 20):
        """
        Plot filled contours using ``ax.contourf`` with optional overlaid contour
        lines and automatic bin/level determination.

        Parameters
        ----------
        x : array-like
            X-coordinates corresponding to ``data``.
        y : array-like
            Y-coordinates corresponding to ``data``.
        data : 2D array-like
            Array of values to visualize.
        colorbar_label : str
            Label for the colorbar.
        cmap : str or Colormap
            Colormap used for the filled contour plot.
        contours : bool, optional
            If True, draw additional black contour lines over the filled contours.
            Default is True.
        vmin : float, optional
            Minimum value for the colormap range. If None, computed from ``data``.
        vmax : float, optional
            Maximum value for the colormap range. If None, computed from ``data``.
        center : float, optional
            Center value for diverging colormaps. Cannot be used simultaneously
            with both ``vmin`` and ``vmax``. If provided, symmetric limits are
            constructed around this value.
        extend : {None, 'both', 'min', 'max'}, optional
            Indicates whether colorbar/contours should extend beyond provided
            limits. Passed directly to ``contourf``. Default is None.
        n_bins_goal : int, optional
            Target number of color bins. The algorithm adjusts step size to reach
            a “nice” bin spacing. Default is 20.

        Notes
        -----
        - If all values in ``data`` are zero and ``vmax`` is not provided, a default
            ``vmax = 1`` is used.
        - Level spacing is internally adjusted to “nice” rounded steps based on
            order-of-magnitude heuristics.
        
        """
        # Handle edge case where all data values are zero
        if np.all(data == 0) and vmax is None:
            vmax = 1

        def find_better_step(step_orig):

            sign = step_orig / abs(step_orig)
            step_orig = abs(step_orig)

            order_of_magnitude = 10 ** math.floor(math.log10(step_orig)) 

            step = round(step_orig / order_of_magnitude)
            if step > 5:
                step = 1
                order_of_magnitude *= 10
            elif step == 3:
                step = 2 
            elif step == 4:
                step = 5
                
            step = step * order_of_magnitude * sign

            return step, order_of_magnitude

        #Determining levels
        lev_in_contour = 2 # Number of colored bins in one contour bin

        min = vmin if vmin is not None else data.min()
        max = vmax if vmax is not None else data.max()

        diff = max - min
        step = diff / n_bins_goal

        if (vmin is None or vmax is None) and center is None:

            step, order_of_magnitude = find_better_step(step)

            if vmin is None and vmax is None:
                vmin = round_down(min, order_of_magnitude * 10)
                diff = max - vmin

            diff = round_up(diff, step)

            if vmin is None:
                vmin = max - diff
        elif center is not None:
            max_dev = np.max([max - center, center - min])

            step, order_of_magnitude = find_better_step(2 * max_dev / n_bins_goal)

            max_dev = round_up(max_dev, step * 2)
            vmin = center - max_dev
            diff = 2 * max_dev
        
            

        n_bins = int(diff / step)

        levels = [vmin + i*step for i in range(n_bins + 1)]
        levels_contour = [vmin + lev_in_contour*i*step for i in range(int(n_bins / lev_in_contour) + 1)]

        # Create filled contour plot
        c = self.ax.contourf(x, y, data, cmap=cmap, levels=levels, extend=extend) #levels=levels, vmin=vmin, vmax=vmax

        if contours:
            cs = self.ax.contour(x, y, data, colors='k', levels=levels_contour, extend=extend, linewidths=self.zeitreihe.contour_linewidth) #levels=int(levels/3)
            self.ax.clabel(cs, cs.levels)

        # Update maximum x value
        self.max_x = x.max().values

        self.add_colorbar(c, colorbar_label, yticks=levels_contour)
        #self.cbar.add_lines(cs)

        # Set y-axis limits
        self.ax.set_ylim(y.min(), y.max())

    def add_hatches(self, time, mask, hatch_spacing=1/20):
        """
        Add diagonal hatch marks to highlight intervals where ``mask`` is True.

        Consecutive True values in ``mask`` are grouped into continuous time
        segments. For each segment, diagonal 45° lines are drawn using a blended
        data/axes transform, along with vertical boundary lines at the segment
        edges. A corresponding legend entry is also added.

        Parameters
        ----------
        time : array-like of datetime64
            Time values aligned with ``mask``.
        mask : array-like of bool
            Boolean mask marking intervals to be hatched.
        hatch_spacing : float, optional
            Spacing between hatch lines in axes coordinates. Default is ``1/20``.

        Returns
        -------
        None
            The hatches and boundaries are drawn directly on ``self.ax``.
        """
        trans = transforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)

        start_times = []
        durations = []

        last_true = False
        for i in range(mask.size):
            if mask[i] == True and not last_true:
                start_times.append(time[i])
                last_true = True
            elif mask[i] == False and last_true:
                durations.append(time[i] - start_times[-1])
                last_true = False

        if last_true:
            durations.append(time[i-1] - start_times[-1])

        m = (self.ax_position[2] * self.zeitreihe.fig.get_figwidth()) / (self.ax_position[3] * self.zeitreihe.fig.get_figheight())

        def f(x, l): #Linerar function for Lines with 45° angle
            y = m * ((x - np.datetime64(self.zeitreihe.date)) / np.timedelta64(1, 'D')) + l
            return y

        for i in range(len(start_times)):
            x_start = start_times[i]
            x_end = x_start + durations[i]

            for l in np.arange(-m*1, 1, hatch_spacing):
                y_start = f(x_start, l)
                y_end = f(x_end, l)

                self.ax.plot([x_start, x_end], [y_start, y_end], color='black', linewidth=0.5, transform=trans)#, zorder=9

            self.ax.axvline(x=x_start, color='k', linewidth=0.5)
            self.ax.axvline(x=x_end, color='k', linewidth=0.5)

        # Add object to legend (can be improved; not vectorized)
        self.custom_labels.append('Rain Flag')
        self.custom_handles.append(patches.Patch(
                linewidth=0.5,
                hatch_linewidth=0.5,
                edgecolor='black',
                facecolor='none',
                hatch='//',
                rasterized = False,
            ))

    def set_best_ylim(self, step_of_step = 0.5, number_of_ticks = 10, fix_min=None, fix_max=None):
        """
        Automatically set y-axis limits and tick spacing based on plotted data.

        The function determines suitable y-limits and a uniform tick step by
        scanning all visible line data. Optional fixed minimum or maximum values
        can be enforced, and the tick step is adjusted iteratively until the
        desired number of ticks fits the data range.

        Parameters
        ----------
        step_of_step : float, optional
            Increment used when gradually increasing the tick spacing. Default is 0.5.
        number_of_ticks : int, optional
            Desired number of y-axis ticks. Default is 10.
        fix_min : float, optional
            If provided, forces the lower y-limit to this value.
        fix_max : float, optional
            If provided, forces the upper y-limit to this value.

        Returns
        -------
        None
            Y-axis limits and ticks are updated directly on ``self.ax``.
        """
        if fix_max is not None and fix_min is not None:
            ymin_new = fix_min
            ymax_new = fix_max
            step  = (ymax_new - ymin_new) / number_of_ticks    

        elif fix_max is None and fix_min is not None:
            ymax = max((np.nanmax(line.get_ydata()) for line in self.ax.get_lines() 
                       if line._label[0] != '_' and line.get_ydata().size > 0 and not np.all(np.isnan(line.get_ydata()))),
                       default=fix_min+1)

            #Kleinen Puffer einbauen
            ymax += 0.02 * (ymax - fix_min)

            step = step_of_step
            ymin_new = fix_min

            ymax_new = ymin_new + number_of_ticks*step
            while ymax_new <= ymax:
                step += step_of_step
                ymax_new = ymin_new + number_of_ticks*step

        elif fix_max is not None and fix_min is None:
            ymin = min((np.nanmin(line.get_ydata()) for line in self.ax.get_lines() 
                       if line._label[0] != '_' and line.get_ydata().size > 0 and not np.all(np.isnan(line.get_ydata()))),
                       default=fix_max-1)

            #Kleinen Puffer einbauen
            ymin -= 0.02 * (fix_max - ymin)

            step = step_of_step
            ymax_new = fix_max

            ymin_new = ymax_new - number_of_ticks*step
            while ymin_new >= ymin:
                step += step_of_step
                ymin_new = ymax_new - number_of_ticks*step

        elif fix_max is None and fix_min is None:
            ymax = max((np.nanmax(line.get_ydata()) for line in self.ax.get_lines() 
                       if line._label[0] != '_' and line.get_ydata().size > 0 and not np.all(np.isnan(line.get_ydata()))),
                       default=1)
            ymin = min((np.nanmin(line.get_ydata()) for line in self.ax.get_lines() 
                       if line._label[0] != '_' and line.get_ydata().size > 0 and not np.all(np.isnan(line.get_ydata()))),
                       default=0)

            #Kleinen Puffer einbauen
            ymax += 0.02 * (ymax - ymin)
            ymin -= 0.02 * (ymax - ymin)

            step = round_up((ymax - ymin)/number_of_ticks, step_of_step)

            order_of_magnitude = 10 ** math.floor(math.log10(step)) 

            ymin_new = round_down(ymin, 10 * order_of_magnitude)

            ymax_new = ymin_new + number_of_ticks*step
            while ymax_new <= ymax:
                step += step_of_step
                order_of_magnitude = 10 ** math.floor(math.log10(step)) 

                ymin_new = round_down(ymin, 10 * order_of_magnitude)

                ymax_new = ymin_new + number_of_ticks*step

        self.ax.set_ylim(ymin_new,ymax_new)
        self.ax.set_yticks(np.arange(ymin_new, ymax_new + step, step=step))
    
class TimeSeriesPlot:
    """
    Container class for building multi-axis time series plots.

    The class manages a figure, creates aligned axes, and provides utilities for
    adding data, annotations, gray background stripes, and customized legends.
    It also offers convenience methods for saving or displaying the final plot.

    Parameters
    ----------
    date : datetime-like
        The reference date for the plot. Used in axis scaling and in the title.
    title : str
        Main title describing the plot content.
    pre_title : str, optional
        Text placed before the main title. Defaults to the institute/location
        description if omitted.

    Notes
    -----
    - Figures are automatically closed when the object is deleted.
    """
    def __init__(self, date, title, pre_title=None):
        self.date = date
        self.num_of_axes = 0
        self.color_axes = 'k'
        if pre_title is None:
            pre_title = 'Meteorological Institute of Ludwig Maximilian University (Munich, Germany; 48.148 N / 11.573 E)'
        self.title = f"{pre_title}:  {title}: {self.date:%d-%m-%Y}"

        self.titlesize = 9.5
        self.fs = 9.5
        self.spinewidth = 0.3
        self.contour_linewidth = 0.5

        #Initialize plot
        self.fig = plt.figure(figsize=(15.075,4.12))
        self.axes = []
    
    def add_axis(self, ylabel=None, spine_color=None, lower_ylabel=False):
        """
        Create and append a new axis to the figure.

        Parameters
        ----------
        ylabel : str, optional
            Label for the y-axis.
        spine_color : str or None, optional
            Color of the axis spine. If None, a default color is used.
        lower_ylabel : bool, optional
            If True, places the y-label below the axis.

        Returns
        -------
        Axis
            The created axis object.
        """
        axis = Axis(self, self.num_of_axes, ylabel, spine_color, lower_ylabel)
        self.axes.append(axis)
        self.num_of_axes += 1
        return axis

    def add_gray_stripes(self, alpha=0.25):
        """
        Add a shaded gray region after the last data point.

        Useful for visually separating the plotted day from the following
        day or marking incomplete data.

        Parameters
        ----------
        alpha : float, optional
            Transparency of the gray shading. Default is 0.25.

        Returns
        -------
        None
        """
        if self.axes[0].max_x is not None:
            max_x = self.axes[0].max_x
        else:
            max_x = max( max(np.nanmax(line.get_xdata()) for line in axis.ax.get_lines() if line._label[0] != '_' and line.get_ydata().size > 0) for axis in self.axes )
        
        self.axes[0].ax.axvspan(max_x + np.timedelta64(60, 's'), self.date + np.timedelta64(1, 'D'), color='lightgray', alpha=alpha) #np.timedelta64(30, 's')
      
    def create_legend(self, use_handels=False):
        """
        Draw a custom text-based legend outside the plot area.

        Handles from all axes are collected and placed manually via
        `ax.text` and small line/patch markers.

        Parameters
        ----------
        use_handels : bool, optional
            If True, draw small example line segments next to each label.
            Default is False.

        Returns
        -------
        None
        """
        handles = []
        labels = []

        # Iterate over all axes and extract handles and labels
        for axis in self.axes:
            handles_i, labels_i = axis.ax.get_legend_handles_labels()
            handles.extend(handles_i)  # Add handles to the full list
            labels.extend(labels_i)  # Add labels to the full list

        if len(labels) > 12:
            labelspacing=0.0475#0.33 * self.fs / 500
            y_koord = 1.0#1.0
        else:
            labelspacing=0.062#0.8 * self.fs / 500
            y_koord = 0.95#1.0
        if self.num_of_axes == 1:
            x_koord = 1.018
        else:
            x_koord = 1.028

        # Build custom legend manually
        for handle, label in zip(handles, labels):

            if type(handle) == lines.Line2D:
                color = handle.get_color()

                self.axes[0].ax.text(x_koord, y_koord, label, fontsize=self.fs, color=color, ha='left', va='top', transform=self.axes[0].ax.transAxes)

                if use_handels:
                    self.axes[0].ax.plot([x_koord, x_koord + 0.05], [y_koord, y_koord], transform=self.axes[0].ax.transAxes, clip_on=False)

            elif type(handle) == patches.Rectangle:

                    
                width_axes =  1.1 * self.fs / (self.axes[0].ax_position[2] * self.fig.get_figwidth() * self.fig.dpi)
                height_axes = 1.1 * self.fs / (self.axes[0].ax_position[3] * self.fig.get_figheight() * self.fig.dpi)

                new_rect = patches.Rectangle((x_koord, y_koord - height_axes), width_axes, height_axes,
                            linewidth=handle.get_linewidth(),
                            hatch_linewidth=handle.get_hatch_linewidth(),
                            edgecolor=handle.get_edgecolor(),
                            facecolor=handle.get_facecolor(),
                            hatch=handle.get_hatch(),
                            transform=self.axes[0].ax.transAxes,
                            clip_on=False
                            )
                
                self.axes[0].ax.add_patch(new_rect)

                self.axes[0].ax.text(x_koord + width_axes * 1.5 , y_koord, label, fontsize=self.fs, ha='left', va='top', transform=self.axes[0].ax.transAxes)
    
            y_koord -= labelspacing + label.count("\n") * 0.04

    def create_legend_with_handles(self):
        """
        Create a standard Matplotlib legend including custom handles.

        Collects both native plot handles and user-defined custom handles
        (e.g., from hatched patches) and places the legend to the right
        of the first axis.

        Returns
        -------
        None
        """
        handles = []
        labels = []

        # Iterate over all axes and extract handles and labels
        for axis in self.axes:
            handles_i, labels_i = axis.ax.get_legend_handles_labels()
            handles.extend(handles_i)  # Add handles to the full list
            labels.extend(labels_i)  # Add labels to the full list

            labels.extend(axis.custom_labels)
            handles.extend(axis.custom_handles)
        
        self.axes[0].ax.legend(handles, labels, loc='upper left', frameon=False, handlelength=0.7, borderpad=0, handletextpad=0.5, borderaxespad=1, bbox_to_anchor=(1, 1))

    def save(self, file_name, **kwargs):
        """
        Save the figure to the specified file and close it.

        Parameters
        ----------
        file_name : str
            Destination file path.
        **kwargs
            Additional arguments forwarded to `fig.savefig`.

        Returns
        -------
        None
        """
        self.fig.savefig(file_name, **kwargs)
        self.close()
        logger.info(f"{file_name} plotted") 

    def show(self, **kwargs):
        """
        Display the figure as an inline SVG in IPython environments.

        The figure is rendered into an in-memory buffer and closed
        afterwards to free resources.

        Parameters
        ----------
        **kwargs
            Additional arguments forwarded to `fig.savefig`.

        Returns
        -------
        None
        """
        from IPython.display import SVG, display

        buffer = io.BytesIO()
        self.fig.savefig(buffer, format="svg", **kwargs) #, bbox_inches="tight"
        self.close()

        buffer.seek(0)

        display(SVG(buffer.getvalue()))

    def close(self):
        """Close the figure associated with this plot."""
        plt.close(self.fig)

    def __del__(self):
        """Destructor to ensure the figure is closed when the object is deleted."""
        self.close()


class TimeSeriesCreator:
    """Static methods to create specific time series plots."""
    @staticmethod
    def lwp(date, pred_NN, pred_MLR=None, ds_hatpro=None, pre_title=None, retrieval_label='MIM-Retrieval'):
        lwp = TimeSeriesPlot(date, 'Liquid Water Path', pre_title=pre_title)
        ax = lwp.add_axis(lower_ylabel=True, ylabel='[g/$m^2$]')

        ax.plot(pred_NN.time, pred_NN.lwp, label=retrieval_label)

        if ds_hatpro is not None:
            lwp_rpg = ds_hatpro['lwp']
            lwp_rpg = lwp_rpg.rolling(time=5, center=True, min_periods=1).mean()
            ax.plot(lwp_rpg.time, lwp_rpg, label='RPG-Retrieval')

        if pred_MLR is not None:
            ax.plot(pred_MLR.time, pred_MLR.lwp, label=f'{retrieval_label}\n(MLR)')

        ax.set_best_ylim(step_of_step=10, fix_min=0) #, fix_min=0

        rain_flag_mask = pred_NN['rain_flag'] > 0
        ax.add_hatches(rain_flag_mask.time.values, rain_flag_mask.values)

        lwp.add_gray_stripes(alpha=1)
        lwp.create_legend_with_handles()

        return lwp

    @staticmethod
    def iwv(date, pred_NN, ds_hatpro, pre_title=None, retrieval_label='MIM-Retrieval'):
        iwv = TimeSeriesPlot(date, 'Integrated Water Vapour', pre_title=pre_title)
        ax = iwv.add_axis(lower_ylabel=True, ylabel='[kg/$m^2$]')

        ax.plot(pred_NN.time, pred_NN['iwv'], label=retrieval_label)

        if ds_hatpro is not None:
            iwv_rpg = ds_hatpro['iwv']
            iwv_rpg = iwv_rpg.rolling(time=5, center=True, min_periods=1).mean()
            ax.plot(iwv_rpg.time, iwv_rpg, label='RPG-Retrieval')
        #ax.plot(pred_MLR.time, pred_MLR['iwv_from_ah'], label='MIM-Retrieval\n(MLR)')

        ax.set_best_ylim(step_of_step=1, fix_min=0)

        rain_flag_mask = pred_NN['rain_flag'] > 0
        ax.add_hatches(rain_flag_mask.time.values, rain_flag_mask.values)

        iwv.add_gray_stripes(alpha=1)
        iwv.create_legend_with_handles()

        return iwv

    @staticmethod
    def temp(date, data, pre_title=None, retrieval_label='MIM-Retrieval'):
        temp = TimeSeriesPlot(date, f'Temperature - {retrieval_label}', pre_title=pre_title)
        ax = temp.add_axis(ylabel='Height [km]')
        ax.plot_filled_contours(data.time, data.height/1000, data.T.values.T, 'Temperature [K]', 'RdBu_r') #coolwarm
        rain_flag_mask = data['rain_flag'] > 0
        ax.add_hatches(rain_flag_mask.time.values, rain_flag_mask.values)
        temp.add_gray_stripes(alpha=1)
        return temp

    @staticmethod
    def temp_an(date, data, pre_title=None, retrieval_label='MIM-Retrieval'):
        temp_an = TimeSeriesPlot(date, f'Temperature Anomaly - {retrieval_label}', pre_title=pre_title) #Perturbation, Deviation?
        ax = temp_an.add_axis(ylabel='Height [km]')

        T_an = data.T - data.T.mean(dim='time')
        ax.plot_filled_contours(data.time, data.height/1000, T_an.values.T, 'Temperature Anomaly to Daily Average [K]', 'RdBu_r', center = 0) #coolwarm

        rain_flag_mask = data['rain_flag'] > 0
        ax.add_hatches(rain_flag_mask.time.values, rain_flag_mask.values)

        temp_an.add_gray_stripes(alpha=1)
        return temp_an

    @staticmethod
    def dT(date, data, pre_title=None, retrieval_label='MIM-Retrieval'):
        dT = TimeSeriesPlot(date, f'Stability - {retrieval_label}', pre_title=pre_title)
        ax = dT.add_axis(ylabel='Height [km]')

        label = 'unstable   <-   Stability   ->     stable\n(Temperature Gradient minus Dry Adiabatic [K/km])'
        ax.plot_filled_contours(data.time, data.height/1000, data['T_grad'].values.T + 9.8, label, 'RdBu', vmin=-20, vmax=20, extend='both') #

        rain_flag_mask = data['rain_flag'] > 0
        ax.add_hatches(rain_flag_mask.time.values, rain_flag_mask.values)

        dT.add_gray_stripes(alpha=1)
        return dT

    @staticmethod
    def rh(date, data, pre_title=None, retrieval_label='MIM-Retrieval'):
        rh = TimeSeriesPlot(date, f'Relative Humidity - {retrieval_label}', pre_title=pre_title)
        ax = rh.add_axis(ylabel='Height [km]')

        # Custom colormap, because 'Blues' does not start fully at white
        colors = [(1, 1, 1), (0.5, 0.75, 0.9), (0, 0.2, 0.5)]  # RGB values: white -> blue
        blue_cmap = LinearSegmentedColormap.from_list("white_to_blue", colors)

        ax.plot_filled_contours(data.time, data.height/1000, data['rh'].values.T, 'Relative Humidity [%]', blue_cmap, vmin=0, vmax=100)

        rain_flag_mask = data['rain_flag'] > 0
        ax.add_hatches(rain_flag_mask.time.values, rain_flag_mask.values)

        rh.add_gray_stripes(alpha=1)
        return rh

    @staticmethod
    def ah(date, data, pre_title=None, retrieval_label='MIM-Retrieval'):
        ah = TimeSeriesPlot(date, f'Absolute Humidity - {retrieval_label}', pre_title=pre_title)
        ax = ah.add_axis(ylabel='Height [km]')

        # Custom colormap, because 'Blues' does not start fully at white
        colors = [(1, 1, 1), (0.5, 0.75, 0.9), (0, 0.2, 0.5)]  # RGB values: white -> blue
        blue_cmap = LinearSegmentedColormap.from_list("white_to_blue", colors)

        ax.plot_filled_contours(data.time, data.height/1000, data['ah'].values.T, 'Absolute Humidity [g/m³]', blue_cmap, vmin=0)

        rain_flag_mask = data['rain_flag'] > 0
        ax.add_hatches(rain_flag_mask.time.values, rain_flag_mask.values)

        ah.add_gray_stripes(alpha=1)
        return ah

    @staticmethod
    def lwc(date, data, pre_title=None, retrieval_label='MIM-Retrieval'):
        lwc = TimeSeriesPlot(date, f'Liquid Water Content - {retrieval_label}', pre_title=pre_title)
        ax = lwc.add_axis(ylabel='Height [km]')

        # Custom colormap, because 'Blues' does not start fully at white
        colors = [(1, 1, 1), (0.5, 0.75, 0.9), (0, 0.2, 0.5)]  # RGB values: white -> blue
        blue_cmap = LinearSegmentedColormap.from_list("white_to_blue", colors)

        ax.plot_filled_contours(data.time, data.height/1000, data['lwc'].values.T, 'Liquid Water Content [g/m³]', blue_cmap, contours=False, vmin=0)

        rain_flag_mask = data['rain_flag'] > 0
        ax.add_hatches(rain_flag_mask.time.values, rain_flag_mask.values)

        lwc.add_gray_stripes(alpha=1)
        return lwc
    

def plot_err(err_ds_height, label_dic=None, err='RMSE', zero_xlim=True, xlabel_T='Temperature in K', xlabel_RH='Relative Humidity in %', xlabel_LWC='LWC in g/m³', colors=None):
    """
    Plot vertical error profiles (e.g., RMSE, MAE) for multiple models.

    Three subplots are generated if liquid water content (``lwc``) is present,
    otherwise two. Each subplot shows the error of one variable (T, RH, LWC)
    against height for all models in the dataset. Colors are kept consistent
    across subplots.

    Parameters
    ----------
    err_ds_height : xarray.Dataset
        Dataset containing error profiles per model with dimensions
        ``height`` and ``model``. Expected variables include ``T`` and ``rh``,
        and optionally ``lwc``.
    label_dic : dict, optional
        Mapping from model names to legend labels. If None, model names are
        used directly.
    err : str, optional
        Error metric name used as the figure title. Default is ``'RMSE'``.
    zero_xlim : bool, optional
        If True and ``err`` ∈ {``RMSE``, ``MAE``}, force x-limits to start at
        zero. Otherwise a vertical zero line is drawn.
    xlabel_T : str, optional
        X-axis label for the temperature subplot.
    xlabel_RH : str, optional
        X-axis label for the relative humidity subplot.
    xlabel_LWC : str, optional
        X-axis label for the LWC subplot (if present).
    colors : dict, optional
        Optional dictionary mapping models to plot colors. New models are added
        with their automatically chosen Matplotlib color.

    Returns
    -------
    dict
        Dictionary mapping models to the colors used for plotting for later use.
    """
    if label_dic is None:
        label_dic = {str(model.values): str(model.values) for model in err_ds_height.model}

    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(12, 8))
    plt.suptitle(err, y=0.92)
    
    if 'lwc' in err_ds_height:
        plt.subplot(1, 3, 1)
    else:
        plt.subplot(1, 2, 1)

    if colors is None:
        colors = {}

    for model in err_ds_height.model.values:
        line = plt.plot(err_ds_height.T.sel(model=model), err_ds_height.height/1000, label=label_dic[model], color=colors.get(model, None))
        colors[model] = line[0].get_color()

    
    if err in ['RMSE','MAE'] and zero_xlim:
        plt.xlim(left=0)
    else:
        plt.axvline(x=0, color='black')
    plt.xlabel(xlabel_T)
    plt.ylabel("Height above ground in km")
    plt.ylim(bottom=0)
    plt.grid()

    if 'lwc' in err_ds_height:
        plt.subplot(1, 3, 2)
    else:
        plt.subplot(1, 2, 2)

    for model in err_ds_height.model.values:
        plt.plot(err_ds_height.rh.sel(model=model), err_ds_height.height/1000, label=label_dic[model], color=colors[model])

    if err in ['RMSE','MAE'] and zero_xlim:
        plt.xlim(left=0)
    else:
        plt.axvline(x=0, color='black')
    plt.xlabel(xlabel_RH)
    plt.ylim(bottom=0)
    plt.grid()

    if 'lwc' in err_ds_height:
        plt.subplot(1, 3, 3)
        for model in err_ds_height.model.values:
            plt.plot(err_ds_height.lwc.sel(model=model), err_ds_height.height/1000, label=label_dic[model], color=colors[model])

        if err in ['RMSE','MAE'] and zero_xlim:
            plt.xlim(left=0)
        else:
            plt.axvline(x=0, color='black')
        plt.xlabel(xlabel_LWC)
        plt.ylim(bottom=0)
        plt.grid()

    plt.legend()
    plt.show()
    return colors

def plot_scatter_lwp(pred, data):
    """
    Plot scatter plots comparing predicted and form radiosonde profiles calculated Liquid Water Path (LWP).
    """

    combined_ds = xr.Dataset({'pred_lwp': pred.lwp, 'data_lwp': data.lwp})
    
    plt.figure(figsize=(10, 5))
    plt.suptitle(pred.model.values, y=0.95)

    plt.subplot(1, 2, 1)
    for station in pred.station:
        plt.scatter(combined_ds.data_lwp.sel(station=station), combined_ds.pred_lwp.sel(station=station), s=3, edgecolor='none', facecolor='#1f77b4')
    plt.plot(np.arange(combined_ds.pred_lwp.max()), color='red')
    plt.ylabel('Predicted LWP in g/m²')
    plt.xlabel('Radiosonde LWP in g/m²')
    plt.grid()

    plt.subplot(1, 2, 2)
    for station in pred.station:
        plt.scatter(combined_ds.data_lwp.sel(station=station), combined_ds.pred_lwp.sel(station=station), s=3, edgecolor='none', facecolor='#1f77b4')
    plt.plot(np.arange(combined_ds.pred_lwp.max()), color='red')
    plt.ylabel('Predicted LWP in g/m²')
    plt.xlabel('Radiosonde LWP in g/m²')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylim(-20, 200)
    plt.xlim(-20, 200)
    plt.grid()
    plt.show()
