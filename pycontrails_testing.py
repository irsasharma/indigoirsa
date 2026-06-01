import numpy as np
import pandas as pd
import xarray as xr
from pycontrails.core import Flight, MetDataset, models
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.core.fuel import Fuel
from pycontrails.core.fuel import SAFBlend
from pycontrails.models.ps_model import PSFlight
from pycontrails.models.emissions import Emissions


df = pd.DataFrame()
df["longitude"] = np.linspace(0, 50, 100)
df["latitude"] = np.linspace(0, 10, 100)
df["altitude"] = 11000
df["time"] = pd.date_range("2022-03-01 00:00:00", "2022-03-01 02:00:00", periods=100)

fl = Flight(data=df, flight_id="ABC")

print("Flight object created:")
print(fl)
fl.attrs["aircraft_type"] = "A320"

print("\nFlight altitude:", fl["altitude"][0])
print("Number of trajectory points:", len(fl))


# defining met dataset 
met_standard = xr.Dataset (
    data_vars= {
        "eastward_wind": (
            ("time", "level", "latitude", "longitude"), 
            #used this to set atmospheric variables
            np.full((24, 2, 20, 20), 10.0),
            #to set constant met values 
        ),
        "northward_wind": (
            ("time", "level", "latitude", "longitude"),
            np.full((24, 2, 20, 20), 10.0),
        ),
       
       "air_temperature": (
            ("time", "level", "latitude", "longitude"),
            np.full((24, 2, 20, 20), 233.0),
       ),
       #relative humidity
       "specific_humidity": (
            ("time", "level", "latitude", "longitude"),
            np.full((24, 2, 20, 20), 0.0002),
        ),
    },


    coords= {"longitude": np.linspace(-10, 60, 20),
        "latitude": np.linspace(-10, 20, 20),
        "level": [250, 200],
        "time": pd.date_range(start= "2022-03-01", end= "2022-03-02", periods=24),
    }

    )

met = MetDataset(met_standard)
print(met)

saf=SAFBlend(0.3)
fl_saf=fl.copy()

fl_saf.attrs["aircraft_type"] = "A320"
fl_saf.fuel = saf

fl.downselect_met(met)
fl_saf.downselect_met(met)
fl["air_temperature"] = models.interpolate_met(met, fl, "air_temperature")
fl["specific_humidity"] = models.interpolate_met(met, fl, "specific_humidity")
fl["true_airspeed"] = fl.segment_groundspeed()

fl_saf["air_temperature"] = models.interpolate_met(met, fl_saf, "air_temperature")
fl_saf["specific_humidity"] = models.interpolate_met(met, fl_saf, "specific_humidity")
fl_saf["true_airspeed"] = fl_saf.segment_groundspeed()

# get aircraft performance data using ps model
ps = PSFlight(met=met)
fl = ps.eval(fl)
fl_saf = ps.eval(fl_saf)

#calculate emissions from flights
emissions = Emissions()
fl = emissions.eval(fl)
fl_saf = emissions.eval(fl_saf)

#saf simulation next
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print((fl.dataframe["nvpm_ei_n"]-fl_saf.dataframe["nvpm_ei_n"]) / fl.dataframe["nvpm_ei_n"])
