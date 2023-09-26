"""Script converts raster PRISM files to netcdf for easier loading"""

import src.params
import src.utils
import numpy as np
import pandas as pd
import xarray as xr
import progressbar
import os.path
import argparse


def open_prism_file(prism_fp, year, month, day):
    """open single day's worth of prism data"""

    ## Get filepath
    # prism_fp = f"/vortexfs1/share/clidex/data/obs/hydro/PRISM/prism"
    folder_name = f"PRISM_ppt_stable_4kmD2_{year}0101_{year}1231_bil"
    file_name = f"PRISM_ppt_stable_4kmD2_{year}{month:02d}{day:02d}_bil.bil"
    fp = f"{prism_fp}/{folder_name}/{file_name}"

    ## Open data and fix labels
    data = xr.open_dataarray(fp, engine="rasterio").rename("P")
    data = data.rename({"y": "latitude", "x": "longitude", "band": "time"})
    data["time"] = pd.DatetimeIndex([f"{year}-{month}-{day}"])

    ## Round latitude and longitude values
    ## different files have slightly different values
    data["latitude"] = np.round(data.latitude, 5)
    data["longitude"] = np.round(data.longitude, 5)

    ## shift longitude from [-180,180) to [0,360)
    data["longitude"] = data["longitude"] + 360

    ## label units
    data.attrs.update({"units": "mm"})

    return data


def load_prism_data_month(prism_fp, year, month):
    """Load single month of prism data"""

    ## Get list of dates for the given month
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month+1:02d}-01"
    dates = pd.date_range(start=start_date, end=end_date, freq="1D")[:-1]

    ## Load data for each day
    data = [open_prism_file(prism_fp, year, month, d) for d in dates.day]
    data = xr.concat(data, dim="time")
    return data


def load_prism_data_year(prism_fp, year, months=None):
    """Load prism data for given year (only include given months)"""
    if months is None:
        months = np.arange(1, 13)

    data = [load_prism_data_month(prism_fp, year, m) for m in months]
    data = xr.concat(data, dim="time")
    return data


def load_prism_data(prism_fp, years, months=None, chunkby="lonlat"):
    """Load prism data for specified years and months"""

    ## Specify 'months' if necessary (default to all months)
    if months is None:
        months = np.arange(1, 13)

    ## Parse chunking input
    if chunkby is None:
        chunks = {}
    elif chunkby == "lonlat":
        chunks = {"latitude": 300, "longitude": 300}
    elif chunkby == "time":
        chunks = {"time": 91}

    ## Load data for each year sequentially, and concatenate
    data = [
        load_prism_data_year(prism_fp, y, months).chunk(chunks)
        for y in progressbar.progressbar(years)
    ]
    data = xr.concat(data, dim="time").drop("spatial_ref")

    ## Un-chunk dimensions other than the one specified in 'chunkby'
    if (chunkby == "lonlat") | (chunkby is None):
        data = data.chunk({"time": len(data.time)})
    elif chunkby == "time":
        data = data.chunk(
            {"longitude": len(data.longitude), "latitude": len(data.latitude)}
        )

    return data

    
def move_prism_data(in_fp):
    """wrapper function converts .bil files to single .nc file.
    Function returns xr.dataset containing the prism data"""

    ## Filename for saving .nc file
    fname = f"{src.params.DATA_RAW_FP}/prism.nc"

    ## First, check if file already exists
    if os.path.isfile(fname):
        return xr.open_dataarray(fname)

    ## If not, proceed
    else:
        print("Opening PRISM data...")
        extended_range = [src.params.MONTH_RANGE[0] - 1, src.params.MONTH_RANGE[1] + 1]
        months = np.arange(extended_range[0], extended_range[1] + 1)
        data = load_prism_data(
            prism_fp=in_fp, years=np.arange(1981, 2021), months=months, chunkby=None
        )
        print("Loading into memory...")
        data.load()
        print("Done.")

        ## save to file
        data.to_netcdf(fname)

        return data


def fix_gleam_longitudes(gleam_data):
    """convert gleam longitudes from [-180,180) to [0,360)"""

    ## Find and update negative longitudes
    neg_idx = gleam_data.lon.values < 0
    gleam_data.lon.values[neg_idx] = gleam_data.lon.values[neg_idx] + 360
    gleam_data = gleam_data.roll({"lon": 720}, roll_coords=True)

    return gleam_data


def move_gleam_data(in_fp):
    """Move data from server (in_fp) to data directory (out_fp)."""

    ## check to see if file exists:
    out_fp = f"{src.params.DATA_RAW_FP}/gleam.nc"
    if os.path.isfile(out_fp):
        return xr.open_dataset(out_fp)

    else:
        ## Open GLEAM data
        gleam = xr.open_dataset(in_fp)

        ## update longitude coordinates
        gleam = fix_gleam_longitudes(gleam)

        ## trim data in space
        gleam = gleam.sel(lat=slice(60, 5), lon=slice(210, 330))

        ## rename coordinates to match ERA5
        gleam = gleam.rename({"lon": "longitude", "lat": "latitude"})

        ## save to file
        gleam.to_netcdf(out_fp)

        return gleam


def main():
    """Load the .bil files sequentially, then save as single netcdf file."""

    ## Parse user-supplied arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prism_fp",
        default=f"/vortexfs1/share/clidex/data/obs/hydro/PRISM/prism",
        type=str,
    )
    parser.add_argument(
        "--gleam_fp",
        default=f"/vortexfs1/share/clidex/data/obs/hydro/GLEAM/E_1980-2022_GLEAM_v3.7a_MO.nc",
        type=str,
    )
    args = parser.parse_args()

    #### PRISM ####
    ## Convert .bil files to netcdf
    print("Converting PRISM '.bil' files to .nc")
    prism_data = move_prism_data(in_fp=args.prism_fp)

    #### GLEAM ####
    print("Moving GLEAM data to project directory")
    gleam_data = move_gleam_data(in_fp=args.gleam_fp) 

    return


if __name__ == "__main__":
    main()
