import src.utils
import src.params


def trim(data):
    """trim data to specified months and lon/lats"""

    ## Specify lon/lat/time ranges
    lon_range = [210, 330]
    lat_range = [60, 5]
    month_range = [2, 9]

    ## trim in space
    data_ = data.sel(
        longitude=slice(*lon_range),
        latitude=slice(*lat_range),
    )

    ## trim in time
    data_ = src.utils.get_month_range(data_, *month_range)

    return data_


def convert_m3_to_m(data):
    """convert units from m^3 to m, by dividing by gridcell area.
    TO-DO: this function messes up variables which are
    not on a 2-D grid (e.g., north_loss and south_loss)"""
    # Get size of each gridcell (m^2)
    gridsize = src.utils.get_gridsize(
        lat=data.latitude.values,
        lon=data.longitude.values,
        dlat=1.0,
        dlon=1.0,
    )
    gridcell_area = gridsize["A"]

    # divide data (m^3) by gridcell size (m^2)
    return data / gridcell_area


def aggregate(data):
    """sum data over masked areas and divide by area of Midwest"""

    ###### Compute spatially aggregated quantities
    masks = src.utils.get_masks(
        lat=data.latitude.values,
        lon=data.longitude.values,
    )

    # get area of each gridcell
    gridsize = src.utils.get_gridsize(
        lat=data.latitude.values,
        lon=data.longitude.values,
        dlat=1.0,
        dlon=1.0,
    )
    gridcell_area = gridsize["A"]

    # Get size of Midwest region in m^2
    midwest_size = (masks.sel(mask="Midwest") * gridcell_area).sum(
        ["latitude", "longitude"]
    )

    # Get totals for each masked region (m^3)
    data_agg = (data * masks).sum(["latitude", "longitude"])

    # Convert to m, dividing (m^3) by (m^2)
    return data_agg / midwest_size.values.item()


if __name__ == "__main__":
    import xarray as xr
    import numpy as np
    import os
    import shutil

    ## create temporary folder for intermediate results
    temp_fp = f"{src.params.DATA_RAW_FP}/temp"
    if os.path.isdir(temp_fp):
        print("Warning: temporary directory already exists!")
    else:
        os.mkdir(temp_fp)

    ## Process each year of data sequentially
    for year in np.arange(1979, 2021):
        print(f"Processing year {year}")

        for vartype in ["fluxes", "storage"]:
            ## Load data
            data = xr.open_dataset(f"{os.environ['WAM_FP']}/{vartype}_daily_{year}.nc")

            # Trim to smaller area, and convert from m^3 to m
            data_trim = trim(data).compute()
            data_trim = convert_m3_to_m(data_trim)
            data_trim.to_netcdf(f"{temp_fp}/wam_{vartype}_trim_{year}.nc")

            # Aggregate over specified regions
            data_agg = aggregate(data)
            data_agg.to_netcdf(f"{temp_fp}/wam_{vartype}_agg_{year}.nc")

    print("Merging data")
    for vartype in ["fluxes", "storage"]:
        ## merge data
        data_trim_all = xr.open_mfdataset(
            f"{temp_fp}/wam_{vartype}_trim_*.nc"
        ).compute()
        data_agg_all = xr.open_mfdataset(f"{temp_fp}/wam_{vartype}_agg_*.nc").compute()

        ## save to file
        data_trim_all.to_netcdf(f"{src.params.DATA_PREPPED_FP}/wam_{vartype}_trim.nc")
        data_agg_all.to_netcdf(f"{src.params.DATA_PREPPED_FP}/wam_{vartype}_agg.nc")

    ## Delete temporary folder
    shutil.rmtree(temp_fp)
