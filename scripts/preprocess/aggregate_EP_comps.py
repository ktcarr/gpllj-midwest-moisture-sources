"""Script converts raster PRISM files to netcdf for easier loading"""

import src.params
import src.utils
import xarray as xr
import os.path


def get_midwest_precip_prism(prism_data):
    """Get area-averaged precip in Midwest
    using the PRISM dataset."""

    ## Filepath for saving results
    fname = f"{src.params.DATA_RAW_FP}/prism_agg.nc"

    ## check to see if file already exists
    if os.path.isfile(fname):
        p_midwest = xr.open_dataarray(fname)

    ## if not, compute area-average
    else:
        ## Trim data around the Midwest for convenience
        lon_min, lat_min = src.utils.midwest_vertices.min(0)
        lon_max, lat_max = src.utils.midwest_vertices.max(0)
        prism_data_trim = prism_data.sel(
            longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min)
        )

        ## Average over midwest
        p_midwest = aggregate_data(prism_data_trim, dlat=1 / 24, dlon=1 / 24)
        p_midwest = p_midwest.squeeze().compute()

        ## save to file
        p_midwest.to_netcdf(fname)

    return p_midwest


def aggregate_data(data_mm, dlat, dlon):
    """sum data over masked areas and divide by area of Midwest"""

    ## Create mask for Midwest
    mask = src.utils.makeMask(
        vertices=src.utils.midwest_vertices,
        lat=data_mm["latitude"],
        lon=data_mm["longitude"],
        name="Midwest",
    )

    ## mask out great lakes
    ## assumes data_mm has NaNs over water
    land_mask = 1 + 0.0 * data_mm.isel(time=0).compute()
    mask = mask * land_mask

    # get area of each gridcell
    gridsize = src.utils.get_gridsize(
        lat=data_mm.latitude.values,
        lon=data_mm.longitude.values,
        dlat=dlat,
        dlon=dlon,
    )
    gridcell_area = gridsize["A"]

    # Get size of Midwest region in m^2
    midwest_size = (mask * gridcell_area).sum(["latitude", "longitude"])

    # convert height to volume (mm to mm * m^2)
    data_mm_m2 = data_mm * gridcell_area

    # Get totals for each masked region (mm * m^2)
    data_agg_mm_m2 = (data_mm_m2 * mask).sum(["latitude", "longitude"])

    # convert back to mm, by dividing (mm * m^2) by m^2
    data_agg_mm = data_agg_mm_m2 / midwest_size.values.item()

    # label array with units
    data_agg_mm.attrs.update({"units": "mm (over Midwest)"})

    return data_agg_mm


def get_midwest_evap_gleam(gleam_data):
    """Compute area-averaged evaporation in Midwest
    based on the GLEAM dataset"""

    ## Filepath for saving results
    fname = f"{src.params.DATA_RAW_FP}/gleam_agg.nc"

    ## Check if file exists
    if os.path.isfile(fname):
        e_midwest = xr.open_dataset(fname)

    else:
        e_midwest = aggregate_data(gleam_data["E"], dlat=0.25, dlon=0.25)
        e_midwest.to_netcdf(fname)

    return e_midwest


def main():
    #### PRISM ####
    print(f"\nAggregating PRISM over the Midwest")
    prism_data = xr.open_dataarray(f"{src.params.DATA_RAW_FP}/prism.nc")
    p_midwest = get_midwest_precip_prism(prism_data)

    #### GLEAM ####
    print("Aggregating GLEAM data over the Midwest")
    gleam_data = xr.open_dataset(f"{src.params.DATA_RAW_FP}/gleam.nc")
    e_midwest = get_midwest_evap_gleam(gleam_data)

    return


if __name__ == "__main__":
    main()
