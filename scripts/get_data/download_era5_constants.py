import cdsapi
import src.params

## download ERA5 data
c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": "lake_cover",
        "year": "2000",
        "month": "05",
        "day": "15",
        "time": "12:00",
        "grid": "1.0/1.0",
    },
    f"{src.params.DATA_CONST_FP}/lake_cover.nc",
)

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": "land_sea_mask",
        "year": "2000",
        "month": "05",
        "day": "15",
        "time": "12:00",
        "grid": "1.0/1.0",
    },
    f"{src.params.DATA_CONST_FP}/lsm.nc",
)

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": "geopotential",
        "year": "2000",
        "month": "05",
        "day": "15",
        "time": "12:00",
        "grid": "1.0/1.0",
    },
    f"{src.params.DATA_CONST_FP}/Z_surf_lores.nc",
)

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": "geopotential",
        "year": "2000",
        "month": "05",
        "day": "15",
        "time": "12:00",
        "grid": "0.25/0.25",
    },
    f"{src.params.DATA_CONST_FP}/Z_surf_hires.nc",
)
