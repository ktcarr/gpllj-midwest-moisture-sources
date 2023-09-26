import xarray as xr
import numpy as np
import src.params
import tqdm


def phi_area(phi):
    """Area north of a given latitude, in meters"""
    R = 6.37e6  # radius of earth
    return 2 * np.pi * R**2 * (1 - np.sin(phi))


def lwa_vec(z):
    """Compute LWA given Z500 field. Return dictionary with CWA/AWA/LWA, and other fields"""
    ## Constants
    R = 6.37e6  # radius of earth (m)# Remove data at pole (for numerical reasons)
    dtheta = np.deg2rad(1.0)
    dphi = np.deg2rad(1.0)
    phi = np.deg2rad(z.latitude.values)
    dx = R * np.cos(phi) * dtheta
    dy = R * dphi * np.ones_like(z.isel(time=0).values)
    dA = dx[:, None] * dy

    # print("Computing phi areas")
    phi_areas = phi_area(phi)  # Compute area of phi and Z500 contours
    check_phi_cwa = phi[:, None] < phi[None, :]

    # print("Computing z areas")
    N = 180  # number of Z contours to check
    z_vals = np.linspace(5e3, 6e3, N)
    z_areas = (z.values[..., None] < z_vals) * dA[..., None]
    z_areas = np.sum(z_areas, (1, 2))

    # print("Obtaining Z_eq")
    diff = phi_areas[None, ...] - z_areas[..., None]
    ze_vals = z_vals[np.argmin(diff**2, axis=1)]

    # print("Compute z anomalies")
    z_prime = z.values[..., None] - ze_vals[:, None, None, ...]
    check_z_cwa = z_prime < 0

    #print("Compute CWA and AWA")
    # Get areas to integrate over
    cwa_idx = check_z_cwa & check_phi_cwa[:, None, :]  # todo: fix colon on this line
    awa_idx = (~check_z_cwa) & (~check_phi_cwa[:, None, :])

    ## Print compute inside of integral
    cwa = -z_prime * np.cos(phi)[None, :, None, None] * cwa_idx * dphi
    awa = z_prime * np.cos(phi)[None, :, None, None] * awa_idx * dphi

    # Integrate and switch dimenions to (time, lat, lon)
    cwa = np.transpose(cwa.sum(1), (0, 2, 1))
    awa = np.transpose(awa.sum(1), (0, 2, 1))

    # multiply by radius of Earth and weight by latitude
    cwa *= R / np.cos(phi)[None, :, None]
    awa *= R / np.cos(phi)[None, :, None]
    lwa = cwa + awa

    # print('Filtering bad values')
    # is_valid = (z.values[...,None] > z_vals).any(1).all(1)
    # z_max = z_vals[is_valid.sum(1)]
    # is_valid_ze = ze_vals < z_max[:,None]
    # cwa[~is_valid_ze] = np.nan
    # awa[~is_valid_ze] = np.nan
    # lwa               = cwa + awa

    print("Put in DataArray")
    coords = {"latitude": z.latitude, "longitude": z.longitude, "time": z.time}
    dims = ["time", "latitude", "longitude"]
    lwa = xr.DataArray(lwa, coords=coords, dims=dims)
    cwa = xr.DataArray(cwa, coords=coords, dims=dims)
    awa = xr.DataArray(awa, coords=coords, dims=dims)

    coords["lat_idx"] = z.latitude
    dims.append("lat_idx")
    cwa_idx = xr.DataArray(cwa_idx, coords=coords, dims=dims)
    awa_idx = xr.DataArray(awa_idx, coords=coords, dims=dims)
    lwa_idx = xr.DataArray(cwa_idx | awa_idx, coords=coords, dims=dims)
    z_equiv = xr.DataArray(
        ze_vals,
        coords={"time": z.time, "latitude": z.latitude},
        dims=["time", "latitude"],
    )

    ## Put everything in single dataarray
    results = xr.merge(
        [
            lwa.rename("lwa"),
            cwa.rename("cwa"),
            awa.rename("awa"),
            lwa_idx.rename("lwa_idx"),
            cwa_idx.rename("cwa_idx"),
            awa_idx.rename("awa_idx"),
            z_equiv.rename("z_equiv"),
        ]
    )

    return results


def get_lwa_oneyear(year):
    """
    Compute LWA for single year of data
    """

    ## Load geopotential data
    z = xr.open_dataarray(f"{src.params.DATA_RAW_FP}/Z_500.nc")
    z = z.sel(time=f"{year}", latitude=slice(89, 0))
    z = z.squeeze("level", drop=True).compute()

    ## Convert from geopotential to geopotential height
    g = 9.80665  # acceleration due to gravity (m/s^2)
    z = z / g  # convert to geopotential height (units: meters)

    ## Compute LWA and trim
    lwa = lwa_vec(z) 
    lwa = lwa.sel(latitude=slice(42, 30), longitude=slice(240, 258))
    
    return lwa


def main():
    ## Loop through years of data, computing LWA
    lwa = []
    for year in tqdm.tqdm(np.arange(1979, 2021)):
        lwa.append(get_lwa_oneyear(year)[["cwa", "awa"]])
    lwa = xr.concat(lwa, dim="time")

    ## save to file
    lwa["cwa"].to_netcdf(f"{src.params.DATA_PREPPED_FP}/cwa.nc")
    lwa["awa"].to_netcdf(f"{src.params.DATA_PREPPED_FP}/awa.nc")

    return


if __name__ == "__main__":
    main()
