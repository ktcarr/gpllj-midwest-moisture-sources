"""
TO-DO:
    - compute monthly and seasonal means of all variables
        - probably best to do this for all variables separately 
        (so that we don't have to re-run it if something changes)
    - also: daily detrend

"""

import pandas as pd
import xarray as xr
import src.utils
import src.params
import os.path


def aggregate_monthly(data, mean=True):
    """
    resamples data to monthly.
    Unlike xr.DataArray.resample, handles case where
    not all months are present in data (e.g., only AMJ).
    'mean' is a boolean specifying whether to take mean;
    if false, take the sum.
    """

    # define helper function to get monthly means/sums for single year of data
    if mean:
        aggregate_ = lambda data: data.groupby("time.month").mean()
    else:
        aggregate_ = lambda data: data.groupby("time.month").sum()

    # apply  to all years in dataset
    data_agg = data.groupby("time.year").map(aggregate_)

    # reset the "time" dimension (stack "year" and "month")
    data_agg = data_agg.stack(time=["year", "month"])

    # get new time index
    time_idx = [f"{t.year.item()}-{t.month.item():02d}-01" for t in data_agg.time]
    time_idx = pd.DatetimeIndex(time_idx)

    # assign time index to data
    data_agg = data_agg.drop_vars(["time", "year", "month"])
    data_agg["time"] = time_idx

    return data_agg


def prep(data, mean, daily_detrend, trim_month_range):
    """
    preprocess data. Inputs are:
        - data: xarray dataset
        - mean: boolean indicating whether to take mean (otherwise, take sum)
        - daily_detrend: boolean indicating whether to detrend daily data
        - trim_month_range: trim data to these months before processing
    Returns dictionary with the following:
        - monthly mean
        - detrended monthly mean
        - seasonal mean
        - detrended seasonal mean
        - detrended daily data
    """
    ## trim data to specified month range
    data_trimmed = src.utils.get_month_range(data, *trim_month_range)

    ## compute monthly means/sums
    data_monthly = aggregate_monthly(data_trimmed, mean=mean)

    ## compute seasonal means/sums
    data_seasonal = {}
    seasons = ["amj", "mjj", "jja"]
    for season, month_range in zip(seasons, [[4, 6], [5, 7], [6, 8]]):
        data_seasonal_ = src.utils.get_month_range(data_monthly, *month_range)
        data_seasonal_ = data_seasonal_.groupby("time.year")
        if mean:
            data_seasonal[season] = data_seasonal_.mean()
        else:
            data_seasonal[season] = data_seasonal_.sum()

    ## Detrend
    detrend = src.utils.detrend_dim  # function used to detrend
    monthly_detrend = data_monthly.groupby("time.month").map(detrend)
    seasonal_detrend = {}
    for season in seasons:
        seasonal_detrend[season] = detrend(data_seasonal[season], dim="year")

    ## Put results in dictionary
    data_prepped = {"monthly": data_monthly, "monthly_detrend": monthly_detrend}
    for season in seasons:
        data_prepped[f"seasonal_{season}"] = data_seasonal[season]
        data_prepped[f"seasonal_detrend_{season}"] = seasonal_detrend[season]

    ## Optional: daily_detrend:
    if daily_detrend:
        print("Doing daily detrend....")
        data_prepped["daily_detrend"] = data_trimmed.groupby("time.dayofyear").map(
            detrend
        )
        print("Done with daily detrend.")

    return data_prepped


def prep_and_save(
    name,
    load_fp=src.params.DATA_RAW_FP,
    trim_month_range=(3, 9),
    mean=True,
    daily_detrend=True,
):
    """convenience function to load, prep, and save prepped data"""

    ## Check if data has already been pre-processed
    if os.path.isfile(f"{src.params.DATA_PREPPED_FP}/{name}_monthly.nc"):
        print(f"\nData for {name} has already been prepped.")
        return

    else:
        print(f"\nPre-processing {name}")

        ## Load the data
        data = xr.open_dataset(f"{load_fp}/{name}.nc")

        ## Remove unnecessary variables if they exist
        vars_to_drop = ["time_bnds", "spatial_ref"]
        vars_to_drop = [x for x in vars_to_drop if x in list(data)]
        data = data.drop_vars(vars_to_drop)

        ## Do the pre-processing
        data_prepped = prep(
            data,
            mean=mean,
            daily_detrend=daily_detrend,
            trim_month_range=trim_month_range,
        )

        ## Save to file
        for item in list(data_prepped):
            data_prepped[item].to_netcdf(
                f"{src.params.DATA_PREPPED_FP}/{name}_{item}.nc"
            )
        return


if __name__ == "__main__":
    from os.path import join
    import numpy as np
    from progressbar import progressbar

    ###### Compute GPLLJ index using Kirtman region/index ###

    if os.path.isfile(f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg.nc"):
        pass

    else:
        print(f"\nAveraging UVQ over GPLLJ region")

        # load data
        uvq_gpllj = xr.open_dataset(f"{src.params.DATA_RAW_FP}/uvq_gpllj.nc")

        # Select Kirtman region
        uvq_gpllj_avg = uvq_gpllj.sel(
            level=850,
            latitude=slice(35, 25),
            time=slice(None, "2021-03-31"),
            longitude=slice(258, 263),
        )

        # Then avg over lon/lat
        uvq_gpllj_avg = uvq_gpllj_avg.mean(["latitude", "longitude"])

        # Resample to daily
        uvq_gpllj_avg_daily = uvq_gpllj_avg.resample(time="1D").mean()

        # save to file
        uvq_gpllj_avg_daily.to_netcdf(f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg.nc")

    ####### Pre-process data #####################
    prep_and_save(
        "uvq_gpllj_avg", load_fp=src.params.DATA_PREPPED_FP, trim_month_range=(1, 12)
    )
    prep_and_save("wam_fluxes_trim", load_fp=src.params.DATA_PREPPED_FP, mean=False)
    prep_and_save(
        "wam_fluxes_agg",
        load_fp=src.params.DATA_PREPPED_FP,
        mean=False,
        trim_month_range=(1, 12),
    )
    prep_and_save("wam_storage_trim", load_fp=src.params.DATA_PREPPED_FP)
    prep_and_save("Z")
    prep_and_save("uvq_wide", daily_detrend=False)
    prep_and_save("vq_cross", daily_detrend=False)
    prep_and_save("helmholtz", load_fp=src.params.DATA_PREPPED_FP)
    prep_and_save("sm")
    prep_and_save("sp", daily_detrend=False)
    prep_and_save("tcwv", daily_detrend=False)
    prep_and_save("ivt", daily_detrend=False)
    prep_and_save("prism", mean=False)
    prep_and_save("prism_agg", mean=False)
    prep_and_save("gleam", mean=False, daily_detrend=False)
    prep_and_save("gleam_agg", mean=False, daily_detrend=False)

    ################# Classify GPLLJ days ####################

    if os.path.isfile(f"{src.params.DATA_PREPPED_FP}/times_v850.nc"):
        pass
    else:
        print(f"\nCompute GPLLJ indices using Kirtman region")
        # Load GPLLJ data
        fp = f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg_daily_detrend.nc"
        gpllj_idx = xr.open_dataset(fp)["v"]

        # Standardize (to give variance of 1)
        gpllj_idx = (
            gpllj_idx.groupby("time.dayofyear")
            / gpllj_idx.groupby("time.dayofyear").std()
        )

        # Get times where threshold is exceeded
        thresh = 1  # threshold for GPLLJ event (standard deviations)
        times_v850 = gpllj_idx[gpllj_idx > thresh].time.values
        times_v850 = pd.DatetimeIndex(times_v850)
        np.save(
            f"{src.params.DATA_PREPPED_FP}/times_v850.npy", times_v850
        )  ## save times to file

        # Look at opposite case (weak GPLLJ)
        times_v850_neg = gpllj_idx[gpllj_idx < -thresh].time.values
        times_v850_neg = pd.DatetimeIndex(times_v850_neg)
        np.save(
            f"{src.params.DATA_PREPPED_FP}/times_v850_neg.npy", times_v850_neg
        )  ## save times to file

    ###### Evaluate Bonner-Whitman criterion for GPLLJ days ####
    if os.path.isfile(f"{src.params.DATA_PREPPED_FP}/times_coupled.npy"):
        pass
    else:
        print("Get list of coupled/uncoupled GPLLJ days")
        ## load in GPLLJ data
        uvq_gpllj = xr.open_dataset(f"{src.params.DATA_RAW_FP}/uvq_gpllj.nc")

        ## trim to specified area
        is_6_utc = uvq_gpllj.time.dt.hour == 6
        lat_range = [42, 30]
        lon_range = [258, 268]
        v_trimmed = uvq_gpllj.v.sel(
            time=is_6_utc, latitude=slice(*lat_range), longitude=slice(*lon_range)
        )
        shear = v_trimmed.sel(level=850) - v_trimmed.sel(level=700)

        #### Load wave activity data #####
        cwa = xr.open_dataarray(f"{src.params.DATA_PREPPED_FP}/cwa.nc")

        # Get dates of jet events which satisfy BW jet criterion
        bw_percentile = 75
        bw_area_prop = (
            0.1  # setting as .25 leads to a more "reasonable" proportion of jet events
        )

        cwa_percentile = 66
        cwa_area_prop = 1 / 3

        times_coupled = []
        times_uncoupled = []
        for m in progressbar(range(4, 9)):
            # Apply Bonner-Whiteman criteria
            is_m = shear.time.dt.month == m
            shear_ = shear.sel(time=is_m)
            v_ = v_trimmed.sel(time=is_m, level=850)
            shear_thresh = np.percentile(shear_.values, q=bw_percentile, axis=0)
            v_thresh = np.percentile(v_.values, q=bw_percentile, axis=0)
            cell_bool = (shear_ > shear_thresh) & (v_ > v_thresh)
            prop_of_cells = cell_bool.mean(["latitude", "longitude"])
            times_ = shear_.time.sel(time=prop_of_cells.values > bw_area_prop)

            # Apply LWA criteria
            cwa_ = cwa.sel(time=times_)
            cwa_thresh = np.percentile(cwa_.values, q=cwa_percentile, axis=0)
            prop_of_cells = (cwa_ > cwa_thresh).mean(["latitude", "longitude"])
            times_coupled_ = cwa_.time.sel(time=prop_of_cells.values > cwa_area_prop)
            times_uncoupled_ = cwa_.time.sel(time=prop_of_cells.values <= cwa_area_prop)

            # Append to list
            times_coupled += list(times_coupled_.values)
            times_uncoupled += list(times_uncoupled_.values)

        # Sort, and convert to datetime index
        times_coupled = pd.DatetimeIndex(sorted(times_coupled))
        times_uncoupled = pd.DatetimeIndex(sorted(times_uncoupled))

        # Adjust label on the time series to make comparison to other things easier
        times_coupled = times_coupled - pd.Timedelta("6h")
        times_uncoupled = times_uncoupled - pd.Timedelta("6h")

        np.save(f"{src.params.DATA_PREPPED_FP}/times_coupled.npy", times_coupled)
        np.save(f"{src.params.DATA_PREPPED_FP}/times_uncoupled.npy", times_uncoupled)
