"""
Script produces the following figures:
    - "synoptic_tracked-moisture.png"
    - "synoptic_precip.png"
    - "synoptic_precip_vec.png"
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import xarray as xr
import progressbar
import pandas as pd
import seaborn as sns
import numpy as np
import dask
import matplotlib.pyplot as plt
import cmocean
import warnings
import shapely.errors
import os
import src.utils
import src.params
import argparse


## Functions to load data
def load_wam_data(month_range):
    """Load output from moisture tracking model"""

    print("Loading WAM data...")

    ## open data
    wam_data = xr.open_mfdataset(
        [
            f"{src.params.DATA_PREPPED_FP}/wam_fluxes_trim_daily_detrend.nc",
            f"{src.params.DATA_PREPPED_FP}/wam_storage_trim_daily_detrend.nc",
        ]
    ).compute()

    ## rename level to avoid confusion with geopotential height
    wam_data = wam_data.rename({"level": "wam_level"})

    ## convert from m to mm
    mm_per_m = 1000.0
    wam_data = wam_data * mm_per_m

    ## trim to month range
    wam_data = src.utils.trim_and_fix_dates(wam_data, month_range)

    return wam_data


def load_prism_data(month_range):
    """Load PRISM data"""

    print("Loading PRISM data...")

    ## ignore large chunksize warning
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        prism = xr.open_dataset(
            f"{src.params.DATA_PREPPED_FP}/prism_daily_detrend.nc",
            chunks={"latitude": 300, "longitude": 300},
        )
        prism = prism.rename(
            {"P": "prism", "latitude": "latitude_prism", "longitude": "longitude_prism"}
        )

        ## trim to month range
        prism = src.utils.trim_and_fix_dates(prism, month_range)

    return prism


def load_geopotential_height(month_range):
    """Load geopotential height data"""

    Z = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/Z_daily_detrend.nc")
    Z /= 9.8  # convert to geopotential meters

    ## trim to month range
    Z = src.utils.trim_and_fix_dates(Z, month_range)

    return Z


def load_helmholtz(month_range):
    """helmholtz decomposition of IVT"""

    helmholtz = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/helmholtz_daily_detrend.nc"
    )

    ## trim to month range
    helmholtz = src.utils.trim_and_fix_dates(helmholtz, month_range)

    return helmholtz


def load_soil_moisture(month_range):
    """Load soil moisture data from ERA5"""

    sm = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/sm_daily_detrend.nc")
    sm = sm["swvl1"] * 70  # convert from m^3/m^3 to mm (soil layer is 7cm=70mm)
    sm = sm.to_dataset().rename({"lon": "longitude", "lat": "latitude"})

    ## trim to month range
    sm = src.utils.trim_and_fix_dates(sm, month_range=month_range)

    return sm


def load_data(month_range, load_prism):
    """Function to load all datasets combined"""

    ## Get expanded month range (to take lagged composites)
    expanded_month_range = (month_range[0] - 1, month_range[1] + 1)

    ## ignore large chunksize warning
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        if load_prism:
            ## Load each dataset
            data = xr.merge(
                [
                    load_wam_data(expanded_month_range),
                    load_prism_data(expanded_month_range),
                    load_geopotential_height(expanded_month_range),
                    load_helmholtz(expanded_month_range),
                    load_soil_moisture(expanded_month_range),
                ]
            )
        else:
            ## Load each dataset EXCEPT prism
            data = xr.merge(
                [
                    load_wam_data(expanded_month_range),
                    load_geopotential_height(expanded_month_range),
                    load_helmholtz(expanded_month_range),
                    load_soil_moisture(expanded_month_range),
                ]
            )

    return data


## Plotting functions
def mod_synoptic_panel_wide(ax, plot_domains=True):
    """
    Set up plotting panel for single panel of synoptic plot
    """

    ### Set up plotting background
    ax, gl = src.utils.plot_setup_ax(
        ax=ax,
        plot_range=[150, 320, 10, 60],
        xticks=[160, -170, -140, -110, -80, -50],
        yticks=[15, 35, 55],
        alpha=0.1,
        plot_topo=False,
    )
    gl.bottom_labels = False

    ## Outline GPLLJ and Midwest regions
    if plot_domains:
        ax = src.utils.add_patches_to_ax(ax)

    return ax, gl


def get_composite_times_all(month_range):
    """Get set of days on which a GPLLJ event occurs. Returns
    dictionary with each key corresponding to a different type
    of GPLLJ event"""

    times = {}
    for cat in ["v850", "v850_neg", "coupled", "uncoupled"]:
        times[cat] = src.utils.load_and_filter(
            f"{src.params.DATA_PREPPED_FP}/times_{cat}.npy", month_range
        )
    return times


def get_composite_times(gpllj_category, month_range):
    """Get set of days on which a GPLLJ event occurred for specific
    category of GPLLJ event"""

    ## Load times for all categories
    times = get_composite_times_all(month_range)

    return times[gpllj_category]


def get_composite(data, gpllj_category, month_range):
    """Get composite, based on specified GPLLJ category"""

    ## filepath for saving composite
    suffix = f"{gpllj_category}_{src.params.season}"
    composite_fp = f"{src.params.DATA_PREPPED_FP}/composite_{suffix}.nc"

    ## check for existence of file
    if not os.path.isfile(composite_fp):
        print(f"Computing composite for {gpllj_category} GPLLJs")

        ## Get list of times for creating composite
        composite_times = get_composite_times(gpllj_category, month_range)

        ## Compute the composite
        composite = src.utils.get_lagged_composites(
            data,
            composite_times,
            lags=np.arange(-14, 6, 1),
        )
        composite = src.utils.remove_nan(composite)
        composite.to_netcdf(composite_fp)

    else:
        composite = xr.open_dataset(composite_fp)

    return composite


def get_composite_agg(gpllj_category, month_range, load_prism):
    """Get composite with aggregated data"""

    ## filepath for saving composite and monte-carlo bounds
    suffix = f"{gpllj_category}_{src.params.season}"
    composite_fp = f"{src.params.DATA_PREPPED_FP}/composite_agg_{suffix}.nc"
    lb_fp = f"{src.params.DATA_PREPPED_FP}/composite_agg_lb_{suffix}.nc"
    ub_fp = f"{src.params.DATA_PREPPED_FP}/composite_agg_ub_{suffix}.nc"

    ## check for existence of file

    if not os.path.isfile(composite_fp):
        ### Get set of times for computing composite
        composite_times = get_composite_times(gpllj_category, month_range)

        ### Load aggregated WAM output
        wam_agg = xr.open_dataset(
            f"{src.params.DATA_PREPPED_FP}/wam_fluxes_agg_daily_detrend.nc"
        )
        expanded_month_range = (month_range[0] - 1, month_range[1] + 1)
        wam_agg = src.utils.get_month_range(wam_agg, *expanded_month_range)
        wam_agg *= 1000  # convert from m to mm

        if load_prism:
            ### Load PRISM data
            prism_agg = (
                xr.open_dataarray(
                    f"{src.params.DATA_PREPPED_FP}/prism_agg_daily_detrend.nc"
                )
                .drop("mask")
                .rename("prism")
            )

            ### Combine data
            data_agg = xr.merge([wam_agg, prism_agg])

        else:
            data_agg = wam_agg

        ### Compute composite
        composite = src.utils.get_lagged_composites(
            data_agg,
            composite_times,
            lags=np.arange(-14, 4),
        )

        ### Montecarlo test to get sig. bounds
        mc_comps = []  # list to hold results of each simulation
        nsims = 1000  # number of simulations for montecarlo composite
        np.random.seed(0)  # seed RNG for reproducibility
        for sim in progressbar.progressbar(range(nsims)):
            rand_times = np.random.choice(data_agg.time, size=len(composite_times))
            rand_times = pd.DatetimeIndex(rand_times)
            mc_comps.append(data_agg.sel(time=rand_times).mean("time"))
        mc_comps = xr.concat(mc_comps, dim=pd.Index(np.arange(nsims), name="sim"))

        ## Compute confidence upper/lower bounds
        ub = mc_comps.quantile(q=0.95, dim="sim")
        lb = mc_comps.quantile(q=0.05, dim="sim")

        ## save to file
        composite.to_netcdf(composite_fp)
        ub.to_netcdf(ub_fp)
        lb.to_netcdf(lb_fp)

    else:
        composite = xr.open_dataset(composite_fp)
        ub = xr.open_dataset(ub_fp)
        lb = xr.open_dataset(lb_fp)

    return composite, ub, lb


def get_surf_mask():
    """Return array with ones where surface pressure is greater than 850,
    and NaNs elsewhere"""

    ## Load surface pressure climatology
    sp_clim = xr.open_dataarray(
        f"{src.params.DATA_PREPPED_FP}/sp_seasonal_{src.params.season}.nc"
    )
    sp_clim = sp_clim.mean("year")

    ## Create mask
    mask = xr.ones_like(sp_clim)
    mask = mask.where(sp_clim < 85000.0)

    return mask


def plot_z(ax, z):
    """Plot geopotential height at 850 and 200 hPa"""

    ## Get plotting background
    ax, gl = mod_synoptic_panel_wide(ax, plot_domains=False)

    ### Plot data
    ## geopotential height @ 200 hPa
    cs = ax.contour(
        z.longitude,
        z.latitude,
        z.sel(level=200),
        extend="both",
        colors="k",
        levels=src.utils.make_cb_range(150, 15),
        transform=ccrs.PlateCarree(),
    )

    # plot
    cp = ax.contourf(
        z.longitude,
        z.latitude,
        z.sel(level=850),
        extend="both",
        cmap="cmo.balance",
        levels=src.utils.make_cb_range(50, 5),
        transform=ccrs.PlateCarree(),
    )

    # add stippling to mark where surface is below 850 hPa
    # z850_masked = mask_z850(z.sel(level=850))
    surf_mask = get_surf_mask()
    hatch = ax.contourf(
        surf_mask.longitude,
        surf_mask.latitude,
        surf_mask,
        colors="none",
        hatches=["...."],
        levels=[0.5, 1.5],
        transform=ccrs.PlateCarree(),
    )

    return ax, gl, cp


def plot_satrack_z(ax, satrack, z):
    """Plot tracked moisture and geopotential height"""

    ## Get plotting background
    ax, gl = mod_synoptic_panel_wide(ax)

    ### Plot data
    ## geopotential height
    cs = ax.contour(
        z.longitude,
        z.latitude,
        z,
        extend="both",
        colors="k",
        levels=src.utils.make_cb_range(150, 15),
        transform=ccrs.PlateCarree(),
    )

    ## Tracked moisture
    cp = ax.contourf(
        satrack.longitude,
        satrack.latitude,
        satrack,
        extend="both",
        cmap="cmo.balance_r",
        levels=src.utils.make_cb_range(3, 0.3),
        transform=ccrs.PlateCarree(),
    )

    return ax, gl, cp


def plot_precip(ax, precip, levels=src.utils.make_cb_range(5, 0.5)):
    """Set up plotting background and plot precip"""

    ### Set up plotting background
    ax, gl = src.utils.plot_setup_ax(
        ax=ax,
        plot_range=[230, 310, 15, 53],
        xticks=[-125, -105, -85, -65],
        yticks=[20, 30, 40, 50],
        alpha=0.1,
        plot_topo=False,
    )
    gl.bottom_labels = False

    ## Plot precip
    cp = ax.contourf(
        precip.longitude,
        precip.latitude,
        precip,
        extend="both",
        cmap="cmo.balance_r",
        levels=levels,
        transform=ccrs.PlateCarree(),
    )

    ## Outline GPLLJ region
    ax = src.utils.add_patches_to_ax(ax)

    return ax, gl, cp


def add_sm_to_ax(ax, sm):
    """Plot soil moisture contours on existing ax"""

    ## soil moisture
    cs = ax.contour(
        sm.longitude,
        sm.latitude,
        sm,
        extend="both",
        colors="k",
        levels=src.utils.make_cb_range(5, 0.5),
        transform=ccrs.PlateCarree(),
    )
    return ax


def add_flux_to_ax(ax, u, v):
    """Plot vectors on existing ax"""

    n = 3
    scale = 240
    x, y = np.meshgrid(u.longitude[::n].values, u.latitude[::n].values)
    qv = ax.quiver(
        x,
        y,
        u.values[::n, ::n],
        v.values[::n, ::n],
        pivot="middle",
        scale=scale,
        color="k",
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

    ## Add quiverkey
    ax.add_patch(
        mpatches.Rectangle(
            xy=[292, 40],
            width=17,
            height=12,
            facecolor="white",
            edgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=1.5,
            linewidth=src.params.plot_params["border_width"],
        )
    )

    qk = ax.quiverkey(
        qv,
        X=0.88,
        Y=0.7,
        U=scale / 20,
        label=f"{scale/20}" + r" $\frac{kg}{m \cdot s}$",
        fontproperties={"size": mpl.rcParams["legend.fontsize"]},
    )
    qk.set_zorder(1.55)

    return ax, qv


def plot_setup_agg(ax, composite, ub, lb, varnames, masks, labels, colors):
    """Plot lagged composite of aggregated data"""

    ## Modify axis labels
    xticks = np.arange(-12, 6, 3)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.axhline(0, lw=0.7, c="k", ls="--")
    ax.axvline(0, lw=0.7, c="k", ls="--")
    ax.set_ylabel(r"Anomaly ($mm$)")
    ax.set_xlim([-14, 3])

    for varname, mask, label, color in zip(varnames, masks, labels, colors):
        y = composite[varname].sel(mask=mask)
        ub_ = ub[varname].sel(mask=mask).item() * np.ones(len(composite.lag))
        lb_ = lb[varname].sel(mask=mask).item() * np.ones(len(composite.lag))
        p = ax.plot(composite.lag, y, label=label, color=color)
        x, y1, y2 = src.utils.get_fill_between(y, lb=lb_, ub=ub_, idx=composite.lag)
        ax.fill_between(x, y1, y2, alpha=0.1, color=color)

    ax.legend(loc="best")

    return ax


def plot_agg_e_v_p(ax, composite, ub, lb):
    """Plot composite of aggregate E and P anomalies"""
    ax = plot_setup_agg(
        ax=ax,
        composite=composite,
        ub=ub,
        lb=lb,
        varnames=["E_track", "P"],
        masks=["Total", "Midwest"],
        labels=["Tracked Evap.", "Precip"],
        colors=[sns.color_palette("colorblind")[i] for i in [5, 9]],
    )
    ax.set_ylim([-1.2, 3.2])
    ticks = np.arange(-1, 4, 1)
    labels = [f"{t:.1f}" for t in ticks]
    ax.set_yticks(ticks=ticks, labels=labels)

    return ax


def plot_agg_oc_v_la(ax, composite, ub, lb):
    """Plot composite of aggregate ocean and land anomalies"""

    ax = plot_setup_agg(
        ax=ax,
        composite=composite,
        ub=ub,
        lb=lb,
        varnames=["E_track", "E_track", "E_track"],
        masks=["Ocean", "Land", "Midwest"],
        labels=["Ocean", "Land", "Midwest"],
        colors=[sns.color_palette("colorblind")[i] for i in [0, 2, 1]],
    )

    ax.set_ylim([-0.12, 0.42])
    ticks = np.arange(-0.1, 0.5, 0.1)
    labels = [f"{t:.1f}" for t in ticks]
    ax.set_yticks(ticks=ticks, labels=labels)

    return ax


def plot_agg_atl_v_pac(ax, composite, ub, lb):
    """Plot composite of aggregate atlantic and pacific anomalies"""
    ax = plot_setup_agg(
        ax=ax,
        composite=composite,
        ub=ub,
        lb=lb,
        varnames=["E_track", "E_track"],
        masks=["Atlantic", "Pacific"],
        labels=["Atlantic", "Pacific"],
        colors=[sns.color_palette("mako")[i] for i in [3, 0]],
    )
    ax.set_ylim([-0.12, 0.32])
    ticks = np.arange(-0.1, 0.4, 0.1)
    labels = [f"{t:.1f}" for t in ticks]
    ax.set_yticks(ticks=ticks, labels=labels)

    return ax


def plot_multiday_comp(ax, comp, plot_ivt=False):
    """Plot composite using the same style as 'plot_etrack_coef'
    in 'plot_regression.py' script"""
    ## Begin plotting
    ax, gl = src.utils.plot_setup_ax(
        ax=ax,
        plot_range=[210, 330, 5, 60],
        xticks=[-140, -115, -90, -65, -40],
        yticks=[10, 25, 40, 55],
        alpha=0.1,
        plot_topo=True,
    )

    ## Plot region outlines
    ax = src.utils.add_patches_to_ax(ax)

    ## Tracked moisture
    cp = ax.contourf(
        comp["E_track"].longitude,
        comp["E_track"].latitude,
        comp["E_track"],
        levels=src.utils.make_cb_range(1, 0.1),
        extend="both",
        cmap="cmo.balance_r",
        transform=ccrs.PlateCarree(),
    )

    ### divergent component of IVT
    if plot_ivt:
        n = 5
        scale = 60
        x, y = np.meshgrid(comp.longitude[::n].values, comp.latitude[::n].values)
        qv1 = ax.quiver(
            x,
            y,
            comp["u_chi"].values[::n, ::n],
            comp["v_chi"].values[::n, ::n],
            pivot="middle",
            scale=scale,
            color="k",
            alpha=1,
            transform=ccrs.PlateCarree(),
        )

        ## Quiver key (and background)
        ax.add_patch(
            mpatches.Rectangle(
                xy=[211, 6],
                width=24,
                height=14,
                facecolor="white",
                edgecolor="black",
                transform=ccrs.PlateCarree(),
                linewidth=src.params.plot_params["border_width"],
                zorder=1.06,
            )
        )
        qk = ax.quiverkey(
            qv1,
            X=0.1,
            Y=0.05,
            U=scale / 20,
            label=f"{scale/20}" + r" $\frac{kg}{m \cdot s}$",
            fontproperties={"size": mpl.rcParams["legend.fontsize"]},
        )

    return ax, gl, cp


def plot_setup_monthly(yticks, ylim):
    """make a blank canvas for plotting monthly data"""

    fig, ax = plt.subplots(figsize=(4.2, 3))

    ax.axhline(0, ls="--", c="k")
    xticks = np.arange(1, 13, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylim(ylim)
    ax.set_xlim([0.5, 12.5])
    ax.set_xlabel("Month")

    return fig, ax


def print_counts_for_month_range(month_range):
    """Print out the counts for each category for specified month_range"""

    ## Load times for each GPLLJ category
    times = get_composite_times_all(month_range=month_range)

    ## Print out stats
    print(f"Number of V850:       {len(times['v850'])}")
    print(f"Number of B-W:       {len(times['coupled'])+len(times['uncoupled'])}")
    print(f"Number of coupled:    {len(times['coupled'])}")
    print(f"Number of uncoupled: {len(times['uncoupled'])}")
    print(
        f"V850 & coupled:       {len(set(times['v850']).intersection(times['coupled']))}"
    )
    print(
        f"V850 & uncoupled:     {len(set(times['v850']).intersection(times['uncoupled']))}"
    )
    return


def get_counts_by_month():
    """Count number of GPLLJ events by month, by GPLLJ category"""

    ## Categories and months to count for
    cats = ["v850", "coupled", "uncoupled"]
    months = np.arange(4, 9)

    ## Function to count number of events for given category in given month
    count = lambda cat, m: len(get_composite_times(cat, month_range=(m, m)))

    ## Loop through categories and months
    counts_by_month = {cat: np.array([count(cat, m) for m in months]) for cat in cats}

    return months, counts_by_month


def plot_counts_by_month(fig):
    """Plot count of each GPLLJ category by month"""

    ## Get the count for each category by month
    months, counts_by_month = get_counts_by_month()

    ## Setup plot
    ax = fig.add_subplot()

    ## Add data
    for cat in list(counts_by_month):
        ax.plot(months, counts_by_month[cat], "-o", label=cat)

    ## Set axis limits and add legend
    ax.set_ylim([-20, None])
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.legend()

    return fig, ax


def get_multiday_comp(comp, lags=(-14, 2)):
    """Get multiday composite by summing/averaging over
    multiple lag times"""

    ## List of variables to average and sum over
    mean_vars = ["z", "u_chi", "v_chi", "Sa_track"]
    sum_vars = ["E_track"]

    ## Compute mean/sums
    comp_mean = comp[mean_vars].sel(lag=slice(*lags)).mean("lag")
    comp_sum = comp[sum_vars].sel(lag=slice(*lags)).sum("lag")

    return xr.merge([comp_mean, comp_sum])


def load_agg_composite(gpllj_cat):
    """Load composite and monte-carlo bounds for given GPLLJ category"""

    ## Get filepaths for composite and lower/upper bounds
    fp_prefix = f"{src.params.DATA_PREPPED_FP}/composite_agg"
    suffix = f"{gpllj_cat}_{src.params.season}"
    fp_comp = f"{fp_prefix}_{suffix}.nc"
    fp_lb = f"{fp_prefix}_lb_{suffix}.nc"
    fp_ub = f"{fp_prefix}_ub_{suffix}.nc"

    ## convenience function to open dataset and filter for Midwest mask
    load = lambda fp: xr.open_dataset(fp).sel(mask="Midwest")

    ## Put data in dictionary
    comp_and_bounds = {
        f"comp_{gpllj_cat}": load(fp_comp),
        f"lb_{gpllj_cat}": load(fp_lb),
        f"ub_{gpllj_cat}": load(fp_ub),
    }
    return comp_and_bounds


def plot_agg_precip(ax, comp, ub, lb):
    """Plot composite of Midwest P anomalies for ERA5 and PRISM"""

    ## Modify axis labels
    xticks = np.arange(-12, 6, 3)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.axhline(0, lw=0.5, c="k", ls="-")
    ax.axvline(0, lw=0.5, c="k", ls="-")
    ax.set_xlim([-14, 3])

    varnames = ["P", "prism"]
    labels = ["ERA5", "PRISM"]
    # colors = [sns.color_palette("colorblind")[i] for i in [0, 1]]
    colors = ["k", "k"]
    lss = ["-", "--"]

    for varname, label, color, ls in zip(varnames, labels, colors, lss):
        y = comp[varname]
        ub_ = ub[varname].item() * np.ones(len(comp.lag))
        lb_ = lb[varname].item() * np.ones(len(comp.lag))
        p = ax.plot(comp.lag, y, label=label, color=color, ls=ls)
        # x, y1, y2 = src.utils.get_fill_between(y, lb=lb_, ub=ub_, idx=comp.lag)
        # ax.fill_between(x, y1, y2, alpha=0.1, color=color)

    ax.legend(loc="best")

    return ax


def plot_agg_precip_comp():
    """Plot comparison of aggregate anomalies composite
    on GPLLJ events for each type of GPLLJ category"""

    ## Load data
    gpllj_cats = ["v850", "coupled", "uncoupled"]
    labels = [r"a) $v_{850}$", "b) Coupled", "c) Uncoupled"]

    data = {
        **load_agg_composite(gpllj_cats[0]),
        **load_agg_composite(gpllj_cats[1]),
        **load_agg_composite(gpllj_cats[2]),
    }

    ## Make the plot
    # Set up the plotting canvas
    aspect = 3
    width = src.params.plot_params["max_width"]
    height = width / aspect
    fig, axs = plt.subplots(1, 3, figsize=(width, height), layout="constrained")
    axs[0].set_ylabel(r"Anomaly ($mm$)")

    # Plot the data
    for i, (ax, cat, label) in enumerate(zip(axs, gpllj_cats, labels)):
        ax = plot_agg_precip(
            ax, comp=data[f"comp_{cat}"], ub=data[f"ub_{cat}"], lb=data[f"lb_{cat}"]
        )
        ax.set_xlabel("Lag (days)")
        ax.set_ylim([-1.2, 3.2])
        ax = src.utils.label_subplot(
            ax, label=label, posn=(0.65, 0.9), transform=ax.transAxes
        )

    return fig, axs


if __name__ == "__main__":
    ## Ignore shapely warning
    warnings.filterwarnings("ignore", category=shapely.errors.ShapelyDeprecationWarning)

    ## Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpllj_category",
        default="v850",
        choices=["v850", "v850_neg", "coupled", "uncoupled"],
        dest="gpllj_category",
        type=str,
    )
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--plot_prism", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    ## Set custom plot style
    src.params.set_plot_style()  # set plotting style for figures

    ## define GPLLJ categories
    gpllj_cats = ["v850", "coupled", "uncoupled", "v850_neg"]

    #### Compute composites for spatial data ####
    ## Load data (only necessary if composites don't exist already)
    files_to_check = os.listdir(f"{src.params.DATA_PREPPED_FP}")
    files_exist = np.array(
        [
            f"composite_{cat}_{src.params.season}.nc" in files_to_check
            for cat in gpllj_cats
        ]
    ).all()
    if files_exist:
        print("Composites are precomputed.")
        data = None

    else:
        print("Loading data...")
        data = load_data(month_range=src.params.MONTH_RANGE, load_prism=args.plot_prism)
        print("Done.")

    ## Get composites for each type of event
    comps = {
        cat: get_composite(data, cat, month_range=src.params.MONTH_RANGE)
        for cat in gpllj_cats
    }
    multiday_comps = {cat: get_multiday_comp(comps[cat]) for cat in gpllj_cats}

    ## Get composites for user-specified GPLLJ category
    comp = comps[args.gpllj_category]

    #### Compute composites for aggregated data ####
    print(f"\nComputing composites for aggregated data...")
    comps_agg = {
        cat: get_composite_agg(
            cat, month_range=src.params.MONTH_RANGE, load_prism=args.plot_prism
        )
        for cat in gpllj_cats
    }
    comp_agg, ub, lb = comps_agg[args.gpllj_category]
    print("Done.")

    ################## Begin plots ################

    ###### 1. tracked moisture/geopotential height ####

    ## Set figure layout with gridspec
    aspect = 1.25
    width = src.params.plot_params["max_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs0 = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[4.5, 0.3, 2.7])
    gs00 = gs0[0].subgridspec(nrows=4, ncols=1)
    gs01 = gs0[-1].subgridspec(nrows=5, ncols=1, height_ratios=[1, 0.1, 1, 0.1, 1])

    # Plot tracked moisture
    labels = ["a)", "b)", "c)", "d)"]
    for i, (lag, label) in enumerate(zip([-4, -2, 0, 2], labels)):
        ax = fig.add_subplot(
            gs00[i], projection=ccrs.PlateCarree(central_longitude=180)
        )
        ax, gl, cp = plot_satrack_z(
            ax=ax,
            satrack=comp["Sa_track"].sel(lag=lag).sum("wam_level"),
            z=comp["z"].sel(lag=lag, level=200),
        )
        gl.top_labels = i == 0  # only plot top labels if i==0

        ## Add label to subplot
        ax = src.utils.label_subplot(
            ax,
            label=f"{label} {lag} days",
            posn=(167, 17),
            transform=ccrs.PlateCarree(),
        )

    ## At bottom of left panel, add labels and colorbar
    # gl.bottom_labels = True
    cb = fig.colorbar(cp, ax=ax, orientation="horizontal", pad=0.05, fraction=0.1)
    cb = src.utils.mod_colorbar(
        cb, label=r"Tracked moisture ($mm$)", ticks=np.arange(-3, 4, 1)
    )

    ###### 1b. Aggregate anomalies
    ax1 = fig.add_subplot(gs01[0])
    ax1 = plot_agg_e_v_p(ax1, composite=comp_agg, ub=ub, lb=lb)
    ax1.xaxis.set_ticklabels([])

    ax2 = fig.add_subplot(gs01[2])
    ax2 = plot_agg_oc_v_la(ax2, composite=comp_agg, ub=ub, lb=lb)
    ax2.xaxis.set_ticklabels([])

    ax3 = fig.add_subplot(gs01[4])
    ax3 = plot_agg_atl_v_pac(ax3, composite=comp_agg, ub=ub, lb=lb)
    ax3.set_xlabel("Lag (days)")

    #### Add labels to aggregate subplots
    ax1 = src.utils.label_subplot(
        ax1, label="e)", posn=(0.92, 0.1), transform=ax1.transAxes
    )
    ax2 = src.utils.label_subplot(
        ax2, label="f)", posn=(0.92, 0.9), transform=ax2.transAxes
    )
    ax3 = src.utils.label_subplot(
        ax3, label="g)", posn=(0.92, 0.9), transform=ax3.transAxes
    )

    ## Save to file
    is_supp = args.gpllj_category != "v850"  # supplementary unless category is 'v850'
    src.utils.save(
        fig=fig, fname=f"synoptic_composite_{args.gpllj_category}", is_supp=is_supp
    )
    plt.close(fig)

    ###### 1.5. geopotential height at 850 and 200 hPa ######
    ## Set figure layout with gridspec
    aspect = 1.25 * 4.5 / 7.5
    width = src.params.plot_params["max_width"] * 4.5 / 7.5
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs00 = fig.add_gridspec(nrows=4, ncols=1)

    # Plot tracked moisture
    labels = ["a)", "b)", "c)", "d)"]
    for i, (lag, label) in enumerate(zip([-4, -2, 0, 2], labels)):
        ax = fig.add_subplot(
            gs00[i], projection=ccrs.PlateCarree(central_longitude=180)
        )
        ax, gl, cp = plot_z(
            ax=ax,
            z=comp["z"].sel(lag=lag),
        )

        ## Add label to subplot
        ax = src.utils.label_subplot(
            ax,
            label=f"{label} {lag} days",
            posn=(167, 17),
            transform=ccrs.PlateCarree(),
        )

    ## At bottom of left panel, add labels and colorbar
    gl.bottom_labels = True
    cb = fig.colorbar(cp, ax=ax, orientation="horizontal", pad=0.05, fraction=0.1)
    cb = src.utils.mod_colorbar(
        cb, label=r"$Z_{850}$ (gpm)", ticks=np.arange(-50, 75, 25)
    )

    ## Save to file
    src.utils.save(fig=fig, fname=f"synoptic_composite_Z_{args.gpllj_category}")
    plt.close(fig)

    ###### 2. Comparison of tracked evap. anoms. preceding ######
    ######    each type of GPLLJ event ######

    ## Plotting
    proj = ccrs.PlateCarree(central_longitude=180)
    aspect = 1.75
    width = src.params.plot_params["max_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(nrows=2, ncols=2)

    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax1, gl1, cp1 = plot_multiday_comp(ax1, comp=multiday_comps["v850"])
    gl1.bottom_labels = False

    ax2 = fig.add_subplot(gs[1, 0], projection=proj)
    ax2, gl2, cp2 = plot_multiday_comp(ax2, comp=multiday_comps["v850_neg"])

    ax3 = fig.add_subplot(gs[0, 1], projection=proj)
    ax3, gl3, cp3 = plot_multiday_comp(ax3, comp=multiday_comps["coupled"])
    gl3.bottom_labels = False
    gl3.left_labels = False

    ax4 = fig.add_subplot(gs[1, 1], projection=proj)
    ax4, gl4, cp4 = plot_multiday_comp(ax4, comp=multiday_comps["uncoupled"])
    gl4.left_labels = False

    ## colorbar
    cb = fig.colorbar(
        cp4, ax=[ax2, ax4], orientation="horizontal", pad=0.07, fraction=0.08
    )
    cb = src.utils.mod_colorbar(
        cb, label=r"Contribution ($mm$)", ticks=[-1.0, -0.5, 0.0, 0.5, 1.0]
    )

    ## label subplots
    ax1 = src.utils.label_subplot(
        ax=ax1,
        label=r"a) GPLLJ idx $> 1~\sigma$",
        posn=(0.75, 0.85),
        transform=ax1.transAxes,
    )
    ax2 = src.utils.label_subplot(
        ax=ax2,
        label=r"b) GPLLJ idx $<-1~\sigma$",
        posn=(0.75, 0.85),
        transform=ax2.transAxes,
    )
    ax3 = src.utils.label_subplot(
        ax=ax3, label="c) coupled", posn=(0.85, 0.85), transform=ax3.transAxes
    )
    ax4 = src.utils.label_subplot(
        ax=ax4, label="d) uncoupled", posn=(0.85, 0.85), transform=ax4.transAxes
    )

    src.utils.save(fig=fig, fname=f"synoptic_composite_comparison", is_supp=False)
    plt.close(fig)

    ###### 3. Precip / {IVT, soil moisture} ######

    ## Set figure layout with gridspec
    aspect = 1
    width = src.params.plot_params["twocol_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(nrows=4, ncols=2)

    # Plot precip/IVT
    labels = ["a)", "b)", "c)", "d)"]
    for i, (lag, label) in enumerate(zip([-1, 0, 1, 2], labels)):
        ax = fig.add_subplot(
            gs[i, 0], projection=ccrs.PlateCarree(central_longitude=180)
        )

        # first, plot precip
        ax, gl, cp = plot_precip(
            ax=ax,
            precip=comp["P"].sel(lag=lag),
        )

        # next, plot divergent moisture flux
        ax, qv = add_flux_to_ax(
            ax=ax,
            u=comp["u_chi"].sel(lag=lag),
            v=comp["v_chi"].sel(lag=lag),
        )

        ## Add label to subplot
        ax = src.utils.label_subplot(
            ax,
            label=f"{label} {lag} days",
            posn=(0.15, 0.12),
            transform=ax.transAxes,
        )
    gl.bottom_labels = True

    # Plot precip/soil moisture
    labels = ["e)", "f)", "g)", "h)"]
    for i, (lag, label) in enumerate(zip([-1, 1, 3, 5], labels)):
        ax = fig.add_subplot(
            gs[i, 1], projection=ccrs.PlateCarree(central_longitude=180)
        )

        # first, plot precip
        ax, gl, cp = plot_precip(
            ax=ax,
            precip=comp["P"].sel(lag=lag),
        )
        gl.left_labels = False

        # next, plot divergent moisture flux
        ax = add_sm_to_ax(ax=ax, sm=comp["swvl1"].sel(lag=lag))

        ## Add label to subplot
        ax = src.utils.label_subplot(
            ax,
            label=f"{label} {lag} days",
            posn=(0.15, 0.12),
            transform=ax.transAxes,
        )
    gl.bottom_labels = True

    ## At bottom of fig, add labels and colorbar
    cb = fig.colorbar(
        cp,
        ax=[fig.axes[3], fig.axes[7]],
        orientation="horizontal",
        pad=0.05,
        fraction=0.1,
    )
    cb = src.utils.mod_colorbar(
        cb,
        label=r"Precip ($mm$)",
        ticks=np.round(np.arange(-5, 7.5, 2.5), 1),
    )

    ## Save to file
    src.utils.save(fig=fig, fname=f"synoptic_composite_precip_{args.gpllj_category}")
    plt.close(fig)

    ###### 4. Plot count of GPLLJ events by month #######
    aspect = 1.25
    width = src.params.plot_params["twocol_width"] / 2.5
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    fig, ax = plot_counts_by_month(fig)
    src.utils.save(fig, fname="gpllj_counts_by_month")
    plt.close(fig)

    ## Only make the following plots if specified
    if args.plot_prism:
        ###### 5. PRISM / ERA5 precip ######
        ## Set figure layout with gridspec
        aspect = 1
        width = src.params.plot_params["twocol_width"]
        height = width / aspect
        fig = plt.figure(figsize=(width, height), layout="constrained")
        gs = fig.add_gridspec(nrows=4, ncols=2)

        # Plot precip/IVT
        labels_era5 = ["a)", "b)", "c)", "d)"]
        labels_prism = ["e)", "f)", "g)", "h)"]

        for j, (name, labels) in enumerate(
            zip(["P", "prism"], [labels_era5, labels_prism])
        ):
            # Rename lon/lat coords for prism and apply land-sea mask to ERA5
            if name == "prism":
                ## rename PRISM coordinates
                precip = comp[name].rename(
                    {"longitude_prism": "longitude", "latitude_prism": "latitude"}
                )
            else:
                ## Load LSM
                era5_lsm = xr.open_dataarray(f"{src.params.DATA_CONST_FP}/lsm.nc")
                era5_lsm = era5_lsm.isel(time=0).sel(
                    longitude=comp.longitude, latitude=comp.latitude
                )

                ## apply LSM to precip data
                precip = comp[name] * era5_lsm

            for i, (lag, label) in enumerate(zip([-1, 0, 1, 2], labels)):
                ax = fig.add_subplot(
                    gs[i, j], projection=ccrs.PlateCarree(central_longitude=180)
                )

                ax, gl, cp = plot_precip(
                    ax=ax,
                    precip=precip.sel(lag=lag),
                    levels=src.utils.make_cb_range(7.5, 0.75),
                )

                ## Add label to subplot
                if j == 0:
                    ax = src.utils.label_subplot(
                        ax,
                        label=f"{label} {lag} days",
                        posn=(0.15, 0.12),
                        transform=ax.transAxes,
                    )
            gl.bottom_labels = True

        ## At bottom of fig, add labels and colorbar
        cb = fig.colorbar(
            cp,
            ax=[fig.axes[3], fig.axes[7]],
            orientation="horizontal",
            pad=0.05,
            fraction=0.1,
        )
        cb = src.utils.mod_colorbar(
            cb,
            label=r"Precip ($mm$)",
            ticks=np.round(np.arange(-7.5, 10, 2.5), 1),
        )

        ## save to file
        src.utils.save(fig=fig, fname=f"synoptic_composite_prism_{args.gpllj_category}")
        plt.close(fig)

        ###### 6. compare aggregate ERA5/PRISM precip. anomalies ######
        fig, axs = plot_agg_precip_comp()
        src.utils.save(fig=fig, fname=f"agg_precip_comp")
        plt.close(fig)

    #### Print out counts of GPLLJ per month, if desired
    if args.verbose:
        print_counts_for_month_range(src.params.MONTH_RANGE)
