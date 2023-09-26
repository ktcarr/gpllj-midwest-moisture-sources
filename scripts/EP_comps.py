import src.utils
import src.params
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import seaborn as sns
import cmocean
import progressbar
import os.path
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs


def get_filepath_suffix(is_detrend, is_seasonal):
    """get suffix for filename,
    given boolean options 'is_detrend' and 'is_seasonal'"""

    ## get season
    season = src.params.season

    if is_detrend & is_seasonal:
        suffix = f"_seasonal_detrend_{season}"
    elif is_detrend:
        suffix = "_daily_detrend"
    elif is_seasonal:
        suffix = f"_seasonal_{season}"
    else:
        suffix = ""

    return suffix


def get_filepath_prefix(is_prism, is_agg):
    """get prefix for filename"""

    if is_prism:
        prefix = "prism_agg" if is_agg else "prism"
    else:
        prefix = "wam_fluxes_agg" if is_agg else "wam_fluxes_trim"

    return prefix


def load_agg_data(detrend, seasonal, month_range=[1, 12]):
    """convenience wrapper function for loading spatially aggregated data"""
    agg_data = load_data(
        agg=True, detrend=detrend, seasonal=seasonal, month_range=month_range
    )
    return agg_data


def load_data(agg, detrend, seasonal, month_range):
    """Load aggregated data for ERA and PRISM"""

    ## Get filepaths for data
    data_fp = src.params.DATA_PREPPED_FP
    suffix = get_filepath_suffix(is_detrend=detrend, is_seasonal=seasonal)
    prefix_prism = get_filepath_prefix(is_prism=True, is_agg=agg)
    prefix_era5 = get_filepath_prefix(is_prism=False, is_agg=agg)

    ## Load the data
    # PRISM
    p_prism = xr.open_dataarray(f"{data_fp}/{prefix_prism}{suffix}.nc")

    # ERA5
    p_era5 = xr.open_dataset(f"{data_fp}/{prefix_era5}{suffix}.nc")["P"]

    ## convert ERA5 from m to mm
    mm_per_m = 1000.0
    p_era5 = p_era5 * mm_per_m

    ## handle diff lon/lat grids
    if not agg:
        p_prism = p_prism.rename(
            {"latitude": "latitude_prism", "longitude": "longitude_prism"}
        )
        p_era5 = p_era5.rename(
            {"latitude": "latitude_era5", "longitude": "longitude_era5"}
        )
    else:
        p_era5 = p_era5.sel(mask="Midwest")

    ## Make sure times align on the datasets
    time_coord = "year" if seasonal else "time"
    p_era5 = p_era5.sel({time_coord: p_prism[time_coord]})

    ## Merge data into single dataset
    p = xr.merge([p_prism.rename("prism"), p_era5.rename("era5")])

    ## Select specified month range (only if not using seasonal data
    if not seasonal:
        p = src.utils.get_month_range(p, *month_range)

    return p


def plot_scatter_and_diff(p_era5, p_prism, xlim, xticks, yticks, s):
    """Plot scatter of era vs. prism, and differnce vs. prism"""
    ## Compute correlation
    r, p = scipy.stats.pearsonr(p_prism, p_era5)

    ## Points for one-to-one line
    x = np.linspace(p_prism.min(), p_prism.max())

    ## Make plot
    width = src.params.plot_params["twocol_width"]
    aspect = 2.75
    height = width / aspect
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(width, height), layout="constrained")

    ## Set plots to equal aspect
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    #### First plot #####
    ax1.set_xlabel("PRISM (mm)")
    ax1.set_ylabel("ERA5 (mm)")
    ax1.set_title("PRISM vs. ERA5 precip.")

    #### display stats in legend
    stats1 = r"$r=$" + f"{r:.3f}\n"
    stats2 = r"$p=$" + f"{p:.2e}"
    ax1.legend(title=stats1 + stats2)

    ## set axis limits
    ax1.set_xlim(xlim)
    ax1.set_ylim(xlim)
    ax1.set_xticks(xticks)
    ax1.set_yticks(xticks)

    ## Add data
    ax1.scatter(p_prism, p_era5, s=s)
    ax1.plot(x, x, ls="--", c="k", lw=0.5, alpha=0.6)

    #### Second plot ####
    ax2.set_xlabel("PRISM (mm)")
    ax2.set_ylabel("ERA5 minus PRISM (mm)")
    ax2.set_title("Difference")

    ## Add data
    diff = p_era5 - p_prism
    ax2.scatter(p_prism, diff, s=s)
    ax2.plot(x, 0 * x, ls="--", c="k", lw=0.5, alpha=0.6)

    ## Set axis limits for second plot
    yscale = np.abs(diff).max() * 1.1
    ax2.set_ylim([-yscale, yscale])
    ax2.set_xlim(xlim)

    ## tick values
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)

    return fig, [ax1, ax2]


def plot_agg_comp_seasonal():
    """Plot comparison of ERA5 and PRISM data at SEASONAL timescales"""

    ## Load daily-frequency precip data
    p_agg = load_agg_data(
        detrend=True, seasonal=True, month_range=src.params.MONTH_RANGE
    )

    ## Make plot
    fig, axs = plot_scatter_and_diff(
        p_prism=p_agg["prism"],
        p_era5=p_agg["era5"],
        xlim=[-110, 90],
        xticks=np.arange(-100, 100, 50),
        yticks=np.arange(-50, 75, 25),
        s=2.0,
    )

    return fig, axs


def plot_agg_comp_daily():
    """Plot comparison of ERA5 and PRISM data at DAILY timescales"""

    ## Load daily-frequency precip data
    p_agg = load_agg_data(detrend=True, seasonal=False)

    ## Make plot
    fig, axs = plot_scatter_and_diff(
        p_prism=p_agg["prism"],
        p_era5=p_agg["era5"],
        xlim=[-6, 17],
        xticks=np.arange(-5, 20, 5),
        yticks=np.arange(-5, 15, 5),
        s=0.25,
    )

    return fig, axs


def plot_timeseries_comp():
    """Plot comparison of seasonal total ERA5 and PRISM rainfall in Midwest"""

    ## Load data
    p_agg = load_agg_data(detrend=False, seasonal=True)
    p_agg_detrend = load_agg_data(detrend=True, seasonal=True)

    ## Create blank canvas for plotting
    width = src.params.plot_params["twocol_width"]
    aspect = 2.75
    height = width / aspect
    fig, [ax, ax2] = plt.subplots(1, 2, figsize=(width, height), layout="constrained")

    ## Plot before detrending
    ax.plot(p_agg.year, p_agg["prism"], label="PRISM")
    ax.plot(p_agg.year, p_agg["era5"], label="ERA5")
    ax.set_xticks(np.arange(1980, 2030, 10))
    ax.set_ylabel("AMJ precip. (mm)")
    ax.set_xlabel("Year")
    ax.legend(loc="lower right")

    ## Plot after detrending
    ax2.axhline(0, ls="--", c="k")
    ax2.plot(p_agg_detrend.year, p_agg_detrend["prism"], label="PRISM")
    ax2.plot(p_agg_detrend.year, p_agg_detrend["era5"], label="ERA5")
    ax2.set_xticks(np.arange(1980, 2030, 10))
    ax2.set_xlabel("Year")

    ## Label subplots
    ax = src.utils.label_subplot(
        ax, label="a)", posn=(0.1, 0.1), transform=ax.transAxes
    )
    ax2 = src.utils.label_subplot(
        ax2, label="b)", posn=(0.1, 0.1), transform=ax2.transAxes
    )

    return fig, [ax, ax2]


def plot_evap(ax, evap, levels=np.arange(0, 160, 10), cmap="cmo.rain"):
    """Set up plotting background and plot precip"""

    ### Set up plotting background
    ax, gl = src.utils.plot_setup_ax(
        ax=ax,
        plot_range=[233, 292, 22, 53],
        xticks=[-120, -105, -90, -75],
        yticks=[30, 40, 50],
        alpha=0.1,
        plot_topo=False,
    )

    ## Plot precip
    cp = ax.contourf(
        evap["longitude"],
        evap["latitude"],
        evap,
        extend="both",
        cmap=cmap,
        levels=levels,
        transform=ccrs.PlateCarree(),
    )

    ## Outline GPLLJ region
    ax.add_patch(
        mpatches.Rectangle(
            xy=[258, 25],
            width=5,
            height=10,
            facecolor="none",
            edgecolor="k",
            lw=mpl.rcParams["lines.linewidth"] * 2 / 3,
            transform=ccrs.PlateCarree(),
        )
    )

    # Plot outline of Midwest region
    ax.add_patch(
        mpatches.Polygon(
            xy=src.utils.midwest_vertices,
            closed=True,
            edgecolor="w",
            facecolor="none",
            ls="--",
            lw=mpl.rcParams["lines.linewidth"] * 2 / 3,
            transform=ccrs.PlateCarree(),
        )
    )

    return ax, gl, cp


def plot_E_comp_spatial():
    """Plot comparison of seasonal mean evaporation b/n ERA5 and GLEAM"""

    ## Load data (ordered as: ERA5, GLEAM, GLEAM_regrid, difference
    datasets = load_and_regrid_E()

    ## begin plotting
    aspect = 2
    width = src.params.plot_params["max_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")

    axs = []
    gls = []
    cps = []
    labels = ["a) ERA5", "b) GLEAM", "c) GLEAM*", r"d) ERA5 $-$ GLEAM"]
    # datasets = [E_era5, E_gleam, E_gleam_regrid, E_diff]
    for i, (data, label) in enumerate(zip(datasets, labels)):
        if i < 3:
            cmap = "cmo.rain"
            levels = np.arange(0, 480, 30)
        else:
            cmap = "cmo.balance_r"
            levels = src.utils.make_cb_range(150, 15)

        ## Setup plotting background
        ax = fig.add_subplot(
            2, 2, i + 1, projection=ccrs.PlateCarree(central_longitude=180)
        )

        ## Plot data
        evap = data["E"].groupby("time.year").sum("time").mean("year")
        ax, gl, cp = plot_evap(ax, evap=evap, cmap=cmap, levels=levels)
        gl.bottom_labels = False

        ## Label subplots
        ax = src.utils.label_subplot(
            ax, label=label, posn=(0.8, 0.1), transform=ax.transAxes
        )

        ## Append to record
        axs.append(ax)
        gls.append(gl)
        cps.append(cp)

    gls[1].left_labels = False
    gls[3].left_labels = False
    gls[0].top_labels = True
    gls[1].top_labels = True

    cb_mean = fig.colorbar(cps[1], ax=axs[1], orientation="vertical")
    cb_mean = src.utils.mod_colorbar(
        cb_mean, label=r"$mm$", ticks=np.arange(0, 540, 90), is_horiz=False
    )
    cb_anom = fig.colorbar(cps[3], ax=axs[3], orientation="vertical")
    cb_anom = src.utils.mod_colorbar(
        cb_anom, label=r"$mm$", ticks=np.arange(-150, 225, 75), is_horiz=False
    )

    return fig, axs


def load_E():
    """Load monthly evaporation data for ERA5 and GLEAM"""

    ## Open ERA5 data
    era = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/wam_fluxes_trim_monthly.nc")
    era = era.sel(time=slice("1980", None))[["E"]]  # .rename({"E":"era5"})

    ## Convert ERA5 units from m to mm
    mm_per_m = 1000.0
    era["E"] = era["E"] * mm_per_m
    era["E"].attrs.update({"units": "mm"})

    ## Open GLEAM data
    gleam = xr.open_dataset(f"{src.params.DATA_RAW_FP}/gleam.nc")

    ## Change GLEAM time index labels to month start (MS)
    years = gleam.time.dt.year.values
    months = gleam.time.dt.month.values
    dates_MS = pd.to_datetime([f"{y}-{m}" for y, m in zip(years, months)])
    gleam["time"] = dates_MS

    ## Select same dates as ERA5
    gleam = gleam.sel(time=era.time)

    ## Load ERA5 lsm (proportion of gridcell which is land)
    era5_lsm = xr.open_dataarray(f"{src.params.DATA_CONST_FP}/lsm.nc")
    era5_lsm = (
        era5_lsm.isel(time=0)
        .sel(longitude=era.longitude, latitude=era.latitude)
        .rename("mask")
    )

    ## convert continuous mask to binary
    era5_lsm = 1.0 * (era5_lsm > 0.5)

    ## filter out data over ocean
    era["E"] = era["E"] * era5_lsm

    ## compute GLEAM mask
    gleam_mask = ~np.isnan(gleam["E"].isel(time=0)).rename("mask")

    return xr.merge([era, era5_lsm]), xr.merge([gleam, gleam_mask])


def load_and_regrid_E():
    """Load ERA5 and GLEAM datasets, and perform regridding necessary for plotting"""

    ## Load data
    print("Loading and re-gridding evaporation data...")
    E_era5, E_gleam = load_E()
    print("Done.")

    ## subset in time for season of interest
    E_era5 = src.utils.get_month_range(E_era5, *src.params.MONTH_RANGE)
    E_gleam = src.utils.get_month_range(E_gleam, *src.params.MONTH_RANGE)

    ## Regrid GLEAM to match ERA5
    regridder = xe.Regridder(
        E_gleam,
        E_era5[["mask", "longitude", "latitude"]],
        "conservative_normed",
    )
    E_gleam_regrid = regridder(E_gleam, keep_attrs=True)

    ## Compute difference between ERA5 and GLEAM
    E_diff = E_era5 - E_gleam_regrid

    return E_era5, E_gleam, E_gleam_regrid, E_diff


def load_scatter_data():
    """Load data for scatter plot comparisons"""

    ## Load data
    # GLEAM
    gleam = xr.open_dataarray(
        f"{src.params.DATA_PREPPED_FP}/gleam_agg_monthly_detrend.nc"
    )
    gleam = gleam.squeeze().drop("mask").rename("gleam")

    # PRISM
    prism = xr.open_dataarray(
        f"{src.params.DATA_PREPPED_FP}/prism_agg_monthly_detrend.nc"
    )
    prism = prism.squeeze().drop("mask").rename("prism")

    # ERA5
    era = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/wam_fluxes_agg_monthly_detrend.nc"
    )
    era = era[["E", "P"]].sel(mask="Midwest").rename({"E": "E_era", "P": "P_era"})
    mm_per_m = 1000.0
    era *= mm_per_m

    # GPLLJ
    gpllj = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg_monthly_detrend.nc"
    )
    gpllj = gpllj["v"].rename("gpllj")

    # Merge data and filter months
    data_agg = xr.merge([gleam, prism, era, gpllj])
    data_agg = src.utils.get_month_range(data_agg, *src.params.MONTH_RANGE)

    return data_agg


data_agg = load_scatter_data()


def get_month_str(month_number):
    """converting month integers to string"""

    month_strs = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month_dict = dict(zip(np.arange(1, 13), month_strs))

    return month_dict[month_number]


def get_corr_by_month(y0, y1):
    """Get correlation for each month"""

    months = sorted(np.unique(y0.time.dt.month))
    corr = {"all": scipy.stats.pearsonr(y0, y1)}
    for m in months:
        y0_m = src.utils.get_month_range(y0, m, m)
        y1_m = src.utils.get_month_range(y1, m, m)
        corr.update({m: scipy.stats.pearsonr(y0_m, y1_m)})

    return corr


def scatter_by_month(ax, y0, y1, s, by_month=True):
    """scatter data, stratifying by month if desired"""

    ## scatter by month if desired
    if by_month:
        months = sorted(np.unique(y0.time.dt.month))
        markers = ["x", "^", "."]
        for i, m in enumerate(months):
            ax.scatter(
                src.utils.get_month_range(y0, m, m),
                src.utils.get_month_range(y1, m, m),
                marker=markers[i],
                s=s,
                label=get_month_str(m),
                alpha=0.7,
            )
    else:
        ax.scatter(y0, y1, s=s)

    return ax


def get_one_to_one(y0, y1):
    """Get limits of data (for plotting best fit line"""
    y_all = np.concatenate([y0.values, y1.values])
    y_lim = np.max(np.abs(y_all))
    return y_lim


def print_stats(y0, y1, z):
    """Print correlation and bias stats:"""

    ## Get correlation between two variables
    corr_abs = get_corr_by_month(y0, y1)

    ## Get correlation between their difference
    ## and a third variable
    corr_bias = get_corr_by_month(y1 - y0, z)

    ## Print out results
    for corr, name in zip([corr_abs, corr_bias], ["Comp.", "Bias"]):
        print(f"\n{name}")
        for m in list(corr):
            print(f"{m}: r={corr[m][0]:.2f} (p={corr[m][1]:.1e})")
    return


def plot_evap_comp(data):
    """Plot comparison of evaporation datasets"""

    #### Print out stats
    print("\nEvap. stats:")
    year_range = ("1980", "2020")
    print_stats(
        y0=data["gleam"].sel(time=slice(*year_range)),
        y1=data["E_era"].sel(time=slice(*year_range)),
        z=data["gpllj"].sel(time=slice(*year_range)),
    )

    #### Setup figure
    width = src.params.plot_params["twocol_width"]
    aspect = 2.5
    height = width / aspect
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(width, height), layout="constrained")

    #### Scatter obs. vs reanalysis
    ax0 = scatter_by_month(ax0, y0=data["gleam"], y1=data["E_era"], s=7.5)

    ## Plot one-to-one line
    lim = np.max(np.abs(data["E_era"]))
    ax0.plot([-lim, lim], [-lim, lim], ls="--", c="k", lw=0.5, alpha=0.6)

    ## Format plot
    ticks = np.arange(-20, 30, 10)
    ax0.set_xlim([-22, 22])
    ax0.set_ylim([-22, 22])
    ax0.set_xticks(ticks)
    ax0.set_yticks(ticks)
    ax0.set_xlabel("GLEAM (mm)")
    ax0.set_ylabel("ERA5 (mm)")
    ax0.set_aspect("equal")
    ax0.legend()

    #### Scatter GPLLJ vs. Error
    ax1 = scatter_by_month(
        ax1, y0=data["gpllj"], y1=data["E_era"] - data["gleam"], s=7.5
    )

    ## Plot axes
    ax1.axhline(0, ls="--", c="k", lw=0.5, alpha=0.5)
    ax1.axvline(0, ls="--", c="k", lw=0.5, alpha=0.5)

    ## Format plot
    ax1.set_aspect(5 / 22)
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-22, 22])
    ax1.set_xlabel("GPLLJ anomaly (m/s)")
    ax1.set_ylabel(r"ERA5 $-$ GLEAM (mm)")

    ## Add subplot labels
    ax0 = src.utils.label_subplot(
        ax0, label="a)", posn=(0.9, 0.1), transform=ax0.transAxes
    )
    ax1 = src.utils.label_subplot(
        ax1, label="b)", posn=(0.9, 0.1), transform=ax1.transAxes
    )

    return fig, [ax0, ax1]


def plot_precip_comp(data):
    """Plot comparison of precipitation datasets"""

    #### Print out stats
    print("\n\nPRISM vs. ERA:")
    year_range = ("1981", "2020")
    print_stats(
        y0=data["prism"].sel(time=slice(*year_range)),
        y1=data["P_era"].sel(time=slice(*year_range)),
        z=data["gpllj"].sel(time=slice(*year_range)),
    )

    #### Setup figure
    width = src.params.plot_params["twocol_width"]
    aspect = 2.5
    height = width / aspect
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(width, height), layout="constrained")

    #### Scatter obs. vs reanalysis
    ax0 = scatter_by_month(ax0, y0=data["prism"], y1=data["P_era"], s=7.5)

    ## Plot one-to-one line
    lim = np.max(np.abs(data["P_era"]))
    ax0.plot([-lim, lim], [-lim, lim], ls="--", c="k", lw=0.5, alpha=0.6)

    ## Format plot
    ticks = np.arange(-60, 90, 30)
    ax0.set_xlim([-68, 68])
    ax0.set_ylim([-68, 68])
    ax0.set_xticks(ticks)
    ax0.set_yticks(ticks)
    ax0.set_xlabel("PRISM (mm)")
    ax0.set_ylabel("ERA5 (mm)")
    ax0.set_aspect("equal")
    ax0.legend()

    #### Scatter GPLLJ vs. Error
    ax1 = scatter_by_month(
        ax1, y0=data["gpllj"], y1=data["P_era"] - data["prism"], s=7.5
    )

    ## Plot axes
    ax1.axhline(0, ls="--", c="k", lw=0.5, alpha=0.5)
    ax1.axvline(0, ls="--", c="k", lw=0.5, alpha=0.5)

    ## Format plot
    ax1.set_aspect(5 / 22)
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-22, 22])
    ax1.set_xlabel("GPLLJ anomaly (m/s)")
    ax1.set_ylabel(r"ERA5 $-$ PRISM (mm)")

    ## Add subplot labels
    ax0 = src.utils.label_subplot(
        ax0, label="a)", posn=(0.9, 0.1), transform=ax0.transAxes
    )
    ax1 = src.utils.label_subplot(
        ax1, label="b)", posn=(0.9, 0.1), transform=ax1.transAxes
    )

    return fig, [ax0, ax1]


def main():
    if src.params.season == "jja":
        ## missing values for august, so only run for amj/mjj
        pass

    else:
        #### set plotting style
        src.params.set_plot_style()

        #### Plot comparison of timeseries
        fig, axs = plot_timeseries_comp()
        src.utils.save(fig, fname="prism-era5_seasonal-timeseries")
        plt.close(fig)

        #### Plot comparison with daily data
        fig, axs = plot_agg_comp_daily()
        src.utils.save(fig, fname="prism-era5_daily-scatter")
        plt.close(fig)

        #### Plot comparison with seasonal data
        fig, axs = plot_agg_comp_seasonal()
        src.utils.save(fig, fname="prism-era5_seasonal-scatter")
        plt.close(fig)

        #### Plot comparison of seasonal timeseries
        fig, axs = plot_timeseries_comp()
        src.utils.save(fig, fname="prism-era5_seasonal-timeseries")
        plt.close(fig)

        #### Plot spatial comparison of ERA5 and GLEAM
        fig, axs = plot_E_comp_spatial()
        src.utils.save(fig, fname="gleam-era5_spatial-comp")
        plt.close(fig)

        #### Load monthly aggregated data (ERA5, PRISM, GLEAM)
        #### for comparison.
        data_agg = load_scatter_data()

        #### Plot comparison of monthly evap. in ERA5 and GLEAM
        fig, axs = plot_evap_comp(data_agg)
        src.utils.save(fig, fname="gleam-era5_monthly-scatter")
        plt.close(fig)

        #### Plot comparison of monthly precip. in ERA5 and PRISM
        fig, axs = plot_precip_comp(data_agg)
        src.utils.save(fig, fname="prism-era5_monthly_scatter")
        plt.close(fig)

    return


if __name__ == "__main__":
    main()
