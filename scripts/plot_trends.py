import src.utils
import xarray as xr
import numpy as np
import pandas as pd
import os
import seaborn as sns
from progressbar import progressbar
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.params
import argparse


def prep_for_plot(y, standardize=True):
    """Convenience function: get necessary data for plotting"""

    ## Get linear best-fit on data
    fit = src.utils.compute_stats(x=y.year.values, y=y.values)

    ## Compile results
    results = {
        "x_plot": y.year,
        "y_plot": y,
        "xfit": fit["x_"],
        "yfit": fit["y_"],
        "coef": 10 * fit["coef"][0].item(),  # convert from per year to per decade
    }

    return results


def load_data_seasonal(plot_var):

    season=src.params.season

    ## Load WAM
    wam_agg = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/wam_fluxes_agg_seasonal_{season}.nc"
    )
    wam_agg = wam_agg["E_track"]
    wam_agg *= 1000  # convert from m to mm

    ## Load GPLLJ
    gpllj = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg_seasonal_{season}.nc")[
        plot_var
    ]
    gpllj = gpllj.sel(year=slice(None, 2020))

    return wam_agg, gpllj


def load_data_monthly():
    ## Load WAM
    wam_agg = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/wam_fluxes_agg_monthly.nc")
    wam_agg = wam_agg["E_track"]
    wam_agg *= 1000  # convert from m to mm

    ## Load GPLLJ
    gpllj = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg_monthly.nc")[
        args.plot_var
    ]
    gpllj = gpllj.sel(time=slice(None, "2020-12-31"))

    ## merge data
    data = xr.merge([wam_agg, gpllj])

    return data


def add_timeseries_to_ax(ax, ts, color, label, standardize=True):
    """Add timeseries and best fit to ax object.
    'color' is the color of the line to plot"""

    ## Get plot data
    y_plot = ts["y_plot"]
    yfit = ts["yfit"]

    ## standardize if desired
    if standardize:
        ## function to standardize
        mean_ = y_plot.values.mean()
        std_ = y_plot.values.std()
        standardize_fn = lambda x: (x - mean_) / std_

        ## apply to data
        y_plot = standardize_fn(y_plot)
        yfit = standardize_fn(yfit)

    ## First, plot raw timeseries
    ax.plot(ts["x_plot"], y_plot, c=color, label=label)

    ## Next, plot best-fit line
    ax.plot(ts["xfit"], yfit, c=color, ls="--")

    return ax


def plot_ocean_gpllj_timeseries(ax, res_ocean, res_gpllj):
    """Plot standardized timeseries for ocean and GPLLJ"""

    ## Label plot
    ax.set_ylabel("Standard devs.")
    ax.set_xlabel("Year")
    ax.set_xticks(np.arange(1980, 2030, 10))
    ax = src.utils.plot_zero_line(ax)

    ## Plot data
    colors = sns.color_palette("colorblind")
    ax = add_timeseries_to_ax(
        ax, ts=res_ocean, color=colors[0], label="Ocean", standardize=True
    )
    ax = add_timeseries_to_ax(
        ax, ts=res_gpllj, color="k", label="GPLLJ", standardize=True
    )

    ax.legend(loc="lower right")

    return ax


def plot_ocean_land_timeseries(ax, res_ocean, res_land):
    """Plot standardized timeseries for ocean and GPLLJ"""

    ## Labl plot
    ax.set_ylabel(r"$mm$")
    ax.set_xlabel("Year")
    ax.set_xticks(np.arange(1980, 2030, 10))

    ## Plot data
    colors = sns.color_palette("colorblind")
    ax = add_timeseries_to_ax(
        ax, ts=res_ocean, color=colors[0], label="Ocean", standardize=False
    )
    ax = add_timeseries_to_ax(
        ax, ts=res_land, color=colors[2], label="Land", standardize=False
    )

    ## if desired, include zero line
    ax = src.utils.plot_zero_line(ax)
    ax.set_ylim([-25, None])

    ## legend
    ax.legend(loc=(0.6, 0.2))

    return ax


def get_coef_and_mc_bounds(data, plot_var="v"):
    """get regression coefficient and MC bounds"""

    months = np.arange(1, 13)
    coefs = []
    mc_bounds = []
    mc_fp = f"{src.params.DATA_PREPPED_FP}/mc-bounds_trend_{plot_var}.nc"

    for month in progressbar(months):
        ## Select data from relevant month
        data_ = src.utils.get_month_range(data, month, month)

        ## Get index to regress onto (year, in this case)
        idx = data_.time.dt.year.values.astype(float)
        idx = xr.DataArray(idx, dims=["time"], coords={"time": data_.time})

        ## Function used to compute regression coefficient
        compute_coef_ = lambda data, idx: src.utils.ls_fit_xr(Y_data=data, idx=idx).sel(
            coef="m"
        )

        ## Do the computation
        coefs.append(compute_coef_(data_, idx))

        ## MC test (if it doesn't exist)
        if not os.path.isfile(mc_fp):
            mc_bounds.append(
                src.utils.get_mc_bounds(
                    data_,
                    idx,
                    compute_coef=compute_coef_,
                    nsims=1000,
                    seed=month * 100,
                )
            )

    ## concatenate coefficients
    coefs = xr.concat(coefs, dim=pd.Index(months, name="month"))

    ## Save/load monte-carlo bounds
    if not os.path.isfile(mc_fp):
        mc_bounds = xr.concat(mc_bounds, dim=pd.Index(months, name="month"))
        mc_bounds.to_netcdf(mc_fp)
    else:
        mc_bounds = xr.open_dataset(mc_fp)

    return coefs, mc_bounds


def plot_monthly_trends(ax, data, coefs, mc_bounds, plot_var):
    ax = src.utils.plot_setup_monthly_ax(
        ax,
        yticks=np.round(np.arange(-0.4, 0.6, 0.2), 1),
        ylim=[-0.45, 0.45],
        zero_line_width=1 / 3,
    )

    ## Non-dimensionalize data by dividing by monthly
    ## standard deviation.to have consistent units
    ## Also, multiply by 10 to convert from 1/year to 1/decade
    get_norm_factor = lambda x: 10 / x.groupby("time.month").std()
    gpllj_norm_factor = get_norm_factor(data[plot_var])
    ocean_norm_factor = get_norm_factor(data["E_track"].sel(mask="Ocean"))
    land_norm_factor = get_norm_factor(data["E_track"].sel(mask="Land"))

    ## plot gpllj curve
    ax = src.utils.add_data_with_sig(
        ax,
        x=coefs[plot_var] * gpllj_norm_factor,
        bounds=mc_bounds[plot_var] * gpllj_norm_factor,
        color="k",
        label="GPLLJ",
    )

    ## plot ocean and land curves
    factors = [ocean_norm_factor, land_norm_factor]
    for mask, color_idx, factor in zip(["Ocean", "Land"], [0, 2], factors):
        ax = src.utils.add_data_with_sig(
            ax,
            x=coefs["E_track"].sel(mask=mask) * factor,
            bounds=mc_bounds["E_track"].sel(mask=mask) * factor,
            color=sns.color_palette()[color_idx],
            label=mask,
        )

    ax.set_ylabel("Standard devs.")
    ax.set_xlabel("Month")
    ax.legend(loc="upper right", framealpha=1)

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_var", default="v", choices=["v", "q"])
    args = parser.parse_args()

    ## Set custom plot style
    src.params.set_plot_style()

    # Load data
    wam_agg_seasonal, gpllj_seasonal = load_data_seasonal(plot_var=args.plot_var)

    # Compute correlation before/after detrending
    for mask in ["Ocean", "Land", "Lake"]:
        print(f"Stats for {mask}:")
        a = wam_agg_seasonal.sel(mask=mask)
        b = gpllj_seasonal
        stats = src.utils.compute_stats(a.values, b.values)
        print("Before detrending:")
        print(f"r = {stats['rho']:.3f} (p = {stats['pval']:.2e})")

        a = src.utils.detrend_dim(wam_agg_seasonal.sel(mask=mask), dim="year")
        b = src.utils.detrend_dim(gpllj_seasonal, dim="year")
        stats = src.utils.compute_stats(a.values, b.values)
        print(f"\nAfter detrending:")
        print(f"r = {stats['rho']:.3f} (p = {stats['pval']:.2e})")
        print()

    # Prep timeseries for plotting
    res_ocean = prep_for_plot(wam_agg_seasonal.sel(mask="Ocean"))
    res_land = prep_for_plot(wam_agg_seasonal.sel(mask="Land"))
    res_gpllj = prep_for_plot(gpllj_seasonal)

    # Get monthly trends
    data_monthly = load_data_monthly()
    coefs, bounds = get_coef_and_mc_bounds(data_monthly, plot_var=args.plot_var)

    # trim to exclude dec. (spin-up error)
    coefs = coefs.sel(month=slice(None, 11))
    bounds = bounds.sel(month=slice(None, 11))

    ##### Three-paneled plot ####
    aspect = 3.5
    width = src.params.plot_params["max_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    ax1 = fig.add_subplot(1, 3, 1)
    ax1 = plot_ocean_gpllj_timeseries(ax1, res_ocean, res_gpllj)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2 = plot_ocean_land_timeseries(ax2, res_ocean, res_land)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3 = plot_monthly_trends(ax3, data_monthly, coefs, bounds, args.plot_var)

    # Add labels to subplots
    ax1 = src.utils.label_subplot(
        ax1, label="a)", posn=(0.1, 0.9), transform=ax1.transAxes
    )
    ax2 = src.utils.label_subplot(
        ax2, label="b)", posn=(0.1, 0.9), transform=ax2.transAxes
    )
    ax3 = src.utils.label_subplot(
        ax3, label="c)", posn=(0.1, 0.9), transform=ax3.transAxes
    )

    fig.get_layout_engine().set(wspace=0.05)
    src.utils.save(fig=fig, fname=f"trends_over_time_{args.plot_var}", is_supp=False)
    plt.close(fig)

    # Print out trends
    print(f"Ocean trend: {np.round(res_ocean['coef'],1)}" + r" $\frac{mm}{decade}$")
    print(f"Land trend:  {np.round(res_land['coef'],1)}" + r" $\frac{mm}{decade}$")
    if args.plot_var == "v":
        print(
            f"GPLLJ trend:   {np.round(res_gpllj['coef'],2)}" + r" $\frac{m/s}{decade}$"
        )
    else:
        print(
            f"GPLLJ trend:   {np.round(1000*res_gpllj['coef'],2)}"
            + r" $\frac{g/kg}{decade}$"
        )
