import src.utils
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse
import xarray as xr
import pandas as pd
import seaborn as sns
import os
from progressbar import progressbar
import src.params


def broadcast_array(x, shape):
    new_dims = []
    for i, s in enumerate(shape):
        if s not in x.shape:
            new_dims.append(i)
    if len(new_dims) >= (len(shape) - 1):  # check to see if none of dimensions match
        new_dims = new_dims[1:]
    return np.expand_dims(x, new_dims)


def get_composite_da(data, **kwargs):
    """Create composite from dataarray"""
    idx = kwargs["idx"]
    n = kwargs["n"]

    sorted_idx = np.argsort(idx.values, axis=0)
    idx_hi = broadcast_array(sorted_idx[::-1][:n], data.shape)
    idx_lo = broadcast_array(sorted_idx[:n], data.shape)

    ## Create composite
    data_ = data.transpose("time", ...)  # make sure time is first dimension
    composite = deepcopy(data_.isel(time=0))
    composite.values = np.nanmean(
        np.take_along_axis(data_.values, idx_hi, axis=0), 0
    ) - np.nanmean(np.take_along_axis(data_.values, idx_lo, axis=0), 0)
    return composite / 2


def get_composite(data, idx, n=10):
    """Create composite from dataset or dataarray"""
    return src.utils.xr_wrapper(get_composite_da, data, idx=idx, n=n)


def load_data_regression(detrend, standardize):
    """Load monthly aggregated data.
    'detrend' and 'standardize' are boolean options specifying whether
    to use data which has been preprocessed in that way"""

    ### Load aggregated WAM output
    if detrend:
        wam_agg = xr.open_dataset(
            f"{src.params.DATA_PREPPED_FP}/wam_fluxes_agg_monthly_detrend.nc"
        )
    else:
        wam_agg = xr.open_dataset(
            f"{src.params.DATA_PREPPED_FP}/wam_fluxes_agg_monthly.nc"
        )

    wam_agg = wam_agg["E_track"]  # only care about tracked moisture here
    wam_agg *= 1000  # convert from m to mm

    ## Load GPLLJ index
    if detrend:
        gpllj_idx = xr.open_dataset(
            f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg_monthly_detrend.nc"
        )
    else:
        gpllj_idx = xr.open_dataset(
            f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg_monthly.nc"
        )
    gpllj_idx = gpllj_idx["v"].sel(time=slice(None, "2020-12-31"))

    ## Standardize gpllj index if desired
    if standardize:
        gpllj_idx = (
            gpllj_idx.groupby("time.month") / gpllj_idx.groupby("time.month").std()
        )

    return {"wam_agg": wam_agg, "gpllj_idx": gpllj_idx}


def load_data_clim():
    """load data for monthly climatology"""

    ### Load aggregated WAM output
    wam_agg = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/wam_fluxes_agg_monthly.nc")
    wam_agg *= 1000  # convert from m to mm

    ### Compute climatology and standard deviation
    mean = wam_agg.groupby("time.month").mean("time")
    std = wam_agg.groupby("time.month").std("time")

    ## For convenience, compute clim ± std
    lb = mean - std
    ub = mean + std

    return {"mean": mean, "std": std, "lb": lb, "ub": ub}


def compute_coef(data, idx, plot_type):
    """Based on user input, define function to compute coefficient
    for a given month"""
    if plot_type == "regression":
        return src.utils.ls_fit_xr(Y_data=data, idx=idx).sel(coef="m")

    elif plot_type == "composite":
        return get_composite(data, idx, n=10)

    else:
        print(f"Error: {args.plot_type} is not a valid plot type.")
        return


def get_coefs_mc_bounds(gpllj_idx, wam_agg, plot_type, suffix):
    """get regression/composite coefficients"""

    ## Specify function used to compute coefficients
    compute_coef_ = lambda data, idx: compute_coef(data, idx, plot_type)

    ## Compute regression/composite coefficients separately for each month
    months = np.arange(1, 13)
    coefs = []
    mc_bounds = []
    mc_fp = f"{src.params.DATA_PREPPED_FP}/mc-bounds_{suffix}.nc"

    for month in progressbar(months):
        ## Select data from relevant month
        gpllj_idx_ = src.utils.get_month_range(gpllj_idx, month, month)
        wam_agg_ = src.utils.get_month_range(wam_agg, month, month)

        ## Compute coefficient
        coefs.append(compute_coef_(wam_agg_, gpllj_idx_))

        ## MC test (if it doesn't exist)
        if not os.path.isfile(mc_fp):
            mc_bounds.append(
                src.utils.get_mc_bounds(
                    wam_agg_,
                    gpllj_idx_,
                    compute_coef=compute_coef_,
                    nsims=1000,
                    seed=month,
                )
            )

    ## concatenate coefficients
    coefs = xr.concat(coefs, dim=pd.Index(months, name="month"))

    ## Save/load monte-carlo bounds
    if not os.path.isfile(mc_fp):
        mc_bounds = xr.concat(mc_bounds, dim=pd.Index(months, name="month"))
        mc_bounds.to_netcdf(mc_fp)
    else:
        mc_bounds = xr.open_dataset(mc_fp)["E_track"]

    return coefs, mc_bounds


def plot_clim_ep(ax, clim):
    """Plot climatology of ocean vs. land moisture sources"""

    ## Get E_track and P_midwest
    e = clim["mean"]["E_track"].sel(mask="Total")
    p = clim["mean"]["P"].sel(mask="Midwest")

    ## print out total error
    pct_error = (e.sum() - p.sum()) / p.sum()
    pct_error_dec = (e.sum() - p.sum()) / p.sel(month=12).sum()
    print(f"\nE-P budget error:         {100*pct_error:.2f}%")
    print(f"E-P budget error (Dec.): {100*pct_error_dec:.2f}%\n")

    ## set up plotting canvas
    ax = src.utils.plot_setup_monthly_ax(
        ax=ax,
        ylim=[-5, 130],
        yticks=np.arange(0, 150, 25),
        zero_line_width=2 / 3,
    )

    ## plot the data
    ax.plot(e.month, e, c="k", ls="--", label="Tracked evap.")
    ax.plot(p.month, p, c="k", ls="-", label="Midwest precip.")
    ax.set_xlim([None, 12.5])

    ## Labels
    ax.legend()
    ax.set_ylabel(r"Contribution ($mm$)")
    ax.set_xlabel("Month")

    return ax


def plot_clim_oc_v_la(ax, clim):
    """Plot climatology of ocean vs. land moisture sources"""

    ## set up plotting canvas
    ax = src.utils.plot_setup_monthly_ax(
        ax=ax,
        ylim=[-5, 90],
        yticks=np.arange(0, 100, 20),
        zero_line_width=2 / 3,
    )

    ## get colors and labels for plot
    colors = [sns.color_palette("colorblind")[i] for i in [0, 2, 1]]
    masks = ["Ocean", "Land", "Midwest"]

    ## Plot the data
    for mask, color in zip(masks, colors):
        ax.plot(
            clim["mean"].month,
            clim["mean"]["E_track"].sel(mask=mask),
            label=mask,
            c=color,
        )
        ax.fill_between(
            clim["mean"].month,
            clim["ub"]["E_track"].sel(mask=mask),
            clim["lb"]["E_track"].sel(mask=mask),
            alpha=0.1,
            color=color,
        )
    ax.legend()

    return ax


def plot_clim_atl_v_pac(ax, clim):
    """Plot climatology of atlantic vs pacific moisture source"""

    colors = sns.color_palette("mako")
    colors = [colors[i] for i in [0, 3]]

    ax = src.utils.plot_setup_monthly_ax(
        ax=ax,
        ylim=[-2, 52],
        yticks=np.arange(0, 60, 10),
        zero_line_width=2 / 3,
    )

    for mask, color in zip(["Pacific", "Atlantic"], colors):
        ax.plot(
            clim["mean"].month,
            clim["mean"]["E_track"].sel(mask=mask),
            label=mask,
            c=color,
        )
        ax.fill_between(
            clim["mean"].month,
            clim["ub"]["E_track"].sel(mask=mask),
            clim["lb"]["E_track"].sel(mask=mask),
            alpha=0.1,
            color=color,
        )
    ax.legend()

    return ax


def plot_coef_oc_v_la(ax, coefs, mc_bounds):
    """Plot coefficients of ocean vs. land composite or regressed on GPLLJ"""

    colors = sns.color_palette("colorblind")
    masks = ["Ocean", "Midwest", "Land"]
    yticks = np.arange(-6, 12, 3)
    ylim = [-7, 10]

    ax = src.utils.plot_setup_monthly_ax(
        ax=ax, yticks=yticks, ylim=ylim, zero_line_width=2 / 3
    )
    for mask, color in zip(masks, colors):
        ax = src.utils.add_data_with_sig(
            ax,
            x=coefs.sel(mask=mask),
            bounds=mc_bounds.sel(mask=mask),
            color=color,
            label=mask,
        )

    return ax


def plot_coef_atl_v_pac(ax, coefs, mc_bounds):
    """Plot coefficients for atlantic v. pacific composite or regressed on GPLLJ"""

    colors = [sns.color_palette("mako")[i] for i in [0, 3]]
    masks = ["Pacific", "Atlantic"]
    yticks = np.arange(-4, 8, 2)
    ylim = [-4.5, 6.5]

    ax = src.utils.plot_setup_monthly_ax(
        ax=ax, yticks=yticks, ylim=ylim, zero_line_width=2 / 3
    )
    for mask, color in zip(masks, colors):
        ax = src.utils.add_data_with_sig(
            ax,
            x=coefs.sel(mask=mask),
            bounds=mc_bounds.sel(mask=mask),
            color=color,
            label=mask,
        )
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot_type", default="regression", choices=["regression", "composite"]
    )
    parser.add_argument(
        "--detrend", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--standardize", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    src.params.set_plot_style()

    print("Plotting options")
    print(f"Plot_type:   {args.plot_type}")
    print(f"Detrend:     {args.detrend}")
    print(f"Standardize: {args.standardize}")

    ## Create suffix for naming files
    suffix = (
        f"detrend-{args.detrend}_"
        + f"standardize-{args.standardize}_"
        + f"plottype-{args.plot_type}_"
        + f"{src.params.season}"
    )

    ## Load data
    clim = load_data_clim()
    data = load_data_regression(detrend=args.detrend, standardize=args.standardize)

    ## compute coefficients
    coefs, mc_bounds = get_coefs_mc_bounds(
        data["gpllj_idx"], data["wam_agg"], plot_type=args.plot_type, suffix=suffix
    )

    #### First, plot E_track and Midwest P residual

    aspect = 1.25
    width = src.params.plot_params["onecol_width"]
    height = width / aspect
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")

    ## Climatology
    ax = plot_clim_ep(ax, clim)

    # Save figure
    src.utils.save(fig=fig, fname="ep_budget_agg", is_supp=True)
    plt.close(fig)

    #### Next, trim to exclude december (spin-up error)
    coefs = coefs.sel(month=slice(None, 11))
    mc_bounds = mc_bounds.sel(month=slice(None, 11))
    for varname in list(clim):
        clim[varname] = clim[varname].sel(month=slice(None, 11))

    ## Plot
    aspect = 1.25
    width = src.params.plot_params["twothirds_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(ncols=3, nrows=2, width_ratios=[1, 0.1, 1])

    ## Climatology
    ax1 = fig.add_subplot(gs[0, 0])
    ax1 = plot_clim_oc_v_la(ax1, clim)
    ax1.set_ylabel(r"Contribution ($mm$)")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2 = plot_clim_atl_v_pac(ax2, clim)
    ax2.set_ylabel(r"Contribution ($mm$)")
    ax2.set_xlabel("Month")

    # Regression coefficients
    if args.standardize:
        ylabel = r"Regression coef. ($\frac{mm}{\sigma}$)"
    else:
        ylabel = r"Regression coef. ($\frac{mm}{m/s}$)"

    ax3 = fig.add_subplot(gs[0, 2])
    ax3 = plot_coef_oc_v_la(ax3, coefs, mc_bounds)
    ax3.set_ylabel(ylabel)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4 = plot_coef_atl_v_pac(ax4, coefs, mc_bounds)
    ax4.set_ylabel(ylabel)
    ax4.set_xlabel("Month")

    # Add labels to subplots
    label_posn = (0.1, 0.9)
    ax1 = src.utils.label_subplot(
        ax1, label="a)", posn=label_posn, transform=ax1.transAxes
    )
    ax2 = src.utils.label_subplot(
        ax2, label="b)", posn=label_posn, transform=ax2.transAxes
    )
    ax3 = src.utils.label_subplot(
        ax3, label="c)", posn=label_posn, transform=ax3.transAxes
    )
    ax4 = src.utils.label_subplot(
        ax4, label="d)", posn=label_posn, transform=ax4.transAxes
    )

    # Save figure
    src.utils.save(fig=fig, fname="moisture_sources_seasonal_agg", is_supp=False)
    plt.close(fig)
