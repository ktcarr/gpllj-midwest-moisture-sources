"""
Script creates the following figures:
    - "reynolds_seasonal-cycle.png"
    - "reynolds_timeseries.png"
"""

import pandas as pd
from copy import deepcopy
from os.path import join
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib as mpl
import src.utils
import src.params
import argparse


def unstack_time(x):
    """Function 'unstacks' time dimension in dataarray"""
    year = x.time.dt.year.values
    month = x.time.dt.month.values
    day = x.time.dt.day.values
    timeofday = x.time.dt.time.values
    time_idx = pd.MultiIndex.from_arrays(
        [year, month, day, timeofday], names=("year", "month", "day", "timeofday")
    )
    x_unstack = deepcopy(x)
    x_unstack = x_unstack.assign_coords({"time": time_idx}).unstack("time")
    return x_unstack


def bar(x):
    """Compute monthly average"""
    return x.mean(["day", "timeofday"])


def clim(x):
    """compute climatology"""
    return bar(x).mean("year")


def prime(x):
    """Deviation of monthly of instantaneous value from monthly avg"""
    return x - bar(x)


def hat(x):
    """Deviation of monthly average from climatology"""
    return bar(x) - clim(x)


def get_reynolds_decomp():
    """Compute reynolds decomposition of AMJ moisture flux in GPLLJ region"""

    ## Load $u,v,q$ data in GPLLJ region
    uvq = xr.open_dataset(f"{src.params.DATA_RAW_FP}/uvq_gpllj.nc")
    uvq = uvq.sel(level=850, latitude=slice(35, 25), longitude=slice(258, 263))
    uvq["q"] = uvq["q"] * 1000  # convert from kg/kg to g/kg

    # Unstack time dimension
    q = unstack_time(uvq["q"])
    v = unstack_time(uvq["v"])

    # Compute anomalies
    anom_total = bar(q * v) - clim(q * v)
    anom_thermo = hat(q) * clim(v)
    anom_dyn = clim(q) * hat(v)

    anom_cov_mon = hat(q) * hat(v) - (hat(q) * hat(v)).mean("year")
    anom_cov_hf = bar(prime(q) * prime(v)) - clim(prime(q) * prime(v))

    # Average over AMJ months and GPLLJ region area
    area_avg = lambda x: x.sel(month=slice(*src.params.MONTH_RANGE)).mean(
        ["month", "latitude", "longitude"]
    )
    amj_decomp = {
        "cov_mon": area_avg(anom_cov_mon),
        "cov_hf": area_avg(anom_cov_hf),
        "dyn": area_avg(anom_dyn),
        "thermo": area_avg(anom_thermo),
    }

    ### Next, compute climatology/variance
    mean_flow_clim = (bar(q) * bar(v)).mean("year").mean(["latitude", "longitude"])
    mean_flow_std = (bar(q) * bar(v)).std("year").mean(["latitude", "longitude"])
    eddy_cov_clim = (
        bar(prime(q) * prime(v)).mean("year").mean(["latitude", "longitude"])
    )
    eddy_cov_std = bar(prime(q) * prime(v)).std("year").mean(["latitude", "longitude"])
    monthly_clim = {
        "mean_flow_clim": mean_flow_clim,
        "mean_flow_std": mean_flow_std,
        "eddy_cov_clim": eddy_cov_clim,
        "eddy_cov_std": eddy_cov_std,
    }

    return amj_decomp, monthly_clim


def plot_amj_reynolds_decomp(ax):
    """Plot components of AMJ moisture flux over time"""

    ## Get decomp
    decomp, _ = get_reynolds_decomp()

    ## Plot data
    year = decomp["dyn"].year
    ax.plot(year, decomp["cov_mon"], label="Cov. (Seasonal)")
    ax.plot(year, decomp["cov_hf"], label="Cov. (High-freq.)")
    ax.plot(year, decomp["dyn"], label="Dynamic", c="k")
    ax.plot(year, decomp["thermo"], label="Thermo.", c="k", ls="--")
    ax.axhline(0, ls="-", lw=1, c="k", alpha=0.3)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Flux $\left(\frac{m}{s}\cdot\frac{g}{kg}\right)$")
    ax.legend()

    return ax


def plot_monthly_reynolds_decomp(ax):
    """plot mean flow vs. eddy contribution for each month"""

    ## Get decomposition
    _, clim = get_reynolds_decomp()

    ## Set up the plot
    ax = src.utils.plot_setup_monthly_ax(ax, yticks=np.arange(0, 90, 15), ylim=[-5, 80])

    p2 = ax.plot(
        np.arange(1, 13),
        clim["eddy_cov_clim"],
        label="Cov. (HF)",
        c=sns.color_palette("colorblind")[1],
    )
    p1 = ax.plot(np.arange(1, 13), clim["mean_flow_clim"], label="Mean", c="k")

    # Plot the shading (standard deviation)
    ax.fill_between(
        np.arange(1, 13),
        clim["mean_flow_clim"] + clim["mean_flow_std"],
        clim["mean_flow_clim"] - clim["mean_flow_std"],
        alpha=0.15,
        color=p1[0].get_color(),
    )

    # Plot the shading (standard deviation)
    ax.fill_between(
        np.arange(1, 13),
        clim["eddy_cov_clim"] + clim["eddy_cov_std"],
        clim["eddy_cov_clim"] - clim["eddy_cov_std"],
        alpha=0.15,
        color=p2[0].get_color(),
    )

    ax.axhline(0, ls="--", c="k", lw=mpl.rcParams["lines.linewidth"] / 2)
    ax.set_xlabel("Month")
    ax.set_ylabel(r"$\frac{g}{kg}\cdot\frac{m}{s}$")

    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ## Set plots to specified scale
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_scale", default=1.0, type=float)
    args = parser.parse_args()
    src.params.set_plot_style(scale=args.plot_scale)  # set plotting style for figures

    ## Make plot
    aspect = 1.33
    width = plot_params["onecol_width"]
    height = width / height
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    ax = plot_monthly_reynolds_decomp(ax)
    ax.legend(loc="upper right")
    src.utils.save(fig=fig, fname="reynolds_seasonal-cycle")
    plt.close(fig)
