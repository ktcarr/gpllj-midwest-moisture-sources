from os.path import join
import xarray as xr
import numpy as np
import matplotlib as mpl
import cmocean
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import src.utils
import src.params
from reynolds_decomp import plot_amj_reynolds_decomp

"""
Produces the following figures:
    - "v-and-q_seasonal-cycle.png"
    - "uv850_and_tcwv.png"
    - "v-and-q_cross-section.png"
"""


def load_data():
    ## subset of months to average over

    # Load TCWV data
    # Note: units of kg/m^3 are equivalent to mm!
    # To convert: TCWV from kg/m^2 to mm, divide by density of water and multiply by 1000 mm / m)
    # kg/m^2 * (m^3 / 1000 kg) * (1000 mm / m) = mm
    tcwv = xr.open_dataarray(f"{src.params.DATA_RAW_FP}/tcwv.nc")
    tcwv = src.utils.get_month_range(tcwv, *src.params.MONTH_RANGE).mean("time")

    # Load uvq data
    uvq = xr.open_dataset(f"{src.params.DATA_RAW_FP}/uvq_wide.nc")
    uvq = src.utils.get_month_range(uvq, *src.params.MONTH_RANGE).mean("time")
    uvq = uvq.sel(level=850)

    ## Load surface pressure data
    sp = xr.open_dataarray(f"{src.params.DATA_RAW_FP}/sp.nc")
    sp = src.utils.get_month_range(sp, *src.params.MONTH_RANGE).mean("time")
    sp = sp.interp({"longitude": uvq.longitude, "latitude": uvq.latitude})
    sp = sp / 100  # convert Pa to hPa

    # ## erase values below surface
    for varname in ["u", "v", "q"]:
        uvq[varname].values[sp < 850] = np.nan

    ## Crosss-section data
    lat = 30
    lon_range = (257, 270)

    # Load uvq cross-section data
    uvq_cross = xr.open_dataset(f"{src.params.DATA_RAW_FP}/vq_cross.nc")
    uvq_cross = uvq_cross.sel(latitude=lat, longitude=slice(*lon_range))
    uvq_cross = src.utils.get_month_range(uvq_cross, *src.params.MONTH_RANGE).mean(
        "time"
    )

    # Load surface pressure data
    sp_cross = xr.open_dataarray(f"{src.params.DATA_RAW_FP}/sp.nc")
    sp_cross = sp_cross.sel(latitude=lat, longitude=slice(*lon_range))
    sp_cross = src.utils.get_month_range(sp_cross, *src.params.MONTH_RANGE).mean("time")

    ## Unit conversions
    sp_cross = sp_cross / 100  # convert Pa to hPa
    uvq_cross["q"] = uvq_cross["q"] * 1000  # convert kg/kg to g/kg

    # interpolate to higher vertical resolution
    new_levels = np.concatenate(
        [uvq_cross.level.values[:-5], np.arange(900, 1006.25, 6.25)]
    )
    uvq_cross = uvq_cross.interp({"level": new_levels, "longitude": sp_cross.longitude})

    ## erase values below surface
    x, z = np.meshgrid(sp_cross.longitude, new_levels)
    for varname in ["v", "q"]:
        uvq_cross[varname].values[z > sp_cross.values[None, :]] = np.nan

    return {"tcwv": tcwv, "uvq": uvq, "uvq_cross": uvq_cross, "sp_cross": sp_cross}


def plot_tcwv_clim(ax, tcwv, uvq):
    """Plot AMJ climatology for TCWV and UVQ in continental U.S."""

    ## setup background for the plot
    ax, gl = src.utils.plot_setup_ax(
        ax,
        plot_range=[235, 290, 15, 52],
        xticks=[-120, -105, -90, -75],
        yticks=[20, 30, 40, 50],
        alpha=0.1,
        plot_topo=True,
    )

    ## Plot TCWV
    cp = ax.contourf(
        tcwv.longitude,
        tcwv.latitude,
        tcwv,
        extend="max",
        cmap="cmo.rain",
        levels=np.arange(0, 60, 5),
        transform=ccrs.PlateCarree(),
    )

    ## Plot outline of Midwest region
    ax.add_patch(
        mpatches.Polygon(
            xy=src.utils.midwest_vertices,
            closed=True,
            edgecolor="w",
            facecolor="none",
            ls="--",
            zorder=1.005,
            lw=mpl.rcParams["lines.linewidth"] * 2 / 3,
            transform=ccrs.PlateCarree(),
            # path_effects=[
            #     pe.Stroke(
            #         linewidth=mpl.rcParams["lines.linewidth"] * 4/3,
            #         foreground="k",
            #     ),
            #     pe.Normal(),
            # ]
        )
    )

    ## GPLLJ outline region
    ax.add_patch(
        mpatches.Rectangle(
            xy=[258, 25],
            width=5,
            height=10,
            edgecolor="black",
            facecolor="none",
            ls="--",
            zorder=1.005,
            lw=mpl.rcParams["lines.linewidth"] * 2 / 3,
            transform=ccrs.PlateCarree(),
        )
    )

    ## Plot cross-section location
    ax.plot(
        [257, 270],
        [30, 30],
        transform=ccrs.PlateCarree(),
        c="white",
        ls=":",
        lw=mpl.rcParams["lines.linewidth"] * 2 / 3,
    )

    n = 4
    scale = 1e2
    x, y = np.meshgrid(uvq.longitude[::n].values, uvq.latitude[::n].values)
    qv = ax.quiver(
        x,
        y,
        uvq["u"][::n, ::n].values,
        uvq["v"][::n, ::n].values,
        scale=scale,
        color="k",
        alpha=1,
        transform=ccrs.PlateCarree(),
        zorder=1.05,
    )

    # Add legend for quiver plot
    ax.add_patch(
        mpatches.Rectangle(
            xy=[236, 16],
            width=9,
            height=9,
            facecolor="white",
            edgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=1.06,
            linewidth=src.params.plot_params["border_width"],
        )
    )
    qk = ax.quiverkey(
        qv,
        X=0.1,
        Y=0.05,
        U=scale / 20,
        label=f"{scale/20}" + r" $\frac{m}{s}$",
        fontproperties={"size": mpl.rcParams["legend.fontsize"]},
    )

    return ax, cp


def plot_vq_cross(ax, uvq_cross, sp_cross):
    """plot cross section of v and q in GPLLJ region"""

    ## Label axes
    ax.set_ylabel("hPa")
    ax.set_xlabel("Longitude")
    ax.set_ylim([600, 1000])
    ax.grid(False)

    ## Set xticks
    # ticks = np.array([260,265,270])
    ticks = np.array([258, 263, 268])
    xticklabels = [f"{360-t}" + r"$^{\circ}$W" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(xticklabels)

    ## Plot meridional velocity (colors)
    cp = ax.contourf(
        uvq_cross.longitude,
        uvq_cross.level,
        uvq_cross["v"],
        cmap="cmo.amp",
        extend="max",
        levels=np.arange(0, 9, 1),
    )

    ## Plot specific humidity (contours)
    cs = ax.contour(
        uvq_cross.longitude,
        uvq_cross.level,
        uvq_cross["q"],
        colors="w",
        extend="both",
        levels=np.arange(0, 14, 1),
        alpha=0.6,
    )

    ## Plot surface pressure
    ax.plot(sp_cross.longitude, sp_cross, c="k")

    ## Plot boundaries of LLJ region
    ax.axvline(258, c="k", ls="--", lw=mpl.rcParams["lines.linewidth"] * 2 / 3)
    ax.axvline(263, c="k", ls="--", lw=mpl.rcParams["lines.linewidth"] * 2 / 3)

    plt.gca().invert_yaxis()

    return ax, cp


def get_plot_data(X):
    """Utility function: get annual cycle normalized by max value"""
    mean = X.groupby("time.month").mean("time")
    std = X.groupby("time.month").std("time")

    # Compute maximum and normalize by it
    max_ = mean.max().item()
    mean_norm = mean / max_
    std_norm = std / max_

    return {"mean": mean_norm, "std": std_norm, "max": max_}


def plot_vq_clim(ax):
    """Plot monthly climatology of v and q in GPLLJ region"""
    # Load data
    gpllj_avg = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg_monthly.nc"
    )
    gpllj_avg["q"] = gpllj_avg["q"] * 1000  # convert from kg/kg to g/kg

    v_plot = get_plot_data(gpllj_avg["v"])
    q_plot = get_plot_data(gpllj_avg["q"])

    data = [v_plot, q_plot]
    labels = [r"$v$", r"$q$"]
    units = [r"$m/s$", r"$g/kg$"]

    # Plot parameters
    alpha = 0.2
    lw = 2
    colors = [sns.color_palette("colorblind")[i] for i in [1, 2]]

    ### Modify axis
    ax = src.utils.plot_setup_monthly_ax(
        ax, ylim=[0, 1.3], yticks=np.round(np.arange(0.2, 1.4, 0.2), 1)
    )

    ### Begin plotting
    ax.set_xlabel("Month")
    ax.set_ylabel("Normalized units")

    for i, (X, label, unit) in enumerate(zip(data, labels, units)):
        # Plot the `mean' line
        ax.plot(
            np.arange(1, 13),
            X["mean"],
            label=f'{label} (max = {X["max"]:.1f} {unit})',
            lw=lw,
            c=colors[i],
        )

        # Plot the shading (standard deviation)
        ax.fill_between(
            X["mean"].month,
            X["mean"] + X["std"],
            X["mean"] - X["std"],
            alpha=alpha,
            color=colors[i],
        )
    ax.legend(loc="lower center")

    return ax


if __name__ == "__main__":
    ## Set plots to specified style
    src.params.set_plot_style()

    #### Figure 1 #########################################################################
    data = load_data()

    # make plot
    aspect = 3  # width divided by height
    width = src.params.plot_params["twocol_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1.5, 0.1, 1])

    # Plot TCWV and uv850
    ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=180))
    ax1, cp1 = plot_tcwv_clim(ax1, tcwv=data["tcwv"], uvq=data["uvq"])
    cb1 = fig.colorbar(cp1, orientation="vertical", pad=0.02, label=r"TCWV ($mm$)")
    cb1 = src.utils.mod_colorbar(
        cb1,
        label=r"TCWV ($mm$)",
        ticks=np.arange(0, 60, 10),
        is_horiz=False,
    )

    # Plot q and v cross-section in GPLLJ region
    ax2 = fig.add_subplot(gs[2])
    ax2, cp2 = plot_vq_cross(ax2, data["uvq_cross"], data["sp_cross"])
    cb2 = fig.colorbar(cp2, orientation="vertical", pad=0.02)
    cb2 = src.utils.mod_colorbar(
        cb2,
        label=r"$m/s$",
        ticks=np.arange(0, 10, 2),
        is_horiz=False,
    )

    ## Label subplots
    ax1 = src.utils.label_subplot(
        ax=ax1, label="a)", posn=(-121, 48), transform=ccrs.PlateCarree()
    )
    ax2 = src.utils.label_subplot(ax=ax2, label="b)", posn=(258.5, 640))

    # Save figure
    src.utils.save(fig=fig, fname=f"fluxes_spatial", is_supp=False)
    plt.close(fig)

    # ##### Figure 2 #########################################################

    # make plot
    aspect = 2.9  # width divided by height
    width = src.params.plot_params["twocol_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1, 0.15, 1])

    # Plot v and q
    ax1 = fig.add_subplot(gs[0])
    ax1 = plot_vq_clim(ax1)

    # Plot timeseries of Reynolds decomposition
    ax2 = fig.add_subplot(gs[2])
    ax2 = plot_amj_reynolds_decomp(ax2)

    # Label subplots
    ax1 = src.utils.label_subplot(ax=ax1, label="a)", posn=(1.5, 1.15))
    ax2 = src.utils.label_subplot(ax=ax2, label="b)", posn=(1981, 10))

    # Save figure
    src.utils.save(fig=fig, fname="fluxes_gpllj", is_supp=False)
    plt.close(fig)
