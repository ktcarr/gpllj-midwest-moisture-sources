"""
Script produces the following figures:
    - "uq_rocky_flux_{27,32,37}.png"
    - "topo_cross_{27,32,37}.png"
"""

import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean
import src.utils
import src.params

############## Functions ################
def interp_plevels(sp, n_levels=37, top_level=100):
    """
    Linear interpolate pressure levels from surface to top level.
    Units of top level are Pa (not hPa).
    Note: WAM model configuration for this project uses boundary=29
    """

    # get pressure difference between levels
    dp = (sp - top_level) / (n_levels - 1)

    # compute pressure based on dp
    p = np.stack([sp - n * dp for n in np.arange(n_levels)[::-1]], axis=0)

    return p


def get_pressure_boundary(sp, boundary_idx, n_levels=37, top_level=100):
    """Get boundary between upper and lower layers in two-layer model.
    Uses the same function as WAM2layers script"""
    plevels = interp_plevels(sp=sp, n_levels=n_levels, top_level=top_level)
    return plevels[boundary_idx]


def plot_uq_cross(ax, flux_, sp_):
    """Make plot for specified flux and surface pressure.
    'flux_' must be 2d (x-z) and 'sp_' must be 1d (x)"""

    ## erase values below surface
    x_grid, z_grid = np.meshgrid(flux_.longitude, flux_.level)
    flux_.values[z_grid > sp_.values[None, ...]] = np.nan

    cp = ax.pcolormesh(
        x_grid, z_grid, flux_, cmap="cmo.balance", vmax=40, vmin=-40, shading="auto"
    )

    ax.set_ylim([100, 1000])
    ax.set_xlabel("Longitude")

    ## plot surface pressure
    ax.plot(sp_.longitude, sp_, c="k", ls="-", label="surface")

    ## compute boundary between levels
    p_bound = get_pressure_boundary(sp=sp_, boundary_idx=29)
    ax.plot(sp_.longitude, p_bound, c="k", ls="--", label="boundary")

    plt.gca().invert_yaxis()
    return ax, cp


def load_composite():
    """Load all data"""

    uq_rockies = xr.open_dataset(f"{src.params.DATA_RAW_FP}/uq_rockies.nc")
    uq_rockies = src.utils.get_month_range(uq_rockies, *src.params.MONTH_RANGE)

    # remove hour from daily time index
    uq_rockies["time"] = uq_rockies["time"] - pd.Timedelta("10H")

    # Compute zonal moisture flux
    uq_rockies["uq"] = uq_rockies["u"] * uq_rockies["q"]
    uq_rockies["uq"] *= 1000  # convert from (m/s * kg/kg) to (m/s * g/kg)

    ## Get times of GPLLJ events
    times_v850 = src.utils.load_and_filter(
        f"{src.params.DATA_PREPPED_FP}/times_v850.npy", month_range=src.params.MONTH_RANGE
    )

    ## Composite on GPLLJ events
    uq_rocky_composite = src.utils.get_lagged_composites(
        uq_rockies,
        times_v850,
        lags=[0],
        verbose=False,
    )
    return uq_rocky_composite


def load_surf_data():
    """Load data with surface information"""

    lon_range = (245, 260)

    ## surface pressure
    sp = xr.open_dataarray(f"{src.params.DATA_RAW_FP}/sp.nc")
    sp = sp / 100  # convert from Pa to hPa
    sp_clim = src.utils.get_month_range(sp, *src.params.MONTH_RANGE).mean("time")

    ## Geopotential data
    z_hires = xr.open_dataset(f"{src.params.DATA_CONST_FP}/Z_surf_hires.nc")["z"].isel(time=0, drop=True)
    z_lores = xr.open_dataset(f"{src.params.DATA_CONST_FP}/Z_surf_lores.nc")["z"].isel(time=0, drop=True)

    # Trim and convert to geopotential meters
    z_hires = z_hires / 9.8
    z_lores = z_lores / 9.8

    ## Hi-res topography
    topo = xr.open_dataset(f"{src.params.DATA_CONST_FP}/topo.nc")["bath"]
    topo = topo.sel(X=slice(*lon_range), Y=[27, 32, 37])

    return {"sp_clim": sp_clim, "z_lores": z_lores, "z_hires": z_hires, "topo": topo}


def plot_topo(ax, lat):
    """Plot topography at specified latitude"""

    ## Load data
    surf = load_surf_data()

    ## Make plot
    ax.plot(surf["topo"].X, surf["topo"].sel(Y=lat), label=r"$0.08^{\circ}$ res.")
    ax.plot(
        surf["z_hires"].longitude,
        surf["z_hires"].sel(latitude=lat),
        label=r"$0.25^{\circ}$ res.",
    )
    ax.plot(
        surf["z_lores"].longitude,
        surf["z_lores"].sel(latitude=lat),
        label=r"$1.0^{\circ}$ res.",
        c="k",
        ls="-",
    )

    ax.set_ylim([-10, 3700])
    ax.set_xlim([245, 260])
    ax.set_xlabel("Longitude")

    return ax


def plot_pressure(ax, surf, pres, color_levels):
    """make plot with specified pressure and color levels"""
    ax, gl = src.utils.plot_setup_ax(
        ax=ax,
        plot_range=[230, 305, 15, 60],
        xticks=[-115, -90, -65],
        yticks=[25, 40, 55],
        alpha=0.1,
    )

    ## 1000m and 2000m contours for geopotential height
    ax.contour(
        surf["z_lores"].longitude,
        surf["z_lores"].latitude,
        surf["z_lores"],
        levels=[-10000, 1000, 2000],
        extend="both",
        colors="k",
        linewidths=mpl.rcParams["contour.linewidth"] / 2,
        transform=ccrs.PlateCarree(),
    )

    ## surface pressure
    cp = ax.contourf(
        surf["sp_clim"].longitude,
        surf["sp_clim"].latitude,
        pres,
        cmap="cmo.amp_r",
        extend="both",
        levels=color_levels,
        transform=ccrs.PlateCarree(),
    )

    ## plot cross-sections
    ax.plot(lon_range, [27, 27], transform=ccrs.PlateCarree(), c="k", ls="--")
    ax.plot(lon_range, [32, 32], transform=ccrs.PlateCarree(), c="k", ls="--")
    ax.plot(lon_range, [37, 37], transform=ccrs.PlateCarree(), c="k", ls="--")
    return ax, gl, cp


if __name__ == "__main__":

    ## Set plots to specified scale
    src.params.set_plot_style()

    lon_range = (245, 260)
    ## Load data
    comp = load_composite()
    surf = load_surf_data()

    #### Plot UQ cross-section
    aspect = 2.92
    width = src.params.plot_params["twocol_width"]
    height = width/aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(ncols=3, nrows=1)
    for i, (lat, label) in enumerate(zip([27, 32, 37], ["a)", "b)", "c)"])):
        ax = fig.add_subplot(gs[i])
        ax, cp = plot_uq_cross(
            ax=ax,
            flux_=comp["uq"].sel(lag=0, latitude=lat),
            sp_=surf["sp_clim"].sel(latitude=lat, longitude=comp.longitude),
        )

        ax = src.utils.label_subplot(
            ax,
            label=f"{label} {lat}" + r"$^{\circ}$N",
            posn=(0.17, 0.9),
            transform=ax.transAxes,
        )

    ## Customize subplots
    fig.axes[0].set_ylabel("Pressure (hPa)")
    fig.axes[1].legend()
    fig.axes[1].yaxis.set_ticklabels([])
    fig.axes[2].yaxis.set_ticklabels([])

    cb = fig.colorbar(
        cp,
        ax=fig.axes[-1],
        orientation="vertical",
        pad=0.02,
    )
    cb = src.utils.mod_colorbar(
        cb,
        label=r"$\left(\frac{g}{kg}\right)\left(\frac{m}{s}\right)$",
        ticks=np.arange(-40, 60, 20),
        is_horiz=False,
    )

    src.utils.save(fig=fig, fname=f"uq_rocky_flux")
    plt.close(fig)

    #### Plot surface topography
    aspect = 2.92
    width = src.params.plot_params["twocol_width"]
    height = width/aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(ncols=3, nrows=1)
    for i, (lat, label) in enumerate(zip([27, 32, 37], ["a)", "b)", "c)"])):
        ax = fig.add_subplot(gs[i])
        ax = plot_topo(ax=ax, lat=lat)

        ax = src.utils.label_subplot(
            ax,
            label=f"{label} {lat}" + r"$^{\circ}$N",
            posn=(0.65, 0.1),
            transform=ax.transAxes,
        )

    ## Customiaxe subplots
    fig.axes[0].set_ylabel("Meters")
    fig.axes[1].yaxis.set_ticklabels([])
    fig.axes[2].yaxis.set_ticklabels([])
    fig.axes[1].legend()

    ## Save to file
    src.utils.save(fig=fig, fname="topo_cross_section")
    plt.close(fig)

    ##### Plot surface, boundary pressure
    aspect = 2.66
    width = src.params.plot_params["twocol_width"]
    height = width/aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(ncols=2, nrows=1)
    ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax1, gl1, cp1 = plot_pressure(
        ax=ax1, surf=surf, pres=surf["sp_clim"], color_levels=np.arange(700, 1030, 30)
    )
    cb1 = fig.colorbar(cp1, ax=ax1, orientation="horizontal", pad=0.02, fraction=0.05)
    cb1 = src.utils.mod_colorbar(
        cb1, label="Surface pressure (hPa)", ticks=np.arange(700, 1060, 60)
    )

    p_bound = get_pressure_boundary(sp=surf["sp_clim"], boundary_idx=29)
    ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
    ax2, gl2, cp2 = plot_pressure(
        ax2, surf=surf, pres=p_bound, color_levels=np.arange(620, 840, 20)
    )
    gl2.left_labels = False
    cb2 = fig.colorbar(cp2, ax=ax2, orientation="horizontal", pad=0.02, fraction=0.05)
    cb2 = src.utils.mod_colorbar(
        cb2, label="Boundary pressure (hPa)", ticks=np.arange(620, 860, 40)
    )

    ax1 = src.utils.label_subplot(
        ax1, label="a)", posn=(0.1, 0.1), transform=ax1.transAxes
    )
    ax2 = src.utils.label_subplot(
        ax2, label="b)", posn=(0.1, 0.1), transform=ax2.transAxes
    )

    ## Plot domain outlines in each panel
    ax1 = src.utils.add_patches_to_ax(ax1)
    ax2 = src.utils.add_patches_to_ax(ax2)

    src.utils.save(fig=fig, fname="pressure_surfaces")
    plt.close(fig)
