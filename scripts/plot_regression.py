"""
Produces the following figures:
    - "tracked-moisture-clim.png"
    - "spatial_regression_precip.png"
    - "spatial_regression_e-track.png"
"""

import xarray as xr
import numpy as np
import cmocean
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import src.utils
import src.params
import argparse


def get_masked_uv(u, v, pvals_u, pvals_v, alpha=0.05):
    """Return masked version of u and v"""
    is_sig = (pvals_u < alpha) | (pvals_v < alpha)
    return u.where(is_sig), v.where(is_sig)


def load_data(standardize):
    """Load data used for climatology and regression plots"""

    ## get season from params file
    season = src.params.season

    fluxes = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/wam_fluxes_trim_seasonal_{season}.nc"
    )
    storage = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/wam_storage_trim_seasonal_{season}.nc"
    )

    # Convert from m to mm
    fluxes *= 1000
    storage *= 1000

    # Geopotential height
    Z = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/Z_seasonal_{season}.nc")["z"]
    Z = Z.sel(level=850, longitude=slice(210, None))
    Z /= 9.8  # convert to geopotential meters

    # IVT
    ivt = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/ivt_seasonal_{season}.nc")
    ivt = ivt[["p71.162", "p72.162"]].rename({"p71.162": "u", "p72.162": "v"})

    # Helmholtz decomposition of IVT
    helmholtz = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/helmholtz_seasonal_{season}.nc"
    )

    ###### Data pre-processing #####

    #### Get climatology
    fluxes_clim = fluxes.mean("year")
    Z_clim = Z.mean("year")
    ivt_clim = ivt.mean("year")

    # Remove geopotential height values below surface
    sp = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/sp_seasonal_{season}.nc")
    Z_clim = src.utils.mask_z850(Z_clim)

    #### Load (detrended) data for regression
    gpllj_fp = f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg_seasonal_detrend_{season}.nc"
    gpllj_detrend = xr.open_dataset(gpllj_fp)["v"]

    Z_detrend = xr.open_dataarray(
        f"{src.params.DATA_PREPPED_FP}/Z_seasonal_detrend_{season}.nc"
    )
    Z_detrend = Z_detrend.sel(longitude=slice(210, None))
    Z_detrend /= 9.8  # convert from geopotential to geopotential meters

    ivt_detrend = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/ivt_seasonal_detrend_{season}.nc"
    )
    ivt_detrend = ivt_detrend[["p71.162", "p72.162"]].rename(
        {"p71.162": "u_ivt", "p72.162": "v_ivt"}
    )

    fluxes_detrend = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/wam_fluxes_trim_seasonal_detrend_{season}.nc"
    )
    fluxes_detrend *= 1000  # convert from m to cm

    helmholtz_detrend = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/helmholtz_seasonal_detrend_{season}.nc"
    )

    ## merge data
    data_clim = xr.merge([fluxes_clim, Z_clim, ivt_clim])
    data_detrend = xr.merge([fluxes_detrend, Z_detrend, ivt_detrend, helmholtz_detrend])

    ## Remove 2021 from detrended data (for regression)
    data_detrend = data_detrend.sel(year=slice(None, 2020))
    gpllj_detrend = gpllj_detrend.sel(year=slice(None, 2020))

    ## standardize data if desired
    if standardize:
        print(
            f"Std. dev. before standardizing: {gpllj_detrend.std().values.item():.2f} m/s"
        )
        gpllj_detrend /= gpllj_detrend.std()
        print(
            f"Std. dev. after standardizing:  {gpllj_detrend.std().values.item():.2f} m/s"
        )

    return data_clim, data_detrend, gpllj_detrend


def plot_etrack_clim(ax, clim):
    """Plot spatial climatology of AMJ moisture sources"""

    ax, gl = src.utils.plot_setup_ax(
        ax=ax,
        plot_range=[210, 330, 5, 60],
        xticks=[-140, -115, -90, -65, -40],
        yticks=[10, 25, 40, 55],
        alpha=0.1,
        plot_topo=True,
    )

    gl.bottom_labels = False
    gl.top_labels = True

    #### Plot tracked moisture
    cp = ax.contourf(
        clim["E_track"].longitude,
        clim["E_track"].latitude,
        clim["E_track"],
        extend="max",
        cmap="cmo.rain",
        levels=np.arange(0, 85, 5),
        transform=ccrs.PlateCarree(),
    )

    #### Plot Z_850
    cs = ax.contour(
        clim["z"].longitude,
        clim["z"].latitude,
        clim["z"],
        extend="both",
        linewidths=mpl.rcParams["lines.linewidth"] / 2,
        colors="k",
        levels=np.arange(1380, 1580, 20),
        transform=ccrs.PlateCarree(),
        alpha=0.5,
        zorder=1.01,
    )

    #### Plot fluxes
    n = 5  # plot a vector every n gridpoints
    scale = 4e3
    x, y = np.meshgrid(clim["u"].longitude[::n].values, clim["v"].latitude[::n].values)
    qv = ax.quiver(
        x,
        y,
        clim["u"].values[::n, ::n],
        clim["v"].values[::n, ::n],
        scale=scale,
        color="k",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        width=0.003,
    )

    ## Add quiver key (and background)
    ax.add_patch(
        mpatches.Rectangle(
            xy=[211, 6],
            width=24,
            height=14,
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
        label=f"{scale/20}" + r" $\frac{kg}{m \cdot s}$",
        fontproperties={"size": mpl.rcParams["legend.fontsize"]},
    )

    ## Add patches for Midwest/GPLLJ region
    ax = src.utils.add_patches_to_ax(ax)

    return ax, cp


def plot_precip_coef(ax, coefs, pvals, mask_ivt=False):
    ## Mask some coefficients for plotting
    # load masks
    masks = src.utils.get_masks(lat=coefs.latitude.values, lon=coefs.longitude.values)
    land_lake_mask = masks.sel(mask=["Land", "Lake"]).sum("mask")

    ## Load precip and mask out ocean
    P_coef = coefs["P"] * land_lake_mask

    # Geopotential height: mask out values below surface
    Z_coef = coefs["z"].sel(level=850)
    Z_coef = src.utils.mask_z850(Z_coef)

    if mask_ivt:
        # Only plot significant IVT
        u_chi_coef, v_chi_coef = get_masked_uv(
            coefs["u_chi"],
            coefs["v_chi"],
            pvals["u_chi"],
            pvals["v_chi"],
            alpha=0.05,
        )
    else:
        u_chi_coef = coefs["u_chi"]
        v_chi_coef = coefs["v_chi"]

    ## Plot setup
    ax, gl = src.utils.plot_setup_ax(
        ax=ax,
        plot_range=[230, 300, 15, 53],
        yticks=[20, 35, 50],
        xticks=[-125, -105, -85, -65],
        alpha=0.1,
        plot_topo=True,
    )

    ## Plot Midwest region and GPLLJ
    ax = src.utils.add_patches_to_ax(ax)

    ## Precip
    cp = ax.contourf(
        P_coef.longitude,
        P_coef.latitude,
        P_coef,
        levels=src.utils.make_cb_range(60, 7.5),
        extend="both",
        cmap="cmo.balance_r",
        transform=ccrs.PlateCarree(),
    )

    ## IVT potential
    c = ax.contour(
        coefs["velocity_potential"].longitude,
        coefs["velocity_potential"].latitude,
        coefs["velocity_potential"] / 1e6,
        levels=src.utils.make_cb_range(10, 2),
        transform=ccrs.PlateCarree(),
        colors="k",
        linewidths=mpl.rcParams["lines.linewidth"] * 2 / 3,
        alpha=1,
        zorder=1.01,
    )

    ## IVT
    n = 3
    scale = 90
    x, y = np.meshgrid(coefs.longitude[::n].values, coefs.latitude[::n].values)
    qv1 = ax.quiver(
        x,
        y,
        u_chi_coef.values[::n, ::n],
        v_chi_coef.values[::n, ::n],
        pivot="middle",
        scale=scale,
        color="k",
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

    ## Quiver key (and background)
    ax.add_patch(
        mpatches.Rectangle(
            xy=[230.5, 16],
            width=13,
            height=10,
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
        U=scale / 18,
        label=f"{scale/18}" + r" $\frac{kg}{m \cdot s}$",
        fontproperties={"size": mpl.rcParams["legend.fontsize"]},
    )

    return ax, cp


def plot_etrack_coef(ax, coefs, pvals, mask_ivt=False):
    # Geopotential height: mask out values below surface
    Z_coef = coefs["z"].sel(level=850)
    Z_coef = src.utils.mask_z850(Z_coef)

    if mask_ivt:
        # Only plot significant IVT
        u_ivt_coef, v_ivt_coef = get_masked_uv(
            coefs["u_ivt"],
            coefs["v_ivt"],
            pvals["u_ivt"],
            pvals["v_ivt"],
            alpha=0.05,
        )
    else:
        u_ivt_coef = coefs["u_ivt"]
        v_ivt_coef = coefs["v_ivt"]

    ## Begin plotting
    ax, gl = src.utils.plot_setup_ax(
        ax=ax,
        plot_range=[210, 330, 5, 60],
        xticks=[-140, -115, -90, -65, -40],
        yticks=[10, 25, 40, 55],
        alpha=0.1,
        plot_topo=True,
    )
    gl.left_labels = False
    gl.bottom_labels = False
    gl.top_labels = True

    ## Tracked moisture
    cp = ax.contourf(
        coefs["E_track"].longitude,
        coefs["E_track"].latitude,
        coefs["E_track"],
        levels=src.utils.make_cb_range(8, 0.8),
        extend="both",
        cmap="cmo.balance_r",
        transform=ccrs.PlateCarree(),
    )

    #### Geopotential height
    c = ax.contour(
        Z_coef.longitude,
        Z_coef.latitude,
        Z_coef,
        levels=src.utils.make_cb_range(20, 2),
        transform=ccrs.PlateCarree(),
        colors="k",
        linewidths=mpl.rcParams["lines.linewidth"] / 2,
        alpha=0.5,
    )

    ## IVT
    n = 4
    scale = 600
    x, y = np.meshgrid(coefs.longitude[::n].values, coefs.latitude[::n].values)
    qv = ax.quiver(
        x,
        y,
        u_ivt_coef.values[::n, ::n],
        v_ivt_coef.values[::n, ::n],
        pivot="middle",
        scale=scale,
        color="k",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        zorder=3,
        width=0.003,
    )

    ## Quiver key and background
    ax.add_patch(
        mpatches.Rectangle(
            xy=[211, 6],
            width=24,
            height=14,
            facecolor="white",
            edgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=3.05,
            linewidth=src.params.plot_params["border_width"],
        )
    )
    qk = ax.quiverkey(
        qv,
        X=0.1,
        Y=0.05,
        U=scale / 20,
        label=f"{scale/20}" + r" $\frac{kg}{m \cdot s}$",
        fontproperties={"size": mpl.rcParams["legend.fontsize"]},
    )

    ## Add midwest/GPLLJ regions
    ax = src.utils.add_patches_to_ax(ax)

    return ax, cp


def load_data_gpllj_precip(detrend, remove_outliers):
    """Function to load GPLLJ and precip data for scatter plot.
    detrend is a boolean specifying whether to load detrended data.
    Function returns a tuple of 1-D numpy arrays, in format (x,y)"""

    suffix = "_detrend" if detrend else ""
    suffix += f"_{src.params.season}"

    ## Load precip
    precip = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/wam_fluxes_agg_seasonal{suffix}.nc"
    )
    precip = precip["P"].sel(mask="Midwest") * 1000  # convert to mm

    ## Load GPLLJ
    gpllj = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg_seasonal{suffix}.nc"
    )["v"]
    gpllj = gpllj.sel(year=slice(None, 2020))

    ## Convert to numpy
    gpllj = gpllj.values
    precip = precip.values

    ## remove outliers if desired
    if remove_outliers:
        ## Compute deviation from mean
        gpllj_score = (gpllj - np.mean(gpllj)) / np.std(gpllj)
        precip_score = (precip - np.mean(precip)) / np.std(precip)

        ## Remove values more than 3 std. devs. from mean
        valid_idx = (np.abs(gpllj_score) <= 3) & (np.abs(precip_score) <= 3)
        gpllj = gpllj[valid_idx]
        precip = precip[valid_idx]

    return gpllj, precip


def scatter_gpllj_precip(ax, remove_outliers):
    """Scatter plot of GPLLJ vs. precip"""

    ## Function to load data
    load_data_fn = lambda detrend: load_data_gpllj_precip(detrend, remove_outliers)

    ## Create scatter plot
    ax, plot_stats = src.utils.scatter_ax(ax, load_data=load_data_fn, plot_detrend=True)

    ## Plot zero lines
    ax.axhline(0, ls="--", c="k")
    ax.axvline(0, ls="--", c="k")

    ## Add labels
    ax.set_xlabel("GPLLJ (m/s)")
    ax.set_ylabel(r"Midwest precip. ($mm$)")

    ## print correlation coef. in legend
    ax.legend(title=r"$r=$" + f"{plot_stats['rho'].item():.2f}", loc="lower right")

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--standardize", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--remove_outliers", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    src.params.set_plot_style()  # set plot style for figures

    ## Load data
    data_clim, data_detrend, gpllj_detrend = load_data(args.standardize)

    ## Compute regression coefficients
    coefs = src.utils.ls_fit_xr(Y_data=data_detrend, idx=gpllj_detrend)
    coefs = coefs.sel(coef="m")  # don't care about intercept

    rho = src.utils.rho_xr(Y_data=data_detrend, idx=gpllj_detrend)
    pvals = src.utils.get_pvals_xr(rho_vals=rho, n=42)

    ########### make E-track plot #############
    aspect = 3
    width = src.params.plot_params["max_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(ncols=2, nrows=1)

    # define projection
    proj = ccrs.PlateCarree(central_longitude=180)

    # Plot etrack clim
    ax1 = fig.add_subplot(gs[0], projection=proj)
    ax1, cp1 = plot_etrack_clim(ax1, clim=data_clim)

    # Plot etrack regression
    ax2 = fig.add_subplot(gs[1], projection=proj)
    ax2, cp2 = plot_etrack_coef(ax2, coefs=coefs, pvals=pvals)

    ## Add colorbars
    cb1 = fig.colorbar(cp1, ax=ax1, orientation="horizontal", pad=0.05, fraction=0.06)
    cb1 = src.utils.mod_colorbar(
        cb1, label=r"Contribution ($mm$)", ticks=np.arange(0, 100, 20)
    )

    cb2 = fig.colorbar(cp2, ax=ax2, orientation="horizontal", pad=0.05, fraction=0.06)
    cb2 = src.utils.mod_colorbar(
        cb2,
        label=r"Regression coef. $\left(\frac{mm}{\sigma}\right)$",
        ticks=np.arange(-8, 12, 4),
    )

    ## Add subplot labels
    ax1 = src.utils.label_subplot(
        ax1, label="a)", posn=(-142, 53), transform=ccrs.PlateCarree()
    )
    ax2 = src.utils.label_subplot(
        ax2, label="b)", posn=(-142, 53), transform=ccrs.PlateCarree()
    )

    # Save figure
    src.utils.save(fig=fig, fname="moisture_sources_seasonal_spatial", is_supp=False)
    plt.close(fig)

    ############# precip plot ##############
    aspect = 3.3
    width = src.params.plot_params["twocol_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1.66, 0.15, 1])

    # Plot precip
    ax1 = fig.add_subplot(gs[0], projection=proj)
    ax1, cp1 = plot_precip_coef(ax1, coefs=coefs, pvals=pvals)

    # Plot scatter of gpllj vs. Midwest precip
    ax2 = fig.add_subplot(gs[2])
    ax2 = scatter_gpllj_precip(ax2, remove_outliers=args.remove_outliers)

    # Add colorbar
    ticks = np.arange(-60, 90, 30)
    cb = fig.colorbar(cp1, ax=ax1, orientation="vertical", pad=0.02, fraction=0.06)
    cb = src.utils.mod_colorbar(
        cb,
        label=r"Regression coef. $\left(\frac{mm}{\sigma}\right)$",
        ticks=ticks,
        is_horiz=False,
    )

    # Add subplot labels
    ax1 = src.utils.label_subplot(
        ax1, label="a)", posn=(-125, 48), transform=ccrs.PlateCarree()
    )
    ax2 = src.utils.label_subplot(
        ax2, label="b)", posn=(0.1, 0.9), transform=ax2.transAxes
    )

    # Save figure
    src.utils.save(fig=fig, fname=f"precip_regression", is_supp=False)
    plt.close(fig)
