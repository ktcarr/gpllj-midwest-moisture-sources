import matplotlib.pyplot as plt
import numpy as np


def get_iso(data, x):
    """Given a dataset and the value of an isoline, get coordinates of isoline"""

    levels = np.array([x])

    fig, ax = plt.subplots()
    p = ax.contour(data["longitude"], data["latitude"], data, levels=levels, colors="k")
    plt.close(fig)
    coords = p.get_paths()[0].vertices

    return coords


def get_ridge_lonlat(data, x):
    """Get lon/lat coords of western edge of NASH"""

    ## Get coordinates of the ridge
    ridge_coords = get_iso(data, x)
    longitude = ridge_coords[:, 0]

    ## Get coordinate with minimum longitude
    coords = ridge_coords[np.argmin(longitude), :]

    return coords


def pad_with_nan(x, new_length):
    """function to pad an array with NaNs to desired length.
    Hard-coded to pad 2-d array along axis 0"""
    if len(x) > new_length:
        print("Error: array is too long")
        return
    else:
        pad_length = new_length - x.shape[0]
        npad = ((0, pad_length), (0, 0))
        return np.pad(x, npad, "constant", constant_values=np.nan)


def get_iso_and_pad(data, x, new_length=300):
    """Helper function to get isoline and pad with NaN values"""
    ridge_coords = get_iso(data, x)
    return pad_with_nan(ridge_coords, new_length=new_length)


def get_ridge_posn(Z, x):
    """Get position of NASH. 'Z' is geopotential data, and 'x' is
    scalar value to define contour outline of the NASH. x=1540 is
    a good choice for AMJ"""

    # Get ridge position and NASH coordinates for each year
    ridge_posn = []
    nash_coords = []
    for y in Z.year:
        ridge_posn.append(get_ridge_lonlat(Z.sel(year=y), x=x))
        nash_coords.append(get_iso_and_pad(Z.sel(year=y), x=x, new_length=300))

    # convert to DataArray
    ridge_posn = xr.DataArray(
        np.stack(ridge_posn, axis=0),
        coords=({"year": Z.year, "posn": ["x", "y"]}),
        dims=["year", "posn"],
    )
    nash_coords = xr.DataArray(
        np.stack(nash_coords, axis=0),
        coords={"year": Z.year, "idx": np.arange(300), "posn": ["x", "y"]},
        dims=["year", "idx", "posn"],
    )

    return ridge_posn, nash_coords


def load_nash_coords():
    """wrapper function which loads geopotential data and computes
    NASH coords"""

    ## Load geopotential height
    season = src.params.season
    Z = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/Z_seasonal_{season}.nc")["z"]
    Z = Z.sel(level=850, longitude=slice(250, None), year=slice(None, 2020))
    Z /= 9.8  # convert to geopotential meters

    ## Compute position of westernmost NASH extent
    # first, get season (mean state depends on season)
    is_amj = src.params.MONTH_RANGE[0] == 4
    is_mjj = src.params.MONTH_RANGE[0] == 5
    is_jja = src.params.MONTH_RANGE[0] == 6

    # define contour value to look for
    if is_amj:
        x = 1540
    elif is_mjj:
        x = 1550
    elif is_jja:
        x = 1560
    else:
        print("Warning: not a valid season!")
        x = 1540

    ridge_posn, nash_coords = get_ridge_posn(Z, x=x)

    return ridge_posn, nash_coords


def plot_nash_scatter(ax):
    """add scatter plot of NASH western longitude to plot"""

    ## Load data
    ridge_posn, nash_coords = load_nash_coords()

    ## Specify years to plot in bold
    years_to_plot = np.arange(1980, 2020, 15)

    ## Set up the plot
    ax, gl = src.utils.plot_setup_ax(
        ax=ax,
        plot_range=[256, 296, 17, 36],
        xticks=[-100, -90, -80, -70],
        yticks=[20, 25, 30, 35],
        alpha=0.1,
        plot_topo=True,
    )

    ## Add GPLLJ region outline for context
    ax.add_patch(
        mpatches.Rectangle(
            xy=[258, 25],
            width=5,
            height=10,
            facecolor="none",
            edgecolor="k",
            lw=mpl.rcParams["lines.linewidth"] / 1.5,
            transform=ccrs.PlateCarree(),
        )
    )

    ## Scatter ridge posn for all years
    ax.scatter(
        ridge_posn.sel(posn="x"),
        ridge_posn.sel(posn="y"),
        transform=ccrs.PlateCarree(),
        color="r",
        s=2,
        alpha=1,
    )

    ## Plot bold examples
    colors = sns.color_palette("mako")
    for i, y in enumerate(years_to_plot):
        ax.plot(
            nash_coords.sel(year=y, posn="x").values.T,
            nash_coords.sel(year=y, posn="y").values.T,
            transform=ccrs.PlateCarree(),
            # c="k",
            color=colors[2 * i],
            label=y,
        )
        ax.scatter(
            ridge_posn.sel(year=y, posn="x"),
            ridge_posn.sel(year=y, posn="y"),
            transform=ccrs.PlateCarree(),
            # c="k"
            color=colors[2 * i],
            s=12,
        )

    ax.legend(loc="upper right", framealpha=1)

    return ax


def load_nash_idx(detrend):
    """Convenience function to load NASH index.
    'detrend' is a boolean specifying whether to detrend"""

    ridge_posn, _ = load_nash_coords()
    x = ridge_posn.sel(posn="x", year=slice(None, 2020)).rename("ridge_posn")
    x -= 360  # Convert longitude to degrees west

    if detrend:
        x = src.utils.detrend_dim(x, dim="year")

    return x


def load_gpllj_ridge_posn(detrend):
    """Function to load data for scatter plot.
    'detrend is a boolean specifying whether to load
    detrended data. Returns tuple of 1-D numpy arrays,
    in the form (x,y)"""

    # Load NASH index
    x = load_nash_idx(detrend=detrend)

    # Load GPLLJ data
    suffix = "_detrend" if detrend else ""
    suffix += f"_{src.params.season}"
    y = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/uvq_gpllj_avg_seasonal{suffix}.nc"
    )
    y = y["v"].sel(year=slice(None, 2020))

    return x.values, y.values


def load_wam_ridge_posn(detrend, mask):
    """Function to load data for scatter plot.
    'detrend is a boolean specifying whether to load
    detrended data. Returns tuple of 1-D numpy arrays,
    in the form (x,y)"""

    ## Load NASH index
    x = load_nash_idx(detrend=detrend)

    ## Load WAM_data
    suffix = "_detrend" if detrend else ""
    suffix += f"_{src.params.season}"
    y = xr.open_dataset(
        f"{src.params.DATA_PREPPED_FP}/wam_fluxes_agg_seasonal{suffix}.nc"
    )
    y = y["E_track"].sel(mask=mask) * 1000  # convert to mm

    return x.values, y.values


def plot_nash_gpllj_scatter(ax):
    """scatter plot of GPLLJ strength vs. NASH position"""

    ax, plot_stats = src.utils.scatter_ax(ax, load_gpllj_ridge_posn, plot_detrend=False)

    # Plot
    ax.set_xlabel(r"Westernmost NASH extent")
    ax.set_ylabel(r"GPLLJ ($m/s$)")

    ## Set ticks
    ticks = [-90, -80, -70]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{-t}" + r"$^{\circ}$W" for t in ticks])

    # ## label with correlation coefficient
    # ax.legend(title=r"$r=$" + f"{plot_stats['rho'].item():.2f}", loc="upper right")

    return ax


def plot_nash_ocean_scatter(ax, mask):
    """scatter plot of NASH position vs. oceanic moisture contribution"""

    ## wrapper function to load data from specified region
    load_fn = lambda detrend: load_wam_ridge_posn(detrend, mask)

    ax, plot_stats = src.utils.scatter_ax(ax, load_fn, plot_detrend=False)

    # Plot
    ax.set_xlabel(r"Westernmost NASH extent")

    ## Print corr/coef in legend
    # ax.legend(
    #     title=f"source = {mask}\n" + r"$corr.=$"+f"{plot_stats['rho'].item():.2f}",
    #     loc="lower left"
    # )

    ## Set ticks
    ticks = [-90, -80, -70]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{-t}" + r"$^{\circ}$W" for t in ticks])

    return ax


if __name__ == "__main__":
    import xarray as xr
    import src.params
    import src.utils
    import cartopy.crs as ccrs
    import matplotlib.patches as mpatches
    import matplotlib as mpl
    import seaborn as sns
    from scipy.stats import pearsonr

    src.params.set_plot_style()

    # Create figure
    aspect = 3.1
    width = src.params.plot_params["twocol_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")

    gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[2, 1])

    # Plot NASH western extent
    ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude=180))
    ax1 = plot_nash_scatter(ax1)

    # Scatter NASH vs. GPLLJ
    ax2 = fig.add_subplot(gs[1], aspect=7.5)
    print("NASH vs. GPLLJ")
    ax2 = plot_nash_gpllj_scatter(ax2)

    # Add labels to subplots
    ax1 = src.utils.label_subplot(
        ax1, label="a)", posn=(-102, 19), transform=ccrs.PlateCarree()
    )
    ax2 = src.utils.label_subplot(
        ax2, label="b)", posn=(0.1, 0.1), transform=ax2.transAxes
    )

    # Save figure
    src.utils.save(fig=fig, fname=f"nash_v_gpllj")
    plt.close(fig)

    ## nash vs. ocean contribution
    aspect = 3.1
    width = src.params.plot_params["twocol_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")

    ax1 = fig.add_subplot(1, 3, 1)
    print("NASH vs. land contribution")
    ax1 = plot_nash_ocean_scatter(ax1, mask="Land")
    ax1.set_ylabel(r"Moisture contribution ($mm$)")
    ax1 = src.utils.label_subplot(ax1, label="a)", posn=(-93, 178))

    ax2 = fig.add_subplot(1, 3, 2)
    print("NASH vs. Atlantic contribution")
    ax2 = plot_nash_ocean_scatter(ax2, mask="Atlantic")
    ax2 = src.utils.label_subplot(ax2, label="b)", posn=(-93, 122))

    ax3 = fig.add_subplot(1, 3, 3)
    print("NASH vs. Pacific contribution")
    ax3 = plot_nash_ocean_scatter(ax3, mask="Pacific")
    ax1 = src.utils.label_subplot(ax3, label="c)", posn=(-93, 126))

    src.utils.save(fig=fig, fname="nash_v_moisture_source")
    plt.close(fig)
