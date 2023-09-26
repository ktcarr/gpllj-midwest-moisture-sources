import xarray as xr
import numpy as np
from copy import deepcopy
import pandas as pd
import os
import numpy.linalg
import scipy.stats
from progressbar import progressbar
from matplotlib.path import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import src.params
import warnings


################# Constants ##################

## Set outlines for specified regions

atlantic_vertices = np.array(
    [
        [265.0, 17.0],
        [275.0, 13.0],
        [275.0, 11.0],
        [277.0, 9.0],
        [279.0, 8.0],
        [283.0, 8.0],
        [300.0, -30.0],
        [15.0, -30.0],
        [15.0, 30.0],
        [355.0, 30.0],
        [355.0, 38.0],
        [15.0, 60.0],
        [284.0, 60.0],
        [284.0, 42.0],
        [275.0, 35.0],
        [265.0, 35.0],
        [260.0, 25.0],
    ]
)

pacific_vertices = np.array(
    [
        [300.0, -30.0],
        [283.0, 8.0],
        [279.0, 8.0],
        [277.0, 9.0],
        [275.0, 11.0],
        [275.0, 13.0],
        [265.0, 17.0],
        [237.0, 60.0],
        [120.0, 60.0],
        [120.0, -30.0],
    ]
)

midwest_vertices = np.array(
    [
        [256.0, 41.0],
        [256.0, 49.0],
        [265.0, 49.0],
        [270.0, 48.0],
        [276.0, 46.0],
        [279.5, 42.0],
        [279.5, 40.5],
        [277.5, 38.5],
        [276.5, 36.5],
        [265.5, 36.5],
        [265.5, 37.0],
        [258.0, 37.0],
        [258.0, 41.0],
    ]
)

################ Functions ######################


def make_cb_range(amp, delta):
    """Make colorbar_range for cmo.balance"""
    return np.concatenate(
        [np.arange(-amp, 0, delta), np.arange(delta, amp + delta, delta)]
    )


def save(fig, fname, is_supp=True):
    """Convenience function for saving to results folder"""

    ## get filepath for saving
    if is_supp:
        save_fp = f"{src.params.SAVE_FP}/supp"
    else:
        save_fp = f"{src.params.SAVE_FP}"

    ## Save the file.
    # Warning filter gets rid of shapely/cartopy warning
    # about intersections.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(f"{save_fp}/{fname}.png")
    return


def get_gridsize(lat, lon, dlat, dlon):
    """Grid the size of each gridcell"""

    ## Constants
    dlat_rad = dlat / 180.0 * np.pi
    dlon_rad = dlon / 180.0 * np.pi
    R = 6.378e6  # earth radius (meters)

    ## height of gridcell doesn't depend on longitude
    dy = R * dlat_rad  # unit: meters
    dy *= np.ones([len(lat), len(lon)])

    ## Compute width of gridcell
    lat_rad = lat / 180 * np.pi  # latitude in radians
    dx = R * np.cos(lat_rad) * dlon_rad
    dx = dx[:, None] * np.ones([len(lat), len(lon)])

    ## Compute area
    A = dx * dy

    ## Put in dataset
    coords = {"latitude": lat, "longitude": lon}
    dims = ("latitude", "longitude")
    grid_dims = xr.Dataset(
        {"A": (dims, A), "dx": (dims, dx), "dy": (dims, dy)}, coords=coords
    )

    return grid_dims


def get_month_range(data, m1, m2):
    """Get subset of data bounded by given months (closed interval)."""
    months = data.time.dt.month
    criteria = (months >= m1) & (months <= m2)
    return data.sel(time=criteria)


def filter_times(times, month_range):
    """only get times in specified range"""
    months = pd.DatetimeIndex(times).month
    in_range = (months >= month_range[0]) & (months <= month_range[1])
    return times[in_range]


def load_and_filter(fp, month_range):
    """wrapper function to load and filter time data"""
    return filter_times(np.load(fp), month_range)


def trim_and_fix_dates(data, month_range):
    """get specified month range, and remove hour from 'time'
    variable (makes merging with other datasets possible
    """
    data_trimmed = get_month_range(data, *month_range)
    new_dates = pd.DatetimeIndex(data_trimmed.time.dt.date)
    return data_trimmed.assign_coords({"time": new_dates})


def remove_nan_da(data):
    """remove NaN values from dataarray (replace with zeros)"""
    isnan_idx = np.isnan(data.values)
    data.values[isnan_idx] = 0.0
    return data


def remove_nan(data):
    """remove NaN values from dataarray or dataset"""
    return xr_wrapper(remove_nan_da, data)


def switch_lon_range(lon, neg_to_pos=True):
    """
    Convert longitudes from: [-180,180) to [0,360) and vice versa.
    If neg_to_pos:
        [-180,180) => [0,360)
    else:
        [0,360) => [-180,180)
    """
    lon_ = deepcopy(lon)
    if neg_to_pos:
        lon_[lon_ < 0] = lon_[lon_ < 0] + 360
    else:
        lon_[lon_ > 180] = lon_[lon_ > 180] - 360
    return lon_


def makeMask(vertices, lat, lon, name=None, crosses_pm=False):
    """Function returns a mask with with 1s representing the area inside of vertices
    'vertices' is an Nx2 array, representing boundaries of a region.
    'lat' and 'lon' are length-N arrays
    'name' is a string denoting the name of the mask (for xarray)
    'cross_pm' is a boolean indicating whether vertices cross the prime meridian"""

    lon_ = deepcopy(lon)
    vertices_ = deepcopy(vertices)

    ### Handle case where region outline crosses prime meridian
    if crosses_pm:
        lon_ = switch_lon_range(lon, neg_to_pos=False)
        vertices_[:, :1] = switch_lon_range(vertices[:, :1], neg_to_pos=False)

    # create 2-D grid from lat/lon coords
    lon_lat_grid = np.meshgrid(lon_, lat)

    # next, get pairs of lon/lat
    t = zip(lon_lat_grid[0].flatten(), lon_lat_grid[1].flatten())

    # convert back to array
    t = np.array(list(t))  # convert to array

    # convert vertices into a matplotlib Path
    path = Path(vertices_)
    mask = path.contains_points(t).reshape(len(lat), len(lon))  # create mask
    mask = mask.astype(float)

    # Convert to xr.DataArray
    coords = {"latitude": lat, "longitude": lon, "mask": [f"{name}"]}
    dims = ["mask", "latitude", "longitude"]
    return xr.DataArray(mask[None, ...], coords=coords, dims=dims)


def load_lsm():
    """load LSM from filepath.
    Note: NO conversion to bool!"""
    lsm = xr.open_dataarray(f"{src.params.DATA_CONST_FP}/lsm.nc")
    lsm = lsm.isel(time=0, drop=True)
    return lsm


def load_lake_cover():
    """load lake_cover from filepath.
    Note: NO conversion to bool!"""
    lake_cover = xr.open_dataarray(f"{src.params.DATA_CONST_FP}/lake_cover.nc")
    lake_cover = lake_cover.isel(time=0, drop=True)
    return lake_cover


def get_masks(lat, lon):
    """Get all masks, from specified lon/lat"""

    ## First, load land-sea mask and lake_cover
    land_mask = load_lsm()
    lake_mask = load_lake_cover()

    ## Select specified lat/lons
    land_mask = land_mask.sel(latitude=lat, longitude=lon)
    lake_mask = lake_mask.sel(latitude=lat, longitude=lon)

    ## combine into 'land and lake' mask
    land_lake_mask = land_mask + lake_mask

    ## 'Total mask' (ones everywhere!)
    total_mask = xr.ones_like(land_lake_mask)

    ## Ocean is everything else
    ocean_mask = total_mask - land_lake_mask

    ## Atlantic
    atlantic_mask = makeMask(
        atlantic_vertices, lat, lon, crosses_pm=True, name="Atlantic"
    )
    atlantic_mask *= ocean_mask

    ## Pacific
    pacific_mask = makeMask(
        pacific_vertices, lat, lon, crosses_pm=False, name="Pacific"
    )
    pacific_mask *= ocean_mask * (1 - atlantic_mask.squeeze())

    ## Midwest (note conversion to bool)
    midwest_mask = makeMask(
        midwest_vertices, lat, lon, crosses_pm=False, name="Midwest"
    )
    midwest_mask *= land_mask.astype("bool")

    # concatenate them all
    masks = xr.concat(
        [
            atlantic_mask,
            pacific_mask,
            midwest_mask,
            lake_mask.expand_dims({"mask": ["Lake"]}),
            ocean_mask.expand_dims({"mask": ["Ocean"]}),
            land_mask.expand_dims({"mask": ["Land"]}),
            total_mask.expand_dims({"mask": ["Total"]}),
        ],
        dim="mask",
    )

    return masks


def mask_z850(z):
    """Mask out Z850 below surface pressure climatology"""

    ## Load surface pressure climatology
    season = src.params.season
    sp = xr.open_dataarray(f"{src.params.DATA_PREPPED_FP}/sp_seasonal_{season}.nc")

    ## Compute climatology
    sp_clim = sp.mean("year")

    return z.where(sp_clim > 85000.0)


#################### Regression functions ##############


def detrend_dim(data, dim="time", deg=1):
    """
    Detrend along a single dimension.
    'data' is an xr.Dataset or xr.DataArray
    """
    ## First, check if data is a dataarray. If so,
    ## convert to dataset
    is_dataarray = type(data) is xr.DataArray
    if is_dataarray:
        data = xr.Dataset({data.name: data})

    ## Compute the linear best fit
    p = data.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(data[dim], p)

    ## Subtract the linear best fit for each variable
    data_detrended = np.nan * xr.ones_like(data)
    varnames = list(data)
    for varname in varnames:
        ## Get best fit for given variable, and make sure dtype matches original
        fit_ = fit[f"{varname}_polyfit_coefficients"].astype(data[varname].dtype)

        ## Subtract from original data
        data_detrended[varname] = data[varname] - fit_

    if is_dataarray:
        varname = list(data_detrended)[0]
        data_detrended = data_detrended[varname]

    return data_detrended


def get_rho(x, Y):
    """Get correlation between many timeseries (Y) and single timeseries (x)"""
    Y = Y - np.nanmean(Y, axis=0)
    x = x - np.nanmean(x)
    Y[np.isnan(Y)] = 0.0
    x[np.isnan(x)] = 0.0
    rho = (x.T @ Y) / np.sqrt(np.sum(Y**2, axis=0) * np.dot(x, x))
    return rho


def ls_fit(x, Y):
    """Function solves least-squares to get coefficients for linear projection of x onto Y
    returns coef, a vector of the best-fit coefficients"""
    # Add array of ones to feature vector (to represent constant term in linear model)
    x = add_constant(x)
    coef = numpy.linalg.inv(x.T @ x) @ x.T @ Y
    return coef


def compute_stats(x, y):
    """convenience function computes regression coefficient
    and correlation between two variables.
    Also gets points associated with
    best fit line for x and y. Expects two, 1-D arrays
    as inputs. Returns dictionary with regression
    and correlation coefficients, and
    x and y points for plotting the best fit line"""

    ## Compute the stats
    coef = ls_fit(x[:, None], y[:, None])
    # rho = get_rho(x, y[:,None])
    rho, p = scipy.stats.pearsonr(x, y)

    ## Get bestfit line
    x_ = np.linspace(x.min(), x.max(), 50)
    x_ = add_constant(x_[:, None])
    y_ = x_ @ coef

    return {
        "coef": coef,
        "rho": rho,
        "pval": p,
        "x_": x_[:, 0],
        "y_": y_,
        "x": x,
        "y": y,
    }


def scatter_ax(ax, load_data, plot_detrend=False):
    """Convenience function to add scatter plot to specified ax.
    'load_data' is a function which takes in a single boolean
    argument ('detrend') and returns two 1-D numpy arrays.
    'plot_detrend' is a boolean specifying whether to plot the
    detrended data"""

    ## Compute the stats
    stats = {
        "detrend": compute_stats(*load_data(detrend=True)),
        "nodetrend": compute_stats(*load_data(detrend=False)),
    }

    ## print out correlation before/after detrending
    print("\n")
    print(f"corr. before detrend: {stats['nodetrend']['rho'].item():.3f}")
    print(f"pval. before detrend: {stats['nodetrend']['pval'].item():.2e}")
    print(f"corr. after detrend:  {stats['detrend']['rho'].item():.3f}")
    print(f"pval. after detrend:  {stats['detrend']['pval'].item():.2e}")
    print("\n")

    # Plotting
    plot_stats = stats["detrend" if plot_detrend else "nodetrend"]
    ax.scatter(plot_stats["x"], plot_stats["y"])
    ax.plot(plot_stats["x_"], plot_stats["y_"].squeeze(), c="k")

    return ax, plot_stats


def get_pvals(r, n=40):
    """get p-values given r-values and n"""
    t = r * np.sqrt(n - 2) / (1 - r**2)  # t-statistic
    pval = scipy.stats.t.sf(np.abs(t), n - 2) * 2
    return pval


def add_constant(x):
    """Add a column of ones to matrix, representing constant coefficient for mat. mult."""
    x = np.append(x, np.ones([x.shape[0], 1]), axis=1)
    return x


def get_mc_bounds(data, idx, compute_coef, seed, nsims=1000):
    """
    Compute monte-carlo upper and lower bounds. Args:
        - data: xr.DataArray
        - idx: xr.DataArray (1-D array)
        - compute_coef: function which takes in (data, idx)
        - nsims: integer; number of simulations to run
    """

    ## empty list to hold results
    coef_rand = []

    ## Create random number generator
    rng = np.random.default_rng(seed=seed)

    ## Perform simulations
    for n in np.arange(nsims):
        # shufle data
        data_rand = shuffle_xr(data, rng=rng)
        idx_rand = shuffle_xr(idx, rng=rng)

        # compute coefficients
        coef_rand.append(compute_coef(data_rand, idx_rand))

    # concatenate results
    coef_rand = xr.concat(coef_rand, dim=pd.Index(np.arange(nsims), name="sim"))

    ub = coef_rand.quantile(q=0.975, dim="sim")
    lb = coef_rand.quantile(q=0.025, dim="sim")

    mc_bounds = xr.concat([ub, lb], dim=pd.Index(["upper", "lower"], name="bound"))
    return mc_bounds


def label_subplot(ax, label, posn, transform=None):
    """Add label to subplot in custom style"""
    posnx, posny = posn
    if transform is None:
        ax.text(
            posnx,
            posny,
            label,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="w",
                edgecolor="k",
                linewidth=src.params.plot_params["border_width"],
            ),
        )
    else:
        ax.text(
            posnx,
            posny,
            label,
            ha="center",
            va="center",
            transform=transform,
            bbox=dict(
                facecolor="w",
                edgecolor="k",
                linewidth=src.params.plot_params["border_width"],
            ),
        )
    return ax


def mod_colorbar(cb, label, ticks, is_horiz=True):
    """Modify colorbar with specified, label and ticks"""

    cb.set_label(label=label, fontsize=mpl.rcParams["legend.fontsize"])
    cb.ax.tick_params(
        labelsize=mpl.rcParams["legend.fontsize"],
        width=src.params.plot_params["tick_width"],
        length=src.params.plot_params["tick_length"],
    )

    if is_horiz:
        cb.ax.set_xticks(ticks)
        cb.ax.set_xticklabels(ticks)
    else:
        cb.ax.set_yticks(ticks)
        cb.ax.set_yticklabels(ticks)

    return cb


def plot_setup_monthly_ax(ax, yticks, ylim, zero_line_width=0):
    """Modify axis for plotting monthly data"""

    if zero_line_width > 0:
        ax = plot_zero_line(ax, scale=zero_line_width)
    xticks = np.arange(1, 13, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylim(ylim)
    ax.set_xlim([0.5, 11.5])

    return ax


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


def add_data_with_sig(ax, x, bounds, color, label):
    """
    Add monthly data to plot. Args:
        - 'ax' is the plot object to which the data is added
        - 'x' is the data to plot (xr.DataArray)
        - 'bounds' are the significance bounds (xr.DataArray)
        - 'color' is the color of the plot
    """

    ## plot center curve
    ax.plot(x.month, x, label=label, c=color)

    ## plot bounds
    lw = 1.0 * mpl.rcParams["lines.linewidth"]
    alpha = 1.0
    ls = ":"
    ax.plot(x.month, bounds.sel(bound="upper"), alpha=alpha, c=color, lw=lw, ls=ls)
    ax.plot(x.month, bounds.sel(bound="lower"), alpha=alpha, c=color, lw=lw, ls=ls)

    ## shade area where curve is outside of the bounds
    x, y1, y2 = get_fill_between(
        x,
        idx=x.month,
        lb=bounds.sel(bound="lower"),
        ub=bounds.sel(bound="upper"),
    )
    ax.fill_between(x, y1, y2, alpha=0.1, color=color)

    return ax


def shuffle_xr_da(data, rng, dim="time"):
    """shuffle a DataArray along specified dimension"""
    data_permuted = deepcopy(data).transpose(dim, ...)
    data_permuted.values = rng.permuted(data_permuted.values, axis=0)
    return data_permuted


def shuffle_xr(data, rng):
    """Shuffle dataset or dataarray in time"""
    return xr_wrapper(shuffle_xr_da, data, rng=rng)


def xr_wrapper(xr_da_func, ds, **kwargs):
    """wrapper function which applies dataarray function to dataset"""
    if type(ds) is xr.DataArray:
        return xr_da_func(ds, **kwargs)

    else:
        results = []
        varnames = list(ds)
        for varname in varnames:
            result = xr_da_func(ds[varname], **kwargs)
            results.append(result)

        return xr.merge(results)


def ls_fit_xr_da(Y_data, **kwargs):
    """xarray wrapper for ls_fit (dataarray version).
    'E_data' and 'y_data' are both dataarrays"""

    # Parse inputs
    idx = kwargs["idx"]

    ## Identify regression dimension
    regr_dim = idx.dims[0]
    other_dims = [d for d in Y_data.dims if d != regr_dim]

    ## Stack the non-regression dimensions, if they exist
    if len(other_dims) > 0:
        Y_stack = Y_data.stack(other_dims=other_dims)
    else:
        Y_stack = Y_data

    ## Empty array to hold results
    coefs = Y_stack.isel({regr_dim: slice(None, 2)})
    coefs = coefs.rename({regr_dim: "coef"})
    coefs["coef"] = pd.Index(["m", "b"], name="coef")

    ## perform regression
    coefs.values = ls_fit(idx.values[:, None], Y_stack.values)
    return coefs.unstack()


def rho_xr_da(Y_data, **kwargs):
    """xarray wrapper for rho (dataarray version).
    'idx' and 'y_data' are both dataarrays"""

    # Parse inputs
    idx = kwargs["idx"]

    ## Stack the non-regression dimensions
    regr_dim = idx.dims[0]
    other_dims = [d for d in Y_data.dims if d != regr_dim]
    Y_stack = Y_data.stack(other_dims=other_dims)

    ## Empty array to hold results
    coef = Y_stack.isel({regr_dim: 0}, drop=True)

    ## compute correlation
    coef.values = get_rho(idx.values, Y_stack.values)

    return coef.unstack()


def get_pvals_xr_da(rho_vals, **kwargs):
    """xarray wrapper for get_pvals (dataarray version).
    'n' is an integer (numer of samples) and 'rho_vals' is a dataarray"""

    ## Parse inputs
    n = kwargs["n"]

    ## Empty array to hold results
    pvals = np.nan * xr.ones_like(rho_vals)

    ## compute correlation
    pvals.values = get_pvals(r=rho_vals, n=n)

    return pvals


def get_hatch_xr_da(pvals, **kwargs):
    """xarray wrapper for get_hatch"""
    alpha = kwargs["alpha"]

    hatch = xr.ones_like(pvals)
    hatch.values[pvals.values > alpha] = np.nan

    return hatch


def ls_fit_xr(Y_data, idx):
    """applies least-squares fit to index and xr.Dataset/xr.DataArray"""
    return xr_wrapper(ls_fit_xr_da, ds=Y_data, idx=idx)


def rho_xr(Y_data, idx):
    """applies correlation coefficient to index and xr.Dataset/xr.DataArray"""
    return xr_wrapper(rho_xr_da, ds=Y_data, idx=idx)


def get_pvals_xr(rho_vals, n):
    return xr_wrapper(get_pvals_xr_da, ds=rho_vals, n=n)


def get_hatch_xr(pvals, alpha):
    return xr_wrapper(get_hatch_xr_da, ds=pvals, alpha=alpha)


########### Function to get composite ###########
def get_lagged_composites(X, times, lags, double_count=False, verbose=False):
    """Get lagged composite on specified time"""
    composites = []
    for l in progressbar(lags):
        ## Get times for specified composite
        time_sel = times + pd.Timedelta(f"{l}D")

        ## Optionally avoid double-counting days
        if (not double_count) & (l != 0):
            time_sel = [
                t for t in time_sel if (t not in times)
            ]  # exclude days that already have jet

        ## Optionally print out number of times included in composite
        if verbose:
            print(f"{len(time_sel)}/{len(times)}")

        composites.append(X.sel(time=time_sel).mean("time"))

    ## Merge different lags
    composites = xr.concat(composites, dim=pd.Index(lags, name="lag"))

    return composites


######### Plotting #########


def plot_zero_line(ax, scale=1 / 3):
    """plot line to mark zero on the y-axis"""

    ax.axhline(0, ls="dashdot", c="k", lw=mpl.rcParams["lines.linewidth"] * scale)

    return ax


def add_patches_to_ax(
    ax, ls="--", lw_scale=1 / 2, color="magenta", use_path_effects=True
):
    """add midwest vertices and GPLLJ region to plot"""

    ## add shaded 'background' to lines if desired
    if use_path_effects:
        path_effects = [
            pe.Stroke(
                linewidth=mpl.rcParams["lines.linewidth"] * 2 * lw_scale,
                foreground="k",
            ),
            pe.Normal(),
        ]

    else:
        path_effects = [pe.Normal()]

    ## Midwest region
    ax.add_patch(
        mpatches.Polygon(
            xy=src.utils.midwest_vertices,
            closed=True,
            edgecolor=color,
            ls="--",
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=1.005,
            lw=mpl.rcParams["lines.linewidth"] * lw_scale,
            alpha=1,
            path_effects=path_effects,
        )
    )

    ## GPLLJ region
    ax.add_patch(
        mpatches.Rectangle(
            xy=[258, 25],
            width=5,
            height=10,
            facecolor="none",
            edgecolor=color,
            ls="--",
            transform=ccrs.PlateCarree(),
            zorder=1.005,
            lw=mpl.rcParams["lines.linewidth"] * lw_scale,
            alpha=1,
            path_effects=path_effects,
        )
    )

    return ax


def plot_setup_ax(ax, plot_range, xticks, yticks, alpha, plot_topo=False):
    """Create map background for plotting spatial data"""

    ax.set_extent(plot_range, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=mpl.rcParams["contour.linewidth"])

    gl = ax.gridlines(
        draw_labels=True,
        linestyle="--",
        alpha=alpha,
        linewidth=src.params.plot_params["gridline_width"],
        color="k",
        zorder=1.05,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": mpl.rcParams["xtick.labelsize"]}
    gl.ylabel_style = {"size": mpl.rcParams["ytick.labelsize"]}
    gl.ylocator = mticker.FixedLocator(yticks)
    gl.xlocator = mticker.FixedLocator(xticks)

    ## modify distance between labels and ticks
    gl.xpadding = src.params.plot_params["lonlat_label_pad"]
    gl.ypadding = src.params.plot_params["lonlat_label_pad"]

    if plot_topo:
        # load topography map
        topo = xr.open_dataset(f"{src.params.DATA_CONST_FP}/topo.nc")
        topo = topo["bath"].sel(X=slice(230, 300), Y=slice(15, 52))

        ## Plot 1000m contour of topography
        ax.contour(
            topo.X,
            topo.Y,
            topo,
            extend="both",
            colors="k",
            levels=[-10000, 1000],
            transform=ccrs.PlateCarree(),
            linewidths=mpl.rcParams["contour.linewidth"] / 2,
        )

    return ax, gl


def plot_setup(figsize, plot_range, xticks, yticks, alpha):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent(plot_range, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=src.params.plot_params["border_width"])

    gl = ax.gridlines(
        draw_labels=True,
        linestyle="--",
        alpha=alpha,
        linewidth=src.params.plot_params["gridline_width"],
        color="k",
        zorder=1.05,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": mpl.rcParams["xtick.labelsize"]}
    gl.ylabel_style = {"size": mpl.rcParams["ytick.labelsize"]}
    gl.ylocator = mticker.FixedLocator(yticks)
    gl.xlocator = mticker.FixedLocator(xticks)
    return fig, ax


def get_quiver_args(u, v, n=1, mask_zeros=True):
    """
    get arguments for quiver plot. Args:
        - 'u' and 'v' are xarray dataarray inputs.
          They must have latitude and longitude coordinates.
        - 'n' is the frequency to plot.
        - 'mask_zeros' is a boolean
    """

    ## Get grid for plotting
    x, y = np.meshgrid(u.longitude[::n].values, u.latitude[::n].values)

    ## Get downsampled values
    u_ = u.values[::n, ::n]
    v_ = v.values[::n, ::n]

    ## If desired, mask out zero values
    mask = (u_ == 0) & (v_ == 0)

    return x[~mask], y[~mask], u_[~mask], v_[~mask]


def get_fill_between(a, lb, ub, idx=np.arange(1, 13)):
    """Given index and two series (lower bound and upper bound),
    return indices and subset of each series where lambda(a,b) is true"""
    new_idx = np.linspace(idx[0], idx[-1], 500)
    a_ = np.interp(new_idx, idx, a)
    lo_ = np.interp(new_idx, idx, lb)
    hi_ = np.interp(new_idx, idx, ub)
    is_hi = a_ > hi_
    is_lo = a_ < lo_

    upper = deepcopy(a_)
    lower = deepcopy(a_)

    upper[is_hi] = a_[is_hi]
    lower[is_hi] = hi_[is_hi]

    upper[is_lo] = lo_[is_lo]
    lower[is_lo] = a_[is_lo]

    return new_idx, upper, lower
