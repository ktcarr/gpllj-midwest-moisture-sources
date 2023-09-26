from os import environ

#### User-set parameters
## choose season (one of "amj", "mjj", "jja")
season = "amj"

# months for seasonal averages
if season == "amj":
    MONTH_RANGE = (4, 6)

elif season == "mjj":
    MONTH_RANGE = (5, 7)

elif season == "jja":
    MONTH_RANGE = (6, 8)

else:
    print(f"{season} is not a valid season")

## Set filepaths
DATA_RAW_FP = f"{environ['DATA_FP']}/raw"
DATA_CONST_FP = f"{environ['DATA_FP']}/constants"
DATA_PREPPED_FP = f"{environ['DATA_FP']}/prepped"
SAVE_FP = f"{environ['SAVE_FP']}/{season}"
WAM_FP = f"{environ['WAM_FP']}"

## Plot resolution and scaling
DPI = 1200  # set to 1200 for publication-quality
PLOT_SCALE = 0.85  # plot scaling

## Plotting parameters
plot_params = {
    "gridline_width": 0.8 * PLOT_SCALE,
    "border_width": 0.35 * PLOT_SCALE,
    "tick_width": 0.0 * PLOT_SCALE,
    "tick_length": 2.3 * PLOT_SCALE,
    "lonlat_label_pad": 2.5 * PLOT_SCALE,
    "twothirds_width": 4.5,
    "twocol_width": 5.5,
    "onecol_width": 3.2,
    "max_width": 6.4,
}


def set_plot_style():
    """Set plot style."""

    import matplotlib as mpl
    import seaborn as sns

    sns.set()
    sns.set_palette("colorblind")

    ## Manually change some parameters
    mpl.rcParams["figure.dpi"] = DPI  # set figure resolution (dots per inch)
    mpl.rcParams["hatch.linewidth"] = 0.15 * PLOT_SCALE
    mpl.rcParams["axes.labelsize"] = 9 * PLOT_SCALE
    mpl.rcParams["axes.titlesize"] = 11 * PLOT_SCALE
    mpl.rcParams["xtick.labelsize"] = 9 * PLOT_SCALE
    mpl.rcParams["ytick.labelsize"] = 9 * PLOT_SCALE
    mpl.rcParams["font.size"] = 9 * PLOT_SCALE
    mpl.rcParams["lines.linewidth"] = 1 * PLOT_SCALE
    mpl.rcParams["legend.fontsize"] = 7 * PLOT_SCALE
    mpl.rcParams["legend.title_fontsize"] = 7 * PLOT_SCALE
    mpl.rcParams["patch.linewidth"] = 1 * PLOT_SCALE
    mpl.rcParams["contour.linewidth"] = 0.5 * PLOT_SCALE
    mpl.rcParams["axes.labelpad"] = 4 * PLOT_SCALE  # space between label and axis
    mpl.rcParams["xtick.major.pad"] = (
        0 * PLOT_SCALE
    )  # distance to major tick label in points
    mpl.rcParams["ytick.major.pad"] = mpl.rcParams["xtick.major.pad"]
    mpl.rcParams["lines.markersize"] = 3 * PLOT_SCALE

    return
