import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import src.utils
import src.params
import os


def mod_ax(ax):
    """add vert/horiz lines, specify ticks, and add labels to ax"""
    ax.axhline(0, ls="-", c="gray", lw=0.7)
    ax.axvline(0, ls="-", c="gray", lw=0.7)
    ax.set_xticks(ticks=np.arange(-12, 6, 3))
    ax.set_xlim([-13, 3])
    ax.set_xlabel("Lag (days)")

    return ax


def load_data():
    agg_r1 = xr.open_dataset(f"./composite_agg_v850_amj_r1.nc")
    agg_r2 = xr.open_dataset(f"{src.params.DATA_PREPPED_FP}/composite_agg_v850_amj.nc")

    return agg_r1, agg_r2


def main():
    ## set plotting style
    sns.set()
    src.params.set_plot_style()

    ## Load data
    agg_r1, agg_r2 = load_data()

    ## Make plot
    aspect = 3.5
    width = src.params.plot_params["max_width"]
    height = width / aspect
    fig = plt.figure(figsize=(width, height), layout="constrained")
    gs = fig.add_gridspec(nrows=1, ncols=5, width_ratios=[1, 0.1, 1, 0.1, 1])

    ## E and P
    ax0 = fig.add_subplot(gs[0])
    colors = [sns.color_palette("colorblind")[i] for i in [5, 9]]
    masks = ["Total", "Midwest"]
    ax0.plot(agg_r1.lag, agg_r1["E_track"].sel(mask="Total"), c=colors[0], ls="--")
    ax0.plot(agg_r1.lag, agg_r1["P"].sel(mask="Midwest"), c=colors[1], ls="--")
    ax0.plot(
        agg_r2.lag,
        agg_r2["E_track"].sel(mask="Total"),
        c=colors[0],
        label="Tracked evap.",
    )
    ax0.plot(
        agg_r2.lag,
        agg_r2["P"].sel(mask="Midwest"),
        c=colors[1],
        label="Midwest precip.",
    )
    ax0.legend()
    ax0 = mod_ax(ax0)
    ax0.set_ylabel("Anomaly ($mm$)")

    ## Oc and La
    ax2 = fig.add_subplot(gs[2])
    colors = [sns.color_palette()[i] for i in [0, 2, 1]]
    masks = ["Ocean", "Land", "Midwest"]
    for c, m in zip(colors, masks):
        ax2.plot(agg_r2.lag, agg_r2["E_track"].sel(mask=m), c=c, label=f"{m}")
        ax2.plot(agg_r1.lag, agg_r1["E_track"].sel(mask=m), c=c, ls="--")
    ax2.legend()
    ax2 = mod_ax(ax2)

    ## Atl and Pac
    ax4 = fig.add_subplot(gs[4])
    colors = [sns.color_palette("mako")[i] for i in [3, 0]]
    masks = ["Atlantic", "Pacific"]
    for c, m in zip(colors, masks):
        ax4.plot(agg_r2.lag, agg_r2["E_track"].sel(mask=m), c=c, label=f"{m}")
        ax4.plot(agg_r1.lag, agg_r1["E_track"].sel(mask=m), c=c, ls="--")
    ax4.legend()
    ax4 = mod_ax(ax4)

    ## Label subplots
    ax0 = src.utils.label_subplot(
        ax=ax0, label="a)", posn=(0.1, 0.1), transform=ax0.transAxes
    )
    ax2 = src.utils.label_subplot(
        ax=ax2, label="b)", posn=(0.1, 0.1), transform=ax2.transAxes
    )
    ax4 = src.utils.label_subplot(
        ax=ax4, label="c)", posn=(0.1, 0.1), transform=ax4.transAxes
    )

    ## save
    src.utils.save(fig=fig, fname="r2_r1_comp", is_supp=True)
    plt.close(fig)

    return


if __name__ == "__main__":
    src.params.set_plot_style()
    main()
