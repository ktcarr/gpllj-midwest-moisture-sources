import numpy as np

def fit(X, Y):
    """Get correlation coefficient and regression coefficient"""
    Cxy = X.T @ Y
    Cxx = X.T @ X
    Cyy = Y.T @ Y
    varx = np.diag(Cxx)
    vary = np.diag(Cyy)
    coef = Cxy / varx[:, None]
    corr = Cxy / np.sqrt(varx[:, None] * vary[None, :])
    return coef

def ls_fit2(X,Y):
    m = fit(X-X.mean(0), Y-Y.mean(0))
    b = Y.mean(0) - m @ X.mean(0)[:,None]
    return np.concatenate([m,b],axis=0)


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import src.utils
    from src.params import set_plot_style
    set_plot_style()

    x = np.random.randn(10,1)+np.random.randn(1,1) 
    y = np.random.randn(10,1)+np.random.randn(1,1)

    coef1 = ls_fit2(x,y)
    coef2 = src.utils.ls_fit(x,y)

    print()
    print(coef1)
    print()
    print(coef2)

    
    x_ = np.stack([np.linspace(-5,5,50), np.ones(50)],axis=1)
    y1_ = x_ @ coef1
    y2_ = x_ @ coef2
    
    fig,ax = plt.subplots(figsize=(4,3))
    ax.scatter(x, y, c="k")
    ax.plot(x_[:,0],y1_.squeeze(), label="fit")
    ax.plot(x_[:,0],y2_.squeeze(), label="ls_fit")

    ax.axhline(0,ls='--',c='k',lw=1)
    ax.axvline(0,ls='--',c='k',lw=1)

    ax.legend()

    plt.show() 






#############################################################################
#############################################################################
#############################################################################
# from os.path import join
# import xarray as xr
# import numpy as np
# import pandas as pd
# 
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import cartopy.crs as ccrs
# import matplotlib.patches as mpatches
# import cmocean
# 
# import src.utils
# from src.params import plot_params, set_plot_style, DATA_FP
# set_plot_style(scale=1.) # set plotting style for figures
# 
# #### Check masks work properly ####
# from src.utils import get_masks, switch_lon_range, midwest_vertices, atlantic_vertices, pacific_vertices
# from matplotlib.path import Path
# from copy import deepcopy
# import warnings
# 
# lat = np.arange(80,-31,-1)
# lon = np.arange(0,360)
# 
# masks=get_masks(lat, lon, lsm_fp=f"../data/lsm.nc")
# atlantic_vertices_neg = np.concatenate(
#     [
#         switch_lon_range(atlantic_vertices[:,:1], neg_to_pos=False),
#         atlantic_vertices[:,1:]
#     ], 
# axis=1)
# 
# 
# warnings.filterwarnings('ignore')
# fig = plt.figure(figsize=(5,3))
# 
# ## Set up plot
# ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
# ax.set_extent([100, 359, -30, 81], crs=ccrs.PlateCarree())
# ax.coastlines(linewidth=plot_params["border_width"]/2)
# gl = ax.gridlines(
#     draw_labels=True,
#     linestyle='--',
#     alpha=.1,
#     linewidth=plot_params["gridline_width"],
#     color='k',
#     zorder=1.05
# )
# gl.top_labels = False
# gl.right_labels = False
# gl.ylocator = mticker.FixedLocator([-30,0, 30,60])
# gl.xlocator = mticker.FixedLocator([-120,-80,-40,0, 40])
# gl.xlabel_style = {'size': mpl.rcParams['xtick.labelsize']}
# gl.ylabel_style = {'size': mpl.rcParams['ytick.labelsize']}
# 
# 
# ax.contourf(masks.longitude, masks.latitude, 
#     masks.sel(mask=['Lake']).sum('mask'), transform=ccrs.PlateCarree())
# ax.scatter(atlantic_vertices_neg[:,:1], atlantic_vertices_neg[:,1:], transform=ccrs.PlateCarree())
# 
# # ax.contourf(masks.longitude, masks.latitude, 
# #     masks.sel(mask=['Lake']).sum('mask'), transform=ccrs.PlateCarree())
# # ax.scatter(lake_vertices[:,:1], lake_vertices[:,1:], transform=ccrs.PlateCarree())
# 
# ax.add_patch(
#     mpatches.Polygon(
#         xy=atlantic_vertices_neg,
#         closed=True, 
#         edgecolor='r',
#         facecolor='none', 
#         ls='--', 
#         zorder=1.005,
#         lw=.5,
#         transform=ccrs.PlateCarree()
#     )
# )
# 
# plt.show()
# 
# 
# 
# ############### Compare old/new results #######################
# from src.params import plot_params, set_plot_style, DATA_FP
# set_plot_style(scale=1.)
# 
# 
# topo = xr.open_dataset(join(DATA_FP, 'topo.nc'))
# topo = topo['bath'].sel(X=slice(230,300), Y=slice(15,52))
# 
# def plot_setup():
#     """create blank plot to verify results"""
#     # make plot
#     fig = plt.figure(figsize=(5,3))
# 
#     ## Set up plot
#     ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
#     ax.set_extent([235, 290, 15, 52], crs=ccrs.PlateCarree())
#     ax.coastlines(linewidth=plot_params["border_width"]/2)
#     gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=.1, linewidth=plot_params["gridline_width"], color='k', zorder=1.05)
#     gl.top_labels = False
#     gl.right_labels = False
#     gl.ylocator = mticker.FixedLocator([20,30,40,50])
#     gl.xlocator = mticker.FixedLocator([-120,-105,-90,-75])
#     gl.xlabel_style = {'size': mpl.rcParams['xtick.labelsize']}
#     gl.ylabel_style = {'size': mpl.rcParams['ytick.labelsize']}
# 
#     ## Plot 1000m contour of topography
#     ax.contour(topo.X, topo.Y, topo,
#                extend='both', colors='k',levels=[-10000,1000],
#                transform=ccrs.PlateCarree(),linewidths=.7)
#     return fig, ax
# 
# 
# ### load new results
# fluxes_new = xr.open_dataset('/vortexfs1/scratch/kcarr/WAM2layer-data/output2/fluxes_daily_1998.nc')
# storage_new = xr.open_dataset('/vortexfs1/scratch/kcarr/WAM2layer-data/output2/storage_daily_1998.nc')
# 
# # fluxes_old = xr.open_dataset('/vortexfs1/home/kcarr/gpllj-moisture-tracking/data/WAM_output/fluxes_daily_1998.nc')
# # storage_old = xr.open_dataset('/vortexfs1/home/kcarr/gpllj-moisture-tracking/data/WAM_output/storage_daily_1998.nc')
# 
# ## Load old results
# import scipy.io as sio
# data_old = sio.loadmat('/vortexfs1/home/kcarr/iap-2021/moisture-track-output/all/E_track_continental_daily_full1998-timetracking0.mat')
# 
# lonnrs        = np.arange(200,  360) # for larger domain
# latnrs        = np.arange(20,    96)
# latitude_old  = np.arange(90,   -91, -1)[np.array([int(i) for i in latnrs])]
# longitude_old = np.arange(360)[np.array([int(i) for i in lonnrs])]
# 
# Sa_track_top = data_old["Sa_track_top_per_day"]
# Sa_track_down = data_old["Sa_track_down_per_day"]
# W_top = data_old["W_top_per_day"]
# W_down = data_old["W_down_per_day"]
# E_track = data_old["E_track_per_day"]
# P = data_old["P_per_day"]
# 
# ## Stack data with two layers
# Sa_track = np.stack([Sa_track_down, Sa_track_top], axis=1)
# W = np.stack([W_down, W_top], axis=1)
# 
# ## define coordinates and dimensions for xarray
# coords=dict(
#             longitude=(["longitude"], longitude_old),
#             latitude=(["latitude"], latitude_old),
#             time=(["time"], pd.date_range(start="1998-01-01",freq="D",periods=365)),
#             level=(["level"], ["down","top"])
# )
# 
# dims_1d = ["time","longitude"]
# dims_2d = ["time","latitude","longitude"]
# dims_3d = ["time","level","latitude","longitude"]
# 
# ## Convert to dataset
# storage_old = xr.Dataset(
#     data_vars=dict(
#         Sa_track=(dims_3d, Sa_track),
#         W=(dims_3d, W)
#     ),
#     coords=coords
# )
# 
# fluxes_old = xr.Dataset(
#     data_vars=dict(
#         E_track=(dims_2d, E_track),
#         P=(dims_2d, P)
#     ),
#     coords=coords
# )
# 
# ## Get subset matching new data
# fluxes_old = fluxes_old.sel(time=fluxes_new.time)
# storage_old = storage_old.sel(time=storage_new.time)
# 
# 
# ########## Plot comparison ###########
# for t in np.arange(0,6,2):
#     print(t)
#     for data, label in zip([storage_old['Sa_track'], storage_new['Sa_track']], ['old','new']):
# 
#         fig,ax = plot_setup()
#         ax.set_title(label)
# 
#         # make the plot
#         cp = ax.contourf(
#             data.longitude, 
#             data.latitude,
#             data.sum('level').isel(time=t),
#             cmap="cmo.rain",
#             transform=ccrs.PlateCarree(),
#             levels=np.arange(0,3e7,3e6),
#             extend="max"
#         )
# 
#         # colorbar
#         cb = fig.colorbar(cp, orientation='vertical')
# 
#         plt.show()
# 
# for t in np.arange(0,6,2):
#     print(t)
#     for data, label in zip([fluxes_old['E_track'], fluxes_new['E_track']], ['old','new']):
# 
#         fig,ax = plot_setup()
#         ax.set_title(label)
# 
#         # make the plot
#         cp = ax.contourf(
#             data.longitude, 
#             data.latitude,
#             data.isel(time=t),
#             cmap="cmo.rain",
#             transform=ccrs.PlateCarree(),
#             levels=np.arange(0,3e7,3e6),
#             extend="max"
#         )
# 
#         # colorbar
#         cb = fig.colorbar(cp, orientation='vertical')
# 
#         plt.show()
