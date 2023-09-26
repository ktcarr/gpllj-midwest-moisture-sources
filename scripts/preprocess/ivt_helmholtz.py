import xarray as xr
import windspharm.xarray
import src.params

def main():
    """Perform helmholtz decomposition of data, and put in xarray dataset"""

    print("Loading data")
    data = xr.open_dataset(f"{src.params.DATA_RAW_FP}/ivt.nc")
    data = data.rename({"p71.162": "ivt_u", "p72.162": "ivt_v"})
    data.load();

    print("Computing Legendre polynomials")
    vw = windspharm.xarray.VectorWind(u=data["ivt_u"], v=data["ivt_v"])

    print("Helmholtz decomposition")
    uchi, vchi, upsi, vpsi = vw.helmholtz()
    _, div = vw.vrtdiv()
    _, pot = vw.sfvp()

    print("Merging into single dataset")
    data = xr.merge([uchi, vchi, upsi, vpsi, div, pot])
    data = data.sel(latitude=slice(60,5), longitude=slice(210,330))

    print("Saving to file")
    data.to_netcdf(f"{src.params.DATA_PREPPED_FP}/helmholtz.nc")
    print("Done")

    return 


if __name__ == "__main__":
    main()
