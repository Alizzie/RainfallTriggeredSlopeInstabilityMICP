import numpy as np, pandas as pd, xarray as xr
import bucket_model as bm, constants as const

region_id, e, n = 65, 2720193, 1079228  # Colrerio point coords
year = 2014
ds = xr.open_dataset(
    f"data/rhiresD/ogd-surface-derived-grid-archive.rhiresd_ch01h.swiss.lv95_{year}0101000000_{year}1231000000.nc"
)
precip = (
    ds["RhiresD"]
    .sel(E=e, N=n, method="nearest")
    .sel(time=slice("2014-07-19", "2014-11-30"))
)
rain = np.nan_to_num(precip.values, nan=0.0)

S = bm.calculate_daily_saturation(
    rain,
    n=const.N,
    n_perp=const.H_PERP,
    m0=0.60,
    s_pp_onset=0.76,
    drainage_rate=1.959,
    et_rate=1.513,
)
S = pd.Series(S, index=pd.to_datetime(precip.time.values))
rain = pd.Series(rain, index=S.index)

window = slice("2014-11-08", "2014-11-20")
print(
    pd.DataFrame({"rain_mm": rain[window].round(1), "saturation": S[window].round(3)})
)
print(f"\nPeak rain in window: {rain[window].max():.1f} mm/day")
print(f"Peak saturation in window: {S[window].max():.3f}")
