"""
slope_grid.py - slope (%) from ground elevation on a regular lat/lon grid (used by step 06).

Method (as specified):
  * each point is placed on its grid cell from lat and lon (the grid is about 70 m);
  * grid steps are converted to metres:  dy = dlat * 111320 ;  dx = dlon * 111320 * cos(lat);
  * central differences of elevation in x and y where both neighbours exist; a one-sided
    difference where only one neighbour exists (grid edge, or next to a missing cell);
  * slope_pct = 100 * sqrt((dz/dx)^2 + (dz/dy)^2)    (% = rise per 100 m run);
  * if the x or the y direction has no neighbour at all, the slope is left empty (NaN).

The function also returns a report about the grid (spacing, how complete it is, how regular it is),
so you can check the "regular, complete grid" assumption before trusting the numbers.
"""
import numpy as np

M_PER_DEG = 111_320.0


def _axis_diff(prev, cur, nxt, step):
    """Central difference where both neighbours exist, else one-sided, else NaN."""
    fp, fn = np.isfinite(prev), np.isfinite(nxt)
    with np.errstate(invalid="ignore"):
        d = np.where(fp & fn, (nxt - prev) / (2.0 * step),
                     np.where(fn, (nxt - cur) / step,
                              np.where(fp, (cur - prev) / step, np.nan)))
    return d


def slope_pct_from_points(lat, lon, elev, decimals=6, chunk_rows=400, max_cells=600_000_000,
                          max_offset_frac=0.25):
    """lat, lon, elev: 1-D arrays, ONE row per point (no repeated dates). Returns (slope_pct, report).
    slope_pct is a float32 array aligned with the input; NaN where it cannot be calculated.
    Raises ValueError if the points do not sit on a regular grid."""
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    elev = np.asarray(elev, dtype="float32")
    latr, lonr = np.round(lat, decimals), np.round(lon, decimals)
    ulat, ulon = np.unique(latr), np.unique(lonr)
    if len(ulat) < 2 or len(ulon) < 2:
        raise ValueError("need at least 2 different latitudes and 2 different longitudes")
    dlat, dlon = float(np.median(np.diff(ulat))), float(np.median(np.diff(ulon)))
    iy = np.rint((latr - ulat[0]) / dlat).astype("int64")
    ix = np.rint((lonr - ulon[0]) / dlon).astype("int64")
    off = np.maximum(np.abs(latr - (ulat[0] + iy * dlat)) / dlat,
                     np.abs(lonr - (ulon[0] + ix * dlon)) / dlon)
    worst = float(off.max())
    ny, nx = int(iy.max()) + 1, int(ix.max()) + 1
    cell_id = iy * nx + ix
    n_collisions = int(len(cell_id) - len(np.unique(cell_id)))
    report = dict(n_points=int(len(lat)), n_rows=ny, n_cols=nx, dlat_deg=dlat, dlon_deg=dlon,
                  dy_m=dlat * M_PER_DEG, dx_m_mid_lat=dlon * M_PER_DEG * np.cos(np.radians(lat.mean())),
                  grid_filled_pct=100.0 * len(lat) / float(ny * nx),
                  worst_offset_fraction_of_cell=worst, cells_with_2plus_points=n_collisions)
    if worst > max_offset_frac:
        raise ValueError(f"points are not on a regular grid (worst offset {worst:.2f} of a cell); "
                         f"report so far: {report}")
    if ny * nx > max_cells:
        raise ValueError(f"grid of {ny} x {nx} cells is too large for memory")

    grid = np.full((ny, nx), np.nan, dtype="float32")
    grid[iy, ix] = elev
    lat_rows = ulat[0] + np.arange(ny) * dlat
    dy = dlat * M_PER_DEG
    out = np.full((ny, nx), np.nan, dtype="float32")
    for r0 in range(0, ny, chunk_rows):
        r1 = min(r0 + chunk_rows, ny)
        blk = np.full((r1 - r0 + 2, nx + 2), np.nan, dtype="float32")
        a, b = max(r0 - 1, 0), min(r1 + 1, ny)
        s = a - (r0 - 1)
        blk[s:s + (b - a), 1:-1] = grid[a:b]
        cur = blk[1:-1, 1:-1]
        dx = (dlon * M_PER_DEG * np.cos(np.radians(lat_rows[r0:r1])))[:, None]
        dzdx = _axis_diff(blk[1:-1, :-2], cur, blk[1:-1, 2:], dx)
        dzdy = _axis_diff(blk[:-2, 1:-1], cur, blk[2:, 1:-1], dy)
        sl = 100.0 * np.sqrt(dzdx ** 2 + dzdy ** 2)
        sl[~np.isfinite(cur)] = np.nan
        out[r0:r1] = sl.astype("float32")
    slope = out[iy, ix]
    report["n_slope_missing"] = int(np.isnan(slope).sum())
    ok = slope[np.isfinite(slope)]
    if len(ok):
        report["slope_pct_p50_p95_max"] = tuple(float(x) for x in np.percentile(ok, [50, 95, 100]))
    return slope, report
