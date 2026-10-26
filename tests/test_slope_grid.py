"""Run:  python tests/test_slope_grid.py   (synthetic grids; no real data needed)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from slope_grid import slope_pct_from_points, M_PER_DEG

d = 0.0007                                  # about 78 m
lats = 20.0 + d * np.arange(60)
lons = 85.0 + d * np.arange(80)
LA, LO = np.meshgrid(lats, lons, indexing="ij")
lat, lon = LA.ravel(), LO.ravel()
# plane: rises 2 m per 100 m northwards and 1 m per 100 m eastwards  -> slope = sqrt(2^2+1^2) = 2.236 %
z = 0.02 * (lat - 20.0) * M_PER_DEG + 0.01 * (lon - 85.0) * M_PER_DEG * np.cos(np.radians(lat))
sl, rep = slope_pct_from_points(lat, lon, z, chunk_rows=17)       # small chunks test the halo handling
want = np.sqrt(0.02 ** 2 + 0.01 ** 2) * 100
print("plane: min/max slope", np.nanmin(sl), np.nanmax(sl), "expected", want)
assert np.allclose(sl, want, rtol=0.02), "plane slope wrong (interior or edges)"
assert rep["n_slope_missing"] == 0 and rep["cells_with_2plus_points"] == 0

# holes: remove some points; isolated point must be empty, neighbours of holes still get a value
keep = np.ones(len(lat), bool)
rng = np.random.default_rng(1)
keep[rng.choice(len(lat), 300, replace=False)] = False
sl2, rep2 = slope_pct_from_points(lat[keep], lon[keep], z[keep], chunk_rows=23)
fin = np.isfinite(sl2)
print("with holes: points", keep.sum(), "slope available", fin.sum(), "grid filled %.1f%%" % rep2["grid_filled_pct"])
assert np.allclose(sl2[fin], want, rtol=0.02)

iso_lat = np.array([20.0, 20.0 + 5 * d]); iso_lon = np.array([85.0, 85.0 + 5 * d])
la3 = np.concatenate([lat, [20.0 + 70 * d]]); lo3 = np.concatenate([lon, [85.0 + 90 * d]])
sl3, _ = slope_pct_from_points(la3, lo3, np.concatenate([z, [50.0]]))
assert np.isnan(sl3[-1]), "isolated point must have empty slope"

# irregular points must be refused
try:
    slope_pct_from_points(lat + rng.uniform(0, d * 0.6, len(lat)), lon, z)
    raise SystemExit("irregular grid was NOT detected")
except ValueError as e:
    print("irregular grid refused:", str(e)[:60], "...")
print("SLOPE TESTS PASSED")
