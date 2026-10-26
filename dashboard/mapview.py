"""mapview.py - free maps (OpenStreetMap + Esri satellite) with folium. Switch to Google later here only."""
import folium
import numpy as np

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def _base(lat, lon, zoom):
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(ESRI, attr="Esri World Imagery", name="Satellite (Esri)").add_to(m)
    return m


def pick_map(lat, lon, zoom=11):
    """Map the user clicks on to set the farm location."""
    m = _base(lat, lon, zoom)
    folium.Marker([lat, lon], tooltip="Farm").add_to(m)
    folium.LayerControl().add_to(m)
    return m


def sm_overlay_map(tif_path, lat, lon, half_deg=0.02, vmin=0.10, vmax=0.45):
    """Map with the 70 m soil moisture around the farm. Returns (map, (min, max) shown, or None)."""
    import matplotlib
    import rasterio
    from rasterio.windows import from_bounds
    m = _base(lat, lon, 15)
    rng = None
    with rasterio.open(tif_path) as src:
        w, s, e, n = lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg
        win = from_bounds(w, s, e, n, src.transform)
        arr = src.read(1, window=win, boundless=True, fill_value=np.nan, out_shape=(300, 300)).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    good = np.isfinite(arr)
    if good.any():
        rng = (float(np.nanmin(arr)), float(np.nanmax(arr)))
        rgba = matplotlib.colormaps["RdYlBu"]((np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin))
        rgba[~good, 3] = 0.0
        rgba[good, 3] = 0.65
        folium.raster_layers.ImageOverlay(image=rgba, bounds=[[s, w], [n, e]], name="70 m soil moisture",
                                          origin="upper").add_to(m)
    folium.Marker([lat, lon], tooltip="Farm").add_to(m)
    folium.LayerControl().add_to(m)
    return m, rng
