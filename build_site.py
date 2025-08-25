#!/usr/bin/env python3
# Static site generator for thermal datasets
# - Scans RGB + radiometric TIFF images (timestamp-matched)
# - Converts radiometric DN -> °C: temp = DN/40 - 100
# - Colorizes with fixed range and saves per-pixel °C (Float32)
# - Builds a Leaflet + Bootstrap site with map, RGB/Thermal viewers,
#   4:6 height split (RGB:Thermal), placeholders, and colorbar.
# Requirements: numpy, pillow, tifffile, matplotlib
# Usage:
#   python build_site.py /path/to/input_root /path/to/site_out
#
# Notes:
# * Uses pathlib everywhere
# * Edit PAGE_TITLE / FOOTER_TEXT below to customize top/bottom text

from __future__ import annotations

# ---------- stdlib ----------
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import argparse
import hashlib
import json
import shutil

# ---------- third-party ----------
import numpy as np
from PIL import Image, ExifTags
import tifffile as tiff
from matplotlib import cm, pyplot as plt

# ---------- Editable globals ----------
PAGE_TITLE  = "Thermal Image Map for Shaoyoukeng 2025"
FOOTER_TEXT = "Click a point to load its images. Hover thermal to read temperature. Click to lock/unlock value."

# Temperature rendering (°C)
RENDER_MIN   = 24.0
RENDER_MAX   = 50.0
COLORMAP_NAME = "turbo"

# Filename suffix rules
RGB_SUFFIX               = "-visible.jpg"
RADIOMETRIC_JPG_SUFFIX   = "-radiometric.jpg"
RADIOMETRIC_TIF_SUFFIX   = "-radiometric.tif"
RADIOMETRIC_TIFF_SUFFIX  = "-radiometric.tiff"


# ---------- Helpers ----------
def sha1_name(p: Path) -> str:
    """Stable hashed id from full path string."""
    return hashlib.sha1(str(p).encode("utf-8")).hexdigest()


def ensure_dir(p: Path) -> None:
    """Create directory if missing."""
    p.mkdir(parents=True, exist_ok=True)


def find_triples(root: Path) -> Dict[str, Dict[str, Path]]:
    """Scan dataset and group files by the common timestamp stem."""
    stems: Dict[str, Dict[str, Path]] = {}
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        name = fp.name
        if name.endswith(RGB_SUFFIX):
            stems.setdefault(name[:-len(RGB_SUFFIX)], {})["rgb"] = fp
        elif name.endswith(RADIOMETRIC_JPG_SUFFIX):
            stems.setdefault(name[:-len(RADIOMETRIC_JPG_SUFFIX)], {})["rjpg"] = fp
        elif name.endswith(RADIOMETRIC_TIF_SUFFIX):
            stems.setdefault(name[:-len(RADIOMETRIC_TIF_SUFFIX)], {})["tif"] = fp
        elif name.endswith(RADIOMETRIC_TIFF_SUFFIX):
            stems.setdefault(name[:-len(RADIOMETRIC_TIFF_SUFFIX)], {})["tif"] = fp
    return stems


def read_tiff_temperature(tif_path: Path) -> np.ndarray:
    """Read radiometric TIFF and convert DN to °C with DN' = DN/40 - 100."""
    with tiff.TiffFile(str(tif_path)) as tf:
        arr = tf.asarray().astype(np.float32)
    return arr / 40.0 - 100.0


def colorize(temp: np.ndarray, vmin: float, vmax: float, cmap_name: str = COLORMAP_NAME) -> Image.Image:
    """Colorize temperature array using a matplotlib colormap."""
    t = np.clip((temp - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
    rgb = (cm.get_cmap(cmap_name)(t)[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)


def save_float32_bin(temp: np.ndarray, out_path: Path) -> None:
    """Save temperature (°C) array as little-endian Float32 binary."""
    temp.astype("<f4").tofile(out_path)


# ---------- EXIF helpers (RGB metadata fallback) ----------
GPSTAGS = ExifTags.GPSTAGS

def get_exif_dict(img_path: Path) -> Dict[str, Any]:
    """Load EXIF as a name->value dict (best effort)."""
    try:
        with Image.open(img_path) as im:
            exif = im._getexif() or {}
        return {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    except Exception:
        return {}

def parse_gps(exif: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Parse EXIF GPS into (lon, lat) in degrees."""
    gps_info = exif.get("GPSInfo")
    if not gps_info:
        return None
    gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}

    def _to_deg(val):
        try:
            n, d = val
            return float(n) / float(d) if d else 0.0
        except Exception:
            return float(val)

    def dms_to_deg(dms):
        d = _to_deg(dms[0]); m = _to_deg(dms[1]); s = _to_deg(dms[2])
        return d + m/60.0 + s/3600.0

    try:
        lat = dms_to_deg(gps["GPSLatitude"])
        lon = dms_to_deg(gps["GPSLongitude"])
        if gps.get("GPSLatitudeRef", "N") in ("S", b"S"): lat = -lat
        if gps.get("GPSLongitudeRef", "E") in ("W", b"W"): lon = -lon
        return (lon, lat)
    except Exception:
        return None

def meta_from_rgb(rgb_path: Optional[Path]) -> Dict[str, Any]:
    """Extract camera, datetime, and GPS from the RGB EXIF."""
    if not rgb_path or not rgb_path.exists():
        return {}
    exif = get_exif_dict(rgb_path)
    meta = {
        "camera": " ".join([str(exif.get("Make", "")).strip(), str(exif.get("Model", "")).strip()]).strip(),
        "datetime": str(exif.get("DateTimeOriginal") or exif.get("DateTime") or ""),
    }
    gps = parse_gps(exif)
    if gps:
        meta["_gps"] = {"lon": gps[0], "lat": gps[1]}
    return meta


# ---------- Build colorbar image ----------
def build_colorbar_png(out_path: Path, vmin: float, vmax: float, cmap_name: str = COLORMAP_NAME) -> None:
    """Render a slim vertical colorbar PNG (transparent background)."""
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(0.7, 3.2), dpi=160)
    ax = fig.add_axes([0.4, 0.05, 0.3, 0.9])
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect="auto", cmap=cmap_name, origin="lower", extent=[0, 1, vmin, vmax])
    ax.yaxis.tick_right()
    ax.set_xticks([])
    ax.set_ylabel("°C", rotation=0, labelpad=14, va="center")
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------- HTML/CSS/JS templates ----------
def make_index_html() -> str:
    """Main HTML (Leaflet + Bootstrap + our layout)."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{PAGE_TITLE}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet-tilelayer-wmts@1.0.0/leaflet-tilelayer-wmts.js"></script>
  <!-- Marker clustering -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
  <link rel="stylesheet" href="assets/css/styles.css" />
</head>
<body class="bg-light">
<div class="container-fluid g-2" style="height:100vh;">
  <div class="row gx-2 gy-2 h-100">
    <div class="col-12" style="height:10%;">
      <div class="h-100 d-flex align-items-center justify-content-between px-3 rounded bg-white shadow-sm">
        <h4 class="mb-0">{PAGE_TITLE}</h4>
        <div class="small text-muted">Static demo • Leaflet + Bootstrap</div>
      </div>
    </div>
    <div class="col-12" style="height:80%;">
      <div class="row h-100 gx-2">
        <div class="col-md-6 col-12 h-100">
          <div id="map" class="rounded bg-white shadow-sm h-100"></div>
        </div>
        <div class="col-md-6 col-12 h-100">
          <div class="d-flex flex-column h-100">
            <div class="flex-fill mb-2">
              <div id="rgbView" class="viewer rounded bg-white shadow-sm position-relative">
                <img id="rgbImg" class="fit-contain" alt="RGB" />
                <div class="viewer-title">RGB</div>
                <div id="rgbPlaceholder" class="placeholder">Click the mark on the map to show image</div>
              </div>
            </div>
            <div class="flex-fill">
              <div id="thermView" class="viewer rounded bg-white shadow-sm position-relative">
                <img id="thermImg" class="fit-contain" alt="Thermal" />
                <div id="thermOverlay" class="therm-overlay">—</div>
                <div class="colorbar"><img src="assets/img/colorbar.png" alt="Colorbar"/></div>
                <div class="viewer-title">Thermal ({RENDER_MIN:.0f}–{RENDER_MAX:.0f} °C)</div>
                <div id="thermPlaceholder" class="placeholder">Click the mark on the map to show image</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="col-12" style="height:10%;">
      <div class="h-100 d-flex align-items-center justify-content-between px-3 rounded bg-white shadow-sm">
        <div class="small">{FOOTER_TEXT}</div>
        <div class="small text-muted">© Your Org</div>
      </div>
    </div>
  </div>
</div>
<script src="assets/js/main.js"></script>
</body>
</html>
"""


def make_styles_css() -> str:
    """Minimal CSS + fit-contain images, placeholders and colorbar."""
    return """
html, body { height: 100%; }
#map { width: 100%; }
.viewer { overflow: hidden; position: relative; }
.viewer-title { position:absolute; top:8px; left:12px; background:rgba(255,255,255,0.85); padding:2px 8px; border-radius:6px; font-weight:600; font-size:0.9rem; }
.therm-overlay { position:absolute; right:10%; bottom:10%; background:rgba(0,0,0,0.6); color:white; padding:6px 10px; border-radius:8px; font-variant-numeric: tabular-nums; }
.colorbar { position:absolute; left:8px; bottom:8px; background:rgba(255,255,255,0.85); padding:4px; border-radius:8px; }
.colorbar img { display:block; width:46px; height:auto; }
.placeholder { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#777; font-style:italic; user-select:none; pointer-events:none; }
.leaflet-container { height: 100%; border-radius: 0.75rem; }
/* Ensure viewer images always fit fully inside their frames */
.viewer img.fit-contain {
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  object-position: center center;
  display: block;
}
"""


def make_main_js() -> str:
    """Client JS (map, markers, 4:6 height sync, placeholders, per-pixel readout)."""
    return r"""
let DB = {}; let DN_CACHE = {}; let map, markers; let locked = false;

function syncHeights() {
  // Set RGB:Thermal heights to 40% : 60% of map height
  const mapEl = document.getElementById('map');
  const rgbView = document.getElementById('rgbView');
  const thermView = document.getElementById('thermView');
  if (!mapEl || !rgbView || !thermView) return;
  const rect = mapEl.getBoundingClientRect();
  const gap = 8; // px space between viewers
  const usable = Math.max(160, rect.height - gap);
  rgbView.style.height = `${Math.floor(usable * 0.4)}px`;
  thermView.style.height = `${Math.floor(usable * 0.6)}px`;
}

function setPlaceholders(visible) {
  document.getElementById('rgbPlaceholder').style.display = visible ? 'flex' : 'none';
  document.getElementById('thermPlaceholder').style.display = visible ? 'flex' : 'none';
}

window.addEventListener('DOMContentLoaded', async () => {
  // Load DB + points
  DB = await fetch('data/db.json').then(r => r.json());
  const fc = await fetch('data/points.geojson').then(r => r.json());

  // Map + layers
  map = L.map('map', { zoomControl: true });
  const osm  = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '&copy; OpenStreetMap'});
  const esri = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom: 19, attribution: 'Esri'});
  osm.addTo(map); L.control.layers({'OSM': osm, 'Satellite': esri}, {}).addTo(map);

  // Marker clustering; disable at high zoom to show single markers
  markers = L.markerClusterGroup({ disableClusteringAtZoom: 18 });
  const colors = {}; const palette = ['red','blue','green','purple','orange','darkred','cadetblue'];

  (fc.features || []).forEach((f) => {
    const p = f.properties || {}; const c = f.geometry.coordinates; const cam = p.camera || 'camera';
    if (!(cam in colors)) colors[cam] = palette[Object.keys(colors).length % palette.length];
    const icon = new L.Icon({
      iconUrl: `https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-${colors[cam]}.png`,
      iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34],
      shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png', shadowSize: [41,41]
    });
    const m = L.marker([c[1], c[0]], { icon });
    const html = `<div class="d-flex align-items-center">
        <img src="${p.thumb}" width="64" height="64" style="object-fit:cover;border-radius:6px;margin-right:8px;" />
        <div><div><strong>${cam}</strong></div><div class="small text-muted">${p.datetime || ''}</div><div class="small">ID: ${p.id.slice(0,8)}</div></div></div>`;
    m.bindPopup(html); m.on('click', () => loadShot(p.id)); markers.addLayer(m);
  });
  map.addLayer(markers);

  // Fit to points, then zoom in one extra level to reduce clustering
  try {
    const bounds = L.geoJSON(fc).getBounds();
    if (bounds.isValid()) { map.fitBounds(bounds.pad(0.1)); map.once('moveend', () => { map.setZoom(map.getZoom()+1); }); }
    else { map.setView([23.5,121], 8); }
  } catch { map.setView([23.5,121], 8); }

  // Initial layout + placeholders
  syncHeights(); setPlaceholders(true);
  setTimeout(() => { map.invalidateSize(); syncHeights(); }, 50);
  window.addEventListener('resize', () => { map.invalidateSize(); syncHeights(); });
  map.on('resize', () => { syncHeights(); });

  // Thermal hover/click readout
  const thermImg = document.getElementById('thermImg'); const overlay = document.getElementById('thermOverlay');
  thermImg.addEventListener('mousemove', (ev) => { if (!locked) showValueAtEvent(ev, overlay); });
  thermImg.addEventListener('click', (ev) => { if (!locked) { showValueAtEvent(ev, overlay); locked = true; } else { locked = false; } });
  thermImg.addEventListener('mouseleave', () => { if (!locked) overlay.textContent = '—'; });
});

async function loadShot(id) {
  const rec = DB[id]; if (!rec) return;
  const rgbImg = document.getElementById('rgbImg'); const thermImg = document.getElementById('thermImg');
  rgbImg.src = rec.rgb || ''; thermImg.src = rec.thermal_color; thermImg.dataset.id = id;
  setPlaceholders(false);
  if (!DN_CACHE[id]) {
    const buf = await fetch(rec.thermal_dn).then(r => r.arrayBuffer());
    DN_CACHE[id] = { w: rec.size.w, h: rec.size.h, data: new Float32Array(buf) };
  }
}

function showValueAtEvent(ev, overlay) {
  const img = ev.currentTarget; const id = img.dataset.id;
  if (!id || !DN_CACHE[id]) { overlay.textContent = '—'; return; }
  const dn = DN_CACHE[id]; const rect = img.getBoundingClientRect();
  const xCss = ev.clientX - rect.left; const yCss = ev.clientY - rect.top;
  const scale   = Math.min(rect.width / dn.w, rect.height / dn.h);
  const renderW = dn.w * scale; const renderH = dn.h * scale;
  const xOffset = (rect.width - renderW) / 2; const yOffset = (rect.height - renderH) / 2;
  const x = Math.floor((xCss - xOffset) / scale); const y = Math.floor((yCss - yOffset) / scale);
  if (x < 0 || y < 0 || x >= dn.w || y >= dn.h) { overlay.textContent = '—'; return; }
  const idx = y * dn.w + x; const t = dn.data[idx];
  overlay.textContent = isFinite(t) ? `${t.toFixed(2)} °C` : '—';
}
"""


# ---------- Build routine ----------
def build_site(input_root: Path, out_dir: Path) -> Dict[str, Any]:
    """Generate the static web app into out_dir."""
    # 1) Prepare folders
    for d in [
        out_dir / "assets/css",
        out_dir / "assets/js",
        out_dir / "assets/img",
        out_dir / "data",
        out_dir / "media/rgb",
        out_dir / "media/thermal_color",
        out_dir / "media/thermal_dn",
        out_dir / "media/thumbs",
    ]:
        ensure_dir(d)

    # 2) Write assets (HTML/CSS/JS + colorbar image)
    (out_dir / "index.html").write_text(make_index_html(), encoding="utf-8")
    (out_dir / "assets/css/styles.css").write_text(make_styles_css(), encoding="utf-8")
    (out_dir / "assets/js/main.js").write_text(make_main_js(), encoding="utf-8")
    build_colorbar_png(out_dir / "assets/img/colorbar.png", RENDER_MIN, RENDER_MAX, COLORMAP_NAME)

    # 3) Build database + GeoJSON
    db: Dict[str, Any] = {}
    features: List[Dict[str, Any]] = []

    triples = find_triples(input_root)

    for stem, files in sorted(triples.items()):
        tif_path = files.get("tif")
        rgb_path = files.get("rgb")

        # Skip if missing radiometric TIFF
        if not tif_path:
            continue

        # Read temperature °C and render
        temp = read_tiff_temperature(tif_path)
        h, w = temp.shape
        shot_id = sha1_name(tif_path)

        color_im = colorize(temp, RENDER_MIN, RENDER_MAX)
        color_im.save(out_dir / "media/thermal_color" / f"{shot_id}.jpg", quality=92)
        save_float32_bin(temp, out_dir / "media/thermal_dn" / f"{shot_id}.bin")

        # Copy RGB (hashed filename) if present
        rgb_rel = None
        if rgb_path and rgb_path.exists():
            rgb_hash = sha1_name(rgb_path) + ".jpg"
            shutil.copy2(rgb_path, out_dir / "media/rgb" / rgb_hash)
            rgb_rel = f"media/rgb/{rgb_hash}"

        # Metadata from RGB EXIF
        meta = meta_from_rgb(rgb_path)

        # Thumbnail (from thermal color)
        thumb = color_im.copy()
        thumb.thumbnail((512, 512))
        thumb.save(out_dir / "media/thumbs" / f"{shot_id}.jpg", quality=85)

        # Record in db
        db[shot_id] = {
            "id": shot_id,
            "stem": stem,
            "rgb": rgb_rel,
            "thermal_color": f"media/thermal_color/{shot_id}.jpg",
            "thermal_dn": f"media/thermal_dn/{shot_id}.bin",
            "size": {"w": w, "h": h},
            "meta": meta,
        }

        # Map point from RGB GPS (if available)
        gps = meta.get("_gps")
        if gps:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [gps["lon"], gps["lat"]]},
                "properties": {
                    "id": shot_id,
                    "camera": meta.get("camera", ""),
                    "datetime": meta.get("datetime", ""),
                    "thumb": f"media/thumbs/{shot_id}.jpg",
                }
            })

    # 4) Write outputs
    (out_dir / "data/db.json").write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "data/points.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"shots_indexed": len(db), "features": len(features)}


# ---------- CLI ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build static thermal site")
    parser.add_argument("input_root", type=Path, help="Folder with images")
    parser.add_argument("out_dir",    type=Path, help="Output folder for static site")
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    out_dir    = args.out_dir.expanduser().resolve()

    if not input_root.exists():
        raise SystemExit(f"Input not found: {input_root}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    stats = build_site(input_root, out_dir)
    print(json.dumps(stats, indent=2))
    print(f"Done. Open {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
