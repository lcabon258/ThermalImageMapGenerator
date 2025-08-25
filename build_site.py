#!/usr/bin/env python3
# Static site generator for thermal datasets (Leaflet + Bootstrap)
# - Finds triplets: *-visible.jpg, *-radiometric.(tif|tiff)
# - DN → °C: temp = DN/40 - 100, renders fixed range pseudocolor (24–50 °C)
# - Saves Float32 °C buffer for per-pixel readout in the browser
# - Map with clustering (disabled at high zoom), RGB/Thermal split 4:6
# - Uses Jinja2 templates for HTML + JS + CSS
#
# Usage:
#   pip install numpy pillow tifffile matplotlib Jinja2
#   python build_site.py /path/to/input_root /path/to/site_out

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
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ---------- Editable globals ----------
PAGE_TITLE  = "Thermal Image Demonstration"  # title in header
FOOTER_TEXT = "Click a point to load its images. Hover thermal to read temperature. Click to lock/unlock value."

# Temperature render settings (°C)
RENDER_MIN   = 24.0
RENDER_MAX   = 50.0
COLORMAP_NAME = "turbo"

# Map/UX config passed into templates
SPLIT = {"rgb": 0.4, "thermal": 0.6}  # heights relative to map height
CLUSTERING_OFF_ZOOM  = 18             # disable clustering at/after this zoom
EXTRA_ZOOM_AFTER_FIT = 1              # extra zoom after fitBounds

# Optional theming passed to CSS template
VIEWER_TITLE_BG = "rgba(255,255,255,0.85)"
COLORBAR_WIDTH  = "46px"

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

# ---------- Colorbar image ----------
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

    # 2) Jinja2 env + shared context
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),  # js/css won't be auto-escaped
    )
    ctx = {
        "page_title": PAGE_TITLE,
        "footer_text": FOOTER_TEXT,
        "render_min": RENDER_MIN,
        "render_max": RENDER_MAX,
        "split": SPLIT,
        "clustering_off_zoom": CLUSTERING_OFF_ZOOM,
        "extra_zoom_after_fit": EXTRA_ZOOM_AFTER_FIT,
        "colorbar_url": "assets/img/colorbar.png",
        "viewer_title_bg": VIEWER_TITLE_BG,
        "colorbar_width": COLORBAR_WIDTH,
    }

    # 3) Render HTML + JS + CSS from templates
    (out_dir / "index.html").write_text(env.get_template("index.html.j2").render(**ctx), encoding="utf-8")
    (out_dir / "assets/js/main.js").write_text(env.get_template("js/main.js.j2").render(**ctx), encoding="utf-8")
    (out_dir / "assets/css/styles.css").write_text(env.get_template("css/styles.css.j2").render(**ctx), encoding="utf-8")

    # 4) Colorbar image
    build_colorbar_png(out_dir / "assets/img/colorbar.png", RENDER_MIN, RENDER_MAX, COLORMAP_NAME)

    # 5) Build DB + GeoJSON
    db: Dict[str, Any] = {}
    features: List[Dict[str, Any]] = []
    triples = find_triples(input_root)

    for stem, files in sorted(triples.items()):
        tif_path = files.get("tif")
        rgb_path = files.get("rgb")
        if not tif_path:
            continue  # radiometric TIFF is required

        # Convert to °C, render color, and save Float32 buffer
        temp = read_tiff_temperature(tif_path)
        h, w = temp.shape
        shot_id = sha1_name(tif_path)

        color_im = colorize(temp, RENDER_MIN, RENDER_MAX)
        color_im.save(out_dir / "media/thermal_color" / f"{shot_id}.jpg", quality=92)
        save_float32_bin(temp, out_dir / "media/thermal_dn" / f"{shot_id}.bin")

        # Copy RGB (hashed) if present
        rgb_rel = None
        if rgb_path and rgb_path.exists():
            rgb_hash = sha1_name(rgb_path) + ".jpg"
            shutil.copy2(rgb_path, out_dir / "media/rgb" / rgb_hash)
            rgb_rel = f"media/rgb/{rgb_hash}"

        # Metadata from RGB EXIF
        meta = meta_from_rgb(rgb_path)

        # Thumbnail from thermal color
        thumb = color_im.copy()
        thumb.thumbnail((512, 512))
        thumb.save(out_dir / "media/thumbs" / f"{shot_id}.jpg", quality=85)

        # Record in DB
        db[shot_id] = {
            "id": shot_id,
            "stem": stem,
            "rgb": rgb_rel,
            "thermal_color": f"media/thermal_color/{shot_id}.jpg",
            "thermal_dn": f"media/thermal_dn/{shot_id}.bin",
            "size": {"w": w, "h": h},
            "meta": meta,
        }

        # GeoJSON point if GPS present
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

    # 6) Write DB + points
    (out_dir / "data/db.json").write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "data/points.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"shots_indexed": len(db), "features": len(features)}

# ---------- CLI ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build static thermal site (Jinja2 templated HTML/JS/CSS)")
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
