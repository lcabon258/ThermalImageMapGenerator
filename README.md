# Thermal Image Map Generator

Build a static, interactive website to explore radiometric thermal images alongside RGB imagery on a map. This script scans a folder of images, converts radiometric TIFFs to Celsius, renders colorized previews, extracts GPS from RGB EXIF for map markers, and outputs a self-contained site using Leaflet and Bootstrap. HTML/CSS/JS are rendered from Jinja2 templates in `templates/`.

## Features

- Scans timestamp-matched image triples by filename stem:
  - RGB: `*-visible.jpg`
  - Radiometric TIFF: `*-radiometric.tif` or `*-radiometric.tiff`
  - Radiometric JPEG: `*-radiometric.jpg` (discovered but not used)
- Converts sensor DN to temperature (°C): `temp = DN/40 - 100`
- Colorizes temperatures with a fixed range and colormap (defaults to Turbo)
- Saves per-pixel temperatures as little-endian Float32 `.bin`
- Builds a static site (from Jinja2 templates) with:
  - Leaflet map + marker clustering
  - Side-by-side viewers (RGB on top, Thermal below; 40/60 height split; configurable)
  - Hover to read per-pixel temperature; click to lock/unlock readout
  - Colorbar, thumbnails, and popups with camera/time

## Requirements

- Python 3.9+
- Packages: `numpy`, `pillow`, `tifffile`, `matplotlib`, `Jinja2`

Install into a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you don't want to use the requirements file:

```powershell
pip install numpy pillow tifffile matplotlib Jinja2
```

## Usage

Basic build:

```powershell
# From the repo root
python .\build_site.py "D:\\path\\to\\input_root" "D:\\path\\to\\site_out"
```

- `input_root`: Folder tree containing your images (script scans recursively).
- `site_out`: Destination folder for the generated static site (it will be replaced).

When the build finishes, open `site_out\index.html` in a browser. For best compatibility (especially with external map tiles), serve via a local HTTP server:

```powershell
# Serve the built site on http://localhost:8000
python -m http.server -d "D:\\path\\to\\site_out" 8000
```

Then visit `http://localhost:8000/` in your browser.

## Input data expectations

- Files are matched by a common stem (typically a timestamp) and required suffixes:
  - RGB image: `-visible.jpg` (used for EXIF and optional display)
  - Radiometric TIFF: `-radiometric.tif` or `-radiometric.tiff` (required)
  - Radiometric JPEG: `-radiometric.jpg` (optional, not used)
- GPS coordinates are parsed from the RGB EXIF data. If missing, the point won't appear on the map, but the shot will still be processed and available for browsing via the popup of other points.

Example names (same stem):

```
2025-08-02_14-31-22-visible.jpg
2025-08-02_14-31-22-radiometric.tif
```

## What gets generated

Under `site_out/`:

- `index.html` — Main page (Leaflet + Bootstrap)
- `assets/`
  - `css/styles.css` — Layout and viewer styles
  - `js/main.js` — Map, viewer logic, hover readout
  - `img/colorbar.png` — Vertical colorbar (°C)
- `data/`
  - `db.json` — Lookup by shot id with paths, sizes, metadata
  - `points.geojson` — Map points (from RGB GPS)
- `media/`
  - `rgb/<hash>.jpg` — Copied RGB images (content-hash filename)
  - `thermal_color/<id>.jpg` — Colorized thermal previews
  - `thermal_dn/<id>.bin` — Float32 little-endian per-pixel temperatures (°C)
  - `thumbs/<id>.jpg` — Thumbnails for popups

Notes:
- Temperature range for colorization defaults to 24–50 °C (configurable).
- The `.bin` layout is row-major Float32 LE, length = width × height.

## Customization

Edit constants near the top of `build_site.py`:

- `PAGE_TITLE` — Page title text
- `FOOTER_TEXT` — Footer help text
- `RENDER_MIN` / `RENDER_MAX` — Color scale limits in °C
- `COLORMAP_NAME` — Matplotlib colormap (e.g. `turbo`, `inferno`, `magma`)
- Filename suffixes if your dataset uses different naming:
  - `RGB_SUFFIX` (default `-visible.jpg`)
  - `RADIOMETRIC_TIF_SUFFIX` / `RADIOMETRIC_TIFF_SUFFIX`

Templating (Jinja2):
- Edit HTML layout in `templates/index.html.j2`.
- Edit map/app logic in `templates/js/main.js.j2`.
- Edit styles/theme in `templates/css/styles.css.j2`.
- Build-time variables passed into templates include: `page_title`, `footer_text`, `render_min`, `render_max`, `split` (RGB/Thermal height ratio), `clustering_off_zoom`, and `extra_zoom_after_fit`.

## How it works (quick overview)

1. Recursively scans `input_root`, grouping files by common stem and known suffixes.
2. Loads radiometric TIFFs and converts DN to °C via `DN/40 - 100`.
3. Renders colorized JPEGs and writes Float32 `.bin` temperature arrays.
4. Extracts EXIF (camera, datetime, GPS) from RGB when available.
5. Emits a static site and data files. The browser loads `db.json` and `points.geojson` to populate the map and viewers. Hover on the thermal image to read the temperature at the cursor; click to lock/unlock the value.

## Troubleshooting

- No markers on the map: Check that RGB files have GPS EXIF. Shots without GPS are processed but omitted from `points.geojson`.
- Nothing processed: Ensure filenames match the expected suffixes and that radiometric TIFFs exist.
- Very large TIFFs: Processing uses float arrays; use 64-bit Python and sufficient RAM.
- Map tiles blocked when opening `index.html` directly: Serve via a local HTTP server as shown above.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements

- Leaflet, Leaflet.markercluster for mapping
- Bootstrap for layout
- Matplotlib for color mapping
- Pillow and tifffile for image I/O
- Jinja2 for templating
