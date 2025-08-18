# YAPMS Map Export

This folder contains a small pipeline to turn your margins CSV into a YAPMS map and export it as a PNG.

Contents:
- build_centroid_map_json.py — reads a CSV (abbr, pres_margin or relative_margin, electoral_votes) and builds a SavedMap JSON for the USA presidential blank map, output as base64.
- export_yapms_png.py — launches a headless browser to open the /embed route in YAPMS and saves a PNG.

Quick usage:
1) Start your YAPMS dev server locally (default http://localhost:5173) so the /embed route is available.
2) Generate base64 JSON from your CSV:
   python yapms-map-export/build_centroid_map_json.py presidential_margins_future.csv --relative > map.b64
3) Export PNG:
   python yapms-map-export/export_yapms_png.py (Get-Content map.b64) --out out.png --url http://localhost:5173

Notes:
- ME/NE statewide entries are expanded to at-large and district regions with equal margins if district-level rows aren't present.
- Margin bins default to [5,10,20,+inf] percent. Override with --bins 0.03 0.06 0.12 to change shading tiers.
- Install Playwright once and ensure browsers are installed: pip install -r requirements.txt; python -m playwright install
