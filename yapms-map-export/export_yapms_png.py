import asyncio
import base64
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

from playwright.async_api import async_playwright
import utils  # type: ignore
from typing import Optional


def _colorize_svg(svg_text: str, saved_map: dict) -> str:
    """Colorize a YAPMS USA SVG by matching on the `region` attribute.

    The upstream SVG uses attributes like region="ca", short-name="ca" on each path,
    not element ids. Previously we looked up by id which left fills unchanged (black).
    """
    # Build region -> color mapping from SavedMap JSON
    colors: dict[str, str] = {}
    tossup_color = saved_map.get("tossup", {}).get("margins", [{}])[0].get("color", "#cccccc")
    candidates = {str(c["id"]): c for c in saved_map.get("candidates", [])}
    # Also build region -> EV value mapping for label updates
    ev_values: dict[str, int] = {}
    for r in saved_map.get("regions", []):
        rid = (r.get("id") or "").lower()
        if not rid:
            continue
        # Capture EV value for text labels
        try:
            ev_values[rid] = int(r.get("value") or 0)
        except Exception:
            ev_values[rid] = 0
        cand_list = r.get("candidates") or []
        if not cand_list:
            colors[rid] = tossup_color
            continue
        c0 = cand_list[0]
        cid = str(c0.get("id"))
        m = int(c0.get("margin", 0))
        cand = candidates.get(cid)
        if not cand:
            colors[rid] = tossup_color
            continue
        margins = cand.get("margins", [])
        col = margins[m]["color"] if 0 <= m < len(margins) else tossup_color
        colors[rid] = col

    # Parse SVG and set fill for elements with matching region/short-name/id. Update style if present.
    try:
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        root = ET.fromstring(svg_text)
        # Default all region fills to tossup to avoid black default fills
        for el in root.iter():
            key = (el.attrib.get("region") or el.attrib.get("short-name") or el.attrib.get("id") or "").lower()
            if key:
                el.set("fill", tossup_color)
        # Then override specific regions
        for el in root.iter():
            key = (el.attrib.get("region") or el.attrib.get("short-name") or el.attrib.get("id") or "").lower()
            if not key:
                continue
            col = colors.get(key)
            if not col:
                continue
            el.set("fill", col)
            style = el.attrib.get("style")
            if style:
                new_style = re.sub(r"fill:\s*#[0-9a-fA-F]{3,8}", f"fill:{col}", style)
                if new_style == style:
                    if not new_style.endswith(";"):
                        new_style += ";"
                    new_style += f"fill:{col}"
                el.set("style", new_style)
        # Update EV labels: <text for-region="xx"> (or a nearby ancestor <g for-region>) contains
        # <tspan map-type="value-text">00</tspan>. Some SVGs put the for-region on a parent <g>.
        for el in root.iter():
            # Look for text nodes with a region to target
            if not el.tag.endswith('text'):
                continue
            reg = (el.attrib.get("for-region") or el.attrib.get("region") or "").lower()
            # If the text itself doesn't carry the region, try to find an ancestor (e.g., a <g>)
            if not reg:
                for anc in root.iter():
                    # quick containment test: is `el` inside this ancestor's subtree?
                    found = False
                    for child in anc.iter():
                        if child is el:
                            found = True
                            break
                    if not found:
                        continue
                    reg = (anc.attrib.get("for-region") or anc.attrib.get("region") or anc.attrib.get("short-name") or "").lower()
                    if reg:
                        break
            if not reg:
                continue
            ev = ev_values.get(reg)
            # skip missing or zero EV entries
            if ev is None or ev == 0:
                continue
            # Find nested tspans with map-type="value-text" and set their text
            for tsp in el.iter():
                if tsp.tag.endswith('tspan') and tsp.attrib.get("map-type") == "value-text":
                    tsp.text = str(ev)
        return ET.tostring(root, encoding="unicode")
    except Exception:
        # If parsing fails, fall back to original text unchanged
        return svg_text


async def export_png(base64_data: str, out_path: Path, base_url: str = "inline", year: Optional[str] = None):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1400, "height": 900})

        if base_url.lower() == "inline":
            # Inline mode: colorize the USA SVG locally and render without any server
            try:
                saved_json = json.loads(base64.b64decode(base64_data).decode("utf-8"))
            except Exception as e:
                raise RuntimeError(f"Invalid base64 SavedMap JSON: {e}")

            #year = str(saved_json.get("map", {}).get("year", "2024312"))
            # Prefer the explicit `year` parameter. Do NOT extract year from saved_json
            # to avoid incorrect fallback values like '2024312'. If no year was passed,
            # use a harmless placeholder.
            year = str(year) if year is not None else "unknown"
            svg_dir = (
                Path(__file__).resolve().parent
                / "yapms"
                / "apps"
                / "yapms"
                / "src"
                / "lib"
                / "assets"
                / "maps"
                / "usa"
            )

            def _read_valid_svg(path: Path) -> str:
                try:
                    txt = path.read_text(encoding="utf-8")
                except Exception:
                    return ""
                if "<svg" in txt.lower():
                    return txt
                return ""

            # Prefer asset matching saved map year, but fall back to any usable usa-presidential blank SVG
            preferred = svg_dir / f"usa-presidential-{year}-blank.svg"
            svg_text = _read_valid_svg(preferred)
            chosen_svg_path = preferred
            if not svg_text:
                # Try known 2028 canonical asset
                cand2028 = svg_dir / "usa-presidential-2028-blank.svg"
                svg_text = _read_valid_svg(cand2028)
                if svg_text:
                    chosen_svg_path = cand2028
            if not svg_text:
                # Scan for any usa-presidential-*-blank.svg that actually contains markup
                for cand in sorted(svg_dir.glob("usa-presidential-*-blank.svg")):
                    svg_text = _read_valid_svg(cand)
                    if svg_text:
                        chosen_svg_path = cand
                        break
            if not svg_text:
                raise RuntimeError(f"No valid SVG asset found in {svg_dir}")

            colored_svg = _colorize_svg(svg_text, saved_json)

            # Compute battle bar totals
            dem_ev = rep_ev = toss_ev = 0
            for r in saved_json.get("regions", []):
                ev = int(r.get("value") or 0)
                cands = r.get("candidates") or []
                if not ev:
                    continue
                if not cands:
                    toss_ev += ev
                else:
                    cid = str((cands[0] or {}).get("id"))
                    if cid == "0":
                        dem_ev += ev
                    elif cid == "1":
                        rep_ev += ev
                    else:
                        toss_ev += ev
                total_ev = max(1, dem_ev + rep_ev + toss_ev)
                threshold_ev = 270
                threshold_pct = max(0.0, min(100.0, (threshold_ev / total_ev) * 100.0))

                # Prepare Tossup label (hide if zero)
                toss_label_html = f'<span class="toss">Tossup {toss_ev}</span>' if toss_ev else ''

                # Popular vote (national margin) PV bar
                try:
                        national_margin = float(saved_json.get("national_margin", 0.0) or 0.0)
                except Exception:
                        national_margin = 0.0

                d_pct = (1.0 + national_margin) / 2.0
                r_pct = (1.0 - national_margin) / 2.0
                d_pct = max(0.0, min(1.0, d_pct))
                r_pct = max(0.0, min(1.0, r_pct))

                pv_dem_percent = d_pct * 100.0
                pv_rep_percent = r_pct * 100.0
                pv_label = utils.lean_str(national_margin)

                pv_bar_html = f"""
<!-- Popular vote (PV) bar -->
<div style="margin-top:8px;width:1200px;">
    <!-- PV lean label (above the bar) -->
    <div style="text-align:center;color:#ffffff;font-weight:800;font-size:13px;margin-bottom:6px;">{pv_label}</div>
    <div class="battle pv" style="width:100%;">
        <div class="bar pv-bar" style="position:relative;height:24px;background:#0b1220;border-radius:8px;overflow:hidden;">
            <!-- Democrat portion (left) -->
            <div style="position:absolute;left:0;top:0;bottom:0;width:{pv_dem_percent:.3f}%;background:#183ebf;"></div>
            <!-- Republican portion (right) -->
            <div style="position:absolute;right:0;top:0;bottom:0;width:{pv_rep_percent:.3f}%;background:#bf1d29;"></div>
            <!-- Center dashed line at 50% -->
            <div style="position:absolute;left:50%;top:-8px;bottom:-8px;width:0;border-left:3px dashed rgba(255,255,255,0.9);pointer-events:none;"></div>
            <!-- Democrat percentage (left) -->
            <div style="position:absolute;left:10px;top:50%;transform:translateY(-50%);color:#cfe6ff;font-weight:700;font-size:12px;">{pv_dem_percent:.1f}%</div>
            <!-- Republican percentage (right) -->
            <div style="position:absolute;right:10px;top:50%;transform:translateY(-50%);color:#ffc8c8;font-weight:700;font-size:12px;">{pv_rep_percent:.1f}%</div>
        </div>
    </div>
</div>
"""

            # Dark theme + fine borders for region seams
            # Add a year header above the map for quick identification
            year_header_html = f"<div style=\"position:relative;width:1200px;text-align:center;color:#ffffff;font-weight:900;font-size:26px;margin-bottom:6px;\">{year}</div>"

            html = f"""
            <!doctype html>
            <meta charset=\"utf-8\">
            <style>
                            :root {{ color-scheme: dark; }}
                            html, body {{ margin: 0; padding: 0; background: #0d1117; }}
                            .wrap {{
                                display: flex;
                                flex-direction: column;
                                align-items: center;
                                justify-content: center;
                                width: 100vw;
                                height: 100vh;
                                gap: 16px;
                            }}
              svg {{ width: 1200px; height: auto; background: transparent; }}
              svg [region], svg [short-name] {{
                stroke: #334155;
                stroke-width: .6;
                vector-effect: non-scaling-stroke;
              }}
                            svg text {{
                                /* stronger contrast: white fill with dark outline */
                                fill: #ffffff;
                                font: 700 12px system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, \"Apple Color Emoji\", \"Segoe UI Emoji\";
                                paint-order: stroke fill;
                                stroke: rgba(0,0,0,0.85);
                                stroke-width: 1.6px;
                                stroke-linejoin: round;
                                stroke-linecap: round;
                                letter-spacing: .2px;
                            }}
                            /* EV number (tspan with map-type=value-text) get an extra-thick outline to stay readable on very light fills */
                            svg text tspan[map-type="value-text"] {{
                                font-weight: 800;
                                stroke-width: 2.2px;
                                font-size: 11px;
                                fill: #ffffff;
                            }}

                            /* Battle bar */
                            .battle {{ width: 1200px; color: #e2e8f0; font: 600 12px system-ui, Segoe UI, Roboto; }}
                            .bar {{
                                position: relative;
                                display: flex;
                                height: 18px;
                                border-radius: 6px;
                                overflow: hidden;
                                background: #111827;
                                box-shadow: inset 0 0 0 1px rgba(148,163,184,.25);
                            }}
                            .seg {{ flex-basis: 0; }}
                            .seg.dem {{ background: #183ebf; }}  /* D_called */
                            .seg.toss {{ background: {saved_json.get('tossup', {}).get('margins', [{}])[0].get('color', '#C3B1E1')}; }}
                            .seg.rep {{ background: #bf1d29; }}  /* R_called */
                            .threshold {{
                                position: absolute;
                                top: -4px;
                                bottom: -4px;
                                width: 0;
                                left: {threshold_pct:.3f}%;
                                border-left: 2px dashed rgba(226,232,240,.9);
                                pointer-events: none;
                            }}
                            .threshold::after {{
                                content: '270';
                                position: absolute;
                                top: -18px;
                                left: 4px;
                                color: #94a3b8;
                                font-weight: 700;
                                letter-spacing: .2px;
                            }}
                            .labels {{
                                display: flex;
                                justify-content: space-between;
                                margin-top: 6px;
                            }}
                            .labels .dem {{ color: #93c5fd; }}
                            .labels .toss {{ color: #d8c7f1; }}
                            .labels .rep {{ color: #fca5a5; }}
            </style>
                        <div class=\"wrap\">
                            {year_header_html}
                            {colored_svg}
                            <div class=\"battle\">
                                <div class=\"bar\">
                                    <div class=\"seg dem\" style=\"flex-grow: {dem_ev};\"></div>
                                    <div class=\"seg toss\" style=\"flex-grow: {toss_ev};\"></div>
                                    <div class=\"seg rep\" style=\"flex-grow: {rep_ev};\"></div>
                                    <div class=\"threshold\"></div>
                                </div>
                                <div class=\"labels\">
                                    <span class=\"dem\">D {dem_ev}</span>
                                    {toss_label_html}
                                    <span class=\"rep\">R {rep_ev}</span>
                                </div>
                            </div>
                            {pv_bar_html}
                        </div>
            """

            await page.set_content(html, wait_until="domcontentloaded")
            await page.wait_for_selector("svg", state="attached", timeout=10000)
        else:
            # Remote mode via dev server embed route
            url = f"{base_url}/embed?" + urlencode({"data": base64_data})
            # Retry a few times in case the server is still starting.
            last_err = None
            for attempt in range(3):
                try:
                    resp = await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    if resp is None or not resp.ok:
                        raise RuntimeError(f"HTTP not ok: {resp.status if resp else 'no response'}")
                    break
                except Exception as e:
                    last_err = e
                    await page.wait_for_timeout(1000)
            else:
                raise RuntimeError(f"Failed to load {url}: {last_err}")
            # Wait for map SVG container to be present and stable
            try:
                await page.wait_for_selector("svg[data-map]", state="attached", timeout=12000)
            except Exception:
                await page.wait_for_selector("svg", state="attached", timeout=6000)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            await page.wait_for_timeout(300)

        await page.screenshot(path=str(out_path), full_page=False)
        await browser.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export a YAPMS map to PNG using the embed route")
    parser.add_argument("base64_json", help="Base64-encoded SavedMap JSON")
    parser.add_argument("--out", type=Path, default=Path("map.png"), help="Output PNG path")
    parser.add_argument("--url", default="inline", help="YAPMS base URL (dev server) or 'inline' to render locally")
    parser.add_argument("--year", default=None, help="Year string to display on the exported image (overrides any value inside the SavedMap)")
    args = parser.parse_args()
    asyncio.run(export_png(args.base64_json, args.out, args.url, year=args.year))


if __name__ == "__main__":
    main()
