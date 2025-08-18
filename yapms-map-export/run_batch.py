import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

# Local imports
from yapms_from_centroid_csvs import process_year_folder  # noqa: E402


def wait_for_http(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """Poll a URL until it returns a 200 OK or timeout."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 400:
                    return True
        except Exception as e:
            last_err = e
        time.sleep(interval)
    if last_err:
        sys.stderr.write(f"Failed to reach {url}: {last_err}\n")
    return False


def start_dev_server(app_dir: Path, port: int) -> subprocess.Popen:
    """Start the YAPMS dev server (npm run dev) on the given port.

    Returns a Popen handle. Caller is responsible for stopping it.
    """
    env = os.environ.copy()
    # Force Vite to use the requested port
    cmd = ["npm", "run", "dev", "--", "--port", str(port)]
    return subprocess.Popen(cmd, cwd=str(app_dir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def kill_process_tree(p: subprocess.Popen):
    """Terminate a spawned process and its children (Windows-friendly)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            p.send_signal(signal.SIGTERM)
    except Exception:
        pass


async def main():
    parser = argparse.ArgumentParser(description="Batch export YAPMS centroid maps, optionally starting the dev server.")
    parser.add_argument("year_folder", type=Path, help="Folder with centroid CSVs, e.g., energy_predict/2028")
    parser.add_argument("--out", type=Path, default=None, help="Output folder for PNGs (defaults to year_folder)")
    parser.add_argument("--inline", action="store_true", help="Render inline (no server). Overrides --start-server/--url.")
    parser.add_argument("--start-server", action="store_true", help="Start the local YAPMS dev server automatically")
    parser.add_argument("--url", default=None, help="Server base URL (e.g., http://127.0.0.1:5173). If not provided and --start-server is set, defaults to http://127.0.0.1:5173")
    parser.add_argument("--app-dir", type=Path, default=Path(__file__).resolve().parent / "yapms" / "apps" / "yapms", help="Path to the YAPMS app (for starting the dev server)")
    parser.add_argument("--port", type=int, default=5173, help="Port for dev server when using --start-server")
    args = parser.parse_args()

    year_dir = args.year_folder
    out_dir = args.out if args.out else year_dir

    base_url = "inline"
    proc: subprocess.Popen | None = None

    if not args.inline:
        if args.url:
            base_url = args.url
        elif args.start_server:
            # Start server ourselves
            app_dir = args.app_dir
            if not app_dir.exists():
                raise SystemExit(f"App directory not found: {app_dir}")
            proc = start_dev_server(app_dir, args.port)
            try:
                # Wait until the server responds
                probe = f"http://127.0.0.1:{args.port}/embed"
                ok = wait_for_http(probe, timeout=60)
                if not ok:
                    raise SystemExit("Dev server did not become ready in time.")
            except Exception:
                if proc:
                    kill_process_tree(proc)
                raise
            base_url = f"http://127.0.0.1:{args.port}"
        else:
            # No server requested or provided; default to inline
            base_url = "inline"

    try:
        await process_year_folder(year_dir, out_dir, base_url)
    finally:
        if proc:
            kill_process_tree(proc)


if __name__ == "__main__":
    # Default convenience: process every subfolder in the repo-level energy_predict/ using inline rendering.
    repo_root = Path(__file__).resolve().parent.parent
    energy_root = repo_root / "energy_predict"
    if energy_root.exists() and energy_root.is_dir():
        # Process each immediate subdirectory (e.g., 2028, 2032)
        dirs = sorted([p for p in energy_root.iterdir() if p.is_dir()])
        if not dirs:
            print(f"No subfolders found in {energy_root}")
        for d in dirs:
            print(f"Processing folder: {d}")
            try:
                # Use the folder itself as the out directory and inline rendering
                asyncio.run(process_year_folder(d, d, "inline"))
            except Exception as e:
                print(f"Failed processing {d}: {e}", file=sys.stderr)
    else:
        # Fallback: run the normal CLI-driven main
        asyncio.run(main())
    # example manual run:
    # & "E:/coding projects/historical-PVI/.venv/Scripts/python.exe" "e:/coding projects/historical-PVI/yapms-map-export/run_batch.py"
