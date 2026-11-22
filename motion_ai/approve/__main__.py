"""python -m motion_ai.approve output/<package>.json [--port 8765] [--clips-dir DIR]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .media import default_clips_dir
from .server import serve
from .store import approvals_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m motion_ai.approve",
                                 description="Serve a local review page for a highlights package.")
    ap.add_argument("manifest", type=Path, help="package manifest JSON (never modified)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; keep it local)")
    ap.add_argument("--clips-dir", type=Path, default=None,
                    help="directory holding per-clip mp4s (default: ../runtime_1080/clips "
                         "relative to the manifest, if found)")
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    import json
    manifest = json.loads(args.manifest.read_text())
    clips_dir = args.clips_dir or default_clips_dir(args.manifest, manifest)
    app, httpd = serve(args.manifest, clips_dir, args.host, args.port)
    n_clips = len(manifest.get("clips") or [])
    n_media = sum(app.media.has(c.get("id")) for c in manifest.get("clips") or [])
    print(f"Reviewing {args.manifest} ({n_clips} clips, {n_media} with video, clips dir {clips_dir})")
    print(f"Decisions -> {approvals_path(args.manifest)}")
    print(f"Open http://{args.host}:{args.port}/   (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
