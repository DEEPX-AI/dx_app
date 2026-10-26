#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Fetch the Visual Place Recognition toy dataset into ``sample/vpr/``.

Why this dataset and not EigenPlaces' own: ``gmberton/EigenPlaces`` ships code only.
It trains on SF-XL, which is hundreds of GB behind an access request, and its
``datasets/test_dataset.py`` expects a ``database/`` + ``queries/`` pair that the repo
does not contain. The EigenPlaces README points at ``gmberton/VPR-methods-evaluation``
as the standardised evaluation codebase, and THAT repo ships exactly such a pair as a
deliberately lightweight demo -- 17 database images and 5 queries, MIT, same author.
So this is EigenPlaces' own recommended sample data, one repo along.

NO GROUND TRUTH: upstream runs this set with ``--no_labels``. The canonical VPR
filename convention encodes position as ``@utm_east@utm_north@...@.jpg``, and these 22
files carry none, so top-k retrieval can be shown but Recall@k cannot be computed. A
geo-referenced set (St Lucia, SPED, ...) comes from ``gmberton/VPR-datasets-downloader``;
``--layout vpr`` below writes the folder shape those datasets use, so a real dataset can
be dropped in beside this one without touching the example.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPO = "gmberton/VPR-methods-evaluation"
REF = "master"
RAW = f"https://raw.githubusercontent.com/{REPO}/{REF}"
LICENSE_URL = f"{RAW}/LICENSE"

# The toy dataset's own file list, from the repo tree. Named explicitly rather than
# discovered so that a partial fetch is an error instead of a smaller database.
DATABASE = [f"db{i}.jpg" for i in range(1, 18)]   # 17
QUERIES = [f"q{i}.jpg" for i in range(1, 6)]      # 5


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dx_app-setup"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_one(args: tuple[str, Path]) -> tuple[Path, int, str]:
    url, dest = args
    blob = get(url)
    # A GitHub 404 page is HTML with a 200 on some mirrors; a JPEG starts with FFD8.
    if not blob.startswith(b"\xff\xd8"):
        raise ValueError(f"{url} did not return a JPEG (first bytes {blob[:8]!r})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return dest, len(blob), hashlib.sha256(blob).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "sample" / "vpr"),
                    help="destination directory (default: sample/vpr)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="re-download files that already exist")
    args = ap.parse_args()

    out = Path(args.out)
    jobs: list[tuple[str, Path]] = []
    for folder, names in (("database", DATABASE), ("queries", QUERIES)):
        for n in names:
            dest = out / folder / n
            if dest.exists() and not args.force:
                continue
            jobs.append((f"{RAW}/toy_dataset/{folder}/{n}", dest))

    if not jobs:
        print(f"[DXAPP] [INFO] sample/vpr already complete "
              f"({len(DATABASE)} database + {len(QUERIES)} queries)")
    else:
        print(f"[DXAPP] [INFO] fetching {len(jobs)} image(s) from {REPO}@{REF}")
        failed: list[str] = []
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for job, fut in zip(jobs, [ex.submit(fetch_one, j) for j in jobs]):
                try:
                    dest, size, digest = fut.result()
                    print(f"  [OK] {dest.relative_to(out)}  {size:>7,d} B  sha256:{digest}")
                except (urllib.error.URLError, ValueError, OSError) as e:
                    failed.append(f"{job[0]}: {e}")
        if failed:
            print("[DXAPP] [ERROR] fetch failed:", file=sys.stderr)
            for f in failed:
                print(f"    {f}", file=sys.stderr)
            return 1

    # Provenance travels with the images: they are third-party MIT content, and the
    # example's correctness claim depends on which set produced a gallery.
    try:
        lic = get(LICENSE_URL).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        lic = "(LICENSE could not be fetched; see https://github.com/%s)" % REPO
    (out / "SOURCE.md").write_text(
        f"""# sample/vpr -- Visual Place Recognition toy dataset

Fetched by `scripts/fetch_vpr_toy_dataset.py` from
<https://github.com/{REPO}/tree/{REF}/toy_dataset>.

* `database/` -- {len(DATABASE)} images, the places to be recognised.
* `queries/`  -- {len(QUERIES)} images, the same places from a different viewpoint.

That repo is the standardised VPR evaluation codebase referenced by the EigenPlaces
README; this is the toy set it ships for a quick unlabelled run.

**No ground truth.** The filenames carry no `@utm_east@utm_north@` fields, so top-k
retrieval is demonstrable but Recall@k is not computable. For a geo-referenced set use
<https://github.com/gmberton/VPR-datasets-downloader> and point the gallery builder at
`datasets/<name>/images/test/database`.

## Upstream licence ({REPO})

```
{lic.strip()}
```
""", encoding="utf-8")

    db = sorted((out / "database").glob("*.jpg"))
    qs = sorted((out / "queries").glob("*.jpg"))
    print(f"[DXAPP] [INFO] sample/vpr ready: {len(db)} database, {len(qs)} queries "
          f"-> {out}")
    if len(db) != len(DATABASE) or len(qs) != len(QUERIES):
        print(f"[DXAPP] [ERROR] expected {len(DATABASE)} database and "
              f"{len(QUERIES)} queries", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
