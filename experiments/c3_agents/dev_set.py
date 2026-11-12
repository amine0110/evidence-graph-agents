"""Build the C3 development set from the XD-Violence training split.

S3 may only be changed on these clips, never on the 799 test clips. The
training videos come as large zip files on the dataset's SharePoint; this
script reads a zip's index over HTTP range requests and downloads only the
chosen clips. The pick is stratified by first label with a fixed seed:
--normal normal clips and --per-class clips of each violent class (fewer if
the zip holds fewer). Training clips have video-level labels only (in the file
name), no frame labels.

  python dev_set.py --share-url <link to training zip 1-1004> --out data/XD-Violence/dev
  python dev_set.py --share-url <link to zip 3320-3954> --out data/XD-Violence/dev \
      --youtube-only --normal 20 --per-class 8

The test set is about half YouTube clips (names starting v=); zip 1-1004 holds
films only, so the second call adds YouTube clips. The list hash in
dev_set.json covers every clip in the folder.

Writes the clips, dev_set.json (names, sizes, SHA-256 of the list) and stops.
"""

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from http.cookiejar import CookieJar
from pathlib import Path

SEED = 2026
CLASSES = {"A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot", "B5": "abuse", "B6": "accident",
           "G": "explosion"}


class RangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, opener, url, size):
        self.opener, self.url, self.size, self.pos = opener, url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(8):  # SharePoint throttles with 429/503
            try:
                data = self.opener.open(req, timeout=300).read()
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                if attempt == 7:
                    raise
                time.sleep(min(30 * 2 ** attempt, 600))
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_share(share_url):
    """(opener with the share cookies, direct download URL, size) for a SharePoint share link."""
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    html = opener.open(share_url, timeout=120).read().decode("utf-8", "ignore")
    url = json.loads('"' + re.search(r'"\.downloadUrl"\s*:\s*"([^"]+)"', html).group(1) + '"')
    head = opener.open(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=120)
    size = int(head.headers["Content-Range"].split("/")[1])
    return opener, url, size


def first_label(name):
    m = re.search(r"label_([A-Z0-9]+)", name)
    return m.group(1) if m else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--share-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--normal", type=int, default=40)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--youtube-only", action="store_true", help="only clips whose name starts with v=")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    opener, url, size = open_share(args.share_url)
    archive = zipfile.ZipFile(io.BufferedReader(RangeFile(opener, url, size), buffer_size=8 << 20))
    members = [i for i in archive.infolist() if i.filename.endswith(".mp4") and first_label(Path(i.filename).name)
               and (not args.youtube_only or Path(i.filename).name.startswith("v="))]
    by_class = {}
    for info in members:
        by_class.setdefault(CLASSES.get(first_label(Path(info.filename).name), "other"), []).append(info)
    print({k: len(v) for k, v in sorted(by_class.items())}, "zip bytes", size)

    rng = random.Random(SEED)
    pick = []
    for name, infos in sorted(by_class.items()):
        if name == "other":
            continue
        infos = sorted(infos, key=lambda i: i.filename)
        pick += rng.sample(infos, min(args.normal if name == "normal" else args.per_class, len(infos)))
    for k, info in enumerate(sorted(pick, key=lambda i: i.filename), 1):
        target = out / Path(info.filename).name
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        with archive.open(info) as src, open(target.with_suffix(".part"), "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
        target.with_suffix(".part").rename(target)
        print(f"{k}/{len(pick)} {target.name} {info.file_size / 1e6:.1f} MB", flush=True)

    videos = sorted(out.glob("*.mp4"))
    lines = "".join(f"{v.name}\t{v.stat().st_size}\n" for v in videos)
    meta = {"name": "C3 dev set: stratified picks from XD-Violence training videos (films from 1-1004, "
                    "YouTube clips from 3320-3954)", "seed": SEED,
            "clips": len(videos),
            "classes": {c: sum(CLASSES.get(first_label(v.name)) == c for v in videos) for c in CLASSES.values()},
            "sha256": hashlib.sha256(lines.encode()).hexdigest(), "files": [v.name for v in videos]}
    (out / "dev_set.json").write_text(json.dumps(meta, indent=2) + "\n")
    print({k: meta[k] for k in ("clips", "classes", "sha256")})


if __name__ == "__main__":
    main()
