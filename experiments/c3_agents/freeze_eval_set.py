"""Freeze the C3 evaluation set: every clip of the XD-Violence test split.

Writes eval_set.json with the clip count, the class counts and a SHA-256 over
the sorted lines "<file name>\t<size in bytes>". Rerun with --check to confirm
that a copy of the videos (on the Mac or on the cluster) is the frozen set.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def describe(videos_root):
    videos = sorted(Path(videos_root).glob("*.mp4"))
    lines = "".join(f"{v.name}\t{v.stat().st_size}\n" for v in videos)
    classes = Counter("normal" if re.search(r"label_A\.mp4$", v.name) else "violent" for v in videos)
    return {"name": "XD-Violence test split, all clips", "clips": len(videos), "classes": dict(classes),
            "hash": "sha256 of sorted '<file>\\t<bytes>' lines",
            "sha256": hashlib.sha256(lines.encode()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-root", required=True)
    parser.add_argument("--out", default=str(Path(__file__).parent / "frozen" / "eval_set.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    found = describe(args.videos_root)
    if args.check:
        frozen = json.loads(Path(args.out).read_text())
        ok = frozen["sha256"] == found["sha256"]
        print("matches the frozen set" if ok else f"DIFFERS: {found['clips']} clips, {found['sha256']}")
        raise SystemExit(0 if ok else 1)
    Path(args.out).write_text(json.dumps(found, indent=2) + "\n")
    print(found)


if __name__ == "__main__":
    main()
