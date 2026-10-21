"""Download ATP match CSVs (2010-2026) from a pinned Hugging Face mirror of Jeff Sackmann's data.

Pinning the commit SHA means everyone downloads identical files. Standard library only (uses curl: python.org Python on macOS lacks SSL certificates).
Run from the project root:  python src/download_data.py
"""
import hashlib
import json
import subprocess
from pathlib import Path

REPO = "Aneeshers/tennis-sackmann-archive"
SHA = "8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993"  # dataset commit, last modified 2026-06-25
YEARS = range(2010, 2027)
EXTRA = ["UPSTREAM_README.md", "matches_data_dictionary.txt"]  # provenance + column docs
RAW = Path("data/raw")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    files = [f"atp_matches_{y}.csv" for y in YEARS] + EXTRA
    manifest = {"repo": REPO, "commit": SHA, "files": {}}
    for name in files:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{SHA}/atp/{name}"
        data = subprocess.run(["curl", "-fsSL", url], check=True, capture_output=True).stdout
        (RAW / name).write_bytes(data)
        manifest["files"][name] = hashlib.sha256(data).hexdigest()
        print(f"{name}: {len(data):,} bytes")
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
