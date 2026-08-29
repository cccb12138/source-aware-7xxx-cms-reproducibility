"""Generate a SHA-256 manifest for repository files."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path != OUTPUT and ".git" not in path.parts
    )
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Relative_Path", "Bytes", "SHA256"])
        for path in files:
            writer.writerow([path.relative_to(ROOT).as_posix(), path.stat().st_size, digest(path)])
    print(f"Wrote {len(files)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
