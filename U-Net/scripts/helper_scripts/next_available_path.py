from pathlib import Path
from typing import Optional

def next_available_path(path: Path) -> Path:
    """
    Returns `path` if it does not exist, otherwise appends
    an incrementing integer suffix before the extension.

    Example:
        file.csv  -> file0.csv, file1.csv, ...
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 0
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def find_latest_path(directory: Path, name: str, ext: str = "csv") -> Optional[Path]:
    """
    Returns the highest-numbered file matching:
        name.ext, name_0.ext, name_1.ext, ...

    Returns None if no matching file exists.
    """
    latest = None
    idx = -1

    for p in directory.glob(f"{name}*.{ext}"):
        suffix = p.stem[len(name):]

        if suffix == "":
            i = 0
        elif suffix.startswith("_") and suffix[1:].isdigit():
            i = int(suffix[1:])
        else:
            continue

        if i > idx:
            idx = i
            latest = p

    return latest