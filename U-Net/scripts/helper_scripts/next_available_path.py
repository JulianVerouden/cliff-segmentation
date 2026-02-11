from pathlib import Path

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