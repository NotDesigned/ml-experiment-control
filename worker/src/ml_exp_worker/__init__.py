"""Standard-library execution programs and their sealed source resources."""
from pathlib import Path


def source_path(name: str) -> Path:
    """Locate one bundled execution program without importing server code."""
    return Path(__file__).with_name(name)
