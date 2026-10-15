from pathlib import Path


MOTIONS_DIR = Path(__file__).resolve().parent / "motions"


def resolve_motion_path(metadata_path: str | Path) -> Path:
    original_path = Path(metadata_path).expanduser()
    bundled_path = MOTIONS_DIR / original_path.name
    if bundled_path.is_file():
        return bundled_path
    if original_path.is_file():
        return original_path
    raise FileNotFoundError(
        f"Reference motion not found: checked {bundled_path} and {original_path}"
    )
