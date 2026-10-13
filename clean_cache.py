import argparse
import os
from pathlib import Path
import shutil


def is_junction(path: Path) -> bool:
    return getattr(os.path, "isjunction", lambda _: False)(path)


def validate_directory(value: str) -> Path:
    root = Path(__file__).resolve().parent
    directory = Path(value)
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve(strict=True)

    if directory.parent != root:
        raise ValueError("Wrong cache path, outside PIRB directory")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    return directory


def clean(directory: Path) -> int:
    removed = 0
    for dataset in sorted(directory.iterdir()):
        if dataset.is_symlink() or is_junction(dataset):
            continue
        if not dataset.is_dir() or dataset.resolve() != dataset:
            continue
        for child in sorted(dataset.iterdir()):
            if child.name == "results" or not child.is_dir():
                continue
            if child.is_symlink():
                child.unlink()
            elif is_junction(child):
                os.rmdir(child)
            else:
                shutil.rmtree(child)
            print(f"Removed: {child}")
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="cache")
    args = parser.parse_args()
    try:
        directory = validate_directory(args.directory)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Number of directories removed: {clean(directory)}")


if __name__ == "__main__":
    main()
