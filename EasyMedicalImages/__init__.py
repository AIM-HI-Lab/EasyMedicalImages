import os
from importlib.metadata import version, PackageNotFoundError

from .ImagesRunner.CTImagesLibrary import CTImagesLibrary
from .ImagesRunner.MRIImagesLibrary import MRIImagesLibrary

try:
    __version__ = version("easy-medical-images")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "CTImagesLibrary",
    "MRIImagesLibrary",
    "HighLevel",
    "go_one_level_deep",
    "__version__",
]


def go_one_level_deep(debug=False, options="folders", directory_path=""):
    """One-level (non-recursive) listing of directory_path's children.

    Module-level convenience mirroring HighLevel.go_one_level_deep, but without
    building the _core extension or constructing a HighLevel. Same parameter
    order and defaults as the method, except an empty directory_path means the
    current working directory: the instance method falls back to the HighLevel's
    path_to_folder there, and a free function has none. options is one of
    "folders", "files", or "all". Returns a list of full entry paths.
    """
    options = options.lower()
    if options not in ("folders", "files", "all"):
        raise ValueError(f"Invalid options: {options!r}")
    root = directory_path or "."
    paths = []
    with os.scandir(root) as entries:
        for entry in entries:
            if options == "folders" and not entry.is_dir():
                continue
            if options == "files" and not entry.is_file():
                continue
            if debug:
                print(entry.path)
            paths.append(entry.path)
    return paths


# HighLevel lives in the compiled _core. Keep it lazy so importing the package
# does not require the extension to be built.
def __getattr__(name):
    if name == "HighLevel":
        from ._core import HighLevel
        return HighLevel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")