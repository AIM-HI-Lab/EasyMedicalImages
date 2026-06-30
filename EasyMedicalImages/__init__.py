from importlib.metadata import version, PackageNotFoundError

from .ImagesRunner.CTImagesLibrary import CTImagesLibrary
from .ImagesRunner.MRIImagesLibrary import MRIImagesLibrary

try:
    __version__ = version("easy-medical-images")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["CTImagesLibrary", "MRIImagesLibrary", "HighLevel", "__version__"]

# HighLevel lives in the compiled _core. Keep it lazy so importing the package
# does not require the extension to be built.
def __getattr__(name):
    if name == "HighLevel":
        from ._core import HighLevel
        return HighLevel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")