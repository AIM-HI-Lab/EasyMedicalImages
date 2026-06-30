from . import ImagesRunner, DicomToNIFTI, ThreeDimensionalSubsectionOfLibrary
from importlib.metadata import version, PackageNotFoundError
# from ._core import HighLevel

try:
    __version__ = version("easy-medical-images")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

def __getattr__(name):
    if name == "HighLevel":
        from ._core import HighLevel
        return HighLevel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# def