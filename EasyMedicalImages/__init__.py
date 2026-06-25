from . import ImagesRunner, DicomToNIFTI, ThreeDimensionalSubsectionOfLibrary
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("easy-medical-images")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"