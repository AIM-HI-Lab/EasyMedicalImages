"""
Companion to test_reorient.py (keep this file named lowercase 'conftest.py' so
pytest discovers it on every OS). It registers the --image-folder flag.

Place it in the same folder as test_reorient.py (EasyMedicalImages/Test/). Without it,
point test_reorient.py at your data via the IMAGES_FOLDER environment variable
instead, and don't pass --image-folder.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--image-folder",
        action="store",
        default=None,
        help="Path to the folder containing image.nii.gz and the segmentation.",
    )