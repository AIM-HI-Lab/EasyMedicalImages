"""
Pytest checks for the RAS reorientation done in CTImagesLibrary._mutate_image.

Layout this expects:
    EasyMedicalImages/
        ImagesRunner/
            CTImagesLibrary.py    <- the library
        Test/
            conftest.py           <- adds the --image-folder flag (keep it lowercase)
            test_reorient.py      <- this file (keep it lowercase, no spaces)

This file lives in EasyMedicalImages/Test/ while CTImagesLibrary.py lives in
ImagesRunner/. The sys.path line below makes the import resolve no matter
where you launch pytest from.

Run it (from the EasyMedicalImages project root, or from inside Test/):

  # with the --image-folder flag (needs conftest.py beside this file):
  python -B -m pytest "Test/test_reorient.py" --image-folder "W:\\...\\your_folder" -p no:cacheprovider

  # or from inside Test/:
  python -B -m pytest test_reorient.py --image-folder "W:\\...\\your_folder" -p no:cacheprovider

  # or via an environment variable instead of the flag (no conftest needed):
  #   PowerShell:  $env:IMAGES_FOLDER = "W:\\...\\your_folder"
  #   cmd:         set IMAGES_FOLDER=W:\\...\\your_folder
  #   bash:        export IMAGES_FOLDER="/path/to/your_folder"

The instance (load + isotropic resample + RAS reorient) is built ONCE per
session and shared across every test. Nothing here writes to your data; the
-B and -p no:cacheprovider flags also stop pytest/Python from writing their own
bytecode and cache files.
"""

import os
import sys

import numpy as np
import nibabel as nib
import pytest

# CTImagesLibrary.py lives in ImagesRunner/; make it importable from Test/.
_IMAGES_RUNNER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ImagesRunner",
)
sys.path.insert(0, _IMAGES_RUNNER)
from CTImagesLibrary import CTImagesLibrary as il  # noqa: E402  (after sys.path tweak)


# --- helpers -------------------------------------------------------------

def grid_center_world(shape, affine):
    """World (mm) coordinate of the geometric center of a voxel grid."""
    c = (np.array(shape[:3], float) - 1.0) / 2.0
    return (affine @ np.r_[c, 1.0])[:3]


def mask_centroid_world(mask, affine):
    """World (mm) centroid of the non-zero voxels of a mask, or None if empty."""
    idx = np.argwhere(np.asarray(mask) > 0)
    if idx.size == 0:
        return None
    ijk1 = np.c_[idx.astype(float), np.ones(len(idx))]
    return (affine @ ijk1.T).T[:, :3].mean(axis=0)


def _resolve_folder(request):
    """Image folder from --image-folder if registered, else $IMAGES_FOLDER."""
    folder = None
    try:
        folder = request.config.getoption("--image-folder")
    except (ValueError, KeyError):
        folder = None  # option not registered (conftest.py not present)
    return folder or os.environ.get("IMAGES_FOLDER")


# --- fixtures ------------------------------------------------------------

@pytest.fixture(scope="session")
def instance(request):
    """Build the CTImagesLibrary instance once for the whole test session."""
    folder = _resolve_folder(request)
    if not folder:
        pytest.skip(
            'No image folder given. Pass --image-folder "PATH" (with conftest.py) '
            "or set the IMAGES_FOLDER environment variable."
        )
    return il(path_to_folder=folder, verbose=False)


@pytest.fixture(scope="session")
def vox_mm(instance):
    """Finest raw voxel spacing (mm): tolerance for physical-position checks."""
    return float(np.linalg.norm(instance.raw_img_affine[:3, :3], axis=0).min())


# --- orientation is RAS --------------------------------------------------

def test_image_affine_is_ras(instance):
    assert nib.aff2axcodes(instance.mutated_img_affine) == ("R", "A", "S")


def test_seg_affine_is_ras(instance):
    assert nib.aff2axcodes(instance.mutated_seg_affine) == ("R", "A", "S")


# --- image and seg share one grid (still co-registered) ------------------

def test_img_seg_affines_match(instance):
    assert np.allclose(instance.mutated_img_affine, instance.mutated_seg_affine)


def test_img_seg_shapes_match(instance):
    assert instance.mutated_img_data.shape == instance.mutated_seg_data.shape


# --- physical position preserved -----------------------------------------

def test_grid_center_preserved(instance, vox_mm):
    raw = grid_center_world(instance.raw_img_data.shape, instance.raw_img_affine)
    mut = grid_center_world(instance.mutated_img_data.shape, instance.mutated_img_affine)
    drift = float(np.linalg.norm(raw - mut))
    assert drift < vox_mm, f"grid center moved {drift:.3f} mm (voxel ~{vox_mm:.3f} mm)"


def test_segmentation_nonempty(instance):
    assert np.any(instance.raw_seg_data > 0), "raw segmentation has no non-zero voxels"


def test_tumor_centroid_preserved(instance, vox_mm):
    raw = mask_centroid_world(instance.raw_seg_data, instance.raw_seg_affine)
    mut = mask_centroid_world(instance.mutated_seg_data, instance.mutated_seg_affine)
    assert raw is not None and mut is not None, "segmentation is empty"
    drift = float(np.linalg.norm(raw - mut))
    assert drift < vox_mm, f"tumor centroid moved {drift:.3f} mm (voxel ~{vox_mm:.3f} mm)"


# --- label values intact -------------------------------------------------

def test_labels_preserved(instance):
    raw_labels = set(np.unique(instance.raw_seg_data).tolist())
    mut_labels = set(np.unique(instance.mutated_seg_data).tolist())
    new = sorted(mut_labels - raw_labels)
    assert mut_labels <= raw_labels, f"reorientation introduced new label values: {new}"