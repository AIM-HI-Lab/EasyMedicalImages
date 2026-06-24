"""
Test.py - sanity checks for the RAS reorientation done in _mutate_image.

Run from EasyMedicalImages/Test/ (or pass the path as an argument):

    python Test.py "W:\\...\\your_image_folder"
    # or just `python Test.py` and paste the path when prompted

It builds a CTImagesLibrary instance (which loads + mutates the volumes) and then
checks the properties the reorientation fix is responsible for. Each check prints
PASS/FAIL; the process exits non-zero if anything fails.
"""

import os
import sys

import numpy as np
import nibabel as nib

_IMAGES_RUNNER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ImagesRunner",
)
sys.path.insert(0, _IMAGES_RUNNER)
from CTImagesLibrary import CTImagesLibrary as il  # noqa: E402  (after sys.path tweak)


def grid_center_world(shape, affine):
    """World (mm) coordinate of the geometric center of a voxel grid."""
    c = (np.array(shape[:3], float) - 1.0) / 2.0
    return (affine @ np.r_[c, 1.0])[:3]


def mask_centroid_world(mask, affine):
    """World (mm) centroid of the non-zero voxels of a mask."""
    idx = np.argwhere(np.asarray(mask) > 0)
    if idx.size == 0:
        return None
    ijk1 = np.c_[idx.astype(float), np.ones(len(idx))]
    return (affine @ ijk1.T).T[:, :3].mean(axis=0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else input("Path to image folder: ")
    inst = il(path_to_folder=path, verbose=False)

    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}" + (f"   ({detail})" if detail else ""))

    # voxel scale used as the tolerance for "did it move in physical space"
    vox_mm = float(np.linalg.norm(inst.raw_img_affine[:3, :3], axis=0).min())

    print("\nShapes")
    print(f"   raw img {inst.raw_img_data.shape} -> mutated img {inst.mutated_img_data.shape}")
    print(f"   raw seg {inst.raw_seg_data.shape} -> mutated seg {inst.mutated_seg_data.shape}")

    print("\n1) Orientation is RAS")
    img_codes = nib.aff2axcodes(inst.mutated_img_affine)
    seg_codes = nib.aff2axcodes(inst.mutated_seg_affine)
    check("mutated image affine is RAS", img_codes == ("R", "A", "S"), f"got {img_codes}")
    check("mutated seg affine is RAS", seg_codes == ("R", "A", "S"), f"got {seg_codes}")

    print("\n2) Image and seg share one grid (co-registered)")
    check("img/seg affines match", np.allclose(inst.mutated_img_affine, inst.mutated_seg_affine))
    check(
        "img/seg shapes match",
        inst.mutated_img_data.shape == inst.mutated_seg_data.shape,
        f"{inst.mutated_img_data.shape} vs {inst.mutated_seg_data.shape}",
    )

    print("\n3) Physical position preserved (resampling bookkeeping)")
    raw_c = grid_center_world(inst.raw_img_data.shape, inst.raw_img_affine)
    mut_c = grid_center_world(inst.mutated_img_data.shape, inst.mutated_img_affine)
    d_center = float(np.linalg.norm(raw_c - mut_c))
    check(
        "grid center fixed in world space",
        d_center < vox_mm,
        f"moved {d_center:.3f} mm (voxel ~{vox_mm:.3f} mm)",
    )

    print("\n4) Segmentation stays physically aligned (the reorientation test)")
    raw_t = mask_centroid_world(inst.raw_seg_data, inst.raw_seg_affine)
    mut_t = mask_centroid_world(inst.mutated_seg_data, inst.mutated_seg_affine)
    if raw_t is None or mut_t is None:
        check("segmentation is non-empty", False, "no non-zero voxels in seg")
    else:
        d_tumor = float(np.linalg.norm(raw_t - mut_t))
        check(
            "tumor centroid fixed in world space",
            d_tumor < vox_mm,
            f"moved {d_tumor:.3f} mm (voxel ~{vox_mm:.3f} mm)",
        )

    print("\n5) Label values intact")
    raw_labels = set(np.unique(inst.raw_seg_data).tolist())
    mut_labels = set(np.unique(inst.mutated_seg_data).tolist())
    check(
        "no labels invented by reorientation",
        mut_labels <= raw_labels,
        f"raw={sorted(raw_labels)[:6]}... mutated={sorted(mut_labels)[:6]}...",
    )

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed.")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()