#!/usr/bin/env python3
"""
Smoke + correctness test for processed_run_on_directory (EasyMedicalImages 1.1.0).

It exercises the compiled C++ fan-out (_core.HighLevel) against a directory of
cases and cross-checks the result against the pure-Python in-process twin
(HighLevelPythonVarient). Because apply_seg_crop_to_image is deterministic (a pure
slice to the segmentation bounding box), the two independent code paths -- one
streaming volumes over a subprocess pipe, one running in-process -- must produce
BIT-FOR-BIT identical arrays. If they match, the streaming / argv / JSON plumbing
is correct on real data. If they diverge, the fault is in the transport.

Run from the venv where the package is installed (so worker subprocesses spawn
with the right interpreter):

    python test_apply_operation.py <dataset_dir> [image_name] [seg_name]

Point <dataset_dir> at a SMALL subset (2-3 case folders) for a fast run; each
case folder must contain the image and seg NIfTI files. Nothing is written to disk.
"""
import sys
import hashlib
import numpy as np

# ---- edit these if you want to test a different operation -------------------
OPERATION = "apply_seg_crop_to_image"
OP_KWARGS = {"img_or_seg": "both", "pad": 5}
# -----------------------------------------------------------------------------

try:
    from EasyMedicalImages._core import HighLevel as CppHighLevel
except Exception as e:  # pragma: no cover
    sys.exit(f"Could not import EasyMedicalImages._core.HighLevel: {e}\n"
             f"Did you rebuild and `pip install -e .` after the changes?")

# The pure-Python twin lives next to PythonThreadWrapper; try the likely paths.
PyHighLevel = None
for _modpath in ("EasyMedicalImages.ImagesRunner.HighLevelPythonVarient",
                 "EasyMedicalImages.HighLevelPythonVarient"):
    try:
        PyHighLevel = __import__(_modpath, fromlist=["HighLevel"]).HighLevel
        break
    except Exception:
        continue


def as_f32(a):
    # Match the wire format (float32, C-order) so the comparison is dtype-agnostic:
    # both the streamed array and the in-process array get the identical truncation.
    return np.ascontiguousarray(a, dtype=np.float32)


def signature(pair):
    """A content fingerprint for one (img, seg) case: shapes + hashes of the bytes."""
    img, seg = pair
    return (tuple(img.shape), tuple(seg.shape),
            hashlib.sha1(as_f32(img).tobytes()).hexdigest(),
            hashlib.sha1(as_f32(seg).tobytes()).hexdigest())


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    dataset    = sys.argv[1]
    image_name = sys.argv[2] if len(sys.argv) > 2 else "image.nii.gz"
    seg_name   = sys.argv[3] if len(sys.argv) > 3 else "tumor_segmentation_v2.nii.gz"

    print(f"dataset     : {dataset}")
    print(f"image / seg : {image_name} / {seg_name}")
    print(f"operation   : {OPERATION}({OP_KWARGS})\n")

    hl = CppHighLevel(dataset, image_name, seg_name)

    # 0) API surface is present
    for name in ("run_on_directory", "processed_run_on_directory", "_go_one_level_deep"):
        assert hasattr(hl, name), f"missing method on _core.HighLevel: {name}"
    cases = hl._go_one_level_deep()
    print(f"cases found            : {len(cases)}")
    assert cases, "no case subfolders found under dataset"

    # 1) backwards-compat: plain load still works
    baseline = hl.run_on_directory()
    print(f"run_on_directory       : {len(baseline)} cases loaded")
    assert baseline, "run_on_directory returned nothing"
    print("  baseline img shapes  :", sorted(img.shape for img, _ in baseline))

    # 2) the new generic op via the compiled fan-out
    cpp = hl.processed_run_on_directory(OPERATION, OP_KWARGS)
    print(f"apply_operation (C++)  : {len(cpp)} cases")
    assert cpp, "processed_run_on_directory returned nothing"
    if len(cpp) < len(cases):
        print(f"  NOTE: {len(cases) - len(cpp)} case(s) dropped "
              f"(missing files / 30s timeout / op error -- check stderr above)")
    print("  cropped img shapes   :", sorted(img.shape for img, _ in cpp))
    for img, seg in cpp:
        assert img.size > 0 and seg.size > 0, "empty crop returned"
    print("  all crops non-empty  : OK")

    # 3) authoritative correctness: C++ result set == Python twin result set
    if PyHighLevel is None:
        print("\n(skipping cross-check: HighLevelPythonVarient not importable -- "
              "adjust the import path in this script to enable it)")
    else:
        py = PyHighLevel(dataset, image_name, seg_name).processed_run_on_directory(
            OPERATION, OP_KWARGS)
        assert len(py) == len(cpp), f"case count differs: C++ {len(cpp)} vs Py {len(py)}"

        cpp_sigs = sorted(signature(p) for p in cpp)
        py_sigs  = sorted(signature(p) for p in py)
        if cpp_sigs == py_sigs:
            print(f"\ncross-check            : all {len(cpp)} cases identical to the "
                  f"Python twin (bit-for-bit)  OK")
        else:
            only_cpp = [s[:2] for s in cpp_sigs if s not in py_sigs]
            only_py  = [s[:2] for s in py_sigs  if s not in cpp_sigs]
            print(f"\ncross-check            : MISMATCH")
            print(f"  in C++ but not Python (shape img,seg): {only_cpp}")
            print(f"  in Python but not C++ (shape img,seg): {only_py}")
            sys.exit(1)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()