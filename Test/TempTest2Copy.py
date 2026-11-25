"""
Smoke test for EasyMedicalImages.

Proves the two sublibraries agree: HighLevel loads a whole directory of cases in
parallel, CTImagesLibrary loads one case on its own, and for the same case they
should produce the same image and segmentation arrays.

Run it on a parent folder of case folders, then point it at ONE of those case
folders when prompted.
"""

import numpy as np
import EasyMedicalImages as EMI
import time as tym

print(f"Using EasyMedicalImages {EMI.__version__} from {EMI.__file__}")


def same_volume(a, b):
    # HighLevel returns float32; CTImagesLibrary keeps float64. Compare at the
    # lower precision so this is an apples-to-apples check, not f32 vs f64
    # (np.array_equal on mixed dtypes reports False even when the values match).
    return np.array_equal(a.astype(np.float32), b.astype(np.float32))


# 1) Batch load the whole directory.
parent = input("Path to the FOLDER OF cases: ").strip()
start=tym.time()
batch = EMI.HighLevel(path_to_folder=parent, verbose=True, verbose_workers=True, thread_deadline_time=180).run_on_directory()
print(f"\n==== RESULT ====")
print(f"Total time: {tym.time() - start:.2f} s")
print(f"Batch type: {type(batch)}")
# print(f"Batch: {batch}")

print(f"\nHighLevel loaded {len(batch)} case(s).")
# Each entry is a TUPLE of (image, seg), not a single array.
img0, seg0 = batch[0]
print(f"Each entry is a {type(batch[0]).__name__}: ({img0.dtype} image, {seg0.dtype} seg)")

# 2) Load ONE case on its own. It must be one of the folders inside `parent`.
case = input("\nPath to ONE of those case folders: ").strip()
ct = EMI.CTImagesLibrary(path_to_folder=case, verbose=True)
single_img = ct.mutated_img_data   # the array, not the object
single_seg = ct.mutated_seg_data
print(f"\nCTImagesLibrary loaded image {single_img.shape} {single_img.dtype}")

# 3) Batch order is not guaranteed to match folder order, so search the batch
#    for the case we loaded singly and confirm image AND seg match.
hit = None
for i, (img, seg) in enumerate(batch):
    if img.shape == single_img.shape and same_volume(img, single_img):
        hit = (i, same_volume(seg, single_seg))
        break

print("\n==== RESULT ====")
if hit is None:
    print("FAIL: that case was not found in the batch output.")
    print("      Is the single-case path actually one of the folders inside the parent?")
else:
    i, seg_ok = hit
    print(f"PASS: single case matches batch entry [{i}] (image identical).")
    print(f"      segmentation also matches: {seg_ok}")