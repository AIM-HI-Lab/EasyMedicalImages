import EasyMedicalImages as EMI
import numpy as np
import nibabel as nib

hl = EMI.HighLevel(
    path_to_folder=r"W:\Shared\LRI\Labs\AIM-HI-Lab\research-projects\honn\smaller-kidney-tumor-ct-sample-batch",
    thread_deadline_time=128.2,
)

original_pairs = hl.run_on_directory(return_affines=True)
print(len(original_pairs))
# return_affines=True -> pair is ((img, img_aff), (seg, seg_aff)); the array is [side][0]
print(len(original_pairs), original_pairs[0][0][0].shape, original_pairs[0][1][0].shape)

print(f"\n------------ affine check (run_on_directory) ------------\n")

img_data, img_aff = original_pairs[0][0]
seg_data, seg_aff = original_pairs[0][1]

print(f"img affine:\n{img_aff}")
print(f"seg affine:\n{seg_aff}")
print(f"shape:  img {img_aff.shape}   seg {seg_aff.shape}")
print(f"dtype:  img {img_aff.dtype}   seg {seg_aff.dtype}")

print("4x4:", img_aff.shape == (4, 4) and seg_aff.shape == (4, 4))
print("finite (no NaN/inf):", np.isfinite(img_aff).all() and np.isfinite(seg_aff).all())
print("bottom row [0,0,0,1]:",
      np.allclose(img_aff[3], [0, 0, 0, 1]) and np.allclose(seg_aff[3], [0, 0, 0, 1]))
print("direction block invertible:", abs(np.linalg.det(img_aff[:3, :3])) > 1e-6)

spacing = np.linalg.norm(img_aff[:3, :3], axis=0)  # voxel size per axis = column norms of the 3x3
print(f"voxel spacing (mm): {spacing}")
print("spacing > 0 & finite:", np.all(spacing > 0) and np.all(np.isfinite(spacing)))

# CTImagesLibrary reorients to RAS, so the mutated affine that came back over the pipe should be RAS
print("img axcodes:", nib.aff2axcodes(img_aff), " expected ('R', 'A', 'S')")
print("seg axcodes:", nib.aff2axcodes(seg_aff))

# same case -> image and seg live on the same voxel grid, so their affines should match
print(f"img vs seg affine max abs diff: {np.max(np.abs(img_aff - seg_aff))}")
print("img affine == seg affine:", np.allclose(img_aff, seg_aff))

print(f"\n--- sweeping all {len(original_pairs)} cases ---")
all_ok = True
for i, ((_, ia), (_, sa)) in enumerate(original_pairs):
    ok_ras = nib.aff2axcodes(ia) == ('R', 'A', 'S')
    ok_finite = np.isfinite(ia).all() and np.isfinite(sa).all()
    ok_match = np.allclose(ia, sa)
    if not (ok_ras and ok_finite and ok_match):
        all_ok = False
        print(f"case {i}: RAS={ok_ras} finite={ok_finite} img_aff==seg_aff={ok_match}")
print("all cases passed" if all_ok else "some cases flagged above")

print(f"\n------------ new method ------------\n")

print([m for m in dir(hl) if not m.startswith("__")])
# processed_run_on_directory returns bare (img, seg) even with return_affines=True
# (the flag is currently ignored on that path), so there are no affines to check here
pairs = hl.processed_run_on_directory(
    "apply_seg_crop_to_image", {"img_or_seg": "both", "pad": 5}, return_affines=True
)
print(len(pairs), pairs[0][0].shape, pairs[0][1].shape)

print(
    f"final results:\n"
    f"Original:\n{len(original_pairs), original_pairs[0][0][0].shape, original_pairs[0][1][0].shape}\n"
    f"New Method:\n{len(pairs), pairs[0][0].shape, pairs[0][1].shape}"
)

print(EMI.go_one_level_deep())