# EasyMedicalImages
The easy way to manipulate medical images

#### ---------------------------------------------------------------------------

This is a library that focuses on taking CT scans (as NIFTI Images) and transforming them into arrays in python; giving the user easy ways to manipulate them as well. This can easily be used for downstream usage of CT images within a Python script.

## Installation
Requires Python 3.12 or newer.

```bash
pip3 install easy-medical-images
```

The high-level batch loader (`HighLevel`) is a compiled C++ extension. The published wheels include it, so a normal `pip install` is all you need.

Python 3.14 and 3.14t are supported on all platforms. Python 3.15 is also supported on all platforms, but python 3.15t is not supported on Windows at this time due to a required dependency not supporting it yet.

## Quick start: load a whole directory

```python
import EasyMedicalImages as EMI

path_to_folder_of_ct_scans = input("Input path to CT scans: ")
instance = EMI.HighLevel(path_to_folder=path_to_folder_of_ct_scans)

img_seg_pairs_mega_list = instance.run_on_directory()  # list of (image, seg) array pairs
'''
img_seg_pairs_mega_list=[(img1, seg1),(img2, seg2)...]
'''

# Each entry is one case
img, seg = img_seg_pairs_mega_list[0]
print(f"{len(img_seg_pairs_mega_list)} cases loaded")
print(f"First image shape: {img.shape}, seg shape: {seg.shape}")
```

Every entry in `img_seg_pairs_mega_list` is a tuple of two NumPy arrays: the image and its segmentation. Both are already resampled to isotropic spacing and reoriented to RAS, so they are ready to use.

To access all of the images or all of the segs while keeping order, one can simply do:
```python
all_images=[images for images,_ in img_seg_pairs_mega_list]
all_segmentations=[segmentations for _, segmentations in img_seg_pairs_mega_list]

# Both of the below will print the entire array
print(f"Images: {all_images}")
print(f"Segmentation Masks: {all_segmentations}")

# This just shows the shape, easier to comprehend.
print([(img.shape, seg.shape) for img, seg in img_seg_pairs_mega_list])
```

## Working with a single case
```python
from EasyMedicalImages import CTImagesLibrary

case = CTImagesLibrary(path_to_folder="path/to/one/case")

# The voxels are automatically processed upon calling of the instance, unlike the batch version.
print(case.mutated_img_data.shape)
print(case.mutated_seg_data.shape)

# Display three orthogonal slices (slice numbers are 1-based, ordered X, Y, Z)
case.display_slice(slice_number=(175, 279, 57), img_or_seg="img")

# Get those same slices back as arrays
slice_x, slice_y, slice_z = case.return_slice(slice_number=(175, 279, 57), img_or_seg="img")

# Crop the whole volume down to the segmented region (modifies the mutated arrays in place)
case.apply_seg_crop_to_image(img_or_seg="both")
print(case.mutated_img_data.shape)  # smaller now
```

As stated earlier, building `CTImagesLibrary` automatically: changes the working directory into `path_to_folder`, loads the image and segmentation, and builds isotropic, RAS-reoriented copies. After construction, both the raw and processed arrays are available as attributes via `your_instancename.raw_img_data`, or for seg `your_instancename.raw_seg_data`.

## Expected data layout
Each case lives in its own folder containing an image and a segmentation. By default these are named `image.nii.gz` and `tumor_segmentation_v2.nii.gz`, but you can override both names (see `image_path` and `seg_path` below).

```
parent_folder/
├── case_0001/
│   ├── image.nii.gz
│   └── tumor_segmentation_v2.nii.gz
├── case_0002/
│   ├── image.nii.gz
│   └── tumor_segmentation_v2.nii.gz
└── ...
```

#### Remeber to put the proper root for the input directory path! For example, if your files live on a cloud.

`CTImagesLibrary` operates on a single case folder. `HighLevel` operates on a parent folder and loads every case folder inside it.

## More detail on batch loading vs single image
Generally, this library assumes the user wants a batch of something rather than a specific image. This is reflected in how you first load in your specific instance, and then can add modifiers how you wish.

There are two main sub libraries:

1. `HighLevel`: point it at a parent directory and it loads every case in parallel (after the method 'run_on_directory' is called on the instance). This will return a list of `(image, seg)` array pairs. Use this to pull a whole dataset into memory quickly.
2. `CTImagesLibrary`: load one case and get the full manipulation toolkit (slicing, cropping, reorientation, display). Use this when you want to work with an individual scan.
   See below for examples.

## Affines and voxel spacing
The 4x4 affine is what maps voxel indices to physical (mm) space, so it is the key to any real-world measurement: voxel spacing, distances, and volumes. Both sublibraries can hand you the affines; from there the spacing is a one-liner.

### Getting affines from `HighLevel`
Pass `return_affines=True` to `run_on_directory` and every case comes back with its affines attached. Each entry becomes `((image, image_affine), (seg, seg_affine))`:

```python
import EasyMedicalImages as EMI

instance = EMI.HighLevel(path_to_folder="path/to/parent")
pairs = instance.run_on_directory(return_affines=True)

# Unpack the first case
(img, img_affine), (seg, seg_affine) = pairs[0]
print(img.shape, img_affine.shape)   # e.g. (512, 512, 340) (4, 4)
```

To pull all the images or all the affines while keeping order:
```python
all_images  = [img        for (img, _), _        in pairs]
all_affines = [img_affine for (_, img_affine), _ in pairs]
```

### Getting affines from `CTImagesLibrary`
A single-case instance already exposes them as attributes, so no flag is needed:

```python
from EasyMedicalImages import CTImagesLibrary

case = CTImagesLibrary(path_to_folder="path/to/one/case")

img_affine = case.mutated_img_affine   # processed (isotropic, RAS) grid
seg_affine = case.mutated_seg_affine
# case.raw_img_affine / case.raw_seg_affine hold the originals, before resampling
```

### Voxel spacing and volume
The voxel spacing along each axis is the length of each column of the affine's top-left 3x3 block, which is exactly how the library measures spacing internally when it resamples:

```python
import numpy as np

spacing = np.linalg.norm(img_affine[:3, :3], axis=0)   # (sx, sy, sz) in mm
print(f"voxel spacing (mm): {spacing}")
```

Because the processed (`mutated_*`) arrays are isotropic, those three numbers are equal. The physical volume of a single voxel is the product of the spacings, or equivalently the absolute determinant of that same 3x3 block (the determinant form also stays correct for anisotropic or rotated grids, such as the raw ones):

```python
voxel_volume_mm3 = abs(np.linalg.det(img_affine[:3, :3]))
print(f"voxel volume: {voxel_volume_mm3:.4f} mm^3")
```

That is all you need for physical measurements. For example, the volume of a segmented region is its non-background voxel count times the voxel volume:

```python
n_voxels = np.count_nonzero(seg)          # every voxel where seg != 0
volume_mm3 = n_voxels * voxel_volume_mm3
print(f"segmented volume: {volume_mm3:.1f} mm^3  ({volume_mm3 / 1000:.2f} mL)")
```

## Further detail
This section goes one level deeper on each of the two sublibraries: how you construct it, and what you can call on it once you have an instance. Both are reached through the top-level package (`import EasyMedicalImages as EMI`, then `EMI.HighLevel` or `EMI.CTImagesLibrary`).

### HighLevel
The batch sublibrary. Point it at a parent folder, call `run_on_directory`, and get every case back as `(image, seg)` array pairs. This is the one to reach for when you want a whole dataset in memory at once.

Construct it (default parameters):
```python
HighLevel(
    path_to_folder,
    image_path="image.nii.gz",
    seg_path="tumor_segmentation_v2.nii.gz",
    verbose=False,
    thread_deadline_time=244.4,
    verbose_workers=True
)
```

- `path_to_folder`: parent folder whose immediate subfolders are individual cases.
- `image_path`, `seg_path`: file names to look for inside each case folder.
- `verbose`: passed through to each worker for extra logging.
- `thread_deadline_time`: per-case wall-clock budget in seconds. A worker that runs longer is killed and that case is skipped. This is to prevent a thread, which is just stalling or trying to load a not image, from stoping the whole process.
- `verbose_workers`: print per-worker progress (which case each worker picked up, load confirmations, and the final count).

Methods:

- **`run_on_directory(directory_path="", return_affines=False)`**
  Loads every case in the directory and returns a list of `(image, seg)` tuples, one per case, in the same order as the folders on disk. Both elements are NumPy arrays, already resampled to isotropic spacing and reoriented to RAS. If `directory_path` is given it overrides `path_to_folder` for that one call. `return_affines` will return the affines with the images, making the resulting format `((image, image_affine), (seg, seg_affine))`. This allows easy calculation of grid spacing and 3D object information (see [Affines and voxel spacing](#affines-and-voxel-spacing) above). Cases that fail to load or exceed `thread_deadline_time` are dropped, so the returned list can be shorter than the number of folders in theory (so don't set this too low).

Loading runs in parallel: one worker subprocess per case. On Linux it respects the CPU affinity set by schedulers such as SLURM, so it behaves correctly on a cluster node. On this note, becuase some clusters use cores intead of threads, it has safe logic build in for this. If it detects less than 3 threads or 3 cores, it will simply use half of what it finds (so likely using 1 thread or 1 core, unless something very strange is going on hardware wise). Otherwise, it will use the max number of threads/cores it finds minus 1.

- **`how_many_threads_available()`**
  Returns the number of hardware threads detected (the figure the worker pool is sized from). Honors SLURM cpuset limits on Linux and falls back sensibly when the OS reports nothing.

- **`processed_run_on_directory()`**
  A special version of run_on_directory that applies a method to all items loaded. Presently, only `apply_seg_crop_to_image` is allowed.
  *Example:*
   ```python
   hl = HighLevel(r"W:\...\dataset")           # or however you construct it
   
   pairs = hl.run_on_directory()               # unchanged plain load
   
   pairs = hl.processed_run_on_directory(     # crop every case to the seg bbox
   "apply_seg_crop_to_image",
   {"img_or_seg": "both", "pad": 5},
   )
   # -> list[tuple[np.ndarray, np.ndarray]]
   ```

- **`go_one_level_deep()`**
  A method which will display all of the items in a directory, be them folders, files, or both. If no perameter is specified for directory_path it will simply use the root given at HighLevels class construction.
  *Example:*
   ```python
   hl = HighLevel(root, "image.nii.gz", "tumor_segmentation_v2.nii.gz")
   hl.go_one_level_deep()                          # -> list[str] of subfolder paths
   hl.go_one_level_deep(options="files")           # regular files only
   hl.go_one_level_deep(options="all")             # everything
   hl.go_one_level_deep(directory_path=other_dir)  # scan a different dir this call onl
   ```

### CTImagesLibrary
The single-case sublibrary. Construct it on one case folder and the data is processed and waiting on the instance; from there you slice, crop, reorient, and display. This is the toolkit `HighLevel` runs under the hood for each case, so the arrays you get out of either path match.

Construct it (default parameters):
```python
CTImagesLibrary(
    path_to_folder,
    image_path="image.nii.gz",
    seg_path="tumor_segmentation_v2.nii.gz",
    verbose=True,
    return_affines=False,
)
```

The image and segmentation are loaded and processed during construction, so the attributes below are ready as soon as the instance exists.

Attributes on the instance:
- `raw_img_data`, `raw_seg_data`: the original voxel arrays as loaded.
- `raw_img_affine`, `raw_seg_affine`: the original 4x4 affine matrices.
- `mutated_img_data`, `mutated_seg_data`: isotropic, RAS-reoriented copies. These are usually what you want for machine learning.
- `mutated_img_affine`, `mutated_seg_affine`: affines describing the processed grid.

(See [Affines and voxel spacing](#affines-and-voxel-spacing) for turning any of these affines into voxel spacing or a physical volume.)

methods:

- **`display_slice(slice_number=(), cmap="viridis", view_axes=("X", "Y", "Z"), load_mutated=True, crop=False, pad=5, img_or_seg="img")`**
  Shows one slice per requested axis side by side with matplotlib. `slice_number` is a 1-based `(X, Y, Z)` tuple and must contain exactly three values. Set `load_mutated=False` to view the original (non-isotropic) volume, `crop=True` to trim each slice to its own non-background region (with `pad` voxels of margin), and `img_or_seg="seg"` to view the segmentation instead.


- **`return_slice(slice_number=(), view_axes=("X", "Y", "Z"), load_mutated=True, one_list=False, crop=False, pad=5, img_or_seg="img")`**
  Returns slices as arrays instead of displaying them. With the default three axes it returns `(slice_x, slice_y, slice_z)`; set `one_list=True` to get them as a single array of three instead (convenient for batching). Request a single axis (for example `view_axes=("Z",)`) to get just that one slice back. The arrays match exactly what `display_slice` shows.


- **`apply_seg_crop_to_image(pad=5, return_data=False, img_or_seg="both")`**
  Crops the full 3D volume in place to the bounding box of the segmentation, keeping `pad` voxels of margin. This overwrites `mutated_img_data` and `mutated_seg_data` (and their affines) with the cropped versions. Use `img_or_seg` to crop only `"img"`, only `"seg"`, or `"both"` (the default). If the segmentation is empty, nothing is cropped. Set `return_data=True` to also return the cropped arrays.


- **`apply_seg_crop_to_slice(slice_number=(), one_list=False, pad=5, img_or_seg="img")`**
  The 2D version: returns the X, Y, and Z slices cropped to the segmentation's footprint in each plane. Unlike the 3D version, this runs on the image or the segmentation, not both at once (`img_or_seg` is `"img"` by default). `slice_number` is the same 1-based `(X, Y, Z)` tuple.


## Notes on orientation
The processed arrays (`mutated_img_data` and friends) are in proper RAS orientation. The slices produced by `display_slice` and `return_slice` add a 90-degree flip on the Z axis so they read naturally in matplotlib (similar to how ITK displays them). In short: use the instance arrays when you need correct anatomical orientation, and use `return_slice` when you want something that looks right on screen.

PyPI link: [PyPI](https://pypi.org/project/easy-medical-images/)

GitHub repo: [GitHub](https://github.com/AIM-HI-Lab/EasyMedicalImages)