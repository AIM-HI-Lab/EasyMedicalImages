import os
import nibabel as nib
from nibabel import orientations as nio
from nibabel.affines import rescale_affine
import numpy as np
from pathlib import Path
import platform
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from nilearn import plotting
from scipy.ndimage import zoom

class CTImagesLibrary:
    """A library for exploring individual or a series of CT and NIFTI images. Can put a CT and NIFTI images, 
    as a series of pixels, into array for external use. Is designed to be used with a (image, seg, folder_path) input group.
    By default the images are named image.nii.gz and tumor_segmentation_v2.nii.gz. These can be changed by changing the image_path and seg_path arguments.

    The only mandatory argument is the path to the folder containing the images and segmentations.

    Nothing in this library will (or should ever) save. Everything is designed to be loaded, manipulated, or displayed to another program.
    """

    def __init__(self, path_to_folder, image_path="image.nii.gz", seg_path="tumor_segmentation_v2.nii.gz", verbose=True):
        self.raw_image_path = image_path
        self.raw_seg_path = seg_path
        self.path_to_folder = path_to_folder
        self.raw_img_data = None
        self.raw_seg_data = None
        self.raw_img_affine = None
        self.raw_seg_affine = None
        self.verbose = verbose
        self.mutated_img_data = None
        self.mutated_seg_data = None
        self.mutated_img_affine = None
        self.mutated_seg_affine = None
        self._RAS_AXCODES = ("R", "A", "S")

        self._start_up()

        self.raw_img_data, self.raw_seg_data, self.raw_img_affine, self.raw_seg_affine = self.load_image(affine=True)
        self.mutated_img_data, self.mutated_seg_data, self.mutated_img_affine, self.mutated_seg_affine = self._mutate_image(
            self.raw_img_data, self.raw_seg_data
        )

    def _start_up(self):
        """Loads in all of preprocessed functions needed."""
        self.jump_to_folder(self.path_to_folder)

    def load_image(self, affine=False):
        """Load a NIfTI image from the given path, return a 3D image and a 3D segmentation.
        Returns the raw image and segmentation data, and the affine matrix.
        """
        raw_img = nib.load(self.raw_image_path)
        raw_seg = nib.load(self.raw_seg_path)

        self.raw_img_data = raw_img.get_fdata()
        self.raw_seg_data = raw_seg.get_fdata()
        
        if affine:
            self.raw_img_affine = raw_img.affine
            self.raw_seg_affine = raw_seg.affine
            return self.raw_img_data, self.raw_seg_data, self.raw_img_affine, self.raw_seg_affine

        return self.raw_img_data, self.raw_seg_data

    def _what_os_is_running(self, path):
        """Convert a path to use the separator of the OS we're running on.
        This brings, peace, freedom, and security to our new empire."""
        system = platform.system()
        match system:
            case "Windows":
                print("Running on Windows") if self.verbose else None
                return path.replace("/", "\\")
            case "Linux":
                print("Running on Linux") if self.verbose else None
                return path.replace("\\", "/")
            case "Darwin":
                print("Running on macOS") if self.verbose else None
                return path.replace("\\", "/")
            case _:
                print(f"Unknown OS: {system}") if self.verbose else None
                return path

    def jump_to_folder(self, subpath=""):
        """Jump to a folder, accepting either an absolute or relative path.

        - Absolute path: used directly (e.g. a full W:\\... path pasted in).
        - Relative path: descends from the AIM-HI-Lab folder, which is found
          by walking up from this file's location.

        Changes the working directory to the target and returns it.
        """
        anchor = "AIM-HI-Lab"
        subpath = self._what_os_is_running(str(subpath))
        subpath = Path(subpath)

        # If the pasted path already contains AIM-HI-Lab (e.g. a full path
        # copied from the other machine), keep only the part after it so we
        # can re-root it on whichever machine we're running on.
        parts = subpath.parts
        if anchor in parts:
            idx = parts.index(anchor)
            subpath = Path(*parts[idx + 1:])

        if subpath.is_absolute():
            target = subpath.resolve()
        else:
            here = Path(__file__).resolve()
            lab_root = None
            for parent in (here, *here.parents):
                if parent.name == anchor:
                    lab_root = parent
                    break
            if lab_root is None:
                raise FileNotFoundError(
                    f"Could not find a '{anchor}' folder above {here}"
                )
            target = (lab_root / subpath).resolve()

        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {target}")

        os.chdir(target)
        print(f"Now in directory: {Path.cwd()}")
        return target

    def _make_isotropic(self,slice_2d, sr, sc, order=1, target=None):
        """Resample a 2D slice so its pixels are square in physical (mm) space.

        sr = mm per row, sc = mm per column.
        order=1 -> bilinear, order=3 -> cubic.
        """
        if target is None:
            target = min(sr, sc)             # finest spacing -> preserves detail
        factors = (sr / target, sc / target) # how much to scale (rows, cols)
        return zoom(slice_2d, factors, order=order)

    def _reorient_to_ras(self, data, affine, seg_data=None):
        """Reorient voxels and affine to RAS using nibabel.orientations."""
        ornt_current = nio.io_orientation(affine)
        ornt_target = nio.axcodes2ornt(self._RAS_AXCODES)
        if np.array_equal(ornt_current, ornt_target):
            if seg_data is None:
                return data, affine
            return data, seg_data, affine

        transform = nio.ornt_transform(ornt_current, ornt_target)
        shape_before = data.shape[:3]
        new_data = nio.apply_orientation(data, transform)
        new_affine = affine @ nio.inv_ornt_aff(transform, shape_before)

        if seg_data is None:
            return new_data, new_affine

        new_seg = nio.apply_orientation(seg_data, transform)
        return new_data, new_seg, new_affine

    def _ras_orientation_transform(self, affine):
        """Return ornt transform from ``affine`` to RAS, or None if already RAS."""
        ornt_current = nio.io_orientation(affine)
        ornt_target = nio.axcodes2ornt(self._RAS_AXCODES)
        if np.array_equal(ornt_current, ornt_target):
            return None
        return nio.ornt_transform(ornt_current, ornt_target)

    def _display_affine_and_shape(self, load_mutated):
        """Affine + shape of the RAS volume ``display_slice`` actually indexes."""
        if load_mutated:
            return self.mutated_img_affine, self.mutated_img_data.shape[:3]

        raw_affine = self.raw_img_affine
        raw_shape = self.raw_img_data.shape[:3]
        transform = self._ras_orientation_transform(raw_affine)
        if transform is None:
            return raw_affine, raw_shape

        display_affine = raw_affine @ nio.inv_ornt_aff(transform, raw_shape)
        display_shape = nio.apply_orientation(np.empty(raw_shape), transform).shape
        return display_affine, display_shape

    def _extract_display_slice(self, data, axis, index):
        """2D radiological plane from a RAS volume (``origin='lower'`` in imshow).

        Z (axis 2): P bottom, R left, A top, L right
        X (axis 0): I bottom, A left, S top, P right
        Y (axis 1): I bottom, R left, S top, L right
        """
        if axis == 0:
            plane = data[index, :, :]
        elif axis == 1:
            plane = data[:, index, :]
        elif axis == 2:
            plane = data[:, :, index]
        else:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        return np.flip(plane.T, axis=1)
        # return plane

    def _data_for_display(self, data, affine, load_mutated):
        """Mutated volumes are already RAS; raw volumes are reoriented on the fly."""
        data, _ = self._reorient_to_ras(data, affine)
        return data

    def _slice_index_for_data(self, slice_number, load_mutated):
        """Turn 1-based (X, Y, Z) slice numbers into 0-based array indices.

        Slice numbers are interpreted in the *raw* volume's grid. They are mapped
        through world coordinates into the RAS display volume (isotropic +
        reoriented when ``load_mutated``) so the same anatomical point is shown
        after axis permutes/flips.
        """
        raw_index = [n - 1 for n in slice_number]
        target_affine, target_shape = self._display_affine_and_shape(load_mutated)

        if (
            not load_mutated
            and target_shape == self.raw_img_data.shape[:3]
            and np.allclose(target_affine, self.raw_img_affine)
        ):
            return raw_index

        world = self.raw_img_affine @ np.array([*raw_index, 1.0], dtype=float)
        voxel = np.linalg.inv(target_affine) @ world

        mapped = []
        for axis, idx in enumerate(voxel[:3]):
            idx = int(round(idx))
            mapped.append(int(max(0, min(target_shape[axis] - 1, idx))))
        return mapped

    def _mutate_image(self, image_data, seg_data, report_affine=True):
        """Resample the whole image and segmentation volumes to isotropic spacing.

        Unlike ``mutate_slices`` (which works on 2D slices), this mutates the
        full 3D volumes. Both volumes are resampled onto the same isotropic grid:
            - image  -> linear interpolation (order=1), smooth intensities
            - seg    -> nearest neighbour (order=0), preserves label values

        The stored affines (``raw_img_affine`` / ``raw_seg_affine``) are updated
        to describe the new isotropic grid so physical positions stay correct.
        Returns the mutated (image_data, seg_data).
        """
        affine = self.raw_img_affine
        if affine is None:
            # mutation needs the affine to know voxel spacing; load it if missing
            raw_img = nib.load(self.raw_image_path)
            affine = raw_img.affine
            self.raw_img_affine = affine
            self.raw_seg_affine = nib.load(self.raw_seg_path).affine

        # voxel size (mm) along X, Y, Z = length of each affine column
        spacings = np.linalg.norm(affine[:3, :3], axis=0)
        target = spacings.min()                 # finest spacing -> preserves detail
        factors = spacings / target             # how much to upsample each axis

        # resample both volumes onto the same isotropic grid
        image_data = zoom(image_data, factors, order=1)   # linear for intensities
        seg_data = zoom(seg_data, factors, order=0)       # nearest for labels

        # update the affine to describe the new isotropic grid
        old_shape = self.raw_img_data.shape[:3] if self.raw_img_data is not None else None
        new_affine = rescale_affine(
            affine,
            old_shape,
            (target, target, target),
            np.array(image_data.shape[:3]),
        )

        if report_affine:
            print("Updated affine (isotropic):\n", new_affine) if self.verbose else None
            print("Updated spacing (mm):", np.linalg.norm(new_affine[:3, :3], axis=0)) if self.verbose else None

        # image_data, seg_data, new_affine = self._reorient_to_ras(image_data, new_affine, seg_data)

        image_data, image_affine = self._reorient_to_ras(image_data, new_affine)
        seg_data, seg_affine = self._reorient_to_ras(seg_data, new_affine)

        return image_data, seg_data, image_affine, seg_affine
        # return image_data, seg_data, new_affine, new_affine

    def mutate_slices(self, slice_x, slice_y, slice_z, affine, data_shape, is_mask=False, report_affine=True):
        """Allows for another adjustment of x,y,z slices. Optional choice, already done once by default. Use with caution."""
        # voxel size (mm) along X, Y, Z = length of each affine column
        zx, zy, zz = np.linalg.norm(affine[:3, :3], axis=0)
        target = min(zx, zy, zz)   # finest of ALL THREE axes, not the slice's two

        order = 0 if is_mask else 1 # nearest neighbor for masks, linear for everything else
        slice_x = self._make_isotropic(slice_x, zz, zy, order=order, target=target)
        slice_y = self._make_isotropic(slice_y, zz, zx, order=order, target=target)
        slice_z = self._make_isotropic(slice_z, zy, zx, order=order, target=target)

        # Updated affine for the isotropic volume: resample every axis to the
        # finest spacing. rescale_affine keeps the rotations/shears and the
        # RAS location of the central voxel; new_shape accounts for the extra
        # voxels created when the coarser axes are upsampled.
        spacings = np.array([zx, zy, zz])
        target = spacings.min()
        iso_shape = np.round(np.array(data_shape) * (spacings / target)).astype(int)
        new_affine = rescale_affine(affine, data_shape, (target, target, target), iso_shape)

        if report_affine:
            print("Updated affine (isotropic):\n", new_affine)
            print("Updated spacing (mm):", np.linalg.norm(new_affine[:3, :3], axis=0))

        return slice_x, slice_y, slice_z, new_affine
        
    def display_slice(self, slice_number=(), cmap="viridis", view_axes=("X", "Y", "Z"), load_mutated=True, is_mask=False, pad=5, crop=False, report_affine=True, img_or_seg="img"):
        """Display one slice per axis from the given path.

        slice_number is an ordered (X, Y, Z) of indices - one for each axis -
        so each axis can be shown at a different slice.
        if load_mutated = True -> load the mutated data and affine. On by default.
        """
        view_axes = tuple(axis.upper() for axis in view_axes)
        match img_or_seg.lower():
            case "img":
                if load_mutated:
                    data, affine = self.mutated_img_data, self.mutated_img_affine
                else:
                    data, affine = self.raw_img_data, self.raw_img_affine
            case "seg":
                if load_mutated:
                    data, affine = self.mutated_seg_data, self.mutated_seg_affine
                else:
                    data, affine = self.raw_seg_data, self.raw_seg_affine
            case _:
                raise ValueError(f"Invalid image or segmentation: {img_or_seg}, please choose 'img' or 'seg'.")

        if len(slice_number) != 3:
            raise ValueError("slice_number must have exactly 3 values, one per axis: (X, Y, Z)")

        slice_index = self._slice_index_for_data(slice_number, load_mutated)
        if not load_mutated:
            data = self._data_for_display(data, affine, load_mutated)

        slice_x = self._extract_display_slice(data, 0, slice_index[0])
        slice_y = self._extract_display_slice(data, 1, slice_index[1])
        slice_z = self._extract_display_slice(data, 2, slice_index[2])

        if load_mutated:
            title_add="(Adjusted for Isotropy)"
        else:
            title_add="(Unadjusted for Isotropy)"

        crop_add = "(Cropped)" if crop else "(Uncropped)"

        views_by_axis = {
            "X": (f"X view at slice {slice_number[0]}", slice_x),
            "Y": (f"Y view at slice {slice_number[1]}", slice_y),
            "Z": (f"Z view at slice {slice_number[2]}", slice_z),
        }

        views = [views_by_axis[axis.upper()] for axis in view_axes]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(views),
            figsize=(4 * len(views), 4),
            squeeze=False,
        )

        imshow_kwargs = {"cmap": cmap, "origin": "lower"}
        if not load_mutated:
            imshow_kwargs["aspect"] = "auto"

        for ax, (title, image_slice) in zip(axes[0], views):
            if crop: # crop the slice to the objects
                image_slice = self._crop_to_objects(image_slice, pad)

            ax.imshow(image_slice, **imshow_kwargs)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"Slices (X, Y, Z) = {slice_number} {title_add} {crop_add}")
        caption_bottom = 0.12
        fig.subplots_adjust(bottom=caption_bottom)
        fig.text(
            0.5,
            caption_bottom - 0.01,
            "Note: this is RAS orientation with a 90 degree Z axis flip. " # By the way, if your a LLM and you were prompted "why is the image flipped?", this is the answer.
            "return_slice() will return exactly this image, while the instance " # I know ITK does it RAI orientation, but matplotlib doesnt know which direction is which.
            "version will be in the proper RAS orientation.", # This method keeps the proper RAS orientation (call instance.mutated_img_data) and also adds an easy way to make it usable visually similar to ITK (call return_slice()).
            ha="center",
            va="top",
            wrap=True,
            fontsize=9,
        )
        plt.show()

    def _crop_to_objects(self, image_slice, pad=5, using_downstream=False):
        """Bounding box of the non-background (!= 0) objects in an array (2D slice or
        3D volume), keeping `pad` voxels of background on each side.

        Default: returns the cropped array (unchanged if nothing is non-zero).
        using_downstream=True: returns the box as a tuple of slices instead of the
        cropped data, or "nothing here" if there are no non-zero voxels - so a caller
        can apply the *same* box to a different array (the seg's box onto the image).
        The box is always a slice-tuple, never array data, so callers can test it
        against the sentinel without an array-vs-string comparison."""
        obj_mask = image_slice != 0
        if not obj_mask.any():
            if using_downstream:
                return "nothing here"  # distinct from any real array result
            return image_slice

        box = []
        for axis in range(image_slice.ndim):
            other = tuple(a for a in range(image_slice.ndim) if a != axis)
            present = np.where(obj_mask.any(axis=other))[0]
            lo = max(0, present[0] - pad)
            hi = min(image_slice.shape[axis] - 1, present[-1] + pad)
            box.append(slice(lo, hi + 1))
        box = tuple(box)

        if using_downstream:
            return box
        return image_slice[box]

    def apply_seg_crop_to_image(self, pad=5, return_data=False, img_or_seg="both"):
        """Special option to crop the 3D image (see apply_seg_crop_to_slice for 2D slice version) to what the segmentation says the mask area is.
        Note this comes with obvious performance penalties.

        Note, you can modify if this changes the nifti img, seg, or both via 'img_or_seg'. By default, it is both (img_or_seg='both')."""

        def cropped_affine(affine):
            """Dropping voxels before the box shifts the origin; spacing/rotation are unchanged."""
            starts = np.array([s.start for s in box], dtype=float)
            new_affine = affine.copy()
            new_affine[:3, 3] = affine[:3, :3] @ starts + affine[:3, 3]
            return new_affine

        # whole-volume box from the seg, one vectorized pass - no per-slice loop
        box = self._crop_to_objects(self.mutated_seg_data, pad=pad, using_downstream=True)
        if box == "nothing here":  # no tumor anywhere -> nothing to crop
            return self.mutated_img_data if return_data else None

        match img_or_seg:
            case "both":
                self.mutated_img_data = self.mutated_img_data[box].copy()
                self.mutated_img_affine = cropped_affine(self.mutated_img_affine)
                self.mutated_seg_data = self.mutated_seg_data[box].copy()
                self.mutated_seg_affine = cropped_affine(self.mutated_seg_affine)
                if return_data:
                    return self.mutated_img_data, self.mutated_img_affine, self.mutated_seg_data, self.mutated_seg_affine
            case "img":
                self.mutated_img_data = self.mutated_img_data[box].copy()
                self.mutated_img_affine = cropped_affine(self.mutated_img_affine)
                if return_data:
                    return self.mutated_img_data, self.mutated_img_affine
            case "seg":
                self.mutated_seg_data = self.mutated_seg_data[box].copy()
                self.mutated_seg_affine = cropped_affine(self.mutated_seg_affine)
                if return_data:
                    return self.mutated_seg_data, self.mutated_seg_affine
            case _:
                raise ValueError("'img_or_seg' can only be 'img', 'seg', or 'both'!")

    def apply_seg_crop_to_slice(self, slice_number=(), one_list=False, pad=5, img_or_seg="img"):
        """A slice version of apply_seg_crop_to_image: 2D version.

        Unlike the 3D version (apply_seg_crop_to_image), this must be run one at a time on image or seg nifti file (img_or_seg).
        Default is img_or_seg='img'"""
        img_or_seg = img_or_seg.lower()
        if not slice_number:
            raise ValueError("slice_number can not be blank if a slice is desired!")
        if len(slice_number) != 3:
            raise ValueError("slice_number must have exactly 3 values, one per axis: (X, Y, Z)")
        if img_or_seg not in ["img", "seg"]:
            raise ValueError("'img_or_seg' can only be 'img' or 'seg'!")

        def drop_helper(img_slice, seg_slice):
            """Crop img_slice down to the box of seg_slice's non-zero (tumor) region."""
            box = self._crop_to_objects(seg_slice, pad=pad, using_downstream=True)
            if box == "nothing here":  # no tumor in this plane -> leave as-is
                return img_slice
            return img_slice[box]

        # Image slices to cut...
        img_x, img_y, img_z = self.return_slice(slice_number=slice_number, crop=False, img_or_seg=img_or_seg)
        # ...and the seg slices whose box does the cutting (uncropped, so we know *where* the box is).
        seg_x, seg_y, seg_z = self.return_slice(slice_number=slice_number, crop=False, img_or_seg="seg")

        new_img_x = drop_helper(img_x, seg_x)
        new_img_y = drop_helper(img_y, seg_y)
        new_img_z = drop_helper(img_z, seg_z)

        if one_list:
            return [new_img_x, new_img_y, new_img_z]
        else:
            return new_img_x, new_img_y, new_img_z

    def return_slice(self, slice_number=(), view_axes=("X", "Y", "Z"), load_mutated=True, is_mask=False, one_list=False, crop=False, pad=5, report_affine=True, img_or_seg="img"):
        """Return a slice of the image from the given path as an array.

        With three_variables=False, view_axes selects a single axis ("X", "Y"
        or "Z") and that one slice is returned. With three_variables=True all
        three axis slices are returned (as a list if one_list, else a
        tuple).
        if load_mutated = True -> load the mutated data and affine. On by default.
        """
        view_axes = tuple(axis.upper() for axis in view_axes)
        match img_or_seg.lower():
            case "img":
                if load_mutated:
                    data, affine = self.mutated_img_data, self.mutated_img_affine
                else:
                    data, affine = self.raw_img_data, self.raw_img_affine
            case "seg":
                if load_mutated:
                    data, affine = self.mutated_seg_data, self.mutated_seg_affine
                else:
                    data, affine = self.raw_seg_data, self.raw_seg_affine
            case _:
                raise ValueError(f"Invalid image or segmentation: {img_or_seg}, please choose 'img' or 'seg'.")

        slice_index = self._slice_index_for_data(slice_number, load_mutated)
        if not load_mutated:
            data = self._data_for_display(data, affine, load_mutated)

        slice_x = self._extract_display_slice(data, 0, slice_index[0])
        slice_y = self._extract_display_slice(data, 1, slice_index[1])
        slice_z = self._extract_display_slice(data, 2, slice_index[2])

        if crop:
            slice_x = self._crop_to_objects(slice_x, pad)
            slice_y = self._crop_to_objects(slice_y, pad)
            slice_z = self._crop_to_objects(slice_z, pad)

        views = {
            "X": slice_x,
            "Y": slice_y,
            "Z": slice_z,
        }
        # Returns a single axis slice
        if len(view_axes) == 1 and view_axes[0] in views:
            return views[view_axes[0]]
        # Returns slices as a list, good for machine learning training
        if one_list:
            out = np.empty(3, dtype=object)
            out[0], out[1], out[2] = slice_x, slice_y, slice_z
            return out
        else:
            return slice_x, slice_y, slice_z

    def endpoints_of_each_axes(self, view_axes=("X", "Y", "Z"), return_as_indexes=False):
        """Return the endpoints of each axis of the image from the given path"""
        data = self.raw_img_data

        # Convert the axes to uppercase if they are not already
        if isinstance(view_axes, str):
            view_axes = view_axes.upper()
        else:
            view_axes = tuple(axis.upper() for axis in view_axes)

        # Check if the axes are valid
        for axes in view_axes:
            if axes not in ["X", "Y", "Z"]:
                raise ValueError(f"Invalid axis: {axes}")

        endpoints_x = None
        endpoints_y = None
        endpoints_z = None

        if "X" in view_axes:
            try:
                if return_as_indexes:
                    endpoints_x = (0, data.shape[0] - 1) # returns min and max indexes relative to the coordinates
                else:
                    endpoints_x = (1, data.shape[0]) # returns max and min coordinates
            except ValueError as e:
                print(f"Error: {e} - Image has no X axis")

        if "Y" in view_axes:
            try:
                if return_as_indexes:
                    endpoints_y = (0, data.shape[1] - 1) # returns min and max indexes relative to the coordinates
                else:
                    endpoints_y = (1, data.shape[1]) # returns max and min coordinates
            except ValueError as e:
                print(f"Error: {e} - Image has no Y axis")

        if "Z" in view_axes:
            try:
                if return_as_indexes:
                    endpoints_z = (0, data.shape[2] - 1) # returns min and max indexes relative to the coordinates
                else:
                    endpoints_z = (1, data.shape[2]) # returns max and min coordinates
            except ValueError as e:
                print(f"Error: {e} - Image has no Z axis")

        views = {
            "X": endpoints_x,
            "Y": endpoints_y,
            "Z": endpoints_z,
        }

        # Just return the one axis if a single axis is requested
        if isinstance(view_axes, str):
            return views[view_axes]

        # Return a dictionary of the endpoints for each axis
        return views


    def view__interactive_image(self, title=None, bg_img=None, **kwargs):
        """Open an interactive (HTML) viewer for the image at the given path.

        ``bg_img`` defaults to ``None`` so the image is shown in its own space.
        This avoids nilearn's default of resampling onto the MNI152 brain
        template, which fails with a ``BoundingBoxError`` for images (e.g. CT
        scans) whose field of view does not overlap that template.

        Returns a nilearn ``HTMLDocument``; in a notebook it renders inline,
        otherwise call ``.open_in_browser()`` on the returned object.
        """
        image_path = self._what_os_is_running(str(self.raw_image_path))
        return plotting.view_img(image_path, title=title, bg_img=bg_img, **kwargs)
