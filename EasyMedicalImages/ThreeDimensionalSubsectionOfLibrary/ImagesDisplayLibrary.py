import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class ImagesDisplayLibrary:
    def __init__(self):
        pass

    def three_instances_of_image(self, image_path):
        """Return every slice for each axis as [all_X_slices, all_Y_slices, all_Z_slices].

        The image is loaded once and sliced directly. (Calling return_slice
        per slice would reload the whole file from disk on every call, which is
        very slow for a full volume - especially over a network share.)
        """
        data = self.load_image(image_path)
        x_slices = [data[i, :, :].T for i in range(data.shape[0])]
        y_slices = [data[:, i, :].T for i in range(data.shape[1])]
        z_slices = [data[:, :, i].T for i in range(data.shape[2])]
        return [x_slices, y_slices, z_slices]

    def display_three_image(self, image_path):
        """Render three 3D 'towers' - X, Y, Z - each stacked from all of its slices.

        Each tower has its own slider underneath it. Lowering the slider peels
        layers off the top of the tower, so the new top face shows the slice
        (cross-section) at the current height - letting you scan down through
        the stack one layer at a time.

        Each tower's box is scaled by the real voxel spacing (mm), so
        anisotropic voxels no longer look stretched.
        """
        x_slices, y_slices, z_slices = self.three_instances_of_image(image_path)

        # Voxel sizes (mm) along data axes X, Y, Z. Read the affine straight off
        # the header so we don't load the whole volume a second time.
        affine = nib.load(image_path).affine
        zx, zy, zz = np.linalg.norm(affine[:3, :3], axis=0)

        # Per tower: the stacked volume, plus the (depth, rows, cols) voxel
        # spacings matching how that volume is laid out.
        #   X tower volume axes are (X, Z, Y) -> spacings (zx, zz, zy)
        #   Y tower volume axes are (Y, Z, X) -> spacings (zy, zz, zx)
        #   Z tower volume axes are (Z, Y, X) -> spacings (zz, zy, zx)
        raw_towers = [
            ("X tower", np.stack(x_slices, axis=0), (zx, zz, zy)),
            ("Y tower", np.stack(y_slices, axis=0), (zy, zz, zx)),
            ("Z tower", np.stack(z_slices, axis=0), (zz, zy, zx)),
        ]

        # Downsample rows/cols (and cap depth) once so each slider redraw is fast.
        towers = []
        for name, volume, spacing in raw_towers:
            step_s = max(1, volume.shape[0] // 128)
            step_r = max(1, volume.shape[1] // 64)
            step_c = max(1, volume.shape[2] // 64)
            v = volume[::step_s, ::step_r, ::step_c]
            sd, sr, sc = spacing
            towers.append({
                "name": name,
                "volume": v,
                "step_s": step_s,
                "full_depth": volume.shape[0],
                "vmin": float(np.min(v)),
                "vmax": float(np.max(v)),
                # effective spacing per displayed axis = voxel size x downsample step
                "spacing_depth": sd * step_s,
                "spacing_rows": sr * step_r,
                "spacing_cols": sc * step_c,
            })

        fig = plt.figure(figsize=(15, 7))
        axes3d = []
        sliders = []

        def make_redraw(ax, tower, slider_getter):
            def redraw(_=None):
                cut = int(round(slider_getter()))
                cut = max(2, min(cut, tower["volume"].shape[0]))
                ax.cla()
                self._draw_tower(
                    ax, tower["volume"][:cut], tower["vmin"], tower["vmax"],
                    tower["spacing_cols"], tower["spacing_rows"], tower["spacing_depth"],
                )
                shown = min(cut * tower["step_s"], tower["full_depth"])
                ax.set_title(f"{tower['name']} - layer {shown}/{tower['full_depth']}")
                ax.set_xlabel("width")
                ax.set_ylabel("height")
                ax.set_zlabel("slice #")
                fig.canvas.draw_idle()
            return redraw

        for i, tower in enumerate(towers):
            # 3D plot in the top area of each column.
            ax = fig.add_axes([0.05 + i * 0.32, 0.18, 0.28, 0.74], projection="3d")
            axes3d.append(ax)

            # Slider strip beneath each tower.
            slider_ax = fig.add_axes([0.06 + i * 0.32, 0.06, 0.26, 0.03])
            depth = tower["volume"].shape[0]
            slider = Slider(slider_ax, "height", 2, depth, valinit=depth, valstep=1)
            sliders.append(slider)

            redraw = make_redraw(ax, tower, lambda s=slider: s.val)
            slider.on_changed(redraw)
            redraw()

        # Keep slider references alive so callbacks keep working.
        fig._tower_sliders = sliders
        plt.show()

    def _draw_tower(self, ax, v, vmin, vmax,
                    spacing_cols=1.0, spacing_rows=1.0, spacing_depth=1.0):
        """Texture the 6 outer faces of a (depth, rows, cols) block with its slices.

        spacing_cols/rows/depth are the physical voxel sizes (mm) along the
        width / height / slice-number axes, used to set a physically
        proportional box aspect.
        """
        depth, rows, cols = v.shape

        def face(grid_a, grid_b, grid_c, intensity):
            if vmax > vmin:
                norm = (intensity - vmin) / (vmax - vmin)
            else:
                norm = np.zeros_like(intensity)
            ax.plot_surface(grid_a, grid_b, grid_c, facecolors=plt.cm.gray(norm), shade=False)

        cols_axis = np.arange(cols)
        rows_axis = np.arange(rows)
        depth_axis = np.arange(depth)

        # Top / bottom faces (constant slice #), spanning width (cols) and height (rows).
        gx, gy = np.meshgrid(cols_axis, rows_axis)
        face(gx, gy, np.full((rows, cols), depth - 1), v[-1, :, :])
        face(gx, gy, np.full((rows, cols), 0), v[0, :, :])

        # Front / back faces (constant height), spanning width and slice #.
        gx2, gz2 = np.meshgrid(cols_axis, depth_axis)
        face(gx2, np.full((depth, cols), 0), gz2, v[:, 0, :])
        face(gx2, np.full((depth, cols), rows - 1), gz2, v[:, -1, :])

        # Left / right faces (constant width), spanning height and slice #.
        gy3, gz3 = np.meshgrid(rows_axis, depth_axis)
        face(np.full((depth, rows), 0), gy3, gz3, v[:, :, 0])
        face(np.full((depth, rows), cols - 1), gy3, gz3, v[:, :, -1])

        # Scale the box by physical extent (voxel count x voxel size) so
        # anisotropic voxels are shown in true proportion.
        ax.set_box_aspect((cols * spacing_cols, rows * spacing_rows, depth * spacing_depth))