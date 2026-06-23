from CTImagesLibrary import CTImagesLibrary as il
import numpy as np

input_path = input("Enter the path to the image folder: ")
instance=il(path_to_folder=input_path, verbose=True)

print(f"Data shape: {instance.raw_img_data.shape}\n"); print(f"Mutated data shape: {instance.mutated_img_data.shape}\n")
print(f"Seg shape: {instance.raw_seg_data.shape}\n"); print(f"Mutated seg shape: {instance.mutated_seg_data.shape}\n")
print(f"Affine matrix: \n{instance.raw_img_affine}\n"); print(f"Mutated affine matrix: \n{instance.mutated_img_affine}\n")

big_list = instance.return_slice(slice_number=(175, 279, 57), one_list=True, img_or_seg="img", crop=False)

slice_x, slice_y, slice_z = big_list

# slice_x, slice_y, slice_z = instance.apply_seg_crop_to_slice(slice_number=(175, 279, 57))
# instance.apply_seg_crop_to_image(slice_number=(175, 279, 57), img_or_seg='both')

print(f"Returned slice X shape: {slice_x.shape}")
print(f"Returned slice Y shape: {slice_y.shape}")
print(f"Returned slice Z shape: {slice_z.shape}")

mutated_list=instance.mutated_img_data
print(mutated_list.shape)

# 175, 279, 110
# 100, 22, 44

instance.display_slice(slice_number=(175, 279, 57), img_or_seg="seg", crop=True)

import nibabel as nib
print(nib.aff2axcodes(instance.raw_img_affine))      # e.g. ('L','P','S')
print(nib.aff2axcodes(instance.mutated_img_affine))  # ('R','A','S')

def show_proof_slices_work(slice_x, slice_y, slice_z):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, slc) in zip(
        axes,
        [("X", slice_x), ("Y", slice_y), ("Z", slice_z)],
    ):
        ax.imshow(slc, cmap="viridis", origin="lower")
        ax.set_title(f"{title} view")
        ax.axis("off")
    fig.suptitle("Proof: slices from return_slice()")
    plt.show()

show_proof_slices_work(slice_x, slice_y, slice_z)
