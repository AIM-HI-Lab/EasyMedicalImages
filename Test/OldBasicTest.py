import EasyMedicalImages as EMI

path_to_folder_of_ct_scans = input("Input path to CT scans: ")
instance = EMI.HighLevel(path_to_folder=path_to_folder_of_ct_scans)

img_seg_pairs_mega_list = instance.run_on_directory()  # list of (image, seg) array pairs

all_images=[images for images,_ in img_seg_pairs_mega_list]
all_segmentations=[segmentations for _, segmentations in img_seg_pairs_mega_list]

print(f"Images: {all_images}")
print(f"Segmentation Masks: {all_segmentations}")

print([(img.shape, seg.shape) for img, seg in img_seg_pairs_mega_list])