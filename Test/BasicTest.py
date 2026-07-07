import EasyMedicalImages as EMI
hl = EMI.HighLevel(path_to_folder=r"W:\Shared\LRI\Labs\AIM-HI-Lab\research-projects\honn\kidney-tumor-ct-sample-batch", thread_deadline_time=128.2)
original_pairs=hl.run_on_directory()
print(len(original_pairs), original_pairs[0][0].shape, original_pairs[0][1].shape)

print(f"\n------------ new method ------------\n")

print([m for m in dir(hl) if not m.startswith("__")])
pairs = hl.apply_operation_to_directory("apply_seg_crop_to_image", {"img_or_seg": "both", "pad": 5})
print(len(pairs), pairs[0][0].shape, pairs[0][1].shape)

print(f"final results:\nOriginal:\n{len(original_pairs), original_pairs[0][0].shape, original_pairs[0][1].shape}\nNew Method:\n{len(pairs), pairs[0][0].shape, pairs[0][1].shape}")

# import EasyMedicalImages as EMI
#
# path_to_folder_of_ct_scans = input("Input path to CT scans: ")
# instance = EMI.HighLevel(path_to_folder=path_to_folder_of_ct_scans)
#
# img_seg_pairs_mega_list = instance.run_on_directory()  # list of (image, seg) array pairs
#
# all_images=[images for images,_ in img_seg_pairs_mega_list]
# all_segmentations=[segmentations for _, segmentations in img_seg_pairs_mega_list]
#
# print(f"Images: {all_images}")
# print(f"Segmentation Masks: {all_segmentations}")
#
# print([(img.shape, seg.shape) for img, seg in img_seg_pairs_mega_list])