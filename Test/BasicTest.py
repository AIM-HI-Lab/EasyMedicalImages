import EasyMedicalImages as EMI
hl = EMI.HighLevel(path_to_folder=r"W:\Shared\LRI\Labs\AIM-HI-Lab\research-projects\honn\smaller-kidney-tumor-ct-sample-batch", thread_deadline_time=128.2)
original_pairs=hl.run_on_directory()
print(len(original_pairs), original_pairs[0][0].shape, original_pairs[0][1].shape)

print(f"\n------------ new method ------------\n")

print([m for m in dir(hl) if not m.startswith("__")])
pairs = hl.processed_run_on_directory("apply_seg_crop_to_image", {"img_or_seg": "both", "pad": 5}) #apply_operation_to_directory #processed_run_on_directory
print(len(pairs), pairs[0][0].shape, pairs[0][1].shape)

print(f"final results:\nOriginal:\n{len(original_pairs), original_pairs[0][0].shape, original_pairs[0][1].shape}\nNew Method:\n{len(pairs), pairs[0][0].shape, pairs[0][1].shape}")

print(EMI.go_one_level_deep())