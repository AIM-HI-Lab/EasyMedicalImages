import EasyMedicalImages as EMI
path_to_folder_of_ct_scans=input("Input path to ct scans: ")
instance=EMI.HighLevel(path_to_folder=path_to_folder_of_ct_scans, verbose=True)
megadatalist=instance.run_on_directory()
print(megadatalist) # This outputs all of the images in that directory as arrays!
print(len(megadatalist))

input_path=input("Path to the image: ")
instance = EMI.CTImagesLibrary(path_to_folder=input_path, verbose=True)
instance.display_slice(slice_number=(175, 279, 57), img_or_seg="img", crop=False)
sx, sy, sz = instance.return_slice(slice_number=(175, 279, 57), img_or_seg="img")