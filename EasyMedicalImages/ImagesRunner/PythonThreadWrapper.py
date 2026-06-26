from EasyMedicalImages.ImagesRunner.CTImagesLibrary import CTImagesLibrary

def load_case(folder_path, image_path, seg_path):
    ct = CTImagesLibrary(path_to_folder=folder_path, image_path=image_path, seg_path=seg_path, verbose=False)

    return (ct.mutated_img_data, ct.mutated_seg_data)