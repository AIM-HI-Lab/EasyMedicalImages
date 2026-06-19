import os
import CTImagesLibrary
import MRIImagesLibrary

class HighLevel(CTImagesLibrary, MRIImagesLibrary):
    def __init__(self, path_to_folder, image_path="image.nii.gz", seg_path="tumor_segmentation_v2.nii.gz", verbose=True):
        CTImagesLibrary.__init__(self, path_to_folder, image_path, seg_path, verbose)
        MRIImagesLibrary.__init__(self, path_to_folder, image_path, seg_path, verbose)

    def bla(self):
        pass