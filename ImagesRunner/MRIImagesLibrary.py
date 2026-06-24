import os
from CTImagesLibrary import CTImagesLibrary

class MRIImagesLibrary(CTImagesLibrary):
    def __init__(self, path_to_folder, image_path="image.nii.gz", seg_path="tumor_segmentation_v2.nii.gz", verbose=True):
        super().__init__(path_to_folder, image_path, seg_path, verbose)
        pass

    def bla(self):
        pass

    