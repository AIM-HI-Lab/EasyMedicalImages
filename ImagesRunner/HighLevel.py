import os
from MRIImagesLibrary import MRIImagesLibrary
from CTImagesLibrary import CTImagesLibrary
from pathlib import Path
import numpy as np

class HighLevel(MRIImagesLibrary, CTImagesLibrary):
    def __init__(self, path_to_folder, image_path="image.nii.gz", seg_path="tumor_segmentation_v2.nii.gz", verbose=False):
        self.path_to_folder = path_to_folder
        self.hl_img_path = image_path
        self.hl_seg_path = seg_path
        self.verbose = verbose
        self.img_seg_pairs=[]

    def run_on_directory(self, directory_path=None):
        if directory_path is None:
            directory_path = self.path_to_folder
        else:
            raise ValueError("No directory path provided")

        all_data = self._go_one_level_deep(debug=False, options="folders")
        self._get_all_images_and_segs_from_path(all_data, debug=True)

    def _go_one_level_deep(self, debug=False, options="folders"):
        options = options.lower()
        if options not in ("folders", "files", "all"):
            raise ValueError(f"Invalid options: {options!r}")

        all_data = []
        with os.scandir(self.path_to_folder) as entries:
            for entry in entries:
                if options == "folders" and not entry.is_dir(): # check 1
                    continue
                if options == "files" and not entry.is_file(): # check 2
                    continue
                if debug:
                    print(entry.path)
                all_data.append(entry.path)

        return all_data

    def _get_all_images_and_segs_from_path(self, all_data, return_affines=False, debug=False):
        def ground_work(folder_path, all_data, return_affines=False):
            match return_affines:
                case False:
                    try:
                        image_pair_instance = CTImagesLibrary(
                            path_to_folder=folder_path,
                            image_path=self.hl_img_path,
                            seg_path=self.hl_seg_path,
                            verbose=False,
                        )
                        print(f"Image shape: {image_pair_instance.mutated_img_data.shape}")
                        print(f"Seg shape:   {image_pair_instance.mutated_seg_data.shape}")
                        self.img_seg_pairs.append((image_pair_instance.mutated_img_data, image_pair_instance.mutated_seg_data))
                        print(f"Amazing sucess! ")
                    except Exception as e:
                        print(f"Error in {folder_path}: {e}")
                
                case True:
                    try:
                        image_pair_instance = CTImagesLibrary(
                            path_to_folder=folder_path,
                            image_path=self.hl_img_path,
                            seg_path=self.hl_seg_path,
                            verbose=False,
                        )
                        self.img_seg_pairs.append(((image_pair_instance.mutated_img_data, image_pair_instance.mutated_img_affine), (image_pair_instance.mutated_seg_data, image_pair_instance.mutated_seg_affine)))
                    except Exception as e:
                        print(f"Error in {folder_path}: {e}")
                        
                case _:
                    raise ValueError(f"Invalid return_affines: {return_affines!r}")

        self.img_seg_pairs = []

        i=0
        for folder_path in all_data:
            if debug:
                print(f"\n{folder_path}:")
            ground_work(folder_path, all_data, return_affines=return_affines)
            print(np.array(self.img_seg_pairs[i]).shape)
            # print(np.array(self.img_seg_pairs))
            i=i+1
            if i==2:
                print("Training view (to picture the data):")
                print([(img.shape, seg.shape) for img, seg in self.img_seg_pairs])
                exit()


        return self.img_seg_pairs

if __name__ == "__main__":
    input_path = input("Enter the path to the folder: ")
    high_level = HighLevel(path_to_folder=input_path, verbose=True)
    high_level.run_on_directory()