#include "HighLevel.h"

#include <algorithm>   // std::transform
#include <cctype>      // std::tolower
#include <filesystem>  // std::filesystem (C++17)
#include <iostream>    // std::cout, std::cin
#include <stdexcept>   // std::invalid_argument
#include <utility>     // std::move

namespace fs = std::filesystem;

HighLevel::HighLevel(std::string path_to_folder, std::string image_path, std::string seg_path, bool verbose)
    : path_to_folder(std::move(path_to_folder)),
      hl_img_path(std::move(image_path)),
      hl_seg_path(std::move(seg_path)),
      verbose(verbose) {
    // Mirrors Python __init__: just stores config. img_seg_pairs is
    // default-constructed empty, so there's nothing to clear here.
}

void HighLevel::run_on_directory(const std::string& directory_path) {
    // Use the provided directory if given; otherwise keep the one set at construction.
    if (!directory_path.empty()) {
        path_to_folder = directory_path;
    }

    std::vector<std::string> all_data = _go_one_level_deep(/*debug=*/false, "folders");
    _get_all_images_and_segs_from_path(all_data, /*return_affines=*/false, /*debug=*/true);
}

std::vector<std::string> HighLevel::_go_one_level_deep(bool debug, std::string options) {
    std::transform(options.begin(), options.end(), options.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (options != "folders" && options != "files" && options != "all") {
        throw std::invalid_argument("Invalid options: '" + options + "'");
    }

    std::vector<std::string> all_data;
    for (const fs::directory_entry& entry : fs::directory_iterator(path_to_folder)) {
        if (options == "folders" && !entry.is_directory()) {   // check 1
            continue;
        }
        if (options == "files" && !entry.is_regular_file()) {  // check 2
            continue;
        }
        if (debug) {
            std::cout << entry.path().string() << '\n';
        }
        all_data.push_back(entry.path().string());
    }

    return all_data;
}

void HighLevel::_get_all_images_and_segs_from_path(const std::vector<std::string>& all_data, bool return_affines, bool debug) {
    // Intentionally left blank -- to be implemented.
    (void)all_data;
    (void)return_affines;
    (void)debug;
}

// Translation of `if __name__ == "__main__":`. In a library build you'd usually
// move main() into its own .cpp so HighLevel can be compiled without it.
int main() {
    std::cout << "Enter the path to the folder: ";
    std::string input_path;
    std::getline(std::cin, input_path);

    HighLevel high_level(input_path, "image.nii.gz", "tumor_segmentation_v2.nii.gz", /*verbose=*/true);
    high_level.run_on_directory();
    return 0;
}