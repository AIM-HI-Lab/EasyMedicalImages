#include "HighLevel.h"

#include <iostream>
#include <algorithm>   // std::transform
#include <cctype>      // std::tolower
#include <filesystem>  // std::filesystem (C++17)
#include <stdexcept>   // std::invalid_argument
#include <utility>     // std::move
#include <variant>
#include <thread>
#include <cmath>
// #include <pybind11/embed.h>
// #include <pybind11/numpy.h>
#include <thread>

// namespace py = pybind11;

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
    std::vector<std::string> all_data = _go_one_level_deep(/*debug=*/false, "folders", directory_path);
    _get_all_images_and_segs_from_path(all_data, /*return_affines=*/false, /*debug=*/true);
}

std::vector<std::string> HighLevel::_go_one_level_deep(bool debug, std::string options, const std::string& directory) {
    if (options != "folders" && options != "files" && options != "all") {
        throw std::invalid_argument("Invalid options: '" + options + "'");
    }

    const std::string& scan_dir = directory.empty() ? path_to_folder : directory;

    std::vector<std::string> all_data;
    for (const fs::directory_entry& entry : fs::directory_iterator(scan_dir)) {
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

int HighLevel::how_many_threads_available(bool debug){
    unsigned int threads = std::thread::hardware_concurrency();
    if (threads==0) {
        threads = 1;
        std::cout<<"There is some sort of weird but, and the computer doesn't think you have any threads. Attempting to use single thread mode."<<std::endl;
    }
    std::cout << "Available hardware threads: " << threads << "\n";
    return threads;
}

void HighLevel::_get_all_images_and_segs_from_path(const std::vector<std::string>& all_data, bool return_affines, bool debug) {
    (void)all_data;
    (void)return_affines;
    (void)debug;
    // all_data is a huge list of all of the patient folder names. This is to fight race conditions
    int number_of_threads=how_many_threads_available(true);
    std::cout << "Number of threads: " << number_of_threads << "\n";
    int threads_we_are_using=std::round(number_of_threads*.60);

    // std::vector<std::variant<int, double, std::string>> img_seg_pairs;
    // img_seg_pairs (made in HighLevel.h) is our output answer here. It has all of the images and segs in array format

    for (size_t batch_start = 0;
        batch_start < all_data.size();
        batch_start += threads_we_are_using){
        std::vector<std::thread> workers;

        // Launch one thread for each folder in this batch
        for (size_t i = 0; i < threads_we_are_using && (batch_start + i) < all_data.size(); ++i) {
            const std::string& path_str = all_data[batch_start + i];

            if (debug){
                std::cout << "Thread " << i << "is in folder: "<< path_str << '\n';
            }

            workers.emplace_back([path_str]() {
                workers.emplace_back([&, path_str, batch_start, i]()
                {
                    std::string output_file =
                        "temp_" + std::to_string(batch_start + i) + ".pkl";

                    std::string command =
                        "python EasyMedicalImages/ImagesRunner/PythonThreadWrapper.py "
                        "--folder \"" + path_str + "\" "
                        "--img \"" + hl_img_path + "\" "
                        "--seg \"" + hl_seg_path + "\" "
                        "--output \"" + output_file + "\"";

                    int rc = std::system(command.c_str());

                    if (rc != 0)
                    {
                        std::cerr << "Python failed for " << path_str << '\n';
                        return;
                    }

                    // TODO:
                    // Read output_file and append it to img_seg_pairs.
                });
            });
        }

        // Wait for every thread in this batch to finish
        for (auto& worker : workers){
            worker.join();
        }
    }

}

int main() {
    std::cout << "Enter the path to the folder: ";
    std::string input_path;
    std::getline(std::cin, input_path);

    HighLevel high_level(input_path, "image.nii.gz", "tumor_segmentation_v2.nii.gz", /*verbose=*/true);
    high_level.run_on_directory();
    return 0;
}