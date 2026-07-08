#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <utility>
#include "HighLevel.h"

namespace py = pybind11;

// Zero-copy: move the Volume's buffer onto the heap and hand numpy a capsule
// that frees it. No copy of the (potentially huge) data vector.
static py::array volume_to_numpy(Volume&& v) {
    auto* buf = new std::vector<float>(std::move(v.data));
    py::capsule owner(buf, [](void* p) {
        delete reinterpret_cast<std::vector<float>*>(p);
    });
    std::vector<py::ssize_t> shape(v.shape.begin(), v.shape.end());
    return py::array_t<float>(shape, buf->data(), owner);  // C-order, no copy
}
// 4x4 row-major float64 -> numpy (4,4). Tiny; copied, not zero-copied.
static py::array affine_to_numpy(const std::array<double, 16>& a) {
    py::array_t<double> arr({4, 4});
    std::memcpy(arr.mutable_data(), a.data(), a.size() * sizeof(double));
    return arr;
}

// One (image|seg) side -> bare ndarray, or (ndarray, affine) when asked.
// Reads v.affine BEFORE moving v.data (volume_to_numpy only moves .data).
static py::object side_to_py(Volume&& v, bool with_affine) {
    if (!with_affine)
        return volume_to_numpy(std::move(v));
    py::array aff  = affine_to_numpy(v.affine);
    py::array data = volume_to_numpy(std::move(v));
    return py::make_tuple(std::move(data), std::move(aff));
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for EasyMedicalImages";

    py::class_<HighLevel>(m, "HighLevel")
        .def(py::init<std::string, std::string, std::string, bool, float, bool>(),
            py::arg("path_to_folder"),
            py::arg("image_path")           = "image.nii.gz",
            py::arg("seg_path")             = "tumor_segmentation_v2.nii.gz",
            py::arg("verbose")              = false,
            py::arg("thread_deadline_time") = 100.1f,
            py::arg("verbose_workers")      = true)
        .def("run_on_directory",[](HighLevel& self, const std::string& directory_path, bool return_affines) {
                self.python_exe = py::module_::import("sys").attr("executable").cast<std::string>();
                std::vector<std::pair<Volume, Volume>> pairs;
                {
                    py::gil_scoped_release release;
                    pairs = self.run_on_directory(directory_path, return_affines);
                }
                py::list out;
                for (auto& pr : pairs)
                    out.append(py::make_tuple(side_to_py(std::move(pr.first),  return_affines),
                                              side_to_py(std::move(pr.second), return_affines)));
                return out;  // off: list[(img,seg)]; on: list[((img,aff),(seg,aff))]
             },
             py::arg("directory_path") = "",
             py::arg("return_affines") = false)
        .def("processed_run_on_directory",
        [](HighLevel& self, const std::string& operation, py::dict op_kwargs,
           const std::string& directory_path, bool return_affines) {
                self.python_exe = py::module_::import("sys").attr("executable").cast<std::string>();
                // Serialize kwargs -> JSON here, while we still hold the GIL.
                std::string op_kwargs_json =
                    py::module_::import("json").attr("dumps")(op_kwargs).cast<std::string>();
                std::vector<std::pair<Volume, Volume>> pairs;
                {
                    py::gil_scoped_release release;
                    pairs = self.processed_run_on_directory(operation, op_kwargs_json, directory_path, return_affines);
                }
                py::list out;
                for (auto& pr : pairs)
                    out.append(py::make_tuple(volume_to_numpy(std::move(pr.first)),
                    volume_to_numpy(std::move(pr.second))));
                return out;  // list[tuple[np.ndarray, np.ndarray]]
             },
             py::arg("operation"),
             py::arg("op_kwargs")      = py::dict(),
             py::arg("directory_path") = "",
             py::arg("return_affines") = false)
        .def("go_one_level_deep", &HighLevel::go_one_level_deep,
             py::arg("debug")          = false,
             py::arg("options")        = std::string("folders"),
             py::arg("directory_path") = "")
        .def("how_many_threads_available", &HighLevel::how_many_threads_available);
}