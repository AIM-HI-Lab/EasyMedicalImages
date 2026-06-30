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

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for EasyMedicalImages";

    py::class_<HighLevel>(m, "HighLevel")
        .def(py::init<std::string, std::string, std::string, bool, float>(),
             py::arg("path_to_folder"),
             py::arg("image_path")           = "image.nii.gz",
             py::arg("seg_path")             = "tumor_segmentation_v2.nii.gz",
             py::arg("verbose")              = false,
             py::arg("thread_deadline_time") = 66.0f)
        .def("run_on_directory",
        [](HighLevel& self, const std::string& directory_path) {
                self.python_exe = py::module_::import("sys").attr("executable").cast<std::string>();
                std::vector<std::pair<Volume, Volume>> pairs;
                {
                    py::gil_scoped_release release;
                    pairs = self.run_on_directory(directory_path);
                }
                py::list out;
                for (auto& pr : pairs)
                    out.append(py::make_tuple(volume_to_numpy(std::move(pr.first)),
                    volume_to_numpy(std::move(pr.second))));
                return out;  // list[tuple[np.ndarray, np.ndarray]]
             },
             py::arg("directory_path") = "")
        .def("how_many_threads_available", &HighLevel::how_many_threads_available,
             py::arg("debug") = true);
}