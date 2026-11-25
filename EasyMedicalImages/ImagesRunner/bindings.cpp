#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
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

// ---- Failure reporting (the run methods' on_error argument) -----------------

enum class OnError { Warn, Raise, Ignore };

// Checked before any work starts, so a typo fails at once, not after a long run.
static OnError parse_on_error(const std::string& s) {
    if (s == "warn")   return OnError::Warn;
    if (s == "raise")  return OnError::Raise;
    if (s == "ignore") return OnError::Ignore;
    throw py::value_error("on_error must be \"warn\", \"raise\" or \"ignore\", not \"" + s + "\"");
}

// Folder paths come from std::filesystem::path::string(): UTF-8 on Linux/macOS
// and with MinGW, but the ANSI code page with MSVC. Decode as UTF-8 when valid,
// else with the platform's narrow encoding, so a case folder with an accented
// name can't turn the failure report itself into a crash.
static py::str path_to_py(const std::string& s) {
    const auto n = static_cast<Py_ssize_t>(s.size());
    PyObject* u = PyUnicode_DecodeUTF8(s.data(), n, nullptr);
    if (!u) {
        PyErr_Clear();
#ifdef _WIN32
        u = PyUnicode_DecodeMBCS(s.data(), n, "replace");
#else
        u = PyUnicode_DecodeFSDefaultAndSize(s.data(), n);
#endif
        if (!u) throw py::error_already_set();
    }
    return py::reinterpret_steal<py::str>(u);
}

// Applies the on_error policy to a finished run. `results` is the list the call
// is about to return; in raise mode it rides along on the exception so the cases
// that did load aren't thrown away. The classes live in EasyMedicalImages/errors.py
// so users can import and catch them.
static void report_failures(const std::vector<CaseFailure>& failures, size_t n_cases,
                            OnError policy, const py::list& results) {
    if (failures.empty() || policy == OnError::Ignore) return;

    py::module_ errors = py::module_::import("EasyMedicalImages.errors");
    py::list records;
    for (const auto& f : failures)
        records.append(errors.attr("CaseFailure")(path_to_py(f.folder), f.reason));

    // The message lists the first few; .failures always has every one.
    constexpr size_t kShown = 10;
    py::str msg = py::str("{} of {} case(s) failed and were left out of the results:")
                      .format(failures.size(), n_cases);
    for (size_t i = 0; i < failures.size() && i < kShown; ++i)
        msg = py::str("{}\n  {}: {}").format(msg, records[i].attr("folder"), records[i].attr("reason"));
    if (failures.size() > kShown)
        msg = py::str("{}\n  ... and {} more (full list in .failures)").format(msg, failures.size() - kShown);

    if (policy == OnError::Raise) {
        py::object cls = errors.attr("CaseFailureError");
        py::object exc = cls(msg, records, results);
        PyErr_SetObject(cls.ptr(), exc.ptr());
        throw py::error_already_set();
    }
    // Called from C++ there is no Python frame for this binding, so the default
    // stacklevel attributes the warning to the line that called the run method.
    py::module_::import("warnings").attr("warn")(errors.attr("CaseFailureWarning")(msg, records));
}

// mod_gil_not_used: declare free-threading support so importing _core on a
// 3.14t/3.15t interpreter doesn't silently re-enable the GIL. No effect on
// regular (GIL) builds.
PYBIND11_MODULE(_core, m, py::mod_gil_not_used()) {
    m.doc() = "C++ core for EasyMedicalImages";

    py::class_<HighLevel>(m, "HighLevel")
        // Factory instead of py::init<...> so python_exe is set exactly once, here.
        // The run methods used to rewrite it on every call, which raced with the
        // worker threads of any other call running on the same object.
        .def(py::init([](std::string path_to_folder, std::string image_path, std::string seg_path,
                         bool verbose, float thread_deadline_time, bool verbose_workers) {
                auto hl = std::make_unique<HighLevel>(std::move(path_to_folder), std::move(image_path),
                                                      std::move(seg_path), verbose, thread_deadline_time,
                                                      verbose_workers);
                hl->python_exe = py::module_::import("sys").attr("executable").cast<std::string>();
                return hl;
            }),
            py::arg("path_to_folder"),
            py::arg("image_path")           = "image.nii.gz",
            py::arg("seg_path")             = "tumor_segmentation_v2.nii.gz",
            py::arg("verbose")              = false,
            py::arg("thread_deadline_time") = 100.1f,
            py::arg("verbose_workers")      = true)
        .def("run_on_directory",[](HighLevel& self, const std::string& directory_path, bool return_affines,
                                   const std::string& on_error) {
                const OnError policy = parse_on_error(on_error);
                RunResult res;
                {
                    py::gil_scoped_release release;
                    res = self.run_on_directory(directory_path, return_affines);
                }
                const size_t n_cases = res.pairs.size() + res.failures.size();
                py::list out;
                for (auto& pr : res.pairs)
                    out.append(py::make_tuple(side_to_py(std::move(pr.first),  return_affines),
                                              side_to_py(std::move(pr.second), return_affines)));
                report_failures(res.failures, n_cases, policy, out);
                return out;  // off: list[(img,seg)]; on: list[((img,aff),(seg,aff))]
             },
             py::arg("directory_path") = "",
             py::arg("return_affines") = false,
             py::kw_only(),
             py::arg("on_error")       = "warn")
        .def("processed_run_on_directory",
        [](HighLevel& self, const std::string& operation, py::dict op_kwargs,
           const std::string& directory_path, bool return_affines, const std::string& on_error) {
                const OnError policy = parse_on_error(on_error);
                // Serialize kwargs -> JSON here, while we still hold the GIL.
                std::string op_kwargs_json =
                    py::module_::import("json").attr("dumps")(op_kwargs).cast<std::string>();
                RunResult res;
                {
                    py::gil_scoped_release release;
                    res = self.processed_run_on_directory(operation, op_kwargs_json, directory_path, return_affines);
                }
                const size_t n_cases = res.pairs.size() + res.failures.size();
                py::list out;
                for (auto& pr : res.pairs)
                    out.append(py::make_tuple(volume_to_numpy(std::move(pr.first)),
                    volume_to_numpy(std::move(pr.second))));
                report_failures(res.failures, n_cases, policy, out);
                return out;  // list[tuple[np.ndarray, np.ndarray]]
             },
             py::arg("operation"),
             py::arg("op_kwargs")      = py::dict(),
             py::arg("directory_path") = "",
             py::arg("return_affines") = false,
             py::kw_only(),
             py::arg("on_error")       = "warn")
        .def("go_one_level_deep", &HighLevel::go_one_level_deep,
             py::arg("debug")          = false,
             py::arg("options")        = std::string("folders"),
             py::arg("directory_path") = "")
        .def("how_many_threads_available", &HighLevel::how_many_threads_available);
}