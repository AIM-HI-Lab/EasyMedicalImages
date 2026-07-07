#define NOMINMAX

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "HighLevel.h"

#include <iostream>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <thread>
#include <cmath>
#include <optional>
#include <cstdint>
#include <cstdio>
#include <future>
#include <chrono>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>
#else
#  include <spawn.h>
#  include <unistd.h>
#  include <sys/wait.h>
#  include <signal.h>
extern char **environ;
#endif

namespace fs = std::filesystem;

HighLevel::HighLevel(std::string path_to_folder, std::string image_path, std::string seg_path, bool verbose, float thread_deadline_time, bool verbose_workers):
    path_to_folder(std::move(path_to_folder)),
    hl_img_path(std::move(image_path)),
    hl_seg_path(std::move(seg_path)),
    thread_deadline_time(thread_deadline_time),
    verbose_workers(verbose_workers),
    verbose(verbose) {
    // img_seg_pairs default-constructed empty, so there's nothing to clear here.
    }

std::vector<std::pair<Volume, Volume>> HighLevel::run_on_directory(const std::string& directory_path) {
    std::vector<std::string> all_data = _go_one_level_deep(/*debug=*/false, "folders", directory_path);
    _get_all_images_and_segs_from_path(all_data, /*return_affines=*/false, /*verbose_workers=*/verbose_workers);
    return std::move(img_seg_pairs);  // move, not copy, this vector will probably be huge
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

int HighLevel::how_many_threads_available(){
    unsigned int threads = 0;
    #ifdef __linux__
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0)
            threads = static_cast<unsigned int>(CPU_COUNT(&set));  // respects the SLURM cpuset
    #endif
    if (threads == 0) threads = std::thread::hardware_concurrency();  // fallback
    if (threads==0) { // This works becuase if the threads are still 0 after running, it really is detecting 0 threads.
        threads = 1;
        std::cout<<"Strangely, the computer doesn't think you have any threads. Attempting to use single thread mode."<<std::endl;
    }
    std::cout << "Available hardware threads: " << threads << "\n";
    return threads;
}

namespace {
    #ifdef _WIN32
    constexpr const char* kPythonCmd = "python";
    #else
    constexpr const char* kPythonCmd = "python3";
    #endif

    // A launched child + the read end of its stdout. Unlike popen(), we keep a
    // handle to the *process itself* so we can kill a hung one (popen hands back
    // only a FILE*, with no way to reach the child -> no way to enforce a timeout).
    struct Subprocess {
        std::FILE* stream = nullptr;   // child's stdout, as a FILE* (read_volume reads this)
    #ifdef _WIN32
        HANDLE process = nullptr;
        HANDLE thread  = nullptr;
        HANDLE job     = nullptr;      // Job Object: kills the whole tree on terminate
    #else
        pid_t  pid = -1;               // also the process-group id (we make the child a leader)
    #endif
    };

    #ifndef _WIN32
    // POSIX: pipe + posix_spawnp (exec the program directly -- no /bin/sh wrapper,
    // so SIGKILL hits the program, not a shell that would orphan it). The child is
    // put in its OWN process group so we can signal the whole tree on timeout.
    bool spawn(const std::vector<std::string>& argv, Subprocess& sp) {
        int fds[2];
        if (pipe(fds) != 0) return false;

        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_addclose(&fa, fds[0]);                 // child won't read
        posix_spawn_file_actions_adddup2 (&fa, fds[1], STDOUT_FILENO);  // child stdout -> pipe
        posix_spawn_file_actions_addclose(&fa, fds[1]);

        posix_spawnattr_t at;
        posix_spawnattr_init(&at);
        posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&at, 0);                             // new group, gid == child pid

        std::vector<char*> cargv;                                       // built before spawn (no work in child)
        cargv.reserve(argv.size() + 1);
        for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
        cargv.push_back(nullptr);

        pid_t pid = 0;
        const int rc = posix_spawnp(&pid, cargv[0], &fa, &at, cargv.data(), environ);
        posix_spawn_file_actions_destroy(&fa);
        posix_spawnattr_destroy(&at);
        close(fds[1]);                                                  // parent never writes

        if (rc != 0) { close(fds[0]); return false; }
        sp.pid = pid;
        sp.stream = fdopen(fds[0], "rb");
        if (!sp.stream) { close(fds[0]); return false; }
        return true;
    }

    // Negative pid -> deliver to the whole process group, so a Python worker that
    // itself spawned helpers can't keep the pipe open and stall our reader.
    void kill_child(Subprocess& sp) {
        if (sp.pid > 0) ::kill(-sp.pid, SIGKILL);
    }

    int reap(Subprocess& sp) {
        if (sp.stream) { std::fclose(sp.stream); sp.stream = nullptr; }
        int status = 0;
        if (sp.pid > 0) { waitpid(sp.pid, &status, 0); sp.pid = -1; }
        if (WIFSIGNALED(status)) return -WTERMSIG(status);   // negative -> killed by signal
        if (WIFEXITED(status))   return  WEXITSTATUS(status);
        return -999;
    }

    #else
    // Windows: CreatePipe + CreateProcess. The child is placed in a Job Object so
    // TerminateJobObject kills it AND any descendants (the analogue of the POSIX
    // process-group kill above).
    std::string win_quote(const std::string& a) {
        if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
        std::string out = "\"";
        size_t bs = 0;
        for (char c : a) {
            if (c == '\\') { ++bs; out.push_back(c); continue; }
            if (c == '"')  { out.append(bs + 1, '\\'); out.push_back('"'); }
            else            { out.push_back(c); }
            bs = 0;
        }
        out.append(bs, '\\');
        out.push_back('"');
        return out;
    }

    bool spawn(const std::vector<std::string>& argv, Subprocess& sp) {
        SECURITY_ATTRIBUTES sa{}; sa.nLength = sizeof(sa); sa.bInheritHandle = TRUE;
        HANDLE rd = nullptr, wr = nullptr;
        if (!CreatePipe(&rd, &wr, &sa, 0)) return false;
        SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);              // child must not inherit read end

        std::string cmdline;
        for (size_t i = 0; i < argv.size(); ++i) { if (i) cmdline.push_back(' '); cmdline += win_quote(argv[i]); }
        std::vector<char> buf(cmdline.begin(), cmdline.end()); buf.push_back('\0');  // CreateProcessA may modify

        STARTUPINFOA si{}; si.cb = sizeof(si);
        si.dwFlags    = STARTF_USESTDHANDLES;
        si.hStdOutput = wr;
        si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
        si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);

        PROCESS_INFORMATION pi{};
        // CREATE_SUSPENDED so we can put it in the job before it can spawn anything.
        BOOL ok = CreateProcessA(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP, nullptr, nullptr, &si, &pi);
        CloseHandle(wr);
        if (!ok) { CloseHandle(rd); return false; }

        HANDLE job = CreateJobObjectA(nullptr, nullptr);
        if (job) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli{};
            jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(job, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
            AssignProcessToJobObject(job, pi.hProcess);
        }
        ResumeThread(pi.hThread);

        sp.process = pi.hProcess;
        sp.thread  = pi.hThread;
        sp.job     = job;

        int fd = _open_osfhandle(reinterpret_cast<intptr_t>(rd), _O_RDONLY | _O_BINARY);
        if (fd == -1) { CloseHandle(rd); return false; }
        sp.stream = _fdopen(fd, "rb");                                 // closing this closes fd -> rd
        if (!sp.stream) { _close(fd); return false; }
        return true;
    }

    void kill_child(Subprocess& sp) {
        if (sp.job) TerminateJobObject(sp.job, 1);                     // kills child + descendants
        else if (sp.process) TerminateProcess(sp.process, 1);
    }

    int reap(Subprocess& sp) {
        if (sp.stream) { std::fclose(sp.stream); sp.stream = nullptr; }
        DWORD code = 0;
        if (sp.process) {
            WaitForSingleObject(sp.process, INFINITE);
            GetExitCodeProcess(sp.process, &code);
            CloseHandle(sp.process); sp.process = nullptr;
        }
        if (sp.thread) { CloseHandle(sp.thread); sp.thread = nullptr; }
        if (sp.job)    { CloseHandle(sp.job);    sp.job    = nullptr; }
        return static_cast<int>(code);
    }
    #endif

    // // Per-case wall-clock budget. A worker that blows this is killed and skipped.
    // const auto kLoadTimeout = std::chrono::seconds(static_cast<float>(thread_deadline_time));

    inline bool read_exact(std::FILE* fp, void* dst, size_t bytes) {
        return std::fread(dst, 1, bytes, fp) == bytes;
    }

    // int32 ndim | int64 shape[ndim] | float32 data[prod(shape)]
    // little-endian, C-order.
    // This can also produce an affine if that is inputted
    bool read_volume(std::FILE* fp, Volume& vol, bool want_affine) {
        int32_t ndim = 0;
        if (!read_exact(fp, &ndim, sizeof(ndim)) || ndim <= 0)
            return false;

        vol.shape.resize(static_cast<size_t>(ndim));
        if (!read_exact(fp, vol.shape.data(), static_cast<size_t>(ndim) * sizeof(int64_t)))
            return false;

        int64_t count = 1;
        for (int64_t d : vol.shape) {
            if (d < 0) return false;
            count *= d;
        }

        vol.data.resize(static_cast<size_t>(count));
        if (!read_exact(fp, vol.data.data(), static_cast<size_t>(count) * sizeof(float)))
            return false;

        if (want_affine) // 4x4 float64, written by Python right after the volume data
            return read_exact(fp, vol.affine.data(), vol.affine.size() * sizeof(double));

        return true;
    }

    // What the reader thread hands back. The two volumes live here (not in the
    // worker's frame) so they survive being moved out after the deadline check.
    struct ReadResult {
        bool   ok = false;
        Volume img;
        Volume seg;
    };

}

void HighLevel::_fan_out_workers(const std::vector<std::string>& all_data, bool return_affines, bool verbose_workers, const std::string& operation, const std::string& op_kwargs_json) {
    const auto kLoadTimeout = std::chrono::duration<float>(this->thread_deadline_time);
    const int  hw   = how_many_threads_available();
    const long want = std::lround(static_cast<double>(hw) * 0.80);
    const size_t batch_size = static_cast<size_t>(std::max(1L, want)); // batch size is how many threads we are running, if your confused.
    if (verbose_workers){
        std::cout << "Using up to " << batch_size << " concurrent worker(s)\n";
    }

    // all_data is a huge list of all of the patient folder names. Each thread to 1 independent folder. This is to fight race conditions
    // and input order is preserved.
    std::vector<std::optional<std::pair<Volume, Volume>>> results(all_data.size());

    for (size_t batch_start = 0; batch_start < all_data.size(); batch_start += batch_size) {
        std::vector<std::thread> workers;

        for (size_t i = 0; i < batch_size && (batch_start + i) < all_data.size(); ++i) {
            const size_t      idx      = batch_start + i;
            const std::string path_str = all_data[idx];  // copy: outlives the call frame

            if (verbose_workers)
                std::cout << "Worker " << i << " -> " << path_str << '\n';

            workers.emplace_back([this, path_str, idx, &results, verbose_workers, return_affines, kLoadTimeout, operation, op_kwargs_json]() {
                // argv (not a shell string): posix_spawnp/CreateProcess take it
                // directly, so paths with spaces need no quoting and SIGKILL
                // runs on Python itself rather than a /bin/sh wrapper.
                const std::string interp = this->python_exe.empty()
                           ? std::string(kPythonCmd)
                           : this->python_exe;
                std::vector<std::string> argv = {
                    interp, //std::string(kPythonCmd)
                    "-P",
                    "-m", "EasyMedicalImages.ImagesRunner.PythonThreadWrapper",
                    "--folder", path_str,
                    "--img",    hl_img_path,
                    "--seg",    hl_seg_path,
                };
                if (return_affines) {
                    argv.push_back("--return_affine");
                }
                if (this->verbose) {
                    argv.push_back("--verbose");
                }
                if (!operation.empty()) {
                    argv.push_back("--operation");
                    argv.push_back(operation);
                    argv.push_back("--op_kwargs");
                    argv.push_back(op_kwargs_json);
                }

                Subprocess sp;
                if (!spawn(argv, sp)) {
                    std::cerr << "Failed to launch Python for " << path_str << '\n';
                    return;
                }

                // read_volume() is a *blocking* read, so we run it on a helper
                // thread and let THIS thread enforce the wall-clock deadline.
                std::future<ReadResult> reader = std::async(std::launch::async, [&sp, return_affines]() {
                    ReadResult r;
                    r.ok = read_volume(sp.stream, r.img, return_affines) && read_volume(sp.stream, r.seg, return_affines);
                    if (r.ok) {                       // drain trailing bytes on success
                        char sink[4096];
                        while (std::fread(sink, 1, sizeof(sink), sp.stream) > 0) { }
                    }
                    return r;
                });

                if (reader.wait_for(kLoadTimeout) != std::future_status::ready) {
                    // Over budget: kill the tree (-> stdout closes -> the read
                    // hits EOF), join the reader, reap, and move on. The slot
                    // stays empty, so this case is simply skipped.
                    std::cerr << "Timeout (>" << kLoadTimeout.count() << "s): " << path_str << " -- killing subprocess and skipping\n";
                    kill_child(sp);
                    reader.wait();                    // returns promptly once EOF unblocks the read
                    reap(sp);
                    return;
                }

                ReadResult r  = reader.get();         // ready -> returns immediately
                const int  rc = reap(sp);             // rc == 0 means clean exit

                if (r.ok && rc == 0) {
                    if (verbose_workers)
                        std::cout << "Loaded " << path_str << "  img.ndim=" << r.img.shape.size() << "  seg.ndim=" << r.seg.shape.size() << '\n'<< '\n';
                    results[idx] = std::make_pair(std::move(r.img), std::move(r.seg));
                } else {
                    std::cerr << "Python failed/unreadable for " << path_str << " (rc=" << rc << ")\n";
                }
            });
        }

        for (auto& worker : workers)
            worker.join();
    }

    // Compact to a dense list of successes, in order, matches Python's img_seg_pairs.
    img_seg_pairs.clear();
    img_seg_pairs.reserve(results.size());
    for (auto& slot : results)
        if (slot) img_seg_pairs.push_back(std::move(*slot));

    std::cout << "Collected " << img_seg_pairs.size() << " / " << all_data.size() << " cases\n";
}

// Backwards-compatible thin wrapper: the original plain-load fan-out is just the
// generic one with no operation (so the worker argv is byte-for-byte what it was).
void HighLevel::_get_all_images_and_segs_from_path(const std::vector<std::string>& all_data, bool return_affines, bool verbose_workers) {
    _fan_out_workers(all_data, return_affines, verbose_workers, /*operation=*/"", /*op_kwargs_json=*/"{}");
}

// Generic sibling of run_on_directory (see header). Same directory walk + fan-out,
// but each worker applies `operation` in place before streaming its volumes back.
std::vector<std::pair<Volume, Volume>> HighLevel::apply_operation_to_directory(const std::string& operation, const std::string& op_kwargs_json, const std::string& directory_path, bool return_affines) {
    std::vector<std::string> all_data = _go_one_level_deep(/*debug=*/false, "folders", directory_path);
    _fan_out_workers(all_data, return_affines, verbose_workers, operation, op_kwargs_json);
    return std::move(img_seg_pairs);  // moved out; member left empty (this vector can be huge)
}

int run_test() {
    // This is a manual way to test any changes made to the above class
    std::cout << "Enter the path to the folder: ";
    std::string input_path;
    std::getline(std::cin, input_path);

    HighLevel high_level(input_path, "image.nii.gz", "tumor_segmentation_v2.nii.gz", /*verbose=*/true);
    high_level.run_on_directory();
    return 0;
}