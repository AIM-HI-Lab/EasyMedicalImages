cmake_minimum_required(VERSION 3.30)
project(EasyMedicalImages LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT pybind11_DIR)
    execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
        EasyMedicalImages/ImagesRunner/HighLevel.cpp
        EasyMedicalImages/ImagesRunner/bindings.cpp
)

# Windows (MinGW): fold the GCC runtime into the .pyd so the wheel is
# self-contained. Safe here because a CPython-on-Windows process is MSVC-based
# and nothing else in it links libstdc++, so there is no second copy to clash with.
if (MINGW)
    target_link_options(_core PRIVATE
            -static-libgcc
            -static-libstdc++
            -static
    )
endif()

# NOTE: do NOT static-link libstdc++/libgcc on Linux.
# numpy and scipy load libstdc++ dynamically, so a statically-linked copy inside
# _core gives the process two libstdc++ runtimes and the extension crashes at its
# first iostream/exception use (segfault in how_many_threads_available at the
# std::cout). Link libstdc++ dynamically and let auditwheel (run automatically by
# cibuildwheel) handle manylinux portability. If auditwheel ever complains about
# GLIBCXX/CXXABI symbol versions coming from the C++23 build, bump the manylinux
# image (CIBW_MANYLINUX_X86_64_IMAGE) instead of reintroducing static linking.
#
# (The `if (UNIX AND NOT APPLE)` static-link block that used to live here was the
#  cause of the Linux segfault and has been removed.)

# Lands the .so as EasyMedicalImages/_core...  -> importable as EasyMedicalImages._core
install(TARGETS _core DESTINATION EasyMedicalImages)

#cmake_minimum_required(VERSION 3.30)
#project(EasyMedicalImages)
#
#set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#
##add_executable(HighLevel EasyMedicalImages/ImagesRunner/HighLevel.cpp
##        EasyMedicalImages/ImagesRunner/HighLevel.h)
#
#add_library(EasyMedicalImages
#        EasyMedicalImages/ImagesRunner/HighLevel.cpp
#)
#
#target_include_directories(EasyMedicalImages
#        PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}/EasyMedicalImages
#)