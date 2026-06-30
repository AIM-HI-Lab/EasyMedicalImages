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

if (MINGW)
    target_link_options(_core PRIVATE
            -static-libgcc
            -static-libstdc++
            -static
    )
endif()

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