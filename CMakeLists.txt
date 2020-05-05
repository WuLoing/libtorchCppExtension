cmake_minimum_required (VERSION 3.10)

project(libtorchCppExtension CXX)
set(LIBRARY_TARGET_NAME libtorchCppExtension)

set(Torch_DIR "3rd/libtorch/share/cmake/Torch")
find_package(Torch)
if(TORCH_FOUND)
	include_directories(${TORCH_INCLUDE_DIRS})
    message(STATUS "The libtorch lib is found!") 
endif()

set(${LIBRARY_TARGET_NAME}_SRC 
    libtorch_cpp_extension_test.h
    libtorch_cpp_extension_func.h
    libtorch_cpp_extension_module.h
    main.cpp   
)

add_executable(${LIBRARY_TARGET_NAME} ${${LIBRARY_TARGET_NAME}_SRC})
target_link_libraries(${LIBRARY_TARGET_NAME} ${TORCH_LIBRARIES})
include_directories("../")

if(MSVC)
    include(${CMAKE_SOURCE_DIR}/cmake/CopyTorchDlls.cmake)
    copy_torch_dlls(${LIBRARY_TARGET_NAME})
endif(MSVC)