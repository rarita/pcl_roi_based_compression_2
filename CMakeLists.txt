macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()
# CMakeList.txt: проект CMake для pcl_roi_based_compression_2; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)

# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/libnoise/cmake/Modules/")
# set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")

project ("pcl_roi_based_compression_2")

find_package (PCL CONFIG REQUIRED) # COMPONENTS common io
find_package (mlpack CONFIG REQUIRED)
find_package (armadillo CONFIG REQUIRED)
include("${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/x64-windows/share/libnoise/FindLibNoise.cmake")
# find_package("${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/x64-windows/share/libnoise/")
# find_package(LibNoise)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${MLPACK_INCLUDE_DIRS})
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${LIBNOISE_INCLUDE_DIR})

add_compile_options("/openmp")

# Скрипт не находит LIBNOISE_DIRS, но предполагаю что все лежит там же,
# где лежат другие либы, скомпилированные vcproj
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${ARMADILLO_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
# Так как у скрипта какие-то мистические проблемы с обнаружением libnoiseutils
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LIBNOISE_NOISEUTILS_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/x64-windows/debug/lib/noiseutils.lib")
else()
    set(LIBNOISE_NOISEUTILS_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/x64-windows/lib/noiseutils.lib")
endif()
print_all_variables()

# Добавьте источник в исполняемый файл этого проекта.
add_executable (pcl_roi_based_compression_2 "pcl_roi_based_compression_2.cpp" "pcl_roi_based_compression_2.h")
target_link_libraries(pcl_roi_based_compression_2 PUBLIC ${PCL_LIBRARIES} mlpack::mlpack armadillo ${LIBNOISE_LIBRARIES} ${LIBNOISE_NOISEUTILS_LIBRARY})
# target_include_directories(pcl_roi_based_compression_2 PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/kernel-density")
# TODO: Добавьте тесты и целевые объекты, если это необходимо.
