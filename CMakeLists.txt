cmake_minimum_required(VERSION 3.15)
project(llama_odin LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Global build settings
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Consolidate the vendored llama.cpp/ggml objects into the llama_odin binary
# targets: build upstream as static libraries, then link them into a single
# self-contained libllama_odin shared library (and the static counterpart).
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Force Metal acceleration with shaders compiled into binary bytecode at build
# time on macOS (zero runtime search for .metallib files).
if(APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Vendored upstream llama.cpp (provides llama, ggml, ggml-base, ggml-cpu,
# ggml-metal targets)
# ---------------------------------------------------------------------------

add_subdirectory(vendor/llama.cpp)

# ---------------------------------------------------------------------------
# llama_odin library targets
# ---------------------------------------------------------------------------

add_library(llama_odin SHARED src/llama_odin_ext.cpp)
add_library(llama_odin::llama_odin ALIAS llama_odin)

add_library(llama_odin_static STATIC src/llama_odin_ext.cpp)
if(WIN32)
    # Windows gives both targets the same .lib suffix; disambiguate.
    set_target_properties(llama_odin_static PROPERTIES OUTPUT_NAME llama_odin_static)
else()
    set_target_properties(llama_odin_static PROPERTIES OUTPUT_NAME llama_odin)
endif()

foreach(target llama_odin llama_odin_static)
    target_include_directories(${target}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/src)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
endforeach()

# Link with the vendored upstream library (static; compiled with
# CMAKE_POSITION_INDEPENDENT_CODE=ON). Symbols from the static archives are
# re-exported by the llama_odin shared library through default visibility.
target_link_libraries(llama_odin PRIVATE llama)
target_link_libraries(llama_odin_static PRIVATE llama)

if(APPLE)
    set_target_properties(llama_odin PROPERTIES
        MACOSX_RPATH ON
        INSTALL_NAME_DIR "@rpath"
        SOVERSION 1
        VERSION 1.0.0)
    target_link_libraries(llama_odin PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework Accelerate")
endif()

# ---------------------------------------------------------------------------
# Install / export rules
# ---------------------------------------------------------------------------

include(GNUInstallDirs)

install(TARGETS llama_odin_static
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama_odin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export the shared target for CMake consumers; the static target keeps its
# dependency on the vendored llama objects private and stays out of the
# install-export set.
install(TARGETS llama_odin EXPORT llama_odin-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT llama_odin-targets
    FILE llama_odinTargets.cmake
    NAMESPACE llama_odin::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama_odin)

# ---------------------------------------------------------------------------
# C-ABI sanity test suite (tests/c)
# ---------------------------------------------------------------------------

enable_testing()
add_subdirectory(tests/c)