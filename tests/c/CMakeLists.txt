# C-ABI sanity test suite for libllama_odin (plain C, C-ABI only).
# enable_testing() is called from the root CMakeLists.txt so the test is
# discoverable from the top-level build directory.

add_executable(test_abi test_abi.c)
target_include_directories(test_abi PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/vendor/llama.cpp/ggml/include)
target_link_libraries(test_abi PRIVATE llama_odin::llama_odin)

add_test(NAME test_abi COMMAND test_abi COMMAND_EXPAND_LISTS)
set_tests_properties(test_abi PROPERTIES
    ENVIRONMENT "DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:llama_odin>")