/*
 * test_abi.c — C-ABI sanity test suite for libllama_odin.
 *
 * Plain C, C-ABI only. Exercises the exported llama_* (upstream llama.cpp)
 * and llama_odin_* (wrapper) symbols through the public headers:
 *
 *   - llama_backend_init() / llama_backend_free()        (include/llama.h)
 *   - llama_odin_abi_version() == LLAMA_ODIN_ABI_VERSION (include/llama_odin.h)
 *   - llama_odin_silence_logs()
 *   - GPU backend device enumeration via ggml_backend_dev_*
 *
 * Build & run through CTest: `ctest --test-dir build -R test_abi --output-on-failure`
 * or directly: `build/tests/c/test_abi`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama_odin.h"

static int failures = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (cond) {                                                         \
            printf("ok    - %s\n", (msg));                                  \
        } else {                                                            \
            fprintf(stderr, "FAIL  - %s (at line %d)\n", (msg), __LINE__);  \
            failures++;                                                     \
        }                                                                   \
    } while (0)

int main(void) {
    /* 1. Backend lifecycle: init before any ggml/llama call. */
    llama_backend_init();
    CHECK(1, "llama_backend_init()");

    /* 2. Wrapper ABI version reported by symbol vs. compile-time header. */
    CHECK(llama_odin_abi_version() == LLAMA_ODIN_ABI_VERSION,
          "llama_odin_abi_version() == LLAMA_ODIN_ABI_VERSION (1)");

    /* 3. Internal stderr logging can be silenced without crashing. */
    llama_odin_silence_logs();
    CHECK(1, "llama_odin_silence_logs()");

    /* 4. GPU backend device enumeration via ggml_backend_dev_*. */
    size_t dev_count = ggml_backend_dev_count();
    printf("info  - %zu backend device(s) registered\n", dev_count);
    CHECK(dev_count >= 1, "ggml_backend_dev_count() >= 1");

    size_t gpu_count = 0;
    for (size_t i = 0; i < dev_count; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        CHECK(dev != NULL, "ggml_backend_dev_get() returns non-NULL");

        const char * name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);
        enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        size_t mem_free = 0, mem_total = 0;
        ggml_backend_dev_memory(dev, &mem_free, &mem_total);
        printf("info  - device[%zu]: name='%s' desc='%s' type=%d "
               "free=%zu total=%zu\n",
               i, name ? name : "(null)", desc ? desc : "(null)",
               (int) type, mem_free, mem_total);

        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_count++;
        }
    }
    /* Metal is forced on for the macOS arm64 build, so a GPU device
     * (dedicated or integrated memory) must be discoverable. */
    CHECK(gpu_count >= 1, "at least one GPU/IGPU backend device enumerated");

    /* 5. Symmetric teardown: llama_backend_free() must not crash. */
    llama_backend_free();
    CHECK(1, "llama_backend_free()");

    if (failures != 0) {
        fprintf(stderr, "%d assertion(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("PASS  - all C-ABI sanity assertions passed\n");
    return EXIT_SUCCESS;
}