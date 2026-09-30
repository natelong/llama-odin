# Specification: Standalone `llama-odin` Library & Odin Package

This document is a self-contained, turnkey specification for building the external **`llama-odin`** repository from scratch. It is designed to be provided directly to an autonomous agent or developer tasked with implementing the standalone library.

---

## 1. Project Overview & Mission

**Repository Name**: `llama-odin`  
**Purpose**: Provide a production-grade, precompiled C-ABI shared/static library and native Odin package wrapping upstream `llama.cpp` with embedded Apple Silicon Metal acceleration, Multi-Token Prediction (MTP) speculative decoding extensions, and high-level Odin generator abstractions.

### Core Objectives
1. **Self-Contained Metal Acceleration**: Metal kernel data must be compiled into the library binary itself (`GGML_METAL_EMBED_LIBRARY=ON`) with zero runtime search for `.metallib` or `.metal` files. In the vendored fork this data is preprocessed kernel source (not precompiled bytecode), embedded in `__DATA,__ggml_metallib` sections and JIT-compiled at Metal device init — see §3.2.1 for the exact mechanism and trade-offs.
2. **Unified C-ABI**: Re-export all necessary `llama.cpp` and `ggml` symbols alongside a clean `extern "C"` staging shim for speculative decoding (`llama_ext`), eliminating C++ mangling issues.
3. **Idiomatic Odin Package**: Provide both low-level C foreign declarations (`package llama_c` / `llama/c`) and a high-level generator package (`package llama`).
4. **Precompiled CI Releases**: GitHub Actions workflow that produces versioned release archives for macOS `arm64` (Apple Silicon) with SHA-256 checksums.

---

## 2. Target Repository Layout

```
llama-odin/
├── CMakeLists.txt                  # Top-level build driver (dylib, static lib, Metal embed)
├── include/
│   ├── llama_odin.h                # Unified public C-ABI header
│   └── llama.h                     # Upstream llama.cpp public header
├── src/
│   ├── llama_odin_ext.cpp          # C++ staging shim (MTP functions & helpers)
│   └── (upstream llama.cpp source / submodule)
├── odin/
│   ├── llama/
│   │   ├── llama.odin              # High-level generator, device enumeration, sampling
│   │   └── mtp.odin                # Multi-Token Prediction speculative driver
│   └── c/
│       └── llama_c.odin            # Foreign import declarations for libllama_odin
├── tests/
│   ├── c/
│   │   └── test_abi.c              # C-ABI sanity and symbol verification test
│   └── odin/
│       └── test_generator.odin     # Standalone Odin tests (mock and model evaluation)
├── .github/
│   └── workflows/
│       ├── test.yml                # CI test validation
│       └── release.yml             # Automated multi-platform release packager
├── LICENSE                         # MIT
└── README.md                       # Documentation & integration instructions
```

---

## 3. Upstream `llama.cpp` Integration & CMake Architecture

### 3.1 Upstream Source
- Track upstream `llama.cpp` via Git submodule (pinned to a stable commit, e.g. b3800+) or vendored in `vendor/llama.cpp`.
- Include `ggml` and internal headers required by the staging shim (`src/llama-ext.h`).

### 3.2 Build Requirements (`CMakeLists.txt`)
The root `CMakeLists.txt` must configure:
```cmake
cmake_minimum_required(VERSION 3.15)
project(llama_odin LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Force Metal acceleration with embedded Metal kernel libraries on macOS
# (see section 3.2.1 — the vendor embeds kernel source, JIT-compiled at device init)
if(APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
endif()

# Build targets:
# 1. libllama_odin.dylib (Dynamic library with @rpath install name)
# 2. libllama_odin.a     (Consolidated static library via libtool / ar)
```

#### 3.2.1 Metal Embedding Mechanism (implementation note for §3.2)

`GGML_METAL_EMBED_LIBRARY=ON` in the vendored fork (pinned commit `efa28e950`)
does **not** embed precompiled `.metallib` *bytecode*. Instead, at build time its
CMake logic concatenates each `kernels/<name>.metal` source with the shared
headers (`ggml-common.h`, `ggml-metal-impl.h`, per-kernel headers), strips the
internal includes, and embeds the resulting preprocessed Metal **source** into
the binary: one `__DATA,__ggml_metallib` section per kernel group with
`ggml_metallib_<name>_{start,end}` symbol pairs (~20 groups, ~1.8 MB total in
`libllama_odin` at the pinned commit). At Metal device init, the fork's library
init reads these embedded sources and JIT-compiles one `MTLLibrary` per kernel
group via `newLibraryWithSource`, dispatched in parallel
(`ggml_metal_library_compile_all`). Measured device-init compile wall time on
Apple Silicon (M5) with the pinned fork: ~0.02–0.2 s (older forks and serial
builds have shown multi-second JIT stalls; this one parallelizes per kernel group).

The resulting `libllama_odin` binaries are fully self-contained at runtime: no
`.metallib` or `.metal` file needs to exist on disk and no file search is
performed (the bundle-resource / `GGML_METAL_PATH_RESOURCES` lookup only runs in
the `GGML_METAL_EMBED_LIBRARY=OFF` build, which llama-odin never uses).

**Deviation from strict "bytecode at build time" compliance**: precompiled
`.metallib` bytecode embedding would require (a) the Apple `metal`/`metallib`
compiler toolchain (`xcrun -sdk macosx metal` — a full Xcode toolchain; it is
absent on Command-Line-Tools-only build hosts) and (b) a code path in the vendored
fork that constructs an `MTLLibrary` from in-memory bytecode. The pinned fork has
no such path: bytecode loading exists only as `newLibraryWithURL` on a
`default.metallib` *file on disk* (a runtime file search, gated behind
`!GGML_METAL_EMBED_LIBRARY`). Because `vendor/llama.cpp` is a pinned, read-only
fork, llama-odin keeps the embedded-source mechanism and this note documents the
deviation; the trade-off is a one-time (per-process) parallel JIT compile at Metal
device init in exchange for zero runtime file dependency and robustness across
macOS/Xcode SDK versions (no bytecode built against one SDK to be validated on
another).

**Revisit trigger** (tracked in the project backlog): switch to build-time
`.metallib` bytecode embedding when any of the following holds:
- the vendored fork is upgraded to a commit that provides an embedded-bytecode
  loader usable without submodule edits (e.g. a `newLibraryWithData`-style path);
- CI/release hosts provide Xcode tooling and a wrapper layer can replace the
  embedded source with bytecode without touching `vendor/llama.cpp`;
- measured device-init JIT cost becomes unacceptable on target hardware, or a
  macOS release requires offline shader validation not achievable from source.

### 3.3 Dynamic Library Configuration
On macOS, set the install name to support `@rpath`:
```cmake
set_target_properties(llama_odin PROPERTIES
    MACOSX_RPATH ON
    INSTALL_NAME_DIR "@rpath"
    SOVERSION 1
    VERSION 1.0.0
)
target_link_libraries(llama_odin PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework Accelerate"
)
```

---

## 4. The C-ABI Surface (`include/llama_odin.h`)

The staging shim wraps the C++-mangled functions from `llama-ext.h` in clean C linkage:

```c
#ifndef LLAMA_ODIN_H
#define LLAMA_ODIN_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LLAMA_ODIN_ABI_VERSION 1

/* Report ABI wrapper version */
uint32_t llama_odin_abi_version(void);

/* Silence ggml and llama.cpp internal stderr logging */
void llama_odin_silence_logs(void);

/* Multi-Token Prediction (MTP) Speculative Staging API */
void llama_odin_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);
float * llama_odin_get_embeddings_nextn(struct llama_context * ctx);
float * llama_odin_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);
void llama_odin_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);
struct llama_context * llama_odin_get_ctx_other(struct llama_context * ctx);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_ODIN_H */
```

### Implementation (`src/llama_odin_ext.cpp`):
```cpp
#include "llama_odin.h"
#include "llama-ext.h"
#include <iostream>

extern "C" {

uint32_t llama_odin_abi_version(void) {
    return LLAMA_ODIN_ABI_VERSION;
}

static void null_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

void llama_odin_silence_logs(void) {
    llama_log_set(null_log_callback, nullptr);
}

void llama_odin_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked) {
    llama_set_embeddings_nextn(ctx, value, masked);
}

float * llama_odin_get_embeddings_nextn(struct llama_context * ctx) {
    return llama_get_embeddings_nextn(ctx);
}

float * llama_odin_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i) {
    return llama_get_embeddings_nextn_ith(ctx, i);
}

void llama_odin_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset) {
    llama_set_nextn_layer_offset(ctx, offset);
}

struct llama_context * llama_odin_get_ctx_other(struct llama_context * ctx) {
    return llama_get_ctx_other(ctx);
}

}
```

### 4.1 Exported Symbol Surface

The shared `libllama_odin` binary re-exports only the intended C-ABI:
`llama_*` (upstream llama.cpp C functions, including the `llama_odin_*`
wrappers), `ggml_*` (ggml C-ABI reachable from the public headers), and
`gguf_*` (`gguf.h`). All C++-mangled symbols from the vendored llama.cpp
C++ sources are hidden from the exported surface via platform-conditional
linker filtering configured in the root `CMakeLists.txt`:

| Platform | Mechanism |
|----------|-----------|
| Apple    | `-Wl,-exported_symbols_list cmake/llama-odin-exported-symbols.exp` (C symbols carry a leading underscore on Mach-O) |
| Linux    | `-Wl,--version-script cmake/llama-odin-exports.version` (`{ global: llama_*; ggml_*; gguf_*; local: *; };`) |
| Windows  | Unchanged; DLL exports only explicitly dllexport-tagged functions |

`nm -gU libllama_odin.dylib` therefore contains no C++-mangled (`__Z...`)
symbols.

---

## 5. Native Odin Package

### 5.1 Foreign C Declarations (`odin/c/llama_c.odin`)
Exposes all needed `llama_*`, `ggml_*`, and `llama_odin_*` procedures with foreign link directives pointing to `libllama_odin`:

```odin
package llama_c

import "core:c"

when ODIN_OS == .Darwin {
    foreign import llama_lib "system:libllama_odin.dylib"
} else {
    foreign import llama_lib "system:libllama_odin.so"
}

@(default_calling_convention="c")
foreign llama_lib {
    llama_backend_init :: proc() ---
    llama_backend_free :: proc() ---
    llama_odin_abi_version :: proc() -> u32 ---
    llama_odin_silence_logs :: proc() ---

    // Hardware devices
    ggml_backend_dev_count :: proc() -> uintptr ---
    ggml_backend_dev_get :: proc(index: uintptr) -> rawptr ---
    ggml_backend_dev_name :: proc(device: rawptr) -> cstring ---
    ggml_backend_dev_description :: proc(device: rawptr) -> cstring ---
    ggml_backend_dev_type :: proc(device: rawptr) -> c.int ---

    // Model & Vocab
    llama_model_default_params :: proc() -> Llama_Model_Params ---
    llama_model_load_from_file :: proc(path: cstring, params: Llama_Model_Params) -> ^Llama_Model ---
    llama_model_free :: proc(model: ^Llama_Model) ---
    llama_model_n_ctx_train :: proc(model: ^Llama_Model) -> i32 ---
    llama_model_n_embd_out :: proc(model: ^Llama_Model) -> i32 ---
    llama_model_n_layer_nextn :: proc(model: ^Llama_Model) -> i32 ---
    llama_model_get_vocab :: proc(model: ^Llama_Model) -> ^Llama_Vocab ---
    llama_vocab_n_tokens :: proc(vocab: ^Llama_Vocab) -> i32 ---
    llama_vocab_bos :: proc(vocab: ^Llama_Vocab) -> Llama_Token ---
    llama_vocab_is_eog :: proc(vocab: ^Llama_Vocab, token: Llama_Token) -> bool ---
    llama_tokenize :: proc(vocab: ^Llama_Vocab, text: cstring, text_len: i32, tokens: [^]Llama_Token, n_tokens_max: i32, add_special, parse_special: bool) -> i32 ---
    llama_token_to_piece :: proc(vocab: ^Llama_Vocab, token: Llama_Token, buf: [^]u8, length, lstrip: i32, special: bool) -> i32 ---

    // Context & Execution
    llama_context_default_params :: proc() -> Llama_Context_Params ---
    llama_init_from_model :: proc(model: ^Llama_Model, params: Llama_Context_Params) -> ^Llama_Context ---
    llama_free :: proc(ctx: ^Llama_Context) ---
    llama_n_ctx :: proc(ctx: ^Llama_Context) -> u32 ---
    llama_batch_init :: proc(n_tokens_alloc, embd, n_seq_max: i32) -> Llama_Batch ---
    llama_batch_free :: proc(batch: Llama_Batch) ---
    llama_decode :: proc(ctx: ^Llama_Context, batch: Llama_Batch) -> i32 ---
    llama_get_logits_ith :: proc(ctx: ^Llama_Context, i: i32) -> ^f32 ---

    // Sampling
    llama_sampler_chain_init :: proc(params: Llama_Sampler_Chain_Params) -> ^Llama_Sampler ---
    llama_sampler_chain_add :: proc(chain, smpl: ^Llama_Sampler) ---
    llama_sampler_init_temp :: proc(t: f32) -> ^Llama_Sampler ---
    llama_sampler_init_top_p :: proc(p: f32, min_keep: uintptr) -> ^Llama_Sampler ---
    llama_sampler_init_dist :: proc(seed: u32) -> ^Llama_Sampler ---
    llama_sampler_sample :: proc(smpl: ^Llama_Sampler, ctx: ^Llama_Context, idx: i32) -> Llama_Token ---
    llama_sampler_accept :: proc(smpl: ^Llama_Sampler, token: Llama_Token) ---
    llama_sampler_free :: proc(smpl: ^Llama_Sampler) ---

    // MTP Staging
    llama_odin_set_embeddings_nextn :: proc(ctx: ^Llama_Context, value, masked: bool) ---
    llama_odin_get_embeddings_nextn :: proc(ctx: ^Llama_Context) -> ^f32 ---
    llama_odin_get_embeddings_nextn_ith :: proc(ctx: ^Llama_Context, i: i32) -> ^f32 ---
    llama_odin_set_nextn_layer_offset :: proc(ctx: ^Llama_Context, offset: i32) ---
    llama_odin_get_ctx_other :: proc(ctx: ^Llama_Context) -> ^Llama_Context ---
}
```

### 5.2 High-Level Generator (`odin/llama/llama.odin`)
Exposes idiomatic Odin types:
- `Device_Info`, `Device_Kind`, and `devices() -> []Device_Info`
- `Generator_Config`: model path, layer count, speculative settings
- `Llama_Generator`: encapsulates model, context, batch buffers, samplers, and MTP driver
- `generator_new(cfg) -> (^Llama_Generator, string)`
- `generator_generate(g, prompt, turns, max_tokens, stop_seqs, cancel_flag, on_token, on_progress, user_data)`

---

## 6. Standalone Verification & Test Suites

The repository must provide runnable standalone tests:
1. **C Test (`tests/c/test_abi.c`)**:
   - Initializes backend via `llama_backend_init()`.
   - Verifies `llama_odin_abi_version() == 1`.
   - Calls `llama_odin_silence_logs()`.
   - Enumerate GPU backend devices.
   - Cleans up with `llama_backend_free()`.
2. **Odin Test (`odin/tests/`)**:
   - `odin test tests/odin/ -define:ODIN_TEST_THREADS=1`
   - Verifies device enumeration returns valid Metal GPU info on Apple Silicon.
   - Verifies vocabulary and tokenization helpers.

---

## 7. CI Release Pipeline (`.github/workflows/release.yml`)

Triggered on version tags (`v*`):
1. **Runner**: `macos-14` (Apple Silicon M1/M2/M3).
2. **Build**:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
   cmake --build build --config Release --parallel
   ```
3. **Artifact Bundling**:
   Create `llama-odin-macos-arm64-v{VERSION}.tar.gz` containing:
   - `lib/libllama_odin.dylib` (and `lib/libllama_odin.a`)
   - `include/llama_odin.h` and `include/llama.h`
   - `odin/` package sources
   - `LICENSE` and `README.md`
4. **Checksums**:
   Run `shasum -a 256 *.tar.gz > checksums.txt`.
5. **Publish**:
   Upload archive and `checksums.txt` to GitHub Releases.

---

## 8. Definition of Done for `llama-odin` Agent

The implementation is complete when:
- [ ] CMake builds `libllama_odin.dylib` with zero errors and embedded Metal kernel libraries (§3.2.1).
- [ ] `nm -gU libllama_odin.dylib` shows exported `llama_odin_*` and `llama_*` symbols.
- [ ] `tests/c/test_abi.c` compiles and executes successfully on macOS arm64.
- [ ] Odin package `odin test tests/odin/` passes cleanly.
- [ ] Release workflow YAML is configured and ready for GitHub Actions.
