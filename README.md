# llama-odin

Precompiled C-ABI shared/static library packaging upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) with embedded Apple Silicon Metal acceleration, MTP speculative decoding staging, and idiomatic Odin packages: a foreign layer over the C API (`odin/c/llama_c`) and a high-level generator package (`odin/llama`).

The vendored llama.cpp is compiled as static libraries and linked into a single self-contained `libllama_odin` binary that re-exports only the intended C ABI — no C++-mangled symbols leave the library.

## Prerequisites

- macOS arm64 (Metal is forced on for Apple builds)
- CMake >= 3.15 and a C++17 compiler
- Odin compiler (dev) for the `odin/` packages and Odin tests
- On macOS, the build links `Metal`, `Foundation`, and `Accelerate` frameworks

## Build

```sh
cmake -B build
cmake --build build -j
```

Outputs `build/libllama_odin.dylib` plus its versioned link (`@rpath` install name, `SOVERSION 1`, `VERSION 1.0.0`) and the static `build/libllama_odin.a`. Headers install to `include/`; the shared target is exported for CMake consumers (`llama_odin::llama_odin`); the static target keeps its dependency on the vendored llama objects private.

## Directory structure

```
include/llama_odin.h      C-ABI headers (shim + vendored llama.h, installed)
src/                      C++ shim for llama_*/ggml_*/llama_odin_* exports
cmake/                    export symbol filter lists
tests/c/                  C-ABI test (test_abi)
odin/c/                   llama_c.odin foreign declarations over libllama_odin
odin/llama/               generator + MTP speculative decoding driver
vendor/llama.cpp          pinned llama.cpp fork (git submodule)
tests/odin/               Odin test suite
```

## Usage (Odin)

```odin
package main

import "core:fmt"
import "core:mem"
import llama "odin/llama"

main :: proc() {
	llama.backend_init()
	defer llama.backend_free()

	cfg := llama.Generator_Config_Default()
	cfg.model_path = "gemma-4-E2B-it.gguf"

	g, err := llama.generator_new(cfg)
	if err != "" {
		fmt.eprintln(err)
		return
	}
	defer llama.generator_destroy(g)

	text := llama.generator_generate(g, "Write a haiku about Metal:", 1, 128,
		{"\n"}, nil,
		proc(token_id: llama.Llama_Token, piece: string, user_data: rawptr) -> bool {
			fmt.print(piece)
			return true
		},
		nil, nil)
	defer mem.delete_string(text)
	fmt.println()
}
```

Odin imports are relative to the importing package's directory; adjust the `import` path (or use `-collection`) to match your layout. `mem` is `core:mem`.

`generator_generate` streams every token piece through `on_token` (return `false` to cancel), stops at any `stop_seqs` entry, enforces the `max_tokens` budget across `turns`, and honors an optional `cancel_flag: ^bool`. The returned text is owned by the caller and released with `mem.delete_string`.

Device discovery:

```odin
infos := llama.devices()
defer delete(infos)
for d in infos {
	fmt.printfln("%s: %v MiB free of %v MiB", d.name, d.free_memory >> 20, d.total_memory >> 20)
	mem.delete_string(d.name)
	mem.delete_string(d.description)
}
```

## MTP speculative decoding

Models with multi-token-prediction heads (DeepSeek/GLM/Qwen3.5 single-head, Step35 chain-heads, Gemma4 assistant drafts) or a separate MTP draft GGUF run a speculative decoding loop driven by `odin/llama/mtp.odin`. Configure it through `Generator_Config` — no separate driver setup:

```odin
cfg := llama.Generator_Config_Default()
cfg.model_path = "gemma-4-E2B-it.gguf"
cfg.load_mtp = true        // load the model's nextn (MTP) layers with the model
cfg.mtp_context = true     // create an MTP draft context of the same model
// ... or use a separate draft GGUF instead of mtp_context:
cfg.mtp_model_path = "mtp-gemma-4-it.gguf"
cfg.nextn_layer_offset = 1 // optional: which trained MTP head to decode under
// draft-round tuning (zero values mirror the library defaults):
cfg.mtp_n_draft = 3        // max tokens drafted per round
cfg.mtp_n_min  = 0         // rounds drafting fewer verify nothing
cfg.mtp_p_min  = 0.0       // keep drafting while head top-probability >= p_min

g, err := llama.generator_new(cfg)
// llama.generator_generate dispatches to the MTP loop automatically;
// the same streaming, stop-sequence, and cancellation contract applies.
```

Draft rounds verify every drafted token in one target decode, so output is identical to plain generation. The driver is created and owned by the generator; nothing extra to initialize or free. The `llama_odin_*` staging shims underneath — `set_embeddings_nextn`, `get_embeddings_nextn_ith`, `set_nextn_layer_offset`, `get_ctx_other` — live in `include/llama_odin.h` (§4 of the spec).

## Native C consumers

The full `llama_*`, `ggml_*`, `gguf_*`, and `llama_odin_*` C surface is exported without C++ mangling:

```c
#include <stdio.h>
#include "llama_odin.h"

int main(void) {
    llama_backend_init();
    printf("abi version: %u\n", llama_odin_abi_version());
    llama_odin_silence_logs();
    /* remaining llama.cpp C API is available, e.g. llama_model_load_from_file */
    llama_backend_free();
    return 0;
}
```

```sh
cc hello.c -Iinclude -Lbuild -Wl,-rpath,"$PWD/build" -llama_odin -o hello
./hello
```

## Tests

The C-ABI suite runs through CTest:

```sh
ctest --test-dir build --output-on-failure
```

The Odin suite runs from the repository root after the C build (the repo-root `libllama_odin.dylib` symlinks are how the linker and dyld resolve the library):

```sh
odin test tests/odin/ -define:ODIN_TEST_THREADS=1
```

Model-backed tests need a real GGUF: they use `LLAMA_ODIN_TEST_MODEL` when set, otherwise the offline Hugging Face cache (`ggml-org/gemma-4-E2B-it-GGUF`). Nothing is downloaded; see `tests/odin/README.md`.

## Embedded Metal

GGML Metal kernels are compiled into the binary, not shipped as files: at build time each kernel group's Metal source is concatenated with shared headers and embedded into `__DATA,__ggml_metallib` sections (~1.8 MB); at Metal device init the embedded source is JIT-compiled per kernel group in parallel (~0.02–0.2 s on Apple Silicon). The resulting binaries are self-contained — no `.metallib` or `.metal` file exists on disk and no runtime file search is performed. See `llama-odin-spec.md` §3.2.1 for the embedding mechanism and its deviation note versus build-time `.metallib` bytecode.

## Releases

Version tags (`v*`) trigger a release workflow that publishes a macOS arm64 archive (`llama-odin-macos-arm64-v{VERSION}.tar.gz` with `lib/`, `include/`, `odin/`, `LICENSE`, `README.md`) and a SHA-256 checksums file to GitHub Releases. Design lives in `llama-odin-spec.md` §7.

## License

MIT — see [LICENSE](LICENSE). Vendored llama.cpp keeps its upstream license.