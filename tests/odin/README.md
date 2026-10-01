# Odin test suite

Run from the repository root after a one-time C build
(`cmake -S . -B build && cmake --build build`, which produces
`build/libllama_odin.dylib`):

    odin test tests/odin/ -define:ODIN_TEST_THREADS=1

The two repository-root symlinks (`libllama_odin.dylib` and
`libllama_odin.1.dylib`, both -> `build/libllama_odin.dylib`) are how the
linker and dyld resolve the shared library for this command: `odin test`
links `libllama_odin.dylib` relative to the working directory and the dylib's
install name is `@rpath/libllama_odin.1.dylib`, resolved against the test
binary's directory (`@executable_path` at the repo root).

By default native library logs are silenced by the tests
(`llama.silence_logs`).

## Model-backed tests

The vocabulary, tokenization, and generator tests need a real GGUF. They use
`LLAMA_ODIN_TEST_MODEL` when set, otherwise they pick the lexicographically
smallest non-mmproj/non-mtp `gemma-4-E2B` GGUF under the offline Hugging Face
cache
(`~/.cache/huggingface/hub/models--ggml-org--gemma-4-E2B-it-GGUF/snapshots`).
Nothing is downloaded; when no model file exists, those tests note the skip
on stderr and pass the library-only tests.
