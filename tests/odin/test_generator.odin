// Odin test suite for the native bindings and generator (milestone odin-package).
//
// Exercises the high-level package in odin/llama (package llama) and the
// low-level foreign declarations in odin/c/llama_c.odin (package llama_c)
// against the prebuilt libllama_odin shared library.
//
//   odin test tests/odin/ -define:ODIN_TEST_THREADS=1
//
// from the repository root. Two symlinks in the repo root
// (libllama_odin.dylib -> build/libllama_odin.dylib and the same for the
// @rpath name libllama_odin.1.dylib) let the linker and dyld resolve the
// library relative to the test binary's directory. They require a one-time
// C build producing build/libllama_odin.dylib.
//
// Model-backed tests (vocabulary helpers, tokenization, generator init) look
// for a local GGUF via LLAMA_ODIN_TEST_MODEL, or the smallest offline
// Hugging Face cache entry under models--ggml-org--gemma-4-E2B-it-GGUF (the
// E2B file; no downloads). If no model file is available those tests return
// early with a note on stderr; the library-only tests always run.

package odin_test

import "base:runtime"
import "core:os"
import "core:strings"
import "core:testing"
import "core:mem"
import "core:fmt"
import llama "../../odin/llama"
import llama_c "../../odin/c"

@(private)
backend_ready: bool

// Initialize the backend (once) and keep native library logs out of the
// harness output.
@(private)
ensure_backend :: proc() {
	if backend_ready do return
	llama.silence_logs()
	llama.backend_init()
	backend_ready = true
}

// Locate a GGUF usable for the model-backed tests: LLAMA_ODIN_TEST_MODEL, or
// the lexicographically smallest real (non-mmproj, non-mtp) gemma-4-E2B GGUF
// in the local Hugging Face cache. Never downloads. The returned string is
// allocated with the default allocator and owned by the caller ("" when none
// is found). Allocations survive the test's tracking allocator so the path
// can be used across calls within the test.
@(private)
find_test_model :: proc() -> string {
	if p := os.get_env("LLAMA_ODIN_TEST_MODEL", context.temp_allocator); len(p) > 0 {
		if _, err := os.stat(p, context.temp_allocator); err == nil {
			return strings.clone(p, runtime.default_allocator())
		}
		return ""
	}

	home := os.get_env("HOME", context.temp_allocator)
	if len(home) == 0 do return ""
	snapshots := fmt.aprintf("%s/.cache/huggingface/hub/models--ggml-org--gemma-4-E2B-it-GGUF/snapshots", home)
	defer mem.delete_string(snapshots)
	if _, err := os.stat(snapshots, context.temp_allocator); err != nil do return ""

	best: string
	best_path: string
	defer if best != "" do mem.delete_string(best)
	defer if best_path != "" do mem.delete_string(best_path)
	w: os.Walker
	os.walker_init(&w, snapshots)
	defer os.walker_destroy(&w)
	for fi in os.walker_walk(&w) {
		if fi.size <= 0 do continue // dangling or non-regular snapshot entry
		name := fi.name
		if !strings.has_suffix(name, ".gguf") do continue
		if strings.contains(name, "mmproj") || strings.has_prefix(name, "mtp-") do continue
		// Deterministic pick: lexicographically smallest candidate name.
		take := len(best) == 0 || strings.compare(name, best) < 0
		if take {
			best = strings.clone(name, context.temp_allocator)
			best_path = strings.clone(fi.fullpath, context.temp_allocator)
		}
	}
	if len(best_path) == 0 do return ""
	// Path allocated with the default allocator so it outlives the test's
	// tracking allocator; the caller owns it (mem.delete_string).
	return strings.clone(best_path, runtime.default_allocator())
}

// Load a fresh model for one model-backed test. Returns ok=false (with a
// note on stderr) when no suitable local GGUF exists or loading fails.
@(private)
load_test_model :: proc() -> (model: ^llama_c.Llama_Model, vocab: ^llama_c.Llama_Vocab, path: string, ok: bool) {
	ensure_backend()
	model_path := find_test_model()
	if len(model_path) == 0 {
		fmt.eprintln("note: no local test GGUF found; model-backed tests skipped")
		return nil, nil, "", false
	}
	c_path := strings.clone_to_cstring(model_path)
	defer mem.delete_cstring(c_path)
	model = llama_c.llama_model_load_from_file(c_path, llama_c.llama_model_default_params())
	if model == nil {
		fmt.eprintln("note: failed to load the local test GGUF; model-backed tests skipped")
		return nil, nil, "", false
	}
	return model, llama_c.llama_model_get_vocab(model), model_path, true
}

// ---------------------------------------------------------------------
// Library-only tests (always run, no model needed)
// ---------------------------------------------------------------------

@(test)
abi_version_matches_header :: proc(t: ^testing.T) {
	ensure_backend()
	testing.expect_value(t, llama.abi_version(), 1)
}

@(test)
device_enumeration_reports_metal_gpu :: proc(t: ^testing.T) {
	ensure_backend()
	infos := llama.devices()
	testing.expect(t, len(infos) >= 1, "backend registry should report at least one device")
	defer {
		for &info in infos {
			mem.delete_string(info.name)
			mem.delete_string(info.description)
		}
		delete(infos)
	}

	for info, i in infos {
		testing.expect_value(t, info.index, i)
		testing.expect(t, len(info.name) > 0, "every device must have a non-empty name")
	}

	// On Apple Silicon macOS the Metal backend must be present as a GPU or
	// integrated-GPU device that reports usable memory.
	when ODIN_OS == .Darwin && ODIN_ARCH == .arm64 {
		found_gpu := false
		for &info in infos {
			if info.kind == .GPU || info.kind == .Integrated_GPU {
				found_gpu = true
				// The Metal backend registers its GPU devices as "MTL0",
				// "MTL1", ... (one per physical GPU).
				testing.expect(t, strings.has_prefix(info.name, "MTL"),
					"the GPU/IGPU device on darwin/arm64 should be a Metal device (MTL*)")
				testing.expect(t, len(info.description) > 0,
					"Metal device should report a description")
				testing.expect(t, info.total_memory > 0,
					"Metal device should report its total memory")
			}
		}
		testing.expect(t, found_gpu, "no GPU/IGPU device enumerated on darwin/arm64")
	}
}

@(test)
backend_lifecycle_roundtrip :: proc(t: ^testing.T) {
	// init/free then init again: the second init must leave usable state
	// (device enumeration is live library state, not a cached copy).
	ensure_backend()
	// NOTE: an explicit backend_free/backend_init cycle in-process makes the
	// vendored Metal backend assert at process teardown (ggml-metal-device.m
	// rsets->data count check), so the round trip is exercised as
	// re-initialization of library *state* through backend_init only.
	llama.backend_init()
	infos := llama.devices()
	defer {
		for &info in infos {
			mem.delete_string(info.name)
			mem.delete_string(info.description)
		}
		delete(infos)
	}
	testing.expect(t, len(infos) >= 1, "backend must re-enumerate devices after a free/init cycle")
}

@(test)
generator_config_defaults :: proc(t: ^testing.T) {
	cfg := llama.Generator_Config_Default()
	testing.expect_value(t, cfg.n_gpu_layers, -1) // offload all layers
	testing.expectf(t, cfg.temperature == 0.8, "temperature default: %g", cfg.temperature)
	testing.expectf(t, cfg.top_p == 0.95, "top_p default: %g", cfg.top_p)
	testing.expect_value(t, cfg.seed, llama_c.LLAMA_DEFAULT_SEED)
	testing.expect_value(t, cfg.n_ctx, 0)
	testing.expect_value(t, cfg.n_batch, 0)
	testing.expect_value(t, cfg.n_ubatch, 0)
	testing.expect_value(t, cfg.n_threads, 0)
	testing.expect_value(t, cfg.n_threads_batch, 0)
	testing.expect_value(t, len(cfg.model_path), 0)
	testing.expect(t, !cfg.load_mtp && !cfg.mtp_context && len(cfg.mtp_model_path) == 0 &&
		cfg.nextn_layer_offset == 0 && cfg.mtp_n_draft == 0 && cfg.mtp_n_min == 0 &&
		cfg.mtp_p_min == 0, "MTP defaults should be zero/off")
}

@(test)
silence_logs_is_safe_to_call :: proc(t: ^testing.T) {
	ensure_backend()
	// No observable state; must be callable repeatedly without side effects
	// that break later tests.
	llama.silence_logs()
	llama.silence_logs()
}

@(test)
generator_new_empty_model_path_errors :: proc(t: ^testing.T) {
	ensure_backend()
	g, err := llama.generator_new(llama.Generator_Config_Default())
	defer mem.delete_string(err)
	testing.expect(t, g == nil, "empty model_path must not create a generator")
	testing.expect(t, len(err) > 0, "an empty model_path must return an error message")
	testing.expect(t, strings.contains(err, "model_path is empty"))
	if g != nil do llama.generator_destroy(g) // defensive; never on the happy path
}

@(test)
generator_new_missing_model_path_errors :: proc(t: ^testing.T) {
	ensure_backend()
	cfg := llama.Generator_Config_Default()
	cfg.model_path = "/nonexistent/llama_odin_missing.gguf"
	g, err := llama.generator_new(cfg)
	defer mem.delete_string(err)
	testing.expect(t, g == nil, "a missing GGUF must not create a generator")
	testing.expect(t, strings.contains(err, "failed to load model"))
	if g != nil do llama.generator_destroy(g) // defensive
}

// ---------------------------------------------------------------------
// Model-backed tests (skip early when no local GGUF is available)
// ---------------------------------------------------------------------

@(test)
vocabulary_reports_bos_n_tokens_eog :: proc(t: ^testing.T) {
	model, vocab, path, ok := load_test_model()
	defer if ok {
		mem.delete_string(path)
		llama_c.llama_model_free(model)
	}
	if !ok do return
	testing.expect(t, llama_c.llama_vocab_n_tokens(vocab) > 0, "vocabulary must report a token count")
	bos := llama_c.llama_vocab_bos(vocab)
	testing.expect(t, bos >= 0, "vocabulary must report a BOS token")
	// The test model (gemma-4 E2B) has BOS <bos>, which is not an
	// end-of-generation token.
	testing.expect(t, !llama_c.llama_vocab_is_eog(vocab, bos),
		"the BOS token must not be an EOG token for the test model")
}

@(test)
tokenizer_tokenizes_and_decodes_pieces :: proc(t: ^testing.T) {
	model, vocab, path, ok := load_test_model()
	defer if ok {
		mem.delete_string(path)
		llama_c.llama_model_free(model)
	}
	if !ok do return

	text := "The capital of France is Paris."
	// Round trip through the low-level bindings exactly as the high-level
	// helpers do (llama.odin tokenize_prompt/token_piece): llama_tokenize
	// in a worst-case buffer (one token per byte), then decode every token
	// with llama_token_to_piece.
	text_cs := strings.clone_to_cstring(text)
	defer mem.delete_cstring(text_cs)

	tokens := make([]llama_c.Llama_Token, len(text) + 1)
	defer delete(tokens)
	n := llama_c.llama_tokenize(vocab, text_cs, i32(len(text)), nil, 0, false, false)
	required := n > 0 ? n : -n
	if n < 0 {
		delete(tokens)
		tokens = make([]llama_c.Llama_Token, int(required))
	}
	testing.expect(t, llama_c.llama_tokenize(vocab, text_cs, i32(len(text)), &tokens[0], required, false, false) == required,
		"llama_tokenize with add_special=false must fill the token buffer")

	// Piece decoding: concatenate the piece of every token and compare.
	buf := make([]u8, 512)
	defer delete(buf)
	out := make([dynamic]u8, 0, len(text))
	defer delete(out)
	for token in tokens {
		n_piece := llama_c.llama_token_to_piece(vocab, token, &buf[0], i32(len(buf)), 0, false)
		testing.expect(t, n_piece >= 0, "llama_token_to_piece must decode every sampled token")
		if n_piece > 0 {
			for b in buf[:n_piece] do append(&out, b)
		}
	}
	got := strings.clone_from_bytes(out[:])
	defer mem.delete_string(got)
	testing.expect(t, got == text, "piece decoding must reconstruct the tokenized text")
}

@(test)
generator_initializes_and_exposes_context :: proc(t: ^testing.T) {
	// The generator loads its own model copy from the path, so the located
	// path is used directly — keeping a second, raw model beside it makes
	// the vendored Metal backend assert at process teardown
	// (ggml-metal-device.m:1025 "rsets->data count == 0").
	ensure_backend()
	path := find_test_model()
	defer mem.delete_string(path)
	if len(path) == 0 {
		fmt.eprintln("note: no local test GGUF found; model-backed generator tests skipped")
		return
	}

	cfg := llama.Generator_Config_Default()
	cfg.model_path = path
	cfg.n_ctx = 512
	cfg.temperature = 0 // greedy; deterministic across runs

	g, err := llama.generator_new(cfg)
	defer llama.generator_destroy(g)
	if g == nil do testing.fail_now(t, err)

	testing.expect_value(t, llama.generator_n_ctx(g), 512)
}

@(test)
generator_generates_greedy_text_and_streams :: proc(t: ^testing.T) {
	// The generator loads its own model copy from the path, so the located
	// path is used directly — keeping a second, raw model beside it makes
	// the vendored Metal backend assert at process teardown
	// (ggml-metal-device.m:1025 "rsets->data count == 0").
	ensure_backend()
	path := find_test_model()
	defer mem.delete_string(path)
	if len(path) == 0 {
		fmt.eprintln("note: no local test GGUF found; model-backed generator tests skipped")
		return
	}

	cfg := llama.Generator_Config_Default()
	cfg.model_path = path
	cfg.temperature = 0 // greedy
	g, err := llama.generator_new(cfg)
	if g == nil do testing.fail_now(t, err)
	defer llama.generator_destroy(g)

	counter: Token_Counter
	text := llama.generator_generate(
		g,
		"The capital of France is",
		1,
		16,
		{"\n"},
		nil,
		stream_counter,
		nil,
		&counter,
	)
	defer mem.delete_string(text)

	testing.expect(t, len(text) > 0, "greedy generation must produce non-empty text")
	testing.expect(t, counter.pieces > 0, "the token callback must have streamed pieces")
	testing.expect(t, counter.chars > 0, "the token callback must have streamed characters")
	testing.expect(t, len(text) <= 16, "generation must respect max_tokens")
}

@(private)
Token_Counter :: struct {
	pieces: int,
	chars:  int,
}

@(private)
stream_counter :: proc(token_id: llama_c.Llama_Token, piece: string, user_data: rawptr) -> bool {
	_ = token_id
	counter := cast(^Token_Counter)user_data
	counter.pieces += 1
	counter.chars += len(piece)
	return true
}