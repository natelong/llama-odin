// High-level idiomatic Odin interface to libllama_odin (package `llama`).
//
// This is the layer directly above the foreign C declarations in
// odin/c/llama_c.odin (package llama_c); all C semantics live in that foreign
// layer and include/llama_odin.h — nothing is duplicated here.
//
// Usage sketch:
//
//	backend_init()
//	defer backend_free()
//
//	cfg := Generator_Config_Default()
//	cfg.model_path = "model.gguf"
//	g, err := generator_new(cfg)
//	if err != "" do .. handle .., else defer generator_destroy(g)
//
//	text := generator_generate(g, prompt, 1, 256, {"\n"}, nil,
//	                           proc(id, piece, ud) -> bool { fmt.print(piece); return true },
//	                           nil, nil)

package llama

import "core:c"
import "core:fmt"
import "core:mem"
import "core:strings"
import llama_c "../c"

// ---------------------------------------------------------------------
// Hardware discovery
// ---------------------------------------------------------------------

// Kind of hardware backend device (enum ggml_backend_dev_type).
Device_Kind :: enum {
	CPU, // device using system memory
	GPU, // GPU device using dedicated memory
	Integrated_GPU, // integrated GPU device using host memory
	Accelerator, // accelerator used together with the CPU backend (e.g. BLAS)
	Meta, // "meta" device wrapping multiple other devices
}

// Snapshot of one hardware backend device reported by the ggml backend
// registry (ggml_backend_dev_*).
Device_Info :: struct {
	index:        int,
	name:         string, // short device name, e.g. "Metal"
	description:  string, // longer human-readable description
	kind:         Device_Kind,
	free_memory:  u64, // memory available for offloading, in bytes
	total_memory: u64, // total device memory, in bytes
}

// Enumerate all hardware backend devices (GGML backends) in fixed order.
//
// The returned slice and its strings are allocated with the context
// allocator and owned by the caller: release the strings with
// `mem.delete_string(info.name)` / `mem.delete_string(info.description)` and
// the slice with `delete(infos)`.
devices :: proc() -> []Device_Info {
	n := int(llama_c.ggml_backend_dev_count())
	if n == 0 do return nil

	infos := make([]Device_Info, n)
	for i in 0..<n {
		dev := llama_c.ggml_backend_dev_get(c.size_t(i))
		d := &infos[i]
		d.index = i
		d.name = strings.clone_from_cstring(llama_c.ggml_backend_dev_name(dev))
		d.description = strings.clone_from_cstring(llama_c.ggml_backend_dev_description(dev))
		d.kind = device_kind_from_ggml(llama_c.ggml_backend_dev_type(dev))
		free_mem: c.size_t
		total_mem: c.size_t
		llama_c.ggml_backend_dev_memory(dev, &free_mem, &total_mem)
		d.free_memory, d.total_memory = u64(free_mem), u64(total_mem)
	}
	return infos
}

@(private)
device_kind_from_ggml :: proc(t: llama_c.Ggml_Backend_Dev_Type) -> Device_Kind {
	switch t {
	case .CPU:   return .CPU
	case .GPU:   return .GPU
	case .IGPU:  return .Integrated_GPU
	case .ACCEL: return .Accelerator
	case .META:  return .Meta
	case:        return .CPU
	}
}

// ---------------------------------------------------------------------
// Library lifecycle helpers
// ---------------------------------------------------------------------

// One-time global initialization of the llama.cpp backend. Call once at
// startup before creating generators; release with backend_free.
backend_init :: proc() {
	llama_c.llama_backend_init()
}

// Release global backend state. Must not be called while any generator
// created since backend_init is still alive.
backend_free :: proc() {
	llama_c.llama_backend_free()
}

// Route all native library logs to a null sink.
silence_logs :: proc() {
	llama_c.llama_odin_silence_logs()
}

// ABI version of the native library (llama_odin.h: LLAMA_ODIN_ABI_VERSION).
abi_version :: proc() -> u32 {
	return llama_c.llama_odin_abi_version()
}

// ---------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------

// Configuration for generator_new. Build it with Generator_Config_Default()
// and override individual fields; zero values select the library defaults
// (include/llama.h llama_model_params / llama_context_params).
Generator_Config :: struct {
	model_path: string,

	// Context sizing (u32, 0 = library default; llama_context_params.n_ctx
	// of 0 makes the context use the model's training context size).
	n_ctx:    u32, // 0 = from model
	n_batch:  u32, // 0 = library default; logical maximum llama_decode batch
	n_ubatch: u32, // 0 = library default; physical maximum batch size

	// Hardware offload / threading. n_gpu_layers follows the library
	// semantics: negative = offload all layers (the library default),
	// 0 = keep everything on CPU.
	n_gpu_layers:    i32, // -1 = all layers
	n_threads:       i32, // 0 = library default
	n_threads_batch: i32, // 0 = library default

	// Sampling chain (top_p, temperature, dist are chained in that order).
	// temperature <= 0 makes the temperature sampler pick greedily
	// (llama_sampler_temp_impl treats temp <= 0 as argmax).
	temperature: f32,
	top_p:       f32,
	seed:        u32, // llama_c.LLAMA_DEFAULT_SEED = pseudo-random

	// Speculative / MTP (multi-token prediction) staging settings. The
	// driver logic itself lives in odin/llama/mtp.odin; these fields pass
	// the llama_odin_* settings through at creation time.
	//   load_mtp:           load MTP (next-n) layers with the model
	//   mtp_context:        create an MTP draft context of the same model
	//                       (context type MTP, ctx_other = main context), as
	//                       done for draft-mtp in common/speculative.cpp
	//   mtp_model_path:     optional separate MTP/assistant GGUF file (e.g.
	//                       mtp-gemma-4-it.gguf); when set, a draft model and
	//                       context are created from it in the same way
	//   nextn_layer_offset: llama_odin_set_nextn_layer_offset passthrough,
	//                       applied when a draft context is created
	// Draft-round tuning (Mtp_Config in mtp.odin; zero values mirror the
	// library's common_params_speculative_draft defaults):
	//   mtp_n_draft:        max tokens drafted per round (0 = 3)
	//   mtp_n_min:          rounds drafting fewer than this many tokens
	//                       verify nothing (0 = never verify short drafts)
	//   mtp_p_min:          keep drafting only while the head's top
	//                       probability is >= this (0 = greedy, no gate)
	load_mtp:           bool,
	mtp_context:        bool,
	mtp_model_path:     string,
	nextn_layer_offset: i32,
	mtp_n_draft:        i32,
	mtp_n_min:          i32,
	mtp_p_min:          f32,
}

// Default generator configuration, mirroring the library defaults:
// every layer offloaded to GPU, temperature 0.8, top_p 0.95, random seed.
Generator_Config_Default :: proc() -> Generator_Config {
	return Generator_Config{
		n_gpu_layers = -1,
		temperature  = 0.8,
		top_p        = 0.95,
		seed         = llama_c.LLAMA_DEFAULT_SEED,
	}
}

// A loaded generator: one model, context, token batch buffer and sampler
// chain. Create with generator_new, release with generator_destroy.
Llama_Generator :: struct {
	cfg:     Generator_Config,
	model:   ^llama_c.Llama_Model,
	vocab:   ^llama_c.Llama_Vocab,
	ctx:     ^llama_c.Llama_Context,
	// The MTP draft context created alongside `ctx` when
	// Generator_Config.mtp_context or mtp_model_path is set (context type
	// MTP with ctx_other = ctx, as in common/speculative.cpp); owned by the
	// generator.
	other_ctx:  ^llama_c.Llama_Context,
	// The MTP draft model backing other_ctx when Generator_Config.mtp_model_path
	// is set; owned by the generator and freed after its context.
	mtp_model:  ^llama_c.Llama_Model,
	// The MTP speculative decoding driver (mtp.odin), created when the MTP
	// draft context exists; owned by the generator and freed first on
	// teardown. nil when MTP is not configured.
	mtp:        ^Mtp_Driver,
	batch:      llama_c.Llama_Batch,
	sampler:    ^llama_c.Llama_Sampler,
}

// Token id (llama.h: typedef int32_t llama_token), re-exported from llama_c
// for callback signatures.
Llama_Token :: llama_c.Llama_Token

// Called for every streamed token piece. `piece` is a complete, valid UTF-8
// chunk (all bytes of a rune are only ever streamed together). Return false
// to cancel the rest of generation.
Token_Callback :: #type proc(token_id: llama_c.Llama_Token, piece: string, user_data: rawptr) -> bool

// Called during prompt ingestion with the fraction of prompt tokens decoded
// (in [0, 1]). Return false to cancel the rest of generation.
Progress_Callback :: #type proc(progress: f32, user_data: rawptr) -> bool

// Create a generator from `cfg`. On success returns (g, ""); on failure
// returns (nil, error_message) with no resources left behind. Requires a
// previous backend_init call.
generator_new :: proc(cfg: Generator_Config) -> (^Llama_Generator, string) {
	if len(cfg.model_path) == 0 {
		return nil, "llama.generator_new: model_path is empty"
	}

	g := new(Llama_Generator)
	g.cfg = cfg

	c_path := strings.clone_to_cstring(cfg.model_path)
	defer mem.delete_cstring(c_path)

	mp := llama_c.llama_model_default_params()
	mp.n_gpu_layers = cfg.n_gpu_layers
	mp.load_mtp = cfg.load_mtp

	model := llama_c.llama_model_load_from_file(c_path, mp)
	if model == nil {
		free(g)
		return nil, fmt.aprintf("llama.generator_new: failed to load model %q", cfg.model_path)
	}
	g.model = model
	g.vocab = llama_c.llama_model_get_vocab(model)

	cp := llama_c.llama_context_default_params()
	if cfg.n_ctx > 0 {
		cp.n_ctx = cfg.n_ctx
	}
	if cfg.n_batch > 0 {
		cp.n_batch = cfg.n_batch
	}
	if cfg.n_ubatch > 0 {
		cp.n_ubatch = cfg.n_ubatch
	}
	if cfg.n_threads > 0 {
		cp.n_threads = cfg.n_threads
	}
	if cfg.n_threads_batch > 0 {
		cp.n_threads_batch = cfg.n_threads_batch
	}
	if cfg.mtp_context || len(cfg.mtp_model_path) > 0 {
		// Per-transaction snapshots for KV rollback of rejected draft tokens,
		// needed on models whose memory cannot partially remove positions
		// (recurrent/hybrid parts); the C++ driver's callers request the same
		// (common_context_params_to_llama: n_rs_seq = draft.n_max for MTP).
		cp.n_rs_seq = u32(cfg.mtp_n_draft > 0 ? cfg.mtp_n_draft : 3)
	}

	main_ctx := llama_c.llama_init_from_model(model, cp)
	if main_ctx == nil {
		generator_destroy(g)
		return nil, "llama.generator_new: failed to create context"
	}
	g.ctx = main_ctx

	if cfg.mtp_context || len(cfg.mtp_model_path) > 0 {
		// MTP draft context, wired like common/speculative.cpp (spec_mtp): a
		// context with context type MTP whose ctx_other points back at the
		// main (target) context. Backed by a separate assistant model when
		// mtp_model_path is given, or by the target model itself otherwise.
		cp.ctx_type = .MTP
		cp.ctx_other = main_ctx

		draft_model := model
		if len(cfg.mtp_model_path) > 0 {
			draft_c_path := strings.clone_to_cstring(cfg.mtp_model_path)
			defer mem.delete_cstring(draft_c_path)
			mp2 := llama_c.llama_model_default_params()
			mp2.n_gpu_layers = cfg.n_gpu_layers
			mp2.load_mtp = true
			draft_model = llama_c.llama_model_load_from_file(draft_c_path, mp2)
			if draft_model == nil {
				generator_destroy(g)
				return nil, fmt.aprintf("llama.generator_new: failed to load MTP model %q", cfg.mtp_model_path)
			}
			g.mtp_model = draft_model
		}

		draft_ctx := llama_c.llama_init_from_model(draft_model, cp)
		if draft_ctx == nil {
			generator_destroy(g)
			return nil, "llama.generator_new: failed to create MTP draft context"
		}
		g.other_ctx = draft_ctx

		if cfg.nextn_layer_offset != 0 {
			llama_c.llama_odin_set_nextn_layer_offset(draft_ctx, cfg.nextn_layer_offset)
		}

		// The speculative decoding driver itself: nextn embedding staging
		// flags on both contexts, the draft batch with (token, embedding)
		// inputs, and the cross-decode hidden-row state.
		mtp_err: string
		g.mtp, mtp_err = mtp_driver_new(g)
		if g.mtp == nil {
			generator_destroy(g)
			return nil, fmt.aprintf("llama.generator_new: failed to initialize MTP driver: %s", mtp_err)
		}
	}

	g.batch = llama_c.llama_batch_init(batch_capacity(cfg), 0, 1)
	if g.batch.token == nil {
		generator_destroy(g)
		return nil, "llama.generator_new: failed to allocate token batch"
	}

	sp := llama_c.llama_sampler_chain_default_params()
	sp.no_perf = true
	chain := llama_c.llama_sampler_chain_init(sp)
	if chain == nil {
		generator_destroy(g)
		return nil, "llama.generator_new: failed to create sampler chain"
	}
	if cfg.top_p > 0 && cfg.top_p < 1 {
		llama_c.llama_sampler_chain_add(chain, llama_c.llama_sampler_init_top_p(cfg.top_p, 1))
	}
	llama_c.llama_sampler_chain_add(chain, llama_c.llama_sampler_init_temp(cfg.temperature))
	llama_c.llama_sampler_chain_add(chain, llama_c.llama_sampler_init_dist(cfg.seed))
	g.sampler = chain

	return g, ""
}

// Destroy a generator (sampler chain, batch buffer, contexts, model) and
// free the handle itself. Safe to call with nil or a partially initialized
// generator: every handle is checked and nulled before release.
//
// This does not release global backend state; pair backend_free with
// backend_init at process level.
generator_destroy :: proc(g: ^Llama_Generator) {
	if g == nil do return
	// Free the MTP driver before the contexts it drives.
	if g.mtp != nil {
		mtp_driver_destroy(g.mtp)
		g.mtp = nil
	}
	if g.sampler != nil {
		// A chain owns and frees the samplers added to it.
		llama_c.llama_sampler_free(g.sampler)
		g.sampler = nil
	}
	llama_c.llama_batch_free(g.batch)
	g.batch = llama_c.Llama_Batch{}
	if g.other_ctx != nil {
		// Free the MTP draft context first: it references the main context
		// (ctx_other / shared memory) and must not outlive it.
		llama_c.llama_free(g.other_ctx)
		g.other_ctx = nil
	}
	if g.ctx != nil {
		llama_c.llama_free(g.ctx)
		g.ctx = nil
	}
	if g.mtp_model != nil {
		llama_c.llama_model_free(g.mtp_model)
		g.mtp_model = nil
	}
	if g.model != nil {
		llama_c.llama_model_free(g.model)
		g.model = nil
	}
	free(g)
}

generator_free :: generator_destroy

// Text context size of the generator's context (llama_n_ctx).
generator_n_ctx :: proc(g: ^Llama_Generator) -> u32 {
	if g == nil || g.ctx == nil do return 0
	return llama_c.llama_n_ctx(g.ctx)
}

// Generate text from `prompt` and stream every generated token.
//
//   - `turns`: number of assistant turns to generate. The first turn
//     continues the prompt; after an end-of-generation token, decoding the
//     EOG token itself starts the next turn. `turns <= 0` is treated as 1.
//   - `max_tokens`: total token budget across all turns; `<= 0` is unbounded
//     (runs until the model's end-of-generation token, a stop sequence, or
//     cancellation).
//   - `stop_seqs`: generation stops as soon as any sequence is completed;
//     the matching text is excluded from the result and never (or only
//     partially) streamed. Bytes that could still be part of a stop sequence
//     are held back from the stream until resolved.
//   - `cancel_flag`: when non-nil and dereferencing true, generation stops
//     early. `on_token` and `on_progress` returning false also cancel.
//   - `on_token`: called for every streamed token with its token id and a
//     complete UTF-8 piece; nil disables token streaming.
//   - `on_progress`: called during prompt ingestion with the fraction of
//     prompt tokens decoded; nil disables progress reporting.
//   - `user_data`: passed through to both callbacks.
//
// Returns the full generated text (all turns concatenated, stop sequences
// excluded), allocated with the context allocator; the caller owns it and
// releases it with `mem.delete_string`. Errors from the low layer (tokenize /
// decode failures) stop generation and return the text produced so far.
generator_generate :: proc(
	g:           ^Llama_Generator,
	prompt:      string,
	turns:       i32,
	max_tokens:  i32,
	stop_seqs:   []string,
	cancel_flag: ^bool,
	on_token:    Token_Callback,
	on_progress: Progress_Callback,
	user_data:   rawptr,
) -> string {
	if g == nil || g.model == nil || g.ctx == nil || g.sampler == nil do return ""

	// MTP-configured generators run the speculative decoding loop in
	// mtp.odin (odin/llama/mtp.odin); g.other_ctx exists only when the
	// draft context was created (load_mtp / mtp_context / mtp_model_path).
	if g.other_ctx != nil {
		return mtp_generate(
			g, prompt, turns, max_tokens, stop_seqs, cancel_flag,
			on_token, on_progress, user_data,
		)
	}

	n_turns := max(turns, 1)

	// Generated text (byte buffer) with the originating token id per byte, so
	// streamed pieces can be rune-aligned yet attributed to the right token.
	text_cap := max(1024, len(prompt))
	text := make([dynamic]u8, 0, text_cap)
	defer delete(text)
	text_ids := make([dynamic]llama_c.Llama_Token, 0, text_cap)
	defer delete(text_ids)
	piece_buf := make([dynamic]u8, 0, 512)
	defer delete(piece_buf)
	streamed := 0 // bytes already delivered through on_token

	cancelled := false
	stopped := false
	trunc := len(text) // final text length (moved back when a stop sequence matches)

	prompt_c := strings.clone_to_cstring(prompt)
	defer mem.delete_cstring(prompt_c)

	prompt_tokens := tokenize_prompt(g.vocab, prompt_c, prompt)
	defer delete(prompt_tokens)
	n_prompt := len(prompt_tokens)
	if n_prompt == 0 && len(prompt) > 0 {
		// Tokenization failed.
		cancelled = true
	}

	// Ingest the prompt; the last sampled token seeds the generation loop.
	id: llama_c.Llama_Token = llama_c.LLAMA_TOKEN_NULL
	if n_prompt > 0 {
		id = ingest_prompt(g, prompt_tokens, batch_capacity(g.cfg), cancel_flag, on_progress, user_data, &cancelled)
	}
	cancelled = cancelled || id == llama_c.LLAMA_TOKEN_NULL

	// Generation continues at the position after the ingested prompt tokens.
	pos: u32 = u32(n_prompt)
	turn := 0
	gen_total: i32 = 0
	for !cancelled && !stopped && turn < int(n_turns) {
		if llama_cancelled(cancel_flag) {
			cancelled = true
			break
		}

		if llama_c.llama_vocab_is_eog(g.vocab, id) {
			if turn + 1 >= int(n_turns) do break
			// Decode the EOG token to close this turn and sample the first
			// token of the next one.
			if !decode_token(g, id, pos, true) do break
			pos += 1
			turn += 1
			id = llama_c.llama_sampler_sample(g.sampler, g.ctx, g.batch.n_tokens - 1)
			continue
		}

		piece := token_piece(g.vocab, id, &piece_buf)
		if len(piece) > 0 {
			for _ in 0..<len(piece) do append(&text_ids, id)
			for b in piece do append(&text, b)

			// Stop sequences: only the tail can newly complete a sequence.
			stop_start, hit := stop_suffix_pos(text[:], stop_seqs)
			if hit {
				trunc = stop_start
				stopped = true
				break
			}

			// Hold back a suffix that may still become a stop sequence; only
			// stream complete runes ending before that window.
			limit := len(text) - max(0, max_stop_len(stop_seqs) - 1)
			if !flush_text(text[:], text_ids[:], &streamed, limit, false, on_token, user_data) {
				cancelled = true
				break
			}
		}

		gen_total += 1
		if max_tokens > 0 && gen_total >= max_tokens do break

		if !decode_token(g, id, pos, true) do break
		pos += 1
		id = llama_c.llama_sampler_sample(g.sampler, g.ctx, g.batch.n_tokens - 1)
	}

	// Final flush of every held-back byte of the emitted text.
	if !stopped {
		trunc = len(text)
	}
	if !flush_text(text[:trunc], text_ids[:trunc], &streamed, trunc, true, on_token, user_data) {
		// The callback cancelled; the returned text is still complete.
	}

	return strings.clone_from_bytes(text[:trunc])
}

// ---------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------

// Logical llama_decode batch chunk size kept with the generator: the config
// batch size, or 512 tokens when unset (llama_context_params defaults to
// 2048/512 for n_batch/n_ubatch; a 512-token chunk decodes in a few passes).
@(private)
batch_capacity :: proc(cfg: Generator_Config) -> i32 {
	if cfg.n_batch > 0 do return i32(cfg.n_batch)
	return 512
}

@(private)
max_stop_len :: proc(stop_seqs: []string) -> int {
	n := 0
	for s in stop_seqs do n = max(n, len(s))
	return n
}

// If the current tail of `text` completes any stop sequence, return the byte
// offset where that match begins, and hit = true.
@(private)
stop_suffix_pos :: proc(text: []u8, stop_seqs: []string) -> (int, bool) {
	for s in stop_seqs {
		n := len(s)
		if n == 0 || n > len(text) do continue
		p := len(text) - n
		matched := true
		for j in 0..<n {
			if text[p + j] != s[j] {
				matched = false
				break
			}
		}
		if matched {
			return p, true
		}
	}
	return 0, false
}

// Byte length of the (possibly invalid or partial) rune whose leading byte is
// `b`; continuation bytes and invalid bytes count as 1.
@(private)
rune_byte_len :: proc "contextless" (b: u8) -> int {
	switch {
	case b < 0x80: return 1
	case b < 0xC0: return 1 // stray continuation byte
	case b < 0xE0: return 2
	case b < 0xF0: return 3
	case b < 0xF8: return 4
	case b < 0xFC: return 5
	case:          return 6
	}
}

// Stream complete runes starting at `streamed^` up to byte `limit` through
// `on_token`, attributing each rune to the token id of its first byte. With
// `final`, any trailing incomplete rune is flushed as raw bytes. Returns
// false when the callback requested cancellation. With a nil callback this
// only advances `streamed^`.
@(private)
flush_text :: proc(
	buf:       []u8,
	ids:       []llama_c.Llama_Token,
	streamed:  ^int,
	limit:     int,
	final:     bool,
	on_token:  Token_Callback,
	user_data: rawptr,
) -> bool {
	end := min(limit, len(buf))
	if on_token == nil {
		streamed^ = end
		return true
	}

	i := min(streamed^, end)
	for i < end {
		need := rune_byte_len(buf[i])
		if need > 1 {
			if i + need > len(buf) {
				// Trailing incomplete rune.
				if final {
					// No more bytes will arrive; emit the rest raw.
					piece := strings.clone_from_bytes(buf[i:end])
					keep := on_token(ids[i], piece, user_data)
					mem.delete_string(piece)
					streamed^ = end
					return keep
				}
				break
			}
			if i + need > end {
				// Complete rune, but past the hold-back window.
				break
			}
		}

		piece := strings.clone_from_bytes(buf[i:i + need])
		keep := on_token(ids[i], piece, user_data)
		mem.delete_string(piece)
		i += need
		streamed^ = i
		if !keep {
			return false
		}
	}
	return true
}

// Decode one token at sequence position `pos`, optionally requesting its
// logits. Returns false when llama_decode failed.
@(private)
decode_token :: proc(g: ^Llama_Generator, id: llama_c.Llama_Token, pos: u32, want_logits: bool) -> bool {
	b := &g.batch
	b.n_tokens = 1
	b.token[0]     = id
	b.pos[0]       = llama_c.Llama_Pos(pos)
	b.n_seq_id[0]  = 1
	b.seq_id[0][0] = llama_c.Llama_Seq_Id(0)
	if want_logits {
		b.logits[0] = 1
	} else {
		b.logits[0] = 0
	}
	return llama_c.llama_decode(g.ctx, g.batch) == 0
}

// Decode the prompt in chunks of at most `batch_cap` tokens; only the very
// last token requests logits. `cancel_flag` is checked between chunks and
// `on_progress` reports (and can cancel) ingestion. Returns the token
// sampled for the final prompt position, or LLAMA_TOKEN_NULL on
// cancellation (`cancelled^` set) or decode error.
@(private)
ingest_prompt :: proc(
	g:           ^Llama_Generator,
	tokens:      []llama_c.Llama_Token,
	batch_cap:   i32,
	cancel_flag: ^bool,
	on_progress: Progress_Callback,
	user_data:   rawptr,
	cancelled:   ^bool,
) -> llama_c.Llama_Token {
	pos: u32 = 0
	i := 0
	for i < len(tokens) {
		if llama_cancelled(cancel_flag) {
			cancelled^ = true
			return llama_c.LLAMA_TOKEN_NULL
		}
		n_chunk := min(len(tokens) - i, int(batch_cap))
		b := &g.batch
		b.n_tokens = i32(n_chunk)
		for j in 0..<n_chunk {
			b.token[j]     = tokens[i + j]
			b.pos[j]       = llama_c.Llama_Pos(pos)
			b.n_seq_id[j]  = 1
			b.seq_id[j][0] = llama_c.Llama_Seq_Id(0)
			b.logits[j]    = 0
			pos += 1
		}
		b.logits[n_chunk - 1] = 1

		if llama_c.llama_decode(g.ctx, g.batch) != 0 {
			cancelled^ = true
			return llama_c.LLAMA_TOKEN_NULL
		}

		if on_progress != nil {
			if !on_progress(f32(i + n_chunk) / f32(len(tokens)), user_data) {
				cancelled^ = true
				return llama_c.LLAMA_TOKEN_NULL
			}
		}
		i += n_chunk
	}
	return llama_c.llama_sampler_sample(g.sampler, g.ctx, g.batch.n_tokens - 1)
}

// Detokenize one token into a reusable byte buffer (growing it when needed).
// Special/control tokens (BOS, EOG, ...) yield no piece. Returns a view into
// `buf`; the contents stay valid until the next call.
@(private)
token_piece :: proc(vocab: ^llama_c.Llama_Vocab, id: llama_c.Llama_Token, buf: ^[dynamic]u8) -> []u8 {
	if len(buf^) == 0 {
		resize(buf, 512)
	}
	n := llama_c.llama_token_to_piece(vocab, id, &buf^[0], i32(len(buf^)), 0, false)
	if n < 0 {
		// Buffer smaller than the piece: grow to the required size and retry.
		resize(buf, int(-n))
		n = llama_c.llama_token_to_piece(vocab, id, &buf^[0], i32(len(buf^)), 0, false)
	}
	if n <= 0 do return nil
	return buf^[:n]
}

// Tokenize `prompt` into an owned token buffer (adds BOS/specials like the
// C layer does; llama_tokenize reports a required count of -n when the
// initial buffer, sized worst-case at one token per byte, is too small and
// the allocation is retried once). With an empty prompt an all-special
// buffer (e.g. the BOS token) is returned when the vocabulary has one.
// Returns nil when tokenization ultimately fails. The caller frees the
// returned buffer with `delete` (nil-safe).
@(private)
tokenize_prompt :: proc(vocab: ^llama_c.Llama_Vocab, prompt_cs: cstring, prompt: string) -> []llama_c.Llama_Token {
	buf := max(len(prompt) + 1, 2)
	tokens := make([]llama_c.Llama_Token, buf)
	n := llama_c.llama_tokenize(vocab, prompt_cs, i32(len(prompt)), &tokens[0], i32(buf), true, true)
	if n < 0 {
		buf = int(min(-n, llama_c.llama_vocab_n_tokens(vocab) + 1))
		delete(tokens)
		tokens = make([]llama_c.Llama_Token, buf)
		n = llama_c.llama_tokenize(vocab, prompt_cs, i32(len(prompt)), &tokens[0], i32(buf), true, true)
	}
	if n <= 0 {
		delete(tokens)
		return nil
	}
	return tokens[:n]
}

// Dereference a cancel flag, treating nil as "not cancelled".
@(private)
llama_cancelled :: proc "contextless" (f: ^bool) -> bool {
	return f != nil && f^
}