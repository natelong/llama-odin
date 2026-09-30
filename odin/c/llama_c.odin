// Low-level C-ABI foreign declarations for libllama_odin.
//
// Faithful mapping of include/llama_odin.h (which includes include/llama.h and
// the ggml backend device interface). Opaque C structs are declared here as
// empty structs and used only through pointers; structs passed by value
// (params, batch, sampler chain) are mirrored field-for-field with the C
// layout.
package llama_c

import "core:c"

when ODIN_OS == .Darwin {
	foreign import llama_lib "system:libllama_odin.dylib"
} else when ODIN_OS == .Linux {
	foreign import llama_lib "system:libllama_odin.so"
} else {
	foreign import llama_lib "system:llama_odin.lib"
}

@(default_calling_convention = "c")
foreign llama_lib {

	// ------------------------------------------------------------------
	// ABI version & logging (include/llama_odin.h)
	// ------------------------------------------------------------------

	llama_odin_abi_version :: proc() -> u32 ---
	llama_odin_silence_logs :: proc() ---

	// ------------------------------------------------------------------
	// Backend lifecycle & logging (include/llama.h)
	// ------------------------------------------------------------------

	llama_backend_init :: proc() ---
	llama_backend_free :: proc() ---

	llama_log_set :: proc(log_callback: Ggml_Log_Callback, user_data: rawptr) ---
	llama_log_get :: proc(log_callback: ^Ggml_Log_Callback, user_data: ^rawptr) ---

	// ------------------------------------------------------------------
	// Hardware device enumeration (ggml-backend.h)
	// ------------------------------------------------------------------

	ggml_backend_dev_count :: proc() -> c.size_t ---
	ggml_backend_dev_get :: proc(index: c.size_t) -> ^Ggml_Backend_Dev ---
	ggml_backend_dev_name :: proc(device: ^Ggml_Backend_Dev) -> cstring ---
	ggml_backend_dev_description :: proc(device: ^Ggml_Backend_Dev) -> cstring ---
	ggml_backend_dev_type :: proc(device: ^Ggml_Backend_Dev) -> Ggml_Backend_Dev_Type ---
	ggml_backend_dev_memory :: proc(device: ^Ggml_Backend_Dev, free: ^c.size_t, total: ^c.size_t) ---
	ggml_backend_dev_init :: proc(device: ^Ggml_Backend_Dev, params: cstring) -> ^Ggml_Backend ---

	// ------------------------------------------------------------------
	// Model & vocab (include/llama.h)
	// ------------------------------------------------------------------

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

	llama_tokenize :: proc(
		vocab:         ^Llama_Vocab,
		text:          cstring,
		text_len:      i32,
		tokens:        [^]Llama_Token,
		n_tokens_max:  i32,
		add_special,
		parse_special: bool,
	) -> i32 ---

	llama_token_to_piece :: proc(
		vocab:    ^Llama_Vocab,
		token:    Llama_Token,
		buf:      [^]u8,
		length:   i32,
		lstrip:   i32,
		special:  bool,
	) -> i32 ---

	// ------------------------------------------------------------------
	// Context & execution (include/llama.h)
	// ------------------------------------------------------------------

	llama_context_default_params :: proc() -> Llama_Context_Params ---
	llama_init_from_model :: proc(model: ^Llama_Model, params: Llama_Context_Params) -> ^Llama_Context ---
	llama_free :: proc(ctx: ^Llama_Context) ---

	llama_n_ctx :: proc(ctx: ^Llama_Context) -> u32 ---

	llama_batch_init :: proc(n_tokens_alloc, embd, n_seq_max: i32) -> Llama_Batch ---
	llama_batch_free :: proc(batch: Llama_Batch) ---
	llama_decode :: proc(ctx: ^Llama_Context, batch: Llama_Batch) -> i32 ---
	llama_get_logits_ith :: proc(ctx: ^Llama_Context, i: i32) -> ^f32 ---

	// ------------------------------------------------------------------
	// Sampling (include/llama.h)
	// ------------------------------------------------------------------

	llama_sampler_chain_default_params :: proc() -> Llama_Sampler_Chain_Params ---
	llama_sampler_chain_init :: proc(params: Llama_Sampler_Chain_Params) -> ^Llama_Sampler ---
	llama_sampler_chain_add :: proc(chain, smpl: ^Llama_Sampler) ---
	llama_sampler_init_temp :: proc(t: f32) -> ^Llama_Sampler ---
	llama_sampler_init_top_p :: proc(p: f32, min_keep: c.size_t) -> ^Llama_Sampler ---
	llama_sampler_init_dist :: proc(seed: u32) -> ^Llama_Sampler ---
	llama_sampler_sample :: proc(smpl: ^Llama_Sampler, ctx: ^Llama_Context, idx: i32) -> Llama_Token ---
	llama_sampler_accept :: proc(smpl: ^Llama_Sampler, token: Llama_Token) ---
	llama_sampler_free :: proc(smpl: ^Llama_Sampler) ---

	// ------------------------------------------------------------------
	// MTP staging (include/llama_odin.h)
	// ------------------------------------------------------------------

	llama_odin_set_embeddings_nextn :: proc(ctx: ^Llama_Context, value, masked: c.bool) ---
	llama_odin_get_embeddings_nextn :: proc(ctx: ^Llama_Context) -> ^f32 ---
	llama_odin_get_embeddings_nextn_ith :: proc(ctx: ^Llama_Context, i: i32) -> ^f32 ---
	llama_odin_set_nextn_layer_offset :: proc(ctx: ^Llama_Context, offset: i32) ---
	llama_odin_get_ctx_other :: proc(ctx: ^Llama_Context) -> ^Llama_Context ---
}

// ---------------------------------------------------------------------
// Opaque C structs (used only via pointers)
// ---------------------------------------------------------------------

Llama_Vocab :: struct {}
Llama_Model :: struct {}
Llama_Context :: struct {}
Llama_Sampler :: struct {}

Ggml_Backend_Dev :: struct {}
Ggml_Backend :: struct {}

// ---------------------------------------------------------------------
// Fundamental C types
// ---------------------------------------------------------------------

// typedef int32_t llama_token; (llama.h:69)
Llama_Token :: distinct i32
// typedef int32_t llama_pos;
Llama_Pos :: distinct i32
// typedef int32_t llama_seq_id;
Llama_Seq_Id :: distinct i32

// #define LLAMA_TOKEN_NULL -1
LLAMA_TOKEN_NULL :: -1
// #define LLAMA_ODIN_ABI_VERSION 1
LLAMA_ODIN_ABI_VERSION :: 1

// #define LLAMA_DEFAULT_SEED 0xFFFFFFFF
LLAMA_DEFAULT_SEED :: 0xFFFFFFFF

// typedef enum ggml_log_level { ... } (ggml.h)
Ggml_Log_Level :: enum c.int {
	NONE  = 0,
	DEBUG = 1,
	INFO  = 2,
	WARN  = 3,
	ERROR = 4,
	CONT  = 5,
}

// typedef void (*ggml_log_callback)(enum ggml_log_level, const char *, void *);
Ggml_Log_Callback :: #type proc "c" (level: Ggml_Log_Level, text: cstring, user_data: rawptr)

// typedef bool (*llama_progress_callback)(float progress, void * user_data);
Llama_Progress_Callback :: #type proc "c" (progress: f32, user_data: rawptr) -> bool

// typedef bool (*ggml_abort_callback)(void * data); (ggml.h)
Ggml_Abort_Callback :: #type proc "c" (data: rawptr) -> bool

// typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);
Ggml_Backend_Sched_Eval_Callback :: #type proc "c" (tensor: rawptr, ask: bool, user_data: rawptr) -> bool

// ---------------------------------------------------------------------
// Backend device types (ggml-backend.h)
// ---------------------------------------------------------------------

// enum ggml_backend_dev_type
Ggml_Backend_Dev_Type :: enum c.int {
	CPU    = 0, // CPU device using system memory
	GPU    = 1, // GPU device using dedicated memory
	IGPU   = 2, // integrated GPU device using host memory
	ACCEL  = 3, // accelerator used together with the CPU backend (e.g. BLAS)
	META   = 4, // "meta" device wrapping multiple other devices
}

// ---------------------------------------------------------------------
// llama.h enums (C int, faithful values)
// ---------------------------------------------------------------------

// enum llama_split_mode
Llama_Split_Mode :: enum c.int {
	NONE   = 0, // single GPU
	LAYER  = 1, // split layers and KV across GPUs
	ROW    = 2, // split layers and KV across GPUs, tensor parallelism if supported
	TENSOR = 3,
}

// enum llama_load_mode
Llama_Load_Mode :: enum c.int {
	AUTO       = -1, // auto-detect based on device capabilities
	NONE       =  0, // no special loading mode
	MMAP       =  1, // memory map the model
	MLOCK      =  2, // force system to keep model in RAM
	MMAP_MLOCK =  3, // mmap + mlock
	DIRECT_IO  =  4, // use direct I/O if available
}

// enum llama_lazy_mode
Llama_Lazy_Mode :: enum c.int {
	OFF  = 0, // always read the whole tensor up front
	AUTO = 1, // lazy only for marked tensors larger than 4 GiB (requires mmap)
	ON   = 2, // read the rows of tensors marked by the arch on demand
}

// enum llama_context_type
Llama_Context_Type :: enum c.int {
	DEFAULT = 0,
	MTP     = 1,
}

// enum llama_rope_scaling_type
Llama_Rope_Scaling_Type :: enum c.int {
	UNSPECIFIED = -1,
	NONE        =  0,
	LINEAR      =  1,
	YARN        =  2,
	LONGROPE    =  3,
}

// enum llama_pooling_type
Llama_Pooling_Type :: enum c.int {
	UNSPECIFIED = -1,
	NONE        =  0,
	MEAN        =  1,
	CLS         =  2,
	LAST        =  3,
	RANK        =  4,
}

// enum llama_attention_type
Llama_Attention_Type :: enum c.int {
	UNSPECIFIED = -1,
	CAUSAL      =  0,
	NON_CAUSAL  =  1,
}

// enum llama_flash_attn_type
Llama_Flash_Attn_Type :: enum c.int {
	AUTO     = -1,
	DISABLED =  0,
	ENABLED  =  1,
}

// ---------------------------------------------------------------------
// Value structs (field order and layout must match C exactly)
// ---------------------------------------------------------------------

// typedef struct llama_token_data { ... } (llama.h)
Llama_Token_Data :: struct {
	id:    Llama_Token, // token id
	logit: f32,         // log-odds of the token
	p:     f32,         // probability of the token
}

// typedef struct llama_token_data_array { ... } (llama.h)
Llama_Token_Data_Array :: struct {
	// NOTE: this pointer can be modified by the samplers
	data:     [^]Llama_Token_Data,
	size:     c.size_t,
	selected: c.int64_t, // index in the data array (i.e. not the token id)
	sorted:   c.bool,    // do not assume the data is sorted - always check this flag
}

// typedef struct llama_batch { ... } (llama.h)
Llama_Batch :: struct {
	n_tokens: i32,
	token:    [^]Llama_Token,
	embd:     [^]f32,
	pos:      [^]Llama_Pos,
	n_seq_id: [^]i32,
	seq_id:   [^][^]Llama_Seq_Id,
	logits:   [^]i8,
}

// union { int64_t val_i64; double val_f64; bool val_bool; char val_str[128]; }
// #raw_union to match a C union (no tag byte)
Llama_Model_Kv_Override_Value :: struct #raw_union {
	val_i64:  c.int64_t,
	val_f64:  f64,
	val_bool: c.bool,
	val_str:  [128]c.char,
}

// struct llama_model_kv_override (llama.h)
Llama_Model_Kv_Override :: struct {
	tag:  c.int, // enum llama_model_kv_override_type
	key:  [128]c.char,
	value: Llama_Model_Kv_Override_Value,
}

// struct llama_model_tensor_buft_override (llama.h)
Llama_Model_Tensor_Buft_Override :: struct {
	pattern: cstring,
	buft:    rawptr, // ggml_backend_buffer_type_t
}

// struct llama_model_params (llama.h)
Llama_Model_Params :: struct {
	// NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
	devices: [^]Ggml_Backend_Dev,

	// NULL-terminated list of buffer types to use for tensors that match a pattern
	tensor_buft_overrides: [^]Llama_Model_Tensor_Buft_Override,

	n_gpu_layers: i32, // number of layers to store in VRAM, a negative value means all layers
	split_mode:   Llama_Split_Mode, // how to split the model across multiple GPUs
	load_mode:    Llama_Load_Mode,  // how to load the model
	lazy_mode:    Llama_Lazy_Mode,  // on-demand reading of tensors marked by the arch

	// the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
	main_gpu: i32,

	// proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
	tensor_split: [^]f32,

	// Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
	// If the provided progress_callback returns true, model loading continues.
	// If it returns false, model loading is immediately aborted.
	progress_callback: Llama_Progress_Callback,

	// context pointer passed to the progress callback
	progress_callback_user_data: rawptr,

	// override key-value pairs of the model meta data
	kv_overrides: [^]Llama_Model_Kv_Override,

	// Keep the booleans together to avoid misalignment during copy-by-value.
	vocab_only:    c.bool, // only load the vocabulary, no weights
	check_tensors: c.bool, // validate model tensor data
	use_extra_bufts: c.bool, // use extra buffer types (used for weight repacking)
	no_host:       c.bool, // bypass host buffer allowing extra buffers to be used
	no_alloc:      c.bool, // only load metadata and simulate memory allocations
	load_mtp:      c.bool, // whether to load MTP layers
}

// struct llama_context_params (llama.h)
Llama_Context_Params :: struct {
	n_ctx:  u32, // text context, 0 = from model
	n_batch: u32, // logical maximum batch size that can be submitted to llama_decode
	n_ubatch: u32, // physical maximum batch size
	n_seq_max: u32, // max number of sequences (i.e. distinct states for recurrent models)
	n_rs_seq: u32, // number of recurrent-state snapshots per seq for rollback (0 = no rollback) [EXPERIMENTAL]
	n_outputs_max: u32, // max outputs in a ubatch (0 = n_batch)
	n_outputs_max_per_seq: u32, // max outputs per sequence (0 = n_outputs_max)
	n_threads: i32, // number of threads to use for generation
	n_threads_batch: i32, // number of threads to use for batch processing

	ctx_type:           Llama_Context_Type,      // set the context type (e.g. MTP)
	rope_scaling_type:  Llama_Rope_Scaling_Type, // RoPE scaling type
	pooling_type:       Llama_Pooling_Type,      // whether to pool (sum) embedding results by sequence id
	attention_type:     Llama_Attention_Type,    // attention type to use for embeddings
	flash_attn_type:    Llama_Flash_Attn_Type,   // when to enable Flash Attention

	// ref: https://github.com/ggml-org/llama.cpp/pull/2054
	rope_freq_base:  f32, // RoPE base frequency, 0 = from model
	rope_freq_scale: f32, // RoPE frequency scaling factor, 0 = from model
	yarn_ext_factor: f32, // YaRN extrapolation mix factor, negative = from model
	yarn_attn_factor: f32, // YaRN magnitude scaling factor
	yarn_beta_fast:  f32, // YaRN low correction dim
	yarn_beta_slow:  f32, // YaRN high correction dim
	yarn_orig_ctx:   u32, // YaRN original context size
	defrag_thold:    f32, // [DEPRECATED] defrag the KV cache if holes/size > thold, <= 0 disabled

	cb_eval:          Ggml_Backend_Sched_Eval_Callback,
	cb_eval_user_data: rawptr,

	type_k: c.int, // enum ggml_type; data type for K cache [EXPERIMENTAL]
	type_v: c.int, // enum ggml_type; data type for V cache [EXPERIMENTAL]

	// Abort callback; if it returns true, execution of llama_decode() will be aborted
	abort_callback:      Ggml_Abort_Callback,
	abort_callback_data: rawptr,

	// Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
	embeddings: c.bool, // if true, extract embeddings (together with logits)
	offload_kqv: c.bool, // offload the KQV ops (including the KV cache) to GPU
	no_perf:    c.bool, // measure performance timings
	op_offload: c.bool, // offload host tensor operations to device
	swa_full:   c.bool, // use full-size SWA cache
	kv_unified: c.bool, // use a unified buffer across the input sequences

	// [EXPERIMENTAL] backend sampler chain configuration (the caller keeps the samplers alive)
	samplers: [^]Llama_Sampler_Seq_Config,
	n_samplers: c.size_t,

	// a source/target/parent context; can be used e.g. by sharing results or llama_memory
	// between two contexts
	ctx_other: ^Llama_Context,
}

// struct llama_sampler_seq_config (llama.h)
Llama_Sampler_Seq_Config :: struct {
	seq_id:  Llama_Seq_Id,
	sampler: ^Llama_Sampler,
}

// typedef struct llama_sampler_chain_params { bool no_perf; } (llama.h)
Llama_Sampler_Chain_Params :: struct {
	no_perf: c.bool, // whether to measure performance timings
}

// Compile-time ABI layout guards (mirrors sizeof/offsetof from the C headers,
// computed with clang -Iinclude against llama_odin.h; identical for all C-ABI
// targets this package supports).
#assert(size_of(Llama_Token_Data) == 12)
#assert(size_of(Llama_Token_Data_Array) == 32)
#assert(size_of(Llama_Batch) == 56)
#assert(size_of(Llama_Model_Params) == 80)
#assert(size_of(Llama_Model_Kv_Override) == 264)
#assert(size_of(Llama_Context_Params) == 160)
#assert(size_of(Llama_Sampler_Chain_Params) == 1)
#assert(size_of(Llama_Sampler_Seq_Config) == 16)
#assert(offset_of(Llama_Token_Data_Array, selected) == 16)
#assert(offset_of(Llama_Batch, logits) == 48)
#assert(offset_of(Llama_Model_Params, tensor_split) == 40)
#assert(offset_of(Llama_Model_Params, kv_overrides) == 64)
#assert(offset_of(Llama_Model_Params, vocab_only) == 72)
#assert(offset_of(Llama_Context_Params, n_threads_batch) == 32)
#assert(offset_of(Llama_Context_Params, rope_freq_base) == 56)
#assert(offset_of(Llama_Context_Params, cb_eval) == 88)
#assert(offset_of(Llama_Context_Params, type_k) == 104)
#assert(offset_of(Llama_Context_Params, embeddings) == 128)
#assert(offset_of(Llama_Context_Params, samplers) == 136)
#assert(offset_of(Llama_Context_Params, ctx_other) == 152)