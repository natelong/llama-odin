// Multi-Token Prediction (MTP) speculative decoding driver for
// Llama_Generator (odin/llama/llama.odin).
//
// A single-sequence port of llama.cpp's common/speculative.cpp
// `draft-mtp` implementation, using the llama_odin_* staging shims from
// include/llama_odin.h:
//
//   - llama_odin_set_embeddings_nextn(ctx, true, false) on the target
//     context: every decoded token's hidden row (n_embd_out wide) is
//     stashed unmasked, indexed by its raw batch position.
//   - llama_odin_set_embeddings_nextn(ctx_dft, true, true) on the draft
//     context: hidden rows are stashed only for logits-flagged tokens,
//     read back with llama_odin_get_embeddings_nextn_ith.
//   - llama_odin_set_nextn_layer_offset(ctx_dft, head): selects the
//     trained MTP head (nextn layer) a chain_heads decode runs under.
//
// How the loop works (generator_generate delegates here whenever the
// generator carries an MTP draft context — g.other_ctx != nil):
//
//   - The model GGUF carries one or more extra `nextn` blocks after the
//     trunk (`load_mtp = true`); the draft context's graphs (context type
//     MTP, ctx_other = target context) run only those blocks, consuming
//     (hidden row h, token) pairs and producing the next token's logits
//     plus the next hidden row.
//   - Every target decode (prefill chunk, plain token, verify batch) runs
//     a mtp_process hook: the draft context catches up its KV on the same
//     tokens (unless it shares the target's cache), and the target's
//     hidden rows are stashed for the accept step.
//   - Each round drafts up to n_max tokens greedily from the MTP head
//     (top-k 10, stop when the head's top probability drops under p_min),
//     then verifies the whole batch in one target decode: the sampled
//     token plus the drafts, logits on every row — row r's logits are the
//     target's distribution for pos+r+1. Leading drafts that match the
//     target's own samples are accepted; the first mismatch's target
//     sample replaces the draft token. A draft is only ever streamed when
//     it equals the target's sample at its position, so output quality is
//     identical to plain generation.
//   - mtp_accept re-seeds the next draft with the hidden row of the last
//     accepted position; rejected target-KV positions are rolled back
//     with llama_memory_seq_rm.
//
// Three modes mirror the C++ driver, detected once at driver creation:
//   - single-head (DeepSeek/GLM/Qwen3.5 families): one trained MTP head,
//     separate draft KV growing per step.
//   - chain_heads (e.g. Step35): n_mtp_layers > 1 with separate draft KV —
//     one trained head per draft step, selected per decode with
//     llama_odin_set_nextn_layer_offset, each decoding the full (hidden,
//     token) prefix.
//   - is_mem_shared (e.g. Gemma4 assistant draft models): the draft
//     context shares the target's KV cache, so catch-ups are skipped and
//     all draft steps decode at the sampled position.
//
// Single-sequence only — the generator always drives llama seq 0.
package llama

import "core:fmt"
import "core:math"
import "core:mem"
import "core:strings"
import "base:runtime"
import llama_c "../c"


// The C++ draft sampler keeps the best 10 candidates (common_sampler
// sparams.top_k = 10) and gates drafts on the top candidate's probability.
MTP_DRAFT_TOP_K :: 10

// Draft-round knobs, mirroring llama.cpp's common_params_speculative_draft
// defaults (n_max=3, n_min=0, p_min=0.0).
Mtp_Config :: struct {
	n_max: i32, // maximum tokens drafted and verified per round
	n_min: i32, // rounds drafting fewer than this many tokens verify nothing
	p_min: f32, // draft continues only while the head's top probability >= p_min
}

// Mtp_Config derived from a generator's configuration. The load_mtp /
// mtp_context / mtp_model_path / nextn_layer_offset part of
// Generator_Config already drove the draft context creation and the
// llama_odin_set_nextn_layer_offset passthrough in generator_new; this
// covers the draft-round tuning knobs with the same zero-value semantics.
@(private)
mtp_config_from_generator :: proc(cfg: Generator_Config) -> Mtp_Config {
	return Mtp_Config{
		n_max = cfg.mtp_n_draft > 0 ? cfg.mtp_n_draft : 3,
		n_min = max(cfg.mtp_n_min, 0),
		p_min = cfg.mtp_p_min,
	}
}

// The speculative decoding driver: the generator's target and draft contexts
// plus the cross-decode hidden-row state. Lives as long as its generator
// (created by generator_new when the MTP draft context exists, freed first
// in generator_destroy). Always drives llama seq 0.
Mtp_Driver :: struct {
	g:             ^Llama_Generator,
	ctx_tgt:       ^llama_c.Llama_Context,
	ctx_dft:       ^llama_c.Llama_Context,
	// Width of the nextn hidden rows (llama_model_n_embd_out); the target
	// and draft model must agree on it.
	n_embd:        i32,
	n_vocab:       i32,
	n_mtp_layers:  i32,
	// The draft context shares the target's KV cache (Gemma4 assistant) —
	// llama_odin_get_ctx_other(ctx_dft) == ctx_tgt.
	is_mem_shared: bool,
	// n_mtp_layers > 1 && !is_mem_shared: one trained head per draft step.
	chain_heads:   bool,
	cfg:           Mtp_Config,
	// Effective n_max (chain_heads caps drafts at the trained head count).
	n_max:         i32,

	// The draft context's batch: token AND embedding inputs, both backing
	// the C llama_batch arrays.
	allocator:     runtime.Allocator,

	batch:         llama_c.Llama_Batch,
	batch_cap:     i32,
	batch_token:   []llama_c.Llama_Token, // Odin-owned array behind batch.token

	// Hidden row of the last target-processed token — pairs with the next
	// token (at the next position) in the draft seed decode.
	pending_h:     []f32,
	// Hidden rows of the most recent target batch (row 0 = its first token),
	// consumed by mtp_accept after verification.
	verify_h:      []f32,
	verify_h_rows: i32,

	// Speculative decoding diagnostics (cumulative over generations):
	// verification rounds started and draft tokens drafted / accepted.
	n_verif_rounds:    i64,
	n_draft_tokens:    i64,
	n_accepted_tokens: i64,
}

// Create and stage the MTP driver for a generator that already carries a
// draft context (g.other_ctx). Returns (nil, message) when the driver cannot
// be built; the caller then aborts context creation (generator_new).
mtp_driver_new :: proc(
	g:         ^Llama_Generator,
	allocator := context.allocator,
) -> (d: ^Mtp_Driver, err: string) {
	d = new(Mtp_Driver, allocator)
	d^ = Mtp_Driver{
		g         = g,
		ctx_tgt   = g.ctx,
		ctx_dft   = g.other_ctx,
		cfg       = mtp_config_from_generator(g.cfg),
		allocator = allocator,
	}

	draft_model := g.mtp_model != nil ? g.mtp_model : g.model

	// Staging width: draft and target models must agree on the nextn hidden
	// row width (the C++ driver asserts the same).
	d.n_embd = llama_c.llama_model_n_embd_out(draft_model)
	n_embd_tgt := llama_c.llama_model_n_embd_out(g.model)
	if d.n_embd <= 0 || n_embd_tgt != d.n_embd {
		err = fmt.aprintf(
			"llama.mtp: draft model n_embd_out (%d) does not match target n_embd_out (%d)",
			d.n_embd, n_embd_tgt,
		)
		free(d, allocator)
		return nil, err
	}

	// The draft side vocabulary drives the head's logits row width.
	d.n_vocab = llama_c.llama_vocab_n_tokens(llama_c.llama_model_get_vocab(draft_model))

	// The draft side model's nextn layer count drives the chain_heads mode.
	d.n_mtp_layers = llama_c.llama_model_n_layer_nextn(draft_model)
	if d.n_mtp_layers <= 0 {
		d.n_mtp_layers = 1
	}
	d.is_mem_shared = llama_c.llama_odin_get_ctx_other(d.ctx_dft) == d.ctx_tgt
	d.chain_heads   = d.n_mtp_layers > 1 && !d.is_mem_shared
	d.n_max         = d.cfg.n_max
	if d.chain_heads {
		// Each trained head drafts at most one token per round.
		d.n_max = min(d.n_max, d.n_mtp_layers)
	}

	// Enable nextn embedding staging on both contexts, exactly as the C++
	// driver does in its constructor:
	//   - target: unmasked — every decoded token gets a hidden row, indexed
	//     by its raw batch position;
	//   - draft: masked — rows are stashed only for logits-flagged tokens,
	//     read back with llama_odin_get_embeddings_nextn_ith.
	// Must happen before the first decode on each context (the decode
	// buffers are sized when their graphs are first built).
	llama_c.llama_odin_set_embeddings_nextn(d.ctx_tgt, true, false)
	llama_c.llama_odin_set_embeddings_nextn(d.ctx_dft, true, true)

	// The draft batch needs both token and embedding inputs;
	// llama_batch_init allocates only one of the two, so the token array is
	// allocated here (and hidden from llama_batch_free at teardown).
	d.batch_cap = max(
		batch_capacity(g.cfg),
		d.n_max + 2, // draft prefix rebuilds carry up to n_max + 1 rows
	)
	d.batch = llama_c.llama_batch_init(d.batch_cap, d.n_embd, 1)
	if d.batch.embd == nil {
		// Release whatever the partially allocated batch did get.
		llama_c.llama_batch_free(d.batch)
		d.batch = llama_c.Llama_Batch{}
		err = "llama.mtp: failed to allocate the draft batch"
		free(d, allocator)
		return nil, err
	}
	d.batch_token = make([]llama_c.Llama_Token, d.batch_cap, allocator)
	d.batch.token = ([^]llama_c.Llama_Token)(&d.batch_token[0])

	d.pending_h = make([]f32, d.n_embd, allocator)
	return d, ""
}

// Free the driver and its own resources. The contexts and models belong to
// the enclosing Llama_Generator and are released there (driver first, then
// the draft context, then the target).
mtp_driver_destroy :: proc(d: ^Mtp_Driver) {
	if d == nil do return
	if d.batch_token != nil {
		// llama_batch_free C-frees the batch's arrays; the token array
		// backing batch.token is Odin-allocated, so hide it first (the C++
		// driver frees its malloc'd tokens the same way) and release it
		// through the driver's allocator instead.
		d.batch.token = nil
		delete(d.batch_token, d.allocator)
	}
	llama_c.llama_batch_free(d.batch)
	d.batch = llama_c.Llama_Batch{}
	delete(d.pending_h, d.allocator)
	delete(d.verify_h, d.allocator)
	free(d, d.allocator)
}

// Runs after every target decode (prefill chunk, plain token, verify batch):
// stashes the target's nextn hidden rows and sets the pending row to the
// last one (the hidden state that pairs with the next token at the next
// position) — for verify rounds mtp_accept then overrides it with the
// accepted row. Unless the draft context shares the target's cache, the same
// tokens are also catch-up decoded on the draft context first, with the
// hidden rows shifted by one position (row 0 carries the pending row from
// the previous batch; zeros on the very first chunk).
//
// `batch` holds the target batch's rows (tokens, positions); this must run
// before the next target decode and before the next draft, both of which
// consume the stashed rows.
mtp_process :: proc(d: ^Mtp_Driver, batch: ^llama_c.Llama_Batch) -> bool {
	n := batch.n_tokens
	if n <= 0 do return true

	row := int(d.n_embd)

	if d.is_mem_shared {
		// The draft context shares the target's cache — the target decode
		// already filled it; only the hidden-row stash is needed.
		return mtp_stash_target_rows(d, n)
	}

	// Catch-up decode on the draft context: the same tokens, the target's
	// hidden rows shifted right by one position, row 0 the pending row from
	// the previous batch.
	b := &d.batch
	b.n_tokens = n
	for k in 0..<n {
		b.token[k]     = batch.token[k]
		b.pos[k]       = batch.pos[k]
		b.n_seq_id[k]  = 1
		b.seq_id[k][0] = llama_c.Llama_Seq_Id(0)
		b.logits[k]    = 0
	}
	embd := mem.slice_ptr(b.embd, int(d.batch_cap) * row)
	// Capture the previous batch's pending row for the batch's first
	// embedding row before the stash rewrites it, then shift the target's
	// hidden rows right by one position (row k pairs them with token k-1's
	// hidden row).
	copy(embd[:row], d.pending_h)
	stash_ok := mtp_stash_target_rows(d, n)
	if !stash_ok {
		return false
	}
	stash := d.verify_h
	copy(embd[row : row + row * (int(n) - 1)], stash[:row * (int(n) - 1)])

	mem_dft := llama_c.llama_get_memory(d.ctx_dft)
	for head in 0..<d.n_mtp_layers {
		// The catch-up re-decodes the batch's position region, which carries
		// stale rows from the previous round's draft loop: clear it first so
		// the context's position-consistency checks hold. (Mirrors the C++
		// driver's chain_heads branch; also required for the single-head
		// growing-KV path.)
		llama_c.llama_memory_seq_rm(mem_dft, llama_c.Llama_Seq_Id(0), batch.pos[0], -1)
		if d.chain_heads {
			// Each head is its own decoder layer with its own KV: select it
			// for this decode.
			llama_c.llama_odin_set_nextn_layer_offset(d.ctx_dft, head)
		}
		if llama_c.llama_decode(d.ctx_dft, d.batch) != 0 {
			if d.chain_heads {
				llama_c.llama_odin_set_nextn_layer_offset(d.ctx_dft, 0)
			}
			return false
		}
	}
	if d.chain_heads {
		llama_c.llama_odin_set_nextn_layer_offset(d.ctx_dft, 0) // restore default for non-draft decodes
	}
	return true
}

// Copy one decode's worth of nextn hidden rows (n of them, n_embd_out wide
// each) from the target context into the driver's stash — and set the
// pending row to the last one (the hidden state of the batch's last token;
// mtp_accept overrides it after verify rounds). Returns false when staging
// is missing.
@(private)
mtp_stash_target_rows :: proc(d: ^Mtp_Driver, n: i32) -> bool {
	row := int(d.n_embd)
	needed := int(n) * row
	if len(d.verify_h) < needed {
		if len(d.verify_h) > 0 {
			delete(d.verify_h, d.allocator)
		}
		d.verify_h = make([]f32, needed, d.allocator)
	}
	d.verify_h_rows = n
	verify := d.verify_h[:needed]
	for i in 0..<n {
		h := llama_c.llama_odin_get_embeddings_nextn_ith(d.ctx_tgt, i)
		if h == nil {
			return false
		}
		copy(verify[int(i) * row:(int(i) + 1) * row], mem.slice_ptr(h, row))
	}
	// The pending row for the next draft seed: the last target batch row
	// (the hidden state of the batch's last token).
	copy(d.pending_h, verify[row * (int(n) - 1):row * int(n)])
	return true
}

// Fill one row of the driver batch's token slots with position and seq data
// (batch.n_tokens tracks the row count).
@(private)
mtp_draft_row :: proc(d: ^Mtp_Driver, idx: i32, tok: llama_c.Llama_Token, pos: llama_c.Llama_Pos) {
	b := &d.batch
	if idx >= d.batch_cap do return
	b.token[idx]     = tok
	b.pos[idx]       = pos
	b.n_seq_id[idx]  = 1
	b.seq_id[idx][0] = llama_c.Llama_Seq_Id(0)
	b.logits[idx]    = 0
}

// Drafts tokens for `id_last` (the token just sampled, about to sit at
// `pos0`), up to `n_cap` of them: greedy top-k(10) picks from the MTP head,
// stopping early when the head's top probability drops under p_min. The
// caller owns and deletes the returned buffer; an empty result means
// "draft nothing" (a shorter draft is a normal outcome, not an error).
mtp_draft :: proc(
	d:       ^Mtp_Driver,
	id_last: llama_c.Llama_Token,
	pos0:    llama_c.Llama_Pos,
	n_cap:   i32,
) -> (drafts: [dynamic]llama_c.Llama_Token) {
	if n_cap <= 0 do return drafts

	b := &d.batch
	row := int(d.n_embd)
	embd := mem.slice_ptr(b.embd, int(d.batch_cap) * row)

	// Seed decode: the sampled token at pos0, paired with the pending hidden
	// row (the hidden state of the token at pos0-1).
	mtp_draft_row(d, 0, id_last, pos0)
	b.n_tokens = 1
	b.logits[0] = 1
	copy(embd[:row], d.pending_h)
	i_last: i32 = 0

	// chain_heads: per-step hidden rows for rebuilding the whole prefix under
	// each head (row 0 is the pending row, like the seed).
	chain_h: []f32
	if d.chain_heads {
		chain_h = make([]f32, (int(d.n_max) + 1) * row, context.temp_allocator)
		defer delete(chain_h, context.temp_allocator)
		copy(chain_h[:row], d.pending_h)
	}

	i := 0
	for {
		// The draft region's stale rows must be cleared before the first
		// decode of the round (the catch-up of the last verify batch wrote
		// the positions now being drafted — the context's KV requires
		// positions to be consecutive or strictly forward-jumping). The
		// chain_heads path needs it before every step: each head is its own
		// decoder layer with its own KV, rebuilt from the full prefix under
		// head i (catch-up filled its KV below pos0 only).
		if i == 0 || d.chain_heads {
			llama_c.llama_memory_seq_rm(
				llama_c.llama_get_memory(d.ctx_dft), llama_c.Llama_Seq_Id(0), pos0, -1,
			)
		}
		if d.chain_heads {
			llama_c.llama_odin_set_nextn_layer_offset(d.ctx_dft, i32(i))
		}
		if llama_c.llama_decode(d.ctx_dft, d.batch) != 0 {
			break
		}

		// Greedy top-10 pick from the head's logits at the last row.
		logits := mem.slice_ptr(llama_c.llama_get_logits_ith(d.ctx_dft, i_last), int(d.n_vocab))
		id, p := mtp_top_pick(logits)
		if p < d.cfg.p_min {
			break
		}
		append(&drafts, id)
		if d.n_max <= i32(len(drafts)) || i32(len(drafts)) >= n_cap {
			break
		}

		// The hidden row of the token just drafted feeds the next step.
		h_row := mem.slice_ptr(llama_c.llama_odin_get_embeddings_nextn_ith(d.ctx_dft, i_last), row)
		if d.chain_heads {
			nd := int(len(drafts))
			copy(chain_h[nd * row:(nd + 1) * row], h_row)
			// Rebuild the whole prefix (id_last + drafts so far) under head
			// i+1; only the final row carries logits.
			n_rows := len(drafts) + 1
			b.n_tokens = i32(n_rows)
			for t in 0..<n_rows {
				tok := id_last if t == 0 else drafts[t - 1]
				mtp_draft_row(d, i32(t), tok, pos0 + llama_c.Llama_Pos(t))
				if t == n_rows - 1 {
					b.logits[t] = 1
				} else {
					b.logits[t] = 0
				}
				copy(embd[t * row:(t + 1) * row], chain_h[t * row:(t + 1) * row])
			}
		} else if d.is_mem_shared {
			// Shared-memory draft heads (e.g. Gemma4 assistant) re-read the
			// same position with a fresh hidden row each step.
			mtp_draft_row(d, 0, id, pos0)
			b.n_tokens = 1
			b.logits[0] = 1
			copy(embd[:row], h_row)
		} else {
			// Growing-KV paths re-add only the new token: the KV already
			// holds the prefix, and each step overwrites its own position.
			mtp_draft_row(d, 0, id, pos0 + llama_c.Llama_Pos(i + 1))
			b.n_tokens = 1
			b.logits[0] = 1
			copy(embd[:row], h_row)
		}
		i_last = b.n_tokens - 1
		i += 1
	}

	if d.chain_heads {
		llama_c.llama_odin_set_nextn_layer_offset(d.ctx_dft, 0) // restore default for non-draft decodes
	}
	if i32(len(drafts)) < d.cfg.n_min {
		clear(&drafts)
	}
	return drafts
}

// Records that `n_accepted` draft tokens were accepted by the target: the
// pending row becomes the hidden row of the last accepted position (with
// row 0 of the stashed verify batch being the token before the drafts, row
// n_accepted is the last fully accepted one — the hidden row pairing with
// the next sampled token).
mtp_accept :: proc(d: ^Mtp_Driver, n_accepted: i32) {
	if d.verify_h_rows <= 0 do return
	i_h := min(n_accepted, d.verify_h_rows - 1)
	row := int(d.n_embd)
	copy(d.pending_h, d.verify_h[int(i_h) * row:(int(i_h) + 1) * row])
}

// Greedy top-k pick over a logits row — the C++ draft sampler's policy
// (top-k(10) then greedy): keep the k best logits, softmax over them, and
// return the top candidate with its probability (the p_min gate reads
// exactly this, per drafted token).
mtp_top_pick :: proc(logits: []f32) -> (llama_c.Llama_Token, f32) {
	// Ascending by logit — index 0 is the current worst, the replacement
	// target for anything better.
	klogit: [MTP_DRAFT_TOP_K]f32
	kid:    [MTP_DRAFT_TOP_K]llama_c.Llama_Token
	filled := 0
	for i in 0..<len(logits) {
		l := logits[i]
		if filled < MTP_DRAFT_TOP_K {
			klogit[filled] = l
			kid[filled]    = llama_c.Llama_Token(i)
			filled += 1
			j := filled - 1
			for j > 0 && klogit[j - 1] > klogit[j] {
				klogit[j - 1], klogit[j] = klogit[j], klogit[j - 1]
				kid[j - 1], kid[j]       = kid[j], kid[j - 1]
				j -= 1
			}
		} else if l > klogit[0] {
			klogit[0] = l
			kid[0]    = llama_c.Llama_Token(i)
			j := 0
			for j + 1 < MTP_DRAFT_TOP_K && klogit[j + 1] < klogit[j] {
				klogit[j], klogit[j + 1] = klogit[j + 1], klogit[j]
				kid[j], kid[j + 1]       = kid[j + 1], kid[j]
				j += 1
			}
		}
	}
	if filled == 0 {
		return 0, 0
	}
	// Softmax over the kept candidates; the best is the last (largest).
	max_logit := klogit[filled - 1]
	sum: f64 = 0
	for i in 0..<filled {
		sum += math.exp(f64(klogit[i] - max_logit))
	}
	return kid[filled - 1], f32(1 / sum)
}

// ---------------------------------------------------------------------
// Generation loop
// ---------------------------------------------------------------------

// Decode a single seeded token at `pos` (logits requested) — closing the
// generation loop between rounds — and run the MTP process hook. Sample the
// next token from the resulting row with the generator's sampler chain.
@(private)
mtp_plain_round :: proc(g: ^Llama_Generator, d: ^Mtp_Driver, id: llama_c.Llama_Token, pos: llama_c.Llama_Pos) -> bool {
	batch := &g.batch
	batch.n_tokens = 1
	batch.token[0]     = id
	batch.pos[0]       = pos
	batch.n_seq_id[0]  = 1
	batch.seq_id[0][0] = llama_c.Llama_Seq_Id(0)
	batch.logits[0]    = 1
	if llama_c.llama_decode(g.ctx, g.batch) != 0 {
		return false
	}
	return mtp_process(d, batch)
}

// Sample a token for the position after the row `id` currently occupies,
// after a plain round decoded that row (the MTP-mirror of the plain
// generator's sample step; the sampled token is accepted into the sampler).
@(private)
mtp_next_from_plain :: proc(g: ^Llama_Generator) -> llama_c.Llama_Token {
	s := llama_c.llama_sampler_sample(g.sampler, g.ctx, g.batch.n_tokens - 1)
	llama_c.llama_sampler_accept(g.sampler, s)
	return s
}

// Stream one emitted token with stop-sequence handling, exactly like the
// plain generator loop's emit block: appends the piece (and its token id
// attribution) to the shared buffers, then flushes complete runes up to the
// stop-sequence hold-back window. Returns (matched, stopped) where
// `stopped` means a stop sequence completed and generation must end.
@(private)
mtp_stream_token :: proc(
	g:           ^Llama_Generator,
	id:          llama_c.Llama_Token,
	on_token:    Token_Callback,
	user_data:   rawptr,
	text:        ^[dynamic]u8,
	text_ids:    ^[dynamic]llama_c.Llama_Token,
	piece_buf:   ^[dynamic]u8,
	stop_seqs:   []string,
	streamed:    ^int,
	trunc:       ^int,
	cancelled:   ^bool,
) -> bool {
	piece := token_piece(g.vocab, id, piece_buf)
	if len(piece) > 0 {
		for _ in 0..<len(piece) do append(text_ids, id)
		for b in piece do append(text, b)

		stop_start, hit := stop_suffix_pos(text[:], stop_seqs)
		if hit {
			trunc^ = stop_start
			return true
		}

		limit := len(text) - max(0, max_stop_len(stop_seqs) - 1)
		if !flush_text(text[:], text_ids[:], streamed, limit, false, on_token, user_data) {
			cancelled^ = true
		}
	}
	return false
}

// The MTP generation loop: same request prep and prefill as the plain loop
// (with the draft catch-up per chunk), then multi-token rounds — draft up
// to n_max tokens from the MTP head, verify them in one target decode,
// accept the leading matches. See the file comment for the full loop.
mtp_generate :: proc(
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
	d := g.mtp
	if d == nil {
		// Defensive: a generator without a live driver falls back to plain
		// generation (generator_generate only delegates when its draft
		// context exists, so this should not happen).
		return generator_generate(g, prompt, turns, max_tokens, stop_seqs, cancel_flag, on_token, on_progress, user_data)
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

	// Reset both contexts' KV for this request and clear the cross-decode
	// hidden-row state (the hidden row pending before the first prompt
	// token has no producer — the MTP head never runs at position 0).
	if llama_c.llama_get_memory(g.ctx) != nil {
		llama_c.llama_memory_seq_rm(
			llama_c.llama_get_memory(g.ctx), llama_c.Llama_Seq_Id(0), 0, -1,
		)
	}
	if llama_c.llama_get_memory(d.ctx_dft) != nil {
		llama_c.llama_memory_seq_rm(
			llama_c.llama_get_memory(d.ctx_dft), llama_c.Llama_Seq_Id(0), 0, -1,
		)
	}
	for &h in d.pending_h {
		h = 0
	}
	d.verify_h_rows = 0

	// Ingest the prompt, catching the draft context up per chunk; the last
	// sampled token seeds the generation loop.
	id: llama_c.Llama_Token = llama_c.LLAMA_TOKEN_NULL
	if n_prompt > 0 {
		id = mtp_ingest_prompt(g, d, prompt_tokens, batch_capacity(g.cfg), cancel_flag, on_progress, user_data, &cancelled)
	}
	cancelled = cancelled || id == llama_c.LLAMA_TOKEN_NULL

	// Loop invariants between rounds, matching the plain generator:
	//   - `id` is the most recently sampled token, it sits at `pos`, and it
	//     has been streamed (EOG pieces are empty) but not decoded yet;
	//   - d.pending_h is the hidden row of the token at pos-1.
	pos: llama_c.Llama_Pos = llama_c.Llama_Pos(n_prompt)
	turn := 0
	gen_total: i32 = 0
	n_ctx := int(llama_c.llama_n_ctx(g.ctx))

	for !cancelled && !stopped && turn < int(n_turns) {
		if llama_cancelled(cancel_flag) {
			cancelled = true
			break
		}

		if llama_c.llama_vocab_is_eog(g.vocab, id) {
			if turn + 1 >= int(n_turns) do break
			// Decode the EOG token to close this turn and sample the first
			// token of the next one (plain round — no draft across a turn
			// boundary).
			if !mtp_plain_round(g, d, id, pos) {
				break
			}
			pos += 1
			turn += 1
			id = mtp_next_from_plain(g)
			continue
		}

		// Emit the sampled token.
		if mtp_stream_token(g, id, on_token, user_data, &text, &text_ids, &piece_buf, stop_seqs, &streamed, &trunc, &cancelled) {
			stopped = true
			break
		}
		if cancelled {
			break
		}

		gen_total += 1
		if max_tokens > 0 && gen_total >= max_tokens do break

		// Draft cap: never more than the knob allows, the tokens the
		// generation still has left, or the room left in the context and
		// batch.
		n_cap: i32 = d.n_max
		if max_tokens > 0 {
			if room := max_tokens - gen_total - 1; n_cap > room do n_cap = room
		}
		if room := i32(n_ctx) - i32(pos) - 1; n_cap > room do n_cap = room
		if n_cap > batch_capacity(g.cfg) - 1 do n_cap = batch_capacity(g.cfg) - 1
		if n_cap <= 0 {
			// No room to draft: plain single-token round.
			if !mtp_plain_round(g, d, id, pos) {
				break
			}
			pos += 1
			id = mtp_next_from_plain(g)
			continue
		}

		drafts := mtp_draft(d, id, pos, n_cap)
		defer delete(drafts)
		if len(drafts) == 0 {
			// Nothing worth verifying: plain single-token round.
			if !mtp_plain_round(g, d, id, pos) {
				break
			}
			pos += 1
			id = mtp_next_from_plain(g)
			continue
		}

		// Verify batch: the sampled token plus the drafts, logits on every
		// row (row r's logits are the distribution for position pos+r+1).
		k := len(drafts)
		batch := &g.batch
		batch.n_tokens = i32(k + 1)
		for j in 0..<k + 1 {
			batch.token[j]    = id if j == 0 else drafts[j - 1]
			batch.pos[j]      = pos + llama_c.Llama_Pos(j)
			batch.n_seq_id[j] = 1
			batch.seq_id[j][0] = llama_c.Llama_Seq_Id(0)
			batch.logits[j]   = 1
		}
		if llama_c.llama_decode(g.ctx, g.batch) != 0 {
			break
		}
		if !mtp_process(d, batch) {
			break
		}

		// Accept while the target's own samples match the drafts; a
		// turn-ending EOG or the first mismatch's target sample is the next
		// token. Rows 0..k are all logits-flagged, so the extra sample at
		// row k (the token for the position after the last draft) is always
		// available.
		a := 0
		next_tok: llama_c.Llama_Token
		r: int
		for r = 0; r <= k; r += 1 {
			s := llama_c.llama_sampler_sample(g.sampler, g.ctx, i32(r))
			llama_c.llama_sampler_accept(g.sampler, s)
			if r == k || llama_c.llama_vocab_is_eog(g.vocab, s) || s != drafts[r] {
				next_tok = s
				break
			}
			a = r + 1
		}
		// Speculative decoding diagnostics.
		d.n_verif_rounds += 1
		d.n_draft_tokens += i64(k)
		d.n_accepted_tokens += i64(a)

		mtp_accept(d, i32(a))

		// Stream the accepted drafts — they equal the target's own samples
		// at their positions, in stream order.
		for i in 0..<a {
			if max_tokens > 0 && gen_total >= max_tokens do break
			if mtp_stream_token(g, drafts[i], on_token, user_data, &text, &text_ids, &piece_buf, stop_seqs, &streamed, &trunc, &cancelled) {
				stopped = true
				break
			}
			if cancelled {
				break
			}
			gen_total += 1
		}
		if cancelled || stopped {
			break
		}
		if max_tokens > 0 && gen_total >= max_tokens {
			break
		}

		// Drop the rejected positions from the target KV (a == k: the range
		// starts past the batch — a no-op). The draft context carries no
		// rejected rows: its draft-region positions are rewritten in full by
		// the next round's catch-up and draft decodes.
		llama_c.llama_memory_seq_rm(
			llama_c.llama_get_memory(g.ctx), llama_c.Llama_Seq_Id(0),
			pos + llama_c.Llama_Pos(a + 1), -1,
		)
		pos += llama_c.Llama_Pos(a + 1)
		id = next_tok
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

// Ingest the prompt in chunks of at most `batch_cap` tokens, running the MTP
// process hook (target hidden-row stash + draft catch-up) after each chunk;
// only the very last token requests logits. `cancel_flag` is checked
// between chunks and `on_progress` reports (and can cancel) ingestion.
// Returns the token sampled for the final prompt position, or
// LLAMA_TOKEN_NULL on cancellation (`cancelled^` set) or decode error.
//
// Mirrors ingest_prompt in llama.odin, plus the MTP hook per chunk.
@(private)
mtp_ingest_prompt :: proc(
	g:           ^Llama_Generator,
	d:           ^Mtp_Driver,
	tokens:      []llama_c.Llama_Token,
	batch_cap:   i32,
	cancel_flag: ^bool,
	on_progress: Progress_Callback,
	user_data:   rawptr,
	cancelled:   ^bool,
) -> llama_c.Llama_Token {
	pos: llama_c.Llama_Pos = 0
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
			b.pos[j]       = pos
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
		if !mtp_process(d, b) {
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