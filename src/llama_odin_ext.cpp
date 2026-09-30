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
