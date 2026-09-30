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
