/*
 * glm_moe_infer.m - single-process GLM MoE runtime bootstrap.
 *
 * This is the Flash-MoE-style LargerLM runtime bring-up path. It validates
 * prepared GLM layouts, initializes Metal, opens routed expert layer files,
 * allocates reusable aligned expert buffers, and now carries real-weight probe
 * paths up through full single-token decode, final logits, and argmax token
 * output while keeping the bounded allocation envelope explicit.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAX_EXPERT_BUFFERS 64

typedef struct {
    const char *prepared_dir;
    const char *resident_layout;
    const char *expert_layout;
    int open_experts;
    int mmap_resident;
    int wrap_resident_metal;
    int json;
    int expert_buffer_count;
    int probe_expert_read;
    int probe_all_layers;
    int probe_layer;
    const char *probe_experts_csv;
    const char *probe_weights_csv;
    int probe_layer_moe;
    int probe_router;
    int probe_router_moe;
    int probe_mlp_block;
    int probe_dense_mlp_block;
    int probe_resident_linear;
    int probe_attn_projections;
    int probe_rope_split;
    int probe_mla_attention;
    int probe_attn_output;
    int probe_context1_o_proj_cache_output;
    int build_context1_o_proj_cache_layer;
    int probe_decoder_layer;
    int probe_dense_decoder_layer;
    int probe_decode_layers;
    int probe_final_logits;
    int include_shared_expert;
    const char *decode_layers_csv;
    const char *output_topk_json;
    const char *output_token_json;
    const char *output_next_input_f32;
    const char *output_generated_json;
    const char *generate_request_json;
    const char *prompt_token_ids_csv;
    int prompt_token_count;
    int generate_server_jsonl;
    int generate_steps;
    int generate_token_ids;
    int generate_first_from_input_logits;
    int final_logits_chunk_rows;
    double final_logits_max_chunk_mib;
    int mmap_final_logits;
    double max_embedding_row_mib;
    int skip_final_norm;
    int skip_debug_intermediates;
    const char *resident_tensor_name;
    const char *output_dir;
    const char *q_b_f32;
    const char *k_f32;
    const char *output_q_nope_f32;
    const char *output_q_rope_f32;
    const char *output_q_f32;
    const char *output_k_f32;
    const char *q_nope_f32;
    const char *q_rope_f32;
    const char *mla_kv_b_f32;
    int num_heads;
    int kv_lora_dim;
    int qk_nope_dim;
    int rope_dim;
    int v_head_dim;
    int start_position;
    int context_length;
    int cache_position_offset;
    int batch_tokens;
    double rope_theta;
    int rope_interleave;
    double attention_scale;
    int top_k;
    const char *router_score;
    int router_score_set;
    double routed_scaling_factor;
    int routed_scaling_factor_set;
    int norm_topk_prob;
    int norm_topk_prob_set;
    int router_n_group;
    int router_n_group_set;
    int router_topk_group;
    int router_topk_group_set;
    const char *output_router_json;
    int ignore_router_bias;
    const char *input_f32;
    int input_token_id;
    const char *residual_f32;
    const char *projection_f32;
    const char *output_f32;
    const char *context1_o_proj_cache_layout;
    const char *context1_o_proj_cache_file;
    const char *cache_layout;
    const char *cache_file;
    int cache_position;
    int in_memory_decode_cache;
    int cache_mla_kv_b_f32;
    double max_cache_file_mib;
    double max_cache_read_mib;
    double max_mla_kv_b_cache_mib;
    double max_context1_o_proj_build_gfma;
    int expect_output0_set;
    double expect_output0;
    double rms_norm_eps;
    int probe_repeat;
    double max_live_working_set_mib;
    double min_free_unified_memory_gib;
} LoaderOptions;

typedef struct {
    int fd;
    char *path;
    uint64_t expected_bytes;
    uint64_t actual_bytes;
    uint64_t expert_slot_bytes;
    uint64_t num_experts;
    int layer;
} ExpertFile;

@interface GlmMoeRuntimeContext : NSObject
@property(nonatomic, strong) NSString *residentLayoutPath;
@property(nonatomic, strong) NSString *expertLayoutPath;
@property(nonatomic, strong) NSDictionary *residentLayout;
@property(nonatomic, strong) NSDictionary *expertLayout;
@property(nonatomic, strong) NSString *residentDir;
@property(nonatomic, strong) NSString *expertDir;
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSString *residentBinPath;
@property(nonatomic, strong) NSArray *layers;
@property(nonatomic, strong) NSMutableArray *expertBuffers;
@property(nonatomic, assign) int residentFd;
@property(nonatomic, strong) NSString *decodeCachePath;
@property(nonatomic, assign) int decodeCacheFd;
@property(nonatomic, assign) uint64_t decodeCacheFdOpenCount;
@property(nonatomic, strong) NSString *decodeCacheMemoryPath;
@property(nonatomic, strong) NSMutableData *decodeCacheMemory;
@property(nonatomic, assign) uint64_t decodeCacheMemoryLoadCount;
@property(nonatomic, assign) uint64_t residentLayoutBytes;
@property(nonatomic, assign) uint64_t residentFileBytes;
@property(nonatomic, assign) ExpertFile *expertFiles;
@property(nonatomic, assign) int openedExpertFiles;
@property(nonatomic, assign) uint64_t maxExpertSlotBytes;
@property(nonatomic, assign) uint64_t totalExpertBytes;
@property(nonatomic, assign) uint64_t expertBufferBytes;
@end

static int g_shared_resident_fd = -1;
static NSString *g_shared_resident_path = nil;
static int g_shared_decode_cache_fd = -1;
static NSString *g_shared_decode_cache_path = nil;
static NSMutableData *g_shared_decode_cache_memory = nil;
static void clear_shared_resident_file_if_fd(int fd);
static void clear_shared_decode_cache_file_if_fd(int fd);

@implementation GlmMoeRuntimeContext
- (void)dealloc {
    if (_decodeCacheFd >= 0) {
        clear_shared_decode_cache_file_if_fd(_decodeCacheFd);
        close(_decodeCacheFd);
        _decodeCacheFd = -1;
    }
    if (_decodeCacheMemory && g_shared_decode_cache_memory == _decodeCacheMemory) {
        g_shared_decode_cache_memory = nil;
        g_shared_decode_cache_path = nil;
    }
    if (_residentFd >= 0) {
        clear_shared_resident_file_if_fd(_residentFd);
        close(_residentFd);
        _residentFd = -1;
    }
    if (_expertFiles) {
        for (NSUInteger i = 0; i < [_layers count]; i++) {
            if (_expertFiles[i].fd >= 0) {
                close(_expertFiles[i].fd);
            }
            free(_expertFiles[i].path);
        }
        free(_expertFiles);
        _expertFiles = NULL;
    }
}
@end

typedef struct {
    int *values;
    int count;
} IntList;

typedef struct {
    float *values;
    int count;
} FloatList;

typedef struct {
    int ok;
    uint64_t available_bytes;
    uint64_t total_bytes;
    uint64_t page_size;
} SystemMemorySnapshot;

typedef struct {
    int layer;
    int expert;
    uint64_t bytes;
    uint64_t sample_checksum;
} ProbeReadResult;

typedef struct {
    uint64_t offset;
    uint64_t size;
    uint32_t dim0;
    uint32_t dim1;
    char dtype[16];
} Mxfp4ComponentInfo;

typedef struct {
    uint64_t offset;
    uint64_t size;
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    char dtype[16];
    char name[512];
} ResidentTensor3DComponentInfo;

typedef struct {
    ResidentTensor3DComponentInfo weight;
    ResidentTensor3DComponentInfo scales;
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    uint32_t group_size;
    uint64_t total_bytes;
    char name[512];
} ResidentMxfp4Tensor3DInfo;

typedef struct {
    ResidentMxfp4Tensor3DInfo embed_q;
    ResidentMxfp4Tensor3DInfo unembed_out;
    uint32_t kv_lora_dim;
    uint32_t expected_kv_b_out;
    uint64_t storage_bytes;
    uint64_t source_f32_bytes;
    uint64_t kv_b_f32_bytes;
} MlaAttentionValueSourceInfo;

typedef struct {
    Mxfp4ComponentInfo gate_w;
    Mxfp4ComponentInfo gate_s;
    Mxfp4ComponentInfo up_w;
    Mxfp4ComponentInfo up_s;
    Mxfp4ComponentInfo down_w;
    Mxfp4ComponentInfo down_s;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t group_size;
} Mxfp4ExpertInfo;

typedef struct {
    Mxfp4ExpertInfo local;
    Mxfp4ComponentInfo src_gate_w;
    Mxfp4ComponentInfo src_gate_s;
    Mxfp4ComponentInfo src_up_w;
    Mxfp4ComponentInfo src_up_s;
    Mxfp4ComponentInfo src_down_w;
    Mxfp4ComponentInfo src_down_s;
    uint64_t total_bytes;
} SharedMxfp4Info;

typedef struct {
    uint64_t offset;
    uint64_t size;
    uint32_t num_experts;
    uint32_t hidden_dim;
    char dtype[16];
    char name[512];
} RouterWeightInfo;

typedef struct {
    int present;
    uint64_t offset;
    uint64_t size;
    uint32_t dim;
    char dtype[16];
    char name[512];
} RouterBiasInfo;

typedef struct {
    char score[16];
    float routed_scaling_factor;
    int norm_topk_prob;
    uint32_t n_group;
    uint32_t topk_group;
} RouterTopKOptions;

typedef struct {
    int present;
    uint64_t offset;
    uint64_t size;
    uint32_t dim;
    char dtype[16];
    char name[512];
} ResidentVectorInfo;

typedef struct {
    int ok;
    uint64_t router_bytes_read;
    uint64_t correction_bias_bytes_read;
    double elapsed_seconds;
    double router_read_seconds;
    double kernel_seconds;
    uint32_t top_k;
    int experts[64];
    float weights[64];
    float *logits;
    uint32_t num_experts;
    char router_score[16];
    float routed_scaling_factor;
    int norm_topk_prob;
    uint32_t n_group;
    uint32_t topk_group;
    int used_correction_bias;
    int gpu_topk;
    uint32_t command_buffer_count;
    int fused_with_rmsnorm;
} RouterProbeStats;

typedef struct {
    int ok;
    uint64_t weight_bytes_read;
    double weight_read_seconds;
    double elapsed_seconds;
    float output0;
    uint32_t command_buffer_count;
    int fused_with_router;
    int input_buffer_direct;
} RmsNormProbeStats;

typedef struct {
    Mxfp4ComponentInfo weight;
    Mxfp4ComponentInfo scales;
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    uint64_t total_bytes;
    char name[512];
} ResidentMxfp4MatrixInfo;

typedef struct {
    int ok;
    uint64_t bytes_read;
    double elapsed_seconds;
    double read_seconds;
    double kernel_seconds;
    double output_write_seconds;
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    float output0;
    float output0_abs_error;
    int output0_check_ok;
    int resident_mmap_backed;
} ResidentLinearProbeStats;

typedef struct {
    int ok;
    uint64_t bytes_read;
    uint64_t input_bytes;
    uint64_t residual_bytes;
    uint64_t projection_bytes;
    uint64_t output_bytes;
    uint64_t scratch_bytes;
    double read_seconds;
    double projection_kernel_seconds;
    double residual_add_seconds;
    double projection_write_seconds;
    double output_write_seconds;
    double elapsed_seconds;
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    int fused_matvec_add;
    int context1_o_proj_cache;
    int fused_with_post_attn_norm_router;
    int fused_with_rope_mla;
    int resident_mmap_backed;
    uint32_t command_buffer_count;
    float output0;
} AttnOutputProbeStats;

typedef struct {
    uint64_t offset;
    uint64_t size;
    uint64_t total_bytes;
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t dtype_bytes;
    char dtype[16];
    char name[512];
} Context1OProjCacheMatrixInfo;

typedef struct {
    int ok;
    uint64_t source_bytes_read;
    uint64_t cache_bytes_written;
    uint64_t fma_count;
    double read_seconds;
    double kernel_seconds;
    double write_seconds;
    double elapsed_seconds;
    uint64_t estimated_live_working_set_bytes;
    double max_live_working_set_mib;
    int live_working_set_ok;
    uint32_t hidden_dim;
    uint32_t attention_value_dim;
    uint32_t kv_lora_dim;
    float output0;
} Context1OProjCacheBuildStats;

typedef struct {
    ResidentMxfp4MatrixInfo gate;
    ResidentMxfp4MatrixInfo up;
    ResidentMxfp4MatrixInfo down;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t group_size;
} DenseMlpMxfp4Info;

typedef struct {
    int ok;
    uint64_t bytes_read;
    uint64_t scratch_bytes;
    double elapsed_seconds;
    double rmsnorm_elapsed_seconds;
    double gate_read_seconds;
    double gate_kernel_seconds;
    double up_read_seconds;
    double up_kernel_seconds;
    double swiglu_kernel_seconds;
    double down_read_seconds;
    double down_kernel_seconds;
    double residual_add_seconds;
    double fused_kernel_seconds;
    double output_write_seconds;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t group_size;
    int fused_pipeline;
    uint32_t command_buffer_count;
    uint32_t synchronous_wait_count;
    int async_submitted;
    float output0;
    float output0_abs_error;
    int output0_check_ok;
} DenseMlpProbeStats;

typedef struct {
    int ok;
    uint64_t bytes_read;
    uint64_t scratch_bytes;
    double elapsed_seconds;
    uint32_t hidden_dim;
    uint32_t q_lora_dim;
    uint32_t q_out_dim;
    uint32_t kv_lora_dim;
    uint32_t kv_a_out_dim;
    uint32_t kv_rope_dim;
    uint32_t kv_out_dim;
    int has_kv_b;
    float q_b_output0;
    float kv_a_output0;
    float kv_a_norm_output0;
    float kv_b_output0;
    int cache_append;
    uint64_t cache_position;
    uint64_t cache_write_bytes;
    double cache_write_seconds;
    double fused_pre_cache_seconds;
    int fused_pre_cache;
    uint32_t command_buffer_count;
    uint32_t synchronous_wait_count;
    int async_submitted;
} AttnProjectionProbeStats;

typedef struct {
    int ok;
    uint64_t scratch_bytes;
    double elapsed_seconds;
    uint32_t num_heads;
    uint32_t qk_nope_dim;
    uint32_t rope_dim;
    uint32_t start_position;
    uint32_t batch_tokens;
    float theta;
    int interleave;
    uint64_t q_b_bytes;
    uint64_t k_bytes;
    uint64_t q_nope_bytes;
    uint64_t q_rope_bytes;
    uint32_t command_buffer_count;
    int fused_with_mla;
    int input_buffer_direct;
    float q_output0;
    float k_output0;
} RopeSplitProbeStats;

typedef struct {
    int ok;
    uint64_t raw_cache_bytes;
    uint64_t cache_f32_bytes;
    uint64_t value_storage_bytes;
    uint64_t value_source_f32_bytes;
    uint64_t kv_b_f32_bytes;
    uint64_t value_cache_bytes;
    uint64_t value_cache_total_bytes;
    uint64_t q_nope_bytes;
    uint64_t q_rope_bytes;
    uint64_t output_bytes;
    uint64_t scratch_bytes;
    double cache_read_seconds;
    double value_read_seconds;
    double kernel_seconds;
    double output_write_seconds;
    double elapsed_seconds;
    uint32_t context_length;
    uint32_t num_heads;
    uint32_t kv_lora_dim;
    uint32_t qk_nope_dim;
    uint32_t rope_dim;
    uint32_t v_head_dim;
    uint32_t cache_position_offset;
    float attention_scale;
    float rope_theta;
    int rope_interleave;
    uint32_t command_buffer_count;
    int fused_with_rope;
    int value_cache_enabled;
    int value_cache_hit;
    int value_cache_stored;
    float output0;
} MlaAttentionProbeStats;

typedef struct {
    uint64_t layer;
    uint64_t offset;
    uint64_t total_bytes;
    uint64_t max_context_tokens;
    uint32_t width;
    uint32_t dtype_bytes;
    char dtype[16];
    char kind[32];
} DecodeCacheSegmentInfo;

static void usage(const char *argv0) {
    printf("Usage: %s --prepared DIR [options]\n", argv0);
    printf("       %s --resident-layout PATH --expert-layout PATH [options]\n", argv0);
    printf("Options:\n");
    printf("  --prepared DIR                  prepared package root\n");
    printf("  --resident-layout PATH          resident/layout.json\n");
    printf("  --expert-layout PATH            experts/layout.json\n");
    printf("  --no-open-experts               validate metadata without opening layer files\n");
    printf("  --mmap-resident                 mmap resident.bin read-only\n");
    printf("  --wrap-resident-metal           wrap mmap'd resident.bin as a Metal buffer\n");
    printf("  --expert-buffer-count N         reusable expert buffers to allocate (default: 8, max: 64)\n");
    printf("  --probe-expert-read             pread selected expert slots into Metal buffers\n");
    printf("  --probe-all-layers              read --probe-experts from every expert layer\n");
    printf("  --probe-layer N                 layer id for --probe-expert-read\n");
    printf("  --probe-experts CSV             comma-separated expert ids for --probe-expert-read\n");
    printf("  --probe-layer-moe               run one-layer MXFP4 routed MoE probe\n");
    printf("  --probe-router                  run one-layer BF16 router/top-k probe\n");
    printf("  --probe-router-moe              route then run one-layer MXFP4 MoE probe\n");
    printf("  --probe-mlp-block               RMSNorm, route, routed MoE, residual add\n");
    printf("  --probe-dense-mlp-block         RMSNorm, dense resident MXFP4 MLP, residual add\n");
    printf("  --probe-resident-linear         run one resident MXFP4 matrix-vector probe\n");
    printf("  --probe-attn-projections        run single-token attention projection probe\n");
    printf("  --probe-rope-split              split q_b and rotate q/k RoPE batches\n");
    printf("  --probe-mla-attention           run single-token MLA attention probe\n");
    printf("  --probe-attn-output             run single-token o_proj + residual probe\n");
    printf("  --probe-context1-o-proj-cache-output run latent collapsed o_proj*B_v cache + residual probe\n");
    printf("  --build-context1-o-proj-cache-layer build one collapsed o_proj*B_v cache layer from resident MXFP4\n");
    printf("  --probe-decoder-layer           compose one single-token decoder layer probe\n");
    printf("  --probe-dense-decoder-layer     compose one dense-prefix decoder layer probe\n");
    printf("  --probe-decode-layers           compose a single-token decoder layer list probe\n");
    printf("  --decode-layers CSV             comma-separated layers for --probe-decode-layers\n");
    printf("  --probe-final-logits            stream final RMSNorm + MXFP4 lm_head top-k\n");
    printf("  --output-topk-json PATH         optional final logits top-k JSON output\n");
    printf("  --output-token-json PATH        optional argmax token JSON output\n");
    printf("  --output-next-input-f32 PATH    optional argmax embedding F32 output\n");
    printf("  --output-generated-json PATH    optional greedy multi-step JSON output\n");
    printf("  --generate-request-json PATH    load formal generation request JSON\n");
    printf("  --generate-server-jsonl         start formal generation JSONL service scaffold\n");
    printf("  --generate-token-ids            formal greedy token generation entry\n");
    printf("  --generate-steps N              run N greedy decode steps in one process\n");
    printf("  --generate-first-from-input-logits  generate step 0 from input hidden logits\n");
    printf("  --chunk-rows N                  optional lm_head rows per final-logits chunk\n");
    printf("  --max-chunk-mib N               final-logits lm_head chunk byte cap (default: 64)\n");
    printf("  --mmap-final-logits             mmap lm_head ranges as Metal buffers for final logits\n");
    printf("  --max-embedding-row-mib N       generated-token embedding row cap (default: 64)\n");
    printf("  --skip-final-norm               skip final RMSNorm for final-logits probe\n");
    printf("  --skip-debug-intermediates      avoid decoder-layer debug intermediate files\n");
    printf("  --include-shared-expert         include resident MXFP4 shared expert in MLP probe\n");
    printf("  --resident-tensor-name NAME     full resident tensor name for resident-linear probe\n");
    printf("  --output-dir DIR                output/work directory for attention probes\n");
    printf("  --input-token-id N              decode initial hidden from embedding token id\n");
    printf("  --q-b-f32 PATH                  q_b input for --probe-rope-split\n");
    printf("  --k-f32 PATH                    k_rope input for --probe-rope-split\n");
    printf("  --output-q-nope-f32 PATH        q_nope output for --probe-rope-split\n");
    printf("  --output-q-rope-f32 PATH        unrotated q_rope output for --probe-rope-split\n");
    printf("  --output-q-f32 PATH             rotated q_rope output for --probe-rope-split\n");
    printf("  --output-k-f32 PATH             rotated k_rope output for --probe-rope-split\n");
    printf("  --q-nope-f32 PATH               q_nope input for --probe-mla-attention\n");
    printf("  --q-rope-f32 PATH               rotated q_rope input for --probe-mla-attention\n");
    printf("  --mla-kv-b-f32 PATH             direct F32 KV-B matrix for --probe-mla-attention smoke tests\n");
    printf("  --num-heads N                   attention head count\n");
    printf("  --kv-lora-dim N                 optional KV lora dim for --probe-mla-attention\n");
    printf("  --qk-nope-dim N                 qk nope dim\n");
    printf("  --rope-dim N                    RoPE dim\n");
    printf("  --v-head-dim N                  value head dim for --probe-mla-attention\n");
    printf("  --start-position N              first position for --probe-rope-split\n");
    printf("  --context-length N              context length for --probe-mla-attention\n");
    printf("  --cache-position-offset N       first cache position for MLA RoPE keys\n");
    printf("  --batch-tokens N                batch tokens for --probe-rope-split\n");
    printf("  --rope-theta N                  RoPE theta for --probe-rope-split (default: 10000)\n");
    printf("  --rope-interleave               use interleaved RoPE pairs\n");
    printf("  --attention-scale N             attention scale for --probe-mla-attention (default: auto)\n");
    printf("  --top-k N                       top-k for --probe-router (default: 8)\n");
    printf("  --router-score MODE             sigmoid, softmax, or raw for --probe-router\n");
    printf("  --routed-scaling-factor N       override router scaling factor\n");
    printf("  --norm-topk-prob                normalize selected router weights\n");
    printf("  --no-norm-topk-prob             do not normalize selected router weights\n");
    printf("  --router-n-group N              override router group count\n");
    printf("  --router-topk-group N           override selected router group count\n");
    printf("  --output-router-json PATH       optional router JSON output\n");
    printf("  --ignore-router-bias            ignore correction-bias tensor if present\n");
    printf("  --probe-weights CSV             comma-separated route weights for --probe-layer-moe\n");
    printf("  --input-f32 PATH                input vector for --probe-layer-moe\n");
    printf("  --residual-f32 PATH             residual hidden vector for --probe-attn-output\n");
    printf("  --projection-f32 PATH           optional o_proj-only output for --probe-attn-output\n");
    printf("  --output-f32 PATH               output vector for --probe-layer-moe\n");
    printf("  --context1-o-proj-cache-layout PATH collapsed context=1 o_proj*B_v cache layout\n");
    printf("  --context1-o-proj-cache-file PATH optional collapsed cache backing file override\n");
    printf("  --cache-layout PATH             decode cache layout for attention KV-A append\n");
    printf("  --cache-file PATH               decode cache file for attention KV-A append\n");
    printf("  --in-memory-decode-cache        load decode cache into runtime memory\n");
    printf("  --cache-mla-kv-b-f32            cache absorbed MLA KV-B F32 views in process\n");
    printf("  --position N                    decode cache position for attention KV-A append\n");
    printf("  --max-cache-file-mib N          maximum allowed decode cache file size\n");
    printf("  --max-cache-read-mib N          maximum cache bytes read by MLA attention\n");
    printf("  --max-mla-kv-b-cache-mib N      maximum in-process MLA KV-B F32 cache bytes\n");
    printf("  --max-context1-o-proj-build-gfma N safety cap for one-layer context1 cache builder (default: 1)\n");
    printf("  --expect-output0 N              optional output[0] check for --probe-layer-moe\n");
    printf("  --rms-norm-eps N                RMSNorm epsilon for --probe-mlp-block (default: 1e-5)\n");
    printf("  --probe-repeat N                repeat direct expert reads (default: 1)\n");
    printf("  --max-live-working-set-mib N    fail if estimated live bytes exceed this cap\n");
    printf("  --min-free-unified-memory-gib N recorded safety gate for future runtime\n");
    printf("  --json                          emit JSON\n");
    printf("  --help                          show this message\n");
}

static int parse_positive_int(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || value <= 0 || value > INT32_MAX) {
        fprintf(stderr, "ERROR: %s must be a positive integer\n", name);
        exit(2);
    }
    return (int)value;
}

static int parse_nonnegative_int(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || value < 0 || value > INT_MAX) {
        fprintf(stderr, "ERROR: %s must be a non-negative integer\n", name);
        exit(2);
    }
    return (int)value;
}

static double parse_nonnegative_double(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno || end == text || *end != '\0' || !isfinite(value) || value < 0.0) {
        fprintf(stderr, "ERROR: %s must be a non-negative finite number\n", name);
        exit(2);
    }
    return value;
}

static double parse_positive_double(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno || end == text || *end != '\0' || !isfinite(value) || value <= 0.0) {
        fprintf(stderr, "ERROR: %s must be a positive finite number\n", name);
        exit(2);
    }
    return value;
}

static double parse_finite_double(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno || end == text || *end != '\0' || !isfinite(value)) {
        fprintf(stderr, "ERROR: %s must be a finite number\n", name);
        exit(2);
    }
    return value;
}

static char *dup_json_string_field(NSDictionary *dict, NSString *key, const char *label) {
    id value = dict[key];
    if (value == nil || value == [NSNull null]) {
        return NULL;
    }
    if (![value isKindOfClass:[NSString class]]) {
        fprintf(stderr, "ERROR: generate request field %s must be a string\n", label);
        exit(2);
    }
    const char *utf8 = [(NSString *)value UTF8String];
    char *copy = strdup(utf8 ? utf8 : "");
    if (!copy) {
        fprintf(stderr, "ERROR: out of memory copying generate request field %s\n", label);
        exit(2);
    }
    return copy;
}

static char *dup_json_int_array_csv_field(NSDictionary *dict,
                                          NSString *key,
                                          const char *label,
                                          int *outCount) {
    if (outCount) {
        *outCount = 0;
    }
    id value = dict[key];
    if (value == nil || value == [NSNull null]) {
        return NULL;
    }
    if (![value isKindOfClass:[NSArray class]]) {
        fprintf(stderr, "ERROR: generate request field %s must be an integer array\n",
                label);
        exit(2);
    }
    NSArray *items = (NSArray *)value;
    if ([items count] == 0 || [items count] > INT32_MAX) {
        fprintf(stderr, "ERROR: generate request field %s must be a non-empty array\n",
                label);
        exit(2);
    }
    NSMutableString *csv = [NSMutableString string];
    for (NSUInteger i = 0; i < [items count]; i++) {
        id item = items[i];
        if (![item isKindOfClass:[NSNumber class]]) {
            fprintf(stderr, "ERROR: generate request field %s must contain integers\n",
                    label);
            exit(2);
        }
        long long raw = [(NSNumber *)item longLongValue];
        if (raw < 0 || raw > INT32_MAX) {
            fprintf(stderr,
                    "ERROR: generate request field %s contains an invalid token id\n",
                    label);
            exit(2);
        }
        if (i > 0) {
            [csv appendString:@","];
        }
        [csv appendFormat:@"%lld", raw];
    }
    char *copy = strdup([csv UTF8String]);
    if (!copy) {
        fprintf(stderr, "ERROR: out of memory copying generate request field %s\n",
                label);
        exit(2);
    }
    if (outCount) {
        *outCount = (int)[items count];
    }
    return copy;
}

static int json_int_field(NSDictionary *dict,
                          NSString *key,
                          const char *label,
                          int positive,
                          int *out) {
    id value = dict[key];
    if (value == nil || value == [NSNull null]) {
        return 0;
    }
    if (![value isKindOfClass:[NSNumber class]]) {
        fprintf(stderr, "ERROR: generate request field %s must be an integer\n", label);
        exit(2);
    }
    long long raw = [(NSNumber *)value longLongValue];
    if (raw < 0 || (positive && raw <= 0) || raw > INT32_MAX) {
        fprintf(stderr,
                "ERROR: generate request field %s must be %s integer\n",
                label,
                positive ? "a positive" : "a non-negative");
        exit(2);
    }
    *out = (int)raw;
    return 1;
}

static int json_double_field(NSDictionary *dict, NSString *key, const char *label, double *out) {
    id value = dict[key];
    if (value == nil || value == [NSNull null]) {
        return 0;
    }
    if (![value isKindOfClass:[NSNumber class]]) {
        fprintf(stderr, "ERROR: generate request field %s must be a number\n", label);
        exit(2);
    }
    double raw = [(NSNumber *)value doubleValue];
    if (!isfinite(raw) || raw < 0.0) {
        fprintf(stderr,
                "ERROR: generate request field %s must be a non-negative finite number\n",
                label);
        exit(2);
    }
    *out = raw;
    return 1;
}

static int json_bool_field(NSDictionary *dict, NSString *key, const char *label, int *out) {
    id value = dict[key];
    if (value == nil || value == [NSNull null]) {
        return 0;
    }
    if (![value isKindOfClass:[NSNumber class]]) {
        fprintf(stderr, "ERROR: generate request field %s must be a boolean\n", label);
        exit(2);
    }
    *out = [(NSNumber *)value boolValue] ? 1 : 0;
    return 1;
}

static void apply_generate_request_dictionary(LoaderOptions *options,
                                              NSDictionary *request) {
    char *text = NULL;
    if ((text = dup_json_string_field(request, @"decode_layers", "decode_layers"))) {
        options->decode_layers_csv = text;
    }
    if ((text = dup_json_string_field(request, @"input_f32", "input_f32"))) {
        options->input_f32 = text;
    }
    if ((text = dup_json_string_field(request, @"output_f32", "output_f32"))) {
        options->output_f32 = text;
    }
    if ((text = dup_json_string_field(request, @"output_dir", "output_dir"))) {
        options->output_dir = text;
    }
    if ((text = dup_json_string_field(request, @"output_generated_json", "output_generated_json"))) {
        options->output_generated_json = text;
    }
    if ((text = dup_json_string_field(request, @"output_next_input_f32", "output_next_input_f32"))) {
        options->output_next_input_f32 = text;
    }
    if ((text = dup_json_string_field(request, @"cache_layout", "cache_layout"))) {
        options->cache_layout = text;
    }
    if ((text = dup_json_string_field(request, @"cache_file", "cache_file"))) {
        options->cache_file = text;
    }
    if ((text = dup_json_string_field(request,
                                      @"context1_o_proj_cache_layout",
                                      "context1_o_proj_cache_layout"))) {
        options->context1_o_proj_cache_layout = text;
    }
    if ((text = dup_json_string_field(request,
                                      @"context1_o_proj_cache_file",
                                      "context1_o_proj_cache_file"))) {
        options->context1_o_proj_cache_file = text;
    }
    if ((text = dup_json_int_array_csv_field(request,
                                             @"prompt_token_ids",
                                             "prompt_token_ids",
                                             &options->prompt_token_count))) {
        options->prompt_token_ids_csv = text;
    }
    json_int_field(request, @"input_token_id", "input_token_id", 0, &options->input_token_id);
    json_int_field(request, @"generate_steps", "generate_steps", 1, &options->generate_steps);
    if (!json_int_field(request, @"position", "position", 0, &options->cache_position)) {
        json_int_field(request, @"cache_position", "cache_position", 0, &options->cache_position);
    }
    json_int_field(request, @"context_length", "context_length", 1, &options->context_length);
    json_int_field(request, @"num_heads", "num_heads", 1, &options->num_heads);
    json_int_field(request, @"kv_lora_dim", "kv_lora_dim", 1, &options->kv_lora_dim);
    json_int_field(request, @"qk_nope_dim", "qk_nope_dim", 1, &options->qk_nope_dim);
    json_int_field(request, @"rope_dim", "rope_dim", 1, &options->rope_dim);
    json_int_field(request, @"v_head_dim", "v_head_dim", 1, &options->v_head_dim);
    json_int_field(request, @"cache_position_offset", "cache_position_offset", 0,
                   &options->cache_position_offset);
    json_int_field(request, @"top_k", "top_k", 1, &options->top_k);
    json_int_field(request, @"expert_buffer_count", "expert_buffer_count", 1,
                   &options->expert_buffer_count);
    json_bool_field(request, @"generate_first_from_input_logits",
                    "generate_first_from_input_logits",
                    &options->generate_first_from_input_logits);
    json_bool_field(request, @"skip_debug_intermediates", "skip_debug_intermediates",
                    &options->skip_debug_intermediates);
    json_bool_field(request, @"include_shared_expert", "include_shared_expert",
                    &options->include_shared_expert);
    json_bool_field(request, @"rope_interleave", "rope_interleave",
                    &options->rope_interleave);
    json_bool_field(request, @"in_memory_decode_cache", "in_memory_decode_cache",
                    &options->in_memory_decode_cache);
    json_bool_field(request, @"cache_mla_kv_b_f32", "cache_mla_kv_b_f32",
                    &options->cache_mla_kv_b_f32);
    json_double_field(request, @"rms_norm_eps", "rms_norm_eps", &options->rms_norm_eps);
    json_double_field(request, @"rope_theta", "rope_theta", &options->rope_theta);
    json_double_field(request, @"max_cache_file_mib", "max_cache_file_mib",
                      &options->max_cache_file_mib);
    json_double_field(request, @"max_cache_read_mib", "max_cache_read_mib",
                      &options->max_cache_read_mib);
    json_double_field(request, @"max_mla_kv_b_cache_mib", "max_mla_kv_b_cache_mib",
                      &options->max_mla_kv_b_cache_mib);
    json_double_field(request, @"max_chunk_mib", "max_chunk_mib",
                      &options->final_logits_max_chunk_mib);
    json_bool_field(request, @"mmap_final_logits", "mmap_final_logits",
                    &options->mmap_final_logits);
    json_double_field(request, @"max_embedding_row_mib", "max_embedding_row_mib",
                      &options->max_embedding_row_mib);
    json_double_field(request, @"max_live_working_set_mib", "max_live_working_set_mib",
                      &options->max_live_working_set_mib);
    json_double_field(request, @"min_free_unified_memory_gib", "min_free_unified_memory_gib",
                      &options->min_free_unified_memory_gib);
    options->generate_token_ids = 1;
    options->probe_decode_layers = 1;
}

static void apply_generate_request_json(LoaderOptions *options, const char *path) {
    NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!data) {
        fprintf(stderr, "ERROR: failed to read --generate-request-json %s\n", path);
        exit(2);
    }
    NSError *error = nil;
    id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (![root isKindOfClass:[NSDictionary class]]) {
        fprintf(stderr,
                "ERROR: --generate-request-json must contain a JSON object: %s\n",
                error ? [[error localizedDescription] UTF8String] : "invalid JSON");
        exit(2);
    }
    apply_generate_request_dictionary(options, (NSDictionary *)root);
}

static LoaderOptions parse_options(int argc, char **argv) {
    LoaderOptions options = {
        .prepared_dir = NULL,
        .resident_layout = NULL,
        .expert_layout = NULL,
        .open_experts = 1,
        .mmap_resident = 0,
        .wrap_resident_metal = 0,
        .json = 0,
        .expert_buffer_count = 8,
        .probe_expert_read = 0,
        .probe_all_layers = 0,
        .probe_layer = -1,
        .probe_experts_csv = NULL,
        .probe_weights_csv = NULL,
        .probe_layer_moe = 0,
        .probe_router = 0,
        .probe_router_moe = 0,
        .probe_mlp_block = 0,
        .probe_dense_mlp_block = 0,
        .probe_resident_linear = 0,
        .probe_attn_projections = 0,
        .probe_rope_split = 0,
        .probe_mla_attention = 0,
        .probe_attn_output = 0,
        .probe_context1_o_proj_cache_output = 0,
        .build_context1_o_proj_cache_layer = 0,
        .probe_decoder_layer = 0,
        .probe_dense_decoder_layer = 0,
        .probe_decode_layers = 0,
        .probe_final_logits = 0,
        .include_shared_expert = 0,
        .decode_layers_csv = NULL,
        .output_topk_json = NULL,
        .output_token_json = NULL,
        .output_next_input_f32 = NULL,
        .output_generated_json = NULL,
        .generate_request_json = NULL,
        .prompt_token_ids_csv = NULL,
        .prompt_token_count = 0,
        .generate_server_jsonl = 0,
        .generate_steps = 0,
        .generate_token_ids = 0,
        .generate_first_from_input_logits = 0,
        .final_logits_chunk_rows = 0,
        .final_logits_max_chunk_mib = 64.0,
        .mmap_final_logits = 0,
        .max_embedding_row_mib = 64.0,
        .skip_final_norm = 0,
        .skip_debug_intermediates = 0,
        .resident_tensor_name = NULL,
        .output_dir = NULL,
        .q_b_f32 = NULL,
        .k_f32 = NULL,
        .output_q_nope_f32 = NULL,
        .output_q_rope_f32 = NULL,
        .output_q_f32 = NULL,
        .output_k_f32 = NULL,
        .q_nope_f32 = NULL,
        .q_rope_f32 = NULL,
        .mla_kv_b_f32 = NULL,
        .num_heads = 0,
        .kv_lora_dim = 0,
        .qk_nope_dim = 0,
        .rope_dim = 0,
        .v_head_dim = 0,
        .start_position = -1,
        .context_length = 0,
        .cache_position_offset = 0,
        .batch_tokens = 0,
        .rope_theta = 10000.0,
        .rope_interleave = 0,
        .attention_scale = 0.0,
        .top_k = 8,
        .router_score = NULL,
        .router_score_set = 0,
        .routed_scaling_factor = 0.0,
        .routed_scaling_factor_set = 0,
        .norm_topk_prob = 0,
        .norm_topk_prob_set = 0,
        .router_n_group = 0,
        .router_n_group_set = 0,
        .router_topk_group = 0,
        .router_topk_group_set = 0,
        .output_router_json = NULL,
        .ignore_router_bias = 0,
        .input_f32 = NULL,
        .input_token_id = -1,
        .residual_f32 = NULL,
        .projection_f32 = NULL,
        .output_f32 = NULL,
        .context1_o_proj_cache_layout = NULL,
        .context1_o_proj_cache_file = NULL,
        .cache_layout = NULL,
        .cache_file = NULL,
        .cache_position = -1,
        .in_memory_decode_cache = 0,
        .cache_mla_kv_b_f32 = 0,
        .max_cache_file_mib = 0.0,
        .max_cache_read_mib = 0.0,
        .max_mla_kv_b_cache_mib = 0.0,
        .max_context1_o_proj_build_gfma = 1.0,
        .expect_output0_set = 0,
        .expect_output0 = 0.0,
        .rms_norm_eps = 1.0e-5,
        .probe_repeat = 1,
        .max_live_working_set_mib = 0.0,
        .min_free_unified_memory_gib = 0.0,
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--prepared") == 0 && i + 1 < argc) {
            options.prepared_dir = argv[++i];
        } else if (strcmp(arg, "--resident-layout") == 0 && i + 1 < argc) {
            options.resident_layout = argv[++i];
        } else if (strcmp(arg, "--expert-layout") == 0 && i + 1 < argc) {
            options.expert_layout = argv[++i];
        } else if (strcmp(arg, "--no-open-experts") == 0) {
            options.open_experts = 0;
        } else if (strcmp(arg, "--mmap-resident") == 0) {
            options.mmap_resident = 1;
        } else if (strcmp(arg, "--wrap-resident-metal") == 0) {
            options.wrap_resident_metal = 1;
            options.mmap_resident = 1;
        } else if (strcmp(arg, "--json") == 0) {
            options.json = 1;
        } else if (strcmp(arg, "--expert-buffer-count") == 0 && i + 1 < argc) {
            options.expert_buffer_count = parse_positive_int(
                argv[++i],
                "--expert-buffer-count"
            );
        } else if (strcmp(arg, "--probe-expert-read") == 0) {
            options.probe_expert_read = 1;
        } else if (strcmp(arg, "--probe-all-layers") == 0) {
            options.probe_all_layers = 1;
            options.probe_expert_read = 1;
        } else if (strcmp(arg, "--probe-layer") == 0 && i + 1 < argc) {
            options.probe_layer = parse_nonnegative_int(argv[++i], "--probe-layer");
        } else if (strcmp(arg, "--probe-experts") == 0 && i + 1 < argc) {
            options.probe_experts_csv = argv[++i];
        } else if (strcmp(arg, "--probe-layer-moe") == 0) {
            options.probe_layer_moe = 1;
        } else if (strcmp(arg, "--probe-router") == 0) {
            options.probe_router = 1;
        } else if (strcmp(arg, "--probe-router-moe") == 0) {
            options.probe_router = 1;
            options.probe_router_moe = 1;
        } else if (strcmp(arg, "--probe-mlp-block") == 0) {
            options.probe_router = 1;
            options.probe_router_moe = 1;
            options.probe_mlp_block = 1;
        } else if (strcmp(arg, "--probe-dense-mlp-block") == 0) {
            options.probe_dense_mlp_block = 1;
            options.open_experts = 0;
        } else if (strcmp(arg, "--probe-resident-linear") == 0) {
            options.probe_resident_linear = 1;
        } else if (strcmp(arg, "--probe-attn-projections") == 0) {
            options.probe_attn_projections = 1;
        } else if (strcmp(arg, "--probe-rope-split") == 0) {
            options.probe_rope_split = 1;
        } else if (strcmp(arg, "--probe-mla-attention") == 0) {
            options.probe_mla_attention = 1;
        } else if (strcmp(arg, "--probe-attn-output") == 0) {
            options.probe_attn_output = 1;
        } else if (strcmp(arg, "--probe-context1-o-proj-cache-output") == 0) {
            options.probe_context1_o_proj_cache_output = 1;
        } else if (strcmp(arg, "--build-context1-o-proj-cache-layer") == 0) {
            options.build_context1_o_proj_cache_layer = 1;
        } else if (strcmp(arg, "--probe-decoder-layer") == 0) {
            options.probe_decoder_layer = 1;
        } else if (strcmp(arg, "--probe-dense-decoder-layer") == 0) {
            options.probe_dense_decoder_layer = 1;
            options.open_experts = 0;
        } else if (strcmp(arg, "--probe-decode-layers") == 0) {
            options.probe_decode_layers = 1;
        } else if (strcmp(arg, "--decode-layers") == 0 && i + 1 < argc) {
            options.decode_layers_csv = argv[++i];
        } else if (strcmp(arg, "--probe-final-logits") == 0) {
            options.probe_final_logits = 1;
            options.open_experts = 0;
        } else if (strcmp(arg, "--output-topk-json") == 0 && i + 1 < argc) {
            options.output_topk_json = argv[++i];
        } else if (strcmp(arg, "--output-token-json") == 0 && i + 1 < argc) {
            options.output_token_json = argv[++i];
        } else if (strcmp(arg, "--output-next-input-f32") == 0 && i + 1 < argc) {
            options.output_next_input_f32 = argv[++i];
        } else if (strcmp(arg, "--output-generated-json") == 0 && i + 1 < argc) {
            options.output_generated_json = argv[++i];
        } else if (strcmp(arg, "--generate-request-json") == 0 && i + 1 < argc) {
            options.generate_request_json = argv[++i];
        } else if (strcmp(arg, "--generate-server-jsonl") == 0) {
            options.generate_server_jsonl = 1;
            options.json = 1;
        } else if (strcmp(arg, "--generate-token-ids") == 0) {
            options.generate_token_ids = 1;
            options.probe_decode_layers = 1;
        } else if (strcmp(arg, "--generate-steps") == 0 && i + 1 < argc) {
            options.generate_steps = parse_positive_int(argv[++i], "--generate-steps");
        } else if (strcmp(arg, "--generate-first-from-input-logits") == 0) {
            options.generate_first_from_input_logits = 1;
        } else if (strcmp(arg, "--chunk-rows") == 0 && i + 1 < argc) {
            options.final_logits_chunk_rows = parse_positive_int(argv[++i], "--chunk-rows");
        } else if (strcmp(arg, "--max-chunk-mib") == 0 && i + 1 < argc) {
            options.final_logits_max_chunk_mib = parse_positive_double(argv[++i], "--max-chunk-mib");
        } else if (strcmp(arg, "--mmap-final-logits") == 0) {
            options.mmap_final_logits = 1;
        } else if (strcmp(arg, "--max-embedding-row-mib") == 0 && i + 1 < argc) {
            options.max_embedding_row_mib =
                parse_positive_double(argv[++i], "--max-embedding-row-mib");
        } else if (strcmp(arg, "--skip-final-norm") == 0) {
            options.skip_final_norm = 1;
        } else if (strcmp(arg, "--skip-debug-intermediates") == 0) {
            options.skip_debug_intermediates = 1;
        } else if (strcmp(arg, "--include-shared-expert") == 0) {
            options.include_shared_expert = 1;
        } else if (strcmp(arg, "--resident-tensor-name") == 0 && i + 1 < argc) {
            options.resident_tensor_name = argv[++i];
        } else if (strcmp(arg, "--output-dir") == 0 && i + 1 < argc) {
            options.output_dir = argv[++i];
        } else if (strcmp(arg, "--q-b-f32") == 0 && i + 1 < argc) {
            options.q_b_f32 = argv[++i];
        } else if (strcmp(arg, "--k-f32") == 0 && i + 1 < argc) {
            options.k_f32 = argv[++i];
        } else if (strcmp(arg, "--output-q-nope-f32") == 0 && i + 1 < argc) {
            options.output_q_nope_f32 = argv[++i];
        } else if (strcmp(arg, "--output-q-rope-f32") == 0 && i + 1 < argc) {
            options.output_q_rope_f32 = argv[++i];
        } else if (strcmp(arg, "--output-q-f32") == 0 && i + 1 < argc) {
            options.output_q_f32 = argv[++i];
        } else if (strcmp(arg, "--output-k-f32") == 0 && i + 1 < argc) {
            options.output_k_f32 = argv[++i];
        } else if (strcmp(arg, "--q-nope-f32") == 0 && i + 1 < argc) {
            options.q_nope_f32 = argv[++i];
        } else if (strcmp(arg, "--q-rope-f32") == 0 && i + 1 < argc) {
            options.q_rope_f32 = argv[++i];
        } else if (strcmp(arg, "--mla-kv-b-f32") == 0 && i + 1 < argc) {
            options.mla_kv_b_f32 = argv[++i];
        } else if (strcmp(arg, "--num-heads") == 0 && i + 1 < argc) {
            options.num_heads = parse_positive_int(argv[++i], "--num-heads");
        } else if (strcmp(arg, "--kv-lora-dim") == 0 && i + 1 < argc) {
            options.kv_lora_dim = parse_positive_int(argv[++i], "--kv-lora-dim");
        } else if (strcmp(arg, "--qk-nope-dim") == 0 && i + 1 < argc) {
            options.qk_nope_dim = parse_positive_int(argv[++i], "--qk-nope-dim");
        } else if (strcmp(arg, "--rope-dim") == 0 && i + 1 < argc) {
            options.rope_dim = parse_positive_int(argv[++i], "--rope-dim");
        } else if (strcmp(arg, "--v-head-dim") == 0 && i + 1 < argc) {
            options.v_head_dim = parse_positive_int(argv[++i], "--v-head-dim");
        } else if (strcmp(arg, "--start-position") == 0 && i + 1 < argc) {
            options.start_position = parse_nonnegative_int(argv[++i], "--start-position");
        } else if (strcmp(arg, "--context-length") == 0 && i + 1 < argc) {
            options.context_length = parse_positive_int(argv[++i], "--context-length");
        } else if (strcmp(arg, "--cache-position-offset") == 0 && i + 1 < argc) {
            options.cache_position_offset =
                parse_nonnegative_int(argv[++i], "--cache-position-offset");
        } else if (strcmp(arg, "--batch-tokens") == 0 && i + 1 < argc) {
            options.batch_tokens = parse_positive_int(argv[++i], "--batch-tokens");
        } else if (strcmp(arg, "--rope-theta") == 0 && i + 1 < argc) {
            options.rope_theta = parse_positive_double(argv[++i], "--rope-theta");
        } else if (strcmp(arg, "--rope-interleave") == 0) {
            options.rope_interleave = 1;
        } else if (strcmp(arg, "--attention-scale") == 0 && i + 1 < argc) {
            options.attention_scale = parse_nonnegative_double(argv[++i], "--attention-scale");
        } else if (strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            options.top_k = parse_positive_int(argv[++i], "--top-k");
        } else if (strcmp(arg, "--router-score") == 0 && i + 1 < argc) {
            options.router_score = argv[++i];
            options.router_score_set = 1;
        } else if (strcmp(arg, "--routed-scaling-factor") == 0 && i + 1 < argc) {
            options.routed_scaling_factor = parse_positive_double(
                argv[++i],
                "--routed-scaling-factor"
            );
            options.routed_scaling_factor_set = 1;
        } else if (strcmp(arg, "--norm-topk-prob") == 0) {
            if (options.norm_topk_prob_set && options.norm_topk_prob == 0) {
                fprintf(stderr, "ERROR: --norm-topk-prob conflicts with --no-norm-topk-prob\n");
                exit(2);
            }
            options.norm_topk_prob = 1;
            options.norm_topk_prob_set = 1;
        } else if (strcmp(arg, "--no-norm-topk-prob") == 0) {
            if (options.norm_topk_prob_set && options.norm_topk_prob == 1) {
                fprintf(stderr, "ERROR: --no-norm-topk-prob conflicts with --norm-topk-prob\n");
                exit(2);
            }
            options.norm_topk_prob = 0;
            options.norm_topk_prob_set = 1;
        } else if (strcmp(arg, "--router-n-group") == 0 && i + 1 < argc) {
            options.router_n_group = parse_positive_int(argv[++i], "--router-n-group");
            options.router_n_group_set = 1;
        } else if (strcmp(arg, "--router-topk-group") == 0 && i + 1 < argc) {
            options.router_topk_group =
                parse_positive_int(argv[++i], "--router-topk-group");
            options.router_topk_group_set = 1;
        } else if (strcmp(arg, "--output-router-json") == 0 && i + 1 < argc) {
            options.output_router_json = argv[++i];
        } else if (strcmp(arg, "--ignore-router-bias") == 0) {
            options.ignore_router_bias = 1;
        } else if (strcmp(arg, "--probe-weights") == 0 && i + 1 < argc) {
            options.probe_weights_csv = argv[++i];
        } else if (strcmp(arg, "--input-f32") == 0 && i + 1 < argc) {
            options.input_f32 = argv[++i];
        } else if (strcmp(arg, "--input-token-id") == 0 && i + 1 < argc) {
            options.input_token_id = parse_nonnegative_int(argv[++i], "--input-token-id");
        } else if (strcmp(arg, "--residual-f32") == 0 && i + 1 < argc) {
            options.residual_f32 = argv[++i];
        } else if (strcmp(arg, "--projection-f32") == 0 && i + 1 < argc) {
            options.projection_f32 = argv[++i];
        } else if (strcmp(arg, "--output-f32") == 0 && i + 1 < argc) {
            options.output_f32 = argv[++i];
        } else if (strcmp(arg, "--context1-o-proj-cache-layout") == 0 && i + 1 < argc) {
            options.context1_o_proj_cache_layout = argv[++i];
        } else if (strcmp(arg, "--context1-o-proj-cache-file") == 0 && i + 1 < argc) {
            options.context1_o_proj_cache_file = argv[++i];
        } else if (strcmp(arg, "--cache-layout") == 0 && i + 1 < argc) {
            options.cache_layout = argv[++i];
        } else if (strcmp(arg, "--cache-file") == 0 && i + 1 < argc) {
            options.cache_file = argv[++i];
        } else if (strcmp(arg, "--in-memory-decode-cache") == 0) {
            options.in_memory_decode_cache = 1;
        } else if (strcmp(arg, "--cache-mla-kv-b-f32") == 0) {
            options.cache_mla_kv_b_f32 = 1;
        } else if (strcmp(arg, "--position") == 0 && i + 1 < argc) {
            options.cache_position = parse_nonnegative_int(argv[++i], "--position");
        } else if (strcmp(arg, "--max-cache-file-mib") == 0 && i + 1 < argc) {
            options.max_cache_file_mib = parse_positive_double(
                argv[++i],
                "--max-cache-file-mib"
            );
        } else if (strcmp(arg, "--max-cache-read-mib") == 0 && i + 1 < argc) {
            options.max_cache_read_mib = parse_positive_double(
                argv[++i],
                "--max-cache-read-mib"
            );
        } else if (strcmp(arg, "--max-mla-kv-b-cache-mib") == 0 && i + 1 < argc) {
            options.max_mla_kv_b_cache_mib = parse_positive_double(
                argv[++i],
                "--max-mla-kv-b-cache-mib"
            );
        } else if (strcmp(arg, "--max-context1-o-proj-build-gfma") == 0 && i + 1 < argc) {
            options.max_context1_o_proj_build_gfma = parse_positive_double(
                argv[++i],
                "--max-context1-o-proj-build-gfma"
            );
        } else if (strcmp(arg, "--expect-output0") == 0 && i + 1 < argc) {
            options.expect_output0 = parse_finite_double(argv[++i], "--expect-output0");
            options.expect_output0_set = 1;
        } else if (strcmp(arg, "--rms-norm-eps") == 0 && i + 1 < argc) {
            options.rms_norm_eps = parse_nonnegative_double(argv[++i], "--rms-norm-eps");
        } else if (strcmp(arg, "--probe-repeat") == 0 && i + 1 < argc) {
            options.probe_repeat = parse_positive_int(argv[++i], "--probe-repeat");
        } else if (strcmp(arg, "--max-live-working-set-mib") == 0 && i + 1 < argc) {
            options.max_live_working_set_mib = parse_nonnegative_double(
                argv[++i],
                "--max-live-working-set-mib"
            );
        } else if (strcmp(arg, "--min-free-unified-memory-gib") == 0 && i + 1 < argc) {
            options.min_free_unified_memory_gib = parse_nonnegative_double(
                argv[++i],
                "--min-free-unified-memory-gib"
            );
        } else {
            fprintf(stderr, "ERROR: unknown or incomplete option: %s\n", arg);
            usage(argv[0]);
            exit(2);
        }
    }
    if (options.generate_server_jsonl && options.generate_request_json != NULL) {
        fprintf(stderr,
                "ERROR: --generate-server-jsonl cannot be combined with --generate-request-json\n");
        exit(2);
    }
    if (options.generate_request_json != NULL) {
        apply_generate_request_json(&options, options.generate_request_json);
    }
    if (options.prepared_dir != NULL) {
        if (options.resident_layout == NULL) {
            static char resident_path[4096];
            snprintf(
                resident_path,
                sizeof(resident_path),
                "%s/resident/layout.json",
                options.prepared_dir
            );
            options.resident_layout = resident_path;
        }
        if (options.expert_layout == NULL) {
            static char expert_path[4096];
            snprintf(
                expert_path,
                sizeof(expert_path),
                "%s/experts/layout.json",
                options.prepared_dir
            );
            options.expert_layout = expert_path;
        }
    }
    int context1CacheStandalone =
        options.probe_context1_o_proj_cache_output ||
        options.build_context1_o_proj_cache_layer;
    if (!context1CacheStandalone &&
        (options.resident_layout == NULL || options.expert_layout == NULL)) {
        fprintf(stderr, "ERROR: provide --prepared or both layout paths\n");
        usage(argv[0]);
        exit(2);
    }
    if (options.build_context1_o_proj_cache_layer && options.resident_layout == NULL) {
        fprintf(stderr, "ERROR: --build-context1-o-proj-cache-layer requires --resident-layout or --prepared\n");
        exit(2);
    }
    int inputSourceCount =
        (options.input_f32 != NULL ? 1 : 0) +
        (options.input_token_id >= 0 ? 1 : 0) +
        (options.prompt_token_count > 0 ? 1 : 0);
    if (inputSourceCount > 1) {
        fprintf(stderr,
                "ERROR: provide only one of --input-f32, --input-token-id, or prompt_token_ids\n");
        exit(2);
    }
    if ((options.input_token_id >= 0 || options.prompt_token_count > 0) &&
        !options.probe_decode_layers) {
        fprintf(stderr,
                "ERROR: token-id inputs currently require --generate-token-ids or --probe-decode-layers\n");
        exit(2);
    }
    if (options.expert_buffer_count <= 0 ||
        options.expert_buffer_count > MAX_EXPERT_BUFFERS) {
        fprintf(stderr,
                "ERROR: --expert-buffer-count must be in 1..%d\n",
                MAX_EXPERT_BUFFERS);
        exit(2);
    }
    if (options.probe_expert_read) {
        if (!options.open_experts) {
            fprintf(stderr, "ERROR: --probe-expert-read requires open expert files\n");
            exit(2);
        }
        if (!options.probe_all_layers && options.probe_layer < 0) {
            fprintf(stderr, "ERROR: --probe-expert-read requires --probe-layer\n");
            exit(2);
        }
        if (options.probe_experts_csv == NULL || strlen(options.probe_experts_csv) == 0) {
            fprintf(stderr, "ERROR: --probe-expert-read requires --probe-experts\n");
            exit(2);
        }
    }
    if (options.probe_router || options.probe_decoder_layer) {
        if (options.probe_layer < 0) {
            fprintf(stderr, "ERROR: router probes require --probe-layer\n");
            exit(2);
        }
        if (options.input_f32 == NULL) {
            fprintf(stderr, "ERROR: --probe-router requires --input-f32\n");
            exit(2);
        }
        if (options.top_k <= 0 || options.top_k > 64) {
            fprintf(stderr, "ERROR: --top-k must be in 1..64\n");
            exit(2);
        }
        if (options.router_score_set &&
            strcmp(options.router_score, "sigmoid") != 0 &&
            strcmp(options.router_score, "softmax") != 0 &&
            strcmp(options.router_score, "raw") != 0) {
            fprintf(stderr, "ERROR: --router-score must be sigmoid, softmax, or raw\n");
            exit(2);
        }
    }
    if (options.probe_layer_moe) {
        if (!options.open_experts) {
            fprintf(stderr, "ERROR: --probe-layer-moe requires open expert files\n");
            exit(2);
        }
        if (options.probe_all_layers) {
            fprintf(stderr, "ERROR: --probe-layer-moe cannot be combined with --probe-all-layers\n");
            exit(2);
        }
        if (options.probe_layer < 0) {
            fprintf(stderr, "ERROR: --probe-layer-moe requires --probe-layer\n");
            exit(2);
        }
        if (options.probe_experts_csv == NULL || strlen(options.probe_experts_csv) == 0) {
            fprintf(stderr, "ERROR: --probe-layer-moe requires --probe-experts\n");
            exit(2);
        }
        if (options.probe_weights_csv == NULL || strlen(options.probe_weights_csv) == 0) {
            fprintf(stderr, "ERROR: --probe-layer-moe requires --probe-weights\n");
            exit(2);
        }
        if (options.input_f32 == NULL || options.output_f32 == NULL) {
            fprintf(stderr, "ERROR: --probe-layer-moe requires --input-f32 and --output-f32\n");
            exit(2);
        }
    }
    if (options.probe_router_moe) {
        if (!options.open_experts) {
            fprintf(stderr, "ERROR: --probe-router-moe requires open expert files\n");
            exit(2);
        }
        if (options.probe_all_layers) {
            fprintf(stderr, "ERROR: --probe-router-moe cannot be combined with --probe-all-layers\n");
            exit(2);
        }
        if (options.probe_layer < 0) {
            fprintf(stderr, "ERROR: --probe-router-moe requires --probe-layer\n");
            exit(2);
        }
        if (options.input_f32 == NULL || options.output_f32 == NULL) {
            fprintf(stderr, "ERROR: router-driven MoE probes require --input-f32 and --output-f32\n");
            exit(2);
        }
    }
    if (options.include_shared_expert &&
        !options.probe_mlp_block &&
        !options.probe_decoder_layer &&
        !options.probe_decode_layers) {
        fprintf(stderr,
                "ERROR: --include-shared-expert currently requires --probe-mlp-block, --probe-decoder-layer, or --probe-decode-layers\n");
        exit(2);
    }
    if (options.probe_resident_linear) {
        if (options.resident_tensor_name == NULL || strlen(options.resident_tensor_name) == 0) {
            fprintf(stderr, "ERROR: --probe-resident-linear requires --resident-tensor-name\n");
            exit(2);
        }
        if (options.input_f32 == NULL || options.output_f32 == NULL) {
            fprintf(stderr,
                    "ERROR: --probe-resident-linear requires --input-f32 and --output-f32\n");
            exit(2);
        }
    }
    if (options.probe_dense_mlp_block) {
        if (options.probe_layer < 0) {
            fprintf(stderr, "ERROR: --probe-dense-mlp-block requires --probe-layer\n");
            exit(2);
        }
        if (options.input_f32 == NULL || options.output_f32 == NULL) {
            fprintf(stderr,
                    "ERROR: --probe-dense-mlp-block requires --input-f32 and --output-f32\n");
            exit(2);
        }
    }
    if (options.probe_attn_projections) {
        if (options.probe_layer < 0) {
            fprintf(stderr, "ERROR: --probe-attn-projections requires --probe-layer\n");
            exit(2);
        }
        if (options.input_f32 == NULL || options.output_dir == NULL) {
            fprintf(stderr,
                    "ERROR: --probe-attn-projections requires --input-f32 and --output-dir\n");
            exit(2);
        }
    }
    if (options.probe_decoder_layer &&
        (options.probe_attn_projections ||
         options.probe_rope_split ||
         options.probe_mla_attention ||
         options.probe_attn_output ||
         options.probe_mlp_block ||
         options.probe_router ||
         options.probe_router_moe ||
         options.probe_layer_moe ||
         options.probe_resident_linear ||
         options.probe_dense_mlp_block ||
         options.probe_dense_decoder_layer ||
         options.probe_decode_layers ||
         options.probe_final_logits ||
         options.probe_expert_read)) {
        fprintf(stderr, "ERROR: run --probe-decoder-layer separately from component probes\n");
        exit(2);
    }
    if (options.probe_dense_decoder_layer &&
        (options.probe_attn_projections ||
         options.probe_rope_split ||
         options.probe_mla_attention ||
         options.probe_attn_output ||
         options.probe_mlp_block ||
         options.probe_router ||
         options.probe_router_moe ||
         options.probe_layer_moe ||
         options.probe_resident_linear ||
         options.probe_dense_mlp_block ||
         options.probe_decoder_layer ||
         options.probe_decode_layers ||
         options.probe_final_logits ||
         options.probe_expert_read)) {
        fprintf(stderr, "ERROR: run --probe-dense-decoder-layer separately from other probes\n");
        exit(2);
    }
    if (options.probe_dense_mlp_block &&
        (options.probe_attn_projections ||
         options.probe_rope_split ||
         options.probe_mla_attention ||
         options.probe_attn_output ||
         options.probe_mlp_block ||
         options.probe_router ||
         options.probe_router_moe ||
         options.probe_layer_moe ||
         options.probe_resident_linear ||
         options.probe_expert_read)) {
        fprintf(stderr, "ERROR: run --probe-dense-mlp-block separately from other probes\n");
        exit(2);
    }
    if (options.probe_decode_layers &&
        (options.probe_attn_projections ||
         options.probe_rope_split ||
         options.probe_mla_attention ||
         options.probe_attn_output ||
         options.probe_mlp_block ||
         options.probe_router ||
         options.probe_router_moe ||
         options.probe_layer_moe ||
         options.probe_resident_linear ||
         options.probe_dense_mlp_block ||
         options.probe_decoder_layer ||
         options.probe_dense_decoder_layer ||
         options.probe_final_logits ||
         options.probe_expert_read)) {
        fprintf(stderr, "ERROR: run --probe-decode-layers separately from other probes\n");
        exit(2);
    }
    if (options.probe_final_logits &&
        (options.probe_attn_projections ||
         options.probe_rope_split ||
         options.probe_mla_attention ||
         options.probe_attn_output ||
         options.probe_mlp_block ||
         options.probe_router ||
         options.probe_router_moe ||
         options.probe_layer_moe ||
         options.probe_resident_linear ||
         options.probe_dense_mlp_block ||
         options.probe_decoder_layer ||
         options.probe_dense_decoder_layer ||
         options.probe_decode_layers ||
         options.probe_expert_read)) {
        fprintf(stderr, "ERROR: run --probe-final-logits separately from other probes\n");
        exit(2);
    }
    if (options.output_next_input_f32 &&
        !options.probe_final_logits &&
        !options.probe_decode_layers) {
        fprintf(stderr,
                "ERROR: --output-next-input-f32 requires --probe-final-logits, --generate-token-ids, or --probe-decode-layers\n");
        exit(2);
    }
    if (options.generate_steps > 0 && !options.probe_decode_layers) {
        fprintf(stderr,
                "ERROR: --generate-steps requires --generate-token-ids or --probe-decode-layers\n");
        exit(2);
    }
    if (options.generate_token_ids && options.generate_steps <= 0) {
        fprintf(stderr, "ERROR: --generate-token-ids requires --generate-steps\n");
        exit(2);
    }
    if (options.generate_steps > 0 && options.probe_final_logits) {
        fprintf(stderr, "ERROR: --generate-steps cannot be combined with --probe-final-logits\n");
        exit(2);
    }
    if (options.generate_first_from_input_logits &&
        (!options.probe_decode_layers || options.generate_steps <= 0)) {
        fprintf(stderr,
                "ERROR: --generate-first-from-input-logits requires --probe-decode-layers and --generate-steps\n");
        exit(2);
    }
    if (options.output_generated_json && options.generate_steps <= 0) {
        fprintf(stderr, "ERROR: --output-generated-json requires --generate-steps\n");
        exit(2);
    }
    if (options.probe_attn_projections && options.probe_mla_attention) {
        fprintf(stderr, "ERROR: run --probe-attn-projections and --probe-mla-attention separately\n");
        exit(2);
    }
    if (options.mla_kv_b_f32 != NULL &&
        (!options.probe_mla_attention ||
         options.probe_decoder_layer ||
         options.probe_dense_decoder_layer ||
         options.probe_decode_layers)) {
        fprintf(stderr,
                "ERROR: --mla-kv-b-f32 is only supported for standalone --probe-mla-attention\n");
        exit(2);
    }
    int cache_args_present =
        options.cache_layout != NULL ||
        options.cache_file != NULL ||
        options.cache_position >= 0;
    int append_cache =
        (options.probe_attn_projections && cache_args_present) ||
        options.probe_decoder_layer ||
        options.probe_dense_decoder_layer ||
        options.probe_decode_layers;
    if (append_cache) {
        if (options.cache_layout == NULL ||
            options.cache_file == NULL ||
            options.cache_position < 0) {
            fprintf(stderr,
                    "ERROR: --cache-layout, --cache-file, and --position must be provided together\n");
            exit(2);
        }
        if (options.max_cache_file_mib <= 0.0) {
            fprintf(stderr, "ERROR: cache append requires --max-cache-file-mib\n");
            exit(2);
        }
    } else if (cache_args_present &&
               !options.probe_mla_attention &&
               !options.probe_decoder_layer &&
               !options.probe_dense_decoder_layer &&
               !options.probe_decode_layers) {
        fprintf(stderr,
                "ERROR: cache arguments require attention projection, MLA attention, or decoder-layer probe\n");
        exit(2);
    }
    if (options.probe_mla_attention) {
        if (options.probe_layer < 0) {
            fprintf(stderr, "ERROR: --probe-mla-attention requires --probe-layer\n");
            exit(2);
        }
        if (options.cache_layout == NULL ||
            options.cache_file == NULL ||
            options.q_nope_f32 == NULL ||
            options.q_rope_f32 == NULL ||
            options.output_f32 == NULL) {
            fprintf(stderr,
                    "ERROR: --probe-mla-attention requires cache, q_nope/q_rope, and output paths\n");
            exit(2);
        }
        if (options.num_heads <= 0 ||
            (options.mla_kv_b_f32 != NULL && options.kv_lora_dim <= 0) ||
            options.qk_nope_dim <= 0 ||
            options.rope_dim <= 0 ||
            (options.rope_dim % 2) != 0 ||
            options.v_head_dim <= 0 ||
            options.context_length <= 0 ||
            (uint64_t)options.cache_position_offset +
                    (uint64_t)options.context_length - 1u > UINT32_MAX) {
            fprintf(stderr, "ERROR: --probe-mla-attention dimensions are invalid\n");
            exit(2);
        }
        if (options.max_cache_file_mib <= 0.0 ||
            options.max_cache_read_mib <= 0.0) {
            fprintf(stderr,
                    "ERROR: --probe-mla-attention requires --max-cache-file-mib and --max-cache-read-mib\n");
            exit(2);
        }
    }
    if (options.probe_attn_output) {
        if (options.probe_layer < 0) {
            fprintf(stderr, "ERROR: --probe-attn-output requires --probe-layer\n");
            exit(2);
        }
        if (options.input_f32 == NULL ||
            options.residual_f32 == NULL ||
            options.output_f32 == NULL) {
            fprintf(stderr,
                    "ERROR: --probe-attn-output requires --input-f32, --residual-f32, and --output-f32\n");
            exit(2);
        }
    }
    if (options.probe_decoder_layer) {
        if (!options.open_experts) {
            fprintf(stderr, "ERROR: --probe-decoder-layer requires open expert files\n");
            exit(2);
        }
        if (options.probe_all_layers) {
            fprintf(stderr, "ERROR: --probe-decoder-layer cannot be combined with --probe-all-layers\n");
            exit(2);
        }
        if (options.probe_layer < 0) {
            fprintf(stderr, "ERROR: --probe-decoder-layer requires --probe-layer\n");
            exit(2);
        }
        if (options.input_f32 == NULL ||
            options.output_f32 == NULL ||
            options.output_dir == NULL ||
            options.cache_layout == NULL ||
            options.cache_file == NULL ||
            options.cache_position < 0) {
            fprintf(stderr,
                    "ERROR: --probe-decoder-layer requires input/output, output-dir, cache paths, and --position\n");
            exit(2);
        }
        if (options.num_heads <= 0 ||
            options.kv_lora_dim <= 0 ||
            options.qk_nope_dim <= 0 ||
            options.rope_dim <= 0 ||
            (options.rope_dim % 2) != 0 ||
            options.v_head_dim <= 0 ||
            options.context_length <= 0 ||
            (uint64_t)options.cache_position_offset +
                    (uint64_t)options.context_length - 1u > UINT32_MAX ||
            (uint64_t)options.cache_position > UINT32_MAX) {
            fprintf(stderr, "ERROR: --probe-decoder-layer dimensions/positions are invalid\n");
            exit(2);
        }
        if (options.max_cache_file_mib <= 0.0 ||
            options.max_cache_read_mib <= 0.0) {
            fprintf(stderr,
                    "ERROR: --probe-decoder-layer requires --max-cache-file-mib and --max-cache-read-mib\n");
            exit(2);
        }
    }
    if (options.probe_dense_decoder_layer) {
        if (options.probe_layer < 0) {
            fprintf(stderr, "ERROR: --probe-dense-decoder-layer requires --probe-layer\n");
            exit(2);
        }
        if (options.input_f32 == NULL ||
            options.output_f32 == NULL ||
            options.output_dir == NULL ||
            options.cache_layout == NULL ||
            options.cache_file == NULL ||
            options.cache_position < 0) {
            fprintf(stderr,
                    "ERROR: --probe-dense-decoder-layer requires input/output, output-dir, cache paths, and --position\n");
            exit(2);
        }
        if (options.num_heads <= 0 ||
            options.kv_lora_dim <= 0 ||
            options.qk_nope_dim <= 0 ||
            options.rope_dim <= 0 ||
            (options.rope_dim % 2) != 0 ||
            options.v_head_dim <= 0 ||
            options.context_length <= 0 ||
            (uint64_t)options.cache_position_offset +
                    (uint64_t)options.context_length - 1u > UINT32_MAX ||
            (uint64_t)options.cache_position > UINT32_MAX) {
            fprintf(stderr, "ERROR: --probe-dense-decoder-layer dimensions/positions are invalid\n");
            exit(2);
        }
        if (options.max_cache_file_mib <= 0.0 ||
            options.max_cache_read_mib <= 0.0) {
            fprintf(stderr,
                    "ERROR: --probe-dense-decoder-layer requires --max-cache-file-mib and --max-cache-read-mib\n");
            exit(2);
        }
    }
    if (options.probe_decode_layers) {
        if (options.decode_layers_csv == NULL || strlen(options.decode_layers_csv) == 0) {
            fprintf(stderr, "ERROR: --probe-decode-layers requires --decode-layers\n");
            exit(2);
        }
        if ((options.input_f32 == NULL &&
             options.input_token_id < 0 &&
             options.prompt_token_count <= 0) ||
            options.output_f32 == NULL ||
            options.output_dir == NULL ||
            options.cache_layout == NULL ||
            options.cache_file == NULL ||
            options.cache_position < 0) {
            fprintf(stderr,
                    "ERROR: --generate-token-ids/--probe-decode-layers requires --input-f32, --input-token-id, or prompt_token_ids, output/output-dir, cache paths, and --position\n");
            exit(2);
        }
        if (options.num_heads <= 0 ||
            options.kv_lora_dim <= 0 ||
            options.qk_nope_dim <= 0 ||
            options.rope_dim <= 0 ||
            (options.rope_dim % 2) != 0 ||
            options.v_head_dim <= 0 ||
            options.context_length <= 0 ||
            (uint64_t)options.cache_position_offset +
                    (uint64_t)options.context_length - 1u > UINT32_MAX ||
            (uint64_t)options.cache_position > UINT32_MAX ||
            options.top_k <= 0 ||
            options.top_k > 64) {
            fprintf(stderr, "ERROR: --probe-decode-layers dimensions/positions/top-k are invalid\n");
            exit(2);
        }
        uint64_t promptTokenCount = options.prompt_token_count > 0
            ? (uint64_t)options.prompt_token_count
            : 0;
        uint64_t decodePositions = promptTokenCount > 0
            ? promptTokenCount + (uint64_t)options.generate_steps - 1u
            : (uint64_t)options.generate_steps;
        if (options.generate_steps > 0 &&
            ((uint64_t)options.cache_position + decodePositions - 1u > UINT32_MAX ||
             (uint64_t)options.cache_position_offset +
                     (uint64_t)options.context_length +
                     decodePositions - 2u > UINT32_MAX)) {
            fprintf(stderr, "ERROR: --generate-steps would overflow decode cache positions\n");
            exit(2);
        }
        if (options.max_cache_file_mib <= 0.0 ||
            options.max_cache_read_mib <= 0.0) {
            fprintf(stderr,
                    "ERROR: --probe-decode-layers requires --max-cache-file-mib and --max-cache-read-mib\n");
            exit(2);
        }
    }
    if (options.probe_final_logits) {
        if (options.input_f32 == NULL) {
            fprintf(stderr, "ERROR: --probe-final-logits requires --input-f32\n");
            exit(2);
        }
        if (options.top_k <= 0 || options.top_k > 64) {
            fprintf(stderr, "ERROR: --top-k must be in 1..64 for --probe-final-logits\n");
            exit(2);
        }
        if (options.final_logits_max_chunk_mib <= 0.0) {
            fprintf(stderr, "ERROR: --max-chunk-mib must be positive\n");
            exit(2);
        }
    }
    if (options.probe_rope_split) {
        if (!options.q_b_f32 || !options.k_f32 ||
            !options.output_q_nope_f32 || !options.output_q_rope_f32 ||
            !options.output_q_f32 || !options.output_k_f32) {
            fprintf(stderr, "ERROR: --probe-rope-split requires q/k input and output paths\n");
            exit(2);
        }
        if (options.num_heads <= 0 ||
            options.qk_nope_dim <= 0 ||
            options.rope_dim <= 0 ||
            (options.rope_dim % 2) != 0 ||
            options.start_position < 0 ||
            options.batch_tokens <= 0 ||
            (uint64_t)options.start_position + (uint64_t)options.batch_tokens - 1u > UINT32_MAX) {
            fprintf(stderr,
                    "ERROR: --probe-rope-split dimensions/positions are invalid\n");
            exit(2);
        }
    }
    if (options.cache_mla_kv_b_f32 && options.max_mla_kv_b_cache_mib <= 0.0) {
        fprintf(stderr,
                "ERROR: --cache-mla-kv-b-f32 requires --max-mla-kv-b-cache-mib\n");
        exit(2);
    }
    return options;
}

static int double_gib_to_u64_bytes(double gib, uint64_t *out) {
    if (!isfinite(gib) || gib < 0.0) {
        return 0;
    }
    double bytes = ceil(gib * 1024.0 * 1024.0 * 1024.0);
    if (!isfinite(bytes) || bytes < 0.0 || bytes > (double)UINT64_MAX) {
        return 0;
    }
    *out = (uint64_t)bytes;
    return 1;
}

static int read_system_memory_snapshot(SystemMemorySnapshot *snapshot) {
    if (!snapshot) {
        return 0;
    }
    memset(snapshot, 0, sizeof(*snapshot));
    mach_port_t host = mach_host_self();
    vm_size_t pageSize = 0;
    if (host_page_size(host, &pageSize) != KERN_SUCCESS || pageSize == 0) {
        long fallback = sysconf(_SC_PAGESIZE);
        if (fallback <= 0) {
            mach_port_deallocate(mach_task_self(), host);
            return 0;
        }
        pageSize = (vm_size_t)fallback;
    }
    vm_statistics64_data_t vmStats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(
        host,
        HOST_VM_INFO64,
        (host_info64_t)&vmStats,
        &count
    );
    if (kr != KERN_SUCCESS) {
        mach_port_deallocate(mach_task_self(), host);
        return 0;
    }
    uint64_t availablePages =
        (uint64_t)vmStats.free_count +
        (uint64_t)vmStats.inactive_count +
        (uint64_t)vmStats.speculative_count;
    if (pageSize > 0 && availablePages > UINT64_MAX / (uint64_t)pageSize) {
        mach_port_deallocate(mach_task_self(), host);
        return 0;
    }
    snapshot->available_bytes = availablePages * (uint64_t)pageSize;
    snapshot->page_size = (uint64_t)pageSize;
    uint64_t totalBytes = 0;
    size_t totalSize = sizeof(totalBytes);
    if (sysctlbyname("hw.memsize", &totalBytes, &totalSize, NULL, 0) == 0 &&
        totalSize == sizeof(totalBytes)) {
        snapshot->total_bytes = totalBytes;
    }
    snapshot->ok = 1;
    mach_port_deallocate(mach_task_self(), host);
    return 1;
}

static IntList parse_nonnegative_int_csv(const char *text, const char *name) {
    IntList list = {.values = NULL, .count = 0};
    if (text == NULL || *text == '\0') {
        fprintf(stderr, "ERROR: %s must be a non-empty comma-separated list\n", name);
        exit(2);
    }
    char *copy = strdup(text);
    if (!copy) {
        fprintf(stderr, "ERROR: out of memory\n");
        exit(1);
    }
    int capacity = 8;
    list.values = (int *)calloc((size_t)capacity, sizeof(int));
    if (!list.values) {
        fprintf(stderr, "ERROR: out of memory\n");
        free(copy);
        exit(1);
    }
    char *saveptr = NULL;
    char *token = strtok_r(copy, ",", &saveptr);
    while (token != NULL) {
        while (isspace((unsigned char)*token)) {
            token++;
        }
        char *end = NULL;
        errno = 0;
        long value = strtol(token, &end, 10);
        while (end && isspace((unsigned char)*end)) {
            end++;
        }
        if (errno || end == token || *end != '\0' || value < 0 || value > INT_MAX) {
            fprintf(stderr, "ERROR: %s contains invalid expert id: %s\n", name, token);
            free(copy);
            free(list.values);
            exit(2);
        }
        if (list.count == capacity) {
            capacity *= 2;
            int *grown = (int *)realloc(list.values, (size_t)capacity * sizeof(int));
            if (!grown) {
                fprintf(stderr, "ERROR: out of memory\n");
                free(copy);
                free(list.values);
                exit(1);
            }
            list.values = grown;
        }
        list.values[list.count++] = (int)value;
        token = strtok_r(NULL, ",", &saveptr);
    }
    free(copy);
    if (list.count == 0) {
        fprintf(stderr, "ERROR: %s must contain at least one expert id\n", name);
        free(list.values);
        exit(2);
    }
    return list;
}

static FloatList parse_float_csv(const char *text, const char *name) {
    FloatList list = {.values = NULL, .count = 0};
    if (text == NULL || *text == '\0') {
        fprintf(stderr, "ERROR: %s must be a non-empty comma-separated list\n", name);
        exit(2);
    }
    char *copy = strdup(text);
    if (!copy) {
        fprintf(stderr, "ERROR: out of memory\n");
        exit(1);
    }
    int capacity = 8;
    list.values = (float *)calloc((size_t)capacity, sizeof(float));
    if (!list.values) {
        fprintf(stderr, "ERROR: out of memory\n");
        free(copy);
        exit(1);
    }
    char *saveptr = NULL;
    char *token = strtok_r(copy, ",", &saveptr);
    while (token != NULL) {
        while (isspace((unsigned char)*token)) {
            token++;
        }
        char *end = NULL;
        errno = 0;
        double value = strtod(token, &end);
        while (end && isspace((unsigned char)*end)) {
            end++;
        }
        if (errno || end == token || *end != '\0' || !isfinite(value)) {
            fprintf(stderr, "ERROR: %s contains invalid float: %s\n", name, token);
            free(copy);
            free(list.values);
            exit(2);
        }
        if (list.count == capacity) {
            capacity *= 2;
            float *grown = (float *)realloc(list.values, (size_t)capacity * sizeof(float));
            if (!grown) {
                fprintf(stderr, "ERROR: out of memory\n");
                free(copy);
                free(list.values);
                exit(1);
            }
            list.values = grown;
        }
        list.values[list.count++] = (float)value;
        token = strtok_r(NULL, ",", &saveptr);
    }
    free(copy);
    if (list.count == 0) {
        fprintf(stderr, "ERROR: %s must contain at least one float\n", name);
        free(list.values);
        exit(2);
    }
    return list;
}

static NSDictionary *load_json_dictionary(NSString *path) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data) {
        fprintf(stderr, "ERROR: failed to read JSON %s\n", [path UTF8String]);
        exit(1);
    }
    NSError *error = nil;
    id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (!object || ![object isKindOfClass:[NSDictionary class]]) {
        fprintf(
            stderr,
            "ERROR: failed to parse JSON %s: %s\n",
            [path UTF8String],
            error ? [[error description] UTF8String] : "not an object"
        );
        exit(1);
    }
    return (NSDictionary *)object;
}

static NSString *dirname_string(NSString *path) {
    return [path stringByDeletingLastPathComponent];
}

static NSString *join_path(NSString *base, NSString *child) {
    if ([child isAbsolutePath]) {
        return child;
    }
    return [base stringByAppendingPathComponent:child];
}

static int ensure_output_dir(NSString *outputDir) {
    NSError *error = nil;
    if (![[NSFileManager defaultManager] createDirectoryAtPath:outputDir
                                   withIntermediateDirectories:YES
                                                    attributes:nil
                                                         error:&error]) {
        fprintf(stderr,
                "ERROR: failed to create output dir %s: %s\n",
                [outputDir UTF8String],
                error ? [[error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    return 1;
}

static int write_data_to_output_dir(NSString *outputDir,
                                    NSString *filename,
                                    NSData *data) {
    NSString *path = [outputDir stringByAppendingPathComponent:filename];
    if (![data writeToFile:path atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write %s\n", [path UTF8String]);
        return 0;
    }
    return 1;
}

static uint64_t unsigned_number(id value, const char *name) {
    if (![value isKindOfClass:[NSNumber class]]) {
        fprintf(stderr, "ERROR: %s must be a number\n", name);
        exit(1);
    }
    long long signed_value = [(NSNumber *)value longLongValue];
    if (signed_value < 0) {
        fprintf(stderr, "ERROR: %s must be non-negative\n", name);
        exit(1);
    }
    return (uint64_t)signed_value;
}

static uint64_t file_size_or_die(NSString *path) {
    struct stat st;
    if (stat([path fileSystemRepresentation], &st) != 0) {
        fprintf(
            stderr,
            "ERROR: failed to stat %s: %s\n",
            [path UTF8String],
            strerror(errno)
        );
        exit(1);
    }
    if (st.st_size < 0) {
        fprintf(stderr, "ERROR: negative file size for %s\n", [path UTF8String]);
        exit(1);
    }
    return (uint64_t)st.st_size;
}

static uint64_t round_up_u64(uint64_t value, uint64_t alignment) {
    if (alignment == 0) {
        return value;
    }
    uint64_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

static char *copy_cstr(NSString *value) {
    const char *utf8 = [value UTF8String];
    size_t len = strlen(utf8);
    char *copy = (char *)malloc(len + 1);
    if (!copy) {
        fprintf(stderr, "ERROR: out of memory\n");
        exit(1);
    }
    memcpy(copy, utf8, len + 1);
    return copy;
}

static void register_shared_resident_file(NSString *path, int fd) {
    g_shared_resident_path = path;
    g_shared_resident_fd = fd;
}

static void clear_shared_resident_file_if_fd(int fd) {
    if (g_shared_resident_fd == fd) {
        g_shared_resident_fd = -1;
        g_shared_resident_path = nil;
    }
}

static int open_resident_read_fd(NSString *residentBinPath, int *closeWhenDone) {
    if (closeWhenDone) {
        *closeWhenDone = 0;
    }
    if (g_shared_resident_fd >= 0 &&
        g_shared_resident_path &&
        [g_shared_resident_path isEqualToString:residentBinPath]) {
        return g_shared_resident_fd;
    }
    int fd = open([residentBinPath fileSystemRepresentation], O_RDONLY);
    if (fd >= 0 && closeWhenDone) {
        *closeWhenDone = 1;
    }
    return fd;
}

static void close_resident_read_fd(int fd, int closeWhenDone) {
    if (closeWhenDone && fd >= 0) {
        close(fd);
    }
}

static void register_shared_decode_cache_file(NSString *path, int fd) {
    g_shared_decode_cache_path = path;
    g_shared_decode_cache_fd = fd;
    g_shared_decode_cache_memory = nil;
}

static void register_shared_decode_cache_memory(NSString *path,
                                                NSMutableData *data) {
    g_shared_decode_cache_path = path;
    g_shared_decode_cache_fd = -1;
    g_shared_decode_cache_memory = data;
}

static NSMutableData *shared_decode_cache_memory_for_path(NSString *path) {
    if (g_shared_decode_cache_memory &&
        g_shared_decode_cache_path &&
        [g_shared_decode_cache_path isEqualToString:path]) {
        return g_shared_decode_cache_memory;
    }
    return nil;
}

static void clear_shared_decode_cache_file_if_fd(int fd) {
    if (g_shared_decode_cache_fd == fd) {
        g_shared_decode_cache_fd = -1;
        g_shared_decode_cache_path = nil;
        g_shared_decode_cache_memory = nil;
    }
}

static int open_decode_cache_fd(NSString *cacheFilePath,
                                int writable,
                                int *closeWhenDone) {
    if (closeWhenDone) {
        *closeWhenDone = 0;
    }
    if (g_shared_decode_cache_fd >= 0 &&
        g_shared_decode_cache_path &&
        [g_shared_decode_cache_path isEqualToString:cacheFilePath]) {
        return g_shared_decode_cache_fd;
    }
    int fd = open([cacheFilePath fileSystemRepresentation], writable ? O_WRONLY : O_RDONLY);
    if (fd >= 0 && closeWhenDone) {
        *closeWhenDone = 1;
    }
    return fd;
}

static void close_decode_cache_fd(int fd, int closeWhenDone) {
    if (closeWhenDone && fd >= 0) {
        close(fd);
    }
}

static int ensure_runtime_decode_cache_file(GlmMoeRuntimeContext *runtime,
                                            NSString *cacheFilePath) {
    if (!runtime || !cacheFilePath) {
        return 1;
    }
    if (runtime.decodeCacheFd >= 0 &&
        runtime.decodeCachePath &&
        [runtime.decodeCachePath isEqualToString:cacheFilePath]) {
        register_shared_decode_cache_file(runtime.decodeCachePath, runtime.decodeCacheFd);
        return 1;
    }
    if (runtime.decodeCacheFd >= 0) {
        clear_shared_decode_cache_file_if_fd(runtime.decodeCacheFd);
        close(runtime.decodeCacheFd);
        runtime.decodeCacheFd = -1;
        runtime.decodeCachePath = nil;
    }
    int fd = open([cacheFilePath fileSystemRepresentation], O_RDWR);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open decode cache file %s: %s\n",
                [cacheFilePath UTF8String],
                strerror(errno));
        return 0;
    }
    runtime.decodeCacheFd = fd;
    runtime.decodeCachePath = cacheFilePath;
    runtime.decodeCacheFdOpenCount++;
    register_shared_decode_cache_file(runtime.decodeCachePath, runtime.decodeCacheFd);
    return 1;
}

static GlmMoeRuntimeContext *create_glm_moe_runtime_context(LoaderOptions options,
                                                            int *status) {
    if (status) {
        *status = 1;
    }
    GlmMoeRuntimeContext *ctx = [[GlmMoeRuntimeContext alloc] init];
    ctx.residentFd = -1;
    ctx.decodeCacheFd = -1;
    ctx.expertBuffers = [NSMutableArray array];
    ctx.residentLayoutPath = [NSString stringWithUTF8String:options.resident_layout];
    ctx.expertLayoutPath = [NSString stringWithUTF8String:options.expert_layout];
    ctx.residentLayout = load_json_dictionary(ctx.residentLayoutPath);
    ctx.expertLayout = load_json_dictionary(ctx.expertLayoutPath);
    ctx.residentDir = dirname_string(ctx.residentLayoutPath);
    ctx.expertDir = dirname_string(ctx.expertLayoutPath);

    ctx.device = MTLCreateSystemDefaultDevice();
    if (!ctx.device) {
        fprintf(stderr, "ERROR: no Metal device available\n");
        return nil;
    }

    NSString *weightFile = ctx.residentLayout[@"weight_file"];
    if (![weightFile isKindOfClass:[NSString class]] || [weightFile length] == 0) {
        fprintf(stderr, "ERROR: resident layout missing weight_file\n");
        return nil;
    }
    ctx.residentBinPath = join_path(ctx.residentDir, weightFile);
    ctx.residentLayoutBytes = unsigned_number(
        ctx.residentLayout[@"total_bytes"],
        "resident total_bytes"
    );
    ctx.residentFileBytes = file_size_or_die(ctx.residentBinPath);
    if (ctx.residentFileBytes != ctx.residentLayoutBytes) {
        fprintf(
            stderr,
            "ERROR: resident file bytes %llu do not match layout total_bytes %llu\n",
            (unsigned long long)ctx.residentFileBytes,
            (unsigned long long)ctx.residentLayoutBytes
        );
        return nil;
    }
    ctx.residentFd = open([ctx.residentBinPath fileSystemRepresentation], O_RDONLY);
    if (ctx.residentFd < 0) {
        fprintf(
            stderr,
            "ERROR: failed to open resident file %s: %s\n",
            [ctx.residentBinPath UTF8String],
            strerror(errno)
        );
        return nil;
    }
    register_shared_resident_file(ctx.residentBinPath, ctx.residentFd);

    NSArray *layers = ctx.expertLayout[@"layers"];
    if (![layers isKindOfClass:[NSArray class]] || [layers count] == 0) {
        fprintf(stderr, "ERROR: expert layout missing non-empty layers\n");
        return nil;
    }
    ctx.layers = layers;
    ctx.expertFiles = calloc([layers count], sizeof(ExpertFile));
    if (!ctx.expertFiles) {
        fprintf(stderr, "ERROR: out of memory\n");
        return nil;
    }
    for (NSUInteger i = 0; i < [layers count]; i++) {
        ctx.expertFiles[i].fd = -1;
    }
    for (NSUInteger i = 0; i < [layers count]; i++) {
        NSDictionary *layer = layers[i];
        if (![layer isKindOfClass:[NSDictionary class]]) {
            fprintf(stderr, "ERROR: expert layer entry is not an object\n");
            return nil;
        }
        uint64_t layerId = unsigned_number(layer[@"layer"], "expert layer");
        if (layerId > (uint64_t)INT_MAX) {
            fprintf(stderr, "ERROR: expert layer id exceeds INT_MAX\n");
            return nil;
        }
        uint64_t numExperts = unsigned_number(layer[@"num_experts"], "num_experts");
        uint64_t slotBytes = unsigned_number(
            layer[@"expert_slot_bytes"],
            "expert_slot_bytes"
        );
        NSString *layerFile = layer[@"layer_file"];
        if (![layerFile isKindOfClass:[NSString class]] || [layerFile length] == 0) {
            fprintf(stderr, "ERROR: expert layer missing layer_file\n");
            return nil;
        }
        NSString *layerPath = join_path(ctx.expertDir, layerFile);
        uint64_t expectedBytes = numExperts * slotBytes;
        uint64_t actualBytes = file_size_or_die(layerPath);
        if (actualBytes != expectedBytes) {
            fprintf(
                stderr,
                "ERROR: %s bytes %llu do not match expected %llu\n",
                [layerPath UTF8String],
                (unsigned long long)actualBytes,
                (unsigned long long)expectedBytes
            );
            return nil;
        }
        ctx.expertFiles[i].path = copy_cstr(layerPath);
        ctx.expertFiles[i].expected_bytes = expectedBytes;
        ctx.expertFiles[i].actual_bytes = actualBytes;
        ctx.expertFiles[i].expert_slot_bytes = slotBytes;
        ctx.expertFiles[i].num_experts = numExperts;
        ctx.expertFiles[i].layer = (int)layerId;
        if (options.open_experts) {
            int fd = open([layerPath fileSystemRepresentation], O_RDONLY);
            if (fd < 0) {
                fprintf(
                    stderr,
                    "ERROR: failed to open expert layer %s: %s\n",
                    [layerPath UTF8String],
                    strerror(errno)
                );
                return nil;
            }
            ctx.expertFiles[i].fd = fd;
#ifdef F_RDAHEAD
            (void)fcntl(fd, F_RDAHEAD, 0);
#endif
            ctx.openedExpertFiles++;
        }
        if (slotBytes > ctx.maxExpertSlotBytes) {
            ctx.maxExpertSlotBytes = slotBytes;
        }
        ctx.totalExpertBytes += actualBytes;
    }
    if (status) {
        *status = 0;
    }
    return ctx;
}

static int ensure_runtime_expert_buffers(GlmMoeRuntimeContext *runtime,
                                         uint64_t expertBufferBytes,
                                         int neededCount) {
    if (neededCount < 0 || neededCount > MAX_EXPERT_BUFFERS) {
        fprintf(stderr,
                "ERROR: requested expert buffer count %d exceeds supported range\n",
                neededCount);
        return 0;
    }
    if (neededCount == 0) {
        return 1;
    }
    if (!runtime.expertBuffers) {
        runtime.expertBuffers = [NSMutableArray array];
    }
    if (runtime.expertBufferBytes != 0 &&
        runtime.expertBufferBytes != expertBufferBytes) {
        [runtime.expertBuffers removeAllObjects];
        runtime.expertBufferBytes = 0;
    }
    runtime.expertBufferBytes = expertBufferBytes;
    while ((int)[runtime.expertBuffers count] < neededCount) {
        void *ptr = NULL;
        int rc = posix_memalign(&ptr, 2 * 1024 * 1024, (size_t)expertBufferBytes);
        if (rc != 0 || !ptr) {
            fprintf(stderr, "ERROR: failed to allocate aligned expert buffer\n");
            return 0;
        }
        id<MTLBuffer> buffer =
            [runtime.device newBufferWithBytesNoCopy:ptr
                                              length:(NSUInteger)expertBufferBytes
                                             options:MTLResourceStorageModeShared
                                         deallocator:^(void *pointer, NSUInteger length) {
                                             (void)length;
                                             free(pointer);
                                         }];
        if (!buffer) {
            free(ptr);
            fprintf(stderr, "ERROR: failed to wrap expert buffer as Metal buffer\n");
            return 0;
        }
        [runtime.expertBuffers addObject:buffer];
    }
    return 1;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int env_flag_enabled(const char *name) {
    const char *value = getenv(name);
    if (!value || value[0] == '\0') {
        return 0;
    }
    return strcmp(value, "1") == 0 ||
           strcasecmp(value, "true") == 0 ||
           strcasecmp(value, "yes") == 0 ||
           strcasecmp(value, "on") == 0 ||
           strcasecmp(value, "fast") == 0;
}

static int env_flag_disabled(const char *name) {
    const char *value = getenv(name);
    if (!value || value[0] == '\0') {
        return 0;
    }
    return strcmp(value, "0") == 0 ||
           strcasecmp(value, "false") == 0 ||
           strcasecmp(value, "no") == 0 ||
           strcasecmp(value, "off") == 0 ||
           strcasecmp(value, "slow") == 0 ||
           strcasecmp(value, "scalar") == 0 ||
           strcasecmp(value, "disabled") == 0;
}

static uint64_t fnv1a64_update(uint64_t hash, const uint8_t *data, uint64_t length) {
    for (uint64_t i = 0; i < length; i++) {
        hash ^= (uint64_t)data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t sampled_fnv1a64(const void *data, uint64_t length) {
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t hash = 1469598103934665603ULL;
    hash = fnv1a64_update(hash, (const uint8_t *)&length, sizeof(length));
    if (length <= 16384) {
        return fnv1a64_update(hash, bytes, length);
    }
    hash = fnv1a64_update(hash, bytes, 4096);
    hash = fnv1a64_update(hash, bytes + (length / 2) - 2048, 4096);
    hash = fnv1a64_update(hash, bytes + length - 4096, 4096);
    return hash;
}

static int pread_exact_or_report(int fd,
                                 void *dst,
                                 uint64_t bytes,
                                 uint64_t offset,
                                 const char *path) {
    uint8_t *cursor = (uint8_t *)dst;
    uint64_t done = 0;
    while (done < bytes) {
        uint64_t remaining = bytes - done;
        size_t chunk = remaining > (uint64_t)SSIZE_MAX ? (size_t)SSIZE_MAX : (size_t)remaining;
        ssize_t n = pread(fd, cursor + done, chunk, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            fprintf(stderr, "ERROR: pread %s failed: %s\n", path, strerror(errno));
            return 0;
        }
        if (n == 0) {
            fprintf(stderr,
                    "ERROR: short pread %s at offset %llu after %llu of %llu bytes\n",
                    path,
                    (unsigned long long)(offset + done),
                    (unsigned long long)done,
                    (unsigned long long)bytes);
            return 0;
        }
        done += (uint64_t)n;
    }
    return 1;
}

typedef struct {
    int fd;
    void *dst;
    uint64_t bytes;
    uint64_t offset;
    const char *path;
    int ok;
} ParallelPreadTask;

typedef struct {
    int task_count;
    int worker_count;
    int used_pool;
    int used_serial_fallback;
} ParallelPreadDispatchStats;

typedef struct {
    uint64_t bytes_read;
    double read_seconds;
    uint32_t dispatch_count;
    uint32_t task_count;
    uint32_t max_task_count;
    uint32_t max_worker_count;
    uint32_t pool_dispatch_count;
    uint32_t serial_dispatch_count;
} DirectExpertReadStats;

typedef struct {
    int initialized;
    int worker_count;
    pthread_t workers[MAX_EXPERT_BUFFERS];
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    ParallelPreadTask *tasks;
    int task_count;
    int next_task;
    int completed;
    uint64_t generation;
} ParallelPreadPool;

static ParallelPreadPool g_parallel_pread_pool = {0};

static void *parallel_pread_thread_main(void *arg) {
    ParallelPreadTask *task = (ParallelPreadTask *)arg;
    task->ok = pread_exact_or_report(
        task->fd,
        task->dst,
        task->bytes,
        task->offset,
        task->path
    );
    return NULL;
}

static void parallel_pread_pool_init(void) {
    if (g_parallel_pread_pool.initialized) {
        return;
    }
    pthread_mutex_init(&g_parallel_pread_pool.mutex, NULL);
    pthread_cond_init(&g_parallel_pread_pool.work_ready, NULL);
    pthread_cond_init(&g_parallel_pread_pool.work_done, NULL);
    g_parallel_pread_pool.initialized = 1;
}

static void *parallel_pread_pool_worker_main(void *arg) {
    (void)arg;
    uint64_t seen_generation = 0;
    for (;;) {
        pthread_mutex_lock(&g_parallel_pread_pool.mutex);
        while (g_parallel_pread_pool.generation == seen_generation) {
            pthread_cond_wait(&g_parallel_pread_pool.work_ready,
                              &g_parallel_pread_pool.mutex);
        }
        seen_generation = g_parallel_pread_pool.generation;
        for (;;) {
            int task_index = g_parallel_pread_pool.next_task++;
            if (task_index >= g_parallel_pread_pool.task_count) {
                break;
            }
            ParallelPreadTask *task = &g_parallel_pread_pool.tasks[task_index];
            pthread_mutex_unlock(&g_parallel_pread_pool.mutex);
            parallel_pread_thread_main(task);
            pthread_mutex_lock(&g_parallel_pread_pool.mutex);
            g_parallel_pread_pool.completed++;
            if (g_parallel_pread_pool.completed >= g_parallel_pread_pool.task_count) {
                pthread_cond_signal(&g_parallel_pread_pool.work_done);
            }
        }
        pthread_mutex_unlock(&g_parallel_pread_pool.mutex);
    }
    return NULL;
}

static int parallel_pread_pool_ensure_workers(int desired) {
    parallel_pread_pool_init();
    if (desired > MAX_EXPERT_BUFFERS) {
        desired = MAX_EXPERT_BUFFERS;
    }
    pthread_mutex_lock(&g_parallel_pread_pool.mutex);
    while (g_parallel_pread_pool.worker_count < desired) {
        int index = g_parallel_pread_pool.worker_count;
        int err = pthread_create(&g_parallel_pread_pool.workers[index],
                                 NULL,
                                 parallel_pread_pool_worker_main,
                                 NULL);
        if (err != 0) {
            fprintf(stderr,
                    "WARNING: pthread_create for persistent pread worker failed: %s\n",
                    strerror(err));
            break;
        }
        pthread_detach(g_parallel_pread_pool.workers[index]);
        g_parallel_pread_pool.worker_count++;
    }
    int worker_count = g_parallel_pread_pool.worker_count;
    pthread_mutex_unlock(&g_parallel_pread_pool.mutex);
    return worker_count;
}

static int parallel_pread_exact_or_report(ParallelPreadTask *tasks,
                                          int count,
                                          ParallelPreadDispatchStats *dispatchStats) {
    if (dispatchStats) {
        memset(dispatchStats, 0, sizeof(*dispatchStats));
        dispatchStats->task_count = count;
    }
    if (count <= 0) {
        return 1;
    }
    if (count > MAX_EXPERT_BUFFERS) {
        fprintf(stderr,
                "ERROR: parallel pread task count %d exceeds max %d\n",
                count,
                MAX_EXPERT_BUFFERS);
        return 0;
    }
    if (count == 1) {
        if (dispatchStats) {
            dispatchStats->used_serial_fallback = 1;
        }
        return pread_exact_or_report(
            tasks[0].fd,
            tasks[0].dst,
            tasks[0].bytes,
            tasks[0].offset,
            tasks[0].path
        );
    }
    int worker_count = parallel_pread_pool_ensure_workers(count);
    if (dispatchStats) {
        dispatchStats->worker_count = worker_count;
    }
    if (worker_count <= 0) {
        if (dispatchStats) {
            dispatchStats->used_serial_fallback = 1;
        }
        int fallback_ok = 1;
        for (int i = 0; i < count; i++) {
            parallel_pread_thread_main(&tasks[i]);
            if (!tasks[i].ok) {
                fallback_ok = 0;
            }
        }
        return fallback_ok;
    }
    if (dispatchStats) {
        dispatchStats->used_pool = 1;
    }
    for (int i = 0; i < count; i++) {
        tasks[i].ok = 0;
    }
    pthread_mutex_lock(&g_parallel_pread_pool.mutex);
    g_parallel_pread_pool.tasks = tasks;
    g_parallel_pread_pool.task_count = count;
    g_parallel_pread_pool.next_task = 0;
    g_parallel_pread_pool.completed = 0;
    g_parallel_pread_pool.generation++;
    pthread_cond_broadcast(&g_parallel_pread_pool.work_ready);
    while (g_parallel_pread_pool.completed < count) {
        pthread_cond_wait(&g_parallel_pread_pool.work_done,
                          &g_parallel_pread_pool.mutex);
    }
    g_parallel_pread_pool.tasks = NULL;
    g_parallel_pread_pool.task_count = 0;
    pthread_mutex_unlock(&g_parallel_pread_pool.mutex);
    int ok = 1;
    for (int i = 0; i < count; i++) {
        if (!tasks[i].ok) {
            ok = 0;
        }
    }
    return ok;
}

static void direct_expert_read_stats_add_dispatch(
    DirectExpertReadStats *stats,
    ParallelPreadDispatchStats dispatchStats,
    uint64_t bytesRead,
    double readSeconds
) {
    if (!stats) {
        return;
    }
    stats->bytes_read += bytesRead;
    stats->read_seconds += readSeconds;
    stats->dispatch_count++;
    stats->task_count += (uint32_t)dispatchStats.task_count;
    if ((uint32_t)dispatchStats.task_count > stats->max_task_count) {
        stats->max_task_count = (uint32_t)dispatchStats.task_count;
    }
    if ((uint32_t)dispatchStats.worker_count > stats->max_worker_count) {
        stats->max_worker_count = (uint32_t)dispatchStats.worker_count;
    }
    if (dispatchStats.used_pool) {
        stats->pool_dispatch_count++;
    }
    if (dispatchStats.used_serial_fallback) {
        stats->serial_dispatch_count++;
    }
}

static int read_direct_expert_slots(ExpertFile *expertFile,
                                    IntList experts,
                                    int expertOffset,
                                    int count,
                                    NSArray *expertBuffers,
                                    DirectExpertReadStats *stats) {
    if (count <= 0) {
        return 1;
    }
    if (!expertFile || expertFile->fd < 0 || expertFile->expert_slot_bytes == 0 ||
        !expertBuffers || expertOffset < 0 || expertOffset + count > experts.count) {
        fprintf(stderr, "ERROR: invalid direct expert read request\n");
        return 0;
    }
    if (count > MAX_EXPERT_BUFFERS || count > (int)[expertBuffers count]) {
        fprintf(stderr,
                "ERROR: direct expert read count %d exceeds available buffers\n",
                count);
        return 0;
    }
    if (expertFile->expert_slot_bytes > (uint64_t)NSUIntegerMax) {
        fprintf(stderr, "ERROR: expert slot bytes exceed NSUIntegerMax\n");
        return 0;
    }
    ParallelPreadTask tasks[MAX_EXPERT_BUFFERS];
    for (int i = 0; i < count; i++) {
        int expertId = experts.values[expertOffset + i];
        if (expertId < 0 || (uint64_t)expertId >= expertFile->num_experts) {
            fprintf(stderr,
                    "ERROR: expert id %d is outside layer %d range 0..%llu\n",
                    expertId,
                    expertFile->layer,
                    (unsigned long long)(expertFile->num_experts - 1));
            return 0;
        }
        id<MTLBuffer> buffer = [expertBuffers objectAtIndex:(NSUInteger)i];
        if ((uint64_t)[buffer length] < expertFile->expert_slot_bytes ||
            ![buffer contents]) {
            fprintf(stderr,
                    "ERROR: expert Metal buffer is too small or not CPU-visible\n");
            return 0;
        }
        tasks[i] = (ParallelPreadTask){
            .fd = expertFile->fd,
            .dst = [buffer contents],
            .bytes = expertFile->expert_slot_bytes,
            .offset = (uint64_t)expertId * expertFile->expert_slot_bytes,
            .path = expertFile->path,
            .ok = 0,
        };
    }
    double readStarted = now_seconds();
    ParallelPreadDispatchStats dispatchStats = {0};
    int readOk = parallel_pread_exact_or_report(tasks, count, &dispatchStats);
    double readSeconds = now_seconds() - readStarted;
    if (!readOk) {
        return 0;
    }
    uint64_t bytesRead = 0;
    for (int i = 0; i < count; i++) {
        id<MTLBuffer> buffer = [expertBuffers objectAtIndex:(NSUInteger)i];
        [buffer didModifyRange:NSMakeRange(0, (NSUInteger)tasks[i].bytes)];
        bytesRead += tasks[i].bytes;
    }
    direct_expert_read_stats_add_dispatch(
        stats,
        dispatchStats,
        bytesRead,
        readSeconds
    );
    return 1;
}

static NSString *hex_u64(uint64_t value) {
    return [NSString stringWithFormat:@"0x%016llx", (unsigned long long)value];
}

static float bf16_to_float_cpu(uint16_t value) {
    uint32_t bits = ((uint32_t)value) << 16;
    float out = 0.0f;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t f32_to_bf16_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)((rounded >> 16) & 0xFFFFu);
}

static float mxfp4_e2m1_to_f32_cpu(uint32_t code) {
    static const float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return table[code & 0xFu];
}

static float mxfp4_e8m0_to_f32_cpu(uint8_t value) {
    uint32_t bits = ((uint32_t)value) << 23;
    float out = 0.0f;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static int checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int parse_json_u64_field(NSDictionary *dict,
                                NSString *key,
                                const char *label,
                                int positive,
                                uint64_t *out) {
    id value = dict[key];
    if (![value isKindOfClass:[NSNumber class]]) {
        fprintf(stderr, "ERROR: %s must be a number\n", label);
        return 0;
    }
    long long signedValue = [(NSNumber *)value longLongValue];
    if (signedValue < 0 || (positive && signedValue == 0)) {
        fprintf(stderr, "ERROR: %s must be %s\n",
                label,
                positive ? "positive" : "non-negative");
        return 0;
    }
    *out = (uint64_t)signedValue;
    return 1;
}

static int cache_dtype_bytes(const char *dtype) {
    if (strcmp(dtype, "F32") == 0 ||
        strcmp(dtype, "float32") == 0 ||
        strcmp(dtype, "FLOAT32") == 0) {
        return 4;
    }
    if (strcmp(dtype, "BF16") == 0 ||
        strcmp(dtype, "bfloat16") == 0 ||
        strcmp(dtype, "BFLOAT16") == 0) {
        return 2;
    }
    return 0;
}

static int write_exact_at_path(NSString *path,
                               const void *src,
                               size_t size,
                               off_t offset) {
    NSMutableData *memory = shared_decode_cache_memory_for_path(path);
    if (memory) {
        if (offset < 0 ||
            (uint64_t)offset > (uint64_t)[memory length] ||
            (uint64_t)size > (uint64_t)[memory length] - (uint64_t)offset) {
            fprintf(stderr, "ERROR: in-memory decode cache write is out of bounds\n");
            return 0;
        }
        memcpy((uint8_t *)[memory mutableBytes] + (uint64_t)offset, src, size);
        return 1;
    }
    int closeFd = 0;
    int fd = open_decode_cache_fd(path, 1, &closeFd);
    if (fd < 0) {
        fprintf(stderr, "ERROR: failed to open cache file %s: %s\n",
                [path UTF8String],
                strerror(errno));
        return 0;
    }
    size_t copied = 0;
    while (copied < size) {
        size_t toWrite = size - copied;
        if (toWrite > 8 * 1024 * 1024) {
            toWrite = 8 * 1024 * 1024;
        }
        ssize_t written = pwrite(fd,
                                 (const char *)src + copied,
                                 toWrite,
                                 offset + (off_t)copied);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            fprintf(stderr,
                    "ERROR: pwrite cache file %s failed at %zu/%zu: %s\n",
                    [path UTF8String],
                    copied,
                    size,
                    written == 0 ? "short write" : strerror(errno));
            close_decode_cache_fd(fd, closeFd);
            return 0;
        }
        copied += (size_t)written;
    }
    close_decode_cache_fd(fd, closeFd);
    return 1;
}

static int validate_decode_cache_file_size(NSString *cacheFilePath,
                                           uint64_t expectedBytes) {
    NSMutableData *memory = shared_decode_cache_memory_for_path(cacheFilePath);
    if (memory) {
        if ((uint64_t)[memory length] != expectedBytes) {
            fprintf(stderr,
                    "ERROR: in-memory decode cache size %llu does not match layout total_bytes %llu\n",
                    (unsigned long long)[memory length],
                    (unsigned long long)expectedBytes);
            return 0;
        }
        return 1;
    }
    struct stat st;
    if (stat([cacheFilePath fileSystemRepresentation], &st) != 0) {
        fprintf(stderr, "ERROR: failed to stat decode cache file %s: %s\n",
                [cacheFilePath UTF8String],
                strerror(errno));
        return 0;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size != expectedBytes) {
        fprintf(stderr,
                "ERROR: decode cache file size %llu does not match layout total_bytes %llu\n",
                (unsigned long long)(st.st_size < 0 ? 0 : (uint64_t)st.st_size),
                (unsigned long long)expectedBytes);
        return 0;
    }
    return 1;
}

static int parse_decode_cache_segment_info(NSDictionary *segment,
                                           uint64_t layoutTotalBytes,
                                           NSString *layoutDtype,
                                           uint64_t layoutDtypeBytes,
                                           DecodeCacheSegmentInfo *out) {
    NSString *kind = segment[@"kind"];
    NSString *dtype = segment[@"dtype"];
    if (![kind isKindOfClass:[NSString class]] ||
        ![dtype isKindOfClass:[NSString class]]) {
        fprintf(stderr, "ERROR: cache segment missing kind or dtype\n");
        return 0;
    }
    if (!([kind isEqualToString:@"mla_kv"] ||
          [kind isEqualToString:@"dsa_index"])) {
        fprintf(stderr, "ERROR: unsupported cache segment kind %s\n", [kind UTF8String]);
        return 0;
    }
    if (![dtype isEqualToString:layoutDtype]) {
        fprintf(stderr, "ERROR: cache segment dtype does not match layout dtype\n");
        return 0;
    }

    uint64_t layer = 0;
    uint64_t offset = 0;
    uint64_t width = 0;
    uint64_t dtypeBytes = 0;
    uint64_t tokenStrideBytes = 0;
    uint64_t maxContextTokens = 0;
    uint64_t totalBytes = 0;
    if (!parse_json_u64_field(segment, @"layer", "cache segment layer", 0, &layer) ||
        !parse_json_u64_field(segment, @"offset", "cache segment offset", 0, &offset) ||
        !parse_json_u64_field(segment, @"width", "cache segment width", 1, &width) ||
        !parse_json_u64_field(segment, @"dtype_bytes", "cache segment dtype_bytes", 1, &dtypeBytes) ||
        !parse_json_u64_field(segment, @"token_stride_bytes", "cache segment token_stride_bytes", 1, &tokenStrideBytes) ||
        !parse_json_u64_field(segment, @"max_context_tokens", "cache segment max_context_tokens", 1, &maxContextTokens) ||
        !parse_json_u64_field(segment, @"total_bytes", "cache segment total_bytes", 1, &totalBytes)) {
        return 0;
    }
    int expectedDtypeBytes = cache_dtype_bytes([dtype UTF8String]);
    if (expectedDtypeBytes == 0 ||
        dtypeBytes != (uint64_t)expectedDtypeBytes ||
        dtypeBytes != layoutDtypeBytes) {
        fprintf(stderr, "ERROR: cache segment dtype_bytes does not match dtype\n");
        return 0;
    }
    if (width > UINT32_MAX || dtypeBytes > UINT32_MAX) {
        fprintf(stderr, "ERROR: cache segment width or dtype_bytes is out of range\n");
        return 0;
    }
    uint64_t expectedStride = 0;
    if (!checked_mul_u64(width, dtypeBytes, &expectedStride) ||
        tokenStrideBytes != expectedStride) {
        fprintf(stderr, "ERROR: cache segment token_stride_bytes does not match width*dtype_bytes\n");
        return 0;
    }
    uint64_t expectedTotal = 0;
    if (!checked_mul_u64(tokenStrideBytes, maxContextTokens, &expectedTotal) ||
        totalBytes != expectedTotal) {
        fprintf(stderr, "ERROR: cache segment total_bytes does not match stride*max_context_tokens\n");
        return 0;
    }
    uint64_t segmentEnd = 0;
    if (!checked_add_u64(offset, totalBytes, &segmentEnd) ||
        segmentEnd > layoutTotalBytes) {
        fprintf(stderr, "ERROR: cache segment exceeds layout total_bytes\n");
        return 0;
    }

    memset(out, 0, sizeof(*out));
    out->layer = layer;
    out->offset = offset;
    out->total_bytes = totalBytes;
    out->max_context_tokens = maxContextTokens;
    out->width = (uint32_t)width;
    out->dtype_bytes = (uint32_t)dtypeBytes;
    snprintf(out->dtype, sizeof(out->dtype), "%s", [dtype UTF8String]);
    snprintf(out->kind, sizeof(out->kind), "%s", [kind UTF8String]);
    return 1;
}

static int validate_decode_cache_layout_dictionary(NSDictionary *cacheLayout,
                                                   uint64_t maxCacheFileBytes,
                                                   uint64_t *outTotalBytes) {
    uint64_t version = 0;
    uint64_t totalBytes = 0;
    uint64_t dtypeBytes = 0;
    uint64_t maxContextTokens = 0;
    uint64_t alignment = 0;
    if (!parse_json_u64_field(cacheLayout, @"version", "cache layout version", 1, &version) ||
        !parse_json_u64_field(cacheLayout, @"total_bytes", "cache layout total_bytes", 1, &totalBytes) ||
        !parse_json_u64_field(cacheLayout, @"dtype_bytes", "cache layout dtype_bytes", 1, &dtypeBytes) ||
        !parse_json_u64_field(cacheLayout, @"max_context_tokens", "cache layout max_context_tokens", 1, &maxContextTokens) ||
        !parse_json_u64_field(cacheLayout, @"alignment", "cache layout alignment", 1, &alignment)) {
        return 0;
    }
    (void)maxContextTokens;
    (void)alignment;
    if (version != 1) {
        fprintf(stderr, "ERROR: unsupported cache layout version %llu\n",
                (unsigned long long)version);
        return 0;
    }
    NSString *modelType = cacheLayout[@"model_type"];
    NSString *layoutDtype = cacheLayout[@"dtype"];
    if (![modelType isKindOfClass:[NSString class]] || [modelType length] == 0 ||
        ![layoutDtype isKindOfClass:[NSString class]]) {
        fprintf(stderr, "ERROR: cache layout missing model_type or dtype\n");
        return 0;
    }
    int expectedDtypeBytes = cache_dtype_bytes([layoutDtype UTF8String]);
    if (expectedDtypeBytes == 0 || dtypeBytes != (uint64_t)expectedDtypeBytes) {
        fprintf(stderr, "ERROR: cache layout dtype_bytes does not match dtype\n");
        return 0;
    }
    if (totalBytes > maxCacheFileBytes) {
        fprintf(stderr,
                "ERROR: cache layout total %llu bytes exceeds limit %llu\n",
                (unsigned long long)totalBytes,
                (unsigned long long)maxCacheFileBytes);
        return 0;
    }
    NSArray *segments = cacheLayout[@"segments"];
    if (![segments isKindOfClass:[NSArray class]] || [segments count] == 0) {
        fprintf(stderr, "ERROR: cache layout missing segments array\n");
        return 0;
    }
    NSMutableSet *seen = [NSMutableSet setWithCapacity:[segments count]];
    for (id item in segments) {
        if (![item isKindOfClass:[NSDictionary class]]) {
            fprintf(stderr, "ERROR: cache segment must be a JSON object\n");
            return 0;
        }
        DecodeCacheSegmentInfo segmentInfo = {0};
        if (!parse_decode_cache_segment_info((NSDictionary *)item,
                                             totalBytes,
                                             layoutDtype,
                                             dtypeBytes,
                                             &segmentInfo)) {
            return 0;
        }
        NSString *key = [NSString stringWithFormat:@"%s:%llu",
                         segmentInfo.kind,
                         (unsigned long long)segmentInfo.layer];
        if ([seen containsObject:key]) {
            fprintf(stderr, "ERROR: duplicate cache segment %s\n", [key UTF8String]);
            return 0;
        }
        [seen addObject:key];
    }
    if (outTotalBytes) {
        *outTotalBytes = totalBytes;
    }
    return 1;
}

static int find_decode_cache_segment_info(NSDictionary *cacheLayout,
                                          uint64_t layerId,
                                          NSString *kind,
                                          uint64_t maxCacheFileBytes,
                                          DecodeCacheSegmentInfo *out,
                                          uint64_t *outTotalBytes) {
    uint64_t totalBytes = 0;
    if (!validate_decode_cache_layout_dictionary(cacheLayout,
                                                 maxCacheFileBytes,
                                                 &totalBytes)) {
        return 0;
    }
    NSArray *segments = cacheLayout[@"segments"];
    NSString *layoutDtype = cacheLayout[@"dtype"];
    uint64_t layoutDtypeBytes = 0;
    if (!parse_json_u64_field(cacheLayout,
                              @"dtype_bytes",
                              "cache layout dtype_bytes",
                              1,
                              &layoutDtypeBytes)) {
        return 0;
    }
    for (id item in segments) {
        DecodeCacheSegmentInfo candidate = {0};
        if (!parse_decode_cache_segment_info((NSDictionary *)item,
                                             totalBytes,
                                             layoutDtype,
                                             layoutDtypeBytes,
                                             &candidate)) {
            return 0;
        }
        if (candidate.layer == layerId && strcmp(candidate.kind, [kind UTF8String]) == 0) {
            *out = candidate;
            if (outTotalBytes) {
                *outTotalBytes = totalBytes;
            }
            return 1;
        }
    }
    fprintf(stderr, "ERROR: cache segment %s for layer %llu not found\n",
            [kind UTF8String],
            (unsigned long long)layerId);
    return 0;
}

static int max_cache_file_bytes_from_mib(double mib, uint64_t *out) {
    if (!isfinite(mib) || mib <= 0.0) {
        fprintf(stderr, "ERROR: --max-cache-file-mib must be positive\n");
        return 0;
    }
    double bytes = mib * 1024.0 * 1024.0;
    if (bytes > (double)UINT64_MAX) {
        fprintf(stderr, "ERROR: --max-cache-file-mib is too large\n");
        return 0;
    }
    *out = (uint64_t)bytes;
    return 1;
}

static int positive_mib_to_bytes(double mib, const char *label, uint64_t *out) {
    if (!isfinite(mib) || mib <= 0.0) {
        fprintf(stderr, "ERROR: %s must be positive\n", label);
        return 0;
    }
    double bytes = mib * 1024.0 * 1024.0;
    if (bytes > (double)UINT64_MAX) {
        fprintf(stderr, "ERROR: %s is too large\n", label);
        return 0;
    }
    *out = (uint64_t)bytes;
    return 1;
}

static int parse_context1_o_proj_cache_matrix_info(NSString *layoutPath,
                                                   NSString *cacheFileOverridePath,
                                                   uint64_t layerId,
                                                   uint64_t maxCacheReadBytes,
                                                   Context1OProjCacheMatrixInfo *out,
                                                   NSString **outCacheFilePath) {
    NSDictionary *layout = load_json_dictionary(layoutPath);
    NSString *schema = layout[@"schema"];
    if (![schema isKindOfClass:[NSString class]] ||
        ![schema isEqualToString:@"largerlm.context1_o_proj_bv_cache.v1"]) {
        fprintf(stderr, "ERROR: context1 o_proj cache schema mismatch\n");
        return 0;
    }
    uint64_t version = 0;
    uint64_t totalBytes = 0;
    uint64_t dtypeBytes = 0;
    if (!parse_json_u64_field(layout, @"version", "context1 cache version", 1, &version) ||
        !parse_json_u64_field(layout, @"total_bytes", "context1 cache total_bytes", 0, &totalBytes) ||
        !parse_json_u64_field(layout, @"dtype_bytes", "context1 cache dtype_bytes", 1, &dtypeBytes)) {
        return 0;
    }
    if (version != 1) {
        fprintf(stderr, "ERROR: unsupported context1 cache version %llu\n",
                (unsigned long long)version);
        return 0;
    }
    NSString *dtype = layout[@"dtype"];
    if (![dtype isKindOfClass:[NSString class]]) {
        fprintf(stderr, "ERROR: context1 cache dtype must be a string\n");
        return 0;
    }
    int expectedDtypeBytes = cache_dtype_bytes([dtype UTF8String]);
    if ((expectedDtypeBytes != 2 && expectedDtypeBytes != 4) ||
        dtypeBytes != (uint64_t)expectedDtypeBytes) {
        fprintf(stderr, "ERROR: context1 cache dtype_bytes does not match dtype\n");
        return 0;
    }
    NSDictionary *dims = layout[@"dims"];
    if (![dims isKindOfClass:[NSDictionary class]]) {
        fprintf(stderr, "ERROR: context1 cache dims must be an object\n");
        return 0;
    }
    uint64_t hiddenDim = 0;
    uint64_t kvLoraDim = 0;
    if (!parse_json_u64_field(dims, @"hidden_dim", "context1 cache hidden_dim", 1, &hiddenDim) ||
        !parse_json_u64_field(dims, @"kv_lora_dim", "context1 cache kv_lora_dim", 1, &kvLoraDim)) {
        return 0;
    }
    if (hiddenDim > UINT32_MAX || kvLoraDim > UINT32_MAX) {
        fprintf(stderr, "ERROR: context1 cache dims exceed uint32\n");
        return 0;
    }
    uint64_t perLayerBytes = 0;
    if (!checked_mul_u64(hiddenDim, kvLoraDim, &perLayerBytes) ||
        !checked_mul_u64(perLayerBytes, dtypeBytes, &perLayerBytes)) {
        fprintf(stderr, "ERROR: context1 cache per-layer bytes overflow\n");
        return 0;
    }
    if (maxCacheReadBytes > 0 && perLayerBytes > maxCacheReadBytes) {
        fprintf(stderr,
                "ERROR: context1 cache layer bytes %llu exceed --max-cache-read-mib limit %llu\n",
                (unsigned long long)perLayerBytes,
                (unsigned long long)maxCacheReadBytes);
        return 0;
    }
    NSString *weightFile = layout[@"weight_file"];
    if (![weightFile isKindOfClass:[NSString class]] || [weightFile length] == 0) {
        fprintf(stderr, "ERROR: context1 cache weight_file must be a string\n");
        return 0;
    }
    if ([weightFile isAbsolutePath] ||
        [[weightFile pathComponents] containsObject:@".."]) {
        fprintf(stderr, "ERROR: context1 cache weight_file must be relative\n");
        return 0;
    }
    NSString *cacheFilePath =
        [[layoutPath stringByDeletingLastPathComponent] stringByAppendingPathComponent:weightFile];
    if (cacheFileOverridePath) {
        cacheFilePath = cacheFileOverridePath;
    }
    struct stat st;
    if (stat([cacheFilePath fileSystemRepresentation], &st) != 0) {
        fprintf(stderr, "ERROR: failed to stat context1 cache file %s: %s\n",
                [cacheFilePath UTF8String],
                strerror(errno));
        return 0;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size < totalBytes) {
        fprintf(stderr,
                "ERROR: context1 cache file is smaller than total_bytes: %llu < %llu\n",
                (unsigned long long)(st.st_size < 0 ? 0 : (uint64_t)st.st_size),
                (unsigned long long)totalBytes);
        return 0;
    }

    NSArray *tensors = layout[@"tensors"];
    if (![tensors isKindOfClass:[NSArray class]] || [tensors count] == 0) {
        fprintf(stderr, "ERROR: context1 cache tensors must be non-empty\n");
        return 0;
    }
    uint64_t expectedTotal = 0;
    if (!checked_mul_u64(perLayerBytes, (uint64_t)[tensors count], &expectedTotal) ||
        totalBytes != expectedTotal) {
        fprintf(stderr, "ERROR: context1 cache total_bytes does not match tensor count\n");
        return 0;
    }

    uint64_t previousLayer = 0;
    uint64_t previousEnd = 0;
    int havePrevious = 0;
    int found = 0;
    for (id item in tensors) {
        if (![item isKindOfClass:[NSDictionary class]]) {
            fprintf(stderr, "ERROR: context1 cache tensor must be an object\n");
            return 0;
        }
        NSDictionary *tensor = (NSDictionary *)item;
        uint64_t layer = 0;
        uint64_t offset = 0;
        uint64_t size = 0;
        if (!parse_json_u64_field(tensor, @"layer", "context1 tensor layer", 0, &layer) ||
            !parse_json_u64_field(tensor, @"offset", "context1 tensor offset", 0, &offset) ||
            !parse_json_u64_field(tensor, @"size", "context1 tensor size", 1, &size)) {
            return 0;
        }
        if (havePrevious && layer <= previousLayer) {
            fprintf(stderr, "ERROR: context1 cache tensors must be sorted by numeric layer\n");
            return 0;
        }
        havePrevious = 1;
        previousLayer = layer;
        uint64_t end = 0;
        if (!checked_add_u64(offset, size, &end) || end > totalBytes) {
            fprintf(stderr, "ERROR: context1 cache tensor span exceeds total_bytes\n");
            return 0;
        }
        if (offset < previousEnd) {
            fprintf(stderr, "ERROR: context1 cache tensor spans overlap\n");
            return 0;
        }
        previousEnd = end;
        if (size != perLayerBytes) {
            fprintf(stderr, "ERROR: context1 cache tensor size does not match dims*dtype\n");
            return 0;
        }
        NSString *tensorDtype = tensor[@"dtype"];
        NSString *category = tensor[@"category"];
        NSString *name = tensor[@"name"];
        NSArray *shape = tensor[@"shape"];
        NSString *expectedName =
            [NSString stringWithFormat:@"model.layers.%llu.self_attn.context1_o_proj_bv.weight",
                                       (unsigned long long)layer];
        if (![tensorDtype isKindOfClass:[NSString class]] ||
            ![tensorDtype isEqualToString:dtype] ||
            ![category isKindOfClass:[NSString class]] ||
            ![category isEqualToString:@"context1_attention_output"] ||
            ![name isKindOfClass:[NSString class]] ||
            ![name isEqualToString:expectedName] ||
            ![shape isKindOfClass:[NSArray class]] ||
            [shape count] != 2 ||
            ![shape[0] isKindOfClass:[NSNumber class]] ||
            ![shape[1] isKindOfClass:[NSNumber class]] ||
            [(NSNumber *)shape[0] unsignedLongLongValue] != hiddenDim ||
            [(NSNumber *)shape[1] unsignedLongLongValue] != kvLoraDim) {
            fprintf(stderr, "ERROR: context1 cache tensor metadata is invalid\n");
            return 0;
        }
        if (layer == layerId) {
            memset(out, 0, sizeof(*out));
            out->offset = offset;
            out->size = size;
            out->total_bytes = totalBytes;
            out->out_dim = (uint32_t)hiddenDim;
            out->in_dim = (uint32_t)kvLoraDim;
            out->dtype_bytes = (uint32_t)dtypeBytes;
            snprintf(out->dtype, sizeof(out->dtype), "%s", [dtype UTF8String]);
            snprintf(out->name, sizeof(out->name), "%s", [name UTF8String]);
            found = 1;
        }
    }
    if (!found) {
        fprintf(stderr, "ERROR: context1 cache tensor for layer %llu not found\n",
                (unsigned long long)layerId);
        return 0;
    }
    if (outCacheFilePath) {
        *outCacheFilePath = cacheFilePath;
    }
    return 1;
}

static int decode_cache_layout_total_bytes(NSString *cacheLayoutPath,
                                           uint64_t maxCacheFileBytes,
                                           uint64_t *outTotalBytes) {
    NSDictionary *cacheLayout = load_json_dictionary(cacheLayoutPath);
    if (!cacheLayout ||
        !validate_decode_cache_layout_dictionary(cacheLayout,
                                                 maxCacheFileBytes,
                                                 outTotalBytes)) {
        return 0;
    }
    return 1;
}

static int read_file_exact_into_mutable_data(NSString *path, NSMutableData *data) {
    int fd = open([path fileSystemRepresentation], O_RDONLY);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open decode cache file %s: %s\n",
                [path UTF8String],
                strerror(errno));
        return 0;
    }
    uint8_t *dst = (uint8_t *)[data mutableBytes];
    uint64_t total = (uint64_t)[data length];
    uint64_t copied = 0;
    while (copied < total) {
        uint64_t toRead = total - copied;
        if (toRead > 8 * 1024 * 1024) {
            toRead = 8 * 1024 * 1024;
        }
        ssize_t got = pread(fd, dst + copied, (size_t)toRead, (off_t)copied);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            fprintf(stderr,
                    "ERROR: failed to read decode cache file %s at %llu/%llu: %s\n",
                    [path UTF8String],
                    (unsigned long long)copied,
                    (unsigned long long)total,
                    got == 0 ? "short read" : strerror(errno));
            close(fd);
            return 0;
        }
        copied += (uint64_t)got;
    }
    close(fd);
    return 1;
}

static int ensure_runtime_decode_cache_memory(GlmMoeRuntimeContext *runtime,
                                              NSString *cacheLayoutPath,
                                              NSString *cacheFilePath,
                                              uint64_t maxCacheFileBytes) {
    if (!runtime || !cacheFilePath) {
        return 1;
    }
    uint64_t totalBytes = 0;
    if (!decode_cache_layout_total_bytes(cacheLayoutPath,
                                         maxCacheFileBytes,
                                         &totalBytes) ||
        !validate_decode_cache_file_size(cacheFilePath, totalBytes)) {
        return 0;
    }
    if (totalBytes > (uint64_t)NSUIntegerMax) {
        fprintf(stderr, "ERROR: decode cache is too large for NSMutableData\n");
        return 0;
    }
    if (runtime.decodeCacheMemory &&
        runtime.decodeCacheMemoryPath &&
        [runtime.decodeCacheMemoryPath isEqualToString:cacheFilePath] &&
        (uint64_t)[runtime.decodeCacheMemory length] == totalBytes) {
        register_shared_decode_cache_memory(runtime.decodeCacheMemoryPath,
                                            runtime.decodeCacheMemory);
        return 1;
    }
    if (runtime.decodeCacheFd >= 0) {
        clear_shared_decode_cache_file_if_fd(runtime.decodeCacheFd);
        close(runtime.decodeCacheFd);
        runtime.decodeCacheFd = -1;
        runtime.decodeCachePath = nil;
    }
    NSMutableData *data = [NSMutableData dataWithLength:(NSUInteger)totalBytes];
    if (!data) {
        fprintf(stderr, "ERROR: failed to allocate in-memory decode cache\n");
        return 0;
    }
    if (!read_file_exact_into_mutable_data(cacheFilePath, data)) {
        return 0;
    }
    runtime.decodeCacheMemory = data;
    runtime.decodeCacheMemoryPath = cacheFilePath;
    runtime.decodeCacheMemoryLoadCount++;
    register_shared_decode_cache_memory(runtime.decodeCacheMemoryPath,
                                        runtime.decodeCacheMemory);
    return 1;
}

static int append_kv_a_data_to_decode_cache(NSString *cacheLayoutPath,
                                            NSString *cacheFilePath,
                                            uint64_t layerId,
                                            uint64_t position,
                                            NSData *kvAData,
                                            uint32_t kvAOutDim,
                                            uint64_t maxCacheFileBytes,
                                            uint64_t *outWrittenBytes) {
    if (outWrittenBytes) {
        *outWrittenBytes = 0;
    }
    NSDictionary *cacheLayout = load_json_dictionary(cacheLayoutPath);
    uint64_t totalBytes = 0;
    DecodeCacheSegmentInfo segment = {0};
    if (!find_decode_cache_segment_info(cacheLayout,
                                        layerId,
                                        @"mla_kv",
                                        maxCacheFileBytes,
                                        &segment,
                                        &totalBytes) ||
        !validate_decode_cache_file_size(cacheFilePath, totalBytes)) {
        return 0;
    }
    if (position >= segment.max_context_tokens) {
        fprintf(stderr,
                "ERROR: cache position %llu exceeds max context %llu\n",
                (unsigned long long)position,
                (unsigned long long)segment.max_context_tokens);
        return 0;
    }
    if (segment.width != kvAOutDim) {
        fprintf(stderr,
                "ERROR: cache width %u does not match KV-A output dim %u\n",
                segment.width,
                kvAOutDim);
        return 0;
    }
    uint64_t expectedKvABytes = (uint64_t)kvAOutDim * sizeof(float);
    if ((uint64_t)[kvAData length] != expectedKvABytes) {
        fprintf(stderr,
                "ERROR: KV-A data bytes %llu do not match expected %llu\n",
                (unsigned long long)[kvAData length],
                (unsigned long long)expectedKvABytes);
        return 0;
    }
    uint64_t tokenBytes = (uint64_t)segment.width * (uint64_t)segment.dtype_bytes;
    uint64_t positionBytes = 0;
    uint64_t writeOffset = 0;
    uint64_t writeEnd = 0;
    if (!checked_mul_u64(position, tokenBytes, &positionBytes) ||
        !checked_add_u64(segment.offset, positionBytes, &writeOffset) ||
        !checked_add_u64(writeOffset, tokenBytes, &writeEnd) ||
        writeEnd > totalBytes ||
        tokenBytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "ERROR: cache write offset is out of range\n");
        return 0;
    }
    void *encoded = malloc((size_t)tokenBytes);
    if (!encoded) {
        fprintf(stderr, "ERROR: failed to allocate cache append staging buffer\n");
        return 0;
    }
    const float *src = (const float *)[kvAData bytes];
    if (segment.dtype_bytes == 4) {
        memcpy(encoded, src, (size_t)tokenBytes);
    } else if (segment.dtype_bytes == 2) {
        uint16_t *dst = (uint16_t *)encoded;
        for (uint32_t i = 0; i < segment.width; i++) {
            dst[i] = f32_to_bf16_bits(src[i]);
        }
    } else {
        fprintf(stderr, "ERROR: cache append supports only BF16 or F32\n");
        free(encoded);
        return 0;
    }
    int ok = write_exact_at_path(cacheFilePath,
                                 encoded,
                                 (size_t)tokenBytes,
                                 (off_t)writeOffset);
    free(encoded);
    if (!ok) {
        return 0;
    }
    if (outWrittenBytes) {
        *outWrittenBytes = tokenBytes;
    }
    return 1;
}

static int append_kv_a_buffer_to_decode_cache(NSString *cacheLayoutPath,
                                              NSString *cacheFilePath,
                                              uint64_t layerId,
                                              uint64_t position,
                                              id<MTLBuffer> kvABuffer,
                                              uint32_t kvAOutDim,
                                              uint64_t maxCacheFileBytes,
                                              uint64_t *outWrittenBytes) {
    uint64_t kvABytes = (uint64_t)kvAOutDim * sizeof(float);
    if (!kvABuffer || (uint64_t)[kvABuffer length] < kvABytes) {
        fprintf(stderr,
                "ERROR: KV-A Metal buffer bytes %llu are smaller than expected %llu\n",
                kvABuffer ? (unsigned long long)[kvABuffer length] : 0ull,
                (unsigned long long)kvABytes);
        return 0;
    }
    NSData *kvAData = [NSData dataWithBytes:[kvABuffer contents]
                                     length:(NSUInteger)kvABytes];
    return append_kv_a_data_to_decode_cache(cacheLayoutPath,
                                            cacheFilePath,
                                            layerId,
                                            position,
                                            kvAData,
                                            kvAOutDim,
                                            maxCacheFileBytes,
                                            outWrittenBytes);
}

static int read_decode_cache_segment_f32(NSString *cacheLayoutPath,
                                         NSString *cacheFilePath,
                                         uint64_t layerId,
                                         uint32_t contextLength,
                                         uint32_t expectedWidth,
                                         uint64_t maxCacheFileBytes,
                                         uint64_t maxCacheReadBytes,
                                         float **outValues,
                                         uint64_t *outBytes,
                                         uint64_t *outRawBytes) {
    *outValues = NULL;
    *outBytes = 0;
    if (outRawBytes) {
        *outRawBytes = 0;
    }
    NSDictionary *cacheLayout = load_json_dictionary(cacheLayoutPath);
    uint64_t totalBytes = 0;
    DecodeCacheSegmentInfo segment = {0};
    if (!find_decode_cache_segment_info(cacheLayout,
                                        layerId,
                                        @"mla_kv",
                                        maxCacheFileBytes,
                                        &segment,
                                        &totalBytes) ||
        !validate_decode_cache_file_size(cacheFilePath, totalBytes)) {
        return 0;
    }
    if (contextLength == 0 || contextLength > segment.max_context_tokens) {
        fprintf(stderr,
                "ERROR: context length %u exceeds max context %llu\n",
                contextLength,
                (unsigned long long)segment.max_context_tokens);
        return 0;
    }
    if (segment.width != expectedWidth) {
        fprintf(stderr,
                "ERROR: cache width %u does not match expected %u\n",
                segment.width,
                expectedWidth);
        return 0;
    }
    uint64_t tokenBytes = (uint64_t)segment.width * (uint64_t)segment.dtype_bytes;
    uint64_t rawBytes = 0;
    if (!checked_mul_u64(tokenBytes, (uint64_t)contextLength, &rawBytes) ||
        rawBytes > maxCacheReadBytes ||
        rawBytes > (uint64_t)SIZE_MAX ||
        segment.offset > totalBytes ||
        rawBytes > totalBytes - segment.offset) {
        fprintf(stderr,
                "ERROR: cache read %llu bytes exceeds bounds or limit %llu\n",
                (unsigned long long)rawBytes,
                (unsigned long long)maxCacheReadBytes);
        return 0;
    }
    uint64_t valueCount = 0;
    uint64_t f32Bytes = 0;
    if (!checked_mul_u64((uint64_t)contextLength,
                         (uint64_t)segment.width,
                         &valueCount) ||
        !checked_mul_u64(valueCount, sizeof(float), &f32Bytes) ||
        f32Bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "ERROR: cache f32 expansion byte size overflow\n");
        return 0;
    }
    float *values = (float *)calloc((size_t)valueCount, sizeof(float));
    void *raw = malloc((size_t)rawBytes);
    if (!values || !raw) {
        fprintf(stderr, "ERROR: failed to allocate cache read buffers\n");
        free(values);
        free(raw);
        return 0;
    }
    NSMutableData *memory = shared_decode_cache_memory_for_path(cacheFilePath);
    if (memory) {
        if (segment.offset > (uint64_t)[memory length] ||
            rawBytes > (uint64_t)[memory length] - segment.offset) {
            fprintf(stderr, "ERROR: in-memory decode cache read is out of bounds\n");
            free(values);
            free(raw);
            return 0;
        }
        memcpy(raw,
               (const uint8_t *)[memory bytes] + segment.offset,
               (size_t)rawBytes);
    } else {
        int closeFd = 0;
        int fd = open_decode_cache_fd(cacheFilePath, 0, &closeFd);
        if (fd < 0) {
            fprintf(stderr,
                    "ERROR: failed to open decode cache file %s: %s\n",
                    [cacheFilePath UTF8String],
                    strerror(errno));
            free(values);
            free(raw);
            return 0;
        }
        int readOk = pread_exact_or_report(fd,
                                           raw,
                                           rawBytes,
                                           segment.offset,
                                           [cacheFilePath UTF8String]);
        close_decode_cache_fd(fd, closeFd);
        if (!readOk) {
            free(values);
            free(raw);
            return 0;
        }
    }
    if (segment.dtype_bytes == 4) {
        memcpy(values, raw, (size_t)rawBytes);
    } else if (segment.dtype_bytes == 2) {
        uint16_t *src = (uint16_t *)raw;
        for (uint64_t i = 0; i < valueCount; i++) {
            values[i] = bf16_to_float_cpu(src[i]);
        }
    } else {
        fprintf(stderr, "ERROR: cache read supports only BF16 or F32\n");
        free(values);
        free(raw);
        return 0;
    }
    free(raw);
    *outValues = values;
    *outBytes = f32Bytes;
    if (outRawBytes) {
        *outRawBytes = rawBytes;
    }
    return 1;
}

static int read_decode_cache_prefix_with_current_slot_f32(
    NSString *cacheLayoutPath,
    NSString *cacheFilePath,
    uint64_t layerId,
    uint32_t contextLength,
    uint32_t expectedWidth,
    uint64_t maxCacheFileBytes,
    uint64_t maxCacheReadBytes,
    float **outValues,
    uint64_t *outBytes,
    uint64_t *outRawBytes,
    uint64_t *outCurrentRowOffsetBytes) {
    *outValues = NULL;
    *outBytes = 0;
    if (outRawBytes) {
        *outRawBytes = 0;
    }
    if (outCurrentRowOffsetBytes) {
        *outCurrentRowOffsetBytes = 0;
    }
    if (contextLength == 0) {
        fprintf(stderr, "ERROR: direct current KV-A cache view needs nonzero context\n");
        return 0;
    }
    uint64_t valueCount = 0;
    uint64_t f32Bytes = 0;
    uint64_t rowBytes = (uint64_t)expectedWidth * sizeof(float);
    if (!checked_mul_u64((uint64_t)contextLength,
                         (uint64_t)expectedWidth,
                         &valueCount) ||
        !checked_mul_u64(valueCount, sizeof(float), &f32Bytes) ||
        f32Bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "ERROR: direct current KV-A cache view byte size overflow\n");
        return 0;
    }
    float *values = (float *)calloc((size_t)valueCount, sizeof(float));
    if (!values) {
        fprintf(stderr, "ERROR: failed to allocate direct current KV-A cache view\n");
        return 0;
    }
    uint64_t prefixRawBytes = 0;
    if (contextLength > 1) {
        float *prefixValues = NULL;
        uint64_t prefixF32Bytes = 0;
        if (!read_decode_cache_segment_f32(cacheLayoutPath,
                                           cacheFilePath,
                                           layerId,
                                           contextLength - 1u,
                                           expectedWidth,
                                           maxCacheFileBytes,
                                           maxCacheReadBytes,
                                           &prefixValues,
                                           &prefixF32Bytes,
                                           &prefixRawBytes)) {
            free(values);
            return 0;
        }
        if (prefixF32Bytes > f32Bytes) {
            fprintf(stderr, "ERROR: direct current KV-A prefix exceeds cache view\n");
            free(prefixValues);
            free(values);
            return 0;
        }
        memcpy(values, prefixValues, (size_t)prefixF32Bytes);
        free(prefixValues);
    }
    if (outCurrentRowOffsetBytes) {
        *outCurrentRowOffsetBytes =
            (uint64_t)(contextLength - 1u) * rowBytes;
    }
    if (outRawBytes) {
        *outRawBytes = prefixRawBytes;
    }
    *outValues = values;
    *outBytes = f32Bytes;
    return 1;
}

static int find_layer_vector_info(NSDictionary *residentLayout,
                                  int layerId,
                                  NSString *suffix,
                                  ResidentVectorInfo *out) {
    memset(out, 0, sizeof(*out));
    NSArray *tensors = residentLayout[@"tensors"];
    if (![tensors isKindOfClass:[NSArray class]]) {
        fprintf(stderr, "ERROR: resident layout missing tensors array\n");
        return 0;
    }
    NSString *target = [NSString stringWithFormat:@"model.layers.%d%@", layerId, suffix];
    for (id item in tensors) {
        if (![item isKindOfClass:[NSDictionary class]]) {
            continue;
        }
        NSDictionary *tensor = (NSDictionary *)item;
        NSString *name = tensor[@"name"];
        if (![name isKindOfClass:[NSString class]] || ![name isEqualToString:target]) {
            continue;
        }
        NSString *dtype = tensor[@"dtype"];
        NSArray *shape = tensor[@"shape"];
        if (![dtype isKindOfClass:[NSString class]] ||
            ![shape isKindOfClass:[NSArray class]] ||
            [shape count] != 1) {
            fprintf(stderr, "ERROR: resident vector tensor has invalid dtype or shape\n");
            return 0;
        }
        uint64_t dim = unsigned_number(shape[0], "resident vector dim");
        uint64_t offset = unsigned_number(tensor[@"offset"], "resident vector offset");
        uint64_t size = unsigned_number(tensor[@"size"], "resident vector size");
        uint64_t elemBytes = 0;
        if ([dtype isEqualToString:@"F32"] || [dtype isEqualToString:@"float32"]) {
            elemBytes = sizeof(float);
        } else if ([dtype isEqualToString:@"BF16"] || [dtype isEqualToString:@"bfloat16"]) {
            elemBytes = sizeof(uint16_t);
        } else {
            fprintf(stderr, "ERROR: unsupported resident vector dtype %s\n", [dtype UTF8String]);
            return 0;
        }
        if (dim == 0 || dim > UINT32_MAX || size != dim * elemBytes) {
            fprintf(stderr, "ERROR: resident vector shape/size is invalid\n");
            return 0;
        }
        out->present = 1;
        out->offset = offset;
        out->size = size;
        out->dim = (uint32_t)dim;
        snprintf(out->dtype, sizeof(out->dtype), "%s", [dtype UTF8String]);
        snprintf(out->name, sizeof(out->name), "%s", [name UTF8String]);
        return 1;
    }
    fprintf(stderr, "ERROR: resident vector %s not found\n", [target UTF8String]);
    return 0;
}

static int find_global_vector_info_by_names(NSDictionary *residentLayout,
                                            NSArray *names,
                                            ResidentVectorInfo *out,
                                            const char *label) {
    memset(out, 0, sizeof(*out));
    NSArray *tensors = residentLayout[@"tensors"];
    if (![tensors isKindOfClass:[NSArray class]]) {
        fprintf(stderr, "ERROR: resident layout missing tensors array\n");
        return 0;
    }
    for (NSString *target in names) {
        for (id item in tensors) {
            if (![item isKindOfClass:[NSDictionary class]]) {
                continue;
            }
            NSDictionary *tensor = (NSDictionary *)item;
            NSString *name = tensor[@"name"];
            if (![name isKindOfClass:[NSString class]] || ![name isEqualToString:target]) {
                continue;
            }
            NSString *dtype = tensor[@"dtype"];
            NSArray *shape = tensor[@"shape"];
            if (![dtype isKindOfClass:[NSString class]] ||
                ![shape isKindOfClass:[NSArray class]] ||
                [shape count] != 1) {
                fprintf(stderr, "ERROR: %s vector has invalid dtype or shape\n", label);
                return 0;
            }
            uint64_t dim = unsigned_number(shape[0], "global vector dim");
            uint64_t offset = unsigned_number(tensor[@"offset"], "global vector offset");
            uint64_t size = unsigned_number(tensor[@"size"], "global vector size");
            uint64_t elemBytes = 0;
            if ([dtype isEqualToString:@"F32"] || [dtype isEqualToString:@"float32"]) {
                elemBytes = sizeof(float);
            } else if ([dtype isEqualToString:@"BF16"] || [dtype isEqualToString:@"bfloat16"]) {
                elemBytes = sizeof(uint16_t);
            } else {
                fprintf(stderr, "ERROR: unsupported %s vector dtype %s\n",
                        label,
                        [dtype UTF8String]);
                return 0;
            }
            if (dim == 0 || dim > UINT32_MAX || size != dim * elemBytes) {
                fprintf(stderr, "ERROR: %s vector shape/size is invalid\n", label);
                return 0;
            }
            out->present = 1;
            out->offset = offset;
            out->size = size;
            out->dim = (uint32_t)dim;
            snprintf(out->dtype, sizeof(out->dtype), "%s", [dtype UTF8String]);
            snprintf(out->name, sizeof(out->name), "%s", [name UTF8String]);
            return 1;
        }
    }
    fprintf(stderr, "ERROR: %s vector not found\n", label);
    return 0;
}

static int read_resident_vector_f32(NSString *residentBinPath,
                                    ResidentVectorInfo info,
                                    float **outValues,
                                    uint64_t *outBytesRead) {
    *outValues = NULL;
    if (outBytesRead) {
        *outBytesRead = 0;
    }
    float *values = (float *)calloc(info.dim, sizeof(float));
    if (!values) {
        fprintf(stderr, "ERROR: failed to allocate resident vector copy\n");
        return 0;
    }
    int closeFd = 0;
    int fd = open_resident_read_fd(residentBinPath, &closeFd);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open resident file for vector %s: %s\n",
                [residentBinPath UTF8String],
                strerror(errno));
        free(values);
        return 0;
    }
    if (strcmp(info.dtype, "F32") == 0 || strcmp(info.dtype, "float32") == 0) {
        if (!pread_exact_or_report(fd, values, info.size, info.offset,
                                   [residentBinPath UTF8String])) {
            close_resident_read_fd(fd, closeFd);
            free(values);
            return 0;
        }
    } else {
        uint16_t *raw = (uint16_t *)malloc((size_t)info.size);
        if (!raw) {
            fprintf(stderr, "ERROR: failed to allocate resident BF16 vector copy\n");
            close_resident_read_fd(fd, closeFd);
            free(values);
            return 0;
        }
        if (!pread_exact_or_report(fd, raw, info.size, info.offset,
                                   [residentBinPath UTF8String])) {
            close_resident_read_fd(fd, closeFd);
            free(raw);
            free(values);
            return 0;
        }
        for (uint32_t i = 0; i < info.dim; i++) {
            values[i] = bf16_to_float_cpu(raw[i]);
        }
        free(raw);
    }
    close_resident_read_fd(fd, closeFd);
    if (outBytesRead) {
        *outBytesRead = info.size;
    }
    *outValues = values;
    return 1;
}

static NSString *glm_moe_kernel_source(void) {
    return
        @"#include <metal_stdlib>\n"
        @"using namespace metal;\n"
        @"inline float bf16_to_f32(uint16_t v) { return as_type<float>(uint(v) << 16); }\n"
        @"kernel void glm_rmsnorm_f32(device const float* x [[buffer(0)]],\n"
        @"                            device const float* weight [[buffer(1)]],\n"
        @"                            device float* out [[buffer(2)]],\n"
        @"                            constant uint& dim [[buffer(3)]],\n"
        @"                            constant float& eps [[buffer(4)]],\n"
        @"                            uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid >= dim) return;\n"
        @"  float sumsq = 0.0f;\n"
        @"  for (uint i = 0; i < dim; i++) sumsq += x[i] * x[i];\n"
        @"  float inv = rsqrt(sumsq / float(dim) + eps);\n"
        @"  out[tid] = x[tid] * inv * weight[tid];\n"
        @"}\n"
        @"kernel void glm_router_bf16(device const uint16_t* W [[buffer(0)]],\n"
        @"                            device const float* x [[buffer(1)]],\n"
        @"                            device float* out [[buffer(2)]],\n"
        @"                            constant uint& out_dim [[buffer(3)]],\n"
        @"                            constant uint& in_dim [[buffer(4)]],\n"
        @"                            uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid >= out_dim) return;\n"
        @"  device const uint16_t* row = W + tid * in_dim;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint i = 0; i < in_dim; i++) {\n"
        @"    acc += bf16_to_f32(row[i]) * x[i];\n"
        @"  }\n"
        @"  out[tid] = acc;\n"
        @"}\n"
        @"kernel void glm_context1_o_proj_bf16_matvec_add(device const uint16_t* W [[buffer(0)]],\n"
        @"                                                device const float* x [[buffer(1)]],\n"
        @"                                                device const float* residual [[buffer(2)]],\n"
        @"                                                device float* out [[buffer(3)]],\n"
        @"                                                constant uint& out_dim [[buffer(4)]],\n"
        @"                                                constant uint& in_dim [[buffer(5)]],\n"
        @"                                                uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid >= out_dim) return;\n"
        @"  device const uint16_t* row = W + tid * in_dim;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint i = 0; i < in_dim; i++) acc += bf16_to_f32(row[i]) * x[i];\n"
        @"  out[tid] = acc + residual[tid];\n"
        @"}\n"
        @"kernel void glm_context1_o_proj_f32_matvec_add(device const float* W [[buffer(0)]],\n"
        @"                                               device const float* x [[buffer(1)]],\n"
        @"                                               device const float* residual [[buffer(2)]],\n"
        @"                                               device float* out [[buffer(3)]],\n"
        @"                                               constant uint& out_dim [[buffer(4)]],\n"
        @"                                               constant uint& in_dim [[buffer(5)]],\n"
        @"                                               uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid >= out_dim) return;\n"
        @"  device const float* row = W + tid * in_dim;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint i = 0; i < in_dim; i++) acc += row[i] * x[i];\n"
        @"  out[tid] = acc + residual[tid];\n"
        @"}\n"
        @"inline float mxfp4_value_at(device const uint32_t* W,\n"
        @"                            device const uint32_t* S,\n"
        @"                            uint row,\n"
        @"                            uint col,\n"
        @"                            uint in_dim,\n"
        @"                            uint group_size) {\n"
        @"  uint packed_cols = in_dim / 8u;\n"
        @"  uint num_groups = in_dim / group_size;\n"
        @"  uint packed = W[row * packed_cols + (col >> 3u)];\n"
        @"  uint code = (packed >> ((col & 7u) * 4u)) & 0xFu;\n"
        @"  uint scale_index = row * num_groups + col / group_size;\n"
        @"  uint scale_packed = S[scale_index >> 2u];\n"
        @"  uint scale_raw = (scale_packed >> ((scale_index & 3u) * 8u)) & 0xFFu;\n"
        @"  uint mag = code & 7u;\n"
        @"  float q = mag == 0u ? 0.0f : (mag == 1u ? 0.5f : (mag == 2u ? 1.0f : (mag == 3u ? 1.5f : (mag == 4u ? 2.0f : (mag == 5u ? 3.0f : (mag == 6u ? 4.0f : 6.0f))))));\n"
        @"  if ((code & 8u) != 0u) q = -q;\n"
        @"  return q * as_type<float>(scale_raw << 23);\n"
        @"}\n"
        @"kernel void glm_context1_o_proj_bv_build_bf16(device const uint32_t* oW [[buffer(0)]],\n"
        @"                                                device const uint32_t* oS [[buffer(1)]],\n"
        @"                                                device const uint32_t* uW [[buffer(2)]],\n"
        @"                                                device const uint32_t* uS [[buffer(3)]],\n"
        @"                                                device uint16_t* out [[buffer(4)]],\n"
        @"                                                constant uint& hidden_dim [[buffer(5)]],\n"
        @"                                                constant uint& value_dim [[buffer(6)]],\n"
        @"                                                constant uint& kv_lora_dim [[buffer(7)]],\n"
        @"                                                constant uint& v_head_dim [[buffer(8)]],\n"
        @"                                                constant uint& o_group_size [[buffer(9)]],\n"
        @"                                                constant uint& u_group_size [[buffer(10)]],\n"
        @"                                                uint tid [[thread_position_in_grid]]) {\n"
        @"  uint total = hidden_dim * kv_lora_dim;\n"
        @"  if (tid >= total) return;\n"
        @"  uint row = tid / kv_lora_dim;\n"
        @"  uint kv = tid - row * kv_lora_dim;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint value = 0u; value < value_dim; value++) {\n"
        @"    float o = mxfp4_value_at(oW, oS, row, value, value_dim, o_group_size);\n"
        @"    uint u_row = value;\n"
        @"    float bv = mxfp4_value_at(uW, uS, u_row, kv, kv_lora_dim, u_group_size);\n"
        @"    acc += o * bv;\n"
        @"  }\n"
        @"  uint bits = as_type<uint>(acc);\n"
        @"  uint rounded = bits + 0x7FFFu + ((bits >> 16u) & 1u);\n"
        @"  out[tid] = ushort((rounded >> 16u) & 0xFFFFu);\n"
        @"}\n"
        @"kernel void glm_context1_o_proj_bv_build_f32(device const uint32_t* oW [[buffer(0)]],\n"
        @"                                               device const uint32_t* oS [[buffer(1)]],\n"
        @"                                               device const uint32_t* uW [[buffer(2)]],\n"
        @"                                               device const uint32_t* uS [[buffer(3)]],\n"
        @"                                               device float* out [[buffer(4)]],\n"
        @"                                               constant uint& hidden_dim [[buffer(5)]],\n"
        @"                                               constant uint& value_dim [[buffer(6)]],\n"
        @"                                               constant uint& kv_lora_dim [[buffer(7)]],\n"
        @"                                               constant uint& v_head_dim [[buffer(8)]],\n"
        @"                                               constant uint& o_group_size [[buffer(9)]],\n"
        @"                                               constant uint& u_group_size [[buffer(10)]],\n"
        @"                                               uint tid [[thread_position_in_grid]]) {\n"
        @"  uint total = hidden_dim * kv_lora_dim;\n"
        @"  if (tid >= total) return;\n"
        @"  uint row = tid / kv_lora_dim;\n"
        @"  uint kv = tid - row * kv_lora_dim;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint value = 0u; value < value_dim; value++) {\n"
        @"    float o = mxfp4_value_at(oW, oS, row, value, value_dim, o_group_size);\n"
        @"    float bv = mxfp4_value_at(uW, uS, value, kv, kv_lora_dim, u_group_size);\n"
        @"    acc += o * bv;\n"
        @"  }\n"
        @"  out[tid] = acc;\n"
        @"}\n"
        @"inline float glm_router_score_value(float logit, uint mode) {\n"
        @"  if (mode == 1u || mode == 2u) return logit;\n"
        @"  return 1.0f / (1.0f + exp(-logit));\n"
        @"}\n"
        @"kernel void glm_router_topk_256(device const float* logits [[buffer(0)]],\n"
        @"                                device const float* bias [[buffer(1)]],\n"
        @"                                device uint* ids [[buffer(2)]],\n"
        @"                                device float* weights [[buffer(3)]],\n"
        @"                                constant uint& n [[buffer(4)]],\n"
        @"                                constant uint& k [[buffer(5)]],\n"
        @"                                constant uint& score_mode [[buffer(6)]],\n"
        @"                                constant uint& bias_present [[buffer(7)]],\n"
        @"                                constant uint& norm_topk_prob [[buffer(8)]],\n"
        @"                                constant float& routed_scale [[buffer(9)]],\n"
        @"                                uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid != 0u || n == 0u || n > 256u || k == 0u || k > 64u) return;\n"
        @"  float scores[256];\n"
        @"  float choices[256];\n"
        @"  bool used[256];\n"
        @"  float maxv = logits[0];\n"
        @"  if (score_mode == 2u) {\n"
        @"    for (uint i = 1u; i < n; i++) maxv = max(maxv, logits[i]);\n"
        @"  }\n"
        @"  for (uint i = 0u; i < n; i++) {\n"
        @"    float score = score_mode == 2u ? exp(logits[i] - maxv)\n"
        @"                                  : glm_router_score_value(logits[i], score_mode);\n"
        @"    scores[i] = score;\n"
        @"    choices[i] = score + (bias_present ? bias[i] : 0.0f);\n"
        @"    used[i] = false;\n"
        @"  }\n"
        @"  float sum = 0.0f;\n"
        @"  for (uint out = 0u; out < k; out++) {\n"
        @"    uint best = 0u;\n"
        @"    float best_score = -3.402823466e+38F;\n"
        @"    for (uint i = 0u; i < n; i++) {\n"
        @"      if (!used[i] && choices[i] > best_score) {\n"
        @"        best = i;\n"
        @"        best_score = choices[i];\n"
        @"      }\n"
        @"    }\n"
        @"    used[best] = true;\n"
        @"    ids[out] = best;\n"
        @"    weights[out] = scores[best];\n"
        @"    sum += scores[best];\n"
        @"  }\n"
        @"  for (uint out = 0u; out < k; out++) {\n"
        @"    float w = weights[out];\n"
        @"    if (norm_topk_prob && sum != 0.0f) w /= sum;\n"
        @"    weights[out] = w * routed_scale;\n"
        @"  }\n"
        @"}\n"
        @"constant float mxfp4_e2m1_lut[16] = {\n"
        @"  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,\n"
        @"  0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f\n"
        @"};\n"
        @"inline float mxfp4_e2m1_to_f32(uint q) {\n"
        @"  return mxfp4_e2m1_lut[q & 0xFu];\n"
        @"}\n"
        @"inline uint mxfp4_load_scale_byte(device const uint32_t* S, uint index) {\n"
        @"  uint packed = S[index >> 2u];\n"
        @"  return (packed >> ((index & 3u) * 8u)) & 0xFFu;\n"
        @"}\n"
        @"inline float mxfp4_e8m0_to_f32(uint v) {\n"
        @"  return as_type<float>(uint(v) << 23);\n"
        @"}\n"
        @"inline float mxfp4_dot8_scaled(uint packed, float scale, device const float* x, uint offset) {\n"
        @"  float acc = 0.0f;\n"
        @"  acc += mxfp4_e2m1_to_f32(packed & 0xFu) * x[offset];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 4u) & 0xFu) * x[offset + 1u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 8u) & 0xFu) * x[offset + 2u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 12u) & 0xFu) * x[offset + 3u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 16u) & 0xFu) * x[offset + 4u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 20u) & 0xFu) * x[offset + 5u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 24u) & 0xFu) * x[offset + 6u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 28u) & 0xFu) * x[offset + 7u];\n"
        @"  return acc * scale;\n"
        @"}\n"
        @"inline float mxfp4_dot8_scaled_tg(uint packed, float scale, threadgroup const float* x, uint offset) {\n"
        @"  float acc = 0.0f;\n"
        @"  acc += mxfp4_e2m1_to_f32(packed & 0xFu) * x[offset];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 4u) & 0xFu) * x[offset + 1u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 8u) & 0xFu) * x[offset + 2u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 12u) & 0xFu) * x[offset + 3u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 16u) & 0xFu) * x[offset + 4u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 20u) & 0xFu) * x[offset + 5u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 24u) & 0xFu) * x[offset + 6u];\n"
        @"  acc += mxfp4_e2m1_to_f32((packed >> 28u) & 0xFu) * x[offset + 7u];\n"
        @"  return acc * scale;\n"
        @"}\n"
        @"inline float mxfp4_swiglu_value(float gate_acc, float up_acc) {\n"
        @"  return (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;\n"
        @"}\n"
        @"constant uint GLM_MXFP4_FAST_ROWS = 8u;\n"
        @"kernel void glm_mxfp4_matvec(device const uint32_t* W [[buffer(0)]],\n"
        @"                             device const uint32_t* S [[buffer(1)]],\n"
        @"                             device const float* x [[buffer(2)]],\n"
        @"                             device float* out [[buffer(3)]],\n"
        @"                             constant uint& out_dim [[buffer(4)]],\n"
        @"                             constant uint& in_dim [[buffer(5)]],\n"
        @"                             constant uint& group_size [[buffer(6)]],\n"
        @"                             uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid >= out_dim || group_size == 0u || (group_size & 7u) != 0u) return;\n"
        @"  uint packed_cols = in_dim / 8u;\n"
        @"  uint num_groups = in_dim / group_size;\n"
        @"  uint packed_per_group = group_size / 8u;\n"
        @"  device const uint32_t* row = W + tid * packed_cols;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint g = 0; g < num_groups; g++) {\n"
        @"    float scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(S, tid * num_groups + g));\n"
        @"    uint base_packed = g * packed_per_group;\n"
        @"    uint base_x = g * group_size;\n"
        @"    for (uint p = 0; p < packed_per_group; p++) {\n"
        @"      acc += mxfp4_dot8_scaled(row[base_packed + p], scale, x, base_x + p * 8u);\n"
        @"    }\n"
        @"  }\n"
        @"  out[tid] = acc;\n"
        @"}\n"
        @"kernel void glm_mxfp4_matvec_gs32_simd(device const uint32_t* W [[buffer(0)]],\n"
        @"                                      device const uint32_t* S [[buffer(1)]],\n"
        @"                                      device const float* x [[buffer(2)]],\n"
        @"                                      device float* out [[buffer(3)]],\n"
        @"                                      constant uint& out_dim [[buffer(4)]],\n"
        @"                                      constant uint& in_dim [[buffer(5)]],\n"
        @"                                      constant uint& group_size [[buffer(6)]],\n"
        @"                                      uint tg [[threadgroup_position_in_grid]],\n"
        @"                                      uint simd_lane [[thread_index_in_simdgroup]],\n"
        @"                                      uint simd_group [[simdgroup_index_in_threadgroup]]) {\n"
        @"  if (group_size != 32u || (in_dim & 31u) != 0u) return;\n"
        @"  uint row = tg * GLM_MXFP4_FAST_ROWS + simd_group;\n"
        @"  if (row >= out_dim) return;\n"
        @"  uint packed_cols = in_dim / 8u;\n"
        @"  uint num_groups = in_dim / 32u;\n"
        @"  device const uint32_t* w_row = W + row * packed_cols;\n"
        @"  uint scale_base = row * num_groups;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint col = simd_lane; col < packed_cols; col += 32u) {\n"
        @"    uint g = col >> 2u;\n"
        @"    uint base_x = col * 8u;\n"
        @"    float scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(S, scale_base + g));\n"
        @"    acc += mxfp4_dot8_scaled(w_row[col], scale, x, base_x);\n"
        @"  }\n"
        @"  float sum = simd_sum(acc);\n"
        @"  if (simd_lane == 0u) out[row] = sum;\n"
        @"}\n"
        @"kernel void glm_mxfp4_matvec_add(device const uint32_t* W [[buffer(0)]],\n"
        @"                                 device const uint32_t* S [[buffer(1)]],\n"
        @"                                 device const float* x [[buffer(2)]],\n"
        @"                                 device const float* residual [[buffer(3)]],\n"
        @"                                 device float* out [[buffer(4)]],\n"
        @"                                 constant uint& out_dim [[buffer(5)]],\n"
        @"                                 constant uint& in_dim [[buffer(6)]],\n"
        @"                                 constant uint& group_size [[buffer(7)]],\n"
        @"                                 uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid >= out_dim || group_size == 0u || (group_size & 7u) != 0u) return;\n"
        @"  uint packed_cols = in_dim / 8u;\n"
        @"  uint num_groups = in_dim / group_size;\n"
        @"  uint packed_per_group = group_size / 8u;\n"
        @"  device const uint32_t* row = W + tid * packed_cols;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint g = 0; g < num_groups; g++) {\n"
        @"    float scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(S, tid * num_groups + g));\n"
        @"    uint base_packed = g * packed_per_group;\n"
        @"    uint base_x = g * group_size;\n"
        @"    for (uint p = 0; p < packed_per_group; p++) {\n"
        @"      acc += mxfp4_dot8_scaled(row[base_packed + p], scale, x, base_x + p * 8u);\n"
        @"    }\n"
        @"  }\n"
        @"  out[tid] = acc + residual[tid];\n"
        @"}\n"
        @"kernel void glm_mxfp4_matvec_add_gs32_simd(device const uint32_t* W [[buffer(0)]],\n"
        @"                                          device const uint32_t* S [[buffer(1)]],\n"
        @"                                          device const float* x [[buffer(2)]],\n"
        @"                                          device const float* residual [[buffer(3)]],\n"
        @"                                          device float* out [[buffer(4)]],\n"
        @"                                          constant uint& out_dim [[buffer(5)]],\n"
        @"                                          constant uint& in_dim [[buffer(6)]],\n"
        @"                                          constant uint& group_size [[buffer(7)]],\n"
        @"                                          uint tg [[threadgroup_position_in_grid]],\n"
        @"                                          uint simd_lane [[thread_index_in_simdgroup]],\n"
        @"                                          uint simd_group [[simdgroup_index_in_threadgroup]]) {\n"
        @"  if (group_size != 32u || (in_dim & 31u) != 0u) return;\n"
        @"  uint row = tg * GLM_MXFP4_FAST_ROWS + simd_group;\n"
        @"  if (row >= out_dim) return;\n"
        @"  uint packed_cols = in_dim / 8u;\n"
        @"  uint num_groups = in_dim / 32u;\n"
        @"  device const uint32_t* w_row = W + row * packed_cols;\n"
        @"  uint scale_base = row * num_groups;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint col = simd_lane; col < packed_cols; col += 32u) {\n"
        @"    uint g = col >> 2u;\n"
        @"    uint base_x = col * 8u;\n"
        @"    float scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(S, scale_base + g));\n"
        @"    acc += mxfp4_dot8_scaled(w_row[col], scale, x, base_x);\n"
        @"  }\n"
        @"  float sum = simd_sum(acc);\n"
        @"  if (simd_lane == 0u) out[row] = sum + residual[row];\n"
        @"}\n"
        @"constant uint GLM_MXFP4_INPUT_TILE = 2048u;\n"
        @"kernel void glm_mxfp4_matvec_add_gs32_tiled(device const uint32_t* W [[buffer(0)]],\n"
        @"                                           device const uint32_t* S [[buffer(1)]],\n"
        @"                                           device const float* x [[buffer(2)]],\n"
        @"                                           device const float* residual [[buffer(3)]],\n"
        @"                                           device float* out [[buffer(4)]],\n"
        @"                                           constant uint& out_dim [[buffer(5)]],\n"
        @"                                           constant uint& in_dim [[buffer(6)]],\n"
        @"                                           constant uint& group_size [[buffer(7)]],\n"
        @"                                           uint tg [[threadgroup_position_in_grid]],\n"
        @"                                           uint lid [[thread_position_in_threadgroup]],\n"
        @"                                           uint simd_lane [[thread_index_in_simdgroup]],\n"
        @"                                           uint simd_group [[simdgroup_index_in_threadgroup]]) {\n"
        @"  if (group_size != 32u || (in_dim & 31u) != 0u) return;\n"
        @"  uint row = tg * GLM_MXFP4_FAST_ROWS + simd_group;\n"
        @"  uint packed_cols = in_dim / 8u;\n"
        @"  uint num_groups = in_dim / 32u;\n"
        @"  threadgroup float x_tile[GLM_MXFP4_INPUT_TILE];\n"
        @"  float acc = 0.0f;\n"
        @"  if (row < out_dim) {\n"
        @"    device const uint32_t* w_row = W + row * packed_cols;\n"
        @"    uint scale_base = row * num_groups;\n"
        @"    for (uint tile_base = 0u; tile_base < in_dim; tile_base += GLM_MXFP4_INPUT_TILE) {\n"
        @"      uint tile_elems = min(GLM_MXFP4_INPUT_TILE, in_dim - tile_base);\n"
        @"      for (uint i = lid; i < tile_elems; i += 256u) x_tile[i] = x[tile_base + i];\n"
        @"      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"      uint tile_packed_base = tile_base >> 3u;\n"
        @"      uint tile_packed_count = tile_elems >> 3u;\n"
        @"      for (uint p = simd_lane; p < tile_packed_count; p += 32u) {\n"
        @"        uint col = tile_packed_base + p;\n"
        @"        uint g = col >> 2u;\n"
        @"        float scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(S, scale_base + g));\n"
        @"        acc += mxfp4_dot8_scaled_tg(w_row[col], scale, x_tile, p * 8u);\n"
        @"      }\n"
        @"      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    }\n"
        @"  } else {\n"
        @"    for (uint tile_base = 0u; tile_base < in_dim; tile_base += GLM_MXFP4_INPUT_TILE) {\n"
        @"      uint tile_elems = min(GLM_MXFP4_INPUT_TILE, in_dim - tile_base);\n"
        @"      for (uint i = lid; i < tile_elems; i += 256u) x_tile[i] = x[tile_base + i];\n"
        @"      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    }\n"
        @"  }\n"
        @"  float sum = simd_sum(acc);\n"
        @"  if (row < out_dim && simd_lane == 0u) out[row] = sum + residual[row];\n"
        @"}\n"
        @"kernel void glm_rope_split_batch_f32(device const float* q_b [[buffer(0)]],\n"
        @"                                     device const float* k [[buffer(1)]],\n"
        @"                                     device float* out_q_nope [[buffer(2)]],\n"
        @"                                     device float* out_q_rope [[buffer(3)]],\n"
        @"                                     device float* out_q [[buffer(4)]],\n"
        @"                                     device float* out_k [[buffer(5)]],\n"
        @"                                     constant uint& num_heads [[buffer(6)]],\n"
        @"                                     constant uint& qk_nope_dim [[buffer(7)]],\n"
        @"                                     constant uint& rope_dim [[buffer(8)]],\n"
        @"                                     constant uint& start_position [[buffer(9)]],\n"
        @"                                     constant float& theta [[buffer(10)]],\n"
        @"                                     constant uint& interleave [[buffer(11)]],\n"
        @"                                     constant uint& batch [[buffer(12)]],\n"
        @"                                     uint2 gid [[thread_position_in_grid]]) {\n"
        @"  uint tid = gid.x;\n"
        @"  uint token = gid.y;\n"
        @"  uint q_head_dim = qk_nope_dim + rope_dim;\n"
        @"  uint q_b_count = num_heads * q_head_dim;\n"
        @"  uint q_nope_count = num_heads * qk_nope_dim;\n"
        @"  uint q_rope_count = num_heads * rope_dim;\n"
        @"  uint total = q_b_count + rope_dim;\n"
        @"  if (tid >= total || token >= batch) return;\n"
        @"  uint half_dim = rope_dim / 2u;\n"
        @"  uint position = start_position + token;\n"
        @"  if (tid < q_b_count) {\n"
        @"    uint head = tid / q_head_dim;\n"
        @"    uint head_idx = tid - head * q_head_dim;\n"
        @"    uint q_b_token_base = token * q_b_count;\n"
        @"    float value = q_b[q_b_token_base + tid];\n"
        @"    if (head_idx < qk_nope_dim) {\n"
        @"      out_q_nope[token * q_nope_count + head * qk_nope_dim + head_idx] = value;\n"
        @"      return;\n"
        @"    }\n"
        @"    uint idx = head_idx - qk_nope_dim;\n"
        @"    uint src_base = q_b_token_base + head * q_head_dim + qk_nope_dim;\n"
        @"    out_q_rope[token * q_rope_count + head * rope_dim + idx] = value;\n"
        @"    float rot = 0.0f;\n"
        @"    uint freq_idx = 0;\n"
        @"    if (interleave != 0u) {\n"
        @"      uint pair = idx ^ 1u;\n"
        @"      rot = (idx & 1u) ? q_b[src_base + pair] : -q_b[src_base + pair];\n"
        @"      freq_idx = idx / 2u;\n"
        @"    } else if (idx < half_dim) {\n"
        @"      rot = -q_b[src_base + idx + half_dim];\n"
        @"      freq_idx = idx;\n"
        @"    } else {\n"
        @"      rot = q_b[src_base + idx - half_dim];\n"
        @"      freq_idx = idx - half_dim;\n"
        @"    }\n"
        @"    float angle = float(position) / pow(theta, (2.0f * float(freq_idx)) / float(rope_dim));\n"
        @"    out_q[token * q_rope_count + head * rope_dim + idx] = value * cos(angle) + rot * sin(angle);\n"
        @"    return;\n"
        @"  }\n"
        @"  uint idx = tid - q_b_count;\n"
        @"  uint k_base = token * rope_dim;\n"
        @"  float value = k[k_base + idx];\n"
        @"  float rot = 0.0f;\n"
        @"  uint freq_idx = 0;\n"
        @"  if (interleave != 0u) {\n"
        @"    uint pair = idx ^ 1u;\n"
        @"    rot = (idx & 1u) ? k[k_base + pair] : -k[k_base + pair];\n"
        @"    freq_idx = idx / 2u;\n"
        @"  } else if (idx < half_dim) {\n"
        @"    rot = -k[k_base + idx + half_dim];\n"
        @"    freq_idx = idx;\n"
        @"  } else {\n"
        @"    rot = k[k_base + idx - half_dim];\n"
        @"    freq_idx = idx - half_dim;\n"
        @"  }\n"
        @"  float angle = float(position) / pow(theta, (2.0f * float(freq_idx)) / float(rope_dim));\n"
        @"  out_k[k_base + idx] = value * cos(angle) + rot * sin(angle);\n"
        @"}\n"
        @"kernel void glm_mla_attention_f32(device const float* q_nope [[buffer(0)]],\n"
        @"                                  device const float* q_rope [[buffer(1)]],\n"
        @"                                  device const float* cache [[buffer(2)]],\n"
        @"                                  device const float* kv_b [[buffer(3)]],\n"
        @"                                  device float* out [[buffer(4)]],\n"
        @"                                  constant uint& context_len [[buffer(5)]],\n"
        @"                                  constant uint& num_heads [[buffer(6)]],\n"
        @"                                  constant uint& kv_lora_dim [[buffer(7)]],\n"
        @"                                  constant uint& qk_nope_dim [[buffer(8)]],\n"
        @"                                  constant uint& rope_dim [[buffer(9)]],\n"
        @"                                  constant uint& v_head_dim [[buffer(10)]],\n"
        @"                                  constant float& scale [[buffer(11)]],\n"
        @"                                  constant float& theta [[buffer(12)]],\n"
        @"                                  constant uint& interleave [[buffer(13)]],\n"
        @"                                  constant uint& position_offset [[buffer(14)]],\n"
        @"                                  uint tid [[thread_position_in_grid]]) {\n"
        @"  uint total = num_heads * v_head_dim;\n"
        @"  if (tid >= total) return;\n"
        @"  uint h = tid / v_head_dim;\n"
        @"  uint vd = tid - h * v_head_dim;\n"
        @"  uint head_row_base = h * (qk_nope_dim + v_head_dim);\n"
        @"  uint cache_width = kv_lora_dim + rope_dim;\n"
        @"  uint half_dim = rope_dim / 2u;\n"
        @"  float max_score = -INFINITY;\n"
        @"  for (uint t = 0; t < context_len; t++) {\n"
        @"    device const float* latent = cache + t * cache_width;\n"
        @"    device const float* k_rope_raw = latent + kv_lora_dim;\n"
        @"    float score = 0.0f;\n"
        @"    for (uint d = 0; d < qk_nope_dim; d++) {\n"
        @"      device const float* row = kv_b + (head_row_base + d) * kv_lora_dim;\n"
        @"      float k_nope = 0.0f;\n"
        @"      for (uint r = 0; r < kv_lora_dim; r++) k_nope += row[r] * latent[r];\n"
        @"      score += q_nope[h * qk_nope_dim + d] * k_nope;\n"
        @"    }\n"
        @"    uint pos = position_offset + t;\n"
        @"    for (uint d = 0; d < rope_dim; d++) {\n"
        @"      float x = k_rope_raw[d];\n"
        @"      float rot = 0.0f;\n"
        @"      uint freq_idx = 0;\n"
        @"      if (interleave != 0u) {\n"
        @"        uint pair = d ^ 1u;\n"
        @"        rot = (d & 1u) ? k_rope_raw[pair] : -k_rope_raw[pair];\n"
        @"        freq_idx = d / 2u;\n"
        @"      } else if (d < half_dim) {\n"
        @"        rot = -k_rope_raw[d + half_dim];\n"
        @"        freq_idx = d;\n"
        @"      } else {\n"
        @"        rot = k_rope_raw[d - half_dim];\n"
        @"        freq_idx = d - half_dim;\n"
        @"      }\n"
        @"      float angle = float(pos) / pow(theta, (2.0f * float(freq_idx)) / float(rope_dim));\n"
        @"      float k_rope = x * cos(angle) + rot * sin(angle);\n"
        @"      score += q_rope[h * rope_dim + d] * k_rope;\n"
        @"    }\n"
        @"    score *= scale;\n"
        @"    if (score > max_score) max_score = score;\n"
        @"  }\n"
        @"  float denom = 0.0f;\n"
        @"  float acc = 0.0f;\n"
        @"  device const float* v_row = kv_b + (head_row_base + qk_nope_dim + vd) * kv_lora_dim;\n"
        @"  for (uint t = 0; t < context_len; t++) {\n"
        @"    device const float* latent = cache + t * cache_width;\n"
        @"    device const float* k_rope_raw = latent + kv_lora_dim;\n"
        @"    float score = 0.0f;\n"
        @"    for (uint d = 0; d < qk_nope_dim; d++) {\n"
        @"      device const float* row = kv_b + (head_row_base + d) * kv_lora_dim;\n"
        @"      float k_nope = 0.0f;\n"
        @"      for (uint r = 0; r < kv_lora_dim; r++) k_nope += row[r] * latent[r];\n"
        @"      score += q_nope[h * qk_nope_dim + d] * k_nope;\n"
        @"    }\n"
        @"    uint pos = position_offset + t;\n"
        @"    for (uint d = 0; d < rope_dim; d++) {\n"
        @"      float x = k_rope_raw[d];\n"
        @"      float rot = 0.0f;\n"
        @"      uint freq_idx = 0;\n"
        @"      if (interleave != 0u) {\n"
        @"        uint pair = d ^ 1u;\n"
        @"        rot = (d & 1u) ? k_rope_raw[pair] : -k_rope_raw[pair];\n"
        @"        freq_idx = d / 2u;\n"
        @"      } else if (d < half_dim) {\n"
        @"        rot = -k_rope_raw[d + half_dim];\n"
        @"        freq_idx = d;\n"
        @"      } else {\n"
        @"        rot = k_rope_raw[d - half_dim];\n"
        @"        freq_idx = d - half_dim;\n"
        @"      }\n"
        @"      float angle = float(pos) / pow(theta, (2.0f * float(freq_idx)) / float(rope_dim));\n"
        @"      float k_rope = x * cos(angle) + rot * sin(angle);\n"
        @"      score += q_rope[h * rope_dim + d] * k_rope;\n"
        @"    }\n"
        @"    float weight = exp(score * scale - max_score);\n"
        @"    float value = 0.0f;\n"
        @"    for (uint r = 0; r < kv_lora_dim; r++) value += v_row[r] * latent[r];\n"
        @"    denom += weight;\n"
        @"    acc += weight * value;\n"
        @"  }\n"
        @"  out[tid] = denom == 0.0f ? 0.0f : acc / denom;\n"
        @"}\n"
        @"kernel void glm_mla_attention_context_small_f32(device const float* q_nope [[buffer(0)]],\n"
        @"                                                device const float* q_rope [[buffer(1)]],\n"
        @"                                                device const float* cache [[buffer(2)]],\n"
        @"                                                device const float* kv_b [[buffer(3)]],\n"
        @"                                                device float* out [[buffer(4)]],\n"
        @"                                                constant uint& context_len [[buffer(5)]],\n"
        @"                                                constant uint& num_heads [[buffer(6)]],\n"
        @"                                                constant uint& kv_lora_dim [[buffer(7)]],\n"
        @"                                                constant uint& qk_nope_dim [[buffer(8)]],\n"
        @"                                                constant uint& rope_dim [[buffer(9)]],\n"
        @"                                                constant uint& v_head_dim [[buffer(10)]],\n"
        @"                                                constant float& scale [[buffer(11)]],\n"
        @"                                                constant float& theta [[buffer(12)]],\n"
        @"                                                constant uint& interleave [[buffer(13)]],\n"
        @"                                                constant uint& position_offset [[buffer(14)]],\n"
        @"                                                uint lid [[thread_index_in_threadgroup]],\n"
        @"                                                uint3 tg [[threadgroup_position_in_grid]]) {\n"
        @"  uint h = tg.y;\n"
        @"  if (context_len < 2u || context_len > 32u || h >= num_heads) return;\n"
        @"  uint cache_width = kv_lora_dim + rope_dim;\n"
        @"  uint half_dim = rope_dim / 2u;\n"
        @"  uint head_row_base = h * (qk_nope_dim + v_head_dim);\n"
        @"  threadgroup float weights[32];\n"
        @"  threadgroup float scores[32];\n"
        @"  threadgroup float score_partials[1024];\n"
        @"  for (uint t = 0; t < context_len; t++) {\n"
        @"    device const float* latent = cache + t * cache_width;\n"
        @"    device const float* k_rope_raw = latent + kv_lora_dim;\n"
        @"    float score_partial = 0.0f;\n"
        @"    for (uint d = lid; d < qk_nope_dim; d += v_head_dim) {\n"
        @"      device const float* row = kv_b + (head_row_base + d) * kv_lora_dim;\n"
        @"      float k_nope = 0.0f;\n"
        @"      for (uint r = 0; r < kv_lora_dim; r++) k_nope += row[r] * latent[r];\n"
        @"      score_partial += q_nope[h * qk_nope_dim + d] * k_nope;\n"
        @"    }\n"
        @"    uint pos = position_offset + t;\n"
        @"    for (uint d = lid; d < rope_dim; d += v_head_dim) {\n"
        @"      float x = k_rope_raw[d];\n"
        @"      float rot = 0.0f;\n"
        @"      uint freq_idx = 0;\n"
        @"      if (interleave != 0u) {\n"
        @"        uint pair = d ^ 1u;\n"
        @"        rot = (d & 1u) ? k_rope_raw[pair] : -k_rope_raw[pair];\n"
        @"        freq_idx = d / 2u;\n"
        @"      } else if (d < half_dim) {\n"
        @"        rot = -k_rope_raw[d + half_dim];\n"
        @"        freq_idx = d;\n"
        @"      } else {\n"
        @"        rot = k_rope_raw[d - half_dim];\n"
        @"        freq_idx = d - half_dim;\n"
        @"      }\n"
        @"      float angle = float(pos) / pow(theta, (2.0f * float(freq_idx)) / float(rope_dim));\n"
        @"      float k_rope = x * cos(angle) + rot * sin(angle);\n"
        @"      score_partial += q_rope[h * rope_dim + d] * k_rope;\n"
        @"    }\n"
        @"    score_partials[lid] = score_partial;\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    if (lid == 0u) {\n"
        @"      float score = 0.0f;\n"
        @"      for (uint i = 0; i < v_head_dim; i++) score += score_partials[i];\n"
        @"      scores[t] = score * scale;\n"
        @"    }\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"  }\n"
        @"  if (lid == 0u) {\n"
        @"    float max_score = scores[0];\n"
        @"    for (uint t = 1u; t < context_len; t++) max_score = max(max_score, scores[t]);\n"
        @"    float denom = 0.0f;\n"
        @"    for (uint t = 0; t < context_len; t++) {\n"
        @"      weights[t] = exp(scores[t] - max_score);\n"
        @"      denom += weights[t];\n"
        @"    }\n"
        @"    for (uint t = 0; t < context_len; t++) weights[t] = denom == 0.0f ? 0.0f : weights[t] / denom;\n"
        @"  }\n"
        @"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"  if (lid >= v_head_dim) return;\n"
        @"  float acc = 0.0f;\n"
        @"  device const float* v_row = kv_b + (head_row_base + qk_nope_dim + lid) * kv_lora_dim;\n"
        @"  for (uint t = 0; t < context_len; t++) {\n"
        @"    device const float* latent = cache + t * cache_width;\n"
        @"    float value = 0.0f;\n"
        @"    for (uint r = 0; r < kv_lora_dim; r++) value += v_row[r] * latent[r];\n"
        @"    acc += weights[t] * value;\n"
        @"  }\n"
        @"  out[h * v_head_dim + lid] = acc;\n"
        @"}\n"
        @"kernel void glm_mla_attention_streaming_f32(device const float* q_nope [[buffer(0)]],\n"
        @"                                            device const float* q_rope [[buffer(1)]],\n"
        @"                                            device const float* cache [[buffer(2)]],\n"
        @"                                            device const float* kv_b [[buffer(3)]],\n"
        @"                                            device float* out [[buffer(4)]],\n"
        @"                                            constant uint& context_len [[buffer(5)]],\n"
        @"                                            constant uint& num_heads [[buffer(6)]],\n"
        @"                                            constant uint& kv_lora_dim [[buffer(7)]],\n"
        @"                                            constant uint& qk_nope_dim [[buffer(8)]],\n"
        @"                                            constant uint& rope_dim [[buffer(9)]],\n"
        @"                                            constant uint& v_head_dim [[buffer(10)]],\n"
        @"                                            constant float& scale [[buffer(11)]],\n"
        @"                                            constant float& theta [[buffer(12)]],\n"
        @"                                            constant uint& interleave [[buffer(13)]],\n"
        @"                                            constant uint& position_offset [[buffer(14)]],\n"
        @"                                            uint lid [[thread_index_in_threadgroup]],\n"
        @"                                            uint3 tg [[threadgroup_position_in_grid]]) {\n"
        @"  uint h = tg.y;\n"
        @"  if (context_len < 2u || h >= num_heads) return;\n"
        @"  uint cache_width = kv_lora_dim + rope_dim;\n"
        @"  uint half_dim = rope_dim / 2u;\n"
        @"  uint head_row_base = h * (qk_nope_dim + v_head_dim);\n"
        @"  threadgroup float old_scale_shared;\n"
        @"  threadgroup float weight_shared;\n"
        @"  threadgroup float denom_shared;\n"
        @"  threadgroup float score_partials[1024];\n"
        @"  float acc = 0.0f;\n"
        @"  float running_max = -INFINITY;\n"
        @"  float denom = 0.0f;\n"
        @"  device const float* v_row = kv_b + (head_row_base + qk_nope_dim + lid) * kv_lora_dim;\n"
        @"  for (uint t = 0; t < context_len; t++) {\n"
        @"    device const float* latent = cache + t * cache_width;\n"
        @"    device const float* k_rope_raw = latent + kv_lora_dim;\n"
        @"    float score_partial = 0.0f;\n"
        @"    for (uint d = lid; d < qk_nope_dim; d += v_head_dim) {\n"
        @"      device const float* row = kv_b + (head_row_base + d) * kv_lora_dim;\n"
        @"      float k_nope = 0.0f;\n"
        @"      for (uint r = 0; r < kv_lora_dim; r++) k_nope += row[r] * latent[r];\n"
        @"      score_partial += q_nope[h * qk_nope_dim + d] * k_nope;\n"
        @"    }\n"
        @"    uint pos = position_offset + t;\n"
        @"    for (uint d = lid; d < rope_dim; d += v_head_dim) {\n"
        @"      float x = k_rope_raw[d];\n"
        @"      float rot = 0.0f;\n"
        @"      uint freq_idx = 0;\n"
        @"      if (interleave != 0u) {\n"
        @"        uint pair = d ^ 1u;\n"
        @"        rot = (d & 1u) ? k_rope_raw[pair] : -k_rope_raw[pair];\n"
        @"        freq_idx = d / 2u;\n"
        @"      } else if (d < half_dim) {\n"
        @"        rot = -k_rope_raw[d + half_dim];\n"
        @"        freq_idx = d;\n"
        @"      } else {\n"
        @"        rot = k_rope_raw[d - half_dim];\n"
        @"        freq_idx = d - half_dim;\n"
        @"      }\n"
        @"      float angle = float(pos) / pow(theta, (2.0f * float(freq_idx)) / float(rope_dim));\n"
        @"      float k_rope = x * cos(angle) + rot * sin(angle);\n"
        @"      score_partial += q_rope[h * rope_dim + d] * k_rope;\n"
        @"    }\n"
        @"    score_partials[lid] = score_partial;\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    if (lid == 0u) {\n"
        @"      float score = 0.0f;\n"
        @"      for (uint i = 0; i < v_head_dim; i++) score += score_partials[i];\n"
        @"      float scaled = score * scale;\n"
        @"      float new_max = max(running_max, scaled);\n"
        @"      old_scale_shared = denom == 0.0f ? 0.0f : exp(running_max - new_max);\n"
        @"      weight_shared = exp(scaled - new_max);\n"
        @"      denom = denom * old_scale_shared + weight_shared;\n"
        @"      running_max = new_max;\n"
        @"      denom_shared = denom;\n"
        @"    }\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"    float value = 0.0f;\n"
        @"    for (uint r = 0; r < kv_lora_dim; r++) value += v_row[r] * latent[r];\n"
        @"    acc = acc * old_scale_shared + weight_shared * value;\n"
        @"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"  }\n"
        @"  out[h * v_head_dim + lid] = denom_shared == 0.0f ? 0.0f : acc / denom_shared;\n"
        @"}\n"
        @"kernel void glm_mla_attention_context1_f32(device const float* q_nope [[buffer(0)]],\n"
        @"                                           device const float* q_rope [[buffer(1)]],\n"
        @"                                           device const float* cache [[buffer(2)]],\n"
        @"                                           device const float* kv_b [[buffer(3)]],\n"
        @"                                           device float* out [[buffer(4)]],\n"
        @"                                           constant uint& context_len [[buffer(5)]],\n"
        @"                                           constant uint& num_heads [[buffer(6)]],\n"
        @"                                           constant uint& kv_lora_dim [[buffer(7)]],\n"
        @"                                           constant uint& qk_nope_dim [[buffer(8)]],\n"
        @"                                           constant uint& rope_dim [[buffer(9)]],\n"
        @"                                           constant uint& v_head_dim [[buffer(10)]],\n"
        @"                                           constant float& scale [[buffer(11)]],\n"
        @"                                           constant float& theta [[buffer(12)]],\n"
        @"                                           constant uint& interleave [[buffer(13)]],\n"
        @"                                           constant uint& position_offset [[buffer(14)]],\n"
        @"                                           uint tid [[thread_position_in_grid]]) {\n"
        @"  (void)q_nope; (void)q_rope; (void)scale; (void)theta; (void)interleave; (void)position_offset;\n"
        @"  uint total = num_heads * v_head_dim;\n"
        @"  if (tid >= total || context_len == 0u) return;\n"
        @"  uint h = tid / v_head_dim;\n"
        @"  uint vd = tid - h * v_head_dim;\n"
        @"  uint head_row_base = h * (qk_nope_dim + v_head_dim);\n"
        @"  (void)rope_dim;\n"
        @"  device const float* latent = cache;\n"
        @"  device const float* v_row = kv_b + (head_row_base + qk_nope_dim + vd) * kv_lora_dim;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint r = 0; r < kv_lora_dim; r++) acc += v_row[r] * latent[r];\n"
        @"  out[tid] = acc;\n"
        @"}\n"
        @"constant uint GLM_MXFP4_FAST_MAX_IN = 6144u;\n"
        @"kernel void glm_mxfp4_swiglu_gs32_fast(device const uint32_t* gateW [[buffer(0)]],\n"
        @"                                       device const uint32_t* gateS [[buffer(1)]],\n"
        @"                                       device const uint32_t* upW [[buffer(2)]],\n"
        @"                                       device const uint32_t* upS [[buffer(3)]],\n"
        @"                                       device const float* x [[buffer(4)]],\n"
        @"                                       device float* out [[buffer(5)]],\n"
        @"                                       constant uint& out_dim [[buffer(6)]],\n"
        @"                                       constant uint& in_dim [[buffer(7)]],\n"
        @"                                       constant uint& group_size [[buffer(8)]],\n"
        @"                                       uint tg [[threadgroup_position_in_grid]],\n"
        @"                                       uint lid [[thread_position_in_threadgroup]],\n"
        @"                                       uint simd_lane [[thread_index_in_simdgroup]],\n"
        @"                                       uint simd_group [[simdgroup_index_in_threadgroup]]) {\n"
        @"  if (group_size != 32u || in_dim > GLM_MXFP4_FAST_MAX_IN) return;\n"
        @"  threadgroup float x_shared[6144];\n"
        @"  for (uint i = lid; i < in_dim; i += 256u) x_shared[i] = x[i];\n"
        @"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"  uint row = tg * GLM_MXFP4_FAST_ROWS + simd_group;\n"
        @"  if (row >= out_dim) return;\n"
        @"  uint packed_cols = in_dim / 8u;\n"
        @"  uint num_groups = in_dim / 32u;\n"
        @"  device const uint32_t* gate_row = gateW + row * packed_cols;\n"
        @"  device const uint32_t* up_row = upW + row * packed_cols;\n"
        @"  uint scale_base = row * num_groups;\n"
        @"  float gate_acc = 0.0f;\n"
        @"  float up_acc = 0.0f;\n"
        @"  for (uint col = simd_lane; col < packed_cols; col += 32u) {\n"
        @"    uint g = col >> 2u;\n"
        @"    uint base_x = col * 8u;\n"
        @"    float gate_scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(gateS, scale_base + g));\n"
        @"    float up_scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(upS, scale_base + g));\n"
        @"    gate_acc += mxfp4_dot8_scaled_tg(gate_row[col], gate_scale, x_shared, base_x);\n"
        @"    up_acc += mxfp4_dot8_scaled_tg(up_row[col], up_scale, x_shared, base_x);\n"
        @"  }\n"
        @"  float gate_sum = simd_sum(gate_acc);\n"
        @"  float up_sum = simd_sum(up_acc);\n"
        @"  if (simd_lane == 0u) out[row] = mxfp4_swiglu_value(gate_sum, up_sum);\n"
        @"}\n"
        @"kernel void glm_mxfp4_down_weighted_add_gs32_fast(device const uint32_t* W [[buffer(0)]],\n"
        @"                                                 device const uint32_t* S [[buffer(1)]],\n"
        @"                                                 device const float* x [[buffer(2)]],\n"
        @"                                                 device float* dst [[buffer(3)]],\n"
        @"                                                 constant float& weight [[buffer(4)]],\n"
        @"                                                 constant uint& out_dim [[buffer(5)]],\n"
        @"                                                 constant uint& in_dim [[buffer(6)]],\n"
        @"                                                 constant uint& group_size [[buffer(7)]],\n"
        @"                                                 uint tg [[threadgroup_position_in_grid]],\n"
        @"                                                 uint lid [[thread_position_in_threadgroup]],\n"
        @"                                                 uint simd_lane [[thread_index_in_simdgroup]],\n"
        @"                                                 uint simd_group [[simdgroup_index_in_threadgroup]]) {\n"
        @"  if (group_size != 32u || in_dim > GLM_MXFP4_FAST_MAX_IN) return;\n"
        @"  threadgroup float x_shared[6144];\n"
        @"  for (uint i = lid; i < in_dim; i += 256u) x_shared[i] = x[i];\n"
        @"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        @"  uint row = tg * GLM_MXFP4_FAST_ROWS + simd_group;\n"
        @"  if (row >= out_dim) return;\n"
        @"  uint packed_cols = in_dim / 8u;\n"
        @"  uint num_groups = in_dim / 32u;\n"
        @"  device const uint32_t* w_row = W + row * packed_cols;\n"
        @"  uint scale_base = row * num_groups;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint col = simd_lane; col < packed_cols; col += 32u) {\n"
        @"    uint g = col >> 2u;\n"
        @"    uint base_x = col * 8u;\n"
        @"    float scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(S, scale_base + g));\n"
        @"    acc += mxfp4_dot8_scaled_tg(w_row[col], scale, x_shared, base_x);\n"
        @"  }\n"
        @"  float sum = simd_sum(acc);\n"
        @"  if (simd_lane == 0u) dst[row] += weight * sum;\n"
        @"}\n"
        @"kernel void glm_mxfp4_swiglu_gs32(device const uint32_t* gateW [[buffer(0)]],\n"
        @"                                  device const uint32_t* gateS [[buffer(1)]],\n"
        @"                                  device const uint32_t* upW [[buffer(2)]],\n"
        @"                                  device const uint32_t* upS [[buffer(3)]],\n"
        @"                                  device const float* x [[buffer(4)]],\n"
        @"                                  device float* out [[buffer(5)]],\n"
        @"                                  constant uint& out_dim [[buffer(6)]],\n"
        @"                                  constant uint& in_dim [[buffer(7)]],\n"
        @"                                  constant uint& group_size [[buffer(8)]],\n"
        @"                                  uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid >= out_dim || group_size != 32u) return;\n"
        @"  uint packed_cols = in_dim / 8u;\n"
        @"  uint num_groups = in_dim / 32u;\n"
        @"  device const uint32_t* gate_row = gateW + tid * packed_cols;\n"
        @"  device const uint32_t* up_row = upW + tid * packed_cols;\n"
        @"  uint scale_base = tid * num_groups;\n"
        @"  float gate_acc = 0.0f;\n"
        @"  float up_acc = 0.0f;\n"
        @"  for (uint g = 0; g < num_groups; g++) {\n"
        @"    float gate_scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(gateS, scale_base + g));\n"
        @"    float up_scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(upS, scale_base + g));\n"
        @"    uint base_packed = g * 4u;\n"
        @"    uint base_x = g * 32u;\n"
        @"    gate_acc += mxfp4_dot8_scaled(gate_row[base_packed], gate_scale, x, base_x);\n"
        @"    gate_acc += mxfp4_dot8_scaled(gate_row[base_packed + 1u], gate_scale, x, base_x + 8u);\n"
        @"    gate_acc += mxfp4_dot8_scaled(gate_row[base_packed + 2u], gate_scale, x, base_x + 16u);\n"
        @"    gate_acc += mxfp4_dot8_scaled(gate_row[base_packed + 3u], gate_scale, x, base_x + 24u);\n"
        @"    up_acc += mxfp4_dot8_scaled(up_row[base_packed], up_scale, x, base_x);\n"
        @"    up_acc += mxfp4_dot8_scaled(up_row[base_packed + 1u], up_scale, x, base_x + 8u);\n"
        @"    up_acc += mxfp4_dot8_scaled(up_row[base_packed + 2u], up_scale, x, base_x + 16u);\n"
        @"    up_acc += mxfp4_dot8_scaled(up_row[base_packed + 3u], up_scale, x, base_x + 24u);\n"
        @"  }\n"
        @"  out[tid] = mxfp4_swiglu_value(gate_acc, up_acc);\n"
        @"}\n"
        @"kernel void glm_mxfp4_down_weighted_add_gs32(device const uint32_t* W [[buffer(0)]],\n"
        @"                                            device const uint32_t* S [[buffer(1)]],\n"
        @"                                            device const float* x [[buffer(2)]],\n"
        @"                                            device float* dst [[buffer(3)]],\n"
        @"                                            constant float& weight [[buffer(4)]],\n"
        @"                                            constant uint& out_dim [[buffer(5)]],\n"
        @"                                            constant uint& in_dim [[buffer(6)]],\n"
        @"                                            constant uint& group_size [[buffer(7)]],\n"
        @"                                            uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid >= out_dim || group_size != 32u) return;\n"
        @"  uint packed_cols = in_dim / 8u;\n"
        @"  uint num_groups = in_dim / 32u;\n"
        @"  device const uint32_t* row = W + tid * packed_cols;\n"
        @"  uint scale_base = tid * num_groups;\n"
        @"  float acc = 0.0f;\n"
        @"  for (uint g = 0; g < num_groups; g++) {\n"
        @"    float scale = mxfp4_e8m0_to_f32(mxfp4_load_scale_byte(S, scale_base + g));\n"
        @"    uint base_packed = g * 4u;\n"
        @"    uint base_x = g * 32u;\n"
        @"    acc += mxfp4_dot8_scaled(row[base_packed], scale, x, base_x);\n"
        @"    acc += mxfp4_dot8_scaled(row[base_packed + 1u], scale, x, base_x + 8u);\n"
        @"    acc += mxfp4_dot8_scaled(row[base_packed + 2u], scale, x, base_x + 16u);\n"
        @"    acc += mxfp4_dot8_scaled(row[base_packed + 3u], scale, x, base_x + 24u);\n"
        @"  }\n"
        @"  dst[tid] += weight * acc;\n"
        @"}\n"
        @"kernel void glm_add_inplace_f32(device const float* residual [[buffer(0)]],\n"
        @"                                device float* out [[buffer(1)]],\n"
        @"                                constant uint& n [[buffer(2)]],\n"
        @"                                uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid >= n) return;\n"
        @"  out[tid] += residual[tid];\n"
        @"}\n"
        @"kernel void glm_swiglu_f32(device const float* gate [[buffer(0)]],\n"
        @"                           device const float* up [[buffer(1)]],\n"
        @"                           device float* out [[buffer(2)]],\n"
        @"                           constant uint& n [[buffer(3)]],\n"
        @"                           uint tid [[thread_position_in_grid]]) {\n"
        @"  if (tid >= n) return;\n"
        @"  float g = gate[tid];\n"
        @"  out[tid] = (g / (1.0f + exp(-g))) * up[tid];\n"
        @"}\n";
}

static id<MTLLibrary> g_glm_moe_library = nil;
static id<MTLDevice> g_glm_moe_library_device = nil;
static NSMutableDictionary *g_glm_pipeline_cache = nil;
static id<MTLDevice> g_glm_pipeline_cache_device = nil;
static id<MTLCommandQueue> g_glm_command_queue = nil;
static id<MTLDevice> g_glm_command_queue_device = nil;

static id<MTLComputePipelineState> make_glm_pipeline(id<MTLDevice> device,
                                                      id<MTLLibrary> library,
                                                      NSString *name) {
    if (g_glm_pipeline_cache_device != device) {
        g_glm_pipeline_cache_device = device;
        g_glm_pipeline_cache = [NSMutableDictionary dictionary];
    }
    id<MTLComputePipelineState> cached = [g_glm_pipeline_cache objectForKey:name];
    if (cached) {
        return cached;
    }
    NSError *error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "ERROR: missing Metal function %s\n", [name UTF8String]);
        return nil;
    }
    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn
                                                                             error:&error];
    if (!pipe) {
        fprintf(stderr,
                "ERROR: failed to create pipeline %s: %s\n",
                [name UTF8String],
                error ? [[error localizedDescription] UTF8String] : "unknown");
        return nil;
    }
    [g_glm_pipeline_cache setObject:pipe forKey:name];
    return pipe;
}

static id<MTLLibrary> make_glm_moe_library(id<MTLDevice> device) {
    if (g_glm_moe_library_device == device && g_glm_moe_library) {
        return g_glm_moe_library;
    }
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:glm_moe_kernel_source()
                                                  options:nil
                                                    error:&error];
    if (!library) {
        fprintf(stderr,
                "ERROR: failed to compile GLM MoE kernels: %s\n",
                error ? [[error localizedDescription] UTF8String] : "unknown");
        return nil;
    }
    g_glm_moe_library_device = device;
    g_glm_moe_library = library;
    g_glm_pipeline_cache_device = nil;
    g_glm_pipeline_cache = nil;
    return g_glm_moe_library;
}

static id<MTLCommandQueue> shared_glm_command_queue(id<MTLDevice> device) {
    if (g_glm_command_queue_device == device && g_glm_command_queue) {
        return g_glm_command_queue;
    }
    g_glm_command_queue_device = device;
    g_glm_command_queue = [device newCommandQueue];
    return g_glm_command_queue;
}

static MTLSize threadgroup_1d(id<MTLComputePipelineState> pipe, NSUInteger total_threads) {
    NSUInteger max_threads = pipe.maxTotalThreadsPerThreadgroup;
    if (max_threads == 0 || max_threads > 256) {
        max_threads = 256;
    }
    NSUInteger width = total_threads < max_threads ? total_threads : max_threads;
    if (width == 0) {
        width = 1;
    }
    return MTLSizeMake(width, 1, 1);
}

static int parse_component_shape2(NSArray *shape, uint32_t *dim0, uint32_t *dim1, const char *name) {
    if (![shape isKindOfClass:[NSArray class]] || [shape count] != 2) {
        fprintf(stderr, "ERROR: component %s shape must have rank 2\n", name);
        return 0;
    }
    uint64_t a = unsigned_number(shape[0], "component shape dim0");
    uint64_t b = unsigned_number(shape[1], "component shape dim1");
    if (a == 0 || b == 0 || a > UINT32_MAX || b > UINT32_MAX) {
        fprintf(stderr, "ERROR: component %s shape dims must be in 1..UINT32_MAX\n", name);
        return 0;
    }
    *dim0 = (uint32_t)a;
    *dim1 = (uint32_t)b;
    return 1;
}

static int read_component_info(NSDictionary *component,
                               const char *expected_name,
                               Mxfp4ComponentInfo *out) {
    if (![component isKindOfClass:[NSDictionary class]]) {
        return 0;
    }
    NSString *name = component[@"name"];
    NSString *dtype = component[@"dtype"];
    if (![name isKindOfClass:[NSString class]] ||
        strcmp([name UTF8String], expected_name) != 0 ||
        ![dtype isKindOfClass:[NSString class]]) {
        return 0;
    }
    uint64_t offset = unsigned_number(component[@"offset"], "component offset");
    uint64_t size = unsigned_number(component[@"size"], "component size");
    if (offset % 4 != 0) {
        fprintf(stderr, "ERROR: component %s offset must be 4-byte aligned\n", expected_name);
        return 0;
    }
    uint32_t dim0 = 0;
    uint32_t dim1 = 0;
    if (!parse_component_shape2(component[@"shape"], &dim0, &dim1, expected_name)) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    out->offset = offset;
    out->size = size;
    out->dim0 = dim0;
    out->dim1 = dim1;
    snprintf(out->dtype, sizeof(out->dtype), "%s", [dtype UTF8String]);
    return 1;
}

static int find_component_info(NSArray *components,
                               const char *name,
                               Mxfp4ComponentInfo *out) {
    if (![components isKindOfClass:[NSArray class]]) {
        fprintf(stderr, "ERROR: expert layer components must be an array\n");
        return 0;
    }
    for (id item in components) {
        if (read_component_info(item, name, out)) {
            return 1;
        }
    }
    fprintf(stderr, "ERROR: expert layer missing component %s\n", name);
    return 0;
}

static int validate_mxfp4_component_sizes(Mxfp4ExpertInfo *info) {
    uint64_t hidden = info->hidden_dim;
    uint64_t intermediate = info->intermediate_dim;
    uint64_t group = info->group_size;
    uint64_t gate_weight_bytes = intermediate * (hidden / 8u) * sizeof(uint32_t);
    uint64_t gate_scale_bytes = intermediate * (hidden / group);
    uint64_t down_weight_bytes = hidden * (intermediate / 8u) * sizeof(uint32_t);
    uint64_t down_scale_bytes = hidden * (intermediate / group);
    if (info->gate_w.size != gate_weight_bytes ||
        info->up_w.size != gate_weight_bytes ||
        info->gate_s.size != gate_scale_bytes ||
        info->up_s.size != gate_scale_bytes ||
        info->down_w.size != down_weight_bytes ||
        info->down_s.size != down_scale_bytes) {
        fprintf(stderr, "ERROR: MXFP4 component sizes do not match derived dimensions\n");
        return 0;
    }
    return 1;
}

static int parse_mxfp4_expert_info(NSDictionary *expertLayout,
                                   NSDictionary *layer,
                                   Mxfp4ExpertInfo *out) {
    memset(out, 0, sizeof(*out));
    NSString *quantization = expertLayout[@"quantization"];
    if (![quantization isKindOfClass:[NSString class]] ||
        ![quantization isEqualToString:@"mlx-mxfp4"]) {
        fprintf(stderr, "ERROR: --probe-layer-moe currently supports quantization mlx-mxfp4 only\n");
        return 0;
    }
    uint64_t group = unsigned_number(expertLayout[@"group_size"], "expert layout group_size");
    if (group != 32) {
        fprintf(stderr, "ERROR: --probe-layer-moe currently supports group_size=32 only\n");
        return 0;
    }
    NSArray *components = layer[@"components"];
    if (!find_component_info(components, "gate_proj.weight", &out->gate_w) ||
        !find_component_info(components, "gate_proj.scales", &out->gate_s) ||
        !find_component_info(components, "up_proj.weight", &out->up_w) ||
        !find_component_info(components, "up_proj.scales", &out->up_s) ||
        !find_component_info(components, "down_proj.weight", &out->down_w) ||
        !find_component_info(components, "down_proj.scales", &out->down_s)) {
        return 0;
    }
    if (strcmp(out->gate_w.dtype, "U32") != 0 ||
        strcmp(out->up_w.dtype, "U32") != 0 ||
        strcmp(out->down_w.dtype, "U32") != 0 ||
        strcmp(out->gate_s.dtype, "U8") != 0 ||
        strcmp(out->up_s.dtype, "U8") != 0 ||
        strcmp(out->down_s.dtype, "U8") != 0) {
        fprintf(stderr, "ERROR: MXFP4 routed expert components must be U32 weights and U8 scales\n");
        return 0;
    }
    if (out->gate_w.dim0 != out->up_w.dim0 ||
        out->gate_w.dim1 != out->up_w.dim1 ||
        out->gate_s.dim0 != out->gate_w.dim0 ||
        out->up_s.dim0 != out->up_w.dim0 ||
        out->down_w.dim0 * 1u == 0 ||
        out->down_s.dim0 != out->down_w.dim0) {
        fprintf(stderr, "ERROR: MXFP4 component shapes are inconsistent\n");
        return 0;
    }
    uint64_t hidden_from_gate = (uint64_t)out->gate_w.dim1 * 8u;
    uint64_t intermediate_from_down = (uint64_t)out->down_w.dim1 * 8u;
    if (hidden_from_gate > UINT32_MAX || intermediate_from_down > UINT32_MAX) {
        fprintf(stderr, "ERROR: derived MXFP4 dimensions exceed UINT32_MAX\n");
        return 0;
    }
    out->hidden_dim = (uint32_t)hidden_from_gate;
    out->intermediate_dim = out->gate_w.dim0;
    out->group_size = (uint32_t)group;
    if (out->down_w.dim0 != out->hidden_dim ||
        intermediate_from_down != out->intermediate_dim ||
        out->gate_s.dim1 != out->hidden_dim / out->group_size ||
        out->up_s.dim1 != out->hidden_dim / out->group_size ||
        out->down_s.dim1 != out->intermediate_dim / out->group_size) {
        fprintf(stderr, "ERROR: MXFP4 shapes do not form gate/up/down GLM expert matrices\n");
        return 0;
    }
    if (out->hidden_dim % 32u != 0 || out->intermediate_dim % 32u != 0) {
        fprintf(stderr, "ERROR: MXFP4 group32 kernels require dims divisible by 32\n");
        return 0;
    }
    return validate_mxfp4_component_sizes(out);
}

static int find_resident_component_info(NSDictionary *residentLayout,
                                        NSString *target,
                                        Mxfp4ComponentInfo *out) {
    NSArray *tensors = residentLayout[@"tensors"];
    if (![tensors isKindOfClass:[NSArray class]]) {
        fprintf(stderr, "ERROR: resident layout missing tensors array\n");
        return 0;
    }
    for (id item in tensors) {
        if (![item isKindOfClass:[NSDictionary class]]) {
            continue;
        }
        NSDictionary *tensor = (NSDictionary *)item;
        NSString *name = tensor[@"name"];
        if (![name isKindOfClass:[NSString class]] || ![name isEqualToString:target]) {
            continue;
        }
        NSString *dtype = tensor[@"dtype"];
        NSArray *shape = tensor[@"shape"];
        if (![dtype isKindOfClass:[NSString class]]) {
            fprintf(stderr, "ERROR: resident component %s missing dtype\n", [target UTF8String]);
            return 0;
        }
        uint64_t offset = unsigned_number(tensor[@"offset"], "resident component offset");
        uint64_t size = unsigned_number(tensor[@"size"], "resident component size");
        if (offset % 4 != 0 || size % 4 != 0) {
            fprintf(stderr, "ERROR: resident component %s offset/size must be 4-byte aligned\n",
                    [target UTF8String]);
            return 0;
        }
        uint32_t dim0 = 0;
        uint32_t dim1 = 0;
        if (!parse_component_shape2(shape, &dim0, &dim1, [target UTF8String])) {
            return 0;
        }
        memset(out, 0, sizeof(*out));
        out->offset = offset;
        out->size = size;
        out->dim0 = dim0;
        out->dim1 = dim1;
        snprintf(out->dtype, sizeof(out->dtype), "%s", [dtype UTF8String]);
        return 1;
    }
    fprintf(stderr, "ERROR: resident component %s not found\n", [target UTF8String]);
    return 0;
}

static int resident_tensor_exists(NSDictionary *residentLayout, NSString *target) {
    NSArray *tensors = residentLayout[@"tensors"];
    if (![tensors isKindOfClass:[NSArray class]]) {
        return 0;
    }
    for (id item in tensors) {
        if (![item isKindOfClass:[NSDictionary class]]) {
            continue;
        }
        NSString *name = ((NSDictionary *)item)[@"name"];
        if ([name isKindOfClass:[NSString class]] && [name isEqualToString:target]) {
            return 1;
        }
    }
    return 0;
}

static int parse_component_shape3(NSArray *shape,
                                  uint32_t *dim0,
                                  uint32_t *dim1,
                                  uint32_t *dim2,
                                  const char *name) {
    if (![shape isKindOfClass:[NSArray class]] || [shape count] != 3) {
        fprintf(stderr, "ERROR: component %s shape must have rank 3\n", name);
        return 0;
    }
    uint64_t a = unsigned_number(shape[0], "component shape dim0");
    uint64_t b = unsigned_number(shape[1], "component shape dim1");
    uint64_t c = unsigned_number(shape[2], "component shape dim2");
    if (a == 0 || b == 0 || c == 0 ||
        a > UINT32_MAX || b > UINT32_MAX || c > UINT32_MAX) {
        fprintf(stderr, "ERROR: component %s shape dims must be in 1..UINT32_MAX\n", name);
        return 0;
    }
    *dim0 = (uint32_t)a;
    *dim1 = (uint32_t)b;
    *dim2 = (uint32_t)c;
    return 1;
}

static int find_resident_tensor3d_component_info(NSDictionary *residentLayout,
                                                 NSString *target,
                                                 ResidentTensor3DComponentInfo *out) {
    NSArray *tensors = residentLayout[@"tensors"];
    if (![tensors isKindOfClass:[NSArray class]]) {
        fprintf(stderr, "ERROR: resident layout missing tensors array\n");
        return 0;
    }
    for (id item in tensors) {
        if (![item isKindOfClass:[NSDictionary class]]) {
            continue;
        }
        NSDictionary *tensor = (NSDictionary *)item;
        NSString *name = tensor[@"name"];
        if (![name isKindOfClass:[NSString class]] || ![name isEqualToString:target]) {
            continue;
        }
        NSString *dtype = tensor[@"dtype"];
        NSArray *shape = tensor[@"shape"];
        if (![dtype isKindOfClass:[NSString class]]) {
            fprintf(stderr, "ERROR: resident tensor %s missing dtype\n", [target UTF8String]);
            return 0;
        }
        uint32_t dim0 = 0;
        uint32_t dim1 = 0;
        uint32_t dim2 = 0;
        if (!parse_component_shape3(shape, &dim0, &dim1, &dim2, [target UTF8String])) {
            return 0;
        }
        uint64_t offset = unsigned_number(tensor[@"offset"], "resident tensor offset");
        uint64_t size = unsigned_number(tensor[@"size"], "resident tensor size");
        memset(out, 0, sizeof(*out));
        out->offset = offset;
        out->size = size;
        out->dim0 = dim0;
        out->dim1 = dim1;
        out->dim2 = dim2;
        snprintf(out->dtype, sizeof(out->dtype), "%s", [dtype UTF8String]);
        snprintf(out->name, sizeof(out->name), "%s", [name UTF8String]);
        return 1;
    }
    fprintf(stderr, "ERROR: resident tensor %s not found\n", [target UTF8String]);
    return 0;
}

static int find_resident_mxfp4_tensor3d_info(NSDictionary *residentLayout,
                                             NSString *weightName,
                                             ResidentMxfp4Tensor3DInfo *out) {
    memset(out, 0, sizeof(*out));
    if (![weightName hasSuffix:@".weight"]) {
        fprintf(stderr, "ERROR: resident MXFP4 tensor %s must end with .weight\n",
                [weightName UTF8String]);
        return 0;
    }
    NSString *base = [weightName substringToIndex:([weightName length] - 7)];
    NSString *scaleName = [base stringByAppendingString:@".scales"];
    if (!find_resident_tensor3d_component_info(residentLayout, weightName, &out->weight) ||
        !find_resident_tensor3d_component_info(residentLayout, scaleName, &out->scales)) {
        return 0;
    }
    if (strcmp(out->weight.dtype, "U32") != 0 ||
        strcmp(out->scales.dtype, "U8") != 0) {
        fprintf(stderr, "ERROR: resident MXFP4 tensor requires U32 weight and U8 scales\n");
        return 0;
    }
    if (out->weight.offset % 4 != 0 || out->weight.size % 4 != 0) {
        fprintf(stderr, "ERROR: resident MXFP4 tensor weight offset/size must be 4-byte aligned\n");
        return 0;
    }
    if (out->weight.dim0 != out->scales.dim0 ||
        out->weight.dim1 != out->scales.dim1 ||
        out->weight.dim2 == 0 ||
        out->scales.dim2 == 0 ||
        out->weight.dim2 > UINT32_MAX / 8u) {
        fprintf(stderr, "ERROR: resident MXFP4 tensor shapes are inconsistent\n");
        return 0;
    }
    uint64_t expectedWeightBytes = 0;
    uint64_t expectedScaleBytes = 0;
    uint64_t firstTwo = 0;
    if (!checked_mul_u64((uint64_t)out->weight.dim0,
                         (uint64_t)out->weight.dim1,
                         &firstTwo) ||
        !checked_mul_u64(firstTwo,
                         (uint64_t)out->weight.dim2,
                         &expectedWeightBytes) ||
        !checked_mul_u64(expectedWeightBytes,
                         sizeof(uint32_t),
                         &expectedWeightBytes) ||
        !checked_mul_u64(firstTwo,
                         (uint64_t)out->scales.dim2,
                         &expectedScaleBytes)) {
        fprintf(stderr, "ERROR: resident MXFP4 tensor byte size overflows\n");
        return 0;
    }
    if (out->weight.size != expectedWeightBytes ||
        out->scales.size != expectedScaleBytes) {
        fprintf(stderr, "ERROR: resident MXFP4 tensor component byte sizes are invalid\n");
        return 0;
    }
    uint32_t logicalDim2 = out->weight.dim2 * 8u;
    if (logicalDim2 % out->scales.dim2 != 0) {
        fprintf(stderr, "ERROR: resident MXFP4 tensor scales do not divide logical dim\n");
        return 0;
    }
    uint32_t groupSize = logicalDim2 / out->scales.dim2;
    if (groupSize == 0 || groupSize % 8u != 0) {
        fprintf(stderr, "ERROR: resident MXFP4 tensor group size is invalid\n");
        return 0;
    }
    uint64_t total = 0;
    if (!checked_add_u64(out->weight.size, out->scales.size, &total)) {
        fprintf(stderr, "ERROR: resident MXFP4 tensor total byte size overflows\n");
        return 0;
    }
    out->dim0 = out->weight.dim0;
    out->dim1 = out->weight.dim1;
    out->dim2 = logicalDim2;
    out->group_size = groupSize;
    out->total_bytes = total;
    snprintf(out->name, sizeof(out->name), "%s", out->weight.name);
    return 1;
}

static int resident_mxfp4_tensor3d_f32_bytes(ResidentMxfp4Tensor3DInfo info,
                                             uint64_t *outBytes) {
    uint64_t firstTwo = 0;
    uint64_t count = 0;
    if (!checked_mul_u64((uint64_t)info.dim0, (uint64_t)info.dim1, &firstTwo) ||
        !checked_mul_u64(firstTwo, (uint64_t)info.dim2, &count) ||
        !checked_mul_u64(count, sizeof(float), outBytes)) {
        fprintf(stderr, "ERROR: resident tensor %s f32 byte size overflows\n", info.name);
        return 0;
    }
    return 1;
}

static int read_resident_mxfp4_tensor3d_f32(NSString *residentBinPath,
                                            ResidentMxfp4Tensor3DInfo info,
                                            float **outValues,
                                            uint64_t *outBytes) {
    *outValues = NULL;
    *outBytes = 0;
    uint64_t f32Bytes = 0;
    if (!resident_mxfp4_tensor3d_f32_bytes(info, &f32Bytes) ||
        f32Bytes > (uint64_t)SIZE_MAX ||
        info.weight.size > (uint64_t)SIZE_MAX ||
        info.scales.size > (uint64_t)SIZE_MAX) {
        return 0;
    }
    float *values = (float *)calloc((size_t)(f32Bytes / sizeof(float)), sizeof(float));
    uint32_t *rawWeights = (uint32_t *)malloc((size_t)info.weight.size);
    uint8_t *rawScales = (uint8_t *)malloc((size_t)info.scales.size);
    if (!values || !rawWeights || !rawScales) {
        fprintf(stderr, "ERROR: failed to allocate resident MXFP4 tensor staging buffers\n");
        free(values);
        free(rawWeights);
        free(rawScales);
        return 0;
    }
    int closeFd = 0;
    int fd = open_resident_read_fd(residentBinPath, &closeFd);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open resident file for tensor %s: %s\n",
                [residentBinPath UTF8String],
                strerror(errno));
        free(values);
        free(rawWeights);
        free(rawScales);
        return 0;
    }
    int readOk =
        pread_exact_or_report(fd,
                              rawWeights,
                              info.weight.size,
                              info.weight.offset,
                              [residentBinPath UTF8String]) &&
        pread_exact_or_report(fd,
                              rawScales,
                              info.scales.size,
                              info.scales.offset,
                              [residentBinPath UTF8String]);
    close_resident_read_fd(fd, closeFd);
    if (!readOk) {
        free(values);
        free(rawWeights);
        free(rawScales);
        return 0;
    }
    uint64_t rows = 0;
    if (!checked_mul_u64((uint64_t)info.dim0, (uint64_t)info.dim1, &rows)) {
        free(values);
        free(rawWeights);
        free(rawScales);
        return 0;
    }
    uint32_t packedCols = info.weight.dim2;
    uint32_t groups = info.scales.dim2;
    for (uint64_t row = 0; row < rows; row++) {
        uint64_t wBase = row * (uint64_t)packedCols;
        uint64_t sBase = row * (uint64_t)groups;
        uint64_t outBase = row * (uint64_t)info.dim2;
        for (uint32_t p = 0; p < packedCols; p++) {
            uint32_t packed = rawWeights[wBase + p];
            for (uint32_t i = 0; i < 8u; i++) {
                uint32_t logicalCol = p * 8u + i;
                uint32_t group = logicalCol / info.group_size;
                float scale = mxfp4_e8m0_to_f32_cpu(rawScales[sBase + group]);
                float decoded =
                    mxfp4_e2m1_to_f32_cpu((packed >> (i * 4u)) & 0xFu) * scale;
                values[outBase + logicalCol] = decoded;
            }
        }
    }
    free(rawWeights);
    free(rawScales);
    *outValues = values;
    *outBytes = f32Bytes;
    return 1;
}

static void set_local_component(Mxfp4ComponentInfo src,
                                uint64_t offset,
                                Mxfp4ComponentInfo *dst) {
    *dst = src;
    dst->offset = offset;
}

static int parse_shared_mxfp4_info(NSDictionary *residentLayout,
                                   int layerId,
                                   SharedMxfp4Info *out) {
    memset(out, 0, sizeof(*out));
    NSString *prefix = [NSString stringWithFormat:
        @"model.layers.%d.mlp.shared_experts.",
        layerId
    ];
    NSString *gateW = [prefix stringByAppendingString:@"gate_proj.weight"];
    NSString *gateS = [prefix stringByAppendingString:@"gate_proj.scales"];
    NSString *upW = [prefix stringByAppendingString:@"up_proj.weight"];
    NSString *upS = [prefix stringByAppendingString:@"up_proj.scales"];
    NSString *downW = [prefix stringByAppendingString:@"down_proj.weight"];
    NSString *downS = [prefix stringByAppendingString:@"down_proj.scales"];
    if (!find_resident_component_info(residentLayout, gateW, &out->src_gate_w) ||
        !find_resident_component_info(residentLayout, gateS, &out->src_gate_s) ||
        !find_resident_component_info(residentLayout, upW, &out->src_up_w) ||
        !find_resident_component_info(residentLayout, upS, &out->src_up_s) ||
        !find_resident_component_info(residentLayout, downW, &out->src_down_w) ||
        !find_resident_component_info(residentLayout, downS, &out->src_down_s)) {
        return 0;
    }
    if (strcmp(out->src_gate_w.dtype, "U32") != 0 ||
        strcmp(out->src_up_w.dtype, "U32") != 0 ||
        strcmp(out->src_down_w.dtype, "U32") != 0 ||
        strcmp(out->src_gate_s.dtype, "U8") != 0 ||
        strcmp(out->src_up_s.dtype, "U8") != 0 ||
        strcmp(out->src_down_s.dtype, "U8") != 0) {
        fprintf(stderr, "ERROR: shared expert components must be U32 weights and U8 scales\n");
        return 0;
    }
    uint64_t cursor = 0;
    set_local_component(out->src_gate_w, cursor, &out->local.gate_w);
    cursor += out->src_gate_w.size;
    set_local_component(out->src_gate_s, cursor, &out->local.gate_s);
    cursor += out->src_gate_s.size;
    set_local_component(out->src_up_w, cursor, &out->local.up_w);
    cursor += out->src_up_w.size;
    set_local_component(out->src_up_s, cursor, &out->local.up_s);
    cursor += out->src_up_s.size;
    set_local_component(out->src_down_w, cursor, &out->local.down_w);
    cursor += out->src_down_w.size;
    set_local_component(out->src_down_s, cursor, &out->local.down_s);
    cursor += out->src_down_s.size;
    out->total_bytes = cursor;
    uint64_t hidden_from_gate = (uint64_t)out->local.gate_w.dim1 * 8u;
    uint64_t intermediate_from_down = (uint64_t)out->local.down_w.dim1 * 8u;
    if (hidden_from_gate > UINT32_MAX || intermediate_from_down > UINT32_MAX) {
        fprintf(stderr, "ERROR: shared expert derived dimensions exceed UINT32_MAX\n");
        return 0;
    }
    out->local.hidden_dim = (uint32_t)hidden_from_gate;
    out->local.intermediate_dim = out->local.gate_w.dim0;
    out->local.group_size = 32u;
    if (out->local.up_w.dim0 != out->local.gate_w.dim0 ||
        out->local.up_w.dim1 != out->local.gate_w.dim1 ||
        out->local.down_w.dim0 != out->local.hidden_dim ||
        intermediate_from_down != out->local.intermediate_dim ||
        out->local.gate_s.dim0 != out->local.intermediate_dim ||
        out->local.up_s.dim0 != out->local.intermediate_dim ||
        out->local.down_s.dim0 != out->local.hidden_dim ||
        out->local.gate_s.dim1 != out->local.hidden_dim / out->local.group_size ||
        out->local.up_s.dim1 != out->local.hidden_dim / out->local.group_size ||
        out->local.down_s.dim1 != out->local.intermediate_dim / out->local.group_size) {
        fprintf(stderr, "ERROR: shared expert MXFP4 component shapes are inconsistent\n");
        return 0;
    }
    return validate_mxfp4_component_sizes(&out->local);
}

static int find_resident_mxfp4_matrix_info(NSDictionary *residentLayout,
                                           NSString *weightName,
                                           ResidentMxfp4MatrixInfo *out) {
    memset(out, 0, sizeof(*out));
    if (![weightName hasSuffix:@".weight"]) {
        fprintf(stderr, "ERROR: resident MXFP4 tensor %s must end with .weight\n",
                [weightName UTF8String]);
        return 0;
    }
    NSString *base = [weightName substringToIndex:([weightName length] - 7)];
    NSString *scaleName = [base stringByAppendingString:@".scales"];
    if (!find_resident_component_info(residentLayout, weightName, &out->weight) ||
        !find_resident_component_info(residentLayout, scaleName, &out->scales)) {
        return 0;
    }
    if (strcmp(out->weight.dtype, "U32") != 0 ||
        strcmp(out->scales.dtype, "U8") != 0) {
        fprintf(stderr, "ERROR: resident MXFP4 matrix requires U32 weight and U8 scales\n");
        return 0;
    }
    if (out->weight.dim0 != out->scales.dim0 ||
        out->weight.dim0 == 0 ||
        out->weight.dim1 == 0 ||
        out->scales.dim1 == 0) {
        fprintf(stderr, "ERROR: resident MXFP4 matrix shapes are inconsistent\n");
        return 0;
    }
    uint64_t expectedWeightBytes =
        (uint64_t)out->weight.dim0 * (uint64_t)out->weight.dim1 * sizeof(uint32_t);
    uint64_t expectedScaleBytes =
        (uint64_t)out->scales.dim0 * (uint64_t)out->scales.dim1;
    if (out->weight.size != expectedWeightBytes ||
        out->scales.size != expectedScaleBytes) {
        fprintf(stderr, "ERROR: resident MXFP4 matrix component byte sizes are invalid\n");
        return 0;
    }
    uint64_t logicalInDim = (uint64_t)out->weight.dim1 * 8u;
    if (logicalInDim == 0 || logicalInDim > UINT32_MAX) {
        fprintf(stderr, "ERROR: resident MXFP4 logical input dim is out of range\n");
        return 0;
    }
    if (logicalInDim % out->scales.dim1 != 0) {
        fprintf(stderr, "ERROR: resident MXFP4 scale groups do not divide input dim\n");
        return 0;
    }
    uint64_t groupSize = logicalInDim / out->scales.dim1;
    if (groupSize == 0 || groupSize > UINT32_MAX || (groupSize % 8u) != 0) {
        fprintf(stderr, "ERROR: resident MXFP4 group size is invalid\n");
        return 0;
    }
    out->out_dim = out->weight.dim0;
    out->in_dim = (uint32_t)logicalInDim;
    out->group_size = (uint32_t)groupSize;
    out->total_bytes = out->weight.size + out->scales.size;
    snprintf(out->name, sizeof(out->name), "%s", [weightName UTF8String]);
    return 1;
}

static int find_global_mxfp4_matrix_info_by_names(NSDictionary *residentLayout,
                                                  NSArray *names,
                                                  ResidentMxfp4MatrixInfo *out,
                                                  const char *label) {
    for (NSString *name in names) {
        if (!resident_tensor_exists(residentLayout, name)) {
            continue;
        }
        return find_resident_mxfp4_matrix_info(residentLayout, name, out);
    }
    fprintf(stderr, "ERROR: %s MXFP4 matrix not found\n", label);
    return 0;
}

static int find_dense_mxfp4_matrix_info(NSDictionary *residentLayout,
                                        int layerId,
                                        NSString *component,
                                        ResidentMxfp4MatrixInfo *out) {
    NSArray *suffixes = @[
        [NSString stringWithFormat:@".mlp.%@.weight", component],
        [NSString stringWithFormat:@".mlp.switch_mlp.%@.weight", component],
        [NSString stringWithFormat:@".switch_mlp.%@.weight", component],
    ];
    for (NSString *suffix in suffixes) {
        NSString *name = [NSString stringWithFormat:@"model.layers.%d%@", layerId, suffix];
        if (!resident_tensor_exists(residentLayout, name)) {
            continue;
        }
        return find_resident_mxfp4_matrix_info(residentLayout, name, out);
    }
    fprintf(stderr,
            "ERROR: dense MLP %s matrix not found for layer %d\n",
            [component UTF8String],
            layerId);
    return 0;
}

static int parse_dense_mlp_mxfp4_info(NSDictionary *residentLayout,
                                      int layerId,
                                      DenseMlpMxfp4Info *out) {
    memset(out, 0, sizeof(*out));
    if (!find_dense_mxfp4_matrix_info(residentLayout,
                                      layerId,
                                      @"gate_proj",
                                      &out->gate) ||
        !find_dense_mxfp4_matrix_info(residentLayout,
                                      layerId,
                                      @"up_proj",
                                      &out->up) ||
        !find_dense_mxfp4_matrix_info(residentLayout,
                                      layerId,
                                      @"down_proj",
                                      &out->down)) {
        return 0;
    }
    if (out->gate.in_dim != out->up.in_dim ||
        out->gate.out_dim != out->up.out_dim ||
        out->down.out_dim != out->gate.in_dim ||
        out->down.in_dim != out->gate.out_dim ||
        out->gate.group_size != out->up.group_size ||
        out->gate.group_size != out->down.group_size) {
        fprintf(stderr,
                "ERROR: dense MLP matrix dimensions are inconsistent "
                "(gate out=%u in=%u, up out=%u in=%u, down out=%u in=%u)\n",
                out->gate.out_dim,
                out->gate.in_dim,
                out->up.out_dim,
                out->up.in_dim,
                out->down.out_dim,
                out->down.in_dim);
        return 0;
    }
    out->hidden_dim = out->gate.in_dim;
    out->intermediate_dim = out->gate.out_dim;
    out->group_size = out->gate.group_size;
    return 1;
}

static int find_router_weight_info(NSDictionary *residentLayout,
                                   int layerId,
                                   RouterWeightInfo *out) {
    NSArray *tensors = residentLayout[@"tensors"];
    if (![tensors isKindOfClass:[NSArray class]]) {
        fprintf(stderr, "ERROR: resident layout missing tensors array\n");
        return 0;
    }
    NSString *target = [NSString stringWithFormat:@"model.layers.%d.mlp.gate.weight", layerId];
    for (id item in tensors) {
        if (![item isKindOfClass:[NSDictionary class]]) {
            continue;
        }
        NSDictionary *tensor = (NSDictionary *)item;
        NSString *name = tensor[@"name"];
        if (![name isKindOfClass:[NSString class]] || ![name isEqualToString:target]) {
            continue;
        }
        NSString *dtype = tensor[@"dtype"];
        NSArray *shape = tensor[@"shape"];
        if (![dtype isKindOfClass:[NSString class]] ||
            ![shape isKindOfClass:[NSArray class]] ||
            [shape count] != 2) {
            fprintf(stderr, "ERROR: router tensor has invalid dtype or shape\n");
            return 0;
        }
        if (![dtype isEqualToString:@"BF16"] && ![dtype isEqualToString:@"bfloat16"]) {
            fprintf(stderr, "ERROR: --probe-router currently supports BF16 router weights only\n");
            return 0;
        }
        uint64_t numExperts = unsigned_number(shape[0], "router num experts");
        uint64_t hiddenDim = unsigned_number(shape[1], "router hidden dim");
        uint64_t offset = unsigned_number(tensor[@"offset"], "router offset");
        uint64_t size = unsigned_number(tensor[@"size"], "router size");
        if (numExperts == 0 || hiddenDim == 0 ||
            numExperts > UINT32_MAX || hiddenDim > UINT32_MAX) {
            fprintf(stderr, "ERROR: router dims out of range\n");
            return 0;
        }
        uint64_t expectedSize = numExperts * hiddenDim * sizeof(uint16_t);
        if (size != expectedSize) {
            fprintf(stderr,
                    "ERROR: router BF16 size %llu does not match expected %llu\n",
                    (unsigned long long)size,
                    (unsigned long long)expectedSize);
            return 0;
        }
        memset(out, 0, sizeof(*out));
        out->offset = offset;
        out->size = size;
        out->num_experts = (uint32_t)numExperts;
        out->hidden_dim = (uint32_t)hiddenDim;
        snprintf(out->dtype, sizeof(out->dtype), "%s", [dtype UTF8String]);
        snprintf(out->name, sizeof(out->name), "%s", [name UTF8String]);
        return 1;
    }
    fprintf(stderr, "ERROR: router tensor %s not found\n", [target UTF8String]);
    return 0;
}

static int find_router_bias_info(NSDictionary *residentLayout,
                                 int layerId,
                                 RouterBiasInfo *out) {
    memset(out, 0, sizeof(*out));
    NSArray *tensors = residentLayout[@"tensors"];
    if (![tensors isKindOfClass:[NSArray class]]) {
        fprintf(stderr, "ERROR: resident layout missing tensors array\n");
        return 0;
    }
    NSString *target = [NSString stringWithFormat:
        @"model.layers.%d.mlp.gate.e_score_correction_bias",
        layerId
    ];
    for (id item in tensors) {
        if (![item isKindOfClass:[NSDictionary class]]) {
            continue;
        }
        NSDictionary *tensor = (NSDictionary *)item;
        NSString *name = tensor[@"name"];
        if (![name isKindOfClass:[NSString class]] || ![name isEqualToString:target]) {
            continue;
        }
        NSString *dtype = tensor[@"dtype"];
        NSArray *shape = tensor[@"shape"];
        if (![dtype isKindOfClass:[NSString class]] ||
            ![shape isKindOfClass:[NSArray class]] ||
            [shape count] != 1) {
            fprintf(stderr, "ERROR: router correction-bias tensor has invalid dtype or shape\n");
            return 0;
        }
        if (![dtype isEqualToString:@"F32"] && ![dtype isEqualToString:@"float32"]) {
            fprintf(stderr, "ERROR: router correction bias currently supports F32 only\n");
            return 0;
        }
        uint64_t dim = unsigned_number(shape[0], "router correction-bias dim");
        uint64_t offset = unsigned_number(tensor[@"offset"], "router correction-bias offset");
        uint64_t size = unsigned_number(tensor[@"size"], "router correction-bias size");
        if (dim == 0 || dim > UINT32_MAX || size != dim * sizeof(float)) {
            fprintf(stderr, "ERROR: router correction-bias shape/size is invalid\n");
            return 0;
        }
        out->present = 1;
        out->offset = offset;
        out->size = size;
        out->dim = (uint32_t)dim;
        snprintf(out->dtype, sizeof(out->dtype), "%s", [dtype UTF8String]);
        snprintf(out->name, sizeof(out->name), "%s", [name UTF8String]);
        return 1;
    }
    return 1;
}

static int router_score_mode_is_valid(const char *mode) {
    return mode &&
           (strcmp(mode, "sigmoid") == 0 ||
            strcmp(mode, "softmax") == 0 ||
            strcmp(mode, "raw") == 0);
}

static int resolve_router_topk_options(NSDictionary *residentLayout,
                                       LoaderOptions options,
                                       RouterTopKOptions *out) {
    memset(out, 0, sizeof(*out));
    snprintf(out->score, sizeof(out->score), "sigmoid");
    out->routed_scaling_factor = 1.0f;
    out->norm_topk_prob = 1;
    out->n_group = 1;
    out->topk_group = 1;
    NSDictionary *routerMeta = residentLayout[@"router"];
    if ([routerMeta isKindOfClass:[NSDictionary class]]) {
        NSString *score = routerMeta[@"scoring_func"];
        if ([score isKindOfClass:[NSString class]] && [score length] > 0) {
            snprintf(out->score, sizeof(out->score), "%s", [score UTF8String]);
        }
        NSNumber *scale = routerMeta[@"routed_scaling_factor"];
        if ([scale isKindOfClass:[NSNumber class]]) {
            out->routed_scaling_factor = [scale floatValue];
        }
        NSNumber *norm = routerMeta[@"norm_topk_prob"];
        if ([norm isKindOfClass:[NSNumber class]]) {
            out->norm_topk_prob = [norm boolValue] ? 1 : 0;
        }
        NSNumber *nGroup = routerMeta[@"n_group"];
        if ([nGroup isKindOfClass:[NSNumber class]]) {
            out->n_group = [nGroup unsignedIntValue];
        }
        NSNumber *topkGroup = routerMeta[@"topk_group"];
        if ([topkGroup isKindOfClass:[NSNumber class]]) {
            out->topk_group = [topkGroup unsignedIntValue];
        }
    }
    if (options.router_score_set) {
        snprintf(out->score, sizeof(out->score), "%s", options.router_score);
    }
    if (options.routed_scaling_factor_set) {
        out->routed_scaling_factor = (float)options.routed_scaling_factor;
    }
    if (options.norm_topk_prob_set) {
        out->norm_topk_prob = options.norm_topk_prob ? 1 : 0;
    }
    if (options.router_n_group_set) {
        out->n_group = (uint32_t)options.router_n_group;
    }
    if (options.router_topk_group_set) {
        out->topk_group = (uint32_t)options.router_topk_group;
    }
    if (!router_score_mode_is_valid(out->score)) {
        fprintf(stderr, "ERROR: router scoring_func must be sigmoid, softmax, or raw\n");
        return 0;
    }
    if (!isfinite(out->routed_scaling_factor) || out->routed_scaling_factor <= 0.0f) {
        fprintf(stderr, "ERROR: routed scaling factor must be positive and finite\n");
        return 0;
    }
    if (out->n_group == 0 || out->topk_group == 0 || out->topk_group > out->n_group) {
        fprintf(stderr, "ERROR: invalid router grouping\n");
        return 0;
    }
    return 1;
}

static int encode_glm_router_bf16_with_offset(id<MTLCommandBuffer> cmd,
                                              id<MTLComputePipelineState> pipe,
                                              id<MTLBuffer> router,
                                              NSUInteger routerOffset,
                                              id<MTLBuffer> input,
                                              id<MTLBuffer> logits,
                                              RouterWeightInfo info);

static int encode_glm_router_bf16(id<MTLCommandBuffer> cmd,
                                  id<MTLComputePipelineState> pipe,
                                  id<MTLBuffer> router,
                                  id<MTLBuffer> input,
                                  id<MTLBuffer> logits,
                                  RouterWeightInfo info) {
    return encode_glm_router_bf16_with_offset(cmd,
                                             pipe,
                                             router,
                                             0,
                                             input,
                                             logits,
                                             info);
}

static int encode_glm_router_bf16_with_offset(id<MTLCommandBuffer> cmd,
                                              id<MTLComputePipelineState> pipe,
                                              id<MTLBuffer> router,
                                              NSUInteger routerOffset,
                                              id<MTLBuffer> input,
                                              id<MTLBuffer> logits,
                                              RouterWeightInfo info) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create router command encoder\n");
        return 0;
    }
    uint32_t out_dim = info.num_experts;
    uint32_t in_dim = info.hidden_dim;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:router offset:routerOffset atIndex:0];
    [enc setBuffer:input offset:0 atIndex:1];
    [enc setBuffer:logits offset:0 atIndex:2];
    [enc setBytes:&out_dim length:sizeof(out_dim) atIndex:3];
    [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(info.num_experts, 1, 1)
   threadsPerThreadgroup:threadgroup_1d(pipe, info.num_experts)];
    [enc endEncoding];
    return 1;
}

static int router_score_mode_code(const char *mode, uint32_t *out) {
    if (strcmp(mode, "sigmoid") == 0) {
        *out = 0;
        return 1;
    }
    if (strcmp(mode, "raw") == 0) {
        *out = 1;
        return 1;
    }
    if (strcmp(mode, "softmax") == 0) {
        *out = 2;
        return 1;
    }
    return 0;
}

static int encode_glm_router_topk_256_with_bias_offset(
    id<MTLCommandBuffer> cmd,
    id<MTLComputePipelineState> pipe,
    id<MTLBuffer> logits,
    id<MTLBuffer> bias,
    NSUInteger biasOffset,
    id<MTLBuffer> ids,
    id<MTLBuffer> weights,
    uint32_t numExperts,
    uint32_t topK,
    uint32_t scoreMode,
    int biasPresent,
    int normTopkProb,
    float routedScale);

static int encode_glm_router_topk_256(id<MTLCommandBuffer> cmd,
                                      id<MTLComputePipelineState> pipe,
                                      id<MTLBuffer> logits,
                                      id<MTLBuffer> bias,
                                      id<MTLBuffer> ids,
                                      id<MTLBuffer> weights,
                                      uint32_t numExperts,
                                      uint32_t topK,
                                      uint32_t scoreMode,
                                      int biasPresent,
                                      int normTopkProb,
                                      float routedScale) {
    return encode_glm_router_topk_256_with_bias_offset(cmd,
                                                       pipe,
                                                       logits,
                                                       bias,
                                                       0,
                                                       ids,
                                                       weights,
                                                       numExperts,
                                                       topK,
                                                       scoreMode,
                                                       biasPresent,
                                                       normTopkProb,
                                                       routedScale);
}

static int encode_glm_router_topk_256_with_bias_offset(
    id<MTLCommandBuffer> cmd,
    id<MTLComputePipelineState> pipe,
    id<MTLBuffer> logits,
    id<MTLBuffer> bias,
    NSUInteger biasOffset,
    id<MTLBuffer> ids,
    id<MTLBuffer> weights,
    uint32_t numExperts,
    uint32_t topK,
    uint32_t scoreMode,
    int biasPresent,
    int normTopkProb,
    float routedScale) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create router top-k command encoder\n");
        return 0;
    }
    uint32_t biasFlag = biasPresent ? 1u : 0u;
    uint32_t normFlag = normTopkProb ? 1u : 0u;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:logits offset:0 atIndex:0];
    [enc setBuffer:bias offset:biasOffset atIndex:1];
    [enc setBuffer:ids offset:0 atIndex:2];
    [enc setBuffer:weights offset:0 atIndex:3];
    [enc setBytes:&numExperts length:sizeof(numExperts) atIndex:4];
    [enc setBytes:&topK length:sizeof(topK) atIndex:5];
    [enc setBytes:&scoreMode length:sizeof(scoreMode) atIndex:6];
    [enc setBytes:&biasFlag length:sizeof(biasFlag) atIndex:7];
    [enc setBytes:&normFlag length:sizeof(normFlag) atIndex:8];
    [enc setBytes:&routedScale length:sizeof(routedScale) atIndex:9];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    return 1;
}

static int encode_glm_rmsnorm_with_weight_offset(id<MTLCommandBuffer> cmd,
                                                 id<MTLComputePipelineState> pipe,
                                                 id<MTLBuffer> input,
                                                 id<MTLBuffer> weight,
                                                 NSUInteger weightOffset,
                                                 id<MTLBuffer> output,
                                                 uint32_t dim,
                                                 float eps);

static int encode_glm_rmsnorm(id<MTLCommandBuffer> cmd,
                              id<MTLComputePipelineState> pipe,
                              id<MTLBuffer> input,
                              id<MTLBuffer> weight,
                              id<MTLBuffer> output,
                              uint32_t dim,
                              float eps) {
    return encode_glm_rmsnorm_with_weight_offset(cmd,
                                                 pipe,
                                                 input,
                                                 weight,
                                                 0,
                                                 output,
                                                 dim,
                                                 eps);
}

static int encode_glm_rmsnorm_with_weight_offset(id<MTLCommandBuffer> cmd,
                                                 id<MTLComputePipelineState> pipe,
                                                 id<MTLBuffer> input,
                                                 id<MTLBuffer> weight,
                                                 NSUInteger weightOffset,
                                                 id<MTLBuffer> output,
                                                 uint32_t dim,
                                                 float eps) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create RMSNorm command encoder\n");
        return 0;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:weight offset:weightOffset atIndex:1];
    [enc setBuffer:output offset:0 atIndex:2];
    [enc setBytes:&dim length:sizeof(dim) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
   threadsPerThreadgroup:threadgroup_1d(pipe, dim)];
    [enc endEncoding];
    return 1;
}

static int run_rmsnorm_probe(id<MTLDevice> device,
                             NSString *residentBinPath,
                             ResidentVectorInfo weightInfo,
                             NSString *inputPath,
                             NSData *inputDataOverride,
                             float eps,
                             NSData **outputData,
                             RmsNormProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    *outputData = nil;
    uint64_t inputBytes = (uint64_t)weightInfo.dim * sizeof(float);
    NSData *inputData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
    if (!inputData) {
        fprintf(stderr,
                "ERROR: failed to read RMSNorm input %s\n",
                inputPath ? [inputPath UTF8String] : "<memory>");
        return 0;
    }
    if ((uint64_t)[inputData length] != inputBytes) {
        fprintf(stderr,
                "ERROR: RMSNorm input bytes %llu do not match expected %llu\n",
                (unsigned long long)[inputData length],
                (unsigned long long)inputBytes);
        return 0;
    }
    float *weightValues = NULL;
    uint64_t weightBytesRead = 0;
    double weightReadStarted = now_seconds();
    if (!read_resident_vector_f32(residentBinPath,
                                  weightInfo,
                                  &weightValues,
                                  &weightBytesRead)) {
        return 0;
    }
    stats->weight_read_seconds = now_seconds() - weightReadStarted;
    id<MTLLibrary> library = make_glm_moe_library(device);
    id<MTLComputePipelineState> pipe =
        library ? make_glm_pipeline(device, library, @"glm_rmsnorm_f32") : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> input = [device newBufferWithBytes:[inputData bytes]
                                              length:(NSUInteger)inputBytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight = [device newBufferWithBytes:weightValues
                                               length:(NSUInteger)inputBytes
                                              options:MTLResourceStorageModeShared];
    free(weightValues);
    id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)inputBytes
                                               options:MTLResourceStorageModeShared];
    if (!library || !pipe || !queue || !input || !weight || !output) {
        fprintf(stderr, "ERROR: failed to allocate RMSNorm Metal resources\n");
        return 0;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create RMSNorm command buffer\n");
        return 0;
    }
    if (!encode_glm_rmsnorm(cmd,
                            pipe,
                            input,
                            weight,
                            output,
                            weightInfo.dim,
                            eps)) {
        return 0;
    }
    double started = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    stats->elapsed_seconds = now_seconds() - started;
    stats->command_buffer_count = 1;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: RMSNorm command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    stats->weight_bytes_read = weightBytesRead;
    stats->output0 = ((float *)[output contents])[0];
    stats->ok = 1;
    *outputData = [NSData dataWithBytes:[output contents] length:(NSUInteger)inputBytes];
    return 1;
}

static int use_fast_resident_mxfp4_matvec(ResidentMxfp4MatrixInfo info);
static id<MTLComputePipelineState> make_resident_mxfp4_matvec_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    ResidentMxfp4MatrixInfo info);
static id<MTLComputePipelineState> make_resident_mxfp4_matvec_add_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    ResidentMxfp4MatrixInfo info);
static int encode_glm_mxfp4_matvec_with_offsets(
    id<MTLCommandBuffer> cmd,
    id<MTLComputePipelineState> pipe,
    id<MTLBuffer> weightBuffer,
    NSUInteger weightOffset,
    id<MTLBuffer> scalesBuffer,
    NSUInteger scalesOffset,
    ResidentMxfp4MatrixInfo info,
    id<MTLBuffer> input,
    id<MTLBuffer> output);
static int encode_glm_mxfp4_matvec_add_with_offsets(
    id<MTLCommandBuffer> cmd,
    id<MTLComputePipelineState> pipe,
    id<MTLBuffer> weightBuffer,
    NSUInteger weightOffset,
    id<MTLBuffer> scalesBuffer,
    NSUInteger scalesOffset,
    ResidentMxfp4MatrixInfo info,
    id<MTLBuffer> input,
    id<MTLBuffer> residual,
    id<MTLBuffer> output);

static int encode_glm_mxfp4_matvec(id<MTLCommandBuffer> cmd,
                                   id<MTLComputePipelineState> pipe,
                                   id<MTLBuffer> matrix,
                                   ResidentMxfp4MatrixInfo info,
                                   id<MTLBuffer> input,
                                   id<MTLBuffer> output) {
    return encode_glm_mxfp4_matvec_with_offsets(cmd,
                                                pipe,
                                                matrix,
                                                0,
                                                matrix,
                                                (NSUInteger)info.weight.size,
                                                info,
                                                input,
                                                output);
}

static int encode_glm_mxfp4_matvec_add(id<MTLCommandBuffer> cmd,
                                       id<MTLComputePipelineState> pipe,
                                       id<MTLBuffer> matrix,
                                       ResidentMxfp4MatrixInfo info,
                                       id<MTLBuffer> input,
                                       id<MTLBuffer> residual,
                                       id<MTLBuffer> output) {
    return encode_glm_mxfp4_matvec_add_with_offsets(cmd,
                                                    pipe,
                                                    matrix,
                                                    0,
                                                    matrix,
                                                    (NSUInteger)info.weight.size,
                                                    info,
                                                    input,
                                                    residual,
                                                    output);
}

static int encode_glm_mxfp4_matvec_with_offsets(
    id<MTLCommandBuffer> cmd,
    id<MTLComputePipelineState> pipe,
    id<MTLBuffer> weightBuffer,
    NSUInteger weightOffset,
    id<MTLBuffer> scalesBuffer,
    NSUInteger scalesOffset,
    ResidentMxfp4MatrixInfo info,
    id<MTLBuffer> input,
    id<MTLBuffer> output) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create resident MXFP4 matvec encoder\n");
        return 0;
    }
    uint32_t out_dim = info.out_dim;
    uint32_t in_dim = info.in_dim;
    uint32_t group_size = info.group_size;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:weightBuffer offset:weightOffset atIndex:0];
    [enc setBuffer:scalesBuffer offset:scalesOffset atIndex:1];
    [enc setBuffer:input offset:0 atIndex:2];
    [enc setBuffer:output offset:0 atIndex:3];
    [enc setBytes:&out_dim length:sizeof(out_dim) atIndex:4];
    [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:5];
    [enc setBytes:&group_size length:sizeof(group_size) atIndex:6];
    if (use_fast_resident_mxfp4_matvec(info)) {
        NSUInteger rowGroups = ((NSUInteger)info.out_dim + 7u) / 8u;
        [enc dispatchThreadgroups:MTLSizeMake(rowGroups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake(info.out_dim, 1, 1)
       threadsPerThreadgroup:threadgroup_1d(pipe, info.out_dim)];
    }
    [enc endEncoding];
    return 1;
}

static int encode_glm_mxfp4_matvec_add_with_offsets(
    id<MTLCommandBuffer> cmd,
    id<MTLComputePipelineState> pipe,
    id<MTLBuffer> weightBuffer,
    NSUInteger weightOffset,
    id<MTLBuffer> scalesBuffer,
    NSUInteger scalesOffset,
    ResidentMxfp4MatrixInfo info,
    id<MTLBuffer> input,
    id<MTLBuffer> residual,
    id<MTLBuffer> output) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create resident MXFP4 matvec+add encoder\n");
        return 0;
    }
    uint32_t out_dim = info.out_dim;
    uint32_t in_dim = info.in_dim;
    uint32_t group_size = info.group_size;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:weightBuffer offset:weightOffset atIndex:0];
    [enc setBuffer:scalesBuffer offset:scalesOffset atIndex:1];
    [enc setBuffer:input offset:0 atIndex:2];
    [enc setBuffer:residual offset:0 atIndex:3];
    [enc setBuffer:output offset:0 atIndex:4];
    [enc setBytes:&out_dim length:sizeof(out_dim) atIndex:5];
    [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:6];
    [enc setBytes:&group_size length:sizeof(group_size) atIndex:7];
    if (use_fast_resident_mxfp4_matvec(info)) {
        NSUInteger rowGroups = ((NSUInteger)info.out_dim + 7u) / 8u;
        [enc dispatchThreadgroups:MTLSizeMake(rowGroups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake(info.out_dim, 1, 1)
       threadsPerThreadgroup:threadgroup_1d(pipe, info.out_dim)];
    }
    [enc endEncoding];
    return 1;
}

static int stage_resident_mxfp4_matrix_from_fd(int fd,
                                               NSString *residentBinPath,
                                               ResidentMxfp4MatrixInfo info,
                                               void **outPtr,
                                               uint64_t *outAllocBytes,
                                               double *outReadSeconds) {
    *outPtr = NULL;
    *outAllocBytes = 0;
    if (outReadSeconds) {
        *outReadSeconds = 0.0;
    }
    if (info.total_bytes > NSUIntegerMax) {
        fprintf(stderr, "ERROR: resident MXFP4 matrix buffer exceeds NSUIntegerMax\n");
        return 0;
    }
    uint64_t allocBytes = round_up_u64(info.total_bytes, 2 * 1024 * 1024);
    void *matrixPtr = NULL;
    if (posix_memalign(&matrixPtr, 2 * 1024 * 1024, (size_t)allocBytes) != 0 ||
        !matrixPtr) {
        fprintf(stderr, "ERROR: failed to allocate resident MXFP4 aligned buffer\n");
        return 0;
    }
    double readStarted = now_seconds();
    int readOk =
        pread_exact_or_report(fd,
                              matrixPtr,
                              info.weight.size,
                              info.weight.offset,
                              [residentBinPath UTF8String]) &&
        pread_exact_or_report(fd,
                              (uint8_t *)matrixPtr + info.weight.size,
                              info.scales.size,
                              info.scales.offset,
                              [residentBinPath UTF8String]);
    if (outReadSeconds) {
        *outReadSeconds = now_seconds() - readStarted;
    }
    if (!readOk) {
        free(matrixPtr);
        return 0;
    }
    *outPtr = matrixPtr;
    *outAllocBytes = allocBytes;
    return 1;
}

static int run_resident_linear_probe(id<MTLDevice> device,
                                     NSString *residentBinPath,
                                     id<MTLBuffer> residentMetalBuffer,
                                     ResidentMxfp4MatrixInfo info,
                                     NSString *inputPath,
                                     NSData *inputDataOverride,
                                     NSString *outputPath,
                                     NSData **outputDataOut,
                                     int expectOutput0Set,
                                     double expectOutput0,
                                     ResidentLinearProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (outputDataOut) {
        *outputDataOut = nil;
    }
    stats->out_dim = info.out_dim;
    stats->in_dim = info.in_dim;
    stats->group_size = info.group_size;
    stats->output0_check_ok = 1;
    uint64_t inputBytes = (uint64_t)info.in_dim * sizeof(float);
    uint64_t outputBytes = (uint64_t)info.out_dim * sizeof(float);
    if (inputBytes > NSUIntegerMax ||
        outputBytes > NSUIntegerMax ||
        info.total_bytes > NSUIntegerMax) {
        fprintf(stderr, "ERROR: resident MXFP4 linear buffers exceed NSUIntegerMax\n");
        return 0;
    }
    NSData *inputData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
    if (!inputData) {
        fprintf(stderr, "ERROR: failed to read resident-linear input %s\n",
                inputPath ? [inputPath UTF8String] : "<memory>");
        return 0;
    }
    if ((uint64_t)[inputData length] != inputBytes) {
        fprintf(stderr,
                "ERROR: resident-linear input bytes %llu do not match expected %llu\n",
                (unsigned long long)[inputData length],
                (unsigned long long)inputBytes);
        return 0;
    }

    NSUInteger weightOffset = 0;
    NSUInteger scalesOffset = (NSUInteger)info.weight.size;
    uint64_t allocBytes = round_up_u64(info.total_bytes, 2 * 1024 * 1024);
    void *matrixPtr = NULL;
    id<MTLBuffer> matrix = nil;
    if (residentMetalBuffer) {
        uint64_t residentLength = (uint64_t)[residentMetalBuffer length];
        if (info.weight.offset > (uint64_t)NSUIntegerMax ||
            info.scales.offset > (uint64_t)NSUIntegerMax ||
            info.weight.offset + info.weight.size > residentLength ||
            info.scales.offset + info.scales.size > residentLength) {
            fprintf(stderr,
                    "ERROR: resident MXFP4 matrix spans exceed resident Metal buffer\n");
            return 0;
        }
        matrix = residentMetalBuffer;
        weightOffset = (NSUInteger)info.weight.offset;
        scalesOffset = (NSUInteger)info.scales.offset;
        stats->resident_mmap_backed = 1;
        stats->bytes_read = 0;
    } else {
        if (posix_memalign(&matrixPtr, 2 * 1024 * 1024, (size_t)allocBytes) != 0 ||
            !matrixPtr) {
            fprintf(stderr, "ERROR: failed to allocate resident MXFP4 aligned buffer\n");
            return 0;
        }
        int closeFd = 0;
        int fd = open_resident_read_fd(residentBinPath, &closeFd);
        if (fd < 0) {
            fprintf(stderr,
                    "ERROR: failed to open resident file for MXFP4 linear %s: %s\n",
                    [residentBinPath UTF8String],
                    strerror(errno));
            free(matrixPtr);
            return 0;
        }
        double readStarted = now_seconds();
        int readOk =
            pread_exact_or_report(fd,
                                  matrixPtr,
                                  info.weight.size,
                                  info.weight.offset,
                                  [residentBinPath UTF8String]) &&
            pread_exact_or_report(fd,
                                  (uint8_t *)matrixPtr + info.weight.size,
                                  info.scales.size,
                                  info.scales.offset,
                                  [residentBinPath UTF8String]);
        close_resident_read_fd(fd, closeFd);
        stats->read_seconds = now_seconds() - readStarted;
        if (!readOk) {
            free(matrixPtr);
            return 0;
        }
        stats->bytes_read = info.total_bytes;
    }

    id<MTLLibrary> library = make_glm_moe_library(device);
    id<MTLComputePipelineState> pipe =
        library ? make_resident_mxfp4_matvec_pipeline(device, library, info) : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    if (!matrix) {
        matrix = [device newBufferWithBytesNoCopy:matrixPtr
                                           length:(NSUInteger)allocBytes
                                          options:MTLResourceStorageModeShared
                                      deallocator:^(void *pointer, NSUInteger length) {
                                          (void)length;
                                          free(pointer);
                                      }];
        if (!matrix) {
            free(matrixPtr);
            fprintf(stderr, "ERROR: failed to wrap resident MXFP4 matrix buffer\n");
            return 0;
        }
        [matrix didModifyRange:NSMakeRange(0, (NSUInteger)info.total_bytes)];
    }
    id<MTLBuffer> input = [device newBufferWithBytes:[inputData bytes]
                                              length:(NSUInteger)inputBytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)outputBytes
                                               options:MTLResourceStorageModeShared];
    if (!library || !pipe || !queue || !input || !output) {
        fprintf(stderr, "ERROR: failed to allocate resident-linear Metal resources\n");
        return 0;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create resident-linear command buffer\n");
        return 0;
    }
    if (!encode_glm_mxfp4_matvec_with_offsets(cmd,
                                              pipe,
                                              matrix,
                                              weightOffset,
                                              matrix,
                                              scalesOffset,
                                              info,
                                              input,
                                              output)) {
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    stats->kernel_seconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: resident-linear command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    stats->output0 = ((float *)[output contents])[0];
    if (expectOutput0Set) {
        stats->output0_abs_error = fabsf(stats->output0 - (float)expectOutput0);
        stats->output0_check_ok = stats->output0_abs_error <= 5.0e-3f;
    }
    double writeStarted = now_seconds();
    NSData *outputData = [NSData dataWithBytes:[output contents]
                                        length:(NSUInteger)outputBytes];
    if (outputPath) {
        if (![outputData writeToFile:outputPath atomically:YES]) {
            fprintf(stderr, "ERROR: failed to write resident-linear output %s\n",
                    [outputPath UTF8String]);
            return 0;
        }
    }
    if (outputDataOut) {
        *outputDataOut = outputData;
    }
    stats->output_write_seconds = now_seconds() - writeStarted;
    stats->elapsed_seconds = stats->read_seconds +
                             stats->kernel_seconds +
                             stats->output_write_seconds;
    stats->ok = stats->output0_check_ok;
    if (!stats->output0_check_ok) {
        fprintf(stderr,
                "ERROR: resident-linear output[0] %.6f differs from expected %.6f by %.6f\n",
                stats->output0,
                expectOutput0,
                stats->output0_abs_error);
    }
    return stats->ok;
}

static int encode_glm_add_inplace_f32(id<MTLCommandBuffer> cmd,
                                      id<MTLComputePipelineState> pipe,
                                      id<MTLBuffer> residual,
                                      id<MTLBuffer> output,
                                      uint32_t n) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create F32 add encoder\n");
        return 0;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:residual offset:0 atIndex:0];
    [enc setBuffer:output offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
   threadsPerThreadgroup:threadgroup_1d(pipe, n)];
    [enc endEncoding];
    return 1;
}

static int encode_context1_o_proj_cache_matvec_add(id<MTLCommandBuffer> cmd,
                                                   id<MTLComputePipelineState> pipe,
                                                   id<MTLBuffer> matrix,
                                                   id<MTLBuffer> input,
                                                   id<MTLBuffer> residual,
                                                   id<MTLBuffer> output,
                                                   Context1OProjCacheMatrixInfo info) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create context1 o_proj cache encoder\n");
        return 0;
    }
    uint32_t outDim = info.out_dim;
    uint32_t inDim = info.in_dim;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:matrix offset:0 atIndex:0];
    [enc setBuffer:input offset:0 atIndex:1];
    [enc setBuffer:residual offset:0 atIndex:2];
    [enc setBuffer:output offset:0 atIndex:3];
    [enc setBytes:&outDim length:sizeof(outDim) atIndex:4];
    [enc setBytes:&inDim length:sizeof(inDim) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(outDim, 1, 1)
   threadsPerThreadgroup:threadgroup_1d(pipe, outDim)];
    [enc endEncoding];
    return 1;
}

static int read_span_to_metal_buffer(id<MTLDevice> device,
                                     int fd,
                                     NSString *path,
                                     uint64_t offset,
                                     uint64_t size,
                                     id<MTLBuffer> __strong *outBuffer,
                                     double *readSeconds) {
    if (outBuffer) {
        *outBuffer = nil;
    }
    uint64_t allocBytes = round_up_u64(size, 4096);
    if (allocBytes == 0 || allocBytes > (uint64_t)NSUIntegerMax) {
        fprintf(stderr, "ERROR: span allocation size is invalid\n");
        return 0;
    }
    void *ptr = NULL;
    if (posix_memalign(&ptr, 4096, (size_t)allocBytes) != 0 || !ptr) {
        fprintf(stderr, "ERROR: failed to allocate span buffer\n");
        return 0;
    }
    memset(ptr, 0, (size_t)allocBytes);
    double started = now_seconds();
    int ok = pread_exact_or_report(fd, ptr, size, offset, [path UTF8String]);
    if (readSeconds) {
        *readSeconds += now_seconds() - started;
    }
    if (!ok) {
        free(ptr);
        return 0;
    }
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:ptr
                                                     length:(NSUInteger)allocBytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:^(void *pointer, NSUInteger length) {
                                                    (void)length;
                                                    free(pointer);
                                                }];
    if (!buffer) {
        free(ptr);
        fprintf(stderr, "ERROR: failed to wrap span buffer\n");
        return 0;
    }
    [buffer didModifyRange:NSMakeRange(0, (NSUInteger)size)];
    if (outBuffer) {
        *outBuffer = buffer;
    }
    return 1;
}

static int encode_context1_o_proj_bv_build(id<MTLCommandBuffer> cmd,
                                           id<MTLComputePipelineState> pipe,
                                           id<MTLBuffer> oWeight,
                                           id<MTLBuffer> oScales,
                                           id<MTLBuffer> uWeight,
                                           id<MTLBuffer> uScales,
                                           id<MTLBuffer> output,
                                           ResidentMxfp4MatrixInfo oInfo,
                                           ResidentMxfp4Tensor3DInfo uInfo,
                                           Context1OProjCacheMatrixInfo cacheInfo) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create context1 o_proj cache build encoder\n");
        return 0;
    }
    uint32_t hiddenDim = oInfo.out_dim;
    uint32_t valueDim = oInfo.in_dim;
    uint32_t kvLoraDim = uInfo.dim2;
    uint32_t vHeadDim = uInfo.dim1;
    uint32_t oGroupSize = oInfo.group_size;
    uint32_t uGroupSize = uInfo.group_size;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:oWeight offset:0 atIndex:0];
    [enc setBuffer:oScales offset:0 atIndex:1];
    [enc setBuffer:uWeight offset:0 atIndex:2];
    [enc setBuffer:uScales offset:0 atIndex:3];
    [enc setBuffer:output offset:0 atIndex:4];
    [enc setBytes:&hiddenDim length:sizeof(hiddenDim) atIndex:5];
    [enc setBytes:&valueDim length:sizeof(valueDim) atIndex:6];
    [enc setBytes:&kvLoraDim length:sizeof(kvLoraDim) atIndex:7];
    [enc setBytes:&vHeadDim length:sizeof(vHeadDim) atIndex:8];
    [enc setBytes:&oGroupSize length:sizeof(oGroupSize) atIndex:9];
    [enc setBytes:&uGroupSize length:sizeof(uGroupSize) atIndex:10];
    uint64_t total = (uint64_t)cacheInfo.out_dim * (uint64_t)cacheInfo.in_dim;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
   threadsPerThreadgroup:threadgroup_1d(pipe, (NSUInteger)total)];
    [enc endEncoding];
    return 1;
}

static int run_context1_o_proj_cache_build_layer(id<MTLDevice> device,
                                                 NSString *residentLayoutPath,
                                                 NSString *cacheLayoutPath,
                                                 NSString *cacheFileOverridePath,
                                                 int layerId,
                                                 uint64_t maxCacheReadBytes,
                                                 double maxBuildGFMA,
                                                 uint64_t maxLiveWorkingSetBytes,
                                                 double maxLiveWorkingSetMiB,
                                                 Context1OProjCacheBuildStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (layerId < 0) {
        fprintf(stderr, "ERROR: --probe-layer is required for context1 cache builder\n");
        return 0;
    }
    NSDictionary *residentLayout = load_json_dictionary(residentLayoutPath);
    NSString *residentDir = dirname_string(residentLayoutPath);
    NSString *weightFile = residentLayout[@"weight_file"];
    if (![weightFile isKindOfClass:[NSString class]] || [weightFile length] == 0) {
        fprintf(stderr, "ERROR: resident layout missing weight_file\n");
        return 0;
    }
    NSString *residentBinPath = join_path(residentDir, weightFile);
    ResidentMxfp4MatrixInfo oInfo = {0};
    ResidentMxfp4Tensor3DInfo uInfo = {0};
    NSString *oName = [NSString stringWithFormat:@"model.layers.%d.self_attn.o_proj.weight", layerId];
    NSString *uName = [NSString stringWithFormat:@"model.layers.%d.self_attn.unembed_out.weight", layerId];
    if (!find_resident_mxfp4_matrix_info(residentLayout, oName, &oInfo) ||
        !find_resident_mxfp4_tensor3d_info(residentLayout, uName, &uInfo)) {
        return 0;
    }
    Context1OProjCacheMatrixInfo cacheInfo = {0};
    NSString *cacheFilePath = nil;
    if (!parse_context1_o_proj_cache_matrix_info(cacheLayoutPath,
                                                 cacheFileOverridePath,
                                                 (uint64_t)layerId,
                                                 maxCacheReadBytes,
                                                 &cacheInfo,
                                                 &cacheFilePath)) {
        return 0;
    }
    if (cacheFileOverridePath) {
        cacheFilePath = cacheFileOverridePath;
    }
    if (oInfo.out_dim != cacheInfo.out_dim ||
        oInfo.in_dim != uInfo.dim0 * uInfo.dim1 ||
        uInfo.dim2 != cacheInfo.in_dim) {
        fprintf(stderr,
                "ERROR: context1 cache builder dims mismatch o=(%u,%u) unembed=(%u,%u,%u) cache=(%u,%u)\n",
                oInfo.out_dim,
                oInfo.in_dim,
                uInfo.dim0,
                uInfo.dim1,
                uInfo.dim2,
                cacheInfo.out_dim,
                cacheInfo.in_dim);
        return 0;
    }
    uint64_t fma = 0;
    uint64_t tmp = 0;
    if (!checked_mul_u64((uint64_t)oInfo.out_dim, (uint64_t)oInfo.in_dim, &tmp) ||
        !checked_mul_u64(tmp, (uint64_t)cacheInfo.in_dim, &fma)) {
        fprintf(stderr, "ERROR: context1 cache builder FMA count overflows\n");
        return 0;
    }
    double gfma = (double)fma / 1.0e9;
    if (!isfinite(maxBuildGFMA) || maxBuildGFMA <= 0.0 || gfma > maxBuildGFMA) {
        fprintf(stderr,
                "ERROR: context1 cache builder requires %.6f GFMA, exceeds cap %.6f; raise --max-context1-o-proj-build-gfma intentionally\n",
                gfma,
                maxBuildGFMA);
        return 0;
    }
    uint64_t sourceBytes = 0;
    if (!checked_add_u64(oInfo.total_bytes, uInfo.total_bytes, &sourceBytes)) {
        fprintf(stderr, "ERROR: context1 cache builder source bytes overflow\n");
        return 0;
    }
    if (maxCacheReadBytes > 0 && sourceBytes > maxCacheReadBytes) {
        fprintf(stderr,
                "ERROR: context1 cache builder source bytes %llu exceed --max-cache-read-mib limit %llu\n",
                (unsigned long long)sourceBytes,
                (unsigned long long)maxCacheReadBytes);
        return 0;
    }
    stats->source_bytes_read = sourceBytes;
    stats->cache_bytes_written = cacheInfo.size;
    stats->fma_count = fma;
    stats->hidden_dim = oInfo.out_dim;
    stats->attention_value_dim = oInfo.in_dim;
    stats->kv_lora_dim = cacheInfo.in_dim;
    uint64_t estimatedLiveBytes = 0;
    if (!checked_add_u64(sourceBytes, cacheInfo.size, &estimatedLiveBytes)) {
        fprintf(stderr, "ERROR: context1 cache builder live bytes overflow\n");
        return 0;
    }
    stats->estimated_live_working_set_bytes = estimatedLiveBytes;
    stats->max_live_working_set_mib = maxLiveWorkingSetMiB;
    stats->live_working_set_ok =
        (maxLiveWorkingSetBytes == 0 || estimatedLiveBytes <= maxLiveWorkingSetBytes);
    if (!stats->live_working_set_ok) {
        fprintf(stderr,
                "ERROR: context1 cache builder estimated live bytes %llu exceed --max-live-working-set-mib %.6f\n",
                (unsigned long long)estimatedLiveBytes,
                maxLiveWorkingSetMiB);
        return 0;
    }
    int fd = open([residentBinPath fileSystemRepresentation], O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: failed to open resident file %s: %s\n",
                [residentBinPath UTF8String],
                strerror(errno));
        return 0;
    }
    id<MTLBuffer> oWeight = nil;
    id<MTLBuffer> oScales = nil;
    id<MTLBuffer> uWeight = nil;
    id<MTLBuffer> uScales = nil;
    int readOk =
        read_span_to_metal_buffer(device, fd, residentBinPath, oInfo.weight.offset,
                                  oInfo.weight.size, &oWeight, &stats->read_seconds) &&
        read_span_to_metal_buffer(device, fd, residentBinPath, oInfo.scales.offset,
                                  oInfo.scales.size, &oScales, &stats->read_seconds) &&
        read_span_to_metal_buffer(device, fd, residentBinPath, uInfo.weight.offset,
                                  uInfo.weight.size, &uWeight, &stats->read_seconds) &&
        read_span_to_metal_buffer(device, fd, residentBinPath, uInfo.scales.offset,
                                  uInfo.scales.size, &uScales, &stats->read_seconds);
    close(fd);
    if (!readOk) {
        return 0;
    }
    id<MTLLibrary> library = make_glm_moe_library(device);
    NSString *kernelName = cacheInfo.dtype_bytes == 2
        ? @"glm_context1_o_proj_bv_build_bf16"
        : @"glm_context1_o_proj_bv_build_f32";
    id<MTLComputePipelineState> pipe =
        library ? make_glm_pipeline(device, library, kernelName) : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)cacheInfo.size
                                               options:MTLResourceStorageModeShared];
    if (!library || !pipe || !queue || !output) {
        fprintf(stderr, "ERROR: failed to allocate context1 cache builder Metal resources\n");
        return 0;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create context1 cache builder command buffer\n");
        return 0;
    }
    if (!encode_context1_o_proj_bv_build(cmd,
                                         pipe,
                                         oWeight,
                                         oScales,
                                         uWeight,
                                         uScales,
                                         output,
                                         oInfo,
                                         uInfo,
                                         cacheInfo)) {
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    stats->kernel_seconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: context1 cache builder command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    int outFd = open([cacheFilePath fileSystemRepresentation], O_RDWR);
    if (outFd < 0) {
        fprintf(stderr, "ERROR: failed to open context1 cache output %s: %s\n",
                [cacheFilePath UTF8String],
                strerror(errno));
        return 0;
    }
    double writeStarted = now_seconds();
    uint64_t copied = 0;
    const uint8_t *src = (const uint8_t *)[output contents];
    while (copied < cacheInfo.size) {
        uint64_t toWrite = cacheInfo.size - copied;
        if (toWrite > 8 * 1024 * 1024) {
            toWrite = 8 * 1024 * 1024;
        }
        ssize_t written = pwrite(outFd,
                                 src + copied,
                                 (size_t)toWrite,
                                 (off_t)(cacheInfo.offset + copied));
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            break;
        }
        copied += (uint64_t)written;
    }
    close(outFd);
    stats->write_seconds = now_seconds() - writeStarted;
    if (copied != cacheInfo.size) {
        fprintf(stderr, "ERROR: failed to write context1 cache output layer\n");
        return 0;
    }
    stats->ok = 1;
    stats->source_bytes_read = sourceBytes;
    stats->cache_bytes_written = cacheInfo.size;
    stats->fma_count = fma;
    stats->hidden_dim = oInfo.out_dim;
    stats->attention_value_dim = oInfo.in_dim;
    stats->kv_lora_dim = cacheInfo.in_dim;
    stats->elapsed_seconds =
        stats->read_seconds + stats->kernel_seconds + stats->write_seconds;
    if (cacheInfo.dtype_bytes == 2) {
        stats->output0 = bf16_to_float_cpu(((uint16_t *)[output contents])[0]);
    } else {
        stats->output0 = ((float *)[output contents])[0];
    }
    return 1;
}

static int run_context1_o_proj_cache_output_probe(id<MTLDevice> device,
                                                  NSString *cacheLayoutPath,
                                                  NSString *cacheFileOverridePath,
                                                  int layerId,
                                                  NSString *inputPath,
                                                  NSData *inputDataOverride,
                                                  NSString *residualPath,
                                                  NSData *residualDataOverride,
                                                  NSString *outputPath,
                                                  NSData **outputDataOut,
                                                  id<MTLBuffer> __strong *outputBufferOut,
                                                  uint64_t maxCacheReadBytes,
                                                  AttnOutputProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (outputBufferOut) {
        *outputBufferOut = nil;
    }
    if (layerId < 0) {
        fprintf(stderr, "ERROR: --probe-layer is required for context1 cache output\n");
        return 0;
    }
    Context1OProjCacheMatrixInfo info = {0};
    NSString *cacheFilePath = nil;
    if (!parse_context1_o_proj_cache_matrix_info(cacheLayoutPath,
                                                 cacheFileOverridePath,
                                                 (uint64_t)layerId,
                                                 maxCacheReadBytes,
                                                 &info,
                                                 &cacheFilePath)) {
        return 0;
    }
    stats->out_dim = info.out_dim;
    stats->in_dim = info.in_dim;
    stats->group_size = 0;
    stats->input_bytes = (uint64_t)info.in_dim * sizeof(float);
    stats->residual_bytes = (uint64_t)info.out_dim * sizeof(float);
    stats->projection_bytes = stats->residual_bytes;
    stats->output_bytes = stats->residual_bytes;
    if (stats->input_bytes > NSUIntegerMax ||
        stats->residual_bytes > NSUIntegerMax ||
        stats->output_bytes > NSUIntegerMax ||
        info.size > NSUIntegerMax) {
        fprintf(stderr, "ERROR: context1 o_proj cache buffers exceed NSUIntegerMax\n");
        return 0;
    }
    NSData *inputData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
    NSData *residualData =
        residualDataOverride ?: [NSData dataWithContentsOfFile:residualPath];
    if (!inputData || !residualData) {
        fprintf(stderr, "ERROR: failed to read context1 cache input or residual\n");
        return 0;
    }
    if ((uint64_t)[inputData length] != stats->input_bytes ||
        (uint64_t)[residualData length] != stats->residual_bytes) {
        fprintf(stderr, "ERROR: context1 cache input/residual bytes mismatch\n");
        return 0;
    }
    void *matrixPtr = NULL;
    if (posix_memalign(&matrixPtr, 4096, (size_t)info.size) != 0 || !matrixPtr) {
        fprintf(stderr, "ERROR: failed to allocate context1 o_proj cache matrix buffer\n");
        return 0;
    }
    int fd = open([cacheFilePath fileSystemRepresentation], O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: failed to open context1 cache file %s: %s\n",
                [cacheFilePath UTF8String],
                strerror(errno));
        free(matrixPtr);
        return 0;
    }
    double readStarted = now_seconds();
    int readOk = pread_exact_or_report(fd,
                                       matrixPtr,
                                       info.size,
                                       info.offset,
                                       [cacheFilePath UTF8String]);
    close(fd);
    stats->read_seconds = now_seconds() - readStarted;
    if (!readOk) {
        free(matrixPtr);
        return 0;
    }
    stats->bytes_read = info.size;

    id<MTLLibrary> library = make_glm_moe_library(device);
    NSString *kernelName = info.dtype_bytes == 2
        ? @"glm_context1_o_proj_bf16_matvec_add"
        : @"glm_context1_o_proj_f32_matvec_add";
    id<MTLComputePipelineState> pipe =
        library ? make_glm_pipeline(device, library, kernelName) : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> matrix = [device newBufferWithBytesNoCopy:matrixPtr
                                                     length:(NSUInteger)info.size
                                                    options:MTLResourceStorageModeShared
                                                deallocator:^(void *pointer, NSUInteger length) {
                                                    (void)length;
                                                    free(pointer);
                                                }];
    if (!matrix) {
        free(matrixPtr);
        fprintf(stderr, "ERROR: failed to wrap context1 cache matrix buffer\n");
        return 0;
    }
    [matrix didModifyRange:NSMakeRange(0, (NSUInteger)info.size)];
    id<MTLBuffer> input = [device newBufferWithBytes:[inputData bytes]
                                              length:(NSUInteger)stats->input_bytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> residual = [device newBufferWithBytes:[residualData bytes]
                                                 length:(NSUInteger)stats->residual_bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)stats->output_bytes
                                               options:MTLResourceStorageModeShared];
    if (!library || !pipe || !queue || !input || !residual || !output) {
        fprintf(stderr, "ERROR: failed to allocate context1 o_proj cache Metal resources\n");
        return 0;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create context1 o_proj cache command buffer\n");
        return 0;
    }
    if (!encode_context1_o_proj_cache_matvec_add(cmd,
                                                 pipe,
                                                 matrix,
                                                 input,
                                                 residual,
                                                 output,
                                                 info)) {
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    stats->projection_kernel_seconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: context1 o_proj cache command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    double writeStarted = now_seconds();
    NSData *outputData = [NSData dataWithBytes:[output contents]
                                        length:(NSUInteger)stats->output_bytes];
    if (outputPath && ![outputData writeToFile:outputPath atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write context1 o_proj cache output %s\n",
                [outputPath UTF8String]);
        return 0;
    }
    if (outputDataOut) {
        *outputDataOut = outputData;
    }
    if (outputBufferOut) {
        *outputBufferOut = output;
    }
    stats->output_write_seconds = now_seconds() - writeStarted;
    stats->elapsed_seconds = stats->read_seconds +
                             stats->projection_kernel_seconds +
                             stats->output_write_seconds;
    stats->fused_matvec_add = 1;
    stats->context1_o_proj_cache = 1;
    stats->command_buffer_count = 1;
    stats->output0 = ((float *)[output contents])[0];
    stats->ok = 1;
    return 1;
}

static int run_attention_output_probe(id<MTLDevice> device,
                                      NSString *residentBinPath,
                                      id<MTLBuffer> residentMetalBuffer,
                                      ResidentMxfp4MatrixInfo info,
                                      NSString *inputPath,
                                      NSData *inputDataOverride,
                                      NSString *residualPath,
                                      NSData *residualDataOverride,
                                      NSString *projectionPath,
                                      NSString *outputPath,
                                      NSData **outputDataOut,
                                      id<MTLBuffer> __strong *outputBufferOut,
                                      AttnOutputProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (outputBufferOut) {
        *outputBufferOut = nil;
    }
    stats->out_dim = info.out_dim;
    stats->in_dim = info.in_dim;
    stats->group_size = info.group_size;
    stats->input_bytes = (uint64_t)info.in_dim * sizeof(float);
    stats->residual_bytes = (uint64_t)info.out_dim * sizeof(float);
    stats->projection_bytes = stats->residual_bytes;
    stats->output_bytes = stats->residual_bytes;

    if (stats->input_bytes > NSUIntegerMax ||
        stats->residual_bytes > NSUIntegerMax ||
        stats->output_bytes > NSUIntegerMax) {
        fprintf(stderr, "ERROR: attention-output buffers exceed NSUIntegerMax\n");
        return 0;
    }
    NSData *residualData =
        residualDataOverride ?: [NSData dataWithContentsOfFile:residualPath];
    if (!residualData) {
        fprintf(stderr, "ERROR: failed to read attention-output residual %s\n",
                residualPath ? [residualPath UTF8String] : "<memory>");
        return 0;
    }
    if ((uint64_t)[residualData length] != stats->residual_bytes) {
        fprintf(stderr,
                "ERROR: attention-output residual bytes %llu do not match expected %llu\n",
                (unsigned long long)[residualData length],
                (unsigned long long)stats->residual_bytes);
        return 0;
    }

    if (projectionPath) {
        ResidentLinearProbeStats linearStats = {0};
        NSData *projectionData = nil;
        if (!run_resident_linear_probe(device,
                                       residentBinPath,
                                       residentMetalBuffer,
                                       info,
                                       inputPath,
                                       inputDataOverride,
                                       projectionPath,
                                       &projectionData,
                                       0,
                                       0.0,
                                       &linearStats)) {
            return 0;
        }
        if (!projectionData || (uint64_t)[projectionData length] != stats->projection_bytes) {
            fprintf(stderr, "ERROR: attention-output projection bytes are invalid\n");
            return 0;
        }

        id<MTLLibrary> library = make_glm_moe_library(device);
        id<MTLComputePipelineState> addPipe =
            library ? make_glm_pipeline(device, library, @"glm_add_inplace_f32") : nil;
        id<MTLCommandQueue> queue = shared_glm_command_queue(device);
        id<MTLBuffer> residual = [device newBufferWithBytes:[residualData bytes]
                                                     length:(NSUInteger)stats->residual_bytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [device newBufferWithBytes:[projectionData bytes]
                                                   length:(NSUInteger)stats->output_bytes
                                                  options:MTLResourceStorageModeShared];
        if (!library || !addPipe || !queue || !residual || !output) {
            fprintf(stderr, "ERROR: failed to allocate attention-output Metal resources\n");
            return 0;
        }
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) {
            fprintf(stderr, "ERROR: failed to create attention-output command buffer\n");
            return 0;
        }
        if (!encode_glm_add_inplace_f32(cmd, addPipe, residual, output, info.out_dim)) {
            return 0;
        }
        double addStarted = now_seconds();
        [cmd commit];
        [cmd waitUntilCompleted];
        stats->residual_add_seconds = now_seconds() - addStarted;
        if (cmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr,
                    "ERROR: attention-output residual add failed: %s\n",
                    cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
            return 0;
        }
        double writeStarted = now_seconds();
        NSData *outputData = [NSData dataWithBytes:[output contents]
                                            length:(NSUInteger)stats->output_bytes];
        if (outputPath && ![outputData writeToFile:outputPath atomically:YES]) {
            fprintf(stderr, "ERROR: failed to write attention-output output %s\n",
                    [outputPath UTF8String]);
            return 0;
        }
        if (outputDataOut) {
            *outputDataOut = outputData;
        }
        if (outputBufferOut) {
            *outputBufferOut = output;
        }
        stats->output_write_seconds = now_seconds() - writeStarted;
        stats->bytes_read = linearStats.bytes_read;
        stats->read_seconds = linearStats.read_seconds;
        stats->projection_kernel_seconds = linearStats.kernel_seconds;
        stats->projection_write_seconds = linearStats.output_write_seconds;
        stats->resident_mmap_backed = linearStats.resident_mmap_backed;
        stats->command_buffer_count = 2;
        stats->fused_matvec_add = 0;
        stats->elapsed_seconds = stats->read_seconds +
                                 stats->projection_kernel_seconds +
                                 stats->projection_write_seconds +
                                 stats->residual_add_seconds +
                                 stats->output_write_seconds;
        stats->output0 = ((float *)[output contents])[0];
        stats->ok = 1;
        return 1;
    }

    NSData *inputData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
    if (!inputData) {
        fprintf(stderr, "ERROR: failed to read attention-output input %s\n",
                inputPath ? [inputPath UTF8String] : "<memory>");
        return 0;
    }
    if ((uint64_t)[inputData length] != stats->input_bytes) {
        fprintf(stderr,
                "ERROR: attention-output input bytes %llu do not match expected %llu\n",
                (unsigned long long)[inputData length],
                (unsigned long long)stats->input_bytes);
        return 0;
    }
    int useResidentMetalWeights = residentMetalBuffer != nil;
    NSUInteger weightOffset = 0;
    NSUInteger scalesOffset = (NSUInteger)info.weight.size;
    void *matrixPtr = NULL;
    uint64_t allocBytes = useResidentMetalWeights
        ? 0
        : round_up_u64(info.total_bytes, 2 * 1024 * 1024);
    if (useResidentMetalWeights) {
        uint64_t residentLength = (uint64_t)[residentMetalBuffer length];
        uint64_t weightEnd = 0;
        uint64_t scalesEnd = 0;
        if (!checked_add_u64(info.weight.offset, info.weight.size, &weightEnd) ||
            !checked_add_u64(info.scales.offset, info.scales.size, &scalesEnd) ||
            info.weight.offset > (uint64_t)NSUIntegerMax ||
            info.scales.offset > (uint64_t)NSUIntegerMax ||
            weightEnd > residentLength ||
            scalesEnd > residentLength) {
            fprintf(stderr,
                    "ERROR: resident Metal buffer does not cover attention-output weights\n");
            return 0;
        }
        weightOffset = (NSUInteger)info.weight.offset;
        scalesOffset = (NSUInteger)info.scales.offset;
        stats->resident_mmap_backed = 1;
        stats->bytes_read = 0;
    } else {
        if (posix_memalign(&matrixPtr, 2 * 1024 * 1024, (size_t)allocBytes) != 0 ||
            !matrixPtr) {
            fprintf(stderr, "ERROR: failed to allocate fused attention-output matrix buffer\n");
            return 0;
        }
        int closeFd = 0;
        int fd = open_resident_read_fd(residentBinPath, &closeFd);
        if (fd < 0) {
            fprintf(stderr,
                    "ERROR: failed to open resident file for fused attention-output %s: %s\n",
                    [residentBinPath UTF8String],
                    strerror(errno));
            free(matrixPtr);
            return 0;
        }
        double readStarted = now_seconds();
        int readOk =
            pread_exact_or_report(fd,
                                  matrixPtr,
                                  info.weight.size,
                                  info.weight.offset,
                                  [residentBinPath UTF8String]) &&
            pread_exact_or_report(fd,
                                  (uint8_t *)matrixPtr + info.weight.size,
                                  info.scales.size,
                                  info.scales.offset,
                                  [residentBinPath UTF8String]);
        close_resident_read_fd(fd, closeFd);
        stats->read_seconds = now_seconds() - readStarted;
        if (!readOk) {
            free(matrixPtr);
            return 0;
        }
        stats->bytes_read = info.total_bytes;
    }

    id<MTLLibrary> library = make_glm_moe_library(device);
    id<MTLComputePipelineState> matvecAddPipe =
        library ? make_resident_mxfp4_matvec_add_pipeline(device, library, info) : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> matrix = useResidentMetalWeights
        ? residentMetalBuffer
        : [device newBufferWithBytesNoCopy:matrixPtr
                                    length:(NSUInteger)allocBytes
                                   options:MTLResourceStorageModeShared
                               deallocator:^(void *pointer, NSUInteger length) {
                                   (void)length;
                                   free(pointer);
                               }];
    if (!matrix) {
        free(matrixPtr);
        fprintf(stderr, "ERROR: failed to wrap fused attention-output matrix buffer\n");
        return 0;
    }
    if (!useResidentMetalWeights) {
        [matrix didModifyRange:NSMakeRange(0, (NSUInteger)info.total_bytes)];
    }
    id<MTLBuffer> input = [device newBufferWithBytes:[inputData bytes]
                                              length:(NSUInteger)stats->input_bytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> residual = [device newBufferWithBytes:[residualData bytes]
                                                 length:(NSUInteger)stats->residual_bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)stats->output_bytes
                                               options:MTLResourceStorageModeShared];
    if (!library || !matvecAddPipe || !queue || !input || !residual || !output) {
        fprintf(stderr, "ERROR: failed to allocate fused attention-output Metal resources\n");
        return 0;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create fused attention-output command buffer\n");
        return 0;
    }
    if (!encode_glm_mxfp4_matvec_add_with_offsets(cmd,
                                                  matvecAddPipe,
                                                  matrix,
                                                  weightOffset,
                                                  matrix,
                                                  scalesOffset,
                                                  info,
                                                  input,
                                                  residual,
                                                  output)) {
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    double fusedKernelSeconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: fused attention-output command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    stats->projection_kernel_seconds = fusedKernelSeconds;
    stats->residual_add_seconds = 0.0;
    stats->command_buffer_count = 1;
    stats->fused_matvec_add = 1;
    double writeStarted = now_seconds();
    NSData *outputData = [NSData dataWithBytes:[output contents]
                                        length:(NSUInteger)stats->output_bytes];
    if (outputPath && ![outputData writeToFile:outputPath atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write attention-output output %s\n",
                [outputPath UTF8String]);
        return 0;
    }
    if (outputDataOut) {
        *outputDataOut = outputData;
    }
    if (outputBufferOut) {
        *outputBufferOut = output;
    }
    stats->output_write_seconds = now_seconds() - writeStarted;
    stats->elapsed_seconds = stats->read_seconds +
                             stats->projection_kernel_seconds +
                             stats->output_write_seconds;
    stats->output0 = ((float *)[output contents])[0];
    stats->ok = 1;
    return 1;
}

static NSString *layer_tensor_name(int layerId, NSString *suffix) {
    return [NSString stringWithFormat:@"model.layers.%d%@", layerId, suffix];
}

static uint64_t resident_linear_scratch_bytes(ResidentMxfp4MatrixInfo info) {
    return round_up_u64(info.total_bytes, 2 * 1024 * 1024) +
           (uint64_t)info.in_dim * sizeof(float) +
           (uint64_t)info.out_dim * sizeof(float);
}

static uint64_t resident_linear_scratch_bytes_for_backing(ResidentMxfp4MatrixInfo info,
                                                          int residentMmapBacked) {
    uint64_t matrixBytes = residentMmapBacked
        ? 0
        : round_up_u64(info.total_bytes, 2 * 1024 * 1024);
    return matrixBytes +
           (uint64_t)info.in_dim * sizeof(float) +
           (uint64_t)info.out_dim * sizeof(float);
}

static int use_fast_resident_mxfp4_matvec(ResidentMxfp4MatrixInfo info) {
    const char *flagName = "LARGERLM_GLM_MOE_INFER_FAST_RESIDENT_MXFP4";
    if (env_flag_disabled(flagName)) {
        return 0;
    }
    int explicitlyRequested = env_flag_enabled(flagName);
    if (info.group_size != 32 || (info.in_dim % 32u) != 0u || info.out_dim == 0) {
        if (explicitlyRequested) {
            fprintf(stderr,
                    "WARNING: fast resident MXFP4 matvec requested but unsupported for out=%u in=%u group=%u; using scalar kernel\n",
                    info.out_dim,
                    info.in_dim,
                    info.group_size);
        }
        return 0;
    }
    if (explicitlyRequested) {
        return 1;
    }
    return strstr(info.name, ".self_attn.o_proj.weight") != NULL;
}

static int use_tiled_resident_mxfp4_matvec_add(ResidentMxfp4MatrixInfo info) {
    if (!env_flag_enabled("LARGERLM_GLM_MOE_INFER_TILED_ATTN_OUTPUT_MXFP4")) {
        return 0;
    }
    return use_fast_resident_mxfp4_matvec(info) && info.in_dim > 6144u;
}

static id<MTLComputePipelineState> make_resident_mxfp4_matvec_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    ResidentMxfp4MatrixInfo info) {
    NSString *name = use_fast_resident_mxfp4_matvec(info)
        ? @"glm_mxfp4_matvec_gs32_simd"
        : @"glm_mxfp4_matvec";
    return make_glm_pipeline(device, library, name);
}

static id<MTLComputePipelineState> make_resident_mxfp4_matvec_add_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    ResidentMxfp4MatrixInfo info) {
    NSString *name = use_tiled_resident_mxfp4_matvec_add(info)
        ? @"glm_mxfp4_matvec_add_gs32_tiled"
        : (use_fast_resident_mxfp4_matvec(info)
            ? @"glm_mxfp4_matvec_add_gs32_simd"
            : @"glm_mxfp4_matvec_add");
    return make_glm_pipeline(device, library, name);
}

static uint64_t dense_mlp_scratch_bytes(DenseMlpMxfp4Info info,
                                        ResidentVectorInfo normInfo) {
    uint64_t normBytes =
        (uint64_t)normInfo.dim * sizeof(float) * 3u + normInfo.size;
    uint64_t activationBytes =
        (uint64_t)info.hidden_dim * sizeof(float) * 2u +
        (uint64_t)info.intermediate_dim * sizeof(float) * 3u;
    return normBytes +
           resident_linear_scratch_bytes(info.gate) +
           resident_linear_scratch_bytes(info.up) +
           resident_linear_scratch_bytes(info.down) +
           activationBytes;
}

static int encode_glm_swiglu_f32(id<MTLCommandBuffer> cmd,
                                 id<MTLComputePipelineState> pipe,
                                 id<MTLBuffer> gate,
                                 id<MTLBuffer> up,
                                 id<MTLBuffer> out,
                                 uint32_t dim) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create SwiGLU command encoder\n");
        return 0;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:gate offset:0 atIndex:0];
    [enc setBuffer:up offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&dim length:sizeof(dim) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
   threadsPerThreadgroup:threadgroup_1d(pipe, dim)];
    [enc endEncoding];
    return 1;
}

static int run_dense_mlp_fused_probe(id<MTLDevice> device,
                                     NSString *residentBinPath,
                                     DenseMlpMxfp4Info info,
                                     NSString *inputPath,
                                     NSData *inputDataOverride,
                                     id<MTLBuffer> inputBufferOverride,
                                     ResidentVectorInfo postAttnNormInfo,
                                     NSString *outputPath,
                                     NSData **outputDataOut,
                                     id<MTLBuffer> __strong *outputBufferOut,
                                     float rmsNormEps,
                                     int expectOutput0Set,
                                     double expectOutput0,
                                     int allowAsyncSubmit,
                                     id<MTLCommandBuffer> __strong *pendingCommandOut,
                                     DenseMlpProbeStats *stats) {
    uint64_t hiddenBytes = (uint64_t)info.hidden_dim * sizeof(float);
    uint64_t intermediateBytes = (uint64_t)info.intermediate_dim * sizeof(float);
    if (hiddenBytes > NSUIntegerMax ||
        intermediateBytes > NSUIntegerMax ||
        postAttnNormInfo.dim != info.hidden_dim ||
        info.gate.in_dim != info.hidden_dim ||
        info.up.in_dim != info.hidden_dim ||
        info.down.out_dim != info.hidden_dim ||
        info.gate.out_dim != info.intermediate_dim ||
        info.up.out_dim != info.intermediate_dim ||
        info.down.in_dim != info.intermediate_dim) {
        fprintf(stderr, "ERROR: fused dense MLP dimensions are inconsistent\n");
        return 0;
    }
    if (outputBufferOut) {
        *outputBufferOut = nil;
    }
    if (pendingCommandOut) {
        *pendingCommandOut = nil;
    }
    NSData *residualData = nil;
    if (inputBufferOverride) {
        if ((uint64_t)[inputBufferOverride length] < hiddenBytes) {
            fprintf(stderr,
                    "ERROR: dense MLP input buffer bytes %llu are smaller than hidden bytes %llu\n",
                    (unsigned long long)[inputBufferOverride length],
                    (unsigned long long)hiddenBytes);
            return 0;
        }
    } else {
        residualData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
        if (!residualData) {
            fprintf(stderr, "ERROR: failed to read dense MLP input %s\n",
                    inputPath ? [inputPath UTF8String] : "<memory>");
            return 0;
        }
        if ((uint64_t)[residualData length] != hiddenBytes) {
            fprintf(stderr,
                    "ERROR: dense MLP input bytes %llu do not match hidden bytes %llu\n",
                    (unsigned long long)[residualData length],
                    (unsigned long long)hiddenBytes);
            return 0;
        }
    }

    double totalStarted = now_seconds();
    float *normWeightValues = NULL;
    uint64_t normWeightBytesRead = 0;
    if (!read_resident_vector_f32(residentBinPath,
                                  postAttnNormInfo,
                                  &normWeightValues,
                                  &normWeightBytesRead)) {
        return 0;
    }

    void *gatePtr = NULL;
    void *upPtr = NULL;
    void *downPtr = NULL;
    uint64_t gateAllocBytes = 0;
    uint64_t upAllocBytes = 0;
    uint64_t downAllocBytes = 0;
    int closeFd = 0;
    int fd = open_resident_read_fd(residentBinPath, &closeFd);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open resident file for fused dense MLP %s: %s\n",
                [residentBinPath UTF8String],
                strerror(errno));
        free(normWeightValues);
        return 0;
    }
    int stagedOk =
        stage_resident_mxfp4_matrix_from_fd(fd,
                                            residentBinPath,
                                            info.gate,
                                            &gatePtr,
                                            &gateAllocBytes,
                                            &stats->gate_read_seconds) &&
        stage_resident_mxfp4_matrix_from_fd(fd,
                                            residentBinPath,
                                            info.up,
                                            &upPtr,
                                            &upAllocBytes,
                                            &stats->up_read_seconds) &&
        stage_resident_mxfp4_matrix_from_fd(fd,
                                            residentBinPath,
                                            info.down,
                                            &downPtr,
                                            &downAllocBytes,
                                            &stats->down_read_seconds);
    close_resident_read_fd(fd, closeFd);
    if (!stagedOk) {
        free(normWeightValues);
        free(gatePtr);
        free(upPtr);
        free(downPtr);
        return 0;
    }

    id<MTLLibrary> library = make_glm_moe_library(device);
    id<MTLComputePipelineState> rmsPipe =
        library ? make_glm_pipeline(device, library, @"glm_rmsnorm_f32") : nil;
    id<MTLComputePipelineState> matvecPipe =
        library ? make_resident_mxfp4_matvec_pipeline(device, library, info.gate) : nil;
    id<MTLComputePipelineState> swigluPipe =
        library ? make_glm_pipeline(device, library, @"glm_swiglu_f32") : nil;
    id<MTLComputePipelineState> addPipe =
        library ? make_glm_pipeline(device, library, @"glm_add_inplace_f32") : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> input = inputBufferOverride ?:
        [device newBufferWithBytes:[residualData bytes]
                            length:(NSUInteger)hiddenBytes
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> normWeight = [device newBufferWithBytes:normWeightValues
                                                   length:(NSUInteger)hiddenBytes
                                                  options:MTLResourceStorageModeShared];
    free(normWeightValues);
    id<MTLBuffer> normed = [device newBufferWithLength:(NSUInteger)hiddenBytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> gateMatrix = [device newBufferWithBytesNoCopy:gatePtr
                                                         length:(NSUInteger)gateAllocBytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:^(void *pointer, NSUInteger length) {
                                                        (void)length;
                                                        free(pointer);
                                                    }];
    if (gateMatrix) {
        gatePtr = NULL;
    }
    id<MTLBuffer> upMatrix = [device newBufferWithBytesNoCopy:upPtr
                                                       length:(NSUInteger)upAllocBytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:^(void *pointer, NSUInteger length) {
                                                      (void)length;
                                                      free(pointer);
                                                  }];
    if (upMatrix) {
        upPtr = NULL;
    }
    id<MTLBuffer> downMatrix = [device newBufferWithBytesNoCopy:downPtr
                                                         length:(NSUInteger)downAllocBytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:^(void *pointer, NSUInteger length) {
                                                        (void)length;
                                                        free(pointer);
                                                    }];
    if (downMatrix) {
        downPtr = NULL;
    }
    id<MTLBuffer> gateOut = [device newBufferWithLength:(NSUInteger)intermediateBytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> upOut = [device newBufferWithLength:(NSUInteger)intermediateBytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> act = [device newBufferWithLength:(NSUInteger)intermediateBytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)hiddenBytes
                                               options:MTLResourceStorageModeShared];
    if (!library || !rmsPipe || !matvecPipe || !swigluPipe || !addPipe || !queue ||
        !input || !normWeight || !normed || !gateMatrix || !upMatrix ||
        !downMatrix || !gateOut || !upOut || !act || !output) {
        fprintf(stderr, "ERROR: failed to allocate fused dense MLP Metal resources\n");
        free(gatePtr);
        free(upPtr);
        free(downPtr);
        return 0;
    }
    [gateMatrix didModifyRange:NSMakeRange(0, (NSUInteger)info.gate.total_bytes)];
    [upMatrix didModifyRange:NSMakeRange(0, (NSUInteger)info.up.total_bytes)];
    [downMatrix didModifyRange:NSMakeRange(0, (NSUInteger)info.down.total_bytes)];

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create fused dense MLP command buffer\n");
        return 0;
    }
    if (!encode_glm_rmsnorm(cmd,
                            rmsPipe,
                            input,
                            normWeight,
                            normed,
                            info.hidden_dim,
                            rmsNormEps) ||
        !encode_glm_mxfp4_matvec(cmd,
                                 matvecPipe,
                                 gateMatrix,
                                 info.gate,
                                 normed,
                                 gateOut) ||
        !encode_glm_mxfp4_matvec(cmd,
                                 matvecPipe,
                                 upMatrix,
                                 info.up,
                                 normed,
                                 upOut) ||
        !encode_glm_swiglu_f32(cmd,
                               swigluPipe,
                               gateOut,
                               upOut,
                               act,
                               info.intermediate_dim) ||
        !encode_glm_mxfp4_matvec(cmd,
                                 matvecPipe,
                                 downMatrix,
                                 info.down,
                                 act,
                                 output) ||
        !encode_glm_add_inplace_f32(cmd, addPipe, input, output, info.hidden_dim)) {
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    stats->bytes_read =
        normWeightBytesRead +
        info.gate.total_bytes +
        info.up.total_bytes +
        info.down.total_bytes;
    stats->rmsnorm_elapsed_seconds = 0.0;
    stats->gate_kernel_seconds = 0.0;
    stats->up_kernel_seconds = 0.0;
    stats->swiglu_kernel_seconds = 0.0;
    stats->down_kernel_seconds = 0.0;
    stats->residual_add_seconds = 0.0;
    stats->fused_pipeline = 1;
    stats->command_buffer_count = 1;
    int canSubmitAsync =
        allowAsyncSubmit &&
        pendingCommandOut &&
        outputBufferOut &&
        outputDataOut == NULL &&
        outputPath == nil &&
        !expectOutput0Set;
    if (canSubmitAsync) {
        stats->async_submitted = 1;
        stats->fused_kernel_seconds = now_seconds() - kernelStarted;
        stats->output0_check_ok = 1;
        stats->elapsed_seconds = now_seconds() - totalStarted;
        stats->ok = 1;
        *pendingCommandOut = cmd;
        *outputBufferOut = output;
        return 1;
    }

    [cmd waitUntilCompleted];
    stats->synchronous_wait_count = 1;
    stats->fused_kernel_seconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: fused dense MLP command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }

    double writeStarted = now_seconds();
    NSData *outputData = nil;
    if (outputDataOut || outputPath) {
        outputData = [NSData dataWithBytes:[output contents]
                                    length:(NSUInteger)hiddenBytes];
    }
    if (outputPath && ![outputData writeToFile:outputPath atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write dense MLP output %s\n",
                [outputPath UTF8String]);
        return 0;
    }
    if (outputDataOut) {
        *outputDataOut = outputData;
    }
    if (outputBufferOut) {
        *outputBufferOut = output;
    }
    stats->output_write_seconds = now_seconds() - writeStarted;
    stats->output0 = ((float *)[output contents])[0];
    if (expectOutput0Set) {
        stats->output0_abs_error = fabsf(stats->output0 - (float)expectOutput0);
        stats->output0_check_ok = stats->output0_abs_error <= 5.0e-3f;
    }
    stats->elapsed_seconds = now_seconds() - totalStarted;
    stats->ok = stats->output0_check_ok;
    if (!stats->output0_check_ok) {
        fprintf(stderr,
                "ERROR: dense MLP output[0] %.6f differs from expected %.6f by %.6f\n",
                stats->output0,
                expectOutput0,
                stats->output0_abs_error);
    }
    return stats->ok;
}

static int run_dense_mlp_probe(id<MTLDevice> device,
                               NSString *residentBinPath,
                               int layerId,
                               ResidentVectorInfo postAttnNormInfo,
                               DenseMlpMxfp4Info info,
                               NSString *inputPath,
                               NSData *inputDataOverride,
                               id<MTLBuffer> inputBufferOverride,
                               NSString *outputPath,
                               NSData **outputDataOut,
                               id<MTLBuffer> __strong *outputBufferOut,
                               float rmsNormEps,
                               int expectOutput0Set,
                               double expectOutput0,
                               int allowAsyncSubmit,
                               id<MTLCommandBuffer> __strong *pendingCommandOut,
                               DenseMlpProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    stats->hidden_dim = info.hidden_dim;
    stats->intermediate_dim = info.intermediate_dim;
    stats->group_size = info.group_size;
    stats->output0_check_ok = 1;
    (void)layerId;
    return run_dense_mlp_fused_probe(device,
                                     residentBinPath,
                                     info,
                                     inputPath,
                                     inputDataOverride,
                                     inputBufferOverride,
                                     postAttnNormInfo,
                                     outputPath,
                                     outputDataOut,
                                     outputBufferOut,
                                     rmsNormEps,
                                     expectOutput0Set,
                                     expectOutput0,
                                     allowAsyncSubmit,
                                     pendingCommandOut,
                                     stats);
}

static int run_attention_projection_fused_pre_cache_probe(
    id<MTLDevice> device,
    NSString *residentBinPath,
    ResidentVectorInfo inputNormInfo,
    ResidentVectorInfo qANormInfo,
    ResidentMxfp4MatrixInfo qAInfo,
                                     ResidentMxfp4MatrixInfo qBInfo,
                                     ResidentMxfp4MatrixInfo kvAInfo,
                                     NSString *inputPath,
                                     NSData *inputDataOverride,
                                     id<MTLBuffer> inputBufferOverride,
                                     float rmsNormEps,
                                     NSData **qBDataOut,
    NSData **kvADataOut,
    id<MTLBuffer> __strong *qBBufferOut,
    id<MTLBuffer> __strong *kvABufferOut,
    int waitForCompletion,
    AttnProjectionProbeStats *stats) {
    if (qBBufferOut) {
        *qBBufferOut = nil;
    }
    if (kvABufferOut) {
        *kvABufferOut = nil;
    }
    uint64_t inputBytes = (uint64_t)inputNormInfo.dim * sizeof(float);
    uint64_t qABytes = (uint64_t)qAInfo.out_dim * sizeof(float);
    uint64_t qBBytes = (uint64_t)qBInfo.out_dim * sizeof(float);
    uint64_t kvABytes = (uint64_t)kvAInfo.out_dim * sizeof(float);
    if (inputBytes > NSUIntegerMax ||
        qABytes > NSUIntegerMax ||
        qBBytes > NSUIntegerMax ||
        kvABytes > NSUIntegerMax ||
        inputNormInfo.dim != qAInfo.in_dim ||
        inputNormInfo.dim != kvAInfo.in_dim ||
        qANormInfo.dim != qAInfo.out_dim ||
        qBInfo.in_dim != qANormInfo.dim) {
        fprintf(stderr, "ERROR: fused attention projection dimensions are inconsistent\n");
        return 0;
    }
    NSData *inputData = nil;
    if (inputBufferOverride) {
        if ((uint64_t)[inputBufferOverride length] < inputBytes) {
            fprintf(stderr,
                    "ERROR: attention projection input buffer bytes %llu are smaller than expected %llu\n",
                    (unsigned long long)[inputBufferOverride length],
                    (unsigned long long)inputBytes);
            return 0;
        }
    } else {
        inputData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
        if (!inputData) {
            fprintf(stderr, "ERROR: failed to read attention projection input %s\n",
                    inputPath ? [inputPath UTF8String] : "<memory>");
            return 0;
        }
        if ((uint64_t)[inputData length] != inputBytes) {
            fprintf(stderr,
                    "ERROR: attention projection input bytes %llu do not match expected %llu\n",
                    (unsigned long long)[inputData length],
                    (unsigned long long)inputBytes);
            return 0;
        }
    }

    float *inputNormWeightValues = NULL;
    float *qANormWeightValues = NULL;
    uint64_t inputNormWeightBytes = 0;
    uint64_t qANormWeightBytes = 0;
    if (!read_resident_vector_f32(residentBinPath,
                                  inputNormInfo,
                                  &inputNormWeightValues,
                                  &inputNormWeightBytes) ||
        !read_resident_vector_f32(residentBinPath,
                                  qANormInfo,
                                  &qANormWeightValues,
                                  &qANormWeightBytes)) {
        free(inputNormWeightValues);
        free(qANormWeightValues);
        return 0;
    }

    void *qAPtr = NULL;
    void *qBPtr = NULL;
    void *kvAPtr = NULL;
    uint64_t qAAllocBytes = 0;
    uint64_t qBAllocBytes = 0;
    uint64_t kvAAllocBytes = 0;
    double qAReadSeconds = 0.0;
    double qBReadSeconds = 0.0;
    double kvAReadSeconds = 0.0;
    int closeFd = 0;
    int fd = open_resident_read_fd(residentBinPath, &closeFd);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open resident file for fused attention projection %s: %s\n",
                [residentBinPath UTF8String],
                strerror(errno));
        free(inputNormWeightValues);
        free(qANormWeightValues);
        return 0;
    }
    int stagedOk =
        stage_resident_mxfp4_matrix_from_fd(fd,
                                            residentBinPath,
                                            qAInfo,
                                            &qAPtr,
                                            &qAAllocBytes,
                                            &qAReadSeconds) &&
        stage_resident_mxfp4_matrix_from_fd(fd,
                                            residentBinPath,
                                            qBInfo,
                                            &qBPtr,
                                            &qBAllocBytes,
                                            &qBReadSeconds) &&
        stage_resident_mxfp4_matrix_from_fd(fd,
                                            residentBinPath,
                                            kvAInfo,
                                            &kvAPtr,
                                            &kvAAllocBytes,
                                            &kvAReadSeconds);
    close_resident_read_fd(fd, closeFd);
    if (!stagedOk) {
        free(inputNormWeightValues);
        free(qANormWeightValues);
        free(qAPtr);
        free(qBPtr);
        free(kvAPtr);
        return 0;
    }

    id<MTLLibrary> library = make_glm_moe_library(device);
    id<MTLComputePipelineState> rmsPipe =
        library ? make_glm_pipeline(device, library, @"glm_rmsnorm_f32") : nil;
    id<MTLComputePipelineState> matvecPipe =
        library ? make_resident_mxfp4_matvec_pipeline(device, library, qAInfo) : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> input = inputBufferOverride ?:
        [device newBufferWithBytes:[inputData bytes]
                            length:(NSUInteger)inputBytes
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> inputNormWeight =
        [device newBufferWithBytes:inputNormWeightValues
                            length:(NSUInteger)inputBytes
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> qANormWeight =
        [device newBufferWithBytes:qANormWeightValues
                            length:(NSUInteger)qABytes
                           options:MTLResourceStorageModeShared];
    free(inputNormWeightValues);
    free(qANormWeightValues);
    id<MTLBuffer> qAMatrix = [device newBufferWithBytesNoCopy:qAPtr
                                                       length:(NSUInteger)qAAllocBytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:^(void *pointer, NSUInteger length) {
                                                      (void)length;
                                                      free(pointer);
                                                  }];
    if (qAMatrix) {
        qAPtr = NULL;
    }
    id<MTLBuffer> qBMatrix = [device newBufferWithBytesNoCopy:qBPtr
                                                       length:(NSUInteger)qBAllocBytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:^(void *pointer, NSUInteger length) {
                                                      (void)length;
                                                      free(pointer);
                                                  }];
    if (qBMatrix) {
        qBPtr = NULL;
    }
    id<MTLBuffer> kvAMatrix = [device newBufferWithBytesNoCopy:kvAPtr
                                                        length:(NSUInteger)kvAAllocBytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:^(void *pointer, NSUInteger length) {
                                                       (void)length;
                                                       free(pointer);
                                                   }];
    if (kvAMatrix) {
        kvAPtr = NULL;
    }
    id<MTLBuffer> inputNorm = [device newBufferWithLength:(NSUInteger)inputBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> qA = [device newBufferWithLength:(NSUInteger)qABytes
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> qANorm = [device newBufferWithLength:(NSUInteger)qABytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> qB = [device newBufferWithLength:(NSUInteger)qBBytes
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> kvA = [device newBufferWithLength:(NSUInteger)kvABytes
                                            options:MTLResourceStorageModeShared];
    if (!library || !rmsPipe || !matvecPipe || !queue || !input ||
        !inputNormWeight || !qANormWeight || !qAMatrix || !qBMatrix ||
        !kvAMatrix || !inputNorm || !qA || !qANorm || !qB || !kvA) {
        fprintf(stderr, "ERROR: failed to allocate fused attention projection Metal resources\n");
        free(qAPtr);
        free(qBPtr);
        free(kvAPtr);
        return 0;
    }
    [qAMatrix didModifyRange:NSMakeRange(0, (NSUInteger)qAInfo.total_bytes)];
    [qBMatrix didModifyRange:NSMakeRange(0, (NSUInteger)qBInfo.total_bytes)];
    [kvAMatrix didModifyRange:NSMakeRange(0, (NSUInteger)kvAInfo.total_bytes)];

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create fused attention projection command buffer\n");
        return 0;
    }
    if (!encode_glm_rmsnorm(cmd,
                            rmsPipe,
                            input,
                            inputNormWeight,
                            inputNorm,
                            inputNormInfo.dim,
                            rmsNormEps) ||
        !encode_glm_mxfp4_matvec(cmd,
                                 matvecPipe,
                                 qAMatrix,
                                 qAInfo,
                                 inputNorm,
                                 qA) ||
        !encode_glm_mxfp4_matvec(cmd,
                                 matvecPipe,
                                 kvAMatrix,
                                 kvAInfo,
                                 inputNorm,
                                 kvA) ||
        !encode_glm_rmsnorm(cmd,
                            rmsPipe,
                            qA,
                            qANormWeight,
                            qANorm,
                            qANormInfo.dim,
                            rmsNormEps) ||
        !encode_glm_mxfp4_matvec(cmd,
                                 matvecPipe,
                                 qBMatrix,
                                 qBInfo,
                                 qANorm,
                                 qB)) {
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    int mustWait = waitForCompletion || qBDataOut || kvADataOut;
    if (mustWait) {
        [cmd waitUntilCompleted];
        stats->synchronous_wait_count = 1;
        stats->fused_pre_cache_seconds = now_seconds() - kernelStarted;
        if (cmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr,
                    "ERROR: fused attention projection command failed: %s\n",
                    cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
            return 0;
        }
    } else {
        stats->async_submitted = 1;
        stats->fused_pre_cache_seconds = 0.0;
    }

    if (qBDataOut) {
        *qBDataOut = [NSData dataWithBytes:[qB contents] length:(NSUInteger)qBBytes];
    }
    if (kvADataOut) {
        *kvADataOut = [NSData dataWithBytes:[kvA contents] length:(NSUInteger)kvABytes];
    }
    if (qBBufferOut) {
        *qBBufferOut = qB;
    }
    if (kvABufferOut) {
        *kvABufferOut = kvA;
    }
    stats->bytes_read =
        inputNormWeightBytes +
        qANormWeightBytes +
        qAInfo.total_bytes +
        qBInfo.total_bytes +
        kvAInfo.total_bytes;
    (void)qAReadSeconds;
    (void)qBReadSeconds;
    (void)kvAReadSeconds;
    if (mustWait) {
        stats->q_b_output0 = ((float *)[qB contents])[0];
        stats->kv_a_output0 = ((float *)[kvA contents])[0];
    }
    stats->kv_a_norm_output0 = 0.0f;
    stats->kv_b_output0 = 0.0f;
    stats->fused_pre_cache = 1;
    stats->command_buffer_count = 1;
    return 1;
}

static int run_attention_projection_probe(id<MTLDevice> device,
                                          NSString *residentBinPath,
                                          int layerId,
                                          ResidentVectorInfo inputNormInfo,
                                          ResidentVectorInfo qANormInfo,
                                          ResidentVectorInfo kvANormInfo,
                                          ResidentMxfp4MatrixInfo qAInfo,
                                          ResidentMxfp4MatrixInfo qBInfo,
                                          ResidentMxfp4MatrixInfo kvAInfo,
                                          int hasKVB,
                                          ResidentMxfp4MatrixInfo kvBInfo,
                                          NSString *inputPath,
                                          NSData *inputDataOverride,
                                          id<MTLBuffer> inputBufferOverride,
                                          NSString *outputDir,
                                          NSString *cacheLayoutPath,
                                          NSString *cacheFilePath,
                                          uint64_t cachePosition,
                                          int appendCache,
                                          uint64_t maxCacheFileBytes,
                                          float rmsNormEps,
                                          int writeDebugFiles,
                                          NSData **qBDataOut,
                                          NSData **kvADataOut,
                                          id<MTLBuffer> __strong *qBBufferOut,
                                          id<MTLBuffer> __strong *kvABufferOut,
                                          AttnProjectionProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (qBBufferOut) {
        *qBBufferOut = nil;
    }
    if (kvABufferOut) {
        *kvABufferOut = nil;
    }
    stats->hidden_dim = qAInfo.in_dim;
    stats->q_lora_dim = qAInfo.out_dim;
    stats->q_out_dim = qBInfo.out_dim;
    stats->kv_lora_dim = kvANormInfo.dim;
    stats->kv_a_out_dim = kvAInfo.out_dim;
    stats->kv_rope_dim = kvAInfo.out_dim - kvANormInfo.dim;
    stats->kv_out_dim = hasKVB ? kvBInfo.out_dim : 0;
    stats->has_kv_b = hasKVB;
    stats->cache_append = appendCache ? 1 : 0;
    stats->cache_position = cachePosition;
    if (writeDebugFiles && !ensure_output_dir(outputDir)) {
        return 0;
    }

    double started = now_seconds();
    if (!writeDebugFiles) {
        NSData *qBData = nil;
        NSData *kvAData = nil;
        int wantsKVData = appendCache || kvADataOut;
        int ok = 0;
        if (qBDataOut && wantsKVData) {
            ok = run_attention_projection_fused_pre_cache_probe(device,
                                                                residentBinPath,
                                                                inputNormInfo,
                                                                qANormInfo,
                                                                qAInfo,
                                                                qBInfo,
	                                                                kvAInfo,
	                                                                inputPath,
	                                                                inputDataOverride,
                                                                    inputBufferOverride,
	                                                                rmsNormEps,
                                                                &qBData,
                                                                &kvAData,
                                                                qBBufferOut,
                                                                kvABufferOut,
                                                                0,
                                                                stats);
        } else if (qBDataOut) {
            ok = run_attention_projection_fused_pre_cache_probe(device,
                                                                residentBinPath,
                                                                inputNormInfo,
                                                                qANormInfo,
                                                                qAInfo,
                                                                qBInfo,
	                                                                kvAInfo,
	                                                                inputPath,
	                                                                inputDataOverride,
                                                                    inputBufferOverride,
	                                                                rmsNormEps,
                                                                &qBData,
                                                                nil,
                                                                qBBufferOut,
                                                                kvABufferOut,
                                                                0,
                                                                stats);
        } else if (wantsKVData) {
            ok = run_attention_projection_fused_pre_cache_probe(device,
                                                                residentBinPath,
                                                                inputNormInfo,
                                                                qANormInfo,
                                                                qAInfo,
                                                                qBInfo,
	                                                                kvAInfo,
	                                                                inputPath,
	                                                                inputDataOverride,
                                                                    inputBufferOverride,
	                                                                rmsNormEps,
                                                                nil,
                                                                &kvAData,
                                                                qBBufferOut,
                                                                kvABufferOut,
                                                                0,
                                                                stats);
        } else {
            ok = run_attention_projection_fused_pre_cache_probe(device,
                                                                residentBinPath,
                                                                inputNormInfo,
                                                                qANormInfo,
                                                                qAInfo,
                                                                qBInfo,
	                                                                kvAInfo,
	                                                                inputPath,
	                                                                inputDataOverride,
                                                                    inputBufferOverride,
	                                                                rmsNormEps,
                                                                nil,
                                                                nil,
                                                                qBBufferOut,
                                                                kvABufferOut,
                                                                0,
                                                                stats);
        }
        if (!ok) {
            return 0;
        }
        if (qBDataOut) {
            *qBDataOut = qBData;
        }
        if (kvADataOut) {
            *kvADataOut = kvAData;
        }
        if (appendCache) {
            if (!kvAData) {
                fprintf(stderr, "ERROR: fused attention projection did not return KV-A data for cache append\n");
                return 0;
            }
            double cacheStarted = now_seconds();
            if (!append_kv_a_data_to_decode_cache(cacheLayoutPath,
                                                  cacheFilePath,
                                                  (uint64_t)layerId,
                                                  cachePosition,
                                                  kvAData,
                                                  kvAInfo.out_dim,
                                                  maxCacheFileBytes,
                                                  &stats->cache_write_bytes)) {
                return 0;
            }
            stats->cache_write_seconds = now_seconds() - cacheStarted;
        }
        stats->elapsed_seconds = now_seconds() - started;
        stats->ok = 1;
        if (qBDataOut) {
            *qBDataOut = qBData;
        }
        if (kvADataOut) {
            *kvADataOut = kvAData;
        }
        return 1;
    }

    RmsNormProbeStats inputNormStats = {0};
    RmsNormProbeStats qANormStats = {0};
    RmsNormProbeStats kvANormStats = {0};
    ResidentLinearProbeStats qAStats = {0};
    ResidentLinearProbeStats qBStats = {0};
    ResidentLinearProbeStats kvAStats = {0};
    ResidentLinearProbeStats kvBStats = {0};
    NSData *inputNormData = nil;
    NSData *qAData = nil;
    NSData *qANormData = nil;
    NSData *qBData = nil;
    NSData *kvAData = nil;
    NSData *kvANormData = nil;
    NSData *kvBData = nil;

    if (!run_rmsnorm_probe(device,
                           residentBinPath,
                           inputNormInfo,
                           inputPath,
                           inputDataOverride,
                           rmsNormEps,
                           &inputNormData,
                           &inputNormStats)) {
        return 0;
    }
    if (writeDebugFiles &&
        !write_data_to_output_dir(outputDir, @"attn_input_norm.f32", inputNormData)) {
        return 0;
    }
    if (!run_resident_linear_probe(device,
                                   residentBinPath,
                                   nil,
                                   qAInfo,
                                   nil,
                                   inputNormData,
                                   writeDebugFiles
                                       ? [outputDir stringByAppendingPathComponent:@"attn_q_a.f32"]
                                       : nil,
                                   &qAData,
                                   0,
                                   0.0,
                                   &qAStats)) {
        return 0;
    }
    if (!run_rmsnorm_probe(device,
                           residentBinPath,
                           qANormInfo,
                           nil,
                           qAData,
                           rmsNormEps,
                           &qANormData,
                           &qANormStats)) {
        return 0;
    }
    if (writeDebugFiles &&
        !write_data_to_output_dir(outputDir, @"attn_q_a_norm.f32", qANormData)) {
        return 0;
    }
    if (!run_resident_linear_probe(device,
                                   residentBinPath,
                                   nil,
                                   qBInfo,
                                   nil,
                                   qANormData,
                                   writeDebugFiles
                                       ? [outputDir stringByAppendingPathComponent:@"attn_q_b.f32"]
                                       : nil,
                                   &qBData,
                                   0,
                                   0.0,
                                   &qBStats)) {
        return 0;
    }
    if (!run_resident_linear_probe(device,
                                   residentBinPath,
                                   nil,
                                   kvAInfo,
                                   nil,
                                   inputNormData,
                                   writeDebugFiles
                                       ? [outputDir stringByAppendingPathComponent:@"attn_kv_a.f32"]
                                       : nil,
                                   &kvAData,
                                   0,
                                   0.0,
                                   &kvAStats)) {
        return 0;
    }
    if (appendCache) {
        double cacheStarted = now_seconds();
        if (!append_kv_a_data_to_decode_cache(cacheLayoutPath,
                                              cacheFilePath,
                                              (uint64_t)layerId,
                                              cachePosition,
                                              kvAData,
                                              kvAInfo.out_dim,
                                              maxCacheFileBytes,
                                              &stats->cache_write_bytes)) {
            return 0;
        }
        stats->cache_write_seconds = now_seconds() - cacheStarted;
    }
    uint64_t kvLoraBytes = (uint64_t)kvANormInfo.dim * sizeof(float);
    if ((uint64_t)[kvAData length] < kvLoraBytes) {
        fprintf(stderr, "ERROR: KV-A output is smaller than kv lora dim\n");
        return 0;
    }
    NSData *kvALoraData = [NSData dataWithBytes:[kvAData bytes]
                                         length:(NSUInteger)kvLoraBytes];
    if (!run_rmsnorm_probe(device,
                           residentBinPath,
                           kvANormInfo,
                           nil,
                           kvALoraData,
                           rmsNormEps,
                           &kvANormData,
                           &kvANormStats)) {
        return 0;
    }
    if (writeDebugFiles &&
        !write_data_to_output_dir(outputDir, @"attn_kv_a_norm.f32", kvANormData)) {
        return 0;
    }
    if (hasKVB) {
        if (!run_resident_linear_probe(device,
                                       residentBinPath,
                                       nil,
                                       kvBInfo,
                                       nil,
                                       kvANormData,
                                       writeDebugFiles
                                           ? [outputDir stringByAppendingPathComponent:@"attn_kv_b.f32"]
                                           : nil,
                                       &kvBData,
                                       0,
                                       0.0,
                                       &kvBStats)) {
            return 0;
        }
    }

    stats->bytes_read =
        inputNormStats.weight_bytes_read +
        qANormStats.weight_bytes_read +
        kvANormStats.weight_bytes_read +
        qAStats.bytes_read +
        qBStats.bytes_read +
        kvAStats.bytes_read +
        (hasKVB ? kvBStats.bytes_read : 0);
    stats->q_b_output0 = qBStats.output0;
    stats->kv_a_output0 = kvAStats.output0;
    stats->kv_a_norm_output0 = kvANormStats.output0;
    stats->kv_b_output0 = hasKVB ? kvBStats.output0 : 0.0f;
    stats->fused_pre_cache = 0;
    stats->command_buffer_count = hasKVB ? 7 : 6;
    stats->elapsed_seconds = now_seconds() - started;
    stats->ok = 1;
    if (qBDataOut) {
        *qBDataOut = qBData;
    }
    if (kvADataOut) {
        *kvADataOut = kvAData;
    }
    return 1;
}

static int encode_glm_rope_split_batch_with_offsets(id<MTLCommandBuffer> cmd,
                                                    id<MTLComputePipelineState> pipe,
                                                    id<MTLBuffer> qB,
                                                    NSUInteger qBOffset,
                                                    id<MTLBuffer> k,
                                                    NSUInteger kOffset,
                                                    id<MTLBuffer> outQNope,
                                                    id<MTLBuffer> outQRope,
                                                    id<MTLBuffer> outQ,
                                                    id<MTLBuffer> outK,
                                                    uint32_t numHeads,
                                                    uint32_t qkNopeDim,
                                                    uint32_t ropeDim,
                                                    uint32_t startPosition,
                                                    float theta,
                                                    uint32_t interleave,
                                                    uint32_t batchTokens) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create RoPE split encoder\n");
        return 0;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:qB offset:qBOffset atIndex:0];
    [enc setBuffer:k offset:kOffset atIndex:1];
    [enc setBuffer:outQNope offset:0 atIndex:2];
    [enc setBuffer:outQRope offset:0 atIndex:3];
    [enc setBuffer:outQ offset:0 atIndex:4];
    [enc setBuffer:outK offset:0 atIndex:5];
    [enc setBytes:&numHeads length:sizeof(numHeads) atIndex:6];
    [enc setBytes:&qkNopeDim length:sizeof(qkNopeDim) atIndex:7];
    [enc setBytes:&ropeDim length:sizeof(ropeDim) atIndex:8];
    [enc setBytes:&startPosition length:sizeof(startPosition) atIndex:9];
    [enc setBytes:&theta length:sizeof(theta) atIndex:10];
    [enc setBytes:&interleave length:sizeof(interleave) atIndex:11];
    [enc setBytes:&batchTokens length:sizeof(batchTokens) atIndex:12];
    uint32_t qHeadDim = qkNopeDim + ropeDim;
    uint32_t total = numHeads * qHeadDim + ropeDim;
    NSUInteger width = pipe.maxTotalThreadsPerThreadgroup;
    if (width == 0 || width > 256) {
        width = 256;
    }
    [enc dispatchThreads:MTLSizeMake(total, batchTokens, 1)
   threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [enc endEncoding];
    return 1;
}

static int encode_glm_rope_split_batch(id<MTLCommandBuffer> cmd,
                                       id<MTLComputePipelineState> pipe,
                                       id<MTLBuffer> qB,
                                       id<MTLBuffer> k,
                                       id<MTLBuffer> outQNope,
                                       id<MTLBuffer> outQRope,
                                       id<MTLBuffer> outQ,
                                       id<MTLBuffer> outK,
                                       uint32_t numHeads,
                                       uint32_t qkNopeDim,
                                       uint32_t ropeDim,
                                       uint32_t startPosition,
                                       float theta,
                                       uint32_t interleave,
                                       uint32_t batchTokens) {
    return encode_glm_rope_split_batch_with_offsets(cmd,
                                                    pipe,
                                                    qB,
                                                    0,
                                                    k,
                                                    0,
                                                    outQNope,
                                                    outQRope,
                                                    outQ,
                                                    outK,
                                                    numHeads,
                                                    qkNopeDim,
                                                    ropeDim,
                                                    startPosition,
                                                    theta,
                                                    interleave,
                                                    batchTokens);
}

static uint64_t rope_split_scratch_bytes(uint32_t numHeads,
                                         uint32_t qkNopeDim,
                                         uint32_t ropeDim,
                                         uint32_t batchTokens,
                                         uint64_t *qBBytesOut,
                                         uint64_t *kBytesOut,
                                         uint64_t *qNopeBytesOut,
                                         uint64_t *qRopeBytesOut) {
    uint64_t qHeadDim = (uint64_t)qkNopeDim + ropeDim;
    uint64_t qBBytes =
        (uint64_t)batchTokens * numHeads * qHeadDim * sizeof(float);
    uint64_t kBytes = (uint64_t)batchTokens * ropeDim * sizeof(float);
    uint64_t qNopeBytes =
        (uint64_t)batchTokens * numHeads * qkNopeDim * sizeof(float);
    uint64_t qRopeBytes =
        (uint64_t)batchTokens * numHeads * ropeDim * sizeof(float);
    if (qBBytesOut) *qBBytesOut = qBBytes;
    if (kBytesOut) *kBytesOut = kBytes;
    if (qNopeBytesOut) *qNopeBytesOut = qNopeBytes;
    if (qRopeBytesOut) *qRopeBytesOut = qRopeBytes;
    return qBBytes + kBytes + qNopeBytes + qRopeBytes + qRopeBytes + kBytes;
}

static int run_rope_split_probe(id<MTLDevice> device,
                                NSString *qBPath,
                                NSString *kPath,
                                NSData *qBDataOverride,
                                NSData *kDataOverride,
                                NSString *outQNopePath,
                                NSString *outQRopePath,
                                NSString *outQPath,
                                NSString *outKPath,
                                uint32_t numHeads,
                                uint32_t qkNopeDim,
                                uint32_t ropeDim,
                                uint32_t startPosition,
                                uint32_t batchTokens,
                                float theta,
                                int interleaveFlag,
                                NSData **qNopeDataOut,
                                NSData **qRopeDataOut,
                                RopeSplitProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (qkNopeDim > UINT32_MAX - ropeDim) {
        fprintf(stderr, "ERROR: RoPE q head dim overflows uint32\n");
        return 0;
    }
    uint64_t qBBytes = 0;
    uint64_t kBytes = 0;
    uint64_t qNopeBytes = 0;
    uint64_t qRopeBytes = 0;
    uint64_t scratchBytes = rope_split_scratch_bytes(
        numHeads,
        qkNopeDim,
        ropeDim,
        batchTokens,
        &qBBytes,
        &kBytes,
        &qNopeBytes,
        &qRopeBytes
    );
    if (qBBytes > NSUIntegerMax ||
        kBytes > NSUIntegerMax ||
        qNopeBytes > NSUIntegerMax ||
        qRopeBytes > NSUIntegerMax) {
        fprintf(stderr, "ERROR: RoPE split buffers exceed NSUIntegerMax\n");
        return 0;
    }
    NSData *qBData = qBDataOverride ?: [NSData dataWithContentsOfFile:qBPath];
    NSData *kData = kDataOverride ?: [NSData dataWithContentsOfFile:kPath];
    if (!qBData || !kData) {
        fprintf(stderr, "ERROR: failed to read RoPE split inputs\n");
        return 0;
    }
    if ((uint64_t)[qBData length] != qBBytes || (uint64_t)[kData length] != kBytes) {
        fprintf(stderr,
                "ERROR: RoPE split input bytes mismatch: q_b %llu/%llu k %llu/%llu\n",
                (unsigned long long)[qBData length],
                (unsigned long long)qBBytes,
                (unsigned long long)[kData length],
                (unsigned long long)kBytes);
        return 0;
    }
    id<MTLLibrary> library = make_glm_moe_library(device);
    id<MTLComputePipelineState> pipe =
        library ? make_glm_pipeline(device, library, @"glm_rope_split_batch_f32") : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> qB = [device newBufferWithBytes:[qBData bytes]
                                           length:(NSUInteger)qBBytes
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> k = [device newBufferWithBytes:[kData bytes]
                                          length:(NSUInteger)kBytes
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQNope = [device newBufferWithLength:(NSUInteger)qNopeBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQRope = [device newBufferWithLength:(NSUInteger)qRopeBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQ = [device newBufferWithLength:(NSUInteger)qRopeBytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> outK = [device newBufferWithLength:(NSUInteger)kBytes
                                             options:MTLResourceStorageModeShared];
    if (!library || !pipe || !queue || !qB || !k || !outQNope || !outQRope || !outQ || !outK) {
        fprintf(stderr, "ERROR: failed to allocate RoPE split Metal resources\n");
        return 0;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create RoPE split command buffer\n");
        return 0;
    }
    if (!encode_glm_rope_split_batch(cmd,
                                     pipe,
                                     qB,
                                     k,
                                     outQNope,
                                     outQRope,
                                     outQ,
                                     outK,
                                     numHeads,
                                     qkNopeDim,
                                     ropeDim,
                                     startPosition,
                                     theta,
                                     interleaveFlag ? 1u : 0u,
                                     batchTokens)) {
        return 0;
    }
    double started = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    stats->elapsed_seconds = now_seconds() - started;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: RoPE split command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    NSData *outQNopeData = [NSData dataWithBytes:[outQNope contents]
                                          length:(NSUInteger)qNopeBytes];
    NSData *outQRopeData = [NSData dataWithBytes:[outQRope contents]
                                          length:(NSUInteger)qRopeBytes];
    NSData *outQData = [NSData dataWithBytes:[outQ contents]
                                      length:(NSUInteger)qRopeBytes];
    NSData *outKData = [NSData dataWithBytes:[outK contents]
                                      length:(NSUInteger)kBytes];
    if (outQNopePath || outQRopePath || outQPath || outKPath) {
        if (!outQNopePath || !outQRopePath || !outQPath || !outKPath ||
            ![outQNopeData writeToFile:outQNopePath atomically:YES] ||
            ![outQRopeData writeToFile:outQRopePath atomically:YES] ||
            ![outQData writeToFile:outQPath atomically:YES] ||
            ![outKData writeToFile:outKPath atomically:YES]) {
            fprintf(stderr, "ERROR: failed to write RoPE split outputs\n");
            return 0;
        }
    }
    if (qNopeDataOut) {
        *qNopeDataOut = outQNopeData;
    }
    if (qRopeDataOut) {
        *qRopeDataOut = outQData;
    }
    stats->ok = 1;
    stats->scratch_bytes = scratchBytes;
    stats->num_heads = numHeads;
    stats->qk_nope_dim = qkNopeDim;
    stats->rope_dim = ropeDim;
    stats->start_position = startPosition;
    stats->batch_tokens = batchTokens;
    stats->theta = theta;
    stats->interleave = interleaveFlag ? 1 : 0;
    stats->q_b_bytes = qBBytes;
    stats->k_bytes = kBytes;
    stats->q_nope_bytes = qNopeBytes;
    stats->q_rope_bytes = qRopeBytes;
    stats->command_buffer_count = 1;
    stats->fused_with_mla = 0;
    stats->q_output0 = ((float *)[outQ contents])[0];
    stats->k_output0 = ((float *)[outK contents])[0];
    return 1;
}

static int resolve_mla_attention_value_source(NSDictionary *residentLayout,
                                              int layerId,
                                              uint32_t numHeads,
                                              uint32_t kvLoraDimArg,
                                              uint32_t qkNopeDim,
                                              uint32_t vHeadDim,
                                              MlaAttentionValueSourceInfo *out) {
    memset(out, 0, sizeof(*out));
    NSString *embedQName = layer_tensor_name(layerId, @".self_attn.embed_q.weight");
    NSString *unembedOutName = layer_tensor_name(layerId, @".self_attn.unembed_out.weight");
    ResidentMxfp4Tensor3DInfo embedQ = {0};
    ResidentMxfp4Tensor3DInfo unembedOut = {0};
    if (!find_resident_mxfp4_tensor3d_info(residentLayout, embedQName, &embedQ) ||
        !find_resident_mxfp4_tensor3d_info(residentLayout, unembedOutName, &unembedOut)) {
        return 0;
    }
    uint32_t kvLoraDim = kvLoraDimArg ? kvLoraDimArg : embedQ.dim1;
    if (embedQ.dim0 != numHeads ||
        embedQ.dim1 != kvLoraDim ||
        embedQ.dim2 != qkNopeDim ||
        unembedOut.dim0 != numHeads ||
        unembedOut.dim1 != vHeadDim ||
        unembedOut.dim2 != kvLoraDim) {
        fprintf(stderr,
                "ERROR: absorbed MLA alias shapes do not match expected "
                "embed=[%u,%u,%u] unembed=[%u,%u,%u]\n",
                numHeads,
                kvLoraDim,
                qkNopeDim,
                numHeads,
                vHeadDim,
                kvLoraDim);
        return 0;
    }
    uint64_t qv = 0;
    uint64_t expectedOut = 0;
    uint64_t kvBValues = 0;
    uint64_t kvBBytes = 0;
    uint64_t storageBytes = 0;
    uint64_t embedF32Bytes = 0;
    uint64_t unembedF32Bytes = 0;
    uint64_t sourceF32Bytes = 0;
    if (!checked_add_u64((uint64_t)qkNopeDim, (uint64_t)vHeadDim, &qv) ||
        !checked_mul_u64((uint64_t)numHeads, qv, &expectedOut) ||
        expectedOut > UINT32_MAX ||
        !checked_mul_u64(expectedOut, (uint64_t)kvLoraDim, &kvBValues) ||
        !checked_mul_u64(kvBValues, sizeof(float), &kvBBytes) ||
        !checked_add_u64(embedQ.total_bytes, unembedOut.total_bytes, &storageBytes) ||
        !resident_mxfp4_tensor3d_f32_bytes(embedQ, &embedF32Bytes) ||
        !resident_mxfp4_tensor3d_f32_bytes(unembedOut, &unembedF32Bytes) ||
        !checked_add_u64(embedF32Bytes, unembedF32Bytes, &sourceF32Bytes)) {
        fprintf(stderr, "ERROR: absorbed MLA value source byte size overflows\n");
        return 0;
    }
    out->embed_q = embedQ;
    out->unembed_out = unembedOut;
    out->kv_lora_dim = kvLoraDim;
    out->expected_kv_b_out = (uint32_t)expectedOut;
    out->storage_bytes = storageBytes;
    out->source_f32_bytes = sourceF32Bytes;
    out->kv_b_f32_bytes = kvBBytes;
    return 1;
}

static int read_absorbed_mla_kv_b_f32(NSString *residentBinPath,
                                      MlaAttentionValueSourceInfo source,
                                      uint32_t numHeads,
                                      uint32_t qkNopeDim,
                                      uint32_t vHeadDim,
                                      float **outValues,
                                      uint64_t *outBytes) {
    *outValues = NULL;
    *outBytes = 0;
    if (source.kv_b_f32_bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "ERROR: MLA KV-B f32 view is too large\n");
        return 0;
    }
    uint64_t kvBCount = source.kv_b_f32_bytes / sizeof(float);
    float *kvB = (float *)calloc((size_t)kvBCount, sizeof(float));
    if (!kvB) {
        fprintf(stderr, "ERROR: failed to allocate MLA KV-B f32 view\n");
        return 0;
    }

    float *embed = NULL;
    uint64_t embedBytes = 0;
    if (!read_resident_mxfp4_tensor3d_f32(residentBinPath,
                                          source.embed_q,
                                          &embed,
                                          &embedBytes)) {
        free(kvB);
        return 0;
    }
    uint32_t kvLoraDim = source.kv_lora_dim;
    for (uint32_t h = 0; h < numHeads; h++) {
        uint64_t headRowBase = (uint64_t)h * (uint64_t)(qkNopeDim + vHeadDim);
        for (uint32_t d = 0; d < qkNopeDim; d++) {
            float *row = kvB + (headRowBase + d) * (uint64_t)kvLoraDim;
            for (uint32_t r = 0; r < kvLoraDim; r++) {
                uint64_t src = ((uint64_t)h * kvLoraDim + r) * (uint64_t)qkNopeDim + d;
                row[r] = embed[src];
            }
        }
    }
    free(embed);

    float *unembed = NULL;
    uint64_t unembedBytes = 0;
    if (!read_resident_mxfp4_tensor3d_f32(residentBinPath,
                                          source.unembed_out,
                                          &unembed,
                                          &unembedBytes)) {
        free(kvB);
        return 0;
    }
    for (uint32_t h = 0; h < numHeads; h++) {
        uint64_t headRowBase = (uint64_t)h * (uint64_t)(qkNopeDim + vHeadDim);
        for (uint32_t v = 0; v < vHeadDim; v++) {
            float *row =
                kvB + (headRowBase + qkNopeDim + v) * (uint64_t)kvLoraDim;
            for (uint32_t r = 0; r < kvLoraDim; r++) {
                uint64_t src = ((uint64_t)h * vHeadDim + v) *
                               (uint64_t)kvLoraDim + r;
                row[r] = unembed[src];
            }
        }
    }
    free(unembed);

    if (embedBytes + unembedBytes != source.source_f32_bytes) {
        fprintf(stderr, "ERROR: internal MLA absorbed alias f32 byte estimate mismatch\n");
        free(kvB);
        return 0;
    }
    *outValues = kvB;
    *outBytes = source.kv_b_f32_bytes;
    return 1;
}

#define MAX_MLA_KV_B_CACHE_ENTRIES 128

typedef struct {
    int valid;
    char resident_path[PATH_MAX];
    int layer_id;
    uint32_t num_heads;
    uint32_t qk_nope_dim;
    uint32_t v_head_dim;
    ResidentMxfp4Tensor3DInfo embed_q;
    ResidentMxfp4Tensor3DInfo unembed_out;
    uint64_t bytes;
    float *values;
} MlaKVBMemoryCacheEntry;

typedef struct {
    int enabled;
    uint64_t max_bytes;
    uint64_t bytes;
    MlaKVBMemoryCacheEntry entries[MAX_MLA_KV_B_CACHE_ENTRIES];
    pthread_mutex_t mutex;
} MlaKVBMemoryCache;

static MlaKVBMemoryCache gMlaKVBMemoryCache = {
    .enabled = 0,
    .max_bytes = 0,
    .bytes = 0,
    .entries = {{0}},
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};

static int mla_kv_b_tensor3d_matches(ResidentMxfp4Tensor3DInfo a,
                                     ResidentMxfp4Tensor3DInfo b) {
    return a.dim0 == b.dim0 &&
           a.dim1 == b.dim1 &&
           a.dim2 == b.dim2 &&
           a.group_size == b.group_size &&
           a.total_bytes == b.total_bytes &&
           a.weight.offset == b.weight.offset &&
           a.weight.size == b.weight.size &&
           a.weight.dim0 == b.weight.dim0 &&
           a.weight.dim1 == b.weight.dim1 &&
           a.weight.dim2 == b.weight.dim2 &&
           a.scales.offset == b.scales.offset &&
           a.scales.size == b.scales.size &&
           a.scales.dim0 == b.scales.dim0 &&
           a.scales.dim1 == b.scales.dim1 &&
           a.scales.dim2 == b.scales.dim2 &&
           strcmp(a.weight.dtype, b.weight.dtype) == 0 &&
           strcmp(a.scales.dtype, b.scales.dtype) == 0;
}

static int mla_kv_b_cache_entry_matches(MlaKVBMemoryCacheEntry *entry,
                                        const char *residentPath,
                                        int layerId,
                                        MlaAttentionValueSourceInfo source,
                                        uint32_t numHeads,
                                        uint32_t qkNopeDim,
                                        uint32_t vHeadDim) {
    return entry->valid &&
           entry->layer_id == layerId &&
           entry->num_heads == numHeads &&
           entry->qk_nope_dim == qkNopeDim &&
           entry->v_head_dim == vHeadDim &&
           entry->bytes == source.kv_b_f32_bytes &&
           strcmp(entry->resident_path, residentPath) == 0 &&
           mla_kv_b_tensor3d_matches(entry->embed_q, source.embed_q) &&
           mla_kv_b_tensor3d_matches(entry->unembed_out, source.unembed_out);
}

static void mla_kv_b_memory_cache_clear_locked(void) {
    for (int i = 0; i < MAX_MLA_KV_B_CACHE_ENTRIES; i++) {
        if (gMlaKVBMemoryCache.entries[i].valid) {
            free(gMlaKVBMemoryCache.entries[i].values);
        }
        memset(&gMlaKVBMemoryCache.entries[i], 0,
               sizeof(gMlaKVBMemoryCache.entries[i]));
    }
    gMlaKVBMemoryCache.bytes = 0;
}

static void mla_kv_b_memory_cache_configure(int enabled, uint64_t maxBytes) {
    pthread_mutex_lock(&gMlaKVBMemoryCache.mutex);
    if (!enabled || maxBytes == 0) {
        mla_kv_b_memory_cache_clear_locked();
        gMlaKVBMemoryCache.enabled = 0;
        gMlaKVBMemoryCache.max_bytes = 0;
        pthread_mutex_unlock(&gMlaKVBMemoryCache.mutex);
        return;
    }
    if (!gMlaKVBMemoryCache.enabled || gMlaKVBMemoryCache.max_bytes != maxBytes) {
        mla_kv_b_memory_cache_clear_locked();
    }
    gMlaKVBMemoryCache.enabled = 1;
    gMlaKVBMemoryCache.max_bytes = maxBytes;
    pthread_mutex_unlock(&gMlaKVBMemoryCache.mutex);
}

static uint64_t mla_kv_b_memory_cache_current_bytes(void) {
    pthread_mutex_lock(&gMlaKVBMemoryCache.mutex);
    uint64_t bytes = gMlaKVBMemoryCache.bytes;
    pthread_mutex_unlock(&gMlaKVBMemoryCache.mutex);
    return bytes;
}

static int mla_kv_b_memory_cache_enabled(void) {
    pthread_mutex_lock(&gMlaKVBMemoryCache.mutex);
    int enabled = gMlaKVBMemoryCache.enabled;
    pthread_mutex_unlock(&gMlaKVBMemoryCache.mutex);
    return enabled;
}

static int read_absorbed_mla_kv_b_f32_cached(NSString *residentBinPath,
                                             MlaAttentionValueSourceInfo source,
                                             int layerId,
                                             uint32_t numHeads,
                                             uint32_t qkNopeDim,
                                             uint32_t vHeadDim,
                                             float **outValues,
                                             uint64_t *outBytes,
                                             int *outOwned,
                                             int *outCacheHit,
                                             int *outCacheStored,
                                             uint64_t *outCacheBytes,
                                             uint64_t *outCacheTotalBytes) {
    *outValues = NULL;
    *outBytes = 0;
    *outOwned = 1;
    *outCacheHit = 0;
    *outCacheStored = 0;
    *outCacheBytes = 0;
    *outCacheTotalBytes = mla_kv_b_memory_cache_current_bytes();
    const char *residentPath = [residentBinPath fileSystemRepresentation];
    if (!residentPath || strlen(residentPath) >= PATH_MAX) {
        return read_absorbed_mla_kv_b_f32(residentBinPath,
                                          source,
                                          numHeads,
                                          qkNopeDim,
                                          vHeadDim,
                                          outValues,
                                          outBytes);
    }

    pthread_mutex_lock(&gMlaKVBMemoryCache.mutex);
    if (!gMlaKVBMemoryCache.enabled ||
        source.kv_b_f32_bytes == 0 ||
        source.kv_b_f32_bytes > gMlaKVBMemoryCache.max_bytes) {
        pthread_mutex_unlock(&gMlaKVBMemoryCache.mutex);
        return read_absorbed_mla_kv_b_f32(residentBinPath,
                                          source,
                                          numHeads,
                                          qkNopeDim,
                                          vHeadDim,
                                          outValues,
                                          outBytes);
    }
    for (int i = 0; i < MAX_MLA_KV_B_CACHE_ENTRIES; i++) {
        MlaKVBMemoryCacheEntry *entry = &gMlaKVBMemoryCache.entries[i];
        if (mla_kv_b_cache_entry_matches(entry,
                                         residentPath,
                                         layerId,
                                         source,
                                         numHeads,
                                         qkNopeDim,
                                         vHeadDim)) {
            *outValues = entry->values;
            *outBytes = entry->bytes;
            *outOwned = 0;
            *outCacheHit = 1;
            *outCacheBytes = entry->bytes;
            *outCacheTotalBytes = gMlaKVBMemoryCache.bytes;
            pthread_mutex_unlock(&gMlaKVBMemoryCache.mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&gMlaKVBMemoryCache.mutex);

    float *values = NULL;
    uint64_t bytes = 0;
    if (!read_absorbed_mla_kv_b_f32(residentBinPath,
                                    source,
                                    numHeads,
                                    qkNopeDim,
                                    vHeadDim,
                                    &values,
                                    &bytes)) {
        return 0;
    }
    *outValues = values;
    *outBytes = bytes;
    *outOwned = 1;

    pthread_mutex_lock(&gMlaKVBMemoryCache.mutex);
    if (!gMlaKVBMemoryCache.enabled ||
        bytes == 0 ||
        bytes > gMlaKVBMemoryCache.max_bytes) {
        *outCacheTotalBytes = gMlaKVBMemoryCache.bytes;
        pthread_mutex_unlock(&gMlaKVBMemoryCache.mutex);
        return 1;
    }
    for (int i = 0; i < MAX_MLA_KV_B_CACHE_ENTRIES; i++) {
        MlaKVBMemoryCacheEntry *entry = &gMlaKVBMemoryCache.entries[i];
        if (mla_kv_b_cache_entry_matches(entry,
                                         residentPath,
                                         layerId,
                                         source,
                                         numHeads,
                                         qkNopeDim,
                                         vHeadDim)) {
            free(values);
            *outValues = entry->values;
            *outBytes = entry->bytes;
            *outOwned = 0;
            *outCacheHit = 1;
            *outCacheBytes = entry->bytes;
            *outCacheTotalBytes = gMlaKVBMemoryCache.bytes;
            pthread_mutex_unlock(&gMlaKVBMemoryCache.mutex);
            return 1;
        }
    }
    int freeSlot = -1;
    for (int i = 0; i < MAX_MLA_KV_B_CACHE_ENTRIES; i++) {
        if (!gMlaKVBMemoryCache.entries[i].valid) {
            freeSlot = i;
            break;
        }
    }
    if (freeSlot >= 0 &&
        bytes <= gMlaKVBMemoryCache.max_bytes - gMlaKVBMemoryCache.bytes) {
        MlaKVBMemoryCacheEntry *entry = &gMlaKVBMemoryCache.entries[freeSlot];
        memset(entry, 0, sizeof(*entry));
        entry->valid = 1;
        snprintf(entry->resident_path, sizeof(entry->resident_path), "%s", residentPath);
        entry->layer_id = layerId;
        entry->num_heads = numHeads;
        entry->qk_nope_dim = qkNopeDim;
        entry->v_head_dim = vHeadDim;
        entry->embed_q = source.embed_q;
        entry->unembed_out = source.unembed_out;
        entry->bytes = bytes;
        entry->values = values;
        gMlaKVBMemoryCache.bytes += bytes;
        *outOwned = 0;
        *outCacheStored = 1;
        *outCacheBytes = bytes;
        *outCacheTotalBytes = gMlaKVBMemoryCache.bytes;
    } else {
        *outCacheTotalBytes = gMlaKVBMemoryCache.bytes;
    }
    pthread_mutex_unlock(&gMlaKVBMemoryCache.mutex);
    return 1;
}

static int mla_attention_cache_byte_counts(NSString *cacheLayoutPath,
                                           uint64_t layerId,
                                           uint32_t contextLength,
                                           uint32_t expectedWidth,
                                           uint64_t maxCacheFileBytes,
                                           uint64_t maxCacheReadBytes,
                                           uint64_t *outRawBytes,
                                           uint64_t *outF32Bytes) {
    NSDictionary *cacheLayout = load_json_dictionary(cacheLayoutPath);
    uint64_t totalBytes = 0;
    DecodeCacheSegmentInfo segment = {0};
    if (!find_decode_cache_segment_info(cacheLayout,
                                        layerId,
                                        @"mla_kv",
                                        maxCacheFileBytes,
                                        &segment,
                                        &totalBytes)) {
        return 0;
    }
    (void)totalBytes;
    if (contextLength == 0 || contextLength > segment.max_context_tokens) {
        fprintf(stderr,
                "ERROR: context length %u exceeds max context %llu\n",
                contextLength,
                (unsigned long long)segment.max_context_tokens);
        return 0;
    }
    if (segment.width != expectedWidth) {
        fprintf(stderr,
                "ERROR: cache width %u does not match expected %u\n",
                segment.width,
                expectedWidth);
        return 0;
    }
    uint64_t rawBytes = 0;
    uint64_t valueCount = 0;
    uint64_t f32Bytes = 0;
    if (!checked_mul_u64((uint64_t)segment.width,
                         (uint64_t)segment.dtype_bytes,
                         &rawBytes) ||
        !checked_mul_u64(rawBytes, (uint64_t)contextLength, &rawBytes) ||
        rawBytes > maxCacheReadBytes ||
        !checked_mul_u64((uint64_t)contextLength,
                         (uint64_t)segment.width,
                         &valueCount) ||
        !checked_mul_u64(valueCount, sizeof(float), &f32Bytes)) {
        fprintf(stderr,
                "ERROR: MLA cache byte count exceeds bounds or read limit %llu\n",
                (unsigned long long)maxCacheReadBytes);
        return 0;
    }
    *outRawBytes = rawBytes;
    *outF32Bytes = f32Bytes;
    return 1;
}

static int encode_glm_mla_attention(id<MTLCommandBuffer> cmd,
                                    id<MTLComputePipelineState> pipe,
                                    id<MTLBuffer> qNope,
                                    id<MTLBuffer> qRope,
                                    id<MTLBuffer> cache,
                                    id<MTLBuffer> kvB,
                                    id<MTLBuffer> out,
                                    uint32_t contextLength,
                                    uint32_t numHeads,
                                    uint32_t kvLoraDim,
                                    uint32_t qkNopeDim,
                                    uint32_t ropeDim,
                                    uint32_t vHeadDim,
                                    float attentionScale,
                                    float ropeTheta,
                                    uint32_t ropeInterleave,
                                    uint32_t cachePositionOffset) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create MLA attention encoder\n");
        return 0;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:qNope offset:0 atIndex:0];
    [enc setBuffer:qRope offset:0 atIndex:1];
    [enc setBuffer:cache offset:0 atIndex:2];
    [enc setBuffer:kvB offset:0 atIndex:3];
    [enc setBuffer:out offset:0 atIndex:4];
    [enc setBytes:&contextLength length:sizeof(contextLength) atIndex:5];
    [enc setBytes:&numHeads length:sizeof(numHeads) atIndex:6];
    [enc setBytes:&kvLoraDim length:sizeof(kvLoraDim) atIndex:7];
    [enc setBytes:&qkNopeDim length:sizeof(qkNopeDim) atIndex:8];
    [enc setBytes:&ropeDim length:sizeof(ropeDim) atIndex:9];
    [enc setBytes:&vHeadDim length:sizeof(vHeadDim) atIndex:10];
    [enc setBytes:&attentionScale length:sizeof(attentionScale) atIndex:11];
    [enc setBytes:&ropeTheta length:sizeof(ropeTheta) atIndex:12];
    [enc setBytes:&ropeInterleave length:sizeof(ropeInterleave) atIndex:13];
    [enc setBytes:&cachePositionOffset length:sizeof(cachePositionOffset) atIndex:14];
    if (contextLength >= 2u) {
        if ((NSUInteger)vHeadDim > [pipe maxTotalThreadsPerThreadgroup] ||
            vHeadDim > 1024u) {
            fprintf(stderr,
                    "ERROR: threadgroup MLA v_head_dim %u exceeds max threadgroup threads %lu or 1024 partial slots\n",
                    vHeadDim,
                    (unsigned long)[pipe maxTotalThreadsPerThreadgroup]);
            [enc endEncoding];
            return 0;
        }
        [enc dispatchThreadgroups:MTLSizeMake(1, numHeads, 1)
             threadsPerThreadgroup:MTLSizeMake(vHeadDim, 1, 1)];
    } else {
        NSUInteger totalThreads = (NSUInteger)numHeads * (NSUInteger)vHeadDim;
        [enc dispatchThreads:MTLSizeMake(totalThreads, 1, 1)
       threadsPerThreadgroup:threadgroup_1d(pipe, totalThreads)];
    }
    [enc endEncoding];
    return 1;
}

static int run_mla_attention_probe(id<MTLDevice> device,
                                   NSString *residentBinPath,
                                   MlaAttentionValueSourceInfo valueSource,
                                   NSString *cacheLayoutPath,
                                   NSString *cacheFilePath,
                                   int layerId,
                                   NSString *qNopePath,
                                   NSString *qRopePath,
                                   NSString *directKVBPath,
                                   NSData *qNopeDataOverride,
                                   NSData *qRopeDataOverride,
                                   NSString *outputPath,
                                   uint32_t contextLength,
                                   uint32_t numHeads,
                                   uint32_t qkNopeDim,
                                   uint32_t ropeDim,
                                   uint32_t vHeadDim,
                                   uint32_t cachePositionOffset,
                                   float attentionScale,
                                   float ropeTheta,
                                   int ropeInterleave,
                                   uint64_t maxCacheFileBytes,
                                   uint64_t maxCacheReadBytes,
                                   NSData **outputDataOut,
                                   MlaAttentionProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    uint32_t kvLoraDim = valueSource.kv_lora_dim;
    uint32_t cacheWidth = kvLoraDim + ropeDim;
    uint64_t qNopeBytes = (uint64_t)numHeads * qkNopeDim * sizeof(float);
    uint64_t qRopeBytes = (uint64_t)numHeads * ropeDim * sizeof(float);
    uint64_t outputBytes = (uint64_t)numHeads * vHeadDim * sizeof(float);
    if (qNopeBytes > (uint64_t)NSUIntegerMax ||
        qRopeBytes > (uint64_t)NSUIntegerMax ||
        outputBytes > (uint64_t)NSUIntegerMax) {
        fprintf(stderr, "ERROR: MLA query/output buffer sizes exceed NSUIntegerMax\n");
        return 0;
    }
    NSData *qNopeData =
        qNopeDataOverride ?: [NSData dataWithContentsOfFile:qNopePath];
    NSData *qRopeData =
        qRopeDataOverride ?: [NSData dataWithContentsOfFile:qRopePath];
    if (!qNopeData || !qRopeData) {
        fprintf(stderr, "ERROR: failed to read MLA q_nope/q_rope inputs\n");
        return 0;
    }
    if ((uint64_t)[qNopeData length] != qNopeBytes ||
        (uint64_t)[qRopeData length] != qRopeBytes) {
        fprintf(stderr,
                "ERROR: MLA query byte mismatch: q_nope %llu/%llu q_rope %llu/%llu\n",
                (unsigned long long)[qNopeData length],
                (unsigned long long)qNopeBytes,
                (unsigned long long)[qRopeData length],
                (unsigned long long)qRopeBytes);
        return 0;
    }
    double totalStarted = now_seconds();
    float *cacheF32 = NULL;
    uint64_t cacheF32Bytes = 0;
    uint64_t rawCacheBytes = 0;
    double cacheStarted = now_seconds();
    if (!read_decode_cache_segment_f32(cacheLayoutPath,
                                       cacheFilePath,
                                       (uint64_t)layerId,
                                       contextLength,
                                       cacheWidth,
                                       maxCacheFileBytes,
                                       maxCacheReadBytes,
                                       &cacheF32,
                                       &cacheF32Bytes,
                                       &rawCacheBytes)) {
        return 0;
    }
    stats->cache_read_seconds = now_seconds() - cacheStarted;

    float *kvBF32 = NULL;
    uint64_t kvBF32Bytes = 0;
    int kvBF32Owned = 1;
    int valueCacheHit = 0;
    int valueCacheStored = 0;
    uint64_t valueCacheBytes = 0;
    uint64_t valueCacheTotalBytes = 0;
    double valueStarted = now_seconds();
    if (directKVBPath) {
        NSData *kvBData = [NSData dataWithContentsOfFile:directKVBPath];
        if (!kvBData || (uint64_t)[kvBData length] != valueSource.kv_b_f32_bytes) {
            fprintf(stderr,
                    "ERROR: direct MLA KV-B byte mismatch: %llu/%llu\n",
                    (unsigned long long)(kvBData ? [kvBData length] : 0),
                    (unsigned long long)valueSource.kv_b_f32_bytes);
            free(cacheF32);
            return 0;
        }
        if (valueSource.kv_b_f32_bytes > (uint64_t)SIZE_MAX) {
            fprintf(stderr, "ERROR: direct MLA KV-B F32 view is too large\n");
            free(cacheF32);
            return 0;
        }
        kvBF32 = (float *)malloc((size_t)valueSource.kv_b_f32_bytes);
        if (!kvBF32) {
            fprintf(stderr, "ERROR: failed to allocate direct MLA KV-B copy\n");
            free(cacheF32);
            return 0;
        }
        memcpy(kvBF32, [kvBData bytes], (size_t)valueSource.kv_b_f32_bytes);
        kvBF32Bytes = valueSource.kv_b_f32_bytes;
        kvBF32Owned = 1;
    } else {
        if (!read_absorbed_mla_kv_b_f32_cached(residentBinPath,
                                               valueSource,
                                               layerId,
                                               numHeads,
                                               qkNopeDim,
                                               vHeadDim,
                                               &kvBF32,
                                               &kvBF32Bytes,
                                               &kvBF32Owned,
                                               &valueCacheHit,
                                               &valueCacheStored,
                                               &valueCacheBytes,
                                               &valueCacheTotalBytes)) {
            free(cacheF32);
            return 0;
        }
    }
    stats->value_read_seconds = now_seconds() - valueStarted;
    if (cacheF32Bytes != (uint64_t)contextLength * cacheWidth * sizeof(float) ||
        kvBF32Bytes != valueSource.kv_b_f32_bytes) {
        fprintf(stderr, "ERROR: MLA attention internal byte estimate mismatch\n");
        free(cacheF32);
        if (kvBF32Owned) {
            free(kvBF32);
        }
        return 0;
    }
    if (attentionScale == 0.0f) {
        attentionScale = 1.0f / sqrtf((float)(qkNopeDim + ropeDim));
    }
    id<MTLLibrary> library = make_glm_moe_library(device);
    NSString *mlaKernelName = contextLength == 1
        ? @"glm_mla_attention_context1_f32"
        : (contextLength <= 32
            ? @"glm_mla_attention_context_small_f32"
            : @"glm_mla_attention_streaming_f32");
    id<MTLComputePipelineState> pipe =
        library ? make_glm_pipeline(device, library, mlaKernelName) : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> qNope = [device newBufferWithBytes:[qNopeData bytes]
                                              length:(NSUInteger)qNopeBytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> qRope = [device newBufferWithBytes:[qRopeData bytes]
                                              length:(NSUInteger)qRopeBytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> cache = [device newBufferWithBytesNoCopy:cacheF32
                                                    length:(NSUInteger)cacheF32Bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:^(void *pointer, NSUInteger length) {
                                                   (void)length;
                                                   free(pointer);
                                               }];
    if (cache) {
        cacheF32 = NULL;
    }
    void (^kvBDeallocator)(void *, NSUInteger) = nil;
    if (kvBF32Owned) {
        kvBDeallocator = ^(void *pointer, NSUInteger length) {
            (void)length;
            free(pointer);
        };
    }
    id<MTLBuffer> kvB = [device newBufferWithBytesNoCopy:kvBF32
                                                  length:(NSUInteger)kvBF32Bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:kvBDeallocator];
    if (kvB) {
        kvBF32 = NULL;
    }
    id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)outputBytes
                                               options:MTLResourceStorageModeShared];
    if (!library || !pipe || !queue || !qNope || !qRope || !cache || !kvB || !output) {
        fprintf(stderr, "ERROR: failed to allocate MLA attention Metal resources\n");
        free(cacheF32);
        if (kvBF32Owned) {
            free(kvBF32);
        }
        return 0;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create MLA attention command buffer\n");
        return 0;
    }
    if (!encode_glm_mla_attention(cmd,
                                  pipe,
                                  qNope,
                                  qRope,
                                  cache,
                                  kvB,
                                  output,
                                  contextLength,
                                  numHeads,
                                  kvLoraDim,
                                  qkNopeDim,
                                  ropeDim,
                                  vHeadDim,
                                  attentionScale,
                                  ropeTheta,
                                  ropeInterleave ? 1u : 0u,
                                  cachePositionOffset)) {
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    stats->kernel_seconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: MLA attention command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    double writeStarted = now_seconds();
    NSData *outputData = [NSData dataWithBytes:[output contents]
                                        length:(NSUInteger)outputBytes];
    if (outputPath && ![outputData writeToFile:outputPath atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write MLA attention output %s\n",
                [outputPath UTF8String]);
        return 0;
    }
    if (outputDataOut) {
        *outputDataOut = outputData;
    }
    stats->output_write_seconds = now_seconds() - writeStarted;
    stats->ok = 1;
    stats->raw_cache_bytes = rawCacheBytes;
    stats->cache_f32_bytes = cacheF32Bytes;
    stats->value_storage_bytes = valueSource.storage_bytes;
    stats->value_source_f32_bytes = valueSource.source_f32_bytes;
    stats->kv_b_f32_bytes = kvBF32Bytes;
    stats->value_cache_bytes = valueCacheBytes;
    stats->value_cache_total_bytes = valueCacheTotalBytes;
    stats->q_nope_bytes = qNopeBytes;
    stats->q_rope_bytes = qRopeBytes;
    stats->output_bytes = outputBytes;
    stats->scratch_bytes = rawCacheBytes + cacheF32Bytes +
                           valueSource.storage_bytes +
                           valueSource.source_f32_bytes +
                           kvBF32Bytes + qNopeBytes + qRopeBytes +
                           outputBytes;
    stats->elapsed_seconds = now_seconds() - totalStarted;
    stats->context_length = contextLength;
    stats->num_heads = numHeads;
    stats->kv_lora_dim = kvLoraDim;
    stats->qk_nope_dim = qkNopeDim;
    stats->rope_dim = ropeDim;
    stats->v_head_dim = vHeadDim;
    stats->cache_position_offset = cachePositionOffset;
    stats->attention_scale = attentionScale;
    stats->rope_theta = ropeTheta;
    stats->rope_interleave = ropeInterleave ? 1 : 0;
    stats->command_buffer_count = 1;
    stats->fused_with_rope = 0;
    stats->value_cache_enabled = mla_kv_b_memory_cache_enabled();
    stats->value_cache_hit = valueCacheHit;
    stats->value_cache_stored = valueCacheStored;
    stats->output0 = ((float *)[output contents])[0];
    return 1;
}

static int run_rope_split_mla_attention_fused_probe(
    id<MTLDevice> device,
    NSString *residentBinPath,
    MlaAttentionValueSourceInfo valueSource,
    NSString *cacheLayoutPath,
    NSString *cacheFilePath,
    int layerId,
    NSData *qBData,
    NSData *kData,
    NSString *outputPath,
    uint32_t contextLength,
    uint32_t numHeads,
    uint32_t qkNopeDim,
    uint32_t ropeDim,
    uint32_t vHeadDim,
    uint32_t startPosition,
    uint32_t cachePositionOffset,
    float attentionScale,
    float ropeTheta,
    int ropeInterleave,
    uint64_t maxCacheFileBytes,
    uint64_t maxCacheReadBytes,
    NSData **outputDataOut,
    RopeSplitProbeStats *ropeStats,
    MlaAttentionProbeStats *mlaStats) {
    memset(ropeStats, 0, sizeof(*ropeStats));
    memset(mlaStats, 0, sizeof(*mlaStats));
    if (qkNopeDim > UINT32_MAX - ropeDim) {
        fprintf(stderr, "ERROR: fused RoPE+MLA q head dim overflows uint32\n");
        return 0;
    }
    if (!qBData || !kData) {
        fprintf(stderr, "ERROR: fused RoPE+MLA requires in-memory q_b and k_rope\n");
        return 0;
    }

    uint32_t batchTokens = 1;
    uint64_t qBBytes = 0;
    uint64_t kBytes = 0;
    uint64_t qNopeBytes = 0;
    uint64_t qRopeBytes = 0;
    uint64_t ropeScratchBytes = rope_split_scratch_bytes(
        numHeads,
        qkNopeDim,
        ropeDim,
        batchTokens,
        &qBBytes,
        &kBytes,
        &qNopeBytes,
        &qRopeBytes
    );
    uint32_t kvLoraDim = valueSource.kv_lora_dim;
    uint32_t cacheWidth = kvLoraDim + ropeDim;
    uint64_t outputBytes = (uint64_t)numHeads * vHeadDim * sizeof(float);
    if (qBBytes > NSUIntegerMax ||
        kBytes > NSUIntegerMax ||
        qNopeBytes > NSUIntegerMax ||
        qRopeBytes > NSUIntegerMax ||
        outputBytes > NSUIntegerMax) {
        fprintf(stderr, "ERROR: fused RoPE+MLA buffers exceed NSUIntegerMax\n");
        return 0;
    }
    if ((uint64_t)[qBData length] != qBBytes || (uint64_t)[kData length] != kBytes) {
        fprintf(stderr,
                "ERROR: fused RoPE+MLA input bytes mismatch: q_b %llu/%llu k %llu/%llu\n",
                (unsigned long long)[qBData length],
                (unsigned long long)qBBytes,
                (unsigned long long)[kData length],
                (unsigned long long)kBytes);
        return 0;
    }

    double totalStarted = now_seconds();
    float *cacheF32 = NULL;
    uint64_t cacheF32Bytes = 0;
    uint64_t rawCacheBytes = 0;
    double cacheStarted = now_seconds();
    if (!read_decode_cache_segment_f32(cacheLayoutPath,
                                       cacheFilePath,
                                       (uint64_t)layerId,
                                       contextLength,
                                       cacheWidth,
                                       maxCacheFileBytes,
                                       maxCacheReadBytes,
                                       &cacheF32,
                                       &cacheF32Bytes,
                                       &rawCacheBytes)) {
        return 0;
    }
    mlaStats->cache_read_seconds = now_seconds() - cacheStarted;

    float *kvBF32 = NULL;
    uint64_t kvBF32Bytes = 0;
    int kvBF32Owned = 1;
    int valueCacheHit = 0;
    int valueCacheStored = 0;
    uint64_t valueCacheBytes = 0;
    uint64_t valueCacheTotalBytes = 0;
    double valueStarted = now_seconds();
    if (!read_absorbed_mla_kv_b_f32_cached(residentBinPath,
                                           valueSource,
                                           layerId,
                                           numHeads,
                                           qkNopeDim,
                                           vHeadDim,
                                           &kvBF32,
                                           &kvBF32Bytes,
                                           &kvBF32Owned,
                                           &valueCacheHit,
                                           &valueCacheStored,
                                           &valueCacheBytes,
                                           &valueCacheTotalBytes)) {
        free(cacheF32);
        return 0;
    }
    mlaStats->value_read_seconds = now_seconds() - valueStarted;
    if (cacheF32Bytes != (uint64_t)contextLength * cacheWidth * sizeof(float) ||
        kvBF32Bytes != valueSource.kv_b_f32_bytes ||
        cacheF32Bytes > NSUIntegerMax ||
        kvBF32Bytes > NSUIntegerMax) {
        fprintf(stderr, "ERROR: fused RoPE+MLA internal byte estimate mismatch\n");
        free(cacheF32);
        if (kvBF32Owned) {
            free(kvBF32);
        }
        return 0;
    }
    if (attentionScale == 0.0f) {
        attentionScale = 1.0f / sqrtf((float)(qkNopeDim + ropeDim));
    }

    id<MTLLibrary> library = make_glm_moe_library(device);
    NSString *mlaKernelName = contextLength == 1
        ? @"glm_mla_attention_context1_f32"
        : (contextLength <= 32
            ? @"glm_mla_attention_context_small_f32"
            : @"glm_mla_attention_streaming_f32");
    id<MTLComputePipelineState> ropePipe =
        library ? make_glm_pipeline(device, library, @"glm_rope_split_batch_f32") : nil;
    id<MTLComputePipelineState> mlaPipe =
        library ? make_glm_pipeline(device, library, mlaKernelName) : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> qB = [device newBufferWithBytes:[qBData bytes]
                                           length:(NSUInteger)qBBytes
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> k = [device newBufferWithBytes:[kData bytes]
                                          length:(NSUInteger)kBytes
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQNope = [device newBufferWithLength:(NSUInteger)qNopeBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQRope = [device newBufferWithLength:(NSUInteger)qRopeBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQ = [device newBufferWithLength:(NSUInteger)qRopeBytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> outK = [device newBufferWithLength:(NSUInteger)kBytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> cache = [device newBufferWithBytesNoCopy:cacheF32
                                                    length:(NSUInteger)cacheF32Bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:^(void *pointer, NSUInteger length) {
                                                   (void)length;
                                                   free(pointer);
                                               }];
    if (cache) {
        cacheF32 = NULL;
    }
    void (^kvBDeallocator)(void *, NSUInteger) = nil;
    if (kvBF32Owned) {
        kvBDeallocator = ^(void *pointer, NSUInteger length) {
            (void)length;
            free(pointer);
        };
    }
    id<MTLBuffer> kvB = [device newBufferWithBytesNoCopy:kvBF32
                                                  length:(NSUInteger)kvBF32Bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:kvBDeallocator];
    if (kvB) {
        kvBF32 = NULL;
    }
    id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)outputBytes
                                               options:MTLResourceStorageModeShared];
    if (!library || !ropePipe || !mlaPipe || !queue || !qB || !k || !outQNope ||
        !outQRope || !outQ || !outK || !cache || !kvB || !output) {
        fprintf(stderr, "ERROR: failed to allocate fused RoPE+MLA Metal resources\n");
        free(cacheF32);
        if (kvBF32Owned) {
            free(kvBF32);
        }
        return 0;
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create fused RoPE+MLA command buffer\n");
        return 0;
    }
    if (!encode_glm_rope_split_batch(cmd,
                                     ropePipe,
                                     qB,
                                     k,
                                     outQNope,
                                     outQRope,
                                     outQ,
                                     outK,
                                     numHeads,
                                     qkNopeDim,
                                     ropeDim,
                                     startPosition,
                                     ropeTheta,
                                     ropeInterleave ? 1u : 0u,
                                     batchTokens) ||
        !encode_glm_mla_attention(cmd,
                                  mlaPipe,
                                  outQNope,
                                  outQ,
                                  cache,
                                  kvB,
                                  output,
                                  contextLength,
                                  numHeads,
                                  kvLoraDim,
                                  qkNopeDim,
                                  ropeDim,
                                  vHeadDim,
                                  attentionScale,
                                  ropeTheta,
                                  ropeInterleave ? 1u : 0u,
                                  cachePositionOffset)) {
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    mlaStats->kernel_seconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: fused RoPE+MLA command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }

    double writeStarted = now_seconds();
    NSData *outputData = [NSData dataWithBytes:[output contents]
                                        length:(NSUInteger)outputBytes];
    if (outputPath && ![outputData writeToFile:outputPath atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write fused RoPE+MLA output %s\n",
                [outputPath UTF8String]);
        return 0;
    }
    if (outputDataOut) {
        *outputDataOut = outputData;
    }
    mlaStats->output_write_seconds = now_seconds() - writeStarted;

    ropeStats->ok = 1;
    ropeStats->scratch_bytes = ropeScratchBytes;
    ropeStats->elapsed_seconds = 0.0;
    ropeStats->num_heads = numHeads;
    ropeStats->qk_nope_dim = qkNopeDim;
    ropeStats->rope_dim = ropeDim;
    ropeStats->start_position = startPosition;
    ropeStats->batch_tokens = batchTokens;
    ropeStats->theta = ropeTheta;
    ropeStats->interleave = ropeInterleave ? 1 : 0;
    ropeStats->q_b_bytes = qBBytes;
    ropeStats->k_bytes = kBytes;
    ropeStats->q_nope_bytes = qNopeBytes;
    ropeStats->q_rope_bytes = qRopeBytes;
    ropeStats->command_buffer_count = 0;
    ropeStats->fused_with_mla = 1;
    ropeStats->input_buffer_direct = 0;
    ropeStats->q_output0 = ((float *)[outQ contents])[0];
    ropeStats->k_output0 = ((float *)[outK contents])[0];

    mlaStats->ok = 1;
    mlaStats->raw_cache_bytes = rawCacheBytes;
    mlaStats->cache_f32_bytes = cacheF32Bytes;
    mlaStats->value_storage_bytes = valueSource.storage_bytes;
    mlaStats->value_source_f32_bytes = valueSource.source_f32_bytes;
    mlaStats->kv_b_f32_bytes = kvBF32Bytes;
    mlaStats->value_cache_bytes = valueCacheBytes;
    mlaStats->value_cache_total_bytes = valueCacheTotalBytes;
    mlaStats->q_nope_bytes = qNopeBytes;
    mlaStats->q_rope_bytes = qRopeBytes;
    mlaStats->output_bytes = outputBytes;
    mlaStats->scratch_bytes = rawCacheBytes + cacheF32Bytes +
                              valueSource.storage_bytes +
                              valueSource.source_f32_bytes +
                              kvBF32Bytes + qNopeBytes + qRopeBytes +
                              outputBytes;
    mlaStats->elapsed_seconds = now_seconds() - totalStarted;
    mlaStats->context_length = contextLength;
    mlaStats->num_heads = numHeads;
    mlaStats->kv_lora_dim = kvLoraDim;
    mlaStats->qk_nope_dim = qkNopeDim;
    mlaStats->rope_dim = ropeDim;
    mlaStats->v_head_dim = vHeadDim;
    mlaStats->cache_position_offset = cachePositionOffset;
    mlaStats->attention_scale = attentionScale;
    mlaStats->rope_theta = ropeTheta;
    mlaStats->rope_interleave = ropeInterleave ? 1 : 0;
    mlaStats->command_buffer_count = 1;
    mlaStats->fused_with_rope = 1;
    mlaStats->value_cache_enabled = mla_kv_b_memory_cache_enabled();
    mlaStats->value_cache_hit = valueCacheHit;
    mlaStats->value_cache_stored = valueCacheStored;
    mlaStats->output0 = ((float *)[output contents])[0];
    return 1;
}

static int run_rope_mla_attention_output_fused_probe(
    id<MTLDevice> device,
    NSString *residentBinPath,
    id<MTLBuffer> residentMetalBuffer,
    MlaAttentionValueSourceInfo valueSource,
    NSString *cacheLayoutPath,
    NSString *cacheFilePath,
    int layerId,
    NSData *qBData,
    NSData *kData,
    id<MTLBuffer> qBBufferOverride,
    id<MTLBuffer> kBufferOverride,
    NSUInteger kBufferOffset,
    id<MTLBuffer> cacheBufferOverride,
    NSUInteger cacheBufferOffset,
    ResidentMxfp4MatrixInfo attnInfo,
    NSString *residualPath,
    NSData *residualDataOverride,
    id<MTLBuffer> residualBufferOverride,
    NSString *outputPath,
    uint32_t contextLength,
    uint32_t numHeads,
    uint32_t qkNopeDim,
    uint32_t ropeDim,
    uint32_t vHeadDim,
    uint32_t startPosition,
    uint32_t cachePositionOffset,
    float attentionScale,
    float ropeTheta,
    int ropeInterleave,
    uint64_t maxCacheFileBytes,
    uint64_t maxCacheReadBytes,
    NSData **outputDataOut,
    id<MTLBuffer> __strong *outputBufferOut,
    RopeSplitProbeStats *ropeStats,
    MlaAttentionProbeStats *mlaStats,
    AttnOutputProbeStats *attnStats) {
    memset(ropeStats, 0, sizeof(*ropeStats));
    memset(mlaStats, 0, sizeof(*mlaStats));
    memset(attnStats, 0, sizeof(*attnStats));
    if (outputDataOut) *outputDataOut = nil;
    if (outputBufferOut) *outputBufferOut = nil;
    if (qkNopeDim > UINT32_MAX - ropeDim) {
        fprintf(stderr, "ERROR: fused RoPE/MLA/attn-output q head dim overflows uint32\n");
        return 0;
    }

    uint32_t batchTokens = 1;
    uint64_t qBBytes = 0;
    uint64_t kBytes = 0;
    uint64_t qNopeBytes = 0;
    uint64_t qRopeBytes = 0;
    uint64_t ropeScratchBytes = rope_split_scratch_bytes(
        numHeads,
        qkNopeDim,
        ropeDim,
        batchTokens,
        &qBBytes,
        &kBytes,
        &qNopeBytes,
        &qRopeBytes
    );
    uint32_t kvLoraDim = valueSource.kv_lora_dim;
    uint32_t cacheWidth = kvLoraDim + ropeDim;
    uint64_t mlaOutputBytes = (uint64_t)numHeads * vHeadDim * sizeof(float);
    uint64_t attnInputBytes = (uint64_t)attnInfo.in_dim * sizeof(float);
    uint64_t attnOutputBytes = (uint64_t)attnInfo.out_dim * sizeof(float);
    if (qBBytes > NSUIntegerMax ||
        kBytes > NSUIntegerMax ||
        qNopeBytes > NSUIntegerMax ||
        qRopeBytes > NSUIntegerMax ||
        mlaOutputBytes > NSUIntegerMax ||
        attnOutputBytes > NSUIntegerMax ||
        attnInfo.total_bytes > NSUIntegerMax) {
        fprintf(stderr, "ERROR: fused RoPE/MLA/attn-output buffers exceed NSUIntegerMax\n");
        return 0;
    }
    if (mlaOutputBytes != attnInputBytes) {
        fprintf(stderr,
                "ERROR: fused RoPE/MLA/attn-output dims mismatch mla_out=%llu attn_in=%u\n",
                (unsigned long long)(mlaOutputBytes / sizeof(float)),
                attnInfo.in_dim);
        return 0;
    }
    int directInputBuffers = qBBufferOverride && kBufferOverride;
    if (directInputBuffers) {
        if ((uint64_t)[qBBufferOverride length] < qBBytes ||
            (uint64_t)[kBufferOverride length] < (uint64_t)kBufferOffset + kBytes) {
            fprintf(stderr, "ERROR: fused RoPE/MLA/attn-output direct input buffers are too small\n");
            return 0;
        }
    } else {
        if (!qBData || !kData) {
            fprintf(stderr, "ERROR: fused RoPE/MLA/attn-output requires q_b and k_rope inputs\n");
            return 0;
        }
        if ((uint64_t)[qBData length] != qBBytes || (uint64_t)[kData length] != kBytes) {
            fprintf(stderr,
                    "ERROR: fused RoPE/MLA/attn-output input bytes mismatch: q_b %llu/%llu k %llu/%llu\n",
                    (unsigned long long)[qBData length],
                    (unsigned long long)qBBytes,
                    (unsigned long long)[kData length],
                    (unsigned long long)kBytes);
            return 0;
        }
    }

    NSData *residualData = nil;
    if (residualBufferOverride) {
        if ((uint64_t)[residualBufferOverride length] < attnOutputBytes) {
            fprintf(stderr,
                    "ERROR: fused RoPE/MLA/attn-output residual buffer bytes %llu are smaller than expected %llu\n",
                    (unsigned long long)[residualBufferOverride length],
                    (unsigned long long)attnOutputBytes);
            return 0;
        }
    } else {
        residualData =
            residualDataOverride ?: [NSData dataWithContentsOfFile:residualPath];
        if (!residualData) {
            fprintf(stderr, "ERROR: failed to read fused RoPE/MLA/attn-output residual %s\n",
                    residualPath ? [residualPath UTF8String] : "<memory>");
            return 0;
        }
        if ((uint64_t)[residualData length] != attnOutputBytes) {
            fprintf(stderr,
                    "ERROR: fused RoPE/MLA/attn-output residual bytes %llu do not match expected %llu\n",
                    (unsigned long long)[residualData length],
                    (unsigned long long)attnOutputBytes);
            return 0;
        }
    }

    double totalStarted = now_seconds();
    float *cacheF32 = NULL;
    uint64_t cacheF32Bytes = 0;
    uint64_t rawCacheBytes = 0;
    uint64_t currentCacheRowOffsetBytes = 0;
    uint64_t cacheRowBytes = (uint64_t)cacheWidth * sizeof(float);
    double cacheStarted = now_seconds();
    int directCacheBuffer = cacheBufferOverride != nil;
    if (directCacheBuffer) {
        if ((uint64_t)[cacheBufferOverride length] <
            (uint64_t)cacheBufferOffset + cacheRowBytes) {
            fprintf(stderr,
                    "ERROR: fused RoPE/MLA/attn-output direct cache buffer is too small\n");
            return 0;
        }
        if (contextLength == 1) {
            cacheF32Bytes = cacheRowBytes;
        } else if (!read_decode_cache_prefix_with_current_slot_f32(
                       cacheLayoutPath,
                       cacheFilePath,
                       (uint64_t)layerId,
                       contextLength,
                       cacheWidth,
                       maxCacheFileBytes,
                       maxCacheReadBytes,
                       &cacheF32,
                       &cacheF32Bytes,
                       &rawCacheBytes,
                       &currentCacheRowOffsetBytes)) {
            return 0;
        }
    } else {
        if (!read_decode_cache_segment_f32(cacheLayoutPath,
                                           cacheFilePath,
                                           (uint64_t)layerId,
                                           contextLength,
                                           cacheWidth,
                                           maxCacheFileBytes,
                                           maxCacheReadBytes,
                                           &cacheF32,
                                           &cacheF32Bytes,
                                           &rawCacheBytes)) {
            return 0;
        }
    }
    mlaStats->cache_read_seconds = now_seconds() - cacheStarted;

    float *kvBF32 = NULL;
    uint64_t kvBF32Bytes = 0;
    int kvBF32Owned = 1;
    int valueCacheHit = 0;
    int valueCacheStored = 0;
    uint64_t valueCacheBytes = 0;
    uint64_t valueCacheTotalBytes = 0;
    double valueStarted = now_seconds();
    if (!read_absorbed_mla_kv_b_f32_cached(residentBinPath,
                                           valueSource,
                                           layerId,
                                           numHeads,
                                           qkNopeDim,
                                           vHeadDim,
                                           &kvBF32,
                                           &kvBF32Bytes,
                                           &kvBF32Owned,
                                           &valueCacheHit,
                                           &valueCacheStored,
                                           &valueCacheBytes,
                                           &valueCacheTotalBytes)) {
        free(cacheF32);
        return 0;
    }
    mlaStats->value_read_seconds = now_seconds() - valueStarted;
    if (cacheF32Bytes != (uint64_t)contextLength * cacheWidth * sizeof(float) ||
        kvBF32Bytes != valueSource.kv_b_f32_bytes ||
        cacheF32Bytes > NSUIntegerMax ||
        kvBF32Bytes > NSUIntegerMax) {
        fprintf(stderr, "ERROR: fused RoPE/MLA/attn-output internal byte estimate mismatch\n");
        free(cacheF32);
        if (kvBF32Owned) free(kvBF32);
        return 0;
    }
    if (attentionScale == 0.0f) {
        attentionScale = 1.0f / sqrtf((float)(qkNopeDim + ropeDim));
    }

    int useResidentMetalWeights = residentMetalBuffer != nil;
    NSUInteger weightOffset = 0;
    NSUInteger scalesOffset = (NSUInteger)attnInfo.weight.size;
    uint64_t matrixAllocBytes = useResidentMetalWeights
        ? 0
        : round_up_u64(attnInfo.total_bytes, 2 * 1024 * 1024);
    void *matrixPtr = NULL;
    if (useResidentMetalWeights) {
        uint64_t residentLength = (uint64_t)[residentMetalBuffer length];
        uint64_t weightEnd = 0;
        uint64_t scalesEnd = 0;
        if (!checked_add_u64(attnInfo.weight.offset,
                             attnInfo.weight.size,
                             &weightEnd) ||
            !checked_add_u64(attnInfo.scales.offset,
                             attnInfo.scales.size,
                             &scalesEnd) ||
            attnInfo.weight.offset > (uint64_t)NSUIntegerMax ||
            attnInfo.scales.offset > (uint64_t)NSUIntegerMax ||
            weightEnd > residentLength ||
            scalesEnd > residentLength) {
            fprintf(stderr,
                    "ERROR: resident Metal buffer does not cover fused RoPE/MLA/attn-output weights\n");
            free(cacheF32);
            if (kvBF32Owned) free(kvBF32);
            return 0;
        }
        weightOffset = (NSUInteger)attnInfo.weight.offset;
        scalesOffset = (NSUInteger)attnInfo.scales.offset;
        attnStats->resident_mmap_backed = 1;
        attnStats->bytes_read = 0;
    } else {
        if (posix_memalign(&matrixPtr, 2 * 1024 * 1024, (size_t)matrixAllocBytes) != 0 ||
            !matrixPtr) {
            fprintf(stderr, "ERROR: failed to allocate fused RoPE/MLA/attn-output aligned matrix\n");
            free(cacheF32);
            if (kvBF32Owned) free(kvBF32);
            free(matrixPtr);
            return 0;
        }
        int closeFd = 0;
        int fd = open_resident_read_fd(residentBinPath, &closeFd);
        if (fd < 0) {
            fprintf(stderr,
                    "ERROR: failed to open resident file for fused RoPE/MLA/attn-output %s: %s\n",
                    [residentBinPath UTF8String],
                    strerror(errno));
            free(cacheF32);
            if (kvBF32Owned) free(kvBF32);
            free(matrixPtr);
            return 0;
        }
        double attnReadStarted = now_seconds();
        int attnReadOk =
            pread_exact_or_report(fd,
                                  matrixPtr,
                                  attnInfo.weight.size,
                                  attnInfo.weight.offset,
                                  [residentBinPath UTF8String]) &&
            pread_exact_or_report(fd,
                                  (uint8_t *)matrixPtr + attnInfo.weight.size,
                                  attnInfo.scales.size,
                                  attnInfo.scales.offset,
                                  [residentBinPath UTF8String]);
        attnStats->read_seconds = now_seconds() - attnReadStarted;
        close_resident_read_fd(fd, closeFd);
        if (!attnReadOk) {
            free(cacheF32);
            if (kvBF32Owned) free(kvBF32);
            free(matrixPtr);
            return 0;
        }
        attnStats->bytes_read = attnInfo.total_bytes;
    }

    id<MTLLibrary> library = make_glm_moe_library(device);
    NSString *mlaKernelName = contextLength == 1
        ? @"glm_mla_attention_context1_f32"
        : (contextLength <= 32
            ? @"glm_mla_attention_context_small_f32"
            : @"glm_mla_attention_streaming_f32");
    id<MTLComputePipelineState> ropePipe =
        library ? make_glm_pipeline(device, library, @"glm_rope_split_batch_f32") : nil;
    id<MTLComputePipelineState> mlaPipe =
        library ? make_glm_pipeline(device, library, mlaKernelName) : nil;
    id<MTLComputePipelineState> matvecAddPipe =
        library ? make_resident_mxfp4_matvec_add_pipeline(device, library, attnInfo) : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> qB = directInputBuffers
        ? qBBufferOverride
        : [device newBufferWithBytes:[qBData bytes]
                              length:(NSUInteger)qBBytes
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> k = directInputBuffers
        ? kBufferOverride
        : [device newBufferWithBytes:[kData bytes]
                              length:(NSUInteger)kBytes
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQNope = [device newBufferWithLength:(NSUInteger)qNopeBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQRope = [device newBufferWithLength:(NSUInteger)qRopeBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQ = [device newBufferWithLength:(NSUInteger)qRopeBytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> outK = [device newBufferWithLength:(NSUInteger)kBytes
                                             options:MTLResourceStorageModeShared];
    int useDirectCacheBuffer = directCacheBuffer && contextLength == 1;
    int copyDirectCurrentCacheRow = directCacheBuffer && contextLength > 1;
    id<MTLBuffer> cache = useDirectCacheBuffer
        ? cacheBufferOverride
        : [device newBufferWithBytesNoCopy:cacheF32
                                    length:(NSUInteger)cacheF32Bytes
                                   options:MTLResourceStorageModeShared
                               deallocator:^(void *pointer, NSUInteger length) {
                                   (void)length;
                                   free(pointer);
                               }];
    if (!useDirectCacheBuffer && cache) cacheF32 = NULL;
    void (^kvBDeallocator)(void *, NSUInteger) = nil;
    if (kvBF32Owned) {
        kvBDeallocator = ^(void *pointer, NSUInteger length) {
            (void)length;
            free(pointer);
        };
    }
    id<MTLBuffer> kvB = [device newBufferWithBytesNoCopy:kvBF32
                                                  length:(NSUInteger)kvBF32Bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:kvBDeallocator];
    if (kvB) kvBF32 = NULL;
    id<MTLBuffer> mlaOutput = [device newBufferWithLength:(NSUInteger)mlaOutputBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> matrix = useResidentMetalWeights
        ? residentMetalBuffer
        : [device newBufferWithBytesNoCopy:matrixPtr
                                    length:(NSUInteger)matrixAllocBytes
                                   options:MTLResourceStorageModeShared
                               deallocator:^(void *pointer, NSUInteger length) {
                                   (void)length;
                                   free(pointer);
                               }];
    if (matrix) matrixPtr = NULL;
    id<MTLBuffer> residual = residualBufferOverride ?:
        [device newBufferWithBytes:[residualData bytes]
                            length:(NSUInteger)attnOutputBytes
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> attnOutput = [device newBufferWithLength:(NSUInteger)attnOutputBytes
                                                   options:MTLResourceStorageModeShared];
    if (!library || !ropePipe || !mlaPipe || !matvecAddPipe || !queue ||
        !qB || !k || !outQNope || !outQRope || !outQ || !outK || !cache ||
        !kvB || !mlaOutput || !matrix || !residual || !attnOutput) {
        fprintf(stderr, "ERROR: failed to allocate fused RoPE/MLA/attn-output Metal resources\n");
        free(cacheF32);
        if (kvBF32Owned) free(kvBF32);
        free(matrixPtr);
        return 0;
    }
    if (!useResidentMetalWeights) {
        [matrix didModifyRange:NSMakeRange(0, (NSUInteger)attnInfo.total_bytes)];
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        fprintf(stderr, "ERROR: failed to create fused RoPE/MLA/attn-output command buffer\n");
        return 0;
    }
    if (copyDirectCurrentCacheRow) {
        if (cacheRowBytes > NSUIntegerMax ||
            currentCacheRowOffsetBytes > NSUIntegerMax ||
            cacheBufferOffset > NSUIntegerMax) {
            fprintf(stderr, "ERROR: direct current KV-A cache copy offset exceeds NSUIntegerMax\n");
            return 0;
        }
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (!blit) {
            fprintf(stderr, "ERROR: failed to create direct current KV-A cache blit encoder\n");
            return 0;
        }
        [blit copyFromBuffer:cacheBufferOverride
                sourceOffset:cacheBufferOffset
                    toBuffer:cache
           destinationOffset:(NSUInteger)currentCacheRowOffsetBytes
                        size:(NSUInteger)cacheRowBytes];
        [blit endEncoding];
    }
    if (!encode_glm_rope_split_batch_with_offsets(cmd,
                                                  ropePipe,
                                                  qB,
                                                  0,
                                                  k,
                                                  directInputBuffers ? kBufferOffset : 0,
                                                  outQNope,
                                                  outQRope,
                                                  outQ,
                                                  outK,
                                                  numHeads,
                                                  qkNopeDim,
                                                  ropeDim,
                                                  startPosition,
                                                  ropeTheta,
                                                  ropeInterleave ? 1u : 0u,
                                                  batchTokens) ||
        !encode_glm_mla_attention(cmd,
                                  mlaPipe,
                                  outQNope,
                                  outQ,
                                  cache,
                                  kvB,
                                  mlaOutput,
                                  contextLength,
                                  numHeads,
                                  kvLoraDim,
                                  qkNopeDim,
                                  ropeDim,
                                  vHeadDim,
                                  attentionScale,
                                  ropeTheta,
                                  ropeInterleave ? 1u : 0u,
                                  useDirectCacheBuffer ? 0u : cachePositionOffset) ||
        !encode_glm_mxfp4_matvec_add_with_offsets(
            cmd,
            matvecAddPipe,
            matrix,
            weightOffset,
            matrix,
            scalesOffset,
            attnInfo,
            mlaOutput,
            residual,
            attnOutput)) {
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    double kernelSeconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: fused RoPE/MLA/attn-output command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }

    double writeStarted = now_seconds();
    NSData *outputData = nil;
    if (outputDataOut || outputPath) {
        outputData = [NSData dataWithBytes:[attnOutput contents]
                                    length:(NSUInteger)attnOutputBytes];
    }
    if (outputPath && ![outputData writeToFile:outputPath atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write fused RoPE/MLA/attn-output output %s\n",
                [outputPath UTF8String]);
        return 0;
    }
    if (outputDataOut) *outputDataOut = outputData;
    if (outputBufferOut) *outputBufferOut = attnOutput;
    attnStats->output_write_seconds = now_seconds() - writeStarted;

    ropeStats->ok = 1;
    ropeStats->scratch_bytes = ropeScratchBytes;
    ropeStats->elapsed_seconds = 0.0;
    ropeStats->num_heads = numHeads;
    ropeStats->qk_nope_dim = qkNopeDim;
    ropeStats->rope_dim = ropeDim;
    ropeStats->start_position = startPosition;
    ropeStats->batch_tokens = batchTokens;
    ropeStats->theta = ropeTheta;
    ropeStats->interleave = ropeInterleave ? 1 : 0;
    ropeStats->q_b_bytes = qBBytes;
    ropeStats->k_bytes = kBytes;
    ropeStats->q_nope_bytes = qNopeBytes;
    ropeStats->q_rope_bytes = qRopeBytes;
    ropeStats->command_buffer_count = 0;
    ropeStats->fused_with_mla = 1;
    ropeStats->input_buffer_direct = directInputBuffers ? 1 : 0;
    ropeStats->q_output0 = ((float *)[outQ contents])[0];
    ropeStats->k_output0 = ((float *)[outK contents])[0];

    mlaStats->ok = 1;
    mlaStats->raw_cache_bytes = rawCacheBytes;
    mlaStats->cache_f32_bytes = cacheF32Bytes;
    mlaStats->value_storage_bytes = valueSource.storage_bytes;
    mlaStats->value_source_f32_bytes = valueSource.source_f32_bytes;
    mlaStats->kv_b_f32_bytes = kvBF32Bytes;
    mlaStats->value_cache_bytes = valueCacheBytes;
    mlaStats->value_cache_total_bytes = valueCacheTotalBytes;
    mlaStats->q_nope_bytes = qNopeBytes;
    mlaStats->q_rope_bytes = qRopeBytes;
    mlaStats->output_bytes = mlaOutputBytes;
    mlaStats->scratch_bytes = rawCacheBytes + cacheF32Bytes +
                              valueSource.storage_bytes +
                              valueSource.source_f32_bytes +
                              kvBF32Bytes + qNopeBytes + qRopeBytes +
                              mlaOutputBytes;
    mlaStats->elapsed_seconds = mlaStats->cache_read_seconds +
                                mlaStats->value_read_seconds;
    mlaStats->context_length = contextLength;
    mlaStats->num_heads = numHeads;
    mlaStats->kv_lora_dim = kvLoraDim;
    mlaStats->qk_nope_dim = qkNopeDim;
    mlaStats->rope_dim = ropeDim;
    mlaStats->v_head_dim = vHeadDim;
    mlaStats->cache_position_offset = cachePositionOffset;
    mlaStats->attention_scale = attentionScale;
    mlaStats->rope_theta = ropeTheta;
    mlaStats->rope_interleave = ropeInterleave ? 1 : 0;
    mlaStats->kernel_seconds = 0.0;
    mlaStats->output_write_seconds = 0.0;
    mlaStats->command_buffer_count = 0;
    mlaStats->fused_with_rope = 1;
    mlaStats->value_cache_enabled = mla_kv_b_memory_cache_enabled();
    mlaStats->value_cache_hit = valueCacheHit;
    mlaStats->value_cache_stored = valueCacheStored;
    mlaStats->output0 = ((float *)[mlaOutput contents])[0];

    attnStats->ok = 1;
    attnStats->out_dim = attnInfo.out_dim;
    attnStats->in_dim = attnInfo.in_dim;
    attnStats->group_size = attnInfo.group_size;
    attnStats->input_bytes = attnInputBytes;
    attnStats->residual_bytes = attnOutputBytes;
    attnStats->projection_bytes = attnOutputBytes;
    attnStats->output_bytes = attnOutputBytes;
    attnStats->projection_kernel_seconds = kernelSeconds;
    attnStats->residual_add_seconds = 0.0;
    attnStats->command_buffer_count = 1;
    attnStats->fused_matvec_add = 1;
    attnStats->fused_with_rope_mla = 1;
    attnStats->elapsed_seconds = attnStats->read_seconds +
                                 attnStats->projection_kernel_seconds +
                                 attnStats->output_write_seconds;
    attnStats->output0 = ((float *)[attnOutput contents])[0];
    (void)totalStarted;
    return 1;
}

static float router_score_value(float logit, const char *mode) {
    if (strcmp(mode, "raw") == 0 || strcmp(mode, "softmax") == 0) {
        return logit;
    }
    return 1.0f / (1.0f + expf(-logit));
}

static int compute_router_topk(const float *logits,
                               uint32_t n,
                               uint32_t k,
                               const float *selectionBias,
                               RouterTopKOptions options,
                               int *experts,
                               float *weights) {
    if (!logits || !experts || !weights || k == 0 || k > n || k > 64) {
        return 0;
    }
    float *scores = (float *)calloc(n, sizeof(float));
    float *choiceScores = (float *)calloc(n, sizeof(float));
    uint8_t *allowed = (uint8_t *)calloc(n, sizeof(uint8_t));
    if (!scores || !choiceScores || !allowed) {
        free(scores);
        free(choiceScores);
        free(allowed);
        return 0;
    }
    if (strcmp(options.score, "softmax") == 0) {
        float maxv = logits[0];
        for (uint32_t i = 1; i < n; i++) {
            if (logits[i] > maxv) {
                maxv = logits[i];
            }
        }
        for (uint32_t i = 0; i < n; i++) {
            scores[i] = expf(logits[i] - maxv);
        }
    } else {
        for (uint32_t i = 0; i < n; i++) {
            scores[i] = router_score_value(logits[i], options.score);
        }
    }
    for (uint32_t i = 0; i < n; i++) {
        choiceScores[i] = scores[i] + (selectionBias ? selectionBias[i] : 0.0f);
        allowed[i] = 1;
    }
    if (options.n_group > 1) {
        if (n % options.n_group != 0) {
            fprintf(stderr, "ERROR: invalid router grouping for %u experts\n", n);
            free(scores);
            free(choiceScores);
            free(allowed);
            return 0;
        }
        uint32_t perGroup = n / options.n_group;
        float *groupScores = (float *)calloc(options.n_group, sizeof(float));
        uint8_t *groupUsed = (uint8_t *)calloc(options.n_group, sizeof(uint8_t));
        if (!groupScores || !groupUsed) {
            free(groupScores);
            free(groupUsed);
            free(scores);
            free(choiceScores);
            free(allowed);
            return 0;
        }
        for (uint32_t group = 0; group < options.n_group; group++) {
            float best1 = -INFINITY;
            float best2 = -INFINITY;
            for (uint32_t i = 0; i < perGroup; i++) {
                float value = choiceScores[group * perGroup + i];
                if (value > best1) {
                    best2 = best1;
                    best1 = value;
                } else if (value > best2) {
                    best2 = value;
                }
            }
            groupScores[group] = best1 + best2;
        }
        memset(allowed, 0, n);
        for (uint32_t out = 0; out < options.topk_group; out++) {
            uint32_t best = 0;
            float bestScore = -INFINITY;
            for (uint32_t group = 0; group < options.n_group; group++) {
                if (!groupUsed[group] && groupScores[group] > bestScore) {
                    best = group;
                    bestScore = groupScores[group];
                }
            }
            groupUsed[best] = 1;
            for (uint32_t i = 0; i < perGroup; i++) {
                allowed[best * perGroup + i] = 1;
            }
        }
        free(groupScores);
        free(groupUsed);
    }
    uint8_t usedStack[64] = {0};
    uint8_t *usedDynamic = NULL;
    uint8_t *usedFlags = usedStack;
    if (n > sizeof(usedStack)) {
        usedDynamic = (uint8_t *)calloc(n, sizeof(uint8_t));
        if (!usedDynamic) {
            free(scores);
            free(choiceScores);
            free(allowed);
            return 0;
        }
        usedFlags = usedDynamic;
    }
    for (uint32_t out = 0; out < k; out++) {
        uint32_t best = 0;
        float bestScore = -INFINITY;
        for (uint32_t i = 0; i < n; i++) {
            if (allowed[i] && !usedFlags[i] && choiceScores[i] > bestScore) {
                best = i;
                bestScore = choiceScores[i];
            }
        }
        usedFlags[best] = 1;
        experts[out] = (int)best;
        weights[out] = scores[best];
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < k; i++) {
        sum += weights[i];
    }
    if (options.norm_topk_prob && sum != 0.0f) {
        for (uint32_t i = 0; i < k; i++) {
            weights[i] /= sum;
        }
    }
    for (uint32_t i = 0; i < k; i++) {
        weights[i] *= options.routed_scaling_factor;
    }
    free(usedDynamic);
    free(allowed);
    free(choiceScores);
    free(scores);
    return 1;
}

static NSDictionary *router_probe_dictionary(RouterProbeStats stats,
                                             RouterWeightInfo info,
                                             int layerId,
                                             NSString *inputPath) {
    NSMutableArray *experts = [NSMutableArray arrayWithCapacity:stats.top_k];
    NSMutableArray *weights = [NSMutableArray arrayWithCapacity:stats.top_k];
    for (uint32_t i = 0; i < stats.top_k; i++) {
        [experts addObject:@(stats.experts[i])];
        [weights addObject:@(stats.weights[i])];
    }
    NSMutableArray *logits = [NSMutableArray arrayWithCapacity:stats.num_experts];
    if (stats.logits) {
        for (uint32_t i = 0; i < stats.num_experts; i++) {
            [logits addObject:@(stats.logits[i])];
        }
    }
    return @{
        @"enabled": @YES,
        @"ok": @(stats.ok),
        @"layer": @(layerId),
        @"tensor": [NSString stringWithUTF8String:info.name],
        @"dtype": [NSString stringWithUTF8String:info.dtype],
        @"router_score": [NSString stringWithUTF8String:stats.router_score],
        @"top_k": @(stats.top_k),
        @"norm_topk_prob": @(stats.norm_topk_prob ? YES : NO),
        @"routed_scaling_factor": @(stats.routed_scaling_factor),
        @"n_group": @(stats.n_group),
        @"topk_group": @(stats.topk_group),
        @"topk_backend": stats.gpu_topk ? @"metal" : @"cpu",
        @"used_correction_bias": @(stats.used_correction_bias ? YES : NO),
        @"experts": experts,
        @"weights": weights,
        @"logits": logits,
        @"num_experts": @(info.num_experts),
        @"hidden_dim": @(info.hidden_dim),
        @"router_bytes_read": @(stats.router_bytes_read),
        @"correction_bias_bytes_read": @(stats.correction_bias_bytes_read),
        @"elapsed_seconds": @(stats.elapsed_seconds),
        @"router_read_seconds": @(stats.router_read_seconds),
        @"kernel_seconds": @(stats.kernel_seconds),
        @"command_buffer_count": @(stats.command_buffer_count),
        @"fused_with_rmsnorm": @(stats.fused_with_rmsnorm ? YES : NO),
        @"input_f32": inputPath,
    };
}

static int write_router_probe_json(NSString *path,
                                   RouterProbeStats stats,
                                   RouterWeightInfo info,
                                   int layerId,
                                   NSString *inputPath) {
    NSDictionary *payload = router_probe_dictionary(stats, info, layerId, inputPath);
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:&error];
    if (!data) {
        fprintf(stderr,
                "ERROR: failed to serialize router JSON: %s\n",
                error ? [[error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    if (![data writeToFile:path atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write router JSON %s\n", [path UTF8String]);
        return 0;
    }
    return 1;
}

static int run_router_probe(id<MTLDevice> device,
                            NSString *residentBinPath,
                            RouterWeightInfo info,
                            RouterBiasInfo biasInfo,
                            RouterTopKOptions topkOptions,
                            int layerId,
                            NSString *inputPath,
                            NSData *inputDataOverride,
                            uint32_t topK,
                            NSString *outputRouterJson,
                            int allowGpuTopK,
                            RouterProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    stats->top_k = topK;
    stats->num_experts = info.num_experts;
    snprintf(stats->router_score, sizeof(stats->router_score), "%s", topkOptions.score);
    stats->routed_scaling_factor = topkOptions.routed_scaling_factor;
    stats->norm_topk_prob = topkOptions.norm_topk_prob;
    stats->n_group = topkOptions.n_group;
    stats->topk_group = topkOptions.topk_group;
    uint64_t inputBytes = (uint64_t)info.hidden_dim * sizeof(float);
    uint64_t logitsBytes = (uint64_t)info.num_experts * sizeof(float);
    if (inputBytes > NSUIntegerMax || logitsBytes > NSUIntegerMax || info.size > NSUIntegerMax) {
        fprintf(stderr, "ERROR: router probe buffer exceeds NSUIntegerMax\n");
        return 0;
    }
    NSData *inputData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
    if (!inputData) {
        fprintf(stderr, "ERROR: failed to read router --input-f32 %s\n", [inputPath UTF8String]);
        return 0;
    }
    if ((uint64_t)[inputData length] != inputBytes) {
        fprintf(stderr,
                "ERROR: router input bytes %llu do not match expected %llu\n",
                (unsigned long long)[inputData length],
                (unsigned long long)inputBytes);
        return 0;
    }
    uint64_t routerAllocBytes = round_up_u64(info.size, 2 * 1024 * 1024);
    void *routerPtr = NULL;
    if (posix_memalign(&routerPtr, 2 * 1024 * 1024, (size_t)routerAllocBytes) != 0 ||
        !routerPtr) {
        fprintf(stderr, "ERROR: failed to allocate aligned router buffer\n");
        return 0;
    }
    int closeFd = 0;
    int fd = open_resident_read_fd(residentBinPath, &closeFd);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open resident file for router %s: %s\n",
                [residentBinPath UTF8String],
                strerror(errno));
        free(routerPtr);
        return 0;
    }
    float *selectionBias = NULL;
    if (biasInfo.present) {
        if (biasInfo.dim != info.num_experts || biasInfo.size != logitsBytes) {
            fprintf(stderr, "ERROR: router correction-bias dims do not match router experts\n");
            close_resident_read_fd(fd, closeFd);
            free(routerPtr);
            return 0;
        }
        selectionBias = (float *)malloc((size_t)biasInfo.size);
        if (!selectionBias) {
            fprintf(stderr, "ERROR: failed to allocate router correction-bias copy\n");
            close_resident_read_fd(fd, closeFd);
            free(routerPtr);
            return 0;
        }
    }
    double readStarted = now_seconds();
    if (!pread_exact_or_report(fd, routerPtr, info.size, info.offset, [residentBinPath UTF8String])) {
        close_resident_read_fd(fd, closeFd);
        free(selectionBias);
        free(routerPtr);
        return 0;
    }
    if (selectionBias &&
        !pread_exact_or_report(fd,
                               selectionBias,
                               biasInfo.size,
                               biasInfo.offset,
                               [residentBinPath UTF8String])) {
        close_resident_read_fd(fd, closeFd);
        free(selectionBias);
        free(routerPtr);
        return 0;
    }
    close_resident_read_fd(fd, closeFd);
    stats->router_read_seconds = now_seconds() - readStarted;
    stats->router_bytes_read = info.size;
    stats->correction_bias_bytes_read = selectionBias ? biasInfo.size : 0;
    stats->used_correction_bias = selectionBias ? 1 : 0;
    uint32_t scoreMode = 0;
    int scoreModeOk = router_score_mode_code(topkOptions.score, &scoreMode);
    int useGpuTopK =
        allowGpuTopK &&
        outputRouterJson == nil &&
        scoreModeOk &&
        topK <= 64 &&
        info.num_experts <= 256 &&
        topkOptions.n_group == 1 &&
        topkOptions.topk_group == 1;
    id<MTLBuffer> router = [device newBufferWithBytesNoCopy:routerPtr
                                                     length:(NSUInteger)routerAllocBytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:^(void *pointer, NSUInteger length) {
                                                    (void)length;
                                                    free(pointer);
                                                }];
    if (!router) {
        free(selectionBias);
        free(routerPtr);
        fprintf(stderr, "ERROR: failed to wrap router buffer\n");
        return 0;
    }
    id<MTLLibrary> library = make_glm_moe_library(device);
    id<MTLComputePipelineState> pipe =
        library ? make_glm_pipeline(device, library, @"glm_router_bf16") : nil;
    id<MTLComputePipelineState> topkPipe =
        (library && useGpuTopK) ? make_glm_pipeline(device, library, @"glm_router_topk_256") : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> input = [device newBufferWithBytes:[inputData bytes]
                                              length:(NSUInteger)inputBytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> logits = [device newBufferWithLength:(NSUInteger)logitsBytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> bias = selectionBias
        ? [device newBufferWithBytes:selectionBias
                              length:(NSUInteger)biasInfo.size
                             options:MTLResourceStorageModeShared]
        : nil;
    id<MTLBuffer> selectedIds = useGpuTopK
        ? [device newBufferWithLength:64 * sizeof(uint32_t)
                              options:MTLResourceStorageModeShared]
        : nil;
    id<MTLBuffer> selectedWeights = useGpuTopK
        ? [device newBufferWithLength:64 * sizeof(float)
                              options:MTLResourceStorageModeShared]
        : nil;
    if (!library || !pipe || (useGpuTopK && !topkPipe) || !queue || !input ||
        !logits || (selectionBias && !bias) || (useGpuTopK && (!selectedIds || !selectedWeights))) {
        free(selectionBias);
        fprintf(stderr, "ERROR: failed to allocate router Metal resources\n");
        return 0;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        free(selectionBias);
        fprintf(stderr, "ERROR: failed to create router command buffer\n");
        return 0;
    }
    if (!encode_glm_router_bf16(cmd, pipe, router, input, logits, info)) {
        free(selectionBias);
        return 0;
    }
    if (useGpuTopK &&
        !encode_glm_router_topk_256(cmd,
                                    topkPipe,
                                    logits,
                                    bias,
                                    selectedIds,
                                    selectedWeights,
                                    info.num_experts,
                                    topK,
                                    scoreMode,
                                    selectionBias ? 1 : 0,
                                    topkOptions.norm_topk_prob,
                                    topkOptions.routed_scaling_factor)) {
        free(selectionBias);
        return 0;
    }
    double started = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    stats->kernel_seconds = now_seconds() - started;
    if (cmd.status == MTLCommandBufferStatusError) {
        free(selectionBias);
        fprintf(stderr,
                "ERROR: router command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    if (useGpuTopK) {
        uint32_t *ids = (uint32_t *)[selectedIds contents];
        float *weights = (float *)[selectedWeights contents];
        for (uint32_t i = 0; i < topK; i++) {
            stats->experts[i] = (int)ids[i];
            stats->weights[i] = weights[i];
        }
        stats->gpu_topk = 1;
    } else {
        stats->logits = (float *)malloc((size_t)logitsBytes);
        if (!stats->logits) {
            free(selectionBias);
            fprintf(stderr, "ERROR: failed to allocate router logits copy\n");
            return 0;
        }
        memcpy(stats->logits, [logits contents], (size_t)logitsBytes);
        if (!compute_router_topk(stats->logits,
                                 info.num_experts,
                                 topK,
                                 selectionBias,
                                 topkOptions,
                                 stats->experts,
                                 stats->weights)) {
            free(selectionBias);
            fprintf(stderr, "ERROR: failed to compute router top-k\n");
            return 0;
        }
    }
    free(selectionBias);
    stats->elapsed_seconds = stats->router_read_seconds + stats->kernel_seconds;
    stats->command_buffer_count = 1;
    stats->ok = 1;
    if (outputRouterJson &&
        !write_router_probe_json(outputRouterJson, *stats, info, layerId, inputPath)) {
        return 0;
    }
    return 1;
}

static int run_rmsnorm_router_fused_probe(id<MTLDevice> device,
                                          NSString *residentBinPath,
                                          ResidentVectorInfo normInfo,
                                          RouterWeightInfo routerInfo,
                                          RouterBiasInfo biasInfo,
                                          RouterTopKOptions topkOptions,
                                          int layerId,
                                          NSString *inputPath,
                                          NSData *inputDataOverride,
                                          id<MTLBuffer> inputBufferOverride,
                                          float eps,
                                          uint32_t topK,
                                          NSData **normedDataOut,
                                          id<MTLBuffer> __strong *normedBufferOut,
                                          RmsNormProbeStats *normStats,
                                          RouterProbeStats *routerStats) {
    memset(normStats, 0, sizeof(*normStats));
    memset(routerStats, 0, sizeof(*routerStats));
    if (normedDataOut) {
        *normedDataOut = nil;
    }
    if (normedBufferOut) {
        *normedBufferOut = nil;
    }
    routerStats->top_k = topK;
    routerStats->num_experts = routerInfo.num_experts;
    snprintf(routerStats->router_score, sizeof(routerStats->router_score), "%s",
             topkOptions.score);
    routerStats->routed_scaling_factor = topkOptions.routed_scaling_factor;
    routerStats->norm_topk_prob = topkOptions.norm_topk_prob;
    routerStats->n_group = topkOptions.n_group;
    routerStats->topk_group = topkOptions.topk_group;
    if (normInfo.dim != routerInfo.hidden_dim) {
        fprintf(stderr,
                "ERROR: fused RMSNorm/router dim mismatch norm=%u router=%u\n",
                normInfo.dim,
                routerInfo.hidden_dim);
        return 0;
    }
    uint64_t inputBytes = (uint64_t)normInfo.dim * sizeof(float);
    uint64_t logitsBytes = (uint64_t)routerInfo.num_experts * sizeof(float);
    if (inputBytes > NSUIntegerMax || logitsBytes > NSUIntegerMax ||
        routerInfo.size > NSUIntegerMax) {
        fprintf(stderr, "ERROR: fused RMSNorm/router buffer exceeds NSUIntegerMax\n");
        return 0;
    }
    NSData *inputData = nil;
    if (inputBufferOverride) {
        if ((uint64_t)[inputBufferOverride length] < inputBytes) {
            fprintf(stderr,
                    "ERROR: fused RMSNorm/router input MTLBuffer bytes %llu are smaller than expected %llu\n",
                    (unsigned long long)[inputBufferOverride length],
                    (unsigned long long)inputBytes);
            return 0;
        }
    } else {
        inputData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
        if (!inputData) {
            fprintf(stderr,
                    "ERROR: failed to read fused RMSNorm/router input %s\n",
                    inputPath ? [inputPath UTF8String] : "<memory>");
            return 0;
        }
        if ((uint64_t)[inputData length] != inputBytes) {
            fprintf(stderr,
                    "ERROR: fused RMSNorm/router input bytes %llu do not match expected %llu\n",
                    (unsigned long long)[inputData length],
                    (unsigned long long)inputBytes);
            return 0;
        }
    }
    float *normWeightValues = NULL;
    uint64_t normWeightBytesRead = 0;
    double normWeightReadStarted = now_seconds();
    if (!read_resident_vector_f32(residentBinPath,
                                  normInfo,
                                  &normWeightValues,
                                  &normWeightBytesRead)) {
        return 0;
    }
    normStats->weight_read_seconds = now_seconds() - normWeightReadStarted;
    uint64_t routerAllocBytes = round_up_u64(routerInfo.size, 2 * 1024 * 1024);
    void *routerPtr = NULL;
    if (posix_memalign(&routerPtr, 2 * 1024 * 1024, (size_t)routerAllocBytes) != 0 ||
        !routerPtr) {
        fprintf(stderr, "ERROR: failed to allocate aligned fused router buffer\n");
        free(normWeightValues);
        return 0;
    }
    int closeFd = 0;
    int fd = open_resident_read_fd(residentBinPath, &closeFd);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open resident file for fused router %s: %s\n",
                [residentBinPath UTF8String],
                strerror(errno));
        free(normWeightValues);
        free(routerPtr);
        return 0;
    }
    float *selectionBias = NULL;
    if (biasInfo.present) {
        if (biasInfo.dim != routerInfo.num_experts || biasInfo.size != logitsBytes) {
            fprintf(stderr,
                    "ERROR: fused router correction-bias dims do not match router experts\n");
            close_resident_read_fd(fd, closeFd);
            free(normWeightValues);
            free(routerPtr);
            return 0;
        }
        selectionBias = (float *)malloc((size_t)biasInfo.size);
        if (!selectionBias) {
            fprintf(stderr,
                    "ERROR: failed to allocate fused router correction-bias copy\n");
            close_resident_read_fd(fd, closeFd);
            free(normWeightValues);
            free(routerPtr);
            return 0;
        }
    }
    double readStarted = now_seconds();
    if (!pread_exact_or_report(fd,
                               routerPtr,
                               routerInfo.size,
                               routerInfo.offset,
                               [residentBinPath UTF8String])) {
        close_resident_read_fd(fd, closeFd);
        free(selectionBias);
        free(normWeightValues);
        free(routerPtr);
        return 0;
    }
    if (selectionBias &&
        !pread_exact_or_report(fd,
                               selectionBias,
                               biasInfo.size,
                               biasInfo.offset,
                               [residentBinPath UTF8String])) {
        close_resident_read_fd(fd, closeFd);
        free(selectionBias);
        free(normWeightValues);
        free(routerPtr);
        return 0;
    }
    close_resident_read_fd(fd, closeFd);
    routerStats->router_read_seconds = now_seconds() - readStarted;
    routerStats->router_bytes_read = routerInfo.size;
    routerStats->correction_bias_bytes_read = selectionBias ? biasInfo.size : 0;
    routerStats->used_correction_bias = selectionBias ? 1 : 0;

    uint32_t scoreMode = 0;
    int scoreModeOk = router_score_mode_code(topkOptions.score, &scoreMode);
    int useGpuTopK =
        scoreModeOk &&
        topK <= 64 &&
        routerInfo.num_experts <= 256 &&
        topkOptions.n_group == 1 &&
        topkOptions.topk_group == 1;

    id<MTLLibrary> library = make_glm_moe_library(device);
    id<MTLComputePipelineState> rmsPipe =
        library ? make_glm_pipeline(device, library, @"glm_rmsnorm_f32") : nil;
    id<MTLComputePipelineState> routerPipe =
        library ? make_glm_pipeline(device, library, @"glm_router_bf16") : nil;
    id<MTLComputePipelineState> topkPipe =
        (library && useGpuTopK)
            ? make_glm_pipeline(device, library, @"glm_router_topk_256")
            : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> input = inputBufferOverride ?: [device newBufferWithBytes:[inputData bytes]
                                                                      length:(NSUInteger)inputBytes
                                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> normWeight = [device newBufferWithBytes:normWeightValues
                                                    length:(NSUInteger)inputBytes
                                                   options:MTLResourceStorageModeShared];
    free(normWeightValues);
    id<MTLBuffer> normed = [device newBufferWithLength:(NSUInteger)inputBytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> router = [device newBufferWithBytesNoCopy:routerPtr
                                                     length:(NSUInteger)routerAllocBytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:^(void *pointer, NSUInteger length) {
                                                    (void)length;
                                                    free(pointer);
                                                }];
    if (router) {
        routerPtr = NULL;
    }
    id<MTLBuffer> logits = [device newBufferWithLength:(NSUInteger)logitsBytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> bias = selectionBias
        ? [device newBufferWithBytes:selectionBias
                              length:(NSUInteger)biasInfo.size
                             options:MTLResourceStorageModeShared]
        : nil;
    id<MTLBuffer> selectedIds = useGpuTopK
        ? [device newBufferWithLength:64 * sizeof(uint32_t)
                              options:MTLResourceStorageModeShared]
        : nil;
    id<MTLBuffer> selectedWeights = useGpuTopK
        ? [device newBufferWithLength:64 * sizeof(float)
                              options:MTLResourceStorageModeShared]
        : nil;
    if (!library || !rmsPipe || !routerPipe || (useGpuTopK && !topkPipe) ||
        !queue || !input || !normWeight || !normed || !router || !logits ||
        (selectionBias && !bias) ||
        (useGpuTopK && (!selectedIds || !selectedWeights))) {
        free(selectionBias);
        free(routerPtr);
        fprintf(stderr, "ERROR: failed to allocate fused RMSNorm/router resources\n");
        return 0;
    }
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        free(selectionBias);
        fprintf(stderr, "ERROR: failed to create fused RMSNorm/router command buffer\n");
        return 0;
    }
    if (!encode_glm_rmsnorm(cmd,
                            rmsPipe,
                            input,
                            normWeight,
                            normed,
                            normInfo.dim,
                            eps) ||
        !encode_glm_router_bf16(cmd,
                                routerPipe,
                                router,
                                normed,
                                logits,
                                routerInfo)) {
        free(selectionBias);
        return 0;
    }
    if (useGpuTopK &&
        !encode_glm_router_topk_256(cmd,
                                    topkPipe,
                                    logits,
                                    bias,
                                    selectedIds,
                                    selectedWeights,
                                    routerInfo.num_experts,
                                    topK,
                                    scoreMode,
                                    selectionBias ? 1 : 0,
                                    topkOptions.norm_topk_prob,
                                    topkOptions.routed_scaling_factor)) {
        free(selectionBias);
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    double kernelSeconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        free(selectionBias);
        fprintf(stderr,
                "ERROR: fused RMSNorm/router command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    normStats->weight_bytes_read = normWeightBytesRead;
    normStats->elapsed_seconds = kernelSeconds;
    normStats->output0 = ((float *)[normed contents])[0];
    normStats->command_buffer_count = 1;
    normStats->fused_with_router = 1;
    normStats->input_buffer_direct = inputBufferOverride ? 1 : 0;
    normStats->ok = 1;
    if (normedDataOut) {
        *normedDataOut = [NSData dataWithBytes:[normed contents]
                                        length:(NSUInteger)inputBytes];
    }
    if (normedBufferOut) {
        *normedBufferOut = normed;
    }

    routerStats->kernel_seconds = 0.0;
    routerStats->command_buffer_count = 0;
    routerStats->fused_with_rmsnorm = 1;
    if (useGpuTopK) {
        uint32_t *ids = (uint32_t *)[selectedIds contents];
        float *weights = (float *)[selectedWeights contents];
        for (uint32_t i = 0; i < topK; i++) {
            routerStats->experts[i] = (int)ids[i];
            routerStats->weights[i] = weights[i];
        }
        routerStats->gpu_topk = 1;
    } else {
        routerStats->logits = (float *)malloc((size_t)logitsBytes);
        if (!routerStats->logits) {
            free(selectionBias);
            fprintf(stderr, "ERROR: failed to allocate fused router logits copy\n");
            return 0;
        }
        memcpy(routerStats->logits, [logits contents], (size_t)logitsBytes);
        if (!compute_router_topk(routerStats->logits,
                                 routerInfo.num_experts,
                                 topK,
                                 selectionBias,
                                 topkOptions,
                                 routerStats->experts,
                                 routerStats->weights)) {
            free(selectionBias);
            fprintf(stderr, "ERROR: failed to compute fused router top-k\n");
            return 0;
        }
    }
    free(selectionBias);
    routerStats->elapsed_seconds = routerStats->router_read_seconds;
    routerStats->ok = 1;
    (void)layerId;
    return 1;
}

static int __attribute__((unused)) run_attention_output_rmsnorm_router_fused_probe(
    id<MTLDevice> device,
    NSString *residentBinPath,
    ResidentMxfp4MatrixInfo attnInfo,
    ResidentVectorInfo normInfo,
    RouterWeightInfo routerInfo,
    RouterBiasInfo biasInfo,
    RouterTopKOptions topkOptions,
    int layerId,
    NSString *inputPath,
    NSData *inputDataOverride,
    NSString *residualPath,
    NSData *residualDataOverride,
    NSString *outputPath,
    float eps,
    uint32_t topK,
    NSData **attnOutputDataOut,
    id<MTLBuffer> __strong *attnOutputBufferOut,
    NSData **normedDataOut,
    id<MTLBuffer> __strong *normedBufferOut,
    AttnOutputProbeStats *attnStats,
    RmsNormProbeStats *normStats,
    RouterProbeStats *routerStats) {
    memset(attnStats, 0, sizeof(*attnStats));
    memset(normStats, 0, sizeof(*normStats));
    memset(routerStats, 0, sizeof(*routerStats));
    if (attnOutputDataOut) {
        *attnOutputDataOut = nil;
    }
    if (attnOutputBufferOut) {
        *attnOutputBufferOut = nil;
    }
    if (normedDataOut) {
        *normedDataOut = nil;
    }
    if (normedBufferOut) {
        *normedBufferOut = nil;
    }
    attnStats->out_dim = attnInfo.out_dim;
    attnStats->in_dim = attnInfo.in_dim;
    attnStats->group_size = attnInfo.group_size;
    attnStats->input_bytes = (uint64_t)attnInfo.in_dim * sizeof(float);
    attnStats->residual_bytes = (uint64_t)attnInfo.out_dim * sizeof(float);
    attnStats->projection_bytes = attnStats->residual_bytes;
    attnStats->output_bytes = attnStats->residual_bytes;
    routerStats->top_k = topK;
    routerStats->num_experts = routerInfo.num_experts;
    snprintf(routerStats->router_score, sizeof(routerStats->router_score), "%s",
             topkOptions.score);
    routerStats->routed_scaling_factor = topkOptions.routed_scaling_factor;
    routerStats->norm_topk_prob = topkOptions.norm_topk_prob;
    routerStats->n_group = topkOptions.n_group;
    routerStats->topk_group = topkOptions.topk_group;

    if (attnInfo.out_dim != normInfo.dim ||
        normInfo.dim != routerInfo.hidden_dim) {
        fprintf(stderr,
                "ERROR: fused attn-output/RMSNorm/router dim mismatch attn_out=%u norm=%u router=%u\n",
                attnInfo.out_dim,
                normInfo.dim,
                routerInfo.hidden_dim);
        return 0;
    }
    uint64_t normBytes = (uint64_t)normInfo.dim * sizeof(float);
    uint64_t logitsBytes = (uint64_t)routerInfo.num_experts * sizeof(float);
    if (attnStats->input_bytes > NSUIntegerMax ||
        attnStats->residual_bytes > NSUIntegerMax ||
        attnStats->output_bytes > NSUIntegerMax ||
        normBytes > NSUIntegerMax ||
        logitsBytes > NSUIntegerMax ||
        attnInfo.total_bytes > NSUIntegerMax ||
        routerInfo.size > NSUIntegerMax) {
        fprintf(stderr, "ERROR: fused attn-output/RMSNorm/router buffers exceed NSUIntegerMax\n");
        return 0;
    }

    NSData *inputData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
    if (!inputData) {
        fprintf(stderr, "ERROR: failed to read fused attention-output input %s\n",
                inputPath ? [inputPath UTF8String] : "<memory>");
        return 0;
    }
    if ((uint64_t)[inputData length] != attnStats->input_bytes) {
        fprintf(stderr,
                "ERROR: fused attention-output input bytes %llu do not match expected %llu\n",
                (unsigned long long)[inputData length],
                (unsigned long long)attnStats->input_bytes);
        return 0;
    }
    NSData *residualData =
        residualDataOverride ?: [NSData dataWithContentsOfFile:residualPath];
    if (!residualData) {
        fprintf(stderr, "ERROR: failed to read fused attention-output residual %s\n",
                residualPath ? [residualPath UTF8String] : "<memory>");
        return 0;
    }
    if ((uint64_t)[residualData length] != attnStats->residual_bytes) {
        fprintf(stderr,
                "ERROR: fused attention-output residual bytes %llu do not match expected %llu\n",
                (unsigned long long)[residualData length],
                (unsigned long long)attnStats->residual_bytes);
        return 0;
    }

    uint64_t matrixAllocBytes = round_up_u64(attnInfo.total_bytes, 2 * 1024 * 1024);
    uint64_t routerAllocBytes = round_up_u64(routerInfo.size, 2 * 1024 * 1024);
    void *matrixPtr = NULL;
    void *routerPtr = NULL;
    float *normWeightValues = NULL;
    float *selectionBias = NULL;
    uint64_t normWeightBytesRead = 0;
    if (posix_memalign(&matrixPtr, 2 * 1024 * 1024, (size_t)matrixAllocBytes) != 0 ||
        !matrixPtr ||
        posix_memalign(&routerPtr, 2 * 1024 * 1024, (size_t)routerAllocBytes) != 0 ||
        !routerPtr) {
        fprintf(stderr, "ERROR: failed to allocate fused attn-output/RMSNorm/router aligned buffers\n");
        free(matrixPtr);
        free(routerPtr);
        return 0;
    }
    double normWeightReadStarted = now_seconds();
    if (!read_resident_vector_f32(residentBinPath,
                                  normInfo,
                                  &normWeightValues,
                                  &normWeightBytesRead)) {
        free(matrixPtr);
        free(routerPtr);
        return 0;
    }
    normStats->weight_read_seconds = now_seconds() - normWeightReadStarted;
    if (biasInfo.present) {
        if (biasInfo.dim != routerInfo.num_experts || biasInfo.size != logitsBytes) {
            fprintf(stderr,
                    "ERROR: fused router correction-bias dims do not match router experts\n");
            free(normWeightValues);
            free(matrixPtr);
            free(routerPtr);
            return 0;
        }
        selectionBias = (float *)malloc((size_t)biasInfo.size);
        if (!selectionBias) {
            fprintf(stderr,
                    "ERROR: failed to allocate fused router correction-bias copy\n");
            free(normWeightValues);
            free(matrixPtr);
            free(routerPtr);
            return 0;
        }
    }

    int closeFd = 0;
    int fd = open_resident_read_fd(residentBinPath, &closeFd);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open resident file for fused attn-output/RMSNorm/router %s: %s\n",
                [residentBinPath UTF8String],
                strerror(errno));
        free(selectionBias);
        free(normWeightValues);
        free(matrixPtr);
        free(routerPtr);
        return 0;
    }
    double attnReadStarted = now_seconds();
    int attnReadOk =
        pread_exact_or_report(fd,
                              matrixPtr,
                              attnInfo.weight.size,
                              attnInfo.weight.offset,
                              [residentBinPath UTF8String]) &&
        pread_exact_or_report(fd,
                              (uint8_t *)matrixPtr + attnInfo.weight.size,
                              attnInfo.scales.size,
                              attnInfo.scales.offset,
                              [residentBinPath UTF8String]);
    attnStats->read_seconds = now_seconds() - attnReadStarted;
    if (!attnReadOk) {
        close_resident_read_fd(fd, closeFd);
        free(selectionBias);
        free(normWeightValues);
        free(matrixPtr);
        free(routerPtr);
        return 0;
    }
    attnStats->bytes_read = attnInfo.total_bytes;
    double routerReadStarted = now_seconds();
    int routerReadOk =
        pread_exact_or_report(fd,
                              routerPtr,
                              routerInfo.size,
                              routerInfo.offset,
                              [residentBinPath UTF8String]);
    if (routerReadOk && selectionBias) {
        routerReadOk =
            pread_exact_or_report(fd,
                                  selectionBias,
                                  biasInfo.size,
                                  biasInfo.offset,
                                  [residentBinPath UTF8String]);
    }
    routerStats->router_read_seconds = now_seconds() - routerReadStarted;
    close_resident_read_fd(fd, closeFd);
    if (!routerReadOk) {
        free(selectionBias);
        free(normWeightValues);
        free(matrixPtr);
        free(routerPtr);
        return 0;
    }
    routerStats->router_bytes_read = routerInfo.size;
    routerStats->correction_bias_bytes_read = selectionBias ? biasInfo.size : 0;
    routerStats->used_correction_bias = selectionBias ? 1 : 0;

    uint32_t scoreMode = 0;
    int scoreModeOk = router_score_mode_code(topkOptions.score, &scoreMode);
    int useGpuTopK =
        scoreModeOk &&
        topK <= 64 &&
        routerInfo.num_experts <= 256 &&
        topkOptions.n_group == 1 &&
        topkOptions.topk_group == 1;

    id<MTLLibrary> library = make_glm_moe_library(device);
    id<MTLComputePipelineState> matvecAddPipe =
        library ? make_resident_mxfp4_matvec_add_pipeline(device, library, attnInfo) : nil;
    id<MTLComputePipelineState> rmsPipe =
        library ? make_glm_pipeline(device, library, @"glm_rmsnorm_f32") : nil;
    id<MTLComputePipelineState> routerPipe =
        library ? make_glm_pipeline(device, library, @"glm_router_bf16") : nil;
    id<MTLComputePipelineState> topkPipe =
        (library && useGpuTopK)
            ? make_glm_pipeline(device, library, @"glm_router_topk_256")
            : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> matrix = [device newBufferWithBytesNoCopy:matrixPtr
                                                     length:(NSUInteger)matrixAllocBytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:^(void *pointer, NSUInteger length) {
                                                    (void)length;
                                                    free(pointer);
                                                }];
    if (matrix) {
        matrixPtr = NULL;
    }
    id<MTLBuffer> input = [device newBufferWithBytes:[inputData bytes]
                                              length:(NSUInteger)attnStats->input_bytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> residual = [device newBufferWithBytes:[residualData bytes]
                                                 length:(NSUInteger)attnStats->residual_bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> attnOutput = [device newBufferWithLength:(NSUInteger)attnStats->output_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> normWeight = [device newBufferWithBytes:normWeightValues
                                                   length:(NSUInteger)normBytes
                                                  options:MTLResourceStorageModeShared];
    free(normWeightValues);
    normWeightValues = NULL;
    id<MTLBuffer> normed = [device newBufferWithLength:(NSUInteger)normBytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> router = [device newBufferWithBytesNoCopy:routerPtr
                                                     length:(NSUInteger)routerAllocBytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:^(void *pointer, NSUInteger length) {
                                                    (void)length;
                                                    free(pointer);
                                                }];
    if (router) {
        routerPtr = NULL;
    }
    id<MTLBuffer> logits = [device newBufferWithLength:(NSUInteger)logitsBytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> bias = selectionBias
        ? [device newBufferWithBytes:selectionBias
                              length:(NSUInteger)biasInfo.size
                             options:MTLResourceStorageModeShared]
        : nil;
    id<MTLBuffer> selectedIds = useGpuTopK
        ? [device newBufferWithLength:64 * sizeof(uint32_t)
                              options:MTLResourceStorageModeShared]
        : nil;
    id<MTLBuffer> selectedWeights = useGpuTopK
        ? [device newBufferWithLength:64 * sizeof(float)
                              options:MTLResourceStorageModeShared]
        : nil;
    if (!library || !matvecAddPipe || !rmsPipe || !routerPipe ||
        (useGpuTopK && !topkPipe) || !queue || !matrix || !input ||
        !residual || !attnOutput || !normWeight || !normed || !router ||
        !logits || (selectionBias && !bias) ||
        (useGpuTopK && (!selectedIds || !selectedWeights))) {
        free(selectionBias);
        free(normWeightValues);
        free(matrixPtr);
        free(routerPtr);
        fprintf(stderr, "ERROR: failed to allocate fused attn-output/RMSNorm/router Metal resources\n");
        return 0;
    }
    [matrix didModifyRange:NSMakeRange(0, (NSUInteger)attnInfo.total_bytes)];

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        free(selectionBias);
        fprintf(stderr, "ERROR: failed to create fused attn-output/RMSNorm/router command buffer\n");
        return 0;
    }
    if (!encode_glm_mxfp4_matvec_add(cmd,
                                     matvecAddPipe,
                                     matrix,
                                     attnInfo,
                                     input,
                                     residual,
                                     attnOutput) ||
        !encode_glm_rmsnorm(cmd, rmsPipe, attnOutput, normWeight, normed, normInfo.dim, eps) ||
        !encode_glm_router_bf16(cmd, routerPipe, router, normed, logits, routerInfo)) {
        free(selectionBias);
        return 0;
    }
    if (useGpuTopK &&
        !encode_glm_router_topk_256(cmd,
                                    topkPipe,
                                    logits,
                                    bias,
                                    selectedIds,
                                    selectedWeights,
                                    routerInfo.num_experts,
                                    topK,
                                    scoreMode,
                                    selectionBias ? 1 : 0,
                                    topkOptions.norm_topk_prob,
                                    topkOptions.routed_scaling_factor)) {
        free(selectionBias);
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    [cmd waitUntilCompleted];
    double kernelSeconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        free(selectionBias);
        fprintf(stderr,
                "ERROR: fused attn-output/RMSNorm/router command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }

    attnStats->projection_kernel_seconds = kernelSeconds;
    attnStats->residual_add_seconds = 0.0;
    attnStats->command_buffer_count = 1;
    attnStats->fused_matvec_add = 1;
    attnStats->fused_with_post_attn_norm_router = 1;
    double writeStarted = now_seconds();
    NSData *attnOutputData = nil;
    if (attnOutputDataOut || outputPath) {
        attnOutputData = [NSData dataWithBytes:[attnOutput contents]
                                        length:(NSUInteger)attnStats->output_bytes];
    }
    if (outputPath && ![attnOutputData writeToFile:outputPath atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write fused attention-output output %s\n",
                [outputPath UTF8String]);
        free(selectionBias);
        return 0;
    }
    if (attnOutputDataOut) {
        *attnOutputDataOut = attnOutputData;
    }
    if (attnOutputBufferOut) {
        *attnOutputBufferOut = attnOutput;
    }
    attnStats->output_write_seconds = now_seconds() - writeStarted;
    attnStats->elapsed_seconds = attnStats->read_seconds +
                                 attnStats->projection_kernel_seconds +
                                 attnStats->output_write_seconds;
    attnStats->output0 = ((float *)[attnOutput contents])[0];
    attnStats->ok = 1;

    normStats->weight_bytes_read = normWeightBytesRead;
    normStats->elapsed_seconds = 0.0;
    normStats->output0 = ((float *)[normed contents])[0];
    normStats->command_buffer_count = 0;
    normStats->fused_with_router = 1;
    normStats->input_buffer_direct = 1;
    normStats->ok = 1;
    if (normedDataOut) {
        *normedDataOut = [NSData dataWithBytes:[normed contents]
                                        length:(NSUInteger)normBytes];
    }
    if (normedBufferOut) {
        *normedBufferOut = normed;
    }

    routerStats->kernel_seconds = 0.0;
    routerStats->command_buffer_count = 0;
    routerStats->fused_with_rmsnorm = 1;
    if (useGpuTopK) {
        uint32_t *ids = (uint32_t *)[selectedIds contents];
        float *weights = (float *)[selectedWeights contents];
        for (uint32_t i = 0; i < topK; i++) {
            routerStats->experts[i] = (int)ids[i];
            routerStats->weights[i] = weights[i];
        }
        routerStats->gpu_topk = 1;
    } else {
        routerStats->logits = (float *)malloc((size_t)logitsBytes);
        if (!routerStats->logits) {
            free(selectionBias);
            fprintf(stderr, "ERROR: failed to allocate fused router logits copy\n");
            return 0;
        }
        memcpy(routerStats->logits, [logits contents], (size_t)logitsBytes);
        if (!compute_router_topk(routerStats->logits,
                                 routerInfo.num_experts,
                                 topK,
                                 selectionBias,
                                 topkOptions,
                                 routerStats->experts,
                                 routerStats->weights)) {
            free(selectionBias);
            fprintf(stderr, "ERROR: failed to compute fused router top-k\n");
            return 0;
        }
    }
    free(selectionBias);
    routerStats->elapsed_seconds = routerStats->router_read_seconds;
    routerStats->ok = 1;
    (void)layerId;
    return 1;
}

typedef int (^GlmMoePreWaitWorkBlock)(void);

static int run_rope_mla_attention_output_rmsnorm_router_fused_probe(
    id<MTLDevice> device,
    NSString *residentBinPath,
    id<MTLBuffer> residentMetalBuffer,
    MlaAttentionValueSourceInfo valueSource,
    NSString *cacheLayoutPath,
    NSString *cacheFilePath,
    int layerId,
    NSData *qBData,
    NSData *kData,
    id<MTLBuffer> qBBufferOverride,
    id<MTLBuffer> kBufferOverride,
    NSUInteger kBufferOffset,
    id<MTLBuffer> cacheBufferOverride,
    NSUInteger cacheBufferOffset,
    ResidentMxfp4MatrixInfo attnInfo,
    ResidentVectorInfo normInfo,
    RouterWeightInfo routerInfo,
    RouterBiasInfo biasInfo,
    RouterTopKOptions topkOptions,
    NSString *residualPath,
    NSData *residualDataOverride,
    id<MTLBuffer> residualBufferOverride,
    NSString *outputPath,
    uint32_t contextLength,
    uint32_t numHeads,
    uint32_t qkNopeDim,
    uint32_t ropeDim,
    uint32_t vHeadDim,
    uint32_t startPosition,
    uint32_t cachePositionOffset,
    float attentionScale,
    float ropeTheta,
    int ropeInterleave,
    uint64_t maxCacheFileBytes,
    uint64_t maxCacheReadBytes,
    float eps,
    uint32_t topK,
    NSData **attnOutputDataOut,
    id<MTLBuffer> __strong *attnOutputBufferOut,
    NSData **normedDataOut,
    id<MTLBuffer> __strong *normedBufferOut,
    RopeSplitProbeStats *ropeStats,
    MlaAttentionProbeStats *mlaStats,
    AttnOutputProbeStats *attnStats,
    RmsNormProbeStats *normStats,
    RouterProbeStats *routerStats,
    GlmMoePreWaitWorkBlock preWaitWork) {
    memset(ropeStats, 0, sizeof(*ropeStats));
    memset(mlaStats, 0, sizeof(*mlaStats));
    memset(attnStats, 0, sizeof(*attnStats));
    memset(normStats, 0, sizeof(*normStats));
    memset(routerStats, 0, sizeof(*routerStats));
    if (attnOutputDataOut) *attnOutputDataOut = nil;
    if (attnOutputBufferOut) *attnOutputBufferOut = nil;
    if (normedDataOut) *normedDataOut = nil;
    if (normedBufferOut) *normedBufferOut = nil;

    if (qkNopeDim > UINT32_MAX - ropeDim) {
        fprintf(stderr, "ERROR: fused RoPE/MLA/attn-output q head dim overflows uint32\n");
        return 0;
    }
    uint32_t batchTokens = 1;
    uint64_t qBBytes = 0;
    uint64_t kBytes = 0;
    uint64_t qNopeBytes = 0;
    uint64_t qRopeBytes = 0;
    uint64_t ropeScratchBytes = rope_split_scratch_bytes(
        numHeads,
        qkNopeDim,
        ropeDim,
        batchTokens,
        &qBBytes,
        &kBytes,
        &qNopeBytes,
        &qRopeBytes
    );
    uint32_t kvLoraDim = valueSource.kv_lora_dim;
    uint32_t cacheWidth = kvLoraDim + ropeDim;
    uint64_t mlaOutputBytes = (uint64_t)numHeads * vHeadDim * sizeof(float);
    uint64_t attnInputBytes = (uint64_t)attnInfo.in_dim * sizeof(float);
    uint64_t attnOutputBytes = (uint64_t)attnInfo.out_dim * sizeof(float);
    uint64_t normBytes = (uint64_t)normInfo.dim * sizeof(float);
    uint64_t logitsBytes = (uint64_t)routerInfo.num_experts * sizeof(float);
    if (qBBytes > NSUIntegerMax ||
        kBytes > NSUIntegerMax ||
        qNopeBytes > NSUIntegerMax ||
        qRopeBytes > NSUIntegerMax ||
        mlaOutputBytes > NSUIntegerMax ||
        attnInputBytes > NSUIntegerMax ||
        attnOutputBytes > NSUIntegerMax ||
        normBytes > NSUIntegerMax ||
        logitsBytes > NSUIntegerMax ||
        attnInfo.total_bytes > NSUIntegerMax ||
        routerInfo.size > NSUIntegerMax) {
        fprintf(stderr, "ERROR: fused RoPE/MLA/attn-output buffers exceed NSUIntegerMax\n");
        return 0;
    }
    int directInputBuffers = qBBufferOverride && kBufferOverride;
    if (directInputBuffers) {
        if ((uint64_t)[qBBufferOverride length] < qBBytes ||
            (uint64_t)[kBufferOverride length] < (uint64_t)kBufferOffset + kBytes) {
            fprintf(stderr,
                    "ERROR: fused RoPE/MLA/attn-output direct input buffers are too small\n");
            return 0;
        }
    } else {
        if (!qBData || !kData) {
            fprintf(stderr, "ERROR: fused RoPE/MLA/attn-output requires q_b and k_rope inputs\n");
            return 0;
        }
        if ((uint64_t)[qBData length] != qBBytes || (uint64_t)[kData length] != kBytes) {
            fprintf(stderr,
                    "ERROR: fused RoPE/MLA/attn-output input bytes mismatch: q_b %llu/%llu k %llu/%llu\n",
                    (unsigned long long)[qBData length],
                    (unsigned long long)qBBytes,
                    (unsigned long long)[kData length],
                    (unsigned long long)kBytes);
            return 0;
        }
    }
    if (mlaOutputBytes != attnInputBytes ||
        attnInfo.out_dim != normInfo.dim ||
        normInfo.dim != routerInfo.hidden_dim) {
        fprintf(stderr,
                "ERROR: fused RoPE/MLA/attn-output dims mismatch mla_out=%llu attn_in=%u attn_out=%u norm=%u router=%u\n",
                (unsigned long long)(mlaOutputBytes / sizeof(float)),
                attnInfo.in_dim,
                attnInfo.out_dim,
                normInfo.dim,
                routerInfo.hidden_dim);
        return 0;
    }

    NSData *residualData = nil;
    if (residualBufferOverride) {
        if ((uint64_t)[residualBufferOverride length] < attnOutputBytes) {
            fprintf(stderr,
                    "ERROR: fused RoPE/MLA/attn-output residual buffer bytes %llu are smaller than expected %llu\n",
                    (unsigned long long)[residualBufferOverride length],
                    (unsigned long long)attnOutputBytes);
            return 0;
        }
    } else {
        residualData =
            residualDataOverride ?: [NSData dataWithContentsOfFile:residualPath];
        if (!residualData) {
            fprintf(stderr, "ERROR: failed to read fused RoPE/MLA/attn-output residual %s\n",
                    residualPath ? [residualPath UTF8String] : "<memory>");
            return 0;
        }
        if ((uint64_t)[residualData length] != attnOutputBytes) {
            fprintf(stderr,
                    "ERROR: fused RoPE/MLA/attn-output residual bytes %llu do not match expected %llu\n",
                    (unsigned long long)[residualData length],
                    (unsigned long long)attnOutputBytes);
            return 0;
        }
    }

    float *cacheF32 = NULL;
    uint64_t cacheF32Bytes = 0;
    uint64_t rawCacheBytes = 0;
    uint64_t currentCacheRowOffsetBytes = 0;
    uint64_t cacheRowBytes = (uint64_t)cacheWidth * sizeof(float);
    double cacheStarted = now_seconds();
    int directCacheBuffer = cacheBufferOverride != nil;
    if (directCacheBuffer) {
        cacheF32Bytes = (uint64_t)cacheWidth * sizeof(float);
        if ((uint64_t)[cacheBufferOverride length] <
            (uint64_t)cacheBufferOffset + cacheRowBytes) {
            fprintf(stderr,
                    "ERROR: fused RoPE/MLA/attn-output direct cache buffer is too small\n");
            return 0;
        }
        if (contextLength == 1) {
            cacheF32Bytes = cacheRowBytes;
        } else if (!read_decode_cache_prefix_with_current_slot_f32(
                       cacheLayoutPath,
                       cacheFilePath,
                       (uint64_t)layerId,
                       contextLength,
                       cacheWidth,
                       maxCacheFileBytes,
                       maxCacheReadBytes,
                       &cacheF32,
                       &cacheF32Bytes,
                       &rawCacheBytes,
                       &currentCacheRowOffsetBytes)) {
            return 0;
        }
    } else {
        if (!read_decode_cache_segment_f32(cacheLayoutPath,
                                           cacheFilePath,
                                           (uint64_t)layerId,
                                           contextLength,
                                           cacheWidth,
                                           maxCacheFileBytes,
                                           maxCacheReadBytes,
                                           &cacheF32,
                                           &cacheF32Bytes,
                                           &rawCacheBytes)) {
            return 0;
        }
    }
    mlaStats->cache_read_seconds = now_seconds() - cacheStarted;

    float *kvBF32 = NULL;
    uint64_t kvBF32Bytes = 0;
    int kvBF32Owned = 1;
    int valueCacheHit = 0;
    int valueCacheStored = 0;
    uint64_t valueCacheBytes = 0;
    uint64_t valueCacheTotalBytes = 0;
    double valueStarted = now_seconds();
    if (!read_absorbed_mla_kv_b_f32_cached(residentBinPath,
                                           valueSource,
                                           layerId,
                                           numHeads,
                                           qkNopeDim,
                                           vHeadDim,
                                           &kvBF32,
                                           &kvBF32Bytes,
                                           &kvBF32Owned,
                                           &valueCacheHit,
                                           &valueCacheStored,
                                           &valueCacheBytes,
                                           &valueCacheTotalBytes)) {
        free(cacheF32);
        return 0;
    }
    mlaStats->value_read_seconds = now_seconds() - valueStarted;
    if (cacheF32Bytes != (uint64_t)contextLength * cacheWidth * sizeof(float) ||
        kvBF32Bytes != valueSource.kv_b_f32_bytes ||
        cacheF32Bytes > NSUIntegerMax ||
        kvBF32Bytes > NSUIntegerMax) {
        fprintf(stderr, "ERROR: fused RoPE/MLA/attn-output internal byte estimate mismatch\n");
        free(cacheF32);
        if (kvBF32Owned) free(kvBF32);
        return 0;
    }
    if (attentionScale == 0.0f) {
        attentionScale = 1.0f / sqrtf((float)(qkNopeDim + ropeDim));
    }

    uint32_t scoreMode = 0;
    int scoreModeOk = router_score_mode_code(topkOptions.score, &scoreMode);
    int useGpuTopK =
        scoreModeOk &&
        topK <= 64 &&
        routerInfo.num_experts <= 256 &&
        topkOptions.n_group == 1 &&
        topkOptions.topk_group == 1;
    if (biasInfo.present &&
        (biasInfo.dim != routerInfo.num_experts || biasInfo.size != logitsBytes)) {
        fprintf(stderr,
                "ERROR: fused router correction-bias dims do not match router experts\n");
        free(cacheF32);
        if (kvBF32Owned) free(kvBF32);
        return 0;
    }
    int useResidentMetalWeights = residentMetalBuffer != nil && useGpuTopK;
    if (useResidentMetalWeights) {
        uint64_t residentLength = (uint64_t)[residentMetalBuffer length];
        uint64_t attnWeightEnd = 0;
        uint64_t attnScalesEnd = 0;
        uint64_t routerEnd = 0;
        uint64_t biasEnd = 0;
        if (!checked_add_u64(attnInfo.weight.offset,
                             attnInfo.weight.size,
                             &attnWeightEnd) ||
            !checked_add_u64(attnInfo.scales.offset,
                             attnInfo.scales.size,
                             &attnScalesEnd) ||
            !checked_add_u64(routerInfo.offset,
                             routerInfo.size,
                             &routerEnd) ||
            (biasInfo.present &&
             !checked_add_u64(biasInfo.offset, biasInfo.size, &biasEnd)) ||
            attnInfo.weight.offset > NSUIntegerMax ||
            attnInfo.scales.offset > NSUIntegerMax ||
            routerInfo.offset > NSUIntegerMax ||
            (biasInfo.present && biasInfo.offset > NSUIntegerMax) ||
            attnWeightEnd > residentLength ||
            attnScalesEnd > residentLength ||
            routerEnd > residentLength ||
            (biasInfo.present && biasEnd > residentLength)) {
            fprintf(stderr,
                    "ERROR: resident Metal buffer does not cover fused RoPE/MLA/attn-output weights\n");
            free(cacheF32);
            if (kvBF32Owned) free(kvBF32);
            return 0;
        }
    }

    uint64_t matrixAllocBytes = round_up_u64(attnInfo.total_bytes, 2 * 1024 * 1024);
    uint64_t routerAllocBytes = round_up_u64(routerInfo.size, 2 * 1024 * 1024);
    void *matrixPtr = NULL;
    void *routerPtr = NULL;
    float *normWeightValues = NULL;
    float *selectionBias = NULL;
    uint64_t normWeightBytesRead = 0;
    double normWeightReadStarted = now_seconds();
    if (!read_resident_vector_f32(residentBinPath,
                                  normInfo,
                                  &normWeightValues,
                                  &normWeightBytesRead)) {
        free(cacheF32);
        if (kvBF32Owned) free(kvBF32);
        return 0;
    }
    normStats->weight_read_seconds = now_seconds() - normWeightReadStarted;
    if (!useResidentMetalWeights) {
        if (posix_memalign(&matrixPtr, 2 * 1024 * 1024, (size_t)matrixAllocBytes) != 0 ||
            !matrixPtr ||
            posix_memalign(&routerPtr, 2 * 1024 * 1024, (size_t)routerAllocBytes) != 0 ||
            !routerPtr) {
            fprintf(stderr, "ERROR: failed to allocate fused RoPE/MLA/attn-output aligned buffers\n");
            free(cacheF32);
            if (kvBF32Owned) free(kvBF32);
            free(normWeightValues);
            free(matrixPtr);
            free(routerPtr);
            return 0;
        }
    }
    if (biasInfo.present && !useResidentMetalWeights) {
        if (biasInfo.dim != routerInfo.num_experts || biasInfo.size != logitsBytes) {
            fprintf(stderr,
                    "ERROR: fused router correction-bias dims do not match router experts\n");
            free(cacheF32);
            if (kvBF32Owned) free(kvBF32);
            free(normWeightValues);
            free(matrixPtr);
            free(routerPtr);
            return 0;
        }
        selectionBias = (float *)malloc((size_t)biasInfo.size);
        if (!selectionBias) {
            fprintf(stderr, "ERROR: failed to allocate fused router correction-bias copy\n");
            free(cacheF32);
            if (kvBF32Owned) free(kvBF32);
            free(normWeightValues);
            free(matrixPtr);
            free(routerPtr);
            return 0;
        }
    }

    if (!useResidentMetalWeights) {
        int closeFd = 0;
        int fd = open_resident_read_fd(residentBinPath, &closeFd);
        if (fd < 0) {
            fprintf(stderr,
                    "ERROR: failed to open resident file for fused RoPE/MLA/attn-output %s: %s\n",
                    [residentBinPath UTF8String],
                    strerror(errno));
            free(cacheF32);
            if (kvBF32Owned) free(kvBF32);
            free(selectionBias);
            free(normWeightValues);
            free(matrixPtr);
            free(routerPtr);
            return 0;
        }
        ParallelPreadTask readTasks[4];
        int readTaskCount = 0;
        readTasks[readTaskCount++] = (ParallelPreadTask){
            .fd = fd,
            .dst = matrixPtr,
            .bytes = attnInfo.weight.size,
            .offset = attnInfo.weight.offset,
            .path = [residentBinPath UTF8String],
            .ok = 0,
        };
        readTasks[readTaskCount++] = (ParallelPreadTask){
            .fd = fd,
            .dst = (uint8_t *)matrixPtr + attnInfo.weight.size,
            .bytes = attnInfo.scales.size,
            .offset = attnInfo.scales.offset,
            .path = [residentBinPath UTF8String],
            .ok = 0,
        };
        readTasks[readTaskCount++] = (ParallelPreadTask){
            .fd = fd,
            .dst = routerPtr,
            .bytes = routerInfo.size,
            .offset = routerInfo.offset,
            .path = [residentBinPath UTF8String],
            .ok = 0,
        };
        if (selectionBias) {
            readTasks[readTaskCount++] = (ParallelPreadTask){
                .fd = fd,
                .dst = selectionBias,
                .bytes = biasInfo.size,
                .offset = biasInfo.offset,
                .path = [residentBinPath UTF8String],
                .ok = 0,
            };
        }
        double fusedReadStarted = now_seconds();
        ParallelPreadDispatchStats fusedReadDispatchStats = {0};
        int readOk = parallel_pread_exact_or_report(readTasks,
                                                    readTaskCount,
                                                    &fusedReadDispatchStats);
        double fusedReadSeconds = now_seconds() - fusedReadStarted;
        close_resident_read_fd(fd, closeFd);
        if (!readOk) {
            free(cacheF32);
            if (kvBF32Owned) free(kvBF32);
            free(selectionBias);
            free(normWeightValues);
            free(matrixPtr);
            free(routerPtr);
            return 0;
        }
        attnStats->bytes_read = attnInfo.total_bytes;
        routerStats->router_bytes_read = routerInfo.size;
        routerStats->correction_bias_bytes_read = selectionBias ? biasInfo.size : 0;
        uint64_t routerReadBytes =
            routerStats->router_bytes_read + routerStats->correction_bias_bytes_read;
        uint64_t fusedReadBytes = attnStats->bytes_read + routerReadBytes;
        if (fusedReadBytes > 0) {
            attnStats->read_seconds =
                fusedReadSeconds * ((double)attnStats->bytes_read / (double)fusedReadBytes);
            routerStats->router_read_seconds =
                fusedReadSeconds * ((double)routerReadBytes / (double)fusedReadBytes);
        }
    }
    routerStats->used_correction_bias = biasInfo.present ? 1 : 0;
    routerStats->top_k = topK;
    routerStats->num_experts = routerInfo.num_experts;
    snprintf(routerStats->router_score, sizeof(routerStats->router_score), "%s",
             topkOptions.score);
    routerStats->routed_scaling_factor = topkOptions.routed_scaling_factor;
    routerStats->norm_topk_prob = topkOptions.norm_topk_prob;
    routerStats->n_group = topkOptions.n_group;
    routerStats->topk_group = topkOptions.topk_group;

    id<MTLLibrary> library = make_glm_moe_library(device);
    NSString *mlaKernelName = contextLength == 1
        ? @"glm_mla_attention_context1_f32"
        : (contextLength <= 32
            ? @"glm_mla_attention_context_small_f32"
            : @"glm_mla_attention_streaming_f32");
    id<MTLComputePipelineState> ropePipe =
        library ? make_glm_pipeline(device, library, @"glm_rope_split_batch_f32") : nil;
    id<MTLComputePipelineState> mlaPipe =
        library ? make_glm_pipeline(device, library, mlaKernelName) : nil;
    id<MTLComputePipelineState> matvecAddPipe =
        library ? make_resident_mxfp4_matvec_add_pipeline(device, library, attnInfo) : nil;
    id<MTLComputePipelineState> rmsPipe =
        library ? make_glm_pipeline(device, library, @"glm_rmsnorm_f32") : nil;
    id<MTLComputePipelineState> routerPipe =
        library ? make_glm_pipeline(device, library, @"glm_router_bf16") : nil;
    id<MTLComputePipelineState> topkPipe =
        (library && useGpuTopK)
            ? make_glm_pipeline(device, library, @"glm_router_topk_256")
            : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);

    id<MTLBuffer> qB = directInputBuffers
        ? qBBufferOverride
        : [device newBufferWithBytes:[qBData bytes]
                              length:(NSUInteger)qBBytes
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> k = directInputBuffers
        ? kBufferOverride
        : [device newBufferWithBytes:[kData bytes]
                              length:(NSUInteger)kBytes
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQNope = [device newBufferWithLength:(NSUInteger)qNopeBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQRope = [device newBufferWithLength:(NSUInteger)qRopeBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outQ = [device newBufferWithLength:(NSUInteger)qRopeBytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> outK = [device newBufferWithLength:(NSUInteger)kBytes
                                             options:MTLResourceStorageModeShared];
    int useDirectCacheBuffer = directCacheBuffer && contextLength == 1;
    int copyDirectCurrentCacheRow = directCacheBuffer && contextLength > 1;
    id<MTLBuffer> cache = useDirectCacheBuffer
        ? cacheBufferOverride
        : [device newBufferWithBytesNoCopy:cacheF32
                                    length:(NSUInteger)cacheF32Bytes
                                   options:MTLResourceStorageModeShared
                               deallocator:^(void *pointer, NSUInteger length) {
                                   (void)length;
                                   free(pointer);
                               }];
    if (!useDirectCacheBuffer && cache) cacheF32 = NULL;
    void (^kvBDeallocator)(void *, NSUInteger) = nil;
    if (kvBF32Owned) {
        kvBDeallocator = ^(void *pointer, NSUInteger length) {
            (void)length;
            free(pointer);
        };
    }
    id<MTLBuffer> kvB = [device newBufferWithBytesNoCopy:kvBF32
                                                  length:(NSUInteger)kvBF32Bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:kvBDeallocator];
    if (kvB) kvBF32 = NULL;
    id<MTLBuffer> mlaOutput = [device newBufferWithLength:(NSUInteger)mlaOutputBytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> matrix = useResidentMetalWeights
        ? residentMetalBuffer
        : [device newBufferWithBytesNoCopy:matrixPtr
                                    length:(NSUInteger)matrixAllocBytes
                                   options:MTLResourceStorageModeShared
                               deallocator:^(void *pointer, NSUInteger length) {
                                   (void)length;
                                   free(pointer);
                               }];
    if (matrix) matrixPtr = NULL;
    id<MTLBuffer> residual = residualBufferOverride ?:
        [device newBufferWithBytes:[residualData bytes]
                            length:(NSUInteger)attnOutputBytes
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> attnOutput = [device newBufferWithLength:(NSUInteger)attnOutputBytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> normWeight = [device newBufferWithBytes:normWeightValues
                                                   length:(NSUInteger)normBytes
                                                  options:MTLResourceStorageModeShared];
    free(normWeightValues);
    normWeightValues = NULL;
    id<MTLBuffer> normed = [device newBufferWithLength:(NSUInteger)normBytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> router = useResidentMetalWeights
        ? residentMetalBuffer
        : [device newBufferWithBytesNoCopy:routerPtr
                                    length:(NSUInteger)routerAllocBytes
                                   options:MTLResourceStorageModeShared
                               deallocator:^(void *pointer, NSUInteger length) {
                                   (void)length;
                                   free(pointer);
                               }];
    if (router) routerPtr = NULL;
    id<MTLBuffer> logits = [device newBufferWithLength:(NSUInteger)logitsBytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> bias = nil;
    NSUInteger biasOffset = 0;
    if (useResidentMetalWeights && biasInfo.present) {
        bias = residentMetalBuffer;
        biasOffset = (NSUInteger)biasInfo.offset;
    } else if (selectionBias) {
        bias = [device newBufferWithBytes:selectionBias
                                   length:(NSUInteger)biasInfo.size
                                  options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> selectedIds = useGpuTopK
        ? [device newBufferWithLength:64 * sizeof(uint32_t)
                              options:MTLResourceStorageModeShared]
        : nil;
    id<MTLBuffer> selectedWeights = useGpuTopK
        ? [device newBufferWithLength:64 * sizeof(float)
                              options:MTLResourceStorageModeShared]
        : nil;
    if (!library || !ropePipe || !mlaPipe || !matvecAddPipe ||
        !rmsPipe || !routerPipe || (useGpuTopK && !topkPipe) || !queue ||
        !qB || !k || !outQNope || !outQRope || !outQ || !outK || !cache ||
        !kvB || !mlaOutput || !matrix || !residual || !attnOutput ||
        !normWeight || !normed || !router || !logits ||
        ((selectionBias || (useResidentMetalWeights && biasInfo.present)) && !bias) ||
        (useGpuTopK && (!selectedIds || !selectedWeights))) {
        fprintf(stderr, "ERROR: failed to allocate fused RoPE/MLA/attn-output Metal resources\n");
        free(cacheF32);
        if (kvBF32Owned) free(kvBF32);
        free(selectionBias);
        free(normWeightValues);
        free(matrixPtr);
        free(routerPtr);
        return 0;
    }
    if (!useResidentMetalWeights) {
        [matrix didModifyRange:NSMakeRange(0, (NSUInteger)attnInfo.total_bytes)];
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        free(selectionBias);
        fprintf(stderr, "ERROR: failed to create fused RoPE/MLA/attn-output command buffer\n");
        return 0;
    }
    if (copyDirectCurrentCacheRow) {
        if (cacheRowBytes > NSUIntegerMax ||
            currentCacheRowOffsetBytes > NSUIntegerMax ||
            cacheBufferOffset > NSUIntegerMax) {
            free(selectionBias);
            fprintf(stderr, "ERROR: direct current KV-A cache copy offset exceeds NSUIntegerMax\n");
            return 0;
        }
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (!blit) {
            free(selectionBias);
            fprintf(stderr, "ERROR: failed to create direct current KV-A cache blit encoder\n");
            return 0;
        }
        [blit copyFromBuffer:cacheBufferOverride
                sourceOffset:cacheBufferOffset
                    toBuffer:cache
           destinationOffset:(NSUInteger)currentCacheRowOffsetBytes
                        size:(NSUInteger)cacheRowBytes];
        [blit endEncoding];
    }
    if (!encode_glm_rope_split_batch_with_offsets(cmd,
                                                  ropePipe,
                                                  qB,
                                                  0,
                                                  k,
                                                  directInputBuffers ? kBufferOffset : 0,
                                                  outQNope,
                                                  outQRope,
                                                  outQ,
                                                  outK,
                                                  numHeads,
                                                  qkNopeDim,
                                                  ropeDim,
                                                  startPosition,
                                                  ropeTheta,
                                                  ropeInterleave ? 1u : 0u,
                                                  batchTokens) ||
        !encode_glm_mla_attention(cmd,
                                  mlaPipe,
                                  outQNope,
                                  outQ,
                                  cache,
                                  kvB,
                                  mlaOutput,
                                  contextLength,
                                  numHeads,
                                  kvLoraDim,
                                  qkNopeDim,
                                  ropeDim,
                                  vHeadDim,
                                  attentionScale,
                                  ropeTheta,
                                  ropeInterleave ? 1u : 0u,
                                  useDirectCacheBuffer ? 0u : cachePositionOffset) ||
        !encode_glm_mxfp4_matvec_add_with_offsets(
            cmd,
            matvecAddPipe,
            matrix,
            useResidentMetalWeights ? (NSUInteger)attnInfo.weight.offset : 0,
            matrix,
            useResidentMetalWeights
                ? (NSUInteger)attnInfo.scales.offset
                : (NSUInteger)attnInfo.weight.size,
            attnInfo,
            mlaOutput,
            residual,
            attnOutput) ||
        !encode_glm_rmsnorm_with_weight_offset(
            cmd,
            rmsPipe,
            attnOutput,
            normWeight,
            0,
            normed,
            normInfo.dim,
            eps) ||
        !encode_glm_router_bf16_with_offset(
            cmd,
            routerPipe,
            router,
            useResidentMetalWeights ? (NSUInteger)routerInfo.offset : 0,
            normed,
            logits,
            routerInfo)) {
        free(selectionBias);
        return 0;
    }
    if (useGpuTopK &&
        !encode_glm_router_topk_256_with_bias_offset(
            cmd,
            topkPipe,
            logits,
            bias,
            biasOffset,
            selectedIds,
            selectedWeights,
            routerInfo.num_experts,
            topK,
            scoreMode,
            (biasInfo.present &&
             (useResidentMetalWeights || selectionBias != NULL)) ? 1 : 0,
            topkOptions.norm_topk_prob,
            topkOptions.routed_scaling_factor)) {
        free(selectionBias);
        return 0;
    }
    double kernelStarted = now_seconds();
    [cmd commit];
    int preWaitOk = 1;
    if (preWaitWork) {
        preWaitOk = preWaitWork();
    }
    [cmd waitUntilCompleted];
    double kernelSeconds = now_seconds() - kernelStarted;
    if (cmd.status == MTLCommandBufferStatusError) {
        free(selectionBias);
        fprintf(stderr,
                "ERROR: fused RoPE/MLA/attn-output command failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    if (!preWaitOk) {
        free(selectionBias);
        return 0;
    }

    double writeStarted = now_seconds();
    NSData *attnOutputData = nil;
    if (attnOutputDataOut || outputPath) {
        attnOutputData = [NSData dataWithBytes:[attnOutput contents]
                                        length:(NSUInteger)attnOutputBytes];
    }
    if (outputPath && ![attnOutputData writeToFile:outputPath atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write fused RoPE/MLA/attn-output output %s\n",
                [outputPath UTF8String]);
        free(selectionBias);
        return 0;
    }
    if (attnOutputDataOut) *attnOutputDataOut = attnOutputData;
    if (attnOutputBufferOut) *attnOutputBufferOut = attnOutput;
    attnStats->output_write_seconds = now_seconds() - writeStarted;

    ropeStats->ok = 1;
    ropeStats->scratch_bytes = ropeScratchBytes;
    ropeStats->elapsed_seconds = 0.0;
    ropeStats->num_heads = numHeads;
    ropeStats->qk_nope_dim = qkNopeDim;
    ropeStats->rope_dim = ropeDim;
    ropeStats->start_position = startPosition;
    ropeStats->batch_tokens = batchTokens;
    ropeStats->theta = ropeTheta;
    ropeStats->interleave = ropeInterleave ? 1 : 0;
    ropeStats->q_b_bytes = qBBytes;
    ropeStats->k_bytes = kBytes;
    ropeStats->q_nope_bytes = qNopeBytes;
    ropeStats->q_rope_bytes = qRopeBytes;
    ropeStats->command_buffer_count = 0;
    ropeStats->fused_with_mla = 1;
    ropeStats->input_buffer_direct = directInputBuffers ? 1 : 0;
    ropeStats->q_output0 = ((float *)[outQ contents])[0];
    ropeStats->k_output0 = ((float *)[outK contents])[0];

    mlaStats->ok = 1;
    mlaStats->raw_cache_bytes = rawCacheBytes;
    mlaStats->cache_f32_bytes = cacheF32Bytes;
    mlaStats->value_storage_bytes = valueSource.storage_bytes;
    mlaStats->value_source_f32_bytes = valueSource.source_f32_bytes;
    mlaStats->kv_b_f32_bytes = kvBF32Bytes;
    mlaStats->value_cache_bytes = valueCacheBytes;
    mlaStats->value_cache_total_bytes = valueCacheTotalBytes;
    mlaStats->q_nope_bytes = qNopeBytes;
    mlaStats->q_rope_bytes = qRopeBytes;
    mlaStats->output_bytes = mlaOutputBytes;
    mlaStats->scratch_bytes = rawCacheBytes + cacheF32Bytes +
                              valueSource.storage_bytes +
                              valueSource.source_f32_bytes +
                              kvBF32Bytes + qNopeBytes + qRopeBytes +
                              mlaOutputBytes;
    mlaStats->elapsed_seconds = mlaStats->cache_read_seconds +
                                mlaStats->value_read_seconds;
    mlaStats->context_length = contextLength;
    mlaStats->num_heads = numHeads;
    mlaStats->kv_lora_dim = kvLoraDim;
    mlaStats->qk_nope_dim = qkNopeDim;
    mlaStats->rope_dim = ropeDim;
    mlaStats->v_head_dim = vHeadDim;
    mlaStats->cache_position_offset = cachePositionOffset;
    mlaStats->attention_scale = attentionScale;
    mlaStats->rope_theta = ropeTheta;
    mlaStats->rope_interleave = ropeInterleave ? 1 : 0;
    mlaStats->kernel_seconds = 0.0;
    mlaStats->output_write_seconds = 0.0;
    mlaStats->command_buffer_count = 0;
    mlaStats->fused_with_rope = 1;
    mlaStats->value_cache_enabled = mla_kv_b_memory_cache_enabled();
    mlaStats->value_cache_hit = valueCacheHit;
    mlaStats->value_cache_stored = valueCacheStored;
    mlaStats->output0 = ((float *)[mlaOutput contents])[0];

    attnStats->ok = 1;
    attnStats->out_dim = attnInfo.out_dim;
    attnStats->in_dim = attnInfo.in_dim;
    attnStats->group_size = attnInfo.group_size;
    attnStats->input_bytes = attnInputBytes;
    attnStats->residual_bytes = attnOutputBytes;
    attnStats->projection_bytes = attnOutputBytes;
    attnStats->output_bytes = attnOutputBytes;
    attnStats->projection_kernel_seconds = kernelSeconds;
    attnStats->residual_add_seconds = 0.0;
    attnStats->command_buffer_count = 1;
    attnStats->fused_matvec_add = 1;
    attnStats->fused_with_post_attn_norm_router = 1;
    attnStats->fused_with_rope_mla = 1;
    attnStats->resident_mmap_backed = useResidentMetalWeights;
    attnStats->elapsed_seconds = attnStats->read_seconds +
                                 attnStats->projection_kernel_seconds +
                                 attnStats->output_write_seconds;
    attnStats->output0 = ((float *)[attnOutput contents])[0];

    normStats->ok = 1;
    normStats->weight_bytes_read = normWeightBytesRead;
    normStats->elapsed_seconds = 0.0;
    normStats->output0 = ((float *)[normed contents])[0];
    normStats->command_buffer_count = 0;
    normStats->fused_with_router = 1;
    normStats->input_buffer_direct = 1;
    if (normedDataOut) {
        *normedDataOut = [NSData dataWithBytes:[normed contents]
                                        length:(NSUInteger)normBytes];
    }
    if (normedBufferOut) *normedBufferOut = normed;

    routerStats->kernel_seconds = 0.0;
    routerStats->command_buffer_count = 0;
    routerStats->fused_with_rmsnorm = 1;
    if (useGpuTopK) {
        uint32_t *ids = (uint32_t *)[selectedIds contents];
        float *weights = (float *)[selectedWeights contents];
        for (uint32_t i = 0; i < topK; i++) {
            routerStats->experts[i] = (int)ids[i];
            routerStats->weights[i] = weights[i];
        }
        routerStats->gpu_topk = 1;
    } else {
        routerStats->logits = (float *)malloc((size_t)logitsBytes);
        if (!routerStats->logits) {
            free(selectionBias);
            fprintf(stderr, "ERROR: failed to allocate fused router logits copy\n");
            return 0;
        }
        memcpy(routerStats->logits, [logits contents], (size_t)logitsBytes);
        if (!compute_router_topk(routerStats->logits,
                                 routerInfo.num_experts,
                                 topK,
                                 selectionBias,
                                 topkOptions,
                                 routerStats->experts,
                                 routerStats->weights)) {
            free(selectionBias);
            fprintf(stderr, "ERROR: failed to compute fused router top-k\n");
            return 0;
        }
    }
    free(selectionBias);
    routerStats->elapsed_seconds = routerStats->router_read_seconds;
    routerStats->ok = 1;
    return 1;
}

static int use_fast_mxfp4_moe_kernels(Mxfp4ExpertInfo info) {
    const char *flagName = "LARGERLM_GLM_MOE_INFER_FAST_MXFP4";
    if (env_flag_disabled(flagName)) {
        return 0;
    }
    int explicitlyRequested = env_flag_enabled(flagName);
    if (info.group_size != 32 ||
        info.hidden_dim > 6144 ||
        info.intermediate_dim > 6144) {
        if (explicitlyRequested) {
            fprintf(stderr,
                    "WARNING: fast MXFP4 MoE kernels requested but unsupported for hidden=%u intermediate=%u group=%u; using scalar kernels\n",
                    info.hidden_dim,
                    info.intermediate_dim,
                    info.group_size);
        }
        return 0;
    }
    return 1;
}

static int encode_glm_mxfp4_swiglu(id<MTLCommandBuffer> cmd,
                                   id<MTLComputePipelineState> pipe,
                                   id<MTLBuffer> expert,
                                   Mxfp4ExpertInfo info,
                                   id<MTLBuffer> input,
                                   id<MTLBuffer> act,
                                   int fastKernel) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create swiglu command encoder\n");
        return 0;
    }
    uint32_t out_dim = info.intermediate_dim;
    uint32_t in_dim = info.hidden_dim;
    uint32_t group_size = info.group_size;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:expert offset:(NSUInteger)info.gate_w.offset atIndex:0];
    [enc setBuffer:expert offset:(NSUInteger)info.gate_s.offset atIndex:1];
    [enc setBuffer:expert offset:(NSUInteger)info.up_w.offset atIndex:2];
    [enc setBuffer:expert offset:(NSUInteger)info.up_s.offset atIndex:3];
    [enc setBuffer:input offset:0 atIndex:4];
    [enc setBuffer:act offset:0 atIndex:5];
    [enc setBytes:&out_dim length:sizeof(out_dim) atIndex:6];
    [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:7];
    [enc setBytes:&group_size length:sizeof(group_size) atIndex:8];
    if (fastKernel) {
        NSUInteger rowGroups = ((NSUInteger)info.intermediate_dim + 7u) / 8u;
        [enc dispatchThreadgroups:MTLSizeMake(rowGroups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake(info.intermediate_dim, 1, 1)
       threadsPerThreadgroup:threadgroup_1d(pipe, info.intermediate_dim)];
    }
    [enc endEncoding];
    return 1;
}

static int encode_glm_mxfp4_down_weighted_add(id<MTLCommandBuffer> cmd,
                                              id<MTLComputePipelineState> pipe,
                                              id<MTLBuffer> expert,
                                              Mxfp4ExpertInfo info,
                                              id<MTLBuffer> act,
                                              id<MTLBuffer> accum,
                                              float weight,
                                              int fastKernel) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        fprintf(stderr, "ERROR: failed to create down/add command encoder\n");
        return 0;
    }
    uint32_t out_dim = info.hidden_dim;
    uint32_t in_dim = info.intermediate_dim;
    uint32_t group_size = info.group_size;
    [enc setComputePipelineState:pipe];
    [enc setBuffer:expert offset:(NSUInteger)info.down_w.offset atIndex:0];
    [enc setBuffer:expert offset:(NSUInteger)info.down_s.offset atIndex:1];
    [enc setBuffer:act offset:0 atIndex:2];
    [enc setBuffer:accum offset:0 atIndex:3];
    [enc setBytes:&weight length:sizeof(weight) atIndex:4];
    [enc setBytes:&out_dim length:sizeof(out_dim) atIndex:5];
    [enc setBytes:&in_dim length:sizeof(in_dim) atIndex:6];
    [enc setBytes:&group_size length:sizeof(group_size) atIndex:7];
    if (fastKernel) {
        NSUInteger rowGroups = ((NSUInteger)info.hidden_dim + 7u) / 8u;
        [enc dispatchThreadgroups:MTLSizeMake(rowGroups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake(info.hidden_dim, 1, 1)
       threadsPerThreadgroup:threadgroup_1d(pipe, info.hidden_dim)];
    }
    [enc endEncoding];
    return 1;
}

typedef struct {
    int ok;
    uint64_t expert_bytes_read;
    uint64_t shared_bytes_read;
    double elapsed_seconds;
    double expert_read_seconds;
    uint32_t expert_read_dispatch_count;
    uint32_t expert_read_task_count;
    uint32_t expert_read_max_task_count;
    uint32_t expert_read_max_worker_count;
    uint32_t expert_read_pool_dispatch_count;
    uint32_t expert_read_serial_dispatch_count;
    double kernel_seconds;
    double shared_read_seconds;
    double shared_prefetch_seconds;
    double shared_kernel_seconds;
    double output_write_seconds;
    int residual_add_fused;
    int input_buffer_direct;
    int async_submitted;
    int shared_prefetch_used;
    uint32_t command_buffer_count;
    uint32_t synchronous_wait_count;
    float output0;
    float output0_abs_error;
    int output0_check_ok;
    int fast_mxfp4_kernel;
} LayerMoeProbeStats;

typedef struct {
    int ok;
    uint64_t scratch_bytes;
    double elapsed_seconds;
    uint32_t position;
    uint32_t context_length;
    float output0;
    AttnProjectionProbeStats attn_projection;
    RopeSplitProbeStats rope_split;
    MlaAttentionProbeStats mla_attention;
    AttnOutputProbeStats attn_output;
    RmsNormProbeStats rms_norm;
    RouterProbeStats router;
    LayerMoeProbeStats mlp;
} DecoderLayerProbeStats;

typedef struct {
    int ok;
    uint64_t scratch_bytes;
    double elapsed_seconds;
    uint32_t position;
    uint32_t context_length;
    float output0;
    AttnProjectionProbeStats attn_projection;
    RopeSplitProbeStats rope_split;
    MlaAttentionProbeStats mla_attention;
    AttnOutputProbeStats attn_output;
    DenseMlpProbeStats dense_mlp;
} DenseDecoderLayerProbeStats;

typedef struct {
    int layer;
    int is_dense;
    ExpertFile *expert_file;
    ResidentVectorInfo input_norm;
    ResidentVectorInfo q_a_norm;
    ResidentVectorInfo kv_a_norm;
    ResidentVectorInfo post_attn_norm;
    ResidentMxfp4MatrixInfo q_a;
    ResidentMxfp4MatrixInfo q_b;
    ResidentMxfp4MatrixInfo kv_a;
    ResidentMxfp4MatrixInfo kv_b;
    int has_kv_b;
    MlaAttentionValueSourceInfo mla_value_source;
    ResidentMxfp4MatrixInfo attn_output;
    RouterWeightInfo router;
    RouterBiasInfo router_bias;
    RouterTopKOptions router_topk_options;
    Mxfp4ExpertInfo moe;
    SharedMxfp4Info shared;
    DenseMlpMxfp4Info dense_mlp;
    uint64_t attn_projection_scratch_bytes;
    uint64_t rope_split_scratch_bytes;
    uint64_t mla_attention_scratch_bytes;
    uint64_t attn_output_scratch_bytes;
    uint64_t rmsnorm_scratch_bytes;
    uint64_t router_scratch_bytes;
    uint64_t moe_scratch_bytes;
    uint64_t dense_mlp_scratch_bytes;
    uint64_t shared_storage_bytes;
    uint64_t hot_intermediate_memory_bytes;
    uint64_t layer_scratch_bytes;
} DecodeLayerPlan;

typedef struct {
    int ok;
    int layer;
    int is_dense;
    int input_from_memory;
    int input_buffer_direct;
    int hot_intermediate_tensors;
    uint64_t scratch_bytes;
    uint64_t hot_intermediate_memory_bytes;
    uint64_t expert_bytes_read;
    uint64_t dense_mlp_bytes_read;
    uint64_t shared_bytes_read;
    uint32_t expert_read_dispatch_count;
    uint32_t expert_read_task_count;
    uint32_t expert_read_max_task_count;
    uint32_t expert_read_max_worker_count;
    uint32_t expert_read_pool_dispatch_count;
    uint32_t expert_read_serial_dispatch_count;
    double expert_read_seconds;
    double moe_mlp_kernel_seconds;
    double moe_mlp_output_write_seconds;
    double moe_mlp_overhead_seconds;
    double elapsed_seconds;
    double attn_projection_elapsed_seconds;
    int attn_projection_fused_pre_cache;
    uint32_t attn_projection_command_buffer_count;
    uint32_t attn_projection_synchronous_wait_count;
    int attn_projection_async_submitted;
    double mla_attention_elapsed_seconds;
    double mla_attention_cache_read_seconds;
    double mla_attention_value_read_seconds;
    double mla_attention_kernel_seconds;
    double mla_attention_output_write_seconds;
    uint32_t mla_value_cache_hit_count;
    uint32_t mla_value_cache_store_count;
    uint64_t mla_value_cache_bytes;
    uint64_t mla_value_cache_total_bytes;
    uint64_t attn_output_bytes_read;
    double attn_output_elapsed_seconds;
    double attn_output_read_seconds;
    double attn_output_projection_kernel_seconds;
    uint64_t post_attn_norm_weight_bytes_read;
    double post_attn_norm_weight_read_seconds;
    uint64_t router_bytes_read;
    uint64_t router_correction_bias_bytes_read;
    double router_read_seconds;
    double router_kernel_seconds;
    double mlp_elapsed_seconds;
    double shared_read_seconds;
    double shared_prefetch_seconds;
    int rope_mla_fused;
    int rope_mla_input_buffer_direct;
    uint32_t rope_mla_command_buffer_count;
    int attn_output_fused_matvec_add;
    int attn_output_context1_o_proj_cache;
    int attn_output_resident_mmap_backed;
    int attn_output_norm_router_fused;
    int rope_mla_attn_output_norm_router_fused;
    uint32_t attn_output_command_buffer_count;
    int router_gpu_topk;
    int post_attn_norm_router_fused;
    int attn_output_buffer_direct;
    uint32_t post_attn_norm_command_buffer_count;
    uint32_t router_command_buffer_count;
    uint32_t post_attn_norm_router_command_buffer_count;
    uint32_t router_top_k;
    int router_experts[64];
    int dense_mlp_fused_pipeline;
    uint32_t dense_mlp_command_buffer_count;
    uint32_t dense_mlp_synchronous_wait_count;
    int dense_mlp_async_submitted;
    int moe_mlp_residual_add_fused;
    int moe_mlp_input_buffer_direct;
    int shared_prefetch_used;
    uint32_t moe_mlp_command_buffer_count;
    uint32_t moe_mlp_synchronous_wait_count;
    uint32_t synchronous_wait_count;
    float output0;
} DecodeLayerRunSummary;

static NSArray *expert_routes_payload(DecodeLayerPlan *plans,
                                      DecodeLayerRunSummary *summaries,
                                      int count) {
    NSMutableArray *routes = [NSMutableArray array];
    if (!plans || !summaries || count <= 0) {
        return routes;
    }
    for (int i = 0; i < count; i++) {
        DecodeLayerPlan *plan = &plans[i];
        DecodeLayerRunSummary *summary = &summaries[i];
        if (plan->is_dense || summary->router_top_k == 0) {
            continue;
        }
        NSMutableArray *experts =
            [NSMutableArray arrayWithCapacity:summary->router_top_k];
        for (uint32_t route = 0; route < summary->router_top_k; route++) {
            [experts addObject:@(summary->router_experts[route])];
        }
        [routes addObject:@{
            @"layer": @(plan->layer),
            @"experts": experts,
        }];
    }
    return routes;
}

typedef struct {
    int dense_count;
    int moe_count;
    int memory_input_count;
    int input_buffer_direct_count;
    uint64_t hot_intermediate_memory_bytes;
    uint64_t expert_bytes_read;
    uint64_t dense_mlp_bytes_read;
    uint64_t shared_bytes_read;
    double layer_elapsed_seconds;
    double attn_projection_elapsed_seconds;
    double mla_attention_elapsed_seconds;
    double mla_attention_cache_read_seconds;
    double mla_attention_value_read_seconds;
    double mla_attention_kernel_seconds;
    double mla_attention_output_write_seconds;
    uint32_t mla_value_cache_hit_count;
    uint32_t mla_value_cache_store_count;
    uint64_t mla_value_cache_bytes;
    uint64_t mla_value_cache_total_bytes;
    uint64_t attn_output_bytes_read;
    double attn_output_elapsed_seconds;
    double attn_output_read_seconds;
    double attn_output_projection_kernel_seconds;
    uint64_t post_attn_norm_weight_bytes_read;
    double post_attn_norm_weight_read_seconds;
    uint64_t router_bytes_read;
    uint64_t router_correction_bias_bytes_read;
    double router_read_seconds;
    double router_kernel_seconds;
    double mlp_elapsed_seconds;
    double dense_mlp_elapsed_seconds;
    double moe_mlp_elapsed_seconds;
    double expert_read_seconds;
    double shared_read_seconds;
    double shared_prefetch_seconds;
    double moe_mlp_kernel_seconds;
    double moe_mlp_output_write_seconds;
    double moe_mlp_overhead_seconds;
    double layer_overhead_seconds;
    uint32_t expert_read_dispatch_count;
    uint32_t expert_read_task_count;
    uint32_t expert_read_max_task_count;
    uint32_t expert_read_max_worker_count;
    uint32_t expert_read_pool_dispatch_count;
    uint32_t expert_read_serial_dispatch_count;
    uint32_t attn_projection_command_buffer_count;
    uint32_t attn_projection_synchronous_wait_count;
    uint32_t attn_projection_async_submitted_count;
    uint32_t rope_mla_command_buffer_count;
    uint32_t attn_output_command_buffer_count;
    uint32_t post_attn_norm_command_buffer_count;
    uint32_t router_command_buffer_count;
    uint32_t post_attn_norm_router_command_buffer_count;
    uint32_t dense_mlp_command_buffer_count;
    uint32_t dense_mlp_synchronous_wait_count;
    uint32_t dense_mlp_async_submitted_count;
    uint32_t moe_mlp_command_buffer_count;
    uint32_t moe_mlp_synchronous_wait_count;
    uint32_t shared_prefetch_used_count;
    uint32_t command_buffer_count;
    uint32_t synchronous_wait_count_estimate;
    uint32_t attn_output_context1_o_proj_cache_count;
    uint32_t attn_output_resident_mmap_backed_count;
    uint32_t attn_output_norm_router_fused_count;
    uint32_t rope_mla_attn_output_norm_router_fused_count;
    uint32_t rope_mla_input_buffer_direct_count;
    uint32_t attn_output_buffer_direct_count;
    uint32_t moe_mlp_input_buffer_direct_count;
} DecodeLayersAggregateStats;

static DecodeLayersAggregateStats collect_decode_layers_aggregate_stats(
    DecodeLayerPlan *plans,
    DecodeLayerRunSummary *summaries,
    int count) {
    DecodeLayersAggregateStats stats = {0};
    for (int i = 0; i < count; i++) {
        DecodeLayerPlan *plan = &plans[i];
        DecodeLayerRunSummary *summary = &summaries[i];
        if (plan->is_dense) {
            stats.dense_count++;
            stats.dense_mlp_elapsed_seconds += summary->mlp_elapsed_seconds;
        } else {
            stats.moe_count++;
            stats.moe_mlp_elapsed_seconds += summary->mlp_elapsed_seconds;
            stats.expert_read_seconds += summary->expert_read_seconds;
            stats.shared_read_seconds += summary->shared_read_seconds;
            stats.shared_prefetch_seconds += summary->shared_prefetch_seconds;
            if (summary->shared_prefetch_used) {
                stats.shared_prefetch_used_count++;
            }
            stats.moe_mlp_kernel_seconds += summary->moe_mlp_kernel_seconds;
            stats.moe_mlp_output_write_seconds +=
                summary->moe_mlp_output_write_seconds;
            stats.moe_mlp_overhead_seconds += summary->moe_mlp_overhead_seconds;
        }
        if (summary->input_from_memory) {
            stats.memory_input_count++;
        }
        if (summary->input_buffer_direct) {
            stats.input_buffer_direct_count++;
        }
        stats.hot_intermediate_memory_bytes += summary->hot_intermediate_memory_bytes;
        stats.expert_bytes_read += summary->expert_bytes_read;
        stats.dense_mlp_bytes_read += summary->dense_mlp_bytes_read;
        stats.shared_bytes_read += summary->shared_bytes_read;
        stats.layer_elapsed_seconds += summary->elapsed_seconds;
        stats.attn_projection_elapsed_seconds +=
            summary->attn_projection_elapsed_seconds;
        stats.mla_attention_elapsed_seconds +=
            summary->mla_attention_elapsed_seconds;
        stats.mla_attention_cache_read_seconds +=
            summary->mla_attention_cache_read_seconds;
        stats.mla_attention_value_read_seconds +=
            summary->mla_attention_value_read_seconds;
        stats.mla_attention_kernel_seconds +=
            summary->mla_attention_kernel_seconds;
        stats.mla_attention_output_write_seconds +=
            summary->mla_attention_output_write_seconds;
        stats.mla_value_cache_hit_count += summary->mla_value_cache_hit_count;
        stats.mla_value_cache_store_count += summary->mla_value_cache_store_count;
        stats.mla_value_cache_bytes += summary->mla_value_cache_bytes;
        if (summary->mla_value_cache_total_bytes >
            stats.mla_value_cache_total_bytes) {
            stats.mla_value_cache_total_bytes =
                summary->mla_value_cache_total_bytes;
        }
        stats.attn_output_bytes_read += summary->attn_output_bytes_read;
        stats.attn_output_elapsed_seconds += summary->attn_output_elapsed_seconds;
        stats.attn_output_read_seconds += summary->attn_output_read_seconds;
        stats.attn_output_projection_kernel_seconds +=
            summary->attn_output_projection_kernel_seconds;
        stats.post_attn_norm_weight_bytes_read +=
            summary->post_attn_norm_weight_bytes_read;
        stats.post_attn_norm_weight_read_seconds +=
            summary->post_attn_norm_weight_read_seconds;
        stats.router_bytes_read += summary->router_bytes_read;
        stats.router_correction_bias_bytes_read +=
            summary->router_correction_bias_bytes_read;
        stats.router_read_seconds += summary->router_read_seconds;
        stats.router_kernel_seconds += summary->router_kernel_seconds;
        stats.mlp_elapsed_seconds += summary->mlp_elapsed_seconds;
        double knownLayerSeconds =
            summary->attn_projection_elapsed_seconds +
            summary->mla_attention_elapsed_seconds +
            summary->attn_output_elapsed_seconds +
            summary->mlp_elapsed_seconds;
        double layerOverhead = summary->elapsed_seconds - knownLayerSeconds;
        if (layerOverhead > 0.0) {
            stats.layer_overhead_seconds += layerOverhead;
        }
        stats.attn_projection_command_buffer_count +=
            summary->attn_projection_command_buffer_count;
        stats.attn_projection_synchronous_wait_count +=
            summary->attn_projection_synchronous_wait_count;
        if (summary->attn_projection_async_submitted) {
            stats.attn_projection_async_submitted_count++;
        }
        stats.rope_mla_command_buffer_count +=
            summary->rope_mla_command_buffer_count;
        stats.attn_output_command_buffer_count +=
            summary->attn_output_command_buffer_count;
        if (summary->post_attn_norm_router_fused) {
            stats.post_attn_norm_router_command_buffer_count +=
                summary->post_attn_norm_command_buffer_count;
        } else {
            stats.post_attn_norm_command_buffer_count +=
                summary->post_attn_norm_command_buffer_count;
            stats.router_command_buffer_count += summary->router_command_buffer_count;
        }
        stats.dense_mlp_command_buffer_count +=
            summary->dense_mlp_command_buffer_count;
        stats.dense_mlp_synchronous_wait_count +=
            summary->dense_mlp_synchronous_wait_count;
        if (summary->dense_mlp_async_submitted) {
            stats.dense_mlp_async_submitted_count++;
        }
        stats.moe_mlp_command_buffer_count +=
            summary->moe_mlp_command_buffer_count;
        stats.moe_mlp_synchronous_wait_count +=
            summary->moe_mlp_synchronous_wait_count;
        stats.synchronous_wait_count_estimate +=
            summary->synchronous_wait_count;
        if (summary->attn_output_norm_router_fused) {
            stats.attn_output_norm_router_fused_count++;
        }
        if (summary->attn_output_context1_o_proj_cache) {
            stats.attn_output_context1_o_proj_cache_count++;
        }
        if (summary->attn_output_resident_mmap_backed) {
            stats.attn_output_resident_mmap_backed_count++;
        }
        if (summary->rope_mla_attn_output_norm_router_fused) {
            stats.rope_mla_attn_output_norm_router_fused_count++;
        }
        if (summary->rope_mla_input_buffer_direct) {
            stats.rope_mla_input_buffer_direct_count++;
        }
        if (summary->attn_output_buffer_direct) {
            stats.attn_output_buffer_direct_count++;
        }
        if (summary->moe_mlp_input_buffer_direct) {
            stats.moe_mlp_input_buffer_direct_count++;
        }
        stats.expert_read_dispatch_count += summary->expert_read_dispatch_count;
        stats.expert_read_task_count += summary->expert_read_task_count;
        if (summary->expert_read_max_task_count >
            stats.expert_read_max_task_count) {
            stats.expert_read_max_task_count = summary->expert_read_max_task_count;
        }
        if (summary->expert_read_max_worker_count >
            stats.expert_read_max_worker_count) {
            stats.expert_read_max_worker_count = summary->expert_read_max_worker_count;
        }
        stats.expert_read_pool_dispatch_count +=
            summary->expert_read_pool_dispatch_count;
        stats.expert_read_serial_dispatch_count +=
            summary->expert_read_serial_dispatch_count;
    }
    stats.command_buffer_count =
        stats.attn_projection_command_buffer_count +
        stats.rope_mla_command_buffer_count +
        stats.attn_output_command_buffer_count +
        stats.post_attn_norm_command_buffer_count +
        stats.router_command_buffer_count +
        stats.post_attn_norm_router_command_buffer_count +
        stats.dense_mlp_command_buffer_count +
        stats.moe_mlp_command_buffer_count;
    return stats;
}

static void add_decode_layers_aggregate_payload_fields(
    NSMutableDictionary *payload,
    DecodeLayersAggregateStats stats) {
    payload[@"layer_count"] = @(stats.dense_count + stats.moe_count);
    payload[@"dense_layer_count"] = @(stats.dense_count);
    payload[@"moe_layer_count"] = @(stats.moe_count);
    payload[@"layer_input_buffer_direct_count"] =
        @(stats.input_buffer_direct_count);
    payload[@"layer_elapsed_seconds"] = @(stats.layer_elapsed_seconds);
    payload[@"attn_projection_elapsed_seconds"] =
        @(stats.attn_projection_elapsed_seconds);
    payload[@"mla_attention_elapsed_seconds"] =
        @(stats.mla_attention_elapsed_seconds);
    payload[@"mla_attention_cache_read_seconds"] =
        @(stats.mla_attention_cache_read_seconds);
    payload[@"mla_attention_value_read_seconds"] =
        @(stats.mla_attention_value_read_seconds);
    payload[@"mla_attention_kernel_seconds"] =
        @(stats.mla_attention_kernel_seconds);
    payload[@"mla_attention_output_write_seconds"] =
        @(stats.mla_attention_output_write_seconds);
    payload[@"mla_value_cache_hit_count"] = @(stats.mla_value_cache_hit_count);
    payload[@"mla_value_cache_store_count"] = @(stats.mla_value_cache_store_count);
    payload[@"mla_value_cache_bytes"] = @(stats.mla_value_cache_bytes);
    payload[@"mla_value_cache_total_bytes"] =
        @(stats.mla_value_cache_total_bytes);
    payload[@"attn_output_bytes_read"] = @(stats.attn_output_bytes_read);
    payload[@"attn_output_elapsed_seconds"] =
        @(stats.attn_output_elapsed_seconds);
    payload[@"attn_output_read_seconds"] =
        @(stats.attn_output_read_seconds);
    payload[@"attn_output_projection_kernel_seconds"] =
        @(stats.attn_output_projection_kernel_seconds);
    payload[@"post_attn_norm_weight_bytes_read"] =
        @(stats.post_attn_norm_weight_bytes_read);
    payload[@"post_attn_norm_weight_read_seconds"] =
        @(stats.post_attn_norm_weight_read_seconds);
    payload[@"router_bytes_read"] = @(stats.router_bytes_read);
    payload[@"router_correction_bias_bytes_read"] =
        @(stats.router_correction_bias_bytes_read);
    payload[@"router_read_seconds"] = @(stats.router_read_seconds);
    payload[@"router_kernel_seconds"] = @(stats.router_kernel_seconds);
    payload[@"mlp_elapsed_seconds"] = @(stats.mlp_elapsed_seconds);
    payload[@"dense_mlp_elapsed_seconds"] = @(stats.dense_mlp_elapsed_seconds);
    payload[@"moe_mlp_elapsed_seconds"] = @(stats.moe_mlp_elapsed_seconds);
    payload[@"expert_read_seconds"] = @(stats.expert_read_seconds);
    payload[@"shared_bytes_read"] = @(stats.shared_bytes_read);
    payload[@"shared_read_seconds"] = @(stats.shared_read_seconds);
    payload[@"shared_prefetch_seconds"] = @(stats.shared_prefetch_seconds);
    payload[@"shared_prefetch_used_count"] =
        @(stats.shared_prefetch_used_count);
    payload[@"moe_mlp_kernel_seconds"] = @(stats.moe_mlp_kernel_seconds);
    payload[@"moe_mlp_output_write_seconds"] =
        @(stats.moe_mlp_output_write_seconds);
    payload[@"moe_mlp_overhead_seconds"] = @(stats.moe_mlp_overhead_seconds);
    payload[@"layer_overhead_seconds"] = @(stats.layer_overhead_seconds);
    payload[@"attn_projection_command_buffer_count"] =
        @(stats.attn_projection_command_buffer_count);
    payload[@"attn_projection_synchronous_wait_count"] =
        @(stats.attn_projection_synchronous_wait_count);
    payload[@"attn_projection_async_submitted_count"] =
        @(stats.attn_projection_async_submitted_count);
    payload[@"rope_mla_command_buffer_count"] =
        @(stats.rope_mla_command_buffer_count);
    payload[@"attn_output_command_buffer_count"] =
        @(stats.attn_output_command_buffer_count);
    payload[@"attn_output_context1_o_proj_cache_count"] =
        @(stats.attn_output_context1_o_proj_cache_count);
    payload[@"attn_output_resident_mmap_backed_count"] =
        @(stats.attn_output_resident_mmap_backed_count);
    payload[@"post_attn_norm_command_buffer_count"] =
        @(stats.post_attn_norm_command_buffer_count);
    payload[@"router_command_buffer_count"] =
        @(stats.router_command_buffer_count);
    payload[@"post_attn_norm_router_command_buffer_count"] =
        @(stats.post_attn_norm_router_command_buffer_count);
    payload[@"dense_mlp_command_buffer_count"] =
        @(stats.dense_mlp_command_buffer_count);
    payload[@"dense_mlp_synchronous_wait_count"] =
        @(stats.dense_mlp_synchronous_wait_count);
    payload[@"dense_mlp_async_submitted_count"] =
        @(stats.dense_mlp_async_submitted_count);
    payload[@"moe_mlp_command_buffer_count"] =
        @(stats.moe_mlp_command_buffer_count);
    payload[@"moe_mlp_synchronous_wait_count"] =
        @(stats.moe_mlp_synchronous_wait_count);
    payload[@"attn_output_norm_router_fused_count"] =
        @(stats.attn_output_norm_router_fused_count);
    payload[@"rope_mla_attn_output_norm_router_fused_count"] =
        @(stats.rope_mla_attn_output_norm_router_fused_count);
    payload[@"rope_mla_input_buffer_direct_count"] =
        @(stats.rope_mla_input_buffer_direct_count);
    payload[@"attn_output_buffer_direct_count"] =
        @(stats.attn_output_buffer_direct_count);
    payload[@"moe_mlp_input_buffer_direct_count"] =
        @(stats.moe_mlp_input_buffer_direct_count);
    payload[@"command_buffer_count"] = @(stats.command_buffer_count);
    payload[@"synchronous_wait_count_estimate"] =
        @(stats.synchronous_wait_count_estimate);
    payload[@"expert_read_dispatch_count"] = @(stats.expert_read_dispatch_count);
    payload[@"expert_read_task_count"] = @(stats.expert_read_task_count);
    payload[@"expert_read_max_task_count"] = @(stats.expert_read_max_task_count);
    payload[@"expert_read_max_worker_count"] =
        @(stats.expert_read_max_worker_count);
    payload[@"expert_read_pool_dispatch_count"] =
        @(stats.expert_read_pool_dispatch_count);
    payload[@"expert_read_serial_dispatch_count"] =
        @(stats.expert_read_serial_dispatch_count);
}

typedef struct {
    int ok;
    int input_from_memory;
    int resident_mmap_backed;
    uint64_t bytes_read;
    uint64_t lm_head_bytes_read;
    uint64_t scratch_bytes;
    uint64_t chunk_rows;
    uint64_t chunks;
    double elapsed_seconds;
    double norm_elapsed_seconds;
    double read_seconds;
    double kernel_seconds;
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t group_size;
    uint32_t top_k;
    float scores[64];
    uint64_t ids[64];
    uint32_t count;
} FinalLogitsProbeStats;

typedef struct {
    int ok;
    uint64_t token_id;
    uint64_t bytes_read;
    uint64_t output_bytes;
    double elapsed_seconds;
    double read_seconds;
    double decode_seconds;
    double write_seconds;
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t group_size;
    float output0;
} EmbeddingLookupStats;

static int read_shared_mxfp4_into_buffer(NSString *residentBinPath,
                                         SharedMxfp4Info info,
                                         id<MTLBuffer> buffer,
                                         LayerMoeProbeStats *stats) {
    if (!buffer || info.total_bytes == 0 || info.total_bytes > (uint64_t)[buffer length]) {
        fprintf(stderr, "ERROR: shared expert buffer is too small\n");
        return 0;
    }
    int closeFd = 0;
    int fd = open_resident_read_fd(residentBinPath, &closeFd);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open resident file for shared expert %s: %s\n",
                [residentBinPath UTF8String],
                strerror(errno));
        return 0;
    }
    typedef struct {
        Mxfp4ComponentInfo src;
        Mxfp4ComponentInfo dst;
    } SharedCopy;
    SharedCopy copies[6] = {
        {info.src_gate_w, info.local.gate_w},
        {info.src_gate_s, info.local.gate_s},
        {info.src_up_w, info.local.up_w},
        {info.src_up_s, info.local.up_s},
        {info.src_down_w, info.local.down_w},
        {info.src_down_s, info.local.down_s},
    };
    double started = now_seconds();
    for (int i = 0; i < 6; i++) {
        if (!pread_exact_or_report(fd,
                                   (uint8_t *)[buffer contents] + copies[i].dst.offset,
                                   copies[i].src.size,
                                   copies[i].src.offset,
                                   [residentBinPath UTF8String])) {
            close_resident_read_fd(fd, closeFd);
            return 0;
        }
        stats->shared_bytes_read += copies[i].src.size;
    }
    close_resident_read_fd(fd, closeFd);
    stats->shared_read_seconds += now_seconds() - started;
    [buffer didModifyRange:NSMakeRange(0, (NSUInteger)info.total_bytes)];
    return 1;
}

static int check_completed_command_buffer(id<MTLCommandBuffer> cmd, const char *label) {
    if (!cmd) {
        return 1;
    }
    if (cmd.status != MTLCommandBufferStatusCompleted &&
        cmd.status != MTLCommandBufferStatusError) {
        [cmd waitUntilCompleted];
    }
    if (cmd.status == MTLCommandBufferStatusError) {
        fprintf(stderr,
                "ERROR: %s command failed: %s\n",
                label,
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    return 1;
}

static int run_layer_moe_probe(id<MTLDevice> device,
                               ExpertFile *probeFile,
                               Mxfp4ExpertInfo info,
                               IntList experts,
                               FloatList weights,
                               NSString *inputPath,
                               NSData *inputDataOverride,
                               id<MTLBuffer> inputBufferOverride,
                               NSData *residualData,
                               id<MTLBuffer> residualBufferOverride,
                               NSString *residentBinPath,
                               SharedMxfp4Info *sharedInfo,
                               NSString *outputPath,
                               NSData **outputDataOut,
                               id<MTLBuffer> __strong *outputBufferOut,
                               int expectOutput0Set,
                               double expectOutput0,
                               NSArray *expertBuffers,
                               int prefetchedSharedExpert,
                               NSUInteger prefetchedSharedExpertSlot,
                               LayerMoeProbeStats *prefetchedSharedStats,
                               int allowAsyncSubmit,
                               id<MTLCommandBuffer> __strong *pendingCommandOut,
                               LayerMoeProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (outputBufferOut) {
        *outputBufferOut = nil;
    }
    if (pendingCommandOut) {
        *pendingCommandOut = nil;
    }
    stats->input_buffer_direct = inputBufferOverride ? 1 : 0;
    if (!probeFile || !expertBuffers || [expertBuffers count] == 0 ||
        experts.count <= 0 || experts.count != weights.count) {
        fprintf(stderr, "ERROR: invalid --probe-layer-moe route inputs\n");
        return 0;
    }
    LayerMoeProbeStats sharedPrefetchStats = {0};
    if (prefetchedSharedStats) {
        sharedPrefetchStats = *prefetchedSharedStats;
    }
    if (sharedInfo &&
        (sharedInfo->local.hidden_dim != info.hidden_dim ||
         sharedInfo->local.intermediate_dim != info.intermediate_dim ||
         sharedInfo->local.group_size != info.group_size)) {
        fprintf(stderr, "ERROR: shared expert dims do not match routed expert dims\n");
        return 0;
    }
    if (sharedInfo && !residentBinPath) {
        fprintf(stderr, "ERROR: shared expert requires resident weight path\n");
        return 0;
    }
    if (prefetchedSharedExpert) {
        if (!sharedInfo || prefetchedSharedExpertSlot >= [expertBuffers count] ||
            sharedPrefetchStats.shared_bytes_read == 0) {
            fprintf(stderr, "ERROR: invalid prefetched shared expert state\n");
            return 0;
        }
        stats->shared_prefetch_used = 1;
        stats->shared_prefetch_seconds =
            sharedPrefetchStats.shared_prefetch_seconds > 0.0
                ? sharedPrefetchStats.shared_prefetch_seconds
                : sharedPrefetchStats.shared_read_seconds;
        stats->shared_bytes_read += sharedPrefetchStats.shared_bytes_read;
        stats->shared_read_seconds += sharedPrefetchStats.shared_read_seconds;
    }
    uint64_t inputBytes = (uint64_t)info.hidden_dim * sizeof(float);
    uint64_t actBytes = (uint64_t)info.intermediate_dim * sizeof(float);
    if (inputBytes > NSUIntegerMax || actBytes > NSUIntegerMax) {
        fprintf(stderr, "ERROR: --probe-layer-moe buffers exceed NSUIntegerMax\n");
        return 0;
    }
    NSData *inputData = nil;
    if (inputBufferOverride) {
        if ((uint64_t)[inputBufferOverride length] < inputBytes) {
            fprintf(stderr,
                    "ERROR: input MTLBuffer bytes %llu are smaller than hidden_dim bytes %llu\n",
                    (unsigned long long)[inputBufferOverride length],
                    (unsigned long long)inputBytes);
            return 0;
        }
    } else {
        inputData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
        if (!inputData) {
            fprintf(stderr, "ERROR: failed to read --input-f32 %s\n", [inputPath UTF8String]);
            return 0;
        }
        if ((uint64_t)[inputData length] != inputBytes) {
            fprintf(stderr,
                    "ERROR: --input-f32 bytes %llu do not match hidden_dim bytes %llu\n",
                    (unsigned long long)[inputData length],
                    (unsigned long long)inputBytes);
            return 0;
        }
    }
    if (residualBufferOverride) {
        if ((uint64_t)[residualBufferOverride length] < inputBytes) {
            fprintf(stderr,
                    "ERROR: residual MTLBuffer bytes %llu are smaller than expected %llu\n",
                    (unsigned long long)[residualBufferOverride length],
                    (unsigned long long)inputBytes);
            return 0;
        }
    } else if (residualData && (uint64_t)[residualData length] != inputBytes) {
        fprintf(stderr,
                "ERROR: residual bytes %llu do not match expected %llu\n",
                (unsigned long long)[residualData length],
                (unsigned long long)inputBytes);
        return 0;
    }
    id<MTLLibrary> library = make_glm_moe_library(device);
    if (!library) {
        return 0;
    }
    int fastKernel = use_fast_mxfp4_moe_kernels(info);
    stats->fast_mxfp4_kernel = fastKernel;
    id<MTLComputePipelineState> swiglu =
        make_glm_pipeline(device,
                          library,
                          fastKernel ? @"glm_mxfp4_swiglu_gs32_fast"
                                     : @"glm_mxfp4_swiglu_gs32");
    id<MTLComputePipelineState> down =
        make_glm_pipeline(device,
                          library,
                          fastKernel ? @"glm_mxfp4_down_weighted_add_gs32_fast"
                                     : @"glm_mxfp4_down_weighted_add_gs32");
    int hasResidual = residualBufferOverride || residualData;
    id<MTLComputePipelineState> addPipe =
        hasResidual ? make_glm_pipeline(device, library, @"glm_add_inplace_f32") : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    if (!swiglu || !down || (hasResidual && !addPipe) || !queue) {
        return 0;
    }
    id<MTLBuffer> input = inputBufferOverride ?: [device newBufferWithBytes:[inputData bytes]
                                                                      length:(NSUInteger)inputBytes
                                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> act = [device newBufferWithLength:(NSUInteger)actBytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> accum = [device newBufferWithLength:(NSUInteger)inputBytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> residual = residualBufferOverride ?:
        (residualData
        ? [device newBufferWithBytes:[residualData bytes]
                              length:(NSUInteger)inputBytes
                             options:MTLResourceStorageModeShared]
        : nil);
    if (!input || !act || !accum || (hasResidual && !residual)) {
        fprintf(stderr, "ERROR: failed to allocate --probe-layer-moe Metal buffers\n");
        return 0;
    }
    memset([accum contents], 0, (size_t)inputBytes);
    [accum didModifyRange:NSMakeRange(0, (NSUInteger)inputBytes)];

    double started = now_seconds();
    int totalRoutes = experts.count + (sharedInfo ? 1 : 0);
    int canSubmitAsync =
        allowAsyncSubmit &&
        pendingCommandOut &&
        outputBufferOut &&
        outputDataOut == NULL &&
        outputPath == nil &&
        !expectOutput0Set &&
        totalRoutes <= (int)[expertBuffers count];
    int consumedPrefetchedSharedExpert = 0;
    int routeIndex = 0;
    while (routeIndex < totalRoutes) {
        int batchCount = (int)[expertBuffers count];
        if (batchCount > totalRoutes - routeIndex) {
            batchCount = totalRoutes - routeIndex;
        }
        if (batchCount > MAX_EXPERT_BUFFERS) {
            fprintf(stderr,
                    "ERROR: expert buffer batch %d exceeds max %d\n",
                    batchCount,
                    MAX_EXPERT_BUFFERS);
            return 0;
        }
        int routedReadCount = 0;
        if (routeIndex < experts.count) {
            routedReadCount = experts.count - routeIndex;
            if (routedReadCount > batchCount) {
                routedReadCount = batchCount;
            }
        }
        if (routedReadCount > 0) {
            DirectExpertReadStats readStats = {0};
            if (!read_direct_expert_slots(probeFile,
                                          experts,
                                          routeIndex,
                                          routedReadCount,
                                          expertBuffers,
                                          &readStats)) {
                return 0;
            }
            stats->expert_bytes_read += readStats.bytes_read;
            stats->expert_read_seconds += readStats.read_seconds;
            stats->expert_read_dispatch_count += readStats.dispatch_count;
            stats->expert_read_task_count += readStats.task_count;
            if (readStats.max_task_count > stats->expert_read_max_task_count) {
                stats->expert_read_max_task_count = readStats.max_task_count;
            }
            if (readStats.max_worker_count > stats->expert_read_max_worker_count) {
                stats->expert_read_max_worker_count = readStats.max_worker_count;
            }
            stats->expert_read_pool_dispatch_count += readStats.pool_dispatch_count;
            stats->expert_read_serial_dispatch_count += readStats.serial_dispatch_count;
        }
        for (int slot = routedReadCount; slot < batchCount; slot++) {
            int currentRoute = routeIndex + slot;
            if (currentRoute >= experts.count) {
                id<MTLBuffer> routeBuffer = [expertBuffers objectAtIndex:(NSUInteger)slot];
                if (prefetchedSharedExpert &&
                    routeIndex == 0 &&
                    (NSUInteger)slot == prefetchedSharedExpertSlot) {
                    consumedPrefetchedSharedExpert = 1;
                    continue;
                }
                if (!read_shared_mxfp4_into_buffer(residentBinPath,
                                                   *sharedInfo,
                                                   routeBuffer,
                                                   stats)) {
                    return 0;
                }
            }
        }
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) {
            fprintf(stderr, "ERROR: failed to create --probe-layer-moe command buffer\n");
            return 0;
        }
        for (int slot = 0; slot < batchCount; slot++) {
            int currentRoute = routeIndex + slot;
            id<MTLBuffer> routeBuffer = [expertBuffers objectAtIndex:(NSUInteger)slot];
            Mxfp4ExpertInfo routeInfo =
                currentRoute < experts.count ? info : sharedInfo->local;
            float routeWeight =
                currentRoute < experts.count ? weights.values[currentRoute] : 1.0f;
            int fuseResidualAdd = residual && currentRoute == totalRoutes - 1;
            if (!encode_glm_mxfp4_swiglu(cmd,
                                         swiglu,
                                         routeBuffer,
                                         routeInfo,
                                         input,
                                         act,
                                         fastKernel) ||
                !encode_glm_mxfp4_down_weighted_add(cmd,
                                                    down,
                                                    routeBuffer,
                                                    routeInfo,
                                                    act,
                                                    accum,
                                                    routeWeight,
                                                    fastKernel) ||
                (fuseResidualAdd &&
                 !encode_glm_add_inplace_f32(cmd,
                                             addPipe,
                                             residual,
                                             accum,
                                             info.hidden_dim))) {
                return 0;
            }
        }
        double kernelStarted = now_seconds();
        [cmd commit];
        stats->command_buffer_count++;
        if (residual && routeIndex + batchCount == totalRoutes) {
            stats->residual_add_fused = 1;
        }
        if (canSubmitAsync && routeIndex == 0 && routeIndex + batchCount == totalRoutes) {
            stats->async_submitted = 1;
            stats->kernel_seconds += now_seconds() - kernelStarted;
            stats->elapsed_seconds = now_seconds() - started;
            stats->output0_check_ok = 1;
            stats->ok = 1;
            *pendingCommandOut = cmd;
            *outputBufferOut = accum;
            return 1;
        }
        [cmd waitUntilCompleted];
        stats->synchronous_wait_count++;
        stats->kernel_seconds += now_seconds() - kernelStarted;
        if (!check_completed_command_buffer(cmd, "--probe-layer-moe")) {
            return 0;
        }
        routeIndex += batchCount;
    }
    if (prefetchedSharedExpert && !consumedPrefetchedSharedExpert) {
        fprintf(stderr, "ERROR: prefetched shared expert was not consumed\n");
        return 0;
    }
    stats->elapsed_seconds = now_seconds() - started;
    if (residualData && !stats->residual_add_fused) {
        float *accumValues = (float *)[accum contents];
        const float *residualValues = (const float *)[residualData bytes];
        for (uint64_t i = 0; i < (uint64_t)info.hidden_dim; i++) {
            accumValues[i] += residualValues[i];
        }
    }
    stats->output0 = ((float *)[accum contents])[0];
    stats->output0_check_ok = 1;
    if (expectOutput0Set) {
        stats->output0_abs_error = fabsf(stats->output0 - (float)expectOutput0);
        stats->output0_check_ok = stats->output0_abs_error <= 5.0e-3f;
    }
    double writeStarted = now_seconds();
    NSData *outputData = nil;
    if (outputDataOut || outputPath) {
        outputData = [NSData dataWithBytes:[accum contents] length:(NSUInteger)inputBytes];
    }
    if (outputPath && ![outputData writeToFile:outputPath atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write --output-f32 %s\n", [outputPath UTF8String]);
        return 0;
    }
    if (outputDataOut) {
        *outputDataOut = outputData;
    }
    if (outputBufferOut) {
        *outputBufferOut = accum;
    }
    stats->output_write_seconds = now_seconds() - writeStarted;
    stats->ok = stats->output0_check_ok;
    if (!stats->output0_check_ok) {
        fprintf(stderr,
                "ERROR: --probe-layer-moe output[0] %.6f differs from expected %.6f by %.6f\n",
                stats->output0,
                expectOutput0,
                stats->output0_abs_error);
    }
    return stats->ok;
}

static int write_k_rope_suffix_from_kv_a_data(NSData *kvAData,
                                              NSString *outPath,
                                              uint32_t kvLoraDim,
                                              uint32_t ropeDim,
                                              NSData **outData) {
    uint64_t expectedBytes =
        ((uint64_t)kvLoraDim + (uint64_t)ropeDim) * sizeof(float);
    uint64_t offsetBytes = (uint64_t)kvLoraDim * sizeof(float);
    uint64_t ropeBytes = (uint64_t)ropeDim * sizeof(float);
    if (!kvAData || (uint64_t)[kvAData length] != expectedBytes) {
        fprintf(stderr,
                "ERROR: KV-A data for %s bytes do not match expected %llu\n",
                outPath ? [outPath UTF8String] : "<memory>",
                (unsigned long long)expectedBytes);
        return 0;
    }
    NSData *ropeData = [NSData dataWithBytes:((const uint8_t *)[kvAData bytes]) + offsetBytes
                                      length:(NSUInteger)ropeBytes];
    if (outPath && ![ropeData writeToFile:outPath atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write K RoPE suffix %s\n",
                [outPath UTF8String]);
        return 0;
    }
    if (outData) {
        *outData = ropeData;
    }
    return 1;
}

static NSData *context1_latent_from_kv_a_data(NSData *kvAData,
                                              uint32_t kvLoraDim) {
    uint64_t latentBytes = (uint64_t)kvLoraDim * sizeof(float);
    if (!kvAData || latentBytes > (uint64_t)[kvAData length] ||
        latentBytes > (uint64_t)NSUIntegerMax) {
        fprintf(stderr,
                "ERROR: KV-A data is too small for context1 latent: %llu/%llu\n",
                (unsigned long long)(kvAData ? [kvAData length] : 0),
                (unsigned long long)latentBytes);
        return nil;
    }
    return [kvAData subdataWithRange:NSMakeRange(0, (NSUInteger)latentBytes)];
}

static int run_decoder_layer_probe(id<MTLDevice> device,
                                   NSString *residentBinPath,
                                   int layerId,
                                   ResidentVectorInfo inputNormInfo,
                                   ResidentVectorInfo qANormInfo,
                                   ResidentVectorInfo kvANormInfo,
                                   ResidentVectorInfo postAttnNormInfo,
                                   ResidentMxfp4MatrixInfo qAInfo,
                                   ResidentMxfp4MatrixInfo qBInfo,
                                   ResidentMxfp4MatrixInfo kvAInfo,
                                   int hasKVB,
                                   ResidentMxfp4MatrixInfo kvBInfo,
                                   MlaAttentionValueSourceInfo mlaValueSource,
                                   ResidentMxfp4MatrixInfo attnOutputInfo,
                                   RouterWeightInfo routerInfo,
                                   RouterBiasInfo routerBiasInfo,
                                   RouterTopKOptions routerTopKOptions,
                                   ExpertFile *probeFile,
                                   Mxfp4ExpertInfo moeInfo,
                                   SharedMxfp4Info *sharedInfo,
                                   NSString *inputPath,
                                   NSData *inputDataOverride,
                                   id<MTLBuffer> inputBufferOverride,
                                   NSString *workDir,
                                   NSString *cacheLayoutPath,
                                   NSString *cacheFilePath,
                                   NSString *context1OProjCacheLayoutPath,
                                   NSString *context1OProjCacheFilePath,
                                   uint64_t position,
                                   uint32_t contextLength,
                                   uint32_t numHeads,
                                   uint32_t qkNopeDim,
                                   uint32_t ropeDim,
                                   uint32_t vHeadDim,
                                   uint32_t cachePositionOffset,
                                   float attentionScale,
                                   float ropeTheta,
                                   int ropeInterleave,
                                   uint64_t maxCacheFileBytes,
                                   uint64_t maxCacheReadBytes,
                                   uint32_t topK,
                                   float rmsNormEps,
                                   NSString *outputRouterJson,
                                   NSString *outputPath,
                                   NSData **outputDataOut,
                                   id<MTLBuffer> __strong *outputBufferOut,
                                   int writeDebugFiles,
                                   NSArray *expertBuffers,
                                   id<MTLBuffer> residentMetalBuffer,
                                   int allowAsyncMoeSubmit,
                                   id<MTLCommandBuffer> __strong *pendingMoeCommandOut,
                                   DecoderLayerProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (outputBufferOut) {
        *outputBufferOut = nil;
    }
    stats->position = (uint32_t)position;
    stats->context_length = contextLength;
    if (!ensure_output_dir(workDir)) {
        return 0;
    }
    NSString *attnDir = [workDir stringByAppendingPathComponent:@"attn"];
    NSString *ropeDir = [workDir stringByAppendingPathComponent:@"rope"];
    if (writeDebugFiles && (!ensure_output_dir(attnDir) || !ensure_output_dir(ropeDir))) {
        return 0;
    }

    NSString *qBPath = writeDebugFiles ? [attnDir stringByAppendingPathComponent:@"attn_q_b.f32"] : nil;
    NSString *kRopePath = writeDebugFiles ? [workDir stringByAppendingPathComponent:@"k_rope.f32"] : nil;
    NSString *qNopePath = writeDebugFiles ? [ropeDir stringByAppendingPathComponent:@"q_nope.f32"] : nil;
    NSString *qRopeRawPath = writeDebugFiles ? [ropeDir stringByAppendingPathComponent:@"q_rope.f32"] : nil;
    NSString *qRopePath = writeDebugFiles ? [ropeDir stringByAppendingPathComponent:@"q_rope_rotated.f32"] : nil;
    NSString *kRopeRotatedPath = writeDebugFiles ? [ropeDir stringByAppendingPathComponent:@"k_rope_rotated.f32"] : nil;
    NSString *attnValuePath = writeDebugFiles ? [workDir stringByAppendingPathComponent:@"attn_value.f32"] : nil;
    NSString *attnProjectionPath = writeDebugFiles ? [workDir stringByAppendingPathComponent:@"attn_projection.f32"] : nil;
    NSString *attnOutPath = writeDebugFiles ? [workDir stringByAppendingPathComponent:@"attn_out.f32"] : nil;

    double totalStarted = now_seconds();
    int useContext1OProjCache =
        context1OProjCacheLayoutPath != nil && contextLength == 1u;
    int useFullyFusedAttentionBody =
        outputRouterJson == nil && !writeDebugFiles && !useContext1OProjCache;
    int useDirectCurrentKvACache = useFullyFusedAttentionBody;
    NSData *qBData = nil;
    NSData *kvAData = nil;
    id<MTLBuffer> qBBuffer = nil;
    id<MTLBuffer> kvABuffer = nil;
    if (!run_attention_projection_probe(device,
                                        residentBinPath,
                                        layerId,
                                        inputNormInfo,
                                        qANormInfo,
                                        kvANormInfo,
                                        qAInfo,
                                        qBInfo,
                                        kvAInfo,
                                        hasKVB,
                                        kvBInfo,
                                        inputPath,
                                        inputDataOverride,
                                        inputBufferOverride,
                                        attnDir,
                                        cacheLayoutPath,
                                        cacheFilePath,
                                        position,
                                        useDirectCurrentKvACache ? 0 : 1,
                                        maxCacheFileBytes,
                                        rmsNormEps,
                                        writeDebugFiles,
                                        useFullyFusedAttentionBody ? nil : &qBData,
                                        useFullyFusedAttentionBody ? nil : &kvAData,
                                        useFullyFusedAttentionBody ? &qBBuffer : nil,
                                        useFullyFusedAttentionBody ? &kvABuffer : nil,
                                        &stats->attn_projection)) {
        return 0;
    }
    NSData *kRopeData = nil;
    if (!useFullyFusedAttentionBody && !useContext1OProjCache) {
        if (!write_k_rope_suffix_from_kv_a_data(kvAData,
                                                kRopePath,
                                                mlaValueSource.kv_lora_dim,
                                                ropeDim,
                                                &kRopeData)) {
            return 0;
        }
    }
    NSData *attnValueData = nil;
    NSData *mlpNormedData = nil;
    id<MTLBuffer> mlpNormedBuffer = nil;
    NSData *attnOutData = nil;
    id<MTLBuffer> attnOutBuffer = nil;
    __block LayerMoeProbeStats sharedPrefetchStats = {0};
    __block int sharedPrefetchLoaded = 0;
    NSUInteger sharedPrefetchSlot = (NSUInteger)topK;
    GlmMoePreWaitWorkBlock sharedPrefetchWork = nil;
    uint64_t sharedPrefetchRequiredSlots = (uint64_t)topK + 1u;
    int canPrefetchSharedExpert =
        useFullyFusedAttentionBody &&
        sharedInfo &&
        residentBinPath &&
        expertBuffers &&
        topK > 0 &&
        sharedInfo->total_bytes > 0 &&
        sharedPrefetchRequiredSlots <= (uint64_t)[expertBuffers count];
    if (canPrefetchSharedExpert) {
        id<MTLBuffer> sharedPrefetchBuffer =
            [expertBuffers objectAtIndex:sharedPrefetchSlot];
        if ((uint64_t)[sharedPrefetchBuffer length] >= sharedInfo->total_bytes) {
            sharedPrefetchWork = ^int(void) {
                if (sharedPrefetchLoaded) {
                    return 1;
                }
                if (!read_shared_mxfp4_into_buffer(residentBinPath,
                                                   *sharedInfo,
                                                   sharedPrefetchBuffer,
                                                   &sharedPrefetchStats)) {
                    return 0;
                }
                sharedPrefetchLoaded = 1;
                sharedPrefetchStats.shared_prefetch_used = 1;
                sharedPrefetchStats.shared_prefetch_seconds =
                    sharedPrefetchStats.shared_read_seconds;
                return 1;
            };
        }
    }
    if (useFullyFusedAttentionBody) {
        if (!run_rope_mla_attention_output_rmsnorm_router_fused_probe(
                device,
                residentBinPath,
                residentMetalBuffer,
                mlaValueSource,
                cacheLayoutPath,
                cacheFilePath,
                layerId,
                qBData,
                kRopeData,
                qBBuffer,
                kvABuffer,
                (NSUInteger)mlaValueSource.kv_lora_dim * sizeof(float),
                useDirectCurrentKvACache ? kvABuffer : nil,
                0,
                attnOutputInfo,
                postAttnNormInfo,
                routerInfo,
                routerBiasInfo,
                routerTopKOptions,
	                inputPath,
	                inputDataOverride,
                    inputBufferOverride,
	                attnOutPath,
                contextLength,
                numHeads,
                qkNopeDim,
                ropeDim,
                vHeadDim,
                (uint32_t)position,
                cachePositionOffset,
                attentionScale,
                ropeTheta,
                ropeInterleave,
                maxCacheFileBytes,
                maxCacheReadBytes,
                rmsNormEps,
                topK,
                nil,
                &attnOutBuffer,
                nil,
                &mlpNormedBuffer,
                &stats->rope_split,
                &stats->mla_attention,
                &stats->attn_output,
                &stats->rms_norm,
                &stats->router,
                sharedPrefetchWork)) {
            return 0;
        }
        if (useDirectCurrentKvACache) {
            double cacheStarted = now_seconds();
            if (!append_kv_a_buffer_to_decode_cache(cacheLayoutPath,
                                                    cacheFilePath,
                                                    (uint64_t)layerId,
                                                    position,
                                                    kvABuffer,
                                                    kvAInfo.out_dim,
                                                    maxCacheFileBytes,
                                                    &stats->attn_projection.cache_write_bytes)) {
                return 0;
            }
            stats->attn_projection.cache_append = 1;
            stats->attn_projection.cache_write_seconds =
                now_seconds() - cacheStarted;
        }
    } else if (useContext1OProjCache) {
        NSData *latentData =
            context1_latent_from_kv_a_data(kvAData, mlaValueSource.kv_lora_dim);
        if (!latentData ||
            !run_context1_o_proj_cache_output_probe(
                device,
                context1OProjCacheLayoutPath,
                context1OProjCacheFilePath,
                layerId,
                nil,
                latentData,
                inputPath,
                inputDataOverride,
                attnOutPath,
                &attnOutData,
                &attnOutBuffer,
                maxCacheReadBytes,
                &stats->attn_output)) {
            return 0;
        }
    } else if (writeDebugFiles) {
        NSData *qNopeData = nil;
        NSData *qRopeData = nil;
        if (!run_rope_split_probe(device,
                                  qBPath,
                                  kRopePath,
                                  qBData,
                                  kRopeData,
                                  qNopePath,
                                  qRopeRawPath,
                                  qRopePath,
                                  kRopeRotatedPath,
                                  numHeads,
                                  qkNopeDim,
                                  ropeDim,
                                  (uint32_t)position,
                                  1,
                                  ropeTheta,
                                  ropeInterleave,
                                  &qNopeData,
                                  &qRopeData,
                                  &stats->rope_split)) {
            return 0;
        }
        if (!run_mla_attention_probe(device,
                                     residentBinPath,
                                     mlaValueSource,
                                     cacheLayoutPath,
                                     cacheFilePath,
                                     layerId,
                                     qNopePath,
                                     qRopePath,
                                     nil,
                                     qNopeData,
                                     qRopeData,
                                     attnValuePath,
                                     contextLength,
                                     numHeads,
                                     qkNopeDim,
                                     ropeDim,
                                     vHeadDim,
                                     cachePositionOffset,
                                     attentionScale,
                                     ropeTheta,
                                     ropeInterleave,
                                     maxCacheFileBytes,
                                     maxCacheReadBytes,
                                     &attnValueData,
                                     &stats->mla_attention)) {
            return 0;
        }
    } else if (!useFullyFusedAttentionBody &&
               !run_rope_split_mla_attention_fused_probe(device,
                                                         residentBinPath,
                                                         mlaValueSource,
                                                         cacheLayoutPath,
                                                         cacheFilePath,
                                                         layerId,
                                                         qBData,
                                                         kRopeData,
                                                         attnValuePath,
                                                         contextLength,
                                                         numHeads,
                                                         qkNopeDim,
                                                         ropeDim,
                                                         vHeadDim,
                                                         (uint32_t)position,
                                                         cachePositionOffset,
                                                         attentionScale,
                                                         ropeTheta,
                                                         ropeInterleave,
                                                         maxCacheFileBytes,
                                                         maxCacheReadBytes,
                                                         &attnValueData,
                                                         &stats->rope_split,
                                                         &stats->mla_attention)) {
        return 0;
    }
    if (!useFullyFusedAttentionBody) {
        if (!useContext1OProjCache) {
            if (!run_attention_output_probe(device,
                                            residentBinPath,
                                            residentMetalBuffer,
                                            attnOutputInfo,
                                            attnValuePath,
                                            attnValueData,
                                            inputPath,
                                            inputDataOverride,
                                            attnProjectionPath,
                                            attnOutPath,
                                            &attnOutData,
                                            &attnOutBuffer,
                                            &stats->attn_output)) {
                return 0;
            }
        }
        if (outputRouterJson == nil) {
            if (!run_rmsnorm_router_fused_probe(device,
                                                residentBinPath,
                                                postAttnNormInfo,
                                                routerInfo,
                                                routerBiasInfo,
                                                routerTopKOptions,
                                                layerId,
                                                attnOutPath,
                                                attnOutData,
                                                attnOutBuffer,
                                                rmsNormEps,
                                                topK,
                                                nil,
                                                &mlpNormedBuffer,
                                                &stats->rms_norm,
                                                &stats->router)) {
                return 0;
            }
        } else {
            if (!run_rmsnorm_probe(device,
                                   residentBinPath,
                                   postAttnNormInfo,
                                   attnOutPath,
                                   attnOutData,
                                   rmsNormEps,
                                   &mlpNormedData,
                                   &stats->rms_norm)) {
                return 0;
            }
            if (!run_router_probe(device,
                                  residentBinPath,
                                  routerInfo,
                                  routerBiasInfo,
                                  routerTopKOptions,
                                  layerId,
                                  attnOutPath,
                                  mlpNormedData,
                                  topK,
                                  outputRouterJson,
                                  1,
                                  &stats->router)) {
                return 0;
            }
        }
    }
    if (!attnOutData && !attnOutBuffer) {
        fprintf(stderr, "ERROR: failed to read decoder attention output %s\n",
                attnOutPath ? [attnOutPath UTF8String] : "<memory>");
        return 0;
    }
    IntList routeExperts = {
        .values = stats->router.experts,
        .count = (int)stats->router.top_k,
    };
    FloatList routeWeights = {
        .values = stats->router.weights,
        .count = (int)stats->router.top_k,
    };
    for (uint32_t i = 0; i < stats->router.top_k; i++) {
        if ((uint64_t)stats->router.experts[i] >= probeFile->num_experts) {
            fprintf(stderr,
                    "ERROR: decoder router selected expert %d outside layer expert count %llu\n",
                    stats->router.experts[i],
                    (unsigned long long)probeFile->num_experts);
            return 0;
        }
    }
    if (!run_layer_moe_probe(device,
                             probeFile,
                             moeInfo,
                             routeExperts,
                             routeWeights,
                             attnOutPath,
	                             mlpNormedData,
	                             mlpNormedBuffer,
	                             attnOutData,
                                 attnOutBuffer,
	                             residentBinPath,
	                             sharedInfo,
	                             outputPath,
	                             outputDataOut,
                                 outputBufferOut,
                             0,
                             0.0,
                             expertBuffers,
                             sharedPrefetchLoaded,
                             sharedPrefetchSlot,
                             sharedPrefetchLoaded ? &sharedPrefetchStats : NULL,
                             allowAsyncMoeSubmit,
                             pendingMoeCommandOut,
                             &stats->mlp)) {
        return 0;
    }
    stats->output0 = stats->mlp.output0;
    stats->elapsed_seconds = now_seconds() - totalStarted;
    stats->ok = 1;
    return 1;
}

static int run_dense_decoder_layer_probe(id<MTLDevice> device,
                                         NSString *residentBinPath,
                                         int layerId,
                                         ResidentVectorInfo inputNormInfo,
                                         ResidentVectorInfo qANormInfo,
                                         ResidentVectorInfo kvANormInfo,
                                         ResidentVectorInfo postAttnNormInfo,
                                         ResidentMxfp4MatrixInfo qAInfo,
                                         ResidentMxfp4MatrixInfo qBInfo,
                                         ResidentMxfp4MatrixInfo kvAInfo,
                                         int hasKVB,
                                         ResidentMxfp4MatrixInfo kvBInfo,
                                         MlaAttentionValueSourceInfo mlaValueSource,
                                         ResidentMxfp4MatrixInfo attnOutputInfo,
                                         DenseMlpMxfp4Info denseMlpInfo,
                                         NSString *inputPath,
                                         NSData *inputDataOverride,
                                         id<MTLBuffer> inputBufferOverride,
                                         NSString *workDir,
                                         NSString *cacheLayoutPath,
                                         NSString *cacheFilePath,
                                         NSString *context1OProjCacheLayoutPath,
                                         NSString *context1OProjCacheFilePath,
                                         uint64_t position,
                                         uint32_t contextLength,
                                         uint32_t numHeads,
                                         uint32_t qkNopeDim,
                                         uint32_t ropeDim,
                                         uint32_t vHeadDim,
                                         uint32_t cachePositionOffset,
                                         float attentionScale,
                                         float ropeTheta,
                                         int ropeInterleave,
                                         uint64_t maxCacheFileBytes,
                                         uint64_t maxCacheReadBytes,
                                         float rmsNormEps,
                                         NSString *outputPath,
                                         NSData **outputDataOut,
                                         id<MTLBuffer> __strong *outputBufferOut,
                                         int writeDebugFiles,
                                         id<MTLBuffer> residentMetalBuffer,
                                         int allowAsyncDenseSubmit,
                                         id<MTLCommandBuffer> __strong *pendingDenseCommandOut,
                                         DenseDecoderLayerProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (outputBufferOut) {
        *outputBufferOut = nil;
    }
    stats->position = (uint32_t)position;
    stats->context_length = contextLength;
    if (!ensure_output_dir(workDir)) {
        return 0;
    }
    NSString *attnDir = [workDir stringByAppendingPathComponent:@"attn"];
    NSString *ropeDir = [workDir stringByAppendingPathComponent:@"rope"];
    if (writeDebugFiles && (!ensure_output_dir(attnDir) || !ensure_output_dir(ropeDir))) {
        return 0;
    }

    NSString *qBPath = writeDebugFiles ? [attnDir stringByAppendingPathComponent:@"attn_q_b.f32"] : nil;
    NSString *kRopePath = writeDebugFiles ? [workDir stringByAppendingPathComponent:@"k_rope.f32"] : nil;
    NSString *qNopePath = writeDebugFiles ? [ropeDir stringByAppendingPathComponent:@"q_nope.f32"] : nil;
    NSString *qRopeRawPath = writeDebugFiles ? [ropeDir stringByAppendingPathComponent:@"q_rope.f32"] : nil;
    NSString *qRopePath = writeDebugFiles ? [ropeDir stringByAppendingPathComponent:@"q_rope_rotated.f32"] : nil;
    NSString *kRopeRotatedPath = writeDebugFiles ? [ropeDir stringByAppendingPathComponent:@"k_rope_rotated.f32"] : nil;
    NSString *attnValuePath = writeDebugFiles ? [workDir stringByAppendingPathComponent:@"attn_value.f32"] : nil;
    NSString *attnProjectionPath = writeDebugFiles ? [workDir stringByAppendingPathComponent:@"attn_projection.f32"] : nil;
    NSString *attnOutPath = writeDebugFiles ? [workDir stringByAppendingPathComponent:@"attn_out.f32"] : nil;

    double totalStarted = now_seconds();
    int useContext1OProjCache =
        context1OProjCacheLayoutPath != nil && contextLength == 1u;
    int useFullyFusedAttentionBody = !writeDebugFiles && !useContext1OProjCache;
    int useDirectCurrentKvACache = useFullyFusedAttentionBody;
    NSData *qBData = nil;
    NSData *kvAData = nil;
    id<MTLBuffer> qBBuffer = nil;
    id<MTLBuffer> kvABuffer = nil;
    if (!run_attention_projection_probe(device,
                                        residentBinPath,
                                        layerId,
                                        inputNormInfo,
                                        qANormInfo,
                                        kvANormInfo,
                                        qAInfo,
                                        qBInfo,
                                        kvAInfo,
                                        hasKVB,
                                        kvBInfo,
                                        inputPath,
                                        inputDataOverride,
                                        inputBufferOverride,
                                        attnDir,
                                        cacheLayoutPath,
                                        cacheFilePath,
                                        position,
                                        useDirectCurrentKvACache ? 0 : 1,
                                        maxCacheFileBytes,
                                        rmsNormEps,
                                        writeDebugFiles,
                                        useFullyFusedAttentionBody ? nil : &qBData,
                                        useFullyFusedAttentionBody ? nil : &kvAData,
                                        useFullyFusedAttentionBody ? &qBBuffer : nil,
                                        useFullyFusedAttentionBody ? &kvABuffer : nil,
                                        &stats->attn_projection)) {
        return 0;
    }
    NSData *kRopeData = nil;
    if (!useFullyFusedAttentionBody && !useContext1OProjCache) {
        if (!write_k_rope_suffix_from_kv_a_data(kvAData,
                                                kRopePath,
                                                mlaValueSource.kv_lora_dim,
                                                ropeDim,
                                                &kRopeData)) {
            return 0;
        }
    }
    NSData *attnValueData = nil;
    if (!useContext1OProjCache && writeDebugFiles) {
        NSData *qNopeData = nil;
        NSData *qRopeData = nil;
        if (!run_rope_split_probe(device,
                                  qBPath,
                                  kRopePath,
                                  qBData,
                                  kRopeData,
                                  qNopePath,
                                  qRopeRawPath,
                                  qRopePath,
                                  kRopeRotatedPath,
                                  numHeads,
                                  qkNopeDim,
                                  ropeDim,
                                  (uint32_t)position,
                                  1,
                                  ropeTheta,
                                  ropeInterleave,
                                  &qNopeData,
                                  &qRopeData,
                                  &stats->rope_split)) {
            return 0;
        }
        if (!run_mla_attention_probe(device,
                                     residentBinPath,
                                     mlaValueSource,
                                     cacheLayoutPath,
                                     cacheFilePath,
                                     layerId,
                                     qNopePath,
                                     qRopePath,
                                     nil,
                                     qNopeData,
                                     qRopeData,
                                     attnValuePath,
                                     contextLength,
                                     numHeads,
                                     qkNopeDim,
                                     ropeDim,
                                     vHeadDim,
                                     cachePositionOffset,
                                     attentionScale,
                                     ropeTheta,
                                     ropeInterleave,
                                     maxCacheFileBytes,
                                     maxCacheReadBytes,
                                     &attnValueData,
                                     &stats->mla_attention)) {
            return 0;
        }
    } else if (!useContext1OProjCache &&
               !useFullyFusedAttentionBody &&
               !run_rope_split_mla_attention_fused_probe(device,
                                                         residentBinPath,
                                                         mlaValueSource,
                                                         cacheLayoutPath,
                                                         cacheFilePath,
                                                         layerId,
                                                         qBData,
                                                         kRopeData,
                                                         attnValuePath,
                                                         contextLength,
                                                         numHeads,
                                                         qkNopeDim,
                                                         ropeDim,
                                                         vHeadDim,
                                                         (uint32_t)position,
                                                         cachePositionOffset,
                                                         attentionScale,
                                                         ropeTheta,
                                                         ropeInterleave,
                                                         maxCacheFileBytes,
                                                         maxCacheReadBytes,
                                                         &attnValueData,
                                                         &stats->rope_split,
                                                         &stats->mla_attention)) {
        return 0;
    }
    NSData *attnOutData = nil;
    id<MTLBuffer> attnOutBuffer = nil;
    if (useFullyFusedAttentionBody) {
        if (!run_rope_mla_attention_output_fused_probe(
                device,
                residentBinPath,
                residentMetalBuffer,
                mlaValueSource,
                cacheLayoutPath,
                cacheFilePath,
                layerId,
                qBData,
                kRopeData,
                qBBuffer,
                kvABuffer,
                (NSUInteger)mlaValueSource.kv_lora_dim * sizeof(float),
                useDirectCurrentKvACache ? kvABuffer : nil,
                0,
                attnOutputInfo,
	                inputPath,
	                inputDataOverride,
                    inputBufferOverride,
	                attnOutPath,
                contextLength,
                numHeads,
                qkNopeDim,
                ropeDim,
                vHeadDim,
                (uint32_t)position,
                cachePositionOffset,
                attentionScale,
                ropeTheta,
                ropeInterleave,
                maxCacheFileBytes,
                maxCacheReadBytes,
                nil,
                &attnOutBuffer,
                &stats->rope_split,
                &stats->mla_attention,
                &stats->attn_output)) {
            return 0;
        }
        if (useDirectCurrentKvACache) {
            double cacheStarted = now_seconds();
            if (!append_kv_a_buffer_to_decode_cache(cacheLayoutPath,
                                                    cacheFilePath,
                                                    (uint64_t)layerId,
                                                    position,
                                                    kvABuffer,
                                                    kvAInfo.out_dim,
                                                    maxCacheFileBytes,
                                                    &stats->attn_projection.cache_write_bytes)) {
                return 0;
            }
            stats->attn_projection.cache_append = 1;
            stats->attn_projection.cache_write_seconds =
                now_seconds() - cacheStarted;
        }
    } else {
        if (useContext1OProjCache) {
            NSData *latentData =
                context1_latent_from_kv_a_data(kvAData, mlaValueSource.kv_lora_dim);
            if (!latentData ||
                !run_context1_o_proj_cache_output_probe(
                    device,
                    context1OProjCacheLayoutPath,
                    context1OProjCacheFilePath,
                    layerId,
                    nil,
                    latentData,
                    inputPath,
                    inputDataOverride,
                    attnOutPath,
                    &attnOutData,
                    &attnOutBuffer,
                    maxCacheReadBytes,
                    &stats->attn_output)) {
                return 0;
            }
        } else if (!run_attention_output_probe(device,
                                               residentBinPath,
                                               residentMetalBuffer,
                                               attnOutputInfo,
                                               attnValuePath,
                                               attnValueData,
                                               inputPath,
                                               inputDataOverride,
                                               attnProjectionPath,
                                               attnOutPath,
                                               &attnOutData,
                                               nil,
                                               &stats->attn_output)) {
            return 0;
        }
    }
    if (!attnOutData && !attnOutBuffer) {
        fprintf(stderr, "ERROR: failed to read dense decoder attention output %s\n",
                [attnOutPath UTF8String]);
        return 0;
    }
    if (!run_dense_mlp_probe(device,
                             residentBinPath,
                             layerId,
                             postAttnNormInfo,
	                             denseMlpInfo,
	                             attnOutPath,
	                             attnOutData,
                                 attnOutBuffer,
	                             outputPath,
	                             outputDataOut,
                                 outputBufferOut,
                             rmsNormEps,
                             0,
                             0.0,
                             allowAsyncDenseSubmit,
                             pendingDenseCommandOut,
                             &stats->dense_mlp)) {
        return 0;
    }
    stats->output0 = stats->dense_mlp.output0;
    stats->elapsed_seconds = now_seconds() - totalStarted;
    stats->ok = 1;
    return 1;
}

static ExpertFile *find_expert_file_for_layer(ExpertFile *expertFiles,
                                              NSUInteger expertFileCount,
                                              int layerId) {
    for (NSUInteger i = 0; i < expertFileCount; i++) {
        if (expertFiles[i].layer == layerId) {
            return &expertFiles[i];
        }
    }
    return NULL;
}

static NSDictionary *find_expert_layer_dict(NSArray *layers, int layerId) {
    for (id item in layers) {
        if (![item isKindOfClass:[NSDictionary class]]) {
            continue;
        }
        NSDictionary *layer = (NSDictionary *)item;
        uint64_t current = unsigned_number(layer[@"layer"], "expert layer");
        if (current == (uint64_t)layerId) {
            return layer;
        }
    }
    return nil;
}

static int parse_attention_plan(NSDictionary *residentLayout,
                                NSString *cacheLayoutPath,
                                int layerId,
                                uint32_t numHeads,
                                uint32_t qkNopeDim,
                                uint32_t ropeDim,
                                uint32_t vHeadDim,
                                uint32_t contextLength,
                                uint64_t maxCacheFileBytes,
                                uint64_t maxCacheReadBytes,
                                DecodeLayerPlan *plan) {
    NSString *qAName = layer_tensor_name(layerId, @".self_attn.q_a_proj.weight");
    NSString *qBName = layer_tensor_name(layerId, @".self_attn.q_b_proj.weight");
    NSString *kvAName = layer_tensor_name(layerId,
                                          @".self_attn.kv_a_proj_with_mqa.weight");
    NSString *kvBName = layer_tensor_name(layerId, @".self_attn.kv_b_proj.weight");
    NSString *oProjName = layer_tensor_name(layerId, @".self_attn.o_proj.weight");
    if (!find_layer_vector_info(residentLayout,
                                layerId,
                                @".input_layernorm.weight",
                                &plan->input_norm) ||
        !find_layer_vector_info(residentLayout,
                                layerId,
                                @".self_attn.q_a_layernorm.weight",
                                &plan->q_a_norm) ||
        !find_layer_vector_info(residentLayout,
                                layerId,
                                @".self_attn.kv_a_layernorm.weight",
                                &plan->kv_a_norm) ||
        !find_resident_mxfp4_matrix_info(residentLayout, qAName, &plan->q_a) ||
        !find_resident_mxfp4_matrix_info(residentLayout, qBName, &plan->q_b) ||
        !find_resident_mxfp4_matrix_info(residentLayout, kvAName, &plan->kv_a) ||
        !find_resident_mxfp4_matrix_info(residentLayout, oProjName, &plan->attn_output)) {
        return 0;
    }
    if (plan->q_a.in_dim != plan->kv_a.in_dim ||
        plan->input_norm.dim != plan->q_a.in_dim ||
        plan->q_a_norm.dim != plan->q_a.out_dim ||
        plan->q_b.in_dim != plan->q_a.out_dim ||
        plan->kv_a_norm.dim > plan->kv_a.out_dim) {
        fprintf(stderr, "ERROR: decode layer %d attention dimensions are inconsistent\n", layerId);
        return 0;
    }
    if (resident_tensor_exists(residentLayout, kvBName)) {
        if (!find_resident_mxfp4_matrix_info(residentLayout, kvBName, &plan->kv_b)) {
            return 0;
        }
        if (plan->kv_b.in_dim != plan->kv_a_norm.dim) {
            fprintf(stderr,
                    "ERROR: decode layer %d kv_b input dim does not match kv_a norm dim\n",
                    layerId);
            return 0;
        }
        plan->has_kv_b = 1;
    }
    if (!resolve_mla_attention_value_source(residentLayout,
                                            layerId,
                                            numHeads,
                                            plan->kv_a_norm.dim,
                                            qkNopeDim,
                                            vHeadDim,
                                            &plan->mla_value_source)) {
        return 0;
    }
    uint32_t cacheWidth = plan->mla_value_source.kv_lora_dim + ropeDim;
    uint64_t mlaRawCacheBytes = 0;
    uint64_t mlaCacheF32Bytes = 0;
    if (!mla_attention_cache_byte_counts(cacheLayoutPath,
                                         (uint64_t)layerId,
                                         contextLength,
                                         cacheWidth,
                                         maxCacheFileBytes,
                                         maxCacheReadBytes,
                                         &mlaRawCacheBytes,
                                         &mlaCacheF32Bytes)) {
        return 0;
    }
    uint64_t qNopeBytes = (uint64_t)numHeads * (uint64_t)qkNopeDim * sizeof(float);
    uint64_t qRopeBytes = (uint64_t)numHeads * (uint64_t)ropeDim * sizeof(float);
    uint64_t outputBytes = (uint64_t)numHeads * (uint64_t)vHeadDim * sizeof(float);
    uint64_t qBBytes = (uint64_t)plan->q_b.out_dim * sizeof(float);
    uint64_t kRopeBytes = (uint64_t)ropeDim * sizeof(float);
    plan->attn_projection_scratch_bytes =
        (uint64_t)plan->input_norm.dim * sizeof(float) * 3u +
        plan->input_norm.size +
        (uint64_t)plan->q_a_norm.dim * sizeof(float) * 3u +
        plan->q_a_norm.size +
        (uint64_t)plan->kv_a_norm.dim * sizeof(float) * 3u +
        plan->kv_a_norm.size +
        resident_linear_scratch_bytes(plan->q_a) +
        resident_linear_scratch_bytes(plan->q_b) +
        resident_linear_scratch_bytes(plan->kv_a) +
        (plan->has_kv_b ? resident_linear_scratch_bytes(plan->kv_b) : 0) +
        (uint64_t)plan->kv_a.out_dim * sizeof(float);
    plan->mla_attention_scratch_bytes =
        mlaRawCacheBytes +
        mlaCacheF32Bytes +
        plan->mla_value_source.storage_bytes +
        plan->mla_value_source.source_f32_bytes +
        plan->mla_value_source.kv_b_f32_bytes +
        qNopeBytes +
        qRopeBytes +
        outputBytes;
    plan->attn_output_scratch_bytes =
        resident_linear_scratch_bytes(plan->attn_output) +
        (uint64_t)plan->attn_output.out_dim * sizeof(float);
    plan->hot_intermediate_memory_bytes =
        qBBytes + kRopeBytes + qNopeBytes + qRopeBytes + outputBytes;
    return 1;
}

static int build_decode_layer_plans(NSDictionary *residentLayout,
                                    NSDictionary *expertLayout,
                                    NSArray *expertLayers,
                                    ExpertFile *expertFiles,
                                    NSUInteger expertFileCount,
                                    IntList decodeLayers,
                                    LoaderOptions options,
                                    NSString *cacheLayoutPath,
                                    uint64_t maxCacheFileBytes,
                                    uint64_t maxCacheReadBytes,
                                    uint64_t ropeSplitScratchBytes,
                                    DecodeLayerPlan *plans,
                                    uint64_t *outMaxLayerScratchBytes,
                                    int *outHasMoe) {
    *outMaxLayerScratchBytes = 0;
    *outHasMoe = 0;
    for (int i = 0; i < decodeLayers.count; i++) {
        int layerId = decodeLayers.values[i];
        DecodeLayerPlan *plan = &plans[i];
        memset(plan, 0, sizeof(*plan));
        plan->layer = layerId;
        plan->rope_split_scratch_bytes = ropeSplitScratchBytes;
        if (!parse_attention_plan(residentLayout,
                                  cacheLayoutPath,
                                  layerId,
                                  (uint32_t)options.num_heads,
                                  (uint32_t)options.qk_nope_dim,
                                  (uint32_t)options.rope_dim,
                                  (uint32_t)options.v_head_dim,
                                  (uint32_t)options.context_length,
                                  maxCacheFileBytes,
                                  maxCacheReadBytes,
                                  plan)) {
            return 0;
        }
        plan->expert_file = find_expert_file_for_layer(expertFiles,
                                                       expertFileCount,
                                                       layerId);
        plan->is_dense = plan->expert_file ? 0 : 1;
        if (plan->is_dense) {
            if (!parse_dense_mlp_mxfp4_info(residentLayout,
                                            layerId,
                                            &plan->dense_mlp) ||
                !find_layer_vector_info(residentLayout,
                                        layerId,
                                        @".post_attention_layernorm.weight",
                                        &plan->post_attn_norm)) {
                return 0;
            }
            if (plan->post_attn_norm.dim != plan->dense_mlp.hidden_dim) {
                fprintf(stderr,
                        "ERROR: dense decode layer %d RMSNorm dim %u does not match hidden dim %u\n",
                        layerId,
                        plan->post_attn_norm.dim,
                        plan->dense_mlp.hidden_dim);
                return 0;
            }
            plan->rmsnorm_scratch_bytes =
                (uint64_t)plan->post_attn_norm.dim * sizeof(float) * 3u +
                plan->post_attn_norm.size;
            plan->dense_mlp_scratch_bytes =
                dense_mlp_scratch_bytes(plan->dense_mlp, plan->post_attn_norm);
            plan->layer_scratch_bytes =
                plan->attn_projection_scratch_bytes +
                plan->rope_split_scratch_bytes +
                plan->mla_attention_scratch_bytes +
                plan->attn_output_scratch_bytes +
                plan->dense_mlp_scratch_bytes +
                plan->hot_intermediate_memory_bytes;
        } else {
            *outHasMoe = 1;
            if (!options.open_experts || plan->expert_file->fd < 0) {
                fprintf(stderr,
                        "ERROR: decode layer %d requires open expert files\n",
                        layerId);
                return 0;
            }
            NSDictionary *layerDict = find_expert_layer_dict(expertLayers, layerId);
            if (!layerDict ||
                !parse_mxfp4_expert_info(expertLayout, layerDict, &plan->moe) ||
                !find_layer_vector_info(residentLayout,
                                        layerId,
                                        @".post_attention_layernorm.weight",
                                        &plan->post_attn_norm) ||
                !find_router_weight_info(residentLayout, layerId, &plan->router) ||
                !resolve_router_topk_options(residentLayout,
                                             options,
                                             &plan->router_topk_options)) {
                return 0;
            }
            if (!options.ignore_router_bias &&
                !find_router_bias_info(residentLayout, layerId, &plan->router_bias)) {
                return 0;
            }
            if (plan->post_attn_norm.dim != plan->router.hidden_dim ||
                plan->post_attn_norm.dim != plan->moe.hidden_dim ||
                plan->router.num_experts != plan->expert_file->num_experts) {
                fprintf(stderr,
                        "ERROR: decode layer %d router/expert/RMSNorm dims are inconsistent\n",
                        layerId);
                return 0;
            }
            plan->rmsnorm_scratch_bytes =
                (uint64_t)plan->post_attn_norm.dim * sizeof(float) * 3u +
                plan->post_attn_norm.size;
            plan->router_scratch_bytes =
                round_up_u64(plan->router.size, 2 * 1024 * 1024) +
                (uint64_t)plan->router.hidden_dim * sizeof(float) +
                (uint64_t)plan->router.num_experts * sizeof(float) +
                (uint64_t)plan->router.num_experts * sizeof(float);
            if (plan->router_bias.present) {
                plan->router_scratch_bytes += plan->router_bias.size;
            }
            plan->moe_scratch_bytes =
                (uint64_t)plan->moe.hidden_dim * sizeof(float) * 2u +
                (uint64_t)plan->moe.intermediate_dim * sizeof(float);
            if (options.include_shared_expert) {
                if (!parse_shared_mxfp4_info(residentLayout, layerId, &plan->shared)) {
                    return 0;
                }
                if (plan->shared.local.hidden_dim != plan->moe.hidden_dim ||
                    plan->shared.local.intermediate_dim != plan->moe.intermediate_dim ||
                    plan->shared.local.group_size != plan->moe.group_size) {
                    fprintf(stderr,
                            "ERROR: decode layer %d shared expert dims do not match routed expert dims\n",
                            layerId);
                    return 0;
                }
                plan->shared_storage_bytes = plan->shared.total_bytes;
            }
            plan->layer_scratch_bytes =
                plan->attn_projection_scratch_bytes +
                plan->rope_split_scratch_bytes +
                plan->mla_attention_scratch_bytes +
                plan->attn_output_scratch_bytes +
                plan->rmsnorm_scratch_bytes +
                plan->router_scratch_bytes +
                plan->moe_scratch_bytes +
                plan->hot_intermediate_memory_bytes;
        }
        if (plan->layer_scratch_bytes > *outMaxLayerScratchBytes) {
            *outMaxLayerScratchBytes = plan->layer_scratch_bytes;
        }
    }
    if (decodeLayers.count > 0) {
        uint64_t chainHiddenBytes = 0;
        if (!checked_mul_u64((uint64_t)plans[0].input_norm.dim,
                             sizeof(float),
                             &chainHiddenBytes) ||
            !checked_add_u64(*outMaxLayerScratchBytes,
                             chainHiddenBytes,
                             outMaxLayerScratchBytes)) {
            fprintf(stderr, "ERROR: decode layer memory-chain scratch bytes overflow\n");
            return 0;
        }
    }
    return 1;
}

static int run_decode_layers_probe(id<MTLDevice> device,
                                   NSString *residentBinPath,
                                   DecodeLayerPlan *plans,
                                   int planCount,
                                   NSString *inputPath,
                                   NSData *inputDataOverride,
                                   NSString *outputPath,
                                   NSString *workRoot,
                                   NSString *cacheLayoutPath,
                                   NSString *cacheFilePath,
                                   NSString *context1OProjCacheLayoutPath,
                                   NSString *context1OProjCacheFilePath,
                                   uint64_t position,
                                   uint32_t contextLength,
                                   uint32_t numHeads,
                                   uint32_t qkNopeDim,
                                   uint32_t ropeDim,
                                   uint32_t vHeadDim,
                                   uint32_t cachePositionOffset,
                                   float attentionScale,
                                   float ropeTheta,
                                   int ropeInterleave,
                                   uint64_t maxCacheFileBytes,
                                   uint64_t maxCacheReadBytes,
                                   uint32_t topK,
                                   float rmsNormEps,
                                   NSArray *expertBuffers,
                                   id<MTLBuffer> residentMetalBuffer,
                                   int writeDebugFiles,
                                   DecodeLayerRunSummary *summaries,
                                   NSData **finalOutputDataOut,
                                   float *output0,
                                   double *elapsedSeconds) {
    if (!ensure_output_dir(workRoot)) {
        return 0;
    }
    double totalStarted = now_seconds();
    NSString *currentInputPath = [inputPath copy];
    NSData *currentInputData = inputDataOverride ? [inputDataOverride copy] : nil;
    id<MTLBuffer> currentInputBuffer = nil;
    id<MTLCommandBuffer> pendingLayerCommand = nil;
    NSData *finalOutputData = nil;
    for (int i = 0; i < planCount; i++) {
        DecodeLayerPlan *plan = &plans[i];
        DecodeLayerRunSummary *summary = &summaries[i];
        memset(summary, 0, sizeof(*summary));
        summary->layer = plan->layer;
        summary->is_dense = plan->is_dense;
        summary->input_from_memory = currentInputData ? 1 : 0;
        summary->input_buffer_direct = currentInputBuffer ? 1 : 0;
        summary->scratch_bytes = plan->layer_scratch_bytes;
        summary->hot_intermediate_memory_bytes = plan->hot_intermediate_memory_bytes;
        summary->hot_intermediate_tensors = plan->hot_intermediate_memory_bytes ? 5 : 0;
        NSString *nextInputPath = nil;
        NSData *nextInputData = nil;
        id<MTLBuffer> nextInputBuffer = nil;
        id<MTLCommandBuffer> layerPendingCommand = nil;
        @autoreleasepool {
            NSString *layerName = [NSString stringWithFormat:@"layer_%d", plan->layer];
            NSString *layerWorkDir = [workRoot stringByAppendingPathComponent:layerName];
            NSString *layerOutputPath = (i + 1 == planCount)
                ? outputPath
                : (writeDebugFiles
                    ? [workRoot stringByAppendingPathComponent:
                        [NSString stringWithFormat:@"%@_out.f32", layerName]]
                    : nil);
            NSData *layerOutputData = nil;
            id<MTLBuffer> layerOutputBuffer = nil;
            int wantLayerOutputData = writeDebugFiles || (i + 1 == planCount);
            if (plan->is_dense) {
                DenseDecoderLayerProbeStats denseStats = {0};
                if (!run_dense_decoder_layer_probe(device,
                                                   residentBinPath,
                                                   plan->layer,
                                                   plan->input_norm,
                                                   plan->q_a_norm,
                                                   plan->kv_a_norm,
                                                   plan->post_attn_norm,
                                                   plan->q_a,
                                                   plan->q_b,
                                                   plan->kv_a,
                                                   plan->has_kv_b,
                                                   plan->kv_b,
                                                   plan->mla_value_source,
                                                   plan->attn_output,
                                                   plan->dense_mlp,
                                                   currentInputPath,
                                                   currentInputData,
                                                   currentInputBuffer,
                                                   layerWorkDir,
                                                   cacheLayoutPath,
                                                   cacheFilePath,
                                                   context1OProjCacheLayoutPath,
                                                   context1OProjCacheFilePath,
                                                   position,
                                                   contextLength,
                                                   numHeads,
                                                   qkNopeDim,
                                                   ropeDim,
                                                   vHeadDim,
                                                   cachePositionOffset,
                                                   attentionScale,
                                                   ropeTheta,
                                                   ropeInterleave,
                                                   maxCacheFileBytes,
                                                   maxCacheReadBytes,
                                                   rmsNormEps,
                                                   layerOutputPath,
                                                   wantLayerOutputData ? &layerOutputData : nil,
                                                   &layerOutputBuffer,
                                                   writeDebugFiles,
                                                   residentMetalBuffer,
                                                   (i + 1 < planCount) && !writeDebugFiles,
                                                   &layerPendingCommand,
                                                   &denseStats)) {
                    return 0;
                }
                summary->ok = 1;
                summary->elapsed_seconds = denseStats.elapsed_seconds;
                summary->attn_projection_elapsed_seconds =
                    denseStats.attn_projection.elapsed_seconds;
                summary->attn_projection_fused_pre_cache =
                    denseStats.attn_projection.fused_pre_cache;
                summary->attn_projection_command_buffer_count =
                    denseStats.attn_projection.command_buffer_count;
                summary->attn_projection_synchronous_wait_count =
                    denseStats.attn_projection.synchronous_wait_count;
                summary->attn_projection_async_submitted =
                    denseStats.attn_projection.async_submitted;
                summary->mla_attention_elapsed_seconds =
                    denseStats.mla_attention.elapsed_seconds;
                summary->mla_attention_cache_read_seconds =
                    denseStats.mla_attention.cache_read_seconds;
                summary->mla_attention_value_read_seconds =
                    denseStats.mla_attention.value_read_seconds;
                summary->mla_attention_kernel_seconds =
                    denseStats.mla_attention.kernel_seconds;
                summary->mla_attention_output_write_seconds =
                    denseStats.mla_attention.output_write_seconds;
                summary->mla_value_cache_hit_count =
                    denseStats.mla_attention.value_cache_hit ? 1u : 0u;
                summary->mla_value_cache_store_count =
                    denseStats.mla_attention.value_cache_stored ? 1u : 0u;
                summary->mla_value_cache_bytes =
                    denseStats.mla_attention.value_cache_bytes;
                summary->mla_value_cache_total_bytes =
                    denseStats.mla_attention.value_cache_total_bytes;
                summary->attn_output_elapsed_seconds =
                    denseStats.attn_output.elapsed_seconds;
                summary->attn_output_bytes_read =
                    denseStats.attn_output.bytes_read;
                summary->attn_output_read_seconds =
                    denseStats.attn_output.read_seconds;
                summary->attn_output_projection_kernel_seconds =
                    denseStats.attn_output.projection_kernel_seconds;
                summary->mlp_elapsed_seconds = denseStats.dense_mlp.elapsed_seconds;
                summary->rope_mla_fused =
                    denseStats.rope_split.fused_with_mla &&
                    denseStats.mla_attention.fused_with_rope;
                summary->rope_mla_input_buffer_direct =
                    denseStats.rope_split.input_buffer_direct;
                summary->rope_mla_command_buffer_count =
                    denseStats.rope_split.command_buffer_count +
                    denseStats.mla_attention.command_buffer_count;
                summary->attn_output_fused_matvec_add =
                    denseStats.attn_output.fused_matvec_add;
                summary->attn_output_context1_o_proj_cache =
                    denseStats.attn_output.context1_o_proj_cache;
                summary->attn_output_resident_mmap_backed =
                    denseStats.attn_output.resident_mmap_backed;
                summary->attn_output_command_buffer_count =
                    denseStats.attn_output.command_buffer_count;
                summary->dense_mlp_bytes_read = denseStats.dense_mlp.bytes_read;
                summary->dense_mlp_fused_pipeline =
                    denseStats.dense_mlp.fused_pipeline;
                summary->dense_mlp_command_buffer_count =
                    denseStats.dense_mlp.command_buffer_count;
                summary->dense_mlp_synchronous_wait_count =
                    denseStats.dense_mlp.synchronous_wait_count;
                summary->dense_mlp_async_submitted =
                    denseStats.dense_mlp.async_submitted;
                summary->output0 = denseStats.output0;
            } else {
                DecoderLayerProbeStats moeStats = {0};
                if (!run_decoder_layer_probe(device,
                                             residentBinPath,
                                             plan->layer,
                                             plan->input_norm,
                                             plan->q_a_norm,
                                             plan->kv_a_norm,
                                             plan->post_attn_norm,
                                             plan->q_a,
                                             plan->q_b,
                                             plan->kv_a,
                                             plan->has_kv_b,
                                             plan->kv_b,
                                             plan->mla_value_source,
                                             plan->attn_output,
                                             plan->router,
                                             plan->router_bias,
                                             plan->router_topk_options,
                                             plan->expert_file,
                                             plan->moe,
                                             plan->shared_storage_bytes > 0 ? &plan->shared : NULL,
                                             currentInputPath,
                                             currentInputData,
                                             currentInputBuffer,
                                             layerWorkDir,
                                             cacheLayoutPath,
                                             cacheFilePath,
                                             context1OProjCacheLayoutPath,
                                             context1OProjCacheFilePath,
                                             position,
                                             contextLength,
                                             numHeads,
                                             qkNopeDim,
                                             ropeDim,
                                             vHeadDim,
                                             cachePositionOffset,
                                             attentionScale,
                                             ropeTheta,
                                             ropeInterleave,
                                             maxCacheFileBytes,
                                             maxCacheReadBytes,
                                             topK,
                                             rmsNormEps,
	                                             nil,
	                                             layerOutputPath,
	                                             wantLayerOutputData ? &layerOutputData : nil,
                                             &layerOutputBuffer,
                                             writeDebugFiles,
                                             expertBuffers,
                                             residentMetalBuffer,
                                             (i + 1 < planCount) && !writeDebugFiles,
                                             &layerPendingCommand,
                                             &moeStats)) {
	                    free(moeStats.router.logits);
	                    return 0;
	                }
                summary->ok = 1;
                summary->elapsed_seconds = moeStats.elapsed_seconds;
                summary->attn_projection_elapsed_seconds =
                    moeStats.attn_projection.elapsed_seconds;
                summary->attn_projection_fused_pre_cache =
                    moeStats.attn_projection.fused_pre_cache;
                summary->attn_projection_command_buffer_count =
                    moeStats.attn_projection.command_buffer_count;
                summary->attn_projection_synchronous_wait_count =
                    moeStats.attn_projection.synchronous_wait_count;
                summary->attn_projection_async_submitted =
                    moeStats.attn_projection.async_submitted;
                summary->mla_attention_elapsed_seconds =
                    moeStats.mla_attention.elapsed_seconds;
                summary->mla_attention_cache_read_seconds =
                    moeStats.mla_attention.cache_read_seconds;
                summary->mla_attention_value_read_seconds =
                    moeStats.mla_attention.value_read_seconds;
                summary->mla_attention_kernel_seconds =
                    moeStats.mla_attention.kernel_seconds;
                summary->mla_attention_output_write_seconds =
                    moeStats.mla_attention.output_write_seconds;
                summary->mla_value_cache_hit_count =
                    moeStats.mla_attention.value_cache_hit ? 1u : 0u;
                summary->mla_value_cache_store_count =
                    moeStats.mla_attention.value_cache_stored ? 1u : 0u;
                summary->mla_value_cache_bytes =
                    moeStats.mla_attention.value_cache_bytes;
                summary->mla_value_cache_total_bytes =
                    moeStats.mla_attention.value_cache_total_bytes;
                summary->attn_output_elapsed_seconds =
                    moeStats.attn_output.elapsed_seconds;
                summary->attn_output_bytes_read =
                    moeStats.attn_output.bytes_read;
                summary->attn_output_read_seconds =
                    moeStats.attn_output.read_seconds;
                summary->attn_output_projection_kernel_seconds =
                    moeStats.attn_output.projection_kernel_seconds;
                summary->post_attn_norm_weight_bytes_read =
                    moeStats.rms_norm.weight_bytes_read;
                summary->post_attn_norm_weight_read_seconds =
                    moeStats.rms_norm.weight_read_seconds;
                summary->router_bytes_read =
                    moeStats.router.router_bytes_read;
                summary->router_correction_bias_bytes_read =
                    moeStats.router.correction_bias_bytes_read;
                summary->router_read_seconds =
                    moeStats.router.router_read_seconds;
                summary->router_kernel_seconds =
                    moeStats.router.kernel_seconds;
                summary->mlp_elapsed_seconds = moeStats.mlp.elapsed_seconds;
                summary->rope_mla_fused =
                    moeStats.rope_split.fused_with_mla &&
                    moeStats.mla_attention.fused_with_rope;
                summary->rope_mla_input_buffer_direct =
                    moeStats.rope_split.input_buffer_direct;
                summary->rope_mla_command_buffer_count =
                    moeStats.rope_split.command_buffer_count +
                    moeStats.mla_attention.command_buffer_count;
                summary->attn_output_fused_matvec_add =
                    moeStats.attn_output.fused_matvec_add;
                summary->attn_output_context1_o_proj_cache =
                    moeStats.attn_output.context1_o_proj_cache;
                summary->attn_output_resident_mmap_backed =
                    moeStats.attn_output.resident_mmap_backed;
                summary->attn_output_norm_router_fused =
                    moeStats.attn_output.fused_with_post_attn_norm_router;
                summary->rope_mla_attn_output_norm_router_fused =
                    moeStats.attn_output.fused_with_rope_mla;
                summary->attn_output_command_buffer_count =
                    moeStats.attn_output.command_buffer_count;
                summary->post_attn_norm_router_fused =
                    moeStats.rms_norm.fused_with_router &&
                    moeStats.router.fused_with_rmsnorm;
                summary->attn_output_buffer_direct =
                    moeStats.rms_norm.input_buffer_direct;
                summary->post_attn_norm_command_buffer_count =
                    moeStats.rms_norm.command_buffer_count;
                summary->router_command_buffer_count =
                    moeStats.router.command_buffer_count;
                summary->expert_bytes_read = moeStats.mlp.expert_bytes_read;
                summary->shared_bytes_read = moeStats.mlp.shared_bytes_read;
                summary->shared_read_seconds = moeStats.mlp.shared_read_seconds;
                summary->shared_prefetch_seconds =
                    moeStats.mlp.shared_prefetch_seconds;
                summary->shared_prefetch_used =
                    moeStats.mlp.shared_prefetch_used;
                summary->expert_read_dispatch_count =
                    moeStats.mlp.expert_read_dispatch_count;
                summary->expert_read_task_count =
                    moeStats.mlp.expert_read_task_count;
                summary->expert_read_max_task_count =
                    moeStats.mlp.expert_read_max_task_count;
                summary->expert_read_max_worker_count =
                    moeStats.mlp.expert_read_max_worker_count;
                summary->expert_read_pool_dispatch_count =
                    moeStats.mlp.expert_read_pool_dispatch_count;
                summary->expert_read_serial_dispatch_count =
                    moeStats.mlp.expert_read_serial_dispatch_count;
                summary->expert_read_seconds = moeStats.mlp.expert_read_seconds;
                summary->moe_mlp_kernel_seconds = moeStats.mlp.kernel_seconds;
                summary->moe_mlp_output_write_seconds =
                    moeStats.mlp.output_write_seconds;
                summary->moe_mlp_overhead_seconds =
                    moeStats.mlp.elapsed_seconds -
                    moeStats.mlp.expert_read_seconds -
                    moeStats.mlp.kernel_seconds -
                    moeStats.mlp.output_write_seconds;
                if (summary->moe_mlp_overhead_seconds < 0.0) {
                    summary->moe_mlp_overhead_seconds = 0.0;
                }
                summary->router_gpu_topk = moeStats.router.gpu_topk;
                summary->router_top_k = moeStats.router.top_k;
                for (uint32_t route = 0; route < moeStats.router.top_k; route++) {
                    summary->router_experts[route] = moeStats.router.experts[route];
                }
                summary->moe_mlp_residual_add_fused =
                    moeStats.mlp.residual_add_fused;
                summary->moe_mlp_input_buffer_direct =
                    moeStats.mlp.input_buffer_direct;
                summary->moe_mlp_command_buffer_count =
                    moeStats.mlp.command_buffer_count;
                summary->moe_mlp_synchronous_wait_count =
                    moeStats.mlp.synchronous_wait_count;
                summary->output0 = moeStats.output0;
                free(moeStats.router.logits);
            }
            summary->synchronous_wait_count =
                summary->attn_projection_synchronous_wait_count +
                summary->rope_mla_command_buffer_count +
                summary->attn_output_command_buffer_count +
                summary->post_attn_norm_command_buffer_count +
                summary->router_command_buffer_count +
                summary->post_attn_norm_router_command_buffer_count +
                summary->dense_mlp_synchronous_wait_count +
                summary->moe_mlp_synchronous_wait_count;
            if (!layerOutputData && !layerOutputBuffer) {
                fprintf(stderr,
                        "ERROR: decode layer %d did not return an in-memory output\n",
                        plan->layer);
                return 0;
            }
            nextInputPath = [layerOutputPath copy];
            nextInputData = layerOutputData;
            nextInputBuffer = layerOutputBuffer;
        }
        if (pendingLayerCommand &&
            !check_completed_command_buffer(pendingLayerCommand, "deferred layer output")) {
            return 0;
        }
        pendingLayerCommand = layerPendingCommand;
        currentInputPath = nextInputPath;
        currentInputData = nextInputData;
        currentInputBuffer = nextInputBuffer;
        finalOutputData = currentInputData;
    }
    if (pendingLayerCommand &&
        !check_completed_command_buffer(pendingLayerCommand, "deferred layer output")) {
        return 0;
    }
    if (finalOutputDataOut) {
        *finalOutputDataOut = finalOutputData;
    }
    *elapsedSeconds = now_seconds() - totalStarted;
    if (planCount > 0) {
        *output0 = summaries[planCount - 1].output0;
    }
    return 1;
}

static void update_final_topk(float *scores,
                              uint64_t *ids,
                              uint32_t *count,
                              uint32_t topK,
                              uint64_t tokenId,
                              float score) {
    if (*count < topK) {
        scores[*count] = score;
        ids[*count] = tokenId;
        (*count)++;
        return;
    }
    uint32_t minIdx = 0;
    for (uint32_t i = 1; i < *count; i++) {
        if (scores[i] < scores[minIdx] ||
            (scores[i] == scores[minIdx] && ids[i] > ids[minIdx])) {
            minIdx = i;
        }
    }
    if (score > scores[minIdx] ||
        (score == scores[minIdx] && tokenId < ids[minIdx])) {
        scores[minIdx] = score;
        ids[minIdx] = tokenId;
    }
}

static void sort_final_topk(float *scores, uint64_t *ids, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        for (uint32_t j = i + 1; j < count; j++) {
            if (scores[j] > scores[i] ||
                (scores[j] == scores[i] && ids[j] < ids[i])) {
                float score = scores[i];
                scores[i] = scores[j];
                scores[j] = score;
                uint64_t id = ids[i];
                ids[i] = ids[j];
                ids[j] = id;
            }
        }
    }
}

static NSMutableArray *final_topk_array(FinalLogitsProbeStats stats) {
    NSMutableArray *items = [NSMutableArray arrayWithCapacity:stats.count];
    for (uint32_t i = 0; i < stats.count; i++) {
        [items addObject:@{
            @"token_id": @(stats.ids[i]),
            @"logit": @(stats.scores[i]),
        }];
    }
    return items;
}

static int write_final_logits_topk_json(NSString *path,
                                        FinalLogitsProbeStats stats) {
    NSDictionary *payload = @{
        @"topk": final_topk_array(stats),
        @"elapsed_seconds": @(stats.elapsed_seconds),
        @"read_bytes": @(stats.bytes_read),
        @"chunk_rows": @(stats.chunk_rows),
        @"chunks": @(stats.chunks),
    };
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:&error];
    if (!data) {
        fprintf(stderr,
                "ERROR: failed to serialize final logits top-k JSON: %s\n",
                error ? [[error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    if (![data writeToFile:path atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write final logits top-k JSON %s\n",
                [path UTF8String]);
        return 0;
    }
    return 1;
}

static NSDictionary *final_generated_token_payload(FinalLogitsProbeStats stats) {
    if (stats.count == 0) {
        return nil;
    }
    return @{
        @"token_id": @(stats.ids[0]),
        @"logit": @(stats.scores[0]),
        @"selection": @"argmax",
        @"top_k": @(stats.top_k),
        @"elapsed_seconds": @(stats.elapsed_seconds),
    };
}

static int write_final_logits_token_json(NSString *path,
                                         FinalLogitsProbeStats stats) {
    NSDictionary *payload = final_generated_token_payload(stats);
    if (!payload) {
        fprintf(stderr, "ERROR: final logits token JSON requested with empty top-k\n");
        return 0;
    }
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:&error];
    if (!data) {
        fprintf(stderr,
                "ERROR: failed to serialize generated token JSON: %s\n",
                error ? [[error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    if (![data writeToFile:path atomically:YES]) {
        fprintf(stderr, "ERROR: failed to write generated token JSON %s\n",
                [path UTF8String]);
        return 0;
    }
    return 1;
}

static int write_generated_embedding_f32(NSString *residentBinPath,
                                         ResidentMxfp4MatrixInfo embedding,
                                         uint64_t tokenId,
                                         uint64_t maxRowBytes,
                                         NSString *outputPath,
                                         NSData **outputDataOut,
                                         EmbeddingLookupStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (outputDataOut) {
        *outputDataOut = nil;
    }
    double started = now_seconds();
    if (tokenId >= embedding.out_dim) {
        fprintf(stderr,
                "ERROR: generated token %llu exceeds embedding vocab size %u\n",
                (unsigned long long)tokenId,
                embedding.out_dim);
        return 0;
    }
    uint64_t weightRowBytes = (uint64_t)embedding.weight.dim1 * sizeof(uint32_t);
    uint64_t scaleRowBytes = (uint64_t)embedding.scales.dim1;
    uint64_t rowBytes = 0;
    uint64_t outputBytes = (uint64_t)embedding.in_dim * sizeof(float);
    if (!checked_add_u64(weightRowBytes, scaleRowBytes, &rowBytes) ||
        rowBytes == 0 ||
        rowBytes > maxRowBytes ||
        weightRowBytes > (uint64_t)SIZE_MAX ||
        scaleRowBytes > (uint64_t)SIZE_MAX ||
        outputBytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr,
                "ERROR: generated-token embedding row exceeds configured limits\n");
        return 0;
    }
    uint32_t *rawWeights = (uint32_t *)malloc((size_t)weightRowBytes);
    uint8_t *rawScales = (uint8_t *)malloc((size_t)scaleRowBytes);
    float *values = (float *)malloc((size_t)outputBytes);
    if (!rawWeights || !rawScales || !values) {
        fprintf(stderr, "ERROR: failed to allocate generated-token embedding buffers\n");
        free(rawWeights);
        free(rawScales);
        free(values);
        return 0;
    }
    int closeFd = 0;
    int fd = open_resident_read_fd(residentBinPath, &closeFd);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open resident file for generated-token embedding: %s\n",
                strerror(errno));
        free(rawWeights);
        free(rawScales);
        free(values);
        return 0;
    }
    uint64_t weightOffset =
        embedding.weight.offset + tokenId * weightRowBytes;
    uint64_t scaleOffset =
        embedding.scales.offset + tokenId * scaleRowBytes;
    double readStarted = now_seconds();
    int readOk =
        pread_exact_or_report(fd,
                              rawWeights,
                              weightRowBytes,
                              weightOffset,
                              [residentBinPath UTF8String]) &&
        pread_exact_or_report(fd,
                              rawScales,
                              scaleRowBytes,
                              scaleOffset,
                              [residentBinPath UTF8String]);
    close_resident_read_fd(fd, closeFd);
    stats->read_seconds = now_seconds() - readStarted;
    if (!readOk) {
        free(rawWeights);
        free(rawScales);
        free(values);
        return 0;
    }
    double decodeStarted = now_seconds();
    uint32_t packedCols = embedding.weight.dim1;
    uint32_t groups = embedding.scales.dim1;
    for (uint32_t p = 0; p < packedCols; p++) {
        uint32_t packed = rawWeights[p];
        for (uint32_t i = 0; i < 8u; i++) {
            uint32_t logicalCol = p * 8u + i;
            uint32_t group = logicalCol / embedding.group_size;
            if (group >= groups) {
                fprintf(stderr, "ERROR: generated-token embedding group index overflow\n");
                free(rawWeights);
                free(rawScales);
                free(values);
                return 0;
            }
            float scale = mxfp4_e8m0_to_f32_cpu(rawScales[group]);
            values[logicalCol] =
                mxfp4_e2m1_to_f32_cpu((packed >> (i * 4u)) & 0xFu) * scale;
        }
    }
    stats->decode_seconds = now_seconds() - decodeStarted;
    NSData *data = [NSData dataWithBytes:values length:(NSUInteger)outputBytes];
    if (outputDataOut) {
        *outputDataOut = data;
    }
    if (outputPath) {
        double writeStarted = now_seconds();
        int writeOk = [data writeToFile:outputPath atomically:YES] ? 1 : 0;
        stats->write_seconds = now_seconds() - writeStarted;
        if (!writeOk) {
            fprintf(stderr,
                    "ERROR: failed to write generated-token embedding %s\n",
                    [outputPath UTF8String]);
            free(rawWeights);
            free(rawScales);
            free(values);
            return 0;
        }
    }
    stats->ok = 1;
    stats->token_id = tokenId;
    stats->bytes_read = rowBytes;
    stats->output_bytes = outputBytes;
    stats->elapsed_seconds = now_seconds() - started;
    stats->vocab_size = embedding.out_dim;
    stats->hidden_dim = embedding.in_dim;
    stats->group_size = embedding.group_size;
    stats->output0 = embedding.in_dim > 0 ? values[0] : 0.0f;
    free(rawWeights);
    free(rawScales);
    free(values);
    return 1;
}

static id<MTLBuffer> new_mapped_readonly_file_buffer(id<MTLDevice> device,
                                                     NSString *path,
                                                     uint64_t offset,
                                                     uint64_t size,
                                                     NSUInteger *bufferOffsetOut) {
    if (bufferOffsetOut) {
        *bufferOffsetOut = 0;
    }
    if (size == 0 || offset > (uint64_t)NSUIntegerMax) {
        fprintf(stderr, "ERROR: mmap Metal buffer range is invalid\n");
        return nil;
    }
    long pageSizeLong = sysconf(_SC_PAGESIZE);
    uint64_t pageSize = pageSizeLong > 0 ? (uint64_t)pageSizeLong : 16384ull;
    uint64_t mapOffset = (offset / pageSize) * pageSize;
    uint64_t delta = offset - mapOffset;
    uint64_t mapLength = 0;
    if (!checked_add_u64(delta, size, &mapLength) ||
        mapLength > (uint64_t)NSUIntegerMax ||
        delta > (uint64_t)NSUIntegerMax ||
        mapOffset > (uint64_t)LLONG_MAX) {
        fprintf(stderr, "ERROR: mmap Metal buffer range overflows\n");
        return nil;
    }
    int fd = open([path fileSystemRepresentation], O_RDONLY);
    if (fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open %s for mmap Metal buffer: %s\n",
                [path UTF8String],
                strerror(errno));
        return nil;
    }
    void *ptr = mmap(NULL,
                     (size_t)mapLength,
                     PROT_READ,
                     MAP_PRIVATE,
                     fd,
                     (off_t)mapOffset);
    close(fd);
    if (ptr == MAP_FAILED) {
        fprintf(stderr,
                "ERROR: failed to mmap %s for Metal buffer: %s\n",
                [path UTF8String],
                strerror(errno));
        return nil;
    }
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:ptr
                                                     length:(NSUInteger)mapLength
                                                    options:MTLResourceStorageModeShared
                                                deallocator:^(void *pointer, NSUInteger length) {
                                                    if (pointer) {
                                                        munmap(pointer, length);
                                                    }
                                                }];
    if (!buffer) {
        munmap(ptr, (size_t)mapLength);
        fprintf(stderr, "ERROR: failed to wrap mmap range as Metal buffer\n");
        return nil;
    }
    if (bufferOffsetOut) {
        *bufferOffsetOut = (NSUInteger)delta;
    }
    return buffer;
}

static int run_final_logits_probe(id<MTLDevice> device,
                                  NSString *residentBinPath,
                                  ResidentMxfp4MatrixInfo headInfo,
                                  ResidentVectorInfo finalNormInfo,
                                  NSString *inputPath,
                                  NSData *inputDataOverride,
                                  id<MTLBuffer> residentMetalBuffer,
                                  int mmapFinalLogits,
                                  uint32_t topK,
                                  uint64_t chunkRowsArg,
                                  uint64_t maxChunkBytes,
                                  float rmsNormEps,
                                  int skipFinalNorm,
                                  NSString *outputTopKJson,
                                  NSString *outputTokenJson,
                                  FinalLogitsProbeStats *stats) {
    memset(stats, 0, sizeof(*stats));
    stats->hidden_dim = headInfo.in_dim;
    stats->vocab_size = headInfo.out_dim;
    stats->group_size = headInfo.group_size;
    stats->top_k = topK;
    stats->input_from_memory = inputDataOverride ? 1 : 0;
    if (topK == 0 || topK > 64 || topK > headInfo.out_dim) {
        fprintf(stderr, "ERROR: final logits top-k is invalid\n");
        return 0;
    }
    uint64_t hiddenBytes = (uint64_t)headInfo.in_dim * sizeof(float);
    NSData *normedData = nil;
    double totalStarted = now_seconds();
    if (skipFinalNorm) {
        normedData = inputDataOverride ?: [NSData dataWithContentsOfFile:inputPath];
        if (!normedData) {
            fprintf(stderr, "ERROR: failed to read final logits input %s\n",
                    inputPath ? [inputPath UTF8String] : "<memory>");
            return 0;
        }
    } else {
        RmsNormProbeStats normStats = {0};
        if (!run_rmsnorm_probe(device,
                               residentBinPath,
                               finalNormInfo,
                               inputPath,
                               inputDataOverride,
                               rmsNormEps,
                               &normedData,
                               &normStats)) {
            return 0;
        }
        stats->norm_elapsed_seconds = normStats.elapsed_seconds;
        stats->bytes_read += normStats.weight_bytes_read;
    }
    if ((uint64_t)[normedData length] != hiddenBytes) {
        fprintf(stderr,
                "ERROR: final logits input bytes %llu do not match hidden bytes %llu\n",
                (unsigned long long)[normedData length],
                (unsigned long long)hiddenBytes);
        return 0;
    }
    uint64_t weightRowBytes = (uint64_t)headInfo.weight.dim1 * sizeof(uint32_t);
    uint64_t scaleRowBytes = (uint64_t)headInfo.scales.dim1 * sizeof(uint8_t);
    uint64_t rowBytes = 0;
    if (!checked_add_u64(weightRowBytes, scaleRowBytes, &rowBytes) ||
        rowBytes == 0 ||
        rowBytes > maxChunkBytes) {
        fprintf(stderr,
                "ERROR: one final logits lm_head row %llu bytes exceeds chunk limit %llu\n",
                (unsigned long long)rowBytes,
                (unsigned long long)maxChunkBytes);
        return 0;
    }
    int useResidentMetalWeights = residentMetalBuffer != nil;
    int useMappedFinalLogits = mmapFinalLogits && !useResidentMetalWeights;
    int useStagedFinalLogits = !useResidentMetalWeights && !useMappedFinalLogits;
    id<MTLBuffer> mappedWeightBuffer = nil;
    id<MTLBuffer> mappedScalesBuffer = nil;
    NSUInteger mappedWeightOffset = 0;
    NSUInteger mappedScalesOffset = 0;
    if (useResidentMetalWeights) {
        uint64_t residentLength = (uint64_t)[residentMetalBuffer length];
        uint64_t weightEnd = 0;
        uint64_t scalesEnd = 0;
        if (!checked_add_u64(headInfo.weight.offset, headInfo.weight.size, &weightEnd) ||
            !checked_add_u64(headInfo.scales.offset, headInfo.scales.size, &scalesEnd) ||
            headInfo.weight.offset > (uint64_t)NSUIntegerMax ||
            headInfo.scales.offset > (uint64_t)NSUIntegerMax ||
            weightEnd > residentLength ||
            scalesEnd > residentLength) {
            fprintf(stderr,
                    "ERROR: resident Metal buffer does not cover final logits weights\n");
            return 0;
        }
        stats->resident_mmap_backed = 1;
    }
    if (useMappedFinalLogits) {
        mappedWeightBuffer =
            new_mapped_readonly_file_buffer(device,
                                            residentBinPath,
                                            headInfo.weight.offset,
                                            headInfo.weight.size,
                                            &mappedWeightOffset);
        mappedScalesBuffer =
            new_mapped_readonly_file_buffer(device,
                                            residentBinPath,
                                            headInfo.scales.offset,
                                            headInfo.scales.size,
                                            &mappedScalesOffset);
        if (!mappedWeightBuffer || !mappedScalesBuffer) {
            return 0;
        }
        stats->resident_mmap_backed = 1;
    }
    uint64_t rowsPerChunk = chunkRowsArg
        ? chunkRowsArg
        : ((useResidentMetalWeights || useMappedFinalLogits)
            ? headInfo.out_dim
            : (maxChunkBytes / rowBytes));
    if (rowsPerChunk == 0) {
        rowsPerChunk = 1;
    }
    if (rowsPerChunk > headInfo.out_dim) {
        rowsPerChunk = headInfo.out_dim;
    }
    stats->chunk_rows = rowsPerChunk;

    id<MTLLibrary> library = make_glm_moe_library(device);
    id<MTLComputePipelineState> matvec =
        library ? make_resident_mxfp4_matvec_pipeline(device, library, headInfo) : nil;
    id<MTLCommandQueue> queue = shared_glm_command_queue(device);
    id<MTLBuffer> input = [device newBufferWithBytes:[normedData bytes]
                                              length:(NSUInteger)hiddenBytes
                                             options:MTLResourceStorageModeShared];
    if (!library || !matvec || !queue || !input) {
        fprintf(stderr, "ERROR: failed to allocate final logits Metal resources\n");
        return 0;
    }
    int closeFd = 0;
    int fd = -1;
    if (useStagedFinalLogits) {
        fd = open_resident_read_fd(residentBinPath, &closeFd);
    }
    if (useStagedFinalLogits && fd < 0) {
        fprintf(stderr,
                "ERROR: failed to open resident file for final logits %s: %s\n",
                [residentBinPath UTF8String],
                strerror(errno));
        return 0;
    }
    for (uint64_t rowStart = 0; rowStart < headInfo.out_dim; rowStart += rowsPerChunk) {
        uint64_t rows = rowsPerChunk;
        if (rowStart + rows > headInfo.out_dim) {
            rows = headInfo.out_dim - rowStart;
        }
        uint64_t weightChunkBytes = 0;
        uint64_t scaleChunkBytes = 0;
        uint64_t rawBytes = 0;
        uint64_t outputBytes = rows * sizeof(float);
        if (!checked_mul_u64(rows, weightRowBytes, &weightChunkBytes) ||
            !checked_mul_u64(rows, scaleRowBytes, &scaleChunkBytes) ||
            !checked_add_u64(weightChunkBytes, scaleChunkBytes, &rawBytes)) {
            fprintf(stderr, "ERROR: final logits chunk byte size overflows\n");
            close_resident_read_fd(fd, closeFd);
            return 0;
        }
        uint64_t allocBytes = round_up_u64(rawBytes, 2 * 1024 * 1024);
        if (allocBytes > NSUIntegerMax || outputBytes > NSUIntegerMax) {
            fprintf(stderr, "ERROR: final logits chunk buffers exceed NSUIntegerMax\n");
            if (fd >= 0) {
                close_resident_read_fd(fd, closeFd);
            }
            return 0;
        }
        void *matrixPtr = NULL;
        id<MTLBuffer> matrix = nil;
        NSUInteger weightOffset = 0;
        NSUInteger scalesOffset = (NSUInteger)weightChunkBytes;
        id<MTLBuffer> weightBuffer = nil;
        id<MTLBuffer> scalesBuffer = nil;
        if (useResidentMetalWeights) {
            uint64_t weightChunkOffset = headInfo.weight.offset + rowStart * weightRowBytes;
            uint64_t scaleChunkOffset = headInfo.scales.offset + rowStart * scaleRowBytes;
            if (weightChunkOffset > (uint64_t)NSUIntegerMax ||
                scaleChunkOffset > (uint64_t)NSUIntegerMax) {
                fprintf(stderr, "ERROR: final logits resident chunk offset exceeds NSUIntegerMax\n");
                return 0;
            }
            matrix = residentMetalBuffer;
            weightBuffer = matrix;
            scalesBuffer = matrix;
            weightOffset = (NSUInteger)weightChunkOffset;
            scalesOffset = (NSUInteger)scaleChunkOffset;
        } else if (useMappedFinalLogits) {
            uint64_t weightChunkOffset = 0;
            uint64_t scaleChunkOffset = 0;
            if (!checked_add_u64((uint64_t)mappedWeightOffset,
                                 rowStart * weightRowBytes,
                                 &weightChunkOffset) ||
                !checked_add_u64((uint64_t)mappedScalesOffset,
                                 rowStart * scaleRowBytes,
                                 &scaleChunkOffset) ||
                weightChunkOffset > (uint64_t)NSUIntegerMax ||
                scaleChunkOffset > (uint64_t)NSUIntegerMax) {
                fprintf(stderr, "ERROR: final logits mmap chunk offset overflows\n");
                return 0;
            }
            weightBuffer = mappedWeightBuffer;
            scalesBuffer = mappedScalesBuffer;
            matrix = mappedWeightBuffer;
            weightOffset = (NSUInteger)weightChunkOffset;
            scalesOffset = (NSUInteger)scaleChunkOffset;
        } else {
            if (posix_memalign(&matrixPtr, 2 * 1024 * 1024, (size_t)allocBytes) != 0 ||
                !matrixPtr) {
                fprintf(stderr, "ERROR: failed to allocate final logits chunk buffer\n");
                close_resident_read_fd(fd, closeFd);
                return 0;
            }
            double readStarted = now_seconds();
            int readOk =
                pread_exact_or_report(fd,
                                      matrixPtr,
                                      weightChunkBytes,
                                      headInfo.weight.offset + rowStart * weightRowBytes,
                                      [residentBinPath UTF8String]) &&
                pread_exact_or_report(fd,
                                      (uint8_t *)matrixPtr + weightChunkBytes,
                                      scaleChunkBytes,
                                      headInfo.scales.offset + rowStart * scaleRowBytes,
                                      [residentBinPath UTF8String]);
            stats->read_seconds += now_seconds() - readStarted;
            if (!readOk) {
                free(matrixPtr);
                close_resident_read_fd(fd, closeFd);
                return 0;
            }
            matrix = [device newBufferWithBytesNoCopy:matrixPtr
                                               length:(NSUInteger)allocBytes
                                              options:MTLResourceStorageModeShared
                                          deallocator:^(void *pointer, NSUInteger length) {
                                              (void)length;
                                              free(pointer);
                                          }];
            weightBuffer = matrix;
            scalesBuffer = matrix;
        }
        id<MTLBuffer> output = [device newBufferWithLength:(NSUInteger)outputBytes
                                                   options:MTLResourceStorageModeShared];
        if (!matrix || !output) {
            if (!matrix) {
                free(matrixPtr);
            }
            fprintf(stderr, "ERROR: failed to allocate final logits chunk Metal buffers\n");
            if (fd >= 0) {
                close_resident_read_fd(fd, closeFd);
            }
            return 0;
        }
        if (useStagedFinalLogits) {
            [matrix didModifyRange:NSMakeRange(0, (NSUInteger)rawBytes)];
        }
        ResidentMxfp4MatrixInfo chunkInfo = headInfo;
        chunkInfo.out_dim = (uint32_t)rows;
        chunkInfo.weight.size = weightChunkBytes;
        chunkInfo.scales.size = scaleChunkBytes;
        chunkInfo.total_bytes = rawBytes;
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) {
            fprintf(stderr, "ERROR: failed to create final logits command buffer\n");
            if (fd >= 0) {
                close_resident_read_fd(fd, closeFd);
            }
            return 0;
        }
        if (!encode_glm_mxfp4_matvec_with_offsets(cmd,
                                                  matvec,
                                                  weightBuffer,
                                                  weightOffset,
                                                  scalesBuffer,
                                                  scalesOffset,
                                                  chunkInfo,
                                                  input,
                                                  output)) {
            if (fd >= 0) {
                close_resident_read_fd(fd, closeFd);
            }
            return 0;
        }
        double kernelStarted = now_seconds();
        [cmd commit];
        [cmd waitUntilCompleted];
        stats->kernel_seconds += now_seconds() - kernelStarted;
        if (cmd.status == MTLCommandBufferStatusError) {
            fprintf(stderr,
                    "ERROR: final logits command failed: %s\n",
                    cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
            if (fd >= 0) {
                close_resident_read_fd(fd, closeFd);
            }
            return 0;
        }
        float *logits = (float *)[output contents];
        for (uint64_t i = 0; i < rows; i++) {
            update_final_topk(stats->scores,
                              stats->ids,
                              &stats->count,
                              topK,
                              rowStart + i,
                              logits[i]);
        }
        if (useStagedFinalLogits) {
            stats->bytes_read += rawBytes;
            stats->lm_head_bytes_read += rawBytes;
        }
        stats->chunks++;
    }
    if (fd >= 0) {
        close_resident_read_fd(fd, closeFd);
    }
    sort_final_topk(stats->scores, stats->ids, stats->count);
    stats->elapsed_seconds = now_seconds() - totalStarted;
    stats->ok = 1;
    if (outputTopKJson &&
        !write_final_logits_topk_json(outputTopKJson, *stats)) {
        return 0;
    }
    if (outputTokenJson &&
        !write_final_logits_token_json(outputTokenJson, *stats)) {
        return 0;
    }
    return 1;
}

static int write_json_line(NSDictionary *payload) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload
                                                   options:0
                                                     error:&error];
    if (!data) {
        fprintf(stderr,
                "ERROR: failed to serialize JSONL response: %s\n",
                error ? [[error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    fwrite([data bytes], 1, [data length], stdout);
    fputc('\n', stdout);
    fflush(stdout);
    return 1;
}

static int run_glm_moe_infer_with_runtime(LoaderOptions options,
                                          GlmMoeRuntimeContext *runtime);

typedef struct {
    int fd;
    char *data;
    size_t size;
    size_t capacity;
    int ok;
} StdoutCaptureBuffer;

static void *stdout_capture_reader(void *arg) {
    StdoutCaptureBuffer *capture = (StdoutCaptureBuffer *)arg;
    capture->ok = 1;
    char buffer[65536];
    for (;;) {
        ssize_t n = read(capture->fd, buffer, sizeof(buffer));
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n < 0) {
            capture->ok = 0;
            break;
        }
        if (n == 0) {
            break;
        }
        size_t needed = capture->size + (size_t)n + 1;
        if (needed > capture->capacity) {
            size_t newCapacity = capture->capacity ? capture->capacity : 65536;
            while (newCapacity < needed) {
                if (newCapacity > SIZE_MAX / 2) {
                    capture->ok = 0;
                    close(capture->fd);
                    return NULL;
                }
                newCapacity *= 2;
            }
            char *newData = (char *)realloc(capture->data, newCapacity);
            if (!newData) {
                capture->ok = 0;
                close(capture->fd);
                return NULL;
            }
            capture->data = newData;
            capture->capacity = newCapacity;
        }
        memcpy(capture->data + capture->size, buffer, (size_t)n);
        capture->size += (size_t)n;
        capture->data[capture->size] = '\0';
    }
    close(capture->fd);
    return NULL;
}

static int run_executor_with_captured_stdout(LoaderOptions options,
                                             GlmMoeRuntimeContext *runtime,
                                             char **outStdout,
                                             size_t *outStdoutSize,
                                             int *outExecutorStatus,
                                             NSString **outError) {
    if (outStdout) {
        *outStdout = NULL;
    }
    if (outStdoutSize) {
        *outStdoutSize = 0;
    }
    if (outExecutorStatus) {
        *outExecutorStatus = 1;
    }
    if (outError) {
        *outError = nil;
    }

    int pipeFds[2] = {-1, -1};
    int savedStdout = -1;
    pthread_t readerThread = 0;
    int readerStarted = 0;
    StdoutCaptureBuffer capture = {.fd = -1, .data = NULL, .size = 0, .capacity = 0, .ok = 1};

#define CAPTURE_FAIL(message)                                                   \
    do {                                                                        \
        if (outError) {                                                         \
            *outError = (message);                                              \
        }                                                                       \
        if (pipeFds[0] >= 0 && !readerStarted) {                                \
            close(pipeFds[0]);                                                  \
        }                                                                       \
        if (pipeFds[1] >= 0) {                                                  \
            close(pipeFds[1]);                                                  \
        }                                                                       \
        if (savedStdout >= 0) {                                                 \
            close(savedStdout);                                                 \
        }                                                                       \
        if (readerStarted) {                                                    \
            pthread_join(readerThread, NULL);                                   \
        }                                                                       \
        free(capture.data);                                                     \
        return 0;                                                               \
    } while (0)

    if (pipe(pipeFds) != 0) {
        CAPTURE_FAIL(@"failed to create executor stdout pipe");
    }
    capture.fd = pipeFds[0];
    if (pthread_create(&readerThread, NULL, stdout_capture_reader, &capture) != 0) {
        CAPTURE_FAIL(@"failed to start executor stdout reader");
    }
    readerStarted = 1;
    pipeFds[0] = -1;

    savedStdout = dup(STDOUT_FILENO);
    if (savedStdout < 0) {
        CAPTURE_FAIL(@"failed to duplicate stdout before executor");
    }
    fflush(stdout);
    if (dup2(pipeFds[1], STDOUT_FILENO) < 0) {
        dup2(savedStdout, STDOUT_FILENO);
        CAPTURE_FAIL(@"failed to redirect stdout to executor capture pipe");
    }
    close(pipeFds[1]);
    pipeFds[1] = -1;

    int status = run_glm_moe_infer_with_runtime(options, runtime);
    fflush(stdout);
    dup2(savedStdout, STDOUT_FILENO);
    close(savedStdout);
    savedStdout = -1;
    pthread_join(readerThread, NULL);
    readerStarted = 0;

    if (!capture.ok) {
        free(capture.data);
        if (outError) {
            *outError = @"failed to read executor stdout";
        }
        return 0;
    }
    if (!capture.data) {
        capture.data = (char *)calloc(1, 1);
        if (!capture.data) {
            if (outError) {
                *outError = @"failed to allocate empty executor stdout capture";
            }
            return 0;
        }
    }
    if (outStdout) {
        *outStdout = capture.data;
    } else {
        free(capture.data);
    }
    if (outStdoutSize) {
        *outStdoutSize = capture.size;
    }
    if (outExecutorStatus) {
        *outExecutorStatus = status;
    }
#undef CAPTURE_FAIL
    return 1;
}

static const char *validate_generate_server_request_options(
    const LoaderOptions *options
) {
    if (options->decode_layers_csv == NULL ||
        strlen(options->decode_layers_csv) == 0) {
        return "decode_layers is required";
    }
    int inputSourceCount =
        (options->input_f32 != NULL ? 1 : 0) +
        (options->input_token_id >= 0 ? 1 : 0) +
        (options->prompt_token_count > 0 ? 1 : 0);
    if (inputSourceCount != 1) {
        return "provide exactly one of input_f32, input_token_id, or prompt_token_ids";
    }
    if (options->output_dir == NULL ||
        options->cache_layout == NULL ||
        options->cache_file == NULL ||
        options->cache_position < 0) {
        return "output-dir/cache paths and position are required";
    }
    if (options->context1_o_proj_cache_file != NULL &&
        options->context1_o_proj_cache_layout == NULL) {
        return "context1_o_proj_cache_file requires context1_o_proj_cache_layout";
    }
    if (options->context1_o_proj_cache_layout != NULL &&
        strlen(options->context1_o_proj_cache_layout) == 0) {
        return "context1_o_proj_cache_layout must not be empty";
    }
    if (options->context1_o_proj_cache_file != NULL &&
        strlen(options->context1_o_proj_cache_file) == 0) {
        return "context1_o_proj_cache_file must not be empty";
    }
    if (options->generate_steps <= 0) {
        return "generate_steps must be positive";
    }
    if (options->num_heads <= 0 ||
        options->kv_lora_dim <= 0 ||
        options->qk_nope_dim <= 0 ||
        options->rope_dim <= 0 ||
        (options->rope_dim % 2) != 0 ||
        options->v_head_dim <= 0 ||
        options->context_length <= 0 ||
        options->top_k <= 0 ||
        options->top_k > 64) {
        return "attention dimensions, context_length, and top_k are invalid";
    }
    if (options->expert_buffer_count <= 0 ||
        options->expert_buffer_count > MAX_EXPERT_BUFFERS) {
        return "expert_buffer_count is outside the supported range";
    }
    if (options->max_cache_file_mib <= 0.0 ||
        options->max_cache_read_mib <= 0.0 ||
        options->final_logits_max_chunk_mib <= 0.0 ||
        options->max_embedding_row_mib <= 0.0 ||
        options->max_live_working_set_mib < 0.0 ||
        options->min_free_unified_memory_gib < 0.0) {
        return "memory/cache limits must be positive or non-negative as appropriate";
    }
    uint64_t promptTokenCount = options->prompt_token_count > 0
        ? (uint64_t)options->prompt_token_count
        : 0;
    uint64_t decodePositions = promptTokenCount > 0
        ? promptTokenCount + (uint64_t)options->generate_steps - 1u
        : (uint64_t)options->generate_steps;
    if ((uint64_t)options->cache_position + decodePositions - 1u >
            UINT32_MAX ||
        (uint64_t)options->cache_position_offset +
                (uint64_t)options->context_length +
                decodePositions - 2u >
            UINT32_MAX) {
        return "generate request would overflow decode cache positions";
    }
    return NULL;
}

static void free_generate_request_overrides(const LoaderOptions *base,
                                            LoaderOptions *request) {
#define FREE_OVERRIDE(field)                                                    \
    do {                                                                        \
        if (request->field != NULL && request->field != base->field) {          \
            free((void *)request->field);                                       \
            request->field = NULL;                                              \
        }                                                                       \
    } while (0)
    FREE_OVERRIDE(decode_layers_csv);
    FREE_OVERRIDE(input_f32);
    FREE_OVERRIDE(output_f32);
    FREE_OVERRIDE(output_dir);
    FREE_OVERRIDE(output_generated_json);
    FREE_OVERRIDE(output_next_input_f32);
    FREE_OVERRIDE(prompt_token_ids_csv);
    FREE_OVERRIDE(cache_layout);
    FREE_OVERRIDE(cache_file);
    FREE_OVERRIDE(context1_o_proj_cache_layout);
    FREE_OVERRIDE(context1_o_proj_cache_file);
#undef FREE_OVERRIDE
}

static int run_glm_moe_infer_once(LoaderOptions options);
static int run_glm_moe_infer_with_runtime(LoaderOptions options,
                                          GlmMoeRuntimeContext *runtime);

static int run_context1_o_proj_cache_output_standalone(LoaderOptions options) {
    @autoreleasepool {
        if (options.probe_layer < 0 ||
            options.context1_o_proj_cache_layout == NULL ||
            options.input_f32 == NULL ||
            options.residual_f32 == NULL ||
            options.output_f32 == NULL) {
            fprintf(stderr,
                    "ERROR: --probe-context1-o-proj-cache-output requires "
                    "--probe-layer, --context1-o-proj-cache-layout, --input-f32, "
                    "--residual-f32, and --output-f32\n");
            return 2;
        }
        uint64_t maxCacheReadBytes = 64ull * 1024ull * 1024ull;
        if (options.max_cache_read_mib > 0.0 &&
            !positive_mib_to_bytes(options.max_cache_read_mib,
                                   "--max-cache-read-mib",
                                   &maxCacheReadBytes)) {
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "ERROR: Metal device is not available\n");
            return 1;
        }
        AttnOutputProbeStats stats = {0};
        int ok = run_context1_o_proj_cache_output_probe(
            device,
            [NSString stringWithUTF8String:options.context1_o_proj_cache_layout],
            options.context1_o_proj_cache_file
                ? [NSString stringWithUTF8String:options.context1_o_proj_cache_file]
                : nil,
            options.probe_layer,
            [NSString stringWithUTF8String:options.input_f32],
            nil,
            [NSString stringWithUTF8String:options.residual_f32],
            nil,
            [NSString stringWithUTF8String:options.output_f32],
            nil,
            nil,
            maxCacheReadBytes,
            &stats
        );
        if (options.json) {
            NSMutableDictionary *payload = [NSMutableDictionary dictionary];
            payload[@"schema"] = @"largerlm.glm_moe_infer_context1_o_proj_cache_probe.v1";
            payload[@"ok"] = [NSNumber numberWithBool:(ok ? YES : NO)];
            payload[@"device_name"] = [device name] ?: @"";
            payload[@"layer"] = @(options.probe_layer);
            payload[@"cache_layout"] =
                [NSString stringWithUTF8String:options.context1_o_proj_cache_layout];
            if (options.context1_o_proj_cache_file) {
                payload[@"cache_file_override"] =
                    [NSString stringWithUTF8String:options.context1_o_proj_cache_file];
            }
            payload[@"input_f32"] = [NSString stringWithUTF8String:options.input_f32];
            payload[@"residual_f32"] = [NSString stringWithUTF8String:options.residual_f32];
            payload[@"output_f32"] = [NSString stringWithUTF8String:options.output_f32];
            payload[@"out_dim"] = @(stats.out_dim);
            payload[@"in_dim"] = @(stats.in_dim);
            payload[@"bytes_read"] = @(stats.bytes_read);
            payload[@"input_bytes"] = @(stats.input_bytes);
            payload[@"residual_bytes"] = @(stats.residual_bytes);
            payload[@"output_bytes"] = @(stats.output_bytes);
            payload[@"read_seconds"] = @(stats.read_seconds);
            payload[@"projection_kernel_seconds"] = @(stats.projection_kernel_seconds);
            payload[@"output_write_seconds"] = @(stats.output_write_seconds);
            payload[@"elapsed_seconds"] = @(stats.elapsed_seconds);
            payload[@"command_buffer_count"] = @(stats.command_buffer_count);
            payload[@"output0"] = @(stats.output0);
            if (!write_json_line(payload)) {
                return 1;
            }
        } else {
            printf("LargerLM context=1 o_proj*B_v cache output\n");
            printf("  device:                %s\n", [[device name] UTF8String]);
            printf("  layer:                 %d\n", options.probe_layer);
            printf("  status:                %s\n", ok ? "ok" : "failed");
            printf("  dims:                  out=%u in=%u\n",
                   stats.out_dim,
                   stats.in_dim);
            printf("  bytes read:            %.3f MiB\n",
                   stats.bytes_read / 1048576.0);
            printf("  elapsed:               %.6f s\n", stats.elapsed_seconds);
            printf("  kernel:                %.6f s\n",
                   stats.projection_kernel_seconds);
            printf("  output[0]:             %.6f\n", stats.output0);
        }
        return ok ? 0 : 1;
    }
}

static int run_generate_server_jsonl(LoaderOptions baseOptions) {
    int runtimeStatus = 1;
    GlmMoeRuntimeContext *runtime =
        create_glm_moe_runtime_context(baseOptions, &runtimeStatus);
    if (!runtime) {
        return runtimeStatus;
    }
    if (!write_json_line(@{
            @"schema": @"largerlm.glm_moe_infer_generate_server.v1",
            @"event": @"ready",
            @"ok": @(YES),
            @"execution_ready": @(YES),
            @"execution_status": @"runtime_context_ready",
            @"device_name": [runtime.device name] ?: @"",
            @"resident_layout_path": runtime.residentLayoutPath,
            @"expert_layout_path": runtime.expertLayoutPath,
            @"resident_bytes": @(runtime.residentFileBytes),
            @"resident_fd_opened": @(runtime.residentFd >= 0),
            @"decode_cache_fd_opened": @(runtime.decodeCacheFd >= 0),
            @"decode_cache_fd_open_count": @(runtime.decodeCacheFdOpenCount),
            @"expert_layer_count": @([runtime.layers count]),
            @"expert_files_opened": @(runtime.openedExpertFiles),
            @"expert_buffer_count_runtime_allocated": @([runtime.expertBuffers count]),
        })) {
        return 1;
    }

    char *line = NULL;
    size_t lineCap = 0;
    ssize_t lineLen = 0;
    uint64_t requestCount = 0;
    double started = now_seconds();
    while ((lineLen = getline(&line, &lineCap, stdin)) >= 0) {
        @autoreleasepool {
            while (lineLen > 0 &&
                   (line[lineLen - 1] == '\n' || line[lineLen - 1] == '\r')) {
                line[lineLen - 1] = '\0';
                lineLen--;
            }
            if (lineLen == 0) {
                continue;
            }
            NSData *data = [NSData dataWithBytes:line length:(NSUInteger)lineLen];
            NSError *error = nil;
            id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            if (![obj isKindOfClass:[NSDictionary class]]) {
                requestCount++;
                write_json_line(@{
                    @"schema": @"largerlm.glm_moe_infer_generate_server.v1",
                    @"event": @"request",
                    @"ok": @(NO),
                    @"accepted": @(NO),
                    @"request_index": @(requestCount),
                    @"error": error ? [error localizedDescription] : @"invalid JSON object",
                });
                continue;
            }
            NSDictionary *request = (NSDictionary *)obj;
            NSString *command = request[@"command"];
            if ([command isKindOfClass:[NSString class]] &&
                [command isEqualToString:@"quit"]) {
                break;
            }

            requestCount++;
            LoaderOptions requestOptions = baseOptions;
            requestOptions.generate_request_json = NULL;
            requestOptions.generate_server_jsonl = 0;
            requestOptions.in_memory_decode_cache = 1;
            apply_generate_request_dictionary(&requestOptions, request);
            int dryRun = 0;
            json_bool_field(request, @"dry_run", "dry_run", &dryRun);
            char *executorStdoutJson =
                dup_json_string_field(request,
                                      @"executor_stdout_json",
                                      "executor_stdout_json");
            const char *validationError =
                validate_generate_server_request_options(&requestOptions);
            if (validationError != NULL) {
                write_json_line(@{
                    @"schema": @"largerlm.glm_moe_infer_generate_server.v1",
                    @"event": @"request",
                    @"ok": @(NO),
                    @"accepted": @(NO),
                    @"request_index": @(requestCount),
                    @"error": [NSString stringWithUTF8String:validationError],
                });
                free(executorStdoutJson);
                free_generate_request_overrides(&baseOptions, &requestOptions);
                continue;
            }
            int executorStatus = 0;
            char *executorStdout = NULL;
            size_t executorStdoutSize = 0;
            int executorCaptureOk = 1;
            NSString *executorCaptureError = nil;
            int executorPayloadParsed = 0;
            NSString *executorPayloadParseError = nil;
            int executorStdoutJsonWriteOk = 1;
            NSDictionary *executorPayload = nil;
            double executorStarted = now_seconds();
            if (!dryRun) {
                requestOptions.json = 1;
                executorCaptureOk =
                    run_executor_with_captured_stdout(requestOptions,
                                                      runtime,
                                                      &executorStdout,
                                                      &executorStdoutSize,
                                                      &executorStatus,
                                                      &executorCaptureError);
                if (!executorCaptureOk) {
                    write_json_line(@{
                        @"schema": @"largerlm.glm_moe_infer_generate_server.v1",
                        @"event": @"request",
                        @"ok": @(NO),
                        @"accepted": @(YES),
                        @"executed": @(NO),
                        @"request_index": @(requestCount),
                        @"error": executorCaptureError ?: @"failed to capture executor stdout",
                    });
                    free(executorStdoutJson);
                    free(executorStdout);
                    free_generate_request_overrides(&baseOptions, &requestOptions);
                    continue;
                }
                NSData *executorData =
                    [NSData dataWithBytes:(executorStdout ? executorStdout : "")
                                   length:executorStdoutSize];
                if (executorStdoutJson) {
                    NSString *executorStdoutPath =
                        [NSString stringWithUTF8String:executorStdoutJson];
                    if (![executorData writeToFile:executorStdoutPath atomically:YES]) {
                        executorStdoutJsonWriteOk = 0;
                    }
                }
                if (executorStdoutSize > 0) {
                    NSError *parseError = nil;
                    id parsed = [NSJSONSerialization JSONObjectWithData:executorData
                                                                 options:0
                                                                   error:&parseError];
                    if ([parsed isKindOfClass:[NSDictionary class]]) {
                        executorPayload = (NSDictionary *)parsed;
                        executorPayloadParsed = 1;
                    } else {
                        executorPayloadParseError = parseError
                            ? [parseError localizedDescription]
                            : @"executor stdout was not a JSON object";
                    }
                } else {
                    executorPayloadParseError = @"executor stdout was empty";
                }
            }
            double executorElapsed = now_seconds() - executorStarted;
            NSMutableDictionary *payload = [@{
                @"schema": @"largerlm.glm_moe_infer_generate_server.v1",
                @"event": @"request",
                @"ok": @((dryRun ||
                           (executorStatus == 0 &&
                            executorPayloadParsed &&
                            executorStdoutJsonWriteOk)) ? YES : NO),
                @"accepted": @(YES),
                @"executed": @(dryRun ? NO : YES),
                @"execution_ready": @(dryRun ? NO : YES),
                @"execution_status": dryRun
                    ? @"request_protocol_only"
                    : @"executed_once",
                @"executor_exit_code": @(executorStatus),
                @"executor_elapsed_seconds": @(executorElapsed),
                @"executor_stdout_captured": @(dryRun ? NO : YES),
                @"executor_stdout_bytes": @(executorStdoutSize),
                @"executor_stdout_json": executorStdoutJson
                    ? [NSString stringWithUTF8String:executorStdoutJson]
                    : (id)[NSNull null],
                @"executor_stdout_json_write_ok": @(executorStdoutJsonWriteOk),
                @"executor_payload_parsed": @(executorPayloadParsed),
                @"request_index": @(requestCount),
                @"runtime_entry": @"generate_token_ids",
                @"decode_layers": [NSString stringWithUTF8String:
                    requestOptions.decode_layers_csv],
                @"generate_steps": @(requestOptions.generate_steps),
                @"input_source": requestOptions.prompt_token_count > 0
                    ? @"prompt_token_ids"
                    : (requestOptions.input_token_id >= 0
                        ? @"input_token_id"
                        : @"input_f32"),
                @"prompt_token_count": @(requestOptions.prompt_token_count),
                @"position": @(requestOptions.cache_position),
                @"context_length": @(requestOptions.context_length),
                @"top_k": @(requestOptions.top_k),
                @"expert_buffer_count": @(requestOptions.expert_buffer_count),
                @"expert_buffer_count_runtime_allocated":
                    @([runtime.expertBuffers count]),
                @"decode_cache_fd_opened": @(runtime.decodeCacheFd >= 0),
                @"decode_cache_fd_open_count": @(runtime.decodeCacheFdOpenCount),
                @"decode_cache_fd_path": runtime.decodeCachePath
                    ? runtime.decodeCachePath
                    : (id)[NSNull null],
                @"decode_cache_backend": requestOptions.in_memory_decode_cache
                    ? @"memory"
                    : @"file",
                @"decode_cache_memory_loaded":
                    @(runtime.decodeCacheMemory != nil),
                @"decode_cache_memory_load_count":
                    @(runtime.decodeCacheMemoryLoadCount),
                @"decode_cache_memory_bytes": runtime.decodeCacheMemory
                    ? @((uint64_t)[runtime.decodeCacheMemory length])
                    : @(0),
                @"mla_kv_b_cache_requested":
                    @(requestOptions.cache_mla_kv_b_f32 ? YES : NO),
                @"max_mla_kv_b_cache_mib":
                    @(requestOptions.max_mla_kv_b_cache_mib),
                @"mla_kv_b_cache_current_bytes":
                    @(mla_kv_b_memory_cache_current_bytes()),
                @"runtime_context_ready": @(YES),
                @"runtime_reused": @(dryRun ? NO : YES),
                @"include_shared_expert": @(requestOptions.include_shared_expert ? YES : NO),
                @"max_live_working_set_mib": @(requestOptions.max_live_working_set_mib),
                @"min_free_unified_memory_gib": @(requestOptions.min_free_unified_memory_gib),
                @"next_step": dryRun
                    ? @"send dry_run=false to execute through the persistent runtime with inline executor payload"
                    : @"remove remaining per-request file/cache boundaries and fold prompt prefill into the persistent runtime",
            } mutableCopy];
            if (executorPayloadParseError) {
                payload[@"executor_payload_parse_error"] = executorPayloadParseError;
            }
            if (executorPayload) {
                id admission = executorPayload[@"admission_ok"];
                id available = executorPayload[@"available_unified_memory_ok"];
                id estimated = executorPayload[@"estimated_live_working_set_bytes"];
                id required = executorPayload[@"required_available_memory_bytes"];
                id systemAvailable = executorPayload[@"system_available_memory_bytes"];
                id cacheFdOpened = executorPayload[@"decode_cache_fd_opened"];
                id cacheBackend = executorPayload[@"decode_cache_backend"];
                id cacheMemoryLoaded = executorPayload[@"decode_cache_memory_loaded"];
                id cacheMemoryBytes = executorPayload[@"decode_cache_memory_bytes"];
                id expertPoolReused = executorPayload[@"expert_buffer_pool_reused"];
                id mlaKVBCacheEnabled = executorPayload[@"mla_kv_b_cache_enabled"];
                id mlaKVBCacheCurrentBytes =
                    executorPayload[@"mla_kv_b_cache_current_bytes"];
                id mlaKVBCacheLiveEstimateBytes =
                    executorPayload[@"mla_kv_b_cache_live_estimate_bytes"];
                payload[@"admission_ok"] = admission ?: (id)[NSNull null];
                payload[@"available_unified_memory_ok"] = available ?: (id)[NSNull null];
                payload[@"estimated_live_working_set_bytes"] =
                    estimated ?: (id)[NSNull null];
                payload[@"required_available_memory_bytes"] =
                    required ?: (id)[NSNull null];
                payload[@"system_available_memory_bytes"] =
                    systemAvailable ?: (id)[NSNull null];
                payload[@"executor_decode_cache_fd_opened"] =
                    cacheFdOpened ?: (id)[NSNull null];
                payload[@"executor_decode_cache_backend"] =
                    cacheBackend ?: (id)[NSNull null];
                payload[@"executor_decode_cache_memory_loaded"] =
                    cacheMemoryLoaded ?: (id)[NSNull null];
                payload[@"executor_decode_cache_memory_bytes"] =
                    cacheMemoryBytes ?: (id)[NSNull null];
                payload[@"expert_buffer_pool_reused"] =
                    expertPoolReused ?: (id)[NSNull null];
                payload[@"executor_mla_kv_b_cache_enabled"] =
                    mlaKVBCacheEnabled ?: (id)[NSNull null];
                payload[@"executor_mla_kv_b_cache_current_bytes"] =
                    mlaKVBCacheCurrentBytes ?: (id)[NSNull null];
                payload[@"executor_mla_kv_b_cache_live_estimate_bytes"] =
                    mlaKVBCacheLiveEstimateBytes ?: (id)[NSNull null];
                NSDictionary *generate =
                    [executorPayload[@"probe_generate"] isKindOfClass:[NSDictionary class]]
                        ? executorPayload[@"probe_generate"]
                        : nil;
                if (generate) {
                    payload[@"probe_generate"] = generate;
                    payload[@"probe_generate_ok"] = generate[@"ok"] ?: (id)[NSNull null];
                    payload[@"generated_token_ids"] =
                        generate[@"generated_token_ids"] ?: (id)[NSNull null];
                }
            }
            int wrote = write_json_line(payload);
            free(executorStdoutJson);
            free(executorStdout);
            free_generate_request_overrides(&baseOptions, &requestOptions);
            if (!wrote) {
                free(line);
                return 1;
            }
        }
    }
    free(line);
    if (!write_json_line(@{
            @"schema": @"largerlm.glm_moe_infer_generate_server.v1",
            @"event": @"done",
            @"ok": @(YES),
            @"requests": @(requestCount),
            @"elapsed_seconds": @(now_seconds() - started),
        })) {
        return 1;
    }
    return 0;
}

static int run_context1_o_proj_cache_build_layer_standalone(LoaderOptions options) {
    @autoreleasepool {
        if (options.probe_layer < 0 ||
            options.resident_layout == NULL ||
            options.context1_o_proj_cache_layout == NULL) {
            fprintf(stderr,
                    "ERROR: --build-context1-o-proj-cache-layer requires "
                    "--probe-layer, --resident-layout/--prepared, and "
                    "--context1-o-proj-cache-layout\n");
            return 2;
        }
        uint64_t maxCacheReadBytes = 64ull * 1024ull * 1024ull;
        if (options.max_cache_read_mib > 0.0 &&
            !positive_mib_to_bytes(options.max_cache_read_mib,
                                   "--max-cache-read-mib",
                                   &maxCacheReadBytes)) {
            return 2;
        }
        uint64_t maxLiveWorkingSetBytes = 0;
        if (options.max_live_working_set_mib > 0.0 &&
            !positive_mib_to_bytes(options.max_live_working_set_mib,
                                   "--max-live-working-set-mib",
                                   &maxLiveWorkingSetBytes)) {
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "ERROR: Metal device is not available\n");
            return 1;
        }
        Context1OProjCacheBuildStats stats = {0};
        int ok = run_context1_o_proj_cache_build_layer(
            device,
            [NSString stringWithUTF8String:options.resident_layout],
            [NSString stringWithUTF8String:options.context1_o_proj_cache_layout],
            options.context1_o_proj_cache_file
                ? [NSString stringWithUTF8String:options.context1_o_proj_cache_file]
                : nil,
            options.probe_layer,
            maxCacheReadBytes,
            options.max_context1_o_proj_build_gfma,
            maxLiveWorkingSetBytes,
            options.max_live_working_set_mib,
            &stats
        );
        if (options.json) {
            NSMutableDictionary *payload = [NSMutableDictionary dictionary];
            payload[@"schema"] = @"largerlm.glm_moe_infer_context1_o_proj_cache_build_layer.v1";
            payload[@"ok"] = [NSNumber numberWithBool:(ok ? YES : NO)];
            payload[@"device_name"] = [device name] ?: @"";
            payload[@"layer"] = @(options.probe_layer);
            payload[@"resident_layout"] = [NSString stringWithUTF8String:options.resident_layout];
            payload[@"cache_layout"] =
                [NSString stringWithUTF8String:options.context1_o_proj_cache_layout];
            if (options.context1_o_proj_cache_file) {
                payload[@"cache_file_override"] =
                    [NSString stringWithUTF8String:options.context1_o_proj_cache_file];
            }
            payload[@"hidden_dim"] = @(stats.hidden_dim);
            payload[@"attention_value_dim"] = @(stats.attention_value_dim);
            payload[@"kv_lora_dim"] = @(stats.kv_lora_dim);
            payload[@"source_bytes_read"] = @(stats.source_bytes_read);
            payload[@"cache_bytes_written"] = @(stats.cache_bytes_written);
            payload[@"fma_count"] = @(stats.fma_count);
            payload[@"max_build_gfma"] = @(options.max_context1_o_proj_build_gfma);
            payload[@"estimated_live_working_set_bytes"] =
                @(stats.estimated_live_working_set_bytes);
            payload[@"max_live_working_set_mib"] =
                @(stats.max_live_working_set_mib);
            payload[@"live_working_set_ok"] =
                @(stats.live_working_set_ok ? YES : NO);
            payload[@"read_seconds"] = @(stats.read_seconds);
            payload[@"kernel_seconds"] = @(stats.kernel_seconds);
            payload[@"write_seconds"] = @(stats.write_seconds);
            payload[@"elapsed_seconds"] = @(stats.elapsed_seconds);
            payload[@"output0"] = @(stats.output0);
            if (!write_json_line(payload)) {
                return 1;
            }
        } else {
            printf("LargerLM context=1 o_proj*B_v cache layer build\n");
            printf("  device:                %s\n", [[device name] UTF8String]);
            printf("  layer:                 %d\n", options.probe_layer);
            printf("  status:                %s\n", ok ? "ok" : "failed");
            printf("  dims:                  hidden=%u value=%u kv_lora=%u\n",
                   stats.hidden_dim,
                   stats.attention_value_dim,
                   stats.kv_lora_dim);
            printf("  source read:           %.3f MiB\n",
                   stats.source_bytes_read / 1048576.0);
            printf("  cache written:         %.3f MiB\n",
                   stats.cache_bytes_written / 1048576.0);
            printf("  estimated live:        %.3f MiB\n",
                   stats.estimated_live_working_set_bytes / 1048576.0);
            if (stats.max_live_working_set_mib > 0.0) {
                printf("  live cap:              %.3f MiB\n",
                       stats.max_live_working_set_mib);
                printf("  live cap ok:           %s\n",
                       stats.live_working_set_ok ? "yes" : "no");
            }
            printf("  FMA:                   %llu\n",
                   (unsigned long long)stats.fma_count);
            printf("  elapsed:               %.6f s\n", stats.elapsed_seconds);
            printf("  kernel:                %.6f s\n", stats.kernel_seconds);
            printf("  output[0]:             %.6f\n", stats.output0);
        }
        return ok ? 0 : 1;
    }
}

static int run_glm_moe_infer_once(LoaderOptions options) {
    @autoreleasepool {
        if (options.build_context1_o_proj_cache_layer) {
            return run_context1_o_proj_cache_build_layer_standalone(options);
        }
        if (options.probe_context1_o_proj_cache_output) {
            return run_context1_o_proj_cache_output_standalone(options);
        }
        int runtimeStatus = 1;
        GlmMoeRuntimeContext *runtime =
            create_glm_moe_runtime_context(options, &runtimeStatus);
        if (!runtime) {
            return runtimeStatus;
        }
        return run_glm_moe_infer_with_runtime(options, runtime);
    }
}

static int run_glm_moe_infer_with_runtime(LoaderOptions options,
                                          GlmMoeRuntimeContext *runtime) {
    @autoreleasepool {
        if (!runtime) {
            fprintf(stderr, "ERROR: runtime context is not initialized\n");
            return 1;
        }
        int decodeLayersFinalLogits =
            options.probe_decode_layers &&
            (options.output_topk_json != NULL ||
             options.output_token_json != NULL ||
             options.output_next_input_f32 != NULL ||
             options.generate_steps > 0);
        IntList probeExperts = {.values = NULL, .count = 0};
        FloatList probeWeights = {.values = NULL, .count = 0};
        IntList decodeLayers = {.values = NULL, .count = 0};
        IntList promptTokenIds = {.values = NULL, .count = 0};
        int appendCache = options.cache_layout != NULL;
        uint64_t maxCacheFileBytes = 0;
        uint64_t maxCacheReadBytes = 0;
        uint64_t maxFinalLogitsChunkBytes = 0;
        uint64_t maxEmbeddingRowBytes = 0;
        uint64_t maxMlaKVBMemoryCacheBytes = 0;
        uint64_t decodeCacheMemoryBytes = 0;
        if (appendCache &&
            !max_cache_file_bytes_from_mib(options.max_cache_file_mib, &maxCacheFileBytes)) {
            return 2;
        }
        if (appendCache && options.in_memory_decode_cache) {
            if (!decode_cache_layout_total_bytes(
                    [NSString stringWithUTF8String:options.cache_layout],
                    maxCacheFileBytes,
                    &decodeCacheMemoryBytes)) {
                return 2;
            }
        }
        if ((options.probe_mla_attention ||
             options.probe_decoder_layer ||
             options.probe_dense_decoder_layer ||
             options.probe_decode_layers) &&
            !max_cache_file_bytes_from_mib(options.max_cache_read_mib, &maxCacheReadBytes)) {
            return 2;
        }
        if ((options.probe_final_logits || decodeLayersFinalLogits) &&
            !positive_mib_to_bytes(options.final_logits_max_chunk_mib,
                                   "--max-chunk-mib",
                                   &maxFinalLogitsChunkBytes)) {
            return 2;
        }
        if ((options.output_next_input_f32 ||
             options.generate_steps > 0 ||
             options.input_token_id >= 0) &&
            !positive_mib_to_bytes(options.max_embedding_row_mib,
                                   "--max-embedding-row-mib",
                                   &maxEmbeddingRowBytes)) {
            return 2;
        }
        if (options.cache_mla_kv_b_f32 &&
            !positive_mib_to_bytes(options.max_mla_kv_b_cache_mib,
                                   "--max-mla-kv-b-cache-mib",
                                   &maxMlaKVBMemoryCacheBytes)) {
            return 2;
        }
        mla_kv_b_memory_cache_configure(options.cache_mla_kv_b_f32,
                                        maxMlaKVBMemoryCacheBytes);
        if (options.probe_expert_read || options.probe_layer_moe) {
            probeExperts = parse_nonnegative_int_csv(options.probe_experts_csv, "--probe-experts");
            if (options.probe_expert_read && probeExperts.count > options.expert_buffer_count) {
                fprintf(stderr,
                        "ERROR: --probe-experts count %d exceeds --expert-buffer-count %d\n",
                        probeExperts.count,
                        options.expert_buffer_count);
                return 2;
            }
        }
        if (options.probe_layer_moe) {
            probeWeights = parse_float_csv(options.probe_weights_csv, "--probe-weights");
            if (probeWeights.count != probeExperts.count) {
                fprintf(stderr,
                        "ERROR: --probe-weights count %d must match --probe-experts count %d\n",
                        probeWeights.count,
                        probeExperts.count);
                return 2;
            }
        }
        if (options.probe_decode_layers) {
            decodeLayers = parse_nonnegative_int_csv(options.decode_layers_csv,
                                                     "--decode-layers");
            if (decodeLayers.count <= 0) {
                fprintf(stderr, "ERROR: --decode-layers must include at least one layer\n");
                return 2;
            }
        }
        if (options.prompt_token_ids_csv) {
            promptTokenIds = parse_nonnegative_int_csv(options.prompt_token_ids_csv,
                                                       "prompt_token_ids");
            if (promptTokenIds.count <= 0) {
                fprintf(stderr, "ERROR: prompt_token_ids must include at least one token\n");
                return 2;
            }
        }
        NSString *residentLayoutPath = runtime.residentLayoutPath;
        NSString *expertLayoutPath = runtime.expertLayoutPath;
        NSDictionary *residentLayout = runtime.residentLayout;
        NSDictionary *expertLayout = runtime.expertLayout;
        id<MTLDevice> device = runtime.device;
        NSString *residentBinPath = runtime.residentBinPath;
        uint64_t residentFileBytes = runtime.residentFileBytes;
        NSArray *layers = runtime.layers;
        uint64_t maxExpertSlotBytes = runtime.maxExpertSlotBytes;
        uint64_t totalExpertBytes = runtime.totalExpertBytes;
        ExpertFile *expertFiles = runtime.expertFiles;
        int openedExpertFiles = runtime.openedExpertFiles;
        ExpertFile *probeFile = NULL;
        NSDictionary *probeLayerDict = nil;
        for (NSUInteger i = 0; i < [layers count]; i++) {
            NSDictionary *layer = layers[i];
            if ((options.probe_expert_read ||
                 options.probe_layer_moe ||
                 options.probe_router_moe ||
                 options.probe_decoder_layer) &&
                expertFiles[i].layer == options.probe_layer) {
                probeFile = &expertFiles[i];
                probeLayerDict = layer;
            }
        }
        if (options.probe_expert_read ||
            options.probe_layer_moe ||
            options.probe_router_moe ||
            options.probe_decoder_layer) {
            if (!options.probe_all_layers && probeFile == NULL) {
                fprintf(stderr, "ERROR: --probe-layer %d was not found in expert layout\n",
                        options.probe_layer);
                return 2;
            }
            for (NSUInteger layerIndex = 0; layerIndex < [layers count]; layerIndex++) {
                ExpertFile *candidate = &expertFiles[layerIndex];
                if (!options.probe_all_layers && candidate != probeFile) {
                    continue;
                }
                if (candidate->fd < 0) {
                    fprintf(stderr, "ERROR: probe requires open expert files\n");
                    return 2;
                }
                for (int i = 0; i < probeExperts.count; i++) {
                    if ((uint64_t)probeExperts.values[i] >= candidate->num_experts) {
                        fprintf(stderr,
                                "ERROR: expert id %d is outside layer %d expert count %llu\n",
                                probeExperts.values[i],
                                candidate->layer,
                                (unsigned long long)candidate->num_experts);
                        return 2;
                    }
                }
            }
        }
        Mxfp4ExpertInfo probeMoeInfo = {0};
        uint64_t probeMoeScratchBytes = 0;
        SharedMxfp4Info sharedMoeInfo = {0};
        uint64_t sharedExpertStorageBytes = 0;
        ResidentVectorInfo rmsNormInfo = {0};
        uint64_t rmsNormScratchBytes = 0;
        DenseMlpMxfp4Info denseMlpInfo = {0};
        uint64_t denseMlpScratchBytes = 0;
        RouterWeightInfo routerInfo = {0};
        RouterBiasInfo routerBiasInfo = {0};
        RouterTopKOptions routerTopKOptions = {0};
        uint64_t routerScratchBytes = 0;
        ResidentMxfp4MatrixInfo residentLinearInfo = {0};
        uint64_t residentLinearScratchBytes = 0;
        ResidentVectorInfo attnInputNormInfo = {0};
        ResidentVectorInfo attnQANormInfo = {0};
        ResidentVectorInfo attnKVANormInfo = {0};
        ResidentMxfp4MatrixInfo attnQAInfo = {0};
        ResidentMxfp4MatrixInfo attnQBInfo = {0};
        ResidentMxfp4MatrixInfo attnKVAInfo = {0};
        ResidentMxfp4MatrixInfo attnKVBInfo = {0};
        int attnHasKVB = 0;
        uint64_t attnProjectionScratchBytes = 0;
        uint64_t ropeSplitScratchBytes = 0;
        uint64_t ropeSplitQBBytes = 0;
        uint64_t ropeSplitKBytes = 0;
        uint64_t ropeSplitQNopeBytes = 0;
        uint64_t ropeSplitQRopeBytes = 0;
        MlaAttentionValueSourceInfo mlaValueSource = {0};
        uint64_t mlaAttentionScratchBytes = 0;
        uint64_t mlaRawCacheBytes = 0;
        uint64_t mlaCacheF32Bytes = 0;
        ResidentMxfp4MatrixInfo attnOutputInfo = {0};
        uint64_t attnOutputScratchBytes = 0;
        uint64_t decoderLayerScratchBytes = 0;
        uint64_t denseDecoderLayerScratchBytes = 0;
        DecodeLayerPlan *decodeLayerPlans = NULL;
        DecodeLayerRunSummary *decodeLayerSummaries = NULL;
        uint64_t decodeLayersScratchBytes = 0;
        int decodeLayersHasMoe = 0;
        uint64_t mlaKVBMemoryCachePlannedBytes = 0;
        uint64_t mlaKVBMemoryCacheLiveEstimateBytes =
            options.cache_mla_kv_b_f32 ? maxMlaKVBMemoryCacheBytes : 0;
        ResidentMxfp4MatrixInfo finalLogitsHeadInfo = {0};
        ResidentVectorInfo finalNormInfo = {0};
        uint64_t finalLogitsScratchBytes = 0;
        uint64_t finalLogitsMmapBytes = 0;
        ResidentMxfp4MatrixInfo nextInputEmbeddingInfo = {0};
        uint64_t nextInputEmbeddingScratchBytes = 0;
        if (options.probe_resident_linear) {
            if (!find_resident_mxfp4_matrix_info(
                    residentLayout,
                    [NSString stringWithUTF8String:options.resident_tensor_name],
                    &residentLinearInfo)) {
                return 2;
            }
            residentLinearScratchBytes =
                resident_linear_scratch_bytes_for_backing(
                    residentLinearInfo,
                    options.wrap_resident_metal
                );
        }
        if (options.probe_dense_mlp_block || options.probe_dense_decoder_layer) {
            if (!parse_dense_mlp_mxfp4_info(residentLayout,
                                            options.probe_layer,
                                            &denseMlpInfo) ||
                !find_layer_vector_info(residentLayout,
                                        options.probe_layer,
                                        @".post_attention_layernorm.weight",
                                        &rmsNormInfo)) {
                return 2;
            }
            if (rmsNormInfo.dim != denseMlpInfo.hidden_dim) {
                fprintf(stderr,
                        "ERROR: dense MLP RMSNorm dim %u does not match hidden dim %u\n",
                        rmsNormInfo.dim,
                        denseMlpInfo.hidden_dim);
                return 2;
            }
            rmsNormScratchBytes =
                (uint64_t)rmsNormInfo.dim * sizeof(float) * 3u +
                rmsNormInfo.size;
            denseMlpScratchBytes = dense_mlp_scratch_bytes(denseMlpInfo, rmsNormInfo);
        }
        if (options.probe_attn_projections ||
            options.probe_decoder_layer ||
            options.probe_dense_decoder_layer) {
            NSString *qAName = layer_tensor_name(
                options.probe_layer,
                @".self_attn.q_a_proj.weight"
            );
            NSString *qBName = layer_tensor_name(
                options.probe_layer,
                @".self_attn.q_b_proj.weight"
            );
            NSString *kvAName = layer_tensor_name(
                options.probe_layer,
                @".self_attn.kv_a_proj_with_mqa.weight"
            );
            NSString *kvBName = layer_tensor_name(
                options.probe_layer,
                @".self_attn.kv_b_proj.weight"
            );
            if (!find_layer_vector_info(residentLayout,
                                        options.probe_layer,
                                        @".input_layernorm.weight",
                                        &attnInputNormInfo) ||
                !find_layer_vector_info(residentLayout,
                                        options.probe_layer,
                                        @".self_attn.q_a_layernorm.weight",
                                        &attnQANormInfo) ||
                !find_layer_vector_info(residentLayout,
                                        options.probe_layer,
                                        @".self_attn.kv_a_layernorm.weight",
                                        &attnKVANormInfo) ||
                !find_resident_mxfp4_matrix_info(residentLayout, qAName, &attnQAInfo) ||
                !find_resident_mxfp4_matrix_info(residentLayout, qBName, &attnQBInfo) ||
                !find_resident_mxfp4_matrix_info(residentLayout, kvAName, &attnKVAInfo)) {
                return 2;
            }
            if (attnQAInfo.in_dim != attnKVAInfo.in_dim ||
                attnInputNormInfo.dim != attnQAInfo.in_dim ||
                attnQANormInfo.dim != attnQAInfo.out_dim ||
                attnQBInfo.in_dim != attnQAInfo.out_dim ||
                attnKVANormInfo.dim > attnKVAInfo.out_dim) {
                fprintf(stderr, "ERROR: attention projection dimensions are inconsistent\n");
                return 2;
            }
            if (resident_tensor_exists(residentLayout, kvBName)) {
                if (!find_resident_mxfp4_matrix_info(residentLayout, kvBName, &attnKVBInfo)) {
                    return 2;
                }
                if (attnKVBInfo.in_dim != attnKVANormInfo.dim) {
                    fprintf(stderr, "ERROR: attention kv_b input dim does not match kv_a norm dim\n");
                    return 2;
                }
                attnHasKVB = 1;
            }
            attnProjectionScratchBytes =
                (uint64_t)attnInputNormInfo.dim * sizeof(float) * 3u +
                attnInputNormInfo.size +
                (uint64_t)attnQANormInfo.dim * sizeof(float) * 3u +
                attnQANormInfo.size +
                (uint64_t)attnKVANormInfo.dim * sizeof(float) * 3u +
                attnKVANormInfo.size +
                resident_linear_scratch_bytes(attnQAInfo) +
                resident_linear_scratch_bytes(attnQBInfo) +
                resident_linear_scratch_bytes(attnKVAInfo) +
                (attnHasKVB ? resident_linear_scratch_bytes(attnKVBInfo) : 0);
            if (appendCache) {
                attnProjectionScratchBytes += (uint64_t)attnKVAInfo.out_dim * sizeof(float);
            }
        }
	        if (options.probe_rope_split ||
                options.probe_decoder_layer ||
                options.probe_dense_decoder_layer ||
                options.probe_decode_layers) {
	            ropeSplitScratchBytes = rope_split_scratch_bytes(
	                (uint32_t)options.num_heads,
	                (uint32_t)options.qk_nope_dim,
	                (uint32_t)options.rope_dim,
	                (uint32_t)((options.probe_decoder_layer ||
                                options.probe_dense_decoder_layer ||
                                options.probe_decode_layers)
                                   ? 1
                                   : options.batch_tokens),
	                &ropeSplitQBBytes,
	                &ropeSplitKBytes,
	                &ropeSplitQNopeBytes,
                &ropeSplitQRopeBytes
            );
        }
        if (options.probe_mla_attention ||
            options.probe_decoder_layer ||
            options.probe_dense_decoder_layer) {
            if (options.mla_kv_b_f32 != NULL) {
                uint64_t qv = 0;
                uint64_t expectedOut = 0;
                uint64_t kvBValues = 0;
                uint64_t kvBBytes = 0;
                if (!checked_add_u64((uint64_t)options.qk_nope_dim,
                                     (uint64_t)options.v_head_dim,
                                     &qv) ||
                    !checked_mul_u64((uint64_t)options.num_heads,
                                     qv,
                                     &expectedOut) ||
                    expectedOut > UINT32_MAX ||
                    !checked_mul_u64(expectedOut,
                                     (uint64_t)options.kv_lora_dim,
                                     &kvBValues) ||
                    !checked_mul_u64(kvBValues, sizeof(float), &kvBBytes)) {
                    fprintf(stderr, "ERROR: direct MLA KV-B byte size overflows\n");
                    return 2;
                }
                mlaValueSource.kv_lora_dim = (uint32_t)options.kv_lora_dim;
                mlaValueSource.expected_kv_b_out = (uint32_t)expectedOut;
                mlaValueSource.storage_bytes = 0;
                mlaValueSource.source_f32_bytes = 0;
                mlaValueSource.kv_b_f32_bytes = kvBBytes;
            } else {
                if (!resolve_mla_attention_value_source(residentLayout,
                                                        options.probe_layer,
                                                        (uint32_t)options.num_heads,
                                                        (uint32_t)options.kv_lora_dim,
                                                        (uint32_t)options.qk_nope_dim,
                                                        (uint32_t)options.v_head_dim,
                                                        &mlaValueSource)) {
                    return 2;
                }
            }
            if (options.cache_mla_kv_b_f32 &&
                !checked_add_u64(mlaKVBMemoryCachePlannedBytes,
                                 mlaValueSource.kv_b_f32_bytes,
                                 &mlaKVBMemoryCachePlannedBytes)) {
                fprintf(stderr, "ERROR: MLA KV-B cache planned bytes overflow\n");
                return 2;
            }
            uint32_t cacheWidth = mlaValueSource.kv_lora_dim + (uint32_t)options.rope_dim;
            if (!mla_attention_cache_byte_counts(
                    [NSString stringWithUTF8String:options.cache_layout],
                    (uint64_t)options.probe_layer,
                    (uint32_t)options.context_length,
                    cacheWidth,
                    maxCacheFileBytes,
                    maxCacheReadBytes,
                    &mlaRawCacheBytes,
                    &mlaCacheF32Bytes)) {
                return 2;
            }
            uint64_t qNopeBytes =
                (uint64_t)options.num_heads * (uint64_t)options.qk_nope_dim * sizeof(float);
            uint64_t qRopeBytes =
                (uint64_t)options.num_heads * (uint64_t)options.rope_dim * sizeof(float);
            uint64_t outputBytes =
                (uint64_t)options.num_heads * (uint64_t)options.v_head_dim * sizeof(float);
            mlaAttentionScratchBytes =
                mlaRawCacheBytes +
                mlaCacheF32Bytes +
                mlaValueSource.storage_bytes +
                mlaValueSource.source_f32_bytes +
                mlaValueSource.kv_b_f32_bytes +
                qNopeBytes +
                qRopeBytes +
                outputBytes;
        }
        if (options.probe_attn_output) {
            NSString *oProjName = layer_tensor_name(
                options.probe_layer,
                @".self_attn.o_proj.weight"
            );
            if (!find_resident_mxfp4_matrix_info(residentLayout, oProjName, &attnOutputInfo)) {
                return 2;
            }
            attnOutputScratchBytes =
                resident_linear_scratch_bytes_for_backing(attnOutputInfo,
                                                          options.wrap_resident_metal) +
                (uint64_t)attnOutputInfo.out_dim * sizeof(float);
        }
        if ((options.probe_decoder_layer || options.probe_dense_decoder_layer) &&
            !options.probe_attn_output) {
            NSString *oProjName = layer_tensor_name(
                options.probe_layer,
                @".self_attn.o_proj.weight"
            );
            if (!find_resident_mxfp4_matrix_info(residentLayout, oProjName, &attnOutputInfo)) {
                return 2;
            }
            attnOutputScratchBytes =
                resident_linear_scratch_bytes(attnOutputInfo) +
                (uint64_t)attnOutputInfo.out_dim * sizeof(float);
        }
        if (options.probe_router || options.probe_decoder_layer) {
            if (!find_router_weight_info(residentLayout, options.probe_layer, &routerInfo)) {
                return 2;
            }
            if (!resolve_router_topk_options(residentLayout, options, &routerTopKOptions)) {
                return 2;
            }
            if (!options.ignore_router_bias &&
                !find_router_bias_info(residentLayout, options.probe_layer, &routerBiasInfo)) {
                return 2;
            }
            routerScratchBytes =
                round_up_u64(routerInfo.size, 2 * 1024 * 1024) +
                (uint64_t)routerInfo.hidden_dim * sizeof(float) +
                (uint64_t)routerInfo.num_experts * sizeof(float) +
                (uint64_t)routerInfo.num_experts * sizeof(float);
            if (routerBiasInfo.present) {
                routerScratchBytes += routerBiasInfo.size;
            }
        }
        if (options.probe_layer_moe ||
            options.probe_router_moe ||
            options.probe_decoder_layer) {
            if (probeLayerDict == nil || probeFile == NULL) {
                fprintf(stderr, "ERROR: routed MoE probe layer was not found\n");
                return 2;
            }
            if (!parse_mxfp4_expert_info(expertLayout, probeLayerDict, &probeMoeInfo)) {
                return 2;
            }
            if (options.probe_router_moe || options.probe_decoder_layer) {
                if (routerInfo.num_experts != probeFile->num_experts) {
                    fprintf(stderr,
                            "ERROR: router experts %u do not match layer expert count %llu\n",
                            routerInfo.num_experts,
                            (unsigned long long)probeFile->num_experts);
                    return 2;
                }
                if (routerInfo.hidden_dim != probeMoeInfo.hidden_dim) {
                    fprintf(stderr,
                            "ERROR: router hidden dim %u does not match expert hidden dim %u\n",
                            routerInfo.hidden_dim,
                            probeMoeInfo.hidden_dim);
                    return 2;
                }
            }
            probeMoeScratchBytes =
                (uint64_t)probeMoeInfo.hidden_dim * sizeof(float) * 2u +
                (uint64_t)probeMoeInfo.intermediate_dim * sizeof(float);
        }
        if (options.include_shared_expert && !options.probe_decode_layers) {
            if (!parse_shared_mxfp4_info(residentLayout, options.probe_layer, &sharedMoeInfo)) {
                return 2;
            }
            if (sharedMoeInfo.local.hidden_dim != probeMoeInfo.hidden_dim ||
                sharedMoeInfo.local.intermediate_dim != probeMoeInfo.intermediate_dim ||
                sharedMoeInfo.local.group_size != probeMoeInfo.group_size) {
                fprintf(stderr,
                        "ERROR: shared expert dims hidden=%u intermediate=%u group=%u "
                        "do not match routed hidden=%u intermediate=%u group=%u\n",
                        sharedMoeInfo.local.hidden_dim,
                        sharedMoeInfo.local.intermediate_dim,
                        sharedMoeInfo.local.group_size,
                        probeMoeInfo.hidden_dim,
                        probeMoeInfo.intermediate_dim,
                        probeMoeInfo.group_size);
                return 2;
            }
            sharedExpertStorageBytes = sharedMoeInfo.total_bytes;
        }
        if (options.probe_mlp_block || options.probe_decoder_layer) {
            if (!find_layer_vector_info(residentLayout,
                                        options.probe_layer,
                                        @".post_attention_layernorm.weight",
                                        &rmsNormInfo)) {
                return 2;
            }
            if (rmsNormInfo.dim != routerInfo.hidden_dim ||
                rmsNormInfo.dim != probeMoeInfo.hidden_dim) {
                fprintf(stderr,
                        "ERROR: RMSNorm dim %u does not match router/expert hidden dims %u/%u\n",
                        rmsNormInfo.dim,
                        routerInfo.hidden_dim,
                        probeMoeInfo.hidden_dim);
                return 2;
            }
            rmsNormScratchBytes =
                (uint64_t)rmsNormInfo.dim * sizeof(float) * 3u +
                rmsNormInfo.size;
        }
        if (options.probe_decode_layers) {
            decodeLayerPlans = (DecodeLayerPlan *)calloc((size_t)decodeLayers.count,
                                                         sizeof(DecodeLayerPlan));
            decodeLayerSummaries = (DecodeLayerRunSummary *)calloc(
                (size_t)decodeLayers.count,
                sizeof(DecodeLayerRunSummary)
            );
            if (!decodeLayerPlans || !decodeLayerSummaries) {
                fprintf(stderr, "ERROR: out of memory allocating decode layer plans\n");
                return 1;
            }
            LoaderOptions planOptions = options;
            if (options.generate_steps > 0 && promptTokenIds.count > 0) {
                planOptions.context_length =
                    options.context_length +
                    promptTokenIds.count +
                    options.generate_steps - 2;
            } else if (options.generate_steps > 0) {
                planOptions.context_length =
                    options.context_length + options.generate_steps - 1;
            }
            if (!build_decode_layer_plans(residentLayout,
                                          expertLayout,
                                          layers,
                                          expertFiles,
                                          [layers count],
                                          decodeLayers,
                                          planOptions,
                                          [NSString stringWithUTF8String:options.cache_layout],
                                          maxCacheFileBytes,
                                          maxCacheReadBytes,
                                          ropeSplitScratchBytes,
                                          decodeLayerPlans,
                                          &decodeLayersScratchBytes,
                                          &decodeLayersHasMoe)) {
                return 2;
            }
            if (options.cache_mla_kv_b_f32) {
                for (int i = 0; i < decodeLayers.count; i++) {
                    if (!checked_add_u64(mlaKVBMemoryCachePlannedBytes,
                                         decodeLayerPlans[i].mla_value_source.kv_b_f32_bytes,
                                         &mlaKVBMemoryCachePlannedBytes)) {
                        fprintf(stderr, "ERROR: MLA KV-B cache planned bytes overflow\n");
                        return 2;
                    }
                }
            }
        }
        if (options.probe_final_logits || decodeLayersFinalLogits) {
            if (!find_global_mxfp4_matrix_info_by_names(
                    residentLayout,
                    @[@"lm_head.weight", @"model.lm_head.weight"],
                    &finalLogitsHeadInfo,
                    "lm_head")) {
                return 2;
            }
            if (!options.skip_final_norm) {
                if (!find_global_vector_info_by_names(
                        residentLayout,
                        @[@"model.norm.weight",
                          @"transformer.norm.weight",
                          @"norm.weight"],
                        &finalNormInfo,
                        "final norm")) {
                    return 2;
                }
                if (finalNormInfo.dim != finalLogitsHeadInfo.in_dim) {
                    fprintf(stderr,
                            "ERROR: final norm dim %u does not match lm_head input dim %u\n",
                            finalNormInfo.dim,
                            finalLogitsHeadInfo.in_dim);
                    return 2;
                }
            }
            uint64_t weightRowBytes =
                (uint64_t)finalLogitsHeadInfo.weight.dim1 * sizeof(uint32_t);
            uint64_t scaleRowBytes =
                (uint64_t)finalLogitsHeadInfo.scales.dim1 * sizeof(uint8_t);
            uint64_t rowBytes = 0;
            if (!checked_add_u64(weightRowBytes, scaleRowBytes, &rowBytes) ||
                rowBytes == 0 ||
                rowBytes > maxFinalLogitsChunkBytes) {
                fprintf(stderr, "ERROR: final logits lm_head row exceeds chunk cap\n");
                return 2;
            }
            int finalLogitsUsesMappedWeights =
                options.mmap_final_logits && !options.wrap_resident_metal;
            if (finalLogitsUsesMappedWeights) {
                finalLogitsMmapBytes = finalLogitsHeadInfo.total_bytes;
            }
            uint64_t rowsPerChunk = options.final_logits_chunk_rows > 0
                ? (uint64_t)options.final_logits_chunk_rows
                : ((finalLogitsUsesMappedWeights || options.wrap_resident_metal)
                    ? (uint64_t)finalLogitsHeadInfo.out_dim
                    : (maxFinalLogitsChunkBytes / rowBytes));
            if (rowsPerChunk == 0) {
                rowsPerChunk = 1;
            }
            if (rowsPerChunk > finalLogitsHeadInfo.out_dim) {
                rowsPerChunk = finalLogitsHeadInfo.out_dim;
            }
            uint64_t chunkRawBytes = 0;
            if (!checked_mul_u64(rowsPerChunk, rowBytes, &chunkRawBytes)) {
                fprintf(stderr, "ERROR: final logits chunk size overflows\n");
                return 2;
            }
            uint64_t chunkAllocBytes =
                (finalLogitsUsesMappedWeights || options.wrap_resident_metal)
                    ? 0
                    : round_up_u64(chunkRawBytes, 2 * 1024 * 1024);
            finalLogitsScratchBytes =
                chunkAllocBytes +
                finalLogitsMmapBytes +
                (uint64_t)finalLogitsHeadInfo.in_dim * sizeof(float) +
                rowsPerChunk * sizeof(float);
            if (!options.skip_final_norm) {
                finalLogitsScratchBytes +=
                    (uint64_t)finalNormInfo.dim * sizeof(float) * 3u +
                    finalNormInfo.size;
            }
            if (options.output_next_input_f32 ||
                options.generate_steps > 0 ||
                options.input_token_id >= 0) {
                if (!find_global_mxfp4_matrix_info_by_names(
                        residentLayout,
                        @[@"model.embed_tokens.weight",
                          @"transformer.word_embeddings.weight"],
                        &nextInputEmbeddingInfo,
                        "embedding")) {
                    return 2;
                }
                if (nextInputEmbeddingInfo.in_dim != finalLogitsHeadInfo.in_dim) {
                    fprintf(stderr,
                            "ERROR: embedding hidden dim %u does not match lm_head input dim %u\n",
                            nextInputEmbeddingInfo.in_dim,
                            finalLogitsHeadInfo.in_dim);
                    return 2;
                }
                uint64_t embeddingWeightRowBytes =
                    (uint64_t)nextInputEmbeddingInfo.weight.dim1 * sizeof(uint32_t);
                uint64_t embeddingScaleRowBytes =
                    (uint64_t)nextInputEmbeddingInfo.scales.dim1;
                uint64_t embeddingRowBytes = 0;
                uint64_t embeddingOutputBytes =
                    (uint64_t)nextInputEmbeddingInfo.in_dim * sizeof(float);
                if (!checked_add_u64(embeddingWeightRowBytes,
                                     embeddingScaleRowBytes,
                                     &embeddingRowBytes) ||
                    embeddingRowBytes == 0 ||
                    embeddingRowBytes > maxEmbeddingRowBytes ||
                    !checked_add_u64(embeddingRowBytes,
                                     embeddingOutputBytes,
                                     &nextInputEmbeddingScratchBytes)) {
                    fprintf(stderr,
                            "ERROR: generated-token embedding row exceeds configured cap\n");
                    return 2;
                }
            }
        }
        int activeExpertBufferCount = options.expert_buffer_count;
        if (!options.probe_expert_read &&
            !options.probe_layer_moe &&
            !options.probe_router_moe &&
            (options.probe_router ||
	             options.probe_resident_linear ||
	             options.probe_attn_projections ||
	             options.probe_rope_split ||
	             options.probe_mla_attention ||
	             options.probe_attn_output ||
                 options.probe_dense_mlp_block ||
                 options.probe_dense_decoder_layer ||
                 (options.probe_decode_layers && !decodeLayersHasMoe) ||
                 options.probe_final_logits) &&
            !options.probe_decoder_layer) {
            activeExpertBufferCount = 0;
        } else if ((options.probe_layer_moe ||
                    options.probe_router_moe ||
                    options.probe_decoder_layer ||
                    (options.probe_decode_layers && decodeLayersHasMoe)) &&
	                   !options.probe_expert_read) {
            int routeSlotNeed = options.top_k + (options.include_shared_expert ? 1 : 0);
            if (options.probe_layer_moe && !options.probe_router_moe && probeExperts.count > 0) {
                routeSlotNeed = probeExperts.count + (options.include_shared_expert ? 1 : 0);
            }
            if (routeSlotNeed < 1) {
                routeSlotNeed = 1;
            }
            activeExpertBufferCount =
                options.expert_buffer_count < routeSlotNeed
                    ? options.expert_buffer_count
                    : routeSlotNeed;
        }
        uint64_t expertBufferBytes = round_up_u64(maxExpertSlotBytes, 2 * 1024 * 1024);
        if (options.include_shared_expert && sharedExpertStorageBytes > expertBufferBytes) {
            fprintf(stderr,
                    "ERROR: shared expert bytes %llu exceed reusable expert buffer %llu\n",
                    (unsigned long long)sharedExpertStorageBytes,
                    (unsigned long long)expertBufferBytes);
            return 2;
        }
        if (options.probe_decode_layers && options.include_shared_expert) {
            for (int i = 0; i < decodeLayers.count; i++) {
                if (decodeLayerPlans[i].shared_storage_bytes > expertBufferBytes) {
                    fprintf(stderr,
                            "ERROR: decode layer %d shared expert bytes %llu exceed reusable expert buffer %llu\n",
                            decodeLayerPlans[i].layer,
                            (unsigned long long)decodeLayerPlans[i].shared_storage_bytes,
                            (unsigned long long)expertBufferBytes);
                    return 2;
                }
            }
        }

        int runtimeExpertBufferCountBefore =
            runtime.expertBufferBytes == expertBufferBytes
                ? (int)[runtime.expertBuffers count]
                : 0;
        int liveExpertBufferCount = activeExpertBufferCount > runtimeExpertBufferCountBefore
            ? activeExpertBufferCount
            : runtimeExpertBufferCountBefore;
        uint64_t reusableExpertBytes = expertBufferBytes * (uint64_t)liveExpertBufferCount;
        uint64_t estimatedLiveBytes = reusableExpertBytes;
        if (options.mmap_resident || options.wrap_resident_metal) {
            estimatedLiveBytes += residentFileBytes;
        }
        estimatedLiveBytes += probeMoeScratchBytes;
        estimatedLiveBytes += routerScratchBytes;
        estimatedLiveBytes += rmsNormScratchBytes;
        estimatedLiveBytes += residentLinearScratchBytes;
        estimatedLiveBytes += attnProjectionScratchBytes;
        estimatedLiveBytes += ropeSplitScratchBytes;
        estimatedLiveBytes += mlaAttentionScratchBytes;
        estimatedLiveBytes += attnOutputScratchBytes;
        estimatedLiveBytes += denseMlpScratchBytes;
        estimatedLiveBytes += decodeLayersScratchBytes;
        estimatedLiveBytes += finalLogitsScratchBytes;
        estimatedLiveBytes += nextInputEmbeddingScratchBytes;
        estimatedLiveBytes += decodeCacheMemoryBytes;
        estimatedLiveBytes += mlaKVBMemoryCacheLiveEstimateBytes;
        if (options.probe_decoder_layer) {
            decoderLayerScratchBytes =
                attnProjectionScratchBytes +
                ropeSplitScratchBytes +
                mlaAttentionScratchBytes +
                attnOutputScratchBytes +
                rmsNormScratchBytes +
                routerScratchBytes +
                probeMoeScratchBytes;
            estimatedLiveBytes += 0;
        }
        if (options.probe_dense_decoder_layer) {
            denseDecoderLayerScratchBytes =
                attnProjectionScratchBytes +
                ropeSplitScratchBytes +
                mlaAttentionScratchBytes +
                attnOutputScratchBytes +
                denseMlpScratchBytes;
            estimatedLiveBytes += 0;
        }
        int liveOk = 1;
        if (options.max_live_working_set_mib > 0.0) {
            double capBytes = options.max_live_working_set_mib * 1024.0 * 1024.0;
            liveOk = ((double)estimatedLiveBytes <= capBytes);
        }
        uint64_t minFreeUnifiedMemoryBytes = 0;
        uint64_t requiredAvailableMemoryBytes = 0;
        SystemMemorySnapshot memorySnapshot = {0};
        int freeUnifiedMemoryOk = 1;
        int systemMemorySnapshotOk = 0;
        if (options.min_free_unified_memory_gib > 0.0) {
            if (!double_gib_to_u64_bytes(options.min_free_unified_memory_gib,
                                         &minFreeUnifiedMemoryBytes)) {
                freeUnifiedMemoryOk = 0;
                fprintf(stderr,
                        "ERROR: --min-free-unified-memory-gib could not be converted to bytes\n");
            } else if (!read_system_memory_snapshot(&memorySnapshot)) {
                freeUnifiedMemoryOk = 0;
                fprintf(stderr,
                        "ERROR: could not inspect system available memory for --min-free-unified-memory-gib\n");
            } else {
                systemMemorySnapshotOk = 1;
                if (estimatedLiveBytes > UINT64_MAX - minFreeUnifiedMemoryBytes) {
                    freeUnifiedMemoryOk = 0;
                    requiredAvailableMemoryBytes = UINT64_MAX;
                    fprintf(stderr,
                            "ERROR: required available memory overflows uint64\n");
                } else {
                    requiredAvailableMemoryBytes =
                        estimatedLiveBytes + minFreeUnifiedMemoryBytes;
                    if (memorySnapshot.available_bytes < requiredAvailableMemoryBytes) {
                        freeUnifiedMemoryOk = 0;
                        fprintf(stderr,
                                "ERROR: available unified memory %llu bytes is below required %llu bytes\n",
                                (unsigned long long)memorySnapshot.available_bytes,
                                (unsigned long long)requiredAvailableMemoryBytes);
                    }
                }
            }
        }
        int admissionOk = liveOk && freeUnifiedMemoryOk;

        NSString *runtimeDecodeCachePath = options.cache_file
            ? [NSString stringWithUTF8String:options.cache_file]
            : nil;
        NSString *runtimeDecodeCacheLayoutPath = options.cache_layout
            ? [NSString stringWithUTF8String:options.cache_layout]
            : nil;
        int decodeCacheFdReused = 0;
        int decodeCacheFdReady = 0;
        int decodeCacheMemoryReused = 0;
        int decodeCacheMemoryReady = 0;
        if (admissionOk && runtimeDecodeCachePath) {
            if (options.in_memory_decode_cache) {
                decodeCacheMemoryReused =
                    runtime.decodeCacheMemory != nil &&
                    runtime.decodeCacheMemoryPath &&
                    [runtime.decodeCacheMemoryPath isEqualToString:runtimeDecodeCachePath];
                if (!ensure_runtime_decode_cache_memory(runtime,
                                                        runtimeDecodeCacheLayoutPath,
                                                        runtimeDecodeCachePath,
                                                        maxCacheFileBytes)) {
                    return 1;
                }
                decodeCacheMemoryReady =
                    runtime.decodeCacheMemory != nil &&
                    runtime.decodeCacheMemoryPath &&
                    [runtime.decodeCacheMemoryPath isEqualToString:runtimeDecodeCachePath];
            } else {
                decodeCacheFdReused =
                    runtime.decodeCacheFd >= 0 &&
                    runtime.decodeCachePath &&
                    [runtime.decodeCachePath isEqualToString:runtimeDecodeCachePath];
                if (!ensure_runtime_decode_cache_file(runtime, runtimeDecodeCachePath)) {
                    return 1;
                }
                decodeCacheFdReady =
                    runtime.decodeCacheFd >= 0 &&
                    runtime.decodeCachePath &&
                    [runtime.decodeCachePath isEqualToString:runtimeDecodeCachePath];
            }
        }

        int residentFd = -1;
        void *residentMap = MAP_FAILED;
        id<MTLBuffer> residentMetalBuffer = nil;
        if (admissionOk && options.mmap_resident) {
            residentFd = open([residentBinPath fileSystemRepresentation], O_RDONLY);
            if (residentFd < 0) {
                fprintf(
                    stderr,
                    "ERROR: failed to open resident file %s: %s\n",
                    [residentBinPath UTF8String],
                    strerror(errno)
                );
                return 1;
            }
            residentMap = mmap(NULL, residentFileBytes, PROT_READ, MAP_PRIVATE, residentFd, 0);
            if (residentMap == MAP_FAILED) {
                fprintf(stderr, "ERROR: mmap resident failed: %s\n", strerror(errno));
                return 1;
            }
            if (options.wrap_resident_metal) {
                uint64_t pageAligned = round_up_u64(residentFileBytes, 16384);
                residentMetalBuffer = [device newBufferWithBytesNoCopy:residentMap
                                                                 length:(NSUInteger)pageAligned
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
                if (!residentMetalBuffer) {
                    fprintf(stderr, "ERROR: failed to wrap resident mmap as Metal buffer\n");
                    return 1;
                }
            }
        }
        NSArray *expertBuffers = @[];
        int expertBufferPoolReused = 0;
        if (admissionOk) {
            if (!ensure_runtime_expert_buffers(runtime,
                                               expertBufferBytes,
                                               activeExpertBufferCount)) {
                return 1;
            }
            expertBufferPoolReused =
                activeExpertBufferCount > 0 &&
                runtimeExpertBufferCountBefore >= activeExpertBufferCount;
            if (activeExpertBufferCount > 0) {
                expertBuffers = [runtime.expertBuffers subarrayWithRange:NSMakeRange(
                    0,
                    (NSUInteger)activeExpertBufferCount
                )];
            }
        }
        NSString *resolvedInputPath = options.input_f32
            ? [NSString stringWithUTF8String:options.input_f32]
            : nil;
        NSData *initialInputData = nil;
        EmbeddingLookupStats inputTokenEmbeddingStats = {0};
        int inputTokenEmbeddingOk = 1;
        if (options.input_token_id >= 0) {
            if (!admissionOk) {
                inputTokenEmbeddingOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --input-token-id embedding\n");
            } else {
                inputTokenEmbeddingOk = write_generated_embedding_f32(
                    residentBinPath,
                    nextInputEmbeddingInfo,
                    (uint64_t)options.input_token_id,
                    maxEmbeddingRowBytes,
                    nil,
                    &initialInputData,
                    &inputTokenEmbeddingStats
                );
            }
        }

        int probeOk = 1;
        ProbeReadResult *probeResults = NULL;
        NSUInteger probeReadLayerCount = options.probe_all_layers ? [layers count] : 1;
        size_t probeResultCount = 0;
        uint64_t probeBytesRead = 0;
        double probeElapsedSeconds = 0.0;
        double probeThroughputGiBPerSecond = 0.0;
        DirectExpertReadStats probeReadStats = {0};
        if (options.probe_expert_read) {
            probeResultCount = (size_t)probeReadLayerCount * (size_t)probeExperts.count;
            if (!admissionOk) {
                probeOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-expert-read\n");
            } else {
                probeResults = (ProbeReadResult *)calloc(probeResultCount, sizeof(ProbeReadResult));
                if (!probeResults) {
                    fprintf(stderr, "ERROR: out of memory\n");
                    return 1;
                }
                double start = now_seconds();
                for (int repeat = 0; repeat < options.probe_repeat; repeat++) {
                    for (NSUInteger layerOrdinal = 0; layerOrdinal < probeReadLayerCount; layerOrdinal++) {
                        ExpertFile *currentProbeFile =
                            options.probe_all_layers ? &expertFiles[layerOrdinal] : probeFile;
                        if (!read_direct_expert_slots(currentProbeFile,
                                                      probeExperts,
                                                      0,
                                                      probeExperts.count,
                                                      expertBuffers,
                                                      &probeReadStats)) {
                            return 1;
                        }
                        if (repeat + 1 == options.probe_repeat) {
                            for (int i = 0; i < probeExperts.count; i++) {
                                int expertId = probeExperts.values[i];
                                id<MTLBuffer> buffer =
                                    [expertBuffers objectAtIndex:(NSUInteger)i];
                                void *dst = [buffer contents];
                                if (!dst) {
                                    fprintf(stderr,
                                            "ERROR: expert Metal buffer has no CPU-visible contents\n");
                                    return 1;
                                }
                                size_t resultIndex =
                                    (size_t)layerOrdinal * (size_t)probeExperts.count + (size_t)i;
                                probeResults[resultIndex].layer = currentProbeFile->layer;
                                probeResults[resultIndex].expert = expertId;
                                probeResults[resultIndex].bytes =
                                    currentProbeFile->expert_slot_bytes;
                                probeResults[resultIndex].sample_checksum =
                                    sampled_fnv1a64(dst,
                                                    currentProbeFile->expert_slot_bytes);
                            }
                        }
                    }
                }
                probeBytesRead = probeReadStats.bytes_read;
                probeElapsedSeconds = now_seconds() - start;
                if (probeElapsedSeconds > 0.0) {
                    probeThroughputGiBPerSecond =
                        (probeBytesRead / 1073741824.0) / probeElapsedSeconds;
                }
            }
        }

        ResidentLinearProbeStats residentLinearStats = {0};
        int residentLinearOk = 1;
        if (options.probe_resident_linear) {
            if (!admissionOk) {
                residentLinearOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-resident-linear\n");
            } else {
                residentLinearOk = run_resident_linear_probe(
                    device,
                    residentBinPath,
                    residentMetalBuffer,
                    residentLinearInfo,
                    [NSString stringWithUTF8String:options.input_f32],
                    nil,
                    [NSString stringWithUTF8String:options.output_f32],
                    NULL,
                    options.expect_output0_set,
                    options.expect_output0,
                    &residentLinearStats
                );
            }
        }

        AttnProjectionProbeStats attnProjectionStats = {0};
        int attnProjectionOk = 1;
        if (options.probe_attn_projections) {
            if (!admissionOk) {
                attnProjectionOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-attn-projections\n");
            } else {
                attnProjectionOk = run_attention_projection_probe(
                    device,
                    residentBinPath,
                    options.probe_layer,
                    attnInputNormInfo,
                    attnQANormInfo,
                    attnKVANormInfo,
                    attnQAInfo,
                    attnQBInfo,
                    attnKVAInfo,
                    attnHasKVB,
                    attnKVBInfo,
                    [NSString stringWithUTF8String:options.input_f32],
                    nil,
                    nil,
                    [NSString stringWithUTF8String:options.output_dir],
                    appendCache ? [NSString stringWithUTF8String:options.cache_layout] : nil,
                    appendCache ? [NSString stringWithUTF8String:options.cache_file] : nil,
                    appendCache ? (uint64_t)options.cache_position : 0,
                    appendCache,
                    maxCacheFileBytes,
                    (float)options.rms_norm_eps,
                    1,
                    NULL,
                    NULL,
                    nil,
                    nil,
                    &attnProjectionStats
                );
                attnProjectionStats.scratch_bytes = attnProjectionScratchBytes;
            }
        }

        RopeSplitProbeStats ropeSplitStats = {0};
        int ropeSplitOk = 1;
        if (options.probe_rope_split) {
            if (!admissionOk) {
                ropeSplitOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-rope-split\n");
            } else {
                ropeSplitOk = run_rope_split_probe(
                    device,
                    [NSString stringWithUTF8String:options.q_b_f32],
                    [NSString stringWithUTF8String:options.k_f32],
                    nil,
                    nil,
                    [NSString stringWithUTF8String:options.output_q_nope_f32],
                    [NSString stringWithUTF8String:options.output_q_rope_f32],
                    [NSString stringWithUTF8String:options.output_q_f32],
                    [NSString stringWithUTF8String:options.output_k_f32],
                    (uint32_t)options.num_heads,
                    (uint32_t)options.qk_nope_dim,
                    (uint32_t)options.rope_dim,
                    (uint32_t)options.start_position,
                    (uint32_t)options.batch_tokens,
                    (float)options.rope_theta,
                    options.rope_interleave,
                    NULL,
                    NULL,
                    &ropeSplitStats
                );
            }
        }

        MlaAttentionProbeStats mlaAttentionStats = {0};
        int mlaAttentionOk = 1;
        if (options.probe_mla_attention) {
            if (!admissionOk) {
                mlaAttentionOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-mla-attention\n");
            } else {
                mlaAttentionOk = run_mla_attention_probe(
                    device,
                    residentBinPath,
                    mlaValueSource,
                    [NSString stringWithUTF8String:options.cache_layout],
                    [NSString stringWithUTF8String:options.cache_file],
                    options.probe_layer,
                    [NSString stringWithUTF8String:options.q_nope_f32],
                    [NSString stringWithUTF8String:options.q_rope_f32],
                    options.mla_kv_b_f32 ? [NSString stringWithUTF8String:options.mla_kv_b_f32] : nil,
                    nil,
                    nil,
                    [NSString stringWithUTF8String:options.output_f32],
                    (uint32_t)options.context_length,
                    (uint32_t)options.num_heads,
                    (uint32_t)options.qk_nope_dim,
                    (uint32_t)options.rope_dim,
                    (uint32_t)options.v_head_dim,
                    (uint32_t)options.cache_position_offset,
                    (float)options.attention_scale,
                    (float)options.rope_theta,
                    options.rope_interleave,
                    maxCacheFileBytes,
                    maxCacheReadBytes,
                    NULL,
                    &mlaAttentionStats
                );
                mlaAttentionStats.scratch_bytes = mlaAttentionScratchBytes;
            }
        }

        AttnOutputProbeStats attnOutputStats = {0};
        int attnOutputOk = 1;
        if (options.probe_attn_output) {
            if (!admissionOk) {
                attnOutputOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-attn-output\n");
            } else {
                attnOutputOk = run_attention_output_probe(
                    device,
                    residentBinPath,
                    residentMetalBuffer,
                    attnOutputInfo,
                    [NSString stringWithUTF8String:options.input_f32],
                    nil,
                    [NSString stringWithUTF8String:options.residual_f32],
                    nil,
                    options.projection_f32
                        ? [NSString stringWithUTF8String:options.projection_f32]
                        : nil,
                    [NSString stringWithUTF8String:options.output_f32],
                    NULL,
                    nil,
                    &attnOutputStats
                );
                attnOutputStats.scratch_bytes = attnOutputScratchBytes;
            }
        }

        DenseMlpProbeStats denseMlpStats = {0};
        int denseMlpOk = 1;
        if (options.probe_dense_mlp_block) {
            if (!admissionOk) {
                denseMlpOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-dense-mlp-block\n");
            } else {
                denseMlpOk = run_dense_mlp_probe(
                    device,
                    residentBinPath,
                    options.probe_layer,
                    rmsNormInfo,
                    denseMlpInfo,
                    [NSString stringWithUTF8String:options.input_f32],
                    nil,
                    nil,
                    [NSString stringWithUTF8String:options.output_f32],
                    NULL,
                    nil,
                    (float)options.rms_norm_eps,
                    options.expect_output0_set,
                    options.expect_output0,
                    0,
                    nil,
                    &denseMlpStats
                );
                denseMlpStats.scratch_bytes = denseMlpScratchBytes;
            }
        }

        RmsNormProbeStats rmsNormStats = {0};
        NSData *mlpNormedData = nil;
        NSData *mlpResidualData = nil;
        int rmsNormOk = 1;
        if (options.probe_mlp_block) {
            if (!admissionOk) {
                rmsNormOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-mlp-block RMSNorm\n");
            } else {
                mlpResidualData = [NSData dataWithContentsOfFile:
                    [NSString stringWithUTF8String:options.input_f32]];
                if (!mlpResidualData) {
                    rmsNormOk = 0;
                    fprintf(stderr,
                            "ERROR: failed to read MLP residual input %s\n",
                            options.input_f32);
                } else {
                    rmsNormOk = run_rmsnorm_probe(
                        device,
                        residentBinPath,
                        rmsNormInfo,
                        [NSString stringWithUTF8String:options.input_f32],
                        nil,
                        (float)options.rms_norm_eps,
                        &mlpNormedData,
                        &rmsNormStats
                    );
                }
            }
        }

        RouterProbeStats routerStats = {0};
        int routerOk = 1;
        if (options.probe_router) {
            if (!admissionOk) {
                routerOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-router\n");
            } else if (options.probe_mlp_block && !rmsNormOk) {
                routerOk = 0;
                fprintf(stderr,
                        "ERROR: refusing MLP router because RMSNorm failed\n");
            } else {
                routerOk = run_router_probe(
                    device,
                    residentBinPath,
                    routerInfo,
                    routerBiasInfo,
                    routerTopKOptions,
                    options.probe_layer,
                    [NSString stringWithUTF8String:options.input_f32],
                    options.probe_mlp_block ? mlpNormedData : nil,
                    (uint32_t)options.top_k,
                    options.output_router_json
                        ? [NSString stringWithUTF8String:options.output_router_json]
                        : nil,
                    0,
                    &routerStats
                );
            }
        }

        LayerMoeProbeStats layerMoeStats = {0};
        int layerMoeOk = 1;
        if (options.probe_layer_moe || options.probe_router_moe) {
            if (!admissionOk) {
                layerMoeOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing routed MoE probe\n");
            } else if (options.probe_router_moe && !routerOk) {
                layerMoeOk = 0;
                fprintf(stderr,
                        "ERROR: refusing --probe-router-moe because router probe failed\n");
            } else {
                IntList routeExperts = probeExperts;
                FloatList routeWeights = probeWeights;
                if (options.probe_router_moe) {
                    routeExperts.values = routerStats.experts;
                    routeExperts.count = (int)routerStats.top_k;
                    routeWeights.values = routerStats.weights;
                    routeWeights.count = (int)routerStats.top_k;
                    for (uint32_t i = 0; i < routerStats.top_k; i++) {
                        if ((uint64_t)routerStats.experts[i] >= probeFile->num_experts) {
                            fprintf(stderr,
                                    "ERROR: router selected expert %d outside layer expert count %llu\n",
                                    routerStats.experts[i],
                                    (unsigned long long)probeFile->num_experts);
                            layerMoeOk = 0;
                            break;
                        }
                    }
                }
                if (!layerMoeOk) {
                    /* error already reported */
                } else {
                layerMoeOk = run_layer_moe_probe(
                    device,
                    probeFile,
                    probeMoeInfo,
                    routeExperts,
                    routeWeights,
                    [NSString stringWithUTF8String:options.input_f32],
                    options.probe_mlp_block ? mlpNormedData : nil,
                    nil,
                    options.probe_mlp_block ? mlpResidualData : nil,
                    nil,
                    residentBinPath,
                    options.include_shared_expert ? &sharedMoeInfo : NULL,
                    [NSString stringWithUTF8String:options.output_f32],
                    NULL,
                    nil,
                    options.expect_output0_set,
                    options.expect_output0,
                    expertBuffers,
                    0,
                    0,
                    NULL,
                    0,
                    nil,
                    &layerMoeStats
                );
                }
            }
        }

        DecoderLayerProbeStats decoderLayerStats = {0};
        int decoderLayerOk = 1;
        if (options.probe_decoder_layer) {
            if (!admissionOk) {
                decoderLayerOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-decoder-layer\n");
            } else {
                decoderLayerOk = run_decoder_layer_probe(
                    device,
                    residentBinPath,
                    options.probe_layer,
                    attnInputNormInfo,
                    attnQANormInfo,
                    attnKVANormInfo,
                    rmsNormInfo,
                    attnQAInfo,
                    attnQBInfo,
                    attnKVAInfo,
                    attnHasKVB,
                    attnKVBInfo,
                    mlaValueSource,
                    attnOutputInfo,
                    routerInfo,
                    routerBiasInfo,
                    routerTopKOptions,
                    probeFile,
                    probeMoeInfo,
                    options.include_shared_expert ? &sharedMoeInfo : NULL,
                    [NSString stringWithUTF8String:options.input_f32],
                    nil,
                    nil,
                    [NSString stringWithUTF8String:options.output_dir],
                    [NSString stringWithUTF8String:options.cache_layout],
                    [NSString stringWithUTF8String:options.cache_file],
                    options.context1_o_proj_cache_layout
                        ? [NSString stringWithUTF8String:options.context1_o_proj_cache_layout]
                        : nil,
                    options.context1_o_proj_cache_file
                        ? [NSString stringWithUTF8String:options.context1_o_proj_cache_file]
                        : nil,
                    (uint64_t)options.cache_position,
                    (uint32_t)options.context_length,
                    (uint32_t)options.num_heads,
                    (uint32_t)options.qk_nope_dim,
                    (uint32_t)options.rope_dim,
                    (uint32_t)options.v_head_dim,
                    (uint32_t)options.cache_position_offset,
                    (float)options.attention_scale,
                    (float)options.rope_theta,
                    options.rope_interleave,
                    maxCacheFileBytes,
                    maxCacheReadBytes,
                    (uint32_t)options.top_k,
                    (float)options.rms_norm_eps,
                    options.output_router_json
                        ? [NSString stringWithUTF8String:options.output_router_json]
                        : nil,
                    [NSString stringWithUTF8String:options.output_f32],
                    NULL,
                    nil,
                    1,
                    expertBuffers,
                    residentMetalBuffer,
                    0,
                    nil,
                    &decoderLayerStats
                );
                decoderLayerStats.scratch_bytes = decoderLayerScratchBytes;
            }
        }

        DenseDecoderLayerProbeStats denseDecoderLayerStats = {0};
        int denseDecoderLayerOk = 1;
        if (options.probe_dense_decoder_layer) {
            if (!admissionOk) {
                denseDecoderLayerOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-dense-decoder-layer\n");
            } else {
                denseDecoderLayerOk = run_dense_decoder_layer_probe(
                    device,
                    residentBinPath,
                    options.probe_layer,
                    attnInputNormInfo,
                    attnQANormInfo,
                    attnKVANormInfo,
                    rmsNormInfo,
                    attnQAInfo,
                    attnQBInfo,
                    attnKVAInfo,
                    attnHasKVB,
                    attnKVBInfo,
                    mlaValueSource,
                    attnOutputInfo,
                    denseMlpInfo,
                    [NSString stringWithUTF8String:options.input_f32],
                    nil,
                    nil,
                    [NSString stringWithUTF8String:options.output_dir],
                    [NSString stringWithUTF8String:options.cache_layout],
                    [NSString stringWithUTF8String:options.cache_file],
                    options.context1_o_proj_cache_layout
                        ? [NSString stringWithUTF8String:options.context1_o_proj_cache_layout]
                        : nil,
                    options.context1_o_proj_cache_file
                        ? [NSString stringWithUTF8String:options.context1_o_proj_cache_file]
                        : nil,
                    (uint64_t)options.cache_position,
                    (uint32_t)options.context_length,
                    (uint32_t)options.num_heads,
                    (uint32_t)options.qk_nope_dim,
                    (uint32_t)options.rope_dim,
                    (uint32_t)options.v_head_dim,
                    (uint32_t)options.cache_position_offset,
                    (float)options.attention_scale,
                    (float)options.rope_theta,
                    options.rope_interleave,
                    maxCacheFileBytes,
                    maxCacheReadBytes,
                    (float)options.rms_norm_eps,
                    [NSString stringWithUTF8String:options.output_f32],
                    NULL,
                    nil,
                    1,
                    residentMetalBuffer,
                    0,
                    nil,
                    &denseDecoderLayerStats
                );
                denseDecoderLayerStats.scratch_bytes = denseDecoderLayerScratchBytes;
            }
        }

        int decodeLayersOk = 1;
        float decodeLayersOutput0 = 0.0f;
        double decodeLayersElapsedSeconds = 0.0;
        NSData *decodeLayersFinalOutputData = nil;
        FinalLogitsProbeStats finalLogitsStats = {0};
        EmbeddingLookupStats nextInputEmbeddingStats = {0};
        int finalLogitsOk = 1;
        int decodeLayerGenerateMode =
            options.probe_decode_layers && options.generate_steps > 0;
        NSMutableArray *generateStepPayloads = nil;
        NSMutableArray *promptPrefillStepPayloads = nil;
        if (options.probe_decode_layers) {
            if (!admissionOk) {
                decodeLayersOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-decode-layers\n");
            } else if (!inputTokenEmbeddingOk) {
                decodeLayersOk = 0;
                fprintf(stderr,
                        "ERROR: refusing generate/decode-layers because --input-token-id embedding failed\n");
            } else if (decodeLayerGenerateMode) {
                generateStepPayloads =
                    [NSMutableArray arrayWithCapacity:(NSUInteger)options.generate_steps];
                NSString *currentGenerateInputPath = resolvedInputPath;
                NSData *currentGenerateInputData = initialInputData;
                if (promptTokenIds.count > 0) {
                    promptPrefillStepPayloads =
                        [NSMutableArray arrayWithCapacity:(NSUInteger)promptTokenIds.count];
                    currentGenerateInputPath = nil;
                    currentGenerateInputData = nil;
                    for (int promptIndex = 0; promptIndex < promptTokenIds.count; promptIndex++) {
                        @autoreleasepool {
                            EmbeddingLookupStats promptEmbeddingStats = {0};
                            NSData *promptInputData = nil;
                            if (!write_generated_embedding_f32(
                                    residentBinPath,
                                    nextInputEmbeddingInfo,
                                    (uint64_t)promptTokenIds.values[promptIndex],
                                    maxEmbeddingRowBytes,
                                    nil,
                                    &promptInputData,
                                    &promptEmbeddingStats)) {
                                decodeLayersOk = 0;
                                finalLogitsOk = 0;
                                break;
                            }
                            uint64_t promptPosition =
                                (uint64_t)options.cache_position + (uint64_t)promptIndex;
                            uint32_t promptContextLength =
                                (uint32_t)(options.context_length + promptIndex);
                            NSString *promptWorkRoot =
                                [[NSString stringWithUTF8String:options.output_dir]
                                    stringByAppendingPathComponent:
                                        [NSString stringWithFormat:
                                            @"prompt_prefill_%04d",
                                            promptIndex]];
                            decodeLayersFinalOutputData = nil;
                            decodeLayersElapsedSeconds = 0.0;
                            decodeLayersOutput0 = 0.0f;
                            decodeLayersOk = run_decode_layers_probe(
                                device,
                                residentBinPath,
                                decodeLayerPlans,
                                decodeLayers.count,
                                nil,
                                promptInputData,
                                nil,
                                promptWorkRoot,
                                [NSString stringWithUTF8String:options.cache_layout],
                                [NSString stringWithUTF8String:options.cache_file],
                                options.context1_o_proj_cache_layout
                                    ? [NSString stringWithUTF8String:options.context1_o_proj_cache_layout]
                                    : nil,
                                options.context1_o_proj_cache_file
                                    ? [NSString stringWithUTF8String:options.context1_o_proj_cache_file]
                                    : nil,
                                promptPosition,
                                promptContextLength,
                                (uint32_t)options.num_heads,
                                (uint32_t)options.qk_nope_dim,
                                (uint32_t)options.rope_dim,
                                (uint32_t)options.v_head_dim,
                                (uint32_t)options.cache_position_offset,
                                (float)options.attention_scale,
                                (float)options.rope_theta,
                                options.rope_interleave,
                                maxCacheFileBytes,
                                maxCacheReadBytes,
                                (uint32_t)options.top_k,
                                (float)options.rms_norm_eps,
                                expertBuffers,
                                residentMetalBuffer,
                                !options.skip_debug_intermediates,
                                decodeLayerSummaries,
                                &decodeLayersFinalOutputData,
                                &decodeLayersOutput0,
                                &decodeLayersElapsedSeconds
                            );
                            if (!decodeLayersOk) {
                                finalLogitsOk = 0;
                                break;
                            }
                            DecodeLayersAggregateStats promptDecodeAggregate =
                                collect_decode_layers_aggregate_stats(
                                    decodeLayerPlans,
                                    decodeLayerSummaries,
                                    (int)decodeLayers.count
                                );
                            NSMutableDictionary *prefillPayload =
                                [NSMutableDictionary dictionary];
                            prefillPayload[@"prompt_index"] = @(promptIndex);
                            prefillPayload[@"token_id"] =
                                @(promptTokenIds.values[promptIndex]);
                            prefillPayload[@"cache_position"] = @(promptPosition);
                            prefillPayload[@"context_length"] = @(promptContextLength);
                            prefillPayload[@"decode_elapsed_seconds"] =
                                @(decodeLayersElapsedSeconds);
                            prefillPayload[@"expert_bytes_read"] =
                                @(promptDecodeAggregate.expert_bytes_read);
                            prefillPayload[@"dense_mlp_bytes_read"] =
                                @(promptDecodeAggregate.dense_mlp_bytes_read);
                            prefillPayload[@"expert_routes"] = expert_routes_payload(
                                decodeLayerPlans,
                                decodeLayerSummaries,
                                (int)decodeLayers.count
                            );
                            add_decode_layers_aggregate_payload_fields(
                                prefillPayload,
                                promptDecodeAggregate
                            );
                            prefillPayload[@"input_embedding"] = @{
                                @"ok": @(promptEmbeddingStats.ok ? YES : NO),
                                @"token_id": @(promptEmbeddingStats.token_id),
                                @"bytes_read": @(promptEmbeddingStats.bytes_read),
                                @"output_bytes": @(promptEmbeddingStats.output_bytes),
                                @"elapsed_seconds": @(promptEmbeddingStats.elapsed_seconds),
                                @"read_seconds": @(promptEmbeddingStats.read_seconds),
                                @"decode_seconds": @(promptEmbeddingStats.decode_seconds),
                                @"write_seconds": @(promptEmbeddingStats.write_seconds),
                                @"output0": @(promptEmbeddingStats.output0),
                            };
                            prefillPayload[@"output0"] = @(decodeLayersOutput0);
                            [promptPrefillStepPayloads addObject:prefillPayload];
                            currentGenerateInputPath = nil;
                            currentGenerateInputData = decodeLayersFinalOutputData;
                        }
                    }
                }
                for (int step = 0;
                     decodeLayersOk && finalLogitsOk && step < options.generate_steps;
                     step++) {
                    @autoreleasepool {
                        int promptPrefillMode = promptTokenIds.count > 0;
                        int decodeStepOffset =
                            (promptPrefillMode || options.generate_first_from_input_logits)
                            ? step - 1
                            : step;
                        int stepRunsDecode = decodeStepOffset >= 0;
                        uint64_t stepPosition = stepRunsDecode
                            ? (uint64_t)options.cache_position +
                                (uint64_t)promptTokenIds.count +
                                (uint64_t)decodeStepOffset
                            : (uint64_t)options.cache_position +
                                (promptTokenIds.count > 0
                                    ? (uint64_t)promptTokenIds.count - 1u
                                    : 0u);
                        uint32_t stepContextLength = stepRunsDecode
                            ? (uint32_t)(options.context_length +
                                         promptTokenIds.count +
                                         decodeStepOffset)
                            : (uint32_t)(options.context_length +
                                         (promptTokenIds.count > 0
                                            ? promptTokenIds.count - 1
                                            : 0));
                        NSString *stepWorkRoot =
                            [[NSString stringWithUTF8String:options.output_dir]
                                stringByAppendingPathComponent:
                                    [NSString stringWithFormat:@"generate_step_%04d", step]];
                        NSString *finalGenerateOutputPath = options.output_f32
                            ? [NSString stringWithUTF8String:options.output_f32]
                            : nil;
                        NSString *stepOutputPath =
                            (finalGenerateOutputPath &&
                             step + 1 == options.generate_steps)
                                ? finalGenerateOutputPath
                                : nil;
                        decodeLayersFinalOutputData = nil;
                        decodeLayersElapsedSeconds = 0.0;
                        decodeLayersOutput0 = 0.0f;
                        DecodeLayersAggregateStats stepDecodeAggregate = {0};
                        if (stepRunsDecode) {
                            decodeLayersOk = run_decode_layers_probe(
                                device,
                                residentBinPath,
                                decodeLayerPlans,
                                decodeLayers.count,
                                currentGenerateInputPath,
                                currentGenerateInputData,
                                stepOutputPath,
                                stepWorkRoot,
                                [NSString stringWithUTF8String:options.cache_layout],
                                [NSString stringWithUTF8String:options.cache_file],
                                options.context1_o_proj_cache_layout
                                    ? [NSString stringWithUTF8String:options.context1_o_proj_cache_layout]
                                    : nil,
                                options.context1_o_proj_cache_file
                                    ? [NSString stringWithUTF8String:options.context1_o_proj_cache_file]
                                    : nil,
                                stepPosition,
                                stepContextLength,
                                (uint32_t)options.num_heads,
                                (uint32_t)options.qk_nope_dim,
                                (uint32_t)options.rope_dim,
                                (uint32_t)options.v_head_dim,
                                (uint32_t)options.cache_position_offset,
                                (float)options.attention_scale,
                                (float)options.rope_theta,
                                options.rope_interleave,
                                maxCacheFileBytes,
                                maxCacheReadBytes,
                                (uint32_t)options.top_k,
                                (float)options.rms_norm_eps,
                                expertBuffers,
                                residentMetalBuffer,
                                !options.skip_debug_intermediates,
                                decodeLayerSummaries,
                                &decodeLayersFinalOutputData,
                                &decodeLayersOutput0,
                                &decodeLayersElapsedSeconds
                            );
                            if (decodeLayersOk) {
                                stepDecodeAggregate =
                                    collect_decode_layers_aggregate_stats(
                                        decodeLayerPlans,
                                        decodeLayerSummaries,
                                        (int)decodeLayers.count
                                    );
                            }
                        } else {
                            decodeLayersOk = 1;
                            decodeLayersFinalOutputData = currentGenerateInputData;
                        }
                        if (!decodeLayersOk) {
                            break;
                        }
                        FinalLogitsProbeStats stepLogitsStats = {0};
                        NSString *finalLogitsInputPath = stepRunsDecode
                            ? stepOutputPath
                            : currentGenerateInputPath;
                        NSData *finalLogitsInputData = stepRunsDecode
                            ? decodeLayersFinalOutputData
                            : currentGenerateInputData;
                        finalLogitsOk = run_final_logits_probe(
                            device,
                            residentBinPath,
                            finalLogitsHeadInfo,
                            finalNormInfo,
                            finalLogitsInputPath,
                            finalLogitsInputData,
                            residentMetalBuffer,
                            options.mmap_final_logits,
                            (uint32_t)options.top_k,
                            (uint64_t)options.final_logits_chunk_rows,
                            maxFinalLogitsChunkBytes,
                            (float)options.rms_norm_eps,
                            options.skip_final_norm,
                            nil,
                            nil,
                            &stepLogitsStats
                        );
                        stepLogitsStats.scratch_bytes = finalLogitsScratchBytes;
                        if (!finalLogitsOk || stepLogitsStats.count == 0) {
                            if (stepLogitsStats.count == 0) {
                                fprintf(stderr,
                                        "ERROR: generate step %d produced empty top-k\n",
                                        step);
                            }
                            break;
                        }
                        EmbeddingLookupStats stepEmbeddingStats = {0};
                        NSData *nextInputData = nil;
                        NSString *nextInputOutputPath =
                            (options.output_next_input_f32 &&
                             step + 1 == options.generate_steps)
                                ? [NSString stringWithUTF8String:options.output_next_input_f32]
                                : nil;
                        finalLogitsOk = write_generated_embedding_f32(
                            residentBinPath,
                            nextInputEmbeddingInfo,
                            stepLogitsStats.ids[0],
                            maxEmbeddingRowBytes,
                            nextInputOutputPath,
                            &nextInputData,
                            &stepEmbeddingStats
                        );
                        if (!finalLogitsOk) {
                            break;
                        }
                        finalLogitsStats = stepLogitsStats;
                        nextInputEmbeddingStats = stepEmbeddingStats;
                        NSMutableDictionary *stepPayload =
                            [NSMutableDictionary dictionary];
                        stepPayload[@"step"] = @(step);
                        stepPayload[@"cache_position"] = @(stepPosition);
                        stepPayload[@"context_length"] = @(stepContextLength);
                        stepPayload[@"decode_ran"] = @(stepRunsDecode ? YES : NO);
                        stepPayload[@"first_from_input_logits"] =
                            @(((promptTokenIds.count > 0 ||
                                options.generate_first_from_input_logits) &&
                               step == 0) ? YES : NO);
                        stepPayload[@"input_from_memory"] =
                            @(currentGenerateInputData ? YES : NO);
                        stepPayload[@"generated_token"] =
                            final_generated_token_payload(stepLogitsStats);
                        stepPayload[@"topk"] = final_topk_array(stepLogitsStats);
                        stepPayload[@"decode_elapsed_seconds"] =
                            @(decodeLayersElapsedSeconds);
                        stepPayload[@"expert_bytes_read"] =
                            @(stepDecodeAggregate.expert_bytes_read);
                        stepPayload[@"dense_mlp_bytes_read"] =
                            @(stepDecodeAggregate.dense_mlp_bytes_read);
                        stepPayload[@"expert_routes"] = stepRunsDecode
                            ? expert_routes_payload(
                                decodeLayerPlans,
                                decodeLayerSummaries,
                                (int)decodeLayers.count
                            )
                            : @[];
                        add_decode_layers_aggregate_payload_fields(
                            stepPayload,
                            stepDecodeAggregate
                        );
                        stepPayload[@"final_logits_elapsed_seconds"] =
                            @(stepLogitsStats.elapsed_seconds);
                        stepPayload[@"final_logits_bytes_read"] =
                            @(stepLogitsStats.bytes_read);
                        stepPayload[@"final_logits_lm_head_bytes_read"] =
                            @(stepLogitsStats.lm_head_bytes_read);
                        stepPayload[@"final_logits_read_seconds"] =
                            @(stepLogitsStats.read_seconds);
                        stepPayload[@"final_logits_kernel_seconds"] =
                            @(stepLogitsStats.kernel_seconds);
                        stepPayload[@"final_logits_resident_mmap_backed"] =
                            @(stepLogitsStats.resident_mmap_backed ? YES : NO);
                        stepPayload[@"next_input_embedding"] = @{
                            @"ok": @(stepEmbeddingStats.ok ? YES : NO),
                            @"token_id": @(stepEmbeddingStats.token_id),
                            @"bytes_read": @(stepEmbeddingStats.bytes_read),
                            @"output_bytes": @(stepEmbeddingStats.output_bytes),
                            @"elapsed_seconds": @(stepEmbeddingStats.elapsed_seconds),
                            @"read_seconds": @(stepEmbeddingStats.read_seconds),
                            @"decode_seconds": @(stepEmbeddingStats.decode_seconds),
                            @"write_seconds": @(stepEmbeddingStats.write_seconds),
                            @"output0": @(stepEmbeddingStats.output0),
                        };
                        stepPayload[@"output0"] = @(decodeLayersOutput0);
                        [generateStepPayloads addObject:stepPayload];
                        currentGenerateInputPath = nil;
                        currentGenerateInputData = nextInputData;
                    }
                }
                if ((int)[generateStepPayloads count] != options.generate_steps) {
                    decodeLayersOk = 0;
                    if (finalLogitsOk) {
                        finalLogitsOk = 0;
                    }
                }
                if (decodeLayersOk && finalLogitsOk && options.output_generated_json) {
                    NSDictionary *generatedPayload = @{
                        @"step_count": @(options.generate_steps),
                        @"steps": generateStepPayloads,
                    };
                    NSError *error = nil;
                    NSData *jsonData =
                        [NSJSONSerialization dataWithJSONObject:generatedPayload
                                                        options:NSJSONWritingPrettyPrinted |
                                                                NSJSONWritingSortedKeys
                                                          error:&error];
                    if (!jsonData ||
                        ![jsonData writeToFile:
                            [NSString stringWithUTF8String:options.output_generated_json]
                                      atomically:YES]) {
                        fprintf(stderr,
                                "ERROR: failed to write generated JSON %s: %s\n",
                                options.output_generated_json,
                                error ? [[error localizedDescription] UTF8String] : "unknown");
                        finalLogitsOk = 0;
                    }
                }
            } else {
                decodeLayersOk = run_decode_layers_probe(
                    device,
                    residentBinPath,
                    decodeLayerPlans,
                    decodeLayers.count,
                    resolvedInputPath,
                    initialInputData,
                    [NSString stringWithUTF8String:options.output_f32],
                    [NSString stringWithUTF8String:options.output_dir],
                    [NSString stringWithUTF8String:options.cache_layout],
                    [NSString stringWithUTF8String:options.cache_file],
                    options.context1_o_proj_cache_layout
                        ? [NSString stringWithUTF8String:options.context1_o_proj_cache_layout]
                        : nil,
                    options.context1_o_proj_cache_file
                        ? [NSString stringWithUTF8String:options.context1_o_proj_cache_file]
                        : nil,
                    (uint64_t)options.cache_position,
                    (uint32_t)options.context_length,
                    (uint32_t)options.num_heads,
                    (uint32_t)options.qk_nope_dim,
                    (uint32_t)options.rope_dim,
                    (uint32_t)options.v_head_dim,
                    (uint32_t)options.cache_position_offset,
                    (float)options.attention_scale,
                    (float)options.rope_theta,
                    options.rope_interleave,
                    maxCacheFileBytes,
                    maxCacheReadBytes,
                    (uint32_t)options.top_k,
                    (float)options.rms_norm_eps,
                    expertBuffers,
                    residentMetalBuffer,
                    !options.skip_debug_intermediates,
                    decodeLayerSummaries,
                    &decodeLayersFinalOutputData,
                    &decodeLayersOutput0,
                    &decodeLayersElapsedSeconds
                );
            }
        }

        if ((options.probe_final_logits || decodeLayersFinalLogits) &&
            !decodeLayerGenerateMode) {
            if (!admissionOk) {
                finalLogitsOk = 0;
                fprintf(stderr,
                        "ERROR: admission guard failed; refusing --probe-final-logits\n");
            } else if (decodeLayersFinalLogits && !decodeLayersOk) {
                finalLogitsOk = 0;
                fprintf(stderr,
                        "ERROR: refusing decode-layers final logits because decode layers failed\n");
            } else {
                NSString *finalLogitsInputPath = options.probe_final_logits
                    ? [NSString stringWithUTF8String:options.input_f32]
                    : [NSString stringWithUTF8String:options.output_f32];
                NSData *finalLogitsInputData = options.probe_final_logits
                    ? nil
                    : decodeLayersFinalOutputData;
                finalLogitsOk = run_final_logits_probe(
                    device,
                    residentBinPath,
                    finalLogitsHeadInfo,
                    finalNormInfo,
                    finalLogitsInputPath,
                    finalLogitsInputData,
                    residentMetalBuffer,
                    options.mmap_final_logits,
                    (uint32_t)options.top_k,
                    (uint64_t)options.final_logits_chunk_rows,
                    maxFinalLogitsChunkBytes,
                    (float)options.rms_norm_eps,
                    options.skip_final_norm,
                    options.output_topk_json
                        ? [NSString stringWithUTF8String:options.output_topk_json]
                        : nil,
                    options.output_token_json
                        ? [NSString stringWithUTF8String:options.output_token_json]
                        : nil,
                    &finalLogitsStats
                );
                finalLogitsStats.scratch_bytes = finalLogitsScratchBytes;
                if (finalLogitsOk && options.output_next_input_f32) {
                    if (finalLogitsStats.count == 0) {
                        finalLogitsOk = 0;
                        fprintf(stderr,
                                "ERROR: refusing generated-token embedding because final top-k is empty\n");
                    } else {
                        finalLogitsOk = write_generated_embedding_f32(
                            residentBinPath,
                            nextInputEmbeddingInfo,
                            finalLogitsStats.ids[0],
                            maxEmbeddingRowBytes,
                            [NSString stringWithUTF8String:options.output_next_input_f32],
                            NULL,
                            &nextInputEmbeddingStats
                        );
                    }
                }
            }
        }

        int overallOk =
            admissionOk && inputTokenEmbeddingOk && probeOk && residentLinearOk && attnProjectionOk &&
            ropeSplitOk && mlaAttentionOk && attnOutputOk &&
            denseMlpOk && rmsNormOk && routerOk && layerMoeOk &&
            decoderLayerOk && denseDecoderLayerOk && decodeLayersOk &&
            finalLogitsOk;
        if (options.json) {
            NSMutableDictionary *payload = [NSMutableDictionary dictionary];
            payload[@"schema"] = @"largerlm.glm_moe_infer_loader.v1";
            payload[@"ok"] = @(overallOk);
            payload[@"runtime_entry"] = options.generate_token_ids
                ? @"generate_token_ids"
                : (options.probe_decode_layers ? @"probe_decode_layers" : @"probe");
            payload[@"device_name"] = [device name] ?: @"";
            payload[@"resident_layout_path"] = residentLayoutPath;
            payload[@"resident_weight_file"] = residentBinPath;
            payload[@"resident_bytes"] = @(residentFileBytes);
            payload[@"resident_fd_opened"] = @(runtime.residentFd >= 0);
            payload[@"resident_mmap"] = @(options.mmap_resident);
            payload[@"resident_mmap_actual"] = @(residentMap != MAP_FAILED);
            payload[@"resident_metal_wrapped"] = @(residentMetalBuffer != nil);
            payload[@"decode_cache_backend"] = options.in_memory_decode_cache
                ? @"memory"
                : @"file";
            payload[@"decode_cache_fd_opened"] = @(decodeCacheFdReady);
            payload[@"decode_cache_fd_reused"] = @(decodeCacheFdReused);
            payload[@"decode_cache_fd_open_count"] = @(runtime.decodeCacheFdOpenCount);
            payload[@"decode_cache_fd_path"] = runtime.decodeCachePath
                ? runtime.decodeCachePath
                : (id)[NSNull null];
            payload[@"decode_cache_memory_loaded"] = @(decodeCacheMemoryReady);
            payload[@"decode_cache_memory_reused"] = @(decodeCacheMemoryReused);
            payload[@"decode_cache_memory_load_count"] =
                @(runtime.decodeCacheMemoryLoadCount);
            payload[@"decode_cache_memory_bytes"] = @(decodeCacheMemoryBytes);
            payload[@"decode_cache_memory_path"] = runtime.decodeCacheMemoryPath
                ? runtime.decodeCacheMemoryPath
                : (id)[NSNull null];
            payload[@"expert_layout_path"] = expertLayoutPath;
            payload[@"expert_layer_count"] = @([layers count]);
            payload[@"expert_files_opened"] = @(openedExpertFiles);
            payload[@"expert_total_bytes"] = @(totalExpertBytes);
            payload[@"max_expert_slot_bytes"] = @(maxExpertSlotBytes);
            payload[@"expert_buffer_count"] = @(activeExpertBufferCount);
            payload[@"expert_buffer_count_allocated"] = @([expertBuffers count]);
            payload[@"expert_buffer_count_runtime_allocated"] =
                @([runtime.expertBuffers count]);
            payload[@"expert_buffer_count_live_estimate"] = @(liveExpertBufferCount);
            payload[@"expert_buffer_pool_reused"] = @(expertBufferPoolReused);
            payload[@"expert_buffer_count_requested"] = @(options.expert_buffer_count);
            payload[@"expert_buffer_bytes_each"] = @(expertBufferBytes);
            payload[@"expert_buffer_bytes_total"] = @(reusableExpertBytes);
            payload[@"probe_layer_moe_scratch_bytes"] = @(probeMoeScratchBytes);
            payload[@"probe_shared_expert_storage_bytes"] = @(sharedExpertStorageBytes);
            payload[@"probe_router_scratch_bytes"] = @(routerScratchBytes);
            payload[@"probe_rmsnorm_scratch_bytes"] = @(rmsNormScratchBytes);
            payload[@"probe_resident_linear_scratch_bytes"] = @(residentLinearScratchBytes);
            payload[@"probe_attn_projection_scratch_bytes"] = @(attnProjectionScratchBytes);
            payload[@"probe_rope_split_scratch_bytes"] = @(ropeSplitScratchBytes);
            payload[@"probe_mla_attention_scratch_bytes"] = @(mlaAttentionScratchBytes);
            payload[@"probe_attn_output_scratch_bytes"] = @(attnOutputScratchBytes);
            payload[@"probe_dense_mlp_scratch_bytes"] = @(denseMlpScratchBytes);
            payload[@"probe_decoder_layer_scratch_bytes"] = @(decoderLayerScratchBytes);
            payload[@"probe_dense_decoder_layer_scratch_bytes"] =
                @(denseDecoderLayerScratchBytes);
            payload[@"probe_decode_layers_scratch_bytes"] = @(decodeLayersScratchBytes);
            payload[@"probe_final_logits_scratch_bytes"] = @(finalLogitsScratchBytes);
            payload[@"probe_final_logits_mmap_bytes"] = @(finalLogitsMmapBytes);
            payload[@"probe_next_input_embedding_scratch_bytes"] =
                @(nextInputEmbeddingScratchBytes);
            payload[@"mla_kv_b_cache_enabled"] =
                @(options.cache_mla_kv_b_f32 ? YES : NO);
            payload[@"mla_kv_b_cache_max_bytes"] = @(maxMlaKVBMemoryCacheBytes);
            payload[@"mla_kv_b_cache_planned_bytes"] =
                @(mlaKVBMemoryCachePlannedBytes);
            payload[@"mla_kv_b_cache_live_estimate_bytes"] =
                @(mlaKVBMemoryCacheLiveEstimateBytes);
            payload[@"mla_kv_b_cache_current_bytes"] =
                @(mla_kv_b_memory_cache_current_bytes());
            if (options.input_token_id >= 0) {
                payload[@"probe_input_token_embedding"] = @{
                    @"ok": @(inputTokenEmbeddingStats.ok ? YES : NO),
                    @"token_id": @(inputTokenEmbeddingStats.token_id),
                    @"tensor": [NSString stringWithUTF8String:nextInputEmbeddingInfo.name],
                    @"dtype": @"mlx-mxfp4",
                    @"vocab_size": @(inputTokenEmbeddingStats.vocab_size),
                    @"hidden_dim": @(inputTokenEmbeddingStats.hidden_dim),
                    @"group_size": @(inputTokenEmbeddingStats.group_size),
                    @"bytes_read": @(inputTokenEmbeddingStats.bytes_read),
                    @"output_bytes": @(inputTokenEmbeddingStats.output_bytes),
                    @"elapsed_seconds": @(inputTokenEmbeddingStats.elapsed_seconds),
                    @"read_seconds": @(inputTokenEmbeddingStats.read_seconds),
                    @"decode_seconds": @(inputTokenEmbeddingStats.decode_seconds),
                    @"write_seconds": @(inputTokenEmbeddingStats.write_seconds),
                    @"output0": @(inputTokenEmbeddingStats.output0),
                };
            }
            payload[@"estimated_live_working_set_bytes"] = @(estimatedLiveBytes);
            payload[@"max_live_working_set_mib"] = @(options.max_live_working_set_mib);
            payload[@"live_working_set_ok"] = @(liveOk);
            payload[@"min_free_unified_memory_gib"] = @(options.min_free_unified_memory_gib);
            payload[@"min_free_unified_memory_bytes"] = @(minFreeUnifiedMemoryBytes);
            payload[@"required_available_memory_bytes"] = @(requiredAvailableMemoryBytes);
            payload[@"available_unified_memory_ok"] = @(freeUnifiedMemoryOk);
            payload[@"system_memory_snapshot_ok"] = @(systemMemorySnapshotOk);
            payload[@"system_memory_source"] = systemMemorySnapshotOk
                ? @"host_statistics64"
                : (id)[NSNull null];
            payload[@"system_page_size"] = systemMemorySnapshotOk
                ? @(memorySnapshot.page_size)
                : (id)[NSNull null];
            payload[@"system_available_memory_bytes"] = systemMemorySnapshotOk
                ? @(memorySnapshot.available_bytes)
                : (id)[NSNull null];
            payload[@"system_total_memory_bytes"] =
                (systemMemorySnapshotOk && memorySnapshot.total_bytes > 0)
                    ? @(memorySnapshot.total_bytes)
                    : (id)[NSNull null];
            payload[@"admission_ok"] = @(admissionOk);
            if (options.probe_expert_read) {
                NSMutableDictionary *probePayload = [NSMutableDictionary dictionary];
                NSMutableArray *expertIds = [NSMutableArray arrayWithCapacity:(NSUInteger)probeExperts.count];
                NSMutableArray *resultItems = [NSMutableArray arrayWithCapacity:probeResultCount];
                for (int i = 0; i < probeExperts.count; i++) {
                    [expertIds addObject:@(probeExperts.values[i])];
                }
                if (probeResults) {
                    for (size_t i = 0; i < probeResultCount; i++) {
                        NSMutableDictionary *item = [NSMutableDictionary dictionary];
                        item[@"layer"] = @(probeResults[i].layer);
                        item[@"expert"] = @(probeResults[i].expert);
                        item[@"bytes"] = @(probeResults[i].bytes);
                        item[@"sample_checksum"] = hex_u64(probeResults[i].sample_checksum);
                        [resultItems addObject:item];
                    }
                }
                NSMutableArray *layersRead = [NSMutableArray arrayWithCapacity:probeReadLayerCount];
                if (options.probe_expert_read) {
                    if (probeResults) {
                        for (NSUInteger i = 0; i < probeReadLayerCount; i++) {
                            ExpertFile *currentProbeFile =
                                options.probe_all_layers ? &expertFiles[i] : probeFile;
                            [layersRead addObject:@(currentProbeFile->layer)];
                        }
                    }
                }
                probePayload[@"enabled"] = @(1);
                probePayload[@"ok"] = @(probeOk);
                probePayload[@"all_layers"] = @(options.probe_all_layers);
                probePayload[@"layer"] =
                    options.probe_all_layers ? (id)[NSNull null] : @(options.probe_layer);
                probePayload[@"layers"] = layersRead;
                probePayload[@"layer_count"] = @(probeReadLayerCount);
                probePayload[@"experts"] = expertIds;
                probePayload[@"repeat"] = @(options.probe_repeat);
                probePayload[@"slot_bytes"] =
                    probeFile ? @(probeFile->expert_slot_bytes) : @(maxExpertSlotBytes);
                probePayload[@"bytes_read"] = @(probeBytesRead);
                probePayload[@"elapsed_seconds"] = @(probeElapsedSeconds);
                probePayload[@"throughput_gib_per_second"] = @(probeThroughputGiBPerSecond);
                probePayload[@"expert_read_dispatch_count"] =
                    @(probeReadStats.dispatch_count);
                probePayload[@"expert_read_task_count"] =
                    @(probeReadStats.task_count);
                probePayload[@"expert_read_max_task_count"] =
                    @(probeReadStats.max_task_count);
                probePayload[@"expert_read_max_worker_count"] =
                    @(probeReadStats.max_worker_count);
                probePayload[@"expert_read_pool_dispatch_count"] =
                    @(probeReadStats.pool_dispatch_count);
                probePayload[@"expert_read_serial_dispatch_count"] =
                    @(probeReadStats.serial_dispatch_count);
                probePayload[@"skipped_due_live_cap"] = @(!liveOk);
                if (probeResults) {
                    probePayload[@"results"] = resultItems;
                }
                payload[@"probe_expert_read"] = probePayload;
            }
            if (options.probe_resident_linear) {
                NSMutableDictionary *linearPayload = [NSMutableDictionary dictionary];
                linearPayload[@"enabled"] = @(1);
                linearPayload[@"ok"] = @(residentLinearOk);
                linearPayload[@"tensor"] = [NSString stringWithUTF8String:residentLinearInfo.name];
                linearPayload[@"dtype"] = @"mlx-mxfp4";
                linearPayload[@"out_dim"] = @(residentLinearInfo.out_dim);
                linearPayload[@"in_dim"] = @(residentLinearInfo.in_dim);
                linearPayload[@"group_size"] = @(residentLinearInfo.group_size);
                linearPayload[@"bytes_read"] = @(residentLinearStats.bytes_read);
                linearPayload[@"resident_mmap_backed"] =
                    @(residentLinearStats.resident_mmap_backed ? YES : NO);
                linearPayload[@"scratch_bytes"] = @(residentLinearScratchBytes);
                linearPayload[@"elapsed_seconds"] = @(residentLinearStats.elapsed_seconds);
                linearPayload[@"read_seconds"] = @(residentLinearStats.read_seconds);
                linearPayload[@"kernel_seconds"] = @(residentLinearStats.kernel_seconds);
                linearPayload[@"output_write_seconds"] = @(residentLinearStats.output_write_seconds);
                linearPayload[@"output0"] = @(residentLinearStats.output0);
                linearPayload[@"output0_check_ok"] = @(residentLinearStats.output0_check_ok);
                if (options.expect_output0_set) {
                    linearPayload[@"expected_output0"] = @(options.expect_output0);
                    linearPayload[@"output0_abs_error"] =
                        @(residentLinearStats.output0_abs_error);
                }
                linearPayload[@"input_f32"] = [NSString stringWithUTF8String:options.input_f32];
                linearPayload[@"output_f32"] = [NSString stringWithUTF8String:options.output_f32];
                linearPayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_resident_linear"] = linearPayload;
            }
            if (options.probe_attn_projections) {
                NSMutableDictionary *attnPayload = [NSMutableDictionary dictionary];
                attnPayload[@"enabled"] = @(1);
                attnPayload[@"ok"] = @(attnProjectionOk);
                attnPayload[@"layer"] = @(options.probe_layer);
                attnPayload[@"hidden_dim"] = @(attnProjectionStats.hidden_dim);
                attnPayload[@"q_lora_dim"] = @(attnProjectionStats.q_lora_dim);
                attnPayload[@"q_out_dim"] = @(attnProjectionStats.q_out_dim);
                attnPayload[@"kv_lora_dim"] = @(attnProjectionStats.kv_lora_dim);
                attnPayload[@"kv_a_out_dim"] = @(attnProjectionStats.kv_a_out_dim);
                attnPayload[@"kv_rope_dim"] = @(attnProjectionStats.kv_rope_dim);
                attnPayload[@"kv_out_dim"] = @(attnProjectionStats.kv_out_dim);
                attnPayload[@"has_kv_b"] = @(attnProjectionStats.has_kv_b ? YES : NO);
                attnPayload[@"bytes_read"] = @(attnProjectionStats.bytes_read);
                attnPayload[@"scratch_bytes"] = @(attnProjectionScratchBytes);
                attnPayload[@"elapsed_seconds"] = @(attnProjectionStats.elapsed_seconds);
                attnPayload[@"fused_pre_cache"] =
                    @(attnProjectionStats.fused_pre_cache ? YES : NO);
                attnPayload[@"command_buffer_count"] =
                    @(attnProjectionStats.command_buffer_count);
                attnPayload[@"fused_pre_cache_seconds"] =
                    @(attnProjectionStats.fused_pre_cache_seconds);
                attnPayload[@"q_b_output0"] = @(attnProjectionStats.q_b_output0);
                attnPayload[@"kv_a_output0"] = @(attnProjectionStats.kv_a_output0);
                attnPayload[@"kv_a_norm_output0"] = @(attnProjectionStats.kv_a_norm_output0);
                if (attnProjectionStats.has_kv_b) {
                    attnPayload[@"kv_b_output0"] = @(attnProjectionStats.kv_b_output0);
                }
                attnPayload[@"cache_append"] = @(attnProjectionStats.cache_append ? YES : NO);
                if (attnProjectionStats.cache_append) {
                    attnPayload[@"cache_layout"] = [NSString stringWithUTF8String:options.cache_layout];
                    attnPayload[@"cache_file"] = [NSString stringWithUTF8String:options.cache_file];
                    attnPayload[@"cache_position"] = @(attnProjectionStats.cache_position);
                    attnPayload[@"cache_write_bytes"] = @(attnProjectionStats.cache_write_bytes);
                    attnPayload[@"cache_write_seconds"] = @(attnProjectionStats.cache_write_seconds);
                }
                attnPayload[@"rms_norm_eps"] = @(options.rms_norm_eps);
                attnPayload[@"input_f32"] = [NSString stringWithUTF8String:options.input_f32];
                attnPayload[@"output_dir"] = [NSString stringWithUTF8String:options.output_dir];
                attnPayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_attn_projections"] = attnPayload;
            }
            if (options.probe_rope_split) {
                NSMutableDictionary *ropePayload = [NSMutableDictionary dictionary];
                ropePayload[@"enabled"] = @(1);
                ropePayload[@"ok"] = @(ropeSplitOk);
                ropePayload[@"num_heads"] = @(ropeSplitStats.num_heads);
                ropePayload[@"qk_nope_dim"] = @(ropeSplitStats.qk_nope_dim);
                ropePayload[@"rope_dim"] = @(ropeSplitStats.rope_dim);
                ropePayload[@"start_position"] = @(ropeSplitStats.start_position);
                ropePayload[@"batch_tokens"] = @(ropeSplitStats.batch_tokens);
                ropePayload[@"theta"] = @(ropeSplitStats.theta);
                ropePayload[@"interleave"] = @(ropeSplitStats.interleave ? YES : NO);
                ropePayload[@"q_b_bytes"] = @(ropeSplitQBBytes);
                ropePayload[@"k_bytes"] = @(ropeSplitKBytes);
                ropePayload[@"q_nope_bytes"] = @(ropeSplitQNopeBytes);
                ropePayload[@"q_rope_bytes"] = @(ropeSplitQRopeBytes);
                ropePayload[@"scratch_bytes"] = @(ropeSplitScratchBytes);
                ropePayload[@"elapsed_seconds"] = @(ropeSplitStats.elapsed_seconds);
                ropePayload[@"q_output0"] = @(ropeSplitStats.q_output0);
                ropePayload[@"k_output0"] = @(ropeSplitStats.k_output0);
                ropePayload[@"q_b_f32"] = [NSString stringWithUTF8String:options.q_b_f32];
                ropePayload[@"k_f32"] = [NSString stringWithUTF8String:options.k_f32];
                ropePayload[@"output_q_nope_f32"] =
                    [NSString stringWithUTF8String:options.output_q_nope_f32];
                ropePayload[@"output_q_rope_f32"] =
                    [NSString stringWithUTF8String:options.output_q_rope_f32];
                ropePayload[@"output_q_f32"] =
                    [NSString stringWithUTF8String:options.output_q_f32];
                ropePayload[@"output_k_f32"] =
                    [NSString stringWithUTF8String:options.output_k_f32];
                ropePayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_rope_split"] = ropePayload;
            }
            if (options.probe_mla_attention) {
                NSMutableDictionary *mlaPayload = [NSMutableDictionary dictionary];
                mlaPayload[@"enabled"] = @(1);
                mlaPayload[@"ok"] = @(mlaAttentionOk);
                mlaPayload[@"layer"] = @(options.probe_layer);
                mlaPayload[@"context_length"] = @(mlaAttentionStats.context_length);
                mlaPayload[@"num_heads"] = @(mlaAttentionStats.num_heads);
                mlaPayload[@"kv_lora_dim"] = @(mlaAttentionStats.kv_lora_dim);
                mlaPayload[@"qk_nope_dim"] = @(mlaAttentionStats.qk_nope_dim);
                mlaPayload[@"rope_dim"] = @(mlaAttentionStats.rope_dim);
                mlaPayload[@"v_head_dim"] = @(mlaAttentionStats.v_head_dim);
                mlaPayload[@"cache_position_offset"] =
                    @(mlaAttentionStats.cache_position_offset);
                mlaPayload[@"attention_scale"] = @(mlaAttentionStats.attention_scale);
                mlaPayload[@"rope_theta"] = @(mlaAttentionStats.rope_theta);
                mlaPayload[@"rope_interleave"] =
                    @(mlaAttentionStats.rope_interleave ? YES : NO);
                mlaPayload[@"value_source"] =
                    options.mla_kv_b_f32 ? @"direct-f32" : @"absorbed-alias";
                mlaPayload[@"raw_cache_bytes"] = @(mlaAttentionStats.raw_cache_bytes);
                mlaPayload[@"cache_f32_bytes"] = @(mlaAttentionStats.cache_f32_bytes);
                mlaPayload[@"value_storage_bytes"] =
                    @(mlaAttentionStats.value_storage_bytes);
                mlaPayload[@"value_source_f32_bytes"] =
                    @(mlaAttentionStats.value_source_f32_bytes);
                mlaPayload[@"kv_b_f32_bytes"] = @(mlaAttentionStats.kv_b_f32_bytes);
                mlaPayload[@"value_cache_enabled"] =
                    @(mlaAttentionStats.value_cache_enabled ? YES : NO);
                mlaPayload[@"value_cache_hit"] =
                    @(mlaAttentionStats.value_cache_hit ? YES : NO);
                mlaPayload[@"value_cache_stored"] =
                    @(mlaAttentionStats.value_cache_stored ? YES : NO);
                mlaPayload[@"value_cache_bytes"] =
                    @(mlaAttentionStats.value_cache_bytes);
                mlaPayload[@"value_cache_total_bytes"] =
                    @(mlaAttentionStats.value_cache_total_bytes);
                mlaPayload[@"q_nope_bytes"] = @(mlaAttentionStats.q_nope_bytes);
                mlaPayload[@"q_rope_bytes"] = @(mlaAttentionStats.q_rope_bytes);
                mlaPayload[@"output_bytes"] = @(mlaAttentionStats.output_bytes);
                mlaPayload[@"scratch_bytes"] = @(mlaAttentionScratchBytes);
                mlaPayload[@"cache_read_seconds"] =
                    @(mlaAttentionStats.cache_read_seconds);
                mlaPayload[@"value_read_seconds"] =
                    @(mlaAttentionStats.value_read_seconds);
                mlaPayload[@"kernel_seconds"] = @(mlaAttentionStats.kernel_seconds);
                mlaPayload[@"output_write_seconds"] =
                    @(mlaAttentionStats.output_write_seconds);
                mlaPayload[@"elapsed_seconds"] = @(mlaAttentionStats.elapsed_seconds);
                mlaPayload[@"output0"] = @(mlaAttentionStats.output0);
                mlaPayload[@"cache_layout"] = [NSString stringWithUTF8String:options.cache_layout];
                mlaPayload[@"cache_file"] = [NSString stringWithUTF8String:options.cache_file];
                mlaPayload[@"q_nope_f32"] = [NSString stringWithUTF8String:options.q_nope_f32];
                mlaPayload[@"q_rope_f32"] = [NSString stringWithUTF8String:options.q_rope_f32];
                mlaPayload[@"output_f32"] = [NSString stringWithUTF8String:options.output_f32];
                mlaPayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_mla_attention"] = mlaPayload;
            }
            if (options.probe_attn_output) {
                NSMutableDictionary *outPayload = [NSMutableDictionary dictionary];
                outPayload[@"enabled"] = @(1);
                outPayload[@"ok"] = @(attnOutputOk);
                outPayload[@"layer"] = @(options.probe_layer);
                outPayload[@"tensor"] = [NSString stringWithUTF8String:attnOutputInfo.name];
                outPayload[@"dtype"] = @"mlx-mxfp4";
                outPayload[@"out_dim"] = @(attnOutputStats.out_dim);
                outPayload[@"in_dim"] = @(attnOutputStats.in_dim);
                outPayload[@"group_size"] = @(attnOutputStats.group_size);
                outPayload[@"bytes_read"] = @(attnOutputStats.bytes_read);
                outPayload[@"input_bytes"] = @(attnOutputStats.input_bytes);
                outPayload[@"residual_bytes"] = @(attnOutputStats.residual_bytes);
                outPayload[@"projection_bytes"] = @(attnOutputStats.projection_bytes);
                outPayload[@"output_bytes"] = @(attnOutputStats.output_bytes);
                outPayload[@"scratch_bytes"] = @(attnOutputScratchBytes);
                outPayload[@"resident_mmap_backed"] =
                    @(attnOutputStats.resident_mmap_backed ? YES : NO);
                outPayload[@"read_seconds"] = @(attnOutputStats.read_seconds);
                outPayload[@"projection_kernel_seconds"] =
                    @(attnOutputStats.projection_kernel_seconds);
                outPayload[@"residual_add_seconds"] =
                    @(attnOutputStats.residual_add_seconds);
                outPayload[@"fused_matvec_add"] =
                    @(attnOutputStats.fused_matvec_add ? YES : NO);
                outPayload[@"command_buffer_count"] =
                    @(attnOutputStats.command_buffer_count);
                outPayload[@"projection_write_seconds"] =
                    @(attnOutputStats.projection_write_seconds);
                outPayload[@"output_write_seconds"] =
                    @(attnOutputStats.output_write_seconds);
                outPayload[@"elapsed_seconds"] = @(attnOutputStats.elapsed_seconds);
                outPayload[@"output0"] = @(attnOutputStats.output0);
                outPayload[@"input_f32"] = [NSString stringWithUTF8String:options.input_f32];
                outPayload[@"residual_f32"] =
                    [NSString stringWithUTF8String:options.residual_f32];
                if (options.projection_f32) {
                    outPayload[@"projection_f32"] =
                        [NSString stringWithUTF8String:options.projection_f32];
                }
                outPayload[@"output_f32"] = [NSString stringWithUTF8String:options.output_f32];
                outPayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_attn_output"] = outPayload;
            }
            if (options.probe_dense_mlp_block) {
                NSMutableDictionary *densePayload = [NSMutableDictionary dictionary];
                densePayload[@"enabled"] = @(1);
                densePayload[@"ok"] = @(denseMlpOk);
                densePayload[@"layer"] = @(options.probe_layer);
                densePayload[@"hidden_dim"] = @(denseMlpStats.hidden_dim);
                densePayload[@"intermediate_dim"] = @(denseMlpStats.intermediate_dim);
                densePayload[@"group_size"] = @(denseMlpStats.group_size);
                densePayload[@"gate_tensor"] =
                    [NSString stringWithUTF8String:denseMlpInfo.gate.name];
                densePayload[@"up_tensor"] =
                    [NSString stringWithUTF8String:denseMlpInfo.up.name];
                densePayload[@"down_tensor"] =
                    [NSString stringWithUTF8String:denseMlpInfo.down.name];
                densePayload[@"bytes_read"] = @(denseMlpStats.bytes_read);
                densePayload[@"scratch_bytes"] = @(denseMlpScratchBytes);
                densePayload[@"elapsed_seconds"] = @(denseMlpStats.elapsed_seconds);
                densePayload[@"fused_pipeline"] =
                    @(denseMlpStats.fused_pipeline ? YES : NO);
                densePayload[@"command_buffer_count"] =
                    @(denseMlpStats.command_buffer_count);
                densePayload[@"synchronous_wait_count"] =
                    @(denseMlpStats.synchronous_wait_count);
                densePayload[@"async_submitted"] =
                    @(denseMlpStats.async_submitted ? YES : NO);
                densePayload[@"fused_kernel_seconds"] =
                    @(denseMlpStats.fused_kernel_seconds);
                densePayload[@"rmsnorm_elapsed_seconds"] =
                    @(denseMlpStats.rmsnorm_elapsed_seconds);
                densePayload[@"gate_read_seconds"] = @(denseMlpStats.gate_read_seconds);
                densePayload[@"gate_kernel_seconds"] = @(denseMlpStats.gate_kernel_seconds);
                densePayload[@"up_read_seconds"] = @(denseMlpStats.up_read_seconds);
                densePayload[@"up_kernel_seconds"] = @(denseMlpStats.up_kernel_seconds);
                densePayload[@"swiglu_kernel_seconds"] =
                    @(denseMlpStats.swiglu_kernel_seconds);
                densePayload[@"down_read_seconds"] = @(denseMlpStats.down_read_seconds);
                densePayload[@"down_kernel_seconds"] = @(denseMlpStats.down_kernel_seconds);
                densePayload[@"residual_add_seconds"] =
                    @(denseMlpStats.residual_add_seconds);
                densePayload[@"output_write_seconds"] =
                    @(denseMlpStats.output_write_seconds);
                densePayload[@"output0"] = @(denseMlpStats.output0);
                densePayload[@"output0_check_ok"] =
                    @(denseMlpStats.output0_check_ok ? YES : NO);
                if (options.expect_output0_set) {
                    densePayload[@"expected_output0"] = @(options.expect_output0);
                    densePayload[@"output0_abs_error"] =
                        @(denseMlpStats.output0_abs_error);
                }
                densePayload[@"rms_norm_eps"] = @(options.rms_norm_eps);
                densePayload[@"input_f32"] = [NSString stringWithUTF8String:options.input_f32];
                densePayload[@"output_f32"] = [NSString stringWithUTF8String:options.output_f32];
                densePayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_dense_mlp_block"] = densePayload;
            }
            if (options.probe_decoder_layer) {
                NSMutableDictionary *decoderPayload = [NSMutableDictionary dictionary];
                decoderPayload[@"enabled"] = @(1);
                decoderPayload[@"ok"] = @(decoderLayerOk);
                decoderPayload[@"layer"] = @(options.probe_layer);
                decoderPayload[@"position"] = @(decoderLayerStats.position);
                decoderPayload[@"context_length"] = @(decoderLayerStats.context_length);
                decoderPayload[@"scratch_bytes"] = @(decoderLayerScratchBytes);
                decoderPayload[@"elapsed_seconds"] = @(decoderLayerStats.elapsed_seconds);
                decoderPayload[@"output0"] = @(decoderLayerStats.output0);
                decoderPayload[@"work_dir"] = [NSString stringWithUTF8String:options.output_dir];
                decoderPayload[@"input_f32"] = [NSString stringWithUTF8String:options.input_f32];
                decoderPayload[@"output_f32"] = [NSString stringWithUTF8String:options.output_f32];
                decoderPayload[@"cache_layout"] = [NSString stringWithUTF8String:options.cache_layout];
                decoderPayload[@"cache_file"] = [NSString stringWithUTF8String:options.cache_file];
                decoderPayload[@"cache_write_bytes"] =
                    @(decoderLayerStats.attn_projection.cache_write_bytes);
                decoderPayload[@"cache_write_seconds"] =
                    @(decoderLayerStats.attn_projection.cache_write_seconds);
                decoderPayload[@"attn_projection_elapsed_seconds"] =
                    @(decoderLayerStats.attn_projection.elapsed_seconds);
                decoderPayload[@"attn_projection_fused_pre_cache"] =
                    @(decoderLayerStats.attn_projection.fused_pre_cache ? YES : NO);
                decoderPayload[@"attn_projection_command_buffer_count"] =
                    @(decoderLayerStats.attn_projection.command_buffer_count);
                decoderPayload[@"attn_projection_fused_pre_cache_seconds"] =
                    @(decoderLayerStats.attn_projection.fused_pre_cache_seconds);
                decoderPayload[@"rope_split_elapsed_seconds"] =
                    @(decoderLayerStats.rope_split.elapsed_seconds);
                decoderPayload[@"mla_attention_elapsed_seconds"] =
                    @(decoderLayerStats.mla_attention.elapsed_seconds);
                decoderPayload[@"mla_attention_kernel_seconds"] =
                    @(decoderLayerStats.mla_attention.kernel_seconds);
                decoderPayload[@"mla_value_read_seconds"] =
                    @(decoderLayerStats.mla_attention.value_read_seconds);
                decoderPayload[@"mla_value_cache_enabled"] =
                    @(decoderLayerStats.mla_attention.value_cache_enabled ? YES : NO);
                decoderPayload[@"mla_value_cache_hit"] =
                    @(decoderLayerStats.mla_attention.value_cache_hit ? YES : NO);
                decoderPayload[@"mla_value_cache_stored"] =
                    @(decoderLayerStats.mla_attention.value_cache_stored ? YES : NO);
                decoderPayload[@"mla_value_cache_bytes"] =
                    @(decoderLayerStats.mla_attention.value_cache_bytes);
                decoderPayload[@"mla_value_cache_total_bytes"] =
                    @(decoderLayerStats.mla_attention.value_cache_total_bytes);
                decoderPayload[@"rope_mla_fused"] =
                    @((decoderLayerStats.rope_split.fused_with_mla &&
                       decoderLayerStats.mla_attention.fused_with_rope) ? YES : NO);
                decoderPayload[@"rope_mla_command_buffer_count"] =
                    @(decoderLayerStats.rope_split.command_buffer_count +
                      decoderLayerStats.mla_attention.command_buffer_count);
                decoderPayload[@"attn_output_elapsed_seconds"] =
                    @(decoderLayerStats.attn_output.elapsed_seconds);
                decoderPayload[@"attn_output_bytes_read"] =
                    @(decoderLayerStats.attn_output.bytes_read);
                decoderPayload[@"attn_output_projection_kernel_seconds"] =
                    @(decoderLayerStats.attn_output.projection_kernel_seconds);
                decoderPayload[@"attn_output_residual_add_seconds"] =
                    @(decoderLayerStats.attn_output.residual_add_seconds);
                decoderPayload[@"attn_output_fused_matvec_add"] =
                    @(decoderLayerStats.attn_output.fused_matvec_add ? YES : NO);
                decoderPayload[@"attn_output_context1_o_proj_cache"] =
                    @(decoderLayerStats.attn_output.context1_o_proj_cache ? YES : NO);
                decoderPayload[@"attn_output_resident_mmap_backed"] =
                    @(decoderLayerStats.attn_output.resident_mmap_backed ? YES : NO);
                decoderPayload[@"attn_output_command_buffer_count"] =
                    @(decoderLayerStats.attn_output.command_buffer_count);
                decoderPayload[@"rmsnorm_elapsed_seconds"] =
                    @(decoderLayerStats.rms_norm.elapsed_seconds);
                decoderPayload[@"post_attn_norm_router_fused"] =
                    @((decoderLayerStats.rms_norm.fused_with_router &&
                       decoderLayerStats.router.fused_with_rmsnorm) ? YES : NO);
                decoderPayload[@"post_attn_norm_command_buffer_count"] =
                    @(decoderLayerStats.rms_norm.command_buffer_count);
                decoderPayload[@"router_command_buffer_count"] =
                    @(decoderLayerStats.router.command_buffer_count);
                decoderPayload[@"router"] = router_probe_dictionary(
                    decoderLayerStats.router,
                    routerInfo,
                    options.probe_layer,
                    [NSString stringWithUTF8String:options.output_dir]
                );
                decoderPayload[@"mlp_elapsed_seconds"] =
                    @(decoderLayerStats.mlp.elapsed_seconds);
                decoderPayload[@"mlp_expert_read_seconds"] =
                    @(decoderLayerStats.mlp.expert_read_seconds);
                decoderPayload[@"mlp_kernel_seconds"] =
                    @(decoderLayerStats.mlp.kernel_seconds);
                decoderPayload[@"mlp_output_write_seconds"] =
                    @(decoderLayerStats.mlp.output_write_seconds);
                double mlpOverheadSeconds =
                    decoderLayerStats.mlp.elapsed_seconds -
                    decoderLayerStats.mlp.expert_read_seconds -
                    decoderLayerStats.mlp.kernel_seconds -
                    decoderLayerStats.mlp.output_write_seconds;
                if (mlpOverheadSeconds < 0.0) {
                    mlpOverheadSeconds = 0.0;
                }
                decoderPayload[@"mlp_overhead_seconds"] = @(mlpOverheadSeconds);
                decoderPayload[@"mlp_fast_mxfp4_kernel"] =
                    @(decoderLayerStats.mlp.fast_mxfp4_kernel ? YES : NO);
                decoderPayload[@"mlp_residual_add_fused"] =
                    @(decoderLayerStats.mlp.residual_add_fused ? YES : NO);
                decoderPayload[@"mlp_command_buffer_count"] =
                    @(decoderLayerStats.mlp.command_buffer_count);
                decoderPayload[@"mlp_expert_bytes_read"] =
                    @(decoderLayerStats.mlp.expert_bytes_read);
                decoderPayload[@"mlp_expert_read_dispatch_count"] =
                    @(decoderLayerStats.mlp.expert_read_dispatch_count);
                decoderPayload[@"mlp_expert_read_task_count"] =
                    @(decoderLayerStats.mlp.expert_read_task_count);
                decoderPayload[@"mlp_expert_read_max_task_count"] =
                    @(decoderLayerStats.mlp.expert_read_max_task_count);
                decoderPayload[@"mlp_expert_read_max_worker_count"] =
                    @(decoderLayerStats.mlp.expert_read_max_worker_count);
                decoderPayload[@"mlp_expert_read_pool_dispatch_count"] =
                    @(decoderLayerStats.mlp.expert_read_pool_dispatch_count);
                decoderPayload[@"mlp_expert_read_serial_dispatch_count"] =
                    @(decoderLayerStats.mlp.expert_read_serial_dispatch_count);
                if (options.include_shared_expert) {
                    decoderPayload[@"shared_bytes_read"] =
                        @(decoderLayerStats.mlp.shared_bytes_read);
                    decoderPayload[@"shared_read_seconds"] =
                        @(decoderLayerStats.mlp.shared_read_seconds);
                    decoderPayload[@"shared_prefetch_used"] =
                        @(decoderLayerStats.mlp.shared_prefetch_used ? YES : NO);
                    decoderPayload[@"shared_prefetch_seconds"] =
                        @(decoderLayerStats.mlp.shared_prefetch_seconds);
                    decoderPayload[@"shared_kernel_seconds"] =
                        @(decoderLayerStats.mlp.shared_kernel_seconds);
                }
                decoderPayload[@"include_shared_expert"] =
                    @(options.include_shared_expert ? YES : NO);
                decoderPayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_decoder_layer"] = decoderPayload;
            }
            if (options.probe_dense_decoder_layer) {
                NSMutableDictionary *denseDecoderPayload = [NSMutableDictionary dictionary];
                denseDecoderPayload[@"enabled"] = @(1);
                denseDecoderPayload[@"ok"] = @(denseDecoderLayerOk);
                denseDecoderPayload[@"layer"] = @(options.probe_layer);
                denseDecoderPayload[@"position"] = @(denseDecoderLayerStats.position);
                denseDecoderPayload[@"context_length"] =
                    @(denseDecoderLayerStats.context_length);
                denseDecoderPayload[@"scratch_bytes"] = @(denseDecoderLayerScratchBytes);
                denseDecoderPayload[@"elapsed_seconds"] =
                    @(denseDecoderLayerStats.elapsed_seconds);
                denseDecoderPayload[@"output0"] = @(denseDecoderLayerStats.output0);
                denseDecoderPayload[@"work_dir"] =
                    [NSString stringWithUTF8String:options.output_dir];
                denseDecoderPayload[@"input_f32"] =
                    [NSString stringWithUTF8String:options.input_f32];
                denseDecoderPayload[@"output_f32"] =
                    [NSString stringWithUTF8String:options.output_f32];
                denseDecoderPayload[@"cache_layout"] =
                    [NSString stringWithUTF8String:options.cache_layout];
                denseDecoderPayload[@"cache_file"] =
                    [NSString stringWithUTF8String:options.cache_file];
                denseDecoderPayload[@"cache_write_bytes"] =
                    @(denseDecoderLayerStats.attn_projection.cache_write_bytes);
                denseDecoderPayload[@"cache_write_seconds"] =
                    @(denseDecoderLayerStats.attn_projection.cache_write_seconds);
                denseDecoderPayload[@"attn_projection_elapsed_seconds"] =
                    @(denseDecoderLayerStats.attn_projection.elapsed_seconds);
                denseDecoderPayload[@"attn_projection_fused_pre_cache"] =
                    @(denseDecoderLayerStats.attn_projection.fused_pre_cache ? YES : NO);
                denseDecoderPayload[@"attn_projection_command_buffer_count"] =
                    @(denseDecoderLayerStats.attn_projection.command_buffer_count);
                denseDecoderPayload[@"attn_projection_fused_pre_cache_seconds"] =
                    @(denseDecoderLayerStats.attn_projection.fused_pre_cache_seconds);
                denseDecoderPayload[@"rope_split_elapsed_seconds"] =
                    @(denseDecoderLayerStats.rope_split.elapsed_seconds);
                denseDecoderPayload[@"mla_attention_elapsed_seconds"] =
                    @(denseDecoderLayerStats.mla_attention.elapsed_seconds);
                denseDecoderPayload[@"mla_attention_kernel_seconds"] =
                    @(denseDecoderLayerStats.mla_attention.kernel_seconds);
                denseDecoderPayload[@"mla_value_read_seconds"] =
                    @(denseDecoderLayerStats.mla_attention.value_read_seconds);
                denseDecoderPayload[@"mla_value_cache_enabled"] =
                    @(denseDecoderLayerStats.mla_attention.value_cache_enabled ? YES : NO);
                denseDecoderPayload[@"mla_value_cache_hit"] =
                    @(denseDecoderLayerStats.mla_attention.value_cache_hit ? YES : NO);
                denseDecoderPayload[@"mla_value_cache_stored"] =
                    @(denseDecoderLayerStats.mla_attention.value_cache_stored ? YES : NO);
                denseDecoderPayload[@"mla_value_cache_bytes"] =
                    @(denseDecoderLayerStats.mla_attention.value_cache_bytes);
                denseDecoderPayload[@"mla_value_cache_total_bytes"] =
                    @(denseDecoderLayerStats.mla_attention.value_cache_total_bytes);
                denseDecoderPayload[@"rope_mla_fused"] =
                    @((denseDecoderLayerStats.rope_split.fused_with_mla &&
                       denseDecoderLayerStats.mla_attention.fused_with_rope) ? YES : NO);
                denseDecoderPayload[@"rope_mla_command_buffer_count"] =
                    @(denseDecoderLayerStats.rope_split.command_buffer_count +
                      denseDecoderLayerStats.mla_attention.command_buffer_count);
                denseDecoderPayload[@"attn_output_elapsed_seconds"] =
                    @(denseDecoderLayerStats.attn_output.elapsed_seconds);
                denseDecoderPayload[@"attn_output_bytes_read"] =
                    @(denseDecoderLayerStats.attn_output.bytes_read);
                denseDecoderPayload[@"attn_output_projection_kernel_seconds"] =
                    @(denseDecoderLayerStats.attn_output.projection_kernel_seconds);
                denseDecoderPayload[@"attn_output_residual_add_seconds"] =
                    @(denseDecoderLayerStats.attn_output.residual_add_seconds);
                denseDecoderPayload[@"attn_output_fused_matvec_add"] =
                    @(denseDecoderLayerStats.attn_output.fused_matvec_add ? YES : NO);
                denseDecoderPayload[@"attn_output_context1_o_proj_cache"] =
                    @(denseDecoderLayerStats.attn_output.context1_o_proj_cache ? YES : NO);
                denseDecoderPayload[@"attn_output_resident_mmap_backed"] =
                    @(denseDecoderLayerStats.attn_output.resident_mmap_backed ? YES : NO);
                denseDecoderPayload[@"attn_output_command_buffer_count"] =
                    @(denseDecoderLayerStats.attn_output.command_buffer_count);
                denseDecoderPayload[@"dense_mlp_elapsed_seconds"] =
                    @(denseDecoderLayerStats.dense_mlp.elapsed_seconds);
                denseDecoderPayload[@"dense_mlp_bytes_read"] =
                    @(denseDecoderLayerStats.dense_mlp.bytes_read);
                denseDecoderPayload[@"dense_mlp_fused_pipeline"] =
                    @(denseDecoderLayerStats.dense_mlp.fused_pipeline ? YES : NO);
                denseDecoderPayload[@"dense_mlp_command_buffer_count"] =
                    @(denseDecoderLayerStats.dense_mlp.command_buffer_count);
                denseDecoderPayload[@"dense_mlp_synchronous_wait_count"] =
                    @(denseDecoderLayerStats.dense_mlp.synchronous_wait_count);
                denseDecoderPayload[@"dense_mlp_async_submitted"] =
                    @(denseDecoderLayerStats.dense_mlp.async_submitted ? YES : NO);
                denseDecoderPayload[@"dense_mlp_fused_kernel_seconds"] =
                    @(denseDecoderLayerStats.dense_mlp.fused_kernel_seconds);
                denseDecoderPayload[@"dense_mlp_gate_kernel_seconds"] =
                    @(denseDecoderLayerStats.dense_mlp.gate_kernel_seconds);
                denseDecoderPayload[@"dense_mlp_up_kernel_seconds"] =
                    @(denseDecoderLayerStats.dense_mlp.up_kernel_seconds);
                denseDecoderPayload[@"dense_mlp_down_kernel_seconds"] =
                    @(denseDecoderLayerStats.dense_mlp.down_kernel_seconds);
                denseDecoderPayload[@"dense_mlp_swiglu_kernel_seconds"] =
                    @(denseDecoderLayerStats.dense_mlp.swiglu_kernel_seconds);
                denseDecoderPayload[@"dense_mlp_residual_add_seconds"] =
                    @(denseDecoderLayerStats.dense_mlp.residual_add_seconds);
                denseDecoderPayload[@"dense_mlp_output0"] =
                    @(denseDecoderLayerStats.dense_mlp.output0);
                denseDecoderPayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_dense_decoder_layer"] = denseDecoderPayload;
            }
            if (options.probe_decode_layers) {
                NSMutableDictionary *decodePayload = [NSMutableDictionary dictionary];
                NSMutableArray *layersPayload = [NSMutableArray arrayWithCapacity:
                    (NSUInteger)decodeLayers.count];
                int denseCount = 0;
                int moeCount = 0;
                int memoryInputCount = 0;
                uint64_t hotIntermediateBytes = 0;
                uint64_t expertBytesRead = 0;
                uint64_t denseMlpBytesRead = 0;
                for (int i = 0; i < decodeLayers.count; i++) {
                    DecodeLayerPlan *plan = &decodeLayerPlans[i];
                    DecodeLayerRunSummary *summary = &decodeLayerSummaries[i];
                    if (plan->is_dense) {
                        denseCount++;
                    } else {
                        moeCount++;
                    }
                    expertBytesRead += summary->expert_bytes_read;
                    denseMlpBytesRead += summary->dense_mlp_bytes_read;
                    hotIntermediateBytes += summary->hot_intermediate_memory_bytes;
                    if (summary->input_from_memory) {
                        memoryInputCount++;
                    }
                    NSMutableDictionary *item = [NSMutableDictionary dictionary];
                    item[@"layer"] = @(plan->layer);
                    item[@"kind"] = plan->is_dense ? @"dense" : @"moe";
                    item[@"ok"] = @(summary->ok ? YES : NO);
                    item[@"input_from_memory"] =
                        @(summary->input_from_memory ? YES : NO);
                    item[@"input_buffer_direct"] =
                        @(summary->input_buffer_direct ? YES : NO);
                    item[@"hot_intermediate_tensors"] =
                        @(summary->hot_intermediate_tensors);
                    item[@"hot_intermediate_memory_bytes"] =
                        @(summary->hot_intermediate_memory_bytes);
                    item[@"scratch_bytes"] = @(plan->layer_scratch_bytes);
                    item[@"elapsed_seconds"] = @(summary->elapsed_seconds);
                    double knownLayerSeconds =
                        summary->attn_projection_elapsed_seconds +
                        summary->mla_attention_elapsed_seconds +
                        summary->attn_output_elapsed_seconds +
                        summary->mlp_elapsed_seconds;
                    double layerOverheadSeconds =
                        summary->elapsed_seconds - knownLayerSeconds;
                    if (layerOverheadSeconds < 0.0) {
                        layerOverheadSeconds = 0.0;
                    }
                    item[@"layer_overhead_seconds"] =
                        @(layerOverheadSeconds);
                    item[@"attn_projection_elapsed_seconds"] =
                        @(summary->attn_projection_elapsed_seconds);
                    item[@"attn_projection_fused_pre_cache"] =
                        @(summary->attn_projection_fused_pre_cache ? YES : NO);
                    item[@"attn_projection_command_buffer_count"] =
                        @(summary->attn_projection_command_buffer_count);
                    item[@"attn_projection_synchronous_wait_count"] =
                        @(summary->attn_projection_synchronous_wait_count);
                    item[@"attn_projection_async_submitted"] =
                        @(summary->attn_projection_async_submitted ? YES : NO);
                    item[@"mla_attention_elapsed_seconds"] =
                        @(summary->mla_attention_elapsed_seconds);
                    item[@"mla_attention_cache_read_seconds"] =
                        @(summary->mla_attention_cache_read_seconds);
                    item[@"mla_attention_value_read_seconds"] =
                        @(summary->mla_attention_value_read_seconds);
                    item[@"mla_attention_kernel_seconds"] =
                        @(summary->mla_attention_kernel_seconds);
                    item[@"mla_attention_output_write_seconds"] =
                        @(summary->mla_attention_output_write_seconds);
                    item[@"mla_value_cache_hit_count"] =
                        @(summary->mla_value_cache_hit_count);
                    item[@"mla_value_cache_store_count"] =
                        @(summary->mla_value_cache_store_count);
                    item[@"mla_value_cache_bytes"] =
                        @(summary->mla_value_cache_bytes);
                    item[@"mla_value_cache_total_bytes"] =
                        @(summary->mla_value_cache_total_bytes);
                    item[@"rope_mla_fused"] =
                        @(summary->rope_mla_fused ? YES : NO);
                    item[@"rope_mla_input_buffer_direct"] =
                        @(summary->rope_mla_input_buffer_direct ? YES : NO);
                    item[@"rope_mla_command_buffer_count"] =
                        @(summary->rope_mla_command_buffer_count);
                    item[@"attn_output_elapsed_seconds"] =
                        @(summary->attn_output_elapsed_seconds);
                    item[@"attn_output_bytes_read"] =
                        @(summary->attn_output_bytes_read);
                    item[@"attn_output_read_seconds"] =
                        @(summary->attn_output_read_seconds);
                    item[@"attn_output_projection_kernel_seconds"] =
                        @(summary->attn_output_projection_kernel_seconds);
                    item[@"attn_output_fused_matvec_add"] =
                        @(summary->attn_output_fused_matvec_add ? YES : NO);
                    item[@"attn_output_context1_o_proj_cache"] =
                        @(summary->attn_output_context1_o_proj_cache ? YES : NO);
                    item[@"attn_output_resident_mmap_backed"] =
                        @(summary->attn_output_resident_mmap_backed ? YES : NO);
                    item[@"attn_output_norm_router_fused"] =
                        @(summary->attn_output_norm_router_fused ? YES : NO);
                    item[@"rope_mla_attn_output_norm_router_fused"] =
                        @(summary->rope_mla_attn_output_norm_router_fused ? YES : NO);
                    item[@"attn_output_command_buffer_count"] =
                        @(summary->attn_output_command_buffer_count);
                    item[@"post_attn_norm_router_fused"] =
                        @(summary->post_attn_norm_router_fused ? YES : NO);
                    item[@"attn_output_buffer_direct"] =
                        @(summary->attn_output_buffer_direct ? YES : NO);
                    item[@"post_attn_norm_command_buffer_count"] =
                        @(summary->post_attn_norm_command_buffer_count);
                    item[@"post_attn_norm_weight_bytes_read"] =
                        @(summary->post_attn_norm_weight_bytes_read);
                    item[@"post_attn_norm_weight_read_seconds"] =
                        @(summary->post_attn_norm_weight_read_seconds);
                    item[@"router_bytes_read"] = @(summary->router_bytes_read);
                    item[@"router_correction_bias_bytes_read"] =
                        @(summary->router_correction_bias_bytes_read);
                    item[@"router_read_seconds"] =
                        @(summary->router_read_seconds);
                    item[@"router_kernel_seconds"] =
                        @(summary->router_kernel_seconds);
                    item[@"router_command_buffer_count"] =
                        @(summary->router_command_buffer_count);
                    item[@"mlp_elapsed_seconds"] = @(summary->mlp_elapsed_seconds);
                    item[@"expert_bytes_read"] = @(summary->expert_bytes_read);
                    item[@"shared_bytes_read"] = @(summary->shared_bytes_read);
                    item[@"shared_read_seconds"] = @(summary->shared_read_seconds);
                    item[@"shared_prefetch_seconds"] =
                        @(summary->shared_prefetch_seconds);
                    item[@"shared_prefetch_used"] =
                        @(summary->shared_prefetch_used ? YES : NO);
                    item[@"expert_read_dispatch_count"] =
                        @(summary->expert_read_dispatch_count);
                    item[@"expert_read_task_count"] =
                        @(summary->expert_read_task_count);
                    item[@"expert_read_max_task_count"] =
                        @(summary->expert_read_max_task_count);
                    item[@"expert_read_max_worker_count"] =
                        @(summary->expert_read_max_worker_count);
                    item[@"expert_read_pool_dispatch_count"] =
                        @(summary->expert_read_pool_dispatch_count);
                    item[@"expert_read_serial_dispatch_count"] =
                        @(summary->expert_read_serial_dispatch_count);
                    item[@"expert_read_seconds"] =
                        @(summary->expert_read_seconds);
                    item[@"moe_mlp_kernel_seconds"] =
                        @(summary->moe_mlp_kernel_seconds);
                    item[@"moe_mlp_output_write_seconds"] =
                        @(summary->moe_mlp_output_write_seconds);
                    item[@"moe_mlp_overhead_seconds"] =
                        @(summary->moe_mlp_overhead_seconds);
                    item[@"router_topk_backend"] = summary->is_dense
                        ? @"none"
                        : (summary->router_gpu_topk ? @"metal" : @"cpu");
                    NSMutableArray *selectedExperts =
                        [NSMutableArray arrayWithCapacity:summary->router_top_k];
                    for (uint32_t route = 0;
                         route < summary->router_top_k;
                         route++) {
                        [selectedExperts addObject:@(summary->router_experts[route])];
                    }
                    item[@"selected_experts"] = selectedExperts;
                    item[@"moe_mlp_residual_add_fused"] =
                        @(summary->moe_mlp_residual_add_fused ? YES : NO);
                    item[@"moe_mlp_input_buffer_direct"] =
                        @(summary->moe_mlp_input_buffer_direct ? YES : NO);
                    item[@"moe_mlp_command_buffer_count"] =
                        @(summary->moe_mlp_command_buffer_count);
                    item[@"moe_mlp_synchronous_wait_count"] =
                        @(summary->moe_mlp_synchronous_wait_count);
                    item[@"dense_mlp_bytes_read"] = @(summary->dense_mlp_bytes_read);
                    item[@"dense_mlp_fused_pipeline"] =
                        @(summary->dense_mlp_fused_pipeline ? YES : NO);
                    item[@"dense_mlp_command_buffer_count"] =
                        @(summary->dense_mlp_command_buffer_count);
                    item[@"dense_mlp_synchronous_wait_count"] =
                        @(summary->dense_mlp_synchronous_wait_count);
                    item[@"dense_mlp_async_submitted"] =
                        @(summary->dense_mlp_async_submitted ? YES : NO);
                    item[@"output0"] = @(summary->output0);
                    [layersPayload addObject:item];
                }
                DecodeLayersAggregateStats decodeAggregate =
                    collect_decode_layers_aggregate_stats(
                        decodeLayerPlans,
                        decodeLayerSummaries,
                        (int)decodeLayers.count
                    );
                decodePayload[@"enabled"] = @(1);
                decodePayload[@"ok"] = @(decodeLayersOk);
                decodePayload[@"layer_count"] = @(decodeLayers.count);
                decodePayload[@"dense_layer_count"] = @(denseCount);
                decodePayload[@"moe_layer_count"] = @(moeCount);
                decodePayload[@"memory_chain_enabled"] = @(YES);
                decodePayload[@"memory_input_layer_count"] = @(memoryInputCount);
                decodePayload[@"memory_chain_bytes"] = decodeLayers.count > 0
                    ? @((uint64_t)decodeLayerPlans[0].input_norm.dim * sizeof(float))
                    : @(0);
                decodePayload[@"hot_intermediate_memory_enabled"] = @(YES);
                decodePayload[@"hot_intermediate_memory_bytes"] = @(hotIntermediateBytes);
                decodePayload[@"debug_intermediates_written"] =
                    @(!options.skip_debug_intermediates);
                decodePayload[@"layers"] = layersPayload;
                decodePayload[@"scratch_bytes"] = @(decodeLayersScratchBytes);
                decodePayload[@"elapsed_seconds"] = @(decodeLayersElapsedSeconds);
                decodePayload[@"expert_bytes_read"] = @(expertBytesRead);
                decodePayload[@"dense_mlp_bytes_read"] = @(denseMlpBytesRead);
                add_decode_layers_aggregate_payload_fields(
                    decodePayload,
                    decodeAggregate
                );
                decodePayload[@"output0"] = @(decodeLayersOutput0);
                if (options.input_f32) {
                    decodePayload[@"input_f32"] =
                        [NSString stringWithUTF8String:options.input_f32];
                }
                if (options.input_token_id >= 0) {
                    decodePayload[@"input_token_id"] = @(options.input_token_id);
                }
                if (promptTokenIds.count > 0) {
                    NSMutableArray *promptIds =
                        [NSMutableArray arrayWithCapacity:(NSUInteger)promptTokenIds.count];
                    for (int i = 0; i < promptTokenIds.count; i++) {
                        [promptIds addObject:@(promptTokenIds.values[i])];
                    }
                    decodePayload[@"prompt_token_ids"] = promptIds;
                    decodePayload[@"prompt_token_count"] = @(promptTokenIds.count);
                }
                if (options.output_f32) {
                    decodePayload[@"output_f32"] =
                        [NSString stringWithUTF8String:options.output_f32];
                }
                decodePayload[@"work_dir"] = [NSString stringWithUTF8String:options.output_dir];
                decodePayload[@"cache_layout"] =
                    [NSString stringWithUTF8String:options.cache_layout];
                decodePayload[@"cache_file"] =
                    [NSString stringWithUTF8String:options.cache_file];
                decodePayload[@"include_shared_expert"] =
                    @(options.include_shared_expert ? YES : NO);
                decodePayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_decode_layers"] = decodePayload;
            }
            if (generateStepPayloads) {
                NSMutableArray *tokenIds =
                    [NSMutableArray arrayWithCapacity:[generateStepPayloads count]];
                for (NSDictionary *stepPayload in generateStepPayloads) {
                    NSDictionary *generatedToken = stepPayload[@"generated_token"];
                    if ([generatedToken isKindOfClass:[NSDictionary class]]) {
                        [tokenIds addObject:generatedToken[@"token_id"] ?: @0];
                    }
                }
                NSMutableDictionary *generatePayload =
                    [NSMutableDictionary dictionary];
                generatePayload[@"enabled"] = @(1);
                generatePayload[@"ok"] =
                    @((decodeLayersOk && finalLogitsOk) ? YES : NO);
                generatePayload[@"entry"] = options.generate_token_ids
                    ? @"generate_token_ids"
                    : @"probe_decode_layers";
                generatePayload[@"step_count"] = @(options.generate_steps);
                generatePayload[@"generated_token_ids"] = tokenIds;
                generatePayload[@"steps"] = generateStepPayloads;
                if (promptPrefillStepPayloads) {
                    generatePayload[@"prompt_prefill"] = @{
                        @"enabled": @(YES),
                        @"ok": @(([promptPrefillStepPayloads count] ==
                                  (NSUInteger)promptTokenIds.count) ? YES : NO),
                        @"token_count": @(promptTokenIds.count),
                        @"steps": promptPrefillStepPayloads,
                    };
                }
                generatePayload[@"cache_position_start"] =
                    @((uint64_t)options.cache_position);
                generatePayload[@"context_length_start"] =
                    @((uint32_t)options.context_length);
                generatePayload[@"first_step_from_input_logits"] =
                    @(((options.generate_first_from_input_logits ||
                        promptTokenIds.count > 0)) ? YES : NO);
                generatePayload[@"input_from_memory_after_first_step"] = @(YES);
                if (options.output_generated_json) {
                    generatePayload[@"output_generated_json"] =
                        [NSString stringWithUTF8String:options.output_generated_json];
                }
                payload[@"probe_generate"] = generatePayload;
            }
            if (options.probe_final_logits || decodeLayersFinalLogits) {
                NSMutableDictionary *logitsPayload = [NSMutableDictionary dictionary];
                logitsPayload[@"enabled"] = @(1);
                logitsPayload[@"ok"] = @(finalLogitsOk);
                logitsPayload[@"tensor"] =
                    [NSString stringWithUTF8String:finalLogitsHeadInfo.name];
                logitsPayload[@"norm_tensor"] = options.skip_final_norm
                    ? @"skipped"
                    : [NSString stringWithUTF8String:finalNormInfo.name];
                logitsPayload[@"dtype"] = @"mlx-mxfp4";
                logitsPayload[@"hidden_dim"] = @(finalLogitsStats.hidden_dim);
                logitsPayload[@"vocab_size"] = @(finalLogitsStats.vocab_size);
                logitsPayload[@"group_size"] = @(finalLogitsStats.group_size);
                logitsPayload[@"top_k"] = @(finalLogitsStats.top_k);
                logitsPayload[@"topk"] = final_topk_array(finalLogitsStats);
                logitsPayload[@"bytes_read"] = @(finalLogitsStats.bytes_read);
                logitsPayload[@"lm_head_bytes_read"] =
                    @(finalLogitsStats.lm_head_bytes_read);
                logitsPayload[@"resident_mmap_backed"] =
                    @(finalLogitsStats.resident_mmap_backed ? YES : NO);
                logitsPayload[@"mmap_final_logits"] =
                    @(options.mmap_final_logits ? YES : NO);
                logitsPayload[@"scratch_bytes"] = @(finalLogitsScratchBytes);
                logitsPayload[@"mmap_bytes"] = @(finalLogitsMmapBytes);
                logitsPayload[@"chunk_rows"] = @(finalLogitsStats.chunk_rows);
                logitsPayload[@"chunks"] = @(finalLogitsStats.chunks);
                logitsPayload[@"elapsed_seconds"] = @(finalLogitsStats.elapsed_seconds);
                logitsPayload[@"norm_elapsed_seconds"] =
                    @(finalLogitsStats.norm_elapsed_seconds);
                logitsPayload[@"read_seconds"] = @(finalLogitsStats.read_seconds);
                logitsPayload[@"kernel_seconds"] = @(finalLogitsStats.kernel_seconds);
                if (options.probe_final_logits) {
                    logitsPayload[@"input_f32"] =
                        [NSString stringWithUTF8String:options.input_f32];
                } else if (options.output_f32) {
                    logitsPayload[@"input_f32"] =
                        [NSString stringWithUTF8String:options.output_f32];
                }
                logitsPayload[@"input_from_memory"] =
                    @(finalLogitsStats.input_from_memory ? YES : NO);
                logitsPayload[@"source"] = options.probe_final_logits
                    ? @"standalone"
                    : (finalLogitsStats.input_from_memory
                        ? @"decode_layers_memory_output"
                        : @"decode_layers_output");
                logitsPayload[@"skip_final_norm"] = @(options.skip_final_norm ? YES : NO);
                logitsPayload[@"skipped_due_live_cap"] = @(!liveOk);
                if (options.output_topk_json) {
                    logitsPayload[@"output_topk_json"] =
                        [NSString stringWithUTF8String:options.output_topk_json];
                }
                NSDictionary *generatedToken =
                    final_generated_token_payload(finalLogitsStats);
                if (generatedToken) {
                    logitsPayload[@"generated_token"] = generatedToken;
                }
                if (options.output_token_json) {
                    logitsPayload[@"output_token_json"] =
                        [NSString stringWithUTF8String:options.output_token_json];
                }
                if (options.output_next_input_f32) {
                    logitsPayload[@"output_next_input_f32"] =
                        [NSString stringWithUTF8String:options.output_next_input_f32];
                    logitsPayload[@"next_input_embedding"] = @{
                        @"ok": @(nextInputEmbeddingStats.ok ? YES : NO),
                        @"token_id": @(nextInputEmbeddingStats.token_id),
                        @"tensor": [NSString stringWithUTF8String:nextInputEmbeddingInfo.name],
                        @"dtype": @"mlx-mxfp4",
                        @"vocab_size": @(nextInputEmbeddingStats.vocab_size),
                        @"hidden_dim": @(nextInputEmbeddingStats.hidden_dim),
                        @"group_size": @(nextInputEmbeddingStats.group_size),
                        @"bytes_read": @(nextInputEmbeddingStats.bytes_read),
                        @"output_bytes": @(nextInputEmbeddingStats.output_bytes),
                        @"elapsed_seconds": @(nextInputEmbeddingStats.elapsed_seconds),
                        @"read_seconds": @(nextInputEmbeddingStats.read_seconds),
                        @"decode_seconds": @(nextInputEmbeddingStats.decode_seconds),
                        @"write_seconds": @(nextInputEmbeddingStats.write_seconds),
                        @"output0": @(nextInputEmbeddingStats.output0),
                    };
                }
                payload[@"probe_final_logits"] = logitsPayload;
            }
            if (options.probe_mlp_block) {
                NSMutableDictionary *rmsPayload = [NSMutableDictionary dictionary];
                rmsPayload[@"enabled"] = @(1);
                rmsPayload[@"ok"] = @(rmsNormOk);
                rmsPayload[@"layer"] = @(options.probe_layer);
                rmsPayload[@"tensor"] = [NSString stringWithUTF8String:rmsNormInfo.name];
                rmsPayload[@"dtype"] = [NSString stringWithUTF8String:rmsNormInfo.dtype];
                rmsPayload[@"hidden_dim"] = @(rmsNormInfo.dim);
                rmsPayload[@"eps"] = @(options.rms_norm_eps);
                rmsPayload[@"weight_bytes_read"] = @(rmsNormStats.weight_bytes_read);
                rmsPayload[@"elapsed_seconds"] = @(rmsNormStats.elapsed_seconds);
                rmsPayload[@"output0"] = @(rmsNormStats.output0);
                rmsPayload[@"scratch_bytes"] = @(rmsNormScratchBytes);
                rmsPayload[@"input_f32"] = [NSString stringWithUTF8String:options.input_f32];
                rmsPayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_rmsnorm"] = rmsPayload;
            }
            if (options.probe_router) {
                NSMutableDictionary *routerPayload = [
                    router_probe_dictionary(
                        routerStats,
                        routerInfo,
                        options.probe_layer,
                        [NSString stringWithUTF8String:options.input_f32]
                    ) mutableCopy
                ];
                routerPayload[@"scratch_bytes"] = @(routerScratchBytes);
                routerPayload[@"skipped_due_live_cap"] = @(!liveOk);
                if (options.output_router_json) {
                    routerPayload[@"output_router_json"] =
                        [NSString stringWithUTF8String:options.output_router_json];
                }
                payload[@"probe_router"] = routerPayload;
            }
            if (options.probe_layer_moe || options.probe_router_moe) {
                NSMutableDictionary *moePayload = [NSMutableDictionary dictionary];
                int routeCount = options.probe_router_moe
                    ? (int)routerStats.top_k
                    : probeExperts.count;
                NSMutableArray *expertIds = [NSMutableArray arrayWithCapacity:(NSUInteger)routeCount];
                NSMutableArray *routeWeights = [NSMutableArray arrayWithCapacity:(NSUInteger)routeCount];
                for (int i = 0; i < routeCount; i++) {
                    [expertIds addObject:@(options.probe_router_moe
                        ? routerStats.experts[i]
                        : probeExperts.values[i])];
                    [routeWeights addObject:@(options.probe_router_moe
                        ? routerStats.weights[i]
                        : probeWeights.values[i])];
                }
                moePayload[@"enabled"] = @(1);
                moePayload[@"ok"] = @(layerMoeOk);
                moePayload[@"route_source"] = options.probe_router_moe ? @"router" : @"manual";
                moePayload[@"residual_add"] = @(options.probe_mlp_block ? YES : NO);
                moePayload[@"include_shared_expert"] = @(options.include_shared_expert ? YES : NO);
                moePayload[@"layer"] = @(options.probe_layer);
                moePayload[@"experts"] = expertIds;
                moePayload[@"weights"] = routeWeights;
                moePayload[@"hidden_dim"] = @(probeMoeInfo.hidden_dim);
                moePayload[@"intermediate_dim"] = @(probeMoeInfo.intermediate_dim);
                moePayload[@"group_size"] = @(probeMoeInfo.group_size);
                moePayload[@"scratch_bytes"] = @(probeMoeScratchBytes);
                moePayload[@"expert_bytes_read"] = @(layerMoeStats.expert_bytes_read);
                moePayload[@"shared_bytes_read"] = @(layerMoeStats.shared_bytes_read);
                moePayload[@"elapsed_seconds"] = @(layerMoeStats.elapsed_seconds);
                moePayload[@"expert_read_seconds"] = @(layerMoeStats.expert_read_seconds);
                moePayload[@"expert_read_dispatch_count"] =
                    @(layerMoeStats.expert_read_dispatch_count);
                moePayload[@"expert_read_task_count"] =
                    @(layerMoeStats.expert_read_task_count);
                moePayload[@"expert_read_max_task_count"] =
                    @(layerMoeStats.expert_read_max_task_count);
                moePayload[@"expert_read_max_worker_count"] =
                    @(layerMoeStats.expert_read_max_worker_count);
                moePayload[@"expert_read_pool_dispatch_count"] =
                    @(layerMoeStats.expert_read_pool_dispatch_count);
                moePayload[@"expert_read_serial_dispatch_count"] =
                    @(layerMoeStats.expert_read_serial_dispatch_count);
                moePayload[@"kernel_seconds"] = @(layerMoeStats.kernel_seconds);
                moePayload[@"fast_mxfp4_kernel"] =
                    @(layerMoeStats.fast_mxfp4_kernel ? YES : NO);
                moePayload[@"residual_add_fused"] =
                    @(layerMoeStats.residual_add_fused ? YES : NO);
                moePayload[@"async_submitted"] =
                    @(layerMoeStats.async_submitted ? YES : NO);
                moePayload[@"command_buffer_count"] =
                    @(layerMoeStats.command_buffer_count);
                moePayload[@"synchronous_wait_count"] =
                    @(layerMoeStats.synchronous_wait_count);
                moePayload[@"shared_read_seconds"] = @(layerMoeStats.shared_read_seconds);
                moePayload[@"shared_prefetch_used"] =
                    @(layerMoeStats.shared_prefetch_used ? YES : NO);
                moePayload[@"shared_prefetch_seconds"] =
                    @(layerMoeStats.shared_prefetch_seconds);
                moePayload[@"shared_kernel_seconds"] = @(layerMoeStats.shared_kernel_seconds);
                moePayload[@"output_write_seconds"] = @(layerMoeStats.output_write_seconds);
                moePayload[@"output0"] = @(layerMoeStats.output0);
                moePayload[@"output0_check_ok"] = @(layerMoeStats.output0_check_ok);
                if (options.expect_output0_set) {
                    moePayload[@"expected_output0"] = @(options.expect_output0);
                    moePayload[@"output0_abs_error"] = @(layerMoeStats.output0_abs_error);
                }
                moePayload[@"input_f32"] = [NSString stringWithUTF8String:options.input_f32];
                moePayload[@"output_f32"] = [NSString stringWithUTF8String:options.output_f32];
                moePayload[@"skipped_due_live_cap"] = @(!liveOk);
                payload[@"probe_layer_moe"] = moePayload;
            }
            if (options.probe_mlp_block) {
                NSMutableDictionary *mlpPayload = [NSMutableDictionary dictionary];
                mlpPayload[@"enabled"] = @(1);
                mlpPayload[@"ok"] = @(overallOk);
                mlpPayload[@"layer"] = @(options.probe_layer);
                mlpPayload[@"include_shared_expert"] = @(options.include_shared_expert ? YES : NO);
                mlpPayload[@"residual_add"] = @YES;
                mlpPayload[@"rms_norm_eps"] = @(options.rms_norm_eps);
                mlpPayload[@"input_f32"] = [NSString stringWithUTF8String:options.input_f32];
                mlpPayload[@"output_f32"] = [NSString stringWithUTF8String:options.output_f32];
                payload[@"probe_mlp_block"] = mlpPayload;
            }
            NSData *jsonData = [NSJSONSerialization dataWithJSONObject:payload
                                                               options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                                 error:nil];
            fwrite([jsonData bytes], 1, [jsonData length], stdout);
            fputc('\n', stdout);
        } else {
            printf("LargerLM GLM MoE single-process loader\n");
            printf("  device:                 %s\n", [[device name] UTF8String]);
            printf("  resident layout:        %s\n", [residentLayoutPath UTF8String]);
            printf("  resident weight file:   %s\n", [residentBinPath UTF8String]);
            printf("  resident bytes:         %.3f GiB\n", residentFileBytes / 1073741824.0);
            printf("  resident mmap:          %s requested / %s actual\n",
                   options.mmap_resident ? "yes" : "no",
                   residentMap != MAP_FAILED ? "yes" : "no");
            printf("  resident Metal buffer:  %s\n", residentMetalBuffer ? "yes" : "no");
            printf("  expert layout:          %s\n", [expertLayoutPath UTF8String]);
            printf("  expert layers:          %lu\n", (unsigned long)[layers count]);
            printf("  expert files opened:    %d\n", openedExpertFiles);
            printf("  expert total bytes:     %.3f GiB\n", totalExpertBytes / 1073741824.0);
            printf("  max expert slot:        %.3f MiB\n", maxExpertSlotBytes / 1048576.0);
            printf("  expert buffers:         %d planned / %lu allocated / %d requested x %.3f MiB\n",
                   activeExpertBufferCount,
                   (unsigned long)[expertBuffers count],
                   options.expert_buffer_count,
                   expertBufferBytes / 1048576.0);
            printf("  expert buffer pool:     %lu runtime / %d live-estimate / %s reused\n",
                   (unsigned long)[runtime.expertBuffers count],
                   liveExpertBufferCount,
                   expertBufferPoolReused ? "yes" : "no");
            if (options.probe_layer_moe || options.probe_router_moe) {
                printf("  layer MoE scratch:      %.3f MiB\n",
                       probeMoeScratchBytes / 1048576.0);
            }
            if (options.include_shared_expert) {
                printf("  shared expert storage:  %.3f MiB\n",
                       sharedExpertStorageBytes / 1048576.0);
            }
            if (options.probe_router) {
                printf("  router scratch:         %.3f MiB\n",
                       routerScratchBytes / 1048576.0);
            }
            if (options.probe_mlp_block) {
                printf("  RMSNorm scratch:        %.3f MiB\n",
                       rmsNormScratchBytes / 1048576.0);
            }
            if (options.probe_resident_linear) {
                printf("  resident linear scratch: %.3f MiB\n",
                       residentLinearScratchBytes / 1048576.0);
            }
            if (options.probe_attn_projections) {
                printf("  attn projection scratch: %.3f MiB\n",
                       attnProjectionScratchBytes / 1048576.0);
            }
            if (options.probe_rope_split) {
                printf("  RoPE split scratch:     %.3f MiB\n",
                       ropeSplitScratchBytes / 1048576.0);
            }
            if (options.probe_mla_attention) {
                printf("  MLA attention scratch:  %.3f MiB\n",
                       mlaAttentionScratchBytes / 1048576.0);
            }
            if (options.probe_attn_output) {
                printf("  attn output scratch:    %.3f MiB\n",
                       attnOutputScratchBytes / 1048576.0);
            }
            if (options.probe_dense_mlp_block) {
                printf("  dense MLP scratch:      %.3f MiB\n",
                       denseMlpScratchBytes / 1048576.0);
            }
            if (options.probe_decoder_layer) {
                printf("  decoder layer scratch:  %.3f MiB\n",
                       decoderLayerScratchBytes / 1048576.0);
            }
            if (options.probe_dense_decoder_layer) {
                printf("  dense decoder scratch:  %.3f MiB\n",
                       denseDecoderLayerScratchBytes / 1048576.0);
            }
            if (options.probe_decode_layers) {
                printf("  decode layers scratch:  %.3f MiB\n",
                       decodeLayersScratchBytes / 1048576.0);
            }
            if (options.probe_final_logits || decodeLayersFinalLogits) {
                printf("  final logits scratch:   %.3f MiB\n",
                       finalLogitsScratchBytes / 1048576.0);
            }
            printf("  estimated live bytes:   %.3f GiB\n",
                   estimatedLiveBytes / 1073741824.0);
            if (options.max_live_working_set_mib > 0.0) {
                printf("  live cap:               %.3f MiB (%s)\n",
                       options.max_live_working_set_mib,
                       liveOk ? "ok" : "failed");
            }
            if (options.min_free_unified_memory_gib > 0.0) {
                printf("  min free unified mem:   %.3f GiB reserve (%s)\n",
                       options.min_free_unified_memory_gib,
                       freeUnifiedMemoryOk ? "ok" : "failed");
                if (systemMemorySnapshotOk) {
                    printf("  available unified mem:  %.3f GiB available / %.3f GiB required\n",
                           memorySnapshot.available_bytes / 1073741824.0,
                           requiredAvailableMemoryBytes / 1073741824.0);
                }
            }
            if (options.probe_expert_read) {
                if (probeResults) {
                    if (options.probe_all_layers) {
                        printf("  probe expert read:      all %lu layers, %d experts/layer x %d repeat\n",
                               (unsigned long)probeReadLayerCount,
                               probeExperts.count,
                               options.probe_repeat);
                    } else {
                        printf("  probe expert read:      layer %d, %d experts x %d repeat\n",
                               options.probe_layer,
                               probeExperts.count,
                               options.probe_repeat);
                    }
                    printf("  probe bytes read:       %.3f MiB\n",
                           probeBytesRead / 1048576.0);
                    printf("  probe elapsed:          %.6f s\n", probeElapsedSeconds);
                    printf("  probe throughput:       %.3f GiB/s\n",
                           probeThroughputGiBPerSecond);
                    printf("  probe read dispatch:    %u dispatches, %u tasks, max %u/task, max %u workers\n",
                           probeReadStats.dispatch_count,
                           probeReadStats.task_count,
                           probeReadStats.max_task_count,
                           probeReadStats.max_worker_count);
                } else {
                    printf("  probe expert read:      skipped\n");
                }
            }
            if (options.probe_resident_linear) {
                printf("  probe resident linear:  %s (%s)\n",
                       residentLinearInfo.name,
                       residentLinearOk ? "ok" : "failed");
                printf("  resident linear dims:   out=%u in=%u group=%u\n",
                       residentLinearInfo.out_dim,
                       residentLinearInfo.in_dim,
                       residentLinearInfo.group_size);
                printf("  resident linear read:   %.3f MiB\n",
                       residentLinearStats.bytes_read / 1048576.0);
                printf("  resident linear elapsed: %.6f s\n",
                       residentLinearStats.elapsed_seconds);
                printf("  resident linear kernel: %.6f s\n",
                       residentLinearStats.kernel_seconds);
                printf("  resident linear output[0]: %.6f\n",
                       residentLinearStats.output0);
                if (options.expect_output0_set) {
                    printf("  resident linear output[0] error: %.6f (%s)\n",
                           residentLinearStats.output0_abs_error,
                           residentLinearStats.output0_check_ok ? "ok" : "failed");
                }
            }
            if (options.probe_attn_projections) {
                printf("  probe attn projections: layer %d (%s)\n",
                       options.probe_layer,
                       attnProjectionOk ? "ok" : "failed");
                printf("  attn dims:              hidden=%u q_lora=%u q_out=%u kv_lora=%u kv_a=%u kv_rope=%u\n",
                       attnProjectionStats.hidden_dim,
                       attnProjectionStats.q_lora_dim,
                       attnProjectionStats.q_out_dim,
                       attnProjectionStats.kv_lora_dim,
                       attnProjectionStats.kv_a_out_dim,
                       attnProjectionStats.kv_rope_dim);
                printf("  attn bytes read:        %.3f MiB\n",
                       attnProjectionStats.bytes_read / 1048576.0);
                printf("  attn elapsed:           %.6f s\n",
                       attnProjectionStats.elapsed_seconds);
                printf("  attn q_b[0]:            %.6f\n",
                       attnProjectionStats.q_b_output0);
                printf("  attn kv_a[0]:           %.6f\n",
                       attnProjectionStats.kv_a_output0);
                printf("  attn output dir:        %s\n", options.output_dir);
                if (attnProjectionStats.cache_append) {
                    printf("  attn cache append:      position=%llu bytes=%llu elapsed=%.6f s\n",
                           (unsigned long long)attnProjectionStats.cache_position,
                           (unsigned long long)attnProjectionStats.cache_write_bytes,
                           attnProjectionStats.cache_write_seconds);
                    printf("  attn cache file:        %s\n", options.cache_file);
                }
            }
            if (options.probe_rope_split) {
                printf("  probe RoPE split:       %s\n",
                       ropeSplitOk ? "ok" : "failed");
                printf("  RoPE dims:              heads=%u qk_nope=%u rope=%u batch=%u start=%u\n",
                       ropeSplitStats.num_heads,
                       ropeSplitStats.qk_nope_dim,
                       ropeSplitStats.rope_dim,
                       ropeSplitStats.batch_tokens,
                       ropeSplitStats.start_position);
                printf("  RoPE elapsed:           %.6f s\n",
                       ropeSplitStats.elapsed_seconds);
                printf("  RoPE q_out[0]:          %.6f\n",
                       ropeSplitStats.q_output0);
                printf("  RoPE k_out[0]:          %.6f\n",
                       ropeSplitStats.k_output0);
            }
            if (options.probe_mla_attention) {
                printf("  probe MLA attention:    layer %d (%s)\n",
                       options.probe_layer,
                       mlaAttentionOk ? "ok" : "failed");
                printf("  MLA dims:               context=%u heads=%u kv_lora=%u qk_nope=%u rope=%u v=%u\n",
                       mlaAttentionStats.context_length,
                       mlaAttentionStats.num_heads,
                       mlaAttentionStats.kv_lora_dim,
                       mlaAttentionStats.qk_nope_dim,
                       mlaAttentionStats.rope_dim,
                       mlaAttentionStats.v_head_dim);
                printf("  MLA cache bytes:        raw=%llu f32=%llu\n",
                       (unsigned long long)mlaAttentionStats.raw_cache_bytes,
                       (unsigned long long)mlaAttentionStats.cache_f32_bytes);
                printf("  MLA value source:       absorbed-alias %.3f MiB f32 %.3f MiB\n",
                       mlaAttentionStats.value_storage_bytes / 1048576.0,
                       mlaAttentionStats.value_source_f32_bytes / 1048576.0);
                printf("  MLA elapsed:            %.6f s\n",
                       mlaAttentionStats.elapsed_seconds);
                printf("  MLA kernel:             %.6f s\n",
                       mlaAttentionStats.kernel_seconds);
                printf("  MLA output[0]:          %.6f\n",
                       mlaAttentionStats.output0);
            }
            if (options.probe_attn_output) {
                printf("  probe attn output:      layer %d (%s)\n",
                       options.probe_layer,
                       attnOutputOk ? "ok" : "failed");
                printf("  attn output tensor:     %s\n", attnOutputInfo.name);
                printf("  attn output dims:       out=%u in=%u group=%u\n",
                       attnOutputStats.out_dim,
                       attnOutputStats.in_dim,
                       attnOutputStats.group_size);
                printf("  attn output bytes read: %.3f MiB\n",
                       attnOutputStats.bytes_read / 1048576.0);
                printf("  attn output elapsed:    %.6f s\n",
                       attnOutputStats.elapsed_seconds);
                printf("  attn output o_proj:     %.6f s\n",
                       attnOutputStats.projection_kernel_seconds);
                printf("  attn output residual:   %.6f s\n",
                       attnOutputStats.residual_add_seconds);
                printf("  attn output[0]:         %.6f\n",
                       attnOutputStats.output0);
            }
            if (options.probe_dense_mlp_block) {
                printf("  probe dense MLP:        layer %d (%s)\n",
                       options.probe_layer,
                       denseMlpOk ? "ok" : "failed");
                printf("  dense MLP dims:         hidden=%u intermediate=%u group=%u\n",
                       denseMlpStats.hidden_dim,
                       denseMlpStats.intermediate_dim,
                       denseMlpStats.group_size);
                printf("  dense MLP bytes read:   %.3f MiB\n",
                       denseMlpStats.bytes_read / 1048576.0);
                printf("  dense MLP elapsed:      %.6f s\n",
                       denseMlpStats.elapsed_seconds);
                printf("  dense MLP gate:         read=%.6f s kernel=%.6f s\n",
                       denseMlpStats.gate_read_seconds,
                       denseMlpStats.gate_kernel_seconds);
                printf("  dense MLP up:           read=%.6f s kernel=%.6f s\n",
                       denseMlpStats.up_read_seconds,
                       denseMlpStats.up_kernel_seconds);
                printf("  dense MLP down:         read=%.6f s kernel=%.6f s\n",
                       denseMlpStats.down_read_seconds,
                       denseMlpStats.down_kernel_seconds);
                printf("  dense MLP SwiGLU:       %.6f s\n",
                       denseMlpStats.swiglu_kernel_seconds);
                printf("  dense MLP residual:     %.6f s\n",
                       denseMlpStats.residual_add_seconds);
                printf("  dense MLP output[0]:    %.6f\n",
                       denseMlpStats.output0);
                if (options.expect_output0_set) {
                    printf("  dense MLP output[0] error: %.6f (%s)\n",
                           denseMlpStats.output0_abs_error,
                           denseMlpStats.output0_check_ok ? "ok" : "failed");
                }
            }
            if (options.probe_decoder_layer) {
                printf("  probe decoder layer:    layer %d (%s)\n",
                       options.probe_layer,
                       decoderLayerOk ? "ok" : "failed");
                printf("  decoder position:       %u context=%u\n",
                       decoderLayerStats.position,
                       decoderLayerStats.context_length);
                printf("  decoder elapsed:        %.6f s\n",
                       decoderLayerStats.elapsed_seconds);
                printf("  decoder attn proj:      %.6f s\n",
                       decoderLayerStats.attn_projection.elapsed_seconds);
                printf("  decoder cache append:   bytes=%llu elapsed=%.6f s\n",
                       (unsigned long long)decoderLayerStats.attn_projection.cache_write_bytes,
                       decoderLayerStats.attn_projection.cache_write_seconds);
                printf("  decoder RoPE split:     %.6f s\n",
                       decoderLayerStats.rope_split.elapsed_seconds);
                printf("  decoder MLA:            %.6f s kernel=%.6f s\n",
                       decoderLayerStats.mla_attention.elapsed_seconds,
                       decoderLayerStats.mla_attention.kernel_seconds);
                printf("  decoder attn output:    %.6f s\n",
                       decoderLayerStats.attn_output.elapsed_seconds);
                printf("  decoder MLP:            %.6f s read=%.6f s kernel=%.6f s\n",
                       decoderLayerStats.mlp.elapsed_seconds,
                       decoderLayerStats.mlp.expert_read_seconds,
                       decoderLayerStats.mlp.kernel_seconds);
                printf("  decoder output[0]:      %.6f\n",
                       decoderLayerStats.output0);
            }
            if (options.probe_dense_decoder_layer) {
                printf("  probe dense decoder:    layer %d (%s)\n",
                       options.probe_layer,
                       denseDecoderLayerOk ? "ok" : "failed");
                printf("  dense decoder position: %u context=%u\n",
                       denseDecoderLayerStats.position,
                       denseDecoderLayerStats.context_length);
                printf("  dense decoder elapsed:  %.6f s\n",
                       denseDecoderLayerStats.elapsed_seconds);
                printf("  dense attn proj:        %.6f s\n",
                       denseDecoderLayerStats.attn_projection.elapsed_seconds);
                printf("  dense cache append:     bytes=%llu elapsed=%.6f s\n",
                       (unsigned long long)
                           denseDecoderLayerStats.attn_projection.cache_write_bytes,
                       denseDecoderLayerStats.attn_projection.cache_write_seconds);
                printf("  dense RoPE split:       %.6f s\n",
                       denseDecoderLayerStats.rope_split.elapsed_seconds);
                printf("  dense MLA:              %.6f s kernel=%.6f s\n",
                       denseDecoderLayerStats.mla_attention.elapsed_seconds,
                       denseDecoderLayerStats.mla_attention.kernel_seconds);
                printf("  dense attn output:      %.6f s\n",
                       denseDecoderLayerStats.attn_output.elapsed_seconds);
                printf("  dense MLP:              %.6f s gate=%.6f s up=%.6f s down=%.6f s\n",
                       denseDecoderLayerStats.dense_mlp.elapsed_seconds,
                       denseDecoderLayerStats.dense_mlp.gate_kernel_seconds,
                       denseDecoderLayerStats.dense_mlp.up_kernel_seconds,
                       denseDecoderLayerStats.dense_mlp.down_kernel_seconds);
                printf("  dense decoder output[0]: %.6f\n",
                       denseDecoderLayerStats.output0);
            }
            if (options.probe_decode_layers) {
                printf("  probe decode layers:    %d layers (%s)\n",
                       decodeLayers.count,
                       decodeLayersOk ? "ok" : "failed");
                printf("  decode elapsed:         %.6f s\n",
                       decodeLayersElapsedSeconds);
                printf("  decode output[0]:       %.6f\n",
                       decodeLayersOutput0);
                for (int i = 0; i < decodeLayers.count; i++) {
                    printf("    layer %d %-5s elapsed=%.6f s mlp=%.6f s output[0]=%.6f\n",
                           decodeLayerSummaries[i].layer,
                           decodeLayerSummaries[i].is_dense ? "dense" : "moe",
                           decodeLayerSummaries[i].elapsed_seconds,
                           decodeLayerSummaries[i].mlp_elapsed_seconds,
                           decodeLayerSummaries[i].output0);
                }
            }
            if (options.probe_final_logits || decodeLayersFinalLogits) {
                printf("  probe final logits:     %s\n",
                       finalLogitsOk ? "ok" : "failed");
                printf("  final logits tensor:    %s\n", finalLogitsHeadInfo.name);
                printf("  final logits dims:      vocab=%u hidden=%u group=%u\n",
                       finalLogitsStats.vocab_size,
                       finalLogitsStats.hidden_dim,
                       finalLogitsStats.group_size);
                printf("  final logits chunks:    %llu x %llu rows\n",
                       (unsigned long long)finalLogitsStats.chunks,
                       (unsigned long long)finalLogitsStats.chunk_rows);
                printf("  final logits read:      %.3f MiB\n",
                       finalLogitsStats.bytes_read / 1048576.0);
                printf("  final logits elapsed:   %.6f s\n",
                       finalLogitsStats.elapsed_seconds);
                printf("  final logits top-k:     ");
                for (uint32_t i = 0; i < finalLogitsStats.count; i++) {
                    printf("%llu:%.6g%s",
                           (unsigned long long)finalLogitsStats.ids[i],
                           finalLogitsStats.scores[i],
                           (i + 1 == finalLogitsStats.count) ? "" : ", ");
                }
                printf("\n");
            }
            if (options.probe_mlp_block) {
                printf("  probe RMSNorm:          layer %d (%s)\n",
                       options.probe_layer,
                       rmsNormOk ? "ok" : "failed");
                printf("  RMSNorm eps:            %.9g\n", options.rms_norm_eps);
                printf("  RMSNorm elapsed:        %.6f s\n", rmsNormStats.elapsed_seconds);
                printf("  RMSNorm output[0]:      %.6f\n", rmsNormStats.output0);
            }
            if (options.probe_layer_moe || options.probe_router_moe) {
                int routeCount = options.probe_router_moe
                    ? (int)routerStats.top_k
                    : probeExperts.count;
                printf("  probe layer MoE:        layer %d, %d experts (%s)\n",
                       options.probe_layer,
                       routeCount,
                       layerMoeOk ? "ok" : "failed");
                printf("  MoE dims:               hidden=%u intermediate=%u group=%u\n",
                       probeMoeInfo.hidden_dim,
                       probeMoeInfo.intermediate_dim,
                       probeMoeInfo.group_size);
                printf("  MoE expert bytes read:  %.3f MiB\n",
                       layerMoeStats.expert_bytes_read / 1048576.0);
                if (options.include_shared_expert) {
                    printf("  shared bytes read:      %.3f MiB\n",
                           layerMoeStats.shared_bytes_read / 1048576.0);
                }
                printf("  MoE elapsed:            %.6f s\n", layerMoeStats.elapsed_seconds);
                printf("  MoE expert read:        %.6f s\n", layerMoeStats.expert_read_seconds);
                printf("  MoE read dispatches:    %u (%u tasks, max batch %u, max workers %u)\n",
                       layerMoeStats.expert_read_dispatch_count,
                       layerMoeStats.expert_read_task_count,
                       layerMoeStats.expert_read_max_task_count,
                       layerMoeStats.expert_read_max_worker_count);
                printf("  MoE read pool/serial:   %u pool / %u serial\n",
                       layerMoeStats.expert_read_pool_dispatch_count,
                       layerMoeStats.expert_read_serial_dispatch_count);
                printf("  MoE kernel:             %.6f s\n", layerMoeStats.kernel_seconds);
                if (options.include_shared_expert) {
                    printf("  shared read:            %.6f s\n", layerMoeStats.shared_read_seconds);
                    printf("  shared kernel:          %.6f s\n", layerMoeStats.shared_kernel_seconds);
                }
                printf("  MoE output[0]:          %.6f\n", layerMoeStats.output0);
                if (options.expect_output0_set) {
                    printf("  MoE output[0] error:    %.6f (%s)\n",
                           layerMoeStats.output0_abs_error,
                           layerMoeStats.output0_check_ok ? "ok" : "failed");
                }
            }
            if (options.probe_router) {
                printf("  probe router:           layer %d, top-k %d (%s)\n",
                       options.probe_layer,
                       options.top_k,
                       routerOk ? "ok" : "failed");
                printf("  router tensor:          %s\n", routerInfo.name);
                printf("  router dtype:           %s\n", routerInfo.dtype);
                printf("  router dims:            experts=%u hidden=%u\n",
                       routerInfo.num_experts,
                       routerInfo.hidden_dim);
                printf("  router bytes read:      %.3f MiB\n",
                       routerStats.router_bytes_read / 1048576.0);
                printf("  router elapsed:         %.6f s\n", routerStats.elapsed_seconds);
                printf("  router top-k:           ");
                for (uint32_t i = 0; i < routerStats.top_k; i++) {
                    printf("%d%s", routerStats.experts[i],
                           (i + 1 == routerStats.top_k) ? "" : ",");
                }
                printf("\n");
            }
            printf("  result:                 %s\n", overallOk ? "ok" : "failed");
        }

        free(routerStats.logits);
        free(probeResults);
        free(decodeLayerSummaries);
        free(decodeLayerPlans);
        free(decodeLayers.values);
        free(promptTokenIds.values);
        free(probeWeights.values);
        free(probeExperts.values);
        residentMetalBuffer = nil;
        if (residentMap != MAP_FAILED) {
            munmap(residentMap, residentFileBytes);
        }
        if (residentFd >= 0) {
            close(residentFd);
        }
        return overallOk ? 0 : 1;
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        LoaderOptions options = parse_options(argc, argv);
        if (options.generate_server_jsonl) {
            return run_generate_server_jsonl(options);
        }
        return run_glm_moe_infer_once(options);
    }
}
