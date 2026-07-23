#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <math.h>
#include <stdint.h>

static BOOL has_flag(int argc, const char **argv, const char *flag) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], flag) == 0) {
            return YES;
        }
    }
    return NO;
}

static NSString *truncate_string(NSString *value, NSUInteger limit) {
    if (!value || value.length <= limit) {
        return value ?: @"";
    }
    return [[value substringToIndex:limit] stringByAppendingString:@"..."];
}

static void print_json(NSDictionary *payload) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload
                                                   options:0
                                                     error:&error];
    if (!data || error) {
        fprintf(stderr, "failed to serialize JSON\n");
        return;
    }
    fwrite(data.bytes, 1, data.length, stdout);
    fputc('\n', stdout);
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void compile_mpp_probe(id<MTLDevice> device, NSMutableDictionary *payload) {
    payload[@"mpp_compile_probe_ran"] = @(YES);
    payload[@"mpp_compile_probe_ok"] = @(NO);
    payload[@"mpp_compile_variant"] = @"";
    payload[@"mpp_compile_error"] = @"";

    if (!device) {
        payload[@"mpp_compile_error"] = @"no Metal device";
        return;
    }

    if (@available(macOS 26.0, *)) {
        NSString *body =
            @"using namespace metal;\n"
            @"using namespace mpp::tensor_ops;\n"
            @"kernel void largerlm_mpp_compile_probe(\n"
            @"    device half *a [[buffer(0)]],\n"
            @"    device half *b [[buffer(1)]],\n"
            @"    device half *d [[buffer(2)]]) {\n"
            @"    constexpr int SM = 32;\n"
            @"    constexpr int SN = 32;\n"
            @"    constexpr int K = 32;\n"
            @"    constexpr auto desc = matmul2d_descriptor(SM, SN);\n"
            @"    matmul2d<desc, execution_simdgroup> op;\n"
            @"    auto mA = tensor(a, dextents<int, 2>{K, SM}, array<int, 2>{1, K});\n"
            @"    auto mB = tensor(b, dextents<int, 2>{SN, K}, array<int, 2>{1, SN});\n"
            @"    auto mD = tensor(d, dextents<int, 2>{SN, SM}, array<int, 2>{1, SN});\n"
            @"    op.run(mA, mB, mD);\n"
            @"}\n";

        MTLCompileOptions *options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        NSArray<NSArray<NSString *> *> *variants = @[
            @[@"MetalPerformancePrimitives.framework",
              @"#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"],
            @[@"metal_stdlib", @"#include <metal_stdlib>\n"],
            @[@"metal_stdlib+metal_tensor",
              @"#include <metal_stdlib>\n#include <metal_tensor>\n"],
            @[@"metal_stdlib+metal_mpp",
              @"#include <metal_stdlib>\n#include <metal_mpp>\n"],
            @[@"metal_stdlib+metal_tensor+metal_mpp",
              @"#include <metal_stdlib>\n#include <metal_tensor>\n#include <metal_mpp>\n"],
            @[@"metal_stdlib+metal_performance_primitives",
              @"#include <metal_stdlib>\n#include <metal_performance_primitives>\n"],
        ];

        NSMutableArray<NSString *> *errors = [NSMutableArray array];
        for (NSArray<NSString *> *variant in variants) {
            NSString *name = variant[0];
            NSString *source = [variant[1] stringByAppendingString:body];
            NSError *error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                          options:options
                                                            error:&error];
            if (library) {
                payload[@"mpp_compile_probe_ok"] = @(YES);
                payload[@"mpp_compile_variant"] = name;
                payload[@"mpp_compile_error"] = @"";
                return;
            }
            NSString *message = error.localizedDescription ?: @"unknown compile error";
            [errors addObject:[NSString stringWithFormat:@"%@: %@", name, message]];
        }
        payload[@"mpp_compile_error"] = truncate_string([errors componentsJoinedByString:@"\n"], 1600);
    } else {
        payload[@"mpp_compile_error"] = @"macOS 26.0 runtime is required";
    }
}

static void run_mpp_probe(id<MTLDevice> device, NSMutableDictionary *payload) {
    payload[@"mpp_run_probe_ran"] = @(YES);
    payload[@"mpp_run_probe_ok"] = @(NO);
    payload[@"mpp_run_probe_error"] = @"";
    payload[@"mpp_run_probe_max_abs_error"] = [NSNull null];

    if (!device) {
        payload[@"mpp_run_probe_error"] = @"no Metal device";
        return;
    }

    if (@available(macOS 26.0, *)) {
        NSString *body =
            @"using namespace metal;\n"
            @"using namespace mpp::tensor_ops;\n"
            @"kernel void largerlm_mpp_run_probe(\n"
            @"    device half *a [[buffer(0)]],\n"
            @"    device half *b [[buffer(1)]],\n"
            @"    device half *d [[buffer(2)]]) {\n"
            @"    constexpr int SM = 32;\n"
            @"    constexpr int SN = 32;\n"
            @"    constexpr int K = 32;\n"
            @"    constexpr auto desc = matmul2d_descriptor(SM, SN);\n"
            @"    matmul2d<desc, execution_simdgroup> op;\n"
            @"    auto mA = tensor(a, dextents<int, 2>{K, SM}, array<int, 2>{1, K});\n"
            @"    auto mB = tensor(b, dextents<int, 2>{SN, K}, array<int, 2>{1, SN});\n"
            @"    auto mD = tensor(d, dextents<int, 2>{SN, SM}, array<int, 2>{1, SN});\n"
            @"    op.run(mA, mB, mD);\n"
            @"}\n";

        MTLCompileOptions *options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        NSArray<NSArray<NSString *> *> *variants = @[
            @[@"MetalPerformancePrimitives.framework",
              @"#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"],
            @[@"metal_stdlib+metal_mpp",
              @"#include <metal_stdlib>\n#include <metal_mpp>\n"],
            @[@"metal_stdlib+metal_tensor+metal_mpp",
              @"#include <metal_stdlib>\n#include <metal_tensor>\n#include <metal_mpp>\n"],
            @[@"metal_stdlib+metal_performance_primitives",
              @"#include <metal_stdlib>\n#include <metal_performance_primitives>\n"],
        ];

        NSMutableArray<NSString *> *errors = [NSMutableArray array];
        id<MTLFunction> function = nil;
        NSString *compiledVariant = @"";
        for (NSArray<NSString *> *variant in variants) {
            NSString *name = variant[0];
            NSString *source = [variant[1] stringByAppendingString:body];
            NSError *error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                          options:options
                                                            error:&error];
            if (!library) {
                NSString *message = error.localizedDescription ?: @"unknown compile error";
                [errors addObject:[NSString stringWithFormat:@"%@: %@", name, message]];
                continue;
            }
            function = [library newFunctionWithName:@"largerlm_mpp_run_probe"];
            if (function) {
                compiledVariant = name;
                break;
            }
            [errors addObject:[NSString stringWithFormat:@"%@: function not found", name]];
        }
        if (!function) {
            payload[@"mpp_run_probe_error"] = truncate_string(
                [errors componentsJoinedByString:@"\n"],
                1600
            );
            return;
        }
        payload[@"mpp_run_probe_kernel_variant"] = compiledVariant;
        payload[@"mpp_run_probe_shape"] = @"32x32x32";
        payload[@"mpp_run_probe_dtype"] = @"half";
        payload[@"mpp_run_probe_execution_path"] = @"mpp::tensor_ops::matmul2d";

        NSError *pipelineError = nil;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                     error:&pipelineError];
        if (!pipeline) {
            payload[@"mpp_run_probe_error"] = truncate_string(
                pipelineError.localizedDescription ?: @"failed to create MPP probe pipeline",
                800
            );
            return;
        }

        const NSUInteger elements = 32 * 32;
        NSMutableData *ones = [NSMutableData dataWithLength:elements * sizeof(uint16_t)];
        NSMutableData *zeros = [NSMutableData dataWithLength:elements * sizeof(uint16_t)];
        uint16_t *oneValues = (uint16_t *)ones.mutableBytes;
        for (NSUInteger i = 0; i < elements; i++) {
            oneValues[i] = 0x3c00u;
        }
        id<MTLBuffer> aBuffer = [device newBufferWithBytes:ones.bytes
                                                    length:ones.length
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuffer = [device newBufferWithBytes:ones.bytes
                                                    length:ones.length
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> dBuffer = [device newBufferWithBytes:zeros.bytes
                                                    length:zeros.length
                                                   options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!aBuffer || !bBuffer || !dBuffer || !queue) {
            payload[@"mpp_run_probe_error"] = @"failed to allocate tiny MPP probe buffers";
            return;
        }

        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:aBuffer offset:0 atIndex:0];
        [encoder setBuffer:bBuffer offset:0 atIndex:1];
        [encoder setBuffer:dBuffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) {
            payload[@"mpp_run_probe_error"] = truncate_string(
                command.error.localizedDescription ?: @"MPP probe command failed",
                800
            );
            return;
        }

        uint16_t *actual = (uint16_t *)dBuffer.contents;
        float maxAbsError = 0.0f;
        for (NSUInteger i = 0; i < elements; i++) {
            float value = half_to_float(actual[i]);
            float error = fabsf(value - 32.0f);
            if (error > maxAbsError) {
                maxAbsError = error;
            }
        }
        payload[@"mpp_run_probe_max_abs_error"] = @(maxAbsError);
        if (maxAbsError > 0.05f) {
            payload[@"mpp_run_probe_error"] = [NSString stringWithFormat:
                @"MPP matmul2d result mismatch for %@: max_abs_error=%.6g",
                compiledVariant,
                maxAbsError];
            return;
        }
        payload[@"mpp_run_probe_ok"] = @(YES);
        if ([payload[@"mpp_compile_probe_ok"] isEqual:[NSNull null]] ||
            [payload[@"mpp_compile_probe_ok"] isEqual:@(NO)]) {
            payload[@"mpp_compile_probe_ok"] = @(YES);
            payload[@"mpp_compile_variant"] = compiledVariant;
            payload[@"mpp_compile_error"] = @"";
        }
    } else {
        payload[@"mpp_run_probe_error"] = @"macOS 26.0 runtime is required";
    }
}

static void run_mpsgraph_probe(id<MTLDevice> device, NSMutableDictionary *payload) {
    payload[@"mps_graph_probe_ran"] = @(YES);
    payload[@"mps_graph_probe_ok"] = @(NO);
    payload[@"mps_graph_probe_error"] = @"";

    if (!device) {
        payload[@"mps_graph_probe_error"] = @"no Metal device";
        return;
    }

    @try {
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            payload[@"mps_graph_probe_error"] = @"failed to create Metal command queue";
            return;
        }

        MPSGraph *graph = [MPSGraph new];
        NSArray<NSNumber *> *shape = @[@2, @2];
        MPSGraphTensor *lhs = [graph placeholderWithShape:shape
                                                  dataType:MPSDataTypeFloat32
                                                      name:@"largerlm_probe_lhs"];
        MPSGraphTensor *rhs = [graph placeholderWithShape:shape
                                                  dataType:MPSDataTypeFloat32
                                                      name:@"largerlm_probe_rhs"];
        MPSGraphTensor *product = [graph matrixMultiplicationWithPrimaryTensor:lhs
                                                               secondaryTensor:rhs
                                                                          name:@"largerlm_probe_matmul"];
        if (!lhs || !rhs || !product) {
            payload[@"mps_graph_probe_error"] = @"failed to build MPSGraph matmul";
            return;
        }

        float lhsValues[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float rhsValues[4] = {5.0f, 6.0f, 7.0f, 8.0f};
        id<MTLBuffer> lhsBuffer = [device newBufferWithBytes:lhsValues
                                                      length:sizeof(lhsValues)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> rhsBuffer = [device newBufferWithBytes:rhsValues
                                                      length:sizeof(rhsValues)
                                                     options:MTLResourceStorageModeShared];
        if (!lhsBuffer || !rhsBuffer) {
            payload[@"mps_graph_probe_error"] = @"failed to allocate tiny MPSGraph buffers";
            return;
        }

        MPSGraphTensorData *lhsData = [[MPSGraphTensorData alloc] initWithMTLBuffer:lhsBuffer
                                                                              shape:shape
                                                                           dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *rhsData = [[MPSGraphTensorData alloc] initWithMTLBuffer:rhsBuffer
                                                                              shape:shape
                                                                           dataType:MPSDataTypeFloat32];
        if (!lhsData || !rhsData) {
            payload[@"mps_graph_probe_error"] = @"failed to wrap tiny buffers as MPSGraphTensorData";
            return;
        }

        MPSGraphTensorDataDictionary *results = [graph runWithMTLCommandQueue:queue
                                                                         feeds:@{lhs: lhsData, rhs: rhsData}
                                                                 targetTensors:@[product]
                                                              targetOperations:nil];
        MPSGraphTensorData *productData = results[product];
        if (!productData) {
            payload[@"mps_graph_probe_error"] = @"MPSGraph matmul did not return a result tensor";
            return;
        }

        MPSNDArray *array = [productData mpsndarray];
        if (!array) {
            payload[@"mps_graph_probe_error"] = @"MPSGraph result could not be converted to MPSNDArray";
            return;
        }
        float actual[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        [array readBytes:actual strideBytes:nil];
        float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};
        for (NSUInteger i = 0; i < 4; i++) {
            if (fabsf(actual[i] - expected[i]) > 0.001f) {
                payload[@"mps_graph_probe_error"] = [NSString stringWithFormat:
                    @"MPSGraph matmul result mismatch at %lu: got %.6g expected %.6g",
                    (unsigned long)i, actual[i], expected[i]];
                return;
            }
        }

        payload[@"mps_graph_probe_ok"] = @(YES);
    } @catch (NSException *exception) {
        NSString *message = [NSString stringWithFormat:@"%@: %@",
                             exception.name ?: @"NSException",
                             exception.reason ?: @"unknown MPSGraph exception"];
        payload[@"mps_graph_probe_error"] = truncate_string(message, 800);
    }
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        NSMutableDictionary *payload = [NSMutableDictionary dictionary];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        BOOL shouldRunMPPProbe = has_flag(argc, argv, "--run-mpp-probe");
        BOOL shouldRunMPSGraphProbe = has_flag(argc, argv, "--run-mpsgraph-probe");
        payload[@"ok"] = @(device != nil);
        if (!device) {
            payload[@"device_name"] = @"";
            payload[@"supports_metal4_family"] = @(NO);
            payload[@"responds_new_mtl4_command_queue"] = @(NO);
            payload[@"responds_new_tensor"] = @(NO);
            payload[@"responds_tensor_size_align"] = @(NO);
            payload[@"responds_new_compiler"] = @(NO);
            payload[@"can_allocate_tiny_ml_tensor"] = @(NO);
            payload[@"tensor_error"] = @"no Metal device";
            payload[@"mpp_compile_probe_ran"] = @(has_flag(argc, argv, "--compile-mpp"));
            payload[@"mpp_compile_probe_ok"] = @(NO);
            payload[@"mpp_compile_variant"] = @"";
            payload[@"mpp_compile_error"] = (
                (has_flag(argc, argv, "--compile-mpp") || shouldRunMPPProbe)
                    ? @"no Metal device"
                    : @""
            );
            payload[@"mpp_run_probe_ran"] = @(shouldRunMPPProbe);
            payload[@"mpp_run_probe_ok"] = shouldRunMPPProbe ? @(NO) : [NSNull null];
            payload[@"mpp_run_probe_error"] = shouldRunMPPProbe ? @"no Metal device" : @"";
            payload[@"mpp_run_probe_max_abs_error"] = [NSNull null];
            payload[@"mpp_run_probe_kernel_variant"] = [NSNull null];
            payload[@"mpp_run_probe_shape"] = [NSNull null];
            payload[@"mpp_run_probe_dtype"] = [NSNull null];
            payload[@"mpp_run_probe_execution_path"] = [NSNull null];
            payload[@"mps_graph_probe_ran"] = @(shouldRunMPSGraphProbe);
            payload[@"mps_graph_probe_ok"] = shouldRunMPSGraphProbe ? @(NO) : [NSNull null];
            payload[@"mps_graph_probe_error"] = shouldRunMPSGraphProbe ? @"no Metal device" : @"";
            print_json(payload);
            return 0;
        }

        payload[@"device_name"] = device.name ?: @"";
        BOOL supportsMetal4Family = NO;
        BOOL respondsQueue = [device respondsToSelector:@selector(newMTL4CommandQueue)];
        BOOL respondsTensor = [device respondsToSelector:@selector(newTensorWithDescriptor:error:)];
        BOOL respondsTensorSize = [device respondsToSelector:@selector(tensorSizeAndAlignWithDescriptor:)];
        BOOL respondsCompiler = [device respondsToSelector:@selector(newCompilerWithDescriptor:error:)];
        BOOL canAllocateTinyTensor = NO;
        NSString *tensorError = @"";

        if (@available(macOS 26.0, *)) {
            if ([device respondsToSelector:@selector(supportsFamily:)]) {
                supportsMetal4Family = [device supportsFamily:MTLGPUFamilyMetal4];
            }
            if (respondsTensor && respondsTensorSize) {
                NSInteger dims[2] = {64, 1};
                MTLTensorDescriptor *desc = [MTLTensorDescriptor new];
                desc.dimensions = [[MTLTensorExtents alloc] initWithRank:2 values:dims];
                desc.dataType = MTLTensorDataTypeFloat16;
                desc.usage = MTLTensorUsageMachineLearning;
                desc.storageMode = MTLStorageModePrivate;

                NSError *error = nil;
                id<MTLTensor> tensor = [device newTensorWithDescriptor:desc error:&error];
                canAllocateTinyTensor = (tensor != nil);
                if (error) {
                    tensorError = error.localizedDescription ?: @"unknown tensor error";
                }
            }
        }

        payload[@"supports_metal4_family"] = @(supportsMetal4Family);
        payload[@"responds_new_mtl4_command_queue"] = @(respondsQueue);
        payload[@"responds_new_tensor"] = @(respondsTensor);
        payload[@"responds_tensor_size_align"] = @(respondsTensorSize);
        payload[@"responds_new_compiler"] = @(respondsCompiler);
        payload[@"can_allocate_tiny_ml_tensor"] = @(canAllocateTinyTensor);
        payload[@"tensor_error"] = tensorError;
        payload[@"mpp_compile_probe_ran"] = @(NO);
        payload[@"mpp_compile_probe_ok"] = [NSNull null];
        payload[@"mpp_compile_variant"] = @"";
        payload[@"mpp_compile_error"] = @"";
        payload[@"mpp_run_probe_ran"] = @(NO);
        payload[@"mpp_run_probe_ok"] = [NSNull null];
        payload[@"mpp_run_probe_error"] = @"";
        payload[@"mpp_run_probe_max_abs_error"] = [NSNull null];
        payload[@"mpp_run_probe_kernel_variant"] = [NSNull null];
        payload[@"mpp_run_probe_shape"] = [NSNull null];
        payload[@"mpp_run_probe_dtype"] = [NSNull null];
        payload[@"mpp_run_probe_execution_path"] = [NSNull null];
        payload[@"mps_graph_probe_ran"] = @(NO);
        payload[@"mps_graph_probe_ok"] = [NSNull null];
        payload[@"mps_graph_probe_error"] = @"";
        if (has_flag(argc, argv, "--compile-mpp") || shouldRunMPPProbe) {
            compile_mpp_probe(device, payload);
        }
        if (shouldRunMPPProbe) {
            run_mpp_probe(device, payload);
        }
        if (shouldRunMPSGraphProbe) {
            run_mpsgraph_probe(device, payload);
        }
        print_json(payload);
    }
    return 0;
}
