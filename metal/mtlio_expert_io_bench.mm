#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dispatch/dispatch.h>
#include <vector>

static void usage(const char *program) {
  fprintf(stderr,
          "usage: %s MODEL_DIR [--total-gib N] [--chunk-mib N] "
          "[--inflight N] [--start-command N] [--serial] "
          "[--storage shared|private]\n",
          program);
}

static bool parse_double(const char *text, double *value) {
  char *end = nullptr;
  *value = strtod(text, &end);
  return end != text && *end == '\0' && std::isfinite(*value);
}

static bool parse_u64(const char *text, uint64_t *value) {
  char *end = nullptr;
  unsigned long long parsed = strtoull(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = (uint64_t)parsed;
  return true;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2) {
      usage(argv[0]);
      return 2;
    }
    NSString *modelDir = [NSString stringWithUTF8String:argv[1]];
    double totalGiB = 32.0;
    double chunkMiB = 19.0;
    uint64_t inflight = 8;
    uint64_t startCommand = 0;
    bool serial = false;
    MTLStorageMode storageMode = MTLStorageModeShared;

    for (int i = 2; i < argc; ++i) {
      const char *arg = argv[i];
      if (!strcmp(arg, "--serial")) {
        serial = true;
      } else if (!strcmp(arg, "--total-gib") && i + 1 < argc) {
        if (!parse_double(argv[++i], &totalGiB) || totalGiB <= 0.0) {
          fprintf(stderr, "invalid --total-gib\n");
          return 2;
        }
      } else if (!strcmp(arg, "--chunk-mib") && i + 1 < argc) {
        if (!parse_double(argv[++i], &chunkMiB) || chunkMiB <= 0.0 ||
            chunkMiB > 256.0) {
          fprintf(stderr, "invalid --chunk-mib (must be in (0, 256])\n");
          return 2;
        }
      } else if (!strcmp(arg, "--inflight") && i + 1 < argc) {
        if (!parse_u64(argv[++i], &inflight) || inflight == 0 ||
            inflight > 64) {
          fprintf(stderr, "invalid --inflight (must be in [1, 64])\n");
          return 2;
        }
      } else if (!strcmp(arg, "--start-command") && i + 1 < argc) {
        if (!parse_u64(argv[++i], &startCommand)) {
          fprintf(stderr, "invalid --start-command\n");
          return 2;
        }
      } else if (!strcmp(arg, "--storage") && i + 1 < argc) {
        const char *mode = argv[++i];
        if (!strcmp(mode, "shared")) {
          storageMode = MTLStorageModeShared;
        } else if (!strcmp(mode, "private")) {
          storageMode = MTLStorageModePrivate;
        } else {
          fprintf(stderr, "invalid --storage (expected shared or private)\n");
          return 2;
        }
      } else {
        fprintf(stderr, "unknown or incomplete option: %s\n", arg);
        usage(argv[0]);
        return 2;
      }
    }

    NSError *error = nil;
    NSURL *directoryURL = [NSURL fileURLWithPath:modelDir isDirectory:YES];
    NSArray<NSURL *> *entries = [[NSFileManager defaultManager]
        contentsOfDirectoryAtURL:directoryURL
      includingPropertiesForKeys:@[ NSURLFileSizeKey ]
                         options:NSDirectoryEnumerationSkipsHiddenFiles
                           error:&error];
    if (entries == nil) {
      fprintf(stderr, "failed to list model directory: %s\n",
              error.localizedDescription.UTF8String);
      return 1;
    }
    NSPredicate *shardPredicate =
        [NSPredicate predicateWithBlock:^BOOL(NSURL *url,
                                              NSDictionary *bindings) {
          (void)bindings;
          NSString *name = url.lastPathComponent;
          return [name hasPrefix:@"out-"] &&
                 [name hasSuffix:@".safetensors"];
        }];
    NSArray<NSURL *> *candidateURLs =
        [[entries filteredArrayUsingPredicate:shardPredicate]
            sortedArrayUsingComparator:^NSComparisonResult(NSURL *a, NSURL *b) {
              return [a.lastPathComponent compare:b.lastPathComponent];
            }];
    if (candidateURLs.count == 0) {
      fprintf(stderr, "no out-*.safetensors shards found in %s\n",
              modelDir.UTF8String);
      return 1;
    }

    const uint64_t chunkBytes =
        (uint64_t)llround(chunkMiB * 1024.0 * 1024.0);
    const uint64_t requestedBytes =
        (uint64_t)llround(totalGiB * 1024.0 * 1024.0 * 1024.0);
    const uint64_t commandCount =
        (requestedBytes + chunkBytes - 1) / chunkBytes;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      fprintf(stderr, "no Metal device available\n");
      return 1;
    }
    MTLIOCommandQueueDescriptor *descriptor =
        [MTLIOCommandQueueDescriptor new];
    descriptor.type = serial ? MTLIOCommandQueueTypeSerial
                             : MTLIOCommandQueueTypeConcurrent;
    descriptor.priority = MTLIOPriorityHigh;
    descriptor.maxCommandBufferCount = (NSUInteger)inflight;
    descriptor.maxCommandsInFlight = (NSUInteger)inflight;
    id<MTLIOCommandQueue> queue =
        [device newIOCommandQueueWithDescriptor:descriptor error:&error];
    if (queue == nil) {
      fprintf(stderr, "failed to create MTLIO queue: %s\n",
              error.localizedDescription.UTF8String);
      return 1;
    }

    NSMutableArray<NSURL *> *urls =
        [NSMutableArray arrayWithCapacity:candidateURLs.count];
    NSMutableArray<id<MTLIOFileHandle>> *handles =
        [NSMutableArray arrayWithCapacity:candidateURLs.count];
    std::vector<uint64_t> fileSizes;
    fileSizes.reserve(candidateURLs.count);
    for (NSURL *url in candidateURLs) {
      NSNumber *size = nil;
      if (![url getResourceValue:&size forKey:NSURLFileSizeKey error:&error]) {
        fprintf(stderr, "failed to stat %s: %s\n", url.path.UTF8String,
                error.localizedDescription.UTF8String);
        return 1;
      }
      if (size.unsignedLongLongValue < chunkBytes) continue;
      id<MTLIOFileHandle> handle =
          [device newIOFileHandleWithURL:url error:&error];
      if (handle == nil) {
        fprintf(stderr, "failed to open %s with MTLIO: %s\n",
                url.path.UTF8String, error.localizedDescription.UTF8String);
        return 1;
      }
      [urls addObject:url];
      [handles addObject:handle];
      fileSizes.push_back(size.unsignedLongLongValue);
    }
    if (urls.count == 0) {
      fprintf(stderr, "no shard is large enough for a %llu-byte chunk\n",
              (unsigned long long)chunkBytes);
      return 1;
    }

    NSMutableArray<id<MTLBuffer>> *buffers =
        [NSMutableArray arrayWithCapacity:(NSUInteger)inflight];
    MTLResourceOptions options =
        storageMode == MTLStorageModeShared ? MTLResourceStorageModeShared
                                            : MTLResourceStorageModePrivate;
    for (uint64_t slot = 0; slot < inflight; ++slot) {
      id<MTLBuffer> buffer =
          [device newBufferWithLength:(NSUInteger)chunkBytes options:options];
      if (buffer == nil) {
        fprintf(stderr, "failed to allocate bounded MTLIO slot %llu\n",
                (unsigned long long)slot);
        return 1;
      }
      [buffers addObject:buffer];
    }

    std::atomic<uint64_t> errors(0);
    uint64_t checksum = 0;
    uint64_t completedBytes = 0;
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    for (uint64_t batchStart = 0; batchStart < commandCount;
         batchStart += inflight) {
      @autoreleasepool {
        uint64_t batchCount =
            std::min(inflight, commandCount - batchStart);
        dispatch_group_t group = dispatch_group_create();
        for (uint64_t slot = 0; slot < batchCount; ++slot) {
          uint64_t command = startCommand + batchStart + slot;
          NSUInteger fileIndex = (NSUInteger)(command % urls.count);
          uint64_t round = command / urls.count;
          uint64_t maxOffset = fileSizes[fileIndex] - chunkBytes;
          uint64_t offset = maxOffset == 0 ? 0 : (round * chunkBytes) % maxOffset;
          offset &= ~UINT64_C(4095);
          id<MTLIOCommandBuffer> io = [queue commandBuffer];
          id<MTLBuffer> destination = buffers[(NSUInteger)slot];
          [io loadBuffer:destination
                  offset:0
                    size:(NSUInteger)chunkBytes
            sourceHandle:handles[fileIndex]
      sourceHandleOffset:(NSUInteger)offset];
          dispatch_group_enter(group);
          std::atomic<uint64_t> *errorCounter = &errors;
          [io addCompletedHandler:^(id<MTLIOCommandBuffer> completed) {
            if (completed.status != MTLIOStatusComplete) {
              errorCounter->fetch_add(1, std::memory_order_relaxed);
              fprintf(stderr, "MTLIO command failed: %s\n",
                      completed.error.localizedDescription.UTF8String);
            }
            dispatch_group_leave(group);
          }];
          [io commit];
        }
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        if (errors.load(std::memory_order_relaxed) != 0) return 1;
        if (storageMode == MTLStorageModeShared) {
          for (uint64_t slot = 0; slot < batchCount; ++slot) {
            const uint8_t *bytes =
                (const uint8_t *)[buffers[(NSUInteger)slot] contents];
            checksum += bytes[0];
            checksum += bytes[chunkBytes - 1];
          }
        }
        completedBytes += batchCount * chunkBytes;
      }
    }
    CFAbsoluteTime elapsed = CFAbsoluteTimeGetCurrent() - started;
    double gibPerSecond = (completedBytes / (double)(1024ULL * 1024ULL * 1024ULL)) /
                          elapsed;
    double gbPerSecond = (completedBytes / 1.0e9) / elapsed;
    printf("{\"backend\":\"mtlio\",\"queue\":\"%s\",\"storage\":\"%s\","
           "\"shards\":%lu,\"inflight\":%llu,\"chunk_bytes\":%llu,"
           "\"commands\":%llu,\"start_command\":%llu,\"bytes\":%llu,"
           "\"elapsed_seconds\":%.6f,\"gib_per_second\":%.3f,"
           "\"gb_per_second\":%.3f,\"buffer_bytes\":%llu,"
           "\"checksum\":%llu}\n",
           serial ? "serial" : "concurrent",
           storageMode == MTLStorageModeShared ? "shared" : "private",
           (unsigned long)urls.count, (unsigned long long)inflight,
           (unsigned long long)chunkBytes, (unsigned long long)commandCount,
           (unsigned long long)startCommand,
           (unsigned long long)completedBytes, elapsed, gibPerSecond,
           gbPerSecond, (unsigned long long)(inflight * chunkBytes),
           (unsigned long long)checksum);
  }
  return 0;
}
