#include <fcntl.h>
#include <glob.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifndef F_NOCACHE
#define F_NOCACHE 48
#endif

static void usage(const char *program) {
  fprintf(stderr,
          "usage: %s MODEL_DIR [--total-gib N] [--chunk-mib N] "
          "[--inflight N] [--start-command N]\n",
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
  if (argc < 2) {
    usage(argv[0]);
    return 2;
  }
  std::string modelDir = argv[1];
  double totalGiB = 32.0;
  double chunkMiB = 19.0;
  uint64_t inflight = 8;
  uint64_t startCommand = 0;
  for (int i = 2; i < argc; ++i) {
    const char *arg = argv[i];
    if (!strcmp(arg, "--total-gib") && i + 1 < argc) {
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
    } else {
      fprintf(stderr, "unknown or incomplete option: %s\n", arg);
      usage(argv[0]);
      return 2;
    }
  }

  const uint64_t chunkBytes =
      (uint64_t)llround(chunkMiB * 1024.0 * 1024.0);
  const uint64_t requestedBytes =
      (uint64_t)llround(totalGiB * 1024.0 * 1024.0 * 1024.0);
  const uint64_t commandCount =
      (requestedBytes + chunkBytes - 1) / chunkBytes;

  glob_t matches = {};
  std::string pattern = modelDir + "/out-*.safetensors";
  int globStatus = glob(pattern.c_str(), 0, nullptr, &matches);
  if (globStatus != 0) {
    fprintf(stderr, "no out-*.safetensors shards found in %s\n",
            modelDir.c_str());
    globfree(&matches);
    return 1;
  }
  std::vector<int> descriptors;
  std::vector<uint64_t> fileSizes;
  for (size_t index = 0; index < matches.gl_pathc; ++index) {
    struct stat status = {};
    if (stat(matches.gl_pathv[index], &status) != 0) {
      perror("stat");
      globfree(&matches);
      return 1;
    }
    if ((uint64_t)status.st_size < chunkBytes) continue;
    int descriptor = open(matches.gl_pathv[index], O_RDONLY);
    if (descriptor < 0) {
      perror("open");
      globfree(&matches);
      return 1;
    }
    if (fcntl(descriptor, F_NOCACHE, 1) != 0) {
      perror("fcntl(F_NOCACHE)");
      close(descriptor);
      globfree(&matches);
      return 1;
    }
    descriptors.push_back(descriptor);
    fileSizes.push_back((uint64_t)status.st_size);
  }
  globfree(&matches);
  if (descriptors.empty()) {
    fprintf(stderr, "no shard is large enough for a %llu-byte chunk\n",
            (unsigned long long)chunkBytes);
    return 1;
  }

  std::vector<void *> buffers(inflight, nullptr);
  for (uint64_t worker = 0; worker < inflight; ++worker) {
    if (posix_memalign(&buffers[worker], 4096, chunkBytes) != 0) {
      fprintf(stderr, "failed to allocate bounded pread slot\n");
      return 1;
    }
  }

  std::atomic<uint64_t> errors(0);
  std::vector<uint64_t> checksums(inflight, 0);
  auto started = std::chrono::steady_clock::now();
  std::vector<std::thread> workers;
  workers.reserve(inflight);
  for (uint64_t worker = 0; worker < inflight; ++worker) {
    workers.emplace_back([&, worker]() {
      uint8_t *buffer = static_cast<uint8_t *>(buffers[worker]);
      for (uint64_t local = worker; local < commandCount; local += inflight) {
        uint64_t command = startCommand + local;
        size_t fileIndex = command % descriptors.size();
        uint64_t round = command / descriptors.size();
        uint64_t maxOffset = fileSizes[fileIndex] - chunkBytes;
        uint64_t offset =
            maxOffset == 0 ? 0 : (round * chunkBytes) % maxOffset;
        offset &= ~UINT64_C(4095);
        uint64_t readBytes = 0;
        while (readBytes < chunkBytes) {
          ssize_t count =
              pread(descriptors[fileIndex], buffer + readBytes,
                    chunkBytes - readBytes, (off_t)(offset + readBytes));
          if (count <= 0) {
            errors.fetch_add(1, std::memory_order_relaxed);
            break;
          }
          readBytes += (uint64_t)count;
        }
        if (readBytes == chunkBytes) {
          checksums[worker] += buffer[0];
          checksums[worker] += buffer[chunkBytes - 1];
        }
      }
    });
  }
  for (std::thread &worker : workers) worker.join();
  auto elapsedDuration = std::chrono::steady_clock::now() - started;
  double elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(elapsedDuration)
          .count();

  uint64_t checksum = 0;
  for (uint64_t value : checksums) checksum += value;
  for (void *buffer : buffers) free(buffer);
  for (int descriptor : descriptors) close(descriptor);
  if (errors.load(std::memory_order_relaxed) != 0) {
    fprintf(stderr, "%llu pread commands failed\n",
            (unsigned long long)errors.load());
    return 1;
  }

  uint64_t completedBytes = commandCount * chunkBytes;
  double gibPerSecond =
      (completedBytes / (double)(1024ULL * 1024ULL * 1024ULL)) / elapsed;
  double gbPerSecond = (completedBytes / 1.0e9) / elapsed;
  printf("{\"backend\":\"pread-f_nocache\",\"shards\":%zu,"
         "\"inflight\":%llu,\"chunk_bytes\":%llu,\"commands\":%llu,"
         "\"start_command\":%llu,\"bytes\":%llu,"
         "\"elapsed_seconds\":%.6f,\"gib_per_second\":%.3f,"
         "\"gb_per_second\":%.3f,\"buffer_bytes\":%llu,"
         "\"checksum\":%llu}\n",
         descriptors.size(), (unsigned long long)inflight,
         (unsigned long long)chunkBytes, (unsigned long long)commandCount,
         (unsigned long long)startCommand,
         (unsigned long long)completedBytes, elapsed, gibPerSecond, gbPerSecond,
         (unsigned long long)(inflight * chunkBytes),
         (unsigned long long)checksum);
  return 0;
}
