CC = clang
CXX = clang++
CFLAGS = -O2 -Wall -Wextra -fobjc-arc
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph

TARGET = largerlm-runner
PREFILL_PROBE = prefill-backend-probe
GLM_MOE_INFER = glm_moe_infer
MTLIO_BENCH = mtlio-expert-io-bench
PREAD_BENCH = pread-expert-io-bench

.PHONY: all clean

all: $(TARGET) $(PREFILL_PROBE) $(GLM_MOE_INFER) $(MTLIO_BENCH) $(PREAD_BENCH)

$(TARGET): runner.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) runner.m -o $(TARGET)

$(PREFILL_PROBE): prefill_backend_probe.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) prefill_backend_probe.m -o $(PREFILL_PROBE)

$(GLM_MOE_INFER): glm_moe_infer.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) glm_moe_infer.m -o $(GLM_MOE_INFER)

$(MTLIO_BENCH): mtlio_expert_io_bench.mm
	$(CXX) $(CFLAGS) $(FRAMEWORKS) mtlio_expert_io_bench.mm -o $(MTLIO_BENCH)

$(PREAD_BENCH): pread_expert_io_bench.cc
	$(CXX) -O2 -Wall -Wextra -std=c++17 pread_expert_io_bench.cc -o $(PREAD_BENCH)

clean:
	rm -f $(TARGET) $(PREFILL_PROBE) $(GLM_MOE_INFER) $(MTLIO_BENCH) $(PREAD_BENCH)
