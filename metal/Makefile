CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph

TARGET = largerlm-runner
PREFILL_PROBE = prefill-backend-probe
GLM_MOE_INFER = glm_moe_infer

.PHONY: all clean

all: $(TARGET) $(PREFILL_PROBE) $(GLM_MOE_INFER)

$(TARGET): runner.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) runner.m -o $(TARGET)

$(PREFILL_PROBE): prefill_backend_probe.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) prefill_backend_probe.m -o $(PREFILL_PROBE)

$(GLM_MOE_INFER): glm_moe_infer.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) glm_moe_infer.m -o $(GLM_MOE_INFER)

clean:
	rm -f $(TARGET) $(PREFILL_PROBE) $(GLM_MOE_INFER)
