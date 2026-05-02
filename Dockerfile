# Stage 1: Build Ollama-OV from source
FROM public.ecr.aws/ubuntu/ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git golang cmake build-essential curl libtbb-dev ocl-icd-opencl-dev

# Download and Setup GenAI Runtime (2026.1)
WORKDIR /opt
RUN curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.1/linux/openvino_genai_ubuntu24_2026.1.0.0_x86_64.tar.gz | tar -xz

# Build Ollama-OV
WORKDIR /build
RUN git clone https://github.com/openvinotoolkit/openvino_contrib.git
RUN sed -i 's/OV_GENAI_STREAMMING_STATUS/OV_GENAI_STREAMING_STATUS/g' /build/openvino_contrib/modules/ollama_openvino/genai/genai.go

# --- INJECT PATCHES FOR DRAFT MODEL SUPPORT ---
RUN echo '#include "openvino/genai/llm_pipeline.hpp"' > /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '#include "openvino/genai/c/llm_pipeline.h"' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '#include <string>' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo 'struct ov_genai_llm_pipeline_opaque { std::shared_ptr<ov::genai::LLMPipeline> object; };' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo 'extern "C" {' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '    ov_status_e ov_genai_llm_pipeline_create_with_draft(const char* models_path, const char* device, const char* draft_path, int is_eagle, ov_genai_llm_pipeline** pipe) {' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '        try {' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '            auto draft_model_desc = ov::genai::draft_model(draft_path, device);' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '            auto properties = ov::AnyMap{draft_model_desc};' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '            if (std::string(device) == "NPU") {' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '                properties["MAX_PROMPT_LEN"] = 2048;' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '                properties["MIN_RESPONSE_LEN"] = 256;' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '                if (is_eagle) { properties["num_assistant_tokens"] = 7; }' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '            }' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '            ov_genai_llm_pipeline* _pipe = new ov_genai_llm_pipeline();' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '            _pipe->object = std::make_shared<ov::genai::LLMPipeline>(std::string(models_path), std::string(device), properties);' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '            *pipe = _pipe;' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '        } catch (...) { return (ov_status_e)1; }' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '        return (ov_status_e)0;' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '    }' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp && \
    echo '}' >> /build/openvino_contrib/modules/ollama_openvino/genai/draft_pipeline.cpp

RUN sed -i '/extern int goCallbackBridge(char\* input, void\* ptr);/a extern ov_status_e ov_genai_llm_pipeline_create_with_draft(const char* models_path, const char* device, const char* draft_path, int is_eagle, ov_genai_llm_pipeline** pipe);' /build/openvino_contrib/modules/ollama_openvino/genai/genai.go && \
    sed -i 's/func CreatePipeline(modelsPath string, device string)/func CreatePipeline(modelsPath string, device string, draftPath string, isEagle int)/' /build/openvino_contrib/modules/ollama_openvino/genai/genai.go && \
    sed -i '/cModelsPath := C.CString(modelsPath)/a \\tcDraftPath := C.CString(draftPath)\n\tdefer C.free(unsafe.Pointer(cDraftPath))' /build/openvino_contrib/modules/ollama_openvino/genai/genai.go && \
    sed -i 's/if device == "NPU" {/if draftPath != "" {\n\t\tC.ov_genai_llm_pipeline_create_with_draft(cModelsPath, cDevice, cDraftPath, C.int(isEagle), \&pipeline)\n\t} else if device == "NPU" {/' /build/openvino_contrib/modules/ollama_openvino/genai/genai.go

RUN sed -i 's/s.model = genai.CreatePipeline(ov_model_path, device)/draft_path := filepath.Join(ov_model_path, "draft_model")\n\tis_eagle := 0\n\tif _, err := os.Stat(draft_path); os.IsNotExist(err) {\n\t\tdraft_path = filepath.Join(ov_model_path, "eagle_model")\n\t\tif _, err := os.Stat(draft_path); !os.IsNotExist(err) {\n\t\t\tis_eagle = 1\n\t\t} else {\n\t\t\tdraft_path = ""\n\t\t}\n\t}\n\ts.model = genai.CreatePipeline(ov_model_path, device, draft_path, is_eagle)/' /build/openvino_contrib/modules/ollama_openvino/genai/runner/runner.go
# --- END PATCHES ---

WORKDIR /build/openvino_contrib/modules/ollama_openvino
ENV CGO_ENABLED=1
ENV OpenVINO_DIR=/opt/openvino_genai_ubuntu24_2026.1.0.0_x86_64/runtime
ENV CGO_LDFLAGS="-L${OpenVINO_DIR}/lib/intel64/"
ENV CGO_CFLAGS="-I${OpenVINO_DIR}/include"
ENV CGO_CXXFLAGS="-I${OpenVINO_DIR}/include -std=c++17"
RUN go build -o ollama

# Stage 2: Runtime
FROM public.ecr.aws/ubuntu/ubuntu:24.04
COPY --from=builder /opt/openvino_genai_ubuntu24_2026.1.0.0_x86_64 /opt/ov_genai
COPY --from=builder /build/openvino_contrib/modules/ollama_openvino/ollama /usr/bin/ollama

# Install uv and system dependencies
RUN apt-get update && apt-get install -y python3 curl libtbb12 && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Configure Environment
ENV LD_LIBRARY_PATH="/opt/ov_genai/runtime/lib/intel64"
ENV GODEBUG=cgocheck=0
WORKDIR /app
COPY . .
RUN uv pip install --system --break-system-packages modelscope questionary

ENTRYPOINT ["ollama", "serve"]
