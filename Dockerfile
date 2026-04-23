# Stage 1: Build Ollama-OV from source
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git go-golang cmake build-essential curl

# Download and Setup GenAI Runtime (2026.1)
WORKDIR /opt
RUN curl -L https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.1/linux/openvino_genai_ubuntu24_2026.1.0.0_x86_64.tar.gz | tar -xz

# Build Ollama-OV
WORKDIR /build
RUN git clone https://github.com/openvinotoolkit/openvino_contrib.git
WORKDIR /build/openvino_contrib/modules/ollama_openvino
ENV CGO_ENABLED=1
ENV OpenVINO_DIR=/opt/openvino_genai_ubuntu24_2026.1.0.0_x86_64/runtime
ENV CGO_LDFLAGS="-L${OpenVINO_DIR}/lib/intel64/"
ENV CGO_CFLAGS="-I${OpenVINO_DIR}/include"
RUN go build -o ollama

# Stage 2: Runtime
FROM ubuntu:24.04
COPY --from=builder /opt/openvino_genai_ubuntu24_2026.1.0.0_x86_64 /opt/ov_genai
COPY --from=builder /build/openvino_contrib/modules/ollama_openvino/ollama /usr/bin/ollama

# Install uv and system dependencies
RUN apt-get update && apt-get install -y python3 curl libtbb12 && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Configure Environment
ENV LD_LIBRARY_PATH="/opt/ov_genai/runtime/lib/intel64"
ENV GODEBUG=cgocheck=0
WORKDIR /app
COPY . .
RUN uv pip install --system modelscope questionary

ENTRYPOINT ["ollama", "serve"]
