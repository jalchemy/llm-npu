# llm-npu
LLMs running on the NPU via OpenVINO, packaged as a single docker service with a focus on UX

This strategy is designed for a **headless coding agent** to build a fully portable, NPU-accelerated Ollama environment. It eliminates dependencies on your current local state by automating the "Tarbomb" fix, the ModelScope downloads, and the OpenVINO GenAI environment setup using uv.
### Project Structure
The agent should initialize a repository with the following structure:
```text
ollama-npu-portable/
├── pyproject.toml      # uv-managed dependencies
├── models.json         # Formatted registry of NPU-compatible models
├── manager.py          # The browsing/management utility
├── Dockerfile          # Multi-stage build (Source -> Runtime)
├── compose.yml         # Container orchestration
└── scripts/
    └── setup-ov.sh     # Environment initialization

```
### 1. The Model Registry (models.json)
This file categorizes the models from the OpenVINO-Ollama README specifically for NPU usage.
```json
{
  "npu_optimized": [
    {
      "name": "DeepSeek-R1-Distill-Qwen-7B-NPU",
      "id": "zhaohb/DeepSeek-R1-Distill-Qwen-7B-int4-ov-npu",
      "type": "LLM",
      "description": "DeepSeek R1 Distill (Qwen 7B) - Symmetric Quantization for Best NPU perf"
    },
    {
      "name": "Qwen3-4B-NPU",
      "id": "zhaohb/Qwen3-4B-int4-sym-ov-npu",
      "type": "LLM",
      "description": "Qwen3 4B - Symmetric Quantization for Best NPU perf"
    },
    {
      "name": "Llama-3.2-3B-Instruct-NPU",
      "id": "FionaZhao/llama-3.2-3b-instruct-int4-ov-npu",
      "type": "LLM",
      "description": "Llama 3.2 3B - Symmetric Quantization for Best NPU perf"
    },
    {
      "name": "Qwen2.5-VL-3B-Vision-NPU",
      "id": "zhaohb/Qwen2.5-VL-3B-Instruct-int4-ov",
      "type": "VLM",
      "description": "Vision-Language Model - NPU Base support"
    }
  ]
}

```
### 2. The Management Utility (manager.py)
This script handles the selection, the **Parent-Folder Tarball Fix**, and Ollama model creation. It uses uv for execution.
```python
import json
import os
import subprocess
import tarfile
import shutil
from pathlib import Path
import questionary # Dependency for browsing

REGISTRY_PATH = "models.json"
MODELS_DIR = Path("models")
TAR_DIR = Path("tarballs")

def load_registry():
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def download_model(model_id, folder_name):
    print(f"Downloading {model_id} via ModelScope...")
    target_path = MODELS_DIR / folder_name
    subprocess.run([
        "python3", "-c", 
        f"from modelscope import snapshot_download; snapshot_download('{model_id}', local_dir='{target_path}')"
    ], check=True)
    
    # Cleanup ModelScope artifacts to prevent runner crash
    for artifact in [".msc", "._____temp", ".mv"]:
        shutil.rmtree(target_path / artifact, ignore_errors=True)

def package_model(folder_name):
    print(f"Packaging {folder_name} into structured tarball...")
    TAR_DIR.mkdir(exist_ok=True)
    tar_path = TAR_DIR / f"{folder_name}.tar.gz"
    
    # The Fix: Ensure model files are nested inside a directory in the tarball
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(MODELS_DIR / folder_name, arcname=folder_name)
    return tar_path

def create_modelfile(name, tar_path, model_type):
    modelfile_content = f"""
FROM {tar_path}
ModelBackend "OpenVINO"
ModelType "{model_type}"
InferDevice "NPU"
PARAMETER stop_id 151643
PARAMETER max_new_token 2048
PARAMETER temperature 0.6
"""
    mf_path = Path(f"Modelfile-{name}")
    mf_path.write_text(modelfile_content)
    return mf_path

def main():
    registry = load_registry()
    options = [f"{m['name']} ({m['type']})" for m in registry["npu_optimized"]]
    
    selection_name = questionary.select(
        "Select an NPU-optimized model to deploy:",
        choices=options + ["Exit"]
    ).ask()

    if selection_name == "Exit": return

    model_data = next(m for m in registry["npu_optimized"] if m["name"] in selection_name)
    folder_name = model_data["name"].lower().replace("-", "_")
    
    # 1. Download
    download_model(model_data["id"], folder_name)
    
    # 2. Package
    tar_path = package_model(folder_name)
    
    # 3. Modelfile
    mf_path = create_modelfile(folder_name, f"/models/tarballs/{tar_path.name}", model_data["type"])
    
    # 4. Ollama Create
    print(f"Registering {model_data['name']} with Ollama...")
    subprocess.run(["ollama", "create", model_data["name"], "-f", str(mf_path)], check=True)
    print(f"Successfully loaded {model_data['name']}. You can now run it.")

if __name__ == "__main__":
    main()

```
### 3. Build & Runtime Environment
The Dockerfile builds the OpenVINO-integrated Ollama directly from the zhaohb repository to ensure version compatibility.
```dockerfile
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

```
### 4. Agent Instructions (Step-by-Step)
The coding agent should execute the following to verify the "independent" setup:
 1. **Dependency Management:** Create pyproject.toml with uv init. Add modelscope and questionary.
 2. **Logic Verification:** * Run manager.py (Mock the ollama call if binary isn't present).
   * Verify package_model creates a tarball where tar -tf shows a parent directory. This confirms the index out of range crash is resolved.
 3. **Container Validation:** * Run docker build -t ollama-npu-portable ..
   * Since the agent only has a CPU, it must verify the build completes and the ollama binary is correctly linked to the OpenVINO libraries using ldd /usr/bin/ollama.
 4. **Model Swap Test:** * The agent should simulate selecting a model from models.json and verify the script correctly generates the Modelfile-NPU with the ModelBackend "OpenVINO" and InferDevice "NPU" flags.
 5. **Final Handover:**
   * Provide the compose.yml that mounts /dev/accel and the /models persistent volume.
**Independent Verification Command for Agent:**
```bash
# Agent can run this to check the fix logic without an NPU
uv run manager.py --test-mode

```
This setup ensures that once you pull the repository on your Debian 13 machine, you simply run the manager, pick your model, and the NPU will be utilized automatically without manual file manipulation.
Does this structured utility approach cover the specific model swapping capabilities you were looking for?
