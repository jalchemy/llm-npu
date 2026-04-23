# llm-npu 🚀

Welcome to **llm-npu**! This project provides a fully portable, containerized environment to run Large Language Models (LLMs) and Vision-Language Models (VLMs) directly on your Neural Processing Unit (NPU) using OpenVINO-accelerated Ollama.

Designed with ease of use in mind, it automates model downloading, tarball packaging fixes, and environment setup so you can get NPU acceleration working right out of the box!

## 📦 Features

- **OpenVINO Acceleration:** Run models efficiently using the Intel NPU.
- **Automated Setup:** Single CLI tool (`manager.py`) handles ModelScope downloads, packaging, and Ollama registry.
- **Docker Ready:** Self-contained Docker environment with NPU mapping out-of-the-box.
- **Curated Models:** Includes a `models.json` registry of OpenVINO-optimized and NPU-ready models (e.g., DeepSeek, Qwen, Llama).

## 🛠️ Setup & Requirements

To get started, you will need:
- **[uv](https://github.com/astral-sh/uv)** for fast Python dependency management.
- **[Docker](https://docs.docker.com/engine/install/)** & **[Docker Compose](https://docs.docker.com/compose/install/)** to run the service.
- A compatible NPU device mapped to `/dev/accel` on your host machine.

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/llm-npu.git
   cd llm-npu
   ```

2. **Install dependencies:**
   We use `uv` to manage the project dependencies. Install them easily:
   ```bash
   uv sync
   ```

3. **Build the container:**
   Build the Docker container, which compiles Ollama with OpenVINO GenAI support.
   ```bash
   docker compose build
   ```

## 🎮 Usage

### 1. Download & Register Models

Use the included management utility to browse, download, and register an NPU-optimized model:

```bash
uv run manager.py
```

This interactive script will:
- Let you choose a model from the optimized registry.
- Download it via ModelScope.
- Package it correctly to fix tarball folder structures.
- Generate an Ollama `Modelfile`.
- Register it locally with Ollama.

### 2. Run the Service

Once your models are set up, start the Ollama server with NPU hardware acceleration:

```bash
docker compose up
```

The service will run locally, exposing port `11434`. You can now run queries against the Ollama API, fully accelerated by your NPU!

*(Note: The Docker Compose file automatically mounts `./models` and maps your host's `/dev/accel` device so the OpenVINO runtime can utilize the NPU.)*

## 🧑‍💻 Developer Notes

- **Dependency Management:** This project uses `uv` to manage dependencies (like `modelscope` and `questionary`). Check `pyproject.toml` for the dependency configurations.
- **Management Utility (`manager.py`):** The manager script simplifies handling ModelScope downloads and automates fixing a known OpenVINO archive bug ("tarbomb" fix) by explicitly nesting model artifacts inside a top-level directory in the generated tarball.
- **Tests:** The repository contains a suite of tests located in `test_manager.py`. You can run tests using:
  ```bash
  uv run pytest
  ```
- **NPU vs CPU inference in testing:** When running `manager.py`, the `Modelfile` defaults to `InferDevice "NPU"`. However, during tests (`--test-mode`), it will fallback to `InferDevice "CPU"` so tests can run on devices without NPU hardware or when mocking the setup.

## 📄 License

See the `LICENSE` file for more details.
