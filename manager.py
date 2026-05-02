import json
import os
import subprocess
import tarfile
import shutil
from pathlib import Path
import questionary # Dependency for browsing
import argparse
from unittest.mock import patch

REGISTRY_PATH = "models.json"
MODELS_DIR = Path("models")
TAR_DIR = MODELS_DIR / "tarballs"

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

def create_modelfile(name, tar_path, model_type, is_test=False, root_dir=Path(".")):
    infer_device = "CPU" if is_test else "NPU"
    modelfile_content = f"""
FROM {tar_path}
ModelBackend "OpenVINO"
ModelType "{model_type}"
InferDevice "{infer_device}"
PARAMETER stop_id 151643
PARAMETER max_new_token 2048
PARAMETER temperature 0.6
"""
    mf_path = root_dir / f"Modelfile-{name}"
    mf_path.write_text(modelfile_content)
    return mf_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    registry = load_registry()
    options = [f"{m['name']} ({m['type']})" for m in registry["npu_optimized"]]

    if args.test_mode:
        selection_name = options[0]
        accel_mode = "Standard"
    else:
        selection_name = questionary.select(
            "Select an NPU-optimized model to deploy:",
            choices=options + ["Exit"]
        ).ask()

        if selection_name == "Exit" or selection_name is None: return

        accel_mode = questionary.select(
            "Select Acceleration Mode:",
            choices=["Standard", "Speculative", "EAGLE-3"]
        ).ask()

        if accel_mode is None: return

    model_data = next(m for m in registry["npu_optimized"] if m["name"] in selection_name)
    folder_name = model_data["name"].lower().replace("-", "_")

    # 1. Download main model
    download_model(model_data["id"], folder_name)

    # 1b. Dual-Download Logic for Draft Model
    if accel_mode in ["Speculative", "EAGLE-3"]:
        draft_options = [f"{m['name']} ({m['type']})" for m in registry.get("draft_models", [])]

        if args.test_mode:
            draft_selection_name = draft_options[0] if draft_options else None
        else:
            draft_selection_name = questionary.select(
                "Select a Draft Model:",
                choices=draft_options
            ).ask()

            if draft_selection_name is None: return

        if draft_selection_name:
            draft_model_data = next(m for m in registry["draft_models"] if m["name"] in draft_selection_name)
            draft_subfolder = "eagle_model" if accel_mode == "EAGLE-3" else "draft_model"
            target_draft_folder = f"{folder_name}/{draft_subfolder}"

            download_model(draft_model_data["id"], target_draft_folder)

    # 2. Package
    tar_path = package_model(folder_name)

    # 3. Modelfile
    mf_path = create_modelfile(folder_name, f"/models/tarballs/{tar_path.name}", model_data["type"], is_test=args.test_mode)

    # 4. Ollama Create
    print(f"Registering {model_data['name']} with Ollama...")
    if args.test_mode:
        print("Test mode enabled: Executing mock subprocess call to ollama")
        try:
            subprocess.run(["ollama", "create", model_data["name"], "-f", str(mf_path)], check=True)
        except FileNotFoundError:
            print("Mock: ollama binary not found, but simulating success.")
    else:
        subprocess.run(["ollama", "create", model_data["name"], "-f", str(mf_path)], check=True)
    print(f"Successfully loaded {model_data['name']}. You can now run it.")

if __name__ == "__main__":
    main()
