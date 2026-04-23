import pytest
import os
import tarfile
from pathlib import Path
import manager

def test_load_registry():
    registry = manager.load_registry()
    assert "npu_optimized" in registry
    assert len(registry["npu_optimized"]) > 0

def test_create_modelfile(tmp_path):
    manager.MODELS_DIR = tmp_path / "models"
    manager.TAR_DIR = tmp_path / "tarballs"

    mf_path = manager.create_modelfile("test_model", "/test/tar.gz", "LLM", is_test=True, root_dir=tmp_path)
    assert mf_path.exists()
    content = mf_path.read_text()
    assert 'ModelBackend "OpenVINO"' in content
    assert 'InferDevice "CPU"' in content

def test_package_model(tmp_path):
    manager.MODELS_DIR = tmp_path / "models"
    manager.TAR_DIR = tmp_path / "tarballs"
    manager.MODELS_DIR.mkdir()
    model_folder = manager.MODELS_DIR / "test_folder"
    model_folder.mkdir()
    (model_folder / "test_file.txt").write_text("hello")

    tar_path = manager.package_model("test_folder")
    assert tar_path.exists()

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getnames()
        # Verify there's a parent folder
        assert any(m.startswith("test_folder/") for m in members) or "test_folder" in members
