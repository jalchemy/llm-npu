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

    draft_folder = model_folder / "draft_model"
    draft_folder.mkdir()
    (draft_folder / "draft_file.txt").write_text("draft_hello")

    tar_path = manager.package_model("test_folder")
    assert tar_path.exists()

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getnames()
        # Verify there's a parent folder
        assert any(m.startswith("test_folder/") for m in members) or "test_folder" in members
        assert "test_folder/draft_model/draft_file.txt" in members or any(m.endswith("draft_model/draft_file.txt") for m in members)

def test_main_test_mode_standard(tmp_path, monkeypatch):
    manager.MODELS_DIR = tmp_path / "models"
    manager.TAR_DIR = tmp_path / "tarballs"

    class DummyArgs:
        test_mode = True

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: DummyArgs())

    # Mock download to avoid actual downloads
    def mock_download(model_id, folder_name):
        target = manager.MODELS_DIR / folder_name
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_text("mock")

    monkeypatch.setattr(manager, "download_model", mock_download)

    orig_create_modelfile = manager.create_modelfile
    def mock_create_modelfile(name, tar_path, model_type, is_test=False, root_dir=Path(".")):
        return orig_create_modelfile(name, tar_path, model_type, is_test, root_dir=tmp_path)

    monkeypatch.setattr(manager, "create_modelfile", mock_create_modelfile)

    # Run main logic
    manager.main()

    # Verify tarball and modelfile exists
    assert manager.TAR_DIR.exists()
    assert len(list(manager.TAR_DIR.glob("*.tar.gz"))) == 1

    mf_files = list(tmp_path.glob("Modelfile-*"))
    assert len(mf_files) == 1

def test_main_interactive_speculative(tmp_path, monkeypatch):
    manager.MODELS_DIR = tmp_path / "models"
    manager.TAR_DIR = tmp_path / "tarballs"

    class DummyArgs:
        test_mode = False

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", lambda self: DummyArgs())

    # Mock user input
    class MockQuestionarySelect:
        def __init__(self, message, choices):
            self.message = message
            self.choices = choices

        def ask(self):
            if "NPU-optimized model" in self.message:
                return self.choices[0]
            elif "Acceleration Mode" in self.message:
                return "Speculative"
            elif "Draft Model" in self.message:
                return self.choices[0]
            return None

    monkeypatch.setattr("questionary.select", MockQuestionarySelect)

    # Mock download
    downloaded_paths = []
    def mock_download(model_id, target_folder):
        target = manager.MODELS_DIR / target_folder
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_text("mock")
        downloaded_paths.append(str(target_folder))

    monkeypatch.setattr(manager, "download_model", mock_download)

    # Mock subprocess.run for ollama create
    def mock_run(args, check):
        pass
    monkeypatch.setattr("subprocess.run", mock_run)

    orig_create_modelfile = manager.create_modelfile
    def mock_create_modelfile(name, tar_path, model_type, is_test=False, root_dir=Path(".")):
        return orig_create_modelfile(name, tar_path, model_type, is_test, root_dir=tmp_path)

    monkeypatch.setattr(manager, "create_modelfile", mock_create_modelfile)

    manager.main()

    # Validate both models downloaded
    assert len(downloaded_paths) == 2
    assert "draft_model" in downloaded_paths[1] or "eagle_model" in downloaded_paths[1]

    # Verify tarball and modelfile
    assert len(list(manager.TAR_DIR.glob("*.tar.gz"))) == 1

    with tarfile.open(list(manager.TAR_DIR.glob("*.tar.gz"))[0], "r:gz") as tar:
        members = tar.getnames()
        assert any(m.endswith("draft_model/model.bin") for m in members)
