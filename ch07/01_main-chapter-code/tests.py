# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)


import importlib.util
import json
import os
import subprocess
from unittest.mock import patch


def _load_search_hermes():
    """Dynamically load search_hermes module from the same directory."""
    module_path = os.path.join(os.path.dirname(__file__), "search_hermes.py")
    spec = importlib.util.spec_from_file_location("search_hermes", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpt_class_finetune():
    command = ["python", "ch07/01_main-chapter-code/gpt_instruction_finetuning.py", "--test_mode"]

    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script exited with errors: {result.stderr}"


def test_search_hermes_no_ollama():
    """search_hermes returns None when Ollama is not reachable."""
    sh = _load_search_hermes()
    # Point to a port where nothing is listening so the connection fails fast
    result = sh.search_hermes(url="http://localhost:19999/api/tags")
    assert result is None


def test_search_hermes_filters_hermes_models():
    """search_hermes returns only models whose name contains 'hermes'."""
    import urllib.request

    fake_response_body = json.dumps({
        "models": [
            {"name": "llama3"},
            {"name": "openhermes:latest"},
            {"name": "nous-hermes2:7b"},
            {"name": "mistral"},
        ]
    }).encode("utf-8")

    class _FakeResponse:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with patch.object(urllib.request, "urlopen", return_value=_FakeResponse(fake_response_body)):
        sh = _load_search_hermes()
        result = sh.search_hermes()

    assert result == ["openhermes:latest", "nous-hermes2:7b"]
