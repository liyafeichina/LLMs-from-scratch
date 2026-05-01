# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# Utility script to search for Hermes models available via Ollama

import json
import urllib.request
import urllib.error


def list_local_models(url="http://localhost:11434/api/tags"):
    """Return a list of model names installed locally via Ollama."""
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [model["name"] for model in data.get("models", [])]
    except urllib.error.URLError:
        return None


def search_hermes(url="http://localhost:11434/api/tags"):
    """
    Search for Hermes models available locally through Ollama.

    Returns a list of locally installed model names that contain 'hermes'.
    If Ollama is not reachable, returns None with a descriptive message.
    """
    models = list_local_models(url)

    if models is None:
        print(
            "Could not connect to Ollama at the specified URL.\n"
            "Make sure Ollama is running:\n"
            "  - Start the Ollama application, or\n"
            "  - Run `ollama serve` in a terminal"
        )
        return None

    hermes_models = [m for m in models if "hermes" in m.lower()]
    return hermes_models


def main():
    print("Searching for Hermes models in local Ollama installation...\n")

    all_models = list_local_models()

    if all_models is None:
        print(
            "Could not connect to Ollama at the specified URL.\n"
            "Make sure Ollama is running:\n"
            "  - Start the Ollama application, or\n"
            "  - Run `ollama serve` in a terminal"
        )
        return

    hermes_models = [m for m in all_models if "hermes" in m.lower()]

    if hermes_models:
        print(f"Found {len(hermes_models)} Hermes model(s):")
        for model in hermes_models:
            print(f"  - {model}")
        print(
            "\nTo use a Hermes model for evaluation, pass the model name to "
            "`ollama_evaluate.py`, for example:\n"
            f"  python ollama_evaluate.py --file_path instruction-data-with-response.json "
            f"--model {hermes_models[0]}"
        )
    else:
        print("No Hermes models found locally.")
        if all_models:
            print("\nCurrently installed models:")
            for model in all_models:
                print(f"  - {model}")
        print(
            "\nTo install a Hermes model, run one of the following commands:\n"
            "  ollama pull openhermes\n"
            "  ollama pull nous-hermes2\n"
            "  ollama pull hermes3\n"
            "\nOr visit https://ollama.com/search?q=hermes for all available Hermes models."
        )


if __name__ == "__main__":
    main()
