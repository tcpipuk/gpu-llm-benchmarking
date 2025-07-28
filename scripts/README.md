# Scripts

This directory contains all benchmarking and utility scripts for the GPU LLM Benchmarking project.

1. [Available Scripts](#available-scripts)
2. [Running Locally](#running-locally)
3. [Running on Vast.ai](#running-on-vastai)

## Available Scripts

| Script | About |
|---|---|
| llm_benchmark.py | The main benchmarking script that tests LLM inference performance across different context window
sizes using dual scenario testing (short prompts vs half-context), comprehensive GPU monitoring, and
statistical analysis across multiple runs. |
| run_vast.ai_benchmark.py | Remote benchmarking runner for executing LLM benchmarks on Vast.ai GPU instances. Handles the
complete lifecycle including instance provisioning, Ollama installation and configuration, benchmark
execution, results retrieval, and resource cleanup. |

## Running Locally

For testing on your own hardware:

```bash
# Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or update your existing copy
uv self update

# Path is important to load the .env from the directory above
uv run scripts/llm_benchmark.py
```

Copy the [.env.example](../.env.example) to `.env` locally, and configure it as required. This
script does not install [Ollama](https://github.com/ollama/ollama#ollama) - it expects
[Ollama](https://github.com/ollama/ollama#ollama) to already be running with your desired model
[pre-pulled](https://github.com/ollama/ollama#pull-a-model).

## Running on Vast.ai

For testing across different GPU configurations:

```bash
# Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or update your existing copy
uv self update

# Path is important to load the .env from the directory above
uv run scripts/run_vast.ai_benchmark.py
```

Copy the [.env.example](../.env.example) to `.env` locally, and configure it as required, including
an [API key from Vast.ai](https://cloud.vast.ai/manage-keys/). The script automatically provisions
the cheapest instance meeting the parameters, installs dependencies, executes benchmarks, then
retrieves the results.
