# GPU LLM Benchmarking

Comprehensive benchmarking tool for testing Large Language Model inference performance across
different context window sizes, designed to evaluate GPU performance for informed upgrade decisions.

## Features

- **Dual Scenario Testing**: Tests both short prompts (allocation overhead) and half-context prompts
  (realistic usage)
- **Comprehensive Metrics**: GPU utilization, memory bandwidth, VRAM usage, temperature, power
  consumption, and generation speed
- **Configurable Context Windows**: Test from 8K to 128K contexts with geometric progression
- **Detailed Monitoring**: Real-time GPU metrics captured at 0.5s intervals
- **Statistical Analysis**: Mean and standard deviation calculations across multiple runs
- **Export Ready**: CSV outputs for community analysis and comparison

## Quick Start

### Using Docker (Recommended)

Pull and run the pre-configured Docker image with Ollama:

```bash
docker pull ghcr.io/tomfoster/gpu-llm-benchmark:latest
docker run --gpus all -v ./results:/app/results ghcr.io/tomfoster/gpu-llm-benchmark:latest
```

### Manual Installation

```bash
# Clone the repository
git clone https://git.tomfos.tr/tom/gpu-llm-benchmarking.git
cd gpu-llm-benchmarking

# Install dependencies
pip install requests

# Run quick validation
python scripts/llm_benchmark_python.py --quick

# Run full benchmark
python scripts/llm_benchmark_python.py
```

## Configuration

### Environment Variables

All configuration can be overridden using environment variables:

```bash
# Model configuration
export HF_MODEL="hf.co/unsloth/gemma-3-12b-it-GGUF:Q5_K_XL"
export TEST_ITERATIONS=3

# Context window configuration
export CONTEXT_START=8192
export CONTEXT_END=131072
export CONTEXT_MULTIPLIER=2

# Output directory
export OUTPUT_PATH="./benchmark_results"

# Ollama configuration
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KEEP_ALIVE="10m"
export OLLAMA_KV_CACHE_TYPE="q8_0"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_GPU=1
export OLLAMA_NUM_PARALLEL=1
```

### Docker Environment Variables

When using Docker, pass environment variables with `-e`:

```bash
docker run --gpus all \
  -e HF_MODEL="your-model:tag" \
  -e TEST_ITERATIONS=5 \
  -e CONTEXT_END=65536 \
  -v ./results:/app/results \
  ghcr.io/tomfoster/gpu-llm-benchmark:latest
```

## Benchmark Modes

### Quick Mode

Tests only the starting context length (8K by default) for rapid validation:

```bash
python scripts/llm_benchmark_python.py --quick
# Or with Docker:
docker run --gpus all -v ./results:/app/results ghcr.io/tomfoster/gpu-llm-benchmark:latest --quick
```

### Full Mode

Complete benchmark suite testing all context lengths:

```bash
python scripts/llm_benchmark_python.py
# Or with Docker:
docker run --gpus all -v ./results:/app/results ghcr.io/tomfoster/gpu-llm-benchmark:latest
```

## Output Structure

Results are saved in timestamped directories:

```plain
benchmark_results_20250127_143022/
├── system_info.txt              # GPU specs, CPU info, software versions
├── individual_results.csv       # Raw test data
├── summary_stats.csv           # Statistical analysis
├── benchmark.log               # Execution log
├── prompt_*.txt                # Test prompts
├── response_*.txt              # Generated responses
├── ollama_response_*.json      # Full API responses
└── gpu_monitor_*.csv           # Real-time GPU metrics
```

## Metrics Collected

- **Performance**: Tokens/second, evaluation time, total duration
- **GPU Utilization**: Core usage percentage, memory bandwidth utilization
- **Memory**: VRAM used/free in GB
- **Thermal**: Temperature, power consumption, fan speed
- **Clocks**: Graphics and memory clock speeds
- **States**: GPU performance state (P0-P8)

## Repository Structure

- **[`scripts/`](scripts/)**: Benchmarking scripts and utilities
  - `llm_benchmark_python.py`: Main benchmarking script
  - Additional analysis tools (coming soon)

- **[`results/`](results/)**: Community benchmark results
  - Organised by GPU model and configuration
  - Includes raw data and analysis summaries
  - [View all results](results/)

## Default Test Configuration

- **Model**: Gemma 3 12B Instruct (Q5_K_XL quantization)
- **Context Lengths**: 8K → 16K → 32K → 64K → 128K
- **Iterations**: 3 runs per context length
- **Scenarios**: Short prompts + Half-context prompts

## Requirements

### Hardware

- NVIDIA GPU with CUDA support
- Sufficient VRAM for target model and context lengths
- nvidia-smi installed and accessible

### Software

- Python 3.8+
- Docker (for containerized usage)
- Ollama (automatically installed if not present)
- NVIDIA drivers with nvidia-smi

## Community Contribution

Share your benchmark results to help the community:

1. Run the full benchmark suite
2. Compress the results directory
3. Share on the community forums with your GPU model and system specs
4. Compare results at [gpu-llm-benchmarks.com](https://gpu-llm-benchmarks.com)

## CI/CD

This repository includes GitHub Actions workflows that:

- Build and publish Docker images on each release
- Run validation tests on pull requests
- Ensure benchmark consistency across platforms

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details
