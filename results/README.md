# GPU LLM Benchmark Results

This directory contains comprehensive benchmark results from LLM context length tests across various
GPU configurations. Results are organised by GPU type, model configuration, and test parameters
following a standardised naming convention.

## Directory Structure and Naming Convention

Results are automatically organised into `{GPU_TYPE}_{NUM_GPUS}x_Ollama-{VERSION}_{OLLAMA_MODEL}`
(where all non-dot/hyphen special characters inside fields are replaced with a hyphen to standardise
names):

```plain
results/
├── README.md                   # This file - overview and comparisons
├── RTX-4090_1x_Ollama-0.9.6_hf.co-unsloth-gemma-3-12b-it-GGUF-Q5_K_XL/
│   ├── README.md               # Summary for this specific configuration
│   ├── individual_results.csv  # Raw test data
│   ├── summary_stats.csv       # Statistical analysis
│   ├── system_info.txt         # Hardware and software details
│   └── benchmark.log           # Complete execution log
├── RTX-5090_1x_Ollama-0.9.6_hf.co-unsloth-gemma-3-12b-it-GGUF-Q5-K-XL/
├── RTX-5090_2x_Ollama-0.9.6_hf.co-unsloth-gemma-3-12b-it-GGUF-Q5-K-XL/
```

## Result Documentation

### Individual Result Directories

Each result directory contains a complete benchmark dataset and will include:

- **README.md**: Detailed analysis of that specific test run including:
  - Performance highlights and key findings
  - Interesting outliers or anomalies observed
  - Context length scaling characteristics
  - GPU utilisation patterns and efficiency metrics
  - Comparison with expected theoretical performance

- **Raw Data Files**:
  - `individual_results.csv`: Every test iteration with full metrics
  - `summary_stats.csv`: Statistical summaries grouped by scenario and context length
  - `system_info.txt`: Complete hardware/software configuration for reproducibility
  - `benchmark.log`: Detailed execution logs for debugging

### This README (Comparative Analysis)

This README serves as the central hub for cross-configuration comparisons and analysis, including:

- Performance trends across different GPU generations
- Scaling efficiency with multi-GPU configurations
- Model-specific optimisation insights
- Power efficiency and cost-effectiveness analysis
- Recommendations for different use cases

## Testing Methodology and Data Collection

Our benchmark suite tests LLM inference performance across different context window sizes using two
scenarios: short prompts (allocation overhead testing) and half-context prompts (realistic usage
testing). Each test captures comprehensive GPU metrics including utilisation, memory bandwidth, VRAM
usage, temperature, power consumption, and clock speeds at 0.5-second intervals.

Results include individual test iterations with full metrics, statistical summaries grouped by
scenario and context length, complete hardware/software configurations for reproducibility, and
detailed execution logs. This approach provides insight into GPU performance characteristics,
context length scaling behaviour, and resource utilisation patterns across different hardware
configurations.

## Contributing Results

This repository primarily documents my own benchmark test results across various GPU configurations.

If you'd like to see results for a specific GPU configuration, please open an
[issue](https://git.tomfos.tr/tom/gpu-llm-benchmarking/issues) requesting it and I'll consider
adding it to my testing queue.

Alternatively, if you fork this repository and would like to contribute results back via pull
request, I may include them if they add significant value to the dataset.
