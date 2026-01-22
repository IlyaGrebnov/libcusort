# libcusort

A CUDA radix sort library. Made to be faster than CUB. Single-header, drop-in replacement API, tuned for Blackwell.

## Why?

CUB is the de facto GPU sorting library and is well-tuned, but its codebase prioritizes generality and backward compatibility, making it difficult to follow if you want to understand how GPU radix sort actually works.

libcusort has two goals: **performance** (fastest radix sort on modern NVIDIA GPUs) and **education** (readable, well-commented code that explains the *why* behind each optimization).

## Benchmarks

CUB's radix sort is already memory-bound — the theoretical best case for a sorting algorithm. Squeezing extra performance from an already-optimal implementation is hard; even single-digit percentage gains require careful work.

**RTX 5090, CUDA 13.1, 67M elements:**

### Keys-Only

| Key Type | libcusort | CUB | Speedup |
|----------|-----------|-----|---------|
| int8_t | 0.188ms | 0.251ms | +33.6% |
| int16_t | 0.404ms | 0.495ms | +22.5% |
| int32_t | 1.537ms | 1.609ms | +4.7% |
| int64_t | 6.103ms | 6.171ms | +1.1% |
| float | 1.537ms | 1.618ms | +5.3% |
| double | 6.124ms | 6.347ms | +3.6% |

### Key-Value Pairs

| Key | Value | libcusort | CUB | Speedup |
|-----|-------|-----------|-----|---------|
| int8_t | int8_t | 0.264ms | 0.369ms | +39.5% |
| int8_t | int16_t | 0.328ms | 0.457ms | +39.4% |
| int8_t | int32_t | 0.506ms | 0.518ms | +2.4% |
| int8_t | int64_t | 0.895ms | 0.909ms | +1.6% |
| int8_t | ulonglong2 | 1.637ms | 1.652ms | +0.9% |
| int16_t | int8_t | 0.621ms | 0.692ms | +11.6% |
| int16_t | int16_t | 0.819ms | 0.948ms | +15.8% |
| int16_t | int32_t | 1.178ms | 1.202ms | +2.1% |
| int16_t | int64_t | 1.947ms | 1.963ms | +0.8% |
| int16_t | ulonglong2 | 3.426ms | 3.451ms | +0.7% |
| int32_t | int8_t | 1.959ms | 2.018ms | +3.0% |
| int32_t | int16_t | 2.334ms | 2.365ms | +1.3% |
| int32_t | int32_t | 3.053ms | 3.083ms | +1.0% |
| int32_t | int64_t | 4.557ms | 4.675ms | +2.6% |
| int32_t | ulonglong2 | 7.571ms | 7.653ms | +1.1% |
| int64_t | int8_t | 6.845ms | 6.923ms | +1.1% |
| int64_t | int16_t | 7.572ms | 7.630ms | +0.8% |
| int64_t | int32_t | 9.033ms | 9.089ms | +0.6% |
| int64_t | int64_t | 11.921ms | 12.001ms | +0.7% |
| int64_t | ulonglong2 | 18.055ms | 18.254ms | +1.1% |
| float | int8_t | 1.961ms | 2.032ms | +3.6% |
| float | int16_t | 2.334ms | 2.366ms | +1.4% |
| float | int32_t | 3.053ms | 3.083ms | +1.0% |
| float | int64_t | 4.579ms | 4.677ms | +2.1% |
| float | ulonglong2 | 7.571ms | 7.653ms | +1.1% |
| double | int8_t | 6.884ms | 7.009ms | +1.8% |
| double | int16_t | 7.618ms | 7.854ms | +3.1% |
| double | int32_t | 9.099ms | 9.390ms | +3.2% |
| double | int64_t | 12.038ms | 12.353ms | +2.6% |
| double | ulonglong2 | 18.060ms | 18.263ms | +1.1% |

## Features

- **Single-header** — just `#include "libcusort.cuh"`, no dependencies beyond CUDA runtime
- **CUB-compatible API** — drop-in replacement
- **All key types** — int8/16/32/64, uint8/16/32/64, float, double
- **Values up to 16 bytes** — including `ulonglong2`
- **Stable sort** — equal keys preserve relative order
- **Bit-range sorting** — sort on subset of key bits
- **CUDA Graphs + PDL** — automatic on sm_90+ for reduced launch overhead

## Quick Start

```cpp
#include "libcusort.cuh"

// Allocate input and output buffers
int32_t *d_keys_in, *d_keys_out;
cudaMalloc(&d_keys_in, num_items * sizeof(int32_t));
cudaMalloc(&d_keys_out, num_items * sizeof(int32_t));

// Create double buffer
cusort::DoubleBuffer<int32_t> d_keys(d_keys_in, d_keys_out);

// Query temp storage size
size_t temp_bytes = 0;
cusort::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_keys, num_items);

// Allocate temp storage and sort
void* d_temp;
cudaMalloc(&d_temp, temp_bytes);
cusort::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys, num_items);

// Result is in d_keys.Current()
```

## API

| Function | Description |
|----------|-------------|
| `SortKeys` | Sort keys ascending |
| `SortKeysDescending` | Sort keys descending |
| `SortPairs` | Sort key-value pairs ascending |
| `SortPairsDescending` | Sort key-value pairs descending |

All functions have two variants:
- **DoubleBuffer** — ping-pong buffers, check `selector` for result location
- **Raw pointers** — explicit `d_keys_in` / `d_keys_out`, result always in `_out`

```cpp
// Raw pointer variant
cusort::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys_in, d_keys_out, num_items);
```

Optional parameters: `begin_bit`, `end_bit` (partial-key sorting), `stream`

## Requirements

- CUDA 12.0+ (12.3+ for CUDA Graphs with PDL, 13.0+ for smem spilling)
- SM 7.0+ (Volta and newer)
- C++17
- Tested: MSVC 19.44, GCC 13, Clang 18

## Configuration

Optional compile-time flags:

```cpp
#define CUSORT_DISABLE_CUDA_GRAPHS_AND_PDL 1  // Disable CUDA Graphs (auto-disabled if CUDA < 12.3)
#define CUSORT_DISABLE_SMEM_SPILLING 1        // Disable shared memory spilling (auto-disabled if CUDA < 13.0)
```

## How It Works

libcusort implements the [OneSweep](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus) algorithm with decoupled lookback, achieving single-pass radix digit binning per radix pass.

The source code is extensively commented for educational purposes — explaining not just what the code does, but *why* each optimization exists and how GPU hardware constraints shape the implementation.

Key optimizations:

- **Decoupled lookback** — tiles resolve prefix sums cooperatively
- **CUDA Graph caching** — amortizes ~100μs graph creation to ~5μs replay
- **Programmatic Dependent Launch** — overlaps histogram with first sort pass on Hopper+
- **Fine-grained PTX cache control** — streaming loads/stores to minimize cache pollution
- **Auto-tuned parameters** — 48 key/value/offset combinations tuned for RTX 5090

## License

Apache 2.0

## Acknowledgments

- [OneSweep paper](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus) by Andy Adinets and Duane Merrill
- [GPUSorting](https://github.com/b0nes164/GPUSorting) by b0nes164 for implementation insights
- CUB for API design inspiration
