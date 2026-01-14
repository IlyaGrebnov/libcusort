# libcusort

A CUDA radix sort library. Made to be faster than CUB. Single-header, drop-in replacement API, tuned for Blackwell.

## Why?

CUB is the de facto GPU sorting library and is well-tuned, but its codebase prioritizes generality and backward compatibility, making it difficult to follow if you want to understand how GPU radix sort actually works.

libcusort has two goals: **performance** (fastest radix sort on modern NVIDIA GPUs) and **education** (readable, well-commented code that explains the *why* behind each optimization).

## Benchmarks

CUB's radix sort is already memory-bound — the theoretical best case for a sorting algorithm. Squeezing extra performance from an already-optimal implementation is hard; even single-digit percentage gains require careful work.

**RTX 5090, CUDA 13.1, 67M elements:**

| Type | libcusort | CUB | Δ |
|------|-----------|-----|---|
| int32_t keys | 1.53ms | 1.62ms | **+6%** |
| int64_t keys | 6.12ms | 6.19ms | **+1%** |
| float keys | 1.53ms | 1.62ms | **+6%** |
| int32_t + int32_t pairs | 3.07ms | 3.10ms | **+1%** |

<details>
<summary>Full benchmark results (67M elements, 36 configurations)</summary>

### Keys-Only

| Key Type | libcusort | CUB | Speedup |
|----------|-----------|-----|---------|
| int8_t | 0.192ms | 0.254ms | +32.6% |
| int16_t | 0.413ms | 0.499ms | +20.8% |
| int32_t | 1.529ms | 1.615ms | +5.6% |
| int64_t | 6.124ms | 6.185ms | +1.0% |
| float | 1.529ms | 1.621ms | +6.0% |
| double | 6.156ms | 6.369ms | +3.5% |

### Key-Value Pairs

| Key | Value | libcusort | CUB | Speedup |
|-----|-------|-----------|-----|---------|
| int8_t | int8_t | 0.270ms | 0.372ms | +37.5% |
| int8_t | int16_t | 0.332ms | 0.460ms | +38.6% |
| int8_t | int32_t | 0.508ms | 0.521ms | +2.6% |
| int8_t | int64_t | 0.905ms | 0.912ms | +0.8% |
| int8_t | ulonglong2 | 1.646ms | 1.651ms | +0.3% |
| int16_t | int8_t | 0.627ms | 0.697ms | +11.3% |
| int16_t | int16_t | 0.823ms | 0.955ms | +16.0% |
| int16_t | int32_t | 1.182ms | 1.207ms | +2.2% |
| int16_t | int64_t | 1.962ms | 1.973ms | +0.6% |
| int16_t | ulonglong2 | 3.452ms | 3.459ms | +0.2% |
| int32_t | int8_t | 1.966ms | 2.028ms | +3.2% |
| int32_t | int16_t | 2.341ms | 2.379ms | +1.6% |
| int32_t | int32_t | 3.068ms | 3.100ms | +1.0% |
| int32_t | int64_t | 4.580ms | 4.671ms | +2.0% |
| int32_t | ulonglong2 | 7.641ms | 7.663ms | +0.3% |
| int64_t | int8_t | 6.866ms | 6.931ms | +1.0% |
| int64_t | int16_t | 7.588ms | 7.643ms | +0.7% |
| int64_t | int32_t | 9.054ms | 9.083ms | +0.3% |
| int64_t | int64_t | 11.979ms | 12.003ms | +0.2% |
| int64_t | ulonglong2 | 18.182ms | 18.264ms | +0.5% |
| float | int8_t | 1.965ms | 2.043ms | +4.0% |
| float | int16_t | 2.339ms | 2.377ms | +1.6% |
| float | int32_t | 3.066ms | 3.102ms | +1.2% |
| float | int64_t | 4.595ms | 4.673ms | +1.7% |
| float | ulonglong2 | 7.636ms | 7.663ms | +0.4% |
| double | int8_t | 6.911ms | 7.019ms | +1.6% |
| double | int16_t | 7.639ms | 7.864ms | +2.9% |
| double | int32_t | 9.107ms | 9.380ms | +3.0% |
| double | int64_t | 12.107ms | 12.380ms | +2.3% |
| double | ulonglong2 | 19.682ms | 19.846ms | +0.8% |

</details>

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
