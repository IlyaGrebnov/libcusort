/*--

This file is a part of libcusort, a library for CUDA
accelerated radix sort and segmented sort algorithms.

   Copyright (c) 2025 Ilya Grebnov <ilya.grebnov@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Please see the file LICENSE for full copyright and license details.

--*/

#ifndef LIBCUSORT_CUH
#define LIBCUSORT_CUH 1

#define LIBCUSORT_VERSION_MAJOR          1
#define LIBCUSORT_VERSION_MINOR          0
#define LIBCUSORT_VERSION_PATCH          0
#define LIBCUSORT_VERSION_STRING         "1.0.0"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Debug-only error checking wrapper. In release builds, CUDA calls pass through unchanged
// to avoid function call overhead. In debug builds, each call is checked and errors logged.
#if !defined(NDEBUG) || defined(_DEBUG)
    #include <cstdio>
    inline cudaError_t cusort_cuda_check(cudaError_t err, const char* file, int line, const char* call)
    {
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA error in %s:%d: %s returned %d (%s)\n", file, line, call, (int)err, cudaGetErrorString(err));
        }

        return err;
    }
    #define CUSORT_CUDA_CALL(call) cusort_cuda_check((call), __FILE__, __LINE__, #call)
#else
    #define CUSORT_CUDA_CALL(call) (call)
#endif

// Cross-compiler forcing of inlining. Critical for GPU performance where function call
// overhead in tight loops can dominate; ensures compiler respects inline hints.
#if defined(__CUDACC__)
    #define CUSORT_FORCEINLINE __forceinline__
    #define CUSORT_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define CUSORT_FORCEINLINE __forceinline
    #define CUSORT_RESTRICT __restrict
#elif defined(__GNUC__)
    #define CUSORT_FORCEINLINE __attribute__((always_inline)) inline
    #define CUSORT_RESTRICT __restrict__
#else
    #define CUSORT_FORCEINLINE inline
    #define CUSORT_RESTRICT
#endif

// Full loop unrolling directive. GPU kernels benefit from unrolled loops because:
// 1) Eliminates branch instructions for loop control
// 2) Enables instruction-level parallelism across iterations
// 3) Allows compiler to optimize register allocation across all iterations
#if defined(__CUDACC__)
    #define CUSORT_PRAGMA_UNROLL_FULL _Pragma("unroll")
#elif defined(__GNUC__) && __GNUC__ >= 8
    #define CUSORT_PRAGMA_UNROLL_FULL _Pragma("GCC unroll 65534")
#else
    #define CUSORT_PRAGMA_UNROLL_FULL
#endif

// CUDA execution space qualifiers. These macros allow the header to compile as
// both device code (nvcc) and host-only code (for IDE intellisense, unit tests, etc.).
#if defined(__CUDACC__)
    #define CUSORT_GLOBAL __global__
    #define CUSORT_DEVICE __device__
    #define CUSORT_HOST_DEVICE __host__ __device__
    #define CUSORT_LAUNCH_BOUNDS(...) __launch_bounds__(__VA_ARGS__)
#else
    #define CUSORT_GLOBAL
    #define CUSORT_DEVICE
    #define CUSORT_HOST_DEVICE
    #define CUSORT_LAUNCH_BOUNDS(...)
#endif

// CUDA_ARCH is only defined during device compilation pass. Using this macro allows
// compile-time selection of code paths for different GPU architectures (sm_70, sm_80, etc.).
#if defined(__CUDA_ARCH__)
    #define CUSORT_DEVICE_ARCH __CUDA_ARCH__
#else
    #define CUSORT_DEVICE_ARCH 0
#endif

// CUDA toolkit version detection. CUDA_VERSION is defined by nvcc (e.g., 13010 for CUDA 13.1).
// Used to conditionally enable features requiring specific CUDA versions: shared memory
// spilling (13.0+), async copy (11.0+), and programmatic dependent launch (12.0+).
#if defined(CUDA_VERSION)
    #define CUSORT_CUDA_VERSION CUDA_VERSION
#else
    #define CUSORT_CUDA_VERSION 0
#endif

// Feature toggle: CUDA 13+ can spill registers to shared memory instead of local memory.
// Local memory is actually global memory (slow), while shared memory is on-chip (fast).
// This optimization significantly reduces register pressure penalties on high-occupancy kernels.
#ifndef CUSORT_DISABLE_SMEM_SPILLING
    #if CUSORT_CUDA_VERSION >= 13000
        #define CUSORT_DISABLE_SMEM_SPILLING 0
    #else
        #define CUSORT_DISABLE_SMEM_SPILLING 1
    #endif
#endif

// Feature toggle: Async copy (cp.async) allows memory transfers to overlap with compute.
// Disabled by default: showed promise on Ada (RTX 40xx) but became slower on Blackwell
// (RTX 50xx) after NVIDIA redesigned the memory subsystem with TMA (Tensor Memory
// Accelerator) in Hopper/Blackwell. Kept in code for educational purposes and in case
// it benefits other architectures. Requires 16-byte alignment for cp.async.cg.
#ifndef CUSORT_DISABLE_ASYNC_COPY
    #define CUSORT_DISABLE_ASYNC_COPY 1
#endif

// Feature toggle: CUDA Graphs capture kernel launch sequences and replay them with
// minimal CPU overhead (~5us vs ~15us per launch). Programmatic Dependent Launch (PDL)
// on sm_90+ enables kernel execution overlap between radix passes - the next pass can
// start while the previous finishes. Especially beneficial for overlapping Histogram
// with the first Onesweep pass, hiding the histogram's global memory writes behind sorting.
// Requires CUDA 12.3+ for cudaGraphEdgeData and PDL APIs.
#ifndef CUSORT_DISABLE_CUDA_GRAPHS_AND_PDL
    #if CUSORT_CUDA_VERSION >= 12030
        #define CUSORT_DISABLE_CUDA_GRAPHS_AND_PDL 0
    #else
        #define CUSORT_DISABLE_CUDA_GRAPHS_AND_PDL 1
    #endif
#endif

namespace cusort
{
    // Ping-pong buffer for radix sort passes. Each radix pass reads from one buffer and
    // writes to the other, avoiding in-place scatter conflicts. The selector field tracks
    // which buffer currently holds the "active" data; after each pass, selector ^= 1 swaps roles.
    // This is more efficient than allocating new memory per pass or doing expensive in-place permutation.
    template <typename T> struct DoubleBuffer
    {
        T* d_buffers[2];
        int selector;

        CUSORT_HOST_DEVICE CUSORT_FORCEINLINE DoubleBuffer()
        {
            selector     = 0;
            d_buffers[0] = nullptr;
            d_buffers[1] = nullptr;
        }

        CUSORT_HOST_DEVICE CUSORT_FORCEINLINE DoubleBuffer(T* d_current, T* d_alternate)
        {
            selector     = 0;
            d_buffers[0] = d_current;
            d_buffers[1] = d_alternate;
        }

        CUSORT_HOST_DEVICE CUSORT_FORCEINLINE T* Current() const
        {
            return d_buffers[selector];
        }

        CUSORT_HOST_DEVICE CUSORT_FORCEINLINE T* Alternate() const
        {
            return d_buffers[selector ^ 1];
        }
    };

    namespace internal
    {
        // Sentinel type for keys-only sort (no associated values). Using a distinct type rather
        // than nullptr enables compile-time elimination of value-handling code paths via if constexpr.
        struct NullType {};

#if !CUSORT_DISABLE_CUDA_GRAPHS_AND_PDL
        // Maximum radix passes needed for any supported key type. 64-bit keys with 8-bit radix
        // require ceil(64/8) = 8 passes. This constant sizes static arrays to avoid dynamic allocation.
        static constexpr unsigned int MAX_RADIX_PASSES = 8;

        // Thread-local cache for CUDA graph instances. Building a CUDA graph has ~100us overhead,
        // but replaying a cached graph takes only ~5us. By caching the graph structure and only
        // updating kernel parameters between sorts, we amortize graph creation cost across many calls.
        // Cache invalidation occurs when grid dimensions change (different array size or GPU).
        struct GraphCache
        {
            cudaGraph_t         graph                               = nullptr;
            cudaGraphExec_t     graph_exec                          = nullptr;
            cudaGraphNode_t     memset_node                         = nullptr;
            cudaGraphNode_t     histogram_node                      = nullptr;
            cudaGraphNode_t     onesweep_nodes[MAX_RADIX_PASSES]    = {};
            unsigned int        cached_num_passes                   = 0;
            unsigned int        cached_histogram_grid_size          = 0;
            unsigned int        cached_onesweep_grid_size           = 0;

            GraphCache() = default;
            ~GraphCache() noexcept { destroy(); }

            GraphCache(const GraphCache&) = delete;
            GraphCache& operator=(const GraphCache&) = delete;

            void destroy()
            {
                if (graph_exec) { (void)CUSORT_CUDA_CALL(cudaGraphExecDestroy(graph_exec)); graph_exec = nullptr; }
                if (graph)      { (void)CUSORT_CUDA_CALL(cudaGraphDestroy(graph)); graph = nullptr; }
                cached_num_passes = 0;
            }
        };
#endif

        namespace utils
        {

#if !CUSORT_DISABLE_CUDA_GRAPHS_AND_PDL
            // Empty kernel used solely to query PTX version at runtime. Code compiled for older PTX
            // can still run on newer compute capabilities (forward compatibility), so checking CC alone
            // is insufficient - we must verify the intended PTX version at compilation time. We check
            // both: PTX version gates instruction availability, CC gates hardware features like PDL.
            namespace kernels
            {
                template <typename T> CUSORT_GLOBAL void EmptyKernel(void) { }
            }

            // Query the PTX version the kernel was compiled with. PTX is NVIDIA's virtual ISA;
            // newer PTX versions expose newer hardware features. Cached because cudaFuncGetAttributes
            // is expensive (~10us) and the value never changes during program execution.
            static cudaError_t PtxVersion(int &ptx_version)
            {
                static int cached_ptx_version = 0;

                if (cached_ptx_version == 0)
                {
                    cudaFuncAttributes attr;
                    if (cudaError_t error = CUSORT_CUDA_CALL(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernels::EmptyKernel<void>)))) return error;

                    cached_ptx_version = attr.ptxVersion;
                }

                ptx_version = cached_ptx_version * 10;
                return cudaSuccess;
            }

            // Query the SM (Streaming Multiprocessor) compute capability. Different SM versions
            // have different instruction sets and performance characteristics (e.g., sm_80 has
            // async copy, sm_90 has PDL). Cached to avoid repeated device attribute queries.
            static cudaError_t SmVersion(int &sm_version)
            {
                static int cached_sm_version = 0;

                if (cached_sm_version == 0)
                {
                    int device, major, minor;
                    if (cudaError_t error = CUSORT_CUDA_CALL(cudaGetDevice(&device))) return error;
                    if (cudaError_t error = CUSORT_CUDA_CALL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device))) return error;
                    if (cudaError_t error = CUSORT_CUDA_CALL(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device))) return error;
                    cached_sm_version = major * 10 + minor;
                }

                sm_version = cached_sm_version * 10;
                return cudaSuccess;
            }
#endif

            // Query the number of SMs on the GPU. Used to calculate grid sizes that fully occupy
            // the device. Cached because device topology doesn't change at runtime.
            static cudaError_t SmCount(int &sm_count)
            {
                static int cached_sm_count = 0;

                if (cached_sm_count == 0)
                {
                    int device;
                    if (cudaError_t error = CUSORT_CUDA_CALL(cudaGetDevice(&device))) return error;
                    if (cudaError_t error = CUSORT_CUDA_CALL(cudaDeviceGetAttribute(&cached_sm_count, cudaDevAttrMultiProcessorCount, device))) return error;
                }

                sm_count = cached_sm_count;
                return cudaSuccess;
            }

            // Calculate maximum resident blocks for a kernel. Multiplies SM count by per-SM occupancy
            // (limited by registers, shared memory, and thread count). Used to size grids that
            // maximize GPU utilization without over-subscription that would cause excessive context switching.
            template <typename KernelPtr>
            static cudaError_t MaxResidentBlocks(int &max_blocks, KernelPtr kernel, int block_threads, size_t dynamic_smem_size = 0)
            {
                int sm_count = 0;
                if (cudaError_t error = CUSORT_CUDA_CALL(SmCount(sm_count))) return error;

                int sm_occupancy = 0;
                if (cudaError_t error = CUSORT_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sm_occupancy, kernel, block_threads, dynamic_smem_size))) return error;

                max_blocks = sm_count * sm_occupancy;
                return cudaSuccess;
            }
        } // namespace utils

        namespace traits
        {
            // Type-punning utility for radix sort. Radix sort operates on unsigned bit patterns,
            // but we need to sort signed integers and floats correctly. This trait:
            // 1) Maps each type to an unsigned type of the same size for bit manipulation
            // 2) Provides Encode/Decode to convert between the original type and its bit pattern
            // For floats, we use __float_as_uint which reinterprets bits without conversion.
            template <typename T>
            struct ScalarBits
            {
                static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "ScalarBits only supports integral and floating-point types");
                static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "ScalarBits only supports 1, 2, 4, or 8 byte scalar types");

                using Unsigned =
                    std::conditional_t<sizeof(T) == 1, uint8_t,
                    std::conditional_t<sizeof(T) == 2, uint16_t,
                    std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

                CUSORT_DEVICE CUSORT_FORCEINLINE static Unsigned Encode(T value)
                {
                    if constexpr (std::is_same_v<T, float>)
                    {
                        return static_cast<Unsigned>(__float_as_uint(value));
                    }
                    else if constexpr (std::is_same_v<T, double>)
                    {
                        return static_cast<Unsigned>(__double_as_longlong(value));
                    }
                    else
                    {
                        return static_cast<Unsigned>(value);
                    }
                }

                CUSORT_DEVICE CUSORT_FORCEINLINE static T Decode(Unsigned bits)
                {
                    if constexpr (std::is_same_v<T, float>)
                    {
                        return __uint_as_float(static_cast<unsigned int>(bits));
                    }
                    else if constexpr (std::is_same_v<T, double>)
                    {
                        return __longlong_as_double(static_cast<long long>(bits));
                    }
                    else
                    {
                        return static_cast<T>(bits);
                    }
                }
            };

            // Core transformation logic for radix sort correctness. The "twiddle" operation transforms
            // keys so that unsigned comparison of bit patterns yields correct signed/float ordering:
            //
            // SIGNED INTEGERS: Flip the sign bit. Two's complement has negative numbers with MSB=1,
            //   but we want them to sort before positives. XOR with 0x80...00 maps:
            //   -128 (0x80) -> 0x00, -1 (0xFF) -> 0x7F, 0 (0x00) -> 0x80, 127 (0x7F) -> 0xFF
            //
            // FLOATING POINT: IEEE 754 uses sign-magnitude, not two's complement. For positives,
            //   flip sign bit (like signed int). For negatives, flip ALL bits because larger
            //   magnitude negative floats have larger bit patterns but should sort earlier.
            //   -Inf -> 0x00...00, -0.0 -> 0x7F...FF, +0.0 -> 0x80...00, +Inf -> 0xFF...FF
            //
            // NOTE ON SIGNED ZEROS: IEEE 754 treats +0.0 and -0.0 as equal, but our bit transformation
            //   orders -0.0 < +0.0. This differs from std::sort with default operator<, but matches
            //   C++20 std::strong_order semantics. We preserve this for data roundtrippability:
            //   distinct bit patterns remain distinguishable after sorting.
            //
            // DESCENDING: Invert all bits after the above transformation to reverse sort order.
            //
            // TwiddleForward applies before sorting; TwiddleReverse undoes it when writing output.
            // TwiddleSentinel returns a value that maps to the last bin (0xFF) after transformation,
            // used to fill out-of-bounds lanes so they participate uniformly without affecting offsets.
            template <bool IS_DESCENDING, typename KeyT>
            struct RadixTraits
            {
                // Unsigned type matching key size - used for bit manipulation without changing value semantics
                using UnsignedKeyT = typename ScalarBits<KeyT>::Unsigned;

                // Register-width type for GPU operations. GPU registers are 32-bit minimum, so 8/16-bit keys
                // are promoted to 32-bit for efficient processing. 64-bit keys use native 64-bit registers.
                using UnsignedRegT = std::conditional_t<sizeof(KeyT) <= 4, uint32_t, uint64_t>;

            private:
                static constexpr bool IS_FLOATING_POINT         = std::is_same_v<KeyT, float> || std::is_same_v<KeyT, double>;
                static constexpr bool IS_SIGNED_INT             = std::is_signed_v<KeyT> && !IS_FLOATING_POINT;

                static constexpr int KEY_BITS                   = sizeof(UnsignedKeyT) * 8;
                static constexpr int REG_BITS                   = sizeof(UnsignedRegT) * 8;

                // SIGN_MASK needs the MSB set for each packed key position. For 8-bit keys in 32-bit register:
                //   ALL_ONES >> (32-8) = 0x000000FF
                //   ALL_ONES / 0xFF    = 0x01010101  (replicates 0x01 to each byte via division trick)
                //   << 7               = 0x80808080  (shift to MSB of each byte)
                // This single mask enables parallel twiddle of all packed keys with one XOR instruction.
                static constexpr UnsignedRegT ALL_ZERO          = UnsignedRegT{0u};
                static constexpr UnsignedRegT ALL_ONES          = ~ALL_ZERO;
                static constexpr UnsignedRegT SIGN_MASK         = (ALL_ONES / (ALL_ONES >> (REG_BITS - KEY_BITS))) << (KEY_BITS - 1);

                // For sub-32-bit keys (8/16-bit), multiple keys are packed into one UnsignedRegT.
                // SIGN_MASK is replicated across all packed positions, so one XOR operation
                // transforms all keys simultaneously (e.g., 4 x 8-bit keys in one 32-bit op).
                template <bool IS_FORWARD>
                CUSORT_DEVICE CUSORT_FORCEINLINE static constexpr UnsignedRegT Twiddle(UnsignedRegT v)
                {
                    if constexpr (IS_FLOATING_POINT)
                    {
                        constexpr UnsignedRegT XOR_FOR_NEGATIVE = IS_DESCENDING ? ALL_ZERO   : (IS_FORWARD ? ALL_ONES  : SIGN_MASK);
                        constexpr UnsignedRegT XOR_FOR_POSITIVE = IS_DESCENDING ? ~SIGN_MASK : (IS_FORWARD ? SIGN_MASK : ALL_ONES );
                        return v ^ ((v & SIGN_MASK) ? XOR_FOR_NEGATIVE : XOR_FOR_POSITIVE);
                    }
                    else if constexpr (IS_SIGNED_INT)
                        return v ^ (SIGN_MASK ^ (IS_DESCENDING ? ALL_ONES : ALL_ZERO));
                    else
                        return IS_DESCENDING ? ~v : v;
                }

            public:
                CUSORT_DEVICE CUSORT_FORCEINLINE static UnsignedKeyT Encode(KeyT value) { return ScalarBits<KeyT>::Encode(value); }
                CUSORT_DEVICE CUSORT_FORCEINLINE static KeyT Decode(UnsignedKeyT bits) { return ScalarBits<KeyT>::Decode(bits); }

                CUSORT_DEVICE CUSORT_FORCEINLINE static constexpr UnsignedRegT TwiddleForward(UnsignedRegT v) { return Twiddle<true>(v); }
                CUSORT_DEVICE CUSORT_FORCEINLINE static constexpr UnsignedRegT TwiddleReverse(UnsignedRegT v) { return Twiddle<false>(v); }
                CUSORT_DEVICE CUSORT_FORCEINLINE static constexpr UnsignedRegT TwiddleSentinel() { return TwiddleReverse(ALL_ONES); }
            };
        } // namespace traits

        // Direct PTX assembly for fine-grained control over GPU memory hierarchy.
        // CUDA's default load/store behavior isn't always optimal for radix sort's access patterns.
        // PTX intrinsics let us specify exact caching behavior per memory operation.
        namespace ptx
        {
            // Global memory load cache policies. GPU has L1 (per-SM, fast) and L2 (shared, larger):
            // - ca: Cache at all levels (default) - good for data reused by same thread
            // - cg: Cache global only (L2) - good for data shared across SMs, avoids L1 thrashing
            // - cs: Cache streaming - hint that data won't be reused, evict early
            // - cv: Cache volatile - bypass cache entirely, always fetch from memory
            // - nc: Non-coherent (texture cache) - read-only path with separate cache
            // Radix sort uses 'cs' for input keys (streamed once) and 'cg' for descriptors (cross-tile sharing).
            enum class ld_global_cache_op
            {
                ca,             // ld.global.ca.*                 (cache all levels, L1 + L2, default)
                cg,             // ld.global.cg.*                 (cache global level, L2 only, bypass L1)
                cs,             // ld.global.cs.*                 (cache streaming, evict first)
                cv,             // ld.global.cv.*                 (cache volatile, invalidate L2 line, fetch again)
                l1_no_alloc,    // ld.global.L1::no_allocate.*    (L1 probe only, no allocate on miss)
                nc,             // ld.global.nc.*                 (non-coherent, read-only texture cache)
                nc_l1_no_alloc  // ld.global.nc.L1::no_allocate.* (non-coherent, L1 probe only, no allocate on miss)
            };

            // Global memory store cache policies:
            // - wb: Write-back (default) - write to cache, flush later
            // - wt: Write-through - write to both cache and memory immediately
            // - cg: Cache global (L2 only) - bypass L1, useful when other SMs will read the data
            // - cs: Cache streaming - hint for write-once data, evict early
            // Radix sort uses 'wb' for scattered key/value output: L2 cache coalesces multiple writes
            // to the same 32-byte sector before flushing to DRAM, reducing memory traffic when nearby
            // keys land in the same cache line. Uses 'cg' for descriptors (cross-SM coordination).
            enum class st_global_cache_op
            {
                wb,             // st.global.wb.*                 (write-back all levels, default)
                wt,             // st.global.wt.*                 (write-through, bypass cache to system memory)
                cg,             // st.global.cg.*                 (cache global level, L2 only, bypass L1)
                cs              // st.global.cs.*                 (cache streaming, evict first)
            };

            // Type-specialized PTX load/store wrappers. Each function emits the exact PTX instruction
            // for the specified type and cache policy. Using templates with if constexpr generates
            // only the single instruction needed, with no runtime branching.
            // Vector types (uint4, ulonglong2) enable 16-byte loads/stores for maximum memory bandwidth.
            template <ld_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void ld_global_u8(uint32_t &dst, const void *ptr)
            {
                if constexpr (op == ld_global_cache_op::ca)
                {
                    asm volatile("ld.global.ca.u8 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cg)
                {
                    asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cs)
                {
                    asm volatile("ld.global.cs.u8 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cv)
                {
                    asm volatile("ld.global.cv.u8 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::l1_no_alloc)
                {
                    asm volatile("ld.global.L1::no_allocate.u8 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc)
                {
                    asm volatile("ld.global.nc.u8 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc_l1_no_alloc)
                {
                    asm volatile("ld.global.nc.L1::no_allocate.u8 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
            }

            template <st_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void st_global_u8(void *ptr, uint32_t value)
            {
                if constexpr (op == st_global_cache_op::wb)
                {
                    asm volatile("st.global.wb.u8 [%0], %1;" ::"l"(ptr), "r"(value));
                }
                else if constexpr (op == st_global_cache_op::wt)
                {
                    asm volatile("st.global.wt.u8 [%0], %1;" ::"l"(ptr), "r"(value));
                }
                else if constexpr (op == st_global_cache_op::cg)
                {
                    asm volatile("st.global.cg.u8 [%0], %1;" ::"l"(ptr), "r"(value));
                }
                else if constexpr (op == st_global_cache_op::cs)
                {
                    asm volatile("st.global.cs.u8 [%0], %1;" ::"l"(ptr), "r"(value));
                }
            }

            template <ld_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void ld_global_u16(uint32_t &dst, const void *ptr)
            {
                if constexpr (op == ld_global_cache_op::ca)
                {
                    asm volatile("ld.global.ca.u16 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cg)
                {
                    asm volatile("ld.global.cg.u16 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cs)
                {
                    asm volatile("ld.global.cs.u16 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cv)
                {
                    asm volatile("ld.global.cv.u16 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::l1_no_alloc)
                {
                    asm volatile("ld.global.L1::no_allocate.u16 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc)
                {
                    asm volatile("ld.global.nc.u16 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc_l1_no_alloc)
                {
                    asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
            }

            template <st_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void st_global_u16(void *ptr, uint32_t value)
            {
                if constexpr (op == st_global_cache_op::wb)
                {
                    asm volatile("st.global.wb.u16 [%0], %1;" ::"l"(ptr), "r"(value));
                }
                else if constexpr (op == st_global_cache_op::wt)
                {
                    asm volatile("st.global.wt.u16 [%0], %1;" ::"l"(ptr), "r"(value));
                }
                else if constexpr (op == st_global_cache_op::cg)
                {
                    asm volatile("st.global.cg.u16 [%0], %1;" ::"l"(ptr), "r"(value));
                }
                else if constexpr (op == st_global_cache_op::cs)
                {
                    asm volatile("st.global.cs.u16 [%0], %1;" ::"l"(ptr), "r"(value));
                }
            }

            template <ld_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void ld_global_u32(uint32_t &dst, const void *ptr)
            {
                if constexpr (op == ld_global_cache_op::ca)
                {
                    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cg)
                {
                    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cs)
                {
                    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cv)
                {
                    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::l1_no_alloc)
                {
                    asm volatile("ld.global.L1::no_allocate.u32 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc)
                {
                    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc_l1_no_alloc)
                {
                    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(dst) : "l"(ptr));
                }
            }

            template <st_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void st_global_u32(void *ptr, uint32_t value)
            {
                if constexpr (op == st_global_cache_op::wb)
                {
                    asm volatile("st.global.wb.u32 [%0], %1;" ::"l"(ptr), "r"(value));
                }
                else if constexpr (op == st_global_cache_op::wt)
                {
                    asm volatile("st.global.wt.u32 [%0], %1;" ::"l"(ptr), "r"(value));
                }
                else if constexpr (op == st_global_cache_op::cg)
                {
                    asm volatile("st.global.cg.u32 [%0], %1;" ::"l"(ptr), "r"(value));
                }
                else if constexpr (op == st_global_cache_op::cs)
                {
                    asm volatile("st.global.cs.u32 [%0], %1;" ::"l"(ptr), "r"(value));
                }
            }

            template <ld_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void ld_global_u64(uint64_t &dst, const void *ptr)
            {
                if constexpr (op == ld_global_cache_op::ca)
                {
                    asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cg)
                {
                    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cs)
                {
                    asm volatile("ld.global.cs.u64 %0, [%1];" : "=l"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cv)
                {
                    asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::l1_no_alloc)
                {
                    asm volatile("ld.global.L1::no_allocate.u64 %0, [%1];" : "=l"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc)
                {
                    asm volatile("ld.global.nc.u64 %0, [%1];" : "=l"(dst) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc_l1_no_alloc)
                {
                    asm volatile("ld.global.nc.L1::no_allocate.u64 %0, [%1];" : "=l"(dst) : "l"(ptr));
                }
            }

            template <st_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void st_global_u64(void *ptr, uint64_t value)
            {
                if constexpr (op == st_global_cache_op::wb)
                {
                    asm volatile("st.global.wb.u64 [%0], %1;" ::"l"(ptr), "l"(value));
                }
                else if constexpr (op == st_global_cache_op::wt)
                {
                    asm volatile("st.global.wt.u64 [%0], %1;" ::"l"(ptr), "l"(value));
                }
                else if constexpr (op == st_global_cache_op::cg)
                {
                    asm volatile("st.global.cg.u64 [%0], %1;" ::"l"(ptr), "l"(value));
                }
                else if constexpr (op == st_global_cache_op::cs)
                {
                    asm volatile("st.global.cs.u64 [%0], %1;" ::"l"(ptr), "l"(value));
                }
            }

            template <ld_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void ld_global_v4_u32(uint4 &dst, const void *ptr)
            {
                if constexpr (op == ld_global_cache_op::ca)
                {
                    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cg)
                {
                    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cs)
                {
                    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cv)
                {
                    asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::l1_no_alloc)
                {
                    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc)
                {
                    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc_l1_no_alloc)
                {
                    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w) : "l"(ptr));
                }
            }

            template <st_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void st_global_v4_u32(void *ptr, const uint4 &v)
            {
                if constexpr (op == st_global_cache_op::wb)
                {
                    asm volatile("st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
                }
                else if constexpr (op == st_global_cache_op::wt)
                {
                    asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
                }
                else if constexpr (op == st_global_cache_op::cg)
                {
                    asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
                }
                else if constexpr (op == st_global_cache_op::cs)
                {
                    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
                }
            }

            template <ld_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void ld_global_v2_u64(ulonglong2 &dst, const void *ptr)
            {
                if constexpr (op == ld_global_cache_op::ca)
                {
                    asm volatile("ld.global.ca.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cg)
                {
                    asm volatile("ld.global.cg.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cs)
                {
                    asm volatile("ld.global.cs.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::cv)
                {
                    asm volatile("ld.global.cv.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::l1_no_alloc)
                {
                    asm volatile("ld.global.L1::no_allocate.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc)
                {
                    asm volatile("ld.global.nc.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(ptr));
                }
                else if constexpr (op == ld_global_cache_op::nc_l1_no_alloc)
                {
                    asm volatile("ld.global.nc.L1::no_allocate.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(ptr));
                }
            }

            template <st_global_cache_op op>
            CUSORT_DEVICE CUSORT_FORCEINLINE void st_global_v2_u64(void *ptr, const ulonglong2 &v)
            {
                if constexpr (op == st_global_cache_op::wb)
                {
                    asm volatile("st.global.wb.v2.u64 [%0], {%1,%2};" ::"l"(ptr), "l"(v.x), "l"(v.y));
                }
                else if constexpr (op == st_global_cache_op::wt)
                {
                    asm volatile("st.global.wt.v2.u64 [%0], {%1,%2};" ::"l"(ptr), "l"(v.x), "l"(v.y));
                }
                else if constexpr (op == st_global_cache_op::cg)
                {
                    asm volatile("st.global.cg.v2.u64 [%0], {%1,%2};" ::"l"(ptr), "l"(v.x), "l"(v.y));
                }
                else if constexpr (op == st_global_cache_op::cs)
                {
                    asm volatile("st.global.cs.v2.u64 [%0], {%1,%2};" ::"l"(ptr), "l"(v.x), "l"(v.y));
                }
            }

            template <ld_global_cache_op op, typename T>
            CUSORT_DEVICE CUSORT_FORCEINLINE T ld_global(const T *ptr)
            {
                if constexpr (std::is_same_v<T, uint8_t>)
                {
                    uint32_t tmp; ld_global_u8<op>(tmp, ptr); return static_cast<uint8_t>(tmp);
                }
                else if constexpr (std::is_same_v<T, uint16_t>)
                {
                    uint32_t tmp; ld_global_u16<op>(tmp, ptr); return static_cast<uint16_t>(tmp);
                }
                else if constexpr (std::is_same_v<T, uint32_t>)
                {
                    uint32_t tmp; ld_global_u32<op>(tmp, ptr); return tmp;
                }
                else if constexpr (std::is_same_v<T, uint64_t>)
                {
                    uint64_t tmp; ld_global_u64<op>(tmp, ptr); return tmp;
                }
                else if constexpr (std::is_same_v<T, uint4>)
                {
                    uint4 v; ld_global_v4_u32<op>(v, ptr); return v;
                }
                else if constexpr (std::is_same_v<T, ulonglong2>)
                {
                    ulonglong2 v; ld_global_v2_u64<op>(v, ptr); return v;
                }
                else if constexpr ((std::is_integral_v<T> || std::is_floating_point_v<T>) && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8))
                {
                    using Bits = traits::ScalarBits<T>;
                    using Unsigned = typename Bits::Unsigned;

                    return Bits::Decode(ld_global<op>(reinterpret_cast<const Unsigned *>(ptr)));
                }
                else
                {
                    static_assert(sizeof(T) == 0, "ld_global: Unsupported type T");
                }
            }

            template <st_global_cache_op op, typename T>
            CUSORT_DEVICE CUSORT_FORCEINLINE void st_global(T *ptr, T value)
            {
                if constexpr (std::is_same_v<T, uint8_t>)
                {
                    st_global_u8<op>(ptr, static_cast<uint32_t>(value));
                }
                else if constexpr (std::is_same_v<T, uint16_t>)
                {
                    st_global_u16<op>(ptr, static_cast<uint32_t>(value));
                }
                else if constexpr (std::is_same_v<T, uint32_t>)
                {
                    st_global_u32<op>(ptr, value);
                }
                else if constexpr (std::is_same_v<T, uint64_t>)
                {
                    st_global_u64<op>(ptr, value);
                }
                else if constexpr (std::is_same_v<T, uint4>)
                {
                    st_global_v4_u32<op>(ptr, value);
                }
                else if constexpr (std::is_same_v<T, ulonglong2>)
                {
                    st_global_v2_u64<op>(ptr, value);
                }
                else if constexpr ((std::is_integral_v<T> || std::is_floating_point_v<T>) && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8))
                {
                    using Bits = traits::ScalarBits<T>;
                    using Unsigned = typename Bits::Unsigned;

                    st_global<op>(reinterpret_cast<Unsigned *>(ptr), Bits::Encode(value));
                }
                else
                {
                    static_assert(sizeof(T) == 0, "st_global: Unsupported type T");
                }
            }

            // Warp lane identification. A warp is 32 threads that execute in lockstep (SIMT).
            // Lane ID (0-31) identifies a thread's position within its warp. Using PTX %laneid
            // is faster than threadIdx.x % 32 because it's a hardware register, not a division.
            CUSORT_DEVICE CUSORT_FORCEINLINE unsigned int lane_id()
            {
                unsigned int r;
                asm volatile("mov.u32 %0, %%laneid;" : "=r"(r));
                return r;
            }

            // Warp identification within a thread block. Used to partition shared memory
            // (each warp gets its own histogram slice to avoid conflicts) and coordinate
            // warp-level operations like prefix sums.
            CUSORT_DEVICE CUSORT_FORCEINLINE unsigned int warp_id()
            {
                // NOTE: Unlike %laneid, NVIDIA doesn't provide a %warpid register for the warp's index
                // within a block. The existing %warpid gives the warp's slot on the SM (changes with
                // scheduling), not its logical position in the block. So we compute it manually.                
                return threadIdx.x / 32u;
            }

            // Bitmask of lanes with lower ID than current lane. Hardware register that enables
            // efficient "count predecessors" operations: __popc(lanemask_lt & peers) gives
            // this lane's rank among matching peers for scatter positioning.
            CUSORT_DEVICE CUSORT_FORCEINLINE unsigned int lanemask_lt()
            {
                unsigned int r;
                asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(r));
                return r;
            }

            // Warp-level inclusive prefix sum using shuffle instructions. Each iteration doubles
            // the distance of the shuffle, building a tree-reduction in log2(32)=5 steps.
            // This is the foundational primitive for computing per-digit offsets within a warp.
            // The predicated add (@p) handles the shuffle boundary - lanes that receive invalid
            // data from below the warp don't add, preserving correctness.
            //
            // Example with 8 lanes (actual warp has 32, uses 5 steps):
            //   Input:     [1] [2] [3] [4] [5] [6] [7] [8]    (value at each lane)
            //   Step 1(+1):[1] [3] [5] [7] [9] [11][13][15]   lane[i] += lane[i-1]
            //   Step 2(+2):[1] [3] [6] [10][14][18][22][26]   lane[i] += lane[i-2]
            //   Step 3(+4):[1] [3] [6] [10][15][21][28][36]   lane[i] += lane[i-4]
            //   Output:    [1] [3] [6] [10][15][21][28][36]   inclusive prefix sum
            //
            template <typename T> CUSORT_DEVICE CUSORT_FORCEINLINE T warp_inclusive_scan(T x)
            {
                static_assert((sizeof(T) == 4 || sizeof(T) == 8) && std::is_integral_v<T>, "warp_inclusive_scan only supports 32-bit or 64-bit integer types");

                if constexpr (sizeof(T) == 4)
                {
                    asm volatile(
                        "{ .reg .u32 tmp;                     \n"
                        "  .reg .pred p;                      \n"
                        "  shfl.sync.up.b32 tmp|p, %0,  1, 0, 0xffffffff;\n"
                        "  @p add.u32 %0, %0, tmp;                       \n"
                        "  shfl.sync.up.b32 tmp|p, %0,  2, 0, 0xffffffff;\n"
                        "  @p add.u32 %0, %0, tmp;                       \n"
                        "  shfl.sync.up.b32 tmp|p, %0,  4, 0, 0xffffffff;\n"
                        "  @p add.u32 %0, %0, tmp;                       \n"
                        "  shfl.sync.up.b32 tmp|p, %0,  8, 0, 0xffffffff;\n"
                        "  @p add.u32 %0, %0, tmp;                       \n"
                        "  shfl.sync.up.b32 tmp|p, %0, 16, 0, 0xffffffff;\n"
                        "  @p add.u32 %0, %0, tmp;                       \n"
                        "}                                               \n"
                        : "+r"(x));
                }
                else
                {
                    asm volatile(
                        "{ .reg .b32 lo, hi, tmp_lo, tmp_hi;  \n"
                        "  .reg .pred p;                      \n"
                        "  mov.b64 {lo, hi}, %0;              \n"
                        "  shfl.sync.up.b32 tmp_lo|p, lo, 1, 0, 0xffffffff; \n"
                        "  shfl.sync.up.b32 tmp_hi|p, hi, 1, 0, 0xffffffff; \n"
                        "  @p add.cc.u32 lo, lo, tmp_lo;                    \n"
                        "  @p addc.u32   hi, hi, tmp_hi;                    \n"
                        "  shfl.sync.up.b32 tmp_lo|p, lo, 2, 0, 0xffffffff; \n"
                        "  shfl.sync.up.b32 tmp_hi|p, hi, 2, 0, 0xffffffff; \n"
                        "  @p add.cc.u32 lo, lo, tmp_lo;                    \n"
                        "  @p addc.u32   hi, hi, tmp_hi;                    \n"
                        "  shfl.sync.up.b32 tmp_lo|p, lo, 4, 0, 0xffffffff; \n"
                        "  shfl.sync.up.b32 tmp_hi|p, hi, 4, 0, 0xffffffff; \n"
                        "  @p add.cc.u32 lo, lo, tmp_lo;                    \n"
                        "  @p addc.u32   hi, hi, tmp_hi;                    \n"
                        "  shfl.sync.up.b32 tmp_lo|p, lo, 8, 0, 0xffffffff; \n"
                        "  shfl.sync.up.b32 tmp_hi|p, hi, 8, 0, 0xffffffff; \n"
                        "  @p add.cc.u32 lo, lo, tmp_lo;                    \n"
                        "  @p addc.u32   hi, hi, tmp_hi;                    \n"
                        "  shfl.sync.up.b32 tmp_lo|p, lo, 16, 0, 0xffffffff;\n"
                        "  shfl.sync.up.b32 tmp_hi|p, hi, 16, 0, 0xffffffff;\n"
                        "  @p add.cc.u32 lo, lo, tmp_lo;                    \n"
                        "  @p addc.u32   hi, hi, tmp_hi;                    \n"
                        "  mov.b64 %0, {lo, hi};              \n"
                        "}                                    \n"
                        : "+l"(x));
                }

                return x;
            }

            // Find all warp lanes with matching values. Returns a bitmask where bit i is set if
            // lane i has the same value as the calling lane. Used in ranking phase to group
            // keys by digit - lanes with the same digit coordinate to get consecutive ranks.
            // The 8-bit specialization is unrolled for performance; larger widths use a loop.
            //
            // Why not use __match_any_sync? NVIDIA's hardware match has variable latency that
            // depends on unique value count: best=1 (all same), worst=32 (all different), roughly
            // linear in between. Our ballot-based approach has fixed latency and beats hardware
            // match when there are 4+ unique values - the common case for radix digits. Only in
            // degenerate cases (nearly all values identical) would hardware match win.
            template <int INPUT_BITS> CUSORT_DEVICE CUSORT_FORCEINLINE uint32_t match_any_sync(const uint32_t mask, const uint32_t x)
            {
                if constexpr (INPUT_BITS == 8)
                {
                    uint32_t peers_mask;

                    asm volatile(
                        "{.reg .pred p;                          \n"
                        " .reg .u32 peers;                       \n"
                        " and.b32 %0, %1, 1;                     \n"
                        " setp.ne.u32 p, %0, 0;                  \n"
                        " vote.ballot.sync.b32 %0, p, %2;        \n"
                        " @!p not.b32 %0, %0;                    \n"
                        " and.b32 %0, %0, %2;                    \n"
                        " and.b32 peers, %1, 2;                  \n"
                        " setp.ne.u32 p, peers, 0;               \n"
                        " vote.ballot.sync.b32 peers, p, %2;     \n"
                        " @!p not.b32 peers, peers;              \n"
                        " and.b32 %0, %0, peers;                 \n"
                        " and.b32 peers, %1, 4;                  \n"
                        " setp.ne.u32 p, peers, 0;               \n"
                        " vote.ballot.sync.b32 peers, p, %2;     \n"
                        " @!p not.b32 peers, peers;              \n"
                        " and.b32 %0, %0, peers;                 \n"
                        " and.b32 peers, %1, 8;                  \n"
                        " setp.ne.u32 p, peers, 0;               \n"
                        " vote.ballot.sync.b32 peers, p, %2;     \n"
                        " @!p not.b32 peers, peers;              \n"
                        " and.b32 %0, %0, peers;                 \n"
                        " and.b32 peers, %1, 16;                 \n"
                        " setp.ne.u32 p, peers, 0;               \n"
                        " vote.ballot.sync.b32 peers, p, %2;     \n"
                        " @!p not.b32 peers, peers;              \n"
                        " and.b32 %0, %0, peers;                 \n"
                        " and.b32 peers, %1, 32;                 \n"
                        " setp.ne.u32 p, peers, 0;               \n"
                        " vote.ballot.sync.b32 peers, p, %2;     \n"
                        " @!p not.b32 peers, peers;              \n"
                        " and.b32 %0, %0, peers;                 \n"
                        " and.b32 peers, %1, 64;                 \n"
                        " setp.ne.u32 p, peers, 0;               \n"
                        " vote.ballot.sync.b32 peers, p, %2;     \n"
                        " @!p not.b32 peers, peers;              \n"
                        " and.b32 %0, %0, peers;                 \n"
                        " and.b32 peers, %1, 128;                \n"
                        " setp.ne.u32 p, peers, 0;               \n"
                        " vote.ballot.sync.b32 peers, p, %2;     \n"
                        " @!p not.b32 peers, peers;              \n"
                        " and.b32 %0, %0, peers;                 \n"
                        "}                                       \n"
                        : "=r"(peers_mask) : "r"(x), "r"(mask));

                    return peers_mask;
                }
                else
                {
                    uint32_t peers_mask = mask;

                    CUSORT_PRAGMA_UNROLL_FULL
                    for (uint32_t bit = 0u; bit < INPUT_BITS; bit += 1u)
                    {
                        uint32_t peers, bit_mask = 1u << bit;

                        asm volatile(
                            "{.reg .pred p;                  \n"
                            " and.b32 %0, %1, %2;            \n"
                            " setp.ne.u32 p, %0, 0;          \n"
                            " vote.ballot.sync.b32 %0, p, %3;\n"
                            " @!p not.b32 %0, %0;            \n"
                            "}                               \n"
                            : "=r"(peers) : "r"(x), "r"(bit_mask), "r"(mask));

                        peers_mask &= peers;
                    }

                    return peers_mask;
                }
            }

            // Bitwise rotation for radix digit extraction. Rotating the current radix digit to
            // the LSB position allows extraction with a simple mask (key & 0xFF) instead of
            // a variable shift. This is faster because:
            // 1) The mask is a compile-time constant
            // 2) Funnel shift (__funnelshift_r) executes in 1 cycle on GPU
            // 3) Works uniformly for all key sizes (8/16/32/64-bit)
            // The rotation is reversed when writing output to restore original bit positions.
            //
            // 32-bit case: __funnelshift_r(v, v, s) concatenates [v:v] and extracts 32 bits
            //   starting at bit position s from the right - exactly a rotate right.
            //
            // 64-bit case: No native 64-bit rotate, so we decompose into two 32-bit halves:
            //   1) If s >= 32, swap hi/lo first (equivalent to rotating by 32), then s &= 31
            //   2) For out_lo: extract from [hi:lo] at position s (bits cross the boundary)
            //   3) For out_hi: extract from [lo:hi] at position s (wrapped continuation)
            //   The (s & 32u) check handles the swap; funnelshift uses only low 5 bits of s.
            template <typename T> CUSORT_DEVICE CUSORT_FORCEINLINE T rotate_right(T x, uint32_t s)
            {
                static_assert((sizeof(T) == 4 || sizeof(T) == 8) && std::is_integral_v<T>, "rotate_right only supports 32-bit or 64-bit integral scalar types");

                using Bits = traits::ScalarBits<T>;
                using U    = typename Bits::Unsigned;

                if constexpr (sizeof(T) == 4)
                {
                    const uint32_t v = static_cast<uint32_t>(Bits::Encode(x));
                    const uint32_t r = __funnelshift_r(v, v, s);

                    return Bits::Decode(static_cast<U>(r));
                }
                else
                {
                    const uint64_t v     = static_cast<uint64_t>(Bits::Encode(x));
                    const uint32_t in_lo = static_cast<uint32_t>(v);
                    const uint32_t in_hi = static_cast<uint32_t>(v >> 32);

                    const uint32_t lo = (s & 32u) ? in_hi : in_lo;
                    const uint32_t hi = (s & 32u) ? in_lo : in_hi;

                    const uint32_t out_lo = __funnelshift_r(lo, hi, s);
                    const uint32_t out_hi = __funnelshift_r(hi, lo, s);
                    const uint64_t out = (uint64_t(out_hi) << 32) | uint64_t(out_lo);

                    return Bits::Decode(static_cast<U>(out));
                }
            }

            template <typename T> CUSORT_DEVICE CUSORT_FORCEINLINE T rotate_left(T x, uint32_t s)
            {
                static_assert((sizeof(T) == 4 || sizeof(T) == 8) && std::is_integral_v<T>, "rotate_left only supports 32-bit or 64-bit integral scalar types");

                using Bits = traits::ScalarBits<T>;
                using U    = typename Bits::Unsigned;

                if constexpr (sizeof(T) == 4)
                {
                    const uint32_t v = static_cast<uint32_t>(Bits::Encode(x));
                    const uint32_t r = __funnelshift_l(v, v, s);

                    return Bits::Decode(static_cast<U>(r));
                }
                else
                {
                    const uint64_t v     = static_cast<uint64_t>(Bits::Encode(x));
                    const uint32_t in_lo = static_cast<uint32_t>(v);
                    const uint32_t in_hi = static_cast<uint32_t>(v >> 32);

                    const uint32_t lo = (s & 32u) ? in_hi : in_lo;
                    const uint32_t hi = (s & 32u) ? in_lo : in_hi;

                    const uint32_t out_lo = __funnelshift_l(hi, lo, s);
                    const uint32_t out_hi = __funnelshift_l(lo, hi, s);
                    const uint64_t out = (uint64_t(out_hi) << 32) | uint64_t(out_lo);

                    return Bits::Decode(static_cast<U>(out));
                }
            }

            // Named barrier synchronization for subset of threads. Unlike __syncthreads() which
            // synchronizes all threads in a block, barrier.cta.sync can synchronize just the
            // threads participating in a specific operation (e.g., only RADIX_COUNT threads
            // doing prefix sum). This reduces synchronization overhead when not all threads
            // need to wait.
            //
            // NVIDIA's barrier instruction accepts runtime barrier IDs, but if the compiler
            // cannot prove the barrier count at compile time, it assumes worst case of 16.
            // Each SM has only 16 barrier slots total; consuming all 16 means zero other
            // blocks can run on that SM, resulting in bad occupancy. NVIDIA could have
            // added a mask operand to specify which barriers are actually used, but chose
            // not to. The template recursion is our workaround: manual unrolling generates
            // conditional branches where each path has a compile-time constant barrier ID.
            // This lets the compiler see the actual number of barriers (NUM_BARRIERS) so
            // occupancy isn't artificially limited.
            template <int NUM_BARRIERS, int NUM_THREADS, int B = 0>
            CUSORT_DEVICE CUSORT_FORCEINLINE void barrier_cta_sync(unsigned int barrier_id)
            {
                static_assert(NUM_THREADS > 0 && (NUM_THREADS % 32) == 0, "NUM_THREADS must be a positive multiple of warp size (32)");

                if constexpr (B == NUM_BARRIERS - 1)
                {
                    asm volatile("barrier.cta.sync %0, %1;" ::"n"(B), "n"(NUM_THREADS) : "memory");
                }
                else if (barrier_id == B)
                {
                    asm volatile("barrier.cta.sync %0, %1;" ::"n"(B), "n"(NUM_THREADS) : "memory");
                }
                else
                {
                    barrier_cta_sync<NUM_BARRIERS, NUM_THREADS, B + 1>(barrier_id);
                }
            }

            // Memory fence or delay to prevent compiler from hoisting loads above this point.
            // In lookback, we spin-wait for previous tiles to publish their status. Without
            // this fence, the compiler might hoist the load outside the loop, causing infinite
            // spin. On Volta+ we use nanosleep for power efficiency; older GPUs use threadfence.
            template <typename T> CUSORT_DEVICE CUSORT_FORCEINLINE void delay_or_prevent_hoisting(T delay)
            {
                #if CUSORT_DEVICE_ARCH >= 700
                    __nanosleep(delay);
                #else
                    __threadfence_block(); (void)(delay);
                #endif
            }

            // CUDA 13+ pragma to allow register spilling to shared memory instead of local memory.
            // When a kernel exceeds available registers, excess values "spill" to slower memory.
            // Local memory (the default spill target) is actually global memory with ~400 cycle latency.
            // Shared memory spilling uses on-chip SRAM with ~20 cycle latency - 20x faster.
            // This is critical for our register-heavy kernels that process multiple items per thread.
            CUSORT_DEVICE CUSORT_FORCEINLINE void enable_smem_spilling()
            {
                #if CUSORT_DEVICE_ARCH >= 750 && !CUSORT_DISABLE_SMEM_SPILLING
                    asm volatile(".pragma \"enable_smem_spilling\";");
                #endif
            }

            // Asynchronous memory copy from global to shared memory. On sm_80+, this instruction
            // initiates a DMA transfer that proceeds independently while the SM executes other
            // instructions. The 'cg' variant caches in L2 only (not L1), which is appropriate
            // for data that will be consumed once then discarded. Limited to 16-byte transfers.
            CUSORT_DEVICE CUSORT_FORCEINLINE void cp_async_cg_16(void *smem_ptr, const void *gmem_ptr)
            {
                #if CUSORT_DEVICE_ARCH >= 800 && !CUSORT_DISABLE_ASYNC_COPY
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr))), "l"(gmem_ptr) : "memory");
                #else
                    *reinterpret_cast<ulonglong2 *>(smem_ptr) = *reinterpret_cast<const ulonglong2 *>(gmem_ptr);
                #endif
            }

            // Wait for outstanding async copies to complete. The template parameter N specifies
            // how many copy groups may still be in-flight (0 = wait for all). Used before __syncthreads()
            // to ensure each thread's async copies are visible before the barrier.
            template <unsigned int N> CUSORT_DEVICE CUSORT_FORCEINLINE void cp_async_wait_group()
            {
                #if CUSORT_DEVICE_ARCH >= 800 && !CUSORT_DISABLE_ASYNC_COPY
                    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
                #endif
            }

            // Programmatic Dependent Launch (PDL) on sm_90+. Signals that this kernel's output
            // is ready for dependent kernels to consume. The dependent kernel can start executing
            // its initial phases (e.g., zeroing histograms) while this kernel finishes its final
            // writes. This overlaps kernels without CPU involvement, hiding launch latency.
            CUSORT_DEVICE CUSORT_FORCEINLINE void grid_launch_dependents()
            {
                #if CUSORT_DEVICE_ARCH >= 900 && !CUSORT_DISABLE_CUDA_GRAPHS_AND_PDL
                    cudaTriggerProgrammaticLaunchCompletion();
                #endif
            }

            // Wait for predecessor kernel to signal completion via grid_launch_dependents().
            // Used at the start of each radix pass (except pass 0) to ensure the previous pass
            // has finished writing before we read its output. Combined with PDL, this enables
            // kernel pipelining where passes overlap at their boundaries.
            CUSORT_DEVICE CUSORT_FORCEINLINE void grid_dependents_sync()
            {
                #if CUSORT_DEVICE_ARCH >= 900 && !CUSORT_DISABLE_CUDA_GRAPHS_AND_PDL
                    cudaGridDependencySynchronize();
                #endif
            }
        } // namespace ptx

        // Device-side helper functions for the radix sort algorithm.
        // These are building blocks used by the main histogram and onesweep kernels.
        namespace kernels
        {
            // Decoupled lookback for single-pass parallel prefix sum across tiles.
            // Traditional prefix sum requires a separate kernel pass to combine tile results.
            // Decoupled lookback embeds the cross-tile reduction into the sorting kernel itself:
            //
            // Each tile's descriptor has a 2-bit status (in high bits) and count/prefix (in low bits):
            // - INVALID: Tile hasn't started (initial state, alternates between passes)
            // - PARTIAL: Tile computed local count but hasn't completed lookback
            // - COMPLETE: Tile finished lookback and has its global prefix
            //
            // A tile publishes PARTIAL immediately after computing its local histogram, then
            // walks backwards through predecessor descriptors. When it sees COMPLETE, it adds
            // that prefix and is done. When it sees PARTIAL, it adds that count and continues.
            // This allows tiles to proceed without stalling on slow predecessors.

            // Publish this tile's local digit count with PARTIAL status. Called immediately after
            // histogram accumulation (Phase 3) so successor tiles can start their lookback,
            // overlapping with our remaining local work (ranking, scatter).
            //
            // Uses 'cg' (cache global, L2 only) because descriptors are cross-tile communication:
            // written by one SM and read by different SMs. L1 is per-SM so caching there is useless
            // for the reader. L2 is shared across all SMs, making it the right cache level for
            // inter-tile data. Bypassing L1 also avoids polluting it with data never reused locally.
            template <typename OffsetT>
            CUSORT_DEVICE CUSORT_FORCEINLINE void LookbackPublishPartial(OffsetT *CUSORT_RESTRICT d_descriptors, unsigned int digit_count, unsigned int pass)
            {
                constexpr OffsetT STATUS_PARTIAL = OffsetT{1u} << (sizeof(OffsetT) * 8 - 2);

                (void)pass;
                ptx::st_global<ptx::st_global_cache_op::cg>(d_descriptors, static_cast<OffsetT>(digit_count) + STATUS_PARTIAL);
            }

            // Perform lookback to compute global prefix, then publish COMPLETE status.
            // Walks backward through descriptor array, accumulating counts from PARTIAL tiles
            // until hitting a COMPLETE tile (which includes all prior counts). The exponential
            // backoff delay (8, 16, 32...) reduces memory traffic when spinning on a slow tile.
            // STATUS_INVALID detection handles the alternating-passes scheme where the high bit
            // toggles each pass to distinguish "not started" from previous pass's "complete".
            //
            // The histogram kernel writes in reverse pass order, so when tile 0 of pass X does
            // lookback (reading d_descriptors - RADIX_COUNT), it reads the histogram for pass X.
            // This histogram entry has COMPLETE status with the global digit count as prefix,
            // acting as a "virtual tile -1" that terminates the lookback chain.
            template <typename OffsetT, unsigned int LOOKBACK_STRIDE>
            CUSORT_DEVICE CUSORT_FORCEINLINE OffsetT LookbackComputePrefixAndPublishComplete(OffsetT *CUSORT_RESTRICT d_descriptors, unsigned int digit_count, unsigned int pass)
            {
                // 2-bit status encoding in high bits: 00/10=INVALID, 01=PARTIAL, 11/01=COMPLETE
                // Pass number toggles the high bit so INVALID alternates: pass 0 uses 00, pass 1 uses 10.
                // This avoids needing to memset descriptors between passes - stale COMPLETE from pass N-1
                // reads as INVALID for pass N because the high bit differs.
                constexpr OffsetT STATUS_MASK       = OffsetT{3u} << (sizeof(OffsetT) * 8 - 2);
                constexpr OffsetT STATUS_PARTIAL    = OffsetT{1u} << (sizeof(OffsetT) * 8 - 2);
                constexpr OffsetT PREFIX_MASK       = ~STATUS_MASK;

                const     OffsetT STATUS_INVALID    = OffsetT{pass} << (sizeof(OffsetT) * 8 - 1);
                const     OffsetT STATUS_COMPLETE   = STATUS_INVALID ^ (STATUS_MASK ^ STATUS_PARTIAL);

                OffsetT prefix_sum = 0;

                {
                    OffsetT * descriptors_lookback = d_descriptors;

                    OffsetT descriptor; unsigned int delay = 8;
                    do
                    {
                        descriptors_lookback -= LOOKBACK_STRIDE;

                        do
                        {
                            ptx::delay_or_prevent_hoisting(delay <<= 1);

                            descriptor = ptx::ld_global<ptx::ld_global_cache_op::cg>(descriptors_lookback);

                        } while ((descriptor & STATUS_MASK) == STATUS_INVALID);

                        delay = 0; prefix_sum += (descriptor & PREFIX_MASK);

                    } while ((descriptor & STATUS_MASK) == STATUS_PARTIAL);
                }

                {
                    ptx::st_global<ptx::st_global_cache_op::cg>(d_descriptors, (prefix_sum + static_cast<OffsetT>(digit_count)) + STATUS_COMPLETE);
                }

                return prefix_sum;
            }

            // Block-wide memory copy using warp-striped access pattern. Each warp handles a
            // contiguous chunk of data, with consecutive lanes reading consecutive addresses.
            // This maximizes memory coalescing - the GPU's memory controller can combine
            // aligned sequential accesses into single wide transactions (128 bytes on modern GPUs).
            // The vectorization cascade (uint4 -> uint64_t -> uint32_t -> uint16_t) picks the
            // widest type that maintains alignment, maximizing bytes per instruction.
            template <ptx::ld_global_cache_op LoadOp, ptx::st_global_cache_op StoreOp, typename T, int ITEMS_PER_THREAD>
            CUSORT_DEVICE CUSORT_FORCEINLINE void WarpStripedBlockCopy(const T * CUSORT_RESTRICT d_src, T * CUSORT_RESTRICT d_dst)
            {
                constexpr unsigned int WARP_SIZE = 32;

                const unsigned int warp_id = ptx::warp_id();
                const unsigned int lane_id = ptx::lane_id();

                if constexpr ((sizeof(T) * ITEMS_PER_THREAD) % sizeof(uint4) == 0 && sizeof(T) <= sizeof(uint4))
                {
                    if (sizeof(T) == sizeof(uint4) || ((reinterpret_cast<uintptr_t>(d_src) | reinterpret_cast<uintptr_t>(d_dst)) % sizeof(uint4)) == 0)
                    {
                        constexpr unsigned int VECTORS_PER_THREAD = (sizeof(T) * ITEMS_PER_THREAD) / sizeof(uint4);

                        const uint4* src_vec = reinterpret_cast<const uint4*>(d_src);
                              uint4* dst_vec = reinterpret_cast<      uint4*>(d_dst);

                        const unsigned int thread_offset = warp_id * WARP_SIZE * VECTORS_PER_THREAD + lane_id;

                        src_vec += thread_offset; dst_vec += thread_offset;

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int i = 0; i < VECTORS_PER_THREAD; ++i)
                        {
                            ptx::st_global<StoreOp>(dst_vec + i * WARP_SIZE, ptx::ld_global<LoadOp>(src_vec + i * WARP_SIZE));
                        }

                        return;
                    }
                }

                if constexpr ((sizeof(T) * ITEMS_PER_THREAD) % sizeof(uint64_t) == 0 && sizeof(T) <= sizeof(uint64_t))
                {
                    if (sizeof(T) == sizeof(uint64_t) || ((reinterpret_cast<uintptr_t>(d_src) | reinterpret_cast<uintptr_t>(d_dst)) % sizeof(uint64_t)) == 0)
                    {
                        constexpr unsigned int VECTORS_PER_THREAD = (sizeof(T) * ITEMS_PER_THREAD) / sizeof(uint64_t);

                        const uint64_t* src_vec = reinterpret_cast<const uint64_t*>(d_src);
                              uint64_t* dst_vec = reinterpret_cast<      uint64_t*>(d_dst);

                        const unsigned int thread_offset = warp_id * WARP_SIZE * VECTORS_PER_THREAD + lane_id;

                        src_vec += thread_offset; dst_vec += thread_offset;

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int i = 0; i < VECTORS_PER_THREAD; ++i)
                        {
                            ptx::st_global<StoreOp>(dst_vec + i * WARP_SIZE, ptx::ld_global<LoadOp>(src_vec + i * WARP_SIZE));
                        }

                        return;
                    }
                }

                if constexpr ((sizeof(T) * ITEMS_PER_THREAD) % sizeof(uint32_t) == 0 && sizeof(T) <= sizeof(uint32_t))
                {
                    if (sizeof(T) == sizeof(uint32_t) || ((reinterpret_cast<uintptr_t>(d_src) | reinterpret_cast<uintptr_t>(d_dst)) % sizeof(uint32_t)) == 0)
                    {
                        constexpr unsigned int VECTORS_PER_THREAD = (sizeof(T) * ITEMS_PER_THREAD) / sizeof(uint32_t);

                        const uint32_t* src_vec = reinterpret_cast<const uint32_t*>(d_src);
                              uint32_t* dst_vec = reinterpret_cast<      uint32_t*>(d_dst);

                        const unsigned int thread_offset = warp_id * WARP_SIZE * VECTORS_PER_THREAD + lane_id;

                        src_vec += thread_offset; dst_vec += thread_offset;

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int i = 0; i < VECTORS_PER_THREAD; ++i)
                        {
                            ptx::st_global<StoreOp>(dst_vec + i * WARP_SIZE, ptx::ld_global<LoadOp>(src_vec + i * WARP_SIZE));
                        }

                        return;
                    }
                }

                if constexpr ((sizeof(T) * ITEMS_PER_THREAD) % sizeof(uint16_t) == 0 && sizeof(T) <= sizeof(uint16_t))
                {
                    if (sizeof(T) == sizeof(uint16_t) || ((reinterpret_cast<uintptr_t>(d_src) | reinterpret_cast<uintptr_t>(d_dst)) % sizeof(uint16_t)) == 0)
                    {
                        constexpr unsigned int VECTORS_PER_THREAD = (sizeof(T) * ITEMS_PER_THREAD) / sizeof(uint16_t);

                        const uint16_t* src_vec = reinterpret_cast<const uint16_t*>(d_src);
                              uint16_t* dst_vec = reinterpret_cast<      uint16_t*>(d_dst);

                        const unsigned int thread_offset = warp_id * WARP_SIZE * VECTORS_PER_THREAD + lane_id;

                        src_vec += thread_offset; dst_vec += thread_offset;

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int i = 0; i < VECTORS_PER_THREAD; ++i)
                        {
                            ptx::st_global<StoreOp>(dst_vec + i * WARP_SIZE, ptx::ld_global<LoadOp>(src_vec + i * WARP_SIZE));
                        }

                        return;
                    }
                }

                if constexpr (sizeof(T) < sizeof(uint16_t))
                {
                    const unsigned int thread_offset = warp_id * WARP_SIZE * ITEMS_PER_THREAD + lane_id;

                    d_src += thread_offset; d_dst += thread_offset;

                    CUSORT_PRAGMA_UNROLL_FULL
                    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                    {
                        ptx::st_global<StoreOp>(d_dst + i * WARP_SIZE, ptx::ld_global<LoadOp>(d_src + i * WARP_SIZE));
                    }
                }
            }

            // Load keys into registers with warp-striped layout and sub-word packing.
            // For 8/16-bit keys, multiple keys fit in one 32-bit register (GPU's native size).
            // Packing reduces register count and increases occupancy, but requires careful
            // handling: loads must respect alignment, and the sentinel value fills out-of-bounds
            // slots so they sort to a predictable position (last bucket) without special-casing.
            template <ptx::ld_global_cache_op LoadOp, typename ItemT, typename OffsetT, typename RegisterT, unsigned int REGISTERS_COUNT, RegisterT SENTINEL>
            CUSORT_DEVICE CUSORT_FORCEINLINE void WarpStripedRegisterPackedThreadLoad(
                const ItemT * CUSORT_RESTRICT d_input,
                OffsetT thread_offset,
                OffsetT num_items,
                RegisterT (&items)[REGISTERS_COUNT])
            {
                constexpr unsigned int ITEMS_PER_REGISTER   = sizeof(RegisterT) / sizeof(ItemT);
                constexpr unsigned int WARP_SIZE            = 32;

                static_assert(sizeof(ItemT) == 1 || sizeof(ItemT) == 2 || sizeof(ItemT) == sizeof(RegisterT), "ItemT must be 1, 2 or equal to RegisterT size for correctness");
                static_assert(sizeof(ItemT) >= 4 || sizeof(RegisterT) == 4, "When ItemT is 1 or 2 bytes, RegisterT must be 4 bytes");

                d_input += thread_offset; const OffsetT remaining = ::max(num_items, thread_offset) - thread_offset;

                if (remaining < (REGISTERS_COUNT - 1) * WARP_SIZE * ITEMS_PER_REGISTER + ITEMS_PER_REGISTER)
                {
                    if constexpr (sizeof(ItemT) == 1)
                    {
                        const uint8_t *d_input_reg8 = reinterpret_cast<const uint8_t *>(d_input);

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int r = 0; r < REGISTERS_COUNT; ++r)
                        {
                            OffsetT offset = static_cast<OffsetT>(r * WARP_SIZE * ITEMS_PER_REGISTER);

                            const uint32_t b0 = (remaining > offset + 0) ? static_cast<uint32_t>(ptx::ld_global<LoadOp>(d_input_reg8 + offset + 0)) : ((SENTINEL << 24) >> 24);
                            const uint32_t b1 = (remaining > offset + 1) ? static_cast<uint32_t>(ptx::ld_global<LoadOp>(d_input_reg8 + offset + 1)) : ((SENTINEL << 16) >> 24);
                            const uint32_t b2 = (remaining > offset + 2) ? static_cast<uint32_t>(ptx::ld_global<LoadOp>(d_input_reg8 + offset + 2)) : ((SENTINEL <<  8) >> 24);
                            const uint32_t b3 = (remaining > offset + 3) ? static_cast<uint32_t>(ptx::ld_global<LoadOp>(d_input_reg8 + offset + 3)) : ((SENTINEL <<  0) >> 24);

                            items[r] = static_cast<RegisterT>(__byte_perm(b0, b1, 0x5140) + __byte_perm(b2, b3, 0x4051));
                        }
                    }
                    else if constexpr (sizeof(ItemT) == 2)
                    {
                        const uint16_t *d_input_reg16 = reinterpret_cast<const uint16_t *>(d_input);
                        const uint32_t *d_input_reg32 = reinterpret_cast<const uint32_t *>(d_input);

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int r = 0; r < REGISTERS_COUNT; ++r)
                        {
                            OffsetT offset = static_cast<OffsetT>(r * WARP_SIZE * ITEMS_PER_REGISTER);

                            items[r] = SENTINEL;
                            if (remaining >= offset + 2) { items[r] = static_cast<RegisterT>(ptx::ld_global<LoadOp>(d_input_reg32 + offset / 2)); }
                            if (remaining == offset + 1) { items[r] = static_cast<RegisterT>(ptx::ld_global<LoadOp>(d_input_reg16 + offset)) + ((SENTINEL >> 16) << 16); }
                        }
                    }
                    else
                    {
                        const RegisterT *d_input_reg = reinterpret_cast<const RegisterT *>(d_input);

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int r = 0; r < REGISTERS_COUNT; ++r)
                        {
                            OffsetT offset = static_cast<OffsetT>(r * WARP_SIZE);
                            items[r] = remaining > offset ? ptx::ld_global<LoadOp>(d_input_reg + offset) : SENTINEL;
                        }
                    }
                }
                else
                {
                    const RegisterT *d_input_reg = reinterpret_cast<const RegisterT *>(d_input);

                    CUSORT_PRAGMA_UNROLL_FULL
                    for (unsigned int r = 0; r < REGISTERS_COUNT; ++r)
                    {
                        OffsetT offset = static_cast<OffsetT>(r * WARP_SIZE);
                        items[r] = ptx::ld_global<LoadOp>(d_input_reg + offset);
                    }
                }
            }

            // Store keys from registers to global memory - the reverse of WarpStripedRegisterPackedThreadLoad.
            //
            // Unlike load (which reads from a contiguous, aligned tile), store does scatter and the
            // destination may be unaligned even though it's contiguous. When output isn't aligned to
            // register width (e.g., 4 bytes), we can't do wide register stores.
            //
            // The problem: registers hold packed items in thread order, but we need warp-striped
            // output. Example with 4 uint8 keys per register, 2 registers per thread:
            //
            //   Thread 0 reg[0]: [A0 A1 A2 A3]   Thread 1 reg[0]: [B0 B1 B2 B3]   ...
            //   Thread 0 reg[1]: [A4 A5 A6 A7]   Thread 1 reg[1]: [B4 B5 B6 B7]   ...
            //
            //   Warp-striped output: [A0 B0 C0 ... A1 B1 C1 ... A2 B2 C2 ...]
            //
            // Solution: shuffle sub-register items between lanes so each lane holds the item it
            // needs to write at its position, then do scalar stores. Slower than wide stores but
            // necessary for unaligned scatter destinations.
            template <ptx::st_global_cache_op StoreOp, typename ItemT, typename OffsetT, typename RegisterT, unsigned int REGISTERS_COUNT>
            CUSORT_DEVICE CUSORT_FORCEINLINE void WarpStripedRegisterPackedThreadStore(
                RegisterT (&items)[REGISTERS_COUNT],
                ItemT * CUSORT_RESTRICT d_output,
                OffsetT thread_offset)
            {
                constexpr unsigned int ITEMS_PER_REGISTER   = sizeof(RegisterT) / sizeof(ItemT);
                constexpr unsigned int WARP_SIZE            = 32;

                static_assert(sizeof(ItemT) == 1 || sizeof(ItemT) == 2 || sizeof(ItemT) == sizeof(RegisterT), "ItemT must be 1, 2 or equal to RegisterT size for correctness");
                static_assert(sizeof(ItemT) >= 4 || sizeof(RegisterT) == 4, "When ItemT is 1 or 2 bytes, RegisterT must be 4 bytes");

                d_output += thread_offset;

                if (ITEMS_PER_REGISTER != 1 && reinterpret_cast<uintptr_t>(d_output) % sizeof(RegisterT) != 0)
                {
                    const unsigned int lane_id       = ptx::lane_id();
                    const unsigned int source_lane   = (lane_id / ITEMS_PER_REGISTER);
                    const unsigned int source_shift  = (lane_id % ITEMS_PER_REGISTER) * sizeof(ItemT) * 8;

                    d_output -= lane_id * (ITEMS_PER_REGISTER - 1);

                    CUSORT_PRAGMA_UNROLL_FULL
                    for (unsigned int r = 0; r < REGISTERS_COUNT; ++r)
                    {
                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int s = 0; s < ITEMS_PER_REGISTER; ++s)
                        {
                            const uint32_t source_packed = __shfl_sync(~0u, items[r], source_lane + s * (WARP_SIZE / ITEMS_PER_REGISTER));
                            ptx::st_global<StoreOp>(d_output + (r * ITEMS_PER_REGISTER + s) * WARP_SIZE, static_cast<ItemT>(source_packed >> source_shift));
                        }
                    }
                }
                else
                {
                    RegisterT *d_output_reg = reinterpret_cast<RegisterT *>(d_output);

                    CUSORT_PRAGMA_UNROLL_FULL
                    for (unsigned int r = 0; r < REGISTERS_COUNT; ++r)
                    {
                        ptx::st_global<StoreOp>(d_output_reg + r * WARP_SIZE, items[r]);
                    }
                }
            }

            template <typename HistogramPolicy, bool IS_DESCENDING, typename KeyT, typename OffsetT>
            CUSORT_GLOBAL CUSORT_LAUNCH_BOUNDS(HistogramPolicy::BLOCK_THREADS, HistogramPolicy::BLOCKS_PER_MULTIPROCESSOR) void
            RadixSortHistogramKernel(
                const KeyT * CUSORT_RESTRICT d_keys,
                OffsetT * CUSORT_RESTRICT d_histogram,
                OffsetT num_items,
                unsigned int begin_bit,
                unsigned int partial_digit_mask,
                unsigned int num_passes)
            {
                // On CUDA 13+ and Turing+, allow compiler to spill registers to shared memory instead of local memory for better performance.
                ptx::enable_smem_spilling();

                using RadixTraits                           = traits::RadixTraits<IS_DESCENDING, KeyT>;
                using UnsignedRegT                          = typename RadixTraits::UnsignedRegT;

                constexpr auto LOAD_OP                      = HistogramPolicy::LOAD_OP;

                constexpr unsigned int WARP_SIZE            = HistogramPolicy::WARP_SIZE;
                constexpr unsigned int RADIX_BITS           = HistogramPolicy::RADIX_BITS;
                constexpr unsigned int RADIX_COUNT          = HistogramPolicy::RADIX_COUNT;
                constexpr unsigned int BLOCK_THREADS        = HistogramPolicy::BLOCK_THREADS;
                constexpr unsigned int ITEMS_PER_THREAD     = HistogramPolicy::ITEMS_PER_THREAD;

                constexpr unsigned int FULL_DIGIT_MASK      = RADIX_COUNT - 1;
                constexpr unsigned int MAX_RADIX_PASSES     = (sizeof(KeyT) * 8 + RADIX_BITS - 1) / RADIX_BITS;

                constexpr unsigned int ITEMS_PER_REGISTER   = sizeof(UnsignedRegT) / sizeof(KeyT);
                constexpr unsigned int REGISTERS_PER_THREAD = ITEMS_PER_THREAD / ITEMS_PER_REGISTER;

                constexpr OffsetT TILE_ITEMS                = static_cast<OffsetT>(BLOCK_THREADS) * static_cast<OffsetT>(ITEMS_PER_THREAD);
                constexpr OffsetT WARP_ITEMS                = static_cast<OffsetT>(WARP_SIZE) * static_cast<OffsetT>(ITEMS_PER_THREAD);

                static_assert(ITEMS_PER_REGISTER == 1 || ITEMS_PER_REGISTER == 2 || ITEMS_PER_REGISTER == 4, "ITEMS_PER_REGISTER must be 1, 2, or 4");
                static_assert(ITEMS_PER_THREAD % ITEMS_PER_REGISTER == 0, "ITEMS_PER_THREAD must be a multiple of ITEMS_PER_REGISTER for correctness");
                static_assert(BLOCK_THREADS % RADIX_COUNT == 0, "BLOCK_THREADS must be a multiple of RADIX_COUNT for correctness");

                __shared__ OffsetT s_histogram[MAX_RADIX_PASSES * RADIX_COUNT];

                // PHASE 0: Zero shared memory histogram bins for radix passes.
                {
                    uint4 *s_histogram_uint4 = reinterpret_cast<uint4 *>(s_histogram);
                    const unsigned int limit = num_passes * (RADIX_COUNT * sizeof(OffsetT) / sizeof(uint4));

                    CUSORT_PRAGMA_UNROLL_FULL
                    for (unsigned int i = threadIdx.x; i < MAX_RADIX_PASSES * (RADIX_COUNT * sizeof(OffsetT) / sizeof(uint4)); i += BLOCK_THREADS)
                    {
                        if (i >= limit) break;

                        // Zero 16 bytes per iteration via a single vectorized store instruction.
                        s_histogram_uint4[i] = make_uint4(0, 0, 0, 0);
                    }
                }

                // Ensure all histogram bins are zeroed before any thread starts accumulating digit counts.
                __syncthreads();

                // PHASE 1: Load keys from global memory, apply bit-order transformation, and accumulate per-pass digit counts in shared memory.
                {
                    OffsetT thread_offset = static_cast<OffsetT>(blockIdx.x) * TILE_ITEMS + static_cast<OffsetT>(ptx::warp_id()) * WARP_ITEMS + static_cast<OffsetT>(ptx::lane_id()) * ITEMS_PER_REGISTER;

                    // Use __any_sync so entire warp exits together when all lanes are out-of-bounds, avoiding divergent warp behavior and generating tighter code.
                    for (;__any_sync(~0u, thread_offset < num_items); thread_offset += static_cast<OffsetT>(gridDim.x) * TILE_ITEMS)
                    {
                        UnsignedRegT bit_ordered_keys[REGISTERS_PER_THREAD];

                        // TwiddleSentinel maps to last bin (0xff) after transformation, letting out-of-bounds lanes participate uniformly; their counts don't affect real offsets due to exclusive prefix sum.
                        WarpStripedRegisterPackedThreadLoad<LOAD_OP, KeyT, OffsetT, UnsignedRegT, REGISTERS_PER_THREAD, RadixTraits::TwiddleSentinel()>(d_keys, thread_offset, num_items, bit_ordered_keys);

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int r = 0; r < REGISTERS_PER_THREAD; ++r)
                        {
                            // TwiddleForward converts signed/float keys to unsigned with correct sort order so comparison works bitwise.
                            // Pre-shift by begin_bit here so inner loop shifts become compile-time constants (same total ops, but enables better optimization).
                            bit_ordered_keys[r] = RadixTraits::TwiddleForward(bit_ordered_keys[r]) >> begin_bit;
                        }

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int pass = 0; pass < MAX_RADIX_PASSES; ++pass)
                        {
                            OffsetT *s_pass_histogram       = s_histogram + pass * RADIX_COUNT;
                            const unsigned int digit_mask   = (num_passes == pass + 1) ? partial_digit_mask : FULL_DIGIT_MASK;

                            CUSORT_PRAGMA_UNROLL_FULL
                            for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                            {
                                // For 1-byte/2-byte keys, multiple keys are packed per 32-bit register; r selects the register, s selects the key within it.
                                const unsigned int r        = i / ITEMS_PER_REGISTER;
                                const unsigned int s        = i % ITEMS_PER_REGISTER;
                                const unsigned int shift    = pass * RADIX_BITS + s * (sizeof(KeyT) * 8);
                                const unsigned int digit    = static_cast<unsigned int>(bit_ordered_keys[r] >> shift) & digit_mask;

                                // On Ampere+, this compiles to ATOMS.POPC.INC which is efficient without sharding or warp-leader aggregation.
                                atomicAdd(&s_pass_histogram[digit], static_cast<OffsetT>(1));
                            }

                            if (num_passes == pass + 1) break;
                        }
                    }
                }

                // Launch first pass of Onesweep early; its Phases 0-7 can nicely overlap with our Phase 2 until Onesweep reaches Phase 8 (lookback).
                ptx::grid_launch_dependents();

                // Ensure all digit counts are accumulated before computing prefix sums.
                __syncthreads();

                // PHASE 2: Compute exclusive prefix sum of histogram bins and atomically add to global histogram with lookback status encoding.
                {
                    CUSORT_PRAGMA_UNROLL_FULL
                    for (unsigned int i = threadIdx.x; i < MAX_RADIX_PASSES * RADIX_COUNT; i += BLOCK_THREADS)
                    {
                        // Use of threadIdx.x % RADIX_COUNT instead of i % RADIX_COUNT yield same value but compiler recognizes the former as loop-invariant.
                        const unsigned int bin  = threadIdx.x % RADIX_COUNT;
                        const unsigned int pass = i / RADIX_COUNT;

                        if (num_passes <= pass) return;

                        OffsetT count = s_histogram[i];

                        {
                            // Ensure all threads have read their count before reusing s_histogram as warp staging area.
                            __syncthreads();

                            OffsetT *s_warp_staging = &s_histogram[pass * RADIX_COUNT];

                            // Derive lane_id/warp_id from bin (0..RADIX_COUNT-1) so the prefix sum correctly operates over bins rather than physical threads.
                            const unsigned int lane_id  = bin % WARP_SIZE;
                            const unsigned int warp_id  = bin / WARP_SIZE;

                            OffsetT inclusive = ptx::warp_inclusive_scan(count);
                            count = inclusive - count;

                            if (lane_id == WARP_SIZE - 1) { s_warp_staging[warp_id] = inclusive; }

                            // Ensure warp staging values are visible to all threads before cross-warp accumulation.
                            __syncthreads();

                            CUSORT_PRAGMA_UNROLL_FULL
                            for (unsigned int w = 0; w < RADIX_COUNT / WARP_SIZE - 1; ++w)
                            {
                                if (w < warp_id) { count += s_warp_staging[w]; }
                            }
                        }

                        {
                            // Toggle high bit on alternating passes so STATUS_INVALID and STATUS_COMPLETE have distinct lookback bit patterns across consecutive passes.
                            if (blockIdx.x == 0 && (pass % 2 == 0))
                            {
                                count += (OffsetT{1u} << (sizeof(OffsetT) * 8 - 1));
                            }

                            if (count != 0)
                            {
                                // Store histogram in reverse pass order so d_descriptors[-1] serves as the terminating tile prefix for onesweep lookback across consecutive passes.
                                OffsetT *d_pass_histogram = d_histogram + (num_passes - 1 - pass) * RADIX_COUNT;

                                atomicAdd(&d_pass_histogram[bin], count);
                            }
                        }
                    }
                }
            }

            template <typename OnesweepPolicy, bool IS_DESCENDING, typename KeyT, typename ValueT, typename OffsetT>
            CUSORT_GLOBAL CUSORT_LAUNCH_BOUNDS(OnesweepPolicy::BLOCK_THREADS, OnesweepPolicy::BLOCKS_PER_MULTIPROCESSOR) void
            RadixSortOnesweepKernel(
                const KeyT * CUSORT_RESTRICT d_keys_in,
                KeyT * CUSORT_RESTRICT d_keys_out,
                const ValueT * CUSORT_RESTRICT d_values_in,
                ValueT * CUSORT_RESTRICT d_values_out,
                OffsetT * CUSORT_RESTRICT d_descriptors,
                OffsetT num_items,
                unsigned int current_bit,
                unsigned int digit_mask,
                unsigned int pass)
            {
                // On CUDA 13+ and Turing+, allow compiler to spill registers to shared memory instead of local memory for better performance.
                ptx::enable_smem_spilling();

                using RadixTraits                           = traits::RadixTraits<IS_DESCENDING, KeyT>;
                using UnsignedKeyT                          = typename RadixTraits::UnsignedKeyT;
                using UnsignedRegT                          = typename RadixTraits::UnsignedRegT;

                constexpr bool KEYS_ONLY                    = std::is_same_v<ValueT, NullType>;

                constexpr auto LOAD_OP                      = OnesweepPolicy::LOAD_OP;
                constexpr auto STORE_OP                     = OnesweepPolicy::STORE_OP;

                constexpr unsigned int WARP_SIZE            = OnesweepPolicy::WARP_SIZE;
                constexpr unsigned int RADIX_BITS           = OnesweepPolicy::RADIX_BITS;
                constexpr unsigned int RADIX_COUNT          = OnesweepPolicy::RADIX_COUNT;
                constexpr unsigned int BLOCK_THREADS        = OnesweepPolicy::BLOCK_THREADS;
                constexpr unsigned int ITEMS_PER_THREAD     = OnesweepPolicy::ITEMS_PER_THREAD;

                constexpr unsigned int NUM_WARPS            = BLOCK_THREADS / WARP_SIZE;
                constexpr unsigned int ITEMS_PER_REGISTER   = sizeof(UnsignedRegT) / sizeof(UnsignedKeyT);
                constexpr unsigned int REGISTERS_PER_THREAD = ITEMS_PER_THREAD / ITEMS_PER_REGISTER;

                constexpr OffsetT WARP_ITEMS                = WARP_SIZE * ITEMS_PER_THREAD;
                constexpr OffsetT TILE_ITEMS                = BLOCK_THREADS * ITEMS_PER_THREAD;

                UnsignedRegT bit_ordered_keys[REGISTERS_PER_THREAD];

                // CUDA registers are 32-bit; packing two 16-bit offsets per register avoids wasting half of each register.
                uint32_t offsets[(ITEMS_PER_THREAD + 1) / 2];

                // digit_count computed in Phase 3 is needed in Phases 4, 5, and 8; cannot recompute because warp_histogram gets overwritten.
                unsigned int digit_count;

                // Union reuses same shared memory for histogram (Phases 0-6), then keys scatter (Phase 7-9), then values scatter (Phase 10-11).
                __shared__ union
                {
                    unsigned int warp_histogram[NUM_WARPS * RADIX_COUNT + NUM_WARPS];
                    UnsignedKeyT scattered_keys[TILE_ITEMS];
                    ValueT       scattered_values[TILE_ITEMS];
                } s_mem;

                __shared__ OffsetT s_tile_prefix[RADIX_COUNT];

                static_assert(ITEMS_PER_REGISTER == 1 || ITEMS_PER_REGISTER == 2 || ITEMS_PER_REGISTER == 4, "ITEMS_PER_REGISTER must be 1, 2, or 4");
                static_assert(ITEMS_PER_THREAD % ITEMS_PER_REGISTER == 0, "ITEMS_PER_THREAD must be a multiple of ITEMS_PER_REGISTER for correctness");
                static_assert(TILE_ITEMS <= 65536u, "TILE_ITEMS must fit in 16 bits for offset packing");
                static_assert(BLOCK_THREADS >= RADIX_COUNT, "BLOCK_THREADS must be greater than or equal to RADIX_COUNT for decoupled lookback");

                // PHASE 0: Zero per-warp histogram bins.
                {
                    uint4 *s_warp_histogram = reinterpret_cast<uint4 *>(s_mem.warp_histogram + ptx::warp_id() * RADIX_COUNT);

                    CUSORT_PRAGMA_UNROLL_FULL
                    for (unsigned int i = ptx::lane_id(); i < RADIX_COUNT * sizeof(unsigned int) / sizeof(uint4); i += WARP_SIZE)
                    {
                        // Vectorized 16-byte store reduces instruction count vs individual 4-byte stores.
                        s_warp_histogram[i] = make_uint4(0, 0, 0, 0);
                    }
                }

                // PHASE 1: Load keys, transform for sort direction, rotate digit to LSB.
                // Warp-striped layout enables coalesced global memory access (consecutive lanes read consecutive addresses).
                {
                    OffsetT thread_offset = static_cast<OffsetT>(blockIdx.x) * TILE_ITEMS + static_cast<OffsetT>(ptx::warp_id()) * WARP_ITEMS + static_cast<OffsetT>(ptx::lane_id()) * ITEMS_PER_REGISTER;

                    // For passes > 0, wait for previous pass to finish writing output before reading it as input. Pass 0 needs no sync because histogram kernel doesn't modify keys.
                    if (pass != 0) { ptx::grid_dependents_sync(); }

                    WarpStripedRegisterPackedThreadLoad<LOAD_OP, KeyT, OffsetT, UnsignedRegT, REGISTERS_PER_THREAD, RadixTraits::TwiddleSentinel()>(d_keys_in, thread_offset, num_items, bit_ordered_keys);

                    // For sub-32-bit keys, data is loaded packed but ranking needs one key per lane per iteration; transpose via shuffle.
                    // Example for 16-bit keys (2 per register): Before transpose lane 0 has keys [0,1], lane 1 has [2,3], ...
                    // After transpose: lane 0 has [0,32], lane 1 has [1,33], ... Keys are 32 apart because ranking processes items in warp-sized groups.
                    if constexpr (ITEMS_PER_REGISTER != 1)
                    {
                        const unsigned int lane_id      = ptx::lane_id();
                        const unsigned int source_lane  = lane_id / ITEMS_PER_REGISTER;
                        const unsigned int source_index = lane_id % ITEMS_PER_REGISTER;

                        // __byte_perm(a, b, selector) extracts 4 bytes: selector nibble 0-3 picks from a, 4-7 from b.
                        // For 16-bit keys (2 per register), we need to transpose: lane 0 wants bytes 0-1 from lane 0's
                        // register and bytes 0-1 from lane 16's register. Selector 0x5410 picks: byte0=a[0], byte1=a[1],
                        // byte2=b[0], byte3=b[1]. Adding 0x2222*source_index shifts selection for other sub-key positions.
                        // For 8-bit keys (4 per register), selector 0x0040+0x0011*source_index extracts one byte per lane.
                        const unsigned int selector = ITEMS_PER_REGISTER == 2
                            ? 0x5410u + 0x2222u * source_index
                            : 0x0040u + 0x0011u * source_index;

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int r = 0; r < REGISTERS_PER_THREAD; ++r)
                        {
                            // 16-bit keys: gather from 2 lanes (half-warp apart) and merge bytes; 8-bit keys: gather from 4 lanes.
                            if constexpr (ITEMS_PER_REGISTER == 2)
                            {
                                const UnsignedRegT w0 = __shfl_sync(~0u, bit_ordered_keys[r], source_lane);
                                const UnsignedRegT w1 = __shfl_sync(~0u, bit_ordered_keys[r], source_lane + 16u);

                                bit_ordered_keys[r]   = __byte_perm(w0, w1, selector);
                            }
                            else
                            {
                                const UnsignedRegT b0 = __shfl_sync(~0u, bit_ordered_keys[r], source_lane);
                                const UnsignedRegT b1 = __shfl_sync(~0u, bit_ordered_keys[r], source_lane + 8u);
                                const UnsignedRegT b2 = __shfl_sync(~0u, bit_ordered_keys[r], source_lane + 16u);
                                const UnsignedRegT b3 = __shfl_sync(~0u, bit_ordered_keys[r], source_lane + 24u);

                                const UnsignedRegT w0 = __byte_perm(b0, b1, selector);
                                const UnsignedRegT w1 = __byte_perm(b2, b3, selector);

                                bit_ordered_keys[r]   = __byte_perm(w0, w1, 0x5410u);
                            }

                            // Twiddle+rotate must happen after transpose; doing it before would corrupt bytes during cross-lane shuffle.
                            bit_ordered_keys[r] = ptx::rotate_right(RadixTraits::TwiddleForward(bit_ordered_keys[r]), current_bit);
                        }
                    }
                    else
                    {
                        // For 32/64-bit keys, no transpose needed; just twiddle for correct sort order and rotate digit to LSB for fast extraction.
                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int r = 0; r < REGISTERS_PER_THREAD; ++r)
                        {
                            bit_ordered_keys[r] = ptx::rotate_right(RadixTraits::TwiddleForward(bit_ordered_keys[r]), current_bit);
                        }
                    }
                }

                // PHASE 2: Accumulate digit counts using shared memory atomics.
                {
                    unsigned int *s_warp_histogram = s_mem.warp_histogram + ptx::warp_id() * RADIX_COUNT;

                    CUSORT_PRAGMA_UNROLL_FULL
                    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                    {
                        const unsigned int r        = i / ITEMS_PER_REGISTER;
                        const unsigned int s        = i % ITEMS_PER_REGISTER;
                        const unsigned int shift    = s * (sizeof(KeyT) * 8);
                        const unsigned int digit    = static_cast<unsigned int>(bit_ordered_keys[r] >> shift) & digit_mask;

                        atomicAdd(&s_warp_histogram[digit], 1u);
                    }
                }

                // All warps must finish histogram accumulation before reduction to tile-level counts.
                __syncthreads();

                // PHASE 3: Reduce per-warp histograms to tile-level digit counts, publish partial aggregate to global descriptor for lookback.
                {
                    // Initialize outside if-block so threads >= RADIX_COUNT have defined value (avoids undefined behavior in Phase 4 check).
                    digit_count = 0;

                    if (threadIdx.x < RADIX_COUNT)
                    {
                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int w = 0; w < NUM_WARPS; ++w)
                        {
                            digit_count += (s_mem.warp_histogram + w * RADIX_COUNT)[threadIdx.x];
                        }

                        // Publish partial count early so later tiles can start lookback while we continue local work.
                        LookbackPublishPartial<OffsetT>(d_descriptors + blockIdx.x * RADIX_COUNT + threadIdx.x, digit_count, pass);
                    }
                }

                // PHASE 4: Early exit optimization: if all keys in tile share the same digit, bypass ranking and copy directly to output.
                // Common in nearly-sorted data; skipping ranking saves significant compute when all keys go to the same bucket.
                {
                    // Short-circuit only for 32/64-bit keys; sub-32-bit keys need transpose which complicates early exit logic.
                    if constexpr (ITEMS_PER_REGISTER == 1)
                    {
                        bool short_circuit = (digit_count == TILE_ITEMS) && (static_cast<OffsetT>(blockIdx.x) * TILE_ITEMS + TILE_ITEMS <= num_items);

                        // At most one thread (the digit owner) knows short_circuit is true; all threads must agree to take early exit path together.
                        if (__syncthreads_or(short_circuit))
                        {
                            if (threadIdx.x < RADIX_COUNT)
                            {
                                // Lookback reads d_descriptors[-1] as terminating prefix; histogram must finish writing it first. Later passes already synced in Phase 1.
                                if (pass == 0) { ptx::grid_dependents_sync(); }

                                // Decoupled lookback computes global prefix from previous tiles and marks this tile complete for successors.
                                OffsetT prefix_sum = LookbackComputePrefixAndPublishComplete<OffsetT, RADIX_COUNT>(d_descriptors + blockIdx.x * RADIX_COUNT + threadIdx.x, digit_count, pass);

                                // Only one digit has all keys; that thread publishes prefix so all threads know the output destination.
                                if (short_circuit) { s_tile_prefix[0] = prefix_sum; }
                            }

                            // Undo twiddle and rotation to restore original key bits before writing to output.
                            CUSORT_PRAGMA_UNROLL_FULL
                            for (unsigned int r = 0; r < REGISTERS_PER_THREAD; ++r)
                            {
                                bit_ordered_keys[r] = RadixTraits::TwiddleReverse(ptx::rotate_left(bit_ordered_keys[r], current_bit));
                            }

                            // Wait for the one thread with short_circuit to publish prefix_sum before all threads read it.
                            __syncthreads();

                            // All threads can now read the output destination.
                            OffsetT dst_offset = s_tile_prefix[0];
                            OffsetT lane_offset = static_cast<OffsetT>(ptx::warp_id()) * WARP_ITEMS + static_cast<OffsetT>(ptx::lane_id()) * ITEMS_PER_REGISTER;

                            // All keys go to same bucket, so no ranking/scatter needed; write directly from registers preserving warp-striped order.
                            WarpStripedRegisterPackedThreadStore<STORE_OP, KeyT, OffsetT, UnsignedRegT, REGISTERS_PER_THREAD>(bit_ordered_keys, d_keys_out, dst_offset + lane_offset);

                            if constexpr (!KEYS_ONLY)
                            {
                                // No scatter needed - values stay in same relative order, just copy block to new location.
                                OffsetT src_offset = static_cast<OffsetT>(blockIdx.x) * TILE_ITEMS;
                                WarpStripedBlockCopy<LOAD_OP, STORE_OP, ValueT, ITEMS_PER_THREAD>(d_values_in + src_offset, d_values_out + dst_offset);
                            }

                            return;
                        }
                    }
                }

                // PHASE 5: Compute tile-wide prefix sums and per-warp base offsets.
                // Per-warp base offsets ensure each warp's items land after previous warps' items in each digit bucket.
                {
                    if (threadIdx.x < RADIX_COUNT)
                    {
                        // Extra NUM_WARPS slots allocated at end of warp_histogram for staging; separate from histogram data so no sync needed.
                        unsigned int* s_warp_staging = s_mem.warp_histogram + NUM_WARPS * RADIX_COUNT;

                        unsigned int inclusive = ptx::warp_inclusive_scan(digit_count);
                        unsigned int exclusive = inclusive - digit_count;

                        // Last lane of each warp has the warp's total; publish to staging so other warps can compute cross-warp prefix.
                        if (ptx::lane_id() == WARP_SIZE - 1)
                        {
                            s_warp_staging[ptx::warp_id()] = inclusive;
                        }

                        // Only RADIX_COUNT threads participate; use barrier_cta_sync to avoid full block sync overhead.
                        ptx::barrier_cta_sync<2, RADIX_COUNT>(1);

                        // -1 because no warp needs the last warp's staging value for prefix sum.
                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int w = 0; w < RADIX_COUNT / WARP_SIZE - 1; ++w)
                        {
                            if (w < ptx::warp_id()) { exclusive += s_warp_staging[w]; }
                        }

                        // Store negative so Phase 9 can compute global_offset = s_tile_prefix[digit] + index with a single add.
                        // Cast to OffsetT before negation to ensure correct 64-bit arithmetic when OffsetT is uint64_t.
                        s_tile_prefix[threadIdx.x] = static_cast<OffsetT>(0) - static_cast<OffsetT>(exclusive);

                        // Convert per-warp digit counts to tile-wide base offsets so each warp ranks its items after previous warps' items.
                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int w = 0; w < NUM_WARPS - 1; ++w)
                        {
                            unsigned int count = (s_mem.warp_histogram + w * RADIX_COUNT)[threadIdx.x];
                            (s_mem.warp_histogram + w * RADIX_COUNT)[threadIdx.x] = exclusive;

                            exclusive += count;
                        }

                        // Last warp just needs the final offset; no count to accumulate since nothing follows it.
                        (s_mem.warp_histogram + (NUM_WARPS - 1) * RADIX_COUNT)[threadIdx.x] = exclusive;
                    }
                }

                // Ensure per-warp base offsets in warp_histogram are ready before ranking phase reads them.
                __syncthreads();

                // PHASE 6: Rank each key within its digit bucket.
                // Each key needs a unique memory offset for Phase 7 scatter; group lanes by digit and assign consecutive offsets.
                {
                    unsigned int *s_warp_histogram = s_mem.warp_histogram + ptx::warp_id() * RADIX_COUNT;

                    CUSORT_PRAGMA_UNROLL_FULL
                    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                    {
                        const unsigned int r        = i / ITEMS_PER_REGISTER;
                        const unsigned int s        = i % ITEMS_PER_REGISTER;
                        const unsigned int shift    = s * (sizeof(KeyT) * 8);
                        const unsigned int digit    = static_cast<unsigned int>(bit_ordered_keys[r] >> shift) & digit_mask;

                        // Bitmask of lanes with same digit; used to assign consecutive ranks within each digit group.
                        const unsigned int peers    = ptx::match_any_sync<RADIX_BITS>(~0u, digit);

                        // Count peers with lower lane ID to assign unique rank (0,1,2...) within each digit group for scatter positioning.
                        const unsigned int rank     = __popc(peers & ptx::lanemask_lt());

                        // Only lowest-ranked lane per digit group does atomic; others get result via shuffle. Reduces contention.
                        unsigned int offset; if (rank == 0) { offset = atomicAdd(&s_warp_histogram[digit], __popc(peers)); }

                        // Broadcast base offset from leader (lowest lane in peer group) to all peers; each adds rank to get unique position.
                        offset = __shfl_sync(~0u, offset, __ffs(peers) - 1) + rank;

                        // Pack two 16-bit offsets per register to reduce register pressure and improve occupancy.
                        if ((i & 1) == 0) offsets[i >> 1] = offset; else offsets[i >> 1] += offset * (1u << 16);
                    }
                }

                // Ranking reads warp_histogram; scatter writes to scattered_keys in same union. Must finish reading before overwriting.
                __syncthreads();

                // PHASE 7: Scatter keys to shared memory by rank.
                // Staging in shared memory groups keys by digit, enabling sequential reads in Phase 9 for coalesced global writes.
                {
                    // Undo rotation (but not twiddle) so Phase 9 can extract digit via >> current_bit; twiddle reversed at output time.
                    {
                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int r = 0; r < REGISTERS_PER_THREAD; ++r)
                        {
                            bit_ordered_keys[r] = ptx::rotate_left(bit_ordered_keys[r], current_bit);
                        }
                    }

                    {
                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                        {
                            const unsigned int r        = i / ITEMS_PER_REGISTER;
                            const unsigned int s        = i % ITEMS_PER_REGISTER;
                            const unsigned int shift    = s * (sizeof(KeyT) * 8);
                            const unsigned int offset   = ((i & 1) == 0) ? (offsets[i >> 1] & ((1u << 16) - 1)) : (offsets[i >> 1] >> 16);

                            // Can't write directly to global - each thread's keys scatter to non-contiguous locations (poor coalescing).
                            // Shared memory scatter has no coalescing penalty; then Phase 9 reads sequentially for coalesced global writes.
                            // Shift extracts the key (not digit) from packed register for sub-32-bit key types.
                            s_mem.scattered_keys[offset] = static_cast<UnsignedKeyT>(bit_ordered_keys[r] >> shift);
                        }
                    }
                }

                // PHASE 8: Decoupled lookback to compute global prefix.
                // Tiles publish partial counts early (Phase 3) and complete later; allows overlap with local work instead of blocking.
                {
                    if (threadIdx.x < RADIX_COUNT)
                    {
                        // Lookback reads d_descriptors[-1] as terminating prefix; histogram must finish writing it first. Later passes already synced in Phase 1.
                        if (pass == 0) { ptx::grid_dependents_sync(); }

                        // Decoupled lookback computes global prefix from previous tiles and marks this tile complete for successors.
                        OffsetT prefix_sum = LookbackComputePrefixAndPublishComplete<OffsetT, RADIX_COUNT>(d_descriptors + blockIdx.x * RADIX_COUNT + threadIdx.x, digit_count, pass);

                        // Add to negative exclusive sum stored earlier; result is (global_prefix - tile_exclusive) so Phase 9 index arithmetic works.
                        atomicAdd(&s_tile_prefix[threadIdx.x], prefix_sum);
                    }
                }

                // Ensure global prefix sums are in s_tile_prefix before scatter to global memory.
                __syncthreads();

                // PHASE 9: Write keys to global memory.
                // Sequential reads from shared memory (keys grouped by digit) yield coalesced writes to global (consecutive threads write consecutive addresses).
                {
                    if (static_cast<OffsetT>(blockIdx.x) * TILE_ITEMS + TILE_ITEMS > num_items)
                    {
                        const unsigned int limit = static_cast<unsigned int>(num_items - static_cast<OffsetT>(blockIdx.x) * TILE_ITEMS);

                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                        {
                            const unsigned int index        = threadIdx.x + i * BLOCK_THREADS; if (index >= limit) break;
                            const UnsignedRegT key_bits     = static_cast<UnsignedRegT>(s_mem.scattered_keys[index]);
                            const unsigned int digit        = (static_cast<unsigned int>(key_bits >> current_bit)) & digit_mask;

                            // Keys in shared memory are grouped by digit. Example with digits 0,1,2 having counts [100,150,50]:
                            //   - Digit 0 at indices 0-99, digit 1 at indices 100-249, digit 2 at indices 250-299
                            //   - Tile-exclusive prefix: [0, 100, 250]. Global prefix from lookback: [1000, 5000, 8000]
                            //   - Key with digit 1 at index 150: global_offset = 5000 + (150 - 100) = 5050
                            //   - s_tile_prefix[digit] = global_prefix - tile_exclusive, so: s_tile_prefix[1] + 150 = (5000-100) + 150 = 5050
                            const OffsetT global_offset     = s_tile_prefix[digit] + index;

                            // Reuse bit_ordered_keys registers to save digits needed in Phase 11 (because s_mem will be overwritten by values).
                            if constexpr (!KEYS_ONLY)
                            {
                                const unsigned int r        = i / ITEMS_PER_REGISTER;
                                const unsigned int s        = i % ITEMS_PER_REGISTER;
                                const unsigned int shift    = s * (sizeof(KeyT) * 8);

                                if (shift == 0)
                                {
                                    // Large keys (32/64-bit): save global_offset directly. Small keys (8/16-bit): save digit, recompute offset in Phase 11.
                                    bit_ordered_keys[r] = sizeof(UnsignedKeyT) >= sizeof(OffsetT) ? static_cast<UnsignedRegT>(global_offset) : static_cast<UnsignedRegT>(digit);
                                }
                                else
                                {
                                    bit_ordered_keys[r] += static_cast<UnsignedRegT>(digit) << shift;
                                }
                            }

                            ptx::st_global<STORE_OP>(d_keys_out + global_offset, RadixTraits::Decode(static_cast<UnsignedKeyT>(RadixTraits::TwiddleReverse(key_bits))));
                        }
                    }
                    else
                    {
                        CUSORT_PRAGMA_UNROLL_FULL
                        for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                        {
                            const unsigned int index        = threadIdx.x + i * BLOCK_THREADS;
                            const UnsignedRegT key_bits     = static_cast<UnsignedRegT>(s_mem.scattered_keys[index]);
                            const unsigned int digit        = (static_cast<unsigned int>(key_bits >> current_bit)) & digit_mask;
                            const OffsetT global_offset     = s_tile_prefix[digit] + index;

                            if constexpr (!KEYS_ONLY)
                            {
                                const unsigned int r        = i / ITEMS_PER_REGISTER;
                                const unsigned int s        = i % ITEMS_PER_REGISTER;
                                const unsigned int shift    = s * (sizeof(KeyT) * 8);

                                if (shift == 0)
                                {
                                    bit_ordered_keys[r] = sizeof(UnsignedKeyT) >= sizeof(OffsetT) ? static_cast<UnsignedRegT>(global_offset) : static_cast<UnsignedRegT>(digit);
                                }
                                else
                                {
                                    bit_ordered_keys[r] += static_cast<UnsignedRegT>(digit) << shift;
                                }
                            }

                            ptx::st_global<STORE_OP>(d_keys_out + global_offset, RadixTraits::Decode(static_cast<UnsignedKeyT>(RadixTraits::TwiddleReverse(key_bits))));

                            // Prevent compiler from reordering writes across iterations; keeps writes in order for better coalescing.
                            __syncwarp(~0u);
                        }
                    }
                }

                if constexpr (!KEYS_ONLY)
                {
                    // Ensure keys are written before reusing s_mem union for values scatter.
                    __syncthreads();

                    // PHASE 10: Load values and scatter to shared memory.
                    // Reuse offsets[] from Phase 6 - values need same permutation as keys, no need to recompute ranks.
                    {
                        // Must load values in same warp-striped order as keys in Phase 1, so offsets[] correctly pairs each value with its key.
                        OffsetT thread_offset = static_cast<OffsetT>(blockIdx.x) * TILE_ITEMS + static_cast<OffsetT>(ptx::warp_id()) * WARP_ITEMS + static_cast<OffsetT>(ptx::lane_id());

                        // Advance pointer for simpler indexing and use max() to prevent unsigned underflow when thread is beyond array end.
                        d_values_in += thread_offset; const OffsetT remaining = ::max(num_items, thread_offset) - thread_offset;

                        // Use cp.async.cg for 16-byte values on sm_80+ to overlap memory operations with compute.
                        // Limited to 16 bytes because cp.async.cg (L2-only, bypasses L1) is only available for 16-byte transfers.
                        // Smaller sizes only support cp.async.ca which pollutes L1 cache, hurting overall performance.
                        constexpr bool USE_ASYNC_COPY = (sizeof(ValueT) == 16) && (CUSORT_DEVICE_ARCH >= 800) && !CUSORT_DISABLE_ASYNC_COPY;

                        if (remaining <= (ITEMS_PER_THREAD - 1) * WARP_SIZE)
                        {
                            CUSORT_PRAGMA_UNROLL_FULL
                            for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                            {
                                if (remaining <= i * WARP_SIZE) break;

                                // Reuse offsets[] computed in Phase 6 - values follow same permutation as their keys.
                                const unsigned int offset = ((i & 1) == 0) ? (offsets[i >> 1] & ((1u << 16) - 1)) : (offsets[i >> 1] >> 16);

                                if constexpr (USE_ASYNC_COPY)
                                {
                                    ptx::cp_async_cg_16(&s_mem.scattered_values[offset], d_values_in + i * WARP_SIZE);
                                }
                                else
                                {
                                    s_mem.scattered_values[offset] = ptx::ld_global<LOAD_OP>(d_values_in + i * WARP_SIZE);
                                }
                            }
                        }
                        else
                        {
                            CUSORT_PRAGMA_UNROLL_FULL
                            for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                            {
                                const unsigned int offset = ((i & 1) == 0) ? (offsets[i >> 1] & ((1u << 16) - 1)) : (offsets[i >> 1] >> 16);

                                if constexpr (USE_ASYNC_COPY)
                                {
                                    ptx::cp_async_cg_16(&s_mem.scattered_values[offset], d_values_in + i * WARP_SIZE);
                                }
                                else
                                {
                                    s_mem.scattered_values[offset] = ptx::ld_global<LOAD_OP>(d_values_in + i * WARP_SIZE);
                                }
                            }
                        }

                        // Wait for all in-flight async copies to complete before syncthreads.
                        // wait_group<0> only guarantees data is visible to the current thread, not all threads;
                        // must call before __syncthreads() so each thread's writes are complete before the barrier.
                        if constexpr (USE_ASYNC_COPY) { ptx::cp_async_wait_group<0>(); }
                    }

                    // Ensure all values are scattered to shared memory before writing to global.
                    __syncthreads();

                    // PHASE 11: Write values to global memory.
                    // Same coalescing strategy as Phase 9 - sequential shared memory reads yield coalesced global writes.
                    {
                        if (static_cast<OffsetT>(blockIdx.x) * TILE_ITEMS + TILE_ITEMS > num_items)
                        {
                            const unsigned int limit = static_cast<unsigned int>(num_items - static_cast<OffsetT>(blockIdx.x) * TILE_ITEMS);

                            CUSORT_PRAGMA_UNROLL_FULL
                            for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                            {
                                const unsigned int index        = threadIdx.x + i * BLOCK_THREADS; if (index >= limit) break;
                                const unsigned int r            = i / ITEMS_PER_REGISTER;

                                // Reuse global_offset stashed in bit_ordered_keys during Phase 9 (for large keys) to avoid recomputation.
                                OffsetT global_offset;
                                if constexpr (sizeof(UnsignedKeyT) >= sizeof(OffsetT))
                                {
                                    global_offset = static_cast<OffsetT>(bit_ordered_keys[r]);
                                }
                                else
                                {
                                    // Small keys saved digit in Phase 9; extract it and recompute offset using s_tile_prefix.
                                    const unsigned int s        = i % ITEMS_PER_REGISTER;
                                    const unsigned int shift    = s * (sizeof(KeyT) * 8);
                                    const unsigned int digit    = static_cast<unsigned int>(bit_ordered_keys[r] >> shift) & (RADIX_COUNT - 1);

                                    global_offset = s_tile_prefix[digit] + index;
                                }

                                ptx::st_global<STORE_OP>(d_values_out + global_offset, s_mem.scattered_values[index]);
                            }
                        }
                        else
                        {
                            CUSORT_PRAGMA_UNROLL_FULL
                            for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i)
                            {
                                const unsigned int index        = threadIdx.x + i * BLOCK_THREADS;
                                const unsigned int r            = i / ITEMS_PER_REGISTER;

                                OffsetT global_offset;
                                if constexpr (sizeof(UnsignedKeyT) >= sizeof(OffsetT))
                                {
                                    global_offset = static_cast<OffsetT>(bit_ordered_keys[r]);
                                }
                                else
                                {
                                    const unsigned int s        = i % ITEMS_PER_REGISTER;
                                    const unsigned int shift    = s * (sizeof(KeyT) * 8);
                                    const unsigned int digit    = static_cast<unsigned int>(bit_ordered_keys[r] >> shift) & (RADIX_COUNT - 1);

                                    global_offset = s_tile_prefix[digit] + index;
                                }

                                ptx::st_global<STORE_OP>(d_values_out + global_offset, s_mem.scattered_values[index]);

                                // Prevent compiler from reordering writes across iterations; keeps writes in order for better coalescing.
                                __syncwarp(~0u);
                            }
                        }
                    }
                }

                // Launch next radix pass early; it can begin zeroing histograms (Phase 0) while this pass finishes writing.
                ptx::grid_launch_dependents();
            }
        } // namespace kernels

        // Auto-tuning parameters determined empirically on RTX 5090. Each row in the tuning
        // table specifies optimal (BLOCKS_PER_SM, BLOCK_THREADS, ITEMS_PER_THREAD) for a
        // given (key_size, value_size, offset_size) combination. The tradeoffs are:
        // - More BLOCKS_PER_SM: Better latency hiding but more shared memory pressure
        // - More BLOCK_THREADS: Better occupancy but more synchronization overhead
        // - More ITEMS_PER_THREAD: Better instruction-level parallelism but more registers
        // These values balance these factors for the specific memory access patterns of
        // radix sort on modern GPUs. Different GPUs may benefit from different tuning.
        namespace tuning
        {
            struct OnesweepTuningEntry
            {
                int key_bytes, value_bytes, offset_bytes, BLOCKS_PER_MULTIPROCESSOR, BLOCK_THREADS, ITEMS_PER_THREAD;
            };

            constexpr OnesweepTuningEntry ONESWEEP_TUNING_TABLE[] =
            {
                // Key   Value  Offset  CTA   BT   IPT
                {   1,     0,     4,     3,   256,  44 },
                {   1,     0,     8,     3,   256,  44 },
                {   2,     0,     4,     2,   512,  26 },
                {   2,     0,     8,     2,   512,  26 },
                {   4,     0,     4,     2,   512,  21 },
                {   4,     0,     8,     2,   512,  21 },
                {   8,     0,     4,     1,   384,  11 },
                {   8,     0,     8,     1,   384,  11 },
                {   1,     1,     4,     2,   512,  28 },
                {   1,     1,     8,     2,   512,  28 },
                {   1,     2,     4,     2,   384,  32 },
                {   1,     2,     8,     2,   384,  32 },
                {   1,     4,     4,     1,   384,  28 },
                {   1,     4,     8,     1,   384,  28 },
                {   1,     8,     4,     2,   384,  12 },
                {   1,     8,     8,     2,   384,  12 },
                {   1,    16,     4,     2,   512,   4 },
                {   1,    16,     8,     2,   512,   4 },
                {   2,     1,     4,     2,   384,  32 },
                {   2,     1,     8,     2,   384,  32 },
                {   2,     2,     4,     1,   512,  38 },
                {   2,     2,     8,     1,   512,  38 },
                {   2,     4,     4,     1,   384,  22 },
                {   2,     4,     8,     1,   384,  22 },
                {   2,     8,     4,     2,   512,   6 },
                {   2,     8,     8,     2,   512,   6 },
                {   2,    16,     4,     2,   384,   6 },
                {   2,    16,     8,     2,   384,   6 },
                {   4,     1,     4,     1,   512,  22 },
                {   4,     1,     8,     1,   512,  22 },
                {   4,     2,     4,     1,   512,  22 },
                {   4,     2,     8,     1,   512,  22 },
                {   4,     4,     4,     1,   512,  22 },
                {   4,     4,     8,     1,   512,  22 },
                {   4,     8,     4,     1,   384,  15 },
                {   4,     8,     8,     1,   384,  15 },
                {   4,    16,     4,     2,   384,   6 },
                {   4,    16,     8,     2,   384,   6 },
                {   8,     1,     4,     1,   512,  11 },
                {   8,     1,     8,     1,   512,  11 },
                {   8,     2,     4,     1,   512,  11 },
                {   8,     2,     8,     1,   512,  11 },
                {   8,     4,     4,     1,   256,  22 },
                {   8,     4,     8,     1,   256,  22 },
                {   8,     8,     4,     1,   512,  11 },
                {   8,     8,     8,     1,   512,  11 },
                {   8,    16,     4,     2,   384,   6 },
                {   8,    16,     8,     2,   384,   6 },
            };

            constexpr OnesweepTuningEntry get_onesweep_parameters(int key_bytes, int value_bytes, int offset_bytes)
            {
                for (const auto& e : ONESWEEP_TUNING_TABLE)
                {
                    if (e.key_bytes == key_bytes && e.value_bytes == value_bytes && e.offset_bytes == offset_bytes) return e;
                }

                return {};
            }

            // Base configuration shared by all radix sort policies. The 8-bit radix (256 buckets)
            // is optimal for GPU: large enough to reduce passes (4 for 32-bit, 8 for 64-bit),
            // small enough that per-digit histograms fit in shared memory without bank conflicts.
            struct RadixSortBase
            {
                static constexpr int WARP_SIZE                              = 32;
                static constexpr int RADIX_BITS                             = 8;
                static constexpr int RADIX_COUNT                            = 1 << RADIX_BITS;
            };

            // Tuning parameters for the histogram kernel. OVERSUBSCRIPTION > 1 launches more
            // blocks than can run concurrently. The histogram kernel is memory-bound (streaming
            // keys from global memory). Oversubscription helps with tail latency: when blocks
            // finish histogram computation and write back counts, other blocks can continue
            // reading keys, keeping the memory system fully utilized.
            template <int _BLOCKS_PER_MULTIPROCESSOR, int _BLOCK_THREADS, int _ITEMS_PER_THREAD, int _OVERSUBSCRIPTION = 2, ptx::ld_global_cache_op _LOAD_OP = ptx::ld_global_cache_op::cs>
            struct Histogram : RadixSortBase
            {
                static constexpr int BLOCKS_PER_MULTIPROCESSOR              = _BLOCKS_PER_MULTIPROCESSOR;
                static constexpr int BLOCK_THREADS                          = _BLOCK_THREADS;
                static constexpr int ITEMS_PER_THREAD                       = _ITEMS_PER_THREAD;
                static constexpr int OVERSUBSCRIPTION                       = _OVERSUBSCRIPTION;

                static constexpr ptx::ld_global_cache_op LOAD_OP            = _LOAD_OP;
            };

            // Tuning parameters for the onesweep kernel. Lower BLOCKS_PER_SM than histogram
            // because onesweep is more memory-intensive (scatter to global) and register-heavy
            // (stores keys, offsets, and digit info). The LOAD_OP/STORE_OP control cache behavior.
            template <int _BLOCKS_PER_MULTIPROCESSOR, int _BLOCK_THREADS, int _ITEMS_PER_THREAD, ptx::ld_global_cache_op _LOAD_OP = ptx::ld_global_cache_op::cs, ptx::st_global_cache_op _STORE_OP = ptx::st_global_cache_op::wb>
            struct Onesweep : RadixSortBase
            {
                static constexpr int BLOCKS_PER_MULTIPROCESSOR              = _BLOCKS_PER_MULTIPROCESSOR;
                static constexpr int BLOCK_THREADS                          = _BLOCK_THREADS;
                static constexpr int ITEMS_PER_THREAD                       = _ITEMS_PER_THREAD;

                static constexpr ptx::ld_global_cache_op LOAD_OP            = _LOAD_OP;
                static constexpr ptx::st_global_cache_op STORE_OP           = _STORE_OP;
            };

            // Combines histogram and onesweep policies into a complete radix sort configuration.
            // The two kernels have different optimal parameters, so they're tuned separately.
            template <typename HistogramPolicyT, typename OnesweepPolicyT>
            struct RadixSort : RadixSortBase
            {
                using HistogramPolicy = HistogramPolicyT;
                using OnesweepPolicy  = OnesweepPolicyT;
            };
        } // namespace tuning

        // Internal dispatch layer that handles buffer management and kernel launching.
        // Key responsibilities:
        // 1) Calculate temporary storage requirements (histograms + descriptors + optional staging)
        // 2) Set up ping-pong buffers, handling the case where user wants in-place sort
        //    (simulated via staging buffers since true in-place radix sort is impractical)
        // 3) Launch histogram kernel once, then onesweep kernel for each radix pass
        // 4) Manage CUDA graph caching for reduced launch overhead on repeated sorts
        template <bool IS_DESCENDING, typename KeyT, typename ValueT, typename OffsetT>
        struct DispatchRadixSort
        {
            // Core kernel launch logic. Handles both CUDA Graph path (cached, low-overhead replay)
            // and traditional path (individual kernel launches). The graph path builds a DAG of
            // memset -> histogram -> onesweep[0] -> onesweep[1] -> ... with programmatic dependencies.
            template <typename Policy>
            static cudaError_t Invoke(
                void *d_temp_storage,
                size_t &temp_storage_bytes,
                DoubleBuffer<KeyT> &d_keys,
                DoubleBuffer<ValueT> &d_values,
                OffsetT num_items,
                unsigned int begin_bit,
                unsigned int end_bit,
                cudaStream_t stream,
                bool needs_staging)
            {
                // Cached to avoid repeated cudaOccupancyMaxActiveBlocksPerMultiprocessor calls (~10us each).
                // Static ensures the query runs once per Policy instantiation, not per sort call.
                static int histogram_max_resident_blocks = 0;

                using HistogramPolicy = typename Policy::HistogramPolicy;
                using OnesweepPolicy  = typename Policy::OnesweepPolicy;

                constexpr OffsetT HISTOGRAM_TILE_ITEMS  = HistogramPolicy::BLOCK_THREADS * HistogramPolicy::ITEMS_PER_THREAD;
                constexpr OffsetT ONESWEEP_TILE_ITEMS   = OnesweepPolicy::BLOCK_THREADS * OnesweepPolicy::ITEMS_PER_THREAD;

                constexpr bool KEYS_ONLY                = std::is_same_v<ValueT, NullType>;
                constexpr bool STAGE_VALUES_FIRST       = (!KEYS_ONLY) && (sizeof(ValueT) > sizeof(KeyT));

                // Number of radix passes determined by key bit range. Each pass processes RADIX_BITS (8).
                // partial_digit_mask handles the final pass when remaining bits < RADIX_BITS.
                unsigned int num_bits                   = end_bit - begin_bit;
                unsigned int num_passes                 = (num_bits + Policy::RADIX_BITS - 1) / Policy::RADIX_BITS;
                unsigned int partial_digit_mask         = (1u << (num_bits - (num_passes - 1) * Policy::RADIX_BITS)) - 1u;

                const OffsetT histogram_tiles           = (num_items + HISTOGRAM_TILE_ITEMS - 1) / HISTOGRAM_TILE_ITEMS;
                const OffsetT onesweep_num_tiles        = (num_items + ONESWEEP_TILE_ITEMS - 1) / ONESWEEP_TILE_ITEMS;

                // Staging only matters for multi-pass sorts. Single-pass always lands in correct buffer.
                needs_staging = needs_staging && (num_passes >= 2);

                // Temp storage layout: [histogram counts][tile descriptors][staging larger?][staging smaller?]
                // Histogram: one count per digit (256) per pass. Descriptors: one per tile per digit for lookback.
                // Staging buffers needed when user specifies in/out pointers (not DoubleBuffer API) to ensure
                // output lands in the user's output buffer regardless of pass count parity.
                // Larger type (keys or values) is staged first for natural alignment of both buffers.
                const size_t histogram_bytes            = static_cast<size_t>(num_passes) * Policy::RADIX_COUNT * sizeof(OffsetT);
                const size_t descriptor_bytes           = static_cast<size_t>(onesweep_num_tiles) * Policy::RADIX_COUNT * sizeof(OffsetT);
                const size_t staging_keys_bytes         = (needs_staging) ? static_cast<size_t>(num_items) * sizeof(KeyT) : 0;
                const size_t staging_values_bytes       = (needs_staging && !KEYS_ONLY) ? static_cast<size_t>(num_items) * sizeof(ValueT) : 0;
                const size_t storage_bytes              = histogram_bytes + descriptor_bytes + staging_keys_bytes + staging_values_bytes;

                // Two-phase API pattern: first call queries required size, second call executes.
                // This lets caller allocate exact amount needed without over-provisioning.
                if (d_temp_storage == nullptr)
                {
                    temp_storage_bytes = storage_bytes;
                    return cudaSuccess;
                }

                const auto histogram_kernel             = kernels::RadixSortHistogramKernel<HistogramPolicy, IS_DESCENDING, KeyT, OffsetT>;
                const auto onesweep_kernel              = kernels::RadixSortOnesweepKernel<OnesweepPolicy, IS_DESCENDING, KeyT, ValueT, OffsetT>;

                // Query max resident blocks only once per Policy type. This determines grid sizing
                // to fully occupy the GPU without exceeding occupancy limits.
                if (histogram_max_resident_blocks == 0)
                {
                    if (cudaError_t error = utils::MaxResidentBlocks(histogram_max_resident_blocks, histogram_kernel, HistogramPolicy::BLOCK_THREADS)) return error;
                }

                const unsigned int histogram_grid_size  = static_cast<unsigned int>(std::min(static_cast<OffsetT>(histogram_max_resident_blocks * HistogramPolicy::OVERSUBSCRIPTION), histogram_tiles));
                const unsigned int onesweep_grid_size   = static_cast<unsigned int>(onesweep_num_tiles);

                KeyT   *d_keys_out                      = d_keys.d_buffers[d_keys.selector ^ 1];
                ValueT *d_values_out                    = d_values.d_buffers[d_values.selector ^ 1];
                KeyT   *d_keys_staging                  = reinterpret_cast<KeyT *>(reinterpret_cast<char *>(d_temp_storage) + histogram_bytes + descriptor_bytes + (STAGE_VALUES_FIRST ? staging_values_bytes : 0));
                ValueT *d_values_staging                = reinterpret_cast<ValueT *>(reinterpret_cast<char *>(d_temp_storage) + histogram_bytes + descriptor_bytes + (STAGE_VALUES_FIRST ? 0 : staging_keys_bytes));
                OffsetT *d_histograms                   = reinterpret_cast<OffsetT *>(d_temp_storage);

                // For in/out pointer API, we simulate ping-pong with staging buffers. The selector
                // manipulation ensures the final pass writes to the user's output buffer.
                // Pass parity determines which buffer gets staged: odd passes end in alternate buffer.
                if (needs_staging)
                {
                    d_keys.d_buffers[d_keys.selector ^ 1] = (num_passes % 2 == 1) ? d_keys_out : d_keys_staging;
                    d_values.d_buffers[d_values.selector ^ 1] = (num_passes % 2 == 1) ? d_values_out : d_values_staging;
                }

                // TRADITIONAL PATH: Direct kernel launches without CUDA Graphs.
                // Simpler but higher CPU overhead (~15us per launch vs ~5us for graph replay).
                // Used when CUDA Graphs are disabled or unavailable.
                #if CUSORT_DISABLE_CUDA_GRAPHS_AND_PDL
                {
                    // Zero all temp storage in one call. Histogram counts and descriptors must start at zero.
                    // Descriptors use STATUS_INVALID (which is 0 for pass 0) as initial state.
                    if (cudaError_t error = CUSORT_CUDA_CALL(cudaMemsetAsync(d_temp_storage, 0, storage_bytes, stream))) return error;

                    histogram_kernel<<<histogram_grid_size, HistogramPolicy::BLOCK_THREADS, 0, stream>>>(
                        d_keys.d_buffers[d_keys.selector],
                        d_histograms,
                        num_items,
                        begin_bit,
                        partial_digit_mask,
                        num_passes);

                    // Each pass sorts by one radix digit (8 bits), reading from current buffer and
                    // writing to alternate buffer. Selector XOR swaps buffer roles after each pass.
                    for (unsigned int pass = 0; pass < num_passes; ++pass)
                    {
                        unsigned int current_bit    = begin_bit + pass * Policy::RADIX_BITS;
                        unsigned int digit_mask     = (1u << std::min(static_cast<unsigned int>(Policy::RADIX_BITS), end_bit - current_bit)) - 1u;
                        OffsetT *d_descriptors      = d_histograms + (num_passes - pass) * Policy::RADIX_COUNT;

                        if (pass == 1 && needs_staging)
                        {
                            d_keys.d_buffers[d_keys.selector ^ 1] = (num_passes % 2 == 0) ? d_keys_out : d_keys_staging;
                            d_values.d_buffers[d_values.selector ^ 1] = (num_passes % 2 == 0) ? d_values_out : d_values_staging;
                        }

                        onesweep_kernel<<<onesweep_grid_size, OnesweepPolicy::BLOCK_THREADS, 0, stream>>>(
                            d_keys.d_buffers[d_keys.selector],
                            d_keys.d_buffers[d_keys.selector ^ 1],
                            d_values.d_buffers[d_values.selector],
                            d_values.d_buffers[d_values.selector ^ 1],
                            d_descriptors,
                            num_items,
                            current_bit,
                            digit_mask,
                            pass);

                        d_keys.selector ^= 1; d_values.selector ^= 1;
                    }

                    return cudaSuccess;
                }
                // CUDA GRAPH PATH: Build graph once, replay with updated parameters.
                // Graph construction is expensive (~100us) but replay is fast (~5us).
                // Thread-local cache avoids rebuilding when grid dimensions match previous call.
                #else
                {
                    static thread_local GraphCache cache;

                    // Cache invalidation: graph structure depends on grid dimensions. If num_passes,
                    // histogram_grid_size, or onesweep_grid_size changed, we must rebuild the graph.
                    // This happens when sorting different-sized arrays or switching key types.
                    if ((cache.cached_num_passes != num_passes) || (cache.cached_histogram_grid_size != histogram_grid_size) || (cache.cached_onesweep_grid_size != onesweep_grid_size))
                    {
                        cache.destroy();

                        if (cudaError_t error = CUSORT_CUDA_CALL(cudaGraphCreate(&cache.graph, 0))) return error;

                        KeyT *dummy_keys = nullptr; ValueT *dummy_values = nullptr; OffsetT *dummy_offsets = nullptr;
                        OffsetT dummy_num_items = 0; unsigned int dummy_uint = 0;
                        void *histogram_args[] = { &dummy_keys, &dummy_offsets, &dummy_num_items, &dummy_uint, &dummy_uint, &dummy_uint };
                        void *onesweep_args[] = { &dummy_keys, &dummy_keys, &dummy_values, &dummy_values, &dummy_offsets, &dummy_num_items, &dummy_uint, &dummy_uint, &dummy_uint };

                        cudaMemsetParams memset_params = {};
                        memset_params.dst = d_temp_storage;
                        memset_params.width = storage_bytes;
                        memset_params.height = 1;
                        memset_params.elementSize = 1;

                        cudaKernelNodeParams histogram_params = {};
                        histogram_params.func = reinterpret_cast<void *>(histogram_kernel);
                        histogram_params.gridDim = dim3(histogram_grid_size);
                        histogram_params.blockDim = dim3(HistogramPolicy::BLOCK_THREADS);
                        histogram_params.sharedMemBytes = 0;
                        histogram_params.kernelParams = histogram_args;

                        cudaKernelNodeParams onesweep_params = {};
                        onesweep_params.func = reinterpret_cast<void *>(onesweep_kernel);
                        onesweep_params.gridDim = dim3(onesweep_grid_size);
                        onesweep_params.blockDim = dim3(OnesweepPolicy::BLOCK_THREADS);
                        onesweep_params.kernelParams = onesweep_args;

                        // Programmatic Dependent Launch (PDL) requires PTX 9.0+ and SM 9.0+ (Hopper).
                        // PDL allows GPU-side kernel chaining: histogram signals onesweep[0] can start
                        // without waiting for CPU. When unavailable, fall back to default dependencies.
                        int ptx_version = 0;
                        if (cudaError_t error = utils::PtxVersion(ptx_version)) { cache.destroy(); return error; }

                        int sm_version = 0;
                        if (cudaError_t error = utils::SmVersion(sm_version)) { cache.destroy(); return error; }

                        cudaGraphEdgeData edge_data = {};
                        edge_data.type = static_cast<unsigned char>((std::min(ptx_version, sm_version) >= 900) ? cudaGraphDependencyTypeProgrammatic : cudaGraphDependencyTypeDefault);
                        edge_data.from_port = static_cast<unsigned char>((std::min(ptx_version, sm_version) >= 900) ? cudaGraphKernelNodePortProgrammatic : cudaGraphKernelNodePortDefault);

                        // Build graph DAG: memset -> histogram -> onesweep[0] -> onesweep[1] -> ...
                        // Each onesweep depends on its predecessor. With PDL, dependencies use
                        // cudaGraphDependencyTypeProgrammatic for GPU-side signaling.
                        if (cudaError_t error = CUSORT_CUDA_CALL(cudaGraphAddMemsetNode(&cache.memset_node, cache.graph, nullptr, 0, &memset_params))) { cache.destroy(); return error; }
                        if (cudaError_t error = CUSORT_CUDA_CALL(cudaGraphAddKernelNode(&cache.histogram_node, cache.graph, &cache.memset_node, 1, &histogram_params))) { cache.destroy(); return error; }

                        cudaGraphNode_t prev_node = cache.histogram_node;
                        for (unsigned int pass = 0; pass < num_passes; ++pass)
                        {
                            if (cudaError_t error = CUSORT_CUDA_CALL(cudaGraphAddKernelNode(&cache.onesweep_nodes[pass], cache.graph, nullptr, 0, &onesweep_params))) { cache.destroy(); return error; }
                            if (cudaError_t error = CUSORT_CUDA_CALL(cudaGraphAddDependencies(cache.graph, &prev_node, &cache.onesweep_nodes[pass], &edge_data, 1))) { cache.destroy(); return error; }
                            prev_node = cache.onesweep_nodes[pass];
                        }

                        if (cudaError_t error = CUSORT_CUDA_CALL(cudaGraphInstantiate(&cache.graph_exec, cache.graph))) { cache.destroy(); return error; }

                        cache.cached_num_passes             = num_passes;
                        cache.cached_histogram_grid_size    = histogram_grid_size;
                        cache.cached_onesweep_grid_size     = onesweep_grid_size;
                    }

                    // Graph replay: update parameters without rebuilding graph structure.
                    // Only pointers and sizes change between calls; topology stays the same.
                    {
                        cudaMemsetParams memset_params      = {};
                        memset_params.dst                   = d_temp_storage;
                        memset_params.value                 = 0;
                        memset_params.pitch                 = 0;
                        memset_params.elementSize           = 1;
                        memset_params.width                 = storage_bytes;
                        memset_params.height                = 1;

                        if (cudaError_t error = CUSORT_CUDA_CALL(cudaGraphExecMemsetNodeSetParams(cache.graph_exec, cache.memset_node, &memset_params))) return error;
                    }

                    {
                        void *args[] = { &d_keys.d_buffers[d_keys.selector], &d_histograms, &num_items, &begin_bit, &partial_digit_mask, &num_passes };

                        cudaKernelNodeParams kernel_params  = {};
                        kernel_params.func                  = reinterpret_cast<void *>(histogram_kernel);
                        kernel_params.gridDim               = dim3(histogram_grid_size);
                        kernel_params.blockDim              = dim3(HistogramPolicy::BLOCK_THREADS);
                        kernel_params.sharedMemBytes        = 0;
                        kernel_params.kernelParams          = args;

                        if (cudaError_t error = CUSORT_CUDA_CALL(cudaGraphExecKernelNodeSetParams(cache.graph_exec, cache.histogram_node, &kernel_params))) return error;
                    }

                    // Update each onesweep node's parameters: input/output buffers, descriptor pointer,
                    // current bit position, digit mask. Graph structure unchanged, only data pointers differ.
                    for (unsigned int pass = 0; pass < num_passes; ++pass)
                    {
                        unsigned int current_bit            = begin_bit + pass * Policy::RADIX_BITS;
                        unsigned int digit_mask             = (1u << std::min(static_cast<unsigned int>(Policy::RADIX_BITS), end_bit - current_bit)) - 1u;
                        OffsetT *d_descriptors              = d_histograms + (num_passes - pass) * Policy::RADIX_COUNT;

                        if (pass == 1 && needs_staging)
                        {
                            d_keys.d_buffers[d_keys.selector ^ 1] = (num_passes % 2 == 0) ? d_keys_out : d_keys_staging;
                            d_values.d_buffers[d_values.selector ^ 1] = (num_passes % 2 == 0) ? d_values_out : d_values_staging;
                        }

                        void *args[] =
                        {
                            &d_keys.d_buffers[d_keys.selector],
                            &d_keys.d_buffers[d_keys.selector ^ 1],
                            &d_values.d_buffers[d_values.selector],
                            &d_values.d_buffers[d_values.selector ^ 1],
                            &d_descriptors,
                            &num_items,
                            &current_bit,
                            &digit_mask,
                            &pass
                        };

                        cudaKernelNodeParams kernel_params  = {};
                        kernel_params.func                  = reinterpret_cast<void *>(onesweep_kernel);
                        kernel_params.gridDim               = dim3(onesweep_grid_size);
                        kernel_params.blockDim              = dim3(OnesweepPolicy::BLOCK_THREADS);
                        kernel_params.kernelParams          = args;

                        if (cudaError_t error = CUSORT_CUDA_CALL(cudaGraphExecKernelNodeSetParams(cache.graph_exec, cache.onesweep_nodes[pass], &kernel_params))) return error;

                        d_keys.selector ^= 1; d_values.selector ^= 1;
                    }

                    // Single API call launches entire sort pipeline. GPU executes memset -> histogram ->
                    // onesweep passes with minimal CPU involvement. PDL allows passes to overlap at boundaries.
                    if (cudaError_t error = CUSORT_CUDA_CALL(cudaGraphLaunch(cache.graph_exec, stream))) return error;

                    return cudaSuccess;
                }
                #endif
            }

            // Entry point that selects tuning parameters based on key/value/offset sizes.
            // Uses compile-time policy selection via constexpr lookup into the tuning table.
            // Histogram policy is tuned per-architecture (SM 7.5 uses fewer blocks due to
            // smaller shared memory). Onesweep policy comes from empirical tuning tables.
            static cudaError_t Dispatch(
                void *d_temp_storage,
                size_t &temp_storage_bytes,
                DoubleBuffer<KeyT> &d_keys,
                DoubleBuffer<ValueT> &d_values,
                OffsetT num_items,
                unsigned int begin_bit,
                unsigned int end_bit,
                cudaStream_t stream,
                bool needs_staging)
            {
                static_assert(std::is_arithmetic_v<KeyT> && (sizeof(KeyT) == 1 || sizeof(KeyT) == 2 || sizeof(KeyT) == 4 || sizeof(KeyT) == 8), "KeyT must be an arithmetic type of 1, 2, 4, or 8 bytes");
                static_assert(std::is_same_v<ValueT, NullType> || (sizeof(ValueT) == 1 || sizeof(ValueT) == 2 || sizeof(ValueT) == 4 || sizeof(ValueT) == 8 || sizeof(ValueT) == 16), "ValueT must be NullType or a type of 1, 2, 4, 8, or 16 bytes");
                static_assert(std::is_unsigned_v<OffsetT> && (sizeof(OffsetT) == 4 || sizeof(OffsetT) == 8), "OffsetT must be an unsigned 4 or 8 byte type");

                static constexpr int  KEY_SIZE          = static_cast<int>(sizeof(KeyT));
                static constexpr int  VALUE_SIZE        = !std::is_same_v<ValueT, NullType> ? static_cast<int>(sizeof(ValueT)) : 0;
                static constexpr int  OFFSET_SIZE       = static_cast<int>(sizeof(OffsetT));
                static constexpr auto ONESWEEP_PARAMS   = tuning::get_onesweep_parameters(KEY_SIZE, VALUE_SIZE, OFFSET_SIZE);

                // SM 7.5 (Turing) has smaller shared memory (64KB vs 96KB+) and 1024 threads/block limit,
                // constraining occupancy to 2 CTAs/SM with 512-thread blocks.
                // ITEMS_PER_THREAD scales inversely with key size to balance register pressure:
                // 8-byte keys need more registers per key, so fewer items; 1-byte keys pack 4 per register.
                using HistogramPolicy = tuning::Histogram<(CUSORT_DEVICE_ARCH == 750 ? 2 : 3), 512, (KEY_SIZE == 8) ? 11 : (KEY_SIZE == 4) ? 23 : 44>;
                using OnesweepPolicy  = tuning::Onesweep<ONESWEEP_PARAMS.BLOCKS_PER_MULTIPROCESSOR, ONESWEEP_PARAMS.BLOCK_THREADS, ONESWEEP_PARAMS.ITEMS_PER_THREAD>;
                using RadixSortPolicy = tuning::RadixSort<HistogramPolicy, OnesweepPolicy>;

                return Invoke<RadixSortPolicy>(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, needs_staging);
            }
        };
    } // namespace internal

    // Public API for GPU radix sort. Modeled after CUB's DeviceRadixSort for familiarity.
    //
    // Usage pattern (two-phase):
    //   1) Query temp storage: pass d_temp_storage=nullptr, get required size in temp_storage_bytes
    //   2) Execute sort: allocate temp storage, call again with valid pointer
    //
    // Buffer variants:
    //   - DoubleBuffer API: Caller provides two buffers, output may be in either (check selector)
    //   - In/Out pointer API: Caller specifies exact input and output locations
    //
    // Key types: int8, int16, int32, int64, uint8, uint16, uint32, uint64, float, double
    // Value types: Any 1/2/4/8/16-byte type, or keys-only sort (omit values parameter)
    struct DeviceRadixSort
    {
    private:
        // Internal dispatcher that handles edge cases and type mapping. Responsibilities:
        // 1) Validate bit range (begin_bit <= end_bit within key size)
        // 2) Short-circuit for trivial cases (empty, single-element, or zero-bit sort)
        // 3) Map user's ValueT to an unsigned type of same size (for uniform handling)
        // 4) Select 32-bit or 64-bit OffsetT based on array size (32-bit for <=1B elements)
        // 5) Handle staging requirement for in/out pointer API (needs_staging=true)
        template <bool IS_DESCENDING, typename KeyT, typename ValueT, typename NumItemsT>
        static cudaError_t DispatchSort(
            void *d_temp_storage,
            size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            DoubleBuffer<ValueT> &d_values,
            NumItemsT num_items,
            int begin_bit,
            int end_bit,
            cudaStream_t stream,
            bool needs_staging)
        {
            if (begin_bit < 0 || end_bit < 0 || begin_bit > end_bit || end_bit > static_cast<int>(sizeof(KeyT) * 8)) return cudaErrorInvalidValue;

            // Early exit for trivial cases: 0-1 items or zero-width bit range need no sorting.
            // However, with in/out pointer API (needs_staging), caller expects output in d_keys_out.
            // We must still copy input->output. The two-phase API requires temp_storage_bytes > 0
            // to signal a second call is needed, so we return 1 (unused) to trigger the copy path.
            if (num_items <= 1 || begin_bit == end_bit)
            {
                if (d_temp_storage == nullptr)
                {
                    temp_storage_bytes = (needs_staging && num_items > 0) ? 1 : 0;
                }
                else if (needs_staging && num_items > 0)
                {
                    if (cudaError_t error = CUSORT_CUDA_CALL(cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector ^ 1], d_keys.d_buffers[d_keys.selector], static_cast<size_t>(num_items) * sizeof(KeyT), cudaMemcpyDeviceToDevice, stream))) return error;

                    if constexpr (!std::is_same_v<ValueT, internal::NullType>)
                    {
                        if (cudaError_t error = CUSORT_CUDA_CALL(cudaMemcpyAsync(d_values.d_buffers[d_values.selector ^ 1], d_values.d_buffers[d_values.selector], static_cast<size_t>(num_items) * sizeof(ValueT), cudaMemcpyDeviceToDevice, stream))) return error;
                    }
                }

                return cudaSuccess;
            }

            // Use unsigned long long (not uint64_t) for 64-bit offset to match CUDA's atomicAdd overload.
            // On Linux, uint64_t is 'unsigned long' which doesn't match atomicAdd(unsigned long long*, ...).
            using OffsetT32 = unsigned int;
            using OffsetT64 = unsigned long long;

            // Map user's ValueT to a fixed unsigned type of the same size. Values are opaque payload -
            // we only move bytes, not interpret them. Using canonical types (uint8/16/32/64, ulonglong2)
            // reduces template instantiations: instead of one kernel per user type, we get one per size.
            // This cuts compile time and binary size dramatically for codebases with many value types.
            using MappedValueT =
                std::conditional_t<std::is_same_v<ValueT, internal::NullType>, internal::NullType,
                std::conditional_t<sizeof(ValueT) == 1, uint8_t,
                std::conditional_t<sizeof(ValueT) == 2, uint16_t,
                std::conditional_t<sizeof(ValueT) == 4, uint32_t,
                std::conditional_t<sizeof(ValueT) == 8, uint64_t, ulonglong2>>>>>;

            DoubleBuffer<MappedValueT> d_mapped_values(
                reinterpret_cast<MappedValueT *>(d_values.d_buffers[0]),
                reinterpret_cast<MappedValueT *>(d_values.d_buffers[1]));

            d_mapped_values.selector = d_values.selector;

            // Runtime selection: use 32-bit offsets for arrays up to 2^30 elements (~1 billion).
            // The 2^30 threshold (not 2^32) because decoupled lookback reserves 2 high bits for
            // status flags in the descriptor, leaving only 30 bits for the count value.
            cudaError_t error = num_items <= static_cast<NumItemsT>(1u << 30)
                ? internal::DispatchRadixSort<IS_DESCENDING, KeyT, MappedValueT, OffsetT32>::Dispatch(d_temp_storage, temp_storage_bytes, d_keys, d_mapped_values, static_cast<OffsetT32>(num_items), static_cast<unsigned int>(begin_bit), static_cast<unsigned int>(end_bit), stream, needs_staging)
                : internal::DispatchRadixSort<IS_DESCENDING, KeyT, MappedValueT, OffsetT64>::Dispatch(d_temp_storage, temp_storage_bytes, d_keys, d_mapped_values, static_cast<OffsetT64>(num_items), static_cast<unsigned int>(begin_bit), static_cast<unsigned int>(end_bit), stream, needs_staging);

            d_values.selector = d_mapped_values.selector;

            return error;
        }

    public:
        // Sort key-value pairs in ascending order using DoubleBuffer.
        // Output location depends on number of radix passes; check d_keys.selector after call.
        // begin_bit/end_bit allow sorting on a subset of key bits (e.g., for packed structs).
        template <typename KeyT, typename ValueT, typename NumItemsT>
        static cudaError_t SortPairs(
            void *d_temp_storage,
            size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            DoubleBuffer<ValueT> &d_values,
            NumItemsT num_items,
            int begin_bit = 0,
            int end_bit = sizeof(KeyT) * 8,
            cudaStream_t stream = 0)
        {
            return DispatchSort<false>(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, false);
        }

        // Sort key-value pairs in descending order using DoubleBuffer.
        // Same interface as SortPairs but produces largest-to-smallest ordering.
        template <typename KeyT, typename ValueT, typename NumItemsT>
        static cudaError_t SortPairsDescending(
            void *d_temp_storage,
            size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            DoubleBuffer<ValueT> &d_values,
            NumItemsT num_items,
            int begin_bit = 0,
            int end_bit = sizeof(KeyT) * 8,
            cudaStream_t stream = 0)
        {
            return DispatchSort<true>(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, false);
        }

        // Sort keys only (no associated values) in ascending order using DoubleBuffer.
        // More efficient than SortPairs when values aren't needed - saves memory bandwidth.
        template <typename KeyT, typename NumItemsT>
        static cudaError_t SortKeys(
            void *d_temp_storage,
            size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            NumItemsT num_items,
            int begin_bit = 0,
            int end_bit = sizeof(KeyT) * 8,
            cudaStream_t stream = 0)
        {
            DoubleBuffer<internal::NullType> d_values;
            return DispatchSort<false>(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, false);
        }

        // Sort keys only in descending order using DoubleBuffer.
        template <typename KeyT, typename NumItemsT>
        static cudaError_t SortKeysDescending(
            void *d_temp_storage,
            size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            NumItemsT num_items,
            int begin_bit = 0,
            int end_bit = sizeof(KeyT) * 8,
            cudaStream_t stream = 0)
        {
            DoubleBuffer<internal::NullType> d_values;
            return DispatchSort<true>(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, false);
        }

        // Sort keys with explicit input/output buffers. Output always goes to d_keys_out.
        // Requires additional staging memory internally to simulate in-place operation.
        template <typename KeyT, typename NumItemsT>
        static cudaError_t SortKeys(
            void *d_temp_storage,
            size_t &temp_storage_bytes,
            const KeyT *d_keys_in,
            KeyT *d_keys_out,
            NumItemsT num_items,
            int begin_bit = 0,
            int end_bit = sizeof(KeyT) * 8,
            cudaStream_t stream = 0)
        {
            DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
            DoubleBuffer<internal::NullType> d_values;
            return DispatchSort<false>(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, true);
        }

        // Sort keys descending with explicit input/output buffers.
        template <typename KeyT, typename NumItemsT>
        static cudaError_t SortKeysDescending(
            void *d_temp_storage,
            size_t &temp_storage_bytes,
            const KeyT *d_keys_in,
            KeyT *d_keys_out,
            NumItemsT num_items,
            int begin_bit = 0,
            int end_bit = sizeof(KeyT) * 8,
            cudaStream_t stream = 0)
        {
            DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
            DoubleBuffer<internal::NullType> d_values;
            return DispatchSort<true>(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, true);
        }

        // Sort key-value pairs with explicit input/output buffers. Keys and values are
        // permuted together so d_values_out[i] corresponds to d_keys_out[i] after sort.
        template <typename KeyT, typename ValueT, typename NumItemsT>
        static cudaError_t SortPairs(
            void *d_temp_storage,
            size_t &temp_storage_bytes,
            const KeyT *d_keys_in,
            KeyT *d_keys_out,
            const ValueT *d_values_in,
            ValueT *d_values_out,
            NumItemsT num_items,
            int begin_bit = 0,
            int end_bit = sizeof(KeyT) * 8,
            cudaStream_t stream = 0)
        {
            DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
            DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in), d_values_out);
            return DispatchSort<false>(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, true);
        }

        // Sort key-value pairs descending with explicit input/output buffers.
        template <typename KeyT, typename ValueT, typename NumItemsT>
        static cudaError_t SortPairsDescending(
            void *d_temp_storage,
            size_t &temp_storage_bytes,
            const KeyT *d_keys_in,
            KeyT *d_keys_out,
            const ValueT *d_values_in,
            ValueT *d_values_out,
            NumItemsT num_items,
            int begin_bit = 0,
            int end_bit = sizeof(KeyT) * 8,
            cudaStream_t stream = 0)
        {
            DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
            DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in), d_values_out);
            return DispatchSort<true>(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, begin_bit, end_bit, stream, true);
        }
    };
} // namespace cusort

#endif
