#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <utility>
#include <vector>


#define CUDA_CHECK(call)                                                        
    do {                                                                        
        cudaError_t err__ = (call);                                             
        if (err__ != cudaSuccess) {                                             
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n  in: %s\n",        
                         cudaGetErrorString(err__), __FILE__, __LINE__, #call); 
            std::exit(EXIT_FAILURE);                                            
        }                                                                       
    } while (0)


#define CUDA_CHECK_LAUNCH()                
    do {                                   
        CUDA_CHECK(cudaGetLastError());    
    } while (0)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : n_(n) { CUDA_CHECK(cudaMalloc(&p_, n * sizeof(T))); }
    ~DeviceBuffer() { if (p_) cudaFree(p_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept
        : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (p_) cudaFree(p_);
            p_ = std::exchange(o.p_, nullptr);
            n_ = std::exchange(o.n_, 0);
        }
        return *this;
    }

    T*     get()   const { return p_; }
    size_t size()  const { return n_; }
    size_t bytes() const { return n_ * sizeof(T); }

    void upload(const T* host, size_t count) {
        CUDA_CHECK(cudaMemcpy(p_, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* host, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host, p_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero(cudaStream_t s = 0) { CUDA_CHECK(cudaMemsetAsync(p_, 0, bytes(), s)); }

private:
    T*     p_ = nullptr;
    size_t n_ = 0;
};


template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t n) : n_(n) { CUDA_CHECK(cudaMallocHost(&p_, n * sizeof(T))); }
    ~PinnedBuffer() { if (p_) cudaFreeHost(p_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T*     get()  const { return p_; }
    size_t size() const { return n_; }
    T& operator[](size_t i) { return p_[i]; }
    const T& operator[](size_t i) const { return p_[i]; }

private:
    T*     p_ = nullptr;
    size_t n_ = 0;
};

class GpuTimer {
public:
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&a_)); CUDA_CHECK(cudaEventCreate(&b_)); }
    ~GpuTimer() { cudaEventDestroy(a_); cudaEventDestroy(b_); }
    void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(a_, s)); }
    float stop(cudaStream_t s = 0) {  
        CUDA_CHECK(cudaEventRecord(b_, s));
        CUDA_CHECK(cudaEventSynchronize(b_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a_, b_));
        return ms;
    }
private:
    cudaEvent_t a_{}, b_{};
};

static double wallMs(const std::function<void()>& fn) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static int g_numSMs = 0;


template <typename T>
__device__ __forceinline__ T warpReduceSum(T v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
reduceSumKernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "BLOCK must be a multiple of 32");
    __shared__ float warpSums[BLOCK / 32];

    const size_t tid    = blockIdx.x * (size_t)BLOCK + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * BLOCK;

    float sum = 0.f;


    const size_t n4 = n / 4;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    for (size_t i = tid; i < n4; i += stride) {
        float4 v = in4[i];
        sum += (v.x + v.y) + (v.z + v.w);
    }

    for (size_t i = n4 * 4 + tid; i < n; i += stride) sum += in[i];


    sum = warpReduceSum(sum);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warpSums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < BLOCK / 32) ? warpSums[lane] : 0.f;
        v = warpReduceSum(v);
        if (lane == 0) atomicAdd(out, v);
    }
}
