#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>



// Helper for Warp Reductions
__device__ __forceinline__ float warpSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}


// ============================================================================
// 1. GATHER + RMSNORM KERNELS
// ============================================================================

__global__ void gather_rmsnorm_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float eps,
    int seq_len,
    int hidden_size,
    int num_tokens) {

    int b = blockIdx.y;
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    int seq_idx = indices[b * num_tokens + t];
    const float* row_input = input + ((size_t)b * seq_len + seq_idx) * hidden_size;
    float* row_output = output + (b * num_tokens * hidden_size) + (t * hidden_size);

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        sum_sq += val * val;
    }

    for (int offset = 16; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();
    if (threadIdx.x % 32 == 0) atomicAdd(&shared_sum, sum_sq);
    __syncthreads();

    float inv_rms = rsqrtf(shared_sum / hidden_size + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_output[i] = row_input[i] * inv_rms * weight[i];
    }
}

torch::Tensor gather_rmsnorm_cuda(torch::Tensor input, torch::Tensor indices, torch::Tensor weight, float eps) {
    const int batch = input.size(0);
    const int seq_len = input.size(1);
    const int num_tokens = indices.size(1);
    const int hidden_size = input.size(2);
    auto output = torch::empty({batch, num_tokens, hidden_size}, input.options());

    dim3 grid(num_tokens, batch);
    dim3 block(std::min(hidden_size, 1024));
    gather_rmsnorm_kernel<<<grid, block>>>(
        input.data_ptr<float>(), indices.data_ptr<int64_t>(), weight.data_ptr<float>(),
        output.data_ptr<float>(), eps, seq_len, hidden_size, num_tokens);
    return output;
}

// ============================================================================
// 2. SCATTER + RETURN KERNELS
// ============================================================================

__global__ void scatter_and_return_kernel(
    const float* __restrict__ sparse_values,
    const int64_t* __restrict__ indices,
    float* __restrict__ global_cache,
    float* __restrict__ current_hidden,
    int num_tokens, int seq_len, int hidden_size) {

    int b = blockIdx.y;
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    int seq_idx = indices[b * num_tokens + t];
    int sparse_offset = (b * num_tokens * hidden_size) + (t * hidden_size);
    int full_offset = (b * seq_len * hidden_size) + (seq_idx * hidden_size);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = sparse_values[sparse_offset + i];
        global_cache[full_offset + i] = val;
        current_hidden[full_offset + i] = val;
    }
}

torch::Tensor scatter_and_return_cuda(
    torch::Tensor sparse_values,
    torch::Tensor indices,
    torch::Tensor global_cache,
    torch::Tensor current_hidden) {

    const int batch = sparse_values.size(0);
    const int num_tokens = sparse_values.size(1);
    const int hidden_size = sparse_values.size(2);
    const int seq_len = global_cache.size(1);

    dim3 grid(num_tokens, batch);
    dim3 block(std::min(hidden_size, 1024));
    scatter_and_return_kernel<<<grid, block>>>(
        sparse_values.data_ptr<float>(), indices.data_ptr<int64_t>(),
        global_cache.data_ptr<float>(), current_hidden.data_ptr<float>(),
        num_tokens, seq_len, hidden_size);
    return current_hidden;
}



__global__ void select_gather_rmsnorm_kernel(
    const float* __restrict__ hidden,       // [B, S, H]
    const float* __restrict__ cache_in,     // [B, S, H] (unused; reserved)
    const int64_t* __restrict__ indices,    // [B, T] (Selected Top-K indices)
    const float* __restrict__ weight,       // [H]
    float* __restrict__ gathered_out,       // [B, T, H] or nullptr (skip gather write)
    float* __restrict__ output,             // [B, T, H] normalized
    float eps, int S, int H, int T) {

    int b = blockIdx.y;
    int t = blockIdx.x; // Index in the "Selected" output buffer
    if (t >= T) return;

    // Get the global sequence index from our pre-computed Top-K
    int seq_idx = indices[b * T + t];
    
    // Row pointers (batch stride is S * H, not gridDim.z which is 1 for 2D launches)
    const float* h_row = hidden + ((size_t)b * S + seq_idx) * H;
    float* out_row = output + (b * T * H) + (t * H);
    float* g_row = (gathered_out != nullptr) ? (gathered_out + (b * T * H) + (t * H)) : nullptr;

    // Step 1: Compute Variance (Sum of Squares) for RMSNorm
    // We assume the data is already "selected", so we just Norm it.
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float val = h_row[i];
        sum_sq += val * val;
    }

    // Reduce per-warp partial sums (lane 0 holds warp sum after warpSum).
    sum_sq = warpSum(sum_sq);
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&shared_sum, sum_sq);
    }
    __syncthreads();

    // Step 2: Optional gathered copy + normalized write (single read of h_row).
    // If weight == nullptr, write v * inv_rms only (caller applies weight * x.to(dtype)
    // to match Fast_dLLM_QwenRMSNorm: return self.weight * normalized_fp32.to(input_dtype)).
    float inv_rms = rsqrtf(shared_sum / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = h_row[i];
        if (g_row != nullptr)
            g_row[i] = v;
        float n = v * inv_rms;
        if (weight != nullptr)
            n *= weight[i];
        out_row[i] = n;
    }
}

torch::Tensor fused_select_norm(torch::Tensor hidden, torch::Tensor indices, torch::Tensor weight, float eps) {
    const int B = hidden.size(0);
    const int S = hidden.size(1);
    const int T = indices.size(1);
    const int H = hidden.size(2);
    auto output = torch::empty({B, T, H}, hidden.options());

    dim3 grid(T, B);
    dim3 block(std::min(H, 1024));
    select_gather_rmsnorm_kernel<<<grid, block>>>(
        hidden.data_ptr<float>(), nullptr, indices.data_ptr<int64_t>(),
        weight.data_ptr<float>(), nullptr, output.data_ptr<float>(), eps, S, H, T);
    return output;
}

// Sparse forward: gathered rows + (x * rsqrt) in fp32; Python applies weight * .to(dtype) like QwenRMSNorm.
std::vector<torch::Tensor> fused_gather_input_rmsnorm_pair(
    torch::Tensor hidden, torch::Tensor indices, float eps) {
    const int B = hidden.size(0);
    const int S = hidden.size(1);
    const int T = indices.size(1);
    const int H = hidden.size(2);
    auto gathered = torch::empty({B, T, H}, hidden.options());
    auto normed = torch::empty({B, T, H}, hidden.options());

    dim3 grid(T, B);
    dim3 block(std::min(H, 1024));
    select_gather_rmsnorm_kernel<<<grid, block>>>(
        hidden.data_ptr<float>(), nullptr, indices.data_ptr<int64_t>(),
        nullptr, gathered.data_ptr<float>(), normed.data_ptr<float>(),
        eps, S, H, T);
    return {gathered, normed};
}

// ============================================================================
// 3. BINDINGS
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_rmsnorm_cuda", &gather_rmsnorm_cuda, "Fused Gather and RMSNorm");
    m.def("scatter_and_return_cuda", &scatter_and_return_cuda, "Fused Scatter and Return");
    m.def("fused_select_norm", &fused_select_norm, "Fused Gather + RMSNorm");
    m.def("fused_gather_input_rmsnorm_pair", &fused_gather_input_rmsnorm_pair,
          "Gather + (x*rsqrt) fp32; returns [gathered, normed_pre_weight]; apply weight in Python like QwenRMSNorm");
}