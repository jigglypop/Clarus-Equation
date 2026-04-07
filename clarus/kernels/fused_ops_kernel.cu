// Fused CUDA kernels for ClarusLM.
// Build via torch.utils.cpp_extension or setup.py.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---- TopK SiLU (element-wise, threshold pre-computed) ----------------------

__global__ void topk_silu_fwd_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint8_t* __restrict__ mask,
    const float threshold,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float s = x / (1.0f + expf(-x));
        bool keep = fabsf(s) >= threshold;
        output[idx] = keep ? s : 0.0f;
        mask[idx] = keep ? 1 : 0;
    }
}

__global__ void topk_silu_bwd_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    const uint8_t* __restrict__ mask,
    float* __restrict__ grad_in,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mask[idx]) {
            float x = input[idx];
            float sig = 1.0f / (1.0f + expf(-x));
            grad_in[idx] = grad_out[idx] * sig * (1.0f + x * (1.0f - sig));
        } else {
            grad_in[idx] = 0.0f;
        }
    }
}

// ---- Fused LBO Norm (layer_norm + conformal + projection) ------------------
// Row-parallel: one block per row, uses shared memory for reduction.

__global__ void lbo_norm_fwd_kernel(
    const float* __restrict__ input,     // [n_rows, dim]
    const float* __restrict__ v_eff,     // [rank, dim]
    const float h,
    const float* __restrict__ scale,     // [dim]
    const float* __restrict__ bias,      // [dim]
    float* __restrict__ output,          // [n_rows, dim]
    float* __restrict__ row_curvature,   // [n_rows]
    const int dim,
    const int rank)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x_row = input + row * dim;
    float* o_row = output + row * dim;

    // Step 1: compute mean
    float local_sum = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) {
        local_sum += x_row[j];
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / dim;

    // Step 2: compute variance
    float local_var = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) {
        float d = x_row[j] - mean;
        local_var += d * d;
    }
    smem[tid] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / dim + 1e-5f);

    // Step 3: layer-normalize into shared memory
    float* normed = smem;  // reuse shared mem [dim]
    for (int j = tid; j < dim; j += blockDim.x) {
        normed[j] = (x_row[j] - mean) * inv_std;
    }
    __syncthreads();

    // Step 4: projection (each thread computes partial proj)
    // proj[i] = sum_j normed[j] * v_eff[i*dim+j]
    float* proj = smem + dim;  // [rank] in shared memory
    for (int i = tid; i < rank; i += blockDim.x) {
        float s = 0.0f;
        for (int j = 0; j < dim; j++) {
            s += normed[j] * v_eff[i * dim + j];
        }
        proj[i] = s;
    }
    __syncthreads();

    // Step 5: output = (normed - h*(normed - xW)) * scale + bias
    float local_curv = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) {
        float xw_j = 0.0f;
        for (int i = 0; i < rank; i++) {
            xw_j += proj[i] * v_eff[i * dim + j];
        }
        float lx = normed[j] - xw_j;
        local_curv += lx * lx;
        o_row[j] = (normed[j] - h * lx) * scale[j] + bias[j];
    }

    // Reduce curvature
    smem[tid] = local_curv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        row_curvature[row] = smem[0] / dim;
    }
}

// ---- C++ wrappers ----------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> topk_silu_cuda_fwd(
    torch::Tensor input,
    float threshold)
{
    auto n = input.numel();
    auto output = torch::empty_like(input);
    auto mask = torch::empty({n}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    topk_silu_fwd_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mask.data_ptr<uint8_t>(),
        threshold,
        n);
    return {output, mask};
}

torch::Tensor topk_silu_cuda_bwd(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor mask)
{
    auto n = grad_out.numel();
    auto grad_in = torch::empty_like(grad_out);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    topk_silu_bwd_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        input.data_ptr<float>(),
        mask.data_ptr<uint8_t>(),
        grad_in.data_ptr<float>(),
        n);
    return grad_in;
}

std::tuple<torch::Tensor, torch::Tensor> lbo_norm_cuda_fwd(
    torch::Tensor input,     // [n_rows, dim]
    torch::Tensor v_eff,     // [rank, dim]
    float h,
    torch::Tensor scale,     // [dim]
    torch::Tensor bias,      // [dim]
    int dim,
    int rank)
{
    auto n_rows = input.size(0);
    auto output = torch::empty_like(input);
    auto row_curv = torch::empty({n_rows}, input.options());

    int threads = std::min(256, dim);
    // round up to power of 2 for reduction
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    int smem_bytes = (dim + rank) * sizeof(float);

    lbo_norm_fwd_kernel<<<n_rows, threads, smem_bytes>>>(
        input.data_ptr<float>(),
        v_eff.data_ptr<float>(),
        h,
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        row_curv.data_ptr<float>(),
        dim,
        rank);
    return {output, row_curv};
}

// ---- pybind11 module -------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_silu_fwd", &topk_silu_cuda_fwd, "Fused TopK SiLU forward (CUDA)");
    m.def("topk_silu_bwd", &topk_silu_cuda_bwd, "Fused TopK SiLU backward (CUDA)");
    m.def("lbo_norm_fwd",  &lbo_norm_cuda_fwd,  "Fused LBO Norm forward (CUDA)");
}
