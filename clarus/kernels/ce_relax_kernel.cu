// Metric-aware CE CUDA kernels.
// Phase 1: inference-only custom ops for sparse relax + codebook pull.
// Riemannian geometry: FDT-consistent noise, scale-invariant portal,
// bypass energy integration, Woodbury G^{-1/2} via SVD.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <tuple>
#include <vector>

__global__ void csr_spmv_kernel(
    const float* __restrict__ values,
    const int* __restrict__ col_idx,
    const int* __restrict__ row_ptr,
    const float* __restrict__ x,
    float* __restrict__ out,
    const int dim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) {
        return;
    }

    float acc = 0.0f;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    for (int idx = start; idx < end; ++idx) {
        acc += values[idx] * x[col_idx[idx]];
    }
    out[row] = acc;
}

torch::Tensor csr_spmv_cuda(
    torch::Tensor values,
    torch::Tensor col_idx,
    torch::Tensor row_ptr,
    torch::Tensor x)
{
    auto dim = static_cast<int>(x.numel());
    auto out = torch::zeros_like(x);

    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    csr_spmv_kernel<<<blocks, threads>>>(
        values.data_ptr<float>(),
        col_idx.data_ptr<int>(),
        row_ptr.data_ptr<int>(),
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        dim
    );
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> ce_codebook_pull_cuda_fwd(
    torch::Tensor m,
    torch::Tensor codebook,
    float beta,
    float cb_w)
{
    if (codebook.numel() == 0) {
        return {torch::zeros_like(m), torch::zeros({}, m.options())};
    }

    float beta_safe = std::max(beta, 1e-6f);
    auto logits = beta * torch::matmul(codebook, m);
    auto probs = torch::softmax(logits, 0);
    auto grad = -cb_w * torch::matmul(probs.unsqueeze(0), codebook).squeeze(0);
    auto energy = -(cb_w / beta_safe) * torch::logsumexp(logits, 0);
    return {grad, energy};
}

std::tuple<torch::Tensor, torch::Tensor> natural_direction_cuda(
    torch::Tensor grad,
    torch::Tensor phi,
    torch::Tensor recent_var,
    torch::Tensor metric_basis,
    float lambda0,
    float lambda_phi,
    float lambda_var)
{
    auto diag = (lambda0 + lambda_phi * torch::pow(phi, 2) + lambda_var * recent_var).clamp_min(1e-4);
    auto inv_diag = diag.reciprocal();
    auto inv_diag_grad = grad * inv_diag;

    if (metric_basis.numel() == 0 || metric_basis.size(0) == 0) {
        return {inv_diag_grad, diag};
    }

    auto weighted_basis = metric_basis * inv_diag.unsqueeze(0);
    auto small = torch::eye(metric_basis.size(0), grad.options()) +
                 torch::matmul(metric_basis, weighted_basis.transpose(0, 1));
    auto rhs = torch::matmul(metric_basis, inv_diag_grad).unsqueeze(1);
    auto tmp = std::get<0>(torch::linalg::solve_ex(small, rhs)).squeeze(1);
    auto correction = torch::matmul(metric_basis.transpose(0, 1), tmp);
    auto direction = inv_diag_grad - correction * inv_diag;
    return {direction, diag};
}

torch::Tensor fdt_noise_cuda(
    torch::Tensor z,
    torch::Tensor phi,
    torch::Tensor recent_var,
    torch::Tensor metric_basis,
    float lambda0,
    float lambda_phi,
    float lambda_var)
{
    auto diag = (lambda0 + lambda_phi * torch::pow(phi, 2) + lambda_var * recent_var).clamp_min(1e-4);
    auto inv_sqrt_diag = torch::rsqrt(diag);

    if (metric_basis.numel() == 0 || metric_basis.size(0) == 0) {
        return z * inv_sqrt_diag;
    }

    auto Q = metric_basis * inv_sqrt_diag.unsqueeze(0);
    auto svd_result = torch::linalg::svd(Q, /*full_matrices=*/false);
    auto s_q = std::get<1>(svd_result);
    auto Vh_q = std::get<2>(svd_result);

    auto factors = 1.0 - 1.0 / torch::sqrt(1.0 + torch::pow(s_q, 2));
    auto proj = torch::matmul(Vh_q, z);
    auto corrected = z - torch::matmul(Vh_q.transpose(0, 1), factors * proj);
    return inv_sqrt_diag * corrected;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> ce_relax_cuda_fwd(
    torch::Tensor values,
    torch::Tensor col_idx,
    torch::Tensor row_ptr,
    torch::Tensor b,
    torch::Tensor phi,
    torch::Tensor m0,
    torch::Tensor codebook,
    torch::Tensor metric_basis,
    float portal,
    float bypass,
    float t_wake,
    float beta,
    float cb_w,
    float lambda0,
    float lambda_phi,
    float lambda_var,
    float tau,
    float dt,
    int64_t max_steps,
    float tol,
    float anneal_ratio,
    float noise_scale,
    int64_t seed)
{
    if (seed >= 0) {
        torch::manual_seed(seed);
    }

    float scale = std::max(m0.norm().item<float>(), 1e-8f);
    auto m = m0 / scale;
    auto b_n = b / scale;
    auto phi_n = phi / (phi.norm().item<float>() + 1e-8f);
    auto codebook_n = codebook.numel() ? codebook / scale : codebook;
    auto metric_basis_n = metric_basis;

    auto m1 = m.clone();
    auto m2 = m.clone();
    tau = std::max(tau, 1e-6f);
    float dt_eff = std::min(dt, 0.9f * tau);
    int64_t anneal_end = std::max<int64_t>(1, static_cast<int64_t>(std::round(anneal_ratio * max_steps)));
    float t_eff = t_wake / std::max<int64_t>(1, m.numel());

    std::vector<float> hist_e;
    std::vector<float> hist_delta;
    std::vector<float> hist_e_hop;
    std::vector<float> hist_e_bias;
    std::vector<float> hist_e_portal;
    std::vector<float> hist_e_cb;
    std::vector<float> hist_bypass;
    hist_e.reserve(max_steps);
    hist_delta.reserve(max_steps);
    hist_e_hop.reserve(max_steps);
    hist_e_bias.reserve(max_steps);
    hist_e_portal.reserve(max_steps);
    hist_e_cb.reserve(max_steps);
    hist_bypass.reserve(max_steps);

    auto best_m = m.clone();
    float best_e = std::numeric_limits<float>::infinity();

    for (int64_t k = 0; k < max_steps; ++k) {
        float c_k = (m - 2 * m1 + m2).norm().item<float>();
        auto w_m = csr_spmv_cuda(values, col_idx, row_ptr, m);
        auto grad = w_m + b_n + (portal + c_k * bypass) * phi_n;

        if (codebook_n.numel()) {
            auto cb_out = ce_codebook_pull_cuda_fwd(m, codebook_n, beta, cb_w);
            grad = grad + std::get<0>(cb_out);
        }

        auto recent_var = 0.5f * (torch::pow(m - m1, 2) + torch::pow(m1 - m2, 2));
        auto nat_out = natural_direction_cuda(
            grad, phi_n, recent_var, metric_basis_n,
            lambda0, lambda_phi, lambda_var
        );
        auto nat_grad = std::get<0>(nat_out);

        float t_k = (k < anneal_end) ? t_eff * std::max(0.0f, 1.0f - static_cast<float>(k) / anneal_end) : 0.0f;
        float noise_std = std::sqrt(std::max(0.0f, 2.0f * t_k * dt_eff / tau)) * std::max(0.0f, noise_scale);
        torch::Tensor noise;
        if (noise_std > 0.0f) {
            auto z_raw = torch::randn_like(m);
            noise = noise_std * fdt_noise_cuda(
                z_raw, phi_n, recent_var, metric_basis_n,
                lambda0, lambda_phi, lambda_var
            );
        } else {
            noise = torch::zeros_like(m);
        }

        m2 = m1.clone();
        m1 = m.clone();
        auto dm = (dt_eff / tau) * nat_grad + noise;
        m = m + dm;

        auto w_m_new = csr_spmv_cuda(values, col_idx, row_ptr, m);
        auto e_hop = -0.5 * torch::dot(m, w_m_new);
        auto e_bias = -torch::dot(m, b_n);
        auto e_portal = -portal * torch::dot(m, phi_n);
        auto e_bypass = -bypass * c_k * torch::dot(m, phi_n);
        torch::Tensor e_cb;
        if (codebook_n.numel()) {
            auto cb_out = ce_codebook_pull_cuda_fwd(m, codebook_n, beta, cb_w);
            e_cb = std::get<1>(cb_out);
        } else {
            e_cb = torch::zeros({}, m.options());
        }
        auto e_total = e_hop + e_bias + e_portal + e_cb + e_bypass;

        float e_val = e_total.item<float>();
        float delta = dm.norm().item<float>();
        hist_e.push_back(e_val);
        hist_delta.push_back(delta);
        hist_e_hop.push_back(e_hop.item<float>());
        hist_e_bias.push_back(e_bias.item<float>());
        hist_e_portal.push_back(e_portal.item<float>());
        hist_e_cb.push_back(e_cb.item<float>());
        hist_bypass.push_back(c_k);

        if (e_val < best_e) {
            best_e = e_val;
            best_m = m.clone();
        }

        if (k > 30 && delta < tol) {
            break;
        }
    }

    best_m = best_m * scale;
    auto hist_opts = torch::TensorOptions().dtype(torch::kFloat32);
    return {
        best_m,
        torch::tensor(hist_e, hist_opts),
        torch::tensor(hist_delta, hist_opts),
        torch::tensor(hist_e_hop, hist_opts),
        torch::tensor(hist_e_bias, hist_opts),
        torch::tensor(hist_e_portal, hist_opts),
        torch::tensor(hist_e_cb, hist_opts),
        torch::tensor(hist_bypass, hist_opts)
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ce_codebook_pull_fwd", &ce_codebook_pull_cuda_fwd, "CE codebook pull forward (CUDA)");
    m.def("ce_relax_fwd", &ce_relax_cuda_fwd, "Metric-aware CE relax forward (CUDA)");
}
