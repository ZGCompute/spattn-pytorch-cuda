#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void spattn_forward_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    const int B, const int N, const int H, const int D, const float sparsity
) {
    // simplified kernel for illustration
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * H) return;

    int b = idx / (N * H);
    int n = (idx / H) % N;
    int h = idx % H;

    for (int d = 0; d < D; ++d) {
        out[idx * D + d] = 0.0f;
        for (int i = 0; i < N * sparsity; ++i) {
            float score = 0.0f;
            for (int j = 0; j < D; ++j) {
                score += q[(b*N + n)*H*D + h*D + j] * k[(b*N + i)*H*D + h*D + j];
            }
            out[idx * D + d] += score * v[(b*N + i)*H*D + h*D + d];
        }
    }
}

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, float sparsity) {
    const auto B = q.size(0);
    const auto N = q.size(1);
    const auto H = q.size(2);
    const auto D = q.size(3);
    auto out = torch::zeros_like(q);
    const int threads = 256;
    const int blocks = (B * N * H + threads - 1) / threads;

    spattn_forward_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, H, D, sparsity
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SpAttn forward (CUDA)");
}