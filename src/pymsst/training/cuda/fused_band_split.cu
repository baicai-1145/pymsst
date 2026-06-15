#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <vector>

template <typename scalar_t>
__global__ void band_split_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ normed,
    float* __restrict__ inv_norm,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features,
    int64_t x_stride_m,
    int64_t x_stride_g,
    int64_t x_stride_i) {
  extern __shared__ float shared[];
  const int64_t m = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = threadIdx.x;

  float sum_sq = 0.0f;
  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    const float value = static_cast<float>(x[m * x_stride_m + g * x_stride_g + i * x_stride_i]);
    sum_sq += value * value;
  }
  shared[tid] = sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float scale = sqrtf(static_cast<float>(in_features));
  const float norm = fmaxf(sqrtf(shared[0]), 1.0e-12f);
  const float inv = 1.0f / norm;
  if (tid == 0) {
    inv_norm[m * group_count + g] = inv;
  }

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    const float value = static_cast<float>(x[m * x_stride_m + g * x_stride_g + i * x_stride_i]);
    normed[(m * group_count + g) * in_features + i] = static_cast<scalar_t>(value * inv * scale);
  }
  __syncthreads();

  for (int64_t o = tid; o < out_features; o += blockDim.x) {
    float acc = bias == nullptr ? 0.0f : static_cast<float>(bias[g * out_features + o]);
    for (int64_t i = 0; i < in_features; ++i) {
      const float value = static_cast<float>(x[m * x_stride_m + g * x_stride_g + i * x_stride_i]);
      const float feature = value * inv * scale * static_cast<float>(gamma[g * in_features + i]);
      acc += feature * static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    out[(m * group_count + g) * out_features + o] = static_cast<scalar_t>(acc);
  }
}

std::vector<at::Tensor> band_split_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  const c10::cuda::CUDAGuard device_guard(x.device());

  const auto m_count = x.size(0);
  const auto group_count = x.size(1);
  const auto in_features = x.size(2);
  const auto out_features = weight.size(1);

  auto out = at::empty({m_count, group_count, out_features}, x.options());
  auto normed = at::empty({m_count, group_count, in_features}, x.options());
  auto inv_norm = at::empty({m_count, group_count}, x.options().dtype(at::kFloat));

  const int threads = 256;
  const dim3 blocks(m_count, group_count);
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "band_split_forward_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        band_split_forward_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
            x.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_ptr,
            out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            m_count,
            group_count,
            in_features,
            out_features,
            x.stride(0),
            x.stride(1),
            x.stride(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, normed, inv_norm};
}

template <typename scalar_t>
__global__ void band_split_backward_input_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_x,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features) {
  extern __shared__ float shared[];
  const int64_t m = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(in_features));
  const float inv = inv_norm[m * group_count + g];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float grad_scaled = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
      grad_scaled += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
                     static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    const float grad_normed = grad_scaled * static_cast<float>(gamma[g * in_features + i]);
    dot_partial += grad_normed * static_cast<float>(normed[(m * group_count + g) * in_features + i]);
  }

  shared[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float dot = shared[0];

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float grad_scaled = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
      grad_scaled += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
                     static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    const float grad_normed = grad_scaled * static_cast<float>(gamma[g * in_features + i]);
    const float normed_value = static_cast<float>(normed[(m * group_count + g) * in_features + i]);
    const float value = inv * (grad_normed * scale - normed_value * (dot / scale));
    grad_x[(m * group_count + g) * in_features + i] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void band_split_backward_weight_bias_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const scalar_t* __restrict__ gamma,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features) {
  const int64_t g = blockIdx.x;
  const int64_t o = blockIdx.y;
  const int tid = threadIdx.x;

  if (grad_bias != nullptr && tid == 0) {
    float bias_acc = 0.0f;
    for (int64_t m = 0; m < m_count; ++m) {
      bias_acc += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]);
    }
    grad_bias[g * out_features + o] = static_cast<scalar_t>(bias_acc);
  }

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float weight_acc = 0.0f;
    const float gamma_value = static_cast<float>(gamma[g * in_features + i]);
    for (int64_t m = 0; m < m_count; ++m) {
      weight_acc += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
                    static_cast<float>(normed[(m * group_count + g) * in_features + i]) * gamma_value;
    }
    grad_weight[(g * out_features + o) * in_features + i] = static_cast<scalar_t>(weight_acc);
  }
}

template <typename scalar_t>
__global__ void band_split_backward_gamma_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_gamma,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features) {
  const int64_t g = blockIdx.x;
  const int64_t i = blockIdx.y;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t index = tid; index < m_count * out_features; index += blockDim.x) {
    const int64_t m = index / out_features;
    const int64_t o = index % out_features;
    acc += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
           static_cast<float>(weight[(g * out_features + o) * in_features + i]) *
           static_cast<float>(normed[(m * group_count + g) * in_features + i]);
  }

  extern __shared__ float shared[];
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gamma[g * in_features + i] = static_cast<scalar_t>(shared[0]);
  }
}

std::vector<at::Tensor> band_split_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    bool has_bias) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());

  const auto m_count = grad_out.size(0);
  const auto group_count = grad_out.size(1);
  const auto out_features = grad_out.size(2);
  const auto in_features = normed.size(2);

  auto grad_x = at::empty_like(normed);
  auto grad_gamma = at::empty_like(gamma);
  auto grad_weight = at::empty_like(weight);
  at::Tensor grad_bias = has_bias ? at::empty({group_count, out_features}, grad_out.options()) : at::Tensor();

  const int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_bytes = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "band_split_backward_cuda",
      [&] {
        band_split_backward_input_kernel<scalar_t><<<dim3(m_count, group_count), threads, shared_bytes, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            gamma.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_weight_bias_kernel<scalar_t><<<dim3(group_count, out_features), threads, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            has_bias ? grad_bias.data_ptr<scalar_t>() : nullptr,
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_gamma_kernel<scalar_t><<<dim3(group_count, in_features), threads, shared_bytes, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_gamma.data_ptr<scalar_t>(),
            m_count,
            group_count,
            in_features,
            out_features);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma, grad_weight, grad_bias};
}

template <typename scalar_t>
__global__ void band_split_backward_input_v2_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_x,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* grad_normed_cache = shared + blockDim.x;
  const int64_t m = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(in_features));
  const float inv = inv_norm[m * group_count + g];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float grad_scaled = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
      grad_scaled += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
                     static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    const float grad_normed = grad_scaled * static_cast<float>(gamma[g * in_features + i]);
    grad_normed_cache[i] = grad_normed;
    dot_partial += grad_normed * static_cast<float>(normed[(m * group_count + g) * in_features + i]);
  }

  reduction[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  const float dot = reduction[0];

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    const float normed_value = static_cast<float>(normed[(m * group_count + g) * in_features + i]);
    const float value = inv * (grad_normed_cache[i] * scale - normed_value * (dot / scale));
    grad_x[(m * group_count + g) * in_features + i] = static_cast<scalar_t>(value);
  }
}

std::vector<at::Tensor> band_split_backward_v2_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    bool has_bias) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());

  const auto m_count = grad_out.size(0);
  const auto group_count = grad_out.size(1);
  const auto out_features = grad_out.size(2);
  const auto in_features = normed.size(2);

  const int threads = 256;
  const size_t input_shared_bytes = (threads + static_cast<size_t>(in_features)) * sizeof(float);
  if (input_shared_bytes > 48 * 1024) {
    return band_split_backward_cuda(grad_out, normed, inv_norm, gamma, weight, has_bias);
  }

  auto grad_x = at::empty_like(normed);
  auto grad_gamma = at::empty_like(gamma);
  auto grad_weight = at::empty_like(weight);
  at::Tensor grad_bias = has_bias ? at::empty({group_count, out_features}, grad_out.options()) : at::Tensor();

  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_bytes = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "band_split_backward_v2_cuda",
      [&] {
        band_split_backward_input_v2_kernel<scalar_t><<<dim3(m_count, group_count), threads, input_shared_bytes, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            gamma.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_weight_bias_kernel<scalar_t><<<dim3(group_count, out_features), threads, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            has_bias ? grad_bias.data_ptr<scalar_t>() : nullptr,
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_gamma_kernel<scalar_t><<<dim3(group_count, in_features), threads, shared_bytes, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_gamma.data_ptr<scalar_t>(),
            m_count,
            group_count,
            in_features,
            out_features);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma, grad_weight, grad_bias};
}

template <typename scalar_t>
__global__ void band_split_backward_input_v3_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_x,
    float* __restrict__ grad_scaled_cache,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* grad_normed_cache = shared + blockDim.x;
  const int64_t m = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(in_features));
  const float inv = inv_norm[m * group_count + g];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    float grad_scaled = 0.0f;
    for (int64_t o = 0; o < out_features; ++o) {
      grad_scaled += static_cast<float>(grad_out[(m * group_count + g) * out_features + o]) *
                     static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    const float grad_normed = grad_scaled * static_cast<float>(gamma[g * in_features + i]);
    grad_scaled_cache[(m * group_count + g) * in_features + i] = grad_scaled;
    grad_normed_cache[i] = grad_normed;
    dot_partial += grad_normed * static_cast<float>(normed[(m * group_count + g) * in_features + i]);
  }

  reduction[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  const float dot = reduction[0];

  for (int64_t i = tid; i < in_features; i += blockDim.x) {
    const float normed_value = static_cast<float>(normed[(m * group_count + g) * in_features + i]);
    const float value = inv * (grad_normed_cache[i] * scale - normed_value * (dot / scale));
    grad_x[(m * group_count + g) * in_features + i] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void band_split_backward_gamma_from_scaled_kernel(
    const float* __restrict__ grad_scaled_cache,
    const scalar_t* __restrict__ normed,
    scalar_t* __restrict__ grad_gamma,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features) {
  const int64_t g = blockIdx.x;
  const int64_t i = blockIdx.y;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t m = tid; m < m_count; m += blockDim.x) {
    acc += grad_scaled_cache[(m * group_count + g) * in_features + i] *
           static_cast<float>(normed[(m * group_count + g) * in_features + i]);
  }

  extern __shared__ float shared[];
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gamma[g * in_features + i] = static_cast<scalar_t>(shared[0]);
  }
}

std::vector<at::Tensor> band_split_backward_v3_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    bool has_bias) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());

  const auto m_count = grad_out.size(0);
  const auto group_count = grad_out.size(1);
  const auto out_features = grad_out.size(2);
  const auto in_features = normed.size(2);

  const int threads = 256;
  const size_t input_shared_bytes = (threads + static_cast<size_t>(in_features)) * sizeof(float);
  if (input_shared_bytes > 48 * 1024) {
    return band_split_backward_v2_cuda(grad_out, normed, inv_norm, gamma, weight, has_bias);
  }

  auto grad_x = at::empty_like(normed);
  auto grad_gamma = at::empty_like(gamma);
  auto grad_weight = at::empty_like(weight);
  auto grad_scaled_cache = at::empty({m_count, group_count, in_features}, grad_out.options().dtype(at::kFloat));
  at::Tensor grad_bias = has_bias ? at::empty({group_count, out_features}, grad_out.options()) : at::Tensor();

  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shared_bytes = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "band_split_backward_v3_cuda",
      [&] {
        band_split_backward_input_v3_kernel<scalar_t><<<dim3(m_count, group_count), threads, input_shared_bytes, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            gamma.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_scaled_cache.data_ptr<float>(),
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_weight_bias_kernel<scalar_t><<<dim3(group_count, out_features), threads, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            has_bias ? grad_bias.data_ptr<scalar_t>() : nullptr,
            m_count,
            group_count,
            in_features,
            out_features);
        band_split_backward_gamma_from_scaled_kernel<scalar_t>
            <<<dim3(group_count, in_features), threads, shared_bytes, stream>>>(
                grad_scaled_cache.data_ptr<float>(),
                normed.data_ptr<scalar_t>(),
                grad_gamma.data_ptr<scalar_t>(),
                m_count,
                group_count,
                in_features);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma, grad_weight, grad_bias};
}

template <typename scalar_t>
__global__ void grouped_linear_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t m_count,
    int64_t group_count,
    int64_t in_features,
    int64_t out_features,
    int64_t x_stride_m,
    int64_t x_stride_g,
    int64_t x_stride_i) {
  const int64_t m = blockIdx.x;
  const int64_t g = blockIdx.y;
  const int tid = threadIdx.x;

  for (int64_t o = tid; o < out_features; o += blockDim.x) {
    float acc = bias == nullptr ? 0.0f : static_cast<float>(bias[g * out_features + o]);
    for (int64_t i = 0; i < in_features; ++i) {
      acc += static_cast<float>(x[m * x_stride_m + g * x_stride_g + i * x_stride_i]) *
             static_cast<float>(weight[(g * out_features + o) * in_features + i]);
    }
    out[(m * group_count + g) * out_features + o] = static_cast<scalar_t>(acc);
  }
}

at::Tensor grouped_linear_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  const c10::cuda::CUDAGuard device_guard(x.device());

  const auto m_count = x.size(0);
  const auto group_count = x.size(1);
  const auto in_features = x.size(2);
  const auto out_features = weight.size(1);

  auto out = at::empty({m_count, group_count, out_features}, x.options());
  const int threads = 256;
  const dim3 blocks(m_count, group_count);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "grouped_linear_forward_cuda",
      [&] {
        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value()) {
          bias_ptr = bias.value().data_ptr<scalar_t>();
        }
        grouped_linear_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_ptr,
            out.data_ptr<scalar_t>(),
            m_count,
            group_count,
            in_features,
            out_features,
            x.stride(0),
            x.stride(1),
            x.stride(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}

template <typename scalar_t>
__global__ void rotary_forward_kernel(
    const scalar_t* __restrict__ t,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t seq_len,
    int64_t heads,
    int64_t dim) {
  const int64_t pairs = dim / 2;
  const int64_t total = batch * seq_len * heads * pairs;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    const int64_t pair = index % pairs;
    const int64_t head = (index / pairs) % heads;
    const int64_t seq = (index / (pairs * heads)) % seq_len;
    const int64_t batch_index = index / (pairs * heads * seq_len);
    const int64_t base = ((batch_index * seq_len + seq) * heads + head) * dim + pair * 2;
    const float c = static_cast<float>(cos[seq * pairs + pair]);
    const float s = static_cast<float>(sin[seq * pairs + pair]);
    const float even = static_cast<float>(t[base]);
    const float odd = static_cast<float>(t[base + 1]);
    out[base] = static_cast<scalar_t>(even * c - odd * s);
    out[base + 1] = static_cast<scalar_t>(odd * c + even * s);
  }
}

template <typename scalar_t>
__global__ void rotary_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ grad_t,
    int64_t batch,
    int64_t seq_len,
    int64_t heads,
    int64_t dim) {
  const int64_t pairs = dim / 2;
  const int64_t total = batch * seq_len * heads * pairs;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    const int64_t pair = index % pairs;
    const int64_t head = (index / pairs) % heads;
    const int64_t seq = (index / (pairs * heads)) % seq_len;
    const int64_t batch_index = index / (pairs * heads * seq_len);
    const int64_t base = ((batch_index * seq_len + seq) * heads + head) * dim + pair * 2;
    const float c = static_cast<float>(cos[seq * pairs + pair]);
    const float s = static_cast<float>(sin[seq * pairs + pair]);
    const float grad_even = static_cast<float>(grad_out[base]);
    const float grad_odd = static_cast<float>(grad_out[base + 1]);
    grad_t[base] = static_cast<scalar_t>(grad_even * c + grad_odd * s);
    grad_t[base + 1] = static_cast<scalar_t>(-grad_even * s + grad_odd * c);
  }
}

at::Tensor rotary_forward_cuda(
    const at::Tensor& t,
    const at::Tensor& cos,
    const at::Tensor& sin) {
  const c10::cuda::CUDAGuard device_guard(t.device());
  auto t_contiguous = t.contiguous();
  auto cos_contiguous = cos.contiguous();
  auto sin_contiguous = sin.contiguous();
  auto out = at::empty_like(t_contiguous);

  const auto batch = t_contiguous.size(0);
  const auto seq_len = t_contiguous.size(1);
  const auto heads = t_contiguous.size(2);
  const auto dim = t_contiguous.size(3);
  const int threads = 256;
  const int64_t total_pairs = batch * seq_len * heads * (dim / 2);
  const int blocks = static_cast<int>(std::min<int64_t>((total_pairs + threads - 1) / threads, 65535));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      t.scalar_type(),
      "rotary_forward_cuda",
      [&] {
        rotary_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            t_contiguous.data_ptr<scalar_t>(),
            cos_contiguous.data_ptr<scalar_t>(),
            sin_contiguous.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch,
            seq_len,
            heads,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}

at::Tensor rotary_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& cos,
    const at::Tensor& sin) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto cos_contiguous = cos.contiguous();
  auto sin_contiguous = sin.contiguous();
  auto grad_t = at::empty_like(grad_out_contiguous);

  const auto batch = grad_out_contiguous.size(0);
  const auto seq_len = grad_out_contiguous.size(1);
  const auto heads = grad_out_contiguous.size(2);
  const auto dim = grad_out_contiguous.size(3);
  const int threads = 256;
  const int64_t total_pairs = batch * seq_len * heads * (dim / 2);
  const int blocks = static_cast<int>(std::min<int64_t>((total_pairs + threads - 1) / threads, 65535));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "rotary_backward_cuda",
      [&] {
        rotary_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            cos_contiguous.data_ptr<scalar_t>(),
            sin_contiguous.data_ptr<scalar_t>(),
            grad_t.data_ptr<scalar_t>(),
            batch,
            seq_len,
            heads,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return grad_t;
}

template <typename scalar_t>
__global__ void rms_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gamma,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ normed,
    float* __restrict__ inv_norm,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;

  float sum_sq = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float value = static_cast<float>(x[row * dim + i]);
    sum_sq += value * value;
  }
  shared[tid] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float scale = sqrtf(static_cast<float>(dim));
  const float norm = fmaxf(sqrtf(shared[0]), 1.0e-12f);
  const float inv = 1.0f / norm;
  if (tid == 0) {
    inv_norm[row] = inv;
  }

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float normalized = static_cast<float>(x[row * dim + i]) * inv * scale;
    normed[row * dim + i] = static_cast<scalar_t>(normalized);
    out[row * dim + i] = static_cast<scalar_t>(normalized * static_cast<float>(gamma[i]));
  }
}

template <typename scalar_t>
__global__ void rms_norm_backward_input_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const scalar_t* __restrict__ gamma,
    scalar_t* __restrict__ grad_x,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(dim));
  const float inv = inv_norm[row];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = static_cast<float>(grad_out[row * dim + i]) * static_cast<float>(gamma[i]);
    dot_partial += grad_normed * static_cast<float>(normed[row * dim + i]);
  }
  shared[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float dot = shared[0];

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = static_cast<float>(grad_out[row * dim + i]) * static_cast<float>(gamma[i]);
    const float normed_value = static_cast<float>(normed[row * dim + i]);
    const float value = inv * (grad_normed * scale - normed_value * (dot / scale));
    grad_x[row * dim + i] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void rms_norm_backward_gamma_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    scalar_t* __restrict__ grad_gamma,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t i = blockIdx.x;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t row = tid; row < row_count; row += blockDim.x) {
    acc += static_cast<float>(grad_out[row * dim + i]) * static_cast<float>(normed[row * dim + i]);
  }
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gamma[i] = static_cast<scalar_t>(shared[0]);
  }
}

std::vector<at::Tensor> rms_norm_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contiguous = x.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = x_contiguous.size(-1);
  const auto row_count = x_contiguous.numel() / dim;
  auto out = at::empty_like(x_contiguous);
  auto normed = at::empty_like(x_contiguous);
  auto inv_norm = at::empty({row_count}, x_contiguous.options().dtype(at::kFloat));

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "rms_norm_forward_cuda",
      [&] {
        rms_norm_forward_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            x_contiguous.data_ptr<scalar_t>(),
            gamma_contiguous.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            row_count,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, normed, inv_norm};
}

std::vector<at::Tensor> rms_norm_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto normed_contiguous = normed.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = grad_out_contiguous.size(-1);
  const auto row_count = grad_out_contiguous.numel() / dim;
  auto grad_x = at::empty_like(grad_out_contiguous);
  auto grad_gamma = at::empty_like(gamma_contiguous);

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "rms_norm_backward_cuda",
      [&] {
        rms_norm_backward_input_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            normed_contiguous.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            gamma_contiguous.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            row_count,
            dim);
        rms_norm_backward_gamma_kernel<scalar_t><<<dim, threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            normed_contiguous.data_ptr<scalar_t>(),
            grad_gamma.data_ptr<scalar_t>(),
            row_count,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma};
}

template <typename scalar_t>
__global__ void rms_norm_forward_fp32_gamma_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ gamma,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ normed,
    float* __restrict__ inv_norm,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;

  float sum_sq = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float value = static_cast<float>(x[row * dim + i]);
    sum_sq += value * value;
  }
  shared[tid] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float scale = sqrtf(static_cast<float>(dim));
  const float norm = fmaxf(sqrtf(shared[0]), 1.0e-12f);
  const float inv = 1.0f / norm;
  if (tid == 0) {
    inv_norm[row] = inv;
  }

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float normalized = static_cast<float>(x[row * dim + i]) * inv * scale;
    normed[row * dim + i] = static_cast<scalar_t>(normalized);
    out[row * dim + i] = static_cast<scalar_t>(normalized * gamma[i]);
  }
}

template <typename scalar_t>
__global__ void rms_norm_backward_fp32_gamma_input_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const float* __restrict__ gamma,
    scalar_t* __restrict__ grad_x,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(dim));
  const float inv = inv_norm[row];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = static_cast<float>(grad_out[row * dim + i]) * gamma[i];
    dot_partial += grad_normed * static_cast<float>(normed[row * dim + i]);
  }
  shared[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float dot = shared[0];

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = static_cast<float>(grad_out[row * dim + i]) * gamma[i];
    const float normed_value = static_cast<float>(normed[row * dim + i]);
    const float value = inv * (grad_normed * scale - normed_value * (dot / scale));
    grad_x[row * dim + i] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void rms_norm_backward_fp32_gamma_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normed,
    float* __restrict__ grad_gamma,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t i = blockIdx.x;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t row = tid; row < row_count; row += blockDim.x) {
    acc += static_cast<float>(grad_out[row * dim + i]) * static_cast<float>(normed[row * dim + i]);
  }
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gamma[i] = shared[0];
  }
}

std::vector<at::Tensor> rms_norm_forward_fp32_gamma_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contiguous = x.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = x_contiguous.size(-1);
  const auto row_count = x_contiguous.numel() / dim;
  auto out = at::empty_like(x_contiguous);
  auto normed = at::empty_like(x_contiguous);
  auto inv_norm = at::empty({row_count}, x_contiguous.options().dtype(at::kFloat));

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "rms_norm_forward_fp32_gamma_cuda",
      [&] {
        rms_norm_forward_fp32_gamma_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            x_contiguous.data_ptr<scalar_t>(),
            gamma_contiguous.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            normed.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            row_count,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, normed, inv_norm};
}

std::vector<at::Tensor> rms_norm_backward_fp32_gamma_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto normed_contiguous = normed.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = grad_out_contiguous.size(-1);
  const auto row_count = grad_out_contiguous.numel() / dim;
  auto grad_x = at::empty_like(grad_out_contiguous);
  auto grad_gamma = at::empty_like(gamma_contiguous);

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "rms_norm_backward_fp32_gamma_cuda",
      [&] {
        rms_norm_backward_fp32_gamma_input_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            normed_contiguous.data_ptr<scalar_t>(),
            inv_norm.data_ptr<float>(),
            gamma_contiguous.data_ptr<float>(),
            grad_x.data_ptr<scalar_t>(),
            row_count,
            dim);
        rms_norm_backward_fp32_gamma_kernel<scalar_t><<<dim, threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            normed_contiguous.data_ptr<scalar_t>(),
            grad_gamma.data_ptr<float>(),
            row_count,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma};
}

template <typename scalar_t>
__global__ void rms_norm_forward_mixed_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    float* __restrict__ normed,
    float* __restrict__ inv_norm,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;

  float sum_sq = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float value = static_cast<float>(x[row * dim + i]);
    sum_sq += value * value;
  }
  shared[tid] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float scale = sqrtf(static_cast<float>(dim));
  const float norm = fmaxf(sqrtf(shared[0]), 1.0e-12f);
  const float inv = 1.0f / norm;
  if (tid == 0) {
    inv_norm[row] = inv;
  }

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float normalized = static_cast<float>(x[row * dim + i]) * inv * scale;
    normed[row * dim + i] = normalized;
    out[row * dim + i] = normalized * gamma[i];
  }
}

template <typename scalar_t>
__global__ void rms_norm_backward_mixed_input_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ normed,
    const float* __restrict__ inv_norm,
    const float* __restrict__ gamma,
    scalar_t* __restrict__ grad_x,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const float scale = sqrtf(static_cast<float>(dim));
  const float inv = inv_norm[row];

  float dot_partial = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = grad_out[row * dim + i] * gamma[i];
    dot_partial += grad_normed * normed[row * dim + i];
  }
  shared[tid] = dot_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float dot = shared[0];

  for (int64_t i = tid; i < dim; i += blockDim.x) {
    const float grad_normed = grad_out[row * dim + i] * gamma[i];
    const float value = inv * (grad_normed * scale - normed[row * dim + i] * (dot / scale));
    grad_x[row * dim + i] = static_cast<scalar_t>(value);
  }
}

__global__ void rms_norm_backward_mixed_gamma_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ normed,
    float* __restrict__ grad_gamma,
    int64_t row_count,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t i = blockIdx.x;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t row = tid; row < row_count; row += blockDim.x) {
    acc += grad_out[row * dim + i] * normed[row * dim + i];
  }
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gamma[i] = shared[0];
  }
}

std::vector<at::Tensor> rms_norm_forward_mixed_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto x_contiguous = x.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = x_contiguous.size(-1);
  const auto row_count = x_contiguous.numel() / dim;
  auto float_options = x_contiguous.options().dtype(at::kFloat);
  auto out = at::empty(x_contiguous.sizes(), float_options);
  auto normed = at::empty(x_contiguous.sizes(), float_options);
  auto inv_norm = at::empty({row_count}, float_options);

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "rms_norm_forward_mixed_cuda",
      [&] {
        rms_norm_forward_mixed_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            x_contiguous.data_ptr<scalar_t>(),
            gamma_contiguous.data_ptr<float>(),
            out.data_ptr<float>(),
            normed.data_ptr<float>(),
            inv_norm.data_ptr<float>(),
            row_count,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, normed, inv_norm};
}

std::vector<at::Tensor> rms_norm_backward_mixed_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& x_meta) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto normed_contiguous = normed.contiguous();
  auto gamma_contiguous = gamma.contiguous();

  const auto dim = grad_out_contiguous.size(-1);
  const auto row_count = grad_out_contiguous.numel() / dim;
  auto grad_x = at::empty(grad_out_contiguous.sizes(), x_meta.options());
  auto grad_gamma = at::empty_like(gamma_contiguous);

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_meta.scalar_type(),
      "rms_norm_backward_mixed_cuda",
      [&] {
        rms_norm_backward_mixed_input_kernel<scalar_t><<<row_count, threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<float>(),
            normed_contiguous.data_ptr<float>(),
            inv_norm.data_ptr<float>(),
            gamma_contiguous.data_ptr<float>(),
            grad_x.data_ptr<scalar_t>(),
            row_count,
            dim);
      });
  rms_norm_backward_mixed_gamma_kernel<<<dim, threads, shared_bytes, stream>>>(
      grad_out_contiguous.data_ptr<float>(),
      normed_contiguous.data_ptr<float>(),
      grad_gamma.data_ptr<float>(),
      row_count,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_x, grad_gamma};
}

template <typename scalar_t>
__global__ void attention_gate_compute_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ gate,
    int64_t row_count,
    int64_t heads,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t i = tid; i < dim; i += blockDim.x) {
    acc += static_cast<float>(x[row * dim + i]) * static_cast<float>(weight[head * dim + i]);
  }
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float pre = shared[0] + static_cast<float>(bias[head]);
    gate[row * heads + head] = static_cast<scalar_t>(1.0f / (1.0f + expf(-pre)));
  }
}

template <typename scalar_t>
__global__ void attention_gate_apply_kernel(
    const scalar_t* __restrict__ attn,
    const scalar_t* __restrict__ gate,
    scalar_t* __restrict__ out,
    int64_t row_count,
    int64_t heads,
    int64_t dim_head) {
  const int64_t total = row_count * heads * dim_head;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
    const int64_t dim = index % dim_head;
    const int64_t head = (index / dim_head) % heads;
    const int64_t row = index / (dim_head * heads);
    out[row * heads * dim_head + head * dim_head + dim] =
        static_cast<scalar_t>(static_cast<float>(attn[index]) * static_cast<float>(gate[row * heads + head]));
  }
}

template <typename scalar_t>
__global__ void attention_gate_backward_attn_gate_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ attn,
    const scalar_t* __restrict__ gate,
    scalar_t* __restrict__ grad_attn,
    scalar_t* __restrict__ grad_gate_pre,
    int64_t row_count,
    int64_t heads,
    int64_t dim_head) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int tid = threadIdx.x;
  const float gate_value = static_cast<float>(gate[row * heads + head]);

  float gate_grad_partial = 0.0f;
  for (int64_t d = tid; d < dim_head; d += blockDim.x) {
    const int64_t index = row * heads * dim_head + head * dim_head + d;
    const float grad = static_cast<float>(grad_out[index]);
    const float attn_value = static_cast<float>(attn[index]);
    grad_attn[index] = static_cast<scalar_t>(grad * gate_value);
    gate_grad_partial += grad * attn_value;
  }

  shared[tid] = gate_grad_partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_gate_pre[row * heads + head] = static_cast<scalar_t>(shared[0] * gate_value * (1.0f - gate_value));
  }
}

template <typename scalar_t>
__global__ void attention_gate_backward_x_kernel(
    const scalar_t* __restrict__ grad_gate_pre,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_x,
    int64_t row_count,
    int64_t heads,
    int64_t dim) {
  const int64_t row = blockIdx.x;
  const int64_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= dim) {
    return;
  }

  float acc = 0.0f;
  for (int64_t head = 0; head < heads; ++head) {
    acc += static_cast<float>(grad_gate_pre[row * heads + head]) * static_cast<float>(weight[head * dim + i]);
  }
  grad_x[row * dim + i] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void attention_gate_backward_weight_bias_kernel(
    const scalar_t* __restrict__ grad_gate_pre,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,
    int64_t row_count,
    int64_t heads,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t head = blockIdx.x;
  const int64_t i = blockIdx.y;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  for (int64_t row = tid; row < row_count; row += blockDim.x) {
    const float grad = static_cast<float>(grad_gate_pre[row * heads + head]);
    acc += i == dim ? grad : grad * static_cast<float>(x[row * dim + i]);
  }
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (i == dim) {
      grad_bias[head] = static_cast<scalar_t>(shared[0]);
    } else {
      grad_weight[head * dim + i] = static_cast<scalar_t>(shared[0]);
    }
  }
}

std::vector<at::Tensor> attention_gate_forward_cuda(
    const at::Tensor& attn,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  const c10::cuda::CUDAGuard device_guard(attn.device());
  auto attn_contiguous = attn.contiguous();
  auto x_contiguous = x.contiguous();
  auto weight_contiguous = weight.contiguous();
  auto bias_contiguous = bias.contiguous();

  const auto row_count = attn_contiguous.size(0);
  const auto heads = attn_contiguous.size(1);
  const auto dim_head = attn_contiguous.size(2);
  const auto dim = x_contiguous.size(1);

  auto out = at::empty({row_count, heads * dim_head}, attn_contiguous.options());
  auto gate = at::empty({row_count, heads}, attn_contiguous.options());
  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t total = row_count * heads * dim_head;
  const int apply_blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      attn.scalar_type(),
      "attention_gate_forward_cuda",
      [&] {
        attention_gate_compute_kernel<scalar_t><<<dim3(row_count, heads), threads, shared_bytes, stream>>>(
            x_contiguous.data_ptr<scalar_t>(),
            weight_contiguous.data_ptr<scalar_t>(),
            bias_contiguous.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            row_count,
            heads,
            dim);
        attention_gate_apply_kernel<scalar_t><<<apply_blocks, threads, 0, stream>>>(
            attn_contiguous.data_ptr<scalar_t>(),
            gate.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            row_count,
            heads,
            dim_head);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, gate, attn_contiguous, x_contiguous};
}

std::vector<at::Tensor> attention_gate_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& attn,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& gate) {
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_out_contiguous = grad_out.contiguous();
  auto attn_contiguous = attn.contiguous();
  auto x_contiguous = x.contiguous();
  auto weight_contiguous = weight.contiguous();
  auto gate_contiguous = gate.contiguous();

  const auto row_count = attn_contiguous.size(0);
  const auto heads = attn_contiguous.size(1);
  const auto dim_head = attn_contiguous.size(2);
  const auto dim = x_contiguous.size(1);

  auto grad_attn = at::empty_like(attn_contiguous);
  auto grad_gate_pre = at::empty({row_count, heads}, grad_out_contiguous.options());
  auto grad_x = at::empty_like(x_contiguous);
  auto grad_weight = at::empty_like(weight_contiguous);
  auto grad_bias = at::empty({heads}, grad_out_contiguous.options());

  const int threads = 256;
  const size_t shared_bytes = threads * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int dim_blocks = static_cast<int>((dim + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "attention_gate_backward_cuda",
      [&] {
        attention_gate_backward_attn_gate_kernel<scalar_t><<<dim3(row_count, heads), threads, shared_bytes, stream>>>(
            grad_out_contiguous.data_ptr<scalar_t>(),
            attn_contiguous.data_ptr<scalar_t>(),
            gate_contiguous.data_ptr<scalar_t>(),
            grad_attn.data_ptr<scalar_t>(),
            grad_gate_pre.data_ptr<scalar_t>(),
            row_count,
            heads,
            dim_head);
        attention_gate_backward_x_kernel<scalar_t><<<dim3(row_count, dim_blocks), threads, 0, stream>>>(
            grad_gate_pre.data_ptr<scalar_t>(),
            weight_contiguous.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            row_count,
            heads,
            dim);
        attention_gate_backward_weight_bias_kernel<scalar_t><<<dim3(heads, dim + 1), threads, shared_bytes, stream>>>(
            grad_gate_pre.data_ptr<scalar_t>(),
            x_contiguous.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            row_count,
            heads,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_attn, grad_x, grad_weight, grad_bias};
}
