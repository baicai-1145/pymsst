#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> band_split_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias);

std::vector<at::Tensor> band_split_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    bool has_bias);

at::Tensor grouped_linear_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias);

at::Tensor rotary_forward_cuda(
    const at::Tensor& t,
    const at::Tensor& cos,
    const at::Tensor& sin);

at::Tensor rotary_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& cos,
    const at::Tensor& sin);

std::vector<at::Tensor> rms_norm_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma);

std::vector<at::Tensor> rms_norm_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma);

std::vector<at::Tensor> rms_norm_forward_mixed_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma);

std::vector<at::Tensor> rms_norm_backward_mixed_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& x_meta);

std::vector<at::Tensor> attention_gate_forward_cuda(
    const at::Tensor& attn,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias);

std::vector<at::Tensor> attention_gate_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& attn,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& gate);

std::vector<at::Tensor> band_split_forward(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
  }
  TORCH_CHECK(x.dim() == 3, "x must have shape [M, G, I]");
  TORCH_CHECK(gamma.dim() == 2, "gamma must have shape [G, I]");
  TORCH_CHECK(weight.dim() == 3, "weight must have shape [G, O, I]");
  TORCH_CHECK(gamma.size(0) == x.size(1), "gamma group dimension must match x");
  TORCH_CHECK(gamma.size(1) == x.size(2), "gamma input dimension must match x");
  TORCH_CHECK(weight.size(0) == x.size(1), "weight group dimension must match x");
  TORCH_CHECK(weight.size(2) == x.size(2), "weight input dimension must match x");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 2, "bias must have shape [G, O]");
    TORCH_CHECK(bias.value().size(0) == x.size(1), "bias group dimension must match x");
    TORCH_CHECK(bias.value().size(1) == weight.size(1), "bias output dimension must match weight");
  }
  TORCH_CHECK(gamma.scalar_type() == x.scalar_type(), "gamma dtype must match x");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().scalar_type() == x.scalar_type(), "bias dtype must match x");
  }
  return band_split_forward_cuda(x, gamma, weight, bias);
}

std::vector<at::Tensor> band_split_backward(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& weight,
    bool has_bias) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
  TORCH_CHECK(normed.is_cuda(), "normed must be a CUDA tensor");
  TORCH_CHECK(inv_norm.is_cuda(), "inv_norm must be a CUDA tensor");
  TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(grad_out.dim() == 3, "grad_out must have shape [M, G, O]");
  TORCH_CHECK(normed.dim() == 3, "normed must have shape [M, G, I]");
  TORCH_CHECK(inv_norm.dim() == 2, "inv_norm must have shape [M, G]");
  TORCH_CHECK(gamma.dim() == 2, "gamma must have shape [G, I]");
  TORCH_CHECK(weight.dim() == 3, "weight must have shape [G, O, I]");
  TORCH_CHECK(normed.size(0) == grad_out.size(0), "normed M must match grad_out");
  TORCH_CHECK(normed.size(1) == grad_out.size(1), "normed G must match grad_out");
  TORCH_CHECK(inv_norm.size(0) == grad_out.size(0), "inv_norm M must match grad_out");
  TORCH_CHECK(inv_norm.size(1) == grad_out.size(1), "inv_norm G must match grad_out");
  TORCH_CHECK(gamma.size(0) == grad_out.size(1), "gamma G must match grad_out");
  TORCH_CHECK(gamma.size(1) == normed.size(2), "gamma I must match normed");
  TORCH_CHECK(weight.size(0) == grad_out.size(1), "weight G must match grad_out");
  TORCH_CHECK(weight.size(1) == grad_out.size(2), "weight O must match grad_out");
  TORCH_CHECK(weight.size(2) == normed.size(2), "weight I must match normed");
  TORCH_CHECK(normed.scalar_type() == grad_out.scalar_type(), "normed dtype must match grad_out");
  TORCH_CHECK(gamma.scalar_type() == grad_out.scalar_type(), "gamma dtype must match grad_out");
  TORCH_CHECK(weight.scalar_type() == grad_out.scalar_type(), "weight dtype must match grad_out");
  TORCH_CHECK(inv_norm.scalar_type() == at::kFloat, "inv_norm must be float32");
  return band_split_backward_cuda(grad_out, normed, inv_norm, gamma, weight, has_bias);
}

at::Tensor grouped_linear_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
  }
  TORCH_CHECK(x.dim() == 3, "x must have shape [M, G, I]");
  TORCH_CHECK(weight.dim() == 3, "weight must have shape [G, O, I]");
  TORCH_CHECK(weight.size(0) == x.size(1), "weight group dimension must match x");
  TORCH_CHECK(weight.size(2) == x.size(2), "weight input dimension must match x");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 2, "bias must have shape [G, O]");
    TORCH_CHECK(bias.value().size(0) == x.size(1), "bias group dimension must match x");
    TORCH_CHECK(bias.value().size(1) == weight.size(1), "bias output dimension must match weight");
  }
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().scalar_type() == x.scalar_type(), "bias dtype must match x");
  }
  return grouped_linear_forward_cuda(x, weight, bias);
}

at::Tensor rotary_forward(
    const at::Tensor& t,
    const at::Tensor& cos,
    const at::Tensor& sin) {
  TORCH_CHECK(t.is_cuda(), "t must be a CUDA tensor");
  TORCH_CHECK(cos.is_cuda(), "cos must be a CUDA tensor");
  TORCH_CHECK(sin.is_cuda(), "sin must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 4, "t must have shape [B, N, H, D]");
  TORCH_CHECK(cos.dim() == 4, "cos must have shape [1, N, 1, D/2]");
  TORCH_CHECK(sin.dim() == 4, "sin must have shape [1, N, 1, D/2]");
  TORCH_CHECK(t.size(3) % 2 == 0, "t last dimension must be even");
  TORCH_CHECK(cos.size(0) == 1 && cos.size(2) == 1, "cos must broadcast over batch and heads");
  TORCH_CHECK(sin.size(0) == 1 && sin.size(2) == 1, "sin must broadcast over batch and heads");
  TORCH_CHECK(cos.size(1) == t.size(1), "cos sequence dimension must match t");
  TORCH_CHECK(sin.size(1) == t.size(1), "sin sequence dimension must match t");
  TORCH_CHECK(cos.size(3) == t.size(3) / 2, "cos rotary dimension must match t");
  TORCH_CHECK(sin.size(3) == t.size(3) / 2, "sin rotary dimension must match t");
  TORCH_CHECK(cos.scalar_type() == t.scalar_type(), "cos dtype must match t");
  TORCH_CHECK(sin.scalar_type() == t.scalar_type(), "sin dtype must match t");
  return rotary_forward_cuda(t, cos, sin);
}

at::Tensor rotary_backward(
    const at::Tensor& grad_out,
    const at::Tensor& cos,
    const at::Tensor& sin) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
  TORCH_CHECK(cos.is_cuda(), "cos must be a CUDA tensor");
  TORCH_CHECK(sin.is_cuda(), "sin must be a CUDA tensor");
  TORCH_CHECK(grad_out.dim() == 4, "grad_out must have shape [B, N, H, D]");
  TORCH_CHECK(cos.dim() == 4, "cos must have shape [1, N, 1, D/2]");
  TORCH_CHECK(sin.dim() == 4, "sin must have shape [1, N, 1, D/2]");
  TORCH_CHECK(grad_out.size(3) % 2 == 0, "grad_out last dimension must be even");
  TORCH_CHECK(cos.size(0) == 1 && cos.size(2) == 1, "cos must broadcast over batch and heads");
  TORCH_CHECK(sin.size(0) == 1 && sin.size(2) == 1, "sin must broadcast over batch and heads");
  TORCH_CHECK(cos.size(1) == grad_out.size(1), "cos sequence dimension must match grad_out");
  TORCH_CHECK(sin.size(1) == grad_out.size(1), "sin sequence dimension must match grad_out");
  TORCH_CHECK(cos.size(3) == grad_out.size(3) / 2, "cos rotary dimension must match grad_out");
  TORCH_CHECK(sin.size(3) == grad_out.size(3) / 2, "sin rotary dimension must match grad_out");
  TORCH_CHECK(cos.scalar_type() == grad_out.scalar_type(), "cos dtype must match grad_out");
  TORCH_CHECK(sin.scalar_type() == grad_out.scalar_type(), "sin dtype must match grad_out");
  return rotary_backward_cuda(grad_out, cos, sin);
}

std::vector<at::Tensor> rms_norm_forward(
    const at::Tensor& x,
    const at::Tensor& gamma) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
  TORCH_CHECK(gamma.dim() == 1, "gamma must have shape [D]");
  TORCH_CHECK(gamma.size(0) == x.size(-1), "gamma dimension must match x last dimension");
  TORCH_CHECK(gamma.scalar_type() == x.scalar_type(), "gamma dtype must match x");
  return rms_norm_forward_cuda(x, gamma);
}

std::vector<at::Tensor> rms_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
  TORCH_CHECK(normed.is_cuda(), "normed must be a CUDA tensor");
  TORCH_CHECK(inv_norm.is_cuda(), "inv_norm must be a CUDA tensor");
  TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
  TORCH_CHECK(grad_out.dim() >= 2, "grad_out must have at least 2 dimensions");
  TORCH_CHECK(normed.sizes() == grad_out.sizes(), "normed shape must match grad_out");
  TORCH_CHECK(gamma.dim() == 1, "gamma must have shape [D]");
  TORCH_CHECK(gamma.size(0) == grad_out.size(-1), "gamma dimension must match grad_out last dimension");
  TORCH_CHECK(inv_norm.dim() == 1, "inv_norm must have shape [M]");
  TORCH_CHECK(inv_norm.size(0) * gamma.size(0) == grad_out.numel(), "inv_norm length must match flattened rows");
  TORCH_CHECK(normed.scalar_type() == grad_out.scalar_type(), "normed dtype must match grad_out");
  TORCH_CHECK(gamma.scalar_type() == grad_out.scalar_type(), "gamma dtype must match grad_out");
  TORCH_CHECK(inv_norm.scalar_type() == at::kFloat, "inv_norm must be float32");
  return rms_norm_backward_cuda(grad_out, normed, inv_norm, gamma);
}

std::vector<at::Tensor> rms_norm_forward_mixed(
    const at::Tensor& x,
    const at::Tensor& gamma) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
  TORCH_CHECK(gamma.dim() == 1, "gamma must have shape [D]");
  TORCH_CHECK(gamma.size(0) == x.size(-1), "gamma dimension must match x last dimension");
  TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
  return rms_norm_forward_mixed_cuda(x, gamma);
}

std::vector<at::Tensor> rms_norm_backward_mixed(
    const at::Tensor& grad_out,
    const at::Tensor& normed,
    const at::Tensor& inv_norm,
    const at::Tensor& gamma,
    const at::Tensor& x_meta) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
  TORCH_CHECK(normed.is_cuda(), "normed must be a CUDA tensor");
  TORCH_CHECK(inv_norm.is_cuda(), "inv_norm must be a CUDA tensor");
  TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
  TORCH_CHECK(x_meta.is_cuda(), "x_meta must be a CUDA tensor");
  TORCH_CHECK(grad_out.dim() >= 2, "grad_out must have at least 2 dimensions");
  TORCH_CHECK(normed.sizes() == grad_out.sizes(), "normed shape must match grad_out");
  TORCH_CHECK(gamma.dim() == 1, "gamma must have shape [D]");
  TORCH_CHECK(gamma.size(0) == grad_out.size(-1), "gamma dimension must match grad_out last dimension");
  TORCH_CHECK(inv_norm.dim() == 1, "inv_norm must have shape [M]");
  TORCH_CHECK(inv_norm.size(0) * gamma.size(0) == grad_out.numel(), "inv_norm length must match flattened rows");
  TORCH_CHECK(grad_out.scalar_type() == at::kFloat, "grad_out must be float32");
  TORCH_CHECK(normed.scalar_type() == at::kFloat, "normed must be float32");
  TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
  TORCH_CHECK(inv_norm.scalar_type() == at::kFloat, "inv_norm must be float32");
  return rms_norm_backward_mixed_cuda(grad_out, normed, inv_norm, gamma, x_meta);
}

std::vector<at::Tensor> attention_gate_forward(
    const at::Tensor& attn,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  TORCH_CHECK(attn.is_cuda(), "attn must be a CUDA tensor");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
  TORCH_CHECK(attn.dim() == 3, "attn must have shape [M, H, DH]");
  TORCH_CHECK(x.dim() == 2, "x must have shape [M, D]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [H, D]");
  TORCH_CHECK(bias.dim() == 1, "bias must have shape [H]");
  TORCH_CHECK(attn.size(0) == x.size(0), "attn and x row counts must match");
  TORCH_CHECK(attn.size(1) == weight.size(0), "attn heads must match weight");
  TORCH_CHECK(weight.size(1) == x.size(1), "weight input dimension must match x");
  TORCH_CHECK(bias.size(0) == weight.size(0), "bias heads must match weight");
  TORCH_CHECK(x.scalar_type() == attn.scalar_type(), "x dtype must match attn");
  TORCH_CHECK(weight.scalar_type() == attn.scalar_type(), "weight dtype must match attn");
  TORCH_CHECK(bias.scalar_type() == attn.scalar_type(), "bias dtype must match attn");
  return attention_gate_forward_cuda(attn, x, weight, bias);
}

std::vector<at::Tensor> attention_gate_backward(
    const at::Tensor& grad_out,
    const at::Tensor& attn,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& gate) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
  TORCH_CHECK(attn.is_cuda(), "attn must be a CUDA tensor");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
  TORCH_CHECK(grad_out.dim() == 2, "grad_out must have shape [M, H*DH]");
  TORCH_CHECK(attn.dim() == 3, "attn must have shape [M, H, DH]");
  TORCH_CHECK(x.dim() == 2, "x must have shape [M, D]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [H, D]");
  TORCH_CHECK(gate.dim() == 2, "gate must have shape [M, H]");
  TORCH_CHECK(grad_out.size(0) == attn.size(0), "grad_out rows must match attn");
  TORCH_CHECK(grad_out.size(1) == attn.size(1) * attn.size(2), "grad_out columns must match flattened attn");
  TORCH_CHECK(attn.size(0) == x.size(0), "attn and x row counts must match");
  TORCH_CHECK(attn.size(1) == weight.size(0), "attn heads must match weight");
  TORCH_CHECK(weight.size(1) == x.size(1), "weight input dimension must match x");
  TORCH_CHECK(gate.size(0) == attn.size(0), "gate rows must match attn");
  TORCH_CHECK(gate.size(1) == attn.size(1), "gate heads must match attn");
  TORCH_CHECK(attn.scalar_type() == grad_out.scalar_type(), "attn dtype must match grad_out");
  TORCH_CHECK(x.scalar_type() == grad_out.scalar_type(), "x dtype must match grad_out");
  TORCH_CHECK(weight.scalar_type() == grad_out.scalar_type(), "weight dtype must match grad_out");
  TORCH_CHECK(gate.scalar_type() == grad_out.scalar_type(), "gate dtype must match grad_out");
  return attention_gate_backward_cuda(grad_out, attn, x, weight, gate);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("band_split_forward", &band_split_forward, "Fused BandSplit grouped forward");
  m.def("band_split_backward", &band_split_backward, "Fused BandSplit grouped backward");
  m.def("grouped_linear_forward", &grouped_linear_forward, "Fused grouped linear forward");
  m.def("rotary_forward", &rotary_forward, "Fused rotary embedding forward");
  m.def("rotary_backward", &rotary_backward, "Fused rotary embedding backward");
  m.def("rms_norm_forward", &rms_norm_forward, "Fused RMSNorm forward");
  m.def("rms_norm_backward", &rms_norm_backward, "Fused RMSNorm backward");
  m.def("rms_norm_forward_mixed", &rms_norm_forward_mixed, "Fused mixed-dtype RMSNorm forward");
  m.def("rms_norm_backward_mixed", &rms_norm_backward_mixed, "Fused mixed-dtype RMSNorm backward");
  m.def("attention_gate_forward", &attention_gate_forward, "Fused attention gate forward");
  m.def("attention_gate_backward", &attention_gate_backward, "Fused attention gate backward");
}
