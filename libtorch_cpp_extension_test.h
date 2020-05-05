#pragma once
#include "torch/torch.h"

// 前向传播，两个 Tensor 相加。
inline torch::Tensor Test_forward_cpu(const torch::Tensor& x,
                                      const torch::Tensor& y) {
  AT_ASSERTM(x.sizes() == y.sizes(), "x must be the same size as y");
  torch::Tensor z = torch::zeros(x.sizes());
  z = 2 * x + y;
  return z;
}

// 反向传播
// z对x的导数是2，z对y的导数是1。 z=2*x+y
inline std::vector<torch::Tensor> Test_backward_cpu(
    const torch::Tensor& gradOutput) {
  const torch::Tensor gradOutputX =
      2 * gradOutput * torch::ones(gradOutput.sizes());
  const torch::Tensor gradOutputY =
      gradOutput * torch::ones(gradOutput.sizes());
  return {gradOutputX, gradOutputY};
}