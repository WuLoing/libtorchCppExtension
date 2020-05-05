#pragma once
#include "libtorch_cpp_extension_func.h"
#include "torch/torch.h"

class TestModuleImpl : public torch::nn::Module {
 public:
  TestModuleImpl() {}

  std::vector<at::Tensor> forward(torch::Tensor x, torch::Tensor y) {
    std::vector<at::Tensor> result = TestFunction::apply<TestFunction>(x, y);
  	
    return result;
  }
};
TORCH_MODULE(TestModule);
