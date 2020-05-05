#pragma once
#include "libtorch_cpp_extension_test.h"
#include "torch/torch.h"
using namespace torch::autograd;

class TestFunction : public Function<TestFunction> {

public:
	
  static variable_list forward(AutogradContext *ctx, Variable x, Variable y) {
   std::cout << "i am forward" << std::endl;
    const auto forward_result = Test_forward_cpu(x, y);
    return {forward_result};
  }

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_output) {
    std::cout << "i am backward" << std::endl;
    auto grads = Test_backward_cpu(grad_output[0]);
    return {grads};
  }
};