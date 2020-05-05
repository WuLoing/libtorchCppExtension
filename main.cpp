#include "libtorch_cpp_extension_module.h"
#include "torch/torch.h"

auto main() -> int {
	torch::autograd::Variable x = torch::autograd::Variable(torch::randint(2, {2, 2})).set_requires_grad(true);
  std::cout << "x=" << x << std::endl;

	torch::autograd::Variable y = torch::autograd::Variable(torch::randint(4, {2, 2})).set_requires_grad(true);
  std::cout << "y=" << y << std::endl;

  auto module = TestModule();
  std::vector<at::Tensor> z = module->forward(x, y);
  std::cout << "z=" << z << std::endl;
  std::cout << "z.size=" << z.size() << std::endl;
  z[0].sum().backward();

  std::cout << "x.grad=" << x.grad() << std::endl;
  std::cout << "y.grad=" << y.grad() << std::endl;
  return 0;
}
