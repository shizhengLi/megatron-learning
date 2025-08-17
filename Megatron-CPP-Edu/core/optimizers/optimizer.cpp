#include "optimizer.h"
#include <stdexcept>

namespace megatron {

Optimizer::Optimizer(float learning_rate) : learning_rate_(learning_rate) {
    if (learning_rate <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }
}

void Optimizer::zero_grad(const std::vector<Tensor>& gradients) {
    for (const auto& grad : gradients) {
        const_cast<Tensor&>(grad).zeros();
    }
}

void Optimizer::set_parameters(const std::vector<Tensor>& parameters) {
    parameters_ = parameters;
}

void Optimizer::validate_step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) {
    if (parameters.size() != gradients.size()) {
        throw std::invalid_argument("Number of parameters and gradients must match");
    }
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (parameters[i].shape() != gradients[i].shape()) {
            throw std::invalid_argument("Parameter and gradient shapes must match");
        }
    }
}

} // namespace megatron