#include "sgd.h"
#include <algorithm>

namespace megatron {

SGD::SGD(float learning_rate, float momentum, float weight_decay)
    : Optimizer(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {
    if (momentum < 0.0f || momentum >= 1.0f) {
        throw std::invalid_argument("Momentum must be in range [0, 1)");
    }
    if (weight_decay < 0.0f) {
        throw std::invalid_argument("Weight decay must be non-negative");
    }
}

void SGD::step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) {
    validate_step(parameters, gradients);
    
    // Initialize velocity if needed
    if (momentum_ > 0.0f && velocity_.empty()) {
        initialize_velocity(parameters);
    }
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        Tensor& param = const_cast<Tensor&>(parameters[i]);
        const Tensor& grad = gradients[i];
        
        // Apply weight decay if needed
        Tensor effective_grad = grad;
        if (weight_decay_ > 0.0f) {
            // Create weight decay tensor
            Tensor weight_decay_tensor(param.shape());
            weight_decay_tensor.fill(weight_decay_);
            effective_grad = grad + param * weight_decay_tensor;
        }
        
        if (momentum_ > 0.0f) {
            // Create momentum and learning rate tensors
            Tensor momentum_tensor(velocity_[i].shape());
            momentum_tensor.fill(momentum_);
            Tensor lr_tensor(effective_grad.shape());
            lr_tensor.fill(learning_rate_);
            
            // Update velocity: v = momentum * v + lr * grad
            velocity_[i] = velocity_[i] * momentum_tensor + effective_grad * lr_tensor;
            
            // Update parameters: param = param - v
            param = param - velocity_[i];
        } else {
            // Create learning rate tensor
            Tensor lr_tensor(effective_grad.shape());
            lr_tensor.fill(learning_rate_);
            
            // Standard SGD update: param = param - lr * grad
            param = param - effective_grad * lr_tensor;
        }
    }
}

void SGD::initialize_velocity(const std::vector<Tensor>& parameters) {
    velocity_.clear();
    for (const auto& param : parameters) {
        Tensor velocity(param.shape());
        velocity.zeros();
        velocity_.push_back(velocity);
    }
}

} // namespace megatron