#include "adamw.h"
#include <cmath>

namespace megatron {

AdamW::AdamW(float learning_rate, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(learning_rate), 
      beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay), step_count_(0) {
    
    if (beta1 < 0.0f || beta1 >= 1.0f) {
        throw std::invalid_argument("beta1 must be in range [0, 1)");
    }
    if (beta2 < 0.0f || beta2 >= 1.0f) {
        throw std::invalid_argument("beta2 must be in range [0, 1)");
    }
    if (eps <= 0.0f) {
        throw std::invalid_argument("eps must be positive");
    }
    if (weight_decay < 0.0f) {
        throw std::invalid_argument("Weight decay must be non-negative");
    }
}

void AdamW::step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) {
    validate_step(parameters, gradients);
    
    // Initialize moments if needed
    if (exp_avg_.empty()) {
        initialize_moments(parameters);
    }
    
    step_count_++;
    float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);
    float step_size = learning_rate_ * std::sqrt(bias_correction2) / bias_correction1;
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        Tensor& param = const_cast<Tensor&>(parameters[i]);
        const Tensor& grad = gradients[i];
        
        // Apply weight decay (AdamW style - decoupled weight decay)
        if (weight_decay_ > 0.0f) {
            Tensor weight_decay_tensor(param.shape());
            weight_decay_tensor.fill(1.0f - learning_rate_ * weight_decay_);
            param = param * weight_decay_tensor;
        }
        
        // Create beta tensors
        Tensor beta1_tensor(exp_avg_[i].shape());
        beta1_tensor.fill(beta1_);
        Tensor beta2_tensor(exp_avg_sq_[i].shape());
        beta2_tensor.fill(beta2_);
        Tensor one_minus_beta1_tensor(grad.shape());
        one_minus_beta1_tensor.fill(1.0f - beta1_);
        Tensor one_minus_beta2_tensor(grad.shape());
        one_minus_beta2_tensor.fill(1.0f - beta2_);
        
        // Update first moment (momentum)
        exp_avg_[i] = exp_avg_[i] * beta1_tensor + grad * one_minus_beta1_tensor;
        
        // Update second moment (RMSprop)
        Tensor grad_squared = grad * grad;
        exp_avg_sq_[i] = exp_avg_sq_[i] * beta2_tensor + grad_squared * one_minus_beta2_tensor;
        
        // Compute update
        Tensor eps_tensor(exp_avg_sq_[i].shape());
        eps_tensor.fill(eps_);
        Tensor denom = exp_avg_sq_[i].sqrt() + eps_tensor;
        Tensor update = exp_avg_[i] / denom;
        
        // Create step size tensor
        Tensor step_size_tensor(update.shape());
        step_size_tensor.fill(step_size);
        
        // Apply update
        param = param - update * step_size_tensor;
    }
}

void AdamW::initialize_moments(const std::vector<Tensor>& parameters) {
    exp_avg_.clear();
    exp_avg_sq_.clear();
    
    for (const auto& param : parameters) {
        Tensor first_moment(param.shape());
        Tensor second_moment(param.shape());
        first_moment.zeros();
        second_moment.zeros();
        
        exp_avg_.push_back(first_moment);
        exp_avg_sq_.push_back(second_moment);
    }
}

} // namespace megatron