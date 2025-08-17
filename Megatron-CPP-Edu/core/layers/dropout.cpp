#include "dropout.h"
#include <algorithm>

namespace megatron {

Dropout::Dropout(float p, const std::string& name)
    : Layer(name), p_(p), scale_(1.0f / (1.0f - p)), gen_(std::random_device{}()), dist_(0.0f, 1.0f) {
    
    if (p_ < 0.0f || p_ >= 1.0f) {
        throw std::invalid_argument("Dropout probability must be in [0, 1)");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    if (!training_) {
        // During evaluation, just return the input
        return input;
    }
    
    // Create dropout mask
    mask_ = Tensor(input.shape());
    
    for (int i = 0; i < mask_.size(); ++i) {
        float r = dist_(gen_);
        mask_[i] = (r < p_) ? 0.0f : scale_;
    }
    
    // Apply dropout
    Tensor output = Tensor(input.shape());
    for (int i = 0; i < output.size(); ++i) {
        output[i] = input[i] * mask_[i];
    }
    
    return output;
}

Tensor Dropout::backward(const Tensor& grad_output) {
    if (!training_) {
        // During evaluation, just return the gradient
        return grad_output;
    }
    
    // Apply mask to gradients
    Tensor grad_input = Tensor(grad_output.shape());
    for (int i = 0; i < grad_input.size(); ++i) {
        grad_input[i] = grad_output[i] * mask_[i];
    }
    
    return grad_input;
}

std::vector<Tensor> Dropout::parameters() const {
    // Dropout has no trainable parameters
    return {};
}

std::vector<Tensor> Dropout::gradients() const {
    // Dropout has no gradients
    return {};
}

} // namespace megatron