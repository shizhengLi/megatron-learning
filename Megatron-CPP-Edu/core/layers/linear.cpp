#include "linear.h"
#include <cmath>
#include <random>

namespace megatron {

Linear::Linear(int in_features, int out_features, bool bias, const std::string& name)
    : Layer(name), in_features_(in_features), out_features_(out_features), has_bias_(bias) {
    
    // Initialize weight matrix: [out_features, in_features]
    weight_ = Tensor({out_features, in_features});
    
    // Initialize bias if needed: [out_features]
    if (has_bias_) {
        bias_ = Tensor({out_features});
    }
    
    // Initialize gradient tensors
    weight_grad_ = Tensor({out_features, in_features});
    if (has_bias_) {
        bias_grad_ = Tensor({out_features});
    }
    
    // Initialize parameters
    initialize_parameters();
}

Tensor Linear::forward(const Tensor& input) {
    // Input shape: [batch_size, in_features]
    // Weight shape: [out_features, in_features]
    // Output shape: [batch_size, out_features]
    
    input_ = input;  // Store input for backward pass
    
    // Compute output: input @ weight.T + bias
    output_ = input.matmul(weight_.transpose());
    
    if (has_bias_) {
        // Add bias to each row
        for (int i = 0; i < output_.shape()[0]; ++i) {
            for (int j = 0; j < output_.shape()[1]; ++j) {
                output_[i * output_.shape()[1] + j] += bias_[j];
            }
        }
    }
    
    return output_;
}

Tensor Linear::backward(const Tensor& grad_output) {
    // grad_output shape: [batch_size, out_features]
    // Returns gradients for input: [batch_size, in_features]
    
    // Compute gradients
    compute_gradients(grad_output);
    
    // Compute input gradients: grad_output @ weight
    Tensor grad_input = grad_output.matmul(weight_);
    
    return grad_input;
}

std::vector<Tensor> Linear::parameters() const {
    std::vector<Tensor> params;
    params.push_back(weight_);
    if (has_bias_) {
        params.push_back(bias_);
    }
    return params;
}

std::vector<Tensor> Linear::gradients() const {
    std::vector<Tensor> grads;
    grads.push_back(weight_grad_);
    if (has_bias_) {
        grads.push_back(bias_grad_);
    }
    return grads;
}

void Linear::set_weight(const Tensor& weight) {
    if (weight.shape()[0] != out_features_ || weight.shape()[1] != in_features_) {
        throw std::invalid_argument("Weight tensor has incorrect shape");
    }
    weight_ = weight;
}

void Linear::set_bias(const Tensor& bias) {
    if (!has_bias_) {
        throw std::invalid_argument("This linear layer does not have bias");
    }
    if (bias.shape()[0] != out_features_) {
        throw std::invalid_argument("Bias tensor has incorrect shape");
    }
    bias_ = bias;
}

void Linear::initialize_parameters() {
    // Kaiming He initialization for weights
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / in_features_);
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (int i = 0; i < weight_.size(); ++i) {
        weight_[i] = dist(gen);
    }
    
    // Initialize bias to zeros
    if (has_bias_) {
        bias_.zeros();
    }
    
    // Initialize gradients to zeros
    weight_grad_.zeros();
    if (has_bias_) {
        bias_grad_.zeros();
    }
}

void Linear::compute_gradients(const Tensor& grad_output) {
    // grad_output shape: [batch_size, out_features]
    // input_ shape: [batch_size, in_features]
    
    // Compute weight gradients: grad_output.T @ input
    Tensor grad_weight = grad_output.transpose().matmul(input_);
    weight_grad_ = grad_weight;
    
    // Compute bias gradients: sum grad_output along batch dimension
    if (has_bias_) {
        Tensor grad_bias = grad_output.sum(0);
        bias_grad_ = grad_bias;
    }
}

} // namespace megatron