#include "layer_norm.h"
#include <cmath>

namespace megatron {

LayerNorm::LayerNorm(int normalized_shape, float eps, const std::string& name)
    : Layer(name), 
      normalized_shape_(normalized_shape), 
      eps_(eps) {
    
    // Initialize parameters: weight and bias
    weight_ = Tensor({normalized_shape});
    bias_ = Tensor({normalized_shape});
    
    // Initialize gradients
    weight_grad_ = Tensor({normalized_shape});
    bias_grad_ = Tensor({normalized_shape});
    
    // Initialize parameters
    initialize_parameters();
}

Tensor LayerNorm::forward(const Tensor& input) {
    // Input shape: [batch_size, ..., normalized_shape]
    
    input_ = input;
    
    // Reshape input to 2D: [batch_size * ..., normalized_shape]
    int batch_size = 1;
    for (int i = 0; i < input.dim() - 1; ++i) {
        batch_size *= input.shape()[i];
    }
    
    Tensor input_2d = input.view({batch_size, normalized_shape_});
    
    // Compute mean and variance for each batch
    mean_ = input_2d.mean(1);  // [batch_size]
    var_ = Tensor({batch_size});
    
    for (int i = 0; i < batch_size; ++i) {
        float sum_sq = 0.0f;
        for (int j = 0; j < normalized_shape_; ++j) {
            float diff = input_2d[i * normalized_shape_ + j] - mean_[i];
            sum_sq += diff * diff;
        }
        var_[i] = sum_sq / normalized_shape_;
    }
    
    // Normalize input
    normalized_ = Tensor(input_2d.shape());
    
    for (int i = 0; i < batch_size; ++i) {
        float std = std::sqrt(var_[i] + eps_);
        for (int j = 0; j < normalized_shape_; ++j) {
            normalized_[i * normalized_shape_ + j] = 
                (input_2d[i * normalized_shape_ + j] - mean_[i]) / std;
        }
    }
    
    // Apply weight and bias
    Tensor output_2d = Tensor(input_2d.shape());
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < normalized_shape_; ++j) {
            output_2d[i * normalized_shape_ + j] = 
                normalized_[i * normalized_shape_ + j] * weight_[j] + bias_[j];
        }
    }
    
    // Reshape output back to original dimensions
    return output_2d.view(input.shape());
}

Tensor LayerNorm::backward(const Tensor& grad_output) {
    // grad_output shape: [batch_size, normalized_shape]
    // Returns gradients for input: [batch_size, normalized_shape]
    
    // Compute gradients for weight and bias
    for (int i = 0; i < normalized_shape_; ++i) {
        float weight_grad_sum = 0.0f;
        float bias_grad_sum = 0.0f;
        
        for (int j = 0; j < grad_output.shape()[0]; ++j) {
            weight_grad_sum += grad_output[j * grad_output.shape()[1] + i] * normalized_[j * grad_output.shape()[1] + i];
            bias_grad_sum += grad_output[j * grad_output.shape()[1] + i];
        }
        
        weight_grad_[i] = weight_grad_sum;
        bias_grad_[i] = bias_grad_sum;
    }
    
    // Compute gradients for input (simplified version)
    Tensor grad_input = Tensor(grad_output.shape());
    
    for (int i = 0; i < grad_output.shape()[0]; ++i) {
        float std = std::sqrt(var_[i] + eps_);
        
        for (int j = 0; j < grad_output.shape()[1]; ++j) {
            // Simplified gradient computation
            float grad_norm = grad_output[i * grad_output.shape()[1] + j] * weight_[j];
            
            // Gradient with respect to input
            grad_input[i * grad_output.shape()[1] + j] = grad_norm / std;
        }
    }
    
    return grad_input;
}

std::vector<Tensor> LayerNorm::parameters() const {
    std::vector<Tensor> params;
    params.push_back(weight_);
    params.push_back(bias_);
    return params;
}

std::vector<Tensor> LayerNorm::gradients() const {
    std::vector<Tensor> grads;
    grads.push_back(weight_grad_);
    grads.push_back(bias_grad_);
    return grads;
}

void LayerNorm::initialize_parameters() {
    // Initialize weight to ones
    weight_.ones();
    
    // Initialize bias to zeros
    bias_.zeros();
    
    // Initialize gradients to zeros
    weight_grad_.zeros();
    bias_grad_.zeros();
}

} // namespace megatron