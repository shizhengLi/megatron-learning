#include "linear.h"
#include <iostream>
#include <cmath>
#include <random>

namespace megatron {

Linear::Linear(int in_features, int out_features, bool bias, const std::string& name)
    : Layer(name), 
      in_features_(in_features), 
      out_features_(out_features), 
      has_bias_(bias) {
    
    // Initialize weight matrix: [out_features, in_features]
    weight_ = Tensor({out_features, in_features});
    weight_grad_ = Tensor({out_features, in_features});
    
    // Initialize bias if needed: [out_features]
    if (has_bias_) {
        bias_ = Tensor({out_features});
        bias_grad_ = Tensor({out_features});
    }
    
    // Initialize parameters
    initialize_parameters();
}

Tensor Linear::forward(const Tensor& input) {
    // Input shape: [batch_size, ..., in_features]
    // Weight shape: [out_features, in_features]
    // Output shape: [batch_size, ..., out_features]
    
    input_ = input;  // Store input for backward pass
    
    // Reshape input to 2D if needed: [batch_size * ..., in_features]
    Tensor input_2d = input;
    std::vector<int> original_shape = input.shape();
    
    if (input.dim() > 2) {
        // Flatten all dimensions except the last one
        int flattened_size = 1;
        for (int i = 0; i < input.dim() - 1; ++i) {
            flattened_size *= input.shape()[i];
        }
        input_2d = input.view({flattened_size, input.shape().back()});
    }
    
    // Compute output: input_2d @ weight.T + bias
    Tensor output_2d = input_2d.matmul(weight_.transpose());
    
    if (has_bias_) {
        // Add bias to each row
        for (int i = 0; i < output_2d.shape()[0]; ++i) {
            for (int j = 0; j < output_2d.shape()[1]; ++j) {
                output_2d[i * output_2d.shape()[1] + j] += bias_[j];
            }
        }
    }
    
    // Reshape output back to original dimensions (except last one)
    if (input.dim() > 2) {
        std::vector<int> output_shape = original_shape;
        output_shape.back() = out_features_;
        
        // Manually reshape instead of using view to avoid potential issues
        Tensor output_reshaped(output_shape);
        for (int i = 0; i < output_reshaped.size(); ++i) {
            output_reshaped[i] = output_2d[i];
        }
        output_ = output_reshaped;
    } else {
        output_ = output_2d;
    }
    
    return output_;
}

Tensor Linear::backward(const Tensor& grad_output) {
    // grad_output shape: [batch_size, ..., out_features]
    // Returns gradients for input: [batch_size, ..., in_features]
    
    // Reshape grad_output to 2D if needed
    Tensor grad_output_2d = grad_output;
    
    if (grad_output.dim() > 2) {
        // Flatten all dimensions except the last one
        int flattened_size = 1;
        for (int i = 0; i < grad_output.dim() - 1; ++i) {
            flattened_size *= grad_output.shape()[i];
        }
        grad_output_2d = grad_output.view({flattened_size, grad_output.shape().back()});
    }
    
    // Compute gradients
    compute_gradients(grad_output_2d);
    
    // Compute input gradients: grad_output_2d @ weight
    Tensor grad_input_2d = grad_output_2d.matmul(weight_);
    
    // Reshape grad_input back to original dimensions
    Tensor grad_input = grad_input_2d;
    if (grad_output.dim() > 2) {
        std::vector<int> grad_input_shape = grad_output.shape();
        grad_input_shape.back() = in_features_;
        
        // Manually reshape instead of using view to avoid potential issues
        Tensor grad_input_reshaped(grad_input_shape);
        for (int i = 0; i < grad_input_reshaped.size(); ++i) {
            grad_input_reshaped[i] = grad_input_2d[i];
        }
        grad_input = grad_input_reshaped;
    }
    
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

void Linear::zero_grad() {
    weight_grad_.zeros();
    if (has_bias_) {
        bias_grad_.zeros();
    }
}

void Linear::compute_gradients(const Tensor& grad_output) {
    // grad_output shape: [batch_size, out_features]
    // input_ shape: original input (could be 3D)
    
    // Reshape input_ to 2D if needed for gradient computation
    Tensor input_2d = input_;
    if (input_.dim() > 2) {
        // Flatten all dimensions except the last one
        int flattened_size = 1;
        for (int i = 0; i < input_.dim() - 1; ++i) {
            flattened_size *= input_.shape()[i];
        }
        input_2d = input_.view({flattened_size, input_.shape().back()});
    }
    
    // Compute weight gradients: grad_output.T @ input_2d
    Tensor grad_weight = grad_output.transpose().matmul(input_2d);
    weight_grad_ = grad_weight;
    
    // Compute bias gradients: sum grad_output along batch dimension
    if (has_bias_) {
        Tensor grad_bias = grad_output.sum(0);
        bias_grad_ = grad_bias;
    }
}

} // namespace megatron