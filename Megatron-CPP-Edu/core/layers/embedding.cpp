#include "embedding.h"
#include <cmath>
#include <random>

namespace megatron {

Embedding::Embedding(int vocab_size, int embedding_dim, const std::string& name)
    : Layer(name), 
      vocab_size_(vocab_size), 
      embedding_dim_(embedding_dim) {
    
    // Initialize embedding matrix: [vocab_size, embedding_dim]
    weight_ = Tensor({vocab_size, embedding_dim});
    weight_grad_ = Tensor({vocab_size, embedding_dim});
    
    // Initialize parameters
    initialize_parameters();
}

Tensor Embedding::forward(const Tensor& input) {
    // Input shape: [batch_size, seq_len] with token indices
    // Output shape: [batch_size, seq_len, embedding_dim]
    
    input_ = input;
    
    // Create output tensor
    std::vector<int> output_shape = {input.shape()[0], input.shape()[1], embedding_dim_};
    Tensor output(output_shape);
    
    // Look up embeddings for each token
    for (int i = 0; i < input.shape()[0]; ++i) {  // batch
        for (int j = 0; j < input.shape()[1]; ++j) {  // seq_len
            int input_idx = i * input.shape()[1] + j;
            if (input_idx < 0 || input_idx >= input.size()) {
                throw std::out_of_range("Input index " + std::to_string(input_idx) + " out of range [0, " + std::to_string(input.size()) + ") for tensor shape [" + std::to_string(input.shape()[0]) + ", " + std::to_string(input.shape()[1]) + "]");
            }
            
            int token_idx = static_cast<int>(input[input_idx]);
            
            // Validate token index
            if (token_idx < 0 || token_idx >= vocab_size_) {
                throw std::out_of_range("Token index " + std::to_string(token_idx) + " out of vocabulary range [0, " + std::to_string(vocab_size_) + ")");
            }
            
            // Copy embedding vector
            for (int k = 0; k < embedding_dim_; ++k) {
                int output_idx = i * input.shape()[1] * embedding_dim_ + j * embedding_dim_ + k;
                int weight_idx = token_idx * embedding_dim_ + k;
                
                if (output_idx < 0 || output_idx >= output.size()) {
                    throw std::out_of_range("Output index " + std::to_string(output_idx) + " out of range [0, " + std::to_string(output.size()) + ") for i=" + std::to_string(i) + ", j=" + std::to_string(j) + ", k=" + std::to_string(k) + ", shape=[" + std::to_string(output.shape()[0]) + ", " + std::to_string(output.shape()[1]) + ", " + std::to_string(output.shape()[2]) + "]");
                }
                if (weight_idx < 0 || weight_idx >= weight_.size()) {
                    throw std::out_of_range("Weight index out of range");
                }
                
                output[output_idx] = weight_[weight_idx];
            }
        }
    }
    
    return output;
}

Tensor Embedding::backward(const Tensor& grad_output) {
    // grad_output shape: [batch_size, seq_len, embedding_dim]
    // Returns gradients for input: [batch_size, seq_len]
    
    // Reset gradients
    weight_grad_.zeros();
    
    // Accumulate gradients for embedding weights
    for (int i = 0; i < input_.shape()[0]; ++i) {  // batch
        for (int j = 0; j < input_.shape()[1]; ++j) {  // seq_len
            int token_idx = static_cast<int>(input_[i * input_.shape()[1] + j]);
            
            // Add gradients to the corresponding embedding vector
            for (int k = 0; k < embedding_dim_; ++k) {
                weight_grad_[token_idx * embedding_dim_ + k] += 
                    grad_output[i * input_.shape()[1] * embedding_dim_ + j * embedding_dim_ + k];
            }
        }
    }
    
    // Return gradients for input with same shape as input
    // For embedding layers, input gradients are typically not computed (tokens are discrete)
    // But we return a zero tensor with the correct shape for compatibility
    Tensor grad_input(input_.shape());
    grad_input.zeros();
    return grad_input;
}

std::vector<Tensor> Embedding::parameters() const {
    std::vector<Tensor> params;
    params.push_back(weight_);
    return params;
}

std::vector<Tensor> Embedding::gradients() const {
    std::vector<Tensor> grads;
    grads.push_back(weight_grad_);
    return grads;
}

void Embedding::set_weight(const Tensor& weight) {
    if (weight.shape()[0] != vocab_size_ || weight.shape()[1] != embedding_dim_) {
        throw std::invalid_argument("Weight tensor has incorrect shape");
    }
    weight_ = weight;
}

void Embedding::initialize_parameters() {
    // Initialize embedding weights with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.01f);
    
    for (int i = 0; i < weight_.size(); ++i) {
        weight_[i] = dist(gen);
    }
    
    // Initialize gradients to zeros
    weight_grad_.zeros();
}

} // namespace megatron