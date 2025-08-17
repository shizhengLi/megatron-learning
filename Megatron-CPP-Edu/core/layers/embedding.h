#pragma once

#include "layer.h"

namespace megatron {

class Embedding : public Layer {
public:
    Embedding(int vocab_size, int embedding_dim, const std::string& name = "");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    // Access to embedding weights
    const Tensor& weight() const { return weight_; }
    
    // Set embedding weights
    void set_weight(const Tensor& weight);

private:
    int vocab_size_;
    int embedding_dim_;
    
    Tensor weight_;      // [vocab_size, embedding_dim]
    Tensor weight_grad_; // [vocab_size, embedding_dim]
    
    Tensor input_;       // Store input for backward pass
    
    // Helper methods
    void initialize_parameters();
};

} // namespace megatron