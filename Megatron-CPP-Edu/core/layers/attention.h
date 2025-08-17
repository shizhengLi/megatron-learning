#pragma once

#include "layer.h"
#include <memory>
#include <vector>

namespace megatron {

class MultiHeadAttention : public Layer {
public:
    MultiHeadAttention(int embed_dim, int num_heads, bool dropout = false, float dropout_prob = 0.1f, const std::string& name = "");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    // Reset gradients
    void zero_grad() override;
    
    // Access to attention weights (for visualization/analysis)
    const Tensor& attention_weights() const { return attention_weights_; }

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    bool use_dropout_;
    float dropout_prob_;
    
    // Linear projections for Q, K, V
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear out_proj_;
    
    // Dropout layer
    Dropout dropout_;
    
    // Cache for forward/backward pass
    Tensor input_;
    Tensor q_proj_cache_;
    Tensor k_proj_cache_;
    Tensor v_proj_cache_;
    Tensor attention_weights_;
    Tensor output_cache_;
    
    // Helper methods
    Tensor split_heads(const Tensor& x);
    Tensor combine_heads(const Tensor& x);
    Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v);
    void compute_attention_gradients(const Tensor& grad_output);
};

} // namespace megatron