#pragma once

#include "layer.h"
#include "linear.h"
#include "layer_norm.h"
#include "dropout.h"
#include "attention.h"
#include <memory>

namespace megatron {

class TransformerBlock : public Layer {
public:
    TransformerBlock(int embed_dim, int num_heads, int ff_dim, 
                    bool dropout = false, float dropout_prob = 0.1f, 
                    const std::string& name = "");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    // Reset gradients
    void zero_grad() override;
    
    // Access to attention weights
    const Tensor& attention_weights() const { return attention_.attention_weights(); }

private:
    int embed_dim_;
    int num_heads_;
    int ff_dim_;
    bool use_dropout_;
    float dropout_prob_;
    
    // Multi-head attention
    MultiHeadAttention attention_;
    
    // Layer normalization layers
    LayerNorm norm1_;
    LayerNorm norm2_;
    
    // Feed-forward network
    Linear ff1_;
    Linear ff2_;
    
    // Dropout layers
    Dropout dropout1_;
    Dropout dropout2_;
    
    // Cache for forward/backward pass
    Tensor input_;
    Tensor attn_output_;
    Tensor ff_output_;
    Tensor norm1_output_;
    Tensor norm2_output_;
};

} // namespace megatron