#include "transformer_block.h"
#include <vector>

namespace megatron {

TransformerBlock::TransformerBlock(int embed_dim, int num_heads, int ff_dim, 
                                 bool dropout, float dropout_prob, const std::string& name)
    : Layer(name),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      ff_dim_(ff_dim),
      use_dropout_(dropout),
      dropout_prob_(dropout_prob),
      attention_(embed_dim, num_heads, dropout, dropout_prob, name + "_attention"),
      norm1_(embed_dim, 1e-5f, name + "_norm1"),
      norm2_(embed_dim, 1e-5f, name + "_norm2"),
      ff1_(embed_dim, ff_dim, true, name + "_ff1"),
      ff2_(ff_dim, embed_dim, true, name + "_ff2"),
      dropout1_(dropout_prob, name + "_dropout1"),
      dropout2_(dropout_prob, name + "_dropout2") {
}

Tensor TransformerBlock::forward(const Tensor& input) {
    // Input shape: [batch_size, seq_len, embed_dim]
    input_ = input;
    
    // Multi-head attention with residual connection and layer norm
    // 1. Layer norm
    Tensor norm1_input = norm1_.forward(input);
    
    // 2. Multi-head attention
    attn_output_ = attention_.forward(norm1_input);
    
    // 3. Dropout
    if (use_dropout_) {
        attn_output_ = dropout1_.forward(attn_output_);
    }
    
    // 4. Residual connection
    Tensor residual1 = input + attn_output_;
    norm1_output_ = residual1;
    
    // Feed-forward network with residual connection and layer norm
    // 1. Layer norm
    Tensor norm2_input = norm2_.forward(residual1);
    
    // 2. First linear layer
    Tensor ff1_output = ff1_.forward(norm2_input);
    
    // 3. Activation (ReLU)
    Tensor ff1_activated = ff1_output.relu();
    
    // 4. Second linear layer
    ff_output_ = ff2_.forward(ff1_activated);
    
    // 5. Dropout
    if (use_dropout_) {
        ff_output_ = dropout2_.forward(ff_output_);
    }
    
    // 6. Residual connection
    Tensor residual2 = residual1 + ff_output_;
    norm2_output_ = residual2;
    
    return norm2_output_;
}

Tensor TransformerBlock::backward(const Tensor& grad_output) {
    // grad_output shape: [batch_size, seq_len, embed_dim]
    
    // Backward through second residual connection
    Tensor grad_norm2 = grad_output;
    Tensor grad_ff_residual = grad_output;
    
    // Backward through second dropout
    if (use_dropout_) {
        grad_ff_residual = dropout2_.backward(grad_ff_residual);
    }
    
    // Backward through feed-forward network
    Tensor grad_ff2 = ff2_.backward(grad_ff_residual);
    
    // Backward through ReLU (gradient is 1 for positive inputs, 0 for negative)
    Tensor grad_ff1_activated = grad_ff2;
    for (int i = 0; i < grad_ff1_activated.size(); ++i) {
        if (ff_output_[i] <= 0.0f) {
            grad_ff1_activated[i] = 0.0f;
        }
    }
    
    Tensor grad_ff1 = ff1_.backward(grad_ff1_activated);
    
    // Backward through second layer norm
    Tensor grad_residual1 = norm2_.backward(grad_ff1) + grad_norm2;
    
    // Backward through first residual connection
    Tensor grad_norm1 = grad_residual1;
    Tensor grad_attn_residual = grad_residual1;
    
    // Backward through first dropout
    if (use_dropout_) {
        grad_attn_residual = dropout1_.backward(grad_attn_residual);
    }
    
    // Backward through attention
    Tensor grad_attn = attention_.backward(grad_attn_residual);
    
    // Backward through first layer norm
    Tensor grad_input = norm1_.backward(grad_attn) + grad_norm1;
    
    return grad_input;
}

std::vector<Tensor> TransformerBlock::parameters() const {
    std::vector<Tensor> params;
    
    // Add parameters from attention
    auto attn_params = attention_.parameters();
    params.insert(params.end(), attn_params.begin(), attn_params.end());
    
    // Add parameters from layer norms
    auto norm1_params = norm1_.parameters();
    auto norm2_params = norm2_.parameters();
    params.insert(params.end(), norm1_params.begin(), norm1_params.end());
    params.insert(params.end(), norm2_params.begin(), norm2_params.end());
    
    // Add parameters from feed-forward network
    auto ff1_params = ff1_.parameters();
    auto ff2_params = ff2_.parameters();
    params.insert(params.end(), ff1_params.begin(), ff1_params.end());
    params.insert(params.end(), ff2_params.begin(), ff2_params.end());
    
    return params;
}

std::vector<Tensor> TransformerBlock::gradients() const {
    std::vector<Tensor> grads;
    
    // Add gradients from attention
    auto attn_grads = attention_.gradients();
    grads.insert(grads.end(), attn_grads.begin(), attn_grads.end());
    
    // Add gradients from layer norms
    auto norm1_grads = norm1_.gradients();
    auto norm2_grads = norm2_.gradients();
    grads.insert(grads.end(), norm1_grads.begin(), norm1_grads.end());
    grads.insert(grads.end(), norm2_grads.begin(), norm2_grads.end());
    
    // Add gradients from feed-forward network
    auto ff1_grads = ff1_.gradients();
    auto ff2_grads = ff2_.gradients();
    grads.insert(grads.end(), ff1_grads.begin(), ff1_grads.end());
    grads.insert(grads.end(), ff2_grads.begin(), ff2_grads.end());
    
    return grads;
}

void TransformerBlock::zero_grad() {
    attention_.zero_grad();
    norm1_.zero_grad();
    norm2_.zero_grad();
    ff1_.zero_grad();
    ff2_.zero_grad();
}

} // namespace megatron