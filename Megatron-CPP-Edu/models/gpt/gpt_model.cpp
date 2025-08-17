#include "gpt_model.h"
#include <iostream>

namespace megatron {

GPTModel::GPTModel(int vocab_size, int max_seq_len, int embed_dim, int num_heads, 
                   int num_layers, int ff_dim, bool use_dropout, float dropout_prob, 
                   const std::string& name)
    : Layer(name),
      vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      num_layers_(num_layers),
      ff_dim_(ff_dim),
      use_dropout_(use_dropout),
      dropout_prob_(dropout_prob) {
    
    // Create token embedding
    token_embedding_ = std::make_shared<Embedding>(vocab_size, embed_dim, name + "_token_embed");
    
    // Create position embedding
    position_embedding_ = std::make_shared<Embedding>(max_seq_len, embed_dim, name + "_pos_embed");
    
    // Create transformer blocks
    transformer_blocks_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        transformer_blocks_.push_back(std::make_shared<TransformerBlock>(
            embed_dim, num_heads, ff_dim, use_dropout, dropout_prob, 
            name + "_block_" + std::to_string(i)));
    }
    
    // Create final layer norm
    final_norm_ = std::make_shared<LayerNorm>(embed_dim, 1e-5f, name + "_final_norm");
    
    // Create output projection (language model head)
    output_projection_ = std::make_shared<Linear>(embed_dim, vocab_size, false, name + "_output");
}

Tensor GPTModel::forward(const Tensor& input) {
    // Store input for backward pass
    input_ = input;
    
    // Input shape: [batch_size, seq_len]
    
    // Get token embeddings
    embedded_input_ = token_embedding_->forward(input);
    
    // Create position IDs
    Tensor position_ids = create_position_ids(input);
    
    // Add position embeddings
    Tensor pos_embedded = position_embedding_->forward(position_ids);
    embedded_input_ = embedded_input_ + pos_embedded;
    
    // Pass through transformer blocks
    Tensor x = embedded_input_;
    layer_outputs_.clear();
    layer_outputs_.reserve(num_layers_);
    
    for (int i = 0; i < num_layers; ++i) {
        x = transformer_blocks_[i]->forward(x);
        layer_outputs_.push_back(x);
    }
    
    // Apply final layer norm
    final_norm_output_ = final_norm_->forward(x);
    
    // Apply output projection
    Tensor output = output_projection_->forward(final_norm_output_);
    
    return output;
}

Tensor GPTModel::backward(const Tensor& grad_output) {
    // Gradient flows back through output projection
    Tensor grad_final_norm = output_projection_->backward(grad_output);
    
    // Gradient flows back through final layer norm
    Tensor grad_last_layer = final_norm_->backward(grad_final_norm);
    
    // Gradient flows back through transformer blocks (in reverse order)
    Tensor grad_x = grad_last_layer;
    
    for (int i = num_layers_ - 1; i >= 0; --i) {
        grad_x = transformer_blocks_[i]->backward(grad_x);
    }
    
    // Gradient flows back through embeddings
    // Note: This is simplified - in practice, embedding gradients need special handling
    position_embedding_->backward(grad_x);
    Tensor grad_input = token_embedding_->backward(grad_x);
    
    return grad_input;
}

std::vector<Tensor> GPTModel::parameters() const {
    std::vector<Tensor> params;
    
    // Add embedding parameters
    auto token_params = token_embedding_->parameters();
    params.insert(params.end(), token_params.begin(), token_params.end());
    
    auto pos_params = position_embedding_->parameters();
    params.insert(params.end(), pos_params.begin(), pos_params.end());
    
    // Add transformer block parameters
    for (const auto& block : transformer_blocks_) {
        auto block_params = block->parameters();
        params.insert(params.end(), block_params.begin(), block_params.end());
    }
    
    // Add final norm parameters
    auto norm_params = final_norm_->parameters();
    params.insert(params.end(), norm_params.begin(), norm_params.end());
    
    // Add output projection parameters
    auto output_params = output_projection_->parameters();
    params.insert(params.end(), output_params.begin(), output_params.end());
    
    return params;
}

std::vector<Tensor> GPTModel::gradients() const {
    std::vector<Tensor> grads;
    
    // Add embedding gradients
    auto token_grads = token_embedding_->gradients();
    grads.insert(grads.end(), token_grads.begin(), token_grads.end());
    
    auto pos_grads = position_embedding_->gradients();
    grads.insert(grads.end(), pos_grads.begin(), pos_grads.end());
    
    // Add transformer block gradients
    for (const auto& block : transformer_blocks_) {
        auto block_grads = block->gradients();
        grads.insert(grads.end(), block_grads.begin(), block_grads.end());
    }
    
    // Add final norm gradients
    auto norm_grads = final_norm_->gradients();
    grads.insert(grads.end(), norm_grads.begin(), norm_grads.end());
    
    // Add output projection gradients
    auto output_grads = output_projection_->gradients();
    grads.insert(grads.end(), output_grads.begin(), output_grads.end());
    
    return grads;
}

void GPTModel::zero_grad() {
    // Zero gradients for all components
    token_embedding_->zero_grad();
    position_embedding_->zero_grad();
    
    for (const auto& block : transformer_blocks_) {
        block->zero_grad();
    }
    
    final_norm_->zero_grad();
    output_projection_->zero_grad();
}

Tensor GPTModel::create_position_ids(const Tensor& input) const {
    // Input shape: [batch_size, seq_len]
    // Output shape: [batch_size, seq_len] with values 0, 1, 2, ..., seq_len-1
    
    Tensor position_ids(input.shape());
    
    for (int i = 0; i < input.shape()[0]; ++i) {  // batch_size
        for (int j = 0; j < input.shape()[1]; ++j) {  // seq_len
            position_ids[i * input.shape()[1] + j] = j;
        }
    }
    
    return position_ids;
}

} // namespace megatron