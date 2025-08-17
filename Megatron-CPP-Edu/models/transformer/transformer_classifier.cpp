#include "transformer_classifier.h"
#include <iostream>

namespace megatron {

TransformerClassifier::TransformerClassifier(int vocab_size, int max_seq_len, int embed_dim, 
                                           int num_heads, int num_layers, int ff_dim, int num_classes, 
                                           bool use_dropout, float dropout_prob, const std::string& name)
    : Layer(name),
      vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      num_layers_(num_layers),
      ff_dim_(ff_dim),
      num_classes_(num_classes),
      use_dropout_(use_dropout),
      dropout_prob_(dropout_prob),
      use_cls_token_(true) {  // Default to using [CLS] token
    
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
    
    // Create classification head
    classifier_ = std::make_shared<Linear>(embed_dim, num_classes, true, name + "_classifier");
}

Tensor TransformerClassifier::forward(const Tensor& input) {
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
    
    // Extract final representation
    if (use_cls_token_) {
        final_representation_ = extract_cls_token(x);
    } else {
        final_representation_ = mean_pool(x);
    }
    
    // Apply classification head
    logits_ = classifier_->forward(final_representation_);
    
    return logits_;
}

Tensor TransformerClassifier::backward(const Tensor& grad_output) {
    // Gradient flows back through classifier
    Tensor grad_final_representation = classifier_->backward(grad_output);
    
    // Gradient flows back through pooling
    Tensor grad_last_layer;
    if (use_cls_token_) {
        // For [CLS] token, gradient only flows to the first token position
        grad_last_layer = Tensor(layer_outputs_.back().shape());
        grad_last_layer.zeros();
        
        // Copy gradient to [CLS] token position (assuming it's the first token)
        for (int i = 0; i < grad_final_representation.shape()[0]; ++i) {
            for (int j = 0; j < grad_final_representation.shape()[1]; ++j) {
                grad_last_layer[i * layer_outputs_.back().shape()[1] + j] = 
                    grad_final_representation[i * grad_final_representation.shape()[1] + j];
            }
        }
    } else {
        // For mean pooling, distribute gradient equally
        int seq_len = layer_outputs_.back().shape()[1];
        grad_last_layer = Tensor(layer_outputs_.back().shape());
        
        for (int i = 0; i < grad_final_representation.shape()[0]; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                for (int k = 0; k < grad_final_representation.shape()[1]; ++k) {
                    grad_last_layer[i * seq_len * embed_dim_ + j * embed_dim_ + k] = 
                        grad_final_representation[i * grad_final_representation.shape()[1] + k] / seq_len;
                }
            }
        }
    }
    
    // Gradient flows back through transformer blocks (in reverse order)
    Tensor grad_x = grad_last_layer;
    
    for (int i = num_layers_ - 1; i >= 0; --i) {
        grad_x = transformer_blocks_[i]->backward(grad_x);
    }
    
    // Gradient flows back through embeddings
    position_embedding_->backward(grad_x);
    Tensor grad_input = token_embedding_->backward(grad_x);
    
    return grad_input;
}

std::vector<Tensor> TransformerClassifier::parameters() const {
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
    
    // Add classifier parameters
    auto classifier_params = classifier_->parameters();
    params.insert(params.end(), classifier_params.begin(), classifier_params.end());
    
    return params;
}

std::vector<Tensor> TransformerClassifier::gradients() const {
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
    
    // Add classifier gradients
    auto classifier_grads = classifier_->gradients();
    grads.insert(grads.end(), classifier_grads.begin(), classifier_grads.end());
    
    return grads;
}

void TransformerClassifier::zero_grad() {
    // Zero gradients for all components
    token_embedding_->zero_grad();
    position_embedding_->zero_grad();
    
    for (const auto& block : transformer_blocks_) {
        block->zero_grad();
    }
    
    classifier_->zero_grad();
}

Tensor TransformerClassifier::create_position_ids(const Tensor& input) const {
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

Tensor TransformerClassifier::extract_cls_token(const Tensor& hidden_states) const {
    // Extract the first token ([CLS] token) from each sequence
    // Input shape: [batch_size, seq_len, embed_dim]
    // Output shape: [batch_size, embed_dim]
    
    Tensor cls_representation({hidden_states.shape()[0], hidden_states.shape()[2]});
    
    for (int i = 0; i < hidden_states.shape()[0]; ++i) {
        for (int j = 0; j < hidden_states.shape()[2]; ++j) {
            cls_representation[i * hidden_states.shape()[2] + j] = 
                hidden_states[i * hidden_states.shape()[1] * hidden_states.shape()[2] + j];
        }
    }
    
    return cls_representation;
}

Tensor TransformerClassifier::mean_pool(const Tensor& hidden_states) const {
    // Mean pooling across sequence length
    // Input shape: [batch_size, seq_len, embed_dim]
    // Output shape: [batch_size, embed_dim]
    
    Tensor pooled({hidden_states.shape()[0], hidden_states.shape()[2]});
    pooled.zeros();
    
    int seq_len = hidden_states.shape()[1];
    
    for (int i = 0; i < hidden_states.shape()[0]; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            for (int k = 0; k < hidden_states.shape()[2]; ++k) {
                pooled[i * hidden_states.shape()[2] + k] += 
                    hidden_states[i * seq_len * hidden_states.shape()[2] + j * hidden_states.shape()[2] + k];
            }
        }
        
        // Divide by sequence length
        for (int k = 0; k < hidden_states.shape()[2]; ++k) {
            pooled[i * hidden_states.shape()[2] + k] /= seq_len;
        }
    }
    
    return pooled;
}

} // namespace megatron