#pragma once

#include "core/layers/layer.h"
#include "core/layers/embedding.h"
#include "core/layers/transformer_block.h"
#include "core/layers/linear.h"
#include "core/layers/layer_norm.h"
#include <memory>
#include <vector>

namespace megatron {

class TransformerClassifier : public Layer {
public:
    TransformerClassifier(int vocab_size, int max_seq_len, int embed_dim, int num_heads, 
                          int num_layers, int ff_dim, int num_classes, bool use_dropout = false, 
                          float dropout_prob = 0.1f, const std::string& name = "transformer_classifier");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    void zero_grad() override;
    
    // Get the final representation (before classification head)
    const Tensor& get_final_representation() const { return final_representation_; }
    
    // Get all transformer block outputs
    const std::vector<Tensor>& get_layer_outputs() const { return layer_outputs_; }
    
    // Get classification logits
    const Tensor& get_logits() const { return logits_; }

private:
    int vocab_size_;
    int max_seq_len_;
    int embed_dim_;
    int num_heads_;
    int num_layers_;
    int ff_dim_;
    int num_classes_;
    bool use_dropout_;
    float dropout_prob_;
    
    // Token and position embeddings
    std::shared_ptr<Embedding> token_embedding_;
    std::shared_ptr<Embedding> position_embedding_;
    
    // Transformer blocks
    std::vector<std::shared_ptr<TransformerBlock>> transformer_blocks_;
    
    // Pooling strategy (using [CLS] token representation)
    bool use_cls_token_;
    
    // Classification head
    std::shared_ptr<Linear> classifier_;
    
    // Cache for forward/backward pass
    Tensor input_;
    Tensor embedded_input_;
    Tensor final_representation_;
    std::vector<Tensor> layer_outputs_;
    Tensor logits_;
    
    // Helper method to create position IDs
    Tensor create_position_ids(const Tensor& input) const;
    
    // Helper method to extract [CLS] token representation
    Tensor extract_cls_token(const Tensor& hidden_states) const;
    
    // Helper method for mean pooling
    Tensor mean_pool(const Tensor& hidden_states) const;
};

} // namespace megatron