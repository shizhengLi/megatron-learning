#pragma once

#include "core/layers/layer.h"
#include "core/layers/embedding.h"
#include "core/layers/transformer_block.h"
#include "core/layers/linear.h"
#include "core/layers/layer_norm.h"
#include <memory>
#include <vector>

namespace megatron {

class GPTModel : public Layer {
public:
    GPTModel(int vocab_size, int max_seq_len, int embed_dim, int num_heads, 
             int num_layers, int ff_dim, bool use_dropout = false, 
             float dropout_prob = 0.1f, const std::string& name = "gpt");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    void zero_grad() override;
    
    // Get the final layer norm output
    const Tensor& get_final_norm() const { return final_norm_output_; }
    
    // Get all transformer block outputs
    const std::vector<Tensor>& get_layer_outputs() const { return layer_outputs_; }

private:
    int vocab_size_;
    int max_seq_len_;
    int embed_dim_;
    int num_heads_;
    int num_layers_;
    int ff_dim_;
    bool use_dropout_;
    float dropout_prob_;
    
    // Token and position embeddings
    std::shared_ptr<Embedding> token_embedding_;
    std::shared_ptr<Embedding> position_embedding_;
    
    // Transformer blocks
    std::vector<std::shared_ptr<TransformerBlock>> transformer_blocks_;
    
    // Final layer norm
    std::shared_ptr<LayerNorm> final_norm_;
    
    // Output projection (language model head)
    std::shared_ptr<Linear> output_projection_;
    
    // Cache for forward/backward pass
    Tensor input_;
    Tensor embedded_input_;
    Tensor final_norm_output_;
    std::vector<Tensor> layer_outputs_;
    
    // Helper method to create position IDs
    Tensor create_position_ids(const Tensor& input) const;
};

} // namespace megatron