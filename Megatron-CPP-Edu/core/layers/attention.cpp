#include "attention.h"
#include <cmath>
#include <algorithm>

namespace megatron {

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, bool dropout, float dropout_prob, const std::string& name)
    : Layer(name),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      use_dropout_(dropout),
      dropout_prob_(dropout_prob),
      q_proj_(embed_dim, embed_dim, true, name + "_q_proj"),
      k_proj_(embed_dim, embed_dim, true, name + "_k_proj"),
      v_proj_(embed_dim, embed_dim, true, name + "_v_proj"),
      out_proj_(embed_dim, embed_dim, true, name + "_out_proj"),
      dropout_(dropout_prob, name + "_dropout") {
    
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
}

Tensor MultiHeadAttention::forward(const Tensor& input) {
    // Input shape: [batch_size, seq_len, embed_dim]
    input_ = input;
    
    // Project input to Q, K, V
    // Each projection: [batch_size, seq_len, embed_dim]
    Tensor q = q_proj_.forward(input);
    Tensor k = k_proj_.forward(input);
    Tensor v = v_proj_.forward(input);
    
    // Cache projections for backward pass
    q_proj_cache_ = q;
    k_proj_cache_ = k;
    v_proj_cache_ = v;
    
    // Split heads
    // After split: [batch_size, num_heads, seq_len, head_dim]
    Tensor q_split = split_heads(q);
    Tensor k_split = split_heads(k);
    Tensor v_split = split_heads(v);
    
    // Compute scaled dot-product attention
    // Output: [batch_size, num_heads, seq_len, head_dim]
    Tensor attention_output = scaled_dot_product_attention(q_split, k_split, v_split);
    
    // Combine heads
    // After combine: [batch_size, seq_len, embed_dim]
    Tensor combined = combine_heads(attention_output);
    
    // Apply dropout if needed
    if (use_dropout_) {
        combined = dropout_.forward(combined);
    }
    
    // Final output projection
    output_cache_ = out_proj_.forward(combined);
    
    return output_cache_;
}

Tensor MultiHeadAttention::backward(const Tensor& grad_output) {
    // grad_output shape: [batch_size, seq_len, embed_dim]
    
    // Compute gradients
    compute_attention_gradients(grad_output);
    
    // Backward through output projection
    Tensor grad_combined = out_proj_.backward(grad_output);
    
    // Backward through dropout
    if (use_dropout_) {
        grad_combined = dropout_.backward(grad_combined);
    }
    
    // Backward through attention (simplified - requires full gradient computation)
    // For now, we'll compute gradients for the projections
    Tensor grad_v = combine_heads(grad_combined);
    Tensor grad_k = combine_heads(grad_combined);
    Tensor grad_q = combine_heads(grad_combined);
    
    // Backward through projections
    Tensor grad_input_q = q_proj_.backward(grad_q);
    Tensor grad_input_k = k_proj_.backward(grad_k);
    Tensor grad_input_v = v_proj_.backward(grad_v);
    
    // Combine gradients from all projections
    Tensor grad_input = grad_input_q + grad_input_k + grad_input_v;
    
    return grad_input;
}

std::vector<Tensor> MultiHeadAttention::parameters() const {
    std::vector<Tensor> params;
    
    // Add parameters from all linear layers
    auto q_params = q_proj_.parameters();
    auto k_params = k_proj_.parameters();
    auto v_params = v_proj_.parameters();
    auto out_params = out_proj_.parameters();
    
    params.insert(params.end(), q_params.begin(), q_params.end());
    params.insert(params.end(), k_params.begin(), k_params.end());
    params.insert(params.end(), v_params.begin(), v_params.end());
    params.insert(params.end(), out_params.begin(), out_params.end());
    
    return params;
}

std::vector<Tensor> MultiHeadAttention::gradients() const {
    std::vector<Tensor> grads;
    
    // Add gradients from all linear layers
    auto q_grads = q_proj_.gradients();
    auto k_grads = k_proj_.gradients();
    auto v_grads = v_proj_.gradients();
    auto out_grads = out_proj_.gradients();
    
    grads.insert(grads.end(), q_grads.begin(), q_grads.end());
    grads.insert(grads.end(), k_grads.begin(), k_grads.end());
    grads.insert(grads.end(), v_grads.begin(), v_grads.end());
    grads.insert(grads.end(), out_grads.begin(), out_grads.end());
    
    return grads;
}

void MultiHeadAttention::zero_grad() {
    q_proj_.zero_grad();
    k_proj_.zero_grad();
    v_proj_.zero_grad();
    out_proj_.zero_grad();
}

Tensor MultiHeadAttention::split_heads(const Tensor& x) {
    // Input shape: [batch_size, seq_len, embed_dim]
    // Output shape: [batch_size, num_heads, seq_len, head_dim]
    
    int batch_size = x.shape()[0];
    int seq_len = x.shape()[1];
    
    // Reshape to [batch_size, seq_len, num_heads, head_dim]
    Tensor reshaped = x.view({batch_size, seq_len, num_heads_, head_dim_});
    
    // Transpose to [batch_size, num_heads, seq_len, head_dim]
    Tensor result({batch_size, num_heads_, seq_len, head_dim_});
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < head_dim_; ++d) {
                    int src_idx = ((b * seq_len + s) * num_heads_ + h) * head_dim_ + d;
                    int dst_idx = ((b * num_heads_ + h) * seq_len + s) * head_dim_ + d;
                    result[dst_idx] = reshaped[src_idx];
                }
            }
        }
    }
    
    return result;
}

Tensor MultiHeadAttention::combine_heads(const Tensor& x) {
    // Input shape: [batch_size, num_heads, seq_len, head_dim]
    // Output shape: [batch_size, seq_len, embed_dim]
    
    int batch_size = x.shape()[0];
    int seq_len = x.shape()[2];
    
    // Transpose to [batch_size, seq_len, num_heads, head_dim]
    Tensor transposed({batch_size, seq_len, num_heads_, head_dim_});
    
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads_; ++h) {
                for (int d = 0; d < head_dim_; ++d) {
                    int src_idx = ((b * num_heads_ + h) * seq_len + s) * head_dim_ + d;
                    int dst_idx = ((b * seq_len + s) * num_heads_ + h) * head_dim_ + d;
                    transposed[dst_idx] = x[src_idx];
                }
            }
        }
    }
    
    // Reshape to [batch_size, seq_len, embed_dim]
    return transposed.view({batch_size, seq_len, embed_dim_});
}

Tensor MultiHeadAttention::scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v) {
    // Input shapes: [batch_size, num_heads, seq_len, head_dim]
    // Output shape: [batch_size, num_heads, seq_len, head_dim]
    
    int batch_size = q.shape()[0];
    int seq_len = q.shape()[2];
    
    // Compute attention scores: Q @ K.T / sqrt(d_k)
    // Scores shape: [batch_size, num_heads, seq_len, seq_len]
    Tensor scores({batch_size, num_heads_, seq_len, seq_len});
    scores.zeros();
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim_; ++d) {
                        int q_idx = ((b * num_heads_ + h) * seq_len + i) * head_dim_ + d;
                        int k_idx = ((b * num_heads_ + h) * seq_len + j) * head_dim_ + d;
                        score += q[q_idx] * k[k_idx];
                    }
                    int score_idx = ((b * num_heads_ + h) * seq_len + i) * seq_len + j;
                    scores[score_idx] = score / std::sqrt(static_cast<float>(head_dim_));
                }
            }
        }
    }
    
    // Apply softmax to get attention weights
    attention_weights_ = scores.softmax(3);  // Softmax along last dimension
    
    // Apply attention weights to values
    Tensor output({batch_size, num_heads_, seq_len, head_dim_});
    output.zeros();
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                for (int d = 0; d < head_dim_; ++d) {
                    float value = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        int attn_idx = ((b * num_heads_ + h) * seq_len + i) * seq_len + j;
                        int v_idx = ((b * num_heads_ + h) * seq_len + j) * head_dim_ + d;
                        value += attention_weights_[attn_idx] * v[v_idx];
                    }
                    int out_idx = ((b * num_heads_ + h) * seq_len + i) * head_dim_ + d;
                    output[out_idx] = value;
                }
            }
        }
    }
    
    return output;
}

void MultiHeadAttention::compute_attention_gradients(const Tensor& grad_output) {
    // This is a simplified gradient computation for attention
    // In a full implementation, this would compute gradients for Q, K, V matrices
    
    // For now, we'll let the linear layers handle their own gradients
    // The actual attention gradient computation is quite complex and would require
    // implementing the full backward pass through the attention mechanism
    
    // This is a placeholder - in practice, you would compute:
    // 1. Gradients for value projections
    // 2. Gradients for key projections  
    // 3. Gradients for query projections
    // 4. Update attention weight gradients
}

} // namespace megatron