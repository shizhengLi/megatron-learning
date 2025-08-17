#include "tensor_parallel.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

namespace megatron {

// TensorParallelContext implementation
TensorParallelContext& TensorParallelContext::instance() {
    static TensorParallelContext instance_;
    return instance_;
}

void TensorParallelContext::initialize(int world_size, int rank) {
    world_size_ = world_size;
    rank_ = rank;
}

int TensorParallelContext::get_local_output_dim(int global_output_dim) const {
    if (!is_enabled()) {
        return global_output_dim;
    }
    
    int base_size = global_output_dim / world_size_;
    int remainder = global_output_dim % world_size_;
    
    if (rank_ < remainder) {
        return base_size + 1;
    } else {
        return base_size;
    }
}

int TensorParallelContext::get_local_input_dim(int global_input_dim) const {
    // 对于行并行，输入维度分割方式相同
    return get_local_output_dim(global_input_dim);
}

// ColumnParallelLinear implementation
ColumnParallelLinear::ColumnParallelLinear(int in_features, int out_features, bool bias,
                                         const std::string& name)
    : Layer(name), in_features_(in_features), out_features_(out_features),
      has_bias_(bias), name_(name) {
    
    initialize_parameters();
}

Tensor ColumnParallelLinear::forward(const Tensor& input) {
    input_cache_ = input;
    
    // 本地矩阵乘法: input @ weight.T
    Tensor local_output = input.matmul(weight_.transpose());
    
    // 如果是张量并行，需要all-reduce输出
    if (TensorParallelContext::instance().is_enabled()) {
        all_reduce_output(local_output);
    }
    
    // 添加bias
    if (has_bias_) {
        for (int i = 0; i < local_output.shape()[0]; ++i) {
            for (int j = 0; j < local_output.shape()[1]; ++j) {
                local_output[i * local_output.shape()[1] + j] += bias_[j];
            }
        }
    }
    
    return local_output;
}

Tensor ColumnParallelLinear::backward(const Tensor& grad_output) {
    // 计算权重梯度: grad_output.T @ input
    Tensor grad_weight = grad_output.transpose().matmul(input_cache_);
    weight_grad_ = grad_weight;
    
    // 计算bias梯度
    if (has_bias_) {
        Tensor grad_bias = grad_output.sum(0);
        bias_grad_ = grad_bias;
    }
    
    // 计算输入梯度: grad_output @ weight
    Tensor grad_input = grad_output.matmul(weight_);
    
    // 如果是张量并行，需要all-reduce输入梯度
    if (TensorParallelContext::instance().is_enabled()) {
        all_reduce_grad_input(grad_input);
    }
    
    return grad_input;
}

std::vector<Tensor> ColumnParallelLinear::parameters() const {
    std::vector<Tensor> params;
    params.push_back(weight_);
    if (has_bias_) {
        params.push_back(bias_);
    }
    return params;
}

std::vector<Tensor> ColumnParallelLinear::gradients() const {
    std::vector<Tensor> grads;
    grads.push_back(weight_grad_);
    if (has_bias_) {
        grads.push_back(bias_grad_);
    }
    return grads;
}

std::vector<int> ColumnParallelLinear::get_global_weight_shape() const {
    return {out_features_, in_features_};
}

void ColumnParallelLinear::all_reduce_output(Tensor& output) {
    auto& comm = MPICommunicator::instance();
    comm.all_reduce(output);
}

void ColumnParallelLinear::all_reduce_grad_input(Tensor& grad_input) {
    auto& comm = MPICommunicator::instance();
    comm.all_reduce(grad_input);
}

void ColumnParallelLinear::initialize_parameters() {
    auto& tp_ctx = TensorParallelContext::instance();
    
    // 计算本地输出维度
    int local_out_features = tp_ctx.get_local_output_dim(out_features_);
    
    // 初始化权重: [local_out_features, in_features]
    weight_ = Tensor({local_out_features, in_features_});
    weight_grad_ = Tensor({local_out_features, in_features_});
    
    // Kaiming He初始化
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / in_features_);
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (int i = 0; i < weight_.size(); ++i) {
        weight_[i] = dist(gen);
    }
    
    weight_grad_.zeros();
    
    // 初始化bias
    if (has_bias_) {
        bias_ = Tensor({local_out_features});
        bias_grad_ = Tensor({local_out_features});
        bias_.zeros();
        bias_grad_.zeros();
    }
}

// RowParallelLinear implementation
RowParallelLinear::RowParallelLinear(int in_features, int out_features, bool bias,
                                     const std::string& name)
    : Layer(name), in_features_(in_features), out_features_(out_features),
      has_bias_(bias), name_(name) {
    
    initialize_parameters();
}

Tensor RowParallelLinear::forward(const Tensor& input) {
    input_cache_ = input;
    
    Tensor local_input = input;
    
    // 如果是张量并行，需要all-gather输入
    if (TensorParallelContext::instance().is_enabled()) {
        all_gather_input(local_input);
    }
    
    // 本地矩阵乘法: local_input @ weight.T
    Tensor local_output = local_input.matmul(weight_.transpose());
    
    // 添加bias（只在最后一个设备上添加）
    if (has_bias_ && TensorParallelContext::instance().rank() == TensorParallelContext::instance().world_size() - 1) {
        for (int i = 0; i < local_output.shape()[0]; ++i) {
            for (int j = 0; j < local_output.shape()[1]; ++j) {
                local_output[i * local_output.shape()[1] + j] += bias_[j];
            }
        }
    }
    
    return local_output;
}

Tensor RowParallelLinear::backward(const Tensor& grad_output) {
    // 计算权重梯度: grad_output.T @ input_cache_
    Tensor grad_weight = grad_output.transpose().matmul(input_cache_);
    
    // 如果是张量并行，需要reduce权重梯度
    if (TensorParallelContext::instance().is_enabled()) {
        reduce_grad_weight(grad_weight);
    }
    weight_grad_ = grad_weight;
    
    // 计算bias梯度（只在最后一个设备上计算）
    if (has_bias_ && TensorParallelContext::instance().rank() == TensorParallelContext::instance().world_size() - 1) {
        Tensor grad_bias = grad_output.sum(0);
        bias_grad_ = grad_bias;
    }
    
    // 计算输入梯度: grad_output @ weight
    Tensor grad_input = grad_output.matmul(weight_);
    
    return grad_input;
}

std::vector<Tensor> RowParallelLinear::parameters() const {
    std::vector<Tensor> params;
    params.push_back(weight_);
    if (has_bias_) {
        params.push_back(bias_);
    }
    return params;
}

std::vector<Tensor> RowParallelLinear::gradients() const {
    std::vector<Tensor> grads;
    grads.push_back(weight_grad_);
    if (has_bias_) {
        grads.push_back(bias_grad_);
    }
    return grads;
}

std::vector<int> RowParallelLinear::get_global_weight_shape() const {
    return {out_features_, in_features_};
}

void RowParallelLinear::all_gather_input(Tensor& input) {
    // 简化实现：这里应该执行all-gather操作
    // 实际实现需要根据张量并行的具体策略来收集输入
}

void RowParallelLinear::reduce_grad_weight(Tensor& grad_weight) {
    auto& comm = MPICommunicator::instance();
    comm.reduce(grad_weight, 0);  // reduce到主进程
}

void RowParallelLinear::initialize_parameters() {
    auto& tp_ctx = TensorParallelContext::instance();
    
    // 计算本地输入维度
    int local_in_features = tp_ctx.get_local_input_dim(in_features_);
    
    // 初始化权重: [out_features, local_in_features]
    weight_ = Tensor({out_features_, local_in_features});
    weight_grad_ = Tensor({out_features_, local_in_features});
    
    // Kaiming He初始化
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / local_in_features);
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (int i = 0; i < weight_.size(); ++i) {
        weight_[i] = dist(gen);
    }
    
    weight_grad_.zeros();
    
    // 初始化bias
    if (has_bias_) {
        bias_ = Tensor({out_features_});
        bias_grad_ = Tensor({out_features_});
        bias_.zeros();
        bias_grad_.zeros();
    }
}

// TensorParallelMultiHeadAttention implementation
TensorParallelMultiHeadAttention::TensorParallelMultiHeadAttention(int embed_dim, int num_heads,
                                                                   bool use_causal_mask,
                                                                   const std::string& name)
    : Layer(name), embed_dim_(embed_dim), num_heads_(num_heads),
      use_causal_mask_(use_causal_mask), name_(name) {
    
    head_dim_ = embed_dim / num_heads;
    initialize_parallel_layers();
}

Tensor TensorParallelMultiHeadAttention::forward(const Tensor& input) {
    return forward(input, input, input);
}

Tensor TensorParallelMultiHeadAttention::forward(const Tensor& query, const Tensor& key,
                                                  const Tensor& value) {
    input_cache_ = query;
    
    // 并行投影
    Tensor q = q_proj_->forward(query);
    Tensor k = k_proj_->forward(key);
    Tensor v = v_proj_->forward(value);
    
    // 缓存
    q_cache_ = q;
    k_cache_ = k;
    v_cache_ = v;
    
    // 注意力计算
    Tensor attention_output = scaled_dot_product_attention(q, k, v);
    
    // 并行输出投影
    Tensor output = out_proj_->forward(attention_output);
    
    return output;
}

Tensor TensorParallelMultiHeadAttention::backward(const Tensor& grad_output) {
    // 反向传播通过输出投影
    Tensor grad_attention = out_proj_->backward(grad_output);
    
    // 反向传播通过注意力机制（简化实现）
    Tensor grad_q = grad_attention;
    Tensor grad_k = grad_attention;
    Tensor grad_v = grad_attention;
    
    // 反向传播通过输入投影
    Tensor grad_query = q_proj_->backward(grad_q);
    Tensor grad_key = k_proj_->backward(grad_k);
    Tensor grad_value = v_proj_->backward(grad_v);
    
    // 合并梯度
    Tensor grad_input = grad_query;
    for (int i = 0; i < grad_input.size(); ++i) {
        grad_input[i] += grad_key[i] + grad_value[i];
    }
    
    return grad_input;
}

std::vector<Tensor> TensorParallelMultiHeadAttention::parameters() const {
    std::vector<Tensor> params;
    
    auto q_params = q_proj_->parameters();
    auto k_params = k_proj_->parameters();
    auto v_params = v_proj_->parameters();
    auto out_params = out_proj_->parameters();
    
    params.insert(params.end(), q_params.begin(), q_params.end());
    params.insert(params.end(), k_params.begin(), k_params.end());
    params.insert(params.end(), v_params.begin(), v_params.end());
    params.insert(params.end(), out_params.begin(), out_params.end());
    
    return params;
}

std::vector<Tensor> TensorParallelMultiHeadAttention::gradients() const {
    std::vector<Tensor> grads;
    
    auto q_grads = q_proj_->gradients();
    auto k_grads = k_proj_->gradients();
    auto v_grads = v_proj_->gradients();
    auto out_grads = out_proj_->gradients();
    
    grads.insert(grads.end(), q_grads.begin(), q_grads.end());
    grads.insert(grads.end(), k_grads.begin(), k_grads.end());
    grads.insert(grads.end(), v_grads.begin(), v_grads.end());
    grads.insert(grads.end(), out_grads.begin(), out_grads.end());
    
    return grads;
}

Tensor TensorParallelMultiHeadAttention::scaled_dot_product_attention(const Tensor& q,
                                                                      const Tensor& k,
                                                                      const Tensor& v) {
    // 计算注意力分数: Q @ K.T / sqrt(d_k)
    Tensor scores = q.matmul(k.transpose());
    
    // 缩放
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    for (int i = 0; i < scores.size(); ++i) {
        scores[i] *= scale;
    }
    
    // 因果掩码
    if (use_causal_mask_) {
        int seq_len = q.shape()[1];
        scores = causal_mask(scores, seq_len);
    }
    
    // Softmax
    Tensor attention_weights = scores.softmax(-1);
    attention_weights_cache_ = attention_weights;
    
    // 应用到V: attention_weights @ V
    Tensor output = attention_weights.matmul(v);
    
    return output;
}

Tensor TensorParallelMultiHeadAttention::causal_mask(const Tensor& attention_scores,
                                                     int seq_len) {
    Tensor masked_scores = attention_scores;
    
    // 创建因果掩码
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j > i) {
                masked_scores[i * seq_len + j] = -1e9f;
            }
        }
    }
    
    return masked_scores;
}

Tensor TensorParallelMultiHeadAttention::split_heads(const Tensor& x) {
    // 将输入分割为多个头
    // 简化实现：假设输入已经是正确的形状
    return x;
}

Tensor TensorParallelMultiHeadAttention::combine_heads(const Tensor& x) {
    // 合并多个头的输出
    // 简化实现：假设输入已经是正确的形状
    return x;
}

void TensorParallelMultiHeadAttention::initialize_parallel_layers() {
    int head_dim = embed_dim_ / num_heads_;
    
    // 使用列并行进行Q、K、V投影
    q_proj_ = std::make_shared<ColumnParallelLinear>(embed_dim_, embed_dim_, true, "q_proj");
    k_proj_ = std::make_shared<ColumnParallelLinear>(embed_dim_, embed_dim_, true, "k_proj");
    v_proj_ = std::make_shared<ColumnParallelLinear>(embed_dim_, embed_dim_, true, "v_proj");
    
    // 使用行并行进行输出投影
    out_proj_ = std::make_shared<RowParallelLinear>(embed_dim_, embed_dim_, true, "out_proj");
}

// TensorParallelFFN implementation
TensorParallelFFN::TensorParallelFFN(int embed_dim, int ffn_dim, float dropout,
                                     const std::string& name)
    : Layer(name), embed_dim_(embed_dim), ffn_dim_(ffn_dim), dropout_(dropout), name_(name) {
    
    // 使用列并行进行第一个线性层
    linear1_ = std::make_shared<ColumnParallelLinear>(embed_dim_, ffn_dim, true, "ffn_linear1");
    
    // 使用行并行进行第二个线性层
    linear2_ = std::make_shared<RowParallelLinear>(ffn_dim, embed_dim, true, "ffn_linear2");
    
    dropout_layer_ = std::make_shared<Dropout>(dropout_);
}

Tensor TensorParallelFFN::forward(const Tensor& input) {
    input_cache_ = input;
    
    // FFN: Linear -> GELU -> Dropout -> Linear
    Tensor hidden = linear1_->forward(input);
    hidden = gelu(hidden);
    hidden_cache_ = hidden;
    
    if (is_training()) {
        hidden = dropout_layer_->forward(hidden);
    }
    
    Tensor output = linear2_->forward(hidden);
    
    return output;
}

Tensor TensorParallelFFN::backward(const Tensor& grad_output) {
    // 反向传播通过第二个线性层
    Tensor grad_hidden = linear2_->backward(grad_output);
    
    // 反向传播通过dropout
    if (is_training()) {
        grad_hidden = dropout_layer_->backward(grad_hidden);
    }
    
    // 反向传播通过GELU
    grad_hidden = gelu_backward(hidden_cache_, grad_hidden);
    
    // 反向传播通过第一个线性层
    Tensor grad_input = linear1_->backward(grad_hidden);
    
    return grad_input;
}

std::vector<Tensor> TensorParallelFFN::parameters() const {
    std::vector<Tensor> params;
    
    auto l1_params = linear1_->parameters();
    auto l2_params = linear2_->parameters();
    
    params.insert(params.end(), l1_params.begin(), l1_params.end());
    params.insert(params.end(), l2_params.begin(), l2_params.end());
    
    return params;
}

std::vector<Tensor> TensorParallelFFN::gradients() const {
    std::vector<Tensor> grads;
    
    auto l1_grads = linear1_->gradients();
    auto l2_grads = linear2_->gradients();
    
    grads.insert(grads.end(), l1_grads.begin(), l1_grads.end());
    grads.insert(grads.end(), l2_grads.begin(), l2_grads.end());
    
    return grads;
}

Tensor TensorParallelFFN::gelu(const Tensor& x) {
    Tensor result(x.shape());
    
    for (int i = 0; i < x.size(); ++i) {
        float val = x[i];
        result[i] = 0.5f * val * (1.0f + std::tanh(0.7978845608028654f * (val + 0.044715f * val * val * val)));
    }
    
    return result;
}

Tensor TensorParallelFFN::gelu_backward(const Tensor& x, const Tensor& grad_output) {
    Tensor grad_input(x.shape());
    
    for (int i = 0; i < x.size(); ++i) {
        float val = x[i];
        float tanh_arg = 0.7978845608028654f * (val + 0.044715f * val * val * val);
        float tanh_val = std::tanh(tanh_arg);
        float sech_sq = 1.0f - tanh_val * tanh_val;
        
        float gelu_grad = 0.5f * (1.0f + tanh_val) + 
                         0.5f * val * sech_sq * 0.7978845608028654f * (1.0f + 0.134145f * val * val);
        
        grad_input[i] = grad_output[i] * gelu_grad;
    }
    
    return grad_input;
}

// TensorParallelTransformerBlock implementation
TensorParallelTransformerBlock::TensorParallelTransformerBlock(int embed_dim, int num_heads,
                                                             int ffn_dim, float dropout,
                                                             const std::string& name)
    : Layer(name), embed_dim_(embed_dim), num_heads_(num_heads),
      ffn_dim_(ffn_dim), dropout_(dropout), name_(name) {
    
    // 注意力部分
    attn_norm_ = std::make_shared<LayerNorm>(embed_dim_, 1e-5f);
    attention_ = std::make_shared<TensorParallelMultiHeadAttention>(embed_dim_, num_heads_, true, "self_attention");
    attn_dropout_ = std::make_shared<Dropout>(dropout_);
    
    // FFN部分
    ffn_norm_ = std::make_shared<LayerNorm>(embed_dim_, 1e-5f);
    ffn_ = std::make_shared<TensorParallelFFN>(embed_dim_, ffn_dim_, dropout_, "ffn");
    ffn_dropout_ = std::make_shared<Dropout>(dropout_);
}

Tensor TensorParallelTransformerBlock::forward(const Tensor& input) {
    input_cache_ = input;
    
    // 注意力部分
    Tensor attn_norm = attn_norm_->forward(input);
    Tensor attn_output = attention_->forward(attn_norm);
    attn_output_cache_ = attn_output;
    
    if (is_training()) {
        attn_output = attn_dropout_->forward(attn_output);
    }
    
    Tensor attn_residual = input + attn_output;
    
    // FFN部分
    Tensor ffn_norm = ffn_norm_->forward(attn_residual);
    Tensor ffn_output = ffn_->forward(ffn_norm);
    ffn_output_cache_ = ffn_output;
    
    if (is_training()) {
        ffn_output = ffn_dropout_->forward(ffn_output);
    }
    
    Tensor output = attn_residual + ffn_output;
    
    return output;
}

Tensor TensorParallelTransformerBlock::backward(const Tensor& grad_output) {
    // 反向传播通过FFN残差连接
    Tensor grad_ffn_residual = grad_output;
    Tensor grad_ffn_output = grad_ffn_residual;
    
    if (is_training()) {
        grad_ffn_output = ffn_dropout_->backward(grad_ffn_output);
    }
    
    Tensor grad_ffn_norm = ffn_->backward(grad_ffn_output);
    Tensor grad_attn_residual = ffn_norm_->backward(grad_ffn_norm) + grad_ffn_residual;
    
    // 反向传播通过注意力残差连接
    Tensor grad_attn_output = grad_attn_residual;
    
    if (is_training()) {
        grad_attn_output = attn_dropout_->backward(grad_attn_output);
    }
    
    Tensor grad_attn_norm = attention_->backward(grad_attn_output);
    Tensor grad_input = attn_norm_->backward(grad_attn_norm) + grad_attn_residual;
    
    return grad_input;
}

std::vector<Tensor> TensorParallelTransformerBlock::parameters() const {
    std::vector<Tensor> params;
    
    auto attn_norm_params = attn_norm_->parameters();
    auto attn_params = attention_->parameters();
    auto ffn_norm_params = ffn_norm_->parameters();
    auto ffn_params = ffn_->parameters();
    
    params.insert(params.end(), attn_norm_params.begin(), attn_norm_params.end());
    params.insert(params.end(), attn_params.begin(), attn_params.end());
    params.insert(params.end(), ffn_norm_params.begin(), ffn_norm_params.end());
    params.insert(params.end(), ffn_params.begin(), ffn_params.end());
    
    return params;
}

std::vector<Tensor> TensorParallelTransformerBlock::gradients() const {
    std::vector<Tensor> grads;
    
    auto attn_norm_grads = attn_norm_->gradients();
    auto attn_grads = attention_->gradients();
    auto ffn_norm_grads = ffn_norm_->gradients();
    auto ffn_grads = ffn_->gradients();
    
    grads.insert(grads.end(), attn_norm_grads.begin(), attn_norm_grads.end());
    grads.insert(grads.end(), attn_grads.begin(), attn_grads.end());
    grads.insert(grads.end(), ffn_norm_grads.begin(), ffn_norm_grads.end());
    grads.insert(grads.end(), ffn_grads.begin(), ffn_grads.end());
    
    return grads;
}

// TensorParallelModelBuilder implementation
TensorParallelModelBuilder::TensorParallelModelBuilder(int tensor_parallel_size)
    : tensor_parallel_size_(tensor_parallel_size) {
}

void TensorParallelModelBuilder::set_vocab_size(int vocab_size) {
    vocab_size_ = vocab_size;
}

void TensorParallelModelBuilder::set_max_seq_len(int max_seq_len) {
    max_seq_len_ = max_seq_len;
}

void TensorParallelModelBuilder::set_embed_dim(int embed_dim) {
    embed_dim_ = embed_dim;
}

void TensorParallelModelBuilder::set_num_layers(int num_layers) {
    num_layers_ = num_layers;
}

void TensorParallelModelBuilder::set_num_heads(int num_heads) {
    num_heads_ = num_heads;
}

void TensorParallelModelBuilder::set_ffn_dim(int ffn_dim) {
    ffn_dim_ = ffn_dim;
}

void TensorParallelModelBuilder::set_dropout(float dropout) {
    dropout_ = dropout;
}

std::shared_ptr<Layer> TensorParallelModelBuilder::build_gpt_model() {
    // 创建一个简单的序列容器来表示GPT模型
    class GPTModel : public Layer {
    public:
        GPTModel(std::shared_ptr<TensorParallelEmbedding> embedding,
                 std::shared_ptr<TensorParallelLayerNorm> final_norm,
                 std::vector<std::shared_ptr<TensorParallelTransformerBlock>> blocks,
                 std::shared_ptr<RowParallelLinear> lm_head)
            : embedding_(embedding), final_norm_(final_norm), 
              blocks_(blocks), lm_head_(lm_head) {}
        
        Tensor forward(const Tensor& input) override {
            Tensor x = embedding_->forward(input);
            
            for (auto& block : blocks_) {
                x = block->forward(x);
            }
            
            x = final_norm_->forward(x);
            Tensor output = lm_head_->forward(x);
            
            return output;
        }
        
        Tensor backward(const Tensor& grad_output) override {
            Tensor grad = lm_head_->backward(grad_output);
            grad = final_norm_->backward(grad);
            
            for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
                grad = (*it)->backward(grad);
            }
            
            grad = embedding_->backward(grad);
            return grad;
        }
        
        std::vector<Tensor> parameters() const override {
            std::vector<Tensor> params;
            
            auto emb_params = embedding_->parameters();
            auto norm_params = final_norm_->parameters();
            auto head_params = lm_head_->parameters();
            
            params.insert(params.end(), emb_params.begin(), emb_params.end());
            params.insert(params.end(), norm_params.begin(), norm_params.end());
            params.insert(params.end(), head_params.begin(), head_params.end());
            
            for (auto& block : blocks_) {
                auto block_params = block->parameters();
                params.insert(params.end(), block_params.begin(), block_params.end());
            }
            
            return params;
        }
        
        std::vector<Tensor> gradients() const override {
            std::vector<Tensor> grads;
            
            auto emb_grads = embedding_->gradients();
            auto norm_grads = final_norm_->gradients();
            auto head_grads = lm_head_->gradients();
            
            grads.insert(grads.end(), emb_grads.begin(), emb_grads.end());
            grads.insert(grads.end(), norm_grads.begin(), norm_grads.end());
            grads.insert(grads.end(), head_grads.begin(), head_grads.end());
            
            for (auto& block : blocks_) {
                auto block_grads = block->gradients();
                grads.insert(grads.end(), block_grads.begin(), block_grads.end());
            }
            
            return grads;
        }
        
    private:
        std::shared_ptr<TensorParallelEmbedding> embedding_;
        std::shared_ptr<TensorParallelLayerNorm> final_norm_;
        std::vector<std::shared_ptr<TensorParallelTransformerBlock>> blocks_;
        std::shared_ptr<RowParallelLinear> lm_head_;
    };
    
    auto embedding = build_embedding();
    auto final_norm = build_layer_norm();
    auto blocks = build_transformer_blocks();
    auto lm_head = build_lm_head();
    
    return std::make_shared<GPTModel>(embedding, final_norm, blocks, lm_head);
}

std::shared_ptr<Layer> TensorParallelModelBuilder::build_transformer_classifier(int num_classes) {
    // 构建Transformer分类器（简化实现）
    class TransformerClassifier : public Layer {
    public:
        TransformerClassifier(std::shared_ptr<TensorParallelEmbedding> embedding,
                            std::vector<std::shared_ptr<TensorParallelTransformerBlock>> blocks,
                            std::shared_ptr<TensorParallelLayerNorm> final_norm,
                            std::shared_ptr<RowParallelLinear> classifier)
            : embedding_(embedding), blocks_(blocks), final_norm_(final_norm),
              classifier_(classifier) {}
        
        Tensor forward(const Tensor& input) override {
            Tensor x = embedding_->forward(input);
            
            for (auto& block : blocks_) {
                x = block->forward(x);
            }
            
            x = final_norm_->forward(x);
            
            // 全局平均池化
            Tensor pooled({x.shape()[0], x.shape()[2]});
            for (int i = 0; i < x.shape()[0]; ++i) {
                for (int j = 0; j < x.shape()[2]; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < x.shape()[1]; ++k) {
                        sum += x[i * x.shape()[1] * x.shape()[2] + k * x.shape()[2] + j];
                    }
                    pooled[i * x.shape()[2] + j] = sum / x.shape()[1];
                }
            }
            
            Tensor output = classifier_->forward(pooled);
            return output;
        }
        
        Tensor backward(const Tensor& grad_output) override {
            // 简化的反向传播实现
            Tensor grad = classifier_->backward(grad_output);
            // ... 完整的实现需要考虑池化操作的反向传播
            return grad;
        }
        
        std::vector<Tensor> parameters() const override {
            std::vector<Tensor> params;
            
            auto emb_params = embedding_->parameters();
            auto norm_params = final_norm_->parameters();
            auto classifier_params = classifier_->parameters();
            
            params.insert(params.end(), emb_params.begin(), emb_params.end());
            params.insert(params.end(), norm_params.begin(), norm_params.end());
            params.insert(params.end(), classifier_params.begin(), classifier_params.end());
            
            for (auto& block : blocks_) {
                auto block_params = block->parameters();
                params.insert(params.end(), block_params.begin(), block_params.end());
            }
            
            return params;
        }
        
        std::vector<Tensor> gradients() const override {
            std::vector<Tensor> grads;
            
            auto emb_grads = embedding_->gradients();
            auto norm_grads = final_norm_->gradients();
            auto classifier_grads = classifier_->gradients();
            
            grads.insert(grads.end(), emb_grads.begin(), emb_grads.end());
            grads.insert(grads.end(), norm_grads.begin(), norm_grads.end());
            grads.insert(grads.end(), classifier_grads.begin(), classifier_grads.end());
            
            for (auto& block : blocks_) {
                auto block_grads = block->gradients();
                grads.insert(grads.end(), block_grads.begin(), block_grads.end());
            }
            
            return grads;
        }
        
    private:
        std::shared_ptr<TensorParallelEmbedding> embedding_;
        std::vector<std::shared_ptr<TensorParallelTransformerBlock>> blocks_;
        std::shared_ptr<TensorParallelLayerNorm> final_norm_;
        std::shared_ptr<RowParallelLinear> classifier_;
    };
    
    auto embedding = build_embedding();
    auto final_norm = build_layer_norm();
    auto blocks = build_transformer_blocks();
    auto classifier = std::make_shared<RowParallelLinear>(embed_dim_, num_classes, true, "classifier");
    
    return std::make_shared<TransformerClassifier>(embedding, blocks, final_norm, classifier);
}

std::shared_ptr<TensorParallelEmbedding> TensorParallelModelBuilder::build_embedding() {
    return std::make_shared<TensorParallelEmbedding>(vocab_size_, embed_dim_, "token_embedding");
}

std::shared_ptr<TensorParallelLayerNorm> TensorParallelModelBuilder::build_layer_norm() {
    return std::make_shared<TensorParallelLayerNorm>(embed_dim_, 1e-5f, "final_norm");
}

std::vector<std::shared_ptr<TensorParallelTransformerBlock>> TensorParallelModelBuilder::build_transformer_blocks() {
    std::vector<std::shared_ptr<TensorParallelTransformerBlock>> blocks;
    
    for (int i = 0; i < num_layers_; ++i) {
        auto block = std::make_shared<TensorParallelTransformerBlock>(
            embed_dim_, num_heads_, ffn_dim_, dropout_, 
            "transformer_block_" + std::to_string(i)
        );
        blocks.push_back(block);
    }
    
    return blocks;
}

std::shared_ptr<RowParallelLinear> TensorParallelModelBuilder::build_lm_head() {
    return std::make_shared<RowParallelLinear>(embed_dim_, vocab_size_, true, "lm_head");
}

// tensor_parallel_utils implementation
namespace tensor_parallel_utils {

bool is_tensor_parallel_supported() {
    return distributed_utils::is_distributed_supported();
}

int get_tensor_parallel_world_size() {
    return TensorParallelContext::instance().world_size();
}

int get_tensor_parallel_rank() {
    return TensorParallelContext::instance().rank();
}

void initialize_tensor_parallel(int world_size, int rank) {
    TensorParallelContext::instance().initialize(world_size, rank);
}

void cleanup_tensor_parallel() {
    // 清理张量并行环境
}

Tensor split_tensor(const Tensor& tensor, int dim, int rank, int world_size) {
    // 分割张量的实现
    // 这里需要根据具体的分割策略来实现
    return tensor;  // 简化实现
}

Tensor concatenate_tensors(const std::vector<Tensor>& tensors, int dim) {
    // 合并张量的实现
    // 这里需要根据具体的合并策略来实现
    if (tensors.empty()) {
        return Tensor({});
    }
    return tensors[0];  // 简化实现
}

bool validate_tensor_parallel_config(int world_size, int rank) {
    if (world_size <= 0) {
        return false;
    }
    if (rank < 0 || rank >= world_size) {
        return false;
    }
    return true;
}

void print_tensor_parallel_info() {
    auto& tp_ctx = TensorParallelContext::instance();
    
    std::cout << "=== Tensor Parallel Configuration ===" << std::endl;
    std::cout << "World Size: " << tp_ctx.world_size() << std::endl;
    std::cout << "Rank: " << tp_ctx.rank() << std::endl;
    std::cout << "=====================================" << std::endl;
}

} // namespace tensor_parallel_utils

// Missing implementations for other classes
TensorParallelEmbedding::TensorParallelEmbedding(int vocab_size, int embedding_dim,
                                                 const std::string& name)
    : Layer(name), vocab_size_(vocab_size), embedding_dim_(embedding_dim), name_(name) {
    initialize_parameters();
}

Tensor TensorParallelEmbedding::forward(const Tensor& input) {
    input_cache_ = input;
    
    // 简化的嵌入查找实现
    Tensor output({input.shape()[0], input.shape()[1], embedding_dim_});
    
    for (int i = 0; i < input.shape()[0]; ++i) {
        for (int j = 0; j < input.shape()[1]; ++j) {
            int token_id = static_cast<int>(input[i * input.shape()[1] + j]);
            if (token_id >= 0 && token_id < vocab_size_) {
                for (int k = 0; k < embedding_dim_; ++k) {
                    output[i * input.shape()[1] * embedding_dim_ + j * embedding_dim_ + k] = 
                        weight_[token_id * embedding_dim_ + k];
                }
            }
        }
    }
    
    return output;
}

Tensor TensorParallelEmbedding::backward(const Tensor& grad_output) {
    // 简化的反向传播实现
    weight_grad_.zeros();
    
    for (int i = 0; i < input_cache_.shape()[0]; ++i) {
        for (int j = 0; j < input_cache_.shape()[1]; ++j) {
            int token_id = static_cast<int>(input_cache_[i * input_cache_.shape()[1] + j]);
            if (token_id >= 0 && token_id < vocab_size_) {
                for (int k = 0; k < embedding_dim_; ++k) {
                    weight_grad_[token_id * embedding_dim_ + k] += 
                        grad_output[i * input_cache_.shape()[1] * embedding_dim_ + j * embedding_dim_ + k];
                }
            }
        }
    }
    
    if (TensorParallelContext::instance().is_enabled()) {
        all_reduce_grad_weight(weight_grad_);
    }
    
    return Tensor({});  // 返回空张量，因为嵌入层通常不需要输入梯度
}

std::vector<Tensor> TensorParallelEmbedding::parameters() const {
    return {weight_};
}

std::vector<Tensor> TensorParallelEmbedding::gradients() const {
    return {weight_grad_};
}

void TensorParallelEmbedding::all_reduce_grad_weight(Tensor& grad_weight) {
    auto& comm = MPICommunicator::instance();
    comm.all_reduce(grad_weight);
}

void TensorParallelEmbedding::initialize_parameters() {
    weight_ = Tensor({vocab_size_, embedding_dim_});
    weight_grad_ = Tensor({vocab_size_, embedding_dim_});
    
    // 正态分布初始化
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    for (int i = 0; i < weight_.size(); ++i) {
        weight_[i] = dist(gen);
    }
    
    weight_grad_.zeros();
}

TensorParallelLayerNorm::TensorParallelLayerNorm(int normalized_shape, float eps,
                                                 const std::string& name)
    : Layer(name), normalized_shape_(normalized_shape), eps_(eps), name_(name) {
    weight_ = Tensor({normalized_shape_});
    bias_ = Tensor({normalized_shape_});
    weight_grad_ = Tensor({normalized_shape_});
    bias_grad_ = Tensor({normalized_shape_});
    
    weight_.fill(1.0f);
    bias_.fill(0.0f);
    weight_grad_.zeros();
    bias_grad_.zeros();
}

Tensor TensorParallelLayerNorm::forward(const Tensor& input) {
    input_cache_ = input;
    
    // 计算均值和方差
    Tensor mean({input.shape()[0]});
    Tensor var({input.shape()[0]});
    
    for (int i = 0; i < input.shape()[0]; ++i) {
        float sum = 0.0f;
        int count = 0;
        
        for (int j = 0; j < input.size() / input.shape()[0]; ++j) {
            sum += input[i * (input.size() / input.shape()[0]) + j];
            count++;
        }
        
        mean[i] = sum / count;
        
        float var_sum = 0.0f;
        for (int j = 0; j < input.size() / input.shape()[0]; ++j) {
            float diff = input[i * (input.size() / input.shape()[0]) + j] - mean[i];
            var_sum += diff * diff;
        }
        var[i] = var_sum / count;
    }
    
    mean_cache_ = mean;
    var_cache_ = var;
    
    // 归一化
    Tensor output(input.shape());
    for (int i = 0; i < input.shape()[0]; ++i) {
        float std_dev = std::sqrt(var[i] + eps_);
        for (int j = 0; j < input.size() / input.shape()[0]; ++j) {
            int idx = i * (input.size() / input.shape()[0]) + j;
            output[idx] = ((input[idx] - mean[i]) / std_dev) * weight_[j % normalized_shape_] + bias_[j % normalized_shape_];
        }
    }
    
    return output;
}

Tensor TensorParallelLayerNorm::backward(const Tensor& grad_output) {
    // 简化的LayerNorm反向传播实现
    Tensor grad_input(input_cache_.shape());
    
    for (int i = 0; i < grad_input.size(); ++i) {
        grad_input[i] = grad_output[i] * weight_[i % normalized_shape_];
    }
    
    // 计算权重和偏置的梯度
    for (int i = 0; i < normalized_shape_; ++i) {
        weight_grad_[i] = 0.0f;
        bias_grad_[i] = 0.0f;
        
        for (int j = 0; j < input_cache_.shape()[0]; ++j) {
            int idx = j * normalized_shape_ + i;
            float normalized_val = (input_cache_[idx] - mean_cache_[j]) / std::sqrt(var_cache_[j] + eps_);
            weight_grad_[i] += grad_output[idx] * normalized_val;
            bias_grad_[i] += grad_output[idx];
        }
    }
    
    sync_parameters();
    
    return grad_input;
}

std::vector<Tensor> TensorParallelLayerNorm::parameters() const {
    return {weight_, bias_};
}

std::vector<Tensor> TensorParallelLayerNorm::gradients() const {
    return {weight_grad_, bias_grad_};
}

void TensorParallelLayerNorm::sync_parameters() {
    // 在所有设备上同步LayerNorm的参数
    auto& comm = MPICommunicator::instance();
    comm.broadcast(weight_, 0);
    comm.broadcast(bias_, 0);
}

} // namespace megatron