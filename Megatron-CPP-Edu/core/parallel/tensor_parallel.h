#pragma once

#include "core/tensor/tensor.h"
#include "core/layers/layer.h"
#include "core/layers/linear.h"
#include "core/layers/attention.h"
#include "core/layers/transformer_block.h"
#include "core/layers/dropout.h"
#include "core/layers/layer_norm.h"
#include "core/layers/embedding.h"
#include "../distributed/communication.h"
#include <memory>
#include <vector>

namespace megatron {

// 张量并行上下文
class TensorParallelContext {
public:
    static TensorParallelContext& instance();
    
    void initialize(int world_size, int rank);
    int world_size() const { return world_size_; }
    int rank() const { return rank_; }
    
    // 获取张量并行的分割信息
    int get_local_output_dim(int global_output_dim) const;
    int get_local_input_dim(int global_input_dim) const;
    
    // 检查是否启用张量并行
    bool is_enabled() const { return world_size_ > 1; }
    
private:
    int world_size_ = 1;
    int rank_ = 0;
    
    TensorParallelContext() = default;
};

// 列并行线性层（按输出维度分割）
class ColumnParallelLinear : public Layer {
public:
    ColumnParallelLinear(int in_features, int out_features, bool bias = true,
                        const std::string& name = "column_parallel_linear");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void train(bool is_training) override { 
        if (is_training) {
            Layer::train();
        } else {
            Layer::eval();
        }
    }
    
    // 获取全局参数形状
    std::vector<int> get_global_weight_shape() const;
    
private:
    int in_features_;
    int out_features_;
    bool has_bias_;
    std::string name_;
    
    // 本地参数
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
    
    // 缓存
    Tensor input_cache_;
    
    // 张量并行通信
    void all_reduce_output(Tensor& output);
    void all_reduce_grad_input(Tensor& grad_input);
    
    // 初始化
    void initialize_parameters();
};

// 行并行线性层（按输入维度分割）
class RowParallelLinear : public Layer {
public:
    RowParallelLinear(int in_features, int out_features, bool bias = true,
                      const std::string& name = "row_parallel_linear");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void train(bool is_training) override { 
        if (is_training) {
            Layer::train();
        } else {
            Layer::eval();
        }
    }
    
    // 获取全局参数形状
    std::vector<int> get_global_weight_shape() const;
    
private:
    int in_features_;
    int out_features_;
    bool has_bias_;
    std::string name_;
    
    // 本地参数
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
    
    // 缓存
    Tensor input_cache_;
    
    // 张量并行通信
    void all_gather_input(Tensor& input);
    void reduce_grad_weight(Tensor& grad_weight);
    
    // 初始化
    void initialize_parameters();
};

// 张量并行的多头注意力机制
class TensorParallelMultiHeadAttention : public Layer {
public:
    TensorParallelMultiHeadAttention(int embed_dim, int num_heads,
                                   bool use_causal_mask = false,
                                   const std::string& name = "tp_mha");
    
    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value);
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void train(bool is_training) override { 
        if (is_training) {
            Layer::train();
        } else {
            Layer::eval();
        }
    }
    
private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    bool use_causal_mask_;
    std::string name_;
    
    // 张量并行投影层
    std::shared_ptr<ColumnParallelLinear> q_proj_;
    std::shared_ptr<ColumnParallelLinear> k_proj_;
    std::shared_ptr<ColumnParallelLinear> v_proj_;
    std::shared_ptr<RowParallelLinear> out_proj_;
    
    // 缓存
    Tensor q_cache_;
    Tensor k_cache_;
    Tensor v_cache_;
    Tensor attention_weights_cache_;
    Tensor input_cache_;
    
    // 内部函数
    Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k,
                                       const Tensor& v);
    Tensor causal_mask(const Tensor& attention_scores, int seq_len);
    Tensor split_heads(const Tensor& x);
    Tensor combine_heads(const Tensor& x);
    
    // 初始化
    void initialize_parallel_layers();
};

// 张量并行的FFN层
class TensorParallelFFN : public Layer {
public:
    TensorParallelFFN(int embed_dim, int ffn_dim, float dropout = 0.1,
                      const std::string& name = "tp_ffn");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void train(bool is_training) override { 
        if (is_training) {
            Layer::train();
        } else {
            Layer::eval();
        }
    }
    
private:
    int embed_dim_;
    int ffn_dim_;
    float dropout_;
    std::string name_;
    
    // 张量并行层
    std::shared_ptr<ColumnParallelLinear> linear1_;
    std::shared_ptr<RowParallelLinear> linear2_;
    std::shared_ptr<Dropout> dropout_layer_;
    
    // 缓存
    Tensor hidden_cache_;
    Tensor input_cache_;
    
    // 激活函数
    Tensor gelu(const Tensor& x);
    Tensor gelu_backward(const Tensor& x, const Tensor& grad_output);
};

// 张量并行的Transformer块
class TensorParallelTransformerBlock : public Layer {
public:
    TensorParallelTransformerBlock(int embed_dim, int num_heads, int ffn_dim,
                                  float dropout = 0.1,
                                  const std::string& name = "tp_transformer_block");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void train(bool is_training) override { 
        if (is_training) {
            Layer::train();
        } else {
            Layer::eval();
        }
    }
    
private:
    int embed_dim_;
    int num_heads_;
    int ffn_dim_;
    float dropout_;
    std::string name_;
    
    // 张量并行组件
    std::shared_ptr<LayerNorm> attn_norm_;
    std::shared_ptr<TensorParallelMultiHeadAttention> attention_;
    std::shared_ptr<Dropout> attn_dropout_;
    
    std::shared_ptr<LayerNorm> ffn_norm_;
    std::shared_ptr<TensorParallelFFN> ffn_;
    std::shared_ptr<Dropout> ffn_dropout_;
    
    // 缓存
    Tensor attn_output_cache_;
    Tensor ffn_output_cache_;
    Tensor input_cache_;
};

// 张量并行的嵌入层
class TensorParallelEmbedding : public Layer {
public:
    TensorParallelEmbedding(int vocab_size, int embedding_dim,
                           const std::string& name = "tp_embedding");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void train(bool is_training) override { 
        if (is_training) {
            Layer::train();
        } else {
            Layer::eval();
        }
    }
    
private:
    int vocab_size_;
    int embedding_dim_;
    std::string name_;
    
    // 本地参数
    Tensor weight_;
    Tensor weight_grad_;
    
    // 缓存
    Tensor input_cache_;
    
    // 张量并行通信
    void all_reduce_grad_weight(Tensor& grad_weight);
    
    // 初始化
    void initialize_parameters();
};

// 张量并行的层归一化
class TensorParallelLayerNorm : public Layer {
public:
    TensorParallelLayerNorm(int normalized_shape, float eps = 1e-5,
                           const std::string& name = "tp_layer_norm");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void train(bool is_training) override { 
        if (is_training) {
            Layer::train();
        } else {
            Layer::eval();
        }
    }
    
private:
    int normalized_shape_;
    float eps_;
    std::string name_;
    
    // 参数（在所有设备上保持相同）
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
    
    // 缓存
    Tensor input_cache_;
    Tensor mean_cache_;
    Tensor var_cache_;
    
    // 同步参数
    void sync_parameters();
};

// 张量并行模型构建器
class TensorParallelModelBuilder {
public:
    TensorParallelModelBuilder(int tensor_parallel_size = 1);
    
    // 设置模型配置
    void set_vocab_size(int vocab_size);
    void set_max_seq_len(int max_seq_len);
    void set_embed_dim(int embed_dim);
    void set_num_layers(int num_layers);
    void set_num_heads(int num_heads);
    void set_ffn_dim(int ffn_dim);
    void set_dropout(float dropout);
    
    // 构建张量并行的GPT模型
    std::shared_ptr<Layer> build_gpt_model();
    
    // 构建张量并行的Transformer分类器
    std::shared_ptr<Layer> build_transformer_classifier(int num_classes);
    
private:
    int tensor_parallel_size_;
    
    // 模型配置
    int vocab_size_ = 1000;
    int max_seq_len_ = 512;
    int embed_dim_ = 768;
    int num_layers_ = 12;
    int num_heads_ = 12;
    int ffn_dim_ = 3072;
    float dropout_ = 0.1;
    
    // 构建组件
    std::shared_ptr<TensorParallelEmbedding> build_embedding();
    std::shared_ptr<TensorParallelLayerNorm> build_layer_norm();
    std::vector<std::shared_ptr<TensorParallelTransformerBlock>> build_transformer_blocks();
    std::shared_ptr<RowParallelLinear> build_lm_head();
};

// 张量并行辅助函数
namespace tensor_parallel_utils {
    // 检查是否支持张量并行
    bool is_tensor_parallel_supported();
    
    // 获取当前设备的张量并行信息
    int get_tensor_parallel_world_size();
    int get_tensor_parallel_rank();
    
    // 初始化张量并行环境
    void initialize_tensor_parallel(int world_size, int rank);
    
    // 清理张量并行环境
    void cleanup_tensor_parallel();
    
    // 分割张量
    Tensor split_tensor(const Tensor& tensor, int dim, int rank, int world_size);
    
    // 合并张量
    Tensor concatenate_tensors(const std::vector<Tensor>& tensors, int dim);
    
    // 验证张量并行配置
    bool validate_tensor_parallel_config(int world_size, int rank);
    
    // 打印张量并行信息
    void print_tensor_parallel_info();
}

} // namespace megatron