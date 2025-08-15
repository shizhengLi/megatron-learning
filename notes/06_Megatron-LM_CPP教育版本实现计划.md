# Megatron-LM C++ 教育版本实现计划

## 1. 项目概述

### 1.1 项目目标
基于Megatron-LM的架构设计，实现一个C++教育版本，保留核心功能模块，用于教学和研究目的。

### 1.2 设计原则
- **教育性**：代码结构清晰，注释详细，易于理解
- **简洁性**：去除复杂的优化，保留核心逻辑
- **可扩展性**：模块化设计，便于添加新功能
- **实用性**：能够实际运行，支持小规模训练

### 1.3 技术选型
- **语言**：C++17/20
- **线性代数**：Eigen3 或自定义矩阵运算
- **并行计算**：OpenMP + MPI
- **神经网络**：自定义简单框架
- **构建系统**：CMake

## 2. 系统架构设计

### 2.1 整体架构
```
Megatron-CPP-Edu/
├── core/                    # 核心库
│   ├── tensor/             # 张量操作
│   ├── layers/             # 神经网络层
│   ├── parallel/           # 并行计算
│   ├── optimizers/         # 优化器
│   └── utils/             # 工具函数
├── models/                 # 模型实现
│   ├── transformer/       # Transformer
│   └── gpt/              # GPT模型
├── training/              # 训练逻辑
│   ├── trainer.h         # 训练器
│   └── data_loader.h     # 数据加载
├── distributed/           # 分布式训练
│   ├── communication.h    # 通信模块
│   └── parallel_state.h  # 并行状态
├── examples/              # 示例代码
└── tests/                 # 测试代码
```

### 2.2 核心模块设计

#### 2.2.1 张量系统
```cpp
// core/tensor/tensor.h
#pragma once
#include <vector>
#include <memory>
#include <initializer_list>

namespace megatron {

enum class DeviceType {
    CPU,
    CUDA
};

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT8
};

class Tensor {
public:
    // 构造函数
    Tensor(const std::vector<int>& shape, DataType dtype = DataType::FLOAT32);
    Tensor(std::initializer_list<int> shape, DataType dtype = DataType::FLOAT32);
    
    // 拷贝和移动
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // 基本操作
    void zeros();
    void ones();
    void random_normal(float mean = 0.0f, float std = 1.0f);
    void fill(float value);
    
    // 形状操作
    void reshape(const std::vector<int>& new_shape);
    Tensor view(const std::vector<int>& new_shape) const;
    Tensor transpose() const;
    Tensor slice(int dim, int start, int end) const;
    
    // 数学运算
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    Tensor matmul(const Tensor& other) const;
    Tensor sum(int dim = -1) const;
    Tensor mean(int dim = -1) const;
    Tensor max(int dim = -1) const;
    
    // 激活函数
    Tensor relu() const;
    Tensor gelu() const;
    Tensor softmax(int dim = -1) const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    
    // 属性访问
    const std::vector<int>& shape() const { return shape_; }
    int dim() const { return shape_.size(); }
    int size() const { return size_; }
    DataType dtype() const { return dtype_; }
    DeviceType device() const { return device_; }
    
    // 数据访问
    float* data();
    const float* data() const;
    float& operator[](int index);
    const float& operator[](int index) const;
    
    // 设备操作
    void to(DeviceType device);
    bool is_contiguous() const;
    
    // 内存管理
    Tensor contiguous() const;
    void clone_from(const Tensor& other);
    
private:
    std::vector<int> shape_;
    int size_;
    DataType dtype_;
    DeviceType device_;
    std::shared_ptr<float> data_;
    bool owns_data_;
    
    void allocate_memory();
    void free_memory();
    void compute_size();
};

// 张量操作函数
Tensor add(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor concatenate(const std::vector<Tensor>& tensors, int dim = 0);

} // namespace megatron
```

#### 2.2.2 神经网络层
```cpp
// core/layers/layer.h
#pragma once
#include "core/tensor/tensor.h"
#include <memory>
#include <vector>

namespace megatron {

class Layer {
public:
    virtual ~Layer() = default;
    
    // 前向传播
    virtual Tensor forward(const Tensor& input) = 0;
    
    // 反向传播
    virtual Tensor backward(const Tensor& grad_output) = 0;
    
    // 参数管理
    virtual std::vector<Tensor> parameters() const = 0;
    virtual std::vector<Tensor> gradients() const = 0;
    
    // 模式设置
    virtual void train(bool is_training) { is_training_ = is_training; }
    virtual bool is_training() const { return is_training_; }
    
    // 设备管理
    virtual void to(DeviceType device) = 0;
    
protected:
    bool is_training_ = true;
};

// 线性层
class Linear : public Layer {
public:
    Linear(int in_features, int out_features, bool bias = true);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void to(DeviceType device) override;
    
private:
    int in_features_;
    int out_features_;
    bool use_bias_;
    
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
    
    Tensor input_cache_;
};

// 层归一化
class LayerNorm : public Layer {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void to(DeviceType device) override;
    
private:
    int normalized_shape_;
    float eps_;
    
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
    
    Tensor input_cache_;
    Tensor mean_cache_;
    Tensor var_cache_;
};

// Dropout层
class Dropout : public Layer {
public:
    Dropout(float p = 0.5);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override { return {}; }
    std::vector<Tensor> gradients() const override { return {}; }
    
    void to(DeviceType device) override {}
    
private:
    float p_;
    Tensor mask_;
};

// 嵌入层
class Embedding : public Layer {
public:
    Embedding(int vocab_size, int embedding_dim);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void to(DeviceType device) override;
    
private:
    int vocab_size_;
    int embedding_dim_;
    
    Tensor weight_;
    Tensor weight_grad_;
    
    Tensor input_cache_;
};

} // namespace megatron
```

#### 2.2.3 注意力机制
```cpp
// core/layers/attention.h
#pragma once
#include "layer.h"
#include <memory>

namespace megatron {

class MultiHeadAttention : public Layer {
public:
    MultiHeadAttention(int embed_dim, int num_heads, 
                       bool use_causal_mask = false);
    
    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value);
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void to(DeviceType device) override;
    
    // 位置编码
    void set_positional_encoding(const Tensor& pos_encoding);
    
private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    bool use_causal_mask_;
    
    // 投影层
    std::shared_ptr<Linear> q_proj_;
    std::shared_ptr<Linear> k_proj_;
    std::shared_ptr<Linear> v_proj_;
    std::shared_ptr<Linear> out_proj_;
    
    // 缓存
    Tensor q_cache_;
    Tensor k_cache_;
    Tensor v_cache_;
    Tensor attention_weights_cache_;
    
    // 位置编码
    Tensor positional_encoding_;
    
    // 内部函数
    Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, 
                                       const Tensor& v);
    Tensor causal_mask(const Tensor& attention_scores, int seq_len);
    Tensor split_heads(const Tensor& x);
    Tensor combine_heads(const Tensor& x);
};

class SelfAttention : public MultiHeadAttention {
public:
    SelfAttention(int embed_dim, int num_heads)
        : MultiHeadAttention(embed_dim, num_heads, true) {}
    
    Tensor forward(const Tensor& input) override {
        return MultiHeadAttention::forward(input, input, input);
    }
};

} // namespace megatron
```

#### 2.2.4 Transformer块
```cpp
// core/layers/transformer_block.h
#pragma once
#include "attention.h"
#include <memory>
#include <vector>

namespace megatron {

class TransformerBlock : public Layer {
public:
    TransformerBlock(int embed_dim, int num_heads, int ffn_dim, 
                    float dropout = 0.1);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void to(DeviceType device) override;
    
private:
    int embed_dim_;
    int num_heads_;
    int ffn_dim_;
    float dropout_;
    
    // 注意力部分
    std::shared_ptr<LayerNorm> attn_norm_;
    std::shared_ptr<MultiHeadAttention> attention_;
    std::shared_ptr<Dropout> attn_dropout_;
    
    // FFN部分
    std::shared_ptr<LayerNorm> ffn_norm_;
    std::shared_ptr<Linear> ffn_linear1_;
    std::shared_ptr<Linear> ffn_linear2_;
    std::shared_ptr<Dropout> ffn_dropout_;
    
    // 缓存
    Tensor attn_output_cache_;
    Tensor ffn_output_cache_;
    Tensor input_cache_;
};

class GPT2Block : public TransformerBlock {
public:
    GPT2Block(int embed_dim, int num_heads, int ffn_dim, float dropout = 0.1)
        : TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) {}
};

} // namespace megatron
```

### 2.3 并行计算框架

#### 2.3.1 张量并行
```cpp
// core/parallel/tensor_parallel.h
#pragma once
#include "core/tensor/tensor.h"
#include <memory>
#include <vector>

namespace megatron {

class TensorParallelContext {
public:
    static TensorParallelContext& instance();
    
    void initialize(int world_size, int rank);
    int world_size() const { return world_size_; }
    int rank() const { return rank_; }
    
private:
    int world_size_ = 1;
    int rank_ = 0;
};

class ColumnParallelLinear : public Layer {
public:
    ColumnParallelLinear(int in_features, int out_features, bool bias = true);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void to(DeviceType device) override;
    
private:
    int in_features_;
    int out_features_;
    bool use_bias_;
    
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
    
    Tensor input_cache_;
    
    // 张量并行通信
    void all_reduce_grad(Tensor& grad);
};

class RowParallelLinear : public Layer {
public:
    RowParallelLinear(int in_features, int out_features, bool bias = true);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    void to(DeviceType device) override;
    
private:
    int in_features_;
    int out_features_;
    bool use_bias_;
    
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
    
    Tensor input_cache_;
    
    // 张量并行通信
    void all_gather_output(Tensor& output);
};

} // namespace megatron
```

#### 2.3.2 数据并行
```cpp
// core/parallel/data_parallel.h
#pragma once
#include "core/tensor/tensor.h"
#include <memory>
#include <vector>

namespace megatron {

class DataParallelTrainer {
public:
    DataParallelTrainer(int world_size, int rank);
    
    // 梯度同步
    void synchronize_gradients(std::vector<Tensor>& gradients);
    
    // 广播参数
    void broadcast_parameters(std::vector<Tensor>& parameters);
    
    // 获取本地批大小
    int get_local_batch_size(int global_batch_size) const;
    
private:
    int world_size_;
    int rank_;
    
    // 通信函数
    void all_reduce(Tensor& tensor);
    void broadcast(Tensor& tensor, int root_rank);
};

class DistributedDataParallel {
public:
    DistributedDataParallel(std::shared_ptr<Layer> model, int world_size, int rank);
    
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    
    std::vector<Tensor> parameters() const;
    std::vector<Tensor> gradients() const;
    
    void to(DeviceType device);
    
private:
    std::shared_ptr<Layer> model_;
    int world_size_;
    int rank_;
    
    DataParallelTrainer trainer_;
    Tensor input_cache_;
};

} // namespace megatron
```

### 2.4 优化器系统

#### 2.4.1 优化器基类
```cpp
// core/optimizers/optimizer.h
#pragma once
#include "core/tensor/tensor.h"
#include <vector>
#include <memory>

namespace megatron {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    // 更新参数
    virtual void step(std::vector<Tensor>& parameters, 
                     std::vector<Tensor>& gradients) = 0;
    
    // 设置学习率
    virtual void set_learning_rate(float lr) { learning_rate_ = lr; }
    float learning_rate() const { return learning_rate_; }
    
    // 梯度清零
    virtual void zero_grad(std::vector<Tensor>& gradients);
    
protected:
    float learning_rate_;
    
    // 辅助函数
    void clip_gradients(std::vector<Tensor>& gradients, float max_norm);
};

class SGD : public Optimizer {
public:
    SGD(float lr = 0.01, float momentum = 0.0, float weight_decay = 0.0);
    
    void step(std::vector<Tensor>& parameters, 
             std::vector<Tensor>& gradients) override;
    
private:
    float momentum_;
    float weight_decay_;
    std::vector<Tensor> velocity_;
};

class AdamW : public Optimizer {
public:
    AdamW(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, 
          float weight_decay = 0.01, float eps = 1e-8);
    
    void step(std::vector<Tensor>& parameters, 
             std::vector<Tensor>& gradients) override;
    
private:
    float beta1_;
    float beta2_;
    float weight_decay_;
    float eps_;
    int step_;
    
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
};

} // namespace megatron
```

### 2.5 模型实现

#### 2.5.1 GPT模型
```cpp
// models/gpt/gpt_model.h
#pragma once
#include "core/layers/transformer_block.h"
#include "core/layers/layer.h"
#include "core/optimizers/optimizer.h"
#include <memory>
#include <vector>

namespace megatron {

class GPTModel {
public:
    GPTModel(int vocab_size, int max_seq_len, int embed_dim, 
             int num_layers, int num_heads, int ffn_dim,
             float dropout = 0.1);
    
    // 前向传播
    Tensor forward(const Tensor& input_ids, const Tensor& position_ids);
    
    // 训练
    void train_step(const Tensor& input_ids, const Tensor& targets, 
                   std::shared_ptr<Optimizer> optimizer);
    
    // 生成
    Tensor generate(const Tensor& prompt_ids, int max_length, 
                   float temperature = 1.0f, int top_k = 50);
    
    // 参数管理
    std::vector<Tensor> parameters() const;
    std::vector<Tensor> gradients() const;
    
    // 设备管理
    void to(DeviceType device);
    
    // 模式设置
    void train(bool is_training);
    
private:
    int vocab_size_;
    int max_seq_len_;
    int embed_dim_;
    int num_layers_;
    int num_heads_;
    int ffn_dim_;
    float dropout_;
    
    // 模型组件
    std::shared_ptr<Embedding> token_embedding_;
    std::shared_ptr<Embedding> position_embedding_;
    std::shared_ptr<Dropout> embedding_dropout_;
    
    std::vector<std::shared_ptr<GPT2Block>> transformer_blocks_;
    std::shared_ptr<LayerNorm> final_norm_;
    
    std::shared_ptr<Linear> lm_head_;
    
    // 缓存
    bool is_training_;
    Tensor last_hidden_states_;
    
    // 位置编码
    Tensor create_positional_encoding(int max_seq_len, int embed_dim);
    
    // 损失计算
    Tensor compute_loss(const Tensor& logits, const Tensor& targets);
    
    // 采样函数
    Tensor sample_logits(const Tensor& logits, float temperature, int top_k);
};

} // namespace megatron
```

### 2.6 训练系统

#### 2.6.1 数据加载器
```cpp
// training/data_loader.h
#pragma once
#include "core/tensor/tensor.h"
#include <vector>
#include <string>
#include <memory>

namespace megatron {

struct DataSample {
    std::vector<int> input_ids;
    std::vector<int> target_ids;
    float weight = 1.0f;
};

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual DataSample get_sample(size_t index) = 0;
};

class TextDataset : public Dataset {
public:
    TextDataset(const std::string& file_path, int max_seq_len);
    
    size_t size() const override;
    DataSample get_sample(size_t index) override;
    
private:
    std::vector<DataSample> samples_;
    int max_seq_len_;
    
    void load_file(const std::string& file_path);
    std::vector<int> tokenize(const std::string& text);
};

class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> dataset, int batch_size, 
                bool shuffle = true);
    
    std::vector<DataSample> next_batch();
    bool has_next_batch() const;
    void reset();
    
    int num_batches() const;
    int current_batch() const;
    
private:
    std::shared_ptr<Dataset> dataset_;
    int batch_size_;
    bool shuffle_;
    int current_batch_;
    std::vector<int> indices_;
    
    void shuffle_indices();
    std::vector<DataSample> create_batch(const std::vector<int>& batch_indices);
};

class TensorDataLoader {
public:
    TensorDataLoader(std::shared_ptr<Dataset> dataset, int batch_size,
                     bool shuffle = true);
    
    std::pair<Tensor, Tensor> next_batch();
    bool has_next_batch() const;
    void reset();
    
private:
    DataLoader data_loader_;
    
    Tensor convert_to_tensor(const std::vector<std::vector<int>>& data);
};

} // namespace megatron
```

#### 2.6.2 训练器
```cpp
// training/trainer.h
#pragma once
#include "core/tensor/tensor.h"
#include "core/optimizers/optimizer.h"
#include "training/data_loader.h"
#include <memory>
#include <vector>
#include <string>

namespace megatron {

struct TrainingConfig {
    int batch_size = 32;
    int max_epochs = 10;
    float learning_rate = 0.001;
    float weight_decay = 0.01;
    float grad_clip = 1.0;
    bool use_cuda = false;
    int log_interval = 100;
    int save_interval = 1000;
    std::string save_dir = "./checkpoints";
};

class Trainer {
public:
    Trainer(std::shared_ptr<GPTModel> model, 
             std::shared_ptr<Optimizer> optimizer,
             const TrainingConfig& config);
    
    // 训练循环
    void train(std::shared_ptr<Dataset> dataset);
    
    // 评估
    float evaluate(std::shared_ptr<Dataset> dataset);
    
    // 保存和加载
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
    // 获取训练状态
    float get_current_loss() const { return current_loss_; }
    int get_current_step() const { return current_step_; }
    
private:
    std::shared_ptr<GPTModel> model_;
    std::shared_ptr<Optimizer> optimizer_;
    TrainingConfig config_;
    
    // 训练状态
    int current_epoch_;
    int current_step_;
    float current_loss_;
    std::vector<float> loss_history_;
    
    // 训练步骤
    void train_epoch(std::shared_ptr<TensorDataLoader> data_loader);
    void train_step(const Tensor& input_ids, const Tensor& targets);
    
    // 日志和保存
    void log_training_progress(int step, float loss);
    void save_model();
    
    // 验证
    void validate();
};

} // namespace megatron
```

### 2.7 分布式训练

#### 2.7.1 通信模块
```cpp
// distributed/communication.h
#pragma once
#include "core/tensor/tensor.h"
#include <mpi.h>
#include <vector>

namespace megatron {

class MPICommunicator {
public:
    static MPICommunicator& instance();
    
    void initialize(int* argc, char*** argv);
    void finalize();
    
    int world_size() const;
    int rank() const;
    
    // 基础通信操作
    void send(const Tensor& tensor, int dest_rank, int tag = 0);
    void recv(Tensor& tensor, int src_rank, int tag = 0);
    
    // 集合通信
    void all_reduce(Tensor& tensor);
    void all_gather(const std::vector<Tensor>& send_tensors, 
                   std::vector<Tensor>& recv_tensors);
    void broadcast(Tensor& tensor, int root_rank);
    
    // 障碍同步
    void barrier();
    
private:
    bool initialized_;
    int world_size_;
    int rank_;
    
    MPICommunicator() : initialized_(false), world_size_(1), rank_(0) {}
};

// 简化的分布式训练包装器
class DistributedTrainer {
public:
    DistributedTrainer(std::shared_ptr<GPTModel> model,
                      std::shared_ptr<Optimizer> optimizer,
                      const TrainingConfig& config);
    
    void distributed_train(std::shared_ptr<Dataset> dataset);
    
private:
    std::shared_ptr<GPTModel> model_;
    std::shared_ptr<Optimizer> optimizer_;
    TrainingConfig config_;
    
    MPICommunicator& comm_;
    
    // 分布式数据处理
    std::shared_ptr<Dataset> get_local_dataset(std::shared_ptr<Dataset> global_dataset);
    
    // 分布式训练步骤
    void distributed_train_step(const Tensor& input_ids, const Tensor& targets);
    
    // 梯度同步
    void synchronize_gradients();
};

} // namespace megatron
```

## 3. 实现计划

### 3.1 第一阶段：核心张量系统 (2-3周)

**目标**：实现基础的张量操作和线性代数

**任务**：
1. 实现Tensor类的数据结构和内存管理
2. 实现基本的张量操作（加法、乘法、矩阵乘法）
3. 实现激活函数（ReLU, GELU, Softmax）
4. 编写单元测试

**关键文件**：
- `core/tensor/tensor.h`
- `core/tensor/tensor.cpp`
- `tests/test_tensor.cpp`

**验收标准**：
- 所有张量操作测试通过
- 内存管理无泄漏
- 支持基本的形状操作

### 3.2 第二阶段：神经网络层 (2-3周)

**目标**：实现基础的神经网络层

**任务**：
1. 实现Linear层及其反向传播
2. 实现LayerNorm和Dropout
3. 实现Embedding层
4. 实现基础的注意力机制

**关键文件**：
- `core/layers/layer.h`
- `core/layers/linear.h`
- `core/layers/attention.h`
- `tests/test_layers.cpp`

**验收标准**：
- 所有层的正向和反向传播正确
- 梯度检查通过
- 支持基本的神经网络训练

### 3.3 第三阶段：Transformer实现 (3-4周)

**目标**：实现完整的Transformer架构

**任务**：
1. 实现TransformerBlock
2. 实现位置编码
3. 实现GPT模型
4. 实现基础的文本生成

**关键文件**：
- `core/layers/transformer_block.h`
- `models/gpt/gpt_model.h`
- `tests/test_transformer.cpp`

**验收标准**：
- GPT模型能够正确处理输入
- 文本生成功能正常
- 模型参数量正确

### 3.4 第四阶段：优化器和训练 (2-3周)

**目标**：实现训练系统

**任务**：
1. 实现SGD和AdamW优化器
2. 实现数据加载器
3. 实现训练器
4. 实现基础的训练循环

**关键文件**：
- `core/optimizers/optimizer.h`
- `training/data_loader.h`
- `training/trainer.h`
- `tests/test_training.cpp`

**验收标准**：
- 能够训练简单模型
- 损失函数正确下降
- 支持模型保存和加载

### 3.5 第五阶段：并行计算 (3-4周)

**目标**：实现并行训练功能

**任务**：
1. 实现数据并行
2. 实现简单的张量并行
3. 集成MPI通信
4. 实现分布式训练

**关键文件**：
- `core/parallel/data_parallel.h`
- `core/parallel/tensor_parallel.h`
- `distributed/communication.h`
- `tests/test_parallel.cpp`

**验收标准**：
- 数据并行训练正确
- MPI通信正常工作
- 多GPU训练加速明显

### 3.6 第六阶段：示例和文档 (2-3周)

**目标**：完善示例和文档

**任务**：
1. 编写训练示例
2. 编写生成示例
3. 完善代码文档
4. 编写教程文档

**关键文件**：
- `examples/train_gpt.cpp`
- `examples/generate_text.cpp`
- `README.md`
- `docs/tutorial.md`

**验收标准**：
- 示例代码能够正常运行
- 文档完整清晰
- 用户能够独立运行

## 4. 示例代码

### 4.1 简单训练示例
```cpp
// examples/train_simple_gpt.cpp
#include "models/gpt/gpt_model.h"
#include "training/trainer.h"
#include "training/data_loader.h"
#include <iostream>
#include <memory>

using namespace megatron;

int main(int argc, char** argv) {
    // 初始化MPI
    MPICommunicator::instance().initialize(&argc, &argv);
    
    // 创建简单的数据集
    auto dataset = std::make_shared<SimpleTextDataset>(
        "data/simple_text.txt", 128  // max_seq_len
    );
    
    // 创建GPT模型
    auto model = std::make_shared<GPTModel>(
        1000,  // vocab_size
        128,   // max_seq_len
        256,   // embed_dim
        4,     // num_layers
        8,     // num_heads
        1024,  // ffn_dim
        0.1    // dropout
    );
    
    // 创建优化器
    auto optimizer = std::make_shared<AdamW>(0.001);
    
    // 训练配置
    TrainingConfig config;
    config.batch_size = 32;
    config.max_epochs = 10;
    config.learning_rate = 0.001;
    config.use_cuda = false;  // 教育版本默认使用CPU
    config.log_interval = 10;
    
    // 创建训练器
    Trainer trainer(model, optimizer, config);
    
    // 开始训练
    std::cout << "Starting training..." << std::endl;
    trainer.train(dataset);
    
    // 保存模型
    trainer.save_checkpoint("models/simple_gpt_final.pt");
    
    std::cout << "Training completed!" << std::endl;
    
    // 清理MPI
    MPICommunicator::instance().finalize();
    
    return 0;
}
```

### 4.2 文本生成示例
```cpp
// examples/generate_text.cpp
#include "models/gpt/gpt_model.h"
#include <iostream>
#include <vector>
#include <memory>

using namespace megatron;

int main() {
    // 加载模型
    auto model = std::make_shared<GPTModel>(
        1000,  // vocab_size
        128,   // max_seq_len
        256,   // embed_dim
        4,     // num_layers
        8,     // num_heads
        1024,  // ffn_dim
        0.1    // dropout
    );
    
    // 加载训练好的参数
    model->load_checkpoint("models/simple_gpt_final.pt");
    model->train(false);  // 设置为推理模式
    
    // 输入提示
    std::vector<int> prompt = {1, 2, 3};  // 简单的token IDs
    
    // 生成文本
    std::cout << "Generating text..." << std::endl;
    Tensor generated = model->generate(
        Tensor::from_vector(prompt),
        50,    // max_length
        0.8,   // temperature
        10     // top_k
    );
    
    // 输出结果
    std::cout << "Generated tokens: ";
    for (int i = 0; i < generated.size(); ++i) {
        std::cout << generated[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

### 4.3 并行训练示例
```cpp
// examples/train_parallel_gpt.cpp
#include "models/gpt/gpt_model.h"
#include "training/trainer.h"
#include "training/data_loader.h"
#include "distributed/communication.h"
#include <iostream>
#include <memory>

using namespace megatron;

int main(int argc, char** argv) {
    // 初始化MPI
    MPICommunicator::instance().initialize(&argc, &argv);
    
    int rank = MPICommunicator::instance().rank();
    int world_size = MPICommunicator::instance().world_size();
    
    if (rank == 0) {
        std::cout << "Starting distributed training with " << world_size << " GPUs" << std::endl;
    }
    
    // 创建数据集
    auto dataset = std::make_shared<SimpleTextDataset>(
        "data/large_text.txt", 512
    );
    
    // 创建更大的GPT模型
    auto model = std::make_shared<GPTModel>(
        10000,  // vocab_size
        512,    // max_seq_len
        768,    // embed_dim
        12,     // num_layers
        12,     // num_heads
        3072,   // ffn_dim
        0.1     // dropout
    );
    
    // 创建优化器
    auto optimizer = std::make_shared<AdamW>(0.0001);
    
    // 分布式训练配置
    TrainingConfig config;
    config.batch_size = 16;  // 每个GPU的批大小
    config.max_epochs = 20;
    config.learning_rate = 0.0001;
    config.log_interval = 100;
    config.save_interval = 1000;
    
    // 创建分布式训练器
    auto trainer = std::make_shared<DistributedTrainer>(
        model, optimizer, config
    );
    
    // 开始分布式训练
    trainer->distributed_train(dataset);
    
    if (rank == 0) {
        std::cout << "Distributed training completed!" << std::endl;
    }
    
    // 清理MPI
    MPICommunicator::instance().finalize();
    
    return 0;
}
```

## 5. 构建系统

### 5.1 CMake配置
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(MegatronCPP VERSION 1.0.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(Eigen3 REQUIRED)
find_package(MPI REQUIRED)

# 编译选项
option(USE_CUDA "Use CUDA support" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# 添加子目录
add_subdirectory(core)
add_subdirectory(models)
add_subdirectory(training)
add_subdirectory(distributed)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 安装配置
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY models/ DESTINATION include/models)
install(DIRECTORY training/ DESTINATION include/training)
install(DIRECTORY distributed/ DESTINATION include/distributed)
```

### 5.2 核心库CMake
```cmake
# core/CMakeLists.txt
add_library(megatron_core
    tensor/tensor.cpp
    layers/layer.cpp
    layers/linear.cpp
    layers/attention.cpp
    layers/transformer_block.cpp
    optimizers/optimizer.cpp
    parallel/tensor_parallel.cpp
    parallel/data_parallel.cpp
)

# 包含目录
target_include_directories(megatron_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# 链接依赖
target_link_libraries(megatron_core
    PUBLIC Eigen3::Eigen
    PUBLIC MPI::MPI_CXX
)

# 编译定义
target_compile_definitions(megatron_core
    PRIVATE EIGEN_MPL2_ONLY
)

if(USE_CUDA)
    target_compile_definitions(megatron_core PRIVATE USE_CUDA)
    target_link_libraries(megatron_core PRIVATE CUDA::cudart CUDA::cublas)
endif()

# 安装
install(TARGETS megatron_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
```

## 6. 测试计划

### 6.1 单元测试
```cpp
// tests/test_tensor.cpp
#include <gtest/gtest.h>
#include "core/tensor/tensor.h"
#include <iostream>

using namespace megatron;

TEST(TensorTest, BasicOperations) {
    Tensor a({2, 3});
    Tensor b({2, 3});
    
    a.fill(1.0f);
    b.fill(2.0f);
    
    Tensor c = a + b;
    
    EXPECT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    
    for (int i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c[i], 3.0f);
    }
}

TEST(TensorTest, MatrixMultiplication) {
    Tensor a({2, 3});
    Tensor b({3, 4});
    
    a.fill(1.0f);
    b.fill(2.0f);
    
    Tensor c = a.matmul(b);
    
    EXPECT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 4);
    
    // 检查矩阵乘法结果
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(c[i * 4 + j], 6.0f);  // 1*2*3 = 6
        }
    }
}

TEST(TensorTest, ActivationFunctions) {
    Tensor x({3});
    x[0] = -1.0f;
    x[1] = 0.0f;
    x[2] = 1.0f;
    
    Tensor relu = x.relu();
    EXPECT_FLOAT_EQ(relu[0], 0.0f);
    EXPECT_FLOAT_EQ(relu[1], 0.0f);
    EXPECT_FLOAT_EQ(relu[2], 1.0f);
    
    Tensor sigmoid = x.sigmoid();
    EXPECT_NEAR(sigmoid[0], 0.2689f, 1e-4f);
    EXPECT_NEAR(sigmoid[1], 0.5f, 1e-4f);
    EXPECT_NEAR(sigmoid[2], 0.7311f, 1e-4f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### 6.2 集成测试
```cpp
// tests/test_training.cpp
#include <gtest/gtest.h>
#include "models/gpt/gpt_model.h"
#include "training/trainer.h"
#include "training/data_loader.h"
#include <iostream>

using namespace megatron;

TEST(TrainingTest, SimpleTraining) {
    // 创建简单数据集
    auto dataset = std::make_shared<SimpleTextDataset>(
        "data/test.txt", 32
    );
    
    // 创建小模型
    auto model = std::make_shared<GPTModel>(
        100,   // vocab_size
        32,    // max_seq_len
        64,    // embed_dim
        2,     // num_layers
        4,     // num_heads
        256,   // ffn_dim
        0.1    // dropout
    );
    
    // 创建优化器
    auto optimizer = std::make_shared<AdamW>(0.01);
    
    // 训练配置
    TrainingConfig config;
    config.batch_size = 4;
    config.max_epochs = 2;
    config.learning_rate = 0.01;
    config.log_interval = 1;
    
    // 创建训练器
    Trainer trainer(model, optimizer, config);
    
    // 训练
    trainer.train(dataset);
    
    // 检查损失是否下降
    EXPECT_LT(trainer.get_current_loss(), 10.0f);
}

TEST(TrainingTest, ModelLoading) {
    // 创建模型
    auto model = std::make_shared<GPTModel>(
        100, 32, 64, 2, 4, 256, 0.1
    );
    
    // 保存模型
    model->save_checkpoint("test_model.pt");
    
    // 创建新模型并加载
    auto loaded_model = std::make_shared<GPTModel>(
        100, 32, 64, 2, 4, 256, 0.1
    );
    loaded_model->load_checkpoint("test_model.pt");
    
    // 检查参数是否一致
    auto original_params = model->parameters();
    auto loaded_params = loaded_model->parameters();
    
    ASSERT_EQ(original_params.size(), loaded_params.size());
    
    for (size_t i = 0; i < original_params.size(); ++i) {
        ASSERT_EQ(original_params[i].size(), loaded_params[i].size());
        for (int j = 0; j < original_params[i].size(); ++j) {
            EXPECT_FLOAT_EQ(original_params[i][j], loaded_params[i][j]);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

## 7. 部署和使用

### 7.1 编译和安装
```bash
# 克隆项目
git clone https://github.com/your-username/Megatron-CPP-Edu.git
cd Megatron-CPP-Edu

# 创建构建目录
mkdir build && cd build

# 配置项目
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON

# 编译
make -j$(nproc)

# 运行测试
ctest

# 安装
sudo make install
```

### 7.2 使用示例
```bash
# 训练小模型
./examples/train_simple_gpt

# 生成文本
./examples/generate_text

# 分布式训练
mpirun -np 4 ./examples/train_parallel_gpt
```

## 8. 教学价值

### 8.1 核心概念教学
1. **张量操作**：理解深度学习中的数据结构
2. **神经网络层**：理解前向和反向传播
3. **注意力机制**：理解Transformer的核心创新
4. **并行计算**：理解分布式训练的基本原理
5. **优化算法**：理解模型训练的数学基础

### 8.2 实践技能培养
1. **C++编程**：现代C++特性和最佳实践
2. **内存管理**：理解GPU内存和性能优化
3. **并行编程**：MPI和多线程编程
4. **软件工程**：模块化设计和测试
5. **性能分析**：理解和优化训练性能

### 8.3 研究扩展方向
1. **新架构支持**：添加新的模型架构
2. **高级并行**：实现更复杂的并行策略
3. **量化训练**：添加低精度训练支持
4. **模型压缩**：实现剪枝和量化
5. **分布式优化**：改进通信和负载均衡

## 9. 项目特色

### 9.1 教育特色
- **代码简洁**：去除复杂优化，突出核心概念
- **注释详细**：每个函数和类都有详细说明
- **结构清晰**：模块化设计，易于理解
- **示例丰富**：提供多个完整的使用示例

### 9.2 技术特色
- **现代C++**：使用C++17/20特性
- **类型安全**：强类型系统和编译时检查
- **内存安全**：RAII和智能指针
- **可扩展性**：插件式架构设计
- **跨平台**：支持Linux、macOS、Windows

### 9.3 实用特色
- **完整功能**：支持实际的模型训练
- **性能良好**：针对教育场景优化
- **文档完善**：提供详细的教程和API文档
- **测试覆盖**：全面的单元测试和集成测试
- **社区支持**：开放的贡献和问题讨论

这个C++教育版本的Megatron-LM将为学生和研究者提供一个理解大规模语言模型训练原理的优秀平台，同时保持代码的可读性和可扩展性。

---

*本文档提供了一个完整的C++教育版本Megatron-LM实现计划，涵盖了系统架构、核心模块、实现步骤、示例代码和教学价值等方面。*