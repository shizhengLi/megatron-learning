# Megatron-CPP-Edu 并行计算框架文档

## 目录
1. [概述](#概述)
2. [并行计算基础](#并行计算基础)
3. [张量并行](#张量并行)
4. [数据并行](#数据并行)
5. [流水线并行](#流水线并行)
6. [混合并行](#混合并行)
7. [MPI通信](#mpi通信)
8. [性能优化](#性能优化)
9. [使用示例](#使用示例)
10. [最佳实践](#最佳实践)

## 概述

Megatron-CPP-Edu 实现了一个完整的大规模语言模型并行训练框架，支持多种并行策略，旨在为学生和研究人员提供理解和实践分布式深度学习的平台。

### 设计目标
- **教育性**：代码结构清晰，注释详细，易于理解并行计算的原理
- **实用性**：支持实际的分布式训练，可用于小规模模型训练
- **可扩展性**：模块化设计，便于添加新的并行策略和优化技术
- **完整性**：涵盖从基础通信到高级并行策略的完整实现

### 核心特性
- 支持张量并行（Tensor Parallelism）
- 支持数据并行（Data Parallelism）
- 支持流水线并行（Pipeline Parallelism）
- 支持混合并行（Hybrid Parallelism）
- 基于MPI的高效通信
- 完整的错误处理和资源管理

## 并行计算基础

### 什么是并行计算？

并行计算是指同时使用多个计算资源来解决计算问题的技术。在深度学习中，并行计算主要用于：
1. **加速训练**：通过分布式计算减少训练时间
2. **扩大模型规模**：训练单个GPU无法容纳的大模型
3. **处理大数据集**：处理无法放入单个内存的大规模数据集

### 并行计算的分类

#### 1. 数据并行（Data Parallelism）
- **原理**：将数据分割到多个设备上，每个设备维护完整的模型副本
- **优点**：实现简单，通信开销相对较小
- **缺点**：每个设备需要存储完整的模型参数

#### 2. 模型并行（Model Parallelism）
- **原理**：将模型分割到多个设备上，每个设备处理模型的一部分
- **优点**：可以训练更大的模型
- **缺点**：通信开销大，实现复杂

#### 3. 流水线并行（Pipeline Parallelism）
- **原理**：将模型的不同层分配到不同设备，形成流水线
- **优点**：平衡计算和通信
- **缺点**：存在流水线气泡，效率受限

### 并行度计算

```cpp
// 总并行度 = 数据并行度 × 张量并行度 × 流水线并行度
int total_parallel_degree = data_parallel_size * 
                           tensor_parallel_size * 
                           pipeline_parallel_size;
```

## 张量并行

### 基本概念

张量并行是指将模型中的大张量（如权重矩阵）分割到多个设备上进行计算。Megatron-LM的核心创新就是提出了高效的张量并行策略。

### 实现原理

#### 1. 列并行（Column Parallelism）

对于线性层 `Y = X * A`，其中 `A` 是权重矩阵：

```
将矩阵A按列分割：
A = [A₁, A₂, ..., Aₙ]

每个设备计算：
Yᵢ = X * Aᵢ
```

**代码实现**：
```cpp
class ColumnParallelLinear : public Layer {
public:
    Tensor forward(const Tensor& input) {
        // 本地矩阵乘法: input @ weight.T
        Tensor local_output = input.matmul(weight_.transpose());
        
        // 如果是张量并行，需要all-reduce输出
        if (TensorParallelContext::instance().is_enabled()) {
            all_reduce_output(local_output);
        }
        
        return local_output;
    }
};
```

#### 2. 行并行（Row Parallelism）

对于线性层 `Y = X * A`，其中 `A` 是权重矩阵：

```
将矩阵A按行分割：
A = [A₁; A₂; ...; Aₙ]

每个设备计算：
Yᵢ = X * Aᵢ
```

**代码实现**：
```cpp
class RowParallelLinear : public Layer {
public:
    Tensor forward(const Tensor& input) {
        // 如果是张量并行，需要all-gather输入
        if (TensorParallelContext::instance().is_enabled()) {
            all_gather_input(input);
        }
        
        // 本地矩阵乘法: input @ weight.T
        Tensor local_output = input.matmul(weight_.transpose());
        
        return local_output;
    }
};
```

### 注意力机制的张量并行

在多头注意力机制中，我们使用列并行进行Q、K、V投影，使用行并行进行输出投影：

```cpp
class TensorParallelMultiHeadAttention : public Layer {
private:
    // 使用列并行进行Q、K、V投影
    std::shared_ptr<ColumnParallelLinear> q_proj_;
    std::shared_ptr<ColumnParallelLinear> k_proj_;
    std::shared_ptr<ColumnParallelLinear> v_proj_;
    
    // 使用行并行进行输出投影
    std::shared_ptr<RowParallelLinear> out_proj_;
};
```

### 通信开销分析

张量并行的通信开销主要来自：
1. **All-Reduce**：在列并行中同步梯度
2. **All-Gather**：在行并行中收集输入
3. **Reduce-Scatter**：在某些实现中分散结果

**通信量计算**：
```
列并行通信量：O(batch_size * seq_len * hidden_size)
行并行通信量：O(batch_size * seq_len * hidden_size)
```

## 数据并行

### 基本概念

数据并行是最常用的并行策略，其核心思想是将数据分割到多个设备上，每个设备维护完整的模型副本，独立计算梯度，然后同步梯度更新模型。

### 实现原理

#### 1. 数据分割

```cpp
// 数据分割示例
class DataParallelDataLoader {
public:
    std::shared_ptr<Dataset> get_local_dataset() const {
        // 根据rank和world_size分割数据集
        int samples_per_rank = global_dataset_size / world_size_;
        int start_idx = rank_ * samples_per_rank;
        int end_idx = start_idx + samples_per_rank;
        
        return dataset_->subset(start_idx, end_idx);
    }
};
```

#### 2. 梯度同步

```cpp
class DataParallelTrainer {
public:
    void synchronize_gradients() {
        auto gradients = model_->gradients();
        
        for (auto& grad : gradients) {
            // All-Reduce梯度
            comm_.all_reduce(grad);
            
            // 平均梯度
            for (int i = 0; i < grad.size(); ++i) {
                grad[i] /= world_size_;
            }
        }
    }
};
```

### 分布式数据并行（DDP）

```cpp
class DistributedDataParallel {
public:
    Tensor forward(const Tensor& input) {
        input_cache_ = input;
        return model_->forward(input);
    }
    
    Tensor backward(const Tensor& grad_output) {
        // 反向传播
        Tensor grad_input = model_->backward(grad_output);
        
        // 同步梯度
        sync_gradients();
        
        return grad_input;
    }
    
private:
    void sync_gradients() {
        auto gradients = model_->gradients();
        trainer_.synchronize_gradients(gradients);
    }
};
```

### 梯度累积

对于大批量训练，可以使用梯度累积来减少内存使用：

```cpp
void train_with_gradient_accumulation(int accumulation_steps) {
    for (int step = 0; step < total_steps; ++step) {
        // 前向传播
        Tensor output = model_->forward(input);
        
        // 计算损失
        float loss = compute_loss(output, target);
        
        // 反向传播
        Tensor grad_output = compute_gradient(output, target);
        model_->backward(grad_output);
        
        // 每accumulation_steps步更新一次参数
        if ((step + 1) % accumulation_steps == 0) {
            // 同步梯度
            synchronize_gradients();
            
            // 更新参数
            optimizer_->step();
            
            // 清零梯度
            model_->zero_grad();
        }
    }
}
```

### 性能优化

#### 1. 梯度压缩

```cpp
void compress_gradients(std::vector<Tensor>& gradients) {
    for (auto& grad : gradients) {
        // 量化梯度到8位
        quantize_tensor(grad, INT8);
    }
}
```

#### 2. 异步通信

```cpp
void async_synchronize_gradients() {
    std::vector<MPI_Request> requests;
    
    for (auto& grad : gradients) {
        MPI_Request request;
        MPI_Iallreduce(MPI_IN_PLACE, grad.data(), grad.size(), 
                     MPI_FLOAT, MPI_SUM, comm_, &request);
        requests.push_back(request);
    }
    
    // 等待所有通信完成
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}
```

## 流水线并行

### 基本概念

流水线并行将模型的不同层分配到不同的设备上，形成计算流水线。每个设备负责模型的一部分，数据在设备间流动。

### 实现原理

#### 1. 模型分割

```cpp
class PipelineParallelModel {
public:
    PipelineParallelModel(int pipeline_parallel_size) {
        // 将模型分割到不同的流水线阶段
        split_model_into_stages(pipeline_parallel_size);
    }
    
private:
    std::vector<std::shared_ptr<Layer>> stages_;
    
    void split_model_into_stages(int num_stages) {
        // 将模型层分配到不同的阶段
        int layers_per_stage = total_layers_ / num_stages;
        
        for (int i = 0; i < num_stages; ++i) {
            int start_idx = i * layers_per_stage;
            int end_idx = (i == num_stages - 1) ? total_layers_ : start_idx + layers_per_stage;
            
            auto stage = create_model_stage(start_idx, end_idx);
            stages_.push_back(stage);
        }
    }
};
```

#### 2. 流水线调度

```cpp
class PipelineScheduler {
public:
    void execute_pipeline(const Tensor& input, int num_microbatches) {
        for (int microbatch = 0; microbatch < num_microbatches; ++microbatch) {
            // 前向传播
            for (int stage = 0; stage < num_stages_; ++stage) {
                Tensor stage_input = get_stage_input(stage, microbatch);
                Tensor stage_output = stages_[stage]->forward(stage_input);
                
                // 发送到下一阶段
                if (stage < num_stages_ - 1) {
                    send_to_next_stage(stage_output, stage + 1);
                }
            }
            
            // 反向传播
            for (int stage = num_stages_ - 1; stage >= 0; --stage) {
                Tensor stage_grad = get_stage_gradient(stage, microbatch);
                Tensor stage_input_grad = stages_[stage]->backward(stage_grad);
                
                // 发送到前一阶段
                if (stage > 0) {
                    send_to_previous_stage(stage_input_grad, stage - 1);
                }
            }
        }
    }
};
```

### 流水线策略

#### 1. GPipe（Google Pipeline）

```cpp
class GPipeScheduler : public PipelineScheduler {
public:
    void execute_schedule() {
        // GPipe: 收集所有microbatch的输出，然后反向传播
        for (int microbatch = 0; microbatch < num_microbatches_; ++microbatch) {
            forward_pass(microbatch);
        }
        
        for (int microbatch = num_microbatches_ - 1; microbatch >= 0; --microbatch) {
            backward_pass(microbatch);
        }
    }
};
```

#### 2. PipeDream

```cpp
class PipeDreamScheduler : public PipelineScheduler {
public:
    void execute_schedule() {
        // PipeDream: 1F1B（1个前向，1个反向）调度
        for (int step = 0; step < total_steps_; ++step) {
            if (step < num_stages_) {
                // Warm-up phase: 只做前向
                forward_pass(step);
            } else if (step < total_steps_ - num_stages_) {
                // Steady state: 同时做前向和反向
                int forward_microbatch = step;
                int backward_microbatch = step - num_stages_;
                
                forward_pass(forward_microbatch);
                backward_pass(backward_microbatch);
            } else {
                // Cool-down phase: 只做反向
                int backward_microbatch = step - num_stages_;
                backward_pass(backward_microbatch);
            }
        }
    }
};
```

## 混合并行

### 基本概念

混合并行结合了多种并行策略，通常包括：
- **数据并行**：在最外层进行数据分割
- **张量并行**：在每个节点内进行张量分割
- **流水线并行**：在节点间进行流水线处理

### 架构设计

```
全局架构：
┌─────────────────────────────────────────────────┐
│                   数据并行                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   节点 0    │  │   节点 1    │  │   节点 2    │ │
│  │  ┌─────────┐ │  │  ┌─────────┐ │  │  ┌─────────┐ │
│  │  │张量并行│ │  │  │张量并行│ │  │  │张量并行│ │
│  │  │设备 0-3 │ │  │  │设备 0-3 │ │  │  │设备 0-3 │ │
│  │  └─────────┘ │  │  └─────────┘ │  │  └─────────┘ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────┘
```

### 实现原理

#### 1. 混合并行配置

```cpp
struct HybridParallelConfig {
    int data_parallel_size = 1;
    int tensor_parallel_size = 1;
    int pipeline_parallel_size = 1;
    
    int get_total_world_size() const {
        return data_parallel_size * tensor_parallel_size * pipeline_parallel_size;
    }
    
    bool validate() const {
        int total = get_total_world_size();
        return total > 0 && 
               data_parallel_size > 0 && 
               tensor_parallel_size > 0 && 
               pipeline_parallel_size > 0;
    }
};
```

#### 2. 混合并行训练器

```cpp
class HybridParallelTrainer {
public:
    void hybrid_train_step(const Tensor& inputs, const Tensor& targets) {
        // 1. 数据并行：每个数据并行副本独立训练
        if (data_parallel_size_ > 1) {
            // 执行本地训练步骤
            local_train_step(inputs, targets);
            
            // 同步数据并行梯度
            sync_data_parallel_gradients();
        }
        
        // 2. 张量并行：在节点内同步梯度
        if (tensor_parallel_size_ > 1) {
            sync_tensor_parallel_gradients();
        }
        
        // 3. 流水线并行：协调流水线执行
        if (pipeline_parallel_size_ > 1) {
            execute_pipeline_schedule();
        }
    }
    
private:
    void local_train_step(const Tensor& inputs, const Tensor& targets) {
        // 在本地设备上执行训练步骤
        Tensor output = model_->forward(inputs);
        Tensor grad_output = compute_gradient(output, targets);
        model_->backward(grad_output);
    }
    
    void sync_data_parallel_gradients() {
        // 在数据并行组内同步梯度
        auto gradients = model_->gradients();
        for (auto& grad : gradients) {
            dp_comm_.all_reduce(grad);
            grad /= data_parallel_size_;
        }
    }
    
    void sync_tensor_parallel_gradients() {
        // 在张量并行组内同步梯度
        auto gradients = model_->gradients();
        for (auto& grad : gradients) {
            tp_comm_.all_reduce(grad);
            grad /= tensor_parallel_size_;
        }
    }
};
```

### 通信组管理

```cpp
class ParallelCommGroups {
public:
    void initialize_comm_groups(const HybridParallelConfig& config) {
        // 创建数据并行通信组
        create_data_parallel_comm(config);
        
        // 创建张量并行通信组
        create_tensor_parallel_comm(config);
        
        // 创建流水线并行通信组
        create_pipeline_parallel_comm(config);
    }
    
private:
    MPI_Comm dp_comm_;  // 数据并行通信组
    MPI_Comm tp_comm_;  // 张量并行通信组
    MPI_Comm pp_comm_;  // 流水线并行通信组
    
    void create_data_parallel_comm(const HybridParallelConfig& config) {
        // 数据并行组：相同张量并行和流水线并行的进程在同一组
        int color = get_tensor_parallel_rank() * config.pipeline_parallel_size + 
                   get_pipeline_parallel_rank();
        
        MPI_Comm_split(MPI_COMM_WORLD, color, get_global_rank(), &dp_comm_);
    }
    
    void create_tensor_parallel_comm(const HybridParallelConfig& config) {
        // 张量并行组：相同节点内的进程在同一组
        int color = get_data_parallel_rank() * config.pipeline_parallel_size + 
                   get_pipeline_parallel_rank();
        
        MPI_Comm_split(MPI_COMM_WORLD, color, get_global_rank(), &tp_comm_);
    }
    
    void create_pipeline_parallel_comm(const HybridParallelConfig& config) {
        // 流水线并行组：相同流水线阶段的进程在同一组
        int color = get_data_parallel_rank() * config.tensor_parallel_size + 
                   get_tensor_parallel_rank();
        
        MPI_Comm_split(MPI_COMM_WORLD, color, get_global_rank(), &pp_comm_);
    }
};
```

## MPI通信

### MPI基础

MPI（Message Passing Interface）是并行计算的标准通信接口。Megatron-CPP-Edu使用MPI实现进程间通信。

#### 1. 基本通信操作

```cpp
class MPICommunicator {
public:
    // 点对点通信
    void send(const Tensor& tensor, int dest_rank, int tag = 0) {
        MPI_Send(tensor.data(), tensor.size(), MPI_FLOAT, 
                dest_rank, tag, comm_);
    }
    
    void recv(Tensor& tensor, int src_rank, int tag = 0) {
        MPI_Status status;
        MPI_Recv(tensor.data(), tensor.size(), MPI_FLOAT, 
                src_rank, tag, comm_, &status);
    }
    
    // 集合通信
    void all_reduce(Tensor& tensor) {
        MPI_Allreduce(MPI_IN_PLACE, tensor.data(), tensor.size(), 
                      MPI_FLOAT, MPI_SUM, comm_);
    }
    
    void broadcast(Tensor& tensor, int root_rank) {
        MPI_Bcast(tensor.data(), tensor.size(), MPI_FLOAT, 
                 root_rank, comm_);
    }
    
    void barrier() {
        MPI_Barrier(comm_);
    }
};
```

#### 2. 通信优化

```cpp
class OptimizedMPICommunicator {
public:
    // 异步通信
    void async_all_reduce(Tensor& tensor, MPI_Request* request) {
        MPI_Iallreduce(MPI_IN_PLACE, tensor.data(), tensor.size(),
                       MPI_FLOAT, MPI_SUM, comm_, request);
    }
    
    // 非阻塞通信
    void non_blocking_send(const Tensor& tensor, int dest_rank, 
                          MPI_Request* request) {
        MPI_Isend(tensor.data(), tensor.size(), MPI_FLOAT,
                  dest_rank, 0, comm_, request);
    }
    
    void non_blocking_recv(Tensor& tensor, int src_rank,
                          MPI_Request* request) {
        MPI_Irecv(tensor.data(), tensor.size(), MPI_FLOAT,
                  src_rank, 0, comm_, request);
    }
};
```

### 通信子管理

```cpp
class CommManager {
public:
    // 创建通信子
    void create_communicators(const HybridParallelConfig& config) {
        create_data_parallel_comm(config);
        create_tensor_parallel_comm(config);
        create_model_parallel_comm(config);
    }
    
    // 获取通信子
    MPI_Comm get_data_parallel_comm() const { return dp_comm_; }
    MPI_Comm get_tensor_parallel_comm() const { return tp_comm_; }
    MPI_Comm get_model_parallel_comm() const { return mp_comm_; }
    
private:
    MPI_Comm dp_comm_;  // 数据并行通信子
    MPI_Comm tp_comm_;  // 张量并行通信子
    MPI_Comm mp_comm_;  // 模型并行通信子
    
    void create_data_parallel_comm(const HybridParallelConfig& config) {
        int color = get_model_parallel_id();
        MPI_Comm_split(MPI_COMM_WORLD, color, get_rank(), &dp_comm_);
    }
    
    void create_tensor_parallel_comm(const HybridParallelConfig& config) {
        int color = get_data_parallel_id();
        MPI_Comm_split(MPI_COMM_WORLD, color, get_rank(), &tp_comm_);
    }
};
```

## 性能优化

### 通信优化

#### 1. 通信重叠

```cpp
class OverlappedCommunication {
public:
    void train_step_with_overlap(const Tensor& input, const Tensor& target) {
        // 启动前向传播
        Tensor output = model_->forward(input);
        
        // 计算损失并启动反向传播
        float loss = compute_loss(output, target);
        Tensor grad_output = compute_gradient(output, target);
        
        // 启动异步梯度同步
        std::vector<MPI_Request> requests;
        start_async_gradient_sync(requests);
        
        // 在通信的同时进行其他计算
        do_computation_during_communication();
        
        // 等待通信完成
        wait_for_communication(requests);
        
        // 更新参数
        optimizer_->step();
    }
    
private:
    void start_async_gradient_sync(std::vector<MPI_Request>& requests) {
        auto gradients = model_->gradients();
        for (auto& grad : gradients) {
            MPI_Request request;
            MPI_Iallreduce(MPI_IN_PLACE, grad.data(), grad.size(),
                           MPI_FLOAT, MPI_SUM, comm_, &request);
            requests.push_back(request);
        }
    }
};
```

#### 2. 梯度压缩

```cpp
class GradientCompression {
public:
    void compress_and_sync_gradients(std::vector<Tensor>& gradients) {
        for (auto& grad : gradients) {
            // 压缩梯度
            Tensor compressed = compress_gradient(grad);
            
            // 同步压缩后的梯度
            sync_compressed_gradient(compressed);
            
            // 解压缩
            decompress_gradient(compressed, grad);
        }
    }
    
private:
    Tensor compress_gradient(const Tensor& grad) {
        // 使用8位量化压缩梯度
        Tensor compressed(grad.shape());
        for (int i = 0; i < grad.size(); ++i) {
            compressed[i] = quantize_to_8bit(grad[i]);
        }
        return compressed;
    }
    
    void decompress_gradient(const Tensor& compressed, Tensor& grad) {
        // 解压缩梯度
        for (int i = 0; i < grad.size(); ++i) {
            grad[i] = dequantize_from_8bit(compressed[i]);
        }
    }
};
```

### 内存优化

#### 1. 激活检查点

```cpp
class ActivationCheckpointing {
public:
    Tensor forward_with_checkpoint(const Tensor& input) {
        // 不保存所有中间激活，只保存检查点
        if (should_save_checkpoint()) {
            checkpoints_.push_back(input.clone());
        }
        
        Tensor output = layer_->forward(input);
        
        if (should_save_checkpoint()) {
            checkpoints_.push_back(output.clone());
        }
        
        return output;
    }
    
    Tensor backward_with_recompute(const Tensor& grad_output) {
        // 重新计算中间激活
        Tensor checkpoint = checkpoints_.back();
        checkpoints_.pop_back();
        
        // 从检查点重新计算
        Tensor recomputed = recompute_activation(checkpoint);
        
        // 继续反向传播
        return layer_->backward(recomputed, grad_output);
    }
    
private:
    std::vector<Tensor> checkpoints_;
    
    bool should_save_checkpoint() {
        // 决定是否保存检查点的策略
        return current_layer_ % checkpoint_frequency_ == 0;
    }
};
```

#### 2. 内存分配优化

```cpp
class MemoryPool {
public:
    Tensor allocate_tensor(const std::vector<int>& shape) {
        size_t required_size = calculate_size(shape);
        
        // 查找合适的内存块
        auto it = find_free_block(required_size);
        if (it != free_blocks_.end()) {
            Tensor tensor = it->second;
            free_blocks_.erase(it);
            used_blocks_[tensor] = required_size;
            return tensor;
        }
        
        // 分配新内存
        return Tensor(shape);
    }
    
    void deallocate_tensor(const Tensor& tensor) {
        auto it = used_blocks_.find(tensor);
        if (it != used_blocks_.end()) {
            free_blocks_[it->second] = tensor;
            used_blocks_.erase(it);
        }
    }
    
private:
    std::map<size_t, Tensor> free_blocks_;
    std::map<Tensor, size_t> used_blocks_;
};
```

## 使用示例

### 1. 基本张量并行示例

```cpp
#include "core/parallel/tensor_parallel.h"
#include "core/optimizers/adamw.h"

int main(int argc, char** argv) {
    // 初始化MPI
    MPICommunicator::instance().initialize(&argc, &argv);
    
    // 获取并行信息
    int world_size = MPICommunicator::instance().world_size();
    int rank = MPICommunicator::instance().rank();
    
    // 初始化张量并行
    TensorParallelContext::instance().initialize(world_size, rank);
    
    // 创建张量并行模型
    TensorParallelModelBuilder builder(world_size);
    builder.set_vocab_size(1000);
    builder.set_embed_dim(256);
    builder.set_num_layers(6);
    builder.set_num_heads(8);
    builder.set_ffn_dim(1024);
    
    auto model = builder.build_gpt_model();
    
    // 创建优化器
    auto optimizer = std::make_shared<AdamW>(0.001f);
    
    // 训练循环
    for (int epoch = 0; epoch < 10; ++epoch) {
        Tensor input({32, 128});
        Tensor target({32, 128});
        
        // 前向传播
        Tensor output = model->forward(input);
        
        // 计算损失和梯度
        float loss = compute_loss(output, target);
        Tensor grad_output = compute_gradient(output, target);
        
        // 反向传播
        model->backward(grad_output);
        
        // 更新参数
        optimizer->step(model->parameters(), model->gradients());
        
        // 清零梯度
        model->zero_grad();
        
        if (rank == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }
    
    // 清理
    MPICommunicator::instance().finalize();
    return 0;
}
```

### 2. 数据并行示例

```cpp
#include "core/parallel/data_parallel.h"
#include "core/optimizers/adamw.h"

int main(int argc, char** argv) {
    // 初始化MPI
    MPICommunicator::instance().initialize(&argc, &argv);
    
    // 配置数据并行
    DataParallelConfig dp_config;
    dp_config.world_size = MPICommunicator::instance().world_size();
    dp_config.rank = MPICommunicator::instance().rank();
    dp_config.global_batch_size = 64;
    dp_config.local_batch_size = 16;
    
    // 创建模型
    auto model = std::make_shared<TransformerClassifier>(1000, 128, 256, 8, 6, 1024, 10);
    
    // 创建DDP模型
    auto ddp_model = std::make_shared<DistributedDataParallel>(model, dp_config);
    
    // 创建分布式优化器
    auto optimizer = std::make_shared<AdamW>(0.001f);
    auto dist_optimizer = std::make_shared<DistributedOptimizer>(optimizer, dp_config);
    
    // 创建数据并行训练器
    DataParallelTrainer trainer(dp_config);
    trainer.setup_model_and_optimizer(ddp_model, dist_optimizer);
    
    // 训练循环
    for (int epoch = 0; epoch < 10; ++epoch) {
        Tensor inputs({dp_config.local_batch_size, 128});
        Tensor targets({dp_config.local_batch_size, 10});
        
        // 执行训练步骤
        trainer.train_step(inputs, targets);
        
        if (dp_config.rank == 0) {
            float avg_loss = trainer.get_average_loss();
            std::cout << "Epoch " << epoch << ", Loss: " << avg_loss << std::endl;
        }
    }
    
    // 清理
    MPICommunicator::instance().finalize();
    return 0;
}
```

### 3. 混合并行示例

```cpp
#include "core/parallel/tensor_parallel.h"
#include "core/parallel/data_parallel.h"
#include "distributed/communication.h"

int main(int argc, char** argv) {
    // 初始化MPI
    MPICommunicator::instance().initialize(&argc, &argv);
    
    // 配置混合并行
    HybridParallelConfig config;
    config.data_parallel_size = 2;
    config.tensor_parallel_size = 2;
    config.pipeline_parallel_size = 1;
    
    // 创建混合并行训练器
    HybridParallelTrainer trainer(config);
    
    // 创建模型
    auto model = create_large_model();
    auto optimizer = std::make_shared<AdamW>(0.0001f);
    
    trainer.setup_model_and_optimizer(model, optimizer);
    
    // 训练循环
    for (int epoch = 0; epoch < 10; ++epoch) {
        Tensor inputs({8, 512});
        Tensor targets({8, 1000});
        
        // 执行混合并行训练步骤
        trainer.hybrid_train_step(inputs, targets);
        
        if (trainer.get_global_rank() == 0) {
            std::cout << "Epoch " << epoch << " completed" << std::endl;
        }
    }
    
    // 清理
    MPICommunicator::instance().finalize();
    return 0;
}
```

## 最佳实践

### 1. 并行策略选择

#### 选择标准

1. **模型大小**：
   - 小模型（<1B参数）：数据并行
   - 中等模型（1B-10B参数）：张量并行
   - 大模型（>10B参数）：混合并行

2. **硬件配置**：
   - 单机多GPU：张量并行 + 数据并行
   - 多机多GPU：混合并行

3. **通信带宽**：
   - 高带宽（NVLink）：张量并行
   - 低带宽（以太网）：数据并行

#### 性能评估

```cpp
class ParallelPerformanceAnalyzer {
public:
    void analyze_parallel_efficiency() {
        // 计算强扩展性
        float strong_scaling = calculate_strong_scaling();
        
        // 计算弱扩展性
        float weak_scaling = calculate_weak_scaling();
        
        // 计算通信开销
        float communication_overhead = calculate_communication_overhead();
        
        // 生成性能报告
        generate_performance_report(strong_scaling, weak_scaling, 
                                 communication_overhead);
    }
    
private:
    float calculate_strong_scaling() {
        // 强扩展性：固定问题规模，增加处理器数量
        float single_time = measure_single_node_time();
        float parallel_time = measure_parallel_time();
        
        return single_time / (parallel_time * world_size_);
    }
    
    float calculate_weak_scaling() {
        // 弱扩展性：每个处理器的问题规模固定
        float single_efficiency = measure_single_node_efficiency();
        float parallel_efficiency = measure_parallel_efficiency();
        
        return parallel_efficiency / single_efficiency;
    }
};
```

### 2. 调试和监控

#### 并行调试技巧

```cpp
class ParallelDebugger {
public:
    void verify_gradient_synchronization() {
        // 在所有进程上计算梯度和
        auto gradients = model_->gradients();
        float local_sum = compute_gradient_sum(gradients);
        
        // 同步梯度和
        float global_sum = all_reduce_sum(local_sum);
        
        // 验证梯度是否正确同步
        float expected_sum = local_sum * world_size_;
        if (std::abs(global_sum - expected_sum) > 1e-6f) {
            std::cerr << "Gradient synchronization error!" << std::endl;
        }
    }
    
    void verify_parameter_consistency() {
        // 检查所有进程的参数是否一致
        auto parameters = model_->parameters();
        float local_norm = compute_parameter_norm(parameters);
        
        // 同步参数范数
        float global_norm = all_reduce_max(local_norm);
        
        if (std::abs(local_norm - global_norm) > 1e-6f) {
            std::cerr << "Parameter inconsistency detected!" << std::endl;
        }
    }
};
```

#### 性能监控

```cpp
class ParallelProfiler {
public:
    void profile_training_step() {
        // 记录各个阶段的耗时
        start_timer("forward");
        Tensor output = model_->forward(input);
        stop_timer("forward");
        
        start_timer("backward");
        Tensor grad_input = model_->backward(grad_output);
        stop_timer("backward");
        
        start_timer("communication");
        synchronize_gradients();
        stop_timer("communication");
        
        start_timer("optimization");
        optimizer_->step();
        stop_timer("optimization");
        
        // 生成性能报告
        generate_performance_report();
    }
    
private:
    std::map<std::string, double> timers_;
    
    void start_timer(const std::string& name) {
        timers_[name] = get_current_time();
    }
    
    void stop_timer(const std::string& name) {
        double elapsed = get_current_time() - timers_[name];
        timers_[name] = elapsed;
    }
};
```

### 3. 错误处理

#### 分布式错误处理

```cpp
class ParallelErrorHandler {
public:
    void handle_distributed_errors() {
        try {
            // 执行分布式训练
            execute_distributed_training();
        } catch (const MPIException& e) {
            // 处理MPI通信错误
            handle_mpi_error(e);
        } catch (const DataParallelException& e) {
            // 处理数据并行错误
            handle_data_parallel_error(e);
        } catch (const std::exception& e) {
            // 处理其他错误
            handle_general_error(e);
        }
    }
    
private:
    void handle_mpi_error(const MPIException& e) {
        std::cerr << "MPI Error on rank " << e.get_rank() 
                  << ": " << e.what() << std::endl;
        
        // 同步错误状态
        int error_code = e.get_error_code();
        MPI_Abort(MPI_COMM_WORLD, error_code);
    }
    
    void handle_data_parallel_error(const DataParallelException& e) {
        std::cerr << "Data Parallel Error on rank " << e.get_rank()
                  << ": " << e.what() << std::endl;
        
        // 尝试恢复
        attempt_recovery();
    }
};
```

### 4. 部署建议

#### 环境配置

```bash
# 推荐的MPI配置
export MPI_NUM_THREADS=1
export OMP_NUM_THREADS=8

# 网络优化
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_TX_QUEUE_LEN=1024

# 内存配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
```

#### 启动脚本

```bash
#!/bin/bash
# 启动分布式训练

# 配置参数
NUM_NODES=4
NUM_GPUS_PER_NODE=4
MASTER_ADDR=$(hostname)
MASTER_PORT=29500

# 启动训练
mpirun -np $((NUM_NODES * NUM_GPUS_PER_NODE)) \
    --hostfile hostfile \
    --bind-to core \
    --map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    ./distributed_training \
    --num_nodes $NUM_NODES \
    --num_gpus_per_node $NUM_GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
```

这个文档提供了Megatron-CPP-Edu并行计算框架的完整介绍，涵盖了从基础概念到高级实现的各个方面。通过理解和实践这些内容，用户可以掌握大规模语言模型并行训练的核心技术。