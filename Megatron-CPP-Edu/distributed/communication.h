#pragma once

#include "core/tensor/tensor.h"
#include "core/layers/layer.h"
#include "core/optimizers/optimizer.h"
#include "core/parallel/data_parallel.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <memory>

namespace megatron {

class MPICommunicator {
public:
    static MPICommunicator& instance();
    
    void initialize(int* argc, char*** argv);
    void finalize();
    
    int world_size() const;
    int rank() const;
    bool is_initialized() const;
    
    // 基础通信操作
    void send(const Tensor& tensor, int dest_rank, int tag = 0);
    void recv(Tensor& tensor, int src_rank, int tag = 0);
    
    // 集合通信
    void all_reduce(Tensor& tensor);
    void all_gather(const std::vector<Tensor>& send_tensors, 
                   std::vector<Tensor>& recv_tensors);
    void broadcast(Tensor& tensor, int root_rank);
    void reduce(Tensor& tensor, int root_rank);
    
    // 障碍同步
    void barrier();
    
    // 获取通信器名称
    std::string get_comm_name() const;
    
    // 获取本地设备信息
    int get_local_device_id() const;
    
private:
    bool initialized_;
    int world_size_;
    int rank_;
    MPI_Comm comm_;
    
    MPICommunicator() : initialized_(false), world_size_(1), rank_(0), comm_(MPI_COMM_WORLD) {}
    
    // 内部辅助函数
    void check_initialized() const;
    MPI_Datatype get_mpi_dtype(const Tensor& tensor) const;
    void validate_tensor_shape(const Tensor& tensor) const;
    
    // 禁止拷贝
    MPICommunicator(const MPICommunicator&) = delete;
    MPICommunicator& operator=(const MPICommunicator&) = delete;
};

// 简化的分布式训练包装器
class DistributedTrainer {
public:
    DistributedTrainer(int world_size = 1, int rank = 0);
    
    // 设置模型和优化器
    void setup_model_and_optimizer(std::shared_ptr<Layer> model,
                                 std::shared_ptr<Optimizer> optimizer);
    
    // 分布式训练步骤
    void distributed_train_step(const Tensor& input_ids, const Tensor& targets);
    
    // 梯度同步
    void synchronize_gradients(std::vector<Tensor>& gradients);
    
    // 参数广播
    void broadcast_parameters(std::vector<Tensor>& parameters);
    
    // 获取本地批大小
    int get_local_batch_size(int global_batch_size) const;
    
    // 获取分布式状态
    int world_size() const { return world_size_; }
    int rank() const { return rank_; }
    bool is_distributed() const { return world_size_ > 1; }
    
private:
    int world_size_;
    int rank_;
    
    std::shared_ptr<Layer> model_;
    std::shared_ptr<Optimizer> optimizer_;
    
    MPICommunicator& comm_;
    
    // 内部函数
    void validate_setup() const;
    void all_reduce_gradients(std::vector<Tensor>& gradients);
    void sync_model_parameters();
};

// 分布式数据并行层包装器
class DistributedDataParallel {
public:
    DistributedDataParallel(std::shared_ptr<Layer> model, 
                           const DataParallelConfig& config = DataParallelConfig());
    
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    
    std::vector<Tensor> parameters() const;
    std::vector<Tensor> gradients() const;
    
    void train(bool is_training);
    bool is_training() const;
    
    // 同步相关
    void sync_parameters();
    void sync_gradients();
    
private:
    std::shared_ptr<Layer> model_;
    int world_size_;
    int rank_;
    
    DistributedTrainer trainer_;
    Tensor input_cache_;
    
    // 内部函数
    void initialize_parallel_context();
};

// 并行状态管理
class ParallelState {
public:
    static ParallelState& instance();
    
    void initialize_tensor_parallel(int tensor_parallel_size, int tensor_parallel_rank);
    void initialize_pipeline_parallel(int pipeline_parallel_size, int pipeline_parallel_rank);
    void initialize_data_parallel(int data_parallel_size, int data_parallel_rank);
    
    // 获取并行信息
    int get_tensor_parallel_size() const;
    int get_tensor_parallel_rank() const;
    int get_pipeline_parallel_size() const;
    int get_pipeline_parallel_rank() const;
    int get_data_parallel_size() const;
    int get_data_parallel_rank() const;
    
    // 获取全局信息
    int get_world_size() const;
    int get_global_rank() const;
    
    // 检查是否启用某种并行
    bool is_tensor_parallel_enabled() const;
    bool is_pipeline_parallel_enabled() const;
    bool is_data_parallel_enabled() const;
    
    // 获取通信器
    MPI_Comm get_tensor_parallel_comm() const;
    MPI_Comm get_pipeline_parallel_comm() const;
    MPI_Comm get_data_parallel_comm() const;
    
private:
    // 张量并行状态
    int tensor_parallel_size_;
    int tensor_parallel_rank_;
    MPI_Comm tensor_parallel_comm_;
    
    // 流水线并行状态
    int pipeline_parallel_size_;
    int pipeline_parallel_rank_;
    MPI_Comm pipeline_parallel_comm_;
    
    // 数据并行状态
    int data_parallel_size_;
    int data_parallel_rank_;
    MPI_Comm data_parallel_comm_;
    
    ParallelState();
    
    // 禁止拷贝
    ParallelState(const ParallelState&) = delete;
    ParallelState& operator=(const ParallelState&) = delete;
};

// 分布式训练配置
struct DistributedConfig {
    int world_size = 1;
    int rank = 0;
    int tensor_parallel_size = 1;
    int pipeline_parallel_size = 1;
    int data_parallel_size = 1;
    int global_batch_size = 32;
    int micro_batch_size = 4;
    bool use_fp16 = false;
    bool use_ddp = true;
    std::string master_addr = "localhost";
    int master_port = 29500;
    
    // 验证配置
    bool validate() const;
    
    // 获取本地批大小
    int get_local_batch_size() const;
    
    // 获取梯度累积步数
    int get_gradient_accumulation_steps() const;
};

// 分布式训练辅助函数
namespace distributed_utils {
    // 初始化分布式环境
    void initialize_distributed(const DistributedConfig& config);
    
    // 清理分布式环境
    void cleanup_distributed();
    
    // 获取当前设备
    int get_current_device();
    
    // 设置设备
    void set_device(int device_id);
    
    // 同步所有进程
    void sync_all_processes();
    
    // 只在主进程执行
    void master_do(std::function<void()> func);
    
    // 打印分布式信息
    void print_distributed_info();
    
    // 检查是否支持分布式
    bool is_distributed_supported();
    
    // 获取节点内通信器
    MPI_Comm get_local_comm();
    
    // 获取节点间通信器
    MPI_Comm get_cross_comm();
}

} // namespace megatron