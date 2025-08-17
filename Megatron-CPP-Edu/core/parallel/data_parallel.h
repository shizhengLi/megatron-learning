#pragma once

#include "core/tensor/tensor.h"
#include "core/layers/layer.h"
#include "core/optimizers/optimizer.h"
#include "core/data/dataset.h"
#include "../distributed/communication.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace megatron {

// 数据并行配置
struct DataParallelConfig {
    int world_size = 1;
    int rank = 0;
    int global_batch_size = 32;
    int local_batch_size = 8;
    bool sync_bn = true;
    bool find_unused_parameters = false;
    float bucket_cap_mb = 25.0f;
    
    // 验证配置
    bool validate() const;
    
    // 获取梯度累积步数
    int get_gradient_accumulation_steps() const;
};

// 数据并行训练器
class DataParallelTrainer {
public:
    DataParallelTrainer(const DataParallelConfig& config);
    
    // 设置模型和优化器
    void setup_model_and_optimizer(std::shared_ptr<Layer> model,
                                 std::shared_ptr<Optimizer> optimizer);
    
    // 训练步骤
    void train_step(const Tensor& inputs, const Tensor& targets);
    void train_step_with_loss(const Tensor& inputs, const Tensor& targets, float loss);
    
    // 梯度同步
    void synchronize_gradients();
    void all_reduce_gradients();
    
    // 参数同步
    void synchronize_parameters();
    void broadcast_parameters();
    
    // 获取分布式状态
    int world_size() const { return config_.world_size; }
    int rank() const { return config_.rank; }
    bool is_distributed() const { return config_.world_size > 1; }
    
    // 获取本地批大小
    int get_local_batch_size() const { return config_.local_batch_size; }
    int get_global_batch_size() const { return config_.global_batch_size; }
    
    // 获取训练统计
    float get_average_loss() const;
    int get_global_step() const;
    
private:
    DataParallelConfig config_;
    
    std::shared_ptr<Layer> model_;
    std::shared_ptr<Optimizer> optimizer_;
    
    MPICommunicator& comm_;
    
    // 训练状态
    int global_step_;
    float average_loss_;
    std::vector<float> loss_history_;
    
    // 验证设置
    void validate_setup() const;
    
    // 内部函数
    void reduce_gradients(std::vector<Tensor>& gradients);
    void average_parameters(std::vector<Tensor>& parameters);
    void update_training_stats(float loss);
};

// 分布式数据并行层包装器
class DistributedDataParallel {
public:
    DistributedDataParallel(std::shared_ptr<Layer> model,
                           const DataParallelConfig& config);
    
    // 前向和反向传播
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    
    // 参数管理
    std::vector<Tensor> parameters() const;
    std::vector<Tensor> gradients() const;
    
    // 训练模式
    void train(bool is_training);
    bool is_training() const;
    
    // 同步操作
    void sync_parameters();
    void sync_gradients();
    
    // 获取内部模型
    std::shared_ptr<Layer> get_model() const { return model_; }
    
    // 获取配置
    const DataParallelConfig& get_config() const { return config_; }
    
private:
    std::shared_ptr<Layer> model_;
    DataParallelConfig config_;
    
    DataParallelTrainer trainer_;
    Tensor input_cache_;
    
    // 初始化
    void initialize_ddp();
};

// 数据并行数据加载器
class DataParallelDataLoader {
public:
    DataParallelDataLoader(std::shared_ptr<Dataset> dataset,
                          const DataParallelConfig& config,
                          bool shuffle = true);
    
    // 获取下一个批次
    std::pair<Tensor, Tensor> next_batch();
    bool has_next_batch() const;
    void reset();
    
    // 获取信息
    int num_batches() const;
    int current_batch() const;
    
    // 获取本地数据集
    std::shared_ptr<Dataset> get_local_dataset() const;
    
private:
    std::shared_ptr<Dataset> dataset_;
    std::shared_ptr<Dataset> local_dataset_;
    DataParallelConfig config_;
    bool shuffle_;
    
    int current_batch_;
    std::vector<int> indices_;
    
    // 数据分割
    void partition_dataset();
    void shuffle_indices();
    std::pair<Tensor, Tensor> create_batch(const std::vector<int>& batch_indices);
    
    // 验证
    void validate_config() const;
};

// 混合并行训练器（数据并行 + 张量并行）
class HybridParallelTrainer {
public:
    HybridParallelTrainer(const DataParallelConfig& dp_config,
                         int tensor_parallel_size = 1,
                         int pipeline_parallel_size = 1);
    
    // 设置模型和优化器
    void setup_model_and_optimizer(std::shared_ptr<Layer> model,
                                 std::shared_ptr<Optimizer> optimizer);
    
    // 混合并行训练步骤
    void hybrid_train_step(const Tensor& inputs, const Tensor& targets);
    
    // 梯度同步
    void synchronize_gradients();
    void synchronize_parameters();
    
    // 获取并行状态
    int get_data_parallel_size() const;
    int get_tensor_parallel_size() const;
    int get_pipeline_parallel_size() const;
    int get_global_rank() const;
    int get_local_rank() const;
    
    // 获取配置
    const DataParallelConfig& get_dp_config() const { return dp_config_; }
    
private:
    DataParallelConfig dp_config_;
    int tensor_parallel_size_;
    int pipeline_parallel_size_;
    
    std::shared_ptr<Layer> model_;
    std::shared_ptr<Optimizer> optimizer_;
    
    MPICommunicator& comm_;
    
    // 并行通信器
    MPI_Comm dp_comm_;
    MPI_Comm tp_comm_;
    MPI_Comm pp_comm_;
    
    // 初始化
    void initialize_parallel_communicators();
    void validate_parallel_config() const;
    
    // 内部函数
    void data_parallel_all_reduce(std::vector<Tensor>& gradients);
    void tensor_parallel_all_reduce(std::vector<Tensor>& gradients);
    void pipeline_parallel_schedule();
};

// 分布式优化器包装器
class DistributedOptimizer {
public:
    DistributedOptimizer(std::shared_ptr<Optimizer> base_optimizer,
                        const DataParallelConfig& config);
    
    // 分布式优化步骤
    void step(std::vector<Tensor>& parameters,
             std::vector<Tensor>& gradients);
    
    // 设置学习率
    void set_learning_rate(float lr);
    float get_learning_rate() const;
    
    // 梯度操作
    void zero_grad(std::vector<Tensor>& gradients);
    void clip_gradients(std::vector<Tensor>& gradients, float max_norm);
    
    // 获取基础优化器
    std::shared_ptr<Optimizer> get_base_optimizer() const { return base_optimizer_; }
    
private:
    std::shared_ptr<Optimizer> base_optimizer_;
    DataParallelConfig config_;
    
    // 分布式操作
    void all_reduce_gradients(std::vector<Tensor>& gradients);
    void average_gradients(std::vector<Tensor>& gradients);
    
    // 学习率调度
    float get_current_lr() const;
    void update_lr_scheduler();
};

// 数据并行同步工具
class DataParallelSync {
public:
    DataParallelSync(const DataParallelConfig& config);
    
    // 同步张量
    void sync_tensor(Tensor& tensor);
    void sync_tensor_list(std::vector<Tensor>& tensors);
    
    // 同步统计信息
    void sync_statistics(float& value);
    void sync_statistics(std::vector<float>& values);
    
    // 同步布尔标志
    void sync_flag(bool& flag);
    
    // 聚合操作
    float all_reduce_sum(float value);
    float all_reduce_mean(float value);
    float all_reduce_max(float value);
    float all_reduce_min(float value);
    
    // 广播操作
    void broadcast_value(float& value, int root_rank);
    void broadcast_tensor(Tensor& tensor, int root_rank);
    
    // 障碍同步
    void barrier();
    
private:
    DataParallelConfig config_;
    MPICommunicator& comm_;
    
    // 内部函数
    void validate_tensor(const Tensor& tensor) const;
    void validate_config() const;
};

// 数据并行性能监控
class DataParallelProfiler {
public:
    DataParallelProfiler(const DataParallelConfig& config);
    
    // 开始/结束计时
    void start_timer(const std::string& name);
    void stop_timer(const std::string& name);
    
    // 记录统计信息
    void record_statistic(const std::string& name, float value);
    void record_throughput(float samples_per_second);
    
    // 获取性能报告
    std::string generate_performance_report() const;
    std::map<std::string, float> get_performance_metrics() const;
    
    // 同步性能统计
    void sync_performance_metrics();
    
private:
    DataParallelConfig config_;
    MPICommunicator& comm_;
    
    // 计时器
    std::map<std::string, double> timers_;
    std::map<std::string, bool> timer_running_;
    
    // 统计信息
    std::map<std::string, std::vector<float>> statistics_;
    std::map<std::string, float> aggregated_statistics_;
    
    // 性能指标
    float throughput_samples_per_sec_;
    float throughput_mb_per_sec_;
    
    // 内部函数
    double get_current_time() const;
    void validate_timer_name(const std::string& name) const;
    void aggregate_statistics();
};

// 数据并行辅助函数
namespace data_parallel_utils {
    // 初始化数据并行环境
    void initialize_data_parallel(const DataParallelConfig& config);
    
    // 清理数据并行环境
    void cleanup_data_parallel();
    
    // 分割数据集
    std::shared_ptr<Dataset> partition_dataset(std::shared_ptr<Dataset> dataset,
                                              int rank, int world_size);
    
    // 计算本地批大小
    int calculate_local_batch_size(int global_batch_size, int world_size);
    
    // 检查是否支持数据并行
    bool is_data_parallel_supported();
    
    // 获取数据并行信息
    void print_data_parallel_info(const DataParallelConfig& config);
    
    // 验证数据并行配置
    bool validate_data_parallel_config(const DataParallelConfig& config);
    
    // 创建数据并行模型
    std::shared_ptr<DistributedDataParallel> create_ddp_model(
        std::shared_ptr<Layer> model,
        const DataParallelConfig& config
    );
    
    // 创建数据并行优化器
    std::shared_ptr<DistributedOptimizer> create_distributed_optimizer(
        std::shared_ptr<Optimizer> optimizer,
        const DataParallelConfig& config
    );
    
    // 同步所有进程
    void sync_all_processes();
    
    // 只在主进程执行
    void master_do(const DataParallelConfig& config, std::function<void()> func);
    
    // 收集所有进程的损失
    std::vector<float> gather_all_losses(float local_loss, const DataParallelConfig& config);
    
    // 计算全局平均损失
    float calculate_global_average_loss(float local_loss, const DataParallelConfig& config);
    
    // 检查梯度是否同步
    bool verify_gradient_synchronization(const std::vector<Tensor>& gradients,
                                       const DataParallelConfig& config);
    
    // 检查参数是否同步
    bool verify_parameter_synchronization(const std::vector<Tensor>& parameters,
                                        const DataParallelConfig& config);
}

// 数据并行异常处理
class DataParallelException : public std::runtime_error {
public:
    DataParallelException(const std::string& message, int rank = -1)
        : std::runtime_error(message), rank_(rank) {}
    
    int get_rank() const { return rank_; }
    
private:
    int rank_;
};

// 数据并行检查点
class DataParallelCheckpoint {
public:
    DataParallelCheckpoint(const DataParallelConfig& config);
    
    // 保存检查点
    void save_checkpoint(const std::string& path,
                        const std::vector<Tensor>& parameters,
                        int global_step,
                        float loss);
    
    // 加载检查点
    void load_checkpoint(const std::string& path,
                        std::vector<Tensor>& parameters,
                        int& global_step,
                        float& loss);
    
    // 检查检查点是否存在
    bool checkpoint_exists(const std::string& path) const;
    
    // 删除检查点
    void delete_checkpoint(const std::string& path) const;
    
private:
    DataParallelConfig config_;
    
    // 生成检查点文件名
    std::string get_checkpoint_filename(const std::string& path, int rank) const;
    
    // 验证检查点
    bool validate_checkpoint(const std::string& path) const;
};

} // namespace megatron