#include "communication.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <functional>

namespace megatron {

// MPICommunicator implementation
MPICommunicator& MPICommunicator::instance() {
    static MPICommunicator instance_;
    return instance_;
}

void MPICommunicator::initialize(int* argc, char*** argv) {
    if (initialized_) {
        return;
    }
    
    int initialized = 0;
    MPI_Initialized(&initialized);
    
    if (!initialized) {
        MPI_Init(argc, argv);
    }
    
    MPI_Comm_size(comm_, &world_size_);
    MPI_Comm_rank(comm_, &rank_);
    
    initialized_ = true;
    
    // 只在主进程打印初始化信息
    if (rank_ == 0) {
        std::cout << "MPI initialized with world size: " << world_size_ << std::endl;
    }
}

void MPICommunicator::finalize() {
    if (!initialized_) {
        return;
    }
    
    int finalized = 0;
    MPI_Finalized(&finalized);
    
    if (!finalized) {
        MPI_Finalize();
    }
    
    initialized_ = false;
    world_size_ = 1;
    rank_ = 0;
}

int MPICommunicator::world_size() const {
    check_initialized();
    return world_size_;
}

int MPICommunicator::rank() const {
    check_initialized();
    return rank_;
}

bool MPICommunicator::is_initialized() const {
    return initialized_;
}

void MPICommunicator::send(const Tensor& tensor, int dest_rank, int tag) {
    check_initialized();
    validate_tensor_shape(tensor);
    
    if (dest_rank < 0 || dest_rank >= world_size_) {
        throw std::invalid_argument("Invalid destination rank");
    }
    
    MPI_Datatype dtype = get_mpi_dtype(tensor);
    MPI_Send(tensor.data(), tensor.size(), dtype, dest_rank, tag, comm_);
}

void MPICommunicator::recv(Tensor& tensor, int src_rank, int tag) {
    check_initialized();
    validate_tensor_shape(tensor);
    
    if (src_rank < 0 || src_rank >= world_size_) {
        throw std::invalid_argument("Invalid source rank");
    }
    
    MPI_Datatype dtype = get_mpi_dtype(tensor);
    MPI_Status status;
    MPI_Recv(tensor.data(), tensor.size(), dtype, src_rank, tag, comm_, &status);
}

void MPICommunicator::all_reduce(Tensor& tensor) {
    check_initialized();
    validate_tensor_shape(tensor);
    
    if (world_size_ == 1) {
        return;  // 单进程无需通信
    }
    
    MPI_Datatype dtype = get_mpi_dtype(tensor);
    Tensor buffer(tensor.shape());
    
    // 先执行reduce操作到主进程
    MPI_Reduce(tensor.data(), buffer.data(), tensor.size(), dtype, MPI_SUM, 0, comm_);
    
    // 然后广播结果给所有进程
    MPI_Bcast(buffer.data(), buffer.size(), dtype, 0, comm_);
    
    // 将结果复制回原张量
    for (int i = 0; i < tensor.size(); ++i) {
        tensor[i] = buffer[i] / static_cast<float>(world_size_);
    }
}

void MPICommunicator::all_gather(const std::vector<Tensor>& send_tensors,
                                std::vector<Tensor>& recv_tensors) {
    check_initialized();
    
    if (send_tensors.size() != static_cast<size_t>(world_size_)) {
        throw std::invalid_argument("send_tensors size must match world_size");
    }
    
    recv_tensors.resize(world_size_);
    
    // 简化实现：每个进程发送自己的张量，接收所有张量
    for (int i = 0; i < world_size_; ++i) {
        if (i == rank_) {
            // 发送自己的张量给所有进程
            for (int j = 0; j < world_size_; ++j) {
                if (j != rank_) {
                    send(send_tensors[i], j, 100 + i);
                }
            }
            recv_tensors[i] = send_tensors[i];  // 本进程的数据直接复制
        } else {
            // 从其他进程接收张量
            recv_tensors[i] = Tensor(send_tensors[i].shape());
            recv(recv_tensors[i], i, 100 + i);
        }
    }
}

void MPICommunicator::broadcast(Tensor& tensor, int root_rank) {
    check_initialized();
    validate_tensor_shape(tensor);
    
    if (root_rank < 0 || root_rank >= world_size_) {
        throw std::invalid_argument("Invalid root rank");
    }
    
    MPI_Datatype dtype = get_mpi_dtype(tensor);
    MPI_Bcast(tensor.data(), tensor.size(), dtype, root_rank, comm_);
}

void MPICommunicator::reduce(Tensor& tensor, int root_rank) {
    check_initialized();
    validate_tensor_shape(tensor);
    
    if (root_rank < 0 || root_rank >= world_size_) {
        throw std::invalid_argument("Invalid root rank");
    }
    
    if (world_size_ == 1) {
        return;  // 单进程无需通信
    }
    
    MPI_Datatype dtype = get_mpi_dtype(tensor);
    Tensor buffer(tensor.shape());
    
    MPI_Reduce(tensor.data(), buffer.data(), tensor.size(), dtype, MPI_SUM, root_rank, comm_);
    
    if (rank_ == root_rank) {
        // 在根进程中，将结果复制回原张量
        for (int i = 0; i < tensor.size(); ++i) {
            tensor[i] = buffer[i];
        }
    }
}

void MPICommunicator::barrier() {
    check_initialized();
    MPI_Barrier(comm_);
}

std::string MPICommunicator::get_comm_name() const {
    return "MPI_COMM_WORLD";
}

int MPICommunicator::get_local_device_id() const {
    return rank_;  // 简化实现：每个进程对应一个设备
}

void MPICommunicator::check_initialized() const {
    if (!initialized_) {
        throw std::runtime_error("MPI communicator not initialized");
    }
}

MPI_Datatype MPICommunicator::get_mpi_dtype(const Tensor& tensor) const {
    // 根据张量数据类型返回对应的MPI数据类型
    // 目前只支持float类型
    return MPI_FLOAT;
}

void MPICommunicator::validate_tensor_shape(const Tensor& tensor) const {
    if (tensor.size() == 0) {
        throw std::invalid_argument("Tensor size cannot be zero");
    }
}

// DistributedTrainer implementation
DistributedTrainer::DistributedTrainer(int world_size, int rank)
    : world_size_(world_size), rank_(rank), comm_(MPICommunicator::instance()) {
}

void DistributedTrainer::setup_model_and_optimizer(std::shared_ptr<Layer> model,
                                                   std::shared_ptr<Optimizer> optimizer) {
    model_ = model;
    optimizer_ = optimizer;
}

void DistributedTrainer::distributed_train_step(const Tensor& input_ids, const Tensor& targets) {
    validate_setup();
    
    // 前向传播
    Tensor output = model_->forward(input_ids);
    
    // 计算损失（简化实现）
    float loss = 0.0f;
    for (int i = 0; i < output.size(); ++i) {
        loss += (output[i] - targets[i % targets.size()]) * (output[i] - targets[i % targets.size()]);
    }
    loss /= output.size();
    
    // 创建梯度张量
    Tensor grad_output = output;
    for (int i = 0; i < grad_output.size(); ++i) {
        grad_output[i] = 2.0f * (output[i] - targets[i % targets.size()]) / output.size();
    }
    
    // 反向传播
    Tensor grad_input = model_->backward(grad_output);
    
    // 同步梯度
    auto gradients = model_->gradients();
    synchronize_gradients(gradients);
    
    // 更新参数
    optimizer_->step(model_->parameters(), gradients);
}

void DistributedTrainer::synchronize_gradients(std::vector<Tensor>& gradients) {
    if (world_size_ <= 1) {
        return;
    }
    
    for (auto& grad : gradients) {
        comm_.all_reduce(grad);
    }
}

void DistributedTrainer::broadcast_parameters(std::vector<Tensor>& parameters) {
    if (world_size_ <= 1) {
        return;
    }
    
    for (auto& param : parameters) {
        comm_.broadcast(param, 0);  // 从主进程广播
    }
}

int DistributedTrainer::get_local_batch_size(int global_batch_size) const {
    if (world_size_ <= 1) {
        return global_batch_size;
    }
    
    int local_batch_size = global_batch_size / world_size_;
    if (global_batch_size % world_size_ != 0) {
        local_batch_size += 1;
    }
    
    return local_batch_size;
}

void DistributedTrainer::validate_setup() const {
    if (!model_) {
        throw std::runtime_error("Model not set");
    }
    if (!optimizer_) {
        throw std::runtime_error("Optimizer not set");
    }
}

void DistributedTrainer::all_reduce_gradients(std::vector<Tensor>& gradients) {
    for (auto& grad : gradients) {
        comm_.all_reduce(grad);
    }
}

void DistributedTrainer::sync_model_parameters() {
    auto parameters = model_->parameters();
    broadcast_parameters(parameters);
}

// DistributedDataParallel implementation
DistributedDataParallel::DistributedDataParallel(std::shared_ptr<Layer> model,
                                               const DataParallelConfig& config)
    : model_(model), world_size_(config.world_size), rank_(config.rank),
      trainer_(config.world_size, config.rank) {
    initialize_parallel_context();
}

Tensor DistributedDataParallel::forward(const Tensor& input) {
    input_cache_ = input;
    return model_->forward(input);
}

Tensor DistributedDataParallel::backward(const Tensor& grad_output) {
    Tensor grad_input = model_->backward(grad_output);
    
    // 同步梯度
    sync_gradients();
    
    return grad_input;
}

std::vector<Tensor> DistributedDataParallel::parameters() const {
    return model_->parameters();
}

std::vector<Tensor> DistributedDataParallel::gradients() const {
    return model_->gradients();
}

void DistributedDataParallel::train(bool is_training) {
    if (is_training) {
        model_->train();
    } else {
        model_->eval();
    }
}

bool DistributedDataParallel::is_training() const {
    return model_->is_training();
}

void DistributedDataParallel::sync_parameters() {
    auto parameters = model_->parameters();
    trainer_.broadcast_parameters(parameters);
}

void DistributedDataParallel::sync_gradients() {
    auto gradients = model_->gradients();
    trainer_.synchronize_gradients(gradients);
}

void DistributedDataParallel::initialize_parallel_context() {
    if (world_size_ > 1) {
        sync_parameters();
    }
}

// ParallelState implementation
ParallelState& ParallelState::instance() {
    static ParallelState instance_;
    return instance_;
}

ParallelState::ParallelState()
    : tensor_parallel_size_(1), tensor_parallel_rank_(0),
      pipeline_parallel_size_(1), pipeline_parallel_rank_(0),
      data_parallel_size_(1), data_parallel_rank_(0) {
    
    // 初始化通信器
    tensor_parallel_comm_ = MPI_COMM_WORLD;
    pipeline_parallel_comm_ = MPI_COMM_WORLD;
    data_parallel_comm_ = MPI_COMM_WORLD;
}

void ParallelState::initialize_tensor_parallel(int tensor_parallel_size, int tensor_parallel_rank) {
    tensor_parallel_size_ = tensor_parallel_size;
    tensor_parallel_rank_ = tensor_parallel_rank;
    
    // 创建张量并行的通信器（简化实现）
    tensor_parallel_comm_ = MPI_COMM_WORLD;
}

void ParallelState::initialize_pipeline_parallel(int pipeline_parallel_size, int pipeline_parallel_rank) {
    pipeline_parallel_size_ = pipeline_parallel_size;
    pipeline_parallel_rank_ = pipeline_parallel_rank;
    
    // 创建流水线并行的通信器（简化实现）
    pipeline_parallel_comm_ = MPI_COMM_WORLD;
}

void ParallelState::initialize_data_parallel(int data_parallel_size, int data_parallel_rank) {
    data_parallel_size_ = data_parallel_size;
    data_parallel_rank_ = data_parallel_rank;
    
    // 创建数据并行的通信器（简化实现）
    data_parallel_comm_ = MPI_COMM_WORLD;
}

int ParallelState::get_tensor_parallel_size() const {
    return tensor_parallel_size_;
}

int ParallelState::get_tensor_parallel_rank() const {
    return tensor_parallel_rank_;
}

int ParallelState::get_pipeline_parallel_size() const {
    return pipeline_parallel_size_;
}

int ParallelState::get_pipeline_parallel_rank() const {
    return pipeline_parallel_rank_;
}

int ParallelState::get_data_parallel_size() const {
    return data_parallel_size_;
}

int ParallelState::get_data_parallel_rank() const {
    return data_parallel_rank_;
}

int ParallelState::get_world_size() const {
    return data_parallel_size_ * pipeline_parallel_size_ * tensor_parallel_size_;
}

int ParallelState::get_global_rank() const {
    return data_parallel_rank_ * pipeline_parallel_size_ * tensor_parallel_size_ +
           pipeline_parallel_rank_ * tensor_parallel_size_ +
           tensor_parallel_rank_;
}

bool ParallelState::is_tensor_parallel_enabled() const {
    return tensor_parallel_size_ > 1;
}

bool ParallelState::is_pipeline_parallel_enabled() const {
    return pipeline_parallel_size_ > 1;
}

bool ParallelState::is_data_parallel_enabled() const {
    return data_parallel_size_ > 1;
}

MPI_Comm ParallelState::get_tensor_parallel_comm() const {
    return tensor_parallel_comm_;
}

MPI_Comm ParallelState::get_pipeline_parallel_comm() const {
    return pipeline_parallel_comm_;
}

MPI_Comm ParallelState::get_data_parallel_comm() const {
    return data_parallel_comm_;
}

// DistributedConfig implementation
bool DistributedConfig::validate() const {
    if (world_size <= 0) {
        return false;
    }
    if (rank < 0 || rank >= world_size) {
        return false;
    }
    if (tensor_parallel_size <= 0 || pipeline_parallel_size <= 0 || data_parallel_size <= 0) {
        return false;
    }
    if (tensor_parallel_size * pipeline_parallel_size * data_parallel_size != world_size) {
        return false;
    }
    if (global_batch_size <= 0 || micro_batch_size <= 0) {
        return false;
    }
    if (global_batch_size % micro_batch_size != 0) {
        return false;
    }
    return true;
}

int DistributedConfig::get_local_batch_size() const {
    return global_batch_size / data_parallel_size;
}

int DistributedConfig::get_gradient_accumulation_steps() const {
    return global_batch_size / (data_parallel_size * micro_batch_size);
}

// distributed_utils implementation
namespace distributed_utils {

void initialize_distributed(const DistributedConfig& config) {
    if (!config.validate()) {
        throw std::invalid_argument("Invalid distributed configuration");
    }
    
    // 初始化MPI
    int argc = 0;
    char** argv = nullptr;
    MPICommunicator::instance().initialize(&argc, &argv);
    
    // 设置并行状态
    auto& parallel_state = ParallelState::instance();
    parallel_state.initialize_tensor_parallel(config.tensor_parallel_size, 
                                              config.rank % config.tensor_parallel_size);
    parallel_state.initialize_pipeline_parallel(config.pipeline_parallel_size,
                                               (config.rank / config.tensor_parallel_size) % config.pipeline_parallel_size);
    parallel_state.initialize_data_parallel(config.data_parallel_size,
                                           config.rank / (config.tensor_parallel_size * config.pipeline_parallel_size));
    
    // 打印分布式信息
    if (config.rank == 0) {
        print_distributed_info();
    }
}

void cleanup_distributed() {
    MPICommunicator::instance().finalize();
}

int get_current_device() {
    return MPICommunicator::instance().rank();
}

void set_device(int device_id) {
    // 简化实现：在CPU版本中，设备ID对应进程rank
    // 在GPU版本中，这里会设置CUDA设备
}

void sync_all_processes() {
    MPICommunicator::instance().barrier();
}

void master_do(std::function<void()> func) {
    if (MPICommunicator::instance().rank() == 0) {
        func();
    }
}

void print_distributed_info() {
    auto& comm = MPICommunicator::instance();
    auto& parallel_state = ParallelState::instance();
    
    std::cout << "=== Distributed Training Configuration ===" << std::endl;
    std::cout << "World Size: " << comm.world_size() << std::endl;
    std::cout << "Tensor Parallel: " << parallel_state.get_tensor_parallel_size() << std::endl;
    std::cout << "Pipeline Parallel: " << parallel_state.get_pipeline_parallel_size() << std::endl;
    std::cout << "Data Parallel: " << parallel_state.get_data_parallel_size() << std::endl;
    std::cout << "=========================================" << std::endl;
}

bool is_distributed_supported() {
    // 检查是否支持MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    return initialized || std::getenv("OMPI_COMM_WORLD_SIZE") != nullptr;
}

MPI_Comm get_local_comm() {
    // 简化实现：返回全局通信器
    return MPI_COMM_WORLD;
}

MPI_Comm get_cross_comm() {
    // 简化实现：返回全局通信器
    return MPI_COMM_WORLD;
}

} // namespace distributed_utils

} // namespace megatron