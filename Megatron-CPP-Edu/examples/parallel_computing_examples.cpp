#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <mpi.h>

#include "core/tensor/tensor.h"
#include "core/layers/linear.h"
#include "core/layers/transformer_block.h"
#include "core/optimizers/sgd.h"
#include "core/data/dataset.h"
#include "core/parallel/tensor_parallel.h"
#include "core/parallel/data_parallel.h"
#include "distributed/communication.h"

using namespace megatron;

// 示例1：基础数据并行训练
void example_basic_data_parallel() {
    std::cout << "\n=== 示例1：基础数据并行训练 ===" << std::endl;
    
    // 初始化分布式环境
    DistributedConfig config;
    config.world_size = 4;
    config.rank = MPICommunicator::instance().rank();
    config.global_batch_size = 32;
    config.local_batch_size = 8;
    
    distributed_utils::initialize_distributed(config);
    
    // 创建简单模型
    auto model = std::make_shared<Linear>(784, 10, "data_parallel_model");
    auto optimizer = std::make_shared<SGD>(0.01);
    
    // 创建数据并行训练器
    DataParallelTrainer trainer(config);
    trainer.setup_model_and_optimizer(model, optimizer);
    
    // 模拟训练数据
    Tensor inputs({config.local_batch_size, 784});
    Tensor targets({config.local_batch_size});
    
    // 填充随机数据
    for (int i = 0; i < inputs.size(); ++i) {
        inputs[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < targets.size(); ++i) {
        targets[i] = rand() % 10;
    }
    
    // 执行训练步骤
    for (int step = 0; step < 10; ++step) {
        trainer.train_step(inputs, targets);
        
        if (config.rank == 0 && step % 5 == 0) {
            std::cout << "Step " << step << " completed" << std::endl;
        }
    }
    
    std::cout << "数据并行训练示例完成" << std::endl;
}

// 示例2：张量并行Transformer模型
void example_tensor_parallel_transformer() {
    std::cout << "\n=== 示例2：张量并行Transformer模型 ===" << std::endl;
    
    // 初始化张量并行环境
    int tensor_parallel_size = 2;
    int tensor_parallel_rank = MPICommunicator::instance().rank() % tensor_parallel_size;
    
    tensor_parallel_utils::initialize_tensor_parallel(tensor_parallel_size, tensor_parallel_rank);
    
    // 创建张量并行模型构建器
    TensorParallelModelBuilder builder(tensor_parallel_size);
    builder.set_embed_dim(768);
    builder.set_num_heads(12);
    builder.set_ffn_dim(3072);
    builder.set_num_layers(6);
    
    // 构建张量并行模型
    auto model = builder.build_gpt_model();
    
    // 创建输入数据
    int batch_size = 4;
    int seq_len = 128;
    Tensor input({batch_size, seq_len});
    
    // 填充随机token IDs
    for (int i = 0; i < input.size(); ++i) {
        input[i] = rand() % 1000;  // vocab_size = 1000
    }
    
    // 前向传播
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = model->forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (tensor_parallel_rank == 0) {
        std::cout << "张量并行前向传播时间: " << duration.count() << " ms" << std::endl;
        std::cout << "输出形状: [" << output.shape()[0] << ", " << output.shape()[1] 
                  << ", " << output.shape()[2] << "]" << std::endl;
    }
    
    std::cout << "张量并行Transformer示例完成" << std::endl;
}

// 示例3：混合并行训练
void example_hybrid_parallel_training() {
    std::cout << "\n=== 示例3：混合并行训练 ===" << std::endl;
    
    // 配置混合并行
    DataParallelConfig dp_config;
    dp_config.world_size = 4;
    dp_config.rank = MPICommunicator::instance().rank();
    dp_config.global_batch_size = 64;
    dp_config.local_batch_size = 16;
    
    int tensor_parallel_size = 2;
    int pipeline_parallel_size = 1;
    
    // 创建混合并行训练器
    HybridParallelTrainer trainer(dp_config, tensor_parallel_size, pipeline_parallel_size);
    
    // 创建张量并行模型
    TensorParallelModelBuilder builder(tensor_parallel_size);
    builder.set_embed_dim(512);
    builder.set_num_heads(8);
    builder.set_ffn_dim(2048);
    builder.set_num_layers(4);
    
    auto model = builder.build_gpt_model();
    auto optimizer = std::make_shared<SGD>(0.001);
    
    trainer.setup_model_and_optimizer(model, optimizer);
    
    // 创建训练数据
    Tensor inputs({dp_config.local_batch_size, 128});
    Tensor targets({dp_config.local_batch_size, 128});
    
    // 填充随机数据
    for (int i = 0; i < inputs.size(); ++i) {
        inputs[i] = rand() % 1000;
        targets[i] = rand() % 1000;
    }
    
    // 执行混合并行训练步骤
    for (int step = 0; step < 5; ++step) {
        auto start = std::chrono::high_resolution_clock::now();
        trainer.hybrid_train_step(inputs, targets);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (dp_config.rank == 0) {
            std::cout << "混合并行步骤 " << step << " 时间: " << duration.count() << " ms" << std::endl;
        }
    }
    
    std::cout << "混合并行训练示例完成" << std::endl;
}

// 示例4：分布式数据加载
void example_distributed_data_loading() {
    std::cout << "\n=== 示例4：分布式数据加载 ===" << std::endl;
    
    // 创建模拟数据集
    int total_samples = 1000;
    int feature_dim = 784;
    int num_classes = 10;
    
    std::vector<std::vector<float>> data(total_samples, std::vector<float>(feature_dim));
    std::vector<int> labels(total_samples);
    
    // 生成随机数据
    for (int i = 0; i < total_samples; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            data[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
        labels[i] = rand() % num_classes;
    }
    
    // 创建数据集
    auto dataset = std::make_shared<SimpleDataset>(data, labels);
    
    // 配置数据并行
    DataParallelConfig config;
    config.world_size = 4;
    config.rank = MPICommunicator::instance().rank();
    config.global_batch_size = 32;
    config.local_batch_size = 8;
    
    // 创建分布式数据加载器
    DataParallelDataLoader dataloader(dataset, config, true);
    
    // 模拟数据加载和训练
    int num_epochs = 2;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        dataloader.reset();
        int batch_count = 0;
        
        while (dataloader.has_next_batch()) {
            auto batch = dataloader.next_batch();
            batch_count++;
            
            if (config.rank == 0 && batch_count % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Batch " << batch_count << std::endl;
            }
        }
        
        // 同步所有进程
        distributed_utils::sync_all_processes();
        
        if (config.rank == 0) {
            std::cout << "Epoch " << epoch << " 完成，处理了 " << batch_count << " 个批次" << std::endl;
        }
    }
    
    std::cout << "分布式数据加载示例完成" << std::endl;
}

// 示例5：通信性能测试
void example_communication_performance() {
    std::cout << "\n=== 示例5：通信性能测试 ===" << std::endl;
    
    auto& comm = MPICommunicator::instance();
    int world_size = comm.world_size();
    int rank = comm.rank();
    
    // 测试不同大小的张量通信性能
    std::vector<int> tensor_sizes = {1000, 10000, 100000, 1000000};
    
    for (int size : tensor_sizes) {
        // 创建测试张量
        Tensor tensor({size});
        for (int i = 0; i < size; ++i) {
            tensor[i] = static_cast<float>(i);
        }
        
        // 测试all_reduce性能
        auto start = std::chrono::high_resolution_clock::now();
        comm.all_reduce(tensor);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 收集所有进程的时间
        std::vector<float> all_times(world_size);
        float local_time = duration.count();
        
        MPI_Gather(&local_time, 1, MPI_FLOAT, all_times.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            float avg_time = 0;
            float max_time = 0;
            float min_time = all_times[0];
            
            for (float t : all_times) {
                avg_time += t;
                max_time = std::max(max_time, t);
                min_time = std::min(min_time, t);
            }
            avg_time /= world_size;
            
            std::cout << "张量大小: " << size << std::endl;
            std::cout << "  平均时间: " << avg_time << " μs" << std::endl;
            std::cout << "  最大时间: " << max_time << " μs" << std::endl;
            std::cout << "  最小时间: " << min_time << " μs" << std::endl;
            std::cout << "  带宽: " << (size * 4) / (avg_time / 1e6) / 1e6 << " MB/s" << std::endl;
        }
        
        comm.barrier();
    }
    
    std::cout << "通信性能测试完成" << std::endl;
}

// 示例6：梯度同步验证
void example_gradient_synchronization() {
    std::cout << "\n=== 示例6：梯度同步验证 ===" << std::endl;
    
    auto& comm = MPICommunicator::instance();
    int world_size = comm.world_size();
    int rank = comm.rank();
    
    // 创建简单模型
    auto model = std::make_shared<Linear>(100, 10, "grad_sync_model");
    
    // 获取参数和梯度
    auto parameters = model->parameters();
    auto gradients = model->gradients();
    
    // 为每个进程设置不同的随机梯度
    for (auto& grad : gradients) {
        for (int i = 0; i < grad.size(); ++i) {
            grad[i] = rank + static_cast<float>(i) / grad.size();
        }
    }
    
    // 打印同步前的梯度（只打印第一个参数）
    if (rank == 0) {
        std::cout << "同步前梯度（前5个值）:" << std::endl;
    }
    
    for (int r = 0; r < world_size; ++r) {
        if (rank == r) {
            std::cout << "进程 " << rank << ": ";
            for (int i = 0; i < std::min(5, static_cast<int>(gradients[0].size())); ++i) {
                std::cout << gradients[0][i] << " ";
            }
            std::cout << std::endl;
        }
        comm.barrier();
    }
    
    // 执行梯度同步
    for (auto& grad : gradients) {
        comm.all_reduce(grad);
    }
    
    // 打印同步后的梯度
    if (rank == 0) {
        std::cout << "\n同步后梯度（前5个值）:" << std::endl;
    }
    
    for (int r = 0; r < world_size; ++r) {
        if (rank == r) {
            std::cout << "进程 " << rank << ": ";
            for (int i = 0; i < std::min(5, static_cast<int>(gradients[0].size())); ++i) {
                std::cout << gradients[0][i] << " ";
            }
            std::cout << std::endl;
        }
        comm.barrier();
    }
    
    // 验证同步是否正确
    bool sync_correct = true;
    float expected_sum = 0;
    for (int r = 0; r < world_size; ++r) {
        expected_sum += r + static_cast<float>(0) / gradients[0].size();
    }
    expected_sum /= world_size;
    
    if (std::abs(gradients[0][0] - expected_sum) > 1e-6) {
        sync_correct = false;
    }
    
    if (rank == 0) {
        std::cout << "\n梯度同步验证: " << (sync_correct ? "成功" : "失败") << std::endl;
    }
    
    std::cout << "梯度同步验证示例完成" << std::endl;
}

int main(int argc, char** argv) {
    // 初始化MPI
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (rank == 0) {
        std::cout << "Megatron-CPP-Edu 并行计算示例程序" << std::endl;
        std::cout << "进程数: " << world_size << std::endl;
    }
    
    try {
        // 运行所有示例
        example_basic_data_parallel();
        example_tensor_parallel_transformer();
        example_hybrid_parallel_training();
        example_distributed_data_loading();
        example_communication_performance();
        example_gradient_synchronization();
        
        if (rank == 0) {
            std::cout << "\n所有示例运行完成！" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "进程 " << rank << " 发生错误: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // 清理MPI
    MPI_Finalize();
    
    return 0;
}