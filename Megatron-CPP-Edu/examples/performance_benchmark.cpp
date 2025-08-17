#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mpi.h>

#include "core/tensor/tensor.h"
#include "core/layers/linear.h"
#include "core/layers/transformer_block.h"
#include "core/optimizers/sgd.h"
#include "core/optimizers/adam.h"
#include "core/data/dataset.h"
#include "core/parallel/tensor_parallel.h"
#include "core/parallel/data_parallel.h"
#include "distributed/communication.h"

using namespace megatron;

// 性能基准测试类
class PerformanceBenchmark {
public:
    PerformanceBenchmark(const std::string& output_dir = "./benchmark_results")
        : output_dir_(output_dir) {
        
        // 创建输出目录
        std::string cmd = "mkdir -p " + output_dir;
        system(cmd.c_str());
        
        // 初始化MPI
        int rank = MPICommunicator::instance().rank();
        if (rank == 0) {
            std::cout << "=== Megatron-CPP-Edu Performance Benchmark ===" << std::endl;
        }
    }
    
    ~PerformanceBenchmark() {
        if (MPICommunicator::instance().rank() == 0) {
            std::cout << "Benchmark completed. Results saved to: " << output_dir_ << std::endl;
        }
    }
    
    // 运行所有基准测试
    void run_all_benchmarks() {
        benchmark_tensor_operations();
        benchmark_layer_performance();
        benchmark_communication_performance();
        benchmark_data_parallel_training();
        benchmark_tensor_parallel_training();
        benchmark_hybrid_parallel_training();
        benchmark_scalability_analysis();
        
        generate_performance_report();
    }
    
private:
    std::string output_dir_;
    std::map<std::string, std::vector<double>> timing_results_;
    std::map<std::string, std::map<std::string, double>> detailed_results_;
    
    // 张量操作性能测试
    void benchmark_tensor_operations() {
        std::cout << "\n=== Tensor Operations Benchmark ===" << std::endl;
        
        // 测试不同大小的张量操作
        std::vector<std::vector<int>> tensor_shapes = {
            {1000}, {10000}, {100000}, {1000000}, {10000000}
        };
        
        for (const auto& shape : tensor_shapes) {
            benchmark_tensor_creation(shape);
            benchmark_tensor_arithmetic(shape);
            benchmark_tensor_matrix_ops(shape);
        }
    }
    
    void benchmark_tensor_creation(const std::vector<int>& shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        std::string test_name = "tensor_creation_size_" + std::to_string(size);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 创建张量
        for (int i = 0; i < 100; ++i) {
            Tensor tensor(shape);
            tensor.fill(1.0f);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 100.0);  // 平均时间
        
        if (MPICommunicator::instance().rank() == 0) {
            std::cout << "Tensor creation (size=" << size << "): " 
                      << duration.count() / 100.0 << " μs" << std::endl;
        }
    }
    
    void benchmark_tensor_arithmetic(const std::vector<int>& shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        std::string test_name = "tensor_arithmetic_size_" + std::to_string(size);
        
        Tensor a(shape);
        Tensor b(shape);
        a.fill(1.0f);
        b.fill(2.0f);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 算术操作
        for (int i = 0; i < 100; ++i) {
            Tensor c = a + b;
            Tensor d = a * b;
            Tensor e = a / b;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 300.0);  // 3个操作
        
        if (MPICommunicator::instance().rank() == 0) {
            std::cout << "Tensor arithmetic (size=" << size << "): " 
                      << duration.count() / 300.0 << " μs per operation" << std::endl;
        }
    }
    
    void benchmark_tensor_matrix_ops(const std::vector<int>& shape) {
        if (shape.size() < 2) return;
        
        int size = 1;
        for (int dim : shape) size *= dim;
        std::string test_name = "tensor_matrix_ops_size_" + std::to_string(size);
        
        int m = shape[0];
        int n = shape[1];
        int k = shape.size() > 2 ? shape[2] : n;
        
        Tensor a({m, k});
        Tensor b({k, n});
        a.fill(1.0f);
        b.fill(2.0f);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 矩阵乘法
        for (int i = 0; i < 10; ++i) {
            Tensor c = a.matmul(b);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 10.0);
        
        if (MPICommunicator::instance().rank() == 0) {
            double gflops = (2.0 * m * n * k) / (duration.count() / 10.0 / 1e9);
            std::cout << "Matrix multiplication (size=" << m << "x" << k << "x" << n << "): " 
                      << duration.count() / 10.0 << " μs, " << gflops << " GFLOPS" << std::endl;
        }
    }
    
    // 层性能测试
    void benchmark_layer_performance() {
        std::cout << "\n=== Layer Performance Benchmark ===" << std::endl;
        
        // 测试不同配置的层
        std::vector<std::pair<int, int>> layer_configs = {
            {768, 3072},   // FFN层
            {1024, 4096},  // 大FFN层
            {4096, 16384}, // 超大FFN层
            {1000, 50000}, // 嵌入层
            {768, 768}     // 注意力层
        };
        
        for (const auto& config : layer_configs) {
            benchmark_linear_layer(config.first, config.second);
            benchmark_embedding_layer(config.first, config.second);
        }
        
        benchmark_transformer_block();
    }
    
    void benchmark_linear_layer(int in_features, int out_features) {
        std::string test_name = "linear_layer_" + std::to_string(in_features) + "_" + std::to_string(out_features);
        
        Linear layer(in_features, out_features);
        Tensor input({32, in_features});
        input.fill(1.0f);
        
        // 预热
        for (int i = 0; i < 10; ++i) {
            layer.forward(input);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 前向传播
        for (int i = 0; i < 100; ++i) {
            Tensor output = layer.forward(input);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 100.0);
        
        if (MPICommunicator::instance().rank() == 0) {
            double gflops = (2.0 * 32 * in_features * out_features) / (duration.count() / 100.0 / 1e9);
            std::cout << "Linear layer (" << in_features << "->" << out_features << "): " 
                      << duration.count() / 100.0 << " μs, " << gflops << " GFLOPS" << std::endl;
        }
    }
    
    void benchmark_embedding_layer(int vocab_size, int embedding_dim) {
        std::string test_name = "embedding_layer_" + std::to_string(vocab_size) + "_" + std::to_string(embedding_dim);
        
        Embedding layer(vocab_size, embedding_dim);
        Tensor input({32, 128});
        input.fill(1.0f);
        
        // 预热
        for (int i = 0; i < 10; ++i) {
            layer.forward(input);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 前向传播
        for (int i = 0; i < 100; ++i) {
            Tensor output = layer.forward(input);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 100.0);
        
        if (MPICommunicator::instance().rank() == 0) {
            std::cout << "Embedding layer (" << vocab_size << "x" << embedding_dim << "): " 
                      << duration.count() / 100.0 << " μs" << std::endl;
        }
    }
    
    void benchmark_transformer_block() {
        std::cout << "\n--- Transformer Block Benchmark ---" << std::endl;
        
        std::vector<int> embed_dims = {512, 768, 1024, 1536};
        
        for (int embed_dim : embed_dims) {
            std::string test_name = "transformer_block_" + std::to_string(embed_dim);
            
            int num_heads = embed_dim / 64;
            int ffn_dim = embed_dim * 4;
            
            // 创建Transformer块
            auto attn_norm = std::make_shared<LayerNorm>(embed_dim);
            auto attention = std::make_shared<MultiHeadAttention>(embed_dim, num_heads);
            auto attn_dropout = std::make_shared<Dropout>(0.1);
            
            auto ffn_norm = std::make_shared<LayerNorm>(embed_dim);
            auto linear1 = std::make_shared<Linear>(embed_dim, ffn_dim);
            auto linear2 = std::make_shared<Linear>(ffn_dim, embed_dim);
            auto ffn_dropout = std::make_shared<Dropout>(0.1);
            
            Tensor input({4, 128, embed_dim});
            input.fill(1.0f);
            
            // 预热
            for (int i = 0; i < 5; ++i) {
                // 简化的前向传播
                Tensor x = input;
                x = attn_norm->forward(x);
                x = attention->forward(x);
                x = attn_dropout->forward(x);
                x = ffn_norm->forward(x);
                x = linear1->forward(x);
                x = linear2->forward(x);
                x = ffn_dropout->forward(x);
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // 前向传播
            for (int i = 0; i < 20; ++i) {
                Tensor x = input;
                x = attn_norm->forward(x);
                x = attention->forward(x);
                x = attn_dropout->forward(x);
                x = ffn_norm->forward(x);
                x = linear1->forward(x);
                x = linear2->forward(x);
                x = ffn_dropout->forward(x);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            timing_results_[test_name].push_back(duration.count() / 20.0);
            
            if (MPICommunicator::instance().rank() == 0) {
                std::cout << "Transformer block (embed_dim=" << embed_dim << "): " 
                          << duration.count() / 20.0 << " μs" << std::endl;
            }
        }
    }
    
    // 通信性能测试
    void benchmark_communication_performance() {
        std::cout << "\n=== Communication Performance Benchmark ===" << std::endl;
        
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        if (world_size == 1) {
            std::cout << "Skipping communication benchmark (single process)" << std::endl;
            return;
        }
        
        // 测试不同大小的通信
        std::vector<int> message_sizes = {1024, 10240, 102400, 1024000, 10485760};
        
        for (int size : message_sizes) {
            benchmark_point_to_point(size);
            benchmark_collective_communication(size);
            benchmark_all_reduce(size);
        }
    }
    
    void benchmark_point_to_point(int message_size) {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::string test_name = "point_to_point_size_" + std::to_string(message_size);
        
        Tensor send_tensor({message_size});
        Tensor recv_tensor({message_size});
        send_tensor.fill(static_cast<float>(rank));
        
        // 同步所有进程
        comm.barrier();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 点对点通信测试
        for (int i = 0; i < 10; ++i) {
            if (rank == 0) {
                comm.send(send_tensor, 1, i);
                comm.recv(recv_tensor, 1, i);
            } else if (rank == 1) {
                comm.recv(recv_tensor, 0, i);
                comm.send(send_tensor, 0, i);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 20.0);  // 20次通信
        
        if (rank == 0) {
            double bandwidth = (message_size * 4 * 20) / (duration.count() / 1e6) / 1e6;  // MB/s
            std::cout << "Point-to-point (size=" << message_size << "): " 
                      << duration.count() / 20.0 << " μs, " << bandwidth << " MB/s" << std::endl;
        }
    }
    
    void benchmark_collective_communication(int message_size) {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::string test_name = "collective_comm_size_" + std::to_string(message_size);
        
        Tensor tensor({message_size});
        tensor.fill(static_cast<float>(rank));
        
        // 同步所有进程
        comm.barrier();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 集合通信测试
        for (int i = 0; i < 10; ++i) {
            if (rank == 0) {
                comm.broadcast(tensor, 0);
            } else {
                comm.broadcast(tensor, 0);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 10.0);
        
        if (rank == 0) {
            double bandwidth = (message_size * 4 * 10) / (duration.count() / 1e6) / 1e6;  // MB/s
            std::cout << "Broadcast (size=" << message_size << "): " 
                      << duration.count() / 10.0 << " μs, " << bandwidth << " MB/s" << std::endl;
        }
    }
    
    void benchmark_all_reduce(int message_size) {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::string test_name = "all_reduce_size_" + std::to_string(message_size);
        
        Tensor tensor({message_size});
        tensor.fill(static_cast<float>(rank));
        
        // 同步所有进程
        comm.barrier();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // All-Reduce测试
        for (int i = 0; i < 10; ++i) {
            comm.all_reduce(tensor);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 10.0);
        
        if (rank == 0) {
            double bandwidth = (message_size * 4 * 2 * (world_size - 1) * 10) / (duration.count() / 1e6) / 1e6;  // MB/s
            std::cout << "All-Reduce (size=" << message_size << "): " 
                      << duration.count() / 10.0 << " μs, " << bandwidth << " MB/s" << std::endl;
        }
    }
    
    // 数据并行训练性能测试
    void benchmark_data_parallel_training() {
        std::cout << "\n=== Data Parallel Training Benchmark ===" << std::endl;
        
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        // 测试不同模型大小
        std::vector<int> model_sizes = {1000, 10000, 100000, 1000000};
        
        for (int model_size : model_sizes) {
            benchmark_data_parallel_model(model_size);
        }
    }
    
    void benchmark_data_parallel_model(int model_size) {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::string test_name = "data_parallel_model_" + std::to_string(model_size);
        
        // 创建模型
        auto model = std::make_shared<Linear>(model_size, 10);
        auto optimizer = std::make_shared<Adam>(1e-3);
        
        // 创建数据并行训练器
        DataParallelConfig config;
        config.world_size = world_size;
        config.rank = rank;
        config.global_batch_size = 32;
        config.local_batch_size = 32 / world_size;
        
        DataParallelTrainer trainer(config);
        trainer.setup_model_and_optimizer(model, optimizer);
        
        // 创建训练数据
        Tensor inputs({config.local_batch_size, model_size});
        Tensor targets({config.local_batch_size});
        inputs.fill(1.0f);
        targets.fill(1.0f);
        
        // 预热
        for (int i = 0; i < 5; ++i) {
            trainer.train_step(inputs, targets);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 训练步骤
        for (int i = 0; i < 20; ++i) {
            trainer.train_step(inputs, targets);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 20.0);
        
        if (rank == 0) {
            std::cout << "Data parallel training (model_size=" << model_size << "): " 
                      << duration.count() / 20.0 << " ms per step" << std::endl;
        }
    }
    
    // 张量并行训练性能测试
    void benchmark_tensor_parallel_training() {
        std::cout << "\n=== Tensor Parallel Training Benchmark ===" << std::endl;
        
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        // 测试不同的张量并行大小
        std::vector<int> tp_sizes = {1, 2, 4, 8};
        
        for (int tp_size : tp_sizes) {
            if (world_size >= tp_size) {
                benchmark_tensor_parallel_model(tp_size);
            }
        }
    }
    
    void benchmark_tensor_parallel_model(int tensor_parallel_size) {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::string test_name = "tensor_parallel_size_" + std::to_string(tensor_parallel_size);
        
        // 初始化张量并行环境
        int tp_rank = rank % tensor_parallel_size;
        tensor_parallel_utils::initialize_tensor_parallel(tensor_parallel_size, tp_rank);
        
        // 创建张量并行模型
        TensorParallelModelBuilder builder(tensor_parallel_size);
        builder.set_embed_dim(768);
        builder.set_num_heads(12);
        builder.set_ffn_dim(3072);
        builder.set_num_layers(6);
        
        auto model = builder.build_gpt_model();
        
        // 创建训练数据
        int batch_size = 4;
        int seq_len = 128;
        Tensor inputs({batch_size, seq_len});
        Tensor targets({batch_size, seq_len});
        inputs.fill(1.0f);
        targets.fill(1.0f);
        
        // 预热
        for (int i = 0; i < 5; ++i) {
            Tensor output = model->forward(inputs);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 前向传播
        for (int i = 0; i < 20; ++i) {
            Tensor output = model->forward(inputs);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 20.0);
        
        if (tp_rank == 0) {
            std::cout << "Tensor parallel training (tp_size=" << tensor_parallel_size << "): " 
                      << duration.count() / 20.0 << " ms per forward pass" << std::endl;
        }
    }
    
    // 混合并行训练性能测试
    void benchmark_hybrid_parallel_training() {
        std::cout << "\n=== Hybrid Parallel Training Benchmark ===" << std::endl;
        
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        // 测试不同的混合并行配置
        std::vector<std::tuple<int, int, int>> hybrid_configs = {
            {1, 1, world_size},     // 纯数据并行
            {2, 1, world_size/2},   // 2-way TP + DP
            {4, 1, world_size/4},   // 4-way TP + DP
            {2, 2, world_size/4},   // 2-way TP + 2-way PP + DP
        };
        
        for (const auto& config : hybrid_configs) {
            int tp_size = std::get<0>(config);
            int pp_size = std::get<1>(config);
            int dp_size = std::get<2>(config);
            
            if (tp_size * pp_size * dp_size <= world_size) {
                benchmark_hybrid_parallel_model(tp_size, pp_size, dp_size);
            }
        }
    }
    
    void benchmark_hybrid_parallel_model(int tensor_parallel_size, int pipeline_parallel_size, int data_parallel_size) {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::string test_name = "hybrid_parallel_tp" + std::to_string(tensor_parallel_size) + 
                               "_pp" + std::to_string(pipeline_parallel_size) + 
                               "_dp" + std::to_string(data_parallel_size);
        
        // 创建混合并行训练器
        DataParallelConfig dp_config;
        dp_config.world_size = data_parallel_size;
        dp_config.rank = rank / (tensor_parallel_size * pipeline_parallel_size);
        dp_config.global_batch_size = 32;
        dp_config.local_batch_size = 32 / data_parallel_size;
        
        HybridParallelTrainer trainer(dp_config, tensor_parallel_size, pipeline_parallel_size);
        
        // 创建张量并行模型
        TensorParallelModelBuilder builder(tensor_parallel_size);
        builder.set_embed_dim(512);
        builder.set_num_heads(8);
        builder.set_ffn_dim(2048);
        builder.set_num_layers(4);
        
        auto model = builder.build_gpt_model();
        auto optimizer = std::make_shared<Adam>(1e-4);
        
        trainer.setup_model_and_optimizer(model, optimizer);
        
        // 创建训练数据
        int batch_size = dp_config.local_batch_size;
        int seq_len = 128;
        Tensor inputs({batch_size, seq_len});
        Tensor targets({batch_size, seq_len});
        inputs.fill(1.0f);
        targets.fill(1.0f);
        
        // 预热
        for (int i = 0; i < 5; ++i) {
            trainer.hybrid_train_step(inputs, targets);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 训练步骤
        for (int i = 0; i < 10; ++i) {
            trainer.hybrid_train_step(inputs, targets);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        timing_results_[test_name].push_back(duration.count() / 10.0);
        
        if (dp_config.rank == 0) {
            std::cout << "Hybrid parallel (TP=" << tensor_parallel_size 
                      << ", PP=" << pipeline_parallel_size 
                      << ", DP=" << data_parallel_size << "): " 
                      << duration.count() / 10.0 << " ms per step" << std::endl;
        }
    }
    
    // 可扩展性分析
    void benchmark_scalability_analysis() {
        std::cout << "\n=== Scalability Analysis ===" << std::endl;
        
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        // 强扩展性测试
        benchmark_strong_scaling();
        
        // 弱扩展性测试
        benchmark_weak_scaling();
        
        // 通信开销分析
        benchmark_communication_overhead();
    }
    
    void benchmark_strong_scaling() {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::cout << "\n--- Strong Scaling Analysis ---" << std::endl;
        
        // 固定问题规模，增加处理器数量
        int fixed_model_size = 100000;
        int fixed_batch_size = 32;
        
        for (int np = 1; np <= world_size; np *= 2) {
            if (world_size % np == 0) {
                benchmark_strong_scaling_config(fixed_model_size, fixed_batch_size, np);
            }
        }
    }
    
    void benchmark_strong_scaling_config(int model_size, int batch_size, int num_processes) {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::string test_name = "strong_scaling_np" + std::to_string(num_processes);
        
        if (rank < num_processes) {
            // 创建模型
            auto model = std::make_shared<Linear>(model_size, 10);
            auto optimizer = std::make_shared<Adam>(1e-3);
            
            // 创建数据并行训练器
            DataParallelConfig config;
            config.world_size = num_processes;
            config.rank = rank;
            config.global_batch_size = batch_size;
            config.local_batch_size = batch_size / num_processes;
            
            DataParallelTrainer trainer(config);
            trainer.setup_model_and_optimizer(model, optimizer);
            
            // 创建训练数据
            Tensor inputs({config.local_batch_size, model_size});
            Tensor targets({config.local_batch_size});
            inputs.fill(1.0f);
            targets.fill(1.0f);
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // 训练步骤
            for (int i = 0; i < 20; ++i) {
                trainer.train_step(inputs, targets);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            timing_results_[test_name].push_back(duration.count() / 20.0);
            
            if (rank == 0) {
                std::cout << "Strong scaling (np=" << num_processes << "): " 
                          << duration.count() / 20.0 << " ms per step" << std::endl;
            }
        }
        
        comm.barrier();
    }
    
    void benchmark_weak_scaling() {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::cout << "\n--- Weak Scaling Analysis ---" << std::endl;
        
        // 每个处理器处理固定规模的问题
        int base_model_size = 10000;
        int base_batch_size = 8;
        
        for (int np = 1; np <= world_size; np *= 2) {
            if (world_size % np == 0) {
                benchmark_weak_scaling_config(base_model_size * np, base_batch_size * np, np);
            }
        }
    }
    
    void benchmark_weak_scaling_config(int model_size, int batch_size, int num_processes) {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::string test_name = "weak_scaling_np" + std::to_string(num_processes);
        
        if (rank < num_processes) {
            // 创建模型
            auto model = std::make_shared<Linear>(model_size, 10);
            auto optimizer = std::make_shared<Adam>(1e-3);
            
            // 创建数据并行训练器
            DataParallelConfig config;
            config.world_size = num_processes;
            config.rank = rank;
            config.global_batch_size = batch_size;
            config.local_batch_size = batch_size / num_processes;
            
            DataParallelTrainer trainer(config);
            trainer.setup_model_and_optimizer(model, optimizer);
            
            // 创建训练数据
            Tensor inputs({config.local_batch_size, model_size});
            Tensor targets({config.local_batch_size});
            inputs.fill(1.0f);
            targets.fill(1.0f);
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // 训练步骤
            for (int i = 0; i < 20; ++i) {
                trainer.train_step(inputs, targets);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            timing_results_[test_name].push_back(duration.count() / 20.0);
            
            if (rank == 0) {
                std::cout << "Weak scaling (np=" << num_processes << ", model_size=" << model_size << "): " 
                          << duration.count() / 20.0 << " ms per step" << std::endl;
            }
        }
        
        comm.barrier();
    }
    
    void benchmark_communication_overhead() {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::cout << "\n--- Communication Overhead Analysis ---" << std::endl;
        
        // 测试不同通信开销占比
        std::vector<int> model_sizes = {1000, 10000, 100000, 1000000};
        
        for (int model_size : model_sizes) {
            benchmark_communication_overhead_config(model_size);
        }
    }
    
    void benchmark_communication_overhead_config(int model_size) {
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        std::string test_name = "comm_overhead_model_" + std::to_string(model_size);
        
        // 测试纯计算时间（无通信）
        auto model = std::make_shared<Linear>(model_size, 10);
        Tensor inputs({8, model_size});
        inputs.fill(1.0f);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            Tensor output = model->forward(inputs);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double compute_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100.0;
        
        // 测试数据并行训练时间（包含通信）
        auto optimizer = std::make_shared<Adam>(1e-3);
        DataParallelConfig config;
        config.world_size = world_size;
        config.rank = rank;
        config.global_batch_size = 32;
        config.local_batch_size = 32 / world_size;
        
        DataParallelTrainer trainer(config);
        trainer.setup_model_and_optimizer(model, optimizer);
        
        Tensor targets({config.local_batch_size});
        targets.fill(1.0f);
        
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; ++i) {
            trainer.train_step(inputs, targets);
        }
        end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 20.0;
        
        double comm_overhead = total_time - compute_time;
        double comm_percentage = (comm_overhead / total_time) * 100;
        
        timing_results_[test_name].push_back(total_time);
        detailed_results_[test_name]["compute_time"] = compute_time;
        detailed_results_[test_name]["comm_overhead"] = comm_overhead;
        detailed_results_[test_name]["comm_percentage"] = comm_percentage;
        
        if (rank == 0) {
            std::cout << "Communication overhead (model_size=" << model_size << "): " 
                      << "Total=" << total_time << " μs, "
                      << "Compute=" << compute_time << " μs, "
                      << "Comm=" << comm_overhead << " μs, "
                      << "Comm=" << comm_percentage << "%" << std::endl;
        }
    }
    
    // 生成性能报告
    void generate_performance_report() {
        auto& comm = MPICommunicator::instance();
        int rank = comm.rank();
        
        if (rank == 0) {
            std::cout << "\n=== Generating Performance Report ===" << std::endl;
            
            // 生成详细报告
            generate_detailed_report();
            
            // 生成可视化数据
            generate_visualization_data();
            
            // 生成总结报告
            generate_summary_report();
            
            std::cout << "Performance report generated successfully!" << std::endl;
        }
    }
    
    void generate_detailed_report() {
        std::ofstream report_file(output_dir_ + "/detailed_report.txt");
        
        report_file << "Megatron-CPP-Edu Performance Benchmark Report\n";
        report_file << "==============================================\n\n";
        
        report_file << "System Information:\n";
        report_file << "  MPI World Size: " << MPICommunicator::instance().world_size() << "\n";
        report_file << "  Timestamp: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n\n";
        
        // 张量操作性能
        report_file << "Tensor Operations Performance:\n";
        report_file << "--------------------------------\n";
        for (const auto& [test_name, timings] : timing_results_) {
            if (test_name.find("tensor") == 0 || test_name.find("matrix") != std::string::npos) {
                double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
                report_file << "  " << test_name << ": " << avg_time << " μs\n";
            }
        }
        report_file << "\n";
        
        // 层性能
        report_file << "Layer Performance:\n";
        report_file << "------------------\n";
        for (const auto& [test_name, timings] : timing_results_) {
            if (test_name.find("layer") != std::string::npos || test_name.find("transformer") != std::string::npos) {
                double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
                report_file << "  " << test_name << ": " << avg_time << " μs\n";
            }
        }
        report_file << "\n";
        
        // 通信性能
        report_file << "Communication Performance:\n";
        report_file << "--------------------------\n";
        for (const auto& [test_name, timings] : timing_results_) {
            if (test_name.find("point") != std::string::npos || test_name.find("collective") != std::string::npos || 
                test_name.find("all_reduce") != std::string::npos) {
                double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
                report_file << "  " << test_name << ": " << avg_time << " μs\n";
            }
        }
        report_file << "\n";
        
        // 训练性能
        report_file << "Training Performance:\n";
        report_file << "---------------------\n";
        for (const auto& [test_name, timings] : timing_results_) {
            if (test_name.find("training") != std::string::npos || test_name.find("parallel") != std::string::npos) {
                double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
                report_file << "  " << test_name << ": " << avg_time << " ms\n";
            }
        }
        report_file << "\n";
        
        // 可扩展性分析
        report_file << "Scalability Analysis:\n";
        report_file << "---------------------\n";
        for (const auto& [test_name, timings] : timing_results_) {
            if (test_name.find("scaling") != std::string::npos) {
                double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
                report_file << "  " << test_name << ": " << avg_time << " ms\n";
            }
        }
        report_file << "\n";
        
        // 通信开销分析
        report_file << "Communication Overhead Analysis:\n";
        report_file << "--------------------------------\n";
        for (const auto& [test_name, details] : detailed_results_) {
            if (test_name.find("comm_overhead") != std::string::npos) {
                report_file << "  " << test_name << ":\n";
                report_file << "    Total Time: " << details.at("total_time") << " μs\n";
                report_file << "    Compute Time: " << details.at("compute_time") << " μs\n";
                report_file << "    Comm Overhead: " << details.at("comm_overhead") << " μs\n";
                report_file << "    Comm Percentage: " << details.at("comm_percentage") << "%\n";
            }
        }
        
        report_file.close();
    }
    
    void generate_visualization_data() {
        // 生成CSV格式的可视化数据
        std::ofstream csv_file(output_dir_ + "/performance_data.csv");
        
        csv_file << "Test Name, Average Time (μs), Min Time (μs), Max Time (μs), Std Dev (μs)\n";
        
        for (const auto& [test_name, timings] : timing_results_) {
            double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
            double min_time = *std::min_element(timings.begin(), timings.end());
            double max_time = *std::max_element(timings.begin(), timings.end());
            
            double variance = 0;
            for (double time : timings) {
                variance += (time - avg_time) * (time - avg_time);
            }
            variance /= timings.size();
            double std_dev = std::sqrt(variance);
            
            csv_file << test_name << "," << avg_time << "," << min_time << "," << max_time << "," << std_dev << "\n";
        }
        
        csv_file.close();
        
        // 生成可扩展性数据
        std::ofstream scaling_file(output_dir_ + "/scaling_data.csv");
        scaling_file << "Number of Processes, Strong Scaling Time (ms), Weak Scaling Time (ms), Efficiency (%)\n";
        
        for (const auto& [test_name, timings] : timing_results_) {
            if (test_name.find("strong_scaling") != std::string::npos) {
                int np = std::stoi(test_name.substr(test_name.find("np") + 2));
                double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
                
                // 计算效率
                double base_time = 0;
                for (const auto& [name, t] : timing_results_) {
                    if (name.find("strong_scaling_np1") != std::string::npos) {
                        base_time = std::accumulate(t.begin(), t.end(), 0.0) / t.size();
                        break;
                    }
                }
                double efficiency = (base_time / (np * avg_time)) * 100;
                
                scaling_file << np << "," << avg_time << ",," << efficiency << "\n";
            }
            else if (test_name.find("weak_scaling") != std::string::npos) {
                int np = std::stoi(test_name.substr(test_name.find("np") + 2));
                double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
                
                // 找到对应的弱扩展性时间
                scaling_file << np << ",," << avg_time << ",\n";
            }
        }
        
        scaling_file.close();
    }
    
    void generate_summary_report() {
        std::ofstream summary_file(output_dir_ + "/summary_report.txt");
        
        summary_file << "Megatron-CPP-Edu Performance Summary\n";
        summary_file << "===================================\n\n";
        
        // 计算关键性能指标
        double avg_tensor_op_time = 0;
        double avg_layer_time = 0;
        double avg_comm_time = 0;
        double avg_training_time = 0;
        
        int tensor_ops_count = 0;
        int layer_ops_count = 0;
        int comm_ops_count = 0;
        int training_ops_count = 0;
        
        for (const auto& [test_name, timings] : timing_results_) {
            double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
            
            if (test_name.find("tensor") == 0 || test_name.find("matrix") != std::string::npos) {
                avg_tensor_op_time += avg_time;
                tensor_ops_count++;
            }
            else if (test_name.find("layer") != std::string::npos || test_name.find("transformer") != std::string::npos) {
                avg_layer_time += avg_time;
                layer_ops_count++;
            }
            else if (test_name.find("point") != std::string::npos || test_name.find("collective") != std::string::npos || 
                     test_name.find("all_reduce") != std::string::npos) {
                avg_comm_time += avg_time;
                comm_ops_count++;
            }
            else if (test_name.find("training") != std::string::npos || test_name.find("parallel") != std::string::npos) {
                avg_training_time += avg_time;
                training_ops_count++;
            }
        }
        
        if (tensor_ops_count > 0) avg_tensor_op_time /= tensor_ops_count;
        if (layer_ops_count > 0) avg_layer_time /= layer_ops_count;
        if (comm_ops_count > 0) avg_comm_time /= comm_ops_count;
        if (training_ops_count > 0) avg_training_time /= training_ops_count;
        
        summary_file << "Key Performance Metrics:\n";
        summary_file << "  Average Tensor Operation Time: " << avg_tensor_op_time << " μs\n";
        summary_file << "  Average Layer Operation Time: " << avg_layer_time << " μs\n";
        summary_file << "  Average Communication Time: " << avg_comm_time << " μs\n";
        summary_file << "  Average Training Step Time: " << avg_training_time << " ms\n\n";
        
        // 性能等级评估
        summary_file << "Performance Assessment:\n";
        summary_file << "  Tensor Operations: " << assess_performance(avg_tensor_op_time, 1000) << "\n";
        summary_file << "  Layer Operations: " << assess_performance(avg_layer_time, 10000) << "\n";
        summary_file << "  Communication: " << assess_performance(avg_comm_time, 5000) << "\n";
        summary_file << "  Training Performance: " << assess_performance(avg_training_time * 1000, 50000) << "\n\n";
        
        // 优化建议
        summary_file << "Optimization Recommendations:\n";
        summary_file << "  1. Consider increasing batch size for better GPU utilization\n";
        summary_file << "  2. Use mixed precision training to reduce memory usage\n";
        summary_file << "  3. Enable gradient checkpointing for large models\n";
        summary_file << "  4. Optimize communication overlap with computation\n";
        summary_file << "  5. Consider using tensor parallelism for very large models\n";
        
        summary_file.close();
    }
    
    std::string assess_performance(double value, double threshold) {
        if (value < threshold * 0.5) return "Excellent";
        if (value < threshold) return "Good";
        if (value < threshold * 2) return "Average";
        return "Needs Improvement";
    }
};

int main(int argc, char** argv) {
    // 初始化MPI
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    try {
        // 创建性能基准测试实例
        std::string output_dir = "./benchmark_results";
        if (argc > 1) {
            output_dir = argv[1];
        }
        
        PerformanceBenchmark benchmark(output_dir);
        
        // 运行所有基准测试
        benchmark.run_all_benchmarks();
        
        if (rank == 0) {
            std::cout << "\n=== Benchmark Complete ===" << std::endl;
            std::cout << "Results saved to: " << output_dir << std::endl;
            std::cout << "Files generated:" << std::endl;
            std::cout << "  - detailed_report.txt" << std::endl;
            std::cout << "  - performance_data.csv" << std::endl;
            std::cout << "  - scaling_data.csv" << std::endl;
            std::cout << "  - summary_report.txt" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // 清理MPI
    MPI_Finalize();
    
    return 0;
}