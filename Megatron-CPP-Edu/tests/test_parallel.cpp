#include <gtest/gtest.h>
#include "core/parallel/tensor_parallel.h"
#include "core/parallel/data_parallel.h"
#include "distributed/communication.h"
#include "core/layers/linear.h"
#include "core/optimizers/adamw.h"
#include <iostream>
#include <vector>
#include <memory>

using namespace megatron;

class ParallelComputingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化MPI
        int argc = 0;
        char** argv = nullptr;
        MPICommunicator::instance().initialize(&argc, &argv);
        
        // 初始化张量并行上下文
        auto& comm = MPICommunicator::instance();
        int world_size = comm.world_size();
        int rank = comm.rank();
        
        TensorParallelContext::instance().initialize(world_size, rank);
        
        // 设置测试配置
        tp_config_.world_size = world_size;
        tp_config_.rank = rank;
        
        dp_config_.world_size = world_size;
        dp_config_.rank = rank;
        dp_config_.global_batch_size = 32;
        dp_config_.local_batch_size = 32 / world_size;
    }
    
    void TearDown() override {
        MPICommunicator::instance().finalize();
    }
    
    // 创建测试数据
    Tensor create_test_tensor(const std::vector<int>& shape) {
        Tensor tensor(shape);
        tensor.fill(1.0f);
        return tensor;
    }
    
    // 创建测试模型
    std::shared_ptr<Layer> create_test_model() {
        return std::make_shared<Linear>(10, 5, true);
    }
    
    // 比较两个张量是否近似相等
    bool tensors_approx_equal(const Tensor& a, const Tensor& b, float eps = 1e-6f) {
        if (a.shape() != b.shape()) {
            return false;
        }
        
        for (int i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > eps) {
                return false;
            }
        }
        
        return true;
    }
    
    // 张量并行配置
    struct {
        int world_size;
        int rank;
    } tp_config_;
    
    // 数据并行配置
    DataParallelConfig dp_config_;
};

// MPI通信测试
TEST_F(ParallelComputingTest, MPICommunicationTest) {
    auto& comm = MPICommunicator::instance();
    
    // 测试基本属性
    EXPECT_GT(comm.world_size(), 0);
    EXPECT_GE(comm.rank(), 0);
    EXPECT_LT(comm.rank(), comm.world_size());
    EXPECT_TRUE(comm.is_initialized());
    
    // 测试广播
    Tensor tensor({2, 3});
    if (comm.rank() == 0) {
        tensor.fill(5.0f);
    } else {
        tensor.fill(0.0f);
    }
    
    comm.broadcast(tensor, 0);
    
    // 验证广播结果
    for (int i = 0; i < tensor.size(); ++i) {
        EXPECT_FLOAT_EQ(tensor[i], 5.0f);
    }
    
    // 测试all-reduce
    Tensor reduce_tensor({2, 3});
    reduce_tensor.fill(static_cast<float>(comm.rank() + 1));
    
    comm.all_reduce(reduce_tensor);
    
    // 验证all-reduce结果（应该等于平均值）
    float expected_value = 0.0f;
    for (int i = 0; i < comm.world_size(); ++i) {
        expected_value += static_cast<float>(i + 1);
    }
    expected_value /= comm.world_size();
    
    for (int i = 0; i < reduce_tensor.size(); ++i) {
        EXPECT_NEAR(reduce_tensor[i], expected_value, 1e-6f);
    }
    
    // 测试barrier
    EXPECT_NO_THROW(comm.barrier());
}

// 张量并行上下文测试
TEST_F(ParallelComputingTest, TensorParallelContextTest) {
    auto& tp_ctx = TensorParallelContext::instance();
    
    // 测试基本属性
    EXPECT_EQ(tp_ctx.world_size(), tp_config_.world_size);
    EXPECT_EQ(tp_ctx.rank(), tp_config_.rank);
    
    // 测试本地维度计算
    int global_dim = 10;
    int local_dim = tp_ctx.get_local_output_dim(global_dim);
    
    // 验证本地维度计算
    int total_local_dim = 0;
    for (int i = 0; i < tp_config_.world_size; ++i) {
        total_local_dim += tp_ctx.get_local_output_dim(global_dim);
    }
    EXPECT_EQ(total_local_dim, global_dim);
    
    // 测试是否启用张量并行
    bool tp_enabled = tp_ctx.is_enabled();
    EXPECT_EQ(tp_enabled, tp_config_.world_size > 1);
}

// 列并行线性层测试
TEST_F(ParallelComputingTest, ColumnParallelLinearTest) {
    int in_features = 10;
    int out_features = 8;
    
    auto layer = std::make_shared<ColumnParallelLinear>(in_features, out_features, true, "test_col_linear");
    
    // 测试前向传播
    Tensor input({2, in_features});
    input.fill(1.0f);
    
    Tensor output = layer->forward(input);
    
    // 验证输出形状
    auto& tp_ctx = TensorParallelContext::instance();
    int expected_out_features = tp_ctx.get_local_output_dim(out_features);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], expected_out_features);
    
    // 测试反向传播
    Tensor grad_output(output.shape());
    grad_output.fill(0.1f);
    
    Tensor grad_input = layer->backward(grad_output);
    
    // 验证梯度形状
    EXPECT_EQ(grad_input.shape()[0], 2);
    EXPECT_EQ(grad_input.shape()[1], in_features);
    
    // 测试参数访问
    auto params = layer->parameters();
    auto grads = layer->gradients();
    
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(params.size(), grads.size());
}

// 行并行线性层测试
TEST_F(ParallelComputingTest, RowParallelLinearTest) {
    int in_features = 10;
    int out_features = 6;
    
    auto layer = std::make_shared<RowParallelLinear>(in_features, out_features, true, "test_row_linear");
    
    // 测试前向传播
    Tensor input({2, in_features});
    input.fill(1.0f);
    
    Tensor output = layer->forward(input);
    
    // 验证输出形状
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], out_features);
    
    // 测试反向传播
    Tensor grad_output(output.shape());
    grad_output.fill(0.1f);
    
    Tensor grad_input = layer->backward(grad_output);
    
    // 验证梯度形状
    auto& tp_ctx = TensorParallelContext::instance();
    int expected_in_features = tp_ctx.get_local_input_dim(in_features);
    EXPECT_EQ(grad_input.shape()[0], 2);
    EXPECT_EQ(grad_input.shape()[1], expected_in_features);
    
    // 测试参数访问
    auto params = layer->parameters();
    auto grads = layer->gradients();
    
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(params.size(), grads.size());
}

// 张量并行多头注意力测试
TEST_F(ParallelComputingTest, TensorParallelMultiHeadAttentionTest) {
    int embed_dim = 12;
    int num_heads = 4;
    
    auto layer = std::make_shared<TensorParallelMultiHeadAttention>(embed_dim, num_heads, true, "test_tp_mha");
    
    // 测试前向传播
    Tensor input({2, 8, embed_dim});
    input.fill(0.1f);
    
    Tensor output = layer->forward(input);
    
    // 验证输出形状
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], embed_dim);
    
    // 测试反向传播
    Tensor grad_output(output.shape());
    grad_output.fill(0.01f);
    
    Tensor grad_input = layer->backward(grad_output);
    
    // 验证梯度形状
    EXPECT_EQ(grad_input.shape(), input.shape());
    
    // 测试参数访问
    auto params = layer->parameters();
    auto grads = layer->gradients();
    
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(params.size(), grads.size());
}

// 张量并行FFN测试
TEST_F(ParallelComputingTest, TensorParallelFFNTest) {
    int embed_dim = 8;
    int ffn_dim = 16;
    
    auto layer = std::make_shared<TensorParallelFFN>(embed_dim, ffn_dim, 0.1f, "test_tp_ffn");
    
    // 测试前向传播
    Tensor input({2, embed_dim});
    input.fill(0.1f);
    
    Tensor output = layer->forward(input);
    
    // 验证输出形状
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], embed_dim);
    
    // 测试反向传播
    Tensor grad_output(output.shape());
    grad_output.fill(0.01f);
    
    Tensor grad_input = layer->backward(grad_output);
    
    // 验证梯度形状
    EXPECT_EQ(grad_input.shape(), input.shape());
    
    // 测试参数访问
    auto params = layer->parameters();
    auto grads = layer->gradients();
    
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(params.size(), grads.size());
}

// 张量并行Transformer块测试
TEST_F(ParallelComputingTest, TensorParallelTransformerBlockTest) {
    int embed_dim = 8;
    int num_heads = 2;
    int ffn_dim = 16;
    
    auto layer = std::make_shared<TensorParallelTransformerBlock>(embed_dim, num_heads, ffn_dim, 0.1f, "test_tp_block");
    
    // 测试前向传播
    Tensor input({2, embed_dim});
    input.fill(0.1f);
    
    Tensor output = layer->forward(input);
    
    // 验证输出形状
    EXPECT_EQ(output.shape(), input.shape());
    
    // 测试反向传播
    Tensor grad_output(output.shape());
    grad_output.fill(0.01f);
    
    Tensor grad_input = layer->backward(grad_output);
    
    // 验证梯度形状
    EXPECT_EQ(grad_input.shape(), input.shape());
    
    // 测试参数访问
    auto params = layer->parameters();
    auto grads = layer->gradients();
    
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(params.size(), grads.size());
}

// 张量并行模型构建器测试
TEST_F(ParallelComputingTest, TensorParallelModelBuilderTest) {
    TensorParallelModelBuilder builder(tp_config_.world_size);
    
    // 设置模型配置
    builder.set_vocab_size(100);
    builder.set_embed_dim(8);
    builder.set_num_layers(2);
    builder.set_num_heads(2);
    builder.set_ffn_dim(16);
    
    // 测试构建GPT模型
    auto gpt_model = builder.build_gpt_model();
    EXPECT_NE(gpt_model, nullptr);
    
    // 测试模型前向传播
    Tensor input({2, 10});
    input.fill(1.0f);
    
    Tensor output = gpt_model->forward(input);
    EXPECT_GT(output.size(), 0);
    
    // 测试构建Transformer分类器
    auto classifier = builder.build_transformer_classifier(5);
    EXPECT_NE(classifier, nullptr);
    
    // 测试分类器前向传播
    Tensor class_output = classifier->forward(input);
    EXPECT_GT(class_output.size(), 0);
}

// 数据并行配置测试
TEST_F(ParallelComputingTest, DataParallelConfigTest) {
    // 测试有效配置
    EXPECT_TRUE(dp_config_.validate());
    
    // 测试无效配置
    DataParallelConfig invalid_config;
    invalid_config.world_size = -1;
    EXPECT_FALSE(invalid_config.validate());
    
    invalid_config.world_size = 1;
    invalid_config.rank = 2;
    EXPECT_FALSE(invalid_config.validate());
    
    // 测试梯度累积步数计算
    int accumulation_steps = dp_config_.get_gradient_accumulation_steps();
    EXPECT_GT(accumulation_steps, 0);
}

// 数据并行训练器测试
TEST_F(ParallelComputingTest, DataParallelTrainerTest) {
    DataParallelTrainer trainer(dp_config_);
    
    // 设置模型和优化器
    auto model = create_test_model();
    auto optimizer = std::make_shared<AdamW>(0.001f);
    
    trainer.setup_model_and_optimizer(model, optimizer);
    
    // 测试训练步骤
    Tensor inputs({dp_config_.local_batch_size, 10});
    Tensor targets({dp_config_.local_batch_size, 5});
    
    inputs.fill(0.1f);
    targets.fill(0.2f);
    
    EXPECT_NO_THROW(trainer.train_step(inputs, targets));
    
    // 测试梯度同步
    EXPECT_NO_THROW(trainer.synchronize_gradients());
    
    // 测试参数同步
    EXPECT_NO_THROW(trainer.synchronize_parameters());
    
    // 测试分布式状态
    EXPECT_EQ(trainer.world_size(), dp_config_.world_size);
    EXPECT_EQ(trainer.rank(), dp_config_.rank);
    EXPECT_EQ(trainer.is_distributed(), dp_config_.world_size > 1);
}

// 分布式数据并行测试
TEST_F(ParallelComputingTest, DistributedDataParallelTest) {
    auto model = create_test_model();
    auto ddp_model = std::make_shared<DistributedDataParallel>(model, dp_config_);
    
    // 测试前向传播
    Tensor input({2, 10});
    input.fill(0.1f);
    
    Tensor output = ddp_model->forward(input);
    EXPECT_GT(output.size(), 0);
    
    // 测试反向传播
    Tensor grad_output(output.shape());
    grad_output.fill(0.01f);
    
    Tensor grad_input = ddp_model->backward(grad_output);
    EXPECT_EQ(grad_input.shape(), input.shape());
    
    // 测试参数访问
    auto params = ddp_model->parameters();
    auto grads = ddp_model->gradients();
    
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(params.size(), grads.size());
    
    // 测试同步操作
    EXPECT_NO_THROW(ddp_model->sync_parameters());
    EXPECT_NO_THROW(ddp_model->sync_gradients());
}

// 张量并行工具函数测试
TEST_F(ParallelComputingTest, TensorParallelUtilsTest) {
    // 测试张量并行支持检查
    bool tp_supported = tensor_parallel_utils::is_tensor_parallel_supported();
    EXPECT_TRUE(tp_supported);
    
    // 测试张量并行配置验证
    bool config_valid = tensor_parallel_utils::validate_tensor_parallel_config(
        tp_config_.world_size, tp_config_.rank);
    EXPECT_TRUE(config_valid);
    
    // 测试获取张量并行信息
    int tp_world_size = tensor_parallel_utils::get_tensor_parallel_world_size();
    int tp_rank = tensor_parallel_utils::get_tensor_parallel_rank();
    
    EXPECT_EQ(tp_world_size, tp_config_.world_size);
    EXPECT_EQ(tp_rank, tp_config_.rank);
    
    // 测试张量分割和合并
    Tensor original({4, 6});
    original.fill(1.0f);
    
    Tensor split_tensor = tensor_parallel_utils::split_tensor(original, 1, tp_config_.rank, tp_config_.world_size);
    EXPECT_GT(split_tensor.size(), 0);
    
    std::vector<Tensor> tensors_to_merge = {split_tensor};
    Tensor merged_tensor = tensor_parallel_utils::concatenate_tensors(tensors_to_merge, 1);
    EXPECT_GT(merged_tensor.size(), 0);
}

// 数据并行工具函数测试
TEST_F(ParallelComputingTest, DataParallelUtilsTest) {
    // 测试数据并行支持检查
    bool dp_supported = data_parallel_utils::is_data_parallel_supported();
    EXPECT_TRUE(dp_supported);
    
    // 测试数据并行配置验证
    bool config_valid = data_parallel_utils::validate_data_parallel_config(dp_config_);
    EXPECT_TRUE(config_valid);
    
    // 测试本地批大小计算
    int local_batch_size = data_parallel_utils::calculate_local_batch_size(
        dp_config_.global_batch_size, dp_config_.world_size);
    EXPECT_GT(local_batch_size, 0);
    
    // 测试数据并行信息打印
    EXPECT_NO_THROW(data_parallel_utils::print_data_parallel_info(dp_config_));
    
    // 测试创建DDP模型
    auto model = create_test_model();
    auto ddp_model = data_parallel_utils::create_ddp_model(model, dp_config_);
    EXPECT_NE(ddp_model, nullptr);
    
    // 测试创建分布式优化器
    auto optimizer = std::make_shared<AdamW>(0.001f);
    auto dist_optimizer = data_parallel_utils::create_distributed_optimizer(optimizer, dp_config_);
    EXPECT_NE(dist_optimizer, nullptr);
}

// 分布式通信集成测试
TEST_F(ParallelComputingTest, DistributedCommunicationIntegrationTest) {
    auto& comm = MPICommunicator::instance();
    
    // 测试分布式训练器
    DistributedTrainer dist_trainer(comm.world_size(), comm.rank());
    
    auto model = create_test_model();
    auto optimizer = std::make_shared<AdamW>(0.001f);
    
    dist_trainer.setup_model_and_optimizer(model, optimizer);
    
    // 测试分布式训练步骤
    Tensor inputs({2, 10});
    Tensor targets({2, 5});
    
    inputs.fill(0.1f);
    targets.fill(0.2f);
    
    EXPECT_NO_THROW(dist_trainer.distributed_train_step(inputs, targets));
    
    // 测试梯度同步
    auto gradients = model->gradients();
    EXPECT_NO_THROW(dist_trainer.synchronize_gradients(gradients));
    
    // 测试参数广播
    auto parameters = model->parameters();
    EXPECT_NO_THROW(dist_trainer.broadcast_parameters(parameters));
}

// 混合并行测试
TEST_F(ParallelComputingTest, HybridParallelTest) {
    // 测试混合并行配置
    HybridParallelTrainer hybrid_trainer(dp_config_, 2, 1);
    
    auto model = create_test_model();
    auto optimizer = std::make_shared<AdamW>(0.001f);
    
    hybrid_trainer.setup_model_and_optimizer(model, optimizer);
    
    // 测试混合并行训练步骤
    Tensor inputs({2, 10});
    Tensor targets({2, 5});
    
    inputs.fill(0.1f);
    targets.fill(0.2f);
    
    EXPECT_NO_THROW(hybrid_trainer.hybrid_train_step(inputs, targets));
    
    // 测试并行状态获取
    int dp_size = hybrid_trainer.get_data_parallel_size();
    int tp_size = hybrid_trainer.get_tensor_parallel_size();
    int pp_size = hybrid_trainer.get_pipeline_parallel_size();
    
    EXPECT_GT(dp_size, 0);
    EXPECT_GT(tp_size, 0);
    EXPECT_GT(pp_size, 0);
}

// 性能和正确性验证测试
TEST_F(ParallelComputingTest, PerformanceAndCorrectnessTest) {
    // 测试张量并行性能
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Tensor input({4, 10});
    input.fill(0.1f);
    
    auto tp_model = std::make_shared<TensorParallelTransformerBlock>(10, 2, 20, 0.1f, "perf_test");
    Tensor output = tp_model->forward(input);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_LT(duration.count(), 1000); // 应该在1秒内完成
    
    // 测试数值稳定性
    EXPECT_FALSE(std::isnan(output[0]));
    EXPECT_FALSE(std::isinf(output[0]));
    
    // 测试多次运行的一致性
    Tensor output2 = tp_model->forward(input);
    EXPECT_TRUE(tensors_approx_equal(output, output2, 1e-5f));
}

// 错误处理测试
TEST_F(ParallelComputingTest, ErrorHandlingTest) {
    // 测试无效配置
    EXPECT_THROW(TensorParallelContext::instance().initialize(-1, 0), std::runtime_error);
    
    // 测试无效张量形状
    Tensor empty_tensor;
    auto& comm = MPICommunicator::instance();
    
    // 测试通信错误处理
    if (comm.world_size() > 1) {
        EXPECT_THROW(comm.send(empty_tensor, 1, 0), std::invalid_argument);
    }
    
    // 测试数据并行配置验证
    DataParallelConfig invalid_config;
    invalid_config.world_size = 2;
    invalid_config.rank = 3;
    invalid_config.global_batch_size = 16;
    invalid_config.local_batch_size = 10;
    
    EXPECT_FALSE(data_parallel_utils::validate_data_parallel_config(invalid_config));
}

// 内存管理测试
TEST_F(ParallelComputingTest, MemoryManagementTest) {
    // 测试大张量的内存管理
    Tensor large_tensor({100, 100});
    large_tensor.fill(1.0f);
    
    auto tp_layer = std::make_shared<ColumnParallelLinear>(100, 50, true, "memory_test");
    
    // 测试前向传播内存使用
    Tensor output = tp_layer->forward(large_tensor);
    EXPECT_EQ(output.shape()[0], 100);
    
    // 测试反向传播内存使用
    Tensor grad_output(output.shape());
    grad_output.fill(0.01f);
    
    Tensor grad_input = tp_layer->backward(grad_output);
    EXPECT_EQ(grad_input.shape()[0], 100);
    
    // 验证没有内存泄漏（通过检查参数和梯度的大小）
    auto params = tp_layer->parameters();
    auto grads = tp_layer->gradients();
    
    for (const auto& param : params) {
        EXPECT_GT(param.size(), 0);
    }
    
    for (const auto& grad : grads) {
        EXPECT_GT(grad.size(), 0);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}