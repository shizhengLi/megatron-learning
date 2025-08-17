#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <cmath>

#include "core/tensor/tensor.h"
#include "core/layers/attention.h"
#include "core/layers/transformer_block.h"
#include "core/optimizers/sgd.h"
#include "core/optimizers/adamw.h"
#include "core/loss/cross_entropy_loss.h"
#include "core/training/trainer.h"
#include "core/layers/linear.h"
#include "core/layers/layer_norm.h"
#include "core/layers/dropout.h"
#include "core/layers/embedding.h"

using namespace megatron;

bool test_multi_head_attention() {
    std::cout << "Testing MultiHeadAttention..." << std::endl;
    
    try {
        // Create multi-head attention layer
        MultiHeadAttention attention(512, 8, false, 0.1f, "test_attention");
        
        // Create input tensor [batch_size=2, seq_len=10, embed_dim=512]
        Tensor input({2, 10, 512});
        input.random_normal(0.0f, 0.1f);
        
        // Forward pass
        Tensor output = attention.forward(input);
        
        // Check output shape
        std::vector<int> expected_shape = {2, 10, 512};
        assert(output.shape() == expected_shape);
        assert(output.size() == 2 * 10 * 512);
        
        // Check that output is not zero
        bool all_zero = true;
        for (int i = 0; i < output.size(); ++i) {
            if (std::abs(output[i]) > 1e-6f) {
                all_zero = false;
                break;
            }
        }
        assert(!all_zero);
        
        // Test backward pass
        Tensor grad_output({2, 10, 512});
        grad_output.random_normal(0.0f, 0.1f);
        
        Tensor grad_input = attention.backward(grad_output);
        
        // Check gradient shape
        assert(grad_input.shape() == input.shape());
        
        // Test parameter access
        auto params = attention.parameters();
        auto grads = attention.gradients();
        
        assert(!params.empty());
        assert(params.size() == grads.size());
        
        // Test zero_grad
        attention.zero_grad();
        
        std::cout << "✓ MultiHeadAttention test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "✗ MultiHeadAttention test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_transformer_block() {
    std::cout << "Testing TransformerBlock..." << std::endl;
    
    try {
        // Create transformer block
        TransformerBlock block(512, 8, 2048, true, 0.1f, "test_block");
        
        // Create input tensor [batch_size=2, seq_len=10, embed_dim=512]
        Tensor input({2, 10, 512});
        input.random_normal(0.0f, 0.1f);
        
        // Forward pass
        Tensor output = block.forward(input);
        
        // Check output shape
        std::vector<int> expected_shape = {2, 10, 512};
        assert(output.shape() == expected_shape);
        
        // Check that output is not zero
        bool all_zero = true;
        for (int i = 0; i < output.size(); ++i) {
            if (std::abs(output[i]) > 1e-6f) {
                all_zero = false;
                break;
            }
        }
        assert(!all_zero);
        
        // Test backward pass
        Tensor grad_output({2, 10, 512});
        grad_output.random_normal(0.0f, 0.1f);
        
        Tensor grad_input = block.backward(grad_output);
        
        // Check gradient shape
        assert(grad_input.shape() == input.shape());
        
        // Test parameter access
        auto params = block.parameters();
        auto grads = block.gradients();
        
        assert(!params.empty());
        assert(params.size() == grads.size());
        
        // Test zero_grad
        block.zero_grad();
        
        std::cout << "✓ TransformerBlock test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "✗ TransformerBlock test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_sgd_optimizer() {
    std::cout << "Testing SGD optimizer..." << std::endl;
    
    try {
        // Create test parameters
        Tensor param1({10, 5});
        Tensor param2({5});
        param1.random_normal(0.0f, 1.0f);
        param2.random_normal(0.0f, 1.0f);
        
        std::vector<Tensor> params = {param1, param2};
        
        // Create gradients
        Tensor grad1({10, 5});
        Tensor grad2({5});
        grad1.fill(0.1f);
        grad2.fill(0.1f);
        
        std::vector<Tensor> grads = {grad1, grad2};
        
        // Create SGD optimizer
        SGD sgd(0.01f, 0.9f, 0.001f);
        
        // Store original parameters
        Tensor orig_param1 = param1;
        Tensor orig_param2 = param2;
        
        // Perform optimization step
        sgd.step(params, grads);
        
        // Check that parameters changed
        bool param1_changed = false;
        bool param2_changed = false;
        
        for (int i = 0; i < param1.size(); ++i) {
            if (std::abs(param1[i] - orig_param1[i]) > 1e-6f) {
                param1_changed = true;
                break;
            }
        }
        
        for (int i = 0; i < param2.size(); ++i) {
            if (std::abs(param2[i] - orig_param2[i]) > 1e-6f) {
                param2_changed = true;
                break;
            }
        }
        
        assert(param1_changed);
        assert(param2_changed);
        
        // Test zero_grad
        sgd.zero_grad(grads);
        
        // Check gradients are zero
        for (int i = 0; i < grad1.size(); ++i) {
            assert(std::abs(grad1[i]) < 1e-6f);
        }
        for (int i = 0; i < grad2.size(); ++i) {
            assert(std::abs(grad2[i]) < 1e-6f);
        }
        
        std::cout << "✓ SGD optimizer test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "✗ SGD optimizer test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_adamw_optimizer() {
    std::cout << "Testing AdamW optimizer..." << std::endl;
    
    try {
        // Create test parameters
        Tensor param1({10, 5});
        Tensor param2({5});
        param1.random_normal(0.0f, 1.0f);
        param2.random_normal(0.0f, 1.0f);
        
        std::vector<Tensor> params = {param1, param2};
        
        // Create gradients
        Tensor grad1({10, 5});
        Tensor grad2({5});
        grad1.fill(0.1f);
        grad2.fill(0.1f);
        
        std::vector<Tensor> grads = {grad1, grad2};
        
        // Create AdamW optimizer
        AdamW adamw(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
        
        // Store original parameters
        Tensor orig_param1 = param1;
        Tensor orig_param2 = param2;
        
        // Perform optimization step
        adamw.step(params, grads);
        
        // Check that parameters changed
        bool param1_changed = false;
        bool param2_changed = false;
        
        for (int i = 0; i < param1.size(); ++i) {
            if (std::abs(param1[i] - orig_param1[i]) > 1e-6f) {
                param1_changed = true;
                break;
            }
        }
        
        for (int i = 0; i < param2.size(); ++i) {
            if (std::abs(param2[i] - orig_param2[i]) > 1e-6f) {
                param2_changed = true;
                break;
            }
        }
        
        assert(param1_changed);
        assert(param2_changed);
        
        // Test zero_grad
        adamw.zero_grad(grads);
        
        // Check gradients are zero
        for (int i = 0; i < grad1.size(); ++i) {
            assert(std::abs(grad1[i]) < 1e-6f);
        }
        for (int i = 0; i < grad2.size(); ++i) {
            assert(std::abs(grad2[i]) < 1e-6f);
        }
        
        std::cout << "✓ AdamW optimizer test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "✗ AdamW optimizer test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_cross_entropy_loss() {
    std::cout << "Testing CrossEntropyLoss..." << std::endl;
    
    try {
        // Create cross entropy loss
        CrossEntropyLoss loss_fn;
        
        // Create predictions [batch_size=3, num_classes=5]
        Tensor predictions({3, 5});
        predictions[0] = 2.0f; predictions[1] = 1.0f; predictions[2] = 0.1f; predictions[3] = 0.5f; predictions[4] = 0.3f;
        predictions[5] = 0.5f; predictions[6] = 2.0f; predictions[7] = 1.0f; predictions[8] = 0.3f; predictions[9] = 0.1f;
        predictions[10] = 0.1f; predictions[11] = 0.3f; predictions[12] = 2.0f; predictions[13] = 1.0f; predictions[14] = 0.5f;
        
        // Create targets [batch_size=3] (class indices)
        Tensor targets({3});
        targets[0] = 0;  // First sample: class 0
        targets[1] = 1;  // Second sample: class 1
        targets[2] = 2;  // Third sample: class 2
        
        // Compute loss
        float loss = loss_fn.compute(predictions, targets);
        
        // Loss should be positive
        assert(loss > 0.0f);
        
        // Compute gradients
        Tensor grad_loss = loss_fn.backward(predictions, targets);
        
        // Check gradient shape
        assert(grad_loss.shape() == predictions.shape());
        
        // Gradients should sum to zero (approximately) for each sample
        for (int i = 0; i < 3; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 5; ++j) {
                sum += grad_loss[i * 5 + j];
            }
            assert(std::abs(sum) < 1e-6f);
        }
        
        std::cout << "✓ CrossEntropyLoss test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "✗ CrossEntropyLoss test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_trainer() {
    std::cout << "Testing Trainer..." << std::endl;
    
    try {
        // Create a simple model
        auto linear1 = std::make_shared<Linear>(784, 256, true, "linear1");
        auto dropout = std::make_shared<Dropout>(0.5f, "dropout");
        auto linear2 = std::make_shared<Linear>(256, 10, true, "linear2");
        
        std::vector<std::shared_ptr<Layer>> layers = {linear1, dropout, linear2};
        
        // Create optimizer and loss function
        auto optimizer = std::make_shared<SGD>(0.01f);
        auto loss_fn = std::make_shared<CrossEntropyLoss>();
        
        // Create trainer
        Trainer trainer(layers, optimizer, loss_fn);
        
        // Create training data
        Tensor inputs({2, 784});
        inputs.random_normal(0.0f, 1.0f);
        
        Tensor targets({2});
        targets[0] = 3; targets[1] = 7;
        
        // Perform training step
        float loss = trainer.train_step(inputs, targets);
        
        // Loss should be positive
        assert(loss > 0.0f);
        
        // Check step count
        assert(trainer.get_step_count() == 1);
        
        // Test evaluation mode
        trainer.eval();
        float eval_loss = trainer.evaluate_step(inputs, targets);
        assert(eval_loss > 0.0f);
        
        // Test training mode
        trainer.train();
        
        // Test batch training
        std::vector<Tensor> batch_inputs = {inputs, inputs};
        std::vector<Tensor> batch_targets = {targets, targets};
        
        auto batch_losses = trainer.train_epoch(batch_inputs, batch_targets);
        assert(batch_losses.size() == 2);
        
        for (float batch_loss : batch_losses) {
            assert(batch_loss > 0.0f);
        }
        
        // Test parameter saving/loading (basic test)
        trainer.save_checkpoint("/tmp/test_checkpoint.bin");
        trainer.load_checkpoint("/tmp/test_checkpoint.bin");
        
        std::cout << "✓ Trainer test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "✗ Trainer test failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_training_integration() {
    std::cout << "Testing full training integration..." << std::endl;
    
    try {
        // Create a more complex model with transformer components
        auto embedding = std::make_shared<Embedding>(1000, 512, "embedding");
        auto transformer = std::make_shared<TransformerBlock>(512, 8, 2048, true, 0.1f, "transformer");
        auto linear = std::make_shared<Linear>(512, 10, true, "classifier");
        
        std::vector<std::shared_ptr<Layer>> layers = {embedding, transformer, linear};
        
        // Create AdamW optimizer
        auto optimizer = std::make_shared<AdamW>(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
        auto loss_fn = std::make_shared<CrossEntropyLoss>();
        
        // Create trainer
        Trainer trainer(layers, optimizer, loss_fn);
        
        // Create token inputs [batch_size=2, seq_len=16]
        Tensor tokens({2, 16});
        for (int i = 0; i < tokens.size(); ++i) {
            tokens[i] = rand() % 1000;  // Random token IDs
        }
        
        // Create targets
        Tensor targets({2});
        targets[0] = 3; targets[1] = 7;
        
        // Perform several training steps
        for (int i = 0; i < 5; ++i) {
            float loss = trainer.train_step(tokens, targets);
            assert(loss > 0.0f);
            
            // Loss should generally decrease (though not guaranteed due to randomness)
            if (i > 0) {
                float avg_loss = trainer.get_average_loss();
                assert(avg_loss > 0.0f);
            }
        }
        
        // Test evaluation
        trainer.eval();
        float eval_loss = trainer.evaluate_step(tokens, targets);
        assert(eval_loss > 0.0f);
        
        std::cout << "✓ Training integration test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "✗ Training integration test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "Running Phase 3 Tests..." << std::endl;
    std::cout << "===========================" << std::endl;
    
    int passed = 0;
    int total = 7;
    
    if (test_multi_head_attention()) passed++;
    if (test_transformer_block()) passed++;
    if (test_sgd_optimizer()) passed++;
    if (test_adamw_optimizer()) passed++;
    if (test_cross_entropy_loss()) passed++;
    if (test_trainer()) passed++;
    if (test_training_integration()) passed++;
    
    std::cout << "===========================" << std::endl;
    std::cout << "Phase 3 Tests: " << passed << "/" << total << " passed" << std::endl;
    
    if (passed == total) {
        std::cout << "🎉 All Phase 3 tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests failed!" << std::endl;
        return 1;
    }
}