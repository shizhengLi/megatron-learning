#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <cmath>
#include <chrono>
#include <thread>

#include "models/gpt/gpt_model.h"
#include "models/transformer/transformer_classifier.h"
#include "core/data/dataset.h"
#include "core/evaluation/metrics.h"
#include "core/performance/performance.h"
#include "core/loss/cross_entropy_loss.h"
#include "core/optimizers/adamw.h"
#include "core/training/trainer.h"

using namespace megatron;

bool test_gpt_model_detailed() {
    std::cout << "Testing GPT Model (detailed)..." << std::endl;
    
    try {
        // Create GPT model
        int vocab_size = 1000;
        int max_seq_len = 32;
        int embed_dim = 128;
        int num_heads = 4;
        int num_layers = 2;
        int ff_dim = 512;
        
        std::cout << "Creating GPT model with parameters:" << std::endl;
        std::cout << "  vocab_size: " << vocab_size << std::endl;
        std::cout << "  max_seq_len: " << max_seq_len << std::endl;
        std::cout << "  embed_dim: " << embed_dim << std::endl;
        std::cout << "  num_heads: " << num_heads << std::endl;
        std::cout << "  num_layers: " << num_layers << std::endl;
        std::cout << "  ff_dim: " << ff_dim << std::endl;
        
        auto model = std::make_shared<GPTModel>(
            vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim,
            true, 0.1f, "test_gpt");
        
        std::cout << "Model created successfully" << std::endl;
        
        // Create input tensor with valid token indices
        Tensor input({2, 16});  // batch_size=2, seq_len=16
        std::cout << "Input tensor shape: [" << input.shape()[0] << ", " << input.shape()[1] << "]" << std::endl;
        std::cout << "Input tensor size: " << input.size() << std::endl;
        
        for (int i = 0; i < input.size(); ++i) {
            input[i] = rand() % vocab_size;  // Ensure tokens are within vocab range
            if (i < 10) {  // Print first 10 values
                std::cout << "input[" << i << "] = " << input[i] << std::endl;
            }
        }
        
        std::cout << "Starting forward pass..." << std::endl;
        
        // Forward pass
        Tensor output = model->forward(input);
        
        std::cout << "Forward pass completed successfully" << std::endl;
        std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << ", " << output.shape()[2] << "]" << std::endl;
        std::cout << "Output size: " << output.size() << std::endl;
        
        // Check output shape
        std::vector<int> expected_shape = {2, 16, vocab_size};
        if (output.shape() != expected_shape) {
            std::cout << "Shape mismatch! Expected: [" << expected_shape[0] << ", " << expected_shape[1] << ", " << expected_shape[2] << "]" << std::endl;
            return false;
        }
        
        // Check that output is not zero
        bool all_zero = true;
        for (int i = 0; i < output.size(); ++i) {
            if (std::abs(output[i]) > 1e-6f) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            std::cout << "Output is all zero!" << std::endl;
            return false;
        }
        
        std::cout << "✓ GPT Model test passed" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "✗ GPT Model test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    test_gpt_model_detailed();
    return 0;
}