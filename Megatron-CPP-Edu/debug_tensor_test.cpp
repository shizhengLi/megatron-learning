#include <iostream>
#include <memory>
#include "core/tensor/tensor.h"
#include "core/layers/embedding.h"
#include "core/layers/linear.h"
#include "core/layers/attention.h"
#include "gpt_model.h"

using namespace megatron;

int main() {
    try {
        std::cout << "Testing GPT components..." << std::endl;
        
        // Test 1: Embedding layer
        std::cout << "\n1. Testing embedding layer..." << std::endl;
        int vocab_size = 1000;
        int embed_dim = 128;
        auto embedding = std::make_shared<Embedding>(vocab_size, embed_dim, "test_embed");
        
        Tensor input({2, 16});
        for (int i = 0; i < input.size(); ++i) {
            input[i] = i % vocab_size;
        }
        
        Tensor embedded = embedding->forward(input);
        std::cout << "Embedding output shape: [" << embedded.shape()[0] << ", " << embedded.shape()[1] << ", " << embedded.shape()[2] << "]" << std::endl;
        
        // Test 2: Linear layer
        std::cout << "\n2. Testing linear layer..." << std::endl;
        auto linear = std::make_shared<Linear>(embed_dim, embed_dim, true, "test_linear");
        Tensor linear_out = linear->forward(embedded);
        std::cout << "Linear output shape: [" << linear_out.shape()[0] << ", " << linear_out.shape()[1] << ", " << linear_out.shape()[2] << "]" << std::endl;
        
        // Test 3: Attention layer
        std::cout << "\n3. Testing attention layer..." << std::endl;
        int num_heads = 4;
        auto attention = std::make_shared<MultiHeadAttention>(embed_dim, num_heads, false, 0.0f, "test_attention");
        Tensor attn_out = attention->forward(embedded);
        std::cout << "Attention output shape: [" << attn_out.shape()[0] << ", " << attn_out.shape()[1] << ", " << attn_out.shape()[2] << "]" << std::endl;
        
        // Test 4: Full GPT model
        std::cout << "\n4. Testing full GPT model..." << std::endl;
        int max_seq_len = 32;
        int num_layers = 2;
        int ff_dim = 512;
        
        auto gpt_model = std::make_shared<GPTModel>(
            vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim,
            false, 0.0f, "test_gpt");
        
        Tensor gpt_input({2, 16});
        for (int i = 0; i < gpt_input.size(); ++i) {
            gpt_input[i] = i % vocab_size;
        }
        
        Tensor gpt_output = gpt_model->forward(gpt_input);
        std::cout << "GPT output shape: [" << gpt_output.shape()[0] << ", " << gpt_output.shape()[1] << ", " << gpt_output.shape()[2] << "]" << std::endl;
        
        std::cout << "\n✓ All GPT tests passed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "✗ GPT test failed: " << e.what() << std::endl;
        return 1;
    }
}