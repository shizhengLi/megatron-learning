#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <cmath>

// Forward declarations to avoid header issues
namespace megatron {
    class Tensor {
    public:
        Tensor(const std::vector<int>& shape);
        ~Tensor();
        float& operator[](int index);
        const float& operator[](int index) const;
        const std::vector<int>& shape() const;
        int size() const;
    private:
        std::vector<int> shape_;
        float* data_;
        int size_;
    };
    
    class Layer {
    public:
        virtual ~Layer() = default;
        virtual Tensor forward(const Tensor& input) = 0;
        virtual Tensor backward(const Tensor& grad_output) = 0;
        virtual std::vector<Tensor> parameters() const = 0;
        virtual std::vector<Tensor> gradients() const = 0;
        virtual void zero_grad() = 0;
    };
    
    class GPTModel : public Layer {
    public:
        GPTModel(int vocab_size, int max_seq_len, int embed_dim, int num_heads, 
                 int num_layers, int ff_dim, bool use_dropout, float dropout_prob, 
                 const std::string& name);
        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad_output) override;
        std::vector<Tensor> parameters() const override;
        std::vector<Tensor> gradients() const override;
        void zero_grad() override;
    private:
        int vocab_size_;
        int max_seq_len_;
        int embed_dim_;
        int num_heads_;
        int num_layers_;
        int ff_dim_;
        bool use_dropout_;
        float dropout_prob_;
    };
}

using namespace megatron;

int main() {
    std::cout << "Testing GPT Model Index Issue..." << std::endl;
    
    try {
        // Create GPT model
        int vocab_size = 1000;
        int max_seq_len = 32;
        int embed_dim = 128;
        int num_heads = 4;
        int num_layers = 2;
        int ff_dim = 512;
        
        std::cout << "Creating GPT model..." << std::endl;
        auto model = std::make_shared<GPTModel>(
            vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim,
            true, 0.1f, "test_gpt");
        
        std::cout << "GPT model created successfully" << std::endl;
        
        // Create input tensor with valid token indices
        Tensor input({2, 16});  // batch_size=2, seq_len=16
        std::cout << "Input tensor shape: [" << input.shape()[0] << ", " << input.shape()[1] << "]" << std::endl;
        std::cout << "Input tensor size: " << input.size() << std::endl;
        
        for (int i = 0; i < input.size(); ++i) {
            input[i] = i % vocab_size;  // Ensure tokens are within vocab range
            if (i < 10) {
                std::cout << "input[" << i << "] = " << input[i] << std::endl;
            }
        }
        
        std::cout << "Starting forward pass..." << std::endl;
        // Forward pass
        Tensor output = model->forward(input);
        std::cout << "Forward pass completed" << std::endl;
        
        std::cout << "Output tensor shape: [" << output.shape()[0] << ", " << output.shape()[1] << ", " << output.shape()[2] << "]" << std::endl;
        std::cout << "Output tensor size: " << output.size() << std::endl;
        
        // Try to access output elements
        std::cout << "Checking output elements..." << std::endl;
        for (int i = 0; i < std::min(10, output.size()); ++i) {
            std::cout << "output[" << i << "] = " << output[i] << std::endl;
        }
        
        std::cout << "✓ GPT Model test passed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "✗ GPT Model test failed: " << e.what() << std::endl;
        return 1;
    }
}