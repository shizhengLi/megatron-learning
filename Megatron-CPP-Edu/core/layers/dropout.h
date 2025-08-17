#pragma once

#include "layer.h"
#include <random>

namespace megatron {

class Dropout : public Layer {
public:
    Dropout(float p = 0.5, const std::string& name = "");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    // Set dropout probability
    void set_p(float p) { p_ = p; }
    float get_p() const { return p_; }

private:
    float p_;  // Probability of dropping a unit
    float scale_;  // Scaling factor: 1 / (1 - p)
    
    Tensor mask_;  // Store dropout mask for backward pass
    
    // Random number generation
    std::mt19937 gen_;
    std::uniform_real_distribution<float> dist_;
};

} // namespace megatron