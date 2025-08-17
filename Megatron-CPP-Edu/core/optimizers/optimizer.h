#pragma once

#include "../tensor/tensor.h"
#include <vector>
#include <memory>

namespace megatron {

class Optimizer {
public:
    Optimizer(float learning_rate = 0.001f);
    virtual ~Optimizer() = default;
    
    // Main optimization step
    virtual void step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) = 0;
    
    // Zero out gradients
    virtual void zero_grad(const std::vector<Tensor>& gradients);
    
    // Set learning rate
    void set_learning_rate(float lr) { learning_rate_ = lr; }
    float get_learning_rate() const { return learning_rate_; }
    
    // Set parameters for optimization
    void set_parameters(const std::vector<Tensor>& parameters);
    
protected:
    float learning_rate_;
    std::vector<Tensor> parameters_;
    
    // Helper method to validate parameters and gradients
    void validate_step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients);
};

} // namespace megatron