#pragma once

#include "../tensor/tensor.h"

namespace megatron {

class Loss {
public:
    virtual ~Loss() = default;
    
    // Compute loss and return loss value
    virtual float compute(const Tensor& predictions, const Tensor& targets) = 0;
    
    // Compute loss and gradients
    virtual Tensor backward(const Tensor& predictions, const Tensor& targets) = 0;
    
    // Get the last computed loss value
    float get_last_loss() const { return last_loss_; }

protected:
    float last_loss_ = 0.0f;
};

} // namespace megatron