#pragma once

#include "optimizer.h"

namespace megatron {

class SGD : public Optimizer {
public:
    SGD(float learning_rate = 0.001f, float momentum = 0.0f, float weight_decay = 0.0f);
    
    void step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) override;
    
    // Get/set momentum
    float get_momentum() const { return momentum_; }
    void set_momentum(float momentum) { momentum_ = momentum; }
    
    // Get/set weight decay
    float get_weight_decay() const { return weight_decay_; }
    void set_weight_decay(float weight_decay) { weight_decay_ = weight_decay; }

private:
    float momentum_;
    float weight_decay_;
    std::vector<Tensor> velocity_;
    
    // Initialize velocity buffers
    void initialize_velocity(const std::vector<Tensor>& parameters);
};

} // namespace megatron