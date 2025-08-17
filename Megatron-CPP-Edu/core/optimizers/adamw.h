#pragma once

#include "optimizer.h"

namespace megatron {

class AdamW : public Optimizer {
public:
    AdamW(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, 
          float eps = 1e-8f, float weight_decay = 0.01f);
    
    void step(const std::vector<Tensor>& parameters, const std::vector<Tensor>& gradients) override;
    
    // Get/set hyperparameters
    float get_beta1() const { return beta1_; }
    void set_beta1(float beta1) { beta1_ = beta1; }
    
    float get_beta2() const { return beta2_; }
    void set_beta2(float beta2) { beta2_ = beta2; }
    
    float get_eps() const { return eps_; }
    void set_eps(float eps) { eps_ = eps; }
    
    float get_weight_decay() const { return weight_decay_; }
    void set_weight_decay(float weight_decay) { weight_decay_ = weight_decay; }
    
    // Get current step count
    int get_step_count() const { return step_count_; }

private:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int step_count_;
    
    // First and second moment estimates
    std::vector<Tensor> exp_avg_;
    std::vector<Tensor> exp_avg_sq_;
    
    // Initialize moment estimates
    void initialize_moments(const std::vector<Tensor>& parameters);
};

} // namespace megatron