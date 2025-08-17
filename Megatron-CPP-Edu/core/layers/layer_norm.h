#pragma once

#include "layer.h"

namespace megatron {

class LayerNorm : public Layer {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5, const std::string& name = "");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    // Access to parameters
    const Tensor& weight() const { return weight_; }
    const Tensor& bias() const { return bias_; }

private:
    int normalized_shape_;
    float eps_;
    
    Tensor weight_;      // [normalized_shape]
    Tensor bias_;        // [normalized_shape]
    Tensor weight_grad_; // [normalized_shape]
    Tensor bias_grad_;   // [normalized_shape]
    
    Tensor input_;       // Store input for backward pass
    Tensor mean_;        // Store mean for backward pass
    Tensor var_;         // Store variance for backward pass
    Tensor normalized_;  // Store normalized input for backward pass
    
    // Helper methods
    void initialize_parameters();
};

} // namespace megatron