#pragma once

#include "layer.h"
#include <memory>

namespace megatron {

class Linear : public Layer {
public:
    Linear(int in_features, int out_features, bool bias = true, const std::string& name = "");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
    
    // Access to weights and biases
    const Tensor& weight() const { return weight_; }
    const Tensor& bias() const { return bias_; }
    
    // Set weights and biases (for initialization)
    void set_weight(const Tensor& weight);
    void set_bias(const Tensor& bias);

private:
    int in_features_;
    int out_features_;
    bool has_bias_;
    
    Tensor weight_;      // [out_features, in_features]
    Tensor bias_;        // [out_features]
    Tensor weight_grad_; // [out_features, in_features]
    Tensor bias_grad_;   // [out_features]
    
    Tensor input_;       // Store input for backward pass
    Tensor output_;      // Store output for backward pass
    
    // Helper methods
    void initialize_parameters();
    void compute_gradients(const Tensor& grad_output);
};

} // namespace megatron