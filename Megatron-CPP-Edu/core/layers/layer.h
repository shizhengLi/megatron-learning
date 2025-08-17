#pragma once

#include "../tensor/tensor.h"
#include <string>
#include <vector>

namespace megatron {

class Layer {
public:
    Layer(const std::string& name = "") : name_(name), training_(true) {}
    virtual ~Layer() = default;

    // Forward pass
    virtual Tensor forward(const Tensor& input) = 0;

    // Backward pass (returns gradients for input)
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // Get layer parameters
    virtual std::vector<Tensor> parameters() const = 0;
    virtual std::vector<Tensor> gradients() const = 0;

    // Set training/evaluation mode
    virtual void train() { training_ = true; }
    virtual void eval() { training_ = false; }
    bool is_training() const { return training_; }

    // Get layer name
    const std::string& name() const { return name_; }

    // Reset gradients
    virtual void zero_grad() {
        auto grads = gradients();
        for (auto& grad : grads) {
            grad.zeros();
        }
    }

protected:
    std::string name_;
    bool training_;
};

} // namespace megatron