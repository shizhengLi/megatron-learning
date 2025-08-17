#pragma once

#include "loss.h"

namespace megatron {

class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss();
    
    float compute(const Tensor& predictions, const Tensor& targets) override;
    Tensor backward(const Tensor& predictions, const Tensor& targets) override;
    
    // Get the last computed probabilities (for analysis)
    const Tensor& get_last_probabilities() const { return last_probabilities_; }

private:
    Tensor last_probabilities_;
    Tensor last_targets_one_hot_;
    
    // Helper methods
    Tensor softmax(const Tensor& x);
    Tensor one_hot_encode(const Tensor& targets, int num_classes);
    float compute_cross_entropy(const Tensor& probs, const Tensor& targets_one_hot);
    Tensor compute_gradients(const Tensor& probs, const Tensor& targets_one_hot);
};

} // namespace megatron