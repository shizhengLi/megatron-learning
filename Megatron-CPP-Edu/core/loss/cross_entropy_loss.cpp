#include "cross_entropy_loss.h"
#include <cmath>
#include <algorithm>

namespace megatron {

CrossEntropyLoss::CrossEntropyLoss() = default;

float CrossEntropyLoss::compute(const Tensor& predictions, const Tensor& targets) {
    // predictions shape: [batch_size, num_classes]
    // targets shape: [batch_size] (class indices)
    
    int batch_size = predictions.shape()[0];
    int num_classes = predictions.shape()[1];
    
    // Compute softmax probabilities
    last_probabilities_ = softmax(predictions);
    
    // One-hot encode targets
    last_targets_one_hot_ = one_hot_encode(targets, num_classes);
    
    // Compute cross entropy loss
    last_loss_ = compute_cross_entropy(last_probabilities_, last_targets_one_hot_);
    
    return last_loss_;
}

Tensor CrossEntropyLoss::backward(const Tensor& predictions, const Tensor& targets) {
    // The gradient of cross entropy loss with softmax is: (probs - targets)
    // This is computed efficiently using the cached values from forward pass
    
    if (last_probabilities_.size() == 0) {
        // Compute forward pass if not already done
        compute(predictions, targets);
    }
    
    return compute_gradients(last_probabilities_, last_targets_one_hot_);
}

Tensor CrossEntropyLoss::softmax(const Tensor& x) {
    // x shape: [batch_size, num_classes]
    int batch_size = x.shape()[0];
    int num_classes = x.shape()[1];
    
    Tensor probs(x.shape());
    
    for (int i = 0; i < batch_size; ++i) {
        // Find max for numerical stability
        float max_val = x[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            max_val = std::max(max_val, x[i * num_classes + j]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            float exp_val = std::exp(x[i * num_classes + j] - max_val);
            probs[i * num_classes + j] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (int j = 0; j < num_classes; ++j) {
            probs[i * num_classes + j] /= sum_exp;
        }
    }
    
    return probs;
}

Tensor CrossEntropyLoss::one_hot_encode(const Tensor& targets, int num_classes) {
    // targets shape: [batch_size] (class indices)
    // output shape: [batch_size, num_classes]
    
    int batch_size = targets.shape()[0];
    Tensor one_hot({batch_size, num_classes});
    one_hot.zeros();
    
    for (int i = 0; i < batch_size; ++i) {
        int class_idx = static_cast<int>(targets[i]);
        if (class_idx >= 0 && class_idx < num_classes) {
            one_hot[i * num_classes + class_idx] = 1.0f;
        }
    }
    
    return one_hot;
}

float CrossEntropyLoss::compute_cross_entropy(const Tensor& probs, const Tensor& targets_one_hot) {
    // probs shape: [batch_size, num_classes]
    // targets_one_hot shape: [batch_size, num_classes]
    
    int batch_size = probs.shape()[0];
    int num_classes = probs.shape()[1];
    
    float total_loss = 0.0f;
    
    for (int i = 0; i < batch_size; ++i) {
        float sample_loss = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            if (targets_one_hot[i * num_classes + j] > 0.0f) {
                // Cross entropy: -target * log(prob)
                float prob = std::max(probs[i * num_classes + j], 1e-15f); // Avoid log(0)
                sample_loss -= std::log(prob);
            }
        }
        total_loss += sample_loss;
    }
    
    return total_loss / batch_size;  // Return average loss
}

Tensor CrossEntropyLoss::compute_gradients(const Tensor& probs, const Tensor& targets_one_hot) {
    // Gradient of cross entropy with softmax: (probs - targets) / batch_size
    // This is a beautiful mathematical property that makes implementation simple!
    
    int batch_size = probs.shape()[0];
    Tensor grad = probs - targets_one_hot;
    
    // Average gradients across batch
    for (int i = 0; i < grad.size(); ++i) {
        grad[i] /= batch_size;
    }
    
    return grad;
}

} // namespace megatron