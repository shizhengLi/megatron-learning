#pragma once

#include "../tensor/tensor.h"
#include "../layers/layer.h"
#include "../optimizers/optimizer.h"
#include "../loss/loss.h"
#include <vector>
#include <memory>
#include <functional>

namespace megatron {

class Trainer {
public:
    Trainer(std::vector<std::shared_ptr<Layer>> layers, 
            std::shared_ptr<Optimizer> optimizer, 
            std::shared_ptr<Loss> loss_fn);
    
    // Training methods
    float train_step(const Tensor& inputs, const Tensor& targets);
    std::vector<float> train_epoch(const std::vector<Tensor>& inputs_batch, 
                                  const std::vector<Tensor>& targets_batch);
    
    // Evaluation methods
    float evaluate(const std::vector<Tensor>& inputs_batch, 
                  const std::vector<Tensor>& targets_batch);
    float evaluate_step(const Tensor& inputs, const Tensor& targets);
    
    // Utility methods
    void set_learning_rate(float lr);
    void zero_grad();
    void save_checkpoint(const std::string& filepath);
    void load_checkpoint(const std::string& filepath);
    
    // Training statistics
    float get_average_loss() const { return average_loss_; }
    int get_step_count() const { return step_count_; }
    
    // Training mode control
    void train();
    void eval();

private:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::shared_ptr<Optimizer> optimizer_;
    std::shared_ptr<Loss> loss_fn_;
    
    // Training state
    bool training_mode_;
    int step_count_;
    float average_loss_;
    std::vector<float> loss_history_;
    
    // Helper methods
    Tensor forward_pass(const Tensor& input);
    Tensor backward_pass(const Tensor& grad_loss);
    void update_parameters();
    std::vector<Tensor> get_all_parameters();
    std::vector<Tensor> get_all_gradients();
};

} // namespace megatron