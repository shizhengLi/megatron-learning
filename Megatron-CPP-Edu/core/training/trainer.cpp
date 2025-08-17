#include "trainer.h"
#include <algorithm>
#include <numeric>

namespace megatron {

Trainer::Trainer(std::vector<std::shared_ptr<Layer>> layers, 
                std::shared_ptr<Optimizer> optimizer, 
                std::shared_ptr<Loss> loss_fn)
    : layers_(layers), optimizer_(optimizer), loss_fn_(loss_fn),
      training_mode_(true), step_count_(0), average_loss_(0.0f) {
    
    // Initialize optimizer with model parameters
    auto parameters = get_all_parameters();
    optimizer_->set_parameters(parameters);
}

float Trainer::train_step(const Tensor& inputs, const Tensor& targets) {
    if (!training_mode_) {
        train();  // Ensure we're in training mode
    }
    
    // Forward pass
    Tensor predictions = forward_pass(inputs);
    
    // Compute loss
    float loss = loss_fn_->compute(predictions, targets);
    
    // Backward pass
    Tensor grad_loss = loss_fn_->backward(predictions, targets);
    backward_pass(grad_loss);
    
    // Update parameters
    update_parameters();
    
    // Update statistics
    step_count_++;
    loss_history_.push_back(loss);
    
    // Update average loss (exponential moving average)
    if (step_count_ == 1) {
        average_loss_ = loss;
    } else {
        average_loss_ = 0.9f * average_loss_ + 0.1f * loss;
    }
    
    return loss;
}

std::vector<float> Trainer::train_epoch(const std::vector<Tensor>& inputs_batch, 
                                        const std::vector<Tensor>& targets_batch) {
    if (inputs_batch.size() != targets_batch.size()) {
        throw std::invalid_argument("Number of input and target batches must match");
    }
    
    std::vector<float> batch_losses;
    batch_losses.reserve(inputs_batch.size());
    
    for (size_t i = 0; i < inputs_batch.size(); ++i) {
        float loss = train_step(inputs_batch[i], targets_batch[i]);
        batch_losses.push_back(loss);
    }
    
    return batch_losses;
}

float Trainer::evaluate(const std::vector<Tensor>& inputs_batch, 
                       const std::vector<Tensor>& targets_batch) {
    if (inputs_batch.size() != targets_batch.size()) {
        throw std::invalid_argument("Number of input and target batches must match");
    }
    
    float total_loss = 0.0f;
    int num_samples = inputs_batch.size();
    
    for (size_t i = 0; i < inputs_batch.size(); ++i) {
        float loss = evaluate_step(inputs_batch[i], targets_batch[i]);
        total_loss += loss;
    }
    
    return total_loss / num_samples;
}

float Trainer::evaluate_step(const Tensor& inputs, const Tensor& targets) {
    if (training_mode_) {
        eval();  // Ensure we're in evaluation mode
    }
    
    // Forward pass only (no gradient computation)
    Tensor predictions = forward_pass(inputs);
    
    // Compute loss
    return loss_fn_->compute(predictions, targets);
}

void Trainer::set_learning_rate(float lr) {
    optimizer_->set_learning_rate(lr);
}

void Trainer::zero_grad() {
    auto gradients = get_all_gradients();
    optimizer_->zero_grad(gradients);
}

void Trainer::save_checkpoint(const std::string& filepath) {
    // Simple checkpoint implementation - save parameters to file
    // In a real implementation, this would serialize the model state
    auto parameters = get_all_parameters();
    
    // For now, we'll just create a placeholder implementation
    // In practice, you'd use a proper serialization library
    FILE* file = fopen(filepath.c_str(), "wb");
    if (file) {
        // Write number of parameter tensors
        int num_params = parameters.size();
        fwrite(&num_params, sizeof(int), 1, file);
        
        // Write each parameter tensor
        for (const auto& param : parameters) {
            // Write shape
            int dims = param.dim();
            fwrite(&dims, sizeof(int), 1, file);
            for (int i = 0; i < dims; ++i) {
                int dim_size = param.shape()[i];
                fwrite(&dim_size, sizeof(int), 1, file);
            }
            
            // Write data
            fwrite(param.data(), sizeof(float), param.size(), file);
        }
        
        fclose(file);
    }
}

void Trainer::load_checkpoint(const std::string& filepath) {
    // Simple checkpoint implementation - load parameters from file
    FILE* file = fopen(filepath.c_str(), "rb");
    if (file) {
        // Read number of parameter tensors
        int num_params;
        fread(&num_params, sizeof(int), 1, file);
        
        auto parameters = get_all_parameters();
        if (num_params != parameters.size()) {
            fclose(file);
            throw std::runtime_error("Checkpoint file doesn't match model architecture");
        }
        
        // Read each parameter tensor
        for (auto& param : parameters) {
            // Read shape
            int dims;
            fread(&dims, sizeof(int), 1, file);
            
            std::vector<int> shape(dims);
            for (int i = 0; i < dims; ++i) {
                fread(&shape[i], sizeof(int), 1, file);
            }
            
            // Verify shape matches
            if (shape != param.shape()) {
                fclose(file);
                throw std::runtime_error("Parameter shape mismatch in checkpoint");
            }
            
            // Read data
            fread(param.data(), sizeof(float), param.size(), file);
        }
        
        fclose(file);
    }
}

void Trainer::train() {
    training_mode_ = true;
    for (auto& layer : layers_) {
        layer->train();
    }
}

void Trainer::eval() {
    training_mode_ = false;
    for (auto& layer : layers_) {
        layer->eval();
    }
}

Tensor Trainer::forward_pass(const Tensor& input) {
    Tensor current = input;
    
    for (auto& layer : layers_) {
        current = layer->forward(current);
    }
    
    return current;
}

Tensor Trainer::backward_pass(const Tensor& grad_loss) {
    Tensor current_grad = grad_loss;
    
    // Backward pass through layers in reverse order
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }
    
    return current_grad;
}

void Trainer::update_parameters() {
    auto parameters = get_all_parameters();
    auto gradients = get_all_gradients();
    
    optimizer_->step(parameters, gradients);
}

std::vector<Tensor> Trainer::get_all_parameters() {
    std::vector<Tensor> all_parameters;
    
    for (auto& layer : layers_) {
        auto layer_params = layer->parameters();
        all_parameters.insert(all_parameters.end(), layer_params.begin(), layer_params.end());
    }
    
    return all_parameters;
}

std::vector<Tensor> Trainer::get_all_gradients() {
    std::vector<Tensor> all_gradients;
    
    for (auto& layer : layers_) {
        auto layer_grads = layer->gradients();
        all_gradients.insert(all_gradients.end(), layer_grads.begin(), layer_grads.end());
    }
    
    return all_gradients;
}

} // namespace megatron