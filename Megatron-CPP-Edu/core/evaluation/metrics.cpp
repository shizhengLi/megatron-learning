#include "metrics.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace megatron {

// Helper functions for getting predicted and true classes
std::vector<int> get_predicted_classes(const Tensor& predictions) {
    std::vector<int> classes;
    const float* data = predictions.data();
    
    if (predictions.shape().size() == 2) {
        // 2D tensor: (batch_size, num_classes)
        size_t batch_size = predictions.shape()[0];
        size_t num_classes = predictions.shape()[1];
        
        for (size_t i = 0; i < batch_size; ++i) {
            float max_val = data[i * num_classes];
            int max_idx = 0;
            for (size_t j = 1; j < num_classes; ++j) {
                if (data[i * num_classes + j] > max_val) {
                    max_val = data[i * num_classes + j];
                    max_idx = static_cast<int>(j);
                }
            }
            classes.push_back(max_idx);
        }
    } else {
        // Assume 1D tensor with class indices
        size_t size = 1;
        for (int dim : predictions.shape()) {
            size *= dim;
        }
        for (size_t i = 0; i < size; ++i) {
            classes.push_back(static_cast<int>(data[i]));
        }
    }
    
    return classes;
}

std::vector<int> get_true_classes(const Tensor& targets) {
    std::vector<int> classes;
    const float* data = targets.data();
    size_t size = 1;
    for (int dim : targets.shape()) {
        size *= dim;
    }
    
    for (size_t i = 0; i < size; ++i) {
        classes.push_back(static_cast<int>(data[i]));
    }
    
    return classes;
}

// Accuracy implementation
Accuracy::Accuracy() : correct_(0), total_(0) {}

void Accuracy::update(const Tensor& predictions, const Tensor& targets) {
    // For classification, predictions should be logits, targets should be class indices
    auto pred_classes = get_predicted_classes(predictions);
    auto true_classes = get_true_classes(targets);
    
    for (size_t i = 0; i < pred_classes.size(); ++i) {
        if (pred_classes[i] == true_classes[i]) {
            correct_++;
        }
        total_++;
    }
}

float Accuracy::get_value() const {
    return total_ > 0 ? static_cast<float>(correct_) / total_ : 0.0f;
}

void Accuracy::reset() {
    correct_ = 0;
    total_ = 0;
}

std::string Accuracy::get_name() const {
    return "accuracy";
}

std::string Accuracy::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << get_value();
    return oss.str();
}

// PrecisionRecallF1 implementation
PrecisionRecallF1::PrecisionRecallF1(int num_classes, bool macro_averaged)
    : num_classes_(num_classes), macro_averaged_(macro_averaged),
      true_positives_(num_classes, 0),
      false_positives_(num_classes, 0),
      false_negatives_(num_classes, 0) {}

void PrecisionRecallF1::update(const Tensor& predictions, const Tensor& targets) {
    update_confusion_matrix(predictions, targets);
}

void PrecisionRecallF1::update_confusion_matrix(const Tensor& predictions, const Tensor& targets) {
    auto pred_classes = get_predicted_classes(predictions);
    auto true_classes = get_true_classes(targets);
    
    for (size_t i = 0; i < pred_classes.size(); ++i) {
        int pred = pred_classes[i];
        int true_label = true_classes[i];
        
        if (pred == true_label) {
            true_positives_[true_label]++;
        } else {
            false_positives_[pred]++;
            false_negatives_[true_label]++;
        }
    }
}

float PrecisionRecallF1::get_value() const {
    return get_f1();  // Default to F1 score
}

float PrecisionRecallF1::get_precision() const {
    if (macro_averaged_) {
        float total_precision = 0.0f;
        int valid_classes = 0;
        
        for (int i = 0; i < num_classes_; ++i) {
            float denominator = true_positives_[i] + false_positives_[i];
            if (denominator > 0) {
                total_precision += static_cast<float>(true_positives_[i]) / denominator;
                valid_classes++;
            }
        }
        
        return valid_classes > 0 ? total_precision / valid_classes : 0.0f;
    } else {
        // Micro-averaged precision
        float total_tp = 0.0f, total_fp = 0.0f;
        for (int i = 0; i < num_classes_; ++i) {
            total_tp += true_positives_[i];
            total_fp += false_positives_[i];
        }
        
        return (total_tp + total_fp) > 0 ? total_tp / (total_tp + total_fp) : 0.0f;
    }
}

float PrecisionRecallF1::get_recall() const {
    if (macro_averaged_) {
        float total_recall = 0.0f;
        int valid_classes = 0;
        
        for (int i = 0; i < num_classes_; ++i) {
            float denominator = true_positives_[i] + false_negatives_[i];
            if (denominator > 0) {
                total_recall += static_cast<float>(true_positives_[i]) / denominator;
                valid_classes++;
            }
        }
        
        return valid_classes > 0 ? total_recall / valid_classes : 0.0f;
    } else {
        // Micro-averaged recall
        float total_tp = 0.0f, total_fn = 0.0f;
        for (int i = 0; i < num_classes_; ++i) {
            total_tp += true_positives_[i];
            total_fn += false_negatives_[i];
        }
        
        return (total_tp + total_fn) > 0 ? total_tp / (total_tp + total_fn) : 0.0f;
    }
}

float PrecisionRecallF1::get_f1() const {
    float precision = get_precision();
    float recall = get_recall();
    
    return (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0f;
}

void PrecisionRecallF1::reset() {
    std::fill(true_positives_.begin(), true_positives_.end(), 0);
    std::fill(false_positives_.begin(), false_positives_.end(), 0);
    std::fill(false_negatives_.begin(), false_negatives_.end(), 0);
}

std::string PrecisionRecallF1::get_name() const {
    return macro_averaged_ ? "macro_f1" : "micro_f1";
}

std::string PrecisionRecallF1::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) 
        << "P=" << get_precision() << ", R=" << get_recall() << ", F1=" << get_f1();
    return oss.str();
}

// ConfusionMatrix implementation
ConfusionMatrix::ConfusionMatrix(int num_classes) : num_classes_(num_classes) {
    matrix_ = Tensor({num_classes, num_classes});
    matrix_.zeros();
}

void ConfusionMatrix::update(const Tensor& predictions, const Tensor& targets) {
    auto pred_classes = get_predicted_classes(predictions);
    auto true_classes = get_true_classes(targets);
    
    for (size_t i = 0; i < pred_classes.size(); ++i) {
        int pred = pred_classes[i];
        int true_label = true_classes[i];
        
        if (pred >= 0 && pred < num_classes_ && true_label >= 0 && true_label < num_classes_) {
            matrix_[true_label * num_classes_ + pred]++;
        }
    }
}

void ConfusionMatrix::reset() {
    matrix_.zeros();
}

Tensor ConfusionMatrix::get_matrix() const {
    return matrix_;
}

int ConfusionMatrix::get_value(int true_class, int pred_class) const {
    return matrix_[true_class * num_classes_ + pred_class];
}

void ConfusionMatrix::print() const {
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << "    ";
    for (int i = 0; i < num_classes_; ++i) {
        std::cout << std::setw(4) << i;
    }
    std::cout << std::endl;
    
    for (int i = 0; i < num_classes_; ++i) {
        std::cout << std::setw(4) << i;
        for (int j = 0; j < num_classes_; ++j) {
            std::cout << std::setw(4) << matrix_[i * num_classes_ + j];
        }
        std::cout << std::endl;
    }
}

std::vector<float> ConfusionMatrix::get_per_class_accuracy() const {
    std::vector<float> accuracies(num_classes_);
    
    for (int i = 0; i < num_classes_; ++i) {
        float row_sum = 0.0f;
        float diagonal = matrix_[i * num_classes_ + i];
        
        for (int j = 0; j < num_classes_; ++j) {
            row_sum += matrix_[i * num_classes_ + j];
        }
        
        accuracies[i] = row_sum > 0 ? diagonal / row_sum : 0.0f;
    }
    
    return accuracies;
}

std::vector<float> ConfusionMatrix::get_per_class_precision() const {
    std::vector<float> precisions(num_classes_);
    
    for (int i = 0; i < num_classes_; ++i) {
        float col_sum = 0.0f;
        float diagonal = matrix_[i * num_classes_ + i];
        
        for (int j = 0; j < num_classes_; ++j) {
            col_sum += matrix_[j * num_classes_ + i];
        }
        
        precisions[i] = col_sum > 0 ? diagonal / col_sum : 0.0f;
    }
    
    return precisions;
}

std::vector<float> ConfusionMatrix::get_per_class_recall() const {
    std::vector<float> recalls(num_classes_);
    
    for (int i = 0; i < num_classes_; ++i) {
        float row_sum = 0.0f;
        float diagonal = matrix_[i * num_classes_ + i];
        
        for (int j = 0; j < num_classes_; ++j) {
            row_sum += matrix_[i * num_classes_ + j];
        }
        
        recalls[i] = row_sum > 0 ? diagonal / row_sum : 0.0f;
    }
    
    return recalls;
}

// Perplexity implementation
Perplexity::Perplexity() : total_loss_(0.0f), total_tokens_(0) {}

void Perplexity::update(const Tensor& predictions, const Tensor& targets) {
    // Simplified perplexity calculation
    // In practice, this would use the actual loss from cross-entropy
    float batch_loss = 0.0f;
    int batch_size = predictions.shape()[0];
    
    for (int i = 0; i < batch_size; ++i) {
        // Calculate cross-entropy loss for this sample
        float max_logit = predictions[i * predictions.shape()[1]];
        float sum_exp = 0.0f;
        
        for (int j = 0; j < predictions.shape()[1]; ++j) {
            sum_exp += std::exp(predictions[i * predictions.shape()[1] + j] - max_logit);
        }
        
        int target_class = static_cast<int>(targets[i]);
        float log_prob = predictions[i * predictions.shape()[1] + target_class] - max_logit - std::log(sum_exp);
        batch_loss += -log_prob;
    }
    
    total_loss_ += batch_loss;
    total_tokens_ += batch_size;
}

float Perplexity::get_value() const {
    float avg_loss = total_tokens_ > 0 ? total_loss_ / total_tokens_ : 0.0f;
    return std::exp(avg_loss);
}

void Perplexity::reset() {
    total_loss_ = 0.0f;
    total_tokens_ = 0;
}

std::string Perplexity::get_name() const {
    return "perplexity";
}

std::string Perplexity::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << get_value();
    return oss.str();
}

// LossMetrics implementation
LossMetrics::LossMetrics() : total_loss_(0.0f), count_(0) {}

void LossMetrics::update(const Tensor& predictions, const Tensor& targets, float loss) {
    update_with_loss(loss);
}

void LossMetrics::update(const Tensor& predictions, const Tensor& targets) {
    // For loss metrics, we need the actual loss value
    // This implementation doesn't calculate loss from predictions and targets
    // Use the other update method with explicit loss value
    // For now, do nothing - loss should be set via update_with_loss
}

void LossMetrics::update_with_loss(float loss) {
    total_loss_ += loss;
    count_++;
}

float LossMetrics::get_value() const {
    return count_ > 0 ? total_loss_ / count_ : 0.0f;
}

void LossMetrics::reset() {
    total_loss_ = 0.0f;
    count_ = 0;
}

std::string LossMetrics::get_name() const {
    return "loss";
}

std::string LossMetrics::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << get_value();
    return oss.str();
}

// EvaluationPipeline implementation
EvaluationPipeline::EvaluationPipeline() {
    loss_metrics_ = std::make_shared<LossMetrics>();
    metrics_.push_back(loss_metrics_);
}

void EvaluationPipeline::add_metric(std::shared_ptr<Metrics> metric) {
    metrics_.push_back(metric);
}

void EvaluationPipeline::update(const Tensor& predictions, const Tensor& targets, float loss) {
    loss_metrics_->update_with_loss(loss);
    
    for (auto& metric : metrics_) {
        if (metric != loss_metrics_) {
            metric->update(predictions, targets);
        }
    }
}

std::map<std::string, float> EvaluationPipeline::get_all_values() const {
    std::map<std::string, float> values;
    
    for (const auto& metric : metrics_) {
        values[metric->get_name()] = metric->get_value();
    }
    
    return values;
}

std::string EvaluationPipeline::generate_report() const {
    std::ostringstream oss;
    oss << "Evaluation Report:" << std::endl;
    oss << "==================" << std::endl;
    
    for (const auto& metric : metrics_) {
        oss << metric->get_name() << ": " << metric->to_string() << std::endl;
    }
    
    return oss.str();
}

void EvaluationPipeline::reset() {
    for (auto& metric : metrics_) {
        metric->reset();
    }
}

void EvaluationPipeline::save_results(const std::string& filepath) const {
    std::ofstream file(filepath);
    file << generate_report();
}

// ModelEvaluator implementation
ModelEvaluator::ModelEvaluator() {
    eval_pipeline_ = std::make_shared<EvaluationPipeline>();
}

std::map<std::string, float> ModelEvaluator::evaluate(std::shared_ptr<Layer> model,
                                                      const std::vector<Tensor>& test_inputs,
                                                      const std::vector<Tensor>& test_targets,
                                                      std::shared_ptr<Metrics> metric) {
    if (metric) {
        eval_pipeline_->add_metric(metric);
    }
    
    eval_pipeline_->reset();
    
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        Tensor predictions = get_predictions(model, test_inputs[i]);
        eval_pipeline_->update(predictions, test_targets[i], 0.0f);
    }
    
    return eval_pipeline_->get_all_values();
}

std::map<std::string, float> ModelEvaluator::evaluate_with_loss(std::shared_ptr<Layer> model,
                                                                const std::vector<Tensor>& test_inputs,
                                                                const std::vector<Tensor>& test_targets,
                                                                std::shared_ptr<Metrics> loss_fn) {
    eval_pipeline_->reset();
    
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        Tensor predictions = get_predictions(model, test_inputs[i]);
        float loss = loss_fn->get_value();  // This would need to be calculated properly
        eval_pipeline_->update(predictions, test_targets[i], loss);
    }
    
    return eval_pipeline_->get_all_values();
}

std::map<std::string, float> ModelEvaluator::comprehensive_evaluate(std::shared_ptr<Layer> model,
                                                                     const std::vector<Tensor>& test_inputs,
                                                                     const std::vector<Tensor>& test_targets,
                                                                     int num_classes) {
    eval_pipeline_->reset();
    
    // Add comprehensive metrics
    eval_pipeline_->add_metric(std::make_shared<Accuracy>());
    eval_pipeline_->add_metric(std::make_shared<PrecisionRecallF1>(num_classes, true));
    eval_pipeline_->add_metric(std::make_shared<PrecisionRecallF1>(num_classes, false));
    
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        Tensor predictions = get_predictions(model, test_inputs[i]);
        eval_pipeline_->update(predictions, test_targets[i], 0.0f);
    }
    
    return eval_pipeline_->get_all_values();
}

std::string ModelEvaluator::classification_report(std::shared_ptr<Layer> model,
                                                  const std::vector<Tensor>& test_inputs,
                                                  const std::vector<Tensor>& test_targets,
                                                  int num_classes) {
    ConfusionMatrix cm(num_classes);
    
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        Tensor predictions = get_predictions(model, test_inputs[i]);
        cm.update(predictions, test_targets[i]);
    }
    
    std::ostringstream oss;
    oss << "Classification Report:" << std::endl;
    oss << "======================" << std::endl;
    
    auto accuracies = cm.get_per_class_accuracy();
    auto precisions = cm.get_per_class_precision();
    auto recalls = cm.get_per_class_recall();
    
    oss << "Class\tAccuracy\tPrecision\tRecall" << std::endl;
    for (int i = 0; i < num_classes; ++i) {
        oss << i << "\t" << std::fixed << std::setprecision(4) << accuracies[i]
            << "\t" << precisions[i] << "\t" << recalls[i] << std::endl;
    }
    
    oss << std::endl;
    cm.print();
    
    return oss.str();
}

Tensor ModelEvaluator::get_predictions(std::shared_ptr<Layer> model, const Tensor& inputs) {
    return model->forward(inputs);
}

std::vector<int> ModelEvaluator::get_predicted_classes(const Tensor& predictions) {
    std::vector<int> classes;
    
    if (predictions.dim() == 2) {
        // Classification: find argmax for each sample
        for (int i = 0; i < predictions.shape()[0]; ++i) {
            float max_val = predictions[i * predictions.shape()[1]];
            int max_idx = 0;
            
            for (int j = 1; j < predictions.shape()[1]; ++j) {
                if (predictions[i * predictions.shape()[1] + j] > max_val) {
                    max_val = predictions[i * predictions.shape()[1] + j];
                    max_idx = j;
                }
            }
            
            classes.push_back(max_idx);
        }
    }
    
    return classes;
}

std::vector<int> ModelEvaluator::get_true_classes(const Tensor& targets) {
    std::vector<int> classes;
    
    if (targets.dim() == 1) {
        for (int i = 0; i < targets.size(); ++i) {
            classes.push_back(static_cast<int>(targets[i]));
        }
    }
    
    return classes;
}

} // namespace megatron