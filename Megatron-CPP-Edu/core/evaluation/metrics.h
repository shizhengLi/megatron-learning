#pragma once

#include "core/tensor/tensor.h"
#include "core/layers/layer.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <cmath>

namespace megatron {

// Base metrics class
class Metrics {
public:
    virtual ~Metrics() = default;
    
    // Update metrics with predictions and targets
    virtual void update(const Tensor& predictions, const Tensor& targets) = 0;
    
    // Get current metric value
    virtual float get_value() const = 0;
    
    // Reset metrics
    virtual void reset() = 0;
    
    // Get metric name
    virtual std::string get_name() const = 0;
    
    // Get formatted string representation
    virtual std::string to_string() const = 0;
};

// Accuracy metric for classification
class Accuracy : public Metrics {
public:
    Accuracy();
    
    void update(const Tensor& predictions, const Tensor& targets) override;
    float get_value() const override;
    void reset() override;
    std::string get_name() const override;
    std::string to_string() const override;

private:
    int correct_;
    int total_;
};

// Precision, Recall, F1 metrics for classification
class PrecisionRecallF1 : public Metrics {
public:
    PrecisionRecallF1(int num_classes, bool macro_averaged = true);
    
    void update(const Tensor& predictions, const Tensor& targets) override;
    float get_value() const override;
    void reset() override;
    std::string get_name() const override;
    std::string to_string() const override;
    
    // Get individual metrics
    float get_precision() const;
    float get_recall() const;
    float get_f1() const;

private:
    int num_classes_;
    bool macro_averaged_;
    std::vector<int> true_positives_;
    std::vector<int> false_positives_;
    std::vector<int> false_negatives_;
    
    void update_confusion_matrix(const Tensor& predictions, const Tensor& targets);
};

// Confusion Matrix
class ConfusionMatrix {
public:
    ConfusionMatrix(int num_classes);
    
    void update(const Tensor& predictions, const Tensor& targets);
    void reset();
    
    // Get confusion matrix as tensor
    Tensor get_matrix() const;
    
    // Get specific cell value
    int get_value(int true_class, int pred_class) const;
    
    // Print confusion matrix
    void print() const;
    
    // Get per-class metrics
    std::vector<float> get_per_class_accuracy() const;
    std::vector<float> get_per_class_precision() const;
    std::vector<float> get_per_class_recall() const;

private:
    int num_classes_;
    Tensor matrix_;
};

// Perplexity metric for language models
class Perplexity : public Metrics {
public:
    Perplexity();
    
    void update(const Tensor& predictions, const Tensor& targets) override;
    float get_value() const override;
    void reset() override;
    std::string get_name() const override;
    std::string to_string() const override;

private:
    float total_loss_;
    int total_tokens_;
};

// Loss metrics
class LossMetrics : public Metrics {
public:
    LossMetrics();
    
    void update(const Tensor& predictions, const Tensor& targets) override;
    void update(const Tensor& predictions, const Tensor& targets, float loss);
    void update_with_loss(float loss);
    float get_value() const override;
    void reset() override;
    std::string get_name() const override;
    std::string to_string() const override;

private:
    float total_loss_;
    int count_;
};

// Evaluation pipeline
class EvaluationPipeline {
public:
    EvaluationPipeline();
    
    // Add metric to pipeline
    void add_metric(std::shared_ptr<Metrics> metric);
    
    // Update all metrics
    void update(const Tensor& predictions, const Tensor& targets, float loss = 0.0f);
    
    // Get all metric values
    std::map<std::string, float> get_all_values() const;
    
    // Get formatted report
    std::string generate_report() const;
    
    // Reset all metrics
    void reset();
    
    // Save results to file
    void save_results(const std::string& filepath) const;

private:
    std::vector<std::shared_ptr<Metrics>> metrics_;
    std::shared_ptr<LossMetrics> loss_metrics_;
};

// Model evaluator
class ModelEvaluator {
public:
    ModelEvaluator();
    
    // Evaluate model on dataset
    std::map<std::string, float> evaluate(std::shared_ptr<Layer> model, 
                                          const std::vector<Tensor>& test_inputs,
                                          const std::vector<Tensor>& test_targets,
                                          std::shared_ptr<Metrics> metric = nullptr);
    
    // Evaluate model with loss function
    std::map<std::string, float> evaluate_with_loss(std::shared_ptr<Layer> model,
                                                     const std::vector<Tensor>& test_inputs,
                                                     const std::vector<Tensor>& test_targets,
                                                     std::shared_ptr<Metrics> loss_fn);
    
    // Run complete evaluation suite
    std::map<std::string, float> comprehensive_evaluate(std::shared_ptr<Layer> model,
                                                         const std::vector<Tensor>& test_inputs,
                                                         const std::vector<Tensor>& test_targets,
                                                         int num_classes);
    
    // Generate classification report
    std::string classification_report(std::shared_ptr<Layer> model,
                                      const std::vector<Tensor>& test_inputs,
                                      const std::vector<Tensor>& test_targets,
                                      int num_classes);

private:
    std::shared_ptr<EvaluationPipeline> eval_pipeline_;
    
    Tensor get_predictions(std::shared_ptr<Layer> model, const Tensor& inputs);
    std::vector<int> get_predicted_classes(const Tensor& predictions);
    std::vector<int> get_true_classes(const Tensor& targets);
};

// Utility functions for evaluation
namespace evaluation_utils {
    // Calculate ROC AUC (simplified implementation)
    float calculate_roc_auc(const Tensor& predictions, const Tensor& targets);
    
    // Calculate top-k accuracy
    float top_k_accuracy(const Tensor& predictions, const Tensor& targets, int k);
    
    // Calculate mean squared error
    float mean_squared_error(const Tensor& predictions, const Tensor& targets);
    
    // Calculate mean absolute error
    float mean_absolute_error(const Tensor& predictions, const Tensor& targets);
    
    // Calculate R-squared
    float r_squared(const Tensor& predictions, const Tensor& targets);
    
    // Calculate BLEU score (simplified)
    float calculate_bleu(const std::vector<std::string>& references,
                         const std::vector<std::string>& hypotheses,
                         int n_gram = 4);
}

} // namespace megatron