#pragma once

#include "core/tensor/tensor.h"
#include "core/layers/layer.h"
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <fstream>

namespace megatron {

// Performance profiler for measuring execution time
class Profiler {
public:
    Profiler(const std::string& name);
    ~Profiler();
    
    // Start timing
    void start();
    
    // Stop timing and record elapsed time
    void stop();
    
    // Get elapsed time in microseconds
    double get_elapsed_time() const;
    
    // Get elapsed time in milliseconds
    double get_elapsed_time_ms() const;
    
    // Get profiler name
    const std::string& get_name() const { return name_; }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_;
    double elapsed_time_;
};

// Memory usage tracker
class MemoryTracker {
public:
    MemoryTracker();
    
    // Get current memory usage in bytes
    size_t get_current_memory_usage() const;
    
    // Get peak memory usage in bytes
    size_t get_peak_memory_usage() const;
    
    // Reset memory tracking
    void reset();
    
    // Log memory usage
    void log_memory_usage(const std::string& context) const;

private:
    size_t initial_memory_;
    size_t peak_memory_;
};

// Performance benchmark suite
class BenchmarkSuite {
public:
    BenchmarkSuite(const std::string& name);
    
    // Add a benchmark test
    void add_benchmark(const std::string& name, std::function<void()> benchmark_func);
    
    // Run all benchmarks
    void run_benchmarks(int num_runs = 10);
    
    // Get benchmark results
    std::map<std::string, std::vector<double>> get_results() const;
    
    // Generate benchmark report
    std::string generate_report() const;
    
    // Save results to file
    void save_results(const std::string& filepath) const;

private:
    std::string name_;
    std::map<std::string, std::function<void()>> benchmarks_;
    std::map<std::string, std::vector<double>> results_;
    
    void run_single_benchmark(const std::string& name, std::function<void()> func, int num_runs);
};

// Model performance analyzer
class ModelPerformanceAnalyzer {
public:
    ModelPerformanceAnalyzer();
    
    // Analyze model forward pass performance
    void analyze_forward_pass(std::shared_ptr<Layer> model, const Tensor& input, int num_runs = 100);
    
    // Analyze model backward pass performance
    void analyze_backward_pass(std::shared_ptr<Layer> model, const Tensor& grad_output, int num_runs = 100);
    
    // Analyze full training step performance
    void analyze_training_step(std::shared_ptr<Layer> model, const Tensor& input, 
                              const Tensor& target, int num_runs = 50);
    
    // Get performance metrics
    std::map<std::string, double> get_performance_metrics() const;
    
    // Generate performance report
    std::string generate_performance_report() const;
    
    // Calculate model FLOPs (simplified estimation)
    double estimate_flops(std::shared_ptr<Layer> model) const;
    
    // Calculate model parameter count
    size_t count_parameters(std::shared_ptr<Layer> model) const;

private:
    std::map<std::string, double> performance_metrics_;
    MemoryTracker memory_tracker_;
    
    void analyze_layer_performance(std::shared_ptr<Layer> layer, const std::string& layer_name);
};

// Optimization utilities
class OptimizationUtils {
public:
    // Memory optimization suggestions
    static std::vector<std::string> get_memory_optimization_suggestions(std::shared_ptr<Layer> model);
    
    // Performance optimization suggestions
    static std::vector<std::string> get_performance_optimization_suggestions(std::shared_ptr<Layer> model);
    
    // Model compression suggestions
    static std::vector<std::string> get_compression_suggestions(std::shared_ptr<Layer> model);
    
    // Batch size optimization
    static int find_optimal_batch_size(std::shared_ptr<Layer> model, 
                                      const Tensor& sample_input, 
                                      int max_batch_size = 128);
    
    // Learning rate scheduling suggestions
    static std::vector<float> suggest_learning_rates(float base_lr, int num_suggestions = 5);
};

// Real-time performance monitor
class PerformanceMonitor {
public:
    PerformanceMonitor();
    
    // Start monitoring
    void start_monitoring();
    
    // Stop monitoring
    void stop_monitoring();
    
    // Record a metric
    void record_metric(const std::string& name, double value);
    
    // Get average metric value
    double get_average_metric(const std::string& name) const;
    
    // Get metric statistics
    std::map<std::string, double> get_metric_stats(const std::string& name) const;
    
    // Generate real-time report
    std::string generate_realtime_report() const;
    
    // Check for performance warnings
    std::vector<std::string> get_performance_warnings() const;

private:
    bool is_monitoring_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::map<std::string, std::vector<double>> metric_history_;
    
    void update_metric_history(const std::string& name, double value);
};

// System information utilities
class SystemInfo {
public:
    // Get CPU information
    static std::string get_cpu_info();
    
    // Get memory information
    static std::string get_memory_info();
    
    // Get system load
    static float get_system_load();
    
    // Get available threads
    static int get_available_threads();
    
    // Get cache sizes
    static std::vector<size_t> get_cache_sizes();
    
    // Generate system report
    static std::string generate_system_report();
};

// Benchmark utilities
namespace benchmark_utils {
    // Generate synthetic data for benchmarking
    Tensor generate_synthetic_data(const std::vector<int>& shape, float mean = 0.0f, float std = 1.0f);
    
    // Warm up model (for stable benchmarking)
    void warm_up_model(std::shared_ptr<Layer> model, const Tensor& sample_input, int num_warmup_runs = 10);
    
    // Measure throughput (samples per second)
    double measure_throughput(std::shared_ptr<Layer> model, const Tensor& input, int duration_seconds = 10);
    
    // Measure latency distribution
    std::vector<double> measure_latency_distribution(std::shared_ptr<Layer> model, 
                                                   const Tensor& input, int num_samples = 100);
    
    // Calculate statistics from samples
    std::map<std::string, double> calculate_statistics(const std::vector<double>& samples);
    
    // Compare two models
    std::string compare_models(std::shared_ptr<Layer> model1, std::shared_ptr<Layer> model2, 
                             const Tensor& input, int num_runs = 100);
}

} // namespace megatron