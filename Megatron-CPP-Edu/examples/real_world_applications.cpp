#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>

#include "models/gpt/gpt_model.h"
#include "models/transformer/transformer_classifier.h"
#include "core/data/dataset.h"
#include "core/training/trainer.h"
#include "core/optimizers/adamw.h"
#include "core/loss/cross_entropy_loss.h"
#include "core/evaluation/metrics.h"

using namespace megatron;

// Example 1: Text Classification with Transformer
void run_text_classification_example() {
    std::cout << "=== Text Classification Example ===" << std::endl;
    
    // Create a simple synthetic dataset
    std::vector<std::string> texts = {
        "I love this movie it was great",
        "This film is terrible and boring",
        "Amazing performance by the actors",
        "Worst movie I have ever seen",
        "The plot was engaging and interesting",
        "I fell asleep during the film",
        "Excellent cinematography and direction",
        "Poor acting and weak storyline"
    };
    
    std::vector<int> labels = {1, 0, 1, 0, 1, 0, 1, 0};  // 1=positive, 0=negative
    
    // Create model
    int vocab_size = 1000;
    int max_seq_len = 32;
    int embed_dim = 128;
    int num_heads = 4;
    int num_layers = 2;
    int ff_dim = 512;
    int num_classes = 2;
    
    auto model = std::make_shared<TransformerClassifier>(
        vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim, num_classes,
        true, 0.1f, "sentiment_classifier");
    
    // Create optimizer and loss
    auto optimizer = std::make_shared<AdamW>(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    auto loss_fn = std::make_shared<CrossEntropyLoss>();
    
    // Create trainer
    std::vector<std::shared_ptr<Layer>> layers = {model};
    Trainer trainer(layers, optimizer, loss_fn);
    
    // Create training data (simplified)
    std::vector<Tensor> train_inputs, train_targets;
    
    // Generate synthetic training data
    for (int epoch = 0; epoch < 10; ++epoch) {
        Tensor inputs({2, max_seq_len});
        Tensor targets({2});
        
        // Fill with random token IDs (simplified)
        for (int i = 0; i < inputs.size(); ++i) {
            inputs[i] = rand() % vocab_size;
        }
        
        targets[0] = rand() % num_classes;
        targets[1] = rand() % num_classes;
        
        train_inputs.push_back(inputs);
        train_targets.push_back(targets);
        
        // Train on this batch
        float loss = trainer.train_step(inputs, targets);
        
        if (epoch % 2 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }
    
    std::cout << "Text classification training completed!" << std::endl;
}

// Example 2: Simple Language Model Training
void run_language_model_example() {
    std::cout << "\\n=== Language Model Example ===" << std::endl;
    
    // Create GPT model
    int vocab_size = 1000;
    int max_seq_len = 64;
    int embed_dim = 256;
    int num_heads = 8;
    int num_layers = 4;
    int ff_dim = 1024;
    
    auto model = std::make_shared<GPTModel>(
        vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim,
        true, 0.1f, "simple_gpt");
    
    // Create optimizer and loss
    auto optimizer = std::make_shared<AdamW>(0.0001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    auto loss_fn = std::make_shared<CrossEntropyLoss>();
    
    // Create trainer
    std::vector<std::shared_ptr<Layer>> layers = {model};
    Trainer trainer(layers, optimizer, loss_fn);
    
    // Training loop
    for (int epoch = 0; epoch < 5; ++epoch) {
        // Create synthetic training data
        Tensor inputs({2, max_seq_len});
        Tensor targets({2, max_seq_len});
        
        // Fill with random token IDs
        for (int i = 0; i < inputs.size(); ++i) {
            inputs[i] = rand() % vocab_size;
            targets[i] = rand() % vocab_size;
        }
        
        // Reshape targets for language modeling
        targets.reshape({2 * max_seq_len});
        
        float loss = trainer.train_step(inputs, targets);
        
        if (epoch % 1 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }
    
    std::cout << "Language model training completed!" << std::endl;
}

// Example 3: Model Evaluation
void run_model_evaluation_example() {
    std::cout << "\\n=== Model Evaluation Example ===" << std::endl;
    
    // Create a simple model for evaluation
    int vocab_size = 100;
    int max_seq_len = 16;
    int embed_dim = 64;
    int num_heads = 2;
    int num_layers = 1;
    int ff_dim = 256;
    int num_classes = 3;
    
    auto model = std::make_shared<TransformerClassifier>(
        vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim, num_classes,
        false, 0.0f, "eval_model");
    
    // Create evaluator
    ModelEvaluator evaluator;
    
    // Generate synthetic test data
    std::vector<Tensor> test_inputs, test_targets;
    
    for (int i = 0; i < 20; ++i) {
        Tensor inputs({4, max_seq_len});
        Tensor targets({4});
        
        for (int j = 0; j < inputs.size(); ++j) {
            inputs[j] = rand() % vocab_size;
        }
        
        for (int j = 0; j < targets.size(); ++j) {
            targets[j] = rand() % num_classes;
        }
        
        test_inputs.push_back(inputs);
        test_targets.push_back(targets);
    }
    
    // Run comprehensive evaluation
    auto results = evaluator.comprehensive_evaluate(model, test_inputs, test_targets, num_classes);
    
    std::cout << "Evaluation Results:" << std::endl;
    for (const auto& [metric, value] : results) {
        std::cout << "  " << metric << ": " << std::fixed << std::setprecision(4) << value << std::endl;
    }
    
    // Generate classification report
    std::string report = evaluator.classification_report(model, test_inputs, test_targets, num_classes);
    std::cout << "\\n" << report << std::endl;
}

// Example 4: Performance Benchmarking
void run_performance_benchmark() {
    std::cout << "\\n=== Performance Benchmark ===" << std::endl;
    
    // Test different model sizes
    std::vector<std::tuple<int, int, int>> model_configs = {
        {128, 4, 512},   // Small model
        {256, 8, 1024},  // Medium model
        {512, 8, 2048}   // Large model
    };
    
    for (const auto& [embed_dim, num_heads, ff_dim] : model_configs) {
        std::cout << "\\nBenchmarking model with embed_dim=" << embed_dim 
                  << ", num_heads=" << num_heads << ", ff_dim=" << ff_dim << std::endl;
        
        // Create model
        auto model = std::make_shared<TransformerClassifier>(
            1000, 64, embed_dim, num_heads, 2, ff_dim, 10, false, 0.0f, "benchmark_model");
        
        // Create test input
        Tensor inputs({8, 64});
        inputs.random_normal(0.0f, 1.0f);
        
        // Measure forward pass time
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 10; ++i) {
            Tensor output = model->forward(inputs);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Average forward pass time: " 
                  << duration.count() / 10.0 << " microseconds" << std::endl;
        
        // Measure backward pass time
        Tensor grad_output({8, 10});
        grad_output.random_normal(0.0f, 0.1f);
        
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 10; ++i) {
            Tensor grad_input = model->backward(grad_output);
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Average backward pass time: " 
                  << duration.count() / 10.0 << " microseconds" << std::endl;
    }
}

// Example 5: Real-time Inference
void run_realtime_inference_example() {
    std::cout << "\\n=== Real-time Inference Example ===" << std::endl;
    
    // Create a pre-trained model (simplified)
    auto model = std::make_shared<TransformerClassifier>(
        1000, 32, 128, 4, 2, 512, 5, false, 0.0f, "inference_model");
    
    // Create sample texts for classification
    std::vector<std::string> sample_texts = {
        "This product is amazing!",
        "I hate this service.",
        "It's okay, nothing special.",
        "Best purchase I've made!",
        "Terrible experience overall."
    };
    
    // Simple tokenizer simulation
    auto tokenize_text = [](const std::string& text, int max_len) -> Tensor {
        Tensor result({1, max_len});
        result.fill(0);  // Padding
        
        // Simple word-to-token mapping (simulation)
        std::vector<std::string> words = {"this", "product", "is", "amazing", "i", "hate", 
                                         "service", "it's", "okay", "nothing", "special", 
                                         "best", "purchase", "i've", "made", "terrible", 
                                         "experience", "overall"};
        
        std::istringstream iss(text);
        std::string word;
        int pos = 0;
        
        while (iss >> word && pos < max_len) {
            auto it = std::find(words.begin(), words.end(), word);
            if (it != words.end()) {
                result[pos] = std::distance(words.begin(), it) + 1;  // +1 to avoid 0
                pos++;
            }
        }
        
        return result;
    };
    
    // Classify each text
    std::vector<std::string> class_names = {"Negative", "Neutral", "Positive"};
    
    for (const auto& text : sample_texts) {
        Tensor inputs = tokenize_text(text, 32);
        
        // Inference
        Tensor logits = model->forward(inputs);
        
        // Get predicted class
        int predicted_class = 0;
        float max_logit = logits[0];
        
        for (int i = 1; i < logits.size(); ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                predicted_class = i;
            }
        }
        
        std::cout << "Text: \"" << text << "\"" << std::endl;
        std::cout << "Predicted class: " << class_names[predicted_class] << std::endl;
        std::cout << "Confidence: " << std::fixed << std::setprecision(4) 
                  << max_logit << std::endl << std::endl;
    }
}

int main() {
    std::cout << "Megatron-CPP-Edu Real-world Applications" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        // Run all examples
        run_text_classification_example();
        run_language_model_example();
        run_model_evaluation_example();
        run_performance_benchmark();
        run_realtime_inference_example();
        
        std::cout << "\\n=== All Examples Completed Successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}