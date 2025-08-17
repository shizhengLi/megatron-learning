#pragma once

#include "core/tensor/tensor.h"
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

namespace megatron {

// Simple tokenizer implementation
class SimpleTokenizer {
public:
    SimpleTokenizer();
    
    // Build vocabulary from text
    void build_vocab(const std::vector<std::string>& texts, int max_vocab_size = 10000);
    
    // Encode text to token IDs
    std::vector<int> encode(const std::string& text) const;
    
    // Decode token IDs to text
    std::string decode(const std::vector<int>& tokens) const;
    
    // Get vocabulary size
    int vocab_size() const { return vocab_.size(); }
    
    // Get special tokens
    int pad_token() const { return pad_token_; }
    int unk_token() const { return unk_token_; }
    int cls_token() const { return cls_token_; }
    int sep_token() const { return sep_token_; }
    
    // Save/load vocabulary
    void save_vocab(const std::string& filepath) const;
    void load_vocab(const std::string& filepath);

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> word_to_id_;
    int pad_token_ = 0;
    int unk_token_ = 1;
    int cls_token_ = 2;
    int sep_token_ = 3;
    
    void add_special_tokens();
    std::vector<std::string> tokenize_text(const std::string& text) const;
};

// Dataset class for text classification
class TextClassificationDataset {
public:
    TextClassificationDataset(const std::string& filepath, int max_length = 512);
    
    // Load data from CSV file
    void load_data(const std::string& filepath);
    
    // Get batch of data
    std::pair<Tensor, Tensor> get_batch(int batch_size, bool shuffle = true);
    
    // Get dataset size
    int size() const { return texts_.size(); }
    
    // Get number of classes
    int num_classes() const { return num_classes_; }
    
    // Split dataset
    std::pair<TextClassificationDataset, TextClassificationDataset> split(float train_ratio = 0.8f);

private:
    std::vector<std::string> texts_;
    std::vector<int> labels_;
    std::shared_ptr<SimpleTokenizer> tokenizer_;
    int max_length_;
    int num_classes_;
    int current_index_ = 0;
    
    void preprocess_text(std::string& text);
    std::vector<int> pad_sequence(const std::vector<int>& sequence, int max_length) const;
};

// Dataset class for language modeling
class LanguageModelDataset {
public:
    LanguageModelDataset(const std::string& filepath, int seq_length = 128);
    
    // Load data from text file
    void load_data(const std::string& filepath);
    
    // Get batch of data
    std::pair<Tensor, Tensor> get_batch(int batch_size);
    
    // Get dataset size
    int size() const { return tokens_.size() - seq_length_; }
    
    // Get vocabulary size
    int vocab_size() const { return tokenizer_->vocab_size(); }

private:
    std::vector<int> tokens_;
    std::shared_ptr<SimpleTokenizer> tokenizer_;
    int seq_length_;
    int current_index_ = 0;
    
    void tokenize_text(const std::string& text);
};

// Data loader with batching and shuffling
class DataLoader {
public:
    DataLoader(int batch_size = 32, bool shuffle = true);
    
    // Add data to loader
    void add_data(const Tensor& inputs, const Tensor& targets);
    
    // Get next batch
    std::pair<Tensor, Tensor> next_batch();
    
    // Check if there are more batches
    bool has_more_batches() const;
    
    // Reset loader
    void reset();
    
    // Get number of batches
    int num_batches() const { return num_batches_; }

private:
    int batch_size_;
    bool shuffle_;
    Tensor all_inputs_;
    Tensor all_targets_;
    std::vector<int> indices_;
    int current_batch_ = 0;
    int num_batches_ = 0;
    
    void create_batches();
};

// Text preprocessing utilities
class TextPreprocessor {
public:
    static std::string clean_text(const std::string& text);
    static std::string lowercase(const std::string& text);
    static std::string remove_special_chars(const std::string& text);
    static std::vector<std::string> split_sentences(const std::string& text);
    static std::string join_sentences(const std::vector<std::string>& sentences);
};

} // namespace megatron