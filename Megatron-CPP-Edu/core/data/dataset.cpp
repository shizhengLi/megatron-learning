#include "dataset.h"
#include <fstream>
#include <sstream>
#include <cctype>
#include <unordered_set>
#include <algorithm>
#include <random>

namespace megatron {

// SimpleTokenizer implementation
SimpleTokenizer::SimpleTokenizer() {
    add_special_tokens();
}

void SimpleTokenizer::add_special_tokens() {
    vocab_.push_back("<pad>");
    vocab_.push_back("<unk>");
    vocab_.push_back("<cls>");
    vocab_.push_back("<sep>");
    
    word_to_id_["<pad>"] = pad_token_;
    word_to_id_["<unk>"] = unk_token_;
    word_to_id_["<cls>"] = cls_token_;
    word_to_id_["<sep>"] = sep_token_;
}

std::vector<std::string> SimpleTokenizer::tokenize_text(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current_token;
    
    for (char c : text) {
        if (std::isalnum(c) || c == '\'') {
            current_token += c;
        } else if (!current_token.empty()) {
            tokens.push_back(current_token);
            current_token.clear();
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

void SimpleTokenizer::build_vocab(const std::vector<std::string>& texts, int max_vocab_size) {
    std::unordered_map<std::string, int> word_counts;
    
    // Count word frequencies
    for (const auto& text : texts) {
        auto tokens = tokenize_text(text);
        for (const auto& token : tokens) {
            word_counts[token]++;
        }
    }
    
    // Sort by frequency
    std::vector<std::pair<std::string, int>> sorted_words(word_counts.begin(), word_counts.end());
    std::sort(sorted_words.begin(), sorted_words.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Add to vocabulary
    int current_id = vocab_.size();
    for (const auto& [word, count] : sorted_words) {
        if (current_id >= max_vocab_size) break;
        
        vocab_.push_back(word);
        word_to_id_[word] = current_id++;
    }
}

std::vector<int> SimpleTokenizer::encode(const std::string& text) const {
    auto tokens = tokenize_text(text);
    std::vector<int> token_ids;
    
    for (const auto& token : tokens) {
        auto it = word_to_id_.find(token);
        if (it != word_to_id_.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(unk_token_);
        }
    }
    
    return token_ids;
}

std::string SimpleTokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    
    for (int token_id : tokens) {
        if (token_id >= 0 && token_id < vocab_.size()) {
            if (!text.empty()) text += " ";
            text += vocab_[token_id];
        }
    }
    
    return text;
}

void SimpleTokenizer::save_vocab(const std::string& filepath) const {
    std::ofstream file(filepath);
    for (const auto& word : vocab_) {
        file << word << std::endl;
    }
}

void SimpleTokenizer::load_vocab(const std::string& filepath) {
    std::ifstream file(filepath);
    std::string word;
    vocab_.clear();
    word_to_id_.clear();
    
    while (std::getline(file, word)) {
        int id = vocab_.size();
        vocab_.push_back(word);
        word_to_id_[word] = id;
    }
}

// TextClassificationDataset implementation
TextClassificationDataset::TextClassificationDataset(const std::string& filepath, int max_length)
    : max_length_(max_length), tokenizer_(std::make_shared<SimpleTokenizer>()) {
    load_data(filepath);
}

void TextClassificationDataset::load_data(const std::string& filepath) {
    std::ifstream file(filepath);
    std::string line;
    
    // Skip header if exists
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string text, label_str;
        
        // Simple CSV parsing
        if (std::getline(ss, text, ',') && std::getline(ss, label_str, ',')) {
            preprocess_text(text);
            texts_.push_back(text);
            labels_.push_back(std::stoi(label_str));
        }
    }
    
    // Find number of classes
    if (!labels_.empty()) {
        num_classes_ = *std::max_element(labels_.begin(), labels_.end()) + 1;
    }
    
    // Build vocabulary
    tokenizer_->build_vocab(texts_);
}

void TextClassificationDataset::preprocess_text(std::string& text) {
    text = TextPreprocessor::clean_text(text);
    text = TextPreprocessor::lowercase(text);
    text = TextPreprocessor::remove_special_chars(text);
}

std::vector<int> TextClassificationDataset::pad_sequence(const std::vector<int>& sequence, int max_length) const {
    std::vector<int> padded = sequence;
    
    if (padded.size() > max_length) {
        padded.resize(max_length);
    } else {
        padded.resize(max_length, tokenizer_->pad_token());
    }
    
    return padded;
}

std::pair<Tensor, Tensor> TextClassificationDataset::get_batch(int batch_size, bool shuffle) {
    if (current_index_ >= texts_.size()) {
        current_index_ = 0;
        if (shuffle) {
            std::vector<int> indices(texts_.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            
            // Reorder data
            std::vector<std::string> shuffled_texts(texts_.size());
            std::vector<int> shuffled_labels(labels_.size());
            
            for (int i = 0; i < indices.size(); ++i) {
                shuffled_texts[i] = texts_[indices[i]];
                shuffled_labels[i] = labels_[indices[i]];
            }
            
            texts_ = shuffled_texts;
            labels_ = shuffled_labels;
        }
    }
    
    int end_index = std::min(current_index_ + batch_size, static_cast<int>(texts_.size()));
    int actual_batch_size = end_index - current_index_;
    
    // Prepare batch data
    std::vector<std::vector<int>> batch_tokens;
    std::vector<int> batch_labels;
    
    for (int i = current_index_; i < end_index; ++i) {
        auto tokens = tokenizer_->encode(texts_[i]);
        tokens.insert(tokens.begin(), tokenizer_->cls_token());  // Add CLS token
        auto padded = pad_sequence(tokens, max_length_);
        batch_tokens.push_back(padded);
        batch_labels.push_back(labels_[i]);
    }
    
    // Create tensors
    Tensor inputs({actual_batch_size, max_length_});
    Tensor targets({actual_batch_size});
    
    for (int i = 0; i < actual_batch_size; ++i) {
        for (int j = 0; j < max_length_; ++j) {
            inputs[i * max_length_ + j] = batch_tokens[i][j];
        }
        targets[i] = batch_labels[i];
    }
    
    current_index_ = end_index;
    
    return {inputs, targets};
}

std::pair<TextClassificationDataset, TextClassificationDataset> TextClassificationDataset::split(float train_ratio) {
    int train_size = static_cast<int>(texts_.size() * train_ratio);
    
    TextClassificationDataset train_dataset("", max_length_);
    TextClassificationDataset test_dataset("", max_length_);
    
    train_dataset.tokenizer_ = tokenizer_;
    train_dataset.max_length_ = max_length_;
    train_dataset.num_classes_ = num_classes_;
    
    test_dataset.tokenizer_ = tokenizer_;
    test_dataset.max_length_ = max_length_;
    test_dataset.num_classes_ = num_classes_;
    
    // Split data
    train_dataset.texts_ = std::vector<std::string>(texts_.begin(), texts_.begin() + train_size);
    train_dataset.labels_ = std::vector<int>(labels_.begin(), labels_.begin() + train_size);
    
    test_dataset.texts_ = std::vector<std::string>(texts_.begin() + train_size, texts_.end());
    test_dataset.labels_ = std::vector<int>(labels_.begin() + train_size, labels_.end());
    
    return {train_dataset, test_dataset};
}

// LanguageModelDataset implementation
LanguageModelDataset::LanguageModelDataset(const std::string& filepath, int seq_length)
    : seq_length_(seq_length), tokenizer_(std::make_shared<SimpleTokenizer>()) {
    load_data(filepath);
}

void LanguageModelDataset::load_data(const std::string& filepath) {
    std::ifstream file(filepath);
    std::string text((std::istreambuf_iterator<char>(file)), 
                     std::istreambuf_iterator<char>());
    
    tokenize_text(text);
}

void LanguageModelDataset::tokenize_text(const std::string& text) {
    tokens_ = tokenizer_->encode(text);
}

std::pair<Tensor, Tensor> LanguageModelDataset::get_batch(int batch_size) {
    if (current_index_ + seq_length_ + 1 > tokens_.size()) {
        current_index_ = 0;
    }
    
    int actual_batch_size = std::min(batch_size, size() - current_index_);
    
    Tensor inputs({actual_batch_size, seq_length_});
    Tensor targets({actual_batch_size, seq_length_});
    
    for (int i = 0; i < actual_batch_size; ++i) {
        for (int j = 0; j < seq_length_; ++j) {
            inputs[i * seq_length_ + j] = tokens_[current_index_ + i + j];
            targets[i * seq_length_ + j] = tokens_[current_index_ + i + j + 1];
        }
    }
    
    current_index_ += actual_batch_size;
    
    return {inputs, targets};
}

// DataLoader implementation
DataLoader::DataLoader(int batch_size, bool shuffle)
    : batch_size_(batch_size), shuffle_(shuffle) {}

void DataLoader::add_data(const Tensor& inputs, const Tensor& targets) {
    all_inputs_ = inputs;
    all_targets_ = targets;
    create_batches();
}

void DataLoader::create_batches() {
    int num_samples = all_inputs_.shape()[0];
    num_batches_ = (num_samples + batch_size_ - 1) / batch_size_;
    
    indices_.resize(num_samples);
    std::iota(indices_.begin(), indices_.end(), 0);
    
    if (shuffle_) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_.begin(), indices_.end(), g);
    }
}

std::pair<Tensor, Tensor> DataLoader::next_batch() {
    if (current_batch_ >= num_batches_) {
        reset();
        return next_batch();
    }
    
    int start_idx = current_batch_ * batch_size_;
    int end_idx = std::min(start_idx + batch_size_, static_cast<int>(indices_.size()));
    int actual_batch_size = end_idx - start_idx;
    
    // Extract batch data
    std::vector<int> batch_indices(indices_.begin() + start_idx, indices_.begin() + end_idx);
    
    Tensor batch_inputs(all_inputs_.shape());
    batch_inputs.reshape({actual_batch_size, all_inputs_.shape()[1]});
    
    Tensor batch_targets(all_targets_.shape());
    batch_targets.reshape({actual_batch_size});
    
    for (int i = 0; i < actual_batch_size; ++i) {
        int idx = batch_indices[i];
        
        // Copy input data
        for (int j = 0; j < all_inputs_.shape()[1]; ++j) {
            batch_inputs[i * all_inputs_.shape()[1] + j] = all_inputs_[idx * all_inputs_.shape()[1] + j];
        }
        
        // Copy target data
        batch_targets[i] = all_targets_[idx];
    }
    
    current_batch_++;
    
    return {batch_inputs, batch_targets};
}

bool DataLoader::has_more_batches() const {
    return current_batch_ < num_batches_;
}

void DataLoader::reset() {
    current_batch_ = 0;
    if (shuffle_) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_.begin(), indices_.end(), g);
    }
}

// TextPreprocessor implementation
std::string TextPreprocessor::clean_text(const std::string& text) {
    std::string cleaned = text;
    
    // Remove extra whitespace
    cleaned.erase(std::unique(cleaned.begin(), cleaned.end(), 
                              [](char a, char b) { return a == ' ' && b == ' '; }), 
                  cleaned.end());
    
    // Trim leading/trailing whitespace
    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r\f\v"));
    cleaned.erase(cleaned.find_last_not_of(" \t\n\r\f\v") + 1);
    
    return cleaned;
}

std::string TextPreprocessor::lowercase(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string TextPreprocessor::remove_special_chars(const std::string& text) {
    std::string result;
    for (char c : text) {
        if (std::isalnum(c) || std::isspace(c) || c == '\'' || c == '-') {
            result += c;
        } else {
            result += ' ';
        }
    }
    return result;
}

std::vector<std::string> TextPreprocessor::split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string current;
    
    for (char c : text) {
        current += c;
        if (c == '.' || c == '!' || c == '?') {
            sentences.push_back(current);
            current.clear();
        }
    }
    
    if (!current.empty()) {
        sentences.push_back(current);
    }
    
    return sentences;
}

std::string TextPreprocessor::join_sentences(const std::vector<std::string>& sentences) {
    std::string result;
    for (const auto& sentence : sentences) {
        if (!result.empty()) {
            result += " ";
        }
        result += sentence;
    }
    return result;
}

} // namespace megatron