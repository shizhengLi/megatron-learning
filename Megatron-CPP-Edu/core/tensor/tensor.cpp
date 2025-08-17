#include "tensor.h"
#include <cmath>
#include <random>
#include <iostream>

namespace megatron {

// Constructor implementations
Tensor::Tensor() 
    : shape_({0}), size_(0), dtype_(DataType::FLOAT32), device_(DeviceType::CPU), owns_data_(false) {
    // Default constructor creates empty tensor
}

Tensor::Tensor(const std::vector<int>& shape, DataType dtype) 
    : shape_(shape), dtype_(dtype), device_(DeviceType::CPU), owns_data_(true) {
    validate_shape(shape);
    compute_size();
    allocate_memory();
}

Tensor::Tensor(std::initializer_list<int> shape, DataType dtype)
    : shape_(shape), dtype_(dtype), device_(DeviceType::CPU), owns_data_(true) {
    validate_shape(shape_);
    compute_size();
    allocate_memory();
}

// Copy constructor
Tensor::Tensor(const Tensor& other) 
    : shape_(other.shape_), size_(other.size_), dtype_(other.dtype_),
      device_(other.device_), owns_data_(true) {
    allocate_memory();
    std::copy(other.data(), other.data() + size_, data());
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), size_(other.size_), dtype_(other.dtype_),
      device_(other.device_), data_(std::move(other.data_)), owns_data_(other.owns_data_) {
    other.size_ = 0;
    other.owns_data_ = false;
}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        free_memory();
        shape_ = other.shape_;
        size_ = other.size_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        owns_data_ = true;
        allocate_memory();
        std::copy(other.data(), other.data() + size_, data());
    }
    return *this;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free_memory();
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        data_ = std::move(other.data_);
        owns_data_ = other.owns_data_;
        
        other.size_ = 0;
        other.owns_data_ = false;
    }
    return *this;
}

// Basic operations
void Tensor::zeros() {
    std::fill(data(), data() + size_, 0.0f);
}

void Tensor::ones() {
    std::fill(data(), data() + size_, 1.0f);
}

void Tensor::random_normal(float mean, float std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    for (int i = 0; i < size_; ++i) {
        data()[i] = dist(gen);
    }
}

void Tensor::fill(float value) {
    std::fill(data(), data() + size_, value);
}

// Shape operations
void Tensor::reshape(const std::vector<int>& new_shape) {
    validate_shape(new_shape);
    
    int new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    
    if (new_size != size_) {
        throw std::invalid_argument("New shape must have same total size");
    }
    
    shape_ = new_shape;
}

Tensor Tensor::view(const std::vector<int>& new_shape) const {
    Tensor result = *this;
    result.reshape(new_shape);
    return result;
}

Tensor Tensor::transpose() const {
    if (dim() != 2) {
        throw std::invalid_argument("Transpose only supported for 2D tensors");
    }
    
    Tensor result({shape_[1], shape_[0]}, dtype_);
    for (int i = 0; i < shape_[0]; ++i) {
        for (int j = 0; j < shape_[1]; ++j) {
            result[j * shape_[0] + i] = (*this)[i * shape_[1] + j];
        }
    }
    return result;
}

Tensor Tensor::slice(int dim, int start, int end) const {
    if (dim < 0 || dim >= shape_.size()) {
        throw std::invalid_argument("Invalid dimension");
    }
    if (start < 0 || end > shape_[dim] || start >= end) {
        throw std::invalid_argument("Invalid slice range");
    }
    
    std::vector<int> new_shape = shape_;
    new_shape[dim] = end - start;
    
    Tensor result(new_shape, dtype_);
    
    // Simple implementation for 2D tensors
    if (this->dim() == 2) {
        if (dim == 0) {
            // Slice rows
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < shape_[1]; ++j) {
                    result[(i - start) * shape_[1] + j] = (*this)[i * shape_[1] + j];
                }
            }
        } else {
            // Slice columns
            for (int i = 0; i < shape_[0]; ++i) {
                for (int j = start; j < end; ++j) {
                    result[i * (end - start) + (j - start)] = (*this)[i * shape_[1] + j];
                }
            }
        }
    }
    
    return result;
}

// Mathematical operations
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have same shape for addition");
    }
    
    Tensor result(shape_, dtype_);
    for (int i = 0; i < size_; ++i) {
        result.data()[i] = data()[i] + other.data()[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have same shape for subtraction");
    }
    
    Tensor result(shape_, dtype_);
    for (int i = 0; i < size_; ++i) {
        result.data()[i] = data()[i] - other.data()[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have same shape for multiplication");
    }
    
    Tensor result(shape_, dtype_);
    for (int i = 0; i < size_; ++i) {
        result.data()[i] = data()[i] * other.data()[i];
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have same shape for division");
    }
    
    Tensor result(shape_, dtype_);
    for (int i = 0; i < size_; ++i) {
        if (other.data()[i] != 0.0f) {
            result.data()[i] = data()[i] / other.data()[i];
        } else {
            result.data()[i] = 0.0f; // Avoid division by zero
        }
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (dim() != 2 || other.dim() != 2) {
        throw std::invalid_argument("Matrix multiplication only supported for 2D tensors");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
    }
    
    Tensor result({shape_[0], other.shape_[1]}, dtype_);
    result.zeros();
    
    for (int i = 0; i < shape_[0]; ++i) {
        for (int j = 0; j < other.shape_[1]; ++j) {
            for (int k = 0; k < shape_[1]; ++k) {
                result[i * other.shape_[1] + j] += (*this)[i * shape_[1] + k] * other[k * other.shape_[1] + j];
            }
        }
    }
    
    return result;
}

Tensor Tensor::sum(int dim) const {
    if (dim == -1) {
        // Sum all elements
        Tensor result({1}, dtype_);
        result[0] = 0.0f;
        for (int i = 0; i < size_; ++i) {
            result[0] += (*this)[i];
        }
        return result;
    } else {
        // Sum along specific dimension
        if (dim < 0 || dim >= shape_.size()) {
            throw std::invalid_argument("Invalid dimension");
        }
        
        std::vector<int> new_shape = shape_;
        new_shape.erase(new_shape.begin() + dim);
        Tensor result(new_shape, dtype_);
        result.zeros();
        
        // Simple implementation for 2D tensors
        if (this->dim() == 2) {
            if (dim == 0) {
                // Sum rows
                for (int i = 0; i < shape_[0]; ++i) {
                    for (int j = 0; j < shape_[1]; ++j) {
                        result[j] += (*this)[i * shape_[1] + j];
                    }
                }
            } else {
                // Sum columns
                for (int i = 0; i < shape_[0]; ++i) {
                    for (int j = 0; j < shape_[1]; ++j) {
                        result[i] += (*this)[i * shape_[1] + j];
                    }
                }
            }
        }
        
        return result;
    }
}

Tensor Tensor::mean(int dim) const {
    Tensor sum_result = sum(dim);
    int divisor = (dim == -1) ? size_ : shape_[dim];
    
    // Create divisor tensor with same shape as sum_result
    Tensor divisor_tensor = sum_result;
    divisor_tensor.fill(static_cast<float>(divisor));
    
    return sum_result / divisor_tensor;
}

Tensor Tensor::max(int dim) const {
    if (dim == -1) {
        // Max of all elements
        Tensor result({1}, dtype_);
        result[0] = (*this)[0];
        for (int i = 1; i < size_; ++i) {
            if ((*this)[i] > result[0]) {
                result[0] = (*this)[i];
            }
        }
        return result;
    } else {
        // Max along specific dimension
        if (dim < 0 || dim >= shape_.size()) {
            throw std::invalid_argument("Invalid dimension");
        }
        
        std::vector<int> new_shape = shape_;
        new_shape.erase(new_shape.begin() + dim);
        Tensor result(new_shape, dtype_);
        
        // Simple implementation for 2D tensors
        if (this->dim() == 2) {
            if (dim == 0) {
                // Max of rows
                for (int j = 0; j < shape_[1]; ++j) {
                    result[j] = (*this)[j];
                    for (int i = 1; i < shape_[0]; ++i) {
                        if ((*this)[i * shape_[1] + j] > result[j]) {
                            result[j] = (*this)[i * shape_[1] + j];
                        }
                    }
                }
            } else {
                // Max of columns
                for (int i = 0; i < shape_[0]; ++i) {
                    result[i] = (*this)[i * shape_[1]];
                    for (int j = 1; j < shape_[1]; ++j) {
                        if ((*this)[i * shape_[1] + j] > result[i]) {
                            result[i] = (*this)[i * shape_[1] + j];
                        }
                    }
                }
            }
        }
        
        return result;
    }
}

// Activation functions
Tensor Tensor::relu() const {
    Tensor result(shape_, dtype_);
    for (int i = 0; i < size_; ++i) {
        result.data()[i] = std::max(0.0f, data()[i]);
    }
    return result;
}

Tensor Tensor::gelu() const {
    Tensor result(shape_, dtype_);
    for (int i = 0; i < size_; ++i) {
        float x = data()[i];
        result.data()[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
    return result;
}

Tensor Tensor::softmax(int dim) const {
    if (dim < 0 || dim >= shape_.size()) {
        throw std::invalid_argument("Invalid dimension for softmax");
    }
    
    Tensor result(shape_, dtype_);
    
    // Simple implementation for 2D tensors
    if (this->dim() == 2) {
        if (dim == 1) {
            // Softmax along columns (for each row)
            for (int i = 0; i < shape_[0]; ++i) {
                // Find max for numerical stability
                float max_val = (*this)[i * shape_[1]];
                for (int j = 1; j < shape_[1]; ++j) {
                    max_val = std::max(max_val, (*this)[i * shape_[1] + j]);
                }
                
                // Compute exp and sum
                float sum_exp = 0.0f;
                for (int j = 0; j < shape_[1]; ++j) {
                    float exp_val = std::exp((*this)[i * shape_[1] + j] - max_val);
                    result[i * shape_[1] + j] = exp_val;
                    sum_exp += exp_val;
                }
                
                // Normalize
                for (int j = 0; j < shape_[1]; ++j) {
                    result[i * shape_[1] + j] /= sum_exp;
                }
            }
        }
    }
    
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape_, dtype_);
    for (int i = 0; i < size_; ++i) {
        result.data()[i] = 1.0f / (1.0f + std::exp(-data()[i]));
    }
    return result;
}

Tensor Tensor::tanh() const {
    Tensor result(shape_, dtype_);
    for (int i = 0; i < size_; ++i) {
        result.data()[i] = std::tanh(data()[i]);
    }
    return result;
}

// Data access
float* Tensor::data() {
    return data_.get();
}

const float* Tensor::data() const {
    return data_.get();
}

float& Tensor::operator[](int index) {
    if (index < 0 || index >= size_) {
        throw std::out_of_range("Index out of range");
    }
    return data()[index];
}

const float& Tensor::operator[](int index) const {
    if (index < 0 || index >= size_) {
        throw std::out_of_range("Index out of range");
    }
    return data()[index];
}

// Device operations
void Tensor::to(DeviceType device) {
    if (device == device_) {
        return;
    }
    
    // For now, just update the device type
    // In a real implementation, this would involve memory transfer
    device_ = device;
}

bool Tensor::is_contiguous() const {
    // Simple implementation - assume contiguous for now
    return true;
}

// Memory management
Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    return Tensor(*this);
}

void Tensor::clone_from(const Tensor& other) {
    free_memory();
    shape_ = other.shape_;
    size_ = other.size_;
    dtype_ = other.dtype_;
    device_ = other.device_;
    owns_data_ = true;
    allocate_memory();
    std::copy(other.data(), other.data() + size_, data());
}

// Private methods
void Tensor::allocate_memory() {
    if (size_ > 0) {
        data_ = std::shared_ptr<float>(new float[size_], std::default_delete<float[]>());
    }
}

void Tensor::free_memory() {
    if (owns_data_ && data_) {
        data_.reset();
    }
}

void Tensor::compute_size() {
    size_ = 1;
    for (int dim : shape_) {
        if (dim <= 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
        size_ *= dim;
    }
}

void Tensor::validate_shape(const std::vector<int>& shape) const {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty");
    }
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("Dimensions must be positive");
        }
    }
}

int Tensor::get_element_index(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    
    int index = 0;
    int stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of range");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }
    
    return index;
}

// Utility functions
Tensor add(const Tensor& a, const Tensor& b) {
    return a + b;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    return a * b;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    return a.matmul(b);
}

Tensor concatenate(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot concatenate empty tensor list");
    }
    
    // For now, only support concatenating 2D tensors along dimension 1
    if (dim != 1 || tensors[0].dim() != 2) {
        throw std::invalid_argument("Only concatenating 2D tensors along dimension 1 is supported");
    }
    
    // Check all tensors have same shape except for concatenation dimension
    int rows = tensors[0].shape()[0];
    int total_cols = 0;
    
    for (const auto& tensor : tensors) {
        if (tensor.shape()[0] != rows) {
            throw std::invalid_argument("All tensors must have same number of rows");
        }
        total_cols += tensor.shape()[1];
    }
    
    Tensor result({rows, total_cols}, tensors[0].dtype());
    
    int col_offset = 0;
    for (const auto& tensor : tensors) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < tensor.shape()[1]; ++j) {
                result[i * total_cols + col_offset + j] = tensor[i * tensor.shape()[1] + j];
            }
        }
        col_offset += tensor.shape()[1];
    }
    
    return result;
}

} // namespace megatron