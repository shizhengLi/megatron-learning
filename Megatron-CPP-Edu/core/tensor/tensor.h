#pragma once
#include <vector>
#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>

namespace megatron {

enum class DeviceType {
    CPU,
    CUDA
};

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT8
};

class Tensor {
public:
    // Constructors
    Tensor(); // Default constructor - creates empty tensor
    Tensor(const std::vector<int>& shape, DataType dtype = DataType::FLOAT32);
    Tensor(std::initializer_list<int> shape, DataType dtype = DataType::FLOAT32);
    
    // Copy and move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Basic operations
    void zeros();
    void ones();
    void random_normal(float mean = 0.0f, float std = 1.0f);
    void fill(float value);
    
    // Shape operations
    void reshape(const std::vector<int>& new_shape);
    Tensor view(const std::vector<int>& new_shape) const;
    Tensor transpose() const;
    Tensor slice(int dim, int start, int end) const;
    
    // Mathematical operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    Tensor matmul(const Tensor& other) const;
    Tensor sum(int dim = -1) const;
    Tensor mean(int dim = -1) const;
    Tensor max(int dim = -1) const;
    Tensor sqrt() const;
    
    // Activation functions
    Tensor relu() const;
    Tensor gelu() const;
    Tensor softmax(int dim = -1) const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    
    // Properties
    const std::vector<int>& shape() const { return shape_; }
    int dim() const { return shape_.size(); }
    int size() const { return size_; }
    DataType dtype() const { return dtype_; }
    DeviceType device() const { return device_; }
    
    // Data access
    float* data();
    const float* data() const;
    float& operator[](int index);
    const float& operator[](int index) const;
    
    // Device operations
    void to(DeviceType device);
    bool is_contiguous() const;
    
    // Memory management
    Tensor contiguous() const;
    void clone_from(const Tensor& other);
    
private:
    std::vector<int> shape_;
    int size_;
    DataType dtype_;
    DeviceType device_;
    std::shared_ptr<float> data_;
    bool owns_data_;
    
    void allocate_memory();
    void free_memory();
    void compute_size();
    void validate_shape(const std::vector<int>& shape) const;
    int get_element_index(const std::vector<int>& indices) const;
};

// Utility functions
Tensor add(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor concatenate(const std::vector<Tensor>& tensors, int dim = 0);

} // namespace megatron