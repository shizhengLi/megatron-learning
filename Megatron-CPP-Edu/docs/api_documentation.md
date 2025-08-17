# Megatron-CPP-Edu API 文档

## 目录
1. [核心模块](#1-核心模块)
2. [并行计算模块](#2-并行计算模块)
3. [分布式通信模块](#3-分布式通信模块)
4. [数据加载模块](#4-数据加载模块)
5. [优化器模块](#5-优化器模块)
6. [工具类](#6-工具类)

---

## 1. 核心模块

### 1.1 Tensor 类

#### 构造函数
```cpp
// 默认构造函数
Tensor();

// 指定形状构造
Tensor(const std::vector<int>& shape);

// 指定形状和数据构造
Tensor(const std::vector<int>& shape, const std::vector<float>& data);

// 复制构造函数
Tensor(const Tensor& other);

// 移动构造函数
Tensor(Tensor&& other) noexcept;
```

#### 基本操作
```cpp
// 获取张量信息
std::vector<int> shape() const;
int size() const;
int dim() const;
int shape(int dim) const;
float* data();
const float* data() const;

// 重置张量
void zeros();
void ones();
void fill(float value);
void random_normal(float mean = 0.0f, float std = 1.0f);
void random_uniform(float min = 0.0f, float max = 1.0f);

// 重塑形状
void reshape(const std::vector<int>& new_shape);
Tensor reshape(const std::vector<int>& new_shape) const;

// 转置
Tensor transpose() const;
Tensor transpose(int dim1, int dim2) const;

// 数学运算
Tensor operator+(const Tensor& other) const;
Tensor operator-(const Tensor& other) const;
Tensor operator*(const Tensor& other) const;
Tensor operator/(const Tensor& other) const;
Tensor& operator+=(const Tensor& other);
Tensor& operator-=(const Tensor& other);
Tensor& operator*=(const Tensor& other);
Tensor& operator/=(const Tensor& other);

// 矩阵乘法
Tensor matmul(const Tensor& other) const;

// 归约操作
float sum() const;
float mean() const;
float max() const;
float min() const;
float norm() const;
float norm_squared() const;

// 激活函数
Tensor relu() const;
Tensor sigmoid() const;
Tensor tanh() const;
Tensor gelu() const;
Tensor softmax(int dim = -1) const;

// 张量操作
Tensor slice(int start, int end) const;
Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);
Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);
Tensor permute(const std::vector<int>& dims) const;

// 访问元素
float& operator[](int index);
const float& operator[](int index) const;
float& at(const std::vector<int>& indices);
const float& at(const std::vector<int>& indices) const;
```

#### 静态方法
```cpp
// 创建特殊张量
static Tensor zeros(const std::vector<int>& shape);
static Tensor ones(const std::vector<int>& shape);
static Tensor full(const std::vector<int>& shape, float value);
static Tensor randn(const std::vector<int>& shape, float mean = 0.0f, float std = 1.0f);
static Tensor randint(const std::vector<int>& shape, int min, int max);

// 数学函数
static Tensor exp(const Tensor& x);
static Tensor log(const Tensor& x);
static Tensor sqrt(const Tensor& x);
static Tensor pow(const Tensor& x, float exponent);
static Tensor abs(const Tensor& x);
```

### 1.2 Layer 基类

#### 虚函数接口
```cpp
class Layer {
public:
    // 构造函数
    Layer(const std::string& name = "");
    virtual ~Layer() = default;

    // 前向传播（纯虚函数）
    virtual Tensor forward(const Tensor& input) = 0;

    // 反向传播（纯虚函数）
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // 获取参数（纯虚函数）
    virtual std::vector<Tensor> parameters() const = 0;
    virtual std::vector<Tensor> gradients() const = 0;

    // 训练/评估模式
    virtual void train();
    virtual void eval();
    bool is_training() const;

    // 获取层名称
    const std::string& name() const;

    // 梯度清零
    virtual void zero_grad();

protected:
    std::string name_;
    bool training_;
};
```

### 1.3 具体层实现

#### Linear 层
```cpp
class Linear : public Layer {
public:
    // 构造函数
    Linear(int in_features, int out_features, bool bias = true,
           const std::string& name = "linear");

    // 前向传播
    Tensor forward(const Tensor& input) override;

    // 反向传播
    Tensor backward(const Tensor& grad_output) override;

    // 获取参数
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;

    // 权重初始化
    void init_weights();
    void init_xavier_uniform();
    void init_kaiming_normal();
};
```

#### Embedding 层
```cpp
class Embedding : public Layer {
public:
    // 构造函数
    Embedding(int vocab_size, int embedding_dim,
              const std::string& name = "embedding");

    // 前向传播
    Tensor forward(const Tensor& input) override;

    // 反向传播
    Tensor backward(const Tensor& grad_output) override;

    // 获取参数
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
};
```

#### LayerNorm 层
```cpp
class LayerNorm : public Layer {
public:
    // 构造函数
    LayerNorm(int normalized_shape, float eps = 1e-5,
              const std::string& name = "layer_norm");

    // 前向传播
    Tensor forward(const Tensor& input) override;

    // 反向传播
    Tensor backward(const Tensor& grad_output) override;

    // 获取参数
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;
};
```

#### Dropout 层
```cpp
class Dropout : public Layer {
public:
    // 构造函数
    Dropout(float p = 0.5, const std::string& name = "dropout");

    // 前向传播
    Tensor forward(const Tensor& input) override;

    // 反向传播
    Tensor backward(const Tensor& grad_output) override;

    // 获取参数
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;

    // 设置dropout概率
    void set_p(float p);
    float get_p() const;
};
```

---

## 2. 并行计算模块

### 2.1 张量并行模块

#### TensorParallelContext
```cpp
class TensorParallelContext {
public:
    // 获取单例
    static TensorParallelContext& instance();

    // 初始化
    void initialize(int world_size, int rank);

    // 获取信息
    int world_size() const;
    int rank() const;
    bool is_enabled() const;

    // 获取分割信息
    int get_local_output_dim(int global_output_dim) const;
    int get_local_input_dim(int global_input_dim) const;
};
```

#### ColumnParallelLinear
```cpp
class ColumnParallelLinear : public Layer {
public:
    // 构造函数
    ColumnParallelLinear(int in_features, int out_features, bool bias = true,
                        const std::string& name = "column_parallel_linear");

    // 前向传播
    Tensor forward(const Tensor& input) override;

    // 反向传播
    Tensor backward(const Tensor& grad_output) override;

    // 获取参数
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;

    // 获取全局参数形状
    std::vector<int> get_global_weight_shape() const;

    // 训练模式
    void train(bool is_training) override;
};
```

#### RowParallelLinear
```cpp
class RowParallelLinear : public Layer {
public:
    // 构造函数
    RowParallelLinear(int in_features, int out_features, bool bias = true,
                      const std::string& name = "row_parallel_linear");

    // 前向传播
    Tensor forward(const Tensor& input) override;

    // 反向传播
    Tensor backward(const Tensor& grad_output) override;

    // 获取参数
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;

    // 获取全局参数形状
    std::vector<int> get_global_weight_shape() const;

    // 训练模式
    void train(bool is_training) override;
};
```

#### TensorParallelMultiHeadAttention
```cpp
class TensorParallelMultiHeadAttention : public Layer {
public:
    // 构造函数
    TensorParallelMultiHeadAttention(int embed_dim, int num_heads,
                                   bool use_causal_mask = false,
                                   const std::string& name = "tp_mha");

    // 前向传播
    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value);

    // 反向传播
    Tensor backward(const Tensor& grad_output) override;

    // 获取参数
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;

    // 训练模式
    void train(bool is_training) override;
};
```

#### TensorParallelFFN
```cpp
class TensorParallelFFN : public Layer {
public:
    // 构造函数
    TensorParallelFFN(int embed_dim, int ffn_dim, float dropout = 0.1,
                      const std::string& name = "tp_ffn");

    // 前向传播
    Tensor forward(const Tensor& input) override;

    // 反向传播
    Tensor backward(const Tensor& grad_output) override;

    // 获取参数
    std::vector<Tensor> parameters() const override;
    std::vector<Tensor> gradients() const override;

    // 训练模式
    void train(bool is_training) override;
};
```

#### TensorParallelModelBuilder
```cpp
class TensorParallelModelBuilder {
public:
    // 构造函数
    TensorParallelModelBuilder(int tensor_parallel_size = 1);

    // 设置模型配置
    void set_vocab_size(int vocab_size);
    void set_max_seq_len(int max_seq_len);
    void set_embed_dim(int embed_dim);
    void set_num_layers(int num_layers);
    void set_num_heads(int num_heads);
    void set_ffn_dim(int ffn_dim);
    void set_dropout(float dropout);

    // 构建模型
    std::shared_ptr<Layer> build_gpt_model();
    std::shared_ptr<Layer> build_transformer_classifier(int num_classes);
};
```

### 2.2 数据并行模块

#### DataParallelConfig
```cpp
struct DataParallelConfig {
    int world_size = 1;
    int rank = 0;
    int global_batch_size = 32;
    int local_batch_size = 8;
    bool sync_bn = true;
    bool find_unused_parameters = false;
    float bucket_cap_mb = 25.0f;

    // 验证配置
    bool validate() const;

    // 获取梯度累积步数
    int get_gradient_accumulation_steps() const;
};
```

#### DataParallelTrainer
```cpp
class DataParallelTrainer {
public:
    // 构造函数
    DataParallelTrainer(const DataParallelConfig& config);

    // 设置模型和优化器
    void setup_model_and_optimizer(std::shared_ptr<Layer> model,
                                 std::shared_ptr<Optimizer> optimizer);

    // 训练步骤
    void train_step(const Tensor& inputs, const Tensor& targets);
    void train_step_with_loss(const Tensor& inputs, const Tensor& targets, float loss);

    // 梯度同步
    void synchronize_gradients();
    void all_reduce_gradients();

    // 参数同步
    void synchronize_parameters();
    void broadcast_parameters();

    // 获取信息
    int world_size() const;
    int rank() const;
    bool is_distributed() const;
    int get_local_batch_size() const;
    int get_global_batch_size() const;

    // 获取训练统计
    float get_average_loss() const;
    int get_global_step() const;
};
```

#### DistributedDataParallel
```cpp
class DistributedDataParallel {
public:
    // 构造函数
    DistributedDataParallel(std::shared_ptr<Layer> model,
                           const DataParallelConfig& config);

    // 前向和反向传播
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    // 参数管理
    std::vector<Tensor> parameters() const;
    std::vector<Tensor> gradients() const;

    // 训练模式
    void train(bool is_training);
    bool is_training() const;

    // 同步操作
    void sync_parameters();
    void sync_gradients();

    // 获取内部模型和配置
    std::shared_ptr<Layer> get_model() const;
    const DataParallelConfig& get_config() const;
};
```

#### HybridParallelTrainer
```cpp
class HybridParallelTrainer {
public:
    // 构造函数
    HybridParallelTrainer(const DataParallelConfig& dp_config,
                         int tensor_parallel_size = 1,
                         int pipeline_parallel_size = 1);

    // 设置模型和优化器
    void setup_model_and_optimizer(std::shared_ptr<Layer> model,
                                 std::shared_ptr<Optimizer> optimizer);

    // 混合并行训练步骤
    void hybrid_train_step(const Tensor& inputs, const Tensor& targets);

    // 梯度同步
    void synchronize_gradients();
    void synchronize_parameters();

    // 获取并行状态
    int get_data_parallel_size() const;
    int get_tensor_parallel_size() const;
    int get_pipeline_parallel_size() const;
    int get_global_rank() const;
    int get_local_rank() const;

    // 获取配置
    const DataParallelConfig& get_dp_config() const;
};
```

---

## 3. 分布式通信模块

### 3.1 MPICommunicator

#### 基本操作
```cpp
class MPICommunicator {
public:
    // 获取单例
    static MPICommunicator& instance();

    // 初始化和清理
    void initialize(int* argc, char*** argv);
    void finalize();

    // 获取信息
    int world_size() const;
    int rank() const;
    bool is_initialized() const;

    // 基础通信操作
    void send(const Tensor& tensor, int dest_rank, int tag = 0);
    void recv(Tensor& tensor, int src_rank, int tag = 0);

    // 集合通信
    void all_reduce(Tensor& tensor);
    void all_gather(const std::vector<Tensor>& send_tensors,
                   std::vector<Tensor>& recv_tensors);
    void broadcast(Tensor& tensor, int root_rank);
    void reduce(Tensor& tensor, int root_rank);

    // 障碍同步
    void barrier();

    // 获取信息
    std::string get_comm_name() const;
    int get_local_device_id() const;
};
```

#### 异步通信操作
```cpp
// 异步发送
void isend(const Tensor& tensor, int dest_rank, int tag, MPI_Request& request);

// 异步接收
void irecv(Tensor& tensor, int src_rank, int tag, MPI_Request& request);

// 等待通信完成
void wait(MPI_Request& request);
void wait_all(std::vector<MPI_Request>& requests);

// 测试通信完成
bool test(MPI_Request& request);
bool test_all(std::vector<MPI_Request>& requests);
```

### 3.2 DistributedTrainer

```cpp
class DistributedTrainer {
public:
    // 构造函数
    DistributedTrainer(int world_size = 1, int rank = 0);

    // 设置模型和优化器
    void setup_model_and_optimizer(std::shared_ptr<Layer> model,
                                 std::shared_ptr<Optimizer> optimizer);

    // 分布式训练步骤
    void distributed_train_step(const Tensor& input_ids, const Tensor& targets);

    // 梯度同步
    void synchronize_gradients(std::vector<Tensor>& gradients);

    // 参数广播
    void broadcast_parameters(std::vector<Tensor>& parameters);

    // 获取本地批大小
    int get_local_batch_size(int global_batch_size) const;

    // 获取分布式状态
    int world_size() const;
    int rank() const;
    bool is_distributed() const;
};
```

### 3.3 ParallelState

```cpp
class ParallelState {
public:
    // 获取单例
    static ParallelState& instance();

    // 初始化并行状态
    void initialize_tensor_parallel(int tensor_parallel_size, int tensor_parallel_rank);
    void initialize_pipeline_parallel(int pipeline_parallel_size, int pipeline_parallel_rank);
    void initialize_data_parallel(int data_parallel_size, int data_parallel_rank);

    // 获取并行信息
    int get_tensor_parallel_size() const;
    int get_tensor_parallel_rank() const;
    int get_pipeline_parallel_size() const;
    int get_pipeline_parallel_rank() const;
    int get_data_parallel_size() const;
    int get_data_parallel_rank() const;

    // 获取全局信息
    int get_world_size() const;
    int get_global_rank() const;

    // 检查并行状态
    bool is_tensor_parallel_enabled() const;
    bool is_pipeline_parallel_enabled() const;
    bool is_data_parallel_enabled() const;

    // 获取通信器
    MPI_Comm get_tensor_parallel_comm() const;
    MPI_Comm get_pipeline_parallel_comm() const;
    MPI_Comm get_data_parallel_comm() const;
};
```

---

## 4. 数据加载模块

### 4.1 Dataset 基类

```cpp
class Dataset {
public:
    virtual ~Dataset() = default;

    // 获取数据集大小
    virtual size_t size() const = 0;

    // 获取单个样本
    virtual std::pair<Tensor, Tensor> get_item(size_t index) = 0;

    // 获取批次
    virtual std::pair<Tensor, Tensor> get_batch(const std::vector<size_t>& indices);

    // 数据预处理
    virtual void preprocess();

    // 数据增强
    virtual void augment();
};
```

### 4.2 SimpleDataset

```cpp
class SimpleDataset : public Dataset {
public:
    // 构造函数
    SimpleDataset(const std::vector<std::vector<float>>& data,
                 const std::vector<int>& labels);

    // 实现接口
    size_t size() const override;
    std::pair<Tensor, Tensor> get_item(size_t index) override;

    // 添加数据
    void add_data(const std::vector<float>& sample, int label);
    void add_batch(const std::vector<std::vector<float>>& batch,
                  const std::vector<int>& labels);
};
```

### 4.3 DataLoader

```cpp
class DataLoader {
public:
    // 构造函数
    DataLoader(std::shared_ptr<Dataset> dataset, int batch_size = 32,
               bool shuffle = true, int num_workers = 0);

    // 迭代器接口
    class Iterator {
    public:
        Iterator(DataLoader* loader, size_t index);
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;
        std::pair<Tensor, Tensor> operator*() const;
    };

    // 获取迭代器
    Iterator begin();
    Iterator end();

    // 获取信息
    size_t size() const;
    int batch_size() const;

    // 重置数据加载器
    void reset();

    // 设置随机种子
    void set_seed(int seed);
};
```

### 4.4 DataParallelDataLoader

```cpp
class DataParallelDataLoader {
public:
    // 构造函数
    DataParallelDataLoader(std::shared_ptr<Dataset> dataset,
                          const DataParallelConfig& config,
                          bool shuffle = true);

    // 获取下一个批次
    std::pair<Tensor, Tensor> next_batch();
    bool has_next_batch() const;
    void reset();

    // 获取信息
    int num_batches() const;
    int current_batch() const;

    // 获取本地数据集
    std::shared_ptr<Dataset> get_local_dataset() const;
};
```

---

## 5. 优化器模块

### 5.1 Optimizer 基类

```cpp
class Optimizer {
public:
    // 构造函数
    Optimizer(float learning_rate = 1e-3);

    // 虚析构函数
    virtual ~Optimizer() = default;

    // 优化步骤（纯虚函数）
    virtual void step(std::vector<Tensor>& parameters,
                     std::vector<Tensor>& gradients) = 0;

    // 梯度清零
    virtual void zero_grad(std::vector<Tensor>& gradients);

    // 学习率管理
    void set_learning_rate(float lr);
    float get_learning_rate() const;

    // 获取参数
    std::vector<Tensor> get_parameters() const;
    void set_parameters(const std::vector<Tensor>& parameters);

protected:
    float learning_rate_;
    std::vector<Tensor> parameters_;
};
```

### 5.2 SGD 优化器

```cpp
class SGD : public Optimizer {
public:
    // 构造函数
    SGD(float learning_rate = 1e-3, float momentum = 0.0f,
        float weight_decay = 0.0f, bool nesterov = false);

    // 优化步骤
    void step(std::vector<Tensor>& parameters,
             std::vector<Tensor>& gradients) override;

    // 获取超参数
    float get_momentum() const;
    float get_weight_decay() const;
    bool get_nesterov() const;

    // 设置超参数
    void set_momentum(float momentum);
    void set_weight_decay(float weight_decay);
    void set_nesterov(bool nesterov);
};
```

### 5.3 Adam 优化器

```cpp
class Adam : public Optimizer {
public:
    // 构造函数
    Adam(float learning_rate = 1e-3, float beta1 = 0.9f, float beta2 = 0.999f,
        float eps = 1e-8f, float weight_decay = 0.0f, bool amsgrad = false);

    // 优化步骤
    void step(std::vector<Tensor>& parameters,
             std::vector<Tensor>& gradients) override;

    // 获取超参数
    float get_beta1() const;
    float get_beta2() const;
    float get_eps() const;
    float get_weight_decay() const;
    bool get_amsgrad() const;

    // 设置超参数
    void set_beta1(float beta1);
    void set_beta2(float beta2);
    void set_eps(float eps);
    void set_weight_decay(float weight_decay);
    void set_amsgrad(bool amsgrad);
};
```

### 5.4 DistributedOptimizer

```cpp
class DistributedOptimizer {
public:
    // 构造函数
    DistributedOptimizer(std::shared_ptr<Optimizer> base_optimizer,
                        const DataParallelConfig& config);

    // 分布式优化步骤
    void step(std::vector<Tensor>& parameters,
             std::vector<Tensor>& gradients);

    // 学习率管理
    void set_learning_rate(float lr);
    float get_learning_rate() const;

    // 梯度操作
    void zero_grad(std::vector<Tensor>& gradients);
    void clip_gradients(std::vector<Tensor>& gradients, float max_norm);

    // 获取基础优化器
    std::shared_ptr<Optimizer> get_base_optimizer() const;
};
```

---

## 6. 工具类

### 6.1 并行工具

#### tensor_parallel_utils
```cpp
namespace tensor_parallel_utils {
    // 初始化和清理
    void initialize_tensor_parallel(int world_size, int rank);
    void cleanup_tensor_parallel();

    // 获取信息
    int get_tensor_parallel_world_size();
    int get_tensor_parallel_rank();
    bool is_tensor_parallel_supported();

    // 张量操作
    Tensor split_tensor(const Tensor& tensor, int dim, int rank, int world_size);
    Tensor concatenate_tensors(const std::vector<Tensor>& tensors, int dim);

    // 验证和打印
    bool validate_tensor_parallel_config(int world_size, int rank);
    void print_tensor_parallel_info();
}
```

#### data_parallel_utils
```cpp
namespace data_parallel_utils {
    // 初始化和清理
    void initialize_data_parallel(const DataParallelConfig& config);
    void cleanup_data_parallel();

    // 数据集分割
    std::shared_ptr<Dataset> partition_dataset(std::shared_ptr<Dataset> dataset,
                                              int rank, int world_size);

    // 批次计算
    int calculate_local_batch_size(int global_batch_size, int world_size);

    // 验证和检查
    bool is_data_parallel_supported();
    bool validate_data_parallel_config(const DataParallelConfig& config);
    bool verify_gradient_synchronization(const std::vector<Tensor>& gradients,
                                       const DataParallelConfig& config);

    // 工厂函数
    std::shared_ptr<DistributedDataParallel> create_ddp_model(
        std::shared_ptr<Layer> model, const DataParallelConfig& config);
    std::shared_ptr<DistributedOptimizer> create_distributed_optimizer(
        std::shared_ptr<Optimizer> optimizer, const DataParallelConfig& config);

    // 同步操作
    void sync_all_processes();
    void master_do(const DataParallelConfig& config, std::function<void()> func);
    float calculate_global_average_loss(float local_loss, const DataParallelConfig& config);
}
```

#### distributed_utils
```cpp
namespace distributed_utils {
    // 初始化和清理
    void initialize_distributed(const DistributedConfig& config);
    void cleanup_distributed();

    // 设备管理
    int get_current_device();
    void set_device(int device_id);

    // 同步操作
    void sync_all_processes();
    void master_do(std::function<void()> func);

    // 信息获取
    void print_distributed_info();
    bool is_distributed_supported();

    // 通信器管理
    MPI_Comm get_local_comm();
    MPI_Comm get_cross_comm();
}
```

### 6.2 性能监控

#### DataParallelProfiler
```cpp
class DataParallelProfiler {
public:
    // 构造函数
    DataParallelProfiler(const DataParallelConfig& config);

    // 计时操作
    void start_timer(const std::string& name);
    void stop_timer(const std::string& name);

    // 统计记录
    void record_statistic(const std::string& name, float value);
    void record_throughput(float samples_per_second);

    // 性能报告
    std::string generate_performance_report() const;
    std::map<std::string, float> get_performance_metrics() const;

    // 同步性能统计
    void sync_performance_metrics();
};
```

### 6.3 异常处理

#### DataParallelException
```cpp
class DataParallelException : public std::runtime_error {
public:
    // 构造函数
    DataParallelException(const std::string& message, int rank = -1);

    // 获取进程号
    int get_rank() const;
};
```

### 6.4 检查点管理

#### DataParallelCheckpoint
```cpp
class DataParallelCheckpoint {
public:
    // 构造函数
    DataParallelCheckpoint(const DataParallelConfig& config);

    // 检查点操作
    void save_checkpoint(const std::string& path,
                        const std::vector<Tensor>& parameters,
                        int global_step, float loss);
    void load_checkpoint(const std::string& path,
                        std::vector<Tensor>& parameters,
                        int& global_step, float& loss);

    // 检查点管理
    bool checkpoint_exists(const std::string& path) const;
    void delete_checkpoint(const std::string& path) const;
};
```

---

## 使用示例

### 基本使用流程

```cpp
#include "megatron.h"

int main(int argc, char** argv) {
    // 1. 初始化MPI
    MPI_Init(&argc, &argv);

    try {
        // 2. 初始化分布式环境
        DistributedConfig config;
        config.world_size = 4;
        config.rank = 0;  // 由MPI自动设置
        config.global_batch_size = 32;
        config.local_batch_size = 8;

        distributed_utils::initialize_distributed(config);

        // 3. 创建模型
        auto model = std::make_shared<Linear>(784, 10);

        // 4. 创建优化器
        auto optimizer = std::make_shared<Adam>(1e-3);

        // 5. 创建数据并行训练器
        DataParallelTrainer trainer(config);
        trainer.setup_model_and_optimizer(model, optimizer);

        // 6. 训练循环
        for (int epoch = 0; epoch < 10; ++epoch) {
            // 模拟数据
            Tensor inputs({config.local_batch_size, 784});
            Tensor targets({config.local_batch_size});
            
            // 训练步骤
            trainer.train_step(inputs, targets);
        }

        // 7. 清理
        distributed_utils::cleanup_distributed();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
```

### 张量并行示例

```cpp
// 创建张量并行模型
TensorParallelModelBuilder builder(2);  // 2-way tensor parallel
builder.set_embed_dim(768);
builder.set_num_heads(12);
builder.set_num_layers(6);

auto model = builder.build_gpt_model();

// 正常使用模型
Tensor input({4, 128});
Tensor output = model->forward(input);
```

### 混合并行示例

```cpp
// 配置混合并行
DataParallelConfig dp_config;
dp_config.world_size = 8;
dp_config.rank = 0;
dp_config.global_batch_size = 64;
dp_config.local_batch_size = 8;

HybridParallelTrainer trainer(dp_config, 2, 2);  // 2-way TP, 2-way PP

// 创建张量并行模型
auto model = build_tensor_parallel_model();
auto optimizer = std::make_shared<Adam>(1e-4);

trainer.setup_model_and_optimizer(model, optimizer);

// 混合并行训练
trainer.hybrid_train_step(inputs, targets);
```

---

## 注意事项

1. **MPI初始化**：在使用任何分布式功能之前，必须先初始化MPI
2. **内存管理**：注意Tensor的生命周期，避免内存泄漏
3. **错误处理**：使用try-catch块处理可能的异常
4. **资源清理**：程序结束时记得清理MPI资源
5. **配置验证**：在使用并行配置前先验证其有效性
6. **性能监控**：使用性能监控工具来分析和优化训练性能
7. **梯度同步**：确保在参数更新前正确同步梯度

---

## 版本历史

- **v1.0.0**：初始版本，支持基本的数据并行和张量并行
- **v1.1.0**：添加混合并行支持和性能优化
- **v1.2.0**：完善文档和示例代码
- **v1.3.0**：添加更多优化器和工具类

---

## 许可证

本项目采用MIT许可证，详见LICENSE文件。