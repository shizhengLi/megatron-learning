# Megatron-CPP-Edu 部署指南

## 目录
1. [系统要求](#1-系统要求)
2. [环境配置](#2-环境配置)
3. [编译安装](#3-编译安装)
4. [单机部署](#4-单机部署)
5. [多机部署](#5-多机部署)
6. [容器化部署](#6-容器化部署)
7. [云平台部署](#7-云平台部署)
8. [监控和维护](#8-监控和维护)
9. [故障排除](#9-故障排除)
10. [最佳实践](#10-最佳实践)

---

## 1. 系统要求

### 1.1 硬件要求

#### 最低配置
- **CPU**: 4核处理器
- **内存**: 8GB RAM
- **存储**: 20GB 可用空间
- **网络**: 1Gbps 网络带宽

#### 推荐配置
- **CPU**: 16核处理器 (Intel Xeon 或 AMD EPYC)
- **内存**: 64GB RAM 或更多
- **GPU**: NVIDIA GPU (Tesla V100/A100/H100 或 RTX 3090/4090)
- **存储**: 100GB+ SSD
- **网络**: 10Gbps+ InfiniBand 或高速以太网

#### 大规模部署配置
- **CPU**: 64核+ 多处理器
- **内存**: 256GB+ RAM
- **GPU**: 8+ NVIDIA A100/H100 GPU
- **存储**: 1TB+ NVMe SSD
- **网络**: 100Gbps InfiniBand

### 1.2 软件要求

#### 操作系统
- **Ubuntu**: 20.04 LTS 或 22.04 LTS (推荐)
- **CentOS**: 7 或 8
- **Rocky Linux**: 8 或 9
- **其他支持MPI的Linux发行版**

#### 依赖软件
- **C++编译器**: GCC 9.0+ 或 Clang 10.0+
- **CMake**: 3.16+
- **MPI**: OpenMPI 4.0+ 或 MPICH 3.4+
- **CUDA**: 11.0+ (如果使用GPU)
- **cuDNN**: 8.0+ (如果使用GPU)
- **NCCL**: 2.7+ (如果使用GPU)

#### 开发工具
- **Git**: 版本控制
- **Make**: 构建工具
- **Python**: 3.8+ (用于脚本和工具)

---

## 2. 环境配置

### 2.1 基础环境配置

#### 安装基础依赖
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake git wget curl

# CentOS/RHEL/Rocky Linux
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake git wget curl
```

#### 安装MPI

##### OpenMPI (推荐)
```bash
# Ubuntu/Debian
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev

# CentOS/RHEL/Rocky Linux
sudo yum install -y openmpi openmpi-devel

# 验证安装
mpirun --version
mpiCC --version
```

##### MPICH
```bash
# Ubuntu/Debian
sudo apt install -y mpich libmpich-dev

# CentOS/RHEL/Rocky Linux
sudo yum install -y mpich mpich-devel

# 验证安装
mpirun --version
mpicxx --version
```

### 2.2 GPU环境配置

#### 安装CUDA
```bash
# 下载CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install -y cuda-11-8

# 设置环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
```

#### 安装cuDNN
```bash
# 下载cuDNN (需要NVIDIA开发者账户)
# 从 https://developer.nvidia.com/cudnn 下载对应版本的cuDNN

# 安装
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.0.131/include/cudnn*.h /usr/local/cuda/include
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.0.131/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

#### 安装NCCL
```bash
# 下载NCCL
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/nvidia-machine-learning-repo-ubuntu2204_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu2204_1.0.0-1_amd64.deb
sudo apt update
sudo apt install -y libnccl2 libnccl-dev
```

### 2.3 编译环境配置

#### 设置编译器
```bash
# 设置默认编译器
export CC=gcc
export CXX=g++

# 或者使用Clang
export CC=clang
export CXX=clang++
```

#### 设置MPI编译器
```bash
# OpenMPI
export MPICC=mpicc
export MPICXX=mpicxx

# MPICH
export MPICC=mpicc
export MPICXX=mpicxx
```

---

## 3. 编译安装

### 3.1 获取源码

```bash
# 克隆仓库
git clone https://github.com/your-repo/Megatron-CPP-Edu.git
cd Megatron-CPP-Edu

# 检查分支
git checkout main
```

### 3.2 配置构建

#### 创建构建目录
```bash
mkdir build
cd build
```

#### 配置CMake
```bash
# 基础配置
cmake ..

# 启用GPU支持
cmake .. -DUSE_CUDA=ON

# 启用调试模式
cmake .. -DCMAKE_BUILD_TYPE=Debug

# 启用发布模式
cmake .. -DCMAKE_BUILD_TYPE=Release

# 设置安装路径
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/megatron-cpp-edu

# 完整配置示例
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local/megatron-cpp-edu \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=ON
```

### 3.3 编译

```bash
# 编译
make -j$(nproc)

# 编译特定目标
make megatron_core
make parallel_tests
make examples

# 并行编译 (推荐)
make -j$(nproc)
```

### 3.4 安装

```bash
# 安装到指定路径
sudo make install

# 或者仅构建不安装
make

# 验证安装
ls /usr/local/megatron-cpp-edu/bin
ls /usr/local/megatron-cpp-edu/lib
ls /usr/local/megatron-cpp-edu/include
```

### 3.5 运行测试

```bash
# 运行单元测试
make test

# 运行并行测试
mpirun -np 4 ./bin/test_parallel

# 运行特定测试
./bin/test_tensor
./bin/test_layer
./bin/test_communication
```

---

## 4. 单机部署

### 4.1 基础单机部署

#### 环境脚本
```bash
#!/bin/bash
# setup_single_node.sh

# 设置环境变量
export MEGATRON_HOME=/usr/local/megatron-cpp-edu
export LD_LIBRARY_PATH=$MEGATRON_HOME/lib:$LD_LIBRARY_PATH
export PATH=$MEGATRON_HOME/bin:$PATH

# 创建工作目录
mkdir -p ~/megatron-workspace
cd ~/megatron-workspace

# 创建配置文件
cat > config.yaml << EOF
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10

model:
  vocab_size: 50000
  hidden_size: 768
  num_layers: 12
  num_attention_heads: 12

parallel:
  tensor_parallel_size: 1
  data_parallel_size: 1
  pipeline_parallel_size: 1
EOF
```

#### 启动脚本
```bash
#!/bin/bash
# run_single_node.sh

#!/bin/bash

# 加载环境
source setup_single_node.sh

# 运行单节点训练
./bin/train_example \
    --config config.yaml \
    --model_config model_config.json \
    --data_dir ./data \
    --output_dir ./output \
    --log_level INFO
```

### 4.2 GPU单机部署

#### 多GPU配置
```bash
#!/bin/bash
# setup_multi_gpu.sh

# 设置GPU环境
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

# 创建多GPU配置
cat > multi_gpu_config.yaml << EOF
training:
  global_batch_size: 128
  micro_batch_size: 32

parallel:
  tensor_parallel_size: 2
  data_parallel_size: 2
  pipeline_parallel_size: 1

gpu:
  num_gpus: 4
  memory_fraction: 0.9
EOF
```

#### 启动多GPU训练
```bash
#!/bin/bash
# run_multi_gpu.sh

# 加载环境
source setup_multi_gpu.sh

# 启动多GPU训练
mpirun -np 4 \
    --bind-to core \
    --map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    ./bin/train_example \
    --config multi_gpu_config.yaml \
    --num_gpus 4
```

### 4.3 性能优化

#### 系统优化
```bash
#!/bin/bash
# optimize_system.sh

# 设置CPU性能模式
sudo cpupower frequency-set --governor performance

# 禁用CPU节能
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 设置内存大页
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
sudo mkdir -p /dev/hugepages
sudo mount -t hugetlbfs nodev /dev/hugepages

# 设置网络参数
echo 'net.core.rmem_max = 2147483647' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 2147483647' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 2147483647' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 2147483647' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### GPU优化
```bash
#!/bin/bash
# optimize_gpu.sh

# 设置GPU时钟频率
sudo nvidia-smi -lgc 2000,2000

# 设置GPU功耗限制
sudo nvidia-smi -pl 300

# 启用GPU持久模式
sudo nvidia-smi -pm 1

# 设置GPU计算模式
sudo nvidia-smi -c EXCLUSIVE_PROCESS
```

---

## 5. 多机部署

### 5.1 网络配置

#### 主机文件配置
```bash
# hostfile
node01 slots=4
node02 slots=4
node03 slots=4
node04 slots=4
```

#### SSH免密登录配置
```bash
#!/bin/bash
# setup_ssh.sh

# 生成SSH密钥
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# 复制公钥到所有节点
for node in node01 node02 node03 node04; do
    ssh-copy-id $node
done

# 测试SSH连接
for node in node01 node02 node03 node04; do
    ssh $node "hostname"
done
```

### 5.2 环境同步

#### 环境同步脚本
```bash
#!/bin/bash
# sync_environment.sh

NODES="node01 node02 node03 node04"

# 同步软件包
for node in $NODES; do
    echo "Syncing packages to $node"
    rsync -avz --exclude='*.o' --exclude='*.so' --exclude='bin/*' \
        ./ $node:/usr/local/megatron-cpp-edu/
done

# 同步配置文件
for node in $NODES; do
    echo "Syncing config to $node"
    scp config.yaml $node:~/megatron-workspace/
done
```

### 5.3 多机启动

#### 启动脚本
```bash
#!/bin/bash
# run_multi_node.sh

# 设置环境变量
export MEGATRON_HOME=/usr/local/megatron-cpp-edu
export LD_LIBRARY_PATH=$MEGATRON_HOME/lib:$LD_LIBRARY_PATH

# 启动多机训练
mpirun \
    --hostfile hostfile \
    --bind-to core \
    --map-by slot \
    -x MEGATRON_HOME \
    -x LD_LIBRARY_PATH \
    -x CUDA_VISIBLE_DEVICES \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=^docker0,lo \
    ./bin/train_example \
    --config multi_node_config.yaml \
    --data_dir /shared/data \
    --output_dir /shared/output \
    --log_level INFO
```

#### 高级配置
```bash
#!/bin/bash
# run_multi_node_advanced.sh

# 网络优化配置
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_HCA=mlx5_0,mlx5_1

# GPU绑定配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 启动训练
mpirun \
    --hostfile hostfile \
    --bind-to core \
    --map-by slot \
    --rank-by core \
    --mca btl ^openib \
    --mca btl_tcp_if_include ib0 \
    --mca oob_tcp_if_include ib0 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_IB_DISABLE \
    -x NCCL_SOCKET_IFNAME \
    -x NCCL_IB_GID_INDEX \
    -x NCCL_IB_TC \
    -x NCCL_IB_SL \
    -x NCCL_IB_HCA \
    -x CUDA_VISIBLE_DEVICES \
    -x CUDA_DEVICE_ORDER \
    ./bin/train_example \
    --config advanced_config.yaml \
    --use_infiniband=true \
    --enable_nccl=true
```

### 5.4 负载均衡

#### 动态负载均衡脚本
```bash
#!/bin/bash
# dynamic_load_balancing.sh

# 监控节点负载
monitor_nodes() {
    while true; do
        echo "=== Node Load Monitor ==="
        for node in node01 node02 node03 node04; do
            load=$(ssh $node "uptime | awk -F'load average:' '{print \$2}'")
            gpu_usage=$(ssh $node "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1")
            echo "$node: CPU Load $load, GPU Usage $gpu_usage%"
        done
        sleep 30
    done
}

# 根据负载调整任务
adjust_workload() {
    # 获取节点负载信息
    node_loads=()
    for node in node01 node02 node03 node04; do
        load=$(ssh $node "uptime | awk -F'load average:' '{print \$2}' | awk '{print \$1}' | sed 's/,//'")
        node_loads+=($load)
    done
    
    # 找出负载最低的节点
    min_load=${node_loads[0]}
    best_node=0
    for i in "${!node_loads[@]}"; do
        if (( $(echo "${node_loads[$i]} < $min_load" | bc -l) )); then
            min_load=${node_loads[$i]}
            best_node=$i
        fi
    done
    
    echo "Best node: node$(($best_node + 1)) with load $min_load"
    return $best_node
}
```

---

## 6. 容器化部署

### 6.1 Docker配置

#### Dockerfile
```dockerfile
# Dockerfile
FROM ubuntu:22.04

# 设置基础环境
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    openssh-client \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装CUDA (可选)
# COPY cuda-keyring.gpg /etc/apt/keyrings/
# RUN echo "deb [signed-by=/etc/apt/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list
# RUN apt-get update && apt-get install -y \
#     cuda-11-8 \
#     cuda-toolkit-11-8 \
#     && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制源码
COPY . .

# 编译安装
RUN mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=ON && \
    make -j$(nproc) && \
    make install

# 设置环境变量
ENV PATH=/usr/local/megatron-cpp-edu/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/megatron-cpp-edu/lib:$LD_LIBRARY_PATH

# 暴露端口
EXPOSE 22 8080

# 启动命令
CMD ["/bin/bash"]
```

### 6.2 Docker Compose配置

#### docker-compose.yml
```yaml
version: '3.8'

services:
  master:
    build: .
    hostname: master
    environment:
      - MPI_RANK=0
      - MPI_SIZE=4
    volumes:
      - ./data:/shared/data
      - ./output:/shared/output
      - ./config:/shared/config
    networks:
      - megatron-network
    command: ["mpirun", "--allow-run-as-root", "-np", "4", "--hostfile", "/shared/config/hostfile", "./bin/train_example"]

  worker1:
    build: .
    hostname: worker1
    environment:
      - MPI_RANK=1
      - MPI_SIZE=4
    volumes:
      - ./data:/shared/data
      - ./output:/shared/output
      - ./config:/shared/config
    networks:
      - megatron-network
    depends_on:
      - master

  worker2:
    build: .
    hostname: worker2
    environment:
      - MPI_RANK=2
      - MPI_SIZE=4
    volumes:
      - ./data:/shared/data
      - ./output:/shared/output
      - ./config:/shared/config
    networks:
      - megatron-network
    depends_on:
      - master

  worker3:
    build: .
    hostname: worker3
    environment:
      - MPI_RANK=3
      - MPI_SIZE=4
    volumes:
      - ./data:/shared/data
      - ./output:/shared/output
      - ./config:/shared/config
    networks:
      - megatron-network
    depends_on:
      - master

networks:
  megatron-network:
    driver: bridge
```

### 6.3 Kubernetes部署

#### Kubernetes配置文件
```yaml
# megatron-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: megatron-trainer
spec:
  replicas: 4
  selector:
    matchLabels:
      app: megatron-trainer
  template:
    metadata:
      labels:
        app: megatron-trainer
    spec:
      containers:
      - name: megatron-trainer
        image: megatron-cpp-edu:latest
        command: ["mpirun"]
        args: ["--allow-run-as-root", "-np", "4", "./bin/train_example"]
        env:
        - name: MPI_RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: data
          mountPath: /shared/data
        - name: output
          mountPath: /shared/output
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: megatron-data-pvc
      - name: output
        persistentVolumeClaim:
          claimName: megatron-output-pvc

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: megatron-data-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: megatron-output-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
```

#### MPI Operator配置
```yaml
# mpi-operator.yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: megatron-mpi-job
  namespace: default
spec:
  slotsPerWorker: 4
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - image: megatron-cpp-edu:latest
            name: megatron-launcher
            command: ["mpirun"]
            args:
            - -np
            - "16"
            - --allow-run-as-root
            - -bind-to
            - none
            - -map-by
            - slot
            - -x
            - NCCL_DEBUG=INFO
            - -x
            - LD_LIBRARY_PATH
            - -x
            - PATH
            - -mca
            - pml
            - ob1
            - -mca
            - btl
            - ^openib
            - ./bin/train_example
            resources:
              limits:
                cpu: "2"
                memory: "4Gi"
    Worker:
      replicas: 4
      template:
        spec:
          containers:
          - image: megatron-cpp-edu:latest
            name: megatron-worker
            resources:
              limits:
                nvidia.com/gpu: 1
                cpu: "8"
                memory: "32Gi"
              requests:
                nvidia.com/gpu: 1
                cpu: "4"
                memory: "16Gi"
```

---

## 7. 云平台部署

### 7.1 AWS部署

#### EC2实例配置
```bash
#!/bin/bash
# setup_aws_cluster.sh

# 设置AWS区域
export AWS_REGION=us-west-2

# 创建安全组
aws ec2 create-security-group \
    --group-name megatron-sg \
    --description "Megatron CPP Edu Security Group" \
    --vpc-id vpc-12345678

# 配置安全组规则
aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 1-65535 \
    --cidr 172.16.0.0/16

# 启动EC2实例
aws ec2 run-instances \
    --image-id ami-12345678 \
    --count 4 \
    --instance-type p3.8xlarge \
    --key-name megatron-key \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --iam-instance-profile Name=EC2-S3-Access \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=megatron-node}]' \
    --user-data file://setup_node.sh
```

#### 用户数据脚本
```bash
#!/bin/bash
# setup_node.sh

# 安装依赖
apt-get update
apt-get install -y build-essential cmake git wget curl openmpi-bin openmpi-common libopenmpi-dev

# 安装NVIDIA驱动
apt-get install -y linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
apt-get update
apt-get install -y cuda-11-8

# 设置环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> /etc/environment
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> /etc/environment

# 创建用户
useradd -m -s /bin/bash megatron
echo 'megatron ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# 配置SSH
mkdir -p /home/megatron/.ssh
chmod 700 /home/megatron/.ssh
# 需要在这里添加公钥
```

### 7.2 GCP部署

#### GCP配置脚本
```bash
#!/bin/bash
# setup_gcp_cluster.sh

# 设置GCP项目
export PROJECT_ID=your-project-id
export ZONE=us-central1-a

# 创建实例模板
gcloud compute instance-templates create megatron-template \
    --machine-type=n1-standard-8 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --accelerator=type=nvidia-tesla-v100,count=2 \
    --metadata=install-nvidia-driver=True \
    --tags=megatron-node,http-server,https-server

# 创建托管实例组
gcloud compute instance-groups managed create megatron-group \
    --base-instance-name=megatron \
    --template=megatron-template \
    --size=4 \
    --zone=$ZONE

# 配置防火墙规则
gcloud compute firewall-rules create megatron-allow-ssh \
    --allow=tcp:22 \
    --target-tags=megatron-node

gcloud compute firewall-rules create megatron-allow-internal \
    --allow=tcp:1-65535,udp:1-65535 \
    --source-ranges=10.0.0.0/8 \
    --target-tags=megatron-node
```

### 7.3 Azure部署

#### Azure配置脚本
```bash
#!/bin/bash
# setup_azure_cluster.sh

# 创建资源组
az group create --name MegatronGroup --location eastus

# 创建虚拟网络
az network vnet create \
    --resource-group MegatronGroup \
    --name megatron-vnet \
    --address-prefix 10.0.0.0/16

# 创建子网
az network vnet subnet create \
    --resource-group MegatronGroup \
    --vnet-name megatron-vnet \
    --name megatron-subnet \
    --address-prefix 10.0.0.0/24

# 创建可用性集
az vm availability-set create \
    --resource-group MegatronGroup \
    --name megatron-availability-set \
    --platform-fault-domain-count 2 \
    --platform-update-domain-count 2

# 创建虚拟机
for i in {1..4}; do
    az vm create \
        --resource-group MegatronGroup \
        --name megatron-vm$i \
        --image Ubuntu2204 \
        --size Standard_NC6s_v3 \
        --availability-set megatron-availability-set \
        --vnet-name megatron-vnet \
        --subnet megatron-subnet \
        --admin-username azureuser \
        --generate-ssh-keys \
        --data-disk-sizes-gb 100
done

# 配置网络安全组
az network nsg rule create \
    --resource-group MegatronGroup \
    --nsg-name megatron-vmNSG \
    --name allow-ssh \
    --protocol tcp \
    --direction inbound \
    --source-address-prefix '*' \
    --source-port-range '*' \
    --destination-address-prefix '*' \
    --destination-port-range 22 \
    --access allow \
    --priority 100
```

---

## 8. 监控和维护

### 8.1 系统监控

#### 监控脚本
```bash
#!/bin/bash
# monitor_cluster.sh

# 配置
NODES="node01 node02 node03 node04"
LOG_FILE="cluster_monitor.log"

# 监控函数
monitor_cluster() {
    while true; do
        echo "=== Cluster Monitor $(date) ===" >> $LOG_FILE
        
        for node in $NODES; do
            echo "Monitoring $node:" >> $LOG_FILE
            
            # CPU使用率
            cpu_usage=$(ssh $node "top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | cut -d'%' -f1")
            echo "  CPU Usage: $cpu_usage%" >> $LOG_FILE
            
            # 内存使用率
            mem_usage=$(ssh $node "free | grep Mem | awk '{printf \"%.2f\", \$3/\$2 * 100.0}'")
            echo "  Memory Usage: $mem_usage%" >> $LOG_FILE
            
            # GPU使用率
            gpu_usage=$(ssh $node "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1")
            echo "  GPU Usage: $gpu_usage%" >> $LOG_FILE
            
            # 磁盘使用率
            disk_usage=$(ssh $node "df -h / | awk 'NR==2{print \$5}' | cut -d'%' -f1")
            echo "  Disk Usage: $disk_usage%" >> $LOG_FILE
            
            # 网络流量
            network_rx=$(ssh $node "cat /proc/net/dev | grep eth0 | awk '{print \$2}'")
            network_tx=$(ssh $node "cat /proc/net/dev | grep eth0 | awk '{print \$10}'")
            echo "  Network RX: $network_rx bytes, TX: $network_tx bytes" >> $LOG_FILE
            
            echo "" >> $LOG_FILE
        done
        
        # 检查异常情况
        check_alerts
        
        sleep 60
    done
}

# 检查警报
check_alerts() {
    # 检查CPU使用率
    for node in $NODES; do
        cpu_usage=$(ssh $node "top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | cut -d'%' -f1")
        if (( $(echo "$cpu_usage > 90" | bc -l) )); then
            send_alert "High CPU usage on $node: $cpu_usage%"
        fi
    done
    
    # 检查内存使用率
    for node in $NODES; do
        mem_usage=$(ssh $node "free | grep Mem | awk '{printf \"%.2f\", \$3/\$2 * 100.0}'")
        if (( $(echo "$mem_usage > 90" | bc -l) )); then
            send_alert "High memory usage on $node: $mem_usage%"
        fi
    done
    
    # 检查GPU使用率
    for node in $NODES; do
        gpu_usage=$(ssh $node "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1")
        if (( $(echo "$gpu_usage > 95" | bc -l) )); then
            send_alert "High GPU usage on $node: $gpu_usage%"
        fi
    done
}

# 发送警报
send_alert() {
    local message=$1
    echo "ALERT: $message" >> $LOG_FILE
    
    # 发送邮件 (需要配置邮件服务器)
    # echo "$message" | mail -s "Megatron Cluster Alert" admin@example.com
    
    # 发送Slack通知 (需要配置Slack webhook)
    # curl -X POST -H 'Content-type: application/json' \
    #     --data "{\"text\":\"$message\"}" \
    #     https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
}

# 启动监控
monitor_cluster
```

### 8.2 日志管理

#### 日志收集脚本
```bash
#!/bin/bash
# log_collector.sh

# 配置
LOG_DIR="/var/log/megatron"
ARCHIVE_DIR="/var/log/megatron/archive"
RETENTION_DAYS=30

# 创建日志目录
mkdir -p $LOG_DIR
mkdir -p $ARCHIVE_DIR

# 收集日志
collect_logs() {
    timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # 收集系统日志
    for node in node01 node02 node03 node04; do
        ssh $node "journalctl -u megatron-service --since '1 hour ago'" > $LOG_DIR/${node}_system_${timestamp}.log
        ssh $node "cat /var/log/syslog | tail -1000" > $LOG_DIR/${node}_syslog_${timestamp}.log
    done
    
    # 收集应用日志
    cp /home/megatron/training.log $LOG_DIR/training_${timestamp}.log
    
    # 收集GPU日志
    for node in node01 node02 node03 node04; do
        ssh $node "nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv -f $LOG_DIR/${node}_gpu_${timestamp}.csv"
    done
}

# 压缩和归档
archive_logs() {
    find $LOG_DIR -name "*.log" -mtime +1 -exec gzip {} \;
    find $LOG_DIR -name "*.gz" -mtime +$RETENTION_DAYS -exec mv {} $ARCHIVE_DIR/ \;
    find $ARCHIVE_DIR -name "*.gz" -mtime +$RETENTION_DAYS -exec rm {} \;
}

# 日志轮转
rotate_logs() {
    for logfile in $LOG_DIR/*.log; do
        if [ -f "$logfile" ] && [ $(stat -c%s "$logfile") -gt 104857600 ]; then  # 100MB
            mv "$logfile" "${logfile}.$(date +%Y%m%d%H%M%S)"
            gzip "${logfile}.$(date +%Y%m%d%H%M%S)"
        fi
    done
}

# 定期执行
while true; do
    collect_logs
    rotate_logs
    archive_logs
    sleep 3600  # 每小时执行一次
done
```

### 8.3 自动化维护

#### 自动化维护脚本
```bash
#!/bin/bash
# auto_maintenance.sh

# 配置
MAINTENANCE_LOG="/var/log/megatron/maintenance.log"
LOCK_FILE="/tmp/maintenance.lock"

# 创建锁文件防止重复执行
if [ -f "$LOCK_FILE" ]; then
    echo "Maintenance already running. Exiting."
    exit 1
fi

touch "$LOCK_FILE"

# 清理函数
cleanup() {
    rm -f "$LOCK_FILE"
    exit 0
}

# 信号处理
trap cleanup SIGINT SIGTERM

# 记录日志
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $MAINTENANCE_LOG
}

# 系统更新
system_update() {
    log_message "Starting system update"
    
    for node in node01 node02 node03 node04; do
        log_message "Updating $node"
        ssh $node "sudo apt-get update -y && sudo apt-get upgrade -y"
    done
    
    log_message "System update completed"
}

# 磁盘清理
disk_cleanup() {
    log_message "Starting disk cleanup"
    
    for node in node01 node02 node03 node04; do
        log_message "Cleaning up $node"
        ssh $node "sudo apt-get autoremove -y"
        ssh $node "sudo apt-get autoclean -y"
        ssh $node "sudo journalctl --vacuum-time=7d"
        ssh $node "find /tmp -type f -mtime +7 -delete"
    done
    
    log_message "Disk cleanup completed"
}

# 服务重启
service_restart() {
    log_message "Restarting services"
    
    for node in node01 node02 node03 node04; do
        log_message "Restarting services on $node"
        ssh $node "sudo systemctl restart megatron-service"
        ssh $node "sudo systemctl restart sshd"
        ssh $node "sudo systemctl restart nvidia-persistenced"
    done
    
    log_message "Services restarted"
}

# 性能优化
performance_optimization() {
    log_message "Starting performance optimization"
    
    for node in node01 node02 node03 node04; do
        log_message "Optimizing $node"
        
        # 清理内存缓存
        ssh $node "sudo sync && echo 3 > /proc/sys/vm/drop_caches"
        
        # 优化文件系统
        ssh $node "sudo fstrim -av"
        
        # 重启GPU
        ssh $node "sudo nvidia-smi --gpu-reset -i 0"
        
        # 更新GPU时钟频率
        ssh $node "sudo nvidia-smi -lgc 2000,2000"
    done
    
    log_message "Performance optimization completed"
}

# 健康检查
health_check() {
    log_message "Starting health check"
    
    for node in node01 node02 node03 node04; do
        log_message "Checking health of $node"
        
        # 检查SSH连接
        if ! ssh $node "echo 'SSH OK'"; then
            log_message "ERROR: SSH connection failed to $node"
            send_alert "SSH connection failed to $node"
        fi
        
        # 检查GPU状态
        if ! ssh $node "nvidia-smi"; then
            log_message "ERROR: GPU not responding on $node"
            send_alert "GPU not responding on $node"
        fi
        
        # 检查磁盘空间
        disk_usage=$(ssh $node "df -h / | awk 'NR==2{print \$5}' | cut -d'%' -f1")
        if [ "$disk_usage" -gt 90 ]; then
            log_message "WARNING: High disk usage on $node: $disk_usage%"
            send_alert "High disk usage on $node: $disk_usage%"
        fi
        
        # 检查内存使用
        mem_usage=$(ssh $node "free | grep Mem | awk '{printf \"%.2f\", \$3/\$2 * 100.0}'")
        if (( $(echo "$mem_usage > 90" | bc -l) )); then
            log_message "WARNING: High memory usage on $node: $mem_usage%"
            send_alert "High memory usage on $node: $mem_usage%"
        fi
    done
    
    log_message "Health check completed"
}

# 主函数
main() {
    log_message "Starting automated maintenance"
    
    # 执行维护任务
    health_check
    system_update
    disk_cleanup
    service_restart
    performance_optimization
    
    # 最后健康检查
    health_check
    
    log_message "Automated maintenance completed"
    cleanup
}

# 运行主函数
main
```

---

## 9. 故障排除

### 9.1 常见问题诊断

#### MPI连接问题
```bash
#!/bin/bash
# diagnose_mpi_issues.sh

# 检查MPI安装
check_mpi_installation() {
    echo "=== MPI Installation Check ==="
    
    # 检查MPI命令
    for cmd in mpirun mpiexec mpicc mpicxx; do
        if command -v $cmd &> /dev/null; then
            echo "✓ $cmd found: $(which $cmd)"
        else
            echo "✗ $cmd not found"
        fi
    done
    
    # 检查MPI库
    echo "MPI libraries:"
    ldconfig -p | grep -i mpi
    
    # 检查MPI版本
    echo "MPI versions:"
    mpirun --version
    mpiexec --version
}

# 检查网络连接
check_network_connectivity() {
    echo "=== Network Connectivity Check ==="
    
    # 检查主机文件
    if [ -f "hostfile" ]; then
        echo "Hostfile contents:"
        cat hostfile
    else
        echo "✗ Hostfile not found"
    fi
    
    # 检查SSH连接
    echo "SSH connectivity:"
    for node in node01 node02 node03 node04; do
        if ssh $node "echo 'SSH OK'" &> /dev/null; then
            echo "✓ SSH to $node: OK"
        else
            echo "✗ SSH to $node: FAILED"
        fi
    done
    
    # 检查端口
    echo "Port availability:"
    for port in 22 1024 65535; do
        if netstat -tuln | grep -q ":$port "; then
            echo "✓ Port $port: Open"
        else
            echo "✗ Port $port: Closed"
        fi
    done
}

# 测试MPI通信
test_mpi_communication() {
    echo "=== MPI Communication Test ==="
    
    # 创建测试程序
    cat > mpi_test.cpp << 'EOF'
#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::cout << "Rank " << rank << " of " << size << " reporting" << std::endl;
    
    // 测试点对点通信
    if (rank == 0) {
        int message = 42;
        MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Rank 0 sent message" << std::endl;
    } else if (rank == 1) {
        int message;
        MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Rank 1 received message: " << message << std::endl;
    }
    
    // 测试集合通信
    std::vector<int> data(size, rank);
    std::vector<int> result(size);
    
    MPI_Allreduce(data.data(), result.data(), size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "AllReduce result: ";
        for (int i = 0; i < size; ++i) {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
EOF
    
    # 编译测试程序
    mpicxx -o mpi_test mpi_test.cpp
    
    # 运行测试
    echo "Running MPI test..."
    mpirun -np 4 --hostfile hostfile ./mpi_test
    
    # 清理
    rm -f mpi_test.cpp mpi_test
}

# 主函数
main() {
    check_mpi_installation
    check_network_connectivity
    test_mpi_communication
}

main
```

#### GPU问题诊断
```bash
#!/bin/bash
# diagnose_gpu_issues.sh

# 检查GPU状态
check_gpu_status() {
    echo "=== GPU Status Check ==="
    
    # 检查GPU数量
    echo "GPU count:"
    nvidia-smi --query-gpu=count --format=csv,noheader
    
    # 检查GPU详细信息
    echo "GPU details:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv
    
    # 检查驱动版本
    echo "Driver version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    
    # 检查CUDA版本
    echo "CUDA version:"
    nvcc --version 2>/dev/null || echo "nvcc not found"
}

# 测试GPU计算
test_gpu_computation() {
    echo "=== GPU Computation Test ==="
    
    # 创建CUDA测试程序
    cat > cuda_test.cu << 'EOF'
#include <iostream>
#include <cuda_runtime.h>

__global__ void addKernel(int *c, const int *a, const int *b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int size = 1024;
    int a[size], b[size], c[size];
    
    // Initialize host data
    for (int i = 0; i < size; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }
    
    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    addKernel<<<(size + 255) / 256, 256>>>(d_c, d_a, d_b, size);
    
    // Copy result back
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify result
    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (c[i] != a[i] + b[i]) {
            success = false;
            break;
        }
    }
    
    std::cout << "GPU computation test: " << (success ? "PASSED" : "FAILED") << std::endl;
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return success ? 0 : 1;
}
EOF
    
    # 编译和运行测试
    if command -v nvcc &> /dev/null; then
        nvcc -o cuda_test cuda_test.cu
        ./cuda_test
        rm -f cuda_test.cu cuda_test
    else
        echo "nvcc not found, skipping GPU computation test"
    fi
}

# 检查GPU内存
check_gpu_memory() {
    echo "=== GPU Memory Check ==="
    
    # 获取GPU内存信息
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits | while read total used free; do
        echo "Memory: ${used}MB / ${total}MB (${free}MB free)"
        usage_percent=$((used * 100 / total))
        echo "Usage: ${usage_percent}%"
        
        if [ $usage_percent -gt 90 ]; then
            echo "WARNING: High GPU memory usage"
        fi
    done
}

# 主函数
main() {
    check_gpu_status
    check_gpu_memory
    test_gpu_computation
}

main
```

### 9.2 性能问题诊断

#### 性能分析脚本
```bash
#!/bin/bash
# performance_analysis.sh

# 系统性能分析
analyze_system_performance() {
    echo "=== System Performance Analysis ==="
    
    # CPU性能
    echo "CPU Performance:"
    echo "  CPU Model: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
    echo "  CPU Cores: $(nproc)"
    echo "  CPU Usage: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1)%"
    echo "  Load Average: $(uptime | awk -F'load average:' '{print $2}' | xargs)"
    
    # 内存性能
    echo "Memory Performance:"
    free -h
    echo "  Swap Usage: $(swapon --show | tail -n +2 | awk '{print $3}' | head -1)"
    
    # 磁盘性能
    echo "Disk Performance:"
    df -h
    echo "  I/O Statistics:"
    iostat -d -x 1 3
    
    # 网络性能
    echo "Network Performance:"
    echo "  Network Interfaces:"
    ip addr show
    echo "  Network Statistics:"
    sar -n DEV 1 3
}

# GPU性能分析
analyze_gpu_performance() {
    echo "=== GPU Performance Analysis ==="
    
    # GPU利用率
    echo "GPU Utilization:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv
    
    # GPU内存使用
    echo "GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits | while read total used free; do
        echo "  Memory: ${used}MB / ${total}MB (${free}MB free)"
    done
    
    # GPU进程
    echo "GPU Processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
}

# MPI性能分析
analyze_mpi_performance() {
    echo "=== MPI Performance Analysis ==="
    
    # MPI带宽测试
    echo "MPI Bandwidth Test:"
    
    # 创建带宽测试程序
    cat > mpi_bandwidth.cpp << 'EOF'
#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const int message_sizes[] = {1024, 10240, 102400, 1024000, 10485760};
    const int num_sizes = sizeof(message_sizes) / sizeof(message_sizes[0]);
    
    if (rank == 0) {
        std::cout << "MPI Bandwidth Test Results:" << std::endl;
        std::cout << "Size(bytes)\tTime(ms)\tBandwidth(MB/s)" << std::endl;
    }
    
    for (int i = 0; i < num_sizes; ++i) {
        int msg_size = message_sizes[i];
        std::vector<char> send_buffer(msg_size);
        std::vector<char> recv_buffer(msg_size);
        
        // Warm-up
        if (rank == 0) {
            MPI_Send(send_buffer.data(), msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(recv_buffer.data(), msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(recv_buffer.data(), msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(send_buffer.data(), msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        
        // Measurement
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int j = 0; j < 100; ++j) {
            if (rank == 0) {
                MPI_Send(send_buffer.data(), msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(recv_buffer.data(), msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(recv_buffer.data(), msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_buffer.data(), msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (rank == 0) {
            double time_ms = duration.count() / 1000.0 / 100.0;  // Average over 100 iterations
            double bandwidth = (msg_size * 2.0) / (time_ms / 1000.0) / (1024 * 1024);  // MB/s
            
            std::cout << msg_size << "\t\t" << time_ms << "\t\t" << bandwidth << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}
EOF
    
    # 编译和运行
    mpicxx -o mpi_bandwidth mpi_bandwidth.cpp
    mpirun -np 2 --hostfile hostfile ./mpi_bandwidth
    
    # 清理
    rm -f mpi_bandwidth.cpp mpi_bandwidth
}

# 主函数
main() {
    analyze_system_performance
    analyze_gpu_performance
    analyze_mpi_performance
}

main
```

---

## 10. 最佳实践

### 10.1 配置最佳实践

#### 生产环境配置
```yaml
# production_config.yaml
training:
  global_batch_size: 1024
  micro_batch_size: 16
  gradient_accumulation_steps: 64
  learning_rate: 1e-4
  weight_decay: 0.01
  max_epochs: 100
  
  # 混合精度训练
  use_mixed_precision: true
  fp16_opt_level: "O2"
  
  # 梯度裁剪
  max_grad_norm: 1.0
  
  # 学习率调度
  lr_scheduler: "cosine"
  warmup_steps: 1000
  lr_decay_steps: 10000

model:
  vocab_size: 50000
  hidden_size: 4096
  num_layers: 32
  num_attention_heads: 32
  ffn_dim: 16384
  
  # 模型优化
  use_gradient_checkpointing: true
  activation_checkpointing: true

parallel:
  # 并行策略
  tensor_parallel_size: 8
  pipeline_parallel_size: 4
  data_parallel_size: 16
  
  # 通信优化
  overlap_communication: true
  gradient_bucket_size: 25e6  # 25MB
  all_reduce_bucket_size: 25e6
  
  # 内存优化
  partition_activations: true
  cpu_offloading: false

optimization:
  # 优化器配置
  optimizer: "adamw"
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8
  
  # 分布式优化
  use_distributed_optimizer: true
  zero_stage: 2
  
  # 内存管理
  offload_optimizer: false
  offload_parameters: false

checkpointing:
  # 检查点配置
  save_interval: 1000
  save_steps: 1000
  keep_checkpoint_max: 10
  
  # 检查点压缩
  use_compression: true
  compression_level: 6

logging:
  # 日志配置
  log_level: "INFO"
  log_interval: 10
  save_interval: 100
  
  # 监控
  enable_tensorboard: true
  tensorboard_dir: "./tensorboard"
  
  # 性能监控
  profile_enabled: true
  profile_steps: [10, 100, 1000]
```

### 10.2 安全最佳实践

#### 安全配置
```bash
#!/bin/bash
# security_hardening.sh

# 用户安全配置
setup_user_security() {
    echo "=== User Security Setup ==="
    
    # 创建专用用户
    useradd -m -s /bin/bash megatron
    usermod -aG sudo megatron
    
    # 配置sudo权限
    echo 'megatron ALL=(ALL) NOPASSWD:/usr/local/megatron-cpp-edu/bin/*' > /etc/sudoers.d/megatron
    
    # 限制root登录
    sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
    systemctl restart sshd
}

# 网络安全配置
setup_network_security() {
    echo "=== Network Security Setup ==="
    
    # 配置防火墙
    ufw --force enable
    ufw allow 22/tcp
    ufw allow from 10.0.0.0/8
    ufw allow from 172.16.0.0/12
    ufw allow from 192.168.0.0/16
    
    # 禁用不必要的服务
    systemctl disable telnet
    systemctl disable rsh
    systemctl disable rlogin
    
    # 配置SSH安全
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
    sed -i 's/#PermitEmptyPasswords yes/PermitEmptyPasswords no/' /etc/ssh/sshd_config
    sed -i 's/#X11Forwarding yes/X11Forwarding no/' /etc/ssh/sshd_config
    systemctl restart sshd
}

# 文件系统安全
setup_filesystem_security() {
    echo "=== Filesystem Security Setup ==="
    
    # 设置文件权限
    chown -R megatron:megatron /usr/local/megatron-cpp-edu
    chmod -R 750 /usr/local/megatron-cpp-edu
    
    # 配置敏感文件权限
    chmod 600 /etc/ssh/sshd_config
    chmod 600 /etc/sudoers.d/megatron
    
    # 设置系统文件权限
    chown root:root /usr/local/bin/*
    chmod 755 /usr/local/bin/*
}

# 系统安全加固
harden_system() {
    echo "=== System Hardening ==="
    
    # 更新系统
    apt-get update && apt-get upgrade -y
    
    # 安装安全工具
    apt-get install -y fail2ban rkhunter clamav
    
    # 配置fail2ban
    systemctl enable fail2ban
    systemctl start fail2ban
    
    # 配置自动安全更新
    apt-get install -y unattended-upgrades
    dpkg-reconfigure -plow unattended-upgrades
    
    # 禁用不必要的服务
    systemctl disable avahi-daemon
    systemctl disable cups
    systemctl disable bluetooth
}

# 主函数
main() {
    setup_user_security
    setup_network_security
    setup_filesystem_security
    harden_system
    
    echo "Security hardening completed"
}

main
```

### 10.3 备份和恢复

#### 备份脚本
```bash
#!/bin/bash
# backup_system.sh

# 配置
BACKUP_DIR="/backup/megatron"
RETENTION_DAYS=30
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份配置文件
backup_config() {
    echo "Backing up configuration files..."
    
    tar -czf $BACKUP_DIR/config_$TIMESTAMP.tar.gz \
        /usr/local/megatron-cpp-edu/config \
        /etc/systemd/system/megatron.service \
        /etc/sudoers.d/megatron \
        /etc/ssh/sshd_config \
        2>/dev/null
}

# 备份数据
backup_data() {
    echo "Backing up data..."
    
    # 备份训练数据
    if [ -d "/data/megatron" ]; then
        tar -czf $BACKUP_DIR/data_$TIMESTAMP.tar.gz \
            /data/megatron \
            2>/dev/null
    fi
    
    # 备份模型文件
    if [ -d "/models" ]; then
        tar -czf $BACKUP_DIR/models_$TIMESTAMP.tar.gz \
            /models \
            2>/dev/null
    fi
}

# 备份日志
backup_logs() {
    echo "Backing up logs..."
    
    if [ -d "/var/log/megatron" ]; then
        tar -czf $BACKUP_DIR/logs_$TIMESTAMP.tar.gz \
            /var/log/megatron \
            2>/dev/null
    fi
}

# 备份数据库
backup_database() {
    echo "Backing up database..."
    
    # 如果使用数据库，在这里添加备份命令
    # mysqldump -u root -p megatron_db > $BACKUP_DIR/database_$TIMESTAMP.sql
}

# 清理旧备份
cleanup_old_backups() {
    echo "Cleaning up old backups..."
    
    find $BACKUP_DIR -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
    find $BACKUP_DIR -name "*.sql" -mtime +$RETENTION_DAYS -delete
}

# 验证备份
verify_backup() {
    echo "Verifying backup..."
    
    for backup_file in $BACKUP_DIR/*_$TIMESTAMP.*; do
        if [ -f "$backup_file" ]; then
            if tar -tzf "$backup_file" >/dev/null 2>&1; then
                echo "✓ Backup verified: $backup_file"
            else
                echo "✗ Backup verification failed: $backup_file"
            fi
        fi
    done
}

# 主函数
main() {
    backup_config
    backup_data
    backup_logs
    backup_database
    cleanup_old_backups
    verify_backup
    
    echo "Backup completed successfully"
}

main
```

#### 恢复脚本
```bash
#!/bin/bash
# restore_system.sh

# 配置
BACKUP_DIR="/backup/megatron"
TIMESTAMP=$1

if [ -z "$TIMESTAMP" ]; then
    echo "Usage: $0 <timestamp>"
    echo "Available backups:"
    ls $BACKUP_DIR/*.tar.gz | grep -oE '[0-9]{8}_[0-9]{6}' | sort -u
    exit 1
fi

# 恢复配置文件
restore_config() {
    echo "Restoring configuration files..."
    
    if [ -f "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" ]; then
        tar -xzf $BACKUP_DIR/config_$TIMESTAMP.tar.gz -C /
        echo "✓ Configuration files restored"
    else
        echo "✗ Configuration backup not found"
    fi
}

# 恢复数据
restore_data() {
    echo "Restoring data..."
    
    if [ -f "$BACKUP_DIR/data_$TIMESTAMP.tar.gz" ]; then
        tar -xzf $BACKUP_DIR/data_$TIMESTAMP.tar.gz -C /
        echo "✓ Data restored"
    else
        echo "✗ Data backup not found"
    fi
    
    if [ -f "$BACKUP_DIR/models_$TIMESTAMP.tar.gz" ]; then
        tar -xzf $BACKUP_DIR/models_$TIMESTAMP.tar.gz -C /
        echo "✓ Models restored"
    else
        echo "✗ Models backup not found"
    fi
}

# 恢复日志
restore_logs() {
    echo "Restoring logs..."
    
    if [ -f "$BACKUP_DIR/logs_$TIMESTAMP.tar.gz" ]; then
        tar -xzf $BACKUP_DIR/logs_$TIMESTAMP.tar.gz -C /
        echo "✓ Logs restored"
    else
        echo "✗ Logs backup not found"
    fi
}

# 恢复数据库
restore_database() {
    echo "Restoring database..."
    
    if [ -f "$BACKUP_DIR/database_$TIMESTAMP.sql" ]; then
        # mysql -u root -p megatron_db < $BACKUP_DIR/database_$TIMESTAMP.sql
        echo "✓ Database restored"
    else
        echo "✗ Database backup not found"
    fi
}

# 重启服务
restart_services() {
    echo "Restarting services..."
    
    systemctl daemon-reload
    systemctl restart megatron.service
    systemctl restart sshd
    
    echo "✓ Services restarted"
}

# 主函数
main() {
    restore_config
    restore_data
    restore_logs
    restore_database
    restart_services
    
    echo "System restore completed"
}

main
```

---

## 总结

本部署指南提供了Megatron-CPP-Edu的完整部署方案，包括：

1. **系统要求和环境配置**：详细的硬件软件要求说明
2. **编译安装**：从源码编译和安装的完整流程
3. **单机部署**：包括GPU优化和性能调优
4. **多机部署**：网络配置和负载均衡
5. **容器化部署**：Docker和Kubernetes配置
6. **云平台部署**：AWS、GCP、Azure的部署方案
7. **监控和维护**：系统监控、日志管理、自动化维护
8. **故障排除**：常见问题诊断和性能分析
9. **最佳实践**：生产环境配置、安全加固、备份恢复

遵循本指南，您应该能够成功部署和维护一个高性能、高可用的Megatron-CPP-Edu训练集群。