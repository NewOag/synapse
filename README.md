# synapse

用 Rust 记录机器学习与神经网络学习过程的代码仓库。

## 结构

```
synapse/
├── common/                        # 公共工具库
├── 01_foundations/                # 数学基础
│   ├── linear_algebra/
│   ├── calculus/
│   └── probability/
├── 02_classical_ml/               # 经典机器学习
│   ├── linear_regression/
│   ├── logistic_regression/
│   ├── svm/
│   ├── decision_tree/
│   └── ensemble/
├── 03_neural_networks/            # 神经网络基础
│   ├── perceptron/
│   ├── backpropagation/
│   └── mlp/
├── 04_deep_learning/              # 深度学习
│   ├── cnn/
│   ├── rnn_lstm/
│   ├── attention/
│   └── transformer/
├── 05_optimization/               # 优化器
│   ├── sgd/
│   ├── adam/
│   └── lr_schedulers/
└── 06_projects/                   # 实战项目
```

## 运行

```bash
# 运行某个具体算法
cargo run -p ml-linear-regression

# 编译整个 workspace
cargo build --workspace
```

## 每个算法目录

- `src/main.rs` — 实现代码
- `notes.md` — 原理推导与笔记
