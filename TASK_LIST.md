# 任务清单 / Task List

本仓库基于《从零开始构建大型语言模型》（*Build a Large Language Model From Scratch*）一书，涵盖从数据处理到模型预训练、微调的完整流程。以下是本仓库可完成的任务清单：

---

## 核心章节任务

### 第2章：处理文本数据
- 实现文本分词器（Tokenizer）
- 实现字节对编码（Byte Pair Encoding，BPE）分词器（从零开始）
- 构建用于 GPT 模型训练的数据加载器（DataLoader）
- 理解嵌入层（Embedding Layer）与线性层（Linear Layer）的区别

### 第3章：实现注意力机制
- 实现简单的自注意力机制（Self-Attention）
- 实现因果注意力机制（Causal/Masked Attention）
- 实现多头注意力机制（Multi-Head Attention）
- 比较多种高效多头注意力实现方式
- 理解 PyTorch Buffers 的用法

### 第4章：从零实现 GPT 模型
- 实现 GPT 模型架构（包括 Transformer 块、前馈网络、层归一化等）
- 执行文本生成（Text Generation）
- 分析模型的 FLOPS 性能

### 第5章：在无标注数据上进行预训练
- 对 GPT 模型进行预训练
- 加载和使用 OpenAI 的预训练 GPT-2 权重
- 使用多种方式加载预训练权重（包括内存高效方式）
- 在 Project Gutenberg 数据集上预训练 GPT 模型
- 使用学习率调度器（Learning Rate Scheduler）优化训练循环
- 超参数调优（Hyperparameter Tuning）
- 将 GPT 模型转换为 Llama 架构
- 从零实现 Llama 3.2 模型
- 扩展 Tiktoken BPE 分词器（添加新 Token）
- 使用 PyTorch 性能优化技巧加速 LLM 训练
- 构建用户界面与预训练 LLM 交互

### 第6章：微调用于文本分类
- 对 GPT 模型进行文本分类微调（如垃圾邮件识别）
- 在 IMDB 电影评论数据集（50k 条）上微调不同模型
- 实验微调不同层的效果与更大模型的效果
- 构建用户界面与基于 GPT 的垃圾邮件分类器交互

### 第7章：微调以遵循指令
- 对 GPT 模型进行指令微调（Instruction Finetuning）
- 使用 OpenAI API 和 Ollama 评估指令响应效果
- 构建和改进指令微调数据集
- 利用 Llama 3.1 70B 和 Ollama 生成偏好数据集
- 使用直接偏好优化（DPO）进行 LLM 对齐
- 构建用户界面与指令微调后的 GPT 模型交互

---

## 附录任务

### 附录A：PyTorch 入门
- 学习 PyTorch 基础知识（张量操作、自动微分等）
- 使用分布式数据并行（DDP）进行多 GPU 训练

### 附录D：完善训练循环
- 为训练循环添加梯度裁剪、学习率预热、余弦退火等技巧

### 附录E：使用 LoRA 进行参数高效微调
- 使用低秩自适应（LoRA）对模型进行参数高效微调

---

## 任务概览

| 任务类别 | 具体任务 |
|----------|----------|
| 数据处理 | 分词、BPE、数据加载器 |
| 模型构建 | 注意力机制、GPT架构、Llama架构 |
| 预训练   | GPT预训练、权重加载、超参数调优 |
| 微调     | 文本分类微调、指令微调、LoRA微调、DPO对齐 |
| 评估     | 模型评估、指令响应评估 |
| 部署     | 用户界面构建 |
| 优化     | 训练加速、内存高效权重加载 |
