# Cognitive Diagnosis Models with LLM Alignment

本项目实现了一系列认知诊断模型（Cognitive Diagnosis Models），并结合大型语言模型（LLM）进行知识对齐和增强。项目包含多种经典和创新的认知诊断模型，以及使用 Gemma3 等 LLM 进行语义对齐的工具。

## 📁 项目结构

```
.
├── model/                  # 认知诊断模型实现
│   ├── ncdm.py            # Neural Cognitive Diagnosis Model
│   ├── rcdm.py            # Relational Cognitive Diagnosis Model
│   ├── scdm.py            # Sequential Cognitive Diagnosis Model
│   ├── gcdm.py            # Graph-based Cognitive Diagnosis Model
│   └── rdgt.py            # Response-Driven Graph Tracing Model
├── llm/                    # LLM 对齐与实验代码
│   ├── gemma3.py          # Gemma3 LLM 调用与提示词实验
│   └── ncdm_gemma3_simclr_align.py  # NCDM 与 Gemma3 的 SimCLR 对齐
├── data/                   # 数据集
│   ├── math23k/           # Math23K 数学题目数据集
│   │   ├── raw/           # 原始数据
│   │   └── StudentResponses.json
│   └── openluna/          # OpenLUNA 数据集
│       └── StudentResponses.json
├── picture/                # 实验结果可视化
│   ├── T_sne.png          # t-SNE 降维可视化
│   └── rdgt_comp.png      # RDGT 模型对比图
├── requirements.txt        # Python 依赖
└── README.md              # 项目说明文档
```

## 🚀 主要功能

### 认知诊断模型

- **NCDM (Neural Cognitive Diagnosis Model)**: 基于神经网络的认知诊断模型，通过学生掌握度、题目难度和区分度进行建模
- **RCDM (Relational Cognitive Diagnosis Model)**: 关系型认知诊断模型
- **SCDM (Sequential Cognitive Diagnosis Model)**: 序列认知诊断模型
- **GCDM (Graph-based Cognitive Diagnosis Model)**: 基于图的认知诊断模型
- **RDGT (Response-Driven Graph Tracing)**: 响应驱动的图追踪模型，用于动态认知诊断

### LLM 对齐与增强

- **Gemma3 集成**: 使用本地 Ollama 服务调用 Gemma3 模型
- **提示词工程**: 实现了基于 Bloom 分类法和 SOLO 分类法的提示词设计
- **SimCLR 对齐**: 使用对比学习将 NCDM 的认知表示与 LLM 的语义表示进行对齐

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包说明

| 包名 | 版本 | 用途 |
|------|------|------|
| EduData | 0.0.18 | 教育数据处理 |
| torch | 2.8.0 | 深度学习框架 |
| transformers | - | 模型支持 |
| sentence_transformers | 5.4.1 | 句子嵌入 |
| scikit-learn | 1.8.0 | 机器学习工具 |
| pandas | 3.0.3 | 数据处理 |
| numpy | 2.4.6 | 数值计算 |
| matplotlib | 3.10.9 | 数据可视化 |
| seaborn | 0.13.2 | 统计可视化 |

## 🔧 使用示例

### 1. 运行认知诊断模型

```python
from model.ncdm import NCDM

# 初始化模型
model = NCDM(
    num_students=1000,
    num_items=500,
    num_knowledge=10,
    hidden_dim=64,
    dropout=0.2
)

# 预测
logits = model.predict_logits(
    student_ids=torch.tensor([0, 1]),
    item_ids=torch.tensor([10, 20]),
    knowledge_mask=torch.ones(2, 10)
)
```

### 2. LLM 提示词实验

```bash
cd llm
python gemma3.py
```

该脚本演示了如何使用不同的提示词策略（Bloom vs SOLO 分类法）来分析教育课题。

### 3. NCDM 与 Gemma3 SimCLR 对齐

```bash
python llm/ncdm_gemma3_simclr_align.py \
    --response_path data/math23k/StudentResponses.json \
    --question_path data/math23k/raw/train23k.json \
    --output_path llm/ncdm_gemma3_simclr_alignment.pt
```

## 📊 数据集

### Math23K
- 包含 23,000 道小学数学题目
- 提供学生作答记录
- 支持知识点标注

### OpenLUNA
- 开放学习分析数据集
- 包含学生响应数据
- 适用于认知诊断研究

## 📈 实验结果

项目包含以下可视化结果：

- **t-SNE 可视化** (`picture/T_sne.png`): 展示学生认知表示的降维分布
- **RDGT 对比图** (`picture/rdgt_comp.png`): RDGT 模型与其他模型的对比结果

## 🔬 核心算法

### NCDM 模型架构

```
学生掌握度 (Embedding) ──┐
                         ├──→ 预测网络 → 正确率预测
题目难度 (Embedding) ────┘
题目区分度 (Embedding) ──┘
```

### SimCLR 对齐流程

1. 使用 NCDM 生成学生认知表示
2. 使用 Gemma3 生成题目语义表示
3. 通过对比学习损失函数对齐两种表示
4. 优化模型参数以最大化一致性

## 📝 配置说明

### LLM 服务配置

项目默认使用本地 Ollama 服务：

```python
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:27b"
```

请确保已安装并启动 Ollama 服务：

```bash
ollama serve
ollama pull gemma3:27b
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 📄 许可证

本项目采用 MIT 许可证。

## 📚 参考文献

1. Wang, F., et al. "Neural Cognitive Diagnosis for Intelligent Education Systems." AAAI 2020.
2. Chen, P., et al. "Tracking Knowledge Proficiency of Students with Educational Data." CIKM 2017.
3. Gao, C., et al. "Deep Learning based Knowledge Tracing." EDM 2019.

## 📧 联系方式

如有问题或建议，请通过 Issue 联系。
