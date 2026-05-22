# A cognitive diagnosis framework driven by largeanguage models based on Bloom’s taxonomy anddual-space collaborative alignment

面向智能教育系统，基于布鲁姆认知分类法与语义 - 行为双空间协同对齐（DSCA） 的大语言模型驱动认知诊断框架。

## 📖 项目概述

传统认知诊断模型（CDM）仅能处理数值化答题记录，无法有效建模文本信息。大语言模型（LLM）与认知诊断模型融合面临两大核心难题：

大模型生成的诊断文本缺乏教育理论标准，存在模糊性与主观性；

大模型语义空间与传统诊断模型行为空间存在模态鸿沟，难以有效融合。

本框架通过布鲁姆分类法标准化提示与双空间协同对齐机制解决上述问题。

## ✨ 核心创新

布鲁姆分类法引导提示
基于记忆、理解、应用、分析、评价、创造六大认知层级规范大模型诊断输出，消除文本歧义，贴合教育理论标准。

双空间协同对齐（DSCA）
通过 MLP+SimCLR 对比学习，建立语义空间（LLM）与行为空间（CDM）的双向映射，完整保留跨模态信息。

模型无关・泛化性强
兼容 5 种主流认知诊断模型（NCD/RCD/GCD/SCD/RDGT）与 3 种大模型（Gemma3:27B/Qwen3:30B/GLM-4），AUC 平均提升 4.5%。

## 🧩 框架架构

框架包含两大核心模块：

1、LLM 文本诊断生成模块

学生 / 习题协同信息抽取

布鲁姆分类法驱动语义诊断

2、语义 - 行为双空间协同对齐模块：

语义 / 行为特征提取

双向映射（MLP）

一致性约束（SimCLR 对比学习）

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

```

###  LLM 提示词实验

```bash
cd llm
python gemma3.py
```


```

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
- **RDGT 对比图** (`picture/rdgt_comp.png`): RDGT 模型融合不同LLM的对比结果

## 🔬 核心算法

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

