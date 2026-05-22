import requests
import json
import pandas as pd
import time

# --- 实验配置 ---
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:27b" 

TEST_CASES = [
    {"id": 1, "topic": "甲数除以乙数的商是1.5，如果甲数增加20，则甲数是乙的4倍．原来甲数=．"}
]

# --- 提示词设计 (核心实验变量) ---

# Bloom 分类法：侧重于具体的认知动作阶梯
# 优势论点：通过明确的动作指令（分析、评价、创造），迫使模型进行多维思考
BLOOM_PROMPT = """
任务：作为教育专家，请分析课题《{topic}》。
要求：你必须严格按照 Bloom 分类法的高阶认知维度进行回答：
1. [分析层]：拆解该课题的核心组成要素，并解释它们之间的内在联系。
2. [评价层]：基于科学性或逻辑性，评价该课题在当前研究或应用中的局限性。
3. [创造层]：提出一个基于该课题原理的全新应用场景或改进方案。
"""

# SOLO 分类法：侧重于思维结构的关联性
# 特点：侧重于从单点到整体的结构推演
SOLO_PROMPT = """
任务：作为教育专家，请分析课题《{topic}》。
要求：你必须严格按照 SOLO 分类法的思维结构层次进行回答：
1. [关联水平]：说明该课题内所有相关概念是如何整合为一个有机整体的，不要只列举事实。
2. [拓展抽象水平]：将该课题的原理进行泛化，推导到一个全新的理论层面或跨学科领域。
"""


def call_ollama(prompt):
    """
    调用本地 Ollama API 发送请求
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.3,  # 较低随机性以保证实验可重复
            "num_predict": 800  # 限制长度，防止模型由于参数量小而出现循环
        }
    }

    try:
        start_time = time.time()
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        duration = time.time() - start_time

        content = response.json().get("message", {}).get("content", "")
        return content, duration
    except Exception as e:
        return f"Error: {str(e)}", 0


def run_experiment():
    """
    运行自动化实验流程
    """
    print(f"开始实验 - 使用模型: {MODEL_NAME}")
    all_results = []

    for case in TEST_CASES:
        topic = case["topic"]
        print(f"正在测试课题 {case['id']}: {topic}")

        # 1. 运行 Bloom 实验组
        bloom_content, bloom_time = call_ollama(BLOOM_PROMPT.format(topic=topic))

        # 2. 运行 SOLO 实验组
        solo_content, solo_time = call_ollama(SOLO_PROMPT.format(topic=topic))

        all_results.append({
            "Topic_ID": case["id"],
            "Topic": topic,
            "Bloom_Response": bloom_content,
            "Bloom_Latency": bloom_time,
            "SOLO_Response": solo_content,
            "SOLO_Latency": solo_time
        })

    # 将结果保存为 CSV 供后续分析
    df = pd.DataFrame(all_results)
    df.to_csv("experiment_bloom_vs_solo.csv", index=False, encoding='utf-8-sig')
    print("\n[实验完成] 结果已导出至 experiment_bloom_vs_solo.csv")


if __name__ == "__main__":
    run_experiment()