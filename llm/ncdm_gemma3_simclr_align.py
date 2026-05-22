from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
if str(MODEL_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(MODEL_DIR))

from ncdm import NCDM  # noqa: E402


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_GEMMA_MODEL = "gemma3:27b"
DEFAULT_CACHE_PATH = ROOT_DIR / "llm" / "gemma3_alignment_cache.json"
DEFAULT_RESPONSE_PATH = ROOT_DIR / "data" / "math23k" / "StudentResponses.json"
DEFAULT_QUESTION_PATH = ROOT_DIR / "data" / "math23k" / "raw" / "train23k.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "llm" / "ncdm_gemma3_simclr_alignment.pt"


BLOOM_PROMPT = """
你是一名教育认知诊断专家。请基于下面的数学题，生成一段用于认知诊断的简洁语义描述。

要求：
1. 点出题目涉及的核心知识点。
2. 用 Bloom 分类法概括学生完成本题需要的认知层级，至少覆盖“理解、应用、分析”。
3. 描述常见错误点或薄弱环节。
4. 输出 80 到 140 个中文字符，不要分点，不要输出额外解释。

题目：{question_text}
参考方程：{equation}
"""


@dataclass
class AlignmentSample:
    student_id: int
    item_id: int
    question_id: int
    correct: int
    knowledge_mask: list[float]
    question_text: str
    equation: str


class Math23kAlignmentDataset(Dataset):
    def __init__(self, samples: list[AlignmentSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> AlignmentSample:
        return self.samples[index]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_result(value: Any) -> int:
    text = str(value).strip().lower()
    positive = {"正确", "对", "true", "1", "yes", "correct"}
    negative = {"错误", "错", "false", "0", "no", "wrong"}

    if text in positive:
        return 1
    if text in negative:
        return 0

    return 1 if "正确" in str(value) else 0


def build_synthetic_responses(
    question_path: Path,
    max_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    questions = load_json(question_path)
    rng = random.Random(seed)
    rng.shuffle(questions)

    synthetic: list[dict[str, Any]] = []
    for idx, question in enumerate(questions[:max_samples], start=1):
        synthetic.append(
            {
                "student_id": idx,
                "question_id": int(question["id"]),
                "answer": "",
                "result": "正确" if idx % 2 == 1 else "错误",
            }
        )
    return synthetic


def build_knowledge_mask(question_text: str, equation: str, question_id: int) -> list[float]:
    concepts = [0.0] * 8
    source = f"{question_text} {equation}"

    if "+" in equation or "-" in equation:
        concepts[0] = 1.0
    if "*" in equation or "/" in equation:
        concepts[1] = 1.0
    if "(" in equation and "/" in equation:
        concepts[2] = 1.0
    if "%" in equation or "百分" in source:
        concepts[3] = 1.0
    if "比" in source or "倍" in source or "比例" in source:
        concepts[4] = 1.0
    if "速度" in source or "路程" in source or "相遇" in source or "行" in source:
        concepts[5] = 1.0
    if "面积" in source or "周长" in source or "长方形" in source or "圆" in source:
        concepts[6] = 1.0
    if "平均" in source or "分配" in source or "剩" in source or "还" in source:
        concepts[7] = 1.0

    if sum(concepts) == 0:
        concepts[question_id % len(concepts)] = 1.0

    return concepts


def prepare_samples(
    response_path: Path,
    question_path: Path,
    max_samples: int,
    seed: int,
) -> tuple[list[AlignmentSample], dict[int, int], dict[int, int], int]:
    questions = load_json(question_path)
    if response_path.exists():
        responses = load_json(response_path)
    else:
        print(f"[warn] 未找到作答文件 {response_path}，将基于题库生成临时样本。")
        responses = build_synthetic_responses(question_path, max_samples=max_samples, seed=seed)

    question_map = {int(item["id"]): item for item in questions}
    filtered = [row for row in responses if int(row["question_id"]) in question_map]

    rng = random.Random(seed)
    rng.shuffle(filtered)
    filtered = filtered[:max_samples]

    student_ids = sorted({int(row["student_id"]) for row in filtered})
    item_ids = sorted({int(row["question_id"]) for row in filtered})
    student_to_index = {student_id: idx for idx, student_id in enumerate(student_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    samples: list[AlignmentSample] = []
    for row in filtered:
        question_id = int(row["question_id"])
        question = question_map[question_id]
        knowledge_mask = build_knowledge_mask(
            question_text=question["original_text"],
            equation=question["equation"],
            question_id=question_id,
        )
        samples.append(
            AlignmentSample(
                student_id=student_to_index[int(row["student_id"])],
                item_id=item_to_index[question_id],
                question_id=question_id,
                correct=normalize_result(row.get("result", 0)),
                knowledge_mask=knowledge_mask,
                question_text=question["original_text"],
                equation=question["equation"],
            )
        )

    return samples, student_to_index, item_to_index, len(samples[0].knowledge_mask)


class GemmaSemanticGenerator:
    def __init__(
        self,
        model_name: str,
        cache_path: Path,
        timeout: int = 60,
        offline_text: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_path = cache_path
        self.timeout = timeout
        self.offline_text = offline_text
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            self.cache = load_json(cache_path)

    def _call_gemma(self, question_text: str, equation: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": BLOOM_PROMPT.format(
                        question_text=question_text,
                        equation=equation,
                    ),
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 200,
            },
        }
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=self.timeout)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "").strip()
        return content

    def get_text(self, key: str, question_text: str, equation: str) -> str:
        if key in self.cache:
            return self.cache[key]

        if self.offline_text:
            text = f"题目：{question_text} 方程：{equation}"
        else:
            try:
                text = self._call_gemma(question_text, equation)
            except Exception as exc:
                text = f"题目：{question_text} 方程：{equation}"
                print(f"[warn] Gemma3 调用失败，已回退到原题文本: {exc}")

        self.cache[key] = text
        save_json(self.cache_path, self.cache)
        return text


class HashTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 4096, embed_dim: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _hash_ids(self, text: str, max_tokens: int = 256) -> list[int]:
        compact = text.replace(" ", "")
        token_ids = [ord(ch) % self.vocab_size for ch in compact[:max_tokens]]
        if not token_ids:
            token_ids = [0]
        return token_ids

    def forward(self, texts: list[str]) -> torch.Tensor:
        flat_ids: list[int] = []
        offsets = [0]
        for text in texts:
            ids = self._hash_ids(text)
            flat_ids.extend(ids)
            offsets.append(len(flat_ids))

        input_ids = torch.tensor(flat_ids, dtype=torch.long)
        offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long)
        bag = self.embedding(input_ids, offsets_tensor)
        return self.projection(bag)


class NCDMBehaviorEncoder(nn.Module):
    def __init__(
        self,
        num_students: int,
        num_items: int,
        num_knowledge: int,
        model_hidden_dim: int = 64,
        projection_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.ncdm = NCDM(
            num_students=num_students,
            num_items=num_items,
            num_knowledge=num_knowledge,
            hidden_dim=model_hidden_dim,
            dropout=dropout,
        )
        self.projector = nn.Sequential(
            nn.Linear(num_knowledge, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def behavior_features(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        correctness: torch.Tensor,
    ) -> torch.Tensor:
        mastery = torch.sigmoid(self.ncdm.student_mastery(student_ids))
        difficulty = torch.sigmoid(self.ncdm.item_difficulty(item_ids))
        discrimination = F.softplus(self.ncdm.item_discrimination(item_ids)).squeeze(-1) + 1e-6
        signed_correctness = correctness.float().unsqueeze(-1) * 2.0 - 1.0
        behavior = (mastery - difficulty) * knowledge_mask.float()
        behavior = behavior * discrimination.unsqueeze(-1) * signed_correctness
        return behavior

    def forward(
        self,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
        knowledge_mask: torch.Tensor,
        correctness: torch.Tensor,
    ) -> torch.Tensor:
        features = self.behavior_features(student_ids, item_ids, knowledge_mask, correctness)
        return self.projector(features)


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = F.normalize(left, dim=-1)
        right = F.normalize(right, dim=-1)
        logits = torch.matmul(left, right.T) / self.temperature
        labels = torch.arange(left.size(0), device=left.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def collate_alignment_batch(
    batch: list[AlignmentSample],
    generator: GemmaSemanticGenerator,
) -> dict[str, Any]:
    semantic_texts = []
    for sample in batch:
        key = f"{sample.question_id}"
        semantic_texts.append(generator.get_text(key, sample.question_text, sample.equation))

    return {
        "student_ids": torch.tensor([sample.student_id for sample in batch], dtype=torch.long),
        "item_ids": torch.tensor([sample.item_id for sample in batch], dtype=torch.long),
        "labels": torch.tensor([sample.correct for sample in batch], dtype=torch.float32),
        "knowledge_mask": torch.tensor([sample.knowledge_mask for sample in batch], dtype=torch.float32),
        "semantic_texts": semantic_texts,
    }


def train_alignment(args: argparse.Namespace) -> None:
    samples, student_to_index, item_to_index, num_knowledge = prepare_samples(
        response_path=args.response_path,
        question_path=args.question_path,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    if not samples:
        raise RuntimeError("没有找到可用于对齐训练的样本。")

    generator = GemmaSemanticGenerator(
        model_name=args.gemma_model,
        cache_path=args.cache_path,
        timeout=args.request_timeout,
        offline_text=args.offline_text,
    )

    dataset = Math23kAlignmentDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_alignment_batch(batch, generator),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    behavior_encoder = NCDMBehaviorEncoder(
        num_students=len(student_to_index),
        num_items=len(item_to_index),
        num_knowledge=num_knowledge,
        model_hidden_dim=args.model_hidden_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)
    text_encoder = HashTextEncoder(
        vocab_size=args.text_vocab_size,
        embed_dim=args.projection_dim,
    ).to(device)
    contrastive_loss = NTXentLoss(temperature=args.temperature)

    optimizer = torch.optim.Adam(
        list(behavior_encoder.parameters()) + list(text_encoder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"[info] device={device}, samples={len(samples)}, batch_size={args.batch_size}")
    print(f"[info] 使用模型: NCDM + {args.gemma_model}")
    if args.offline_text:
        print("[info] 当前为离线文本模式，语义侧使用原题文本回退。")

    for epoch in range(1, args.epochs + 1):
        behavior_encoder.train()
        text_encoder.train()

        total_loss = 0.0
        total_diag = 0.0
        total_contrast = 0.0

        for batch in loader:
            student_ids = batch["student_ids"].to(device)
            item_ids = batch["item_ids"].to(device)
            labels = batch["labels"].to(device)
            knowledge_mask = batch["knowledge_mask"].to(device)
            semantic_texts = batch["semantic_texts"]

            optimizer.zero_grad()

            behavior_projection = behavior_encoder(
                student_ids=student_ids,
                item_ids=item_ids,
                knowledge_mask=knowledge_mask,
                correctness=labels,
            )
            text_projection = text_encoder(semantic_texts).to(device)

            diag_loss = behavior_encoder.ncdm.loss(
                student_ids=student_ids,
                item_ids=item_ids,
                knowledge_mask=knowledge_mask,
                labels=labels,
            )
            simclr_loss = contrastive_loss(behavior_projection, text_projection)
            loss = diag_loss + args.contrastive_weight * simclr_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_diag += diag_loss.item()
            total_contrast += simclr_loss.item()

        num_batches = max(len(loader), 1)
        print(
            f"epoch={epoch:02d} "
            f"loss={total_loss / num_batches:.4f} "
            f"diag={total_diag / num_batches:.4f} "
            f"simclr={total_contrast / num_batches:.4f}"
        )

    artifact = {
        "config": vars(args),
        "student_count": len(student_to_index),
        "item_count": len(item_to_index),
        "num_knowledge": num_knowledge,
        "behavior_encoder": behavior_encoder.state_dict(),
        "text_encoder": text_encoder.state_dict(),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output_path)
    print(f"[done] 对齐模型已保存到: {args.output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 NCDM 与 Gemma3 语义文本进行 SimCLR 风格对齐训练。"
    )
    parser.add_argument(
        "--response-path",
        type=Path,
        default=DEFAULT_RESPONSE_PATH,
        help="学生作答记录 JSON 路径。",
    )
    parser.add_argument(
        "--question-path",
        type=Path,
        default=DEFAULT_QUESTION_PATH,
        help="题库 JSON 路径。",
    )
    parser.add_argument(
        "--gemma-model",
        type=str,
        default=DEFAULT_GEMMA_MODEL,
        help="本地 Ollama 中的 Gemma3 模型名。",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Gemma3 生成语义文本的缓存文件。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="训练完成后的权重输出路径。",
    )
    parser.add_argument("--max-samples", type=int, default=128, help="用于训练的最大样本数。")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小。")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数。")
    parser.add_argument("--projection-dim", type=int, default=256, help="共享投影空间维度。")
    parser.add_argument("--model-hidden-dim", type=int, default=64, help="NCDM 隐层维度。")
    parser.add_argument("--text-vocab-size", type=int, default=4096, help="文本哈希词表大小。")
    parser.add_argument("--temperature", type=float, default=0.07, help="NT-Xent 温度。")
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.5,
        help="对比损失在总损失中的权重。",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率。")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="权重衰减。")
    parser.add_argument("--dropout", type=float, default=0.2, help="NCDM dropout。")
    parser.add_argument("--request-timeout", type=int, default=60, help="Gemma3 请求超时秒数。")
    parser.add_argument("--seed", type=int, default=20260522, help="随机种子。")
    parser.add_argument(
        "--offline-text",
        action="store_true",
        help="不调用 Gemma3，直接用题目原文作为语义侧输入，便于本地调试。",
    )
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU。")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    train_alignment(args)


if __name__ == "__main__":
    main()
