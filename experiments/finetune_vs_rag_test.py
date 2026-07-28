"""
RAG vs 파인튜닝 비교 실험.

목적: "멍청한 모델을 파인튜닝하면 RAG가 필요 없어지는가?"를 검증.
- Qwen2.5-0.5B-Instruct를 실제 QA 53개(트릭 질문 제외)로 LoRA 파인튜닝
- 파인튜닝 모델을 (a) 학습에 포함된 질문, (b) 학습에 없던 트릭 질문에 테스트
- 같은 트릭 질문을 RAG(grounding_check 포함) 파이프라인에도 물어봐서 비교

가설: 파인튜닝은 학습한 질문은 잘 재현하지만, 한 번도 안 본 "존재하지 않는 기능"
질문에는 여전히 환각한다. RAG는 학습 없이도 매번 grounding_check로 거절 가능하다.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

from src.evaluation import BENCHMARK_DATASET

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = str(PROJECT_ROOT / "experiments" / "lora_0.5b_catia")

# 트릭 질문은 학습에서 완전히 제외 (한 번도 못 본 상태를 유지해야 실험 의미가 있음)
TRAIN_QA = [d for d in BENCHMARK_DATASET if d["type"] != "Trick/Unanswerable QA"]
TRICK_QA = [d for d in BENCHMARK_DATASET if d["type"] == "Trick/Unanswerable QA"]
print(f"[Data] 학습용 실제 QA: {len(TRAIN_QA)}개 / 홀드아웃 트릭 QA: {len(TRICK_QA)}개 (학습에 전혀 안 씀)")


def format_example(q, a):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"


def build_dataset(tokenizer):
    texts = [format_example(d["question"], d["ground_truth"]) for d in TRAIN_QA]
    enc = tokenizer(texts, truncation=True, max_length=256, padding="max_length")
    enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
    return Dataset.from_dict(enc)


def main():
    print(f"[LoRA] {MODEL_NAME} 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    print("[LoRA] 학습 시작...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[LoRA] 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
