"""
RAG '태스크' 파인튜닝 실험 (사실 암기가 아니라 "문맥 보고 답하는 스킬" 학습).

이전 실험(finetune_vs_rag_test.py)은 질문->정답을 직접 암기시켜서 일반화가
안 됐음. 이번엔 실제 검색된 context를 학습 데이터에 포함시켜서, "문맥에 있으면
답하고, 없으면 거절"하는 행동 자체를 학습시킨다.

검증 방법:
- 학습: 실제 QA 53개(문맥 포함) + 트릭 질문 5개(문맥 포함, 거절 학습)
- 테스트 A: 학습에서 뺀 트릭 질문 3개 (원래 8개 중 나머지)
- 테스트 B: 이 세상에 존재한 적 없는 완전히 새로운 트릭 질문 (진짜 일반화 테스트)
- 테스트 C: 학습에서 뺀 실제 질문으로 정상 답변 유지되는지 확인
"""
import os
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from langchain_chroma import Chroma

from src.evaluation import BENCHMARK_DATASET
from src.vector_store import get_embedding_function
from src.config import CHROMA_DB_DIR_MARKDOWN_ML, MULTILINGUAL_EMBEDDING_MODEL
from src.rag_chain import format_docs_with_pages, UNSUPPORTED_ANSWER

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = str(PROJECT_ROOT / "experiments" / "lora_0.5b_rag_task")

random.seed(42)

real_qa = [d for d in BENCHMARK_DATASET if d["type"] != "Trick/Unanswerable QA"]
trick_qa = [d for d in BENCHMARK_DATASET if d["type"] == "Trick/Unanswerable QA"]
random.shuffle(trick_qa)

train_trick = trick_qa[:5]
holdout_trick = trick_qa[5:]  # 학습에서 뺀, 기존에 있던 트릭 질문 (테스트 A)

print(f"[Data] 실제 QA {len(real_qa)}개 + 학습용 트릭 {len(train_trick)}개 / 홀드아웃 트릭 {len(holdout_trick)}개")


def get_retriever():
    embeddings = get_embedding_function(MULTILINGUAL_EMBEDDING_MODEL)
    vs = Chroma(persist_directory=CHROMA_DB_DIR_MARKDOWN_ML, embedding_function=embeddings)
    return vs.as_retriever(search_kwargs={"k": 4})


SYSTEM_PROMPT = (
    "You are a CATIA manual answer extractor. Answer only from the supplied manual context. "
    "If the context does not explicitly support the answer, reply exactly: "
    "'The provided manual does not contain this information.'"
)


def format_example(context, question, answer):
    user_msg = f"[Manual context]\n{context}\n\n[Question]\n{question}\n\n[Answer]"
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"


def build_training_texts(retriever):
    texts = []
    for d in real_qa + train_trick:
        docs = retriever.invoke(d["question"])
        context = format_docs_with_pages(docs)
        answer = d["ground_truth"] if d["type"] != "Trick/Unanswerable QA" else UNSUPPORTED_ANSWER
        texts.append(format_example(context, d["question"], answer))
    return texts


def main():
    print("[RAG-task LoRA] 검색기 준비 및 학습 데이터(문맥 포함) 생성 중...")
    retriever = get_retriever()
    texts = build_training_texts(retriever)

    print(f"[RAG-task LoRA] {MODEL_NAME} 로딩 중...")
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

    enc = tokenizer(texts, truncation=True, max_length=768, padding="max_length")
    enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
    dataset = Dataset.from_dict(enc)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=6,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)

    print("[RAG-task LoRA] 학습 시작...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[RAG-task LoRA] 저장 완료: {OUTPUT_DIR}")

    # 홀드아웃 트릭 질문 목록을 다음 평가 스크립트에서 쓸 수 있게 파일로 저장
    with open(PROJECT_ROOT / "experiments" / "holdout_trick_questions.txt", "w", encoding="utf-8") as f:
        for d in holdout_trick:
            f.write(d["question"] + "\n")


if __name__ == "__main__":
    main()
