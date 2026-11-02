# train_ft.py - README 7단계 도메인 미세 조정 (텍스트 모델 270M 추가 학습)
# 목적: 강의실 질의-문서 282쌍(data/train.jsonl)으로 EmbeddingGemma 2 텍스트 모델을 더 학습시켜
#       강의실 질문을 맞는 문서에 더 잘 연결하게 만든다. 결과는 models/eg2-classroom 폴더에 저장.
# 실행: ~/eg2-lab 에서 venv를 켠 뒤  nohup python train_ft.py > train.log 2>&1 &
#       (SSH가 끊겨도 학습이 계속되게 nohup + & 로 실행. 진행은 tail -f train.log 로 본다)
# 준비: 같은 폴더에 eval_ir.py, data/train.jsonl, data/eval.jsonl
#       pip install -c ~/eg2/torch-pin.txt datasets accelerate  (학습용 패키지, 내 venv에만 설치)
#
# 수정 기록
# 2026-10-26  v1  README 7단계 train_ft.py 를 바탕으로 작성. README와 달라진 점 (왜):
#                 - 학습용 모델을 float32 로 불러옴 (README는 bfloat16). bfloat16 가중치는 유효 숫자가 약 3자리라
#                   learning_rate 2e-5 정도의 작은 수정값이 반올림돼 사라져 학습이 거의 안 될 수 있다.
#                   계산은 bf16=True 로 여전히 bfloat16 속도를 쓴다 (가중치 보관만 float32).
#                 - 단어 임베딩 표(입력 임베딩)를 고정함. 270M 중 큰 부분이 이 표라서, 고정하면 float32 로 바꿔
#                   늘어난 메모리(기울기·Adam 상태)를 크게 줄인다. 282쌍으로 단어 표까지 바꿀 필요도 적다.
#                 - 중간 체크포인트 저장 끔 (save_strategy="no"). 체크포인트마다 모델+옵티마이저 상태로 수 GB가
#                   디스크에 쓰여 느리고, 마지막 모델만 있으면 되기 때문.
#                 - 학습 시간·GPU 메모리 최대를 마지막에 출력 (README 결과 표의 '학습 시간' 칸용)
#                 - 그 밖(gradient checkpointing 직접 켜기, MatryoshkaLoss, warmup_steps=0.1)은 README Q9 반영본 그대로

# ── 학습이 하는 일 (MultipleNegativesRankingLoss) ─────────────
# 배치 8개 쌍을 한 번에 넣는다. 질의 1번의 정답은 문서 1번이고, 같은 배치의 문서 2~8번은 '오답'으로 쓴다.
# 모델이 질의-정답 유사도는 올리고 질의-오답 유사도는 내리도록 가중치를 조금씩 고친다.
# 그래서 같은 배치에 내용이 거의 같은 문서가 둘 있으면 정답을 오답 취급하게 되어 학습이 꼬인다 (README 데이터 준비 참고).
# MatryoshkaLoss 는 위 계산을 768/512/256/128 차원에서 모두 해서, 학습 뒤에도 벡터를 잘라 쓸 수 있게 한다.
# ────────────────────────────────────────────────────────────

import time                   # 학습 시간 측정
import torch                  # float32 / GPU 메모리 확인
from datasets import load_dataset
# datasets : Hugging Face 데이터 패키지. .jsonl 파일을 학습기(Trainer)가 바로 읽는 형식으로 바꿔 준다.
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
# 괄호 ( ) 로 감싸면 import 를 여러 줄로 나눠 쓸 수 있다
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from eval_ir import build_evaluator   # 같은 폴더 eval_ir.py 의 평가 함수. epoch 마다 점수를 찍는 데 쓴다

# ── 설정 (여기만 바꿔 가며 실험) ──────────────────────────────
TRAIN_DTYPE = torch.float32   # README 그대로 해 보려면 torch.bfloat16 (위 수정 기록 참고)
FREEZE_EMBEDDINGS = True      # True: 단어 임베딩 표 고정 (메모리 절약). False: 전부 학습 (float32면 OOM 가능)
EPOCHS = 3                    # 학습 데이터를 몇 바퀴 볼지. 평가 점수가 떨어지면(과적합) 1~2로
BATCH = 8                     # 한 번에 넣는 쌍 수. 클수록 오답이 많아 학습 효과가 크지만 메모리를 더 씀. OOM이면 4
ACCUM = 2                     # 기울기 누적: BATCH 를 ACCUM 번 모아 한 번 수정 → 실제 효과는 8×2=16쌍 배치. BATCH 4면 4로
LR = 2e-5                     # 학습률(한 번에 가중치를 고치는 크기). 과적합이면 1e-5
OUT = "models/eg2-classroom"  # 학습된 모델을 저장할 폴더 (~/eg2-lab 아래에 생김)

# 원본 모델 불러오기. 텍스트 전용(비전·오디오 인코더 없음)
model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
    model_kwargs={"torch_dtype": TRAIN_DTYPE},
    config_kwargs={"vision_config": None, "audio_config": None})

# model[0] : SentenceTransformer 는 여러 부품(모듈)을 차례로 이은 구조. 0번이 실제 트랜스포머 부품이다.
# getattr(객체, "이름", None) : 객체에 그 이름의 속성이 있으면 꺼내고, 없으면 None.
#   버전에 따라 속성 이름이 auto_model 또는 model 이라서 둘 다 시도한다 (README Q9).
# A or B : A 가 None 이면 B 를 쓴다
hf = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)

# gradient checkpointing : 중간 계산 결과를 다 보관하지 않고 필요할 때 다시 계산해 메모리를 아낀다 (대신 조금 느려짐).
# Trainer 인자로 켜면 Transformers v5 에서 TypeError 가 나서 내부 모델에 직접 켠다 (README Q9).
hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

if FREEZE_EMBEDDINGS:
    # get_input_embeddings() : 단어(토큰) 번호 → 벡터 를 바꿔 주는 큰 표. Hugging Face 모델 공통 함수
    # .requires_grad_(False) : 이 가중치는 기울기를 계산하지 않음 = 학습에서 고치지 않음
    hf.get_input_embeddings().requires_grad_(False)

# 학습할 가중치 수 확인. p.numel() : 가중치 묶음 p 안의 숫자 개수
# sum(식 for p in ... if 조건) : 조건에 맞는 것만 더한다
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"가중치 {total / 1e6:.0f}M 중 학습 {trainable / 1e6:.0f}M, 보관 형식 {TRAIN_DTYPE}")

# data/train.jsonl 읽기. 각 줄의 "anchor"(질의), "positive"(정답 문서) 가 열 이름이 된다
train = load_dataset("json", data_files="data/train.jsonl", split="train")
print(f"학습 쌍 {len(train)}개")

# MatryoshkaLoss 로 감싸야 학습 후에도 256/128차원 절단이 유지된다 (README)
loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model),
                      matryoshka_dims=[768, 512, 256, 128])

args = SentenceTransformerTrainingArguments(
    output_dir="runs/eg2-classroom",       # 학습 로그가 쓰이는 폴더 (체크포인트는 아래에서 껐음)
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    learning_rate=LR,
    warmup_steps=0.1,                      # 처음 10% 동안 학습률을 0에서 천천히 올림. Transformers v5는 비율을 float로 (README)
    bf16=True,                             # 계산은 bfloat16 (빠르고 메모리 적게). 가중치 보관 형식과는 별개
    batch_sampler=BatchSamplers.NO_DUPLICATES,   # 같은 배치에 같은 문서가 두 번 들어가지 않게 (정답이 오답 취급되는 것 방지)
    # prompts : 학습할 때도 평가·검색 때와 같은 안내 문구를 붙인다. 질의 열엔 SearchQuery, 문서 열엔 Document
    prompts={"anchor": model.prompts["SearchQuery"],
             "positive": model.prompts["Document"]},
    eval_strategy="epoch",                 # 한 바퀴(epoch)마다 eval.jsonl 로 점수를 찍는다
    save_strategy="no",                    # 중간 체크포인트 저장 안 함 (수정 기록 참고). 마지막에 OUT 으로 한 번 저장
    logging_steps=5,                       # 5번 수정할 때마다 loss 를 로그에 찍음 (한 epoch 약 18번이라 README 10보다 촘촘히)
    report_to="none",                      # wandb 같은 외부 기록 서비스 안 씀
)

trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                                     loss=loss, evaluator=build_evaluator())

t = time.perf_counter()
trainer.train()                            # 학습 시작. 로그에 loss 와 epoch 별 classroom_cosine_... 점수가 찍힌다
minutes = (time.perf_counter() - t) / 60

model.save_pretrained(OUT)                 # 학습된 모델을 폴더에 저장 (텍스트 전용 설정도 함께 저장됨)
print(f"\n학습 시간 {minutes:.1f}분, GPU 메모리 최대 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"저장: {OUT}  → 평가: python eval_ir.py {OUT}")
