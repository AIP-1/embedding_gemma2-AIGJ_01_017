# check.py - 설치 확인 스크립트 (README 2단계)
# 목적: 필요한 패키지 3개가 설치됐는지, torch가 Jetson GPU를 쓸 수 있는지 확인한다.
# 실행: ~/eg2-lab 에서 venv를 켠 뒤  python check.py
#
# 수정 기록
# 2026-10-23  v1  user_setup.sh가 만든 check.py와 코드는 같고, 파이썬 기초 설명 주석만 추가 (지원 님 요청: 파이썬 복습용)

# ── 파이썬 기초 ─────────────────────────────────────────────
# '#' 뒤는 주석이다. 파이썬은 이 부분을 무시하고, 사람이 읽으라고 쓰는 설명이다.
# import 이름      : 다른 사람이 만든 코드 묶음(패키지)을 불러와서 '이름.기능' 으로 쓸 수 있게 한다.
# import 긴이름 as 짧은이름 : 불러오면서 별명을 붙인다. 아래에서 sentence_transformers 대신 st 로 쓴다.
# ────────────────────────────────────────────────────────────

import torch                         # PyTorch: 딥러닝 계산(행렬 곱 등)을 GPU에서 해 주는 기본 패키지
import transformers                  # Hugging Face transformers: 모델 구조와 가중치를 읽어 오는 패키지
import sentence_transformers as st   # sentence-transformers: 문장/이미지/음성을 임베딩(숫자 벡터)으로 바꾸는 패키지

# print(...)      : 괄호 안의 값을 화면에 출력한다. 쉼표로 여러 개를 주면 띄어쓰기로 이어 붙인다.
# 패키지.__version__ : 설치된 버전 문자열 (예: "2.14.1+cu132")
# torch.cuda.is_available() : GPU(CUDA)를 쓸 수 있으면 True, 아니면 False 를 돌려주는 함수
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("sentence-transformers", st.__version__)   # 6.1.0 이상이어야 EmbeddingGemma 2를 읽을 수 있다

# assert 조건, "메시지" : 조건이 True면 그냥 지나가고, False면 메시지와 함께 에러를 내고 멈춘다.
# 왜 확인하나: 패키지를 설치하다 Jetson 전용 torch가 일반 torch(GPU를 못 쓰는 버전)로
#             덮어쓰이는 사고가 잦다. 그러면 cuda가 False가 되므로 여기서 바로 잡아낸다.
assert torch.cuda.is_available(), "Jetson torch가 덮어쓰여졌는지 확인"
