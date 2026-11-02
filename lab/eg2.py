# eg2.py - 모델 불러오기 공통 코드 (README 2단계, 3단계 이후 모든 실습이 사용)
# 목적: EmbeddingGemma 2 모델을 GPU에 올리는 load() 함수를 한 곳에 만들어 두고,
#       다른 파일에서는  from eg2 import load  한 줄로 가져다 쓴다.
# 실행: 직접 실행하는 파일이 아니다. 다른 .py와 같은 폴더(~/eg2-lab)에 있어야 import가 된다.
#
# 수정 기록
# 2026-10-23  v1  user_setup.sh가 만든 eg2.py와 코드는 같고, 파이썬 기초 설명 주석만 추가 (지원 님 요청: 파이썬 복습용)

import torch                                        # bfloat16 같은 숫자 형식 이름을 쓰려고 불러온다
from sentence_transformers import SentenceTransformer
# from 패키지 import 이름 : 패키지 전체가 아니라 그 안의 'SentenceTransformer' 하나만 꺼내 온다.
#                         그래서 아래에서 st.SentenceTransformer 가 아니라 SentenceTransformer 로 바로 쓴다.

# 변수: '이름 = 값' 으로 값에 이름을 붙인다. 대문자 이름은 "바꾸지 않을 설정값"이라는 관례다.
# 이 문자열은 Hugging Face에 올라온 모델 이름이다. HF_HOME(/ssd/hf)에 받아 둔 파일을 찾을 때도 쓴다.
MODEL_ID = "google/embeddinggemma-2"


# def 함수이름(매개변수들):  → 함수 정의. 아래에 들여쓴(4칸) 줄들이 함수 몸통이다.
#   파이썬은 중괄호 { } 대신 '들여쓰기'로 블록을 구분한다. 들여쓰기가 틀리면 에러가 난다.
# vision=False : 매개변수의 기본값. load() 로 부르면 vision은 False, load(vision=True) 로 부르면 True.
def load(vision=False, audio=False):
    # {} : 빈 딕셔너리(사전). '키 → 값' 쌍을 담는 자료형이다. 예: {"이름": "지원"}
    # 모델에 넘길 설정을 여기에 모은다.
    cfg = {}

    # if 조건: 한 줄짜리는 콜론 뒤에 바로 써도 된다.
    # not vision : vision이 False면 True. 즉 "영상을 안 쓰면"
    # cfg["vision_config"] = None : 사전에 '영상 인코더 설정 = 없음'을 넣는다.
    #   → 모델이 영상 인코더를 아예 만들지 않아 GPU 메모리와 로딩 시간을 아낀다.
    if not vision: cfg["vision_config"] = None   # 미사용 인코더는 로드하지 않음
    if not audio:  cfg["audio_config"] = None    # 음성도 같은 방식

    # return 값 : 함수가 일을 마치고 돌려주는 결과. 부른 쪽에서  model = load()  처럼 받는다.
    # SentenceTransformer(...) : 모델을 읽어 와서 encode() 를 쓸 수 있는 객체를 만든다.
    #   괄호 안 '이름=값' 들은 키워드 인자로, 순서와 상관없이 이름으로 값을 넘긴다.
    return SentenceTransformer(
        MODEL_ID,                    # 어떤 모델을 읽을지
        device="cuda",               # GPU에 올린다 ("cpu"로 바꾸면 CPU에서 돌아가지만 훨씬 느리다)
        model_kwargs={"torch_dtype": torch.bfloat16},  # 숫자를 bfloat16(16비트)로 저장해 메모리를 절반으로
                                                       # float16 금지: 이 모델은 float16에서 값이 넘쳐 결과가 깨진다
        config_kwargs=cfg,           # 위에서 만든 '안 쓸 인코더' 설정을 넘긴다
    )
