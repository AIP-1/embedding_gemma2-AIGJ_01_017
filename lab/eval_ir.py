# eval_ir.py - README 7단계 검색 성능 평가 (학습 전후 모두 이 파일로 잰다)
# 목적: 강의실 도메인 질의 60개(data/eval.jsonl)로 "질의 → 정답 문서"를 얼마나 잘 찾는지 점수로 낸다.
#       같은 파일을 학습 전(원본 모델)과 학습 후(models/eg2-classroom)에 돌려 두 점수를 비교한다.
# 실행: ~/eg2-lab 에서 venv를 켠 뒤
#         python eval_ir.py                          → 원본 모델 google/embeddinggemma-2 평가
#         python eval_ir.py models/eg2-classroom     → 미세 조정한 모델 평가
# 결과: 화면에 표를 찍고, 같은 내용을 ~/eg2-lab/results_step7.md 끝에 덧붙인다 (실행할 때마다 한 묶음씩 쌓임).
# 주의: train_ft.py 가 이 파일의 build_evaluator() 를 가져다 쓴다. 두 파일은 같은 폴더에 있어야 한다.
#
# 수정 기록
# 2026-10-26  v1  README 7단계 eval_ir.py 를 바탕으로 작성. README와 달라진 점 (왜):
#                 - 768차원뿐 아니라 256/128차원 점수도 잼 (MatryoshkaLoss로 학습한 뒤에도 짧은 벡터가 유지되는지 보려고)
#                 - 강의실과 관계없는 일반 문장 8쌍 점수도 잼 (README '일반 성능이 떨어지지 않았는지' 확인을 한 번에 하려고)
#                 - 결과 키를 이름 끝부분으로 찾음 (sentence-transformers 버전에 따라 키 앞부분이 달라져도 동작하게)
#                 - 결과를 results_step7.md 에 저장 (보고서에 붙여 넣기용, 6단계 measure.py 와 같은 방식)
# 2026-10-26  v2  InformationRetrievalEvaluator 를 새 경로에서 가져오도록 변경 (Jetson 실행 시 DeprecationWarning 이 나서. 결과는 같음)

# ── 점수 읽는 법 ──────────────────────────────────────────────
# 질의마다 정답 문서가 딱 1개다. 모델이 문서 20개를 유사도 순으로 줄 세운다.
#   accuracy@1 : 정답이 1등인 질의의 비율 (0~1). 0.8 이면 60개 중 48개가 1등으로 맞음
#   accuracy@3 : 정답이 3등 안에 든 비율. 정답이 1개뿐이라 recall@3 과 같은 값이다
#   ndcg@10    : 정답이 10등 안에서 '얼마나 위에' 있는지 점수. 1등=1.0, 2등=0.63, 3등=0.5 ... 순위가 낮을수록 작아짐
#                (1/log2(순위+1) 을 질의마다 구해 평균)
# ────────────────────────────────────────────────────────────

import datetime               # 결과 파일에 실행 날짜·시각을 적으려고
import json                   # .jsonl 한 줄(문자열)을 파이썬 사전으로 바꾸려고
import sys                    # 실행할 때 넘긴 인자(sys.argv) 읽기
import time                   # 평가에 걸린 시간 측정
import torch                  # bfloat16 이름을 쓰려고
from sentence_transformers import SentenceTransformer
# sentence-transformers 6.2 부터 새 경로. 예전 경로도 아직 되지만 DeprecationWarning(곧 없어질 예정 경고)이 나와서 새 경로를 쓴다.
# try/except : try 블록을 먼저 해 보고, ImportError(못 찾음)가 나면 except 블록(예전 경로)으로 대신한다. 버전이 달라도 동작하게
try:
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
except ImportError:
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
# InformationRetrievalEvaluator : sentence-transformers 가 제공하는 '검색 평가 도구'.
#   질의 목록, 문서 목록, 정답 짝을 주면 모든 질의를 검색해 보고 accuracy@k, ndcg@k 등을 계산해 준다.

# 강의실과 관계없는 일반 문장 (도메인 밖). 학습 데이터에 없는 내용이라
# 미세 조정 후 이 점수가 크게 떨어지면 '강의실 문장에만 맞춰지고 일반 능력을 잃은 것(과적합)'이다.
# (질의, 정답 문서) 짝의 리스트. 튜플 (a, b) : 괄호로 묶은 값 묶음, 리스트와 비슷하지만 바꿀 수 없다.
GENERAL = [
    ("찌개 맛있게 끓이는 비결", "김치찌개는 잘 익은 김치와 돼지고기를 먼저 볶은 뒤 끓이면 맛이 깊어진다."),
    ("공차기 시합 한 판은 얼마나 걸려", "축구 경기는 전반과 후반 각 45분으로 진행된다."),
    ("한글 만든 임금", "세종대왕은 1443년에 훈민정음을 창제했다."),
    ("반려묘는 하루에 얼마나 자", "고양이는 하루에 12시간 이상 잠을 잔다."),
    ("기내에서 귀가 먹먹한 이유", "비행기가 오르내릴 때 기압이 바뀌어 고막 안팎의 압력 차이가 생긴다."),
    ("파이썬에서 여러 값을 담는 자료형 만드는 기호", "파이썬 리스트는 대괄호로 만들고 쉼표로 값을 구분한다."),
    ("감기 걸렸을 때 집에서 할 수 있는 일", "감기에는 충분한 수면과 수분 섭취가 회복에 도움이 된다."),
    ("지구에서 가장 높은 산", "에베레스트산은 해발 약 8,849m로 세계에서 가장 높다."),
]


def make_evaluator(rows, name, dim=None):
    """rows: [{"anchor": 질의, "positive": 정답 문서}, ...] 를 받아 평가 도구를 만든다.
    dim: None 이면 768차원 그대로, 256 이면 벡터 앞 256개만 잘라 평가."""
    # 평가 도구가 원하는 형식 3가지 (모두 사전):
    #   corpus   : 문서 ID → 문서 문장         예: {"d0": "강의실 적정 ..."}
    #   queries  : 질의 ID → 질의 문장         예: {"q0": "교실 온도 몇 도가 적당해"}
    #   relevant : 질의 ID → 정답 문서 ID 집합  예: {"q0": {"d0"}}
    corpus, queries, relevant, doc_id = {}, {}, {}, {}
    # enumerate(리스트) : (번호, 값) 짝으로 하나씩 꺼낸다. i = 0, 1, 2, ...
    for i, r in enumerate(rows):
        # 같은 문서에 질의가 3개씩 붙어 있으므로, 문서 문장이 처음 나올 때만 새 ID 를 준다.
        # 사전.setdefault(키, 기본값) : 키가 있으면 그 값을, 없으면 기본값을 넣고 그 값을 돌려준다.
        #   len(doc_id) = 지금까지 본 서로 다른 문서 수 → d0, d1, d2 ... 순서대로 붙는다
        d = doc_id.setdefault(r["positive"], f"d{len(doc_id)}")
        corpus[d] = r["positive"]
        queries[f"q{i}"] = r["anchor"]
        # set() : 집합. 중복 없는 값 묶음. 정답 문서가 여러 개일 수도 있어서 집합으로 둔다 (여기선 늘 1개)
        relevant.setdefault(f"q{i}", set()).add(d)
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name=name,
        query_prompt_name="SearchQuery",   # 질의는 3단계처럼 '검색 질의' 안내 문구를 붙여 임베딩
        corpus_prompt_name="Document",     # 문서는 '문서' 안내 문구
        accuracy_at_k=[1, 3, 5],           # accuracy@1, @3, @5 를 계산
        truncate_dim=dim)                  # None 이면 768 전체, 숫자면 그 길이로 잘라서


def build_evaluator(path="data/eval.jsonl", name="classroom", dim=None):
    """eval.jsonl 을 읽어 평가 도구를 만든다. train_ft.py 도 이 함수를 쓴다 (README와 같은 이름·인자)."""
    # open(...) 으로 연 파일을 for 로 돌리면 한 줄씩 나온다. json.loads(줄) : JSON 문자열 → 사전
    # 'if l.strip()' : 빈 줄(공백만 있는 줄)은 건너뛴다. 파일 끝에 빈 줄이 있어도 에러가 안 나게
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return make_evaluator(rows, name, dim)


def pick(res, metric):
    """평가 결과 사전에서 'cosine_accuracy@1' 처럼 끝이 metric 인 값을 찾아 돌려준다.
    키 앞부분(예: classroom_cosine_ / classroom_256_cosine_)은 버전에 따라 달라서 끝부분으로 찾는다."""
    # next(값 for ... if 조건) : 조건에 맞는 첫 번째 값 하나만 꺼낸다
    return next(v for k, v in res.items() if k.endswith(f"cosine_{metric}"))


# 이 아래는 'python eval_ir.py' 로 직접 실행할 때만 돈다.
# train_ft.py 가 'from eval_ir import build_evaluator' 로 가져갈 때는 실행되지 않는다.
if __name__ == "__main__":
    # sys.argv : 실행 명령의 단어 목록. 'python eval_ir.py models/eg2-classroom' 이면
    #            ['eval_ir.py', 'models/eg2-classroom']. 인자가 없으면 원본 모델 이름을 쓴다.
    path = sys.argv[1] if len(sys.argv) > 1 else "google/embeddinggemma-2"

    # 모델 불러오기. eg2.py 의 load() 와 같은 설정인데, 모델 경로를 바꿀 수 있게 여기서 직접 부른다.
    # 원본 이름이면 HF_HOME(/ssd/hf)에서, 'models/...' 처럼 폴더 경로면 그 폴더에서 읽는다.
    model = SentenceTransformer(path, device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},              # 학습을 float32로 했어도 평가는 bf16 (3~6단계와 같은 조건)
        config_kwargs={"vision_config": None, "audio_config": None})  # 텍스트 전용

    lines = []                                   # 결과 표의 줄들을 모아 두었다가 화면·파일에 함께 쓴다
    lines.append(f"\n### {path}  ({datetime.datetime.now():%Y-%m-%d %H:%M})\n")
    lines.append("| 평가 세트 | 차원 | accuracy@1 | accuracy@3 | ndcg@10 | 시간 (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    # 강의실 평가 세트를 768 / 256 / 128 차원으로 세 번 잰다. None = 자르지 않음(768)
    for dim in [None, 256, 128]:
        t = time.perf_counter()
        res = build_evaluator(dim=dim)(model)    # 평가 도구를 만들고, 바로 (model) 을 넣어 실행 → 결과 사전
        sec = time.perf_counter() - t
        # 조건 표현식 'A if 조건 else B' : 조건이 참이면 A, 아니면 B
        lines.append(f"| 강의실 (eval.jsonl 60개) | {dim or 768} | {pick(res, 'accuracy@1'):.3f} | "
                     f"{pick(res, 'accuracy@3'):.3f} | {pick(res, 'ndcg@10'):.3f} | {sec:.1f} |")

    # 일반 문장 8쌍 (도메인 밖). 768차원만
    # 리스트 컴프리헨션 [식 for ... in ...] : 튜플 (질의, 문서) 를 사전 형식으로 바꿔 새 리스트를 만든다
    general_rows = [{"anchor": q, "positive": d} for q, d in GENERAL]
    t = time.perf_counter()
    res = make_evaluator(general_rows, "general")(model)
    sec = time.perf_counter() - t
    lines.append(f"| 일반 문장 (8개) | 768 | {pick(res, 'accuracy@1'):.3f} | "
                 f"{pick(res, 'accuracy@3'):.3f} | {pick(res, 'ndcg@10'):.3f} | {sec:.1f} |")

    # torch.cuda.max_memory_allocated() : 이 프로세스가 GPU에서 가장 많이 잡았던 메모리(바이트). 1e9 로 나눠 GB
    lines.append(f"\nGPU 메모리 최대 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    text = "\n".join(lines)                      # 줄들을 줄바꿈으로 이어 붙여 문자열 하나로
    print(text)
    # "a" : 덧붙이기(append) 모드. 기존 내용을 지우지 않고 파일 끝에 쓴다. 파일이 없으면 새로 만든다
    with open("results_step7.md", "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n→ results_step7.md 에 덧붙였습니다")
