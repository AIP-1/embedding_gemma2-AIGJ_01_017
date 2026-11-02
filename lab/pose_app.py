# pose_app.py - 9단계 사람 동작 인식 (만세·한팔들기·손 흔들기·악수) + 행동 패턴 + GRU 모델 (README 9단계)
# 목적: 웹캠 영상에서 사람의 관절 17개를 찾고(YOLO11n-pose), 관절 좌표의 위치 관계와 시간 패턴으로
#       동작을 판정해 브라우저에서 실시간 영상·현재 상태·동작 기록을 보여 준다.
#       사람 사진은 저장하지 않는다. 저장하는 것은 동작 기록과(--record 때) 정규화된 관절 숫자뿐이다.
# 실행: ~/eg2-lab 에서 (pose_model.py가 같은 폴더에 있어야 한다)
#   python pose_app.py --raw                      9-1 프레임마다 판정 (문제 확인)
#   python pose_app.py                            9-2 0.6초 다수결 + 손 흔들기 (개선)
#   python pose_app.py --record data/action/wave.csv     9-3 학습 데이터 녹화
#   python pose_app.py --behavior                 9-5 행동 패턴 (지나감/힐끗 봄/관심 있음/머무름)
#   python pose_app.py --behavior --model models/action.pt --model models/behavior.pt   9-6 학습 모델
#   → 맥북 브라우저에서 http://<JETSON_IP>:8000
#
# 수정 기록
# 2026-10-27  v1  README 9단계 pose_app.py를 바탕으로 작성, 줄별 설명 주석 추가 (지원 님 요청: 파이썬 복습용).
#                 README와 다른 점 (모두 8단계 app.py에서 겪은 문제를 미리 막기 위함):
#                 1) det.track()의 half=True → quantize=16. ultralytics 8.4에서 half가 지원 중단 예정이라 경고가 프레임마다 찍힘
#                 2) uvicorn.run()에 timeout_graceful_shutdown=1. 열린 /stream 연결 때문에 Ctrl+C로 안 꺼지던 문제
#                 3) 종료 신호(stop) + try/finally + join. Ctrl+C 때 'Aborted (core dumped)'가 나던 문제.
#                    --record 중이면 종료 때 CSV를 flush·close 해서 마지막 줄이 잘리지 않게 함
#                 그 밖에 README 코드의 작은 오류 2개 수정:
#                 4) /events에서 tracks를 list()로 복사 후 순회. 웹 요청과 영상 스레드가 동시에 tracks를 고치면
#                    'dictionary changed size during iteration' 오류가 날 수 있어서
#                 5) behavior()에서 최근 2초 기록이 비어 있으면 '이동 아님'으로 처리. 빈 목록의 중앙값(np.median)은
#                    nan + 'Mean of empty slice' 경고를 내기 때문
# 2026-10-27  v2  학습 모델(--model)이 idle을 내면 규칙 라벨을 유지하도록 변경. README 코드는 idle을 "기본"으로 덮어써서,
#                 동작 모델이 배우지 않은 만세·한팔들기를 하는 동안 모델이 idle이라 하면 규칙 판정이 사라졌기 때문
#                 (--model 없이 실행하는 9-1~9-5에는 영향 없음)
# 2026-10-27  v3  학습 모델 판정에 다수결 안정화 추가: 최근 3번의 모델 판정 중 2번 이상 같은 라벨일 때만 바꾼다.
#                 v2 실시간 확인에서 동작 1번에 wave 7, arms_crossed 7, clap 6이 기록됐기 때문. 모델 판정이
#                 확률 0.7 근처에서 켜졌다 꺼졌다 하며 규칙 라벨(기본)과 번갈아 바뀔 때마다 새 동작으로 세어졌다.
#                 또 모델 영문 라벨을 한글로 바꿔 표시(ML_KO). raise_both/raise_one으로 녹화하면 만세/한팔들기로
#                 합쳐져 규칙 결과와 같은 이름으로 집계된다
# 2026-11-02  제출용 사본: 주석과 안내 출력 문구의 Jetson 접속 주소를 <JETSON_IP>, 계정명을 <USER>로 바꿈 (공개 리포에 개인 접속 정보를 올리지 않으려고). 코드 동작은 같음

import time, threading, collections, argparse, csv   # time 시각 / threading 스레드 / collections deque·Counter / argparse 옵션 / csv 파일
import cv2, numpy as np, uvicorn                       # cv2 웹캠·그리기 / numpy 계산 / uvicorn 웹 서버
from fastapi import FastAPI                                          # 웹 앱 뼈대
from fastapi.responses import HTMLResponse, StreamingResponse        # HTML 화면 / 끊임없이 보내는 응답(영상)
from ultralytics import YOLO                                         # 자세 추정 모델(yolo11n-pose)도 YOLO 클래스로 연다
# 같은 폴더의 pose_model.py에서 필요한 이름만 가져온다. ( ) 로 감싸면 여러 줄에 걸쳐 쓸 수 있다
from pose_model import (KP_CONF, NOSE, LS, RS, LW, RW, body, facing_score,
                        frame_features, load_model, predict)

# ---------------- 실행 옵션 (명령줄) ----------------
# --raw          : 프레임마다 판정한 결과를 그대로 표시 (9-1 문제 확인용)
# (기본)         : 0.6초 다수결 안정화 + 손 흔들기(시간 패턴) 판정 (9-2 개선)
# --record F.csv : 관절 특징을 CSV로 저장 (9-3 학습 데이터 수집)
# --behavior     : 정면 응시·이동 속도로 행동 패턴(지나감/힐끗 봄/관심 있음/머무름) 판정 (9-5)
# --model M.pt   : 학습한 GRU 모델로 판정 (9-6). 여러 번 지정 가능 (동작용, 행동용)
ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0, help="웹캠 번호 (/dev/video0 → 0)")
ap.add_argument("--conf", type=float, default=0.5, help="사람 검출 신뢰도")
ap.add_argument("--raw", action="store_true")        # store_true : 옵션을 쓰면 True, 안 쓰면 False (값 없이 쓰는 스위치)
ap.add_argument("--record", default=None, help="예: data/wave.csv")
ap.add_argument("--behavior", action="store_true")
# action="append" : 같은 옵션을 여러 번 쓰면 리스트에 차례로 쌓인다 → ["models/action.pt", "models/behavior.pt"]
ap.add_argument("--model", action="append", default=[], help="예: models/action.pt")
ap.add_argument("--port", type=int, default=8000, help="다른 학생이 8000을 쓰고 있으면 8001 등으로")
args = ap.parse_args()
MODE = "raw" if args.raw else "smooth"               # 화면에 보여 줄 모드 이름

# ---------------- 설정값 ----------------
W, H = 1280, 720      # 웹캠 해상도. fps가 10 미만이면 640, 480으로 낮춘다
SMOOTH_S = 0.6        # 다수결 창(초): 최근 0.6초 판정 중 60% 이상인 라벨만 확정
WAVE_S = 1.5          # 손 흔들기 판정 창(초)
BEH_WIN = 15.0        # 행동 패턴 판정에 쓰는 최근 기록 길이(초)
FACE_TH = 0.6         # facing 점수가 이 값 이상이면 "정면을 보고 있음"
LOOK_S, INTEREST_S, LINGER_S = 1.0, 3.0, 10.0   # 힐끗 봄 기준 1초 / 관심 있음 3초 연속 / 머무름 10초 체류
MOVE_TH = 0.5         # 이동 판정 속도 (어깨 너비/초). 이보다 빠르면 걷는 중
ML_TH = 0.7           # 학습 모델 확률이 이 값 이상일 때만 모델 결과를 쓴다 (아니면 규칙 결과)
# 영상 위 글자는 OpenCV가 한글을 못 그려서 영문으로 바꿔 쓴다 (웹 화면은 한글)
EN = {"만세": "BOTH ARMS UP", "한팔들기": "ONE ARM UP", "손흔들기": "WAVE", "악수": "HANDSHAKE",
      "기본": "-", "지나감": "PASSING", "힐끗 봄": "GLANCE", "관심 있음": "INTERESTED",
      "머무름": "LINGER", "관찰 중": "...", "박수": "CLAP", "팔짱": "ARMS CROSSED"}
# 학습 모델 라벨(파일 이름) → 화면·기록에 쓸 한글 이름 (v3). 여기에 없는 라벨은 파일 이름 그대로 쓴다
ML_KO = {"idle": "기본", "wave": "손흔들기", "clap": "박수", "arms_crossed": "팔짱",
         "raise_both": "만세", "raise_one": "한팔들기",
         "passing": "지나감", "glance": "힐끗 봄", "interested": "관심 있음", "linger": "머무름"}
ML_VOTE = 3           # 모델 판정 다수결에 쓰는 최근 판정 수 (5프레임마다 1번 → 3번이면 약 1.2초) (v3)
# 한 사람의 최종 행동 = 머무는 동안 도달한 가장 높은 단계. 숫자가 클수록 높은 단계
BEH_RANK = {"관찰 중": 0, "지나감": 1, "머무름": 2, "힐끗 봄": 3, "관심 있음": 4}

det = YOLO("yolo11n-pose.pt")         # 자세 추정 모델. 첫 실행 시 ~/eg2-lab에 자동 다운로드 (약 6 MB)
models = [load_model(p) for p in args.model]          # 리스트 컴프리헨션: --model로 받은 파일마다 load_model 실행
# 학습 모델이 필요로 하는 가장 긴 윈도 길이만큼 특징을 보관. 모델이 없으면 1 (default=1 : 빈 리스트일 때의 값)
SEQ_LEN = max([m["window"] for m in models], default=1)
print(f"[pose mode={MODE}] conf={args.conf} behavior={args.behavior} "
      f"models={[(m['task'], m['labels']) for m in models]}")

# ---------------- 스레드가 함께 쓰는 상태 ----------------
frame_lock, ev_lock = threading.Lock(), threading.Lock()   # Lock : 두 스레드가 같은 값을 동시에 고치지 않게 하는 자물쇠
latest_jpeg = None    # 가장 최근에 만든 영상 프레임(JPEG 바이트). 웹 /stream이 꺼내 간다
tracks = {}           # 추적 ID -> 그 사람의 기록 사전 (new_track 참고)
pairs = {}            # (ID_a, ID_b) -> deque[(시각, 악수 조건 만족 여부)]
pair_on = set()       # 지금 "악수 중"으로 확정된 두 사람 쌍
events = collections.deque(maxlen=50)    # 동작 기록. maxlen=50 : 50개가 넘으면 가장 오래된 것부터 자동으로 버림
counts = collections.Counter()           # Counter : 키별 개수를 세는 사전. counts["만세"] += 1
beh_counts = collections.Counter()       # 사람(추적 ID) 단위 최종 행동 개수
stats = {"fps": 0.0, "people": 0}
stop = threading.Event()                 # 종료 신호. stop.set() 하면 반복문들이 빠져나온다 (v1, 8단계 v5와 같은 방식)
rec = None                               # CSV 기록기. --record를 안 쓰면 None
if args.record:
    rec_f = open(args.record, "a", newline="")   # "a" : 이어 쓰기 모드. 같은 파일로 여러 번 녹화하면 뒤에 붙는다
    rec = csv.writer(rec_f)                      # 리스트 하나를 CSV 한 줄로 써 주는 도구


def new_track(now):
    """새로 나타난 사람(추적 ID)의 빈 기록을 만든다."""
    return {"raw": collections.deque(maxlen=90),     # (시각, 프레임 판정 라벨) 최근 90개
            "wave": collections.deque(maxlen=90),    # (시각, 올린 손목의 x) 손 흔들기 판정용
            "beh": collections.deque(maxlen=600),    # (시각, 정면 점수, 속도) 행동 패턴용
            "feat": collections.deque(maxlen=SEQ_LEN),   # 학습 모델에 넣을 최근 특징 벡터들
            "label": "기본", "bstate": "관찰 중", "best": "관찰 중", "ml": {},   # 현재 동작 / 현재 행동 / 최고 행동 / 모델 결과
            "mlh": collections.defaultdict(lambda: collections.deque(maxlen=ML_VOTE)),   # 작업별 최근 모델 판정 (v3)
            "first": now, "seen": now, "prev": None, "look": 0.0, "n": 0}     # 처음·마지막 본 시각, 이전 위치, 정면 누적 시간, 프레임 수


def pose_label(k, b):
    """한 프레임만 보고 자세 라벨을 정한다 (규칙)."""
    (cx, cy), sw = b                                 # 어깨 중심 (x, y)와 어깨 너비를 한 번에 풀기
    # lambda : 한 줄짜리 이름 없는 함수. w(손목 번호)를 받아 "그 손목이 보이고, 어깨보다 0.3 어깨너비 이상 위인가"
    # y는 작을수록 위쪽이라 "위" = y가 cy - 0.3*sw 보다 작음
    raised = lambda w: k[w, 2] > KP_CONF and k[w, 1] < cy - 0.3 * sw
    lu, ru = raised(LW), raised(RW)                  # 왼손목·오른손목이 올라갔는지 (True/False)
    if lu and ru:
        return "만세"
    if lu or ru:
        return "한팔들기"
    return "기본"


def is_wave(hist):
    """올린 손목의 x 좌표(어깨 너비로 정규화)가 좌우로 2번 이상 방향을 바꾸면 손 흔들기"""
    xs = [x for _, x in hist]                        # (시각, x) 목록에서 x만 꺼내기. _ 는 안 쓰는 값
    if len(xs) < 8:                                  # 1.5초 창에 8프레임 미만이면 판단 보류 (최소 약 5 fps)
        return False
    changes, last_dir, anchor = 0, 0, xs[0]          # 방향 전환 횟수, 마지막 방향(+1 오른쪽/-1 왼쪽), 기준점
    for x in xs[1:]:                                 # xs[1:] : 두 번째부터 끝까지
        d = x - anchor
        if abs(d) > 0.15:                            # 기준점에서 어깨 너비 0.15배 이상 움직였을 때만 "움직임"으로 셈 (떨림 무시)
            cur = 1 if d > 0 else -1
            if last_dir and cur != last_dir:         # 이전 방향이 있고(0이 아니고) 방향이 바뀌었으면
                changes += 1
            last_dir, anchor = cur, x                # 방향·기준점 갱신
    return changes >= 2


def behavior(st, now):
    """정면 연속 시간·누적 시간, 이동 여부, 체류 시간으로 행동 패턴을 정한다 (규칙)."""
    h = [x for x in st["beh"] if now - x[0] <= BEH_WIN]   # 최근 15초 기록만
    if len(h) < 5:
        return "관찰 중"
    run = 0.0                                   # 끝(지금)에서부터 정면을 계속 본 시간
    for i in range(len(h) - 1, 0, -1):          # range(시작, 끝, -1) : 뒤에서 앞으로 거꾸로 돌기
        if h[i][1] < FACE_TH:                   # 정면이 아닌 프레임을 만나면 멈춘다
            break
        run = h[-1][0] - h[i - 1][0]
    # 15초 동안 정면을 본 시간의 합 (프레임 사이 시간 간격을 정면인 프레임만 더함)
    look = sum(h[i][0] - h[i - 1][0] for i in range(1, len(h)) if h[i][1] >= FACE_TH)
    recent = [s for t, _, s in h if now - t < 2.0]   # 최근 2초의 속도들
    # 속도의 중앙값으로 이동 여부 판단 (평균보다 튀는 값에 덜 흔들림). 비어 있으면 이동 아님 (v1 수정 5)
    moving = bool(recent) and np.median(recent) > MOVE_TH
    if run >= INTEREST_S and not moving:
        return "관심 있음"
    if moving and LOOK_S <= look < INTEREST_S:
        return "힐끗 봄"
    if moving:
        return "지나감"
    if now - st["first"] >= LINGER_S:
        return "머무름"
    return "관찰 중"


def majority(dq, now, window):
    """최근 window초의 값 중 가장 많은 값과 그 비율 (다수결)"""
    vals = [v for t, v in dq if now - t <= window]
    if not vals:
        return None, 0.0
    v, n = collections.Counter(vals).most_common(1)[0]   # most_common(1) : [(가장 많은 값, 개수)]
    return v, n / len(vals)


def log_event(now, who, label):
    """웹의 동작 기록과 횟수에 한 건 추가"""
    with ev_lock:                                    # with 자물쇠: 이 블록 동안 다른 스레드는 기다린다
        events.appendleft({"ts": now, "who": who, "label": label})   # appendleft : 맨 앞(최신)에 넣기
        counts[label] += 1


def close_track(tid, st, now):
    """추적이 끝난(화면에서 사라진) 사람의 요약을 남긴다 (--behavior일 때만)"""
    if not args.behavior or st["best"] == "관찰 중":
        return
    dwell = st["seen"] - st["first"]                 # 머문 시간
    with ev_lock:
        events.appendleft({"ts": now, "who": f"#{tid}",
                           "label": f"요약: 체류 {dwell:.0f}초, 정면 {st['look']:.0f}초 → {st['best']}"})
        beh_counts[st["best"]] += 1


def handshake_pairs(kps, ids, now):
    """두 사람의 손목이 가까이(어깨 너비 0.5배 이내) 있고 높이가 가슴~허리 사이면 악수 후보"""
    found = set()
    for i in range(len(ids)):                        # 가능한 모든 두 사람 조합 (i < j)
        for j in range(i + 1, len(ids)):
            a, b = kps[i], kps[j]
            ba, bb = body(a), body(b)
            if ba is None or bb is None:
                continue                             # 어깨가 안 보이면 이 쌍은 건너뜀
            sw = (ba[1] + bb[1]) / 2                 # 두 사람 어깨 너비의 평균
            best = 1e9                               # 가장 가까운 손목 거리 (처음엔 아주 큰 수)
            for wa in (LW, RW):                      # 두 사람의 손목 4가지 조합 모두 확인
                for wb in (LW, RW):
                    if a[wa, 2] > KP_CONF and b[wb, 2] > KP_CONF:
                        mid_y = (a[wa, 1] + b[wb, 1]) / 2           # 두 손목의 중간 높이
                        if ba[0][1] < mid_y < ba[0][1] + 2.5 * ba[1]:   # 어깨 아래 ~ 어깨너비 2.5배 아래(허리쯤)
                            best = min(best, np.linalg.norm(a[wa, :2] - b[wb, :2]))   # 두 점 사이 거리
            key = tuple(sorted((ids[i], ids[j])))    # (작은 ID, 큰 ID) 로 순서를 고정해 같은 쌍을 같은 키로
            # setdefault : 키가 없으면 새 deque를 넣고, 있으면 기존 것을 돌려준다
            pairs.setdefault(key, collections.deque(maxlen=60)).append((now, best < 0.5 * sw))
            found.add(key)
    return found


def update_person(tid, k, now):
    """한 사람의 이번 프레임 관절로 기록을 갱신하고 동작·행동 라벨을 정한다."""
    st = tracks.setdefault(tid, new_track(now))      # 처음 보는 ID면 새 기록
    dt = now - st["seen"]                            # 지난 프레임과의 시간 차
    st["seen"] = now
    b = body(k)
    if b is None:                                    # 어깨가 안 보이면 이번 프레임은 판정하지 않음
        return st
    raw = pose_label(k, b)                           # 이 프레임만 본 판정
    fa = facing_score(k)                             # 정면 점수
    # 이동 속도 = 어깨 중심이 움직인 거리 ÷ 어깨 너비 ÷ 시간. linalg.norm : 벡터 길이(거리)
    spd = 0.0 if st["prev"] is None else float(np.linalg.norm(b[0] - st["prev"]) / b[1] / max(dt, 1e-3))
    st["prev"] = b[0]
    st["raw"].append((now, raw))
    st["beh"].append((now, fa, spd))
    if fa >= FACE_TH and st["n"] > 0:
        st["look"] += dt                             # 정면을 본 누적 시간
    st["n"] += 1
    if raw in ("한팔들기", "만세"):           # 손 흔들기용: 더 높이 든 손목의 x (어깨 중심 기준, 어깨 너비 단위)
        w = LW if k[LW, 1] < k[RW, 1] else RW
        st["wave"].append((now, (k[w, 0] - b[0][0]) / b[1]))
    feat = frame_features(k, b, fa, spd)
    st["feat"].append(feat)
    if rec:
        # CSV 한 줄 = 시각, ID, 규칙 라벨, 정면 점수, 속도 + 좌표 34 + 신뢰도 17 (합 56칸). round : 소수점 자리 줄이기
        # feat[:51] : 특징 앞 51개(좌표+신뢰도). 정면·속도는 앞쪽 칸에 따로 썼으므로 뺀다. tolist() : 파이썬 리스트로
        rec.writerow([round(now, 3), tid, raw, round(fa, 3), round(spd, 3)] + feat[:51].round(3).tolist())

    # ---- 동작 (규칙) ----
    if MODE == "raw":
        label = raw                                  # 9-1: 프레임 판정 그대로
    else:
        v, frac = majority(st["raw"], now, SMOOTH_S)
        label = v if frac >= 0.6 else st["label"]    # 60% 미만이면 이전 라벨 유지 (깜빡임 방지)
        recent = [(t, x) for t, x in st["wave"] if now - t <= WAVE_S]
        if label == "한팔들기" and is_wave(recent):
            label = "손흔들기"

    # ---- 학습 모델 (5프레임마다, 계산을 아끼려고) ----
    if models and st["n"] % 5 == 0:                  # % : 나머지. n이 5의 배수일 때만
        seq = list(st["feat"])
        for m in models:
            if len(seq) >= m["window"]:              # 윈도만큼 쌓였을 때만 판정
                lab, p = predict(m, seq[-m["window"]:])   # seq[-30:] : 마지막 30개
                hist = st["mlh"][m["task"]]          # 이 작업(action/behavior)의 최근 판정 3개
                hist.append(ML_KO.get(lab, lab) if p >= ML_TH else None)   # 확신이 낮으면 None ("모르겠음")
                # 다수결 (v3): 3번 중 2번 이상 나온 값으로만 바꾸고, 아니면 이전 결과를 유지 → 깜빡임 방지
                v, n = collections.Counter(hist).most_common(1)[0]
                if n >= 2:
                    st["ml"][m["task"]] = (v, p) if v is not None else None   # None 다수면 규칙 사용
    ml_act = st["ml"].get("action")                  # .get : 키가 없으면 None
    # 모델이 idle(기본)이라고 하면 규칙 라벨을 그대로 둔다 (v2).
    # 모델이 배우지 않은 동작(예: 만세)을 하는 중에 모델이 기본이라 해도 규칙의 "만세"를 지우면 안 된다
    if ml_act and ml_act[0] != "기본":
        label = ml_act[0]                            # 손흔들기, 박수 등 모델이 배운 동작이면 규칙 라벨 대신 사용

    if label != st["label"]:                         # 라벨이 바뀌는 순간에만 기록 (같은 동작을 여러 번 세지 않게)
        if label != "기본":
            log_event(now, f"#{tid}", label)
        st["label"] = label

    # ---- 행동 패턴 ----
    if args.behavior:
        bs = behavior(st, now)
        ml_beh = st["ml"].get("behavior")
        if ml_beh:
            bs = ml_beh[0]
        if bs != st["bstate"]:
            if bs != "관찰 중":
                log_event(now, f"#{tid}", bs)
            st["bstate"] = bs
        if BEH_RANK.get(bs, 1) > BEH_RANK.get(st["best"], 0):   # 모델 라벨(영문)은 순위표에 없어 1로 취급
            st["best"] = bs
    return st


def capture_loop():
    """스레드: 웹캠 → 자세 추정·추적 → 사람마다 판정 → 글자 그린 영상을 JPEG로."""
    global latest_jpeg
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # MJPG로 받아야 1280x720에서 fps가 나온다
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    assert cap.isOpened(), "웹캠을 열 수 없음 (README Q11 참고)"
    t0, n = time.time(), 0
    while not stop.is_set():                         # 종료 신호가 올 때까지
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1); continue
        # track : 사람 검출 + 관절 추정 + 같은 사람에게 같은 ID. quantize=16 : FP16 계산 (v1 수정 1)
        r = det.track(frame, persist=True, conf=args.conf, quantize=16, verbose=False)[0]
        now = time.time()
        vis = r.plot(labels=False)        # 박스 + 관절 뼈대를 그린 이미지 (클래스 글자는 빼고)
        ids, kps, boxes = [], [], []
        if r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().tolist()                          # 추적 ID 목록
            # keypoints.xy : (사람 수, 17, 2) 좌표, keypoints.conf : (사람 수, 17) 신뢰도
            # [..., None] : 마지막에 차원 추가 → (사람 수, 17, 1). 이어 붙여 (사람 수, 17, 3) = [x, y, 신뢰도]
            kps = np.concatenate([r.keypoints.xy.cpu().numpy(),
                                  r.keypoints.conf.cpu().numpy()[..., None]], axis=-1)
            boxes = r.boxes.xyxy.int().tolist()                      # 박스 [x1, y1, x2, y2]
        stats["people"] = len(ids)

        for tid, k, (x1, y1, x2, y2) in zip(ids, kps, boxes):
            st = update_person(tid, k, now)
            txt = f"#{tid} {EN.get(st['label'], st['label'])}"      # EN에 없으면(모델 라벨 wave 등) 그대로
            if args.behavior:
                txt += f" | {EN.get(st['bstate'], st['bstate'])}"
            # 동작 중이거나 관심 있음이면 주황, 아니면 초록 (OpenCV 색은 BGR 순서)
            color = (0, 200, 255) if st["label"] != "기본" or st["bstate"] == "관심 있음" else (0, 255, 0)
            cv2.putText(vis, txt, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ---- 2인 동작 (악수) ----
        active = handshake_pairs(kps, ids, now) if len(ids) >= 2 else set()
        for key in list(pairs):                      # list()로 복사해서 돌아야 도는 중에 pop 해도 안전
            if key not in active:                    # 이번 프레임에 둘 중 하나가 없으면 그 쌍은 정리
                pairs.pop(key); pair_on.discard(key); continue
            if MODE == "raw":
                on = pairs[key][-1][1]               # 이번 프레임 결과 그대로
            else:
                v, frac = majority(pairs[key], now, SMOOTH_S)
                on = bool(v) and frac >= 0.6
            if on and key not in pair_on:            # 꺼져 있다가 켜지는 순간에만 기록
                log_event(now, f"#{key[0]}+#{key[1]}", "악수")
            (pair_on.add if on else pair_on.discard)(key)   # 조건에 따라 add 또는 discard 함수를 골라 실행
        for a, b_ in pair_on:
            if a in ids and b_ in ids:
                ba, bb = boxes[ids.index(a)], boxes[ids.index(b_)]
                x, y = (ba[0] + bb[2]) // 2 - 80, min(ba[1], bb[1]) + 30   # 두 박스 사이 위쪽. // : 정수 나눗셈
                cv2.putText(vis, EN["악수"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

        # 3초 이상 안 보인 사람은 추적 종료 + 요약
        for tid in [t for t, s in tracks.items() if now - s["seen"] > 3]:
            close_track(tid, tracks.pop(tid), now)

        n += 1
        if now - t0 >= 1:                            # 1초마다 fps 계산
            stats["fps"], t0, n = n / (now - t0), now, 0
            if rec:
                rec_f.flush()                        # 버퍼에 쌓인 CSV 줄을 디스크에 실제로 쓴다
        cv2.putText(vis, f"{stats['fps']:.1f} fps | {MODE} | people {stats['people']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()
    cap.release()                                    # 반복이 끝나면(종료 신호) 웹캠을 닫는다 (v1 수정 3)


app = FastAPI()


@app.get("/stream")
def stream():
    def gen():
        while not stop.is_set():                     # 종료 신호가 오면 영상 전송도 끝낸다
            with frame_lock:
                f = latest_jpeg
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/events")
def get_events():
    """웹 화면이 1초마다 가져가는 현재 상태 (JSON)"""
    snap = list(tracks.items())                      # 영상 스레드가 바꾸기 전에 복사 (v1 수정 4)
    with ev_lock:
        total = sum(beh_counts.values())
        return {"mode": MODE + (" +behavior" if args.behavior else "") + (" +model" if models else ""),
                "stats": stats, "counts": dict(counts), "events": list(events)[:20],
                "beh": dict(beh_counts),
                "interest_rate": round(beh_counts["관심 있음"] / total, 2) if total else None,
                "now": {f"#{t}": s["label"] + (f" / {s['bstate']}" if args.behavior else "")
                        for t, s in snap}}


# 브라우저에 보낼 HTML 한 장 (영상 + 1초마다 /events를 읽어 표 갱신)
PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Jetson 동작 인식</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
img{width:100%;max-width:960px;border-radius:8px}table{border-collapse:collapse;margin-top:8px}
td,th{border:1px solid #444;padding:4px 10px;font-size:14px}.chip{display:inline-block;background:#333;
border-radius:12px;padding:4px 10px;margin:2px}</style>
<h2>실시간 영상 <span id=mode></span></h2><img src="/stream">
<h2>현재 상태</h2><div id=now></div>
<h2>동작별 횟수</h2><div id=cnt></div>
<h2>관심도 통계 (사람 단위)</h2><div id=beh></div>
<h2>동작 기록</h2><table><thead><tr><th>시각</th><th>대상</th><th>동작</th></tr></thead><tbody id=ev></tbody></table>
<script>
const chips=o=>Object.entries(o).map(([k,v])=>`<span class=chip>${k} ${v}</span>`).join('')||'-';
async function poll(){const r=await (await fetch('/events')).json();
 document.getElementById('mode').textContent=`[${r.mode}] ${r.stats.fps.toFixed(1)} fps, ${r.stats.people}명`;
 document.getElementById('now').innerHTML=Object.entries(r.now).map(([k,v])=>`<span class=chip>${k}: ${v}</span>`).join('')||'사람 없음';
 document.getElementById('cnt').innerHTML=chips(r.counts);
 document.getElementById('beh').innerHTML=chips(r.beh)+(r.interest_rate!==null?` <b>관심 비율 ${(r.interest_rate*100).toFixed(0)}%</b>`:'');
 document.getElementById('ev').innerHTML=r.events.map(e=>`<tr><td>${new Date(e.ts*1000).toLocaleTimeString()}</td><td>${e.who}</td><td>${e.label}</td></tr>`).join('');}
setInterval(poll,1000);poll();
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    worker = threading.Thread(target=capture_loop, daemon=True)   # 영상 처리 스레드
    worker.start()
    try:
        # timeout_graceful_shutdown=1 : Ctrl+C 후 열린 /stream 연결을 1초만 기다리고 닫는다 (v1 수정 2)
        uvicorn.run(app, host="0.0.0.0", port=args.port, timeout_graceful_shutdown=1)
    except KeyboardInterrupt:
        pass                                         # Ctrl+C는 정상 종료 방법
    finally:                                         # 어떻게 끝나든 꼭 실행 (v1 수정 3)
        stop.set()                                   # 영상 스레드에 "그만" 신호
        worker.join(timeout=5)                       # 하던 프레임을 마치고 끝날 때까지 최대 5초 기다림
        if rec:
            rec_f.flush(); rec_f.close()             # 녹화 파일 마무리
            print(f"녹화 저장: {args.record}")
        print("종료 완료")
