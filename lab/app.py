# app.py - 8단계 웹캠 실시간 물체 추출과 웹 검색 (README 8단계)
# 목적: USB 웹캠 영상에서 물체를 찾아(YOLO) 잘라 내고, 잘라 낸 사진을 EmbeddingGemma 2로 벡터로 바꿔 저장한다.
#       브라우저에서 실시간 영상을 보고, "세탁기" 같은 글로 저장된 물체를 검색한다.
# 실행: ~/eg2-lab 에서  python app.py --mode yolo | relabel | yoloe   → 브라우저로 http://<Jetson IP>:8000
#   yolo    : YOLO 클래스 그대로 사용 (8-1, 문제 확인용 기준선)
#   relabel : YOLO 박스 + EmbeddingGemma 2 재라벨 (8-2, 개선 1)
#   yoloe   : 글로 클래스를 정하는 YOLOE 검출기 (8-3, 개선 2)
# 모델: 사진을 다루므로 원본 EmbeddingGemma 2를 eg2.load(vision=True)로 쓴다.
#       7단계 models/eg2-classroom 은 텍스트 전용이라 여기서 쓰지 않는다.
#
# 수정 기록
# 2026-10-26  v1  README 8단계 app.py를 바탕으로 작성. 모델 로드를 eg2.load(vision=True)로 통일,
#                 --cam 옵션 추가(웹캠 번호), 줄별 설명 주석 추가 (지원 님 요청: 파이썬 복습용)
# 2026-10-26  v2  det.track()의 half=True → quantize=16. ultralytics 8.4에서 half가 지원 중단 예정이라
#                 프레임마다 경고가 찍혀 Uvicorn 시작 줄 등 다른 출력이 묻혔기 때문 (동작은 같음)
# 2026-10-26  v3  uvicorn.run()에 timeout_graceful_shutdown=1 추가. 브라우저의 /stream 연결이 안 끝나서
#                 Ctrl+C 한 번으로 안 꺼지고, 두 번 누르면 긴 Traceback이 찍혔기 때문 (검출·검색 동작은 같음)
# 2026-10-26  v4  크롭마다 crops/<mode>/labels.csv에 (파일, YOLO 라벨, 신뢰도, 최종 라벨, 점수, 시각) 기록.
#                 재라벨 결과가 웹 화면(메모리)에만 있어서 앱을 끄면 8-4 비교에 쓸 수 없었기 때문
# 2026-10-26  v5  종료 신호(stop)를 넣어 Ctrl+C 때 두 스레드가 스스로 끝나고 웹캠을 닫게 함.
#                 스레드가 GPU·웹캠 작업 도중 강제로 끊겨 'terminate called ... Aborted (core dumped)'가 났기 때문

import os, time, threading, collections, argparse   # 한 줄에 여러 표준 모듈을 쉼표로 이어 불러온다
# os: 폴더 만들기·파일 지우기 / time: 시각·잠깐 쉬기 / threading: 여러 일을 동시에(스레드)
# collections: deque(양쪽 끝에서 넣고 빼는 큐) / argparse: 명령줄 옵션(--mode 등) 읽기
import cv2                      # OpenCV: 웹캠 열기, 박스·글자 그리기, JPEG 변환
import numpy as np              # 벡터 계산 (유사도 = 행렬 곱)
import torch                    # 모델 추론 중 기울기 계산 끄기(inference_mode)에 쓴다
import uvicorn                  # FastAPI 앱을 실제로 띄우는 웹 서버
from fastapi import FastAPI                                          # 웹 앱 뼈대
from fastapi.responses import HTMLResponse, StreamingResponse        # HTML 화면 / 끊임없이 보내는 응답(영상)
from fastapi.staticfiles import StaticFiles                          # 폴더 안 파일(크롭 사진)을 그대로 웹에 공개
from ultralytics import YOLO    # 물체 검출기 YOLO
from eg2 import load            # 2단계에서 만든 공통 모델 로드 함수 (같은 폴더의 eg2.py)

# ---------------- 실행 모드 (명령줄 옵션) ----------------
ap = argparse.ArgumentParser()                       # 옵션을 읽어 줄 객체
# choices=[...] : 이 셋 말고 다른 값을 쓰면 에러를 내 준다. default : 옵션을 안 쓰면 이 값
ap.add_argument("--mode", choices=["yolo", "relabel", "yoloe"], default="relabel")
# type=float : 글자로 들어온 "0.35"를 숫자 0.35로 바꾼다. None : "따로 안 정함" → 아래에서 모드별 기본값 사용
ap.add_argument("--conf", type=float, default=None, help="검출 신뢰도 (기본: 모드별 값)")
ap.add_argument("--port", type=int, default=8000, help="웹 포트. 다른 학생이 8000을 쓰고 있으면 8001 등으로")
ap.add_argument("--cam", type=int, default=0, help="웹캠 번호 (/dev/video0 → 0)")
args = ap.parse_args()                               # 실제 명령줄을 읽어 args.mode, args.conf ... 로 담는다
MODE = args.mode

# ---------------- 설정 ----------------
CAM, W, H = args.cam, 1280, 720       # 여러 변수에 한 번에 값 넣기. 웹캠 번호, 가로, 세로 해상도
# {"yolo": 0.5, ...}[MODE] : 사전에서 MODE 키의 값을 바로 꺼낸다.  A if 조건 else B : 조건이 참이면 A
CONF = args.conf if args.conf is not None else {"yolo": 0.5, "relabel": 0.35, "yoloe": 0.25}[MODE]
MIN_SIDE      = 48                    # 크롭 최소 크기(px). 이보다 작은 박스는 너무 흐려서 저장 안 함
REFRESH_S     = 30                    # 같은 추적 ID는 30초에 한 번만 다시 임베딩
SKIP_CLASSES  = {"person"}            # {..} 집합(set). 사람은 개인정보라 크롭·저장하지 않는다 (영상 박스로만 표시)
DIM, MAX_ITEMS = 512, 1000            # 벡터를 512차원으로 잘라 쓰고(Matryoshka), 최대 1000개만 메모리에 보관
CROP_DIR      = f"crops/{MODE}"       # f-문자열: {MODE} 자리에 값이 들어간다 → crops/yolo 등. 모드별로 따로 저장

# YOLO(COCO 80종)에 없는 물체는 비슷한 클래스로 오인식된다 (세탁기→oven, 전지 포장→cell phone).
# relabel: 크롭 임베딩과 아래 라벨 문구의 유사도로 라벨을 다시 붙인다.
# yoloe  : 아래 영문 라벨을 검출 클래스 자체로 사용한다.  (영문 = 영상 표시, 한글 = 웹 표시)
# ( , ) 는 튜플: 값 두 개를 묶은 것. 리스트 안에 (영문, 한글) 튜플을 여러 개 넣었다.
LABELS = [
    ("drum washing machine", "드럼 세탁기"), ("top-load washing machine", "통돌이 세탁기"),
    ("oven", "오븐"), ("microwave", "전자레인지"), ("refrigerator", "냉장고"),
    ("cell phone", "휴대폰"), ("battery pack", "원통형 전지 포장"), ("AA battery", "건전지"),
    ("laptop", "노트북"), ("monitor", "모니터"), ("keyboard", "키보드"), ("mouse", "마우스"),
    ("cup", "컵"), ("water bottle", "물병"), ("book", "책"), ("backpack", "가방"), ("chair", "의자"),
]
LABEL_MIN = 0.20                      # 재라벨 점수가 이보다 낮으면 YOLO 라벨 유지 (웹 카드의 점수를 보고 조정)
os.makedirs(CROP_DIR, exist_ok=True)  # 폴더 만들기. exist_ok=True : 이미 있어도 에러 내지 않음

# ---------------- 모델 불러오기 ----------------
if MODE == "yoloe":
    from ultralytics import YOLOE                 # yoloe 모드에서만 필요하므로 여기서 불러온다
    names = [en for en, _ in LABELS]              # 리스트 컴프리헨션: 튜플을 (en, _)로 풀어 영문만 모은다. _ 는 "안 쓰는 값"
    det = YOLOE("yoloe-11s-seg.pt")               # 첫 실행 시 가중치 자동 다운로드 (현재 폴더에 저장)
    det.set_classes(names, det.get_text_pe(names))  # 영문 이름을 글 임베딩으로 바꿔 검출 클래스로 지정
else:
    det = YOLO("yolo11n.pt")                      # COCO 80종 YOLO (n = 가장 작은 모델). 첫 실행 시 자동 다운로드
KO = dict(LABELS)                                 # (영문, 한글) 튜플 리스트 → {영문: 한글} 사전
print(f"[mode={MODE}] conf={CONF}")
emb = load(vision=True)                           # EmbeddingGemma 2 텍스트+비전 (음성 인코더는 안 올림)

# 라벨 문구를 미리 벡터로 바꿔 둔다 (relabel 모드에서 크롭 벡터와 비교, 다른 모드는 점수 표시용)
with torch.inference_mode():                      # with 블록 안에서는 학습용 기울기 계산을 꺼서 빠르고 메모리 절약
    label_vecs = np.asarray(                      # np.asarray : 결과를 numpy 배열로
        emb.encode([f"a photo of a {en}" for en, _ in LABELS],   # "a photo of a cup" 같은 문장 17개
                   prompt_name="SearchQuery",     # 검색 질의용 프롬프트 (3·4단계와 같음)
                   truncate_dim=DIM, normalize_embeddings=True),  # 512차원으로 자르고 길이 1로 정규화
        dtype=np.float32)                         # 결과 모양: (17, 512)

# ---------------- 스레드끼리 함께 쓰는 값 ----------------
# Lock(자물쇠): 두 스레드가 같은 값을 동시에 고치지 않도록 한 번에 한 스레드만 들어가게 한다.
frame_lock, emb_lock, idx_lock = threading.Lock(), threading.Lock(), threading.Lock()
latest_jpeg = None                         # 웹으로 보낼 가장 최근 프레임(JPEG 바이트)
crop_q = collections.deque(maxlen=64)      # 임베딩 대기 크롭 큐. 64개가 넘으면 가장 오래된 것이 자동으로 버려진다
items = []          # 저장된 물체 목록. 각 항목은 {id, cls, label, lscore, conf, ts, img, vec} 사전
seen = {}           # 추적 ID -> 마지막으로 임베딩한 시각
track_label = {}    # 추적 ID -> (영문 라벨, 한글 라벨, 점수). 영상 박스에 재라벨 결과를 쓰기 위해
stats = {"fps": 0.0, "embed_ms": 0.0}      # 웹 화면에 보여 줄 속도
stop = threading.Event()   # 종료 신호. stop.set() 하면 모든 반복문이 빠져나온다 (v5)


def capture_loop():
    """스레드 1: 웹캠 → YOLO 추적 → 새 물체는 크롭해서 큐에 → 박스 그린 영상을 JPEG로."""
    global latest_jpeg                     # global : 함수 안에서 바깥(모듈) 변수 latest_jpeg 에 새 값을 넣겠다는 선언
    cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)                      # 리눅스 V4L2 방식으로 웹캠 열기
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG로 받아야 1280x720에서 fps가 나온다
    # *"MJPG" : 문자열을 "M","J","P","G" 네 글자로 풀어 함수에 넘긴다
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    assert cap.isOpened(), "웹캠을 열 수 없음 (README Q11 참고)"  # 조건이 거짓이면 이 메시지로 멈춘다
    t0, n = time.time(), 0                 # fps 계산용: 시작 시각, 처리한 프레임 수
    while not stop.is_set():               # 종료 신호가 올 때까지 계속 반복
        ok, frame = cap.read()             # 프레임 한 장 읽기. ok=성공 여부, frame=(높이, 너비, 3) BGR 배열
        if not ok:
            time.sleep(0.1); continue      # 실패하면 0.1초 쉬고 다음 반복으로 (continue)
        # track : 검출 + 같은 물체에 같은 ID를 이어 붙이기. persist=True : 이전 프레임의 추적 정보를 이어 씀
        # quantize=16 : 16비트(FP16)로 계산해 빠르게 (예전 이름 half=True). [0] : 결과 리스트에서 첫(유일한) 프레임 결과
        r = det.track(frame, persist=True, conf=CONF, quantize=16, verbose=False)[0]
        now = time.time()
        if r.boxes.id is not None:         # 추적 ID가 붙은 박스가 있을 때만
            # zip(A, B, C, D) : 네 리스트를 한 칸씩 짝지어 돈다 → 박스 좌표, ID, 클래스 번호, 신뢰도
            for (x1, y1, x2, y2), tid, c, cf in zip(r.boxes.xyxy.int().tolist(),
                    r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                name = det.names[c]        # 클래스 번호 → 이름 (예: 41 → "cup")
                if name in SKIP_CLASSES or min(x2 - x1, y2 - y1) < MIN_SIDE:
                    continue               # 사람이거나 너무 작은 박스는 건너뜀
                if now - seen.get(tid, 0) < REFRESH_S:
                    continue               # 30초 안에 이미 저장한 ID면 건너뜀. get(키, 0) : 없으면 0
                seen[tid] = now
                # frame[y1:y2, x1:x2] : 배열 자르기(슬라이싱)로 박스 부분만. .copy() : 원본과 분리된 복사본
                crop_q.append((tid, name, cf, now, frame[y1:y2, x1:x2].copy()))
        vis = frame.copy()                 # 박스를 그릴 화면용 복사본
        # 추적 ID가 없으면 박스 개수만큼 None으로 채운다. [None] * 3 → [None, None, None]
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for (x1, y1, x2, y2), tid, c, cf in zip(r.boxes.xyxy.int().tolist(), ids,
                r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            yolo = det.names[c]
            # relabel 모드이고 재라벨 결과가 있으면 그 라벨, 아니면 검출기 라벨
            en = track_label[tid][0] if (MODE == "relabel" and tid in track_label) else yolo
            color = (0, 200, 255) if en != yolo else (0, 255, 0)   # BGR 색. 재라벨된 박스는 주황, 아니면 초록
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)       # 두께 2의 사각형
            cv2.putText(vis, f"{en} {cf:.2f}", (x1, max(y1 - 6, 14)),   # {cf:.2f} : 소수 둘째 자리까지
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)        # (OpenCV 기본 글꼴은 한글 불가 → 영문 표시)
        n += 1
        if now - t0 >= 1:                  # 1초마다 fps 갱신
            stats["fps"], t0, n = n / (now - t0), now, 0
        cv2.putText(vis, f"{stats['fps']:.1f} fps | indexed {len(items)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])   # 화면을 JPEG(품질 75)로 압축
        if ok:
            with frame_lock:               # 웹 스트림이 읽는 중에 바꾸지 않도록 자물쇠
                latest_jpeg = buf.tobytes()
    cap.release()                          # 반복이 끝나면 웹캠을 정상적으로 닫는다 (다음 실행이 바로 열 수 있게)


def embed_loop():
    """스레드 2: 큐의 크롭을 최대 8개씩 묶어 임베딩하고 인덱스(items)에 저장."""
    seq = 0                                # 파일 이름 번호
    while not stop.is_set():
        if not crop_q:                     # 큐가 비었으면 (빈 deque는 거짓)
            time.sleep(0.05); continue
        batch = [crop_q.popleft() for _ in range(min(8, len(crop_q)))]   # 앞에서부터 최대 8개 꺼내기
        paths = []
        for tid, name, cf, ts, crop in batch:     # 튜플 5개 값을 각각 이름에 풀어 받기
            seq += 1
            p = f"{CROP_DIR}/{seq:06d}_{name}.jpg"  # {seq:06d} : 6자리, 빈 자리는 0 → 000001
            cv2.imwrite(p, crop)                  # 크롭을 파일로 저장 (웹 카드에 보여 주고, 임베딩 입력으로도 씀)
            paths.append(p)
        t = time.perf_counter()                   # 정밀 타이머 시작
        with emb_lock:                            # 검색(/search)과 동시에 모델을 쓰지 않도록
            # {"image": 경로} 형태로 넣어야 사진으로 인식한다 (4단계 test_vision.py와 같은 방식)
            vecs = emb.encode([{"image": p} for p in paths],
                              truncate_dim=DIM, normalize_embeddings=True)
        stats["embed_ms"] = (time.perf_counter() - t) * 1000 / len(paths)   # 크롭 1개당 ms
        vecs = np.asarray(vecs, dtype=np.float32)  # 모양: (크롭 수, 512)
        sims = vecs @ label_vecs.T                 # @ : 행렬 곱. 정규화했으므로 = 코사인 유사도. 모양 (크롭 수, 17)
        with idx_lock:
            for (tid, name, cf, ts, _), p, v, sim in zip(batch, paths, vecs, sims):
                j = int(sim.argmax())              # 가장 비슷한 라벨의 번호
                if MODE != "relabel":              # yolo / yoloe: 검출기 라벨 그대로 (점수는 참고용)
                    en, ko, ls = name, KO.get(name, name), float(sim[j])
                elif sim[j] >= LABEL_MIN:          # relabel: 점수가 기준 이상이면 라벨 교체
                    en, ko, ls = LABELS[j][0], LABELS[j][1], float(sim[j])
                else:                              # 기준 미만이면 YOLO 라벨 유지
                    en, ko, ls = name, name, float(sim[j])
                track_label[tid] = (en, ko, ls)    # 영상 박스 표시용
                # 결과를 crops/<mode>/labels.csv 에 한 줄씩 덧붙인다 ("a" = append, 끝에 추가).
                # 앱을 꺼도 파일 이름(YOLO 라벨)과 재라벨 결과를 나중에 짝지어 볼 수 있게 하려고.
                with open(f"{CROP_DIR}/labels.csv", "a", encoding="utf-8") as fp:
                    fp.write(f"{os.path.basename(p)},{name},{cf:.2f},{ko},{ls:.3f},"
                             f"{time.strftime('%H:%M:%S', time.localtime(ts))}\n")
                items.append({"id": tid, "cls": name, "label": ko, "lscore": round(ls, 3),
                              "conf": round(cf, 2), "ts": ts, "img": "/" + p, "vec": v})
            while len(items) > MAX_ITEMS:          # 1000개를 넘으면 가장 오래된 것부터 삭제
                old = items.pop(0)                 # pop(0) : 맨 앞 항목을 꺼내며 리스트에서 지움
                try: os.remove(old["img"][1:])     # [1:] : 맨 앞 "/"를 떼어 실제 파일 경로로
                except OSError: pass               # 파일이 이미 없어도 무시


# ---------------- 웹 서버 ----------------
app = FastAPI()
app.mount("/crops", StaticFiles(directory="crops"), name="crops")   # http://.../crops/<mode>/<파일> 로 크롭 사진 제공


def public(it, score=None):
    """items 항목에서 벡터(vec)를 뺀, 웹으로 보낼 값만 고른다."""
    d = {k: it[k] for k in ("id", "cls", "label", "lscore", "conf", "ts", "img")}   # 사전 컴프리헨션
    if score is not None: d["score"] = round(float(score), 3)
    return d


@app.get("/stream")                 # @ 데코레이터: 아래 함수를 "GET /stream 요청이 오면 실행"으로 등록
def stream():
    def gen():                      # 함수 안의 함수. yield 가 있으면 값을 하나씩 계속 내보내는 '제너레이터'
        while not stop.is_set():    # 종료 신호가 오면 영상 전송도 끝낸다
            with frame_lock:
                f = latest_jpeg
            if f:
                # MJPEG: "--frame" 경계로 JPEG를 계속 이어 보내면 브라우저가 영상처럼 갈아 끼운다
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"   # b"..." : 바이트 문자열
            time.sleep(0.05)        # 최대 약 20 fps로 전송
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/search")
def search(q: str, k: int = 12):    # /search?q=세탁기&k=12 의 q, k 가 인자로 들어온다 (q: str 은 타입 표시)
    with idx_lock:
        snap = list(items)          # 계산 중에 목록이 바뀌지 않도록 복사본(스냅샷)
    if not snap:
        return []
    with emb_lock:
        qv = emb.encode(q, prompt_name="SearchQuery", truncate_dim=DIM, normalize_embeddings=True)
    s = np.stack([it["vec"] for it in snap]) @ np.asarray(qv, dtype=np.float32)   # 저장 벡터 전체와 질의의 유사도
    # argsort() : 작은 값부터의 순서(번호). [::-1] : 뒤집어 큰 값부터. [:k] : 앞에서 k개
    return [public(snap[i], s[i]) for i in s.argsort()[::-1][:k]]


@app.get("/recent")
def recent(n: int = 12):
    with idx_lock:
        snap = items[-n:][::-1]     # [-n:] : 끝에서 n개 → 뒤집어서 최신이 앞으로
    return {"mode": MODE, "stats": stats, "total": len(items), "items": [public(it) for it in snap]}


# 브라우저 화면 (HTML + 3초마다 /recent 를 불러 갱신하는 자바스크립트). """...""" 는 여러 줄 문자열
PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Jetson 물체 검색</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
img.live{width:100%;max-width:960px;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px}
.card{background:#222;padding:6px;border-radius:6px;font-size:12px}
.card img{width:100%;height:110px;object-fit:contain;background:#000}
input{font-size:16px;padding:6px;width:60%}button{font-size:16px;padding:6px 12px}</style>
<h2>실시간 영상 <span id=mode></span></h2><img class=live src="/stream">
<h2>텍스트로 물체 검색</h2>
<form onsubmit="go();return false"><input id=q placeholder="예: 빨간 컵, 열려 있는 노트북"><button>검색</button></form>
<div id=res class=grid></div>
<h2>최근 추출 <span id=st></span></h2><div id=rec class=grid></div>
<script>
const card=it=>`<div class=card><img src="${it.img}"><div><b>${it.label}</b> (${it.lscore})<br>YOLO: ${it.cls} ${it.conf}`+
 (it.score!==undefined?`<br>검색 점수 <b>${it.score}</b>`:'')+`<br>${new Date(it.ts*1000).toLocaleTimeString()}</div></div>`;
async function go(){const q=document.getElementById('q').value;if(!q)return;
 const r=await (await fetch('/search?q='+encodeURIComponent(q))).json();
 document.getElementById('res').innerHTML=r.map(card).join('')||'결과 없음';}
async function poll(){const r=await (await fetch('/recent')).json();
 document.getElementById('mode').textContent=`[mode: ${r.mode}]`;
 document.getElementById('st').textContent=`(총 ${r.total}개, 검출 ${r.stats.fps.toFixed(1)} fps, 임베딩 ${r.stats.embed_ms.toFixed(0)} ms/개)`;
 document.getElementById('rec').innerHTML=r.items.map(card).join('');}
setInterval(poll,3000);poll();
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


# 이 파일을 직접 실행했을 때만 아래가 돈다 (다른 파일에서 import 하면 실행 안 됨)
if __name__ == "__main__":
    # daemon=True : 메인(웹 서버)이 Ctrl+C로 끝나면 이 스레드들도 같이 끝난다
    # 스레드를 변수에 담아 두고 시작한다. 나중에 join()으로 끝날 때까지 기다리기 위해서
    workers = [threading.Thread(target=capture_loop, daemon=True),
               threading.Thread(target=embed_loop, daemon=True)]
    for w in workers:
        w.start()
    # host="0.0.0.0" : Jetson 자신뿐 아니라 맥북 등 다른 컴퓨터에서도 접속 가능하게
    # timeout_graceful_shutdown=1 : Ctrl+C 후 열린 연결이 닫히길 최대 1초만 기다린다.
    #   /stream 영상 연결은 브라우저를 닫기 전까지 끝나지 않아, 이 값이 없으면 Ctrl+C를 두 번 눌러야 꺼진다.
    # try ... finally : try 블록이 어떻게 끝나든(정상 종료든 Ctrl+C 예외든) finally 블록은 꼭 실행된다.
    #   uvicorn은 Ctrl+C로 서버를 닫은 뒤 KeyboardInterrupt 예외를 다시 던지므로, 정리 코드를 finally에 둔다.
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, timeout_graceful_shutdown=1)
    except KeyboardInterrupt:
        pass                        # Ctrl+C는 정상 종료 방법이니 오류로 취급하지 않는다
    finally:
        stop.set()                  # 두 스레드에 "그만" 신호
        for w in workers:
            w.join(timeout=5)       # 각 스레드가 하던 프레임·임베딩을 마치고 끝날 때까지 최대 5초 기다림
        print("종료 완료")
