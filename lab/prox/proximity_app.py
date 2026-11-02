# proximity_app.py - 10단계: USB 카메라 2대로 강의실 특정 물체(타겟)에 사람이 접근하는지 감지·알람 (README 10-3~10-8)
# 목적: 두 카메라 영상에서 사람 관절(YOLO11n-pose)을 찾고, 어깨 중심을 삼각측량해 바닥 위 (x, y)와 높이를 구한 뒤
#       타겟까지의 거리 변화로 '접근 시작 / 근접 알람 / 이탈 알람'을 낸다. 결과는 웹 화면(영상 2개, 평면도, 사람 표, 기록).
#       사람 이미지는 저장하지 않는다. 남는 것은 바닥 좌표와 알람 기록뿐이다.
# 준비: calib/cam0.npz, calib/cam1.npz (10-2 결과), calib/world_points.json, 같은 폴더의 stereo.py
# 실행: ~/eg2-lab/prox 에서 (→ 맥북 브라우저에서 http://<JETSON_IP>:8000)
#   python proximity_app.py --cams 0 2 --mode ankle1 --target-xy 3.0,1.0     10-3 카메라 1대 + 발목 (문제 확인)
#   python proximity_app.py --cams 0 2 --mode stereo --target-xy 3.0,1.0     10-4 카메라 2대 + 어깨 삼각측량 (개선)
#   python proximity_app.py --cams 0 2 --target-xy 3.0,1.0                   10-5 fused: + 카메라 1대 추정 (보완)
#   python proximity_app.py --cams 0 2 --target "red fire extinguisher"      10-6 타겟을 글로 찾기 + 접근 알람
#   python proximity_app.py --cams 0 2 --target "cart" --moving              10-8 이동 타겟 (칼만 필터 + TTC)
#   --cams 순서는 calib_extrinsics.py 때와 같아야 한다 (첫째 = cam0, 둘째 = cam1). IP 카메라는 "rtsp://..." 주소로 (v2)
#   예: python proximity_app.py --cams 0 "rtsp://..." --lag 0 300 --target-xy 3.0,1.0
#
# 수정 기록
# 2026-10-27  v1  README 10단계 proximity_app.py를 옮기고 줄별 설명 주석 추가 (지원 님 요청: 파이썬 복습용).
#                 README와 다른 점 (1~3은 8·9단계 app.py/pose_app.py에서 겪은 문제를 미리 막기 위함):
#                 1) det.predict()의 half=True → quantize=16. ultralytics 8.4에서 half가 지원 중단 예정이라 경고가 매 프레임 찍힘
#                 2) uvicorn.run()에 timeout_graceful_shutdown=1. 열린 /stream 연결 때문에 Ctrl+C로 안 꺼지던 문제
#                 3) 종료 신호(stop) + try/finally + join, 카메라 스레드도 stop을 보고 끝나며 카메라를 닫음.
#                    Ctrl+C 때 'Aborted (core dumped)'가 나던 문제
#                 4) 화면 상단에 이 프로세스의 RAM(MB) 표시. 10-7 표에 RAM 칸이 있는데 README 코드에는 측정이 없어서
#                    (/proc/self/status 의 VmRSS = 이 프로그램이 실제로 차지한 메모리)
#                 5) 모델 파일(yolo11n-pose.pt, yoloe-11s-seg.pt)이 상위 폴더 ~/eg2-lab 에 이미 있으면 그것을 쓴다.
#                    prox 폴더에서 실행하면 ultralytics가 같은 파일을 또 내려받기 때문
#                 6) EmbeddingGemma 이미지 임베딩을 encode([{"image": 경로}])[0] (목록 형태)로 호출. 8단계 app.py에서
#                    확인한 방식과 맞춤 (README는 사전 하나를 바로 넘김)
# 2026-10-27  v2  --cams에 IP 카메라 주소(rtsp://...)도 받도록 변경 (stereo.open_source/read_frame). 지원 님 장비가
#                 USB 웹캠 1대 + IP 카메라 1대라서. IP 카메라의 큰 화면은 10-1 보정 때와 똑같이 640x480으로 줄인다.
#                 --lag 옵션 추가: 카메라별 영상 지연(ms)을 받은 시각에서 빼서 '찍힌 시각'에 가깝게 맞춘다.
#                 IP 카메라는 압축·네트워크 때문에 USB보다 수백 ms 늦게 도착하는데, 받은 시각만 비교하면
#                 화면의 '카메라 시간 차'가 실제보다 작게 보이고 걷는 사람의 삼각측량이 어긋나기 때문
# 2026-10-28  v3  world_points.json에 "room"(사람 위치로 인정하는 바닥 범위)이 있으면 그 값을 쓴다. 지원 님 기준점이 책상 근처
#                 좁은 영역에만 있어 기본 범위(기준점+0.5m)로는 통로에 선 사람이 버려지기 때문
# 2026-11-02  제출용 사본: 주석과 안내 출력 문구의 Jetson 접속 주소를 <JETSON_IP>, 계정명을 <USER>로 바꿈 (공개 리포에 개인 접속 정보를 올리지 않으려고). 코드 동작은 같음

import argparse, time, threading, collections, os, json   # 옵션 / 시각 / 스레드 / deque / 파일 존재 확인(v1 수정 5) / (v3) json
import cv2, numpy as np, uvicorn                     # 카메라·그리기 / 계산 / 웹 서버
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
import stereo as S                                   # 같은 폴더의 stereo.py를 S라는 짧은 이름으로 (S.triangulate 처럼 씀)

# ---------------- 실행 옵션 ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--cams", nargs=2, default=["0", "1"], help="calib_extrinsics.py와 같은 순서. USB 번호 또는 rtsp 주소 (v2)")
ap.add_argument("--lag", type=float, nargs=2, default=[0.0, 0.0], help="(v2) 카메라별 영상 지연(ms). 예: --lag 0 300")
ap.add_argument("--mode", choices=["ankle1", "stereo", "fused"], default="fused")   # choices: 이 중 하나만 허용
ap.add_argument("--target", default=None, help='예: "red fire extinguisher"')     # 타겟을 글로 지정 (10-6)
ap.add_argument("--target-xy", default=None, help="예: 3.0,1.2 (직접 측정한 바닥 좌표)")   # 줄자로 잰 타겟 위치 (10-3~10-5)
ap.add_argument("--target-h", type=float, default=0.8, help="한 카메라만 타겟을 볼 때 가정할 높이(m)")
ap.add_argument("--near", type=float, default=1.0, help="근접 알람 거리(m)")
ap.add_argument("--far", type=float, default=3.0, help="이탈 알람 거리(m)")
ap.add_argument("--moving", action="store_true", help="이동 타겟 모드 (10-8)")
ap.add_argument("--ttc", type=float, default=2.0, help="충돌 위험 알람 TTC 기준(초)")
ap.add_argument("--target-every", type=int, default=2, help="이동 타겟 검출 주기(프레임)")
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8000, help="다른 학생이 8000을 쓰고 있으면 8001 등으로")
args = ap.parse_args()                               # 옵션 이름의 - 는 _ 로 바뀐다: --target-xy → args.target_xy
W, H = map(int, args.res.split("x"))
if args.moving and not args.target:
    ap.error("--moving 에는 --target \"문구\"가 필요합니다 (매 프레임 YOLOE로 타겟을 찾기 때문)")

# ---------------- 설정값 ----------------
MAX_SKEW = 0.08       # 두 카메라 프레임 시간 차 허용(초). 넘으면 그 순간은 삼각측량 생략 (사람이 움직이면 위치가 어긋나므로)
H_GUESS = 1.2         # 한 카메라 추정 시 어깨 높이 가정(m): 앉음 약 1.0, 섬 약 1.4의 중간
GATE = 0.8            # 같은 사람으로 이어 붙일 최대 이동(m/프레임)
SMOOTH_S = 0.5        # 위치 중앙값 창(초): 최근 0.5초 위치의 중앙값으로 흔들림을 줄인다
SIT_H = 1.15          # 어깨 높이가 이 값 미만이면 '앉음'
NUM_CAMS = 1 if args.mode == "ankle1" else 2         # ankle1 모드는 카메라 1대만 쓴다
CLOSE_V = 0.5         # 충돌 위험 알람: 접근 속도(m/s) 기준
TGT_LOST_S = 2.0      # 이 시간 이상 타겟을 못 보면 EmbeddingGemma로 다시 식별

cams = [S.load_cam(f"calib/cam{n}.npz") for n in range(NUM_CAMS)]   # 10-2 결과 읽기 (cam0, cam1)
_W = json.load(open("calib/world_points.json", encoding="utf-8"))   # (v3) "room" 키를 읽기 위해 파일 전체를 읽음
ROOM = _W.get("room") or S.room_bounds(_W["points"])  # 강의실 범위 (이 밖의 위치는 버림). (v3) "room"이 있으면 그 값


def model_path(name):
    """(v1 수정 5) ~/eg2-lab 에 이미 받은 모델 파일이 있으면 그 경로, 없으면 이름만 (ultralytics가 자동 다운로드)"""
    up = os.path.join("..", name)
    return up if os.path.exists(up) else name


det = YOLO(model_path("yolo11n-pose.pt"))            # 사람 관절 17개 추정 모델 (9단계와 같은 것)
stop = threading.Event()                             # (v1 수정 3) 종료 신호. stop.set() 하면 모든 반복문이 빠져나온다


# ---------------- 카메라 수신 (최신 프레임만 보관) ----------------
class Cam(threading.Thread):
    """카메라 1대를 계속 읽어 '가장 최근 프레임과 그 시각'만 보관하는 스레드.
    두 카메라를 따로 읽으므로, 처리 쪽은 두 최신 프레임의 시각 차(skew)를 보고 삼각측량 여부를 정한다"""
    def __init__(self, dev, lag_ms=0.0):
        super().__init__(daemon=True)                # daemon=True: 프로그램이 끝나면 이 스레드도 같이 끝남
        # (v2) USB면 MJPG 640x480 + 대기열 1장, IP면 FFmpeg 저지연 수신 (stereo.open_source). 못 열면 README Q11, Q14 참고
        self.cap = S.open_source(dev, W, H)
        self.lag = lag_ms / 1000                     # (v2) ms → 초
        self.lock, self.frame, self.ts = threading.Lock(), None, 0.0   # Lock: 두 스레드가 동시에 frame을 만지지 않게

    def run(self):                                   # start() 하면 별도 스레드에서 이 함수가 돈다
        while not stop.is_set():                     # (v1 수정 3) 종료 신호가 올 때까지
            f = S.read_frame(self.cap, W, H)         # (v2) 읽기 + 640x480으로 맞추기
            if f is not None:
                with self.lock:                      # with lock: 이 블록 동안 다른 스레드는 기다린다
                    self.frame, self.ts = f, time.time() - self.lag   # (v2) 받은 시각 - 지연 = 찍힌 시각(추정)
        self.cap.release()                           # (v1 수정 3) 끝날 때 카메라 닫기

    def get(self):
        """(시각, 프레임 복사본). 아직 프레임이 없으면 (0.0, None)"""
        with self.lock:
            return self.ts, (None if self.frame is None else self.frame.copy())


# ---------------- 접근·이탈 알람 ----------------
class ProximityAlarm:
    """사람 한 명마다 하나씩. 히스테리시스로 경계에서 깜빡이지 않게 하고, 최근 1초 거리 변화 속도로 접근/이탈 판단.
    구역(zone): near(near 안) / mid(사이) / far(far 밖).
    히스테리시스: near로 들어갈 때는 1.0m, 나올 때는 1.0+0.2m를 넘어야 나옴 → 경계에 서 있어도 알람이 반복되지 않음"""
    def __init__(self, near, far, margin=0.2, v_th=0.3):
        self.near, self.far, self.m, self.v_th = near, far, margin, v_th   # v_th: 이 속도(m/s) 넘게 변해야 접근/멀어짐
        self.zone, self.hist, self.trend, self.ttc_on = "mid", [], "정지", False

    def update_ttc(self, closing, ttc, d):
        """이동 타겟(10-8): 빠르게 가까워지고 TTC가 짧으면 충돌 위험 알람 (조건이 풀리면 다시 무장)"""
        danger = closing > CLOSE_V and ttc is not None and ttc < args.ttc
        ev = []
        if danger and not self.ttc_on:               # 위험 상태로 '처음' 들어갈 때만 알람 1회
            ev.append(f"알람: 충돌 위험 TTC {ttc:.1f}초 (접근 {closing:.1f}m/s, {d:.1f}m)")
        if danger:
            self.ttc_on = True
        elif self.ttc_on and (closing < 0.2 or ttc is None or ttc > args.ttc * 1.5):   # 충분히 풀려야 해제 (히스테리시스)
            self.ttc_on = False
        return ev

    def update(self, t, d):
        """시각 t에 타겟까지 거리 d(m)를 넣으면, 새로 생긴 알람 문장 목록을 돌려준다"""
        self.hist = [(tt, dd) for tt, dd in self.hist if t - tt < 1.0] + [(t, d)]   # 최근 1초 기록만 유지
        span = self.hist[-1][0] - self.hist[0][0]                                   # 기록의 시간 폭
        v = (self.hist[-1][1] - self.hist[0][1]) / span if span > 0.3 else 0.0      # 거리 변화 속도(m/s). -면 가까워짐
        trend = "접근 중" if v < -self.v_th else "멀어지는 중" if v > self.v_th else "정지"
        ev = []
        if trend == "접근 중" and self.trend != "접근 중" and d < self.far:          # far 안에서 접근이 '시작'될 때
            ev.append(f"접근 시작 ({d:.1f}m)")
        self.trend = trend
        if self.zone != "near" and d < self.near:                                   # near 안으로 들어옴 → 근접 알람
            self.zone = "near"; ev.append(f"알람: 근접 {d:.1f}m")
        elif self.zone == "near" and d > self.near + self.m:                        # near+0.2m를 넘어야 해제
            self.zone = "mid"
        if self.zone != "far" and d > self.far:                                     # far 밖으로 나감
            if self.zone == "mid" and self.hist[0][1] <= self.far:   # 안쪽에 있다가 나간 경우만 알람 (처음부터 밖이면 X)
                ev.append(f"알람: 이탈 {d:.1f}m")
            self.zone = "far"
        elif self.zone == "far" and d < self.far - self.m:                          # far-0.2m 안으로 와야 mid로
            self.zone = "mid"
        return ev


# ---------------- 바닥 평면 추적 ----------------
class Track:
    """한 사람의 바닥 위치 기록. ID, 최근 위치·높이(최대 60개), 알람 상태, 칼만 필터를 가진다"""
    _next = 1                                        # 클래스 변수: 모든 Track이 공유하는 '다음 ID 번호'

    def __init__(self, xy, h, kind, now):
        self.id, Track._next = Track._next, Track._next + 1   # 내 ID를 받고 다음 번호를 1 올림
        self.hist = collections.deque([(now, xy, h)], maxlen=60)   # deque(maxlen=60): 60개가 넘으면 오래된 것부터 버림
        self.kind, self.last, self.first = kind, now, now    # kind: stereo(삼각측량) / single(1대 추정) / ankle
        self.alarm = ProximityAlarm(args.near, args.far)
        self.kf = S.Kalman2D(xy)
        self.rel = (None, 0.0, None)                 # (거리, 접근 속도, TTC)

    def observe(self, xy, h, kind, now):
        """새 측정을 이어 붙인다"""
        self.kf.predict(now - self.last); self.kf.update(xy)
        self.hist.append((now, xy, h)); self.kind, self.last = kind, now

    @property                                        # @property: tr.xy() 가 아니라 tr.xy 처럼 괄호 없이 쓰는 값
    def xy(self):
        if args.moving:                              # 이동 모드: 칼만 필터 위치 (속도 추정 포함)
            return self.kf.x[:2].copy()
        pts = [p for t, p, _ in self.hist if self.last - t <= SMOOTH_S]   # 최근 0.5초 위치들
        return np.median(pts, axis=0)                # 중앙값: 튀는 값 하나에 덜 흔들림

    @property
    def h(self):
        return float(np.median([hh for t, _, hh in self.hist if self.last - t <= SMOOTH_S]))


tracks, events = [], collections.deque(maxlen=60)    # 지금 추적 중인 사람들 / 최근 기록 60개
ev_lock = threading.Lock()                           # events는 처리 스레드와 웹 요청이 함께 쓰므로 잠금
state = {"target": None, "stats": {}, "people": []}  # 웹 화면에 보낼 현재 상태
tgt = {"kf": None, "h": args.target_h, "last": 0.0, "boxes": [None, None], "how": "",
       "trail": collections.deque(maxlen=50), "ref": None}   # 이동 타겟(10-8) 추적 상태
models = {}           # 이동 모드에서 계속 쓰는 yoloe / emb / qv
jpegs = [None, None]  # 카메라별 최신 결과 화면(JPEG). /stream 이 이것을 보낸다


def log(now, who, msg):
    """기록 표 맨 위에 한 줄 추가"""
    with ev_lock:
        events.appendleft({"ts": now, "who": who, "label": msg})


def rss_mb():
    """(v1 수정 4) 이 프로세스가 차지한 실제 메모리(MB). /proc/self/status 의 'VmRSS:  123456 kB' 줄을 읽는다"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024   # kB → MB
    except OSError:
        pass
    return None


def update_tracks(obs, now):
    """obs: [(xy, h, kind)] → 가장 가까운 트랙에 이어 붙이고(GATE 0.8m 이내), 나머지는 새 트랙(새 사람)"""
    free = list(range(len(tracks)))                  # 이번 프레임에 아직 측정을 받지 않은 트랙 번호
    for xy, h, kind in sorted(obs, key=lambda o: o[2] != "stereo"):     # 삼각측량 결과 우선 (False가 True보다 앞)
        best, bi = GATE, None
        for i in free:                               # 가장 가까운 트랙 찾기
            d = np.linalg.norm(tracks[i].xy - xy)
            if d < best:
                best, bi = d, i
        if bi is None:                               # 가까운 트랙이 없으면 새 사람
            tracks.append(Track(xy, h, kind, now)); log(now, f"#{tracks[-1].id}", "등장")
        else:
            tr = tracks[bi]; tr.observe(xy, h, kind, now)
            free.remove(bi)
    for tr in [t for t in tracks if now - t.last > 2.0]:   # 2초 넘게 안 보인 사람은 퇴장 처리
        tracks.remove(tr); log(now, f"#{tr.id}", f"퇴장 (머문 시간 {tr.last - tr.first:.0f}초)")


# ---------------- 타겟 찾기: YOLOE 후보 → EmbeddingGemma 2 재확인 → 삼각측량 ----------------
def load_target_models():
    """YOLOE(글로 지정한 물체 검출)와 EmbeddingGemma 2(이미지·글 비교)를 필요할 때만 불러온다"""
    import torch                                     # 함수 안 import: --target을 안 쓰면 아예 불러오지 않음
    from ultralytics import YOLOE
    from sentence_transformers import SentenceTransformer
    if not models:                                   # 아직 안 불렀으면
        yoloe = YOLOE(model_path("yoloe-11s-seg.pt"))
        yoloe.set_classes([args.target], yoloe.get_text_pe([args.target]))   # 검출할 클래스 = 타겟 문구 하나
        emb = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                  model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"audio_config": None})
        # 비교 기준이 될 글 벡터: "a photo of a red fire extinguisher"
        qv = np.asarray(emb.encode(f"a photo of a {args.target}", prompt_name="SearchQuery",
                                   normalize_embeddings=True), dtype=np.float32)
        models.update(yoloe=yoloe, emb=emb, qv=qv)  # update: 사전에 여러 키를 한 번에 넣기
    return models["yoloe"], models["emb"], models["qv"]


def embed_crop(emb, f, box, path):
    """박스 부분을 잘라 파일로 저장한 뒤 이미지 벡터로 바꾼다 (타겟 물체 크롭만 저장, 사람 사진 아님)"""
    x1, y1, x2, y2 = box
    cv2.imwrite(path, f[y1:y2, x1:x2])               # 이미지 자르기: [세로 범위, 가로 범위]
    # (v1 수정 6) 목록 형태로 넣고 [0]으로 첫 결과를 꺼낸다
    return np.asarray(emb.encode([{"image": path}], normalize_embeddings=True)[0], dtype=np.float32)


def find_target(frames):
    """고정 타겟 찾기(10-6): 카메라마다 YOLOE 후보 중 문구와 가장 닮은 박스 하나 → 두 카메라 박스 중심을 삼각측량"""
    import torch
    yoloe, emb, qv = load_target_models()
    found = []                                       # 카메라별 (점수, 박스 중심, 박스, 벡터) 또는 None
    for n, f in enumerate(frames):
        r = yoloe.predict(f, conf=0.1, verbose=False)[0]   # conf 0.1: 후보를 넉넉히 받고 EmbeddingGemma가 고름
        best = None
        for i, (x1, y1, x2, y2) in enumerate(r.boxes.xyxy.int().tolist()):
            if min(x2 - x1, y2 - y1) < 16:           # 16px보다 작은 박스는 버림 (임베딩이 의미 없음)
                continue
            v = embed_crop(emb, f, (x1, y1, x2, y2), f"calib/target_cand{n}_{i}.jpg")
            score = float(v @ qv)                    # 코사인 유사도 (둘 다 정규화했으므로 내적 = 유사도)
            if best is None or score > best[0]:
                best = (score, ((x1 + x2) / 2, (y1 + y2) / 2), (x1, y1, x2, y2), v)
        found.append(best)
    ok = [b is not None and b[0] >= 0.15 for b in found]   # 점수 0.15 이상이어야 '찾음'
    if any(ok):                                    # 이동 모드 재식별용 기준 임베딩 (점수가 높은 쪽 크롭)
        tgt["ref"] = max((b for b, o in zip(found, ok) if o), key=lambda b: b[0])[3]
    if not args.moving:
        models.clear()                             # 모델 참조를 지워서
        torch.cuda.empty_cache()                   # 고정 타겟은 한 번 찾고 GPU 메모리 반환
    if NUM_CAMS == 2 and all(ok):                  # 두 카메라 모두 찾음 → 삼각측량
        p0, p1 = S.undistort(cams[0], [found[0][1]])[0], S.undistort(cams[1], [found[1][1]])[0]
        X, err = S.triangulate(cams[0], cams[1], p0, p1)
        how = f"삼각측량 (재투영 {err:.1f}px)"
    elif any(ok):                                  # 한 카메라만 찾음 → 높이 --target-h 평면으로 추정
        n = ok.index(True)
        X = S.ray_plane(cams[n], S.undistort(cams[n], [found[n][1]])[0], args.target_h)
        how = f"cam{n} 단독 추정 (높이 {args.target_h}m 가정)"
    else:
        return None
    return {"xyz": X.round(2).tolist(), "how": how,
            "boxes": [b[2] if o else None for b, o in zip(found, ok)],
            "scores": [round(b[0], 3) if b else None for b in found]}


def set_target(frames):
    """타겟 위치 정하기: --target-xy(직접 지정) 또는 --target(글로 찾기)"""
    if args.target_xy:
        x, y = map(float, args.target_xy.split(","))        # "3.0,1.0" → 3.0, 1.0
        state["target"] = {"xyz": [x, y, 0.0], "how": "직접 지정", "boxes": [None, None], "scores": []}
    elif args.target:
        t = find_target(frames)
        state["target"] = t
        log(time.time(), "타겟", f"찾음: {t['xyz']} / {t['how']} / 점수 {t['scores']}" if t else "찾지 못함")
    if args.moving and state["target"]:            # 이동 모드: 찾은 위치로 타겟 칼만 필터 시작
        x, y, z = state["target"]["xyz"]
        tgt.update(kf=S.Kalman2D((x, y)), last=time.time(), boxes=state["target"]["boxes"],
                   how=state["target"]["how"])
        if z > 0.1:
            tgt["h"] = z
        tgt["trail"].clear()
    retarget.clear()                               # '다시 찾기' 요청 처리 완료


def track_target(frames, now, idx):
    """10-8: 이동 타겟을 매 N프레임 검출 → 예측 위치에 가까운 후보 선택(놓쳤으면 임베딩으로 재식별) → 삼각측량 → 칼만 갱신"""
    kf = tgt["kf"]
    if kf is None:
        return
    if idx % args.target_every == 0:               # % = 나머지. 2프레임마다 한 번만 검출 (속도 절약)
        yoloe, emb, qv = load_target_models()
        lost = now - tgt["last"] > TGT_LOST_S      # 2초 넘게 못 봤으면 '놓침'
        X_pred = [*kf.at(now - tgt["last"]), tgt["h"]]   # 예측 위치 (x, y, 높이). * = 목록 풀어 넣기
        chosen = [None, None]
        for n, f in enumerate(frames):
            r = yoloe.predict(f, conf=0.1, verbose=False)[0]
            boxes = [b for b in r.boxes.xyxy.int().tolist() if min(b[2] - b[0], b[3] - b[1]) >= 16]
            if not boxes:
                continue
            centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
            if not lost:                       # 추적 중: 예측 위치를 화면에 투영해 가장 가까운 후보 (100px 이내)
                pp = S.project_distorted(cams[n], X_pred)
                dists = [np.hypot(c[0] - pp[0], c[1] - pp[1]) for c in centers]   # hypot = √(a²+b²)
                i = int(np.argmin(dists))      # argmin: 가장 작은 값의 번호
                if dists[i] < 100:
                    chosen[n] = (centers[i], boxes[i])
            else:                              # 놓침: 기준 임베딩과 가장 닮은 후보 (큰 박스 5개까지)
                ref = tgt["ref"] if tgt["ref"] is not None else qv
                cand = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))[:5]
                sc = [(float(embed_crop(emb, f, boxes[i], f"calib/reid{n}_{i}.jpg") @ ref), i) for i in cand]
                s_best, i = max(sc)            # (점수, 번호) 중 점수가 가장 큰 것
                if s_best >= 0.15:
                    chosen[n] = (centers[i], boxes[i])
        X, how = None, ""
        if NUM_CAMS == 2 and all(chosen):      # 두 카메라 모두 → 삼각측량 (재투영 15px 미만일 때만 믿음)
            p0, p1 = S.undistort(cams[0], [chosen[0][0]])[0], S.undistort(cams[1], [chosen[1][0]])[0]
            Xt, err = S.triangulate(cams[0], cams[1], p0, p1)
            if err < 15:
                X, how = Xt, "삼각측량"
                tgt["h"] = 0.8 * tgt["h"] + 0.2 * float(Xt[2])     # 타겟 높이는 천천히 갱신
        if X is None and any(chosen):          # 한쪽만 → 마지막 타겟 높이 평면으로 추정
            n = 0 if chosen[0] else 1
            X = S.ray_plane(cams[n], S.undistort(cams[n], [chosen[n][0]])[0], tgt["h"])
            how = f"cam{n} 단독 추정"
        tgt["boxes"] = [c[1] if c else None for c in chosen]
        if X is not None:                      # 측정했으면 칼만 필터 갱신
            kf.predict(now - tgt["last"]); kf.update(X[:2])
            tgt["last"], tgt["how"] = now, how
            tgt["trail"].append([round(float(X[0]), 2), round(float(X[1]), 2)])   # 평면도의 보라색 이동 경로
            if lost:
                log(now, "타겟", "다시 찾음 (임베딩 재식별)")
    pos = kf.at(now - tgt["last"])             # 측정이 없던 프레임은 예측 위치
    state["target"] = {"xyz": [round(float(pos[0]), 2), round(float(pos[1]), 2), round(tgt["h"], 2)],
                       "v": [round(float(kf.x[2]), 2), round(float(kf.x[3]), 2)],
                       "how": tgt["how"] + (" (예측)" if now - tgt["last"] > 0.3 else ""),
                       "lost": now - tgt["last"] > TGT_LOST_S,
                       "boxes": tgt["boxes"], "trail": list(tgt["trail"]), "scores": []}


# ---------------- 메인 처리 ----------------
def draw(n, res):
    """카메라 n의 결과 화면: 관절 + 타겟 박스(보라) + 사람 ID 점(빨강 = 근접, 초록 = 그 외)"""
    vis = res.plot(labels=False, boxes=False)    # 관절 뼈대만 그림
    t = state["target"]
    box = t["boxes"][n] if t and n < len(t["boxes"]) else None
    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, "TARGET", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    for tr in tracks:
        if now_global - tr.last > 0.3:           # 0.3초 넘게 안 보인 사람은 그리지 않음
            continue
        # 바닥 위치(+높이)를 이 카메라 화면 좌표로 투영 → 계산된 위치가 실제 사람과 맞는지 눈으로 확인
        u, v = S.project_distorted(cams[n], [*tr.xy, tr.h if args.mode != "ankle1" else 0.0])
        if 0 <= u < W and 0 <= v < H:
            col = (0, 0, 255) if tr.alarm.zone == "near" else (0, 255, 0)
            cv2.circle(vis, (int(u), int(v)), 6, col, -1)
            cv2.putText(vis, f"#{tr.id}", (int(u) + 8, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(vis, f"cam{n} | {args.mode}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


retarget = threading.Event(); retarget.set()     # '타겟 찾기' 요청. 처음에 set → 첫 프레임에서 타겟을 찾는다
now_global = 0.0                                 # draw()가 쓰는 현재 시각


def process_loop():
    """처리 스레드: 프레임 2장 → 관절 → 상체 좌표 → 삼각측량/추정 → 추적 → 거리·알람 → 화면"""
    global now_global                            # global: 함수 밖 변수 now_global을 바꾸겠다는 선언
    srcs = [Cam(d, lag) for d, lag in zip(args.cams[:NUM_CAMS], args.lag)]   # ankle1이면 첫 카메라만
    for s in srcs:
        s.start()
    t0, n_frames, idx = time.time(), 0, 0
    while not stop.is_set():                     # (v1 수정 3) 종료 신호가 올 때까지
        got = [s.get() for s in srcs]            # [(시각, 프레임), ...]
        if any(f is None for _, f in got):       # 아직 프레임이 없는 카메라가 있으면 잠깐 기다림
            time.sleep(0.05); continue
        frames = [f for _, f in got]
        if retarget.is_set():
            set_target(frames)
        skew = abs(got[0][0] - got[-1][0])       # 두 프레임의 촬영 시각 차(초)
        # 두 프레임을 한 번에(배치) 관절 추정. quantize=16 : FP16 계산 (v1 수정 1)
        results = det.predict(frames, conf=0.5, quantize=16, verbose=False)
        # 사람마다 (17, 3) = 관절 (x, y, 신뢰도). 사람이 없으면 (0, 17, 3) 빈 배열
        kps = [np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], -1)
               if r.keypoints is not None and r.keypoints.conf is not None else np.zeros((0, 17, 3)) for r in results]
        now = now_global = time.time()
        idx += 1
        if args.moving:
            track_target(frames, now, idx)
        obs = []                                 # 이번 프레임의 사람 위치 관측 [(xy, 높이, 방식)]
        if args.mode == "ankle1":                # 10-3: 발목 → 바닥(z=0) 평면
            for k in kps[0]:
                a = S.ankle(k)
                X = S.ray_plane(cams[0], S.undistort(cams[0], [a])[0], 0.0) if a is not None else None
                if X is not None:
                    obs.append((X[:2], 0.0, "ankle"))
        else:                                    # 10-4/10-5: 상체(어깨 중심/코)
            pts = [[(i, S.upper_body(k)) for i, k in enumerate(kk)] for kk in kps]   # 카메라별 [(사람번호, 상체점)]
            pts = [[(i, p) for i, p in pp if p is not None] for pp in pts]           # 상체가 안 보이는 사람 제외
            und = [S.undistort(cams[n], [p for _, p in pts[n]]) for n in range(2)]   # 왜곡 보정
            matched, rest0, rest1 = ([], set(range(len(und[0]))), set(range(len(und[1]))))
            if skew <= MAX_SKEW:                 # 시간 차가 80ms 이하일 때만 두 카메라 짝짓기
                matched, rest0, rest1 = S.match_people(cams[0], cams[1], und[0], und[1])
            for i, j, X, err in matched:
                obs.append((X[:2], float(X[2]), "stereo"))
            if args.mode == "fused":             # 10-5: 짝을 못 찾은 사람은 어깨 높이 1.2m로 가정해 추정
                singles = []
                for n, rest in ((0, rest0), (1, rest1)):
                    for i in rest:
                        X = S.ray_plane(cams[n], und[n][i], H_GUESS)
                        if X is not None:
                            singles.append(X[:2])
                for xy in singles:   # 삼각측량 결과나 이미 추가한 추정과 겹치면(0.6m 이내) 같은 사람으로 보고 버림
                    if all(np.linalg.norm(xy - o[0]) > 0.6 for o in obs):
                        obs.append((xy, H_GUESS, "single"))
        x0, y0, x1, y1 = ROOM
        obs = [o for o in obs if x0 <= o[0][0] <= x1 and y0 <= o[0][1] <= y1]   # 강의실 밖 위치는 버림
        update_tracks(obs, now)

        people = []                              # 웹 표에 보낼 사람 목록
        txy = np.array(state["target"]["xyz"][:2]) if state["target"] else None
        for tr in tracks:
            d = float(np.linalg.norm(tr.xy - txy)) if txy is not None else None   # 타겟까지 바닥 거리(m)
            closing, ttc = 0.0, None
            if d is not None:
                msgs = tr.alarm.update(now, d)
                if args.moving and tgt["kf"] is not None and not state["target"].get("lost"):
                    t_state = np.concatenate([txy, tgt["kf"].x[2:]])                 # 타겟 (x, y, vx, vy)
                    p_state = np.concatenate([tr.kf.at(now - tr.last), tr.kf.x[2:]])  # 사람 (x, y, vx, vy)
                    d, closing, ttc = S.relative_state(p_state, t_state)
                    msgs += tr.alarm.update_ttc(closing, ttc, d)
                for msg in msgs:
                    log(now, f"#{tr.id}", msg)
            people.append({"id": tr.id, "x": round(float(tr.xy[0]), 2), "y": round(float(tr.xy[1]), 2),
                           "h": round(tr.h, 2), "kind": tr.kind,
                           "pose": "-" if tr.kind != "stereo" else ("앉음" if tr.h < SIT_H else "섬"),
                           "d": None if d is None else round(d, 2), "trend": tr.alarm.trend, "zone": tr.alarm.zone,
                           "v": [round(float(tr.kf.x[2]), 2), round(float(tr.kf.x[3]), 2)],
                           "closing": round(closing, 2), "ttc": None if ttc is None else round(ttc, 1),
                           "danger": tr.alarm.ttc_on})
        n_frames += 1
        if now - t0 >= 1:                        # 1초마다 fps·시간 차·RAM 갱신
            state["stats"] = {"fps": round(n_frames / (now - t0), 1), "skew_ms": round(skew * 1000),
                              "mode": args.mode + (" +moving" if args.moving else ""),
                              "ram_mb": rss_mb()}   # (v1 수정 4)
            t0, n_frames = now, 0
        state["people"] = people
        for n in range(NUM_CAMS):
            jpegs[n] = draw(n, results[n])
    for s in srcs:                               # (v1 수정 3) 카메라 스레드가 카메라를 닫을 때까지 잠깐 기다림
        s.join(timeout=2)


# ---------------- 웹 앱 ----------------
app = FastAPI()


@app.get("/stream/{n}")
def stream(n: int):
    """카메라 n의 결과 화면을 MJPEG(JPEG 연속)로 계속 보낸다"""
    def gen():
        while not stop.is_set():                 # (v1 수정 3) 종료 신호가 오면 전송도 끝냄
            f = jpegs[n] if n < len(jpegs) else None
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"   # yield: 하나씩 내보내는 생성기
            time.sleep(0.07)                     # 약 14장/초
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/state")
def get_state():
    """화면이 0.5초마다 가져가는 현재 상태 (타겟, 사람, 기록, 통계)"""
    with ev_lock:
        ev = list(events)[:25]                   # 최근 기록 25개
    return {**state, "room": ROOM, "near": args.near, "far": args.far, "events": ev, "cams": NUM_CAMS,
            "moving": args.moving,
            "cam_pos": [c["C"].round(2).tolist() for c in cams]}   # ** = 사전 펼치기 (state 내용 + 추가 항목)


@app.post("/retarget")
def do_retarget():                               # '타겟 다시 찾기' 버튼
    retarget.set(); return {"ok": True}


# 웹 화면. 평면도(SVG)에 강의실 범위, 카메라 위치(파랑), 타겟(보라), 근접 원(빨강), 이탈 원(점선), 사람 점을 그린다.
# 사람 점: 초록 = 삼각측량, 반투명 노랑 = 카메라 1대 추정(single), 빨강 = 근접 또는 충돌 위험
PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>강의실 접근 감지</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}.row{display:flex;gap:10px;flex-wrap:wrap}
.row>div{flex:1;min-width:300px}img{width:100%;border-radius:6px}svg{background:#1b1b1b;border-radius:6px;width:100%;max-width:640px}
table{border-collapse:collapse}td,th{border:1px solid #444;padding:3px 8px;font-size:13px}button{font-size:14px;padding:4px 10px}
.near{color:#ff5555;font-weight:bold}</style>
<h2>강의실 접근 감지 <span id=st></span></h2>
<p><button onclick="fetch('/retarget',{method:'POST'})">타겟 다시 찾기</button>
<button onclick="snd=true;beep()">알람 소리 켜기</button> <span id=tg></span></p>
<div class=row><div><img src="/stream/0"></div><div id=c1><img src="/stream/1"></div></div>
<div class=row><div><h3>평면도</h3><svg id=map viewBox="0 0 640 480"></svg></div>
<div><h3>사람</h3><table><thead><tr><th>ID</th><th>거리(m)</th><th>추세</th><th>구역</th><th>접근(m/s)</th><th>TTC(s)</th><th>자세</th><th>방식</th></tr></thead><tbody id=pp></tbody></table>
<h3>기록</h3><table><thead><tr><th>시각</th><th>대상</th><th>내용</th></tr></thead><tbody id=ev></tbody></table></div></div>
<script>
let snd=false,lastEv=0;
function beep(){try{const a=new AudioContext(),o=a.createOscillator();o.frequency.value=880;o.connect(a.destination);o.start();o.stop(a.currentTime+0.3);}catch(e){}}
async function poll(){const r=await (await fetch('/state')).json();
 const [x0,y0,x1,y1]=r.room,s=Math.min(600/(x1-x0),440/(y1-y0)),X=x=>20+(x-x0)*s,Y=y=>20+(y-y0)*s;
 const arrow=(x,y,v,c)=>r.moving&&Math.hypot(...v)>0.15?`<line x1=${X(x)} y1=${Y(y)} x2=${X(x+v[0])} y2=${Y(y+v[1])} stroke=${c} stroke-width=2 />`:'';
 if(r.cams<2)document.getElementById('c1').style.display='none';
 document.getElementById('st').textContent=r.stats.fps?`[${r.stats.mode}] ${r.stats.fps} fps, 카메라 시간 차 ${r.stats.skew_ms} ms, RAM ${r.stats.ram_mb} MB`:'';
 let g=`<rect x=${X(x0)} y=${Y(y0)} width=${(x1-x0)*s} height=${(y1-y0)*s} fill=none stroke=#555 />`;
 r.cam_pos.forEach((c,i)=>g+=`<rect x=${X(c[0])-6} y=${Y(c[1])-6} width=12 height=12 fill=#4af /><text x=${X(c[0])+8} y=${Y(c[1])} fill=#4af font-size=12>cam${i}</text>`);
 const t=r.target;
 if(t){const tx=X(t.xyz[0]),ty=Y(t.xyz[1]);
  g+=`<circle cx=${tx} cy=${ty} r=${r.far*s} fill=none stroke=#666 stroke-dasharray=4 /><circle cx=${tx} cy=${ty} r=${r.near*s} fill=#ff000022 stroke=#f55 />`+
     `<rect x=${tx-7} y=${ty-7} width=14 height=14 fill=#f0f /><text x=${tx+10} y=${ty-8} fill=#f0f font-size=12>TARGET</text>`;
  if(t.trail&&t.trail.length>1)g+=`<polyline points="${t.trail.map(p=>X(p[0])+','+Y(p[1])).join(' ')}" fill=none stroke=#f0f stroke-opacity=0.5 />`;
  if(t.v)g+=arrow(t.xyz[0],t.xyz[1],t.v,'#f0f');
  document.getElementById('tg').textContent=`타겟 ${t.xyz.join(', ')} (${t.how})`+(t.v?` 속도 ${Math.hypot(...t.v).toFixed(2)}m/s`:'')+(t.lost?' — 놓침':'');}
 else document.getElementById('tg').textContent='타겟 없음';
 r.people.forEach(p=>{const c=(p.zone=='near'||p.danger)?'#f55':p.kind=='single'?'#aa0':'#5f5';
  g+=arrow(p.x,p.y,p.v,c);
  g+=`<circle cx=${X(p.x)} cy=${Y(p.y)} r=8 fill=${c} ${p.kind=='single'?'fill-opacity=0.5':''} /><text x=${X(p.x)+10} y=${Y(p.y)+4} fill=${c} font-size=13>#${p.id} ${p.d??''}</text>`;});
 document.getElementById('map').innerHTML=g;
 document.getElementById('pp').innerHTML=r.people.map(p=>`<tr class=${(p.zone=='near'||p.danger)?'near':''}><td>#${p.id}</td><td>${p.d??'-'}</td><td>${p.trend}</td><td>${p.zone}</td><td>${p.closing}</td><td>${p.ttc??'-'}</td><td>${p.pose}</td><td>${p.kind}</td></tr>`).join('');
 document.getElementById('ev').innerHTML=r.events.map(e=>`<tr><td>${new Date(e.ts*1000).toLocaleTimeString()}</td><td>${e.who}</td><td>${e.label}</td></tr>`).join('');
 const fresh=r.events.filter(e=>e.ts>lastEv&&e.label.startsWith('알람'));if(fresh.length&&snd)beep();
 if(r.events.length)lastEv=Math.max(lastEv,r.events[0].ts);}
setInterval(poll,500);poll();
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    worker = threading.Thread(target=process_loop, daemon=True)   # 영상 처리는 별도 스레드, 웹 서버는 메인 스레드
    worker.start()
    try:
        # timeout_graceful_shutdown=1 : Ctrl+C 후 열린 /stream 연결을 1초만 기다리고 닫는다 (v1 수정 2)
        uvicorn.run(app, host="0.0.0.0", port=args.port, timeout_graceful_shutdown=1)
    finally:                                     # 어떻게 끝나든 꼭 실행 (v1 수정 3)
        stop.set()                               # 처리·카메라 스레드에 "그만" 신호
        worker.join(timeout=5)                   # 하던 프레임을 마치고 끝날 때까지 최대 5초 기다림
