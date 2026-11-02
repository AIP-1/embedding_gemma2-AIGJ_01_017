# stereo.py - 10단계 공통 모듈: 카메라 보정값 로드, 왜곡 보정, 삼각측량, 카메라 1대 높이 평면 추정, 사람 매칭
# 목적: 카메라 2대로 사람·물체의 강의실 바닥 위치를 구할 때 필요한 "기하 계산"을 한 곳에 모은다.
#       calib_extrinsics.py(10-2)와 proximity_app.py(10-3~10-8)가  import stereo as S  로 가져다 쓴다.
# 실행: 직접 실행하는 파일이 아니다. ~/eg2-lab/prox/ 에 다른 10단계 파일과 같이 둔다.
#
# 좌표계 (강의실 바닥 기준, 단위 m)
#   원점 = 바닥의 한 점(벽 모서리), x = 한 벽을 따라가는 방향, y = 원점에서 x를 바라볼 때 왼쪽, z = 높이(위 +)
#   (2026-10-28 README의 x·y를 바꿈: README 방향은 왼손 좌표계라 solvePnP가 '뒤를 보는 카메라'를 내고 ray_plane이 실패함)
#
# 용어 (자세한 설명은 study/기초개념_학습노트.md 10단계)
#   K    : 카메라 내부 행렬 3x3. 초점거리(fx, fy)와 영상 중심(cx, cy). 렌즈마다 다르다 (10-1에서 구함)
#   dist : 왜곡 계수 5개. 렌즈 가장자리가 휘어 보이는 정도 (10-1에서 구함)
#   R, t : 외부 파라미터. 강의실 좌표 → 카메라 좌표로 바꾸는 회전(3x3)과 이동(3x1) (10-2에서 구함)
#   P    : 투영 행렬 3x4 = K [R | t]. 3D 점을 화면 픽셀로 보내는 행렬
#   C    : 카메라 위치(강의실 좌표) = -Rᵀ t
#
# 수정 기록
# 2026-10-27  v1  README 10단계 stereo.py를 그대로 옮기고 줄별 설명 주석 추가 (지원 님 요청: 파이썬 복습용). 코드 변경 없음
# 2026-10-27  v2  카메라 열기 공통 함수 cam_name / open_source / read_frame 추가. 지원 님 장비가 USB 웹캠 1대 + IP 카메라 1대라
#                 README처럼 장치 번호(0, 2)만 받으면 IP 카메라(rtsp:// 주소)를 열 수 없기 때문. 세 보정·앱 파일이 함께 쓴다
# 2026-10-28  v2.1 주석만 수정: 좌표계 설명을 x=교실 뒤쪽, y=칠판 따라 오른쪽으로 (README 방향은 왼손 좌표계라서). 코드 변경 없음

import json                 # world_points.json(기준점 좌표) 읽기
import os                   # (v2) IP 카메라 수신 옵션을 환경 변수로 넘길 때 사용
import numpy as np          # 행렬·벡터 계산
import cv2                  # OpenCV: 왜곡 보정, 삼각측량, 3D→2D 투영

KP_CONF = 0.5               # 관절 신뢰도 기준. 이보다 낮은 관절은 "안 보임"으로 본다
# COCO 17관절 번호 중 여기서 쓰는 것: 0 코, 5 왼어깨, 6 오른어깨, 15 왼발목, 16 오른발목
NOSE, LS, RS, LA, RA = 0, 5, 6, 15, 16


# ---------------- (v2) 카메라 열기: USB 장치 번호 또는 IP 카메라 주소 ----------------
# 'rtsp_transport;tcp' : 영상을 TCP로 받는다 (UDP는 패킷이 빠지면 화면이 깨짐)
# 'fflags;nobuffer|flags;low_delay' : 수신 버퍼를 줄여 지연(몇 백 ms)을 줄인다
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")


def cam_name(src):
    """보정 파일 이름에 쓸 카메라 이름. "0" → "0" (cam0_intr.npz), "rtsp://..." → "ip" (camip_intr.npz)"""
    return str(src) if str(src).isdigit() else "ip"     # isdigit(): 숫자로만 된 글자인지


def open_source(src, W, H):
    """src가 숫자면 USB 웹캠(/dev/videoN), 아니면 IP 카메라 주소(rtsp://...)로 연다"""
    if str(src).isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)   # USB: 리눅스 카메라 드라이버
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    else:
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)      # IP: OpenCV에 들어 있는 FFmpeg로 스트림 수신
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                  # 오래된 프레임이 쌓이지 않게 (지원 안 하는 카메라는 무시됨)
    assert cap.isOpened(), f"카메라 {src}을 열 수 없음 (USB는 장치 번호, IP는 주소·아이디·비밀번호 확인)"
    return cap


def read_frame(cap, W, H):
    """한 프레임 읽기. 해상도가 W x H가 아니면(IP 카메라는 보통 1920x1080) W x H로 맞춘다.
    보정(10-1)과 실행(10-3~)이 똑같이 줄이므로 K가 줄인 화면 기준으로 계산되어 결과가 맞는다"""
    ok, f = cap.read()
    if not ok:
        return None
    if f.shape[1] != W or f.shape[0] != H:               # shape = (높이, 너비, 3)
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)   # INTER_AREA: 줄일 때 화질이 좋은 방식
    return f


def load_cam(path):
    """calib/camN.npz(10-2 결과)를 읽어 계산에 필요한 값을 사전(dict)으로 돌려준다"""
    d = np.load(path)                                     # npz = 넘파이 배열 여러 개를 묶은 파일
    K, dist, R, t = d["K"], d["dist"], d["R"], d["t"].reshape(3, 1)   # t는 세로 벡터(3x1)로 모양을 맞춘다
    return {"K": K, "dist": dist, "R": R, "t": t,
            "P": K @ np.hstack([R, t]),                   # @ = 행렬 곱. hstack: R(3x3) 옆에 t(3x1)를 붙여 3x4
            "C": (-R.T @ t).ravel(),                      # 카메라 위치. R.T = 전치(회전의 역), ravel = 1차원으로 펴기
            "Kinv": np.linalg.inv(K)}                     # K의 역행렬. 픽셀 → 시선 방향 계산(ray_plane)에 쓴다


def load_world(path):
    """world_points.json의 "points" 목록을 읽는다. 항목 예: {"id": "F1", "xyz": [1.0, 1.0, 0.0]}"""
    with open(path, encoding="utf-8") as f:               # with: 블록이 끝나면 파일을 자동으로 닫는다
        return json.load(f)["points"]


def room_bounds(points, pad=0.5):
    """기준점들을 감싸는 사각형 [x최소, y최소, x최대, y최대]에 여유 pad(m)를 더한 범위.
    이 밖으로 계산된 위치는 잘못된 값으로 보고 버린다 (proximity_app.py)"""
    xy = np.array([p["xyz"][:2] for p in points], dtype=float)   # [:2] = x, y만 (z 제외)
    return [float(xy[:, 0].min() - pad), float(xy[:, 1].min() - pad),   # [:, 0] = 모든 행의 0번 열(x)
            float(xy[:, 0].max() + pad), float(xy[:, 1].max() + pad)]


def undistort(cam, pts):
    """왜곡이 있는 픽셀 좌표 → 이상적인(왜곡 없는) 픽셀 좌표 (N, 2)
    삼각측량·투영 계산은 모두 '왜곡 없는' 좌표로 한다. 화면에 그릴 때만 다시 왜곡을 넣는다(project_distorted)"""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)   # OpenCV가 원하는 모양 (N, 1, 2). -1 = 개수는 알아서
    if len(pts) == 0:                                     # 점이 없으면 빈 (0, 2) 배열
        return np.zeros((0, 2))
    # P=cam["K"] : 결과를 '정규화 좌표'가 아니라 같은 K를 쓴 픽셀 좌표로 돌려받는다
    return cv2.undistortPoints(pts, cam["K"], cam["dist"], P=cam["K"]).reshape(-1, 2)


def project(cam, X):
    """3D 점(강의실 좌표) → 왜곡 없는 픽셀 좌표 (u, v)"""
    x = cam["P"] @ np.append(np.asarray(X, float), 1.0)   # 동차좌표: (x, y, z, 1)로 늘려 P(3x4)와 곱함
    return x[:2] / x[2]                                   # (a, b, c) → (a/c, b/c) = 화면 좌표


def project_distorted(cam, X):
    """3D 점 → 실제 영상 픽셀 좌표 (렌즈 왜곡 포함, 화면에 점을 찍을 때 사용)"""
    rvec, _ = cv2.Rodrigues(cam["R"])                     # 회전 행렬 3x3 → 회전 벡터 3개 (projectPoints 형식)
    p, _ = cv2.projectPoints(np.float64([X]), rvec, cam["t"], cam["K"], cam["dist"])
    return p.reshape(2)


def triangulate(c0, c1, p0, p1):
    """삼각측량: 두 카메라의 왜곡 없는 픽셀 좌표 → 3D 점, 평균 재투영 오차(px)
    재투영 오차 = 구한 3D 점을 다시 각 카메라 화면에 찍었을 때 원래 점과 떨어진 거리.
    두 카메라가 '같은 사람'을 보고 있으면 작고, 다른 사람끼리 짝지으면 커진다 → 매칭 기준(match_people)"""
    X = cv2.triangulatePoints(c0["P"], c1["P"], np.float64(p0).reshape(2, 1), np.float64(p1).reshape(2, 1))
    X = (X[:3] / X[3]).ravel()                            # 결과는 동차좌표 (x, y, z, w) → (x/w, y/w, z/w)
    err = (np.linalg.norm(project(c0, X) - p0) + np.linalg.norm(project(c1, X) - p1)) / 2   # norm = 거리
    return X, float(err)


def ray_plane(cam, p, h):
    """카메라 1대 추정: 픽셀 p(왜곡 없음)를 지나는 시선이 높이 h(m) 평면과 만나는 점.
    카메라 1대로는 거리를 알 수 없으므로 '높이가 h라고 가정'해서 위치를 정한다.
    가정한 높이가 틀리면 위치도 틀린다 (10-5에서 확인)"""
    d = cam["R"].T @ (cam["Kinv"] @ np.array([p[0], p[1], 1.0]))   # 픽셀 → 카메라 시선 방향 → 강의실 좌표 방향
    C = cam["C"]                                          # 시선의 시작점 = 카메라 위치
    if abs(d[2]) < 1e-6:                                  # 시선이 수평이면 평면과 만나지 않음
        return None
    s = (h - C[2]) / d[2]                                 # C + s*d 의 높이가 h가 되는 s
    return C + s * d if s > 0 else None                   # s<0 이면 카메라 뒤쪽 → 무효


def upper_body(k):
    """사람 위치로 쓸 상체 점: 어깨 중심, 어깨가 안 보이면 코. 책상에 가려지지 않는 관절이라 교실에 적합.
    k: 한 사람의 관절 (17, 3) = 각 관절의 (x, y, 신뢰도)"""
    if k[LS, 2] > KP_CONF and k[RS, 2] > KP_CONF:         # 두 어깨가 모두 보이면
        return (k[LS, :2] + k[RS, :2]) / 2                # 두 어깨의 가운데
    if k[NOSE, 2] > KP_CONF:
        return k[NOSE, :2].copy()                         # copy: 원본 배열과 연결을 끊는다
    return None                                           # 둘 다 안 보이면 위치 없음


def ankle(k):
    """보이는 발목의 평균 (10-3 '문제 확인'용. 책상에 가려지면 None)"""
    vis = [k[i, :2] for i in (LA, RA) if k[i, 2] > KP_CONF]   # 신뢰도가 높은 발목만 모은다
    return np.mean(vis, axis=0) if vis else None          # axis=0: 점들의 x끼리, y끼리 평균


def match_people(c0, c1, pts0, pts1, max_err=15.0, h_range=(0.5, 2.1)):
    """두 카메라의 사람(왜곡 없는 상체 좌표)을 재투영 오차가 작은 순으로 1:1 매칭.
    얼굴·옷 같은 외모는 쓰지 않는다. 모든 짝을 삼각측량해 보고
      - 재투영 오차가 max_err(px) 이하이고
      - 계산된 높이가 h_range(0.5~2.1m, 사람 상체가 있을 법한 높이) 안인 짝만 후보로 남긴 뒤
    오차가 작은 짝부터 확정한다 (한 사람은 한 번만).
    반환: [(i, j, X(3,), err)], 매칭 안 된 cam0 인덱스 집합, cam1 인덱스 집합"""
    cands = []
    for i, a in enumerate(pts0):                          # enumerate: (번호, 값)을 함께 꺼낸다
        for j, b in enumerate(pts1):                      # cam0의 i번 사람 × cam1의 j번 사람 모든 조합
            X, err = triangulate(c0, c1, a, b)
            if err <= max_err and h_range[0] <= X[2] <= h_range[1]:
                cands.append((err, i, j, X))
    cands.sort(key=lambda c: c[0])                        # 오차(0번째 값)가 작은 순으로 정렬
    used0, used1, out = set(), set(), []                  # set = 중복 없는 모음. 이미 짝지은 번호 기록
    for err, i, j, X in cands:
        if i in used0 or j in used1:                      # 이미 다른 짝에 쓰인 사람이면 건너뜀
            continue
        out.append((i, j, X, err))
        used0.add(i); used1.add(j)
    # 짝을 못 찾은 사람 = 전체 번호 - 쓰인 번호 (fused 모드에서 카메라 1대 추정에 쓰인다)
    return out, set(range(len(pts0))) - used0, set(range(len(pts1))) - used1


# ---------------- 10-8 이동 타겟: 칼만 필터와 상대 운동 ----------------
class Kalman2D:
    """등속 모델 칼만 필터. 상태 (x, y, vx, vy), 측정 (x, y).
    칼만 필터 = '예측(지금 속도로 움직였다면 여기쯤)'과 '측정(카메라가 본 위치)'을 믿을 만한 정도에 따라 섞어
    위치를 매끄럽게 하고, 직접 잴 수 없는 속도까지 추정하는 방법.
    q: 가속 잡음 (클수록 갑자기 방향·속도가 바뀌는 것을 빨리 따라감, 대신 흔들림↑)
    r: 측정 잡음(m²) (클수록 측정을 덜 믿음 → 매끄럽지만 늦게 따라감)"""
    def __init__(self, xy, q=1.0, r=0.05):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], float)   # 처음 상태: 측정 위치, 속도 0
        self.P = np.diag([r, r, 1.0, 1.0])                  # 상태의 불확실성(공분산). 속도는 처음에 잘 모름(1.0)
        self.q, self.r = q, r

    def predict(self, dt):
        """dt초 지난 뒤의 상태를 예측 (위치 += 속도 × dt)"""
        if dt <= 0:
            return
        F = np.eye(4); F[0, 2] = F[1, 3] = dt               # 상태 전이 행렬: x' = x + vx·dt, y' = y + vy·dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])   # 가속이 위치·속도에 주는 영향
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q * G @ G.T        # 시간이 지나면 불확실성이 커진다

    def update(self, z):
        """측정값 z=(x, y)로 상태를 보정"""
        Hm = np.eye(2, 4)                                   # 측정 행렬: 상태 4개 중 위치 2개만 잰다
        S_ = Hm @ self.P @ Hm.T + self.r * np.eye(2)        # 예측 불확실성 + 측정 잡음
        K = self.P @ Hm.T @ np.linalg.inv(S_)               # 칼만 이득: 측정을 얼마나 믿을지 (0~1 사이 비율 같은 것)
        self.x = self.x + K @ (np.asarray(z, float) - Hm @ self.x)   # 예측과 측정의 차이만큼 보정
        self.P = (np.eye(4) - K @ Hm) @ self.P              # 측정을 반영했으니 불확실성이 줄어든다

    def at(self, dt):
        """dt초 뒤 예측 위치 (상태는 바꾸지 않음)"""
        return self.x[:2] + self.x[2:] * dt


def relative_state(p, t):
    """p(사람), t(타겟): 상태 벡터 (x, y, vx, vy).
    반환: 거리(m), 접근 속도(m/s, +면 가까워짐), TTC(충돌 예상 시간, 초. 멀어지거나 느리면 None)"""
    dp, dv = t[:2] - p[:2], t[2:] - p[2:]                   # 상대 위치, 상대 속도
    d = float(np.linalg.norm(dp))
    closing = -float(dp @ dv) / max(d, 1e-6)                # 상대 속도를 '서로 잇는 방향'으로 투영 (부호 반대 = 접근 +)
    ttc = d / closing if closing > 0.1 else None            # 0.1m/s 이하로 다가오면 TTC가 너무 커서 의미 없음
    return d, closing, ttc
