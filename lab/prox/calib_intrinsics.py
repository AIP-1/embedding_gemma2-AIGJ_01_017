# calib_intrinsics.py - 10-1 렌즈 보정 (카메라마다 1회)
# 목적: 체커보드를 여러 위치·각도로 비추면 자동으로 20장을 모아 카메라 내부 행렬 K와 왜곡 계수 dist를 구해
#       calib/cam<장치번호>_intr.npz 로 저장한다. 웹캠 렌즈는 가장자리가 휘어 보이므로(왜곡) 이 값으로 펴야
#       삼각측량 오차가 작아진다.
#       화면(모니터) 없이 동작한다. 진행 상황은 터미널에, 마지막으로 잡힌 체커보드 사진은 calib/cam<N>_last.jpg.
# 실행: ~/eg2-lab/prox 에서  python calib_intrinsics.py --cam 0                      USB 웹캠 (/dev/video0)
#                            python calib_intrinsics.py --cam "rtsp://아이디:비번@주소/경로"   IP 카메라 (v2)
#       미리보기(v3): 실행하면 맥북 브라우저에서 http://<JETSON_IP>:8010 으로 카메라 화면을 보며 찍을 수 있다
# 결과: 끝에 나오는 재투영 RMS가 0.5px 이하면 양호, 1.0 이상이면 다시 촬영
#
# 수정 기록
# 2026-10-27  v1  README 10-1 calib_intrinsics.py를 옮기고 줄별 설명 주석 추가 (지원 님 요청: 파이썬 복습용).
#                 README와 다른 점: 1) 끝나거나 Ctrl+C로 멈출 때 카메라를 닫는다(try/finally + cap.release()).
#                 열린 채로 끝나면 바로 다음 실행에서 장치가 '사용 중'이 될 수 있어서
#                 2) 5초 동안 체커보드를 못 찾으면 안내를 출력. 화면이 없어서 카메라가 무엇을 보는지 알 수 없기 때문
# 2026-10-27  v2  --cam에 IP 카메라 주소(rtsp://...)도 받도록 변경 (stereo.open_source/read_frame 사용). 지원 님 장비가
#                 USB 웹캠 1대 + IP 카메라 1대라서. IP 카메라 결과는 calib/camip_intr.npz 로 저장되고, 큰 해상도는 640x480으로
#                 줄여서 보정한다 (10-3 이후 실행 때도 똑같이 줄이므로 맞음)
# 2026-10-28  v3  브라우저 미리보기 추가 (--port, 기본 8010). 지원 님 요청: 화면 없이는 체커보드가 카메라에 어떻게
#                 보이는지 몰라서 찍기 어려움. 파이썬 기본 http.server로 최신 프레임을 MJPEG로 보내고, 찾은 꼭짓점과
#                 모은 장수를 화면에 그린다 (초록 = 찾음, 빨강 = 못 찾음)
# 2026-10-28  v4  IP 카메라 첫 결과 RMS 2.37px(기준 1.0 미만). 원인 추정: 보드가 움직이는 중에 찍힌 흐린 사진.
#                 1) 보드가 멈춰 있을 때만 찍는다: 직전 프레임과 꼭짓점 위치 차이가 STILL_PX(1.5px) 미만일 때만 저장.
#                    미리보기에 'HOLD STILL'(노랑) 표시  2) 이미 찍은 위치를 미리보기에 파란 점으로 표시 (빈 곳을 채우도록)
#                 3) 계산 후 사진별 오차를 출력하고, 오차가 큰 사진(중앙값의 2배 초과이면서 1px 초과)을 빼고 다시 계산
# 2026-10-28  v5  v4 결과 RMS 3.6px, 모든 사진 오차가 고르게 큼(1.2~5.4px), 중심 cy=-133 → 흔들림이 아니라 계산 쪽 문제로 봄.
#                 1) 꼭짓점 정밀화(cornerSubPix) 창 크기를 칸 크기에 맞춤. 기존 (11,11)은 23x23px를 보는데, IP 카메라 848 폭을
#                    640으로 줄이면 칸이 그보다 작아져 옆 꼭짓점이 섞일 수 있음(추정)  2) 렌즈 왜곡의 3차 항(k3)을 고정
#                    (CALIB_FIX_K3): 일반 웹캠에서 k3까지 풀면 값이 발산하기 쉬움. --k3 로 풀 수 있음
#                 3) 찍은 원본 사진을 calib/cam<이름>_imgs/ 에 저장하고, --from-dir 로 다시 찍지 않고 재계산 가능
# 2026-11-02  제출용 사본: 주석과 안내 출력 문구의 Jetson 접속 주소를 <JETSON_IP>, 계정명을 <USER>로 바꿈 (공개 리포에 개인 접속 정보를 올리지 않으려고). 코드 동작은 같음

import argparse, os, time   # argparse 명령줄 옵션 / os 폴더 만들기 / time 시각
import threading            # (v3) 미리보기 웹 서버를 따로 돌리기 위한 스레드
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler   # (v3) 파이썬에 기본으로 있는 웹 서버
import cv2, numpy as np     # cv2 카메라·체커보드 검출·보정 / numpy 배열
from stereo import cam_name, open_source, read_frame   # (v2) USB/IP 카메라를 같은 방식으로 열고 읽기

# ---------------- 실행 옵션 ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--cam", required=True)              # required=True: 꼭 줘야 하는 옵션. USB 번호(0) 또는 rtsp:// 주소 (v2: 글자로 받음)
ap.add_argument("--cols", type=int, default=9, help="체커보드 내부 코너 가로 개수")   # 칸 10개 → 안쪽 꼭짓점 9개
ap.add_argument("--rows", type=int, default=6, help="체커보드 내부 코너 세로 개수")   # 칸 7개 → 안쪽 꼭짓점 6개
ap.add_argument("--square", type=float, default=0.025, help="칸 한 변 길이(m)")      # 인쇄 후 자로 재서 넣는다
ap.add_argument("--n", type=int, default=20, help="모을 이미지 수")
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8010, help="(v3) 미리보기 포트. 다른 학생이 쓰고 있으면 8011 등으로")
ap.add_argument("--from-dir", default=None, help="(v5) 카메라 대신 저장된 사진 폴더로 재계산. 예: calib/camip_imgs")
ap.add_argument("--k3", action="store_true", help="(v5) 왜곡 3차 항 k3도 계산 (기본은 0으로 고정)")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))                  # "640x480" → ["640", "480"] → 640, 480
os.makedirs("calib", exist_ok=True)                   # calib 폴더가 없으면 만든다 (있으면 그대로)
NAME = cam_name(args.cam)                             # (v2) 파일 이름용: "0" 또는 "ip"
IMG_DIR = f"calib/cam{NAME}_imgs"                     # (v5) 찍은 원본 사진을 모아 두는 폴더

# 체커보드 꼭짓점의 '실제' 좌표(m). 판 위에 z=0으로 놓고 (0,0), (0.025,0), ... 격자로 만든다
objp = np.zeros((args.rows * args.cols, 3), np.float32)               # 꼭짓점 54개 × (x, y, z)
objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square   # mgrid: 격자 번호 → 칸 크기를 곱해 m로

# ---------------- (v3) 미리보기 웹 서버 ----------------
STILL_PX = 1.5                                        # (v4) 꼭짓점이 프레임 사이에 이 픽셀보다 적게 움직이면 '멈춤'
latest = {"jpg": None}                                # 가장 최근 화면(JPEG 바이트). 딕셔너리로 둬서 스레드끼리 같이 본다


class Preview(BaseHTTPRequestHandler):
    def do_GET(self):                                 # 브라우저가 주소를 열면 호출됨
        self.send_response(200)
        # multipart/x-mixed-replace: 사진을 계속 바꿔 끼워 보내는 방식(MJPEG). 브라우저가 동영상처럼 보여 준다
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                if latest["jpg"] is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest["jpg"] + b"\r\n")
                time.sleep(0.1)                       # 초당 10장이면 보기에 충분
        except (BrokenPipeError, ConnectionResetError):
            pass                                      # 브라우저 탭을 닫으면 연결이 끊김. 오류 아님

    def log_message(self, *a):
        pass                                          # 접속할 때마다 터미널에 찍히는 기록을 끔 (진행 상황이 묻히지 않게)




def show(frame, corners, found, still):
    """(v3) 화면에 꼭짓점과 장수를 그려 미리보기로 보낸다. (v4) 멈춤 여부와 이미 찍은 위치도 표시"""
    view = frame.copy()                               # 원본은 그대로 두고 복사본에 그림
    for pts in img_pts:                               # (v4) 이미 찍은 보드의 꼭짓점을 파란 점으로 → 빈 곳이 보인다
        for x, y in pts.reshape(-1, 2)[::4]:          # [::4]: 4개 중 1개만 그려서 화면이 덜 복잡하게
            cv2.circle(view, (int(x), int(y)), 2, (255, 0, 0), -1)   # (255,0,0) = BGR 파랑, -1 = 속을 채움
    if found:
        cv2.drawChessboardCorners(view, (args.cols, args.rows), corners, True)
    if not found:
        msg, color = "NO BOARD", (0, 0, 255)          # 빨강: 보드를 못 찾음
    elif not still:
        msg, color = "HOLD STILL", (0, 220, 255)      # 노랑: 찾았지만 움직이는 중 (v4)
    else:
        msg, color = "FOUND", (0, 200, 0)             # 초록: 찾았고 멈춰 있음
    cv2.putText(view, f"{len(img_pts)}/{args.n} {msg}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    latest["jpg"] = cv2.imencode(".jpg", view)[1].tobytes()   # JPEG로 압축해 저장


def refine(gray, corners):
    """(v5) 꼭짓점을 소수점 픽셀까지 정밀화. 창 크기를 칸 크기의 약 40%로 맞춰 옆 꼭짓점이 섞이지 않게 한다"""
    g = corners.reshape(args.rows, args.cols, 2)                      # 꼭짓점을 (세로, 가로, xy) 격자 모양으로
    dx = np.linalg.norm(np.diff(g, axis=1), axis=2)                   # 가로로 이웃한 꼭짓점 사이 거리(px)
    dy = np.linalg.norm(np.diff(g, axis=0), axis=2)                   # 세로로 이웃한 꼭짓점 사이 거리(px)
    sq = min(dx.min(), dy.min())                                      # 가장 작게 보이는 칸 크기
    win = int(np.clip(sq * 0.4, 2, 11))                               # 창 반지름: 2~11 사이로 제한
    return cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), crit), win


crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)   # 코너 정밀화 반복 조건 (30회 또는 0.001px)
obj_pts, img_pts = [], []                             # 모은 실제 좌표 / 화면 좌표

if args.from_dir:                                     # ---------------- (v5) 저장된 사진으로 재계산 ----------------
    files = sorted(f for f in os.listdir(args.from_dir) if f.endswith(".png"))   # 사진 파일 이름을 순서대로
    for f in files:
        gray = cv2.imread(os.path.join(args.from_dir, f), cv2.IMREAD_GRAYSCALE)  # 흑백으로 읽기
        found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows),
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"  {f}: 체커보드를 못 찾음, 건너뜀")
            continue
        corners, win = refine(gray, corners)
        obj_pts.append(objp); img_pts.append(corners)
        print(f"  {f}: 정밀화 창 {win}px")
    W, H = gray.shape[1], gray.shape[0]               # 사진 크기를 해상도로 사용
    files_used = files
else:
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Preview)    # 0.0.0.0: 맥북(Tailscale)에서도 접속 가능
    threading.Thread(target=server.serve_forever, daemon=True).start()   # daemon=True: 프로그램이 끝나면 같이 꺼짐
    print(f"미리보기: 맥북 브라우저에서 http://<JETSON_IP>:{args.port}")

    # ---------------- 카메라로 찍기 ----------------
    cap = open_source(args.cam, W, H)                     # (v2) USB면 MJPG 640x480으로, IP면 FFmpeg로 연다 (stereo.py)

    last_t, last_c = 0.0, None                            # 마지막으로 모은 시각·위치
    os.makedirs(IMG_DIR, exist_ok=True)                   # (v5) 원본 사진 폴더
    for f in os.listdir(IMG_DIR):                         # (v5) 지난번 사진은 지운다 (이번 촬영분만 남게)
        os.remove(os.path.join(IMG_DIR, f))
    print(f"체커보드({args.cols}x{args.rows})를 화면 곳곳·여러 각도로 천천히 움직이세요. 목표 {args.n}장")
    last_found = time.time()                              # (v1 추가) 마지막으로 체커보드를 찾은 시각
    prev = None                                           # (v4) 직전 프레임의 꼭짓점 (멈춤 판단용)
    try:
        while len(img_pts) < args.n:                      # 목표 장수를 모을 때까지 반복
            frame = read_frame(cap, W, H)                 # (v2) 한 프레임 읽기 + 640x480으로 맞추기. 실패하면 None
            if frame is None:
                continue                                  # continue: 아래를 건너뛰고 다음 반복으로
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                # 흑백으로 (코너 검출은 밝기만 씀)
            # 체커보드 안쪽 꼭짓점 찾기. FAST_CHECK: 보드가 없으면 빨리 포기 (속도↑)
            found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), cv2.CALIB_CB_FAST_CHECK)
            # (v4) 멈춤 판단: 직전 프레임에서도 찾았고, 꼭짓점들이 평균 STILL_PX 미만으로 움직였으면 멈춘 것
            still = (found and prev is not None
                     and np.linalg.norm(corners.reshape(-1, 2) - prev, axis=1).mean() < STILL_PX)
            prev = corners.reshape(-1, 2).copy() if found else None
            show(frame, corners, found, still)            # (v3) 미리보기 갱신 (보드를 찾았든 못 찾았든 매 프레임)
            if found:
                last_found = time.time()
            elif time.time() - last_found > 5:            # (v1 추가) 5초째 못 찾으면 안내
                print("  체커보드가 안 보입니다: 판 전체가 화면에 들어오게, 빛 반사 없이, 너무 멀지 않게")
                last_found = time.time()
            if not still or time.time() - last_t < 1.0:   # 못 찾았거나 움직이는 중(v4)이거나, 1초가 안 지났으면 건너뜀
                continue
            c = corners.reshape(-1, 2).mean(0)            # 꼭짓점들의 중심 = 보드가 화면 어디에 있나
            if last_c is not None and np.linalg.norm(c - last_c) < 40:     # 비슷한 위치(40px 이내)는 건너뜀
                continue
            cv2.imwrite(f"{IMG_DIR}/{len(img_pts) + 1:02d}.png", frame)   # (v5) 원본 저장 (PNG = 화질 손실 없음). :02d = 01, 02 ...
            corners, win = refine(gray, corners)          # (v5) 꼭짓점 정밀화 (창 크기를 칸 크기에 맞춤)
            obj_pts.append(objp); img_pts.append(corners)  # 실제 좌표와 화면 좌표를 한 쌍으로 저장
            last_t, last_c = time.time(), c
            cv2.drawChessboardCorners(frame, (args.cols, args.rows), corners, True)   # 찾은 꼭짓점을 그려서
            cv2.imwrite(f"calib/cam{NAME}_last.jpg", frame)                       # 확인용 사진으로 저장
            print(f"  {len(img_pts)}/{args.n}  (정밀화 창 {win}px)")
    finally:                                              # (v1 추가) 정상 종료든 Ctrl+C든 카메라를 닫는다
        cap.release()
        server.shutdown()                                 # (v3) 미리보기 서버도 끈다

# ---------------- 보정 계산 ----------------
# calibrateCamera: 여러 장의 (실제 좌표, 화면 좌표) 쌍에서 K와 dist를 구한다.
# rms = 구한 값으로 꼭짓점을 다시 화면에 찍었을 때의 평균 오차(px). 작을수록 정확
FLAGS = 0 if args.k3 else cv2.CALIB_FIX_K3            # (v5) 기본: k3를 0으로 고정
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (W, H), None, None, flags=FLAGS)   # rvecs/tvecs: 사진마다 보드의 자세


def view_errors(K, dist, rvecs, tvecs):
    """(v4) 사진마다 재투영 오차(px)를 구한다: 계산된 값으로 꼭짓점을 다시 그려 실제 위치와의 거리 평균"""
    errs = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)                 # 실제 좌표 → 화면 좌표로 다시 투영
        errs.append(float(np.sqrt(((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2).sum(1).mean())))
    return np.array(errs)


errs = view_errors(K, dist, rvecs, tvecs)
print("\n사진별 오차(px):", " ".join(f"{e:.2f}" for e in errs))
bad = (errs > 2 * np.median(errs)) & (errs > 1.0)     # (v4) 다른 사진보다 유독 큰 것만 '나쁜 사진'으로
if bad.any() and (~bad).sum() >= 10:                  # 빼고도 10장 이상 남을 때만 다시 계산
    print(f"오차가 큰 사진 {int(bad.sum())}장을 빼고 다시 계산합니다: 번호 {[int(k) + 1 for k in np.where(bad)[0]]}")
    obj_pts = [o for o, b in zip(obj_pts, bad) if not b]           # 리스트 컴프리헨션: 나쁜 사진만 빼고 새 리스트
    img_pts = [i for i, b in zip(img_pts, bad) if not b]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (W, H), None, None, flags=FLAGS)
print(f"사용한 사진 {len(img_pts)}장")
np.savez(f"calib/cam{NAME}_intr.npz", K=K, dist=dist, res=np.array([W, H]))   # 여러 배열을 한 파일로 저장
print(f"\n재투영 RMS {rms:.3f}px  (0.5 이하 양호, 1.0 이상이면 다시 촬영)")
print(f"K=\n{K.round(1)}\ndist={dist.ravel().round(3)}\n저장: calib/cam{NAME}_intr.npz")
