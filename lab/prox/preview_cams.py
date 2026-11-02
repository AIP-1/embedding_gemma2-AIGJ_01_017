# preview_cams.py - 10-2 준비: 두 카메라 화면을 나란히 실시간으로 보며 카메라 방향을 맞춘다
# 목적: calib_extrinsics.py는 스냅샷(정지 사진)만 보여 주므로, 카메라를 돌리면서 구도를 확인하기 어렵다.
#       이 스크립트는 두 카메라 영상을 좌우로 붙여 브라우저로 계속 보내 준다. 화면 가운데 십자(+)는 카메라가
#       정면으로 보는 곳이다. 두 카메라의 십자가 모두 타겟 자리를 향하면 방향이 맞은 것.
# 실행: ~/eg2-lab/prox 에서  python preview_cams.py --cams 0 'rtsp://admin:비번@10.42.0.22:554/stream2'
#       → 맥북 브라우저에서 http://<JETSON_IP>:8010   (끝낼 때 Ctrl+C)
# 주의: 아무것도 저장하거나 바꾸지 않는다. 보기만 한다.
#
# 수정 기록
# 2026-10-28  v1  새로 작성. 지원 님 요청: 카메라 각도를 조정하면서 바로 화면으로 확인하고 싶음
# 2026-11-02  제출용 사본: 주석과 안내 출력 문구의 Jetson 접속 주소를 <JETSON_IP>, 계정명을 <USER>로 바꿈 (공개 리포에 개인 접속 정보를 올리지 않으려고). 코드 동작은 같음

import argparse, threading, time                     # 옵션 / 웹 서버 스레드 / 시간
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler   # 파이썬 기본 웹 서버
import cv2, numpy as np                              # 영상 처리 / 배열
from stereo import open_source, read_frame           # USB/IP 카메라 공통 열기·읽기

ap = argparse.ArgumentParser()
ap.add_argument("--cams", nargs=2, required=True)   # 카메라 두 대: USB 번호 또는 rtsp 주소 (calib_extrinsics와 같은 순서)
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8010, help="다른 학생이 쓰고 있으면 8011 등으로")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))

frames = [None, None]                                # 카메라별 최신 화면 (스레드끼리 같이 봄)
stop = threading.Event()                             # Ctrl+C 때 읽기 스레드를 멈추는 신호


def reader(n, src):
    """카메라 n을 계속 읽어 frames[n]에 최신 화면을 넣는다 (카메라마다 따로 돌아서 느린 쪽이 빠른 쪽을 막지 않음)"""
    cap = open_source(src, W, H)
    try:
        while not stop.is_set():
            f = read_frame(cap, W, H)
            if f is not None:
                frames[n] = f
    finally:
        cap.release()                                # 끝나면 카메라를 닫는다


def compose():
    """두 화면을 좌우로 붙이고 가운데 십자와 이름을 그린다"""
    views = []
    for n in (0, 1):
        f = frames[n].copy() if frames[n] is not None else np.zeros((H, W, 3), np.uint8)   # 아직 없으면 검은 화면
        cv2.line(f, (W // 2 - 30, H // 2), (W // 2 + 30, H // 2), (0, 0, 255), 2)   # 가로선 (빨강)
        cv2.line(f, (W // 2, H // 2 - 30), (W // 2, H // 2 + 30), (0, 0, 255), 2)   # 세로선
        cv2.putText(f, f"cam{n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        views.append(f)
    return np.hstack(views)                          # hstack: 가로로 붙이기 → 1280x480


class Preview(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")   # MJPEG: 사진을 계속 바꿔 보냄
        self.end_headers()
        try:
            while not stop.is_set():
                jpg = cv2.imencode(".jpg", compose())[1].tobytes()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.1)                      # 초당 10장
        except (BrokenPipeError, ConnectionResetError):
            pass                                     # 브라우저 탭을 닫으면 끊김. 오류 아님

    def log_message(self, *a):
        pass                                         # 접속 기록 출력 끔


threads = [threading.Thread(target=reader, args=(n, s), daemon=True) for n, s in enumerate(args.cams)]
for t in threads:
    t.start()
server = ThreadingHTTPServer(("0.0.0.0", args.port), Preview)
print(f"맥북 브라우저에서 http://<JETSON_IP>:{args.port}  (왼쪽 cam0, 오른쪽 cam1, 끝낼 때 Ctrl+C)")
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    stop.set()                                       # 읽기 스레드 멈춤 신호
    server.server_close()
    for t in threads:
        t.join(timeout=2)                            # 카메라가 닫힐 때까지 잠깐 기다림
