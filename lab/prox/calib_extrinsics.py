# calib_extrinsics.py - 10-2 카메라 위치 보정 (카메라를 설치한 뒤 1회, 카메라가 움직이면 다시)
# 목적: 강의실에 표시한 기준점(실제 좌표를 줄자로 잰 점)을 웹 화면의 두 카메라 스냅샷 위에서 클릭하면,
#       solvePnP로 각 카메라가 강의실 어디에 어느 방향으로 있는지(R, t)를 계산해 calib/cam0.npz, cam1.npz로 저장한다.
#       이 두 파일이 있어야 proximity_app.py가 삼각측량을 할 수 있다.
# 준비: calib/world_points.json 에 기준점 실제 좌표를 적어 둔다 ("check": true 인 점은 계산에 쓰지 않고 검증에만 사용)
#       calib/cam<장치번호 또는 ip>_intr.npz (10-1 결과) 가 두 카메라 모두 있어야 한다
# 실행: ~/eg2-lab/prox 에서  python calib_extrinsics.py --cams 0 "rtsp://..."   →  맥북 브라우저에서 http://<JETSON_IP>:8000
#       --cams 순서가 cam0, cam1 이 된다. 이후 proximity_app.py에도 같은 순서로 쓴다
#
# 수정 기록
# 2026-10-27  v1  README 10-2 calib_extrinsics.py를 옮기고 줄별 설명 주석 추가 (지원 님 요청: 파이썬 복습용).
#                 README와 다른 점: uvicorn에 timeout_graceful_shutdown=1 (8단계에서 Ctrl+C로 바로 안 꺼지던 문제 예방)
# 2026-10-27  v2  --cams에 IP 카메라 주소(rtsp://...)도 받도록 변경 (stereo.open_source/read_frame/cam_name 사용).
#                 지원 님 장비가 USB 웹캠 1대 + IP 카메라 1대라서. 예: --cams 0 "rtsp://..." → 10-1 결과 cam0_intr.npz, camip_intr.npz를 읽는다
# 2026-10-28  v3  좌표계 거울상 검사 추가. README 좌표계(x=칠판 따라 오른쪽, y=교실 뒤쪽, z=위)는 왼손 좌표계라서,
#                 solvePnP가 오차 0처럼 보이는 '카메라가 뒤를 보는' 답을 내고, 10-5(ray_plane)가 항상 실패함을 시뮬레이션으로 확인.
#                 지원 님 실습은 x=교실 뒤쪽, y=칠판 따라 오른쪽, z=위(오른손 좌표계)로 바꿈. 기준점이 카메라 뒤로 계산되면 경고를 낸다
# 2026-10-28  v4  1) 첫 계산을 SOLVEPNP_ITERATIVE → SOLVEPNP_SQPNP로 변경. 지원 님 기준점(8개, 책상 근처 좁은 영역, 높이 2종)으로
#                    시뮬레이션(클릭 오차 1.5px, 300회)해 보니 ITERATIVE는 17~25%가 엉뚱한 답(위치 오차 >0.5m)으로 빠지고,
#                    SQPNP는 0%였음(전역 최적해를 찾는 방식)  2) world_points.json의 "_카메라_실측"이 있으면 계산 위치와의
#                    차이(cm)를 결과에 함께 보여 준다 (README 10-2 표의 '계산 위치 vs 실측 위치' 칸)
# 2026-11-02  제출용 사본: 주석과 안내 출력 문구의 Jetson 접속 주소를 <JETSON_IP>, 계정명을 <USER>로 바꿈 (공개 리포에 개인 접속 정보를 올리지 않으려고). 코드 동작은 같음

import argparse, json, os, time        # argparse 옵션 / json 기준점·클릭 파일 / os 파일 존재 확인 / time
import cv2, numpy as np, uvicorn       # cv2 카메라·solvePnP / numpy / uvicorn 웹 서버
from fastapi import FastAPI, Body      # 웹 앱, Body = 요청 본문(JSON)을 받는 표시
from fastapi.responses import HTMLResponse, Response   # HTML 화면 / 이미지 같은 일반 응답
from stereo import load_world, undistort, triangulate, load_cam   # 같은 폴더의 stereo.py 함수들
from stereo import cam_name, open_source, read_frame              # (v2) USB/IP 카메라 공통 열기·읽기

# ---------------- 실행 옵션 ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--cams", nargs=2, default=["0", "1"])         # nargs=2: 값 두 개 (v2: 글자로 받아 USB 번호와 rtsp 주소 모두 허용)
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8000, help="다른 학생이 8000을 쓰고 있으면 8001 등으로")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
WORLD = load_world("calib/world_points.json")         # 기준점 목록 [{"id": "F1", "xyz": [...]}, ...]
WJSON = json.load(open("calib/world_points.json", encoding="utf-8"))   # (v4) 카메라 실측 위치 등 다른 항목도 읽기
WMAP = {p["id"]: p for p in WORLD}                    # id로 바로 찾을 수 있게 사전으로: {"F1": {...}, ...}
CLICKS_F = "calib/clicks.json"                        # 클릭한 위치를 저장하는 파일 (다시 실행해도 유지)
# 조건 표현식: A if 조건 else B. 파일이 있으면 읽고, 없으면 빈 클릭 기록으로 시작
clicks = json.load(open(CLICKS_F)) if os.path.exists(CLICKS_F) else {"0": {}, "1": {}}   # {"0": {"F1": [u, v]}, "1": {...}}
snaps = {}                                            # 카메라별 스냅샷 이미지 {"0": 이미지, "1": 이미지}


def open_cam(i):
    """카메라 i(USB 번호 또는 rtsp 주소)를 연다 (v2: stereo.open_source에 맡김)"""
    return open_source(i, W, H)


def take_snaps():
    """두 카메라에서 사진을 한 장씩 찍어 snaps에 넣고 calib/snap0.jpg, snap1.jpg로도 저장"""
    for n, ci in enumerate(args.cams):                # n = 0, 1 (순서), ci = 실제 장치 번호 또는 rtsp 주소
        cap = open_cam(ci)
        f = None
        for _ in range(15):                           # 자동 노출 안정화: 처음 몇 장은 어둡거나 밝으므로 15장 버림
            g = read_frame(cap, W, H)                 # (v2) 읽기 + 640x480으로 맞추기. _ : 쓰지 않는 반복 변수 이름 관례
            if g is not None:                         # 읽기에 실패한 프레임은 건너뛰고 마지막 성공 프레임을 쓴다
                f = g
        assert f is not None, f"카메라 {ci}에서 프레임을 받지 못함"
        cap.release()                                 # 한 대씩 열고 닫는다 (동시에 열 필요 없음)
        snaps[str(n)] = f
        cv2.imwrite(f"calib/snap{n}.jpg", f)


def intr(n):
    """n번째(0 또는 1) 카메라의 10-1 결과(K, dist) 읽기"""
    d = np.load(f"calib/cam{cam_name(args.cams[n])}_intr.npz")  # 10-1 파일 이름 (예: cam0_intr.npz, IP는 camip_intr.npz)
    return d["K"], d["dist"]


def solve():
    """클릭한 점으로 카메라별 위치·방향 계산 → calib/cam0.npz, cam1.npz 저장, 오차 보고"""
    out = {}
    for n in ("0", "1"):
        K, dist = intr(int(n))
        # 계산용 점: 이 카메라에서 클릭했고, world_points에 있고, check(검증용)가 아닌 점
        ids = [i for i in clicks[n] if i in WMAP and not WMAP[i].get("check")]   # .get: 키가 없으면 None
        if len(ids) < 6:
            out[n] = {"error": f"계산용 기준점 {len(ids)}개 (6개 이상 필요)"}; continue
        obj = np.float64([WMAP[i]["xyz"] for i in ids])   # 실제 3D 좌표 (N, 3)
        img = np.float64([clicks[n][i] for i in ids])     # 화면 클릭 좌표 (N, 2)
        # solvePnP: '3D 점들이 화면의 이 위치에 보였다' → 카메라의 회전(rvec)·이동(tvec)을 구한다
        first = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)   # (v4) SQPNP가 없는 옛 OpenCV면 ITERATIVE
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=first)
        # 첫 결과를 초기값으로 한 번 더 다듬는다 (True = 초기값 사용)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)        # 구한 카메라로 기준점을 다시 화면에 찍어 보고
        errs = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)    # 클릭 위치와의 거리(px) = 재투영 오차
        R, _ = cv2.Rodrigues(rvec)                                   # 회전 벡터 → 회전 행렬 3x3
        np.savez(f"calib/cam{n}.npz", K=K, dist=dist, R=R, t=tvec, res=np.array([W, H]))
        C = (-R.T @ tvec).ravel()                                    # 카메라 위치(강의실 좌표, m)
        out[n] = {"pos": C.round(2).tolist(), "reproj": {i: round(float(e), 1) for i, e in zip(ids, errs)},
                  "mean_px": round(float(errs.mean()), 2)}           # zip: 두 목록을 짝지어 함께 꺼냄
        meas = {k[:4]: v for k, v in WJSON.get("_카메라_실측", {}).items()}   # (v4) {"cam0": [x,y,z], "cam1": [...]}
        if f"cam{n}" in meas:
            out[n]["pos_vs_measured_cm"] = round(float(np.linalg.norm(C - np.array(meas[f"cam{n}"]))) * 100, 1)
        depth = (R @ obj.T + tvec)[2]                                # (v3) 카메라에서 본 각 기준점의 앞뒤 거리(+면 카메라 앞)
        if (depth <= 0).any():                                       # 하나라도 카메라 뒤쪽이면 좌표계가 거울상
            out[n]["warning"] = "기준점이 카메라 뒤쪽으로 계산됨: 좌표계가 거울상입니다. 원점에서 x 방향을 바라볼 때 y가 왼쪽인지 확인"
    # 두 카메라 모두 성공했으면, 두 카메라 모두 클릭한 점을 삼각측량해 실제 좌표와 비교 (check 점 포함)
    if "error" not in out.get("0", {"error": 1}) and "error" not in out.get("1", {"error": 1}):
        c0, c1 = load_cam("calib/cam0.npz"), load_cam("calib/cam1.npz")
        ver = {}
        for i in set(clicks["0"]) & set(clicks["1"]) & set(WMAP):   # & = 교집합: 세 곳 모두에 있는 id
            p0, p1 = undistort(c0, [clicks["0"][i]])[0], undistort(c1, [clicks["1"][i]])[0]
            X, _ = triangulate(c0, c1, p0, p1)
            ver[i] = {"check": bool(WMAP[i].get("check")),
                      "err_cm": round(float(np.linalg.norm(X - np.array(WMAP[i]["xyz"]))) * 100, 1)}   # m → cm
        out["verify"] = ver
    return out


# ---------------- 웹 앱 ----------------
app = FastAPI()


@app.get("/img/{n}.jpg")              # 브라우저가 /img/0.jpg 를 요청하면 실행. {n}이 함수 인자 n으로 들어온다
def img(n: str):
    """스냅샷 위에 지금까지 클릭한 점과 이름을 그려서 JPEG로 보낸다"""
    f = snaps[n].copy()                                   # 원본에 그리지 않도록 복사
    for i, (u, v) in clicks[n].items():                   # .items(): (키, 값) 쌍. 값 [u, v]를 바로 풀어 받음
        col = (0, 200, 255) if WMAP.get(i, {}).get("check") else (0, 255, 0)   # BGR 색: 주황 = 검증용, 초록 = 계산용
        cv2.circle(f, (int(u), int(v)), 5, col, -1)       # -1 = 속을 채운 원
        cv2.putText(f, i, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return Response(cv2.imencode(".jpg", f)[1].tobytes(), media_type="image/jpeg")


@app.post("/click")
def click(d: dict = Body(...)):       # 화면에서 점을 클릭하면 {"cam": 0, "id": "F1", "u": .., "v": ..} 가 온다
    clicks[str(d["cam"])][d["id"]] = [float(d["u"]), float(d["v"])]
    json.dump(clicks, open(CLICKS_F, "w"), indent=1)      # 클릭할 때마다 파일에 저장
    return {"ok": True}


@app.post("/undo")
def undo(d: dict = Body(...)):        # 선택한 기준점의 클릭을 지운다
    clicks[str(d["cam"])].pop(d["id"], None)              # pop(키, None): 없으면 오류 대신 None
    json.dump(clicks, open(CLICKS_F, "w"), indent=1)
    return {"ok": True}


@app.post("/snap")
def snap():                           # 새 스냅샷 (사람이 기준점을 가렸을 때 다시 찍기)
    take_snaps(); return {"ok": True}


@app.post("/solve")
def do_solve():                       # '계산' 버튼
    return solve()


@app.get("/points")
def points():                         # 화면의 기준점 드롭다운을 만들 때 쓰는 목록
    return {"points": WORLD, "clicks": clicks}


# 브라우저에 보낼 화면 (HTML + 자바스크립트). 이미지를 클릭하면 실제 이미지 픽셀 좌표로 바꿔 /click 으로 보낸다
PAGE = """<!doctype html><meta charset=utf-8><title>카메라 위치 보정</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}.row{display:flex;gap:12px;flex-wrap:wrap}
.row div{flex:1;min-width:300px}img{width:100%;cursor:crosshair;border:1px solid #444}select,button{font-size:15px;padding:4px 10px}
pre{background:#222;padding:8px;white-space:pre-wrap}</style>
<h2>카메라 위치 보정</h2>
<p>기준점 선택 → 해당 카메라 이미지에서 그 점을 클릭. 초록 = 계산용, 주황 = 검증용(check). 카메라마다 계산용 6개 이상.</p>
<p><select id=pid></select> <button onclick="undo(0)">cam0 선택점 지우기</button> <button onclick="undo(1)">cam1 선택점 지우기</button>
<button onclick="snapNew()">새 스냅샷</button> <button onclick="solve()">계산</button></p>
<div class=row><div>cam0<br><img id=i0 src="/img/0.jpg" onclick="clk(event,0)"></div>
<div>cam1<br><img id=i1 src="/img/1.jpg" onclick="clk(event,1)"></div></div>
<h3>결과</h3><pre id=out>-</pre>
<script>
async function load(){const r=await (await fetch('/points')).json();
 document.getElementById('pid').innerHTML=r.points.map(p=>`<option value=${p.id}>${p.id} (${p.xyz.join(', ')})${p.check?' 검증':''}</option>`).join('');}
function refresh(){for(const n of [0,1])document.getElementById('i'+n).src=`/img/${n}.jpg?`+Date.now();}
async function clk(e,n){const im=e.target,u=e.offsetX*im.naturalWidth/im.clientWidth,v=e.offsetY*im.naturalHeight/im.clientHeight;
 await fetch('/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cam:n,id:pid.value,u,v})});refresh();}
async function undo(n){await fetch('/undo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cam:n,id:pid.value})});refresh();}
async function snapNew(){await fetch('/snap',{method:'POST'});refresh();}
async function solve(){const r=await (await fetch('/solve',{method:'POST'})).json();document.getElementById('out').textContent=JSON.stringify(r,null,1);}
load();
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":            # 이 파일을 직접 실행했을 때만 (import 될 때는 실행 안 됨)
    take_snaps()                      # 시작할 때 스냅샷 한 번
    # timeout_graceful_shutdown=1 : Ctrl+C 후 열린 연결을 1초만 기다리고 끈다 (v1)
    uvicorn.run(app, host="0.0.0.0", port=args.port, timeout_graceful_shutdown=1)
