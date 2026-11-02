# find_rtsp.py - 같은 네트워크에서 IP 카메라(RTSP 554번 포트가 열린 기기)를 찾는다 (10단계 준비용)
# 목적: VIGI 앱에서 카메라 IP를 확인할 수 없을 때, Jetson이 붙은 네트워크의 주소를 하나씩 두드려
#       554번(RTSP) 포트에 응답하는 기기와 그 MAC 주소를 출력한다. 카메라 스티커의 MAC과 비교해 카메라 IP를 찾는다.
# 실행: python find_rtsp.py              (Jetson의 172.x / 192.168.x / 10.x 주소 대역을 자동으로 찾아 검사)
#       python find_rtsp.py 172.27.0.0/16 (검사할 대역을 직접 지정)
#       python find_rtsp.py 172.27.0.0/16 --port 443   (v2: 다른 포트로 검사. 443 = 카메라 웹 설정 화면)
# 주의: 연결만 시도하고 아무것도 보내거나 바꾸지 않는다. 대역이 크면(/16 = 6만여 개) 1~2분 걸린다.
#
# 수정 기록
# 2026-10-28  v1  새로 작성. 학교 free 와이파이에서 VIGI 앱 자동 검색이 카메라를 못 찾았지만, Jetson→휴대폰 ping은
#                 되어(기기 간 격리 없음) Jetson에서 직접 카메라를 찾기 위해
# 2026-10-28  v2  --port 옵션 추가 (기본 554). free 와이파이 172.24~27 전체에서 554가 하나도 안 나와서, 카메라 웹 설정 포트(443)로도
#                 찾아보고 'RTSP만 꺼진 것'인지 '카메라가 이 네트워크에 없는 것'인지 구분하기 위해

import argparse, ipaddress, socket, subprocess, sys     # ipaddress 주소 대역 계산 / socket 포트 연결 / subprocess 명령 실행 / sys 인자
from concurrent.futures import ThreadPoolExecutor   # 여러 주소를 동시에 검사 (하나씩 하면 너무 오래 걸림)

ap = argparse.ArgumentParser()
ap.add_argument("net", nargs="?", default=None, help="검사할 대역. 예: 172.27.0.0/16 (생략하면 자동)")   # nargs="?": 없어도 됨
ap.add_argument("--port", type=int, default=554, help="554 = RTSP, 443 = 웹 설정 화면 (v2)")
args = ap.parse_args()
PORT = args.port     # 검사할 포트 (기본 554 = RTSP)
TIMEOUT = 0.5        # 주소 하나당 기다리는 시간(초)


def local_net():
    """Jetson의 사설 IP 대역을 찾는다. 'ip -4 -o addr' 출력에서 172.16~31 / 192.168 / 10 으로 시작하는 주소 (Tailscale 100.x 제외)"""
    out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        cidr = line.split()[3]                               # 예: 172.27.57.20/16 (4번째 칸)
        net = ipaddress.ip_interface(cidr).network           # → 172.27.0.0/16
        if net.is_private and not str(net).startswith(("127.", "100.")):
            return net, line.split()[1]                      # (대역, 장치 이름 예: wlan0)
    sys.exit("사설 IP 대역을 찾지 못함. 대역을 직접 지정하세요: python find_rtsp.py 172.27.0.0/16")


def probe(ip):
    """ip의 554번 포트에 연결해 본다. 연결되면 ip(문자열), 아니면 None"""
    with socket.socket() as s:
        s.settimeout(TIMEOUT)
        return str(ip) if s.connect_ex((str(ip), PORT)) == 0 else None   # connect_ex: 실패해도 오류 대신 0이 아닌 숫자


if args.net:                                                 # 대역을 직접 준 경우
    net, dev = ipaddress.ip_network(args.net, strict=False), "?"
else:
    net, dev = local_net()
if net.num_addresses > 70000:                                # /16보다 크면 너무 오래 걸리므로 멈춘다
    sys.exit(f"{net} 은 너무 큽니다. /16 이하로 직접 지정하세요")
print(f"{dev} {net} 의 {net.num_addresses}개 주소에서 {PORT}번 포트를 찾는 중... (최대 1~2분)")

with ThreadPoolExecutor(max_workers=512) as ex:              # 512개를 동시에 검사
    found = [ip for ip in ex.map(probe, net.hosts()) if ip]  # hosts(): 대역 안의 기기 주소들

neigh = subprocess.run(["ip", "neigh"], capture_output=True, text=True).stdout   # 연결해 본 기기들의 IP-MAC 표
macs = {l.split()[0]: l.split()[4] for l in neigh.splitlines() if "lladdr" in l}  # {IP: MAC}
if not found:
    print("554번 포트가 열린 기기가 없습니다 (카메라가 이 네트워크에 없거나 막혀 있음)")
for ip in found:
    print(f"  {ip:15s}  MAC {macs.get(ip, '?')}")         # :15s = 15칸에 맞춰 출력
