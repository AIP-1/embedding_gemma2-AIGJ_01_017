# ptz.py - IP 카메라(VIGI C540-W) 방향을 명령으로 돌린다 (ONVIF 표준 제어)
# 목적: 회전형(PTZ) 카메라는 손으로 돌리면 기어가 상할 수 있고, VIGI 앱 목록에 카메라가 없어 앱으로도 못 돌린다.
#       ONVIF(카메라 제어 표준, 포트 2020)로 "왼쪽으로 0.5초 돌기" 같은 명령을 보낸다.
# 실행: ~/eg2-lab/prox 에서  python ptz.py left 0.5     (방향: left right up down, 숫자 = 도는 시간 초)
#       preview_cams.py 화면을 보면서 조금씩 돌린다. 비밀번호는 실행하면 물어본다(화면에 안 보임)
# 주의: 카메라 방향만 바꾼다. 10-2 보정 후에는 절대 쓰지 않는다(돌리면 보정을 다시 해야 함)
#
# 수정 기록
# 2026-10-28  v1  새로 작성. 지원 님 IP 카메라 방향을 맞추기 위해 (웹 설정 화면에 회전 버튼이 없을 경우 대비).
#                 Jetson 밖 인터넷이 없어 카메라 시계가 틀릴 수 있으므로, 인증 시각을 카메라 시계에 맞춰 보낸다

import argparse, base64, datetime, getpass, hashlib, os, re, time   # 옵션 / 인증 계산용 / 비밀번호 입력 / 정규식
import urllib.request                                 # 파이썬 기본 HTTP 요청 (추가 설치 없음)

ap = argparse.ArgumentParser()
ap.add_argument("dir", choices=["left", "right", "up", "down"])   # 돌릴 방향
ap.add_argument("sec", type=float, nargs="?", default=0.3)        # 도는 시간(초). 짧게 여러 번이 안전
ap.add_argument("--ip", default="10.42.0.22")
ap.add_argument("--port", type=int, default=2020)                 # TP-Link 카메라의 ONVIF 포트
ap.add_argument("--user", default="admin")
ap.add_argument("--speed", type=float, default=0.3)               # 0~1. 작을수록 천천히
args = ap.parse_args()
PW = getpass.getpass("카메라 비밀번호: ")                          # 입력해도 화면에 안 보임
URL = f"http://{args.ip}:{args.port}/onvif/service"               # TP-Link 카메라의 ONVIF 주소(추정, 안 되면 device_service)


def post(body, auth=True):
    """SOAP(XML) 요청을 보내고 응답 문자열을 돌려준다. auth=True면 비밀번호 인증 헤더를 붙인다"""
    header = ""
    if auth:
        created = (datetime.datetime.utcnow() + OFFSET).strftime("%Y-%m-%dT%H:%M:%SZ")   # 카메라 시계 기준 현재 시각
        nonce = os.urandom(16)                                    # 매번 다른 무작위 값 (재사용 공격 방지)
        digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + PW.encode()).digest()).decode()
        header = f"""<s:Header><Security xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
<UsernameToken><Username>{args.user}</Username>
<Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</Password>
<Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</Nonce>
<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created>
</UsernameToken></Security></s:Header>"""
    xml = f"""<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">{header}<s:Body>{body}</s:Body></s:Envelope>"""
    req = urllib.request.Request(URL, xml.encode(), {"Content-Type": "application/soap+xml; charset=utf-8"})
    try:
        return urllib.request.urlopen(req, timeout=5).read().decode()
    except urllib.error.HTTPError as e:                           # 401 등: 응답 내용을 같이 보여 준다
        raise SystemExit(f"카메라가 거절함 ({e.code}). 비밀번호를 확인하세요\n{e.read().decode()[:300]}")


# 1) 카메라 시계 읽기 (인증 없이 가능). 카메라 시계가 틀려도 인증이 통과하도록 차이(OFFSET)를 구해 둔다
OFFSET = datetime.timedelta(0)
r = post("<tds:GetSystemDateAndTime/>", auth=False)
v = {k: int(re.search(rf"<tt:{k}>(\d+)</tt:{k}>", r.split("UTCDateTime")[1]).group(1))
     for k in ("Year", "Month", "Day", "Hour", "Minute", "Second")}            # UTC 시각 칸에서 숫자만 뽑기
cam_now = datetime.datetime(v["Year"], v["Month"], v["Day"], v["Hour"], v["Minute"], v["Second"])
OFFSET = cam_now - datetime.datetime.utcnow()

# 2) 영상 프로필 이름(token) 얻기. PTZ 명령은 어느 프로필에 대한 것인지 알려 줘야 한다
r = post("<trt:GetProfiles/>")
token = re.search(r'Profiles[^>]*token="([^"]+)"', r).group(1)

# 3) 방향 속도를 정해 ContinuousMove(계속 돌기) → sec초 후 Stop
sp = args.speed
x, y = {"left": (-sp, 0), "right": (sp, 0), "up": (0, sp), "down": (0, -sp)}[args.dir]
post(f'<tptz:ContinuousMove><tptz:ProfileToken>{token}</tptz:ProfileToken>'
     f'<tptz:Velocity><tt:PanTilt x="{x}" y="{y}"/></tptz:Velocity></tptz:ContinuousMove>')
time.sleep(args.sec)
post(f'<tptz:Stop><tptz:ProfileToken>{token}</tptz:ProfileToken><tptz:PanTilt>true</tptz:PanTilt></tptz:Stop>')
print(f"{args.dir} {args.sec}초 이동 완료")
