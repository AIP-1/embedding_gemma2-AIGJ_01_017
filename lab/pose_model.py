# pose_model.py - 9단계 공통 모듈: 관절 특징 계산 + GRU 동작 분류 모델 (README 9단계)
# 목적: pose_app.py(실시간 앱)와 train_pose.py(9-6 학습)가 "같은 계산"을 쓰도록 한곳에 모아 둔다.
#       특징을 만드는 방법이 학습 때와 실행 때 조금이라도 다르면 모델이 엉뚱하게 판정하기 때문이다.
# 실행: 직접 실행하지 않는다. 같은 폴더(~/eg2-lab)에 두면 다른 .py가 import 해서 쓴다.
#
# 수정 기록
# 2026-10-27  v1  README 9단계 pose_model.py를 그대로 옮기고 줄별 설명 주석 추가 (지원 님 요청: 파이썬 복습용).
#                 torch.load에 weights_only=True를 명시. 저장 파일에 텐서·숫자·글자만 있어 그대로 읽히고,
#                 남이 준 .pt 파일 안의 코드가 실행되는 것을 막는 안전한 방식이기 때문 (동작은 같음)

import numpy as np              # 관절 좌표 배열 계산
import torch                    # 신경망(GRU) 계산
import torch.nn as nn           # nn : 신경망 층(GRU, Linear 등)을 모아 둔 모듈. 줄여서 nn 으로 부른다

KP_CONF = 0.5                   # 관절 신뢰도 기준. 0.5 미만인 관절은 "안 보임"으로 취급
# COCO 17 관절 번호: 0 코, 1/2 눈(왼/오), 3/4 귀, 5/6 어깨, 7/8 팔꿈치, 9/10 손목, 11/12 엉덩이, 13/14 무릎, 15/16 발목
# 왼쪽/오른쪽은 "사진 속 사람" 기준이다 (카메라를 보고 서 있으면 사람의 왼쪽 = 화면의 오른쪽)
# 한 줄에 여러 변수 = 여러 값 : 이름과 번호를 순서대로 짝지어 넣는다. 이후 k[LW] 처럼 이름으로 관절을 가리킨다
NOSE, EYE_L, EYE_R, EAR_L, EAR_R, LS, RS, LW, RW, LH, RH = 0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12
N_FEAT = 34 + 17 + 2            # 한 프레임 특징 수 53 = 정규화 좌표(17관절 × x,y) 34 + 관절 신뢰도 17 + 정면 점수 + 이동 속도


def body(k):
    """어깨 중심 좌표와 어깨 너비를 돌려준다. 어깨가 안 보이면 None.
    k : (17, 3) 배열. 관절마다 [x, y, 신뢰도]. x는 오른쪽으로, y는 아래쪽으로 커진다(픽셀)."""
    if k[LS, 2] < KP_CONF or k[RS, 2] < KP_CONF:     # k[LS, 2] : 왼어깨 행의 2번 열(신뢰도)
        return None                                  # 어깨 하나라도 안 보이면 기준을 못 잡으므로 포기
    c = (k[LS, :2] + k[RS, :2]) / 2                  # :2 → 0,1번 열(x, y)만. 두 어깨의 가운데 점
    # 옆으로 서면 두 어깨 x가 거의 같아져 너비가 0에 가까워진다 → 나눗셈이 폭주하지 않게 최소 20픽셀
    sw = max(abs(k[LS, 0] - k[RS, 0]), 20.0)
    return c, sw                                     # 튜플(묶음)로 두 값을 함께 돌려준다


def facing_score(k):
    """카메라를 정면으로 보는 정도 (0~1, 1이 정면).
    코·두 눈이 보이고, 코가 두 귀(귀가 안 보이면 두 눈)의 한가운데 있을수록 정면이다."""
    if k[NOSE, 2] < KP_CONF or k[EYE_L, 2] < KP_CONF or k[EYE_R, 2] < KP_CONF:
        return 0.0                                   # 얼굴이 안 보이면(뒤돌아섬 등) 정면 아님
    if k[EAR_L, 2] > KP_CONF and k[EAR_R, 2] > KP_CONF:
        l, r = k[EAR_L, 0], k[EAR_R, 0]              # 두 귀의 x 좌표 (폭이 넓어 더 정확)
    else:
        l, r = k[EYE_L, 0], k[EYE_R, 0]              # 귀가 가려지면 두 눈의 x 좌표로 대신
    # yaw(고개 돌림 정도) = 코가 가운데에서 벗어난 거리 ÷ 두 점 사이 폭. 0이면 정면, 0.5면 완전히 옆
    yaw = abs(k[NOSE, 0] - (l + r) / 2) / max(abs(l - r), 1.0)   # max(..., 1.0) : 0으로 나누기 방지
    return float(max(0.0, 1 - 2 * yaw))              # yaw 0 → 1점, yaw 0.5 이상 → 0점. float()로 파이썬 숫자로


def frame_features(k, b, facing, speed):
    """한 프레임의 특징 벡터 (53,) 를 만든다. 학습(9-6)과 실행이 모두 이 함수 결과를 쓴다."""
    c, sw = b                                        # body() 결과를 두 변수로 풀기
    # 모든 관절 좌표에서 어깨 중심을 빼고(원점 이동) 어깨 너비로 나눈다(크기 맞춤) → 거리·위치가 달라도 같은 값
    # flatten() : (17, 2) 2차원 배열을 34개짜리 1줄로 편다 [x0, y0, x1, y1, ...]
    norm = ((k[:, :2] - c) / sw).flatten()
    # concatenate : 여러 배열을 이어 붙여 하나로. 순서 = 좌표 34 + 신뢰도 17 + [정면, 속도] 2
    # astype(np.float32) : 32비트 실수로 (GPU 신경망이 쓰는 형식)
    return np.concatenate([norm, k[:, 2], [facing, speed]]).astype(np.float32)


class PoseGRU(nn.Module):
    """시간 순서대로 들어온 특징 벡터 묶음(윈도)을 읽고 라벨 하나를 고르는 작은 신경망.
    class ... (nn.Module) : PyTorch 신경망은 nn.Module을 물려받아(상속) 만든다."""

    def __init__(self, n_feat, n_cls, hidden=64):    # __init__ : 객체를 만들 때 한 번 실행되는 준비 함수
        super().__init__()                           # 부모(nn.Module)의 준비 작업을 먼저 실행
        # GRU : 프레임을 하나씩 읽으며 '기억(hidden, 64개 숫자)'을 갱신한다.
        # batch_first=True : 입력 모양을 (묶음 크기 B, 시간 T, 특징 F) 순서로 받겠다는 뜻
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_cls)           # 마지막 기억 64개 → 라벨별 점수 n_cls개 (선형 층)

    def forward(self, x):                            # forward : 모델에 입력을 넣었을 때 실제 계산. x: (B, T, F)
        _, h = self.gru(x)                           # GRU는 (매 시점 출력, 마지막 기억)을 준다. 앞은 안 쓰니 _
        return self.fc(h[-1])                        # h[-1] : 마지막 층의 마지막 기억 (B, hidden) → (B, n_cls) 점수


def load_model(path, device="cuda"):
    """train_pose.py가 저장한 .pt 파일을 읽어 바로 쓸 수 있는 사전(dict)으로 돌려준다."""
    # map_location=device : 저장할 때 장치와 상관없이 지금 장치(GPU)로 올린다
    # weights_only=True : 텐서·숫자·글자만 읽는다 (v1에서 명시, 파일 맨 위 수정 기록 참고)
    ck = torch.load(path, map_location=device, weights_only=True)
    m = PoseGRU(N_FEAT, len(ck["labels"]), ck["hidden"]).to(device)   # 같은 구조의 빈 모델을 만들고
    m.load_state_dict(ck["state"])                   # 학습된 가중치를 채워 넣는다
    m.eval()                                         # 평가 모드 (학습용 동작을 끈다)
    ck["model"] = m                                  # 사전에 모델 객체도 함께 넣어 둔다
    # 학습 때 쓴 평균·표준편차. 실행 때도 같은 값으로 특징을 표준화해야 학습 때와 같은 범위가 된다
    ck["mean"] = torch.tensor(ck["mean"], device=device)
    ck["std"] = torch.tensor(ck["std"], device=device)
    return ck                                        # 키: state, labels, task, window, hidden, mean, std, model


@torch.inference_mode()                              # 데코레이터: 이 함수 안에서는 기울기 계산을 꺼 빠르고 메모리 절약
def predict(ck, seq, device="cuda"):
    """seq : (T, 53) 특징 배열 (윈도 하나) → (가장 높은 라벨, 그 확률)"""
    x = torch.tensor(np.asarray(seq), device=device)[None]   # [None] : 맨 앞에 차원 추가 → (1, T, 53), 묶음 크기 1
    x = (x - ck["mean"]) / ck["std"]                 # 표준화: 학습 때와 같은 기준으로 맞추기
    p = torch.softmax(ck["model"](x), -1)[0]         # 점수 → 확률(합 1). [0] : 묶음의 첫(유일한) 결과
    i = int(p.argmax())                              # 확률이 가장 큰 라벨 번호
    return ck["labels"][i], float(p[i])              # (라벨 이름, 확률)
