# make_sounds.py - README 5단계 테스트용 경보음(beep.wav)과 잡음(noise.wav) 만들기
# 목적: README의  ffmpeg -f lavfi ...  명령 대신 파이썬으로 같은 소리를 만든다.
# 실행: ~/eg2-lab 에서 venv를 켠 뒤  python make_sounds.py   → media/beep.wav, media/noise.wav 생성
#
# 수정 기록
# 2026-10-23  v1  새로 작성. 이 Jetson의 NVIDIA판 FFmpeg는 lavfi(소리 생성 기능)를 빼고 빌드돼
#                 'Unknown input format: lavfi' 오류가 나서. numpy·soundfile은 venv에 이미 있어 설치할 것이 없다.

import numpy as np            # 숫자 배열 계산 패키지. 소리는 '시간 순서대로 늘어선 숫자 배열'이다
import soundfile as sf        # 숫자 배열을 .wav 파일로 저장하는 패키지

SR = 16000                    # 샘플레이트: 1초에 숫자 16000개 (모델이 요구하는 형식)

# ── 1. 경보음: 1 kHz 사인파 3초 ──────────────────────────────
# np.arange(N) : 0, 1, 2, ..., N-1 정수 배열.  / SR 로 나누면 각 숫자가 '몇 초 지점'인지가 된다
t = np.arange(3 * SR) / SR    # 3초 → 48000개 시각
# sin(2π × 주파수 × 시각) : 1초에 1000번 오르내리는 파형 = '삐-' 소리. 0.5를 곱해 소리 크기를 절반으로(찢어짐 방지)
beep = 0.5 * np.sin(2 * np.pi * 1000 * t)

# ── 2. 잡음: 핑크 노이즈 5초 ────────────────────────────────
# 흰 잡음(모든 높이의 소리가 같은 세기)을 만든 뒤, 높은 소리일수록 약하게 해서 핑크 잡음(빗소리·웅웅 소리)으로 바꾼다
rng = np.random.default_rng(0)               # 난수 생성기. 0을 주면 실행할 때마다 같은 잡음이 나온다
white = rng.standard_normal(5 * SR)          # 5초 분량 흰 잡음
spec = np.fft.rfft(white)                    # 소리 → 주파수별 성분 (FFT)
f = np.fft.rfftfreq(len(white), 1 / SR)      # 각 성분의 주파수(Hz)
f[0] = 1                                     # 0 Hz로 나누는 것을 막으려고 1로 바꿈
pink = np.fft.irfft(spec / np.sqrt(f), n=len(white))   # 세기를 1/√주파수 로 줄인 뒤 다시 소리로 (핑크 잡음의 정의)
pink = 0.5 * pink / np.abs(pink).max()       # 가장 큰 값이 0.5가 되게 크기를 맞춤

# ── 3. 저장 ─────────────────────────────────────────────────
# subtype="PCM_16" : 16비트 정수 wav (일반적인 wav 형식). 배열이 1차원이라 자동으로 모노(채널 1개)
sf.write("media/beep.wav", beep, SR, subtype="PCM_16")
sf.write("media/noise.wav", pink, SR, subtype="PCM_16")
print("media/beep.wav (3초), media/noise.wav (5초) 생성 완료: 16000 Hz 모노")
