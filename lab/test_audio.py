# test_audio.py - README 5단계 음성 테스트
# 목적: 오디오 인코더를 켠 모델로 '글로 쓴 질의'가 알맞은 소리 파일을 찾는지 확인한다.
#       4단계에서 사진 자리에 있던 것을 소리(.wav)로 바꾼 것이다. 글과 소리도 같은 768차원 벡터 공간에 들어간다.
# 실행: ~/eg2-lab 에서 venv를 켠 뒤  python test_audio.py
# 준비: ~/eg2-lab/media/ 에 speech.wav(말소리), beep.wav(경보음), noise.wav(잡음). mic.wav(직접 녹음)는 있으면 함께 쓴다.
#
# 수정 기록
# 2026-10-23  v1  README 5단계 코드를 바탕으로 작성. 바꾼 점 4가지 (실습 중 막히지 않게 하려고)
#                 1) 16 kHz 모노가 아니면 assert로 멈추는 대신, librosa로 메모리 안에서 16 kHz 모노로 바꿔 넣음
#                    (맥 say 음성은 22 kHz 등 다른 형식이라 매번 ffmpeg로 다시 변환하는 수고를 덜려고)
#                 2) 파일 경로로 넣는 기본 방식이 실패하면 소리 배열(숫자 목록)로 직접 넣는 대체 경로 추가
#                    (4단계에서 torchcodec이 이 Jetson의 NVIDIA판 FFmpeg와 안 맞아 실패했기 때문. 장비 FFmpeg는 건드리지 않음)
#                 3) 오디오 임베딩을 두 번 재서 두 번째 값을 따로 출력 (6단계 표는 '두 번째 실행 값'을 적으라고 해서)
#                 4) 질의마다 1위만이 아니라 전체 순위와 GPU 메모리 최대 사용량 출력 (4단계 v2와 같은 형식, 6단계 '+오디오' 행 기록용)
# 2026-10-23  v2  1) 질의 2개 추가: '사람이 말하는 목소리', '에어컨을 켜 달라는 말'
#                    (v1에서 '덥다고 냉방을 요청하는 목소리'가 noise 0.651 / speech 0.647로 근소하게 틀려서,
#                     모델이 '말소리인지'를 못 가리는 건지 '말의 내용'을 못 알아듣는 건지 나눠 보려고)
#                 2) 기본 방식이 실패하면 그 사실을 기억해 2회차에는 바로 배열 방식으로 감
#                    (v1은 실패 메시지가 두 번 찍히고, 2회차 시간에 실패한 시도 시간이 섞였기 때문)

# ── 4단계와 달라진 점 ────────────────────────────────────────
# 4단계: {"image": 경로} 로 사진을 넣었다.  5단계: {"audio": ...} 로 소리를 넣는다.
# 모델은 소리를 '16 kHz 모노'로만 읽도록 학습됐다.
#   16 kHz = 1초를 16000개의 숫자(샘플)로 기록,  모노 = 채널 1개(스테레오는 2개)
# 형식이 다르면 소리가 빨라지거나 느려진 것처럼 읽혀 결과가 틀어지므로, 넣기 전에 맞춘다.
# ────────────────────────────────────────────────────────────

import glob                   # media/*.wav 같은 패턴으로 파일 목록을 얻는다 (4단계와 같음)
import time                   # 걸린 시간 측정
import torch                  # GPU 메모리 사용량을 읽으려고 불러온다
import soundfile as sf        # wav 파일의 정보(샘플레이트·채널 수)를 읽는 패키지. 'as sf' : 짧은 별명으로 부르겠다는 뜻
import librosa                # 소리를 읽고 샘플레이트를 바꾸는(리샘플링) 패키지. user_setup.sh가 venv에 설치해 둠
from eg2 import load          # 같은 폴더 eg2.py 의 공통 로더

SR = 16000                    # 모델이 요구하는 샘플레이트(Hz). 대문자 = 바꾸지 않을 설정값

# load(audio=True) : 오디오 인코더까지 올린다 (텍스트 270M → 약 570M 파라미터).
# vision은 기본값 False라 비전 인코더는 올리지 않는다 → 4단계보다 메모리를 더 쓰는지 비교할 수 있다.
model = load(audio=True)

# ── 1. 파일 목록 ────────────────────────────────────────────
# "/raw_" 가 들어간 파일은 변환 전 원본이라 뺀다 (README 샘플 명령이 raw_*.wav 를 만들기 때문)
# f for f in ... if 조건 : 조건이 참인 것만 골라 담는 컴프리헨션
files = sorted(f for f in glob.glob("media/*.wav") if "/raw_" not in f)
if len(files) == 0:
    print("media/ 폴더에 .wav 파일이 없습니다. README 5단계 샘플 명령으로 먼저 만드세요.")
    raise SystemExit

# ── 2. 형식 확인 + 16 kHz 모노 배열 만들기 ───────────────────
all_ok = True                 # 모든 파일이 처음부터 16 kHz 모노인지 기록 (기본 방식을 써도 되는지 판단용)
arrays = []                   # 16 kHz 모노로 맞춘 소리 배열들을 담을 리스트
for f in files:
    info = sf.info(f)         # 파일을 다 읽지 않고 머리 부분 정보만 읽는다
    ok = info.samplerate == SR and info.channels == 1   # 두 조건이 모두 참이면 True
    if not ok:
        all_ok = False
    # librosa.load(경로, sr=16000, mono=True) : 읽으면서 16 kHz 모노로 바꿔 준다.
    #   돌려주는 값이 2개라 'y, _ =' 로 받는다. y = 소리 배열(실수 목록), _ = 샘플레이트(이미 아는 값이라 버림)
    y, _ = librosa.load(f, sr=SR, mono=True)
    arrays.append(y)
    # 길이(초) = 샘플 개수 / 1초당 샘플 수.  '원본 형식 → 넣는 형식'을 한 줄로 보여 준다
    note = "그대로" if ok else "→ 16000 Hz 모노로 변환"   # A if 조건 else B : 조건에 따라 A나 B
    print(f"{f.split('/')[-1]:<12} {info.samplerate} Hz, {info.channels}채널, {len(y)/SR:.1f}초  {note}")


# ── 3. 오디오 임베딩 (넣는 방식 3가지를 순서대로 시도) ─────────
path_failed = False           # 기본 방식(파일 경로)이 한 번 실패했는지 기억. 실패했으면 2회차에는 시도하지 않는다


def encode_audio():
    """소리 파일들을 임베딩해 (결과, 방식 이름)을 돌려준다. 앞 방식이 실패하면 다음 방식으로 넘어간다."""
    global path_failed        # global : 함수 밖의 변수 path_failed 값을 함수 안에서 바꾸겠다는 선언 (없으면 새 지역 변수가 된다)
    # 1차: README 방식. 파일 경로를 넣으면 모델 쪽에서 파일을 읽는다 (내부에서 torchcodec을 쓸 수 있음).
    #      형식이 다른 파일이 섞여 있으면 결과가 틀어질 수 있으니 전부 16 kHz 모노일 때만 쓴다.
    if all_ok and not path_failed:
        try:
            emb = model.encode([{"audio": f} for f in files], normalize_embeddings=True)
            return emb, "파일 경로(기본)"
        except Exception as e:
            print(f"기본 방식 실패 ({type(e).__name__}: {str(e).splitlines()[0]}) → 배열로 다시 시도")
            path_failed = True
    # 2차: 우리가 librosa로 읽은 배열을 직접 넣는다. 파일 읽기를 모델에 맡기지 않으므로 torchcodec이 필요 없다.
    try:
        emb = model.encode([{"audio": y} for y in arrays], normalize_embeddings=True)
        return emb, "librosa 배열"
    except Exception as e:
        print(f"배열 입력 실패 ({type(e).__name__}: {str(e).splitlines()[0]}) → 샘플레이트를 함께 넘겨 다시 시도")
    # 3차: 배열과 샘플레이트를 사전 하나로 묶어 넣는다 (Hugging Face 데이터셋이 쓰는 형식)
    emb = model.encode([{"audio": {"array": y, "sampling_rate": SR}} for y in arrays],
                       normalize_embeddings=True)
    return emb, "librosa 배열+샘플레이트"


# 같은 작업을 두 번 잰다. 첫 번째는 GPU 준비(커널 로딩 등) 시간이 섞여 느리다.
t = time.perf_counter()
aud_emb, how = encode_audio()           # 함수가 값 2개를 돌려주므로 변수 2개로 받는다
ms1 = (time.perf_counter() - t) * 1000
t = time.perf_counter()
aud_emb, how = encode_audio()
ms2 = (time.perf_counter() - t) * 1000
total_sec = sum(len(y) for y in arrays) / SR   # 전체 소리 길이(초). sum(...) : 합계
print(f"\n오디오 {len(files)}개 ({total_sec:.1f}초 분량): 1회차 {ms1:.0f} ms, 2회차 {ms2:.0f} ms  [방식: {how}]")

# ── 4. 질의 임베딩 ──────────────────────────────────────────
# 질의마다 정답 파일이 하나씩 정해져 있다. mic.wav를 넣었다면 내용에 맞는 질의를 하나 추가해 봐도 된다.
queries = [
    "덥다고 냉방을 요청하는 목소리",   # 정답: speech.wav
    "날카로운 경보음",                 # 정답: beep.wav
    "웅웅거리는 배경 소음",            # 정답: noise.wav
    "사람이 말하는 목소리",            # 정답: speech.wav  (말소리인지 아닌지만 가리면 맞힌다)
    "에어컨을 켜 달라는 말",           # 정답: speech.wav  (말의 '내용'을 알아들어야 맞힌다)
]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)

# ── 5. 질의 → 소리 검색 ─────────────────────────────────────
# q(질의 5×768) @ aud_emb.T(768×파일수) = 질의별·파일별 유사도 표 (3·4단계와 같은 계산)
sim = q @ aud_emb.T
print()
for qi, row in zip(queries, sim):
    order = row.argsort()[::-1]        # 유사도가 높은 순서의 파일 번호
    top = [(files[i].split('/')[-1], round(float(row[i]), 3)) for i in order]
    print(f"{qi}\n   -> {top}")        # 맨 앞이 1위

# ── 6. 메모리 (6단계 측정 표 '+오디오' 행) ───────────────────
# 4단계 '+비전' 1.20 GB와 비교한다. tegrastats 의 RAM 값도 함께 적는다.
print(f"\nGPU 메모리 최대 사용: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
