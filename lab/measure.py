# measure.py - README 6단계 성능 측정
# 목적: 3~5단계에서 따로따로 잰 값을 같은 조건으로 다시 재서 README 6단계 표 한 장을 채운다.
#       구성(텍스트 전용 / +비전 / +오디오 / 전체)마다 '로드 후 RAM, 지연(2회차), GPU 사용률, GPU 메모리'를 잰다.
# 실행: ~/eg2-lab 에서 venv를 켠 뒤  python measure.py            → 4가지 구성을 차례로 모두 잼 (5~10분)
#                                   python measure.py audio      → 한 구성만 (text / vision / audio / full)
# 준비: 4·5단계 media/ 폴더 그대로 (사진, clip.mp4, speech/beep/noise.wav). 새로 설치할 것 없음.
# 결과: 화면에 표를 찍고, 같은 내용을 ~/eg2-lab/results_step6.md 로 저장한다 (보고서에 붙여 넣기용).
#
# 수정 기록
# 2026-10-23  v1  새로 작성. 3~5단계 값은 재는 방법이 제각각이라(텍스트는 GPU 메모리 없음, 사진은 첫 호출 포함)
#                 같은 방법으로 다시 재려고. 구성마다 새 프로세스에서 로드해 앞 구성의 메모리가 섞이지 않게 함.

# ── 이 파일이 재는 것 ─────────────────────────────────────────
# 1) 로드 후 RAM : 장비 전체에서 쓰는 메모리(MB). tegrastats 의 'RAM 사용/전체'와 같은 값.
#    Jetson은 CPU와 GPU가 8GB를 함께 쓰므로 GPU에 올린 모델도 여기 잡힌다.
#    로드 전 값도 재서 '모델이 늘린 양'을 따로 본다 (다른 학생이 장비를 쓰고 있으면 전체 값이 흔들리기 때문).
# 2) 지연 : 같은 입력을 두 번 encode 해서 두 번째 값 (README 지시. 첫 번째는 GPU 준비 시간이 섞임).
# 3) GPU 사용률 : 두 번째 실행 동안 GPU 바쁨 정도(%)를 0.02초마다 읽어 평균·최대를 낸다. tegrastats 의 GR3D_FREQ 와 같은 값.
# 4) GPU 메모리 최대 : torch 가 GPU에서 가장 많이 잡았던 순간 (4·5단계와 같은 방식).
# ────────────────────────────────────────────────────────────

import glob                   # 파일 이름 패턴으로 목록 얻기 (4·5단계와 같음)
import json                   # 결과를 한 줄 문자열로 바꿔 부모 프로세스에 넘길 때 쓴다
import os                     # 파일 존재 확인
import subprocess             # 이 파일을 '새 프로세스'로 다시 실행할 때 쓰는 기본 패키지
import sys                    # 실행할 때 넘긴 인자(sys.argv)와 지금 쓰는 python 경로(sys.executable)
import threading              # GPU 사용률을 재는 일을 encode 와 '동시에' 돌리려고 (스레드 = 한 프로그램 안의 작은 일꾼)
import time                   # 시간 측정

CONFIGS = ["text", "vision", "audio", "full"]   # 잴 구성 4가지. README 표의 행 순서


# ── 측정 도구 1: 장비 메모리 ──────────────────────────────────
def ram_used_mb():
    """장비 전체 사용 메모리(MB) = 전체 - 쓸 수 있는 양. /proc/meminfo 는 리눅스가 메모리 상태를 적어 두는 파일."""
    info = {}                                    # 빈 사전. '항목 이름 → 값(kB)'을 담는다
    with open("/proc/meminfo") as f:             # 읽기 전용으로 열기. 장비에 아무 영향 없음
        for line in f:                           # 한 줄씩. 예: "MemTotal:  7843456 kB"
            name, value = line.split(":")        # ':' 앞뒤로 쪼개 두 변수에 나눠 담는다
            info[name] = int(value.split()[0])   # " 7843456 kB" → 공백으로 쪼갠 첫 조각 → 정수
    return (info["MemTotal"] - info["MemAvailable"]) // 1024   # kB → MB.  // : 나눗셈 후 소수점 버림


# ── 측정 도구 2: GPU 사용률 ───────────────────────────────────
# Jetson GPU는 바쁨 정도를 sysfs(커널이 장치 상태를 보여 주는 가짜 파일들)에 0~1000으로 적는다 (1000 = 100%).
# 경로가 L4T 버전마다 달라 후보를 여러 개 두고 처음 찾은 것을 쓴다. 읽기만 하므로 sudo 불필요, 장비에 영향 없음.
GPU_LOAD_CANDIDATES = (
    glob.glob("/sys/devices/platform/bus@0/17000000.gpu/load")     # JetPack 6 Orin
    + glob.glob("/sys/devices/platform/17000000.gpu/load")
    + glob.glob("/sys/devices/platform/*/*.gpu/load")
    + glob.glob("/sys/class/devfreq/*gpu*/device/load")
    + glob.glob("/sys/devices/gpu.0/load")
)
GPU_LOAD_FILE = GPU_LOAD_CANDIDATES[0] if GPU_LOAD_CANDIDATES else None   # 하나도 없으면 None


class GpuSampler:
    """with GpuSampler() as g: 블록이 도는 동안 GPU 사용률을 계속 읽어 g.samples 에 모은다."""

    def __enter__(self):                         # with 블록에 들어갈 때 자동으로 불림
        self.samples = []                        # 읽은 값(%)을 담을 리스트
        self.running = True
        self.thread = threading.Thread(target=self._loop)   # _loop 함수를 따로 돌릴 일꾼 만들기
        self.thread.start()                      # 일꾼 출발. 이 줄 다음부터 encode 와 동시에 돈다
        return self

    def _loop(self):
        while self.running and GPU_LOAD_FILE:    # 경로를 못 찾았으면 바로 끝
            try:
                with open(GPU_LOAD_FILE) as f:
                    self.samples.append(int(f.read()) / 10)   # 0~1000 → 0~100 %
            except OSError:                      # 읽기 권한이 없는 장비면 포기
                return
            time.sleep(0.02)                     # 0.02초(20 ms)마다. 텍스트 질의는 0.2초 안팎이라 촘촘히 읽는다

    def __exit__(self, *args):                   # with 블록을 나갈 때 자동으로 불림
        self.running = False                     # 일꾼에게 멈추라고 알리고
        self.thread.join()                       # 멈출 때까지 기다림

    def summary(self):
        """'평균 / 최대' 문자열. 값이 없으면 tegrastats 로 보라고 안내."""
        if not self.samples:
            return "못 읽음(tegrastats)"
        return f"{sum(self.samples) / len(self.samples):.0f} / {max(self.samples):.0f}"


# ── 측정 도구 3: 같은 일을 두 번 하고 2회차를 잼 ────────────────
def run_twice(fn):
    """fn()을 두 번 실행. (1회차 ms, 2회차 ms, 2회차 GPU 사용률 문자열)을 돌려준다."""
    t = time.perf_counter()
    fn()                                         # 1회차: GPU 커널 준비 등이 섞여 느리다
    ms1 = (time.perf_counter() - t) * 1000
    with GpuSampler() as g:                      # 2회차 동안만 GPU 사용률을 읽는다
        t = time.perf_counter()
        fn()                                     # encode 는 결과를 numpy 로 돌려주므로 GPU 계산이 끝난 뒤에 반환된다
        ms2 = (time.perf_counter() - t) * 1000
    return ms1, ms2, g.summary()


# ── 입력 준비 (3~5단계와 같은 데이터) ──────────────────────────
QUERIES = ["수업 중에 졸리고 머리가 맑지 않아요", "여름에 냉방 몇 도로 맞춰야 하나요?"]   # 3단계 질의 2개
VIDEO = "media/clip.mp4"


def read_frames_1fps(path):
    """PyAV로 1초에 1장씩 프레임을 꺼낸다 (test_vision.py v3와 같은 방법. 이 장비는 torchcodec이 안 되므로)."""
    import av
    frames, next_t = [], 0.0                     # 변수 두 개를 한 줄에 만들기
    with av.open(path) as container:
        for frame in container.decode(video=0):
            if frame.time is not None and frame.time >= next_t:
                frames.append(frame.to_image())
                next_t += 1.0
    return frames


def load_audio_arrays():
    """wav 파일들을 16 kHz 모노 배열로 읽는다 (test_audio.py v2와 같은 방법). (배열 리스트, 총 길이 초)"""
    import librosa
    files = sorted(f for f in glob.glob("media/*.wav") if "/raw_" not in f)
    arrays = [librosa.load(f, sr=16000, mono=True)[0] for f in files]   # [0] : (배열, 샘플레이트) 중 배열만
    return arrays, sum(len(y) for y in arrays) / 16000


# ── 구성 하나 재기 (자식 프로세스에서 실행) ─────────────────────
def measure_one(name):
    """구성 하나를 로드하고 잰 뒤, 표의 행들을 리스트로 돌려준다."""
    ram_before = ram_used_mb()                   # 모델을 올리기 전 장비 메모리
    import torch                                 # torch·모델은 여기서야 불러온다 (로드 전 RAM에 안 섞이게)
    from eg2 import load

    vision = name in ("vision", "full")          # in : 오른쪽 묶음 안에 있으면 True
    audio = name in ("audio", "full")
    t = time.perf_counter()
    model = load(vision=vision, audio=audio)     # 3~5단계와 같은 공통 로더
    load_s = time.perf_counter() - t
    model.encode(["준비"])                       # 짧은 문장 하나로 GPU를 한 번 깨워 둔다 (모든 구성 똑같이)
    ram_after = ram_used_mb()                    # 로드 후 장비 메모리
    torch.cuda.reset_peak_memory_stats()         # GPU 메모리 최대값을 0부터 다시 기록 (입력별로 비교하려고)

    # 입력별로 '한 번 encode 하는 함수'를 만든다. lambda : 이름 없는 한 줄 함수
    enc_q = lambda: model.encode(QUERIES, prompt_name="SearchQuery", normalize_embeddings=True)
    jobs = []                                    # (입력 설명, 함수) 목록
    if name == "text":
        jobs.append(("질의 2개", enc_q))
    if vision:
        images = sorted(glob.glob("media/*.jpg") + glob.glob("media/*.jpeg") + glob.glob("media/*.png"))
        frames = read_frames_1fps(VIDEO) if os.path.exists(VIDEO) else []   # 동영상 파일 읽기는 시간에서 뺀다
        enc_img = lambda: model.encode([{"image": p} for p in images], normalize_embeddings=True)
        def enc_vid():
            # 4단계처럼 프레임 리스트를 '동영상'으로 넣고, 안 되면 프레임마다 사진으로 넣는다 (test_vision.py v3의 2·3차 방식)
            try:
                return model.encode({"video": frames}, normalize_embeddings=True)
            except Exception:
                return model.encode([{"image": f} for f in frames], normalize_embeddings=True)
    if audio:
        arrays, audio_sec = load_audio_arrays()  # 파일 읽기·변환도 시간에서 뺀다 (모델 속도만 보려고)
        enc_aud = lambda: model.encode([{"audio": y} for y in arrays], normalize_embeddings=True)
    if name == "vision":
        jobs.append((f"이미지 {len(images)}장", enc_img))
        if frames:
            jobs.append((f"동영상 {len(frames)}초({len(frames)}프레임)", enc_vid))
    if name == "audio":
        jobs.append((f"WAV {len(arrays)}개({audio_sec:.1f}초)", enc_aud))
    if name == "full":
        def enc_all():                           # 혼합 = 질의 + 사진 + 동영상 + 소리를 차례로 한 번씩
            enc_q(); enc_img(); enc_aud()        # ';' : 한 줄에 문장 여러 개
            if frames:
                enc_vid()
        jobs.append(("혼합(질의+이미지+" + ("동영상+" if frames else "") + "WAV)", enc_all))

    rows = []
    for label, fn in jobs:
        torch.cuda.reset_peak_memory_stats()
        ms1, ms2, gpu = run_twice(fn)
        rows.append({
            "구성": {"text": "텍스트 전용", "vision": "+비전", "audio": "+오디오", "full": "전체"}[name],
            "로드 크기": {"text": "270M", "vision": "440M", "audio": "570M", "full": "740M"}[name],
            "로드 시간(s)": round(load_s, 1),
            "로드 후 RAM(MB)": f"{ram_after} (+{ram_after - ram_before})",
            "입력": label,
            "지연 1회차(ms)": round(ms1),
            "지연 2회차(ms)": round(ms2),
            "GPU 사용률 평균/최대(%)": gpu,
            "GPU 메모리 최대(GB)": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        })
    return rows


# ── 표 만들기 ────────────────────────────────────────────────
def to_markdown(rows):
    """행 리스트 → 마크다운 표 문자열. 보고서 .md 에 그대로 붙일 수 있다."""
    keys = list(rows[0].keys())                  # 첫 행의 항목 이름들이 머리줄
    lines = ["| " + " | ".join(keys) + " |",     # " | ".join([...]) : 리스트를 ' | '로 이어 한 문자열로
             "|" + " --- |" * len(keys)]         # 문자열 * 숫자 : 그 횟수만큼 반복
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    return "\n".join(lines)                      # 줄바꿈으로 이어 붙이기


# ── 시작점 ──────────────────────────────────────────────────
# python measure.py --one 구성 : 자식 프로세스 모드. 한 구성만 재고 결과를 'RESULT {...}' 한 줄로 출력
# python measure.py [구성...]  : 부모 모드. 구성마다 자식을 하나씩 띄워(=모델을 새로 올려) 결과를 모아 표로
if __name__ == "__main__":                       # 이 파일을 직접 실행했을 때만 아래가 돈다
    args = sys.argv[1:]                          # sys.argv[0]은 파일 이름이라 뺀다
    if args[:1] == ["--one"]:
        print("RESULT " + json.dumps(measure_one(args[1]), ensure_ascii=False))   # ensure_ascii=False : 한글을 그대로
        raise SystemExit

    names = args or CONFIGS                      # 인자가 없으면(빈 리스트) 4가지 전부
    print(f"GPU 사용률 파일: {GPU_LOAD_FILE or '못 찾음 → GPU 사용률은 tegrastats 값으로 적으세요'}")
    all_rows = []
    for name in names:
        print(f"\n[{name}] 모델 로드 후 측정 중... (장비 RAM 지금 {ram_used_mb()} MB)")
        # 같은 python 으로 이 파일을 '--one name' 인자로 다시 실행. 끝나면 그 프로세스의 메모리는 모두 반납된다
        out = subprocess.run([sys.executable, __file__, "--one", name],
                             capture_output=True, text=True)          # 출력을 화면 대신 변수로 받는다
        result = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if out.returncode != 0 or not result:    # 자식이 에러로 죽었으면 마지막 몇 줄을 보여 주고 다음 구성으로
            print(f"  실패 (종료 코드 {out.returncode})\n" + "\n".join(out.stderr.splitlines()[-8:]))
            continue
        rows = json.loads(result[0][len("RESULT "):])   # 'RESULT ' 뒤의 JSON → 파이썬 리스트
        for r in rows:
            print(f"  {r['입력']}: 2회차 {r['지연 2회차(ms)']} ms, RAM {r['로드 후 RAM(MB)']} MB, "
                  f"GPU {r['GPU 사용률 평균/최대(%)']} %, GPU 메모리 {r['GPU 메모리 최대(GB)']} GB")
        all_rows += rows

    if all_rows:
        table = to_markdown(all_rows)
        print("\n" + table)
        with open("results_step6.md", "w") as f:  # "w" : 쓰기 모드 (있으면 덮어씀). ~/eg2-lab 안에만 생긴다
            f.write(f"# 6단계 측정 결과 ({time.strftime('%Y-%m-%d %H:%M')})\n\n{table}\n")
        print("\nresults_step6.md 에 저장했습니다.")
