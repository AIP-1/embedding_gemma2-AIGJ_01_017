# test_vision.py - README 4단계 사진·영상 테스트
# 목적: 비전 인코더를 켠 모델로 '글로 쓴 질의'가 알맞은 사진과 동영상을 찾는지 확인한다.
#       글과 사진이 같은 벡터 공간에 들어가므로, 3단계처럼 유사도만 비교하면 검색이 된다.
# 실행: ~/eg2-lab 에서 venv를 켠 뒤  python test_vision.py
# 준비: ~/eg2-lab/media/ 폴더에 사진 3~5장(.jpg/.jpeg/.png), 동영상 clip.mp4(10초 내외, 없어도 됨)
#
# 수정 기록
# 2026-10-23  v1  README 4단계 코드를 바탕으로 작성. 바꾼 점 3가지 (실습 중 막히지 않게 하려고)
#                 1) .jpg 외에 .jpeg/.png도 찾음 (맥·폰 사진 확장자가 제각각이라서)
#                 2) 동영상이 없거나 읽기에 실패해도 사진 검색 결과는 끝까지 나오게 함 (README의 '주석 처리' 안내를 자동화)
#                 3) GPU 메모리 최대 사용량을 출력 (6단계 측정 표 '+비전' 행 기록용)
# 2026-10-23  v2  질의를 지원 님 클립 내용(사람 없음 1~3, 두 명 4~5)에 맞게 바꾸고, 상위 3개 대신 전체 순위 출력
#                 (README 질의는 화이트보드 등 클립에 없는 장면이라 정답 판단이 안 됐고, 사진이 5장뿐이라 전체를 보는 게 비교하기 쉬워서)
# 2026-10-23  v3  동영상을 기본 방식(torchcodec)으로 못 읽으면 PyAV(pip 패키지 av)로 1초에 1장씩 직접 꺼내 넣는 대체 경로 추가
#                 (Jetson의 FFmpeg에 libswscale이 없어 torchcodec이 실패하고, apt에도 libswscale9가 없어서.
#                  PyAV는 FFmpeg를 자기 안에 담고 있어 sudo 없이 내 venv에만 설치된다)

# ── 3단계와 달라진 점 ────────────────────────────────────────
# 3단계: 문서(글) ↔ 질의(글).  4단계: 사진·영상 ↔ 질의(글).
# 사진도 encode() 에 넣으면 768개 숫자 벡터가 나온다. 이 벡터와 질의 벡터의 코사인 유사도를 비교한다.
# 사진은 기본 280 토큰(조각)으로 나눠 읽고, 동영상은 1초에 1프레임씩 뽑아 읽는다.
# ────────────────────────────────────────────────────────────

import glob                   # 파일 이름 패턴(예: media/*.jpg)에 맞는 파일 목록을 돌려주는 기본 패키지
import os                     # 파일이 있는지 확인(os.path.exists)할 때 쓰는 기본 패키지
import time                   # 걸린 시간을 재는 기본 패키지 (3단계와 같음)
import torch                  # GPU 메모리 사용량을 읽으려고 불러온다
from eg2 import load          # 같은 폴더 eg2.py 의 공통 로더

# load(vision=True) : 비전 인코더까지 올린다 (텍스트 270M → 약 440M 파라미터).
# 3단계보다 로딩이 더 오래 걸리고 메모리도 더 쓴다.
model = load(vision=True)

# ── 1. 사진 목록 만들기 ─────────────────────────────────────
# glob.glob("media/*.jpg") : media 폴더에서 .jpg로 끝나는 파일 경로들을 리스트로 돌려준다.
# 리스트 + 리스트 : 두 리스트를 이어 붙인다. 확장자 세 종류를 한 리스트로 모은다.
# sorted(...) : 이름순으로 정렬. 실행할 때마다 순서가 같아야 결과를 비교하기 쉽다.
images = sorted(glob.glob("media/*.jpg") + glob.glob("media/*.jpeg") + glob.glob("media/*.png"))

# len(리스트) : 리스트에 든 값의 개수.
# 사진이 하나도 없으면 더 진행할 수 없으므로 안내를 찍고 끝낸다.
if len(images) == 0:
    print("media/ 폴더에 사진(.jpg/.jpeg/.png)이 없습니다. 사진을 넣고 다시 실행하세요.")
    raise SystemExit          # 프로그램을 여기서 끝낸다

# ── 2. 사진 임베딩 ──────────────────────────────────────────
t = time.perf_counter()       # 시작 시각

# [ {"image": p} for p in images ] : 리스트 컴프리헨션. for 문을 한 줄로 쓴 것.
#   images 의 경로 p 하나마다 {"image": p} 사전을 만들어 리스트로 모은다.
#   예: [{"image": "media/a.jpg"}, {"image": "media/b.jpg"}, ...]
#   {"image": 경로} 형태로 넣어야 모델이 '글'이 아니라 '사진'으로 알아듣는다.
# 사진에는 prompt_name 을 주지 않는다 (사진 자체가 문서 역할).
# 결과 img_emb 모양: (사진 개수, 768)
img_emb = model.encode([{"image": p} for p in images], normalize_embeddings=True)

ms_img = (time.perf_counter() - t) * 1000   # 걸린 시간(ms)
print(f"사진 {len(images)}장: {ms_img:.0f} ms (장당 {ms_img/len(images):.0f} ms)")

# ── 3. 동영상 임베딩 (없거나 실패해도 계속 진행) ───────────────
VIDEO = "media/clip.mp4"      # 동영상 경로. 다른 이름이면 여기만 바꾼다
vid_emb = None                # None : '값 없음'. 동영상 단계가 실패하면 그대로 None으로 남는다



def read_frames_1fps(path):
    """PyAV로 동영상을 열어 1초에 1장씩 PIL 사진으로 꺼내 리스트로 돌려준다."""
    import av                                   # PyAV. 함수 안에서 import → 이 경로를 쓸 때만 필요
    frames = []                                 # 꺼낸 사진을 담을 빈 리스트
    next_t = 0.0                                # 다음에 꺼낼 시각(초). 0초, 1초, 2초, ...
    with av.open(path) as container:            # with : 블록이 끝나면 파일을 자동으로 닫는다
        # container.decode(video=0) : 첫 번째 영상 트랙의 프레임을 처음부터 하나씩 풀어 준다 (보통 초당 30장)
        for frame in container.decode(video=0):
            # frame.time : 이 프레임이 영상의 몇 초 지점인지. None일 수 있어 그때는 건너뛴다
            if frame.time is not None and frame.time >= next_t:
                frames.append(frame.to_image())  # to_image() : 프레임 → PIL 사진. append : 리스트 끝에 추가
                next_t += 1.0                    # 다음 목표 시각을 1초 뒤로 (모델 기본값 '초당 1프레임'에 맞춤)
    return frames


# os.path.exists(경로) : 그 파일이 있으면 True
if os.path.exists(VIDEO):
    t = time.perf_counter()
    # try: ... except 에러 as e: ... → try 블록에서 에러가 나도 프로그램이 죽지 않고 except 블록으로 넘어간다.
    # 1차: 모델 기본 방식. 동영상 하나는 리스트가 아니라 사전 하나로 넣는다 → 결과 vid_emb 모양: (768,)
    try:
        vid_emb = model.encode({"video": VIDEO}, normalize_embeddings=True)
        how = "기본(torchcodec)"
    except Exception as e:
        # type(e).__name__ : 에러 종류 이름. 에러 전문은 길어서 첫 줄만 보여 준다 (str(e).splitlines()[0])
        print(f"기본 방식 실패 ({type(e).__name__}: {str(e).splitlines()[0]}) → PyAV로 다시 시도")
        try:
            frames = read_frames_1fps(VIDEO)
            print(f"  PyAV로 {len(frames)}장 추출 (1초에 1장)")
            try:
                # 2차: 꺼낸 사진 리스트를 '동영상'으로 넘긴다. 모델이 여러 장을 한 영상으로 묶어 읽는다
                vid_emb = model.encode({"video": frames}, normalize_embeddings=True)
                how = "PyAV 프레임 → 동영상 입력"
            except Exception as e2:
                # 3차: 그것도 안 되면 각 장을 사진으로 임베딩한 뒤 평균을 낸다 (근사치: 장면 순서 정보는 사라짐)
                print(f"  동영상 입력 실패 ({type(e2).__name__}) → 프레임 평균으로 대체")
                emb = model.encode([{"image": f} for f in frames], normalize_embeddings=True)
                vid_emb = emb.mean(0)                                # mean(0) : 행 방향 평균 → (768,) 하나
                vid_emb = vid_emb / (vid_emb ** 2).sum() ** 0.5      # 평균 내면 길이가 줄어 다시 1로 맞춤
                how = "PyAV 프레임 평균(근사)"
        except Exception as e3:
            # PyAV가 없거나 그것도 실패하면 사진 결과만 낸다
            print(f"동영상 읽기 실패 → 사진 결과만 출력합니다. ({type(e3).__name__}: {e3})")
    if vid_emb is not None:
        print(f"동영상 1개: {(time.perf_counter() - t) * 1000:.0f} ms  [방식: {how}]")
else:
    print(f"{VIDEO} 없음 → 동영상 단계는 건너뜁니다.")

# ── 4. 질의 임베딩 ──────────────────────────────────────────
# 내 사진에 맞게 바꿔도 된다. 질의마다 '정답 사진'이 정해져 있어야 결과를 판단할 수 있다.
# 지원 님 클립 기준 정답: frame_01~03 = 사람 없음 (03은 팔 일부), frame_04~05 = 두 사람
queries = [
    "사람이 아무도 없는 빈 공간",          # 정답: frame_01, 02, 03 이 위쪽
    "두 사람이 함께 있는 장면",            # 정답: frame_04, 05 가 위쪽
    "카메라를 바라보는 사람의 얼굴",        # 정답: frame_04, 05 (한 명은 얼굴이 보임)
]

# 질의는 3단계와 똑같이 SearchQuery 프롬프트를 붙인다. 결과 q 모양: (질의 3개, 768)
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)

# ── 5. 질의 → 사진 검색 ─────────────────────────────────────
# q(3×768) @ img_emb.T(768×사진수) = sim(3×사진수) : 질의별·사진별 유사도 표 (3단계와 같은 계산)
sim = q @ img_emb.T

print()                       # 빈 줄 하나
for qi, row in zip(queries, sim):
    # row.argsort() : 값이 작은 것부터의 '번호' 순서. [::-1] 로 뒤집어 큰 것부터.
    #   → 유사도가 높은 순서로 정렬한 사진 번호 (사진 5장 전체)
    order = row.argsort()[::-1]
    # images[i].split('/')[-1] : 경로를 '/'로 쪼갠 뒤 마지막 조각 = 파일 이름만 (media/a.jpg → a.jpg)
    # round(float(row[i]), 3) : 유사도를 소수점 셋째 자리로 반올림
    top = [(images[i].split('/')[-1], round(float(row[i]), 3)) for i in order]
    print(f"{qi}\n   -> {top}")   # 질의 한 줄, 순위 한 줄로 나눠 보기 쉽게

# ── 6. 질의 → 동영상 유사도 ──────────────────────────────────
# is not None : 동영상 임베딩이 만들어졌을 때만
if vid_emb is not None:
    # q(3×768) @ vid_emb(768,) = 질의 3개 각각과 동영상의 유사도 3개
    # .round(3) : 셋째 자리 반올림
    print("질의-동영상 유사도:", (q @ vid_emb).round(3))

# ── 7. 메모리 (6단계 측정 표 '+비전' 행) ─────────────────────
# torch.cuda.max_memory_allocated() : 이 프로그램이 GPU에서 가장 많이 쓴 순간의 메모리(바이트).
# / 1024**3 : 바이트 → GB.  Jetson은 CPU와 GPU가 메모리를 함께 쓰므로 tegrastats 의 RAM 값도 같이 본다.
print(f"\nGPU 메모리 최대 사용: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
