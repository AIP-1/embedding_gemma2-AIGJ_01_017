# train_pose.py - 9-6단계: --record로 모은 CSV로 GRU 동작/행동 분류 모델 학습 (README 9-6)
# 목적: 규칙(손목이 어깨보다 위 등) 대신, 녹화한 예시 데이터에서 판정 기준을 배우는 작은 모델을 처음부터 학습한다.
#       파일 이름(확장자 제외)이 정답 라벨이다. 예) data/action/wave.csv → "wave"
# 실행: ~/eg2-lab 에서 (pose_model.py가 같은 폴더에 있어야 한다)
#   python train_pose.py --data data/action   --task action   --window 30 --out models/action.pt
#   python train_pose.py --data data/behavior --task behavior --window 40 --stride 5 --split seq --out models/behavior.pt
#   (행동: README는 --window 80 --stride 10이지만, 약 12 fps 환경에서는 80프레임이 6.7초라 짧은 통과가 버려져 40으로 줄임)
#
# 수정 기록
# 2026-10-27  v1  README 9-6 train_pose.py를 그대로 옮기고 줄별 설명 주석 추가 (지원 님 요청: 파이썬 복습용).
#                 라벨별 윈도 수를 "이번 파일에서 나온 개수"로 출력하도록 고침. README 코드는 같은 라벨의
#                 파일이 하나뿐이라 결과는 같지만, 셈 방식을 읽기 쉽게 하기 위함 (학습 내용은 같음)
# 2026-10-27  v2  --split seq 옵션 추가 (기본값 time은 기존과 같음). 행동 데이터는 4~8초짜리 통과를 10번 녹화하므로
#                 각 통과의 뒤 20%(1~2초)가 윈도(40프레임 ≈ 3.3초)보다 짧아 검증 윈도가 0개가 되기 때문.
#                 seq는 나중에 녹화된 통과 20%를 통째로 검증에 쓴다 (처음 보는 통과로 평가하는 효과도 있음)

import argparse, csv, glob, os, collections   # glob : 패턴(*.csv)에 맞는 파일 목록 / os : 경로 다루기
import numpy as np, torch, torch.nn as nn
from pose_model import PoseGRU, N_FEAT          # 실행 앱과 같은 모델 구조·특징 수를 쓴다

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)                         # required=True : 꼭 써야 하는 옵션
ap.add_argument("--task", choices=["action", "behavior"], required=True)   # 동작용인지 행동용인지 (앱이 이 이름으로 구분)
ap.add_argument("--window", type=int, default=30, help="시퀀스 길이(프레임)")
ap.add_argument("--stride", type=int, default=5)                 # 윈도를 몇 프레임씩 밀며 자를지 (작을수록 윈도가 많고 서로 겹침)
ap.add_argument("--epochs", type=int, default=40)                # 학습 데이터를 몇 바퀴 볼지
ap.add_argument("--hidden", type=int, default=64)                # GRU 기억 크기
ap.add_argument("--val", type=float, default=0.2, help="검증용 비율")
# --split time : 각 시퀀스의 뒤 20%를 검증 (긴 녹화 1개짜리 동작 데이터용, README 방식)
# --split seq  : 시퀀스 20%를 통째로 검증 (짧은 통과 여러 번인 행동 데이터용. 뒤 20%는 윈도보다 짧아 버려지므로) (v2)
ap.add_argument("--split", choices=["time", "seq"], default="time")
ap.add_argument("--out", required=True)                          # 저장할 모델 파일 경로
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"            # GPU가 있으면 GPU에서 학습


def load_sequences(path):
    """CSV 한 파일 → 연속된 프레임 묶음(시퀀스) 목록.
    추적 ID별로 시간순 정렬하고, 0.5초 이상 끊기면 다른 시퀀스로 나눈다 (사람이 사라졌다 나타난 경우)."""
    by_tid = collections.defaultdict(list)   # defaultdict(list) : 없는 키를 쓰면 빈 리스트를 자동으로 만들어 주는 사전
    with open(path) as f:                    # with open : 블록이 끝나면 파일을 자동으로 닫는다
        for row in csv.reader(f):            # 한 줄씩 글자 리스트로 읽는다
            if len(row) != 5 + 51:           # 칸 수가 다르면(잘린 줄 등) 건너뜀
                continue
            ts, tid = float(row[0]), row[1]  # 시각(숫자), 추적 ID(글자 그대로)
            fa, spd = float(row[3]), float(row[4])   # 정면 점수, 속도 (row[2] 규칙 라벨은 학습에 안 씀)
            # 특징 순서를 앱의 frame_features()와 똑같이 맞춘다: 좌표 34 + 신뢰도 17 + [정면, 속도]
            feat = np.array([float(x) for x in row[5:]] + [fa, spd], dtype=np.float32)
            by_tid[tid].append((ts, feat))
    seqs = []
    for rows in by_tid.values():
        rows.sort(key=lambda r: r[0])        # 시각 기준 정렬
        cur = [rows[0][1]]                   # 현재 시퀀스를 첫 프레임으로 시작
        # zip(rows, rows[1:]) : (앞 프레임, 다음 프레임) 짝으로 돈다
        for (t0, _), (t1, f) in zip(rows, rows[1:]):
            if t1 - t0 > 0.5:                # 0.5초 넘게 끊겼으면 지금까지를 하나의 시퀀스로 마감
                seqs.append(cur); cur = []
            cur.append(f)
        seqs.append(cur)
    # np.stack : 프레임 벡터 목록 → (프레임 수, 53) 배열. 윈도보다 짧은 시퀀스는 버린다
    return [np.stack(s) for s in seqs if len(s) >= args.window]


def windows(seq):
    """시퀀스를 window 길이로 stride씩 밀며 자른다. 예) 길이 100, window 30, stride 5 → 15개"""
    return [seq[i:i + args.window] for i in range(0, len(seq) - args.window + 1, args.stride)]


files = sorted(glob.glob(os.path.join(args.data, "*.csv")))     # 예: [data/action/clap.csv, data/action/idle.csv, ...]
# os.path.basename : 폴더를 뺀 파일 이름, splitext : (이름, 확장자)로 나눔 → [0] 이름 = 라벨
labels = [os.path.splitext(os.path.basename(p))[0] for p in files]
assert len(labels) >= 2, "라벨(CSV 파일)이 2개 이상 필요"
Xtr, ytr, Xva, yva = [], [], [], []        # 학습 입력·정답, 검증 입력·정답
for ci, p in enumerate(files):             # enumerate : (번호, 값) 짝으로 돈다. ci = 라벨 번호
    n_tr, n_va = 0, 0
    seqs = load_sequences(p)                       # 시퀀스 목록 (나타난 순서 = 시간순)
    n_val_seq = max(1, round(len(seqs) * args.val)) if len(seqs) >= 2 else 0   # 시퀀스 단위로 검증에 쓸 개수 (v2)
    for si, seq in enumerate(seqs):
        if args.split == "seq":
            # 시퀀스 단위 분할 (v2): 마지막 20% 시퀀스(나중에 지나간 사람)는 통째로 검증, 나머지는 통째로 학습
            is_val = si >= len(seqs) - n_val_seq
            tr, va = ([], windows(seq)) if is_val else (windows(seq), [])
        else:
            cut = int(len(seq) * (1 - args.val))      # 시간순 분할: 앞 80% 학습, 뒤 20% 검증 (겹침 방지)
            tr, va = windows(seq[:cut]), windows(seq[cut:])
        Xtr += tr; ytr += [ci] * len(tr); Xva += va; yva += [ci] * len(va)   # [ci] * 3 → [ci, ci, ci]
        n_tr += len(tr); n_va += len(va)
    print(f"{labels[ci]:>15}: 학습 {n_tr}, 검증 {n_va} 윈도")    # :>15 : 15칸 오른쪽 정렬
assert Xtr and Xva, "윈도가 부족함: 녹화 시간을 늘리거나 --window를 줄일 것"

Xtr, Xva = torch.tensor(np.stack(Xtr)), torch.tensor(np.stack(Xva))   # (윈도 수, window, 53) 텐서
ytr, yva = torch.tensor(ytr), torch.tensor(yva)
# 표준화: 특징마다 평균 0, 표준편차 1로 맞춘다. 좌표(-3~3)와 속도(0~5) 등 범위가 달라도 고르게 학습되게.
# 평균·표준편차는 학습 데이터로만 계산하고, 검증과 실행(앱)에도 같은 값을 쓴다 → 모델 파일에 함께 저장
mean = Xtr.reshape(-1, N_FEAT).mean(0)     # reshape(-1, 53) : 모든 프레임을 한 줄로 펴서 열(특징)별 평균
std = Xtr.reshape(-1, N_FEAT).std(0) + 1e-6   # + 1e-6 : 값이 늘 같은 특징의 0으로 나누기 방지
norm = lambda x: ((x - mean) / std).to(dev)   # 표준화 후 GPU로 보내는 짧은 함수

model = PoseGRU(N_FEAT, len(labels), args.hidden).to(dev)
# 라벨마다 윈도 수가 다르면(불균형) 많은 라벨만 맞히려 한다 → 적은 라벨의 틀림에 더 큰 벌점(가중치)
w = 1.0 / torch.bincount(ytr, minlength=len(labels)).float().clamp(min=1)   # bincount : 라벨별 개수. 1/개수
loss_fn = nn.CrossEntropyLoss(weight=(w / w.sum() * len(labels)).to(dev))   # 분류용 loss (클래스 불균형 보정)
# Adam : 기울기로 가중치를 고치는 방법(옵티마이저). lr = 학습률, weight_decay = 가중치가 너무 커지지 않게 (과적합 억제)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best, best_state = -1, None                # 지금까지 최고 검증 정확도와 그때의 가중치
for ep in range(args.epochs):
    model.train()                          # 학습 모드
    perm = torch.randperm(len(Xtr))        # 윈도 순서를 매 epoch 섞는다 (순서를 외우지 않게)
    for i in range(0, len(perm), 64):      # batch 64개씩
        idx = perm[i:i + 64]
        # 약한 잡음 증강: 입력에 작은 무작위 값을 더해 관절 떨림에 강하게 만든다
        x = norm(Xtr[idx]) + 0.05 * torch.randn(len(idx), args.window, N_FEAT, device=dev)
        loss = loss_fn(model(x), ytr[idx].to(dev))    # 예측과 정답의 차이
        opt.zero_grad(); loss.backward(); opt.step()  # 기울기 초기화 → 기울기 계산 → 가중치 고치기 (1 step)
    model.eval()                           # 평가 모드
    with torch.no_grad():                  # 검증은 기울기 계산이 필요 없다
        pred = model(norm(Xva)).argmax(-1).cpu()      # 점수가 가장 큰 라벨 번호
    acc = (pred == yva).float().mean().item()         # 맞힌 비율 (True=1, False=0의 평균)
    if acc > best:                         # 최고 기록이면 가중치를 복사해 둔다 (마지막 epoch가 최고라는 보장이 없으므로)
        best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if ep % 5 == 0 or ep == args.epochs - 1:
        print(f"epoch {ep:3d}  loss {loss.item():.3f}  val acc {acc:.3f}")

model.load_state_dict(best_state)          # 최고 기록 가중치로 되돌린다
with torch.no_grad():
    pred = model(norm(Xva)).argmax(-1).cpu()
# 혼동 행렬: 행 = 정답, 열 = 예측. 대각선이 맞힌 개수, 나머지는 "무엇을 무엇으로 헷갈렸는지"
cm = np.zeros((len(labels), len(labels)), dtype=int)
for t, p in zip(yva.tolist(), pred.tolist()):
    cm[t, p] += 1
print(f"\n최고 검증 정확도 {best:.3f}\n혼동 행렬 (행=정답, 열=예측)")
print(" " * 16 + "".join(f"{l[:8]:>9}" for l in labels))   # 열 제목 (라벨 앞 8글자)
for l, row in zip(labels, cm):
    print(f"{l:>15} " + "".join(f"{v:>9}" for v in row))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)   # models/ 폴더가 없으면 만든다
# 가중치 + 앱이 쓸 정보(라벨 이름, 작업 종류, 윈도 길이, 기억 크기, 표준화 값)를 한 파일에 저장
torch.save({"state": best_state, "labels": labels, "task": args.task, "window": args.window,
            "hidden": args.hidden, "mean": mean.tolist(), "std": std.tolist()}, args.out)
print(f"\n저장: {args.out}  (task={args.task}, window={args.window}, labels={labels})")
