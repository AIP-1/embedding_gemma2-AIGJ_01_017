#!/usr/bin/env bash
# Jetson 계정마다 1회 (sudo 불필요): venv, PyTorch, sentence-transformers, 실습 폴더, 모델 확인
# 사용법: bash user_setup.sh            (system_setup.sh를 먼저 실행해 둘 것)
#        SKIP_MODEL=1 bash user_setup.sh  (모델 다운로드·로드 확인 생략)
# 수정 기록
# 2026-10-23 v1  최초 작성 (jetson_setup.sh에서 계정별 설정만 분리, eg2 단축 명령 추가)
# 2026-10-23     /ssd/hf 쓰기 권한이 없으면 멈추도록 바꿨다가 요청으로 v1 동작(개인 캐시로 대체)으로 되돌림. 현재 내용은 v1과 같음
set -euo pipefail
log(){ echo -e "\n\033[1;32m==> $*\033[0m"; }
[ "$(id -u)" != 0 ] || { echo "sudo 없이 본인 계정으로 실행하세요"; exit 1; }

log "0. 사전 조건 확인"
for c in ffmpeg python3; do command -v $c >/dev/null || { echo "$c 없음: 관리자가 system_setup.sh 먼저 실행"; exit 1; }; done
python3 -c "import ensurepip" 2>/dev/null || { echo "python3-venv 없음: 관리자가 system_setup.sh 먼저 실행"; exit 1; }
PYV=$(python3 -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
case "$PYV" in 3.10) JP=6 ;; 3.12) JP=7 ;; *) echo "지원하지 않는 Python $PYV"; exit 1 ;; esac
echo "$USER / Python $PYV / JetPack $JP"

log "1. 모델 캐시 위치 (HF_HOME)"
if [ -d /ssd/hf ] && [ -w /ssd/hf ]; then HF=/ssd/hf
else HF=$HOME/.cache/huggingface
  [ -d /ssd/hf ] && echo "/ssd/hf 쓰기 권한 없음: aiplab 그룹 추가 후 재로그인했는지 확인 (지금은 개인 캐시 사용)"
fi
export HF_HOME=$HF; mkdir -p "$HF"
sed -i '/^export HF_HOME=/d' ~/.bashrc
echo "export HF_HOME=$HF" >> ~/.bashrc
grep -q "umask 002" ~/.bashrc || [ "$HF" != /ssd/hf ] || echo "umask 002   # 공용 캐시를 그룹이 함께 쓰도록" >> ~/.bashrc
umask 002
echo "HF_HOME=$HF"

log "2. venv (~/eg2)"
[ -d ~/eg2 ] || python3 -m venv ~/eg2
source ~/eg2/bin/activate
pip install -U pip wheel

log "3. Jetson용 PyTorch"
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "이미 GPU torch 설치됨: 건너뜀"
elif [ "$JP" = 6 ]; then
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
else
  pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/cu132
fi
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.get_device_capability(), torch.cuda.is_bf16_supported())
x = torch.rand(1024, 1024, device="cuda"); print((x @ x).sum().item())
PY

log "4. sentence-transformers (torch 버전 고정)"
pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > ~/eg2/torch-pin.txt
pip install -c ~/eg2/torch-pin.txt -U \
  "sentence-transformers[image,audio,video]>=6.1.0" transformers accelerate \
  soundfile librosa av pillow huggingface_hub
if [ "$JP" = 7 ]; then pip uninstall -y torchaudio || true; fi   # README Q7

log "5. 실습 폴더 ~/eg2-lab + check.py / eg2.py"
mkdir -p ~/eg2-lab/media && cd ~/eg2-lab
cat > check.py <<'PY'
import torch, transformers, sentence_transformers as st
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("sentence-transformers", st.__version__)
assert torch.cuda.is_available(), "Jetson torch가 덮어쓰여졌는지 확인"
PY
cat > eg2.py <<'PY'
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"

def load(vision=False, audio=False):
    cfg = {}
    if not vision: cfg["vision_config"] = None
    if not audio:  cfg["audio_config"] = None
    return SentenceTransformer(
        MODEL_ID, device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},  # float16 금지
        config_kwargs=cfg,
    )
PY
python check.py

log "6. 단축 명령 eg2 (venv 활성화 + 실습 폴더 이동)"
grep -q "alias eg2=" ~/.bashrc || echo "alias eg2='source ~/eg2/bin/activate && cd ~/eg2-lab'" >> ~/.bashrc

if [ "${SKIP_MODEL:-0}" != 1 ]; then
  log "7. 모델 확인 (공용 캐시에 있으면 다운로드 생략)"
  hf download google/embeddinggemma-2
  python -c "from eg2 import load; m = load(); print(m.encode('test').shape)"   # 기대값 (768,)
fi
log "완료. 새 터미널에서 eg2 입력 → 실습 시작"
