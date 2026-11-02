#!/usr/bin/env bash
# Jetson Orin Nano Super - EmbeddingGemma 2 실습 환경 자동 설정 (README 1~2단계)
# 사용법: bash jetson_setup.sh            (SSD 포맷은 하지 않음. 마운트된 /ssd가 있으면 사용)
#        SKIP_MODEL=1 bash jetson_setup.sh  (모델 다운로드 생략)
# 수정 기록
# 2026-10-23 v1  최초 작성 (README 1~2단계를 한 파일로)
# 2026-10-23 v2  JetPack 7(R38/R39)에서 sentence-transformers 설치 후 torchaudio 제거 추가 (README Q7: cu130/cu132 충돌)
set -euo pipefail
log(){ echo -e "\n\033[1;32m==> $*\033[0m"; }

log "1. JetPack 확인"
REL=$(head -1 /etc/nv_tegra_release 2>/dev/null || echo "unknown")
echo "$REL"
PYV=$(python3 -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "Python $PYV"
case "$PYV" in
  3.10) JP=6 ;;
  3.12) JP=7 ;;   # R38/R39 (Ubuntu 24.04)
  *) echo "지원하지 않는 Python $PYV (JetPack 6=3.10, 7=3.12)"; exit 1 ;;
esac
echo "JetPack $JP 로 판단"

log "2. 최대 성능 모드"
sudo nvpmodel -q || true
sudo nvpmodel -m 0 || echo "모드 0 설정 실패: nvpmodel -q 출력에서 MAXN 번호 확인"
sudo jetson_clocks || true

log "3. 스왑 8GB"
if mountpoint -q /ssd; then BASE=/ssd; export HF_HOME=/ssd/hf; mkdir -p "$HF_HOME"
  grep -q 'HF_HOME=/ssd/hf' ~/.bashrc || echo 'export HF_HOME=/ssd/hf' >> ~/.bashrc
else BASE=""; echo "/ssd 미마운트: 루트에 스왑·캐시 생성 (README 0단계 참고)"; fi
SWAP=${BASE:-}/swapfile
if swapon --show | grep -q "$SWAP"; then echo "이미 활성: $SWAP"; else
  sudo fallocate -l 8G "$SWAP" && sudo chmod 600 "$SWAP"
  sudo mkswap "$SWAP" && sudo swapon "$SWAP"
  grep -q "^$SWAP " /etc/fstab || echo "$SWAP none swap sw 0 0" | sudo tee -a /etc/fstab
fi
free -h

log "4. 시스템 패키지 + venv (~/eg2)"
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg libsndfile1 libopenblas-dev
[ -d ~/eg2 ] || python3 -m venv ~/eg2
source ~/eg2/bin/activate
pip install -U pip wheel

log "5. Jetson용 PyTorch"
if [ "$JP" = 6 ]; then
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

log "6. sentence-transformers (torch 버전 고정)"
pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > ~/eg2/torch-pin.txt
pip install -c ~/eg2/torch-pin.txt -U \
  "sentence-transformers[image,audio,video]>=6.1.0" transformers accelerate \
  soundfile librosa av pillow huggingface_hub
if [ "$JP" = 7 ]; then
  # README Q7: PyPI torchaudio(cu130)가 cu132 torch와 어긋나 transformers import를 막음
  pip uninstall -y torchaudio || true
fi

log "7. 실습 폴더 + check.py / eg2.py"
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

if [ "${SKIP_MODEL:-0}" != 1 ]; then
  log "8. 모델 다운로드 + 로드 확인"
  hf download google/embeddinggemma-2
  python -c "from eg2 import load; m = load(); print(m.encode('test').shape)"   # 기대값 (768,)
fi
log "완료. 이후: source ~/eg2/bin/activate && cd ~/eg2-lab"
