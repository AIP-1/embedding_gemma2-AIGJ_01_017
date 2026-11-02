#!/usr/bin/env bash
# Jetson 장비당 1회 (sudo 필요): 시스템 패키지, 전원 모드, 스왑, 공용 모델 캐시, 사용자 그룹
# 사용법: sudo bash system_setup.sh 사용자1 사용자2 ...
#   인자로 준 계정을 aiplab(공용 캐시)·video(웹캠) 그룹에 추가한다. 이미 한 항목은 건너뛴다.
# 수정 기록
# 2026-10-23 v1  최초 작성 (jetson_setup.sh에서 장비 공통 설정만 분리, 공용 캐시 /ssd/hf + aiplab 그룹 추가)
set -euo pipefail
log(){ echo -e "\n\033[1;32m==> $*\033[0m"; }
[ "$(id -u)" = 0 ] || { echo "sudo로 실행하세요: sudo bash $0 [사용자...]"; exit 1; }

log "1. 시스템 패키지"
apt update
apt install -y python3-venv python3-pip ffmpeg libsndfile1 libopenblas-dev

log "2. 최대 성능 모드"
nvpmodel -m 0 || echo "모드 0 실패: nvpmodel -q 로 MAXN 번호 확인"
jetson_clocks || true

log "3. 스왑 8GB"
if mountpoint -q /ssd; then SWAP=/ssd/swapfile; else SWAP=/swapfile; fi
if swapon --show | grep -q "$SWAP"; then echo "이미 활성: $SWAP"; else
  [ -f "$SWAP" ] || { fallocate -l 8G "$SWAP"; chmod 600 "$SWAP"; mkswap "$SWAP"; }
  swapon "$SWAP"
fi
grep -q "^$SWAP " /etc/fstab || echo "$SWAP none swap sw 0 0" >> /etc/fstab
free -h

log "4. 공용 모델 캐시 /ssd/hf (그룹 aiplab)"
getent group aiplab >/dev/null || groupadd aiplab
if mountpoint -q /ssd; then
  mkdir -p /ssd/hf
  chgrp -R aiplab /ssd/hf
  chmod -R g+rwX /ssd/hf
  find /ssd/hf -type d -exec chmod g+s {} +     # 새 파일도 aiplab 그룹으로
else
  echo "/ssd 미마운트: 공용 캐시를 만들지 않음 (계정별로 ~/.cache/huggingface 사용)"
fi

log "5. 계정을 aiplab·video 그룹에 추가"
for u in "$@"; do
  id "$u" >/dev/null 2>&1 || { echo "계정 없음: $u"; continue; }
  usermod -aG aiplab,video "$u" && echo "$u 추가 (다시 로그인해야 적용)"
done
log "완료. 각 계정에서: bash user_setup.sh"
