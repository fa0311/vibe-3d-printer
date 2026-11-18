#!/usr/bin/env bash
# 新しいマシンでこのリポジトリを動かすための準備 (何度実行してもよい)。
#   - Bambu Studio (スライサー) を ~/.local/opt/bambustudio に展開
#   - Python の依存を入れる
#   - Bambu のプリセットを CLI 用に完全展開する
set -euo pipefail

BS_VERSION="v02.09.01.70"
BS_ASSET="BambuStudio_ubu24-v02.09.01.70-20261103183404.AppImage"
BS_DIR="${V3D_BAMBU_STUDIO_DIR:-$HOME/.local/opt/bambustudio}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# libwebkit2gtk: Bambu Studio は CLI でも WebKitGTK を読み込む
# それ以外: scripts/studio-gui.sh (仮想画面で Bambu Studio を動かして操作する) が使う
PACKAGES=(libwebkit2gtk-4.1-0 xvfb xfwm4 xdotool wmctrl imagemagick)
MISSING=()
for pkg in "${PACKAGES[@]}"; do dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg"); done
if [ ${#MISSING[@]} -gt 0 ]; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${MISSING[@]}"
fi

if [ ! -f "$BS_DIR/.version" ] || [ "$(cat "$BS_DIR/.version")" != "$BS_VERSION" ]; then
  mkdir -p "$BS_DIR"
  cd "$BS_DIR"
  rm -rf squashfs-root
  curl -fL --retry 3 -o BambuStudio.AppImage \
    "https://github.com/bambulab/BambuStudio/releases/download/$BS_VERSION/$BS_ASSET"
  chmod +x BambuStudio.AppImage
  ./BambuStudio.AppImage --appimage-extract >/dev/null   # FUSE 無しでも動くよう展開して使う
  rm BambuStudio.AppImage
  echo "$BS_VERSION" > .version
fi

cd "$REPO"
uv sync
uv run python -c "from v3d import profiles; print('プリセット展開:', profiles.write_all())"
echo "準備完了。 uv run python -m v3d slice fit-test で動作確認できます。"
