#!/usr/bin/env bash
# 純正 Bambu Studio の画面を、専用の仮想ディスプレイで動かす (クラウド経由で印刷を送るため)。
# 持ち主のデスクトップとは別の、大きさが固定 (1600x1000) の画面なので、ボタンの位置が毎回同じになる。
# 画面は必ず shot で撮って目で確かめてから click すること (手順は CLAUDE.md の「印刷を送る」)。
#
#   open [ファイル]   Bambu Studio を (起動し直して) 開く。.gcode.3mf を渡すとプレビュー状態で開く
#   close             Bambu Studio を普通に閉じる (ログイン状態は保存される)
#   stop              仮想ディスプレイごと全部止める
#   shot <png>        画面全体を撮る
#   click <x> <y>     左クリック
#   key <キー...>     キー入力 (xdotool の書き方。例: Escape, ctrl+o)
#   windows           開いているウィンドウの一覧
#   status            動いているか
set -euo pipefail

export DISPLAY=":${V3D_GUI_DISPLAY:-99}"
SIZE="1600x1000"
RUN="${XDG_RUNTIME_DIR:-/tmp}/v3d-gui"
BS="${V3D_BAMBU_STUDIO:-$HOME/.local/opt/bambustudio/squashfs-root}"
mkdir -p "$RUN"

running() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }
spawn() {  # spawn <名前> <コマンド...>  — このシェルが終わっても動き続ける
  local name=$1; shift
  setsid nohup "$@" >"$RUN/$name.log" 2>&1 &
  echo $! >"$RUN/$name.pid"
}
main_window() {  # Bambu Studio のメインウィンドウ (ダイアログではない、一番大きいもの) の ID
  wmctrl -lG 2>/dev/null | awk '$5 >= 1000 {print $1; exit}'
}
close_studio() {
  running studio || return 0
  local id; id=$(main_window || true)
  [ -n "$id" ] && wmctrl -i -c "$id" || true
  for _ in $(seq 30); do running studio || break; sleep 0.5; done
  if running studio; then kill -- -"$(cat "$RUN/studio.pid")" 2>/dev/null || true; sleep 1; fi
  rm -f "$RUN/studio.pid"
}

case "${1:-}" in
  open)
    if ! running xvfb; then
      spawn xvfb Xvfb "$DISPLAY" -screen 0 "${SIZE}x24" +extension GLX -nolisten tcp -noreset
      for _ in $(seq 50); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.1; done
    fi
    running wm || spawn wm xfwm4 --compositor=off
    close_studio
    # ソフトウェア描画 (この画面には GPU が付いていない)
    spawn studio env __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 \
      dbus-run-session -- "$BS/AppRun" "${@:2}"
    for _ in $(seq 120); do [ -n "$(main_window || true)" ] && break; sleep 0.5; done
    id=$(main_window || true)
    [ -n "$id" ] || { echo "Bambu Studio のウィンドウが開きません ($RUN/studio.log を参照)" >&2; exit 1; }
    wmctrl -i -r "$id" -b add,maximized_vert,maximized_horz
    sleep 8   # ファイルの読み込みと描画が落ち着くのを待つ
    wmctrl -l | cut -d' ' -f5-
    ;;
  close)
    close_studio
    ;;
  stop)
    close_studio
    for name in wm xvfb; do
      if running "$name"; then kill "$(cat "$RUN/$name.pid")" 2>/dev/null || true; fi
      rm -f "$RUN/$name.pid"
    done
    ;;
  shot)
    import -window root "${2:?出力先の PNG}"
    ;;
  click)
    xdotool mousemove "${2:?x}" "${3:?y}" click 1
    ;;
  key)
    xdotool key "${@:2}"
    ;;
  windows)
    wmctrl -l | cut -d' ' -f5-
    ;;
  status)
    for name in xvfb wm studio; do running "$name" && echo "$name: 動作中" || echo "$name: 停止"; done
    ;;
  *)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2
    ;;
esac
