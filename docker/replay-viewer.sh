#!/bin/bash
set -e

# The base image sets LIBGL_ALWAYS_SOFTWARE=1 for the headless game server.
# The replay viewer needs GPU rendering, so unset it.
unset LIBGL_ALWAYS_SOFTWARE

# 用法：
#   /replay-viewer.sh <replay_file>          # 单局直放（原行为）：进容器即播放该局
#   /replay-viewer.sh --menu [file|dir ...]  # 菜单模式：把给定录像（目录按 *.orarep 展开）
#                                            # 拷入 Replays 目录后进 OpenRA 主菜单，
#                                            # 用户在 Extras -> Replays 里自选。
#                                            # 不给文件则直接进菜单（volume 里已有的录像同样可见）。
MODE="single"
if [ "${1:-}" = "--menu" ]; then
  MODE="menu"
  shift
fi

# Tunable settings via environment variables (set by docker_manager.py)
REPLAY_RESOLUTION="${OPENRA_RL_REPLAY_RESOLUTION:-1280x960}"
REPLAY_WIDTH="${REPLAY_RESOLUTION%x*}"
REPLAY_HEIGHT="${REPLAY_RESOLUTION#*x}"
REPLAY_UI_SCALE="${OPENRA_RL_REPLAY_UI_SCALE:-1}"
REPLAY_VIEWPORT="${OPENRA_RL_REPLAY_VIEWPORT_DISTANCE:-Medium}"
REPLAY_MUTE="${OPENRA_RL_REPLAY_MUTE:-True}"

# Copy replay to the expected directory structure so OpenRA can read metadata.
# NOTE: 本镜像构建时 Version 字面量就是 {DEV_VERSION}（见 mods/ra/mod.yaml），
# 所以 Replays/ra/{DEV_VERSION}/ 正是引擎实际扫描的目录，不是占位符 bug。
REPLAY_DIR="/root/.config/openra/Replays/ra/{DEV_VERSION}"
mkdir -p "$REPLAY_DIR"

copy_one() {
  src="$1"
  base="$(basename "$src")"
  # watch_replay.sh 灌入多文件时为防重名加了 seed_<i>__ 前缀，
  # 进 Replays 菜单前剥掉，还原原始文件名，方便在 Extras -> Replays 里辨认。
  case "$base" in
    seed_[0-9]*__*) base="$(echo "$base" | sed 's/^seed_[0-9][0-9]*__//')" ;;
  esac
  cp "$src" "$REPLAY_DIR/$base"
  echo "Replay available: $REPLAY_DIR/$base"
}

LAUNCH_REPLAY_ARG=""
if [ "$MODE" = "menu" ]; then
  staged=0
  for src in "$@"; do
    if [ -d "$src" ]; then
      for f in "$src"/*.orarep; do
        [ -f "$f" ] || continue
        copy_one "$f"
        staged=$((staged + 1))
      done
    elif [ -f "$src" ]; then
      copy_one "$src"
      staged=$((staged + 1))
    else
      echo "WARNING: skip (not found): $src"
    fi
  done
  echo "Menu mode: staged $staged replay(s) this boot; pre-existing ones in the volume are listed too."
  echo "--- $REPLAY_DIR ---"
  ls -lh "$REPLAY_DIR" | head -n 40 || true
  echo "In OpenRA main menu, pick Extras -> Replays."
else
  REPLAY_FILE="${1:-}"
  if [ -z "$REPLAY_FILE" ]; then
    echo "Usage: /replay-viewer.sh <replay_file_path> | /replay-viewer.sh --menu [file|dir ...]"
    exit 1
  fi
  if [ ! -f "$REPLAY_FILE" ]; then
    echo "ERROR: Replay file not found: $REPLAY_FILE"
    exit 1
  fi
  copy_one "$REPLAY_FILE"
  REPLAY_BASENAME=$(basename "$REPLAY_FILE")
  LAUNCH_REPLAY_ARG="Launch.Replay=$REPLAY_DIR/$REPLAY_BASENAME"
  echo "Replay copied to: $REPLAY_DIR/$REPLAY_BASENAME"
fi

# Start Xvfb at configured resolution
echo "Starting Xvfb on display :99 (${REPLAY_WIDTH}x${REPLAY_HEIGHT})..."
Xvfb :99 -screen 0 ${REPLAY_WIDTH}x${REPLAY_HEIGHT}x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 2
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
export DISPLAY=:99

# Start x11vnc with performance optimizations
echo "Starting VNC server on port 5900..."
x11vnc -display :99 -forever -nopw -shared -rfbport 5900 \
    -noxdamage -wait 50 -defer 50 -quiet &
VNC_PID=$!
sleep 1

# Start noVNC (websockify proxy)
echo "Starting noVNC on port 6080..."
websockify --web /usr/share/novnc 6080 localhost:5900 &
NOVNC_PID=$!
sleep 1

echo ""
echo "=== Replay viewer ready (mode: $MODE) ==="
echo "Open in browser: http://localhost:6080/vnc.html"
echo "Press Ctrl+C to stop"
echo ""

# Clean shutdown on signals
cleanup() {
    echo "Shutting down replay viewer..."
    kill $NOVNC_PID 2>/dev/null || true
    kill $VNC_PID 2>/dev/null || true
    kill $XVFB_PID 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Launch OpenRA with rendering settings tuned for VNC replay viewing.
# CPU is managed by Docker --cpus limit (set in docker_manager.py).
# 菜单模式不传 Launch.Replay，停在主菜单由用户进 Extras -> Replays 自选。
OPENRA_ARGS=(
    Engine.EngineDir=/opt/openra
    Game.Mod=ra
    Game.Platform=Default
    Graphics.Mode=Windowed
    Graphics.WindowedSize=${REPLAY_WIDTH},${REPLAY_HEIGHT}
    Graphics.UIScale=${REPLAY_UI_SCALE}
    Graphics.VSync=False
    Graphics.DisableGLDebugMessageCallback=True
    Graphics.ViewportDistance=${REPLAY_VIEWPORT}
    Sound.Mute=${REPLAY_MUTE}
)
if [ -n "$LAUNCH_REPLAY_ARG" ]; then
    OPENRA_ARGS+=("$LAUNCH_REPLAY_ARG")
fi
exec dotnet /opt/openra/bin/OpenRA.dll "${OPENRA_ARGS[@]}"
