#!/bin/bash

# ============================================================================
# Myboat Monte Carlo runner
# Usage:
#   bash boat_monte_carlo.bash HeadOn Geo 30
#   bash boat_monte_carlo.bash NarrowMulti Local 1 camera 12345
#
# Scenarios: HeadOn | Crossing | Overtaking | MultiShip | NarrowMulti
# Modes: Geo | Local
# Results:
#   tmp/monte_carlo/<timestamp>_<scenario>_<mode>/results.csv
# ============================================================================

set -euo pipefail

export PS1="${PS1-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$SCRIPT_DIR}"
cd "$WORKSPACE"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "help" ]; then
  cat <<'EOF'
Usage:
  bash boat_monte_carlo.bash <Scenario> <Mode> [Count] [InputSource] [Seed]

Scenario:
  HeadOn | Crossing | Overtaking | MultiShip | NarrowMulti

Mode:
  Geo
  Local

InputSource:
  lidar | pointcloud_bev | camera | global | ais

Seed:
  integer              fixed base seed; trial N uses Seed + N - 1
  random | none | off  use a time-based seed per trial

Common overrides:
  MC_COUNT=30
  MC_TIMEOUT_S=100
  MC_RESULT_DIR=/path/to/output
  MC_GUI=false
  MC_SEED=12345
  MC_PLOT_TRAJECTORIES=true
  LOCAL_API_BASE=http://host:8000/v1
  MODEL=qwen3-vl-2b-instruct-local
EOF
  exit 0
fi

source ~/.bashrc
CONDA_ENV_NAME=${CONDA_ENV_NAME:-spconv2}
if [ -n "$CONDA_ENV_NAME" ] && [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
fi
ACTIVE_PYTHON_BIN="$(command -v python3 || true)"
if [ ! -f devel/setup.bash ]; then
  echo "[ERROR] devel/setup.bash not found. Build first with: catkin build -j2 && source devel/setup.bash"
  exit 1
fi
source devel/setup.bash

usage() {
  cat <<'EOF'
Usage:
  bash boat_monte_carlo.bash <Scenario> <Mode> [Count] [InputSource] [Seed]

Scenario:
  HeadOn | Crossing | Overtaking | MultiShip | NarrowMulti

Mode:
  Geo
  Local

InputSource:
  lidar | pointcloud_bev | camera | global | ais

Seed:
  integer              fixed base seed; trial N uses Seed + N - 1
  random | none | off  use a time-based seed per trial

Common overrides:
  MC_COUNT=30
  MC_TIMEOUT_S=100
  VLA_TIMEOUT_SCALE=1.0     # VLA modes default to MC_TIMEOUT_S
  MC_GOAL_TOLERANCE_M=4.0
  MC_COLLISION_DISTANCE_M=2.2
  MC_COLREG_EVAL_DISTANCE_M=10.0
  MC_COLREG_MULTI_ALLOWED_VIOLATIONS=1
  MC_RESULT_DIR=/path/to/output
  MC_GUI=false
  MC_SEED=12345
  MC_PLOT_TRAJECTORIES=true
  MC_SENSOR=lidar          # lidar | camera
  VLA_IMAGE_SOURCE=camera  # camera | pointcloud_bev | global | ais
  LOCAL_API_BASE=http://host:8000/v1
  MODEL=qwen3-vl-2b-instruct-local
EOF
}

SCENARIO_INPUT="${1:-${MC_SCENARIO:-HeadOn}}"
ALGORITHM_INPUT="${2:-${MC_ALGORITHM:-Geo}}"
COUNT="${3:-${MC_COUNT:-30}}"
INPUT_SOURCE_ARG="${4:-}"
SEED_ARG="${5:-}"

if [ -n "$INPUT_SOURCE_ARG" ] && [ -z "$SEED_ARG" ]; then
  INPUT_SOURCE_OR_SEED_KEY="${INPUT_SOURCE_ARG,,}"
  case "$INPUT_SOURCE_OR_SEED_KEY" in
    random|none|off|false|unfixed|auto)
      SEED_ARG="$INPUT_SOURCE_ARG"
      INPUT_SOURCE_ARG=""
      ;;
    *)
      if [[ "$INPUT_SOURCE_ARG" =~ ^-?[0-9]+$ ]]; then
        SEED_ARG="$INPUT_SOURCE_ARG"
        INPUT_SOURCE_ARG=""
      fi
      ;;
  esac
fi

SCENARIO_KEY=$(echo "$SCENARIO_INPUT" | tr '[:lower:]' '[:upper:]' | tr -d '_-')
ALGORITHM_RAW=$(echo "$ALGORITHM_INPUT" | tr '[:lower:]' '[:upper:]')
ALGORITHM_KEY=$(echo "$ALGORITHM_RAW" | tr -d ' _+-')

case "$SCENARIO_KEY" in
  HEADON|HEADINGON) SCENARIO_NAME="HeadOn" ;;
  CROSSING) SCENARIO_NAME="Crossing" ;;
  OVERTAKING) SCENARIO_NAME="Overtaking" ;;
  MULTISHIP|MULTI) SCENARIO_NAME="MultiShip" ;;
  NARROWMULTI|NARROW) SCENARIO_NAME="NarrowMulti" ;;
  HELP|-H|--HELP)
    usage
    exit 0
    ;;
  *)
    echo "[ERROR] Unknown scenario: $SCENARIO_INPUT"
    usage
    exit 1
    ;;
esac

case "$ALGORITHM_KEY" in
  GEO)
    ALGORITHM="Geo"
    VLA_PROMPT_MODE=""
    ENABLE_VLA_MODE=0
    ENABLE_LOCAL_LLM_MODE=0
    ;;
  LOCAL)
    ALGORITHM="Local"
    VLA_PROMPT_MODE="trajectory_tokens"
    ENABLE_VLA_MODE=1
    ENABLE_LOCAL_LLM_MODE=1
    ;;
  *)
    echo "[ERROR] Unknown mode: $ALGORITHM_INPUT"
    usage
    exit 1
    ;;
esac

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [ "$COUNT" -le 0 ]; then
  echo "[ERROR] Count must be a positive integer: $COUNT"
  exit 1
fi

RUN_STAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_NAME=$(echo "$ALGORITHM" | tr ' /+' '___')
RESULT_DIR="${MC_RESULT_DIR:-$WORKSPACE/tmp/monte_carlo/${RUN_STAMP}_${SCENARIO_NAME}_${RESULT_NAME}}"
mkdir -p "$RESULT_DIR"

RESULT_CSV="$RESULT_DIR/results.csv"
SUMMARY_TXT="$RESULT_DIR/summary.txt"

MC_TIMEOUT_S="${MC_TIMEOUT_S:-100}"
MC_GOAL_TOLERANCE_M="${MC_GOAL_TOLERANCE_M:-4.0}"
MC_COLLISION_DISTANCE_M="${MC_COLLISION_DISTANCE_M:-2.2}"
MC_NEAR_MISS_DISTANCE_M="${MC_NEAR_MISS_DISTANCE_M:-5.0}"
MC_COLREG_PASS_CLEARANCE_M="${MC_COLREG_PASS_CLEARANCE_M:-4.0}"
MC_COLREG_SUBSTANTIAL_OFFSET_M="${MC_COLREG_SUBSTANTIAL_OFFSET_M:-1.2}"
MC_COLREG_EVAL_DISTANCE_M="${MC_COLREG_EVAL_DISTANCE_M:-10.0}"
MC_COLREG_MULTI_ALLOWED_VIOLATIONS="${MC_COLREG_MULTI_ALLOWED_VIOLATIONS:-1}"
MC_GUI="${MC_GUI:-false}"
MC_SEED="${MC_SEED:-}"
SEED_MODE="random"
SEED_BASE=""
if [ -n "$MC_SEED" ]; then
  if ! [[ "$MC_SEED" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] MC_SEED must be an integer, or leave it empty for random seeds: $MC_SEED"
    exit 1
  fi
  SEED_MODE="fixed"
  SEED_BASE="$MC_SEED"
fi
if [ -n "$SEED_ARG" ]; then
  SEED_ARG_KEY="${SEED_ARG,,}"
  case "$SEED_ARG_KEY" in
    random|none|off|false|unfixed|auto)
      MC_SEED=""
      SEED_MODE="random"
      SEED_BASE=""
      ;;
    fixed)
      if [ -z "$MC_SEED" ]; then
        echo "[ERROR] Seed argument 'fixed' requires MC_SEED to be set, or pass an integer seed."
        exit 1
      fi
      ;;
    *)
      if ! [[ "$SEED_ARG" =~ ^-?[0-9]+$ ]]; then
        echo "[ERROR] Seed must be an integer, random, none, or off: $SEED_ARG"
        exit 1
      fi
      MC_SEED="$SEED_ARG"
      SEED_MODE="fixed"
      SEED_BASE="$MC_SEED"
      ;;
  esac
fi
MC_PLOT_TRAJECTORIES="${MC_PLOT_TRAJECTORIES:-true}"
MC_PLOT_MAX_FILES="${MC_PLOT_MAX_FILES:-0}"
MC_PLOT_LABEL_INTERVAL_S="${MC_PLOT_LABEL_INTERVAL_S:-2.0}"
MC_PLOT_MAX_LABELS_PER_TRACK="${MC_PLOT_MAX_LABELS_PER_TRACK:-3}"
MC_PLOT_MIN_LABEL_GAP_M="${MC_PLOT_MIN_LABEL_GAP_M:-5.0}"
MC_STARTUP_TIMEOUT_S="${MC_STARTUP_TIMEOUT_S:-45}"
MC_CLOCK_STALL_S="${MC_CLOCK_STALL_S:-8.0}"
ENABLE_AVOIDANCE="${ENABLE_AVOIDANCE:-1}"
TRAJECTORY_RECORD_INTERVAL_S="${TRAJECTORY_RECORD_INTERVAL_S:-0.5}"
TARGET_MODEL_Z=${TARGET_MODEL_Z:-0.65}
VLA_SPEED_SCALE=${VLA_SPEED_SCALE:-0.3333333333333333}
VLA_TIMEOUT_SCALE=${VLA_TIMEOUT_SCALE:-1.0}

export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"
if [ "$ENABLE_LOCAL_LLM_MODE" = "1" ]; then
  CLASSIFIER_BACKEND=${CLASSIFIER_BACKEND:-http}
else
  CLASSIFIER_BACKEND=${CLASSIFIER_BACKEND:-http}
fi
if [ "$ENABLE_LOCAL_LLM_MODE" = "1" ]; then
  LOCAL_API_BASE="${LOCAL_API_BASE:-${API_BASE:-http://10.92.157.143:8000/v1}}"
  LOCAL_API_BASE="${LOCAL_API_BASE%/}"
  LLM_API_URL="${LLM_API_URL:-${LOCAL_API_URL:-${LOCAL_API_BASE}/chat/completions}}"
  QWEN_VL_MODEL="${QWEN_VL_MODEL:-${LOCAL_MODEL:-${MODEL:-qwen3-vl-2b-instruct-local}}}"
  LLM_USE_IMAGE_INPUT=${LLM_USE_IMAGE_INPUT:-true}
  LLM_API_FORCE_JSON_MODE=${LLM_API_FORCE_JSON_MODE:-true}
else
  LLM_API_URL="${LLM_API_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions}"
  QWEN_VL_MODEL=${QWEN_VL_MODEL:-qwen3.6-flash}
  LLM_USE_IMAGE_INPUT=${LLM_USE_IMAGE_INPUT:-true}
  LLM_API_FORCE_JSON_MODE=${LLM_API_FORCE_JSON_MODE:-true}
fi
LLM_API_MAX_OUTPUT_TOKENS=${LLM_API_MAX_OUTPUT_TOKENS:-96}
LLM_API_REPEAT_RETRY_ENABLE=${LLM_API_REPEAT_RETRY_ENABLE:-true}
LLM_API_REPEAT_NGRAM_WORDS=${LLM_API_REPEAT_NGRAM_WORDS:-8}
LLM_API_REPEAT_NGRAM_MIN_COUNT=${LLM_API_REPEAT_NGRAM_MIN_COUNT:-3}
LLM_API_NO_JSON_RETRY_CHARS=${LLM_API_NO_JSON_RETRY_CHARS:-180}
LLM_DEBUG_ENABLE=${LLM_DEBUG_ENABLE:-false}
LLM_TERMINAL_IO_ENABLE=${LLM_TERMINAL_IO_ENABLE:-true}
LLM_TERMINAL_CLEAN_MODE=${LLM_TERMINAL_CLEAN_MODE:-true}
VLM_CHECK_ENABLE=${VLM_CHECK_ENABLE:-false}
if [ "$ENABLE_VLA_MODE" = "1" ]; then
  VLA_IMAGE_BUFFER_DURATION_S=${VLA_IMAGE_BUFFER_DURATION_S:-1.0}
  VLA_IMAGE_BUFFER_MAX_FRAMES=${VLA_IMAGE_BUFFER_MAX_FRAMES:-12}
else
  VLA_IMAGE_BUFFER_DURATION_S=${VLA_IMAGE_BUFFER_DURATION_S:-2.0}
  VLA_IMAGE_BUFFER_MAX_FRAMES=${VLA_IMAGE_BUFFER_MAX_FRAMES:-30}
fi
VLA_ANNOTATED_SNAPSHOT_ENABLE=${VLA_ANNOTATED_SNAPSHOT_ENABLE:-false}
LLM_TRIGGER_DISTANCE_M=${LLM_TRIGGER_DISTANCE_M:-20.0}
SENSOR_FRONT_HAZARD_ENABLE=${SENSOR_FRONT_HAZARD_ENABLE:-true}
SENSOR_FRONT_HAZARD_RANGE_M=${SENSOR_FRONT_HAZARD_RANGE_M:-$LLM_TRIGGER_DISTANCE_M}
SENSOR_FRONT_HAZARD_BEARING_DEG=${SENSOR_FRONT_HAZARD_BEARING_DEG:-45.0}
SENSOR_FRONT_HAZARD_MIN_FORWARD_M=${SENSOR_FRONT_HAZARD_MIN_FORWARD_M:-1.0}
LLM_REQUIRE_POINTCLOUD_DISTANCE=${LLM_REQUIRE_POINTCLOUD_DISTANCE:-true}
LLM_CALL_REQUIRES_TRIGGER=${LLM_CALL_REQUIRES_TRIGGER:-false}
LLM_REQUIRE_TRIGGER_LOW_BEFORE_CALL=${LLM_REQUIRE_TRIGGER_LOW_BEFORE_CALL:-false}
MC_VLM_TRUTH_ENABLE=${MC_VLM_TRUTH_ENABLE:-true}
MC_VLM_TRUTH_TCPA_HORIZON_S=${MC_VLM_TRUTH_TCPA_HORIZON_S:-120.0}
MC_VLM_TRUTH_DCPA_DANGER_M=${MC_VLM_TRUTH_DCPA_DANGER_M:-$MC_COLREG_EVAL_DISTANCE_M}
MC_VLM_TRUTH_CLOSE_RANGE_M=${MC_VLM_TRUTH_CLOSE_RANGE_M:-$LLM_TRIGGER_DISTANCE_M}
MC_TRUTH_CPA_WARMUP_S=${MC_TRUTH_CPA_WARMUP_S:-3.0}
MC_TRUTH_CPA_MIN_RANGE_M=${MC_TRUTH_CPA_MIN_RANGE_M:-0.0}
if [ "$ENABLE_VLA_MODE" = "1" ]; then
  LLM_ONCE_PER_TRIGGER=${LLM_ONCE_PER_TRIGGER:-false}
  LLM_DISABLE_AFTER_FIRST_SUCCESS=${LLM_DISABLE_AFTER_FIRST_SUCCESS:-false}
  LLM_UPDATE_INTERVAL_S=${LLM_UPDATE_INTERVAL_S:-6.0}
else
  LLM_ONCE_PER_TRIGGER=${LLM_ONCE_PER_TRIGGER:-true}
  LLM_DISABLE_AFTER_FIRST_SUCCESS=${LLM_DISABLE_AFTER_FIRST_SUCCESS:-true}
  LLM_UPDATE_INTERVAL_S=${LLM_UPDATE_INTERVAL_S:-10.0}
fi
LLM_API_TIMEOUT=${LLM_API_TIMEOUT:-60.0}
LLM_FOV_GATE_ENABLE=${LLM_FOV_GATE_ENABLE:-true}
LLM_FOV_GATE_MARGIN_DEG=${LLM_FOV_GATE_MARGIN_DEG:-6.0}
LLM_FOV_GATE_MIN_FORWARD_M=${LLM_FOV_GATE_MIN_FORWARD_M:-1.0}
LLM_FOV_GATE_MIN_VISIBLE_TRACKS=${LLM_FOV_GATE_MIN_VISIBLE_TRACKS:-1}
LLM_REQUIRE_UNKNOWN_TRACK_IN_FOV=${LLM_REQUIRE_UNKNOWN_TRACK_IN_FOV:-true}
LLM_FOV_GATE_STATE_FALLBACK_ENABLE=${LLM_FOV_GATE_STATE_FALLBACK_ENABLE:-true}
LLM_FOV_GATE_STATE_FALLBACK_AFTER_S=${LLM_FOV_GATE_STATE_FALLBACK_AFTER_S:-0.0}
LLM_FOV_GATE_SOFT_ENABLE=${LLM_FOV_GATE_SOFT_ENABLE:-true}
LLM_FOV_GATE_SOFT_AFTER_S=${LLM_FOV_GATE_SOFT_AFTER_S:-0.0}
LLM_FOV_GATE_SOFT_MAX_TARGETS=${LLM_FOV_GATE_SOFT_MAX_TARGETS:-3}
LLM_FOV_GATE_BYPASS_DISTANCE=${LLM_FOV_GATE_BYPASS_DISTANCE:-true}
if [ "$ENABLE_VLA_MODE" = "1" ]; then
  LLM_FOV_GATE_MAX_DISTANCE_M=${LLM_FOV_GATE_MAX_DISTANCE_M:-35.0}
  LLM_FOV_GATE_STATE_MAX_DISTANCE_M=${LLM_FOV_GATE_STATE_MAX_DISTANCE_M:-35.0}
  LLM_FOV_GATE_REQUIRE_ALL_TRACKS=${LLM_FOV_GATE_REQUIRE_ALL_TRACKS:-false}
  LLM_FOV_GATE_PARTIAL_AFTER_S=${LLM_FOV_GATE_PARTIAL_AFTER_S:-0.5}
else
  LLM_FOV_GATE_MAX_DISTANCE_M=${LLM_FOV_GATE_MAX_DISTANCE_M:-20.0}
  LLM_FOV_GATE_STATE_MAX_DISTANCE_M=${LLM_FOV_GATE_STATE_MAX_DISTANCE_M:-20.0}
  LLM_FOV_GATE_REQUIRE_ALL_TRACKS=${LLM_FOV_GATE_REQUIRE_ALL_TRACKS:-true}
  LLM_FOV_GATE_PARTIAL_AFTER_S=${LLM_FOV_GATE_PARTIAL_AFTER_S:-2.0}
fi
LLM_GATE_STATUS_LOG_INTERVAL_S=${LLM_GATE_STATUS_LOG_INTERVAL_S:-5.0}
LLM_WAIT_IMAGE_ON_ENTRY=${LLM_WAIT_IMAGE_ON_ENTRY:-true}
LLM_DISTANCE_INIT_GRACE_S=${LLM_DISTANCE_INIT_GRACE_S:-2.0}
LLM_PRETRIGGER_ENABLE=${LLM_PRETRIGGER_ENABLE:-true}
LLM_PRETRIGGER_DISTANCE_M=${LLM_PRETRIGGER_DISTANCE_M:-35.0}
PENDING_SAFETY_ENABLE=${PENDING_SAFETY_ENABLE:-true}
PENDING_SAFETY_SPEED_SCALE=${PENDING_SAFETY_SPEED_SCALE:-0.35}
VLA_SHUTDOWN_GRACE_S=${VLA_SHUTDOWN_GRACE_S:-8}
VLA_STARTUP_STRICT=${VLA_STARTUP_STRICT:-true}
VLA_IMAGE_TOPIC=${VLA_IMAGE_TOPIC:-/myboat/sensors/cameras/front_camera/image_raw}
if [ -n "$INPUT_SOURCE_ARG" ]; then
  INPUT_SOURCE_KEY="${INPUT_SOURCE_ARG,,}"
  case "$INPUT_SOURCE_KEY" in
    lidar|lidar_wamv|pointcloud_bev|lidar_bev|bev)
      MC_SENSOR="lidar"
      VLA_IMAGE_SOURCE="pointcloud_bev"
      ;;
    global|global_traj|global_trajectory)
      MC_SENSOR="lidar"
      VLA_IMAGE_SOURCE="global"
      ;;
    ais|ais_traj|ais_trajectory)
      MC_SENSOR="lidar"
      VLA_IMAGE_SOURCE="ais"
      ;;
    camera|rgb)
      MC_SENSOR="camera"
      VLA_IMAGE_SOURCE="camera"
      ;;
    *)
      echo "[ERROR] Unknown input source: $INPUT_SOURCE_ARG (expected lidar, pointcloud_bev, camera, global, or ais)"
      usage
      exit 1
      ;;
  esac
elif [ -n "${VLA_IMAGE_SOURCE:-}" ]; then
  VLA_SOURCE_KEY="${VLA_IMAGE_SOURCE,,}"
  case "$VLA_SOURCE_KEY" in
    pointcloud_bev|lidar_bev|bev)
      MC_SENSOR="${MC_SENSOR:-lidar}"
      VLA_IMAGE_SOURCE="pointcloud_bev"
      ;;
    camera)
      MC_SENSOR="${MC_SENSOR:-camera}"
      VLA_IMAGE_SOURCE="camera"
      ;;
    global|global_traj|global_trajectory)
      MC_SENSOR="${MC_SENSOR:-lidar}"
      VLA_IMAGE_SOURCE="global"
      ;;
    ais|ais_traj|ais_trajectory)
      MC_SENSOR="${MC_SENSOR:-lidar}"
      VLA_IMAGE_SOURCE="ais"
      ;;
    *)
      echo "[ERROR] Unknown VLA_IMAGE_SOURCE: $VLA_IMAGE_SOURCE (expected camera, pointcloud_bev, global, or ais)"
      usage
      exit 1
      ;;
  esac
fi
MC_SENSOR=${MC_SENSOR:-lidar}
MC_SENSOR_KEY="${MC_SENSOR,,}"
case "$MC_SENSOR_KEY" in
  lidar|lidar_wamv)
    MC_SENSOR="lidar"
    DEFAULT_SENSOR_POINTCLOUD_TOPIC="/myboat/sensors/lidar_wamv/points"
    DEFAULT_VLA_IMAGE_SOURCE="pointcloud_bev"
    ;;
  camera|rgb)
    MC_SENSOR="camera"
    DEFAULT_SENSOR_POINTCLOUD_TOPIC="/myboat/sensors/lidar_wamv/points"
    DEFAULT_VLA_IMAGE_SOURCE="camera"
    ;;
  *)
    echo "[ERROR] Unknown MC_SENSOR: $MC_SENSOR (expected lidar or camera)"
    exit 1
    ;;
esac
SENSOR_POINTCLOUD_TOPIC=${SENSOR_POINTCLOUD_TOPIC:-$DEFAULT_SENSOR_POINTCLOUD_TOPIC}
VLA_IMAGE_SOURCE=${VLA_IMAGE_SOURCE:-$DEFAULT_VLA_IMAGE_SOURCE}
VLA_GLOBAL_HISTORY_S=${VLA_GLOBAL_HISTORY_S:-8.0}
VLA_GLOBAL_IMAGE_WIDTH=${VLA_GLOBAL_IMAGE_WIDTH:-1000}
VLA_GLOBAL_IMAGE_HEIGHT=${VLA_GLOBAL_IMAGE_HEIGHT:-1000}
VLA_GLOBAL_MIN_SPAN_M=${VLA_GLOBAL_MIN_SPAN_M:-30.0}
AIS_TARGET_ODOM_TOPIC=${AIS_TARGET_ODOM_TOPIC:-/target_boat/odom}
AIS_TARGET2_ODOM_TOPIC=${AIS_TARGET2_ODOM_TOPIC:-/target_boat_2/odom}
AIS_TARGET3_ODOM_TOPIC=${AIS_TARGET3_ODOM_TOPIC:-/target_boat_3/odom}
VLA_CAMERA_VIDEO_ENABLE=${VLA_CAMERA_VIDEO_ENABLE:-true}
VLA_CAMERA_VIDEO_FRAME_COUNT=${VLA_CAMERA_VIDEO_FRAME_COUNT:-4}
VLA_CAMERA_VIDEO_INTERVAL_S=${VLA_CAMERA_VIDEO_INTERVAL_S:-2.5}
VLA_CAMERA_VIDEO_WINDOW_S=${VLA_CAMERA_VIDEO_WINDOW_S:-10.0}
VLA_CAMERA_VIDEO_BUFFER_MAX_FRAMES=${VLA_CAMERA_VIDEO_BUFFER_MAX_FRAMES:-24}
VLA_BEV_POINTCLOUD_TOPIC=${VLA_BEV_POINTCLOUD_TOPIC:-$SENSOR_POINTCLOUD_TOPIC}
VLA_BEV_CLOUD_MAX_AGE_S=${VLA_BEV_CLOUD_MAX_AGE_S:-1.5}
VLA_BEV_RANGE_FORWARD_M=${VLA_BEV_RANGE_FORWARD_M:-60.0}
VLA_BEV_RANGE_BACKWARD_M=${VLA_BEV_RANGE_BACKWARD_M:-60.0}
VLA_BEV_RANGE_SIDE_M=${VLA_BEV_RANGE_SIDE_M:-35.0}
VLA_BEV_Z_MIN_M=${VLA_BEV_Z_MIN_M:--1.5}
VLA_BEV_Z_MAX_M=${VLA_BEV_Z_MAX_M:-4.0}
VLA_BEV_MAX_POINTS=${VLA_BEV_MAX_POINTS:-120000}
VLA_BEV_IMAGE_WIDTH=${VLA_BEV_IMAGE_WIDTH:-1000}
VLA_BEV_IMAGE_HEIGHT=${VLA_BEV_IMAGE_HEIGHT:-1000}
VLA_BEV_TRACK_BOX_MARGIN_M=${VLA_BEV_TRACK_BOX_MARGIN_M:-1.0}
VLA_BEV_FALLBACK_TO_CAMERA=${VLA_BEV_FALLBACK_TO_CAMERA:-false}
VLA_BEV_WARMUP_S=${VLA_BEV_WARMUP_S:-2.0}
VLA_BEV_WARMUP_MIN_FRAMES=${VLA_BEV_WARMUP_MIN_FRAMES:-3}
VLA_TARGET_HISTORY_WARMUP_S=${VLA_TARGET_HISTORY_WARMUP_S:-2.0}
VLA_TARGET_HISTORY_WARMUP_MIN_SAMPLES=${VLA_TARGET_HISTORY_WARMUP_MIN_SAMPLES:-3}
VLA_TARGET_HISTORY_WARMUP_MIN_READY_TARGETS=${VLA_TARGET_HISTORY_WARMUP_MIN_READY_TARGETS:-1}
VLA_PROMPT_TRACK_INPUT_ENABLE=${VLA_PROMPT_TRACK_INPUT_ENABLE:-false}
VLA_CAMERA_HFOV_DEG=${VLA_CAMERA_HFOV_DEG:-90.0}
VLA_SNAPSHOT_DIR=${VLA_SNAPSHOT_DIR:-$RESULT_DIR/vla_snapshots}
mkdir -p "$VLA_SNAPSHOT_DIR"

echo "trial,scenario,mode,seed,result,reason,duration_s,goal_distance_m,min_distance_m,truth_cpa_time_s,min_truth_tcpa_s,min_truth_dcpa_m,min_tcpa_s,min_dcpa_m,colreg_violation,nmf,ego_x,ego_y,ego_yaw,ego_speed,ego_goal_x,ego_goal_y,target_x,target_y,target_yaw,target_speed,target_turn_rate,target2_x,target2_y,target2_yaw,target2_speed,target2_turn_rate,target3_x,target3_y,target3_yaw,target3_speed,target3_turn_rate,trajectory_csv,vlm_truth_calls,vlm_truth_matches,vlm_truth_mismatches,vlm_truth_accuracy,vlm_truth_csv" > "$RESULT_CSV"

cleanup_ros() {
  pkill -f "roslaunch|gzserver|gzclient|rostopic echo /collision/vlm_check" >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do
    if pgrep -f "roslaunch|gzserver|gzclient" >/dev/null 2>&1; then
      sleep 0.5
    else
      break
    fi
  done
}

cleanup_and_exit() {
  cleanup_ros
  exit 130
}
trap cleanup_and_exit SIGINT SIGTERM

ensure_vla_python_wrapper() {
  local wrapper="$WORKSPACE/devel/.private/myboat_gazebo/lib/myboat_gazebo/colregs_llm_decision_node.py"
  local pybin="${ACTIVE_PYTHON_BIN:-}"
  if [ -z "$pybin" ] || [ ! -x "$pybin" ] || [ ! -f "$wrapper" ]; then
    return 0
  fi
  local current
  current="$(head -n 1 "$wrapper" 2>/dev/null || true)"
  local desired="#!$pybin"
  if [ "$current" = "$desired" ]; then
    return 0
  fi
  local tmp
  tmp="$(mktemp)"
  {
    echo "$desired"
    tail -n +2 "$wrapper"
  } > "$tmp"
  chmod --reference="$wrapper" "$tmp" 2>/dev/null || chmod +x "$tmp"
  mv "$tmp" "$wrapper"
}

scale_float() {
  python3 - "$1" "$2" <<'PY'
import sys
value = float(sys.argv[1])
scale = float(sys.argv[2])
print("%.6f" % (value * scale))
PY
}

effective_timeout() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys
base = float(sys.argv[1])
enable_vla = sys.argv[2] == "1"
speed_scale = max(1e-6, float(sys.argv[3]))
scale_raw = str(sys.argv[4]).strip().lower()
if not enable_vla:
    print("%.3f" % base)
elif scale_raw in ("", "auto"):
    print("%.3f" % (base / speed_scale))
else:
    print("%.3f" % (base * float(scale_raw)))
PY
}

scenario_random_vars() {
  local scenario="$1"
  local seed="$2"
  python3 - "$scenario" "$seed" <<'PY'
import math
import random
import sys

scenario = sys.argv[1]
seed = int(sys.argv[2])
rng = random.Random(seed)

def u(center, half):
    return center + rng.uniform(-half, half)

def speed(center, half, lo=0.05):
    return max(lo, center + rng.uniform(-half, half))

v = {
    "EGO_GOAL_X": -430.0,
    "EGO_GOAL_Y": 260.0,
    "USE_STATIC_BUOYS": 0,
    "ENABLE_EXTRA_BUOYS": 0,
    "SPAWN_TARGET_BOAT": 1,
    "CHANNEL_CONSTRAINTS_ENABLE": 0,
    "CHANNEL_LEFT_Y": 244.0,
    "CHANNEL_RIGHT_Y": 274.0,
    "CHANNEL_X_MIN": -470.0,
    "CHANNEL_X_MAX": -410.0,
    "CHANNEL_X_STEP": 6.0,
    "TARGET_TURN_RATE": 0.0,
    "TARGET2_TURN_RATE": 0.0,
    "TARGET3_TURN_RATE": 0.0,
}

if scenario == "HeadOn":
    v.update(
        ENABLE_MULTI_TARGETS=0,
        EGO_X=u(-470.0, 2.0),
        EGO_Y=u(260.0, 1.2),
        EGO_YAW=u(0.0, 0.04),
        EGO_SPEED=speed(0.80, 0.16),
        TARGET_X=u(-440.0, 4.0),
        TARGET_Y=u(258.0, 2.0),
        TARGET_YAW=u(math.pi, 0.06),
        TARGET_SPEED=speed(0.80, 0.16),
        TARGET2_X=u(-450.0, 2.0),
        TARGET2_Y=u(245.0, 1.0),
        TARGET2_YAW=2.3561945,
        TARGET2_SPEED=0.80,
        TARGET3_X=u(-450.0, 2.0),
        TARGET3_Y=u(275.0, 1.0),
        TARGET3_YAW=-2.3561945,
        TARGET3_SPEED=0.80,
    )
elif scenario == "Crossing":
    v.update(
        ENABLE_MULTI_TARGETS=0,
        EGO_X=u(-470.0, 2.0),
        EGO_Y=u(260.0, 1.5),
        EGO_YAW=u(0.0, 0.04),
        EGO_SPEED=speed(0.80, 0.16),
        TARGET_X=u(-455.0, 3.0),
        TARGET_Y=u(245.0, 4.0),
        TARGET_YAW=u(math.pi / 2.0, 0.06),
        TARGET_SPEED=speed(1.00, 0.24),
        TARGET2_X=u(-450.0, 2.0),
        TARGET2_Y=u(245.0, 1.0),
        TARGET2_YAW=2.3561945,
        TARGET2_SPEED=0.80,
        TARGET3_X=u(-450.0, 2.0),
        TARGET3_Y=u(275.0, 1.0),
        TARGET3_YAW=-2.3561945,
        TARGET3_SPEED=0.80,
    )
elif scenario == "Overtaking":
    v.update(
        ENABLE_MULTI_TARGETS=0,
        EGO_X=u(-470.0, 2.0),
        EGO_Y=u(260.0, 1.2),
        EGO_YAW=u(0.0, 0.03),
        EGO_SPEED=speed(1.30, 0.20),
        TARGET_X=u(-450.0, 4.0),
        TARGET_Y=u(260.0, 1.2),
        TARGET_YAW=u(0.0, 0.03),
        TARGET_SPEED=speed(0.60, 0.16),
        TARGET2_X=u(-450.0, 2.0),
        TARGET2_Y=u(245.0, 1.0),
        TARGET2_YAW=2.3561945,
        TARGET2_SPEED=0.80,
        TARGET3_X=u(-450.0, 2.0),
        TARGET3_Y=u(275.0, 1.0),
        TARGET3_YAW=-2.3561945,
        TARGET3_SPEED=0.80,
    )
elif scenario == "MultiShip":
    v.update(
        ENABLE_MULTI_TARGETS=1,
        EGO_X=u(-470.0, 2.0),
        EGO_Y=u(260.0, 1.2),
        EGO_YAW=u(0.0, 0.04),
        EGO_SPEED=speed(1.00, 0.20),
        TARGET_X=u(-440.0, 4.0),
        TARGET_Y=u(260.0, 2.0),
        TARGET_YAW=u(math.pi, 0.06),
        TARGET_SPEED=speed(1.00, 0.20),
        TARGET_TURN_RATE=u(0.0, 0.006),
        TARGET2_X=u(-440.0, 4.0),
        TARGET2_Y=u(246.0, 2.0),
        TARGET2_YAW=u(2.3561945, 0.08),
        TARGET2_SPEED=speed(0.66, 0.16),
        TARGET2_TURN_RATE=u(0.010, 0.006),
        TARGET3_X=u(-440.0, 4.0),
        TARGET3_Y=u(271.0, 2.0),
        TARGET3_YAW=u(-2.9561945, 0.08),
        TARGET3_SPEED=speed(2.00, 0.30),
        TARGET3_TURN_RATE=u(0.0, 0.006),
    )
elif scenario == "NarrowMulti":
    v.update(
        ENABLE_MULTI_TARGETS=1,
        CHANNEL_CONSTRAINTS_ENABLE=1,
        EGO_X=u(-470.0, 2.0),
        EGO_Y=u(260.0, 1.2),
        EGO_YAW=u(0.0, 0.04),
        EGO_SPEED=speed(1.00, 0.20),
        TARGET_X=u(-440.0, 4.0),
        TARGET_Y=u(260.0, 2.0),
        TARGET_YAW=u(math.pi, 0.06),
        TARGET_SPEED=speed(1.00, 0.20),
        TARGET_TURN_RATE=u(0.0, 0.006),
        TARGET2_X=u(-440.0, 4.0),
        TARGET2_Y=u(246.0, 2.0),
        TARGET2_YAW=u(2.3561945, 0.08),
        TARGET2_SPEED=speed(0.66, 0.16),
        TARGET2_TURN_RATE=u(0.010, 0.006),
        TARGET3_X=u(-440.0, 4.0),
        TARGET3_Y=u(271.0, 2.0),
        TARGET3_YAW=u(-2.9561945, 0.08),
        TARGET3_SPEED=speed(2.00, 0.30),
        TARGET3_TURN_RATE=u(0.0, 0.006),
    )
else:
    raise SystemExit("unknown scenario")

for key in sorted(v):
    val = v[key]
    if isinstance(val, float):
        print("%s=%.7f" % (key, val))
    else:
        print("%s=%s" % (key, val))
PY
}

monitor_trial() {
  local goal_x="$1"
  local goal_y="$2"
  local goal_tol="$3"
  local collision_distance="$4"
  local timeout_s="$5"
  local enable_multi="$6"
  local near_miss_distance="$7"
  local scenario_name="$8"
  local colreg_pass_clearance="$9"
  local colreg_substantial_offset="${10}"
  local colreg_eval_distance="${11}"
  local colreg_multi_allowed_violations="${12}"
  local clock_stall_s="${13}"
  local vlm_truth_csv="${14}"
  local vlm_truth_enable="${15}"
  local vlm_truth_tcpa_horizon_s="${16}"
  local vlm_truth_dcpa_danger_m="${17}"
  local vlm_truth_close_range_m="${18}"
  local truth_cpa_warmup_s="${19}"
  local truth_cpa_min_range_m="${20}"
  python3 - "$goal_x" "$goal_y" "$goal_tol" "$collision_distance" "$timeout_s" "$enable_multi" "$near_miss_distance" "$scenario_name" "$colreg_pass_clearance" "$colreg_substantial_offset" "$colreg_eval_distance" "$colreg_multi_allowed_violations" "$clock_stall_s" "$vlm_truth_csv" "$vlm_truth_enable" "$vlm_truth_tcpa_horizon_s" "$vlm_truth_dcpa_danger_m" "$vlm_truth_close_range_m" "$truth_cpa_warmup_s" "$truth_cpa_min_range_m" <<'PY'
import csv
import json
import math
import os
import sys
import time

import rospy
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64, String

goal_x = float(sys.argv[1])
goal_y = float(sys.argv[2])
goal_tol = float(sys.argv[3])
collision_distance = float(sys.argv[4])
timeout_s = float(sys.argv[5])
enable_multi = sys.argv[6] == "1"
near_miss_distance = float(sys.argv[7])
scenario_name = str(sys.argv[8])
colreg_pass_clearance = float(sys.argv[9])
colreg_substantial_offset = float(sys.argv[10])
colreg_eval_distance = float(sys.argv[11])
colreg_multi_allowed_violations = int(float(sys.argv[12]))
clock_stall_s = float(sys.argv[13])
vlm_truth_csv = str(sys.argv[14])
vlm_truth_enable = str(sys.argv[15]).strip().lower() in ("1", "true", "yes", "on")
vlm_truth_tcpa_horizon_s = float(sys.argv[16])
vlm_truth_dcpa_danger_m = float(sys.argv[17])
vlm_truth_close_range_m = float(sys.argv[18])
truth_cpa_warmup_s = float(sys.argv[19])
truth_cpa_min_range_m = float(sys.argv[20])

state = {
    "odom": None,
    "ego_pose": None,
    "targets": {},
    "encounters": {},
    "goal_dist": float("inf"),
    "min_dist": float("inf"),
    "truth_cpa_time": float("inf"),
    "min_truth_dcpa": float("inf"),
    "collision": False,
    "near_miss_count": 0,
    "near_miss_active_by_target": {},
    "mpc_near_miss_count": 0,
    "mpc_near_miss_event_ids": set(),
    "last_vlm_call_seq": None,
    "vlm_truth_calls": 0,
    "vlm_truth_matches": 0,
    "vlm_truth_mismatches": 0,
    "last_clock_wall": None,
    "last_odom_wall": None,
}

truth_fp = None
truth_writer = None
if vlm_truth_enable and vlm_truth_csv:
    parent = os.path.dirname(vlm_truth_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)
    truth_fp = open(vlm_truth_csv, "w", newline="")
    truth_writer = csv.writer(truth_fp)
    truth_writer.writerow(
        [
            "stamp_s",
            "llm_call_seq",
            "vlm_course_action",
            "vlm_speed_action",
            "standard_course_action",
            "action_match",
            "selected_target",
            "target_name",
            "range_m",
            "bearing_deg",
            "tcpa_s",
            "dcpa_m",
            "encounter",
            "course_diff_deg",
            "closing",
            "reason",
        ]
    )
    truth_fp.flush()

def norm_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def odom_truth(msg):
    p = msg.pose.pose.position
    tw = msg.twist.twist
    return {
        "x": float(p.x),
        "y": float(p.y),
        "yaw": yaw_from_quat(msg.pose.pose.orientation),
        "vx": float(tw.linear.x),
        "vy": float(tw.linear.y),
    }

def forward(yaw):
    return (math.cos(yaw), math.sin(yaw))

def left(yaw):
    return (-math.sin(yaw), math.cos(yaw))

def right(yaw):
    return (math.sin(yaw), -math.cos(yaw))

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

def rel_body(from_xy, to_xy, yaw):
    vec = (to_xy[0] - from_xy[0], to_xy[1] - from_xy[1])
    return dot(vec, forward(yaw)), dot(vec, left(yaw))

def canonical_course_action(action):
    text = str(action or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "KEEP": "KEEP_COURSE",
        "STAND_ON": "KEEP_COURSE",
        "STAND_ON_KEEP_COURSE": "KEEP_COURSE",
        "MAINTAIN_COURSE": "KEEP_COURSE",
        "GIVE_WAY": "TURN_STARBOARD",
        "ALTER_TO_STARBOARD": "TURN_STARBOARD",
        "STARBOARD": "TURN_STARBOARD",
        "RIGHT": "TURN_STARBOARD",
        "TURN_RIGHT": "TURN_STARBOARD",
        "RIGHT_TURN": "TURN_STARBOARD",
        "ALTER_TO_PORT": "TURN_PORT",
        "PORT": "TURN_PORT",
        "LEFT": "TURN_PORT",
        "TURN_LEFT": "TURN_PORT",
        "LEFT_TURN": "TURN_PORT",
    }
    text = aliases.get(text, text)
    return text if text in ("KEEP_COURSE", "TURN_STARBOARD", "TURN_PORT") else "KEEP_COURSE"

def decision_field(decision, key):
    value = decision.get(key, "")
    if value:
        return str(value).strip()
    constraints = decision.get("trajectory_constraints", {})
    if isinstance(constraints, dict):
        value = constraints.get(key, "")
        if value:
            return str(value).strip()
    return ""

def positive_int(value):
    try:
        out = int(value)
    except Exception:
        return None
    return out if out > 0 else None

def finite_or_blank(value):
    try:
        out = float(value)
    except Exception:
        return ""
    return "%.3f" % out if math.isfinite(out) else ""

def truth_metrics(target_name, ego, target):
    rx = target["x"] - ego["x"]
    ry = target["y"] - ego["y"]
    rvx = target.get("vx", 0.0) - ego.get("vx", 0.0)
    rvy = target.get("vy", 0.0) - ego.get("vy", 0.0)
    range_m = math.hypot(rx, ry)
    rel_forward, rel_left = rel_body((ego["x"], ego["y"]), (target["x"], target["y"]), ego["yaw"])
    bearing_deg = math.degrees(math.atan2(rel_left, rel_forward))
    rel_speed_sq = rvx * rvx + rvy * rvy
    rel_speed_mps = math.sqrt(rel_speed_sq)
    if rel_speed_sq > 1e-8:
        tcpa_s = -((rx * rvx + ry * rvy) / rel_speed_sq)
    else:
        tcpa_s = float("inf")
    if math.isfinite(tcpa_s) and tcpa_s >= 0.0:
        cpa_x = rx + rvx * tcpa_s
        cpa_y = ry + rvy * tcpa_s
    else:
        cpa_x = rx
        cpa_y = ry
    dcpa_m = math.hypot(cpa_x, cpa_y)
    closing = range_m > 1e-6 and (rx * rvx + ry * rvy) < 0.0
    ego_course = math.atan2(ego.get("vy", 0.0), ego.get("vx", 0.0))
    if math.hypot(ego.get("vx", 0.0), ego.get("vy", 0.0)) < 0.05:
        ego_course = ego["yaw"]
    target_course = math.atan2(target.get("vy", 0.0), target.get("vx", 0.0))
    if math.hypot(target.get("vx", 0.0), target.get("vy", 0.0)) < 0.05:
        target_course = target["yaw"]
    course_diff_deg = abs(math.degrees(norm_pi(ego_course - target_course)))

    abs_bearing = abs(bearing_deg)
    if abs_bearing <= 15.0 and course_diff_deg >= 135.0:
        encounter = "head_on"
    elif rel_forward > 0.0 and course_diff_deg <= 35.0:
        encounter = "overtaking"
    elif bearing_deg < 0.0:
        encounter = "crossing_give_way"
    elif bearing_deg > 0.0:
        encounter = "crossing_stand_on"
    else:
        encounter = "unknown"

    future_risk = (
        math.isfinite(tcpa_s)
        and 0.0 <= tcpa_s <= vlm_truth_tcpa_horizon_s
        and dcpa_m <= vlm_truth_dcpa_danger_m
    )
    close_risk = range_m <= vlm_truth_close_range_m and closing
    immediate_risk = range_m <= max(collision_distance, vlm_truth_dcpa_danger_m)
    return {
        "target_name": target_name,
        "range_m": range_m,
        "bearing_deg": bearing_deg,
        "relative_speed_mps": rel_speed_mps,
        "tcpa_s": tcpa_s,
        "dcpa_m": dcpa_m,
        "encounter": encounter,
        "course_diff_deg": course_diff_deg,
        "closing": closing,
        "future_risk": future_risk,
        "close_risk": close_risk,
        "immediate_risk": immediate_risk,
    }

def standard_decision_from_truth(metrics):
    if not metrics:
        return "KEEP_COURSE", None, "no_truth_target"

    def turn_away_from_bearing(m, center_default="TURN_STARBOARD"):
        bearing = float(m.get("bearing_deg", 0.0))
        if bearing < -2.0:
            return "TURN_PORT", "target_starboard_turn_port"
        if bearing > 2.0:
            return "TURN_STARBOARD", "target_port_turn_starboard"
        return center_default, "target_center_%s" % center_default.lower()

    def risk_key(m):
        risky = bool(m["future_risk"] or m["close_risk"] or m["immediate_risk"])
        tcpa = m["tcpa_s"] if math.isfinite(m["tcpa_s"]) and m["tcpa_s"] >= 0.0 else 1e9
        return (0 if risky else 1, m["dcpa_m"], tcpa, m["range_m"])

    selected = sorted(metrics, key=risk_key)[0]
    if not (selected["future_risk"] or selected["close_risk"] or selected["immediate_risk"]):
        return "KEEP_COURSE", selected, "no_cpa_or_close_risk"

    encounter = selected["encounter"]
    if encounter == "head_on":
        return "TURN_STARBOARD", selected, "head_on_starboard"
    if encounter == "crossing_give_way":
        action, reason = turn_away_from_bearing(selected)
        return action, selected, "crossing_give_way_" + reason
    if encounter == "overtaking":
        action, reason = turn_away_from_bearing(selected, center_default="TURN_PORT")
        return action, selected, "overtaking_" + reason
    if encounter == "crossing_stand_on":
        if selected["immediate_risk"]:
            action, reason = turn_away_from_bearing(selected)
            return action, selected, "stand_on_immediate_risk_" + reason
        return "KEEP_COURSE", selected, "port_crossing_stand_on"
    action, reason = turn_away_from_bearing(selected)
    return action, selected, "unclassified_risk_" + reason

def vlm_truth_cb(msg):
    if not vlm_truth_enable or truth_writer is None:
        return
    try:
        decision = json.loads(msg.data)
    except Exception:
        return
    if not isinstance(decision, dict):
        return
    call_seq = positive_int(decision.get("llm_call_seq", decision.get("debug_call_id", -1)))
    if call_seq is None or call_seq == state["last_vlm_call_seq"]:
        return
    state["last_vlm_call_seq"] = call_seq

    ego = state.get("ego_pose")
    metrics = []
    if ego is not None:
        for target_name, target in sorted(state["targets"].items()):
            metrics.append(truth_metrics(target_name, ego, target))
    standard_action, selected, reason = standard_decision_from_truth(metrics)
    selected_name = selected["target_name"] if selected else ""
    vlm_course_action = canonical_course_action(decision_field(decision, "course_action"))
    vlm_speed_action = str(decision_field(decision, "speed_action") or "").strip()
    action_match = vlm_course_action == standard_action

    state["vlm_truth_calls"] += 1
    if action_match:
        state["vlm_truth_matches"] += 1
    else:
        state["vlm_truth_mismatches"] += 1

    stamp_s = decision.get("stamp", rospy.Time.now().to_sec())
    rows = metrics or [
        {
            "target_name": "",
            "range_m": float("nan"),
            "bearing_deg": float("nan"),
            "tcpa_s": float("nan"),
            "dcpa_m": float("nan"),
            "encounter": "",
            "course_diff_deg": float("nan"),
            "closing": False,
        }
    ]
    for m in rows:
        truth_writer.writerow(
            [
                finite_or_blank(stamp_s),
                str(call_seq),
                vlm_course_action,
                vlm_speed_action,
                standard_action,
                "true" if action_match else "false",
                "true" if m["target_name"] == selected_name else "false",
                m["target_name"],
                finite_or_blank(m["range_m"]),
                finite_or_blank(m["bearing_deg"]),
                finite_or_blank(m["tcpa_s"]),
                finite_or_blank(m["dcpa_m"]),
                m["encounter"],
                finite_or_blank(m["course_diff_deg"]),
                "true" if m["closing"] else "false",
                reason,
            ]
        )
    truth_fp.flush()

def update_truth_cpa_stats(elapsed_s):
    if elapsed_s < truth_cpa_warmup_s:
        return
    ego = state.get("ego_pose")
    if ego is None:
        return
    for target_name, target in list(state["targets"].items()):
        m = truth_metrics(target_name, ego, target)
        range_m = float(m["range_m"])
        if not math.isfinite(range_m):
            continue
        if truth_cpa_min_range_m > 0.0 and range_m < truth_cpa_min_range_m:
            continue
        if range_m < state["min_truth_dcpa"]:
            state["min_truth_dcpa"] = range_m
            state["truth_cpa_time"] = elapsed_s

def classify_encounter(target_name, ego, target):
    scenario = scenario_name.lower()
    if scenario == "headon":
        return "head_on"
    if scenario == "crossing":
        return "crossing_give_way"
    if scenario == "overtaking":
        return "overtaking"

    ef = forward(ego["yaw"])
    rel = (target["x"] - ego["x"], target["y"] - ego["y"])
    rel_forward = dot(rel, ef)
    rel_lateral = dot(rel, left(ego["yaw"]))
    course_diff = abs(math.degrees(norm_pi(ego["yaw"] - target["yaw"])))

    if course_diff >= 135.0 and rel_forward > 0.0 and abs(rel_lateral) <= 8.0:
        return "head_on"
    if course_diff <= 35.0 and rel_forward > 0.0:
        return "overtaking"
    if rel_lateral < 0.0:
        return "crossing_give_way"
    return "stand_on_or_unclassified"

def ensure_encounter(target_name, ego, target):
    enc = state["encounters"].get(target_name)
    if enc is not None:
        return enc
    kind = classify_encounter(target_name, ego, target)
    enc = {
        "kind": kind,
        "ego_init": dict(ego),
        "target_init": dict(target),
        "min_dist": float("inf"),
        "rel_at_min": (0.0, 0.0),
        "target_forward_at_min": 0.0,
        "max_starboard_offset": 0.0,
        "min_overtake_lateral": float("inf"),
    }
    state["encounters"][target_name] = enc
    return enc

def update_colreg_metrics():
    ego = state.get("ego_pose")
    if ego is None:
        return
    for target_name, target in list(state["targets"].items()):
        enc = ensure_encounter(target_name, ego, target)
        ego_init = enc["ego_init"]
        target_init = enc["target_init"]
        ego_xy = (ego["x"], ego["y"])
        target_xy = (target["x"], target["y"])
        rel = (target_xy[0] - ego_xy[0], target_xy[1] - ego_xy[1])
        dist = math.hypot(rel[0], rel[1])

        starboard_offset = dot(
            (ego_xy[0] - ego_init["x"], ego_xy[1] - ego_init["y"]),
            right(ego_init["yaw"]),
        )
        if starboard_offset > enc["max_starboard_offset"]:
            enc["max_starboard_offset"] = starboard_offset

        if dist < enc["min_dist"]:
            enc["min_dist"] = dist
            enc["rel_at_min"] = rel_body(ego_xy, target_xy, ego_init["yaw"])
            enc["target_forward_at_min"] = dot(
                (ego_xy[0] - target_xy[0], ego_xy[1] - target_xy[1]),
                forward(target_init["yaw"]),
            )

        ego_forward_of_target = dot(
            (ego_xy[0] - target_xy[0], ego_xy[1] - target_xy[1]),
            forward(target_init["yaw"]),
        )
        target_lateral = dot(
            (ego_xy[0] - target_xy[0], ego_xy[1] - target_xy[1]),
            left(target_init["yaw"]),
        )
        if ego_forward_of_target >= -1.0:
            enc["min_overtake_lateral"] = min(
                enc["min_overtake_lateral"],
                abs(target_lateral),
            )

def colreg_violation():
    violations = 0
    evaluated = 0
    scenario = scenario_name.lower()
    for enc in state["encounters"].values():
        kind = enc.get("kind")
        min_dist = enc.get("min_dist", float("inf"))
        if (not math.isfinite(min_dist)) or min_dist > colreg_eval_distance:
            continue
        target_violated = False
        if kind == "head_on":
            # Rule 14: alter to starboard and pass port-to-port. For ego, the
            # target should remain on/pass to port at closest approach.
            target_lateral_at_min = enc["rel_at_min"][1]
            if (
                enc["max_starboard_offset"] < colreg_substantial_offset
                and target_lateral_at_min <= -0.5
            ):
                target_violated = True
        elif kind == "crossing_give_way":
            # Rule 15/16: with target on own starboard, give way and avoid
            # crossing ahead. At CPA ego should be astern of the target.
            ego_forward_of_target = enc["target_forward_at_min"]
            if ego_forward_of_target > 1.0 and min_dist < max(colreg_pass_clearance, near_miss_distance):
                target_violated = True
        elif kind == "overtaking":
            # Rule 13: overtaking vessel keeps out of the way. Require a
            # meaningful lateral passing clearance once ego reaches abeam/ahead.
            lateral = enc["min_overtake_lateral"]
            if (not math.isfinite(lateral)) or lateral < colreg_pass_clearance:
                target_violated = True
        if kind in ("head_on", "crossing_give_way", "overtaking"):
            evaluated += 1
        if target_violated:
            violations += 1
    if scenario in ("multiship", "narrowmulti"):
        return violations > max(0, colreg_multi_allowed_violations)
    return violations > 0

def odom_cb(msg):
    pose = odom_truth(msg)
    state["odom"] = (pose["x"], pose["y"])
    state["last_odom_wall"] = time.monotonic()
    state["ego_pose"] = pose
    state["goal_dist"] = math.hypot(goal_x - pose["x"], goal_y - pose["y"])

def target_odom_cb_factory(target_name):
    def target_odom_cb(msg):
        state["targets"][target_name] = odom_truth(msg)
    return target_odom_cb

def dist_cb_factory(target_name):
    def dist_cb(msg):
        d = float(msg.data)
        if d < state["min_dist"]:
            state["min_dist"] = d
        if d <= collision_distance:
            state["collision"] = True
        near_now = (not state["collision"]) and (collision_distance < d < near_miss_distance)
        was_near = bool(state["near_miss_active_by_target"].get(target_name, False))
        if near_now and not was_near:
            state["near_miss_count"] += 1
        state["near_miss_active_by_target"][target_name] = near_now
    return dist_cb

def mpc_near_miss_cb(msg):
    try:
        data = json.loads(msg.data)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    if str(data.get("event", "")) != "mpc_near_miss_detour":
        return
    event_id = data.get("event_id")
    key = "mpc:%s" % str(event_id)
    if key in state["mpc_near_miss_event_ids"]:
        return
    state["mpc_near_miss_event_ids"].add(key)
    if not state["collision"]:
        state["near_miss_count"] += 1
        state["mpc_near_miss_count"] += 1

def clock_cb(msg):
    state["last_clock_wall"] = time.monotonic()

rospy.init_node("monte_carlo_trial_monitor", anonymous=True)
rospy.Subscriber("/clock", Clock, clock_cb, queue_size=20)
rospy.Subscriber("/myboat/odom", Odometry, odom_cb, queue_size=20)
rospy.Subscriber("/target_boat/odom", Odometry, target_odom_cb_factory("target_boat"), queue_size=20)
rospy.Subscriber("/collision/debug/absolute_distance/target_boat", Float64, dist_cb_factory("target_boat"), queue_size=20)
rospy.Subscriber("/collision/mpc_near_miss_event", String, mpc_near_miss_cb, queue_size=20)
rospy.Subscriber("/collision/llm_decision", String, vlm_truth_cb, queue_size=20)
if enable_multi:
    rospy.Subscriber("/target_boat_2/odom", Odometry, target_odom_cb_factory("target_boat_2"), queue_size=20)
    rospy.Subscriber("/target_boat_3/odom", Odometry, target_odom_cb_factory("target_boat_3"), queue_size=20)
    rospy.Subscriber("/collision/debug/absolute_distance/target_boat_2", Float64, dist_cb_factory("target_boat_2"), queue_size=20)
    rospy.Subscriber("/collision/debug/absolute_distance/target_boat_3", Float64, dist_cb_factory("target_boat_3"), queue_size=20)

start = time.monotonic()
reason = "timeout"
while not rospy.is_shutdown():
    elapsed = time.monotonic() - start
    now_wall = time.monotonic()
    update_truth_cpa_stats(elapsed)
    update_colreg_metrics()
    if state["collision"]:
        reason = "collision_distance"
        break
    if state["odom"] is not None and state["goal_dist"] <= goal_tol:
        reason = "goal_reached"
        break
    if elapsed >= clock_stall_s:
        last_clock = state.get("last_clock_wall")
        if last_clock is None or now_wall - last_clock > clock_stall_s:
            reason = "clock_stalled"
            break
        last_odom = state.get("last_odom_wall")
        if last_odom is not None and now_wall - last_odom > clock_stall_s:
            reason = "odom_stalled"
            break
    if elapsed >= timeout_s:
        reason = "timeout"
        break
    time.sleep(0.1)

duration = time.monotonic() - start
min_dist = state["min_dist"]
if not math.isfinite(min_dist):
    min_dist = -1.0
truth_cpa_time = state["truth_cpa_time"]
if not math.isfinite(truth_cpa_time):
    truth_cpa_time = -1.0
min_truth_dcpa = state["min_truth_dcpa"]
if not math.isfinite(min_truth_dcpa):
    min_truth_dcpa = -1.0
goal_dist = state["goal_dist"]
if not math.isfinite(goal_dist):
    goal_dist = -1.0
result = "SUCCESS" if reason == "goal_reached" and not state["collision"] else "FAIL"
nmf = 0 if state["collision"] else int(state["near_miss_count"])
colreg_violation = colreg_violation()
vlm_truth_calls = int(state["vlm_truth_calls"])
vlm_truth_matches = int(state["vlm_truth_matches"])
vlm_truth_mismatches = int(state["vlm_truth_mismatches"])
vlm_truth_accuracy = (
    float(vlm_truth_matches) / float(vlm_truth_calls)
    if vlm_truth_calls > 0
    else -1.0
)
if truth_fp is not None:
    truth_fp.flush()
print(
    "%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%d,%d,%d,%d,%.3f"
    % (
        result,
        reason,
        duration,
        goal_dist,
        min_dist,
        truth_cpa_time,
        min_truth_dcpa,
        "true" if colreg_violation else "false",
        nmf,
        vlm_truth_calls,
        vlm_truth_matches,
        vlm_truth_mismatches,
        vlm_truth_accuracy,
    )
)
PY
}

launch_vla_if_needed() {
  if [ "$ENABLE_VLA_MODE" != "1" ]; then
    VLA_PID=""
    return
  fi

  if [ "$ENABLE_LOCAL_LLM_MODE" != "1" ] && [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "[WARN] DASHSCOPE_API_KEY is empty; VLM mode may fail to call backend."
  fi

  ensure_vla_python_wrapper

  roslaunch exploration_manager collision_llm.launch \
    python_launch_prefix:="$ACTIVE_PYTHON_BIN" \
    llm_backend:="$CLASSIFIER_BACKEND" \
    api_url:="$LLM_API_URL" \
    api_model:="$QWEN_VL_MODEL" \
    api_max_output_tokens:="$LLM_API_MAX_OUTPUT_TOKENS" \
    api_force_json_mode:="$LLM_API_FORCE_JSON_MODE" \
    api_repeat_retry_enable:="$LLM_API_REPEAT_RETRY_ENABLE" \
    api_repeat_ngram_words:="$LLM_API_REPEAT_NGRAM_WORDS" \
    api_repeat_ngram_min_count:="$LLM_API_REPEAT_NGRAM_MIN_COUNT" \
    api_no_json_retry_chars:="$LLM_API_NO_JSON_RETRY_CHARS" \
    debug_enable:="$LLM_DEBUG_ENABLE" \
    terminal_llm_io_enable:="$LLM_TERMINAL_IO_ENABLE" \
    terminal_clean_mode:="$LLM_TERMINAL_CLEAN_MODE" \
    llm_io_log_path:="$RESULT_DIR/vla_trial_${trial}_io.txt" \
    vla_event_log_path:="$RESULT_DIR/vla_trial_${trial}_events.txt" \
    llm_gate_status_log_interval_s:="$LLM_GATE_STATUS_LOG_INTERVAL_S" \
    vlm_check_enable:="$VLM_CHECK_ENABLE" \
    llm_use_image_input:="$LLM_USE_IMAGE_INPUT" \
    vla_prompt_track_input_enable:="$VLA_PROMPT_TRACK_INPUT_ENABLE" \
    vla_prompt_mode:="$VLA_PROMPT_MODE" \
    vla_image_topic:="$VLA_IMAGE_TOPIC" \
    vla_image_source:="$VLA_IMAGE_SOURCE" \
    vla_global_history_s:="$VLA_GLOBAL_HISTORY_S" \
    vla_global_image_width:="$VLA_GLOBAL_IMAGE_WIDTH" \
    vla_global_image_height:="$VLA_GLOBAL_IMAGE_HEIGHT" \
    vla_global_min_span_m:="$VLA_GLOBAL_MIN_SPAN_M" \
    ais_target_odom_topic:="$AIS_TARGET_ODOM_TOPIC" \
    ais_target2_odom_topic:="$AIS_TARGET2_ODOM_TOPIC" \
    ais_target3_odom_topic:="$AIS_TARGET3_ODOM_TOPIC" \
    vla_goal_x:="$EGO_GOAL_X" \
    vla_goal_y:="$EGO_GOAL_Y" \
    vla_camera_video_enable:="$VLA_CAMERA_VIDEO_ENABLE" \
    vla_camera_video_frame_count:="$VLA_CAMERA_VIDEO_FRAME_COUNT" \
    vla_camera_video_interval_s:="$VLA_CAMERA_VIDEO_INTERVAL_S" \
    vla_camera_video_window_s:="$VLA_CAMERA_VIDEO_WINDOW_S" \
    vla_camera_video_buffer_max_frames:="$VLA_CAMERA_VIDEO_BUFFER_MAX_FRAMES" \
    vla_bev_pointcloud_topic:="$VLA_BEV_POINTCLOUD_TOPIC" \
    vla_bev_cloud_max_age_s:="$VLA_BEV_CLOUD_MAX_AGE_S" \
    vla_bev_range_forward_m:="$VLA_BEV_RANGE_FORWARD_M" \
    vla_bev_range_backward_m:="$VLA_BEV_RANGE_BACKWARD_M" \
    vla_bev_range_side_m:="$VLA_BEV_RANGE_SIDE_M" \
    vla_bev_z_min_m:="$VLA_BEV_Z_MIN_M" \
    vla_bev_z_max_m:="$VLA_BEV_Z_MAX_M" \
    vla_bev_max_points:="$VLA_BEV_MAX_POINTS" \
    vla_bev_image_width:="$VLA_BEV_IMAGE_WIDTH" \
    vla_bev_image_height:="$VLA_BEV_IMAGE_HEIGHT" \
    vla_bev_track_box_margin_m:="$VLA_BEV_TRACK_BOX_MARGIN_M" \
    vla_bev_fallback_to_camera:="$VLA_BEV_FALLBACK_TO_CAMERA" \
    vla_bev_warmup_s:="$VLA_BEV_WARMUP_S" \
    vla_bev_warmup_min_frames:="$VLA_BEV_WARMUP_MIN_FRAMES" \
    vla_target_history_warmup_s:="$VLA_TARGET_HISTORY_WARMUP_S" \
    vla_target_history_warmup_min_samples:="$VLA_TARGET_HISTORY_WARMUP_MIN_SAMPLES" \
    vla_target_history_warmup_min_ready_targets:="$VLA_TARGET_HISTORY_WARMUP_MIN_READY_TARGETS" \
    vla_camera_hfov_deg:="$VLA_CAMERA_HFOV_DEG" \
    vla_image_path:= \
    vla_image_url:= \
    vla_snapshot_on_trigger:=true \
    vla_snapshot_dir:="$VLA_SNAPSHOT_DIR" \
    vla_image_buffer_duration_s:="$VLA_IMAGE_BUFFER_DURATION_S" \
    vla_image_buffer_max_frames:="$VLA_IMAGE_BUFFER_MAX_FRAMES" \
    vla_annotated_snapshot_enable:="$VLA_ANNOTATED_SNAPSHOT_ENABLE" \
    vla_trial_index:="$trial" \
    llm_fov_gate_enable:="$LLM_FOV_GATE_ENABLE" \
    llm_fov_gate_margin_deg:="$LLM_FOV_GATE_MARGIN_DEG" \
    llm_fov_gate_min_forward_m:="$LLM_FOV_GATE_MIN_FORWARD_M" \
    llm_fov_gate_max_distance_m:="$LLM_FOV_GATE_MAX_DISTANCE_M" \
    llm_fov_gate_state_max_distance_m:="$LLM_FOV_GATE_STATE_MAX_DISTANCE_M" \
    llm_fov_gate_min_visible_tracks:="$LLM_FOV_GATE_MIN_VISIBLE_TRACKS" \
    llm_fov_gate_require_all_tracks:="$LLM_FOV_GATE_REQUIRE_ALL_TRACKS" \
    llm_require_unknown_track_in_fov:="$LLM_REQUIRE_UNKNOWN_TRACK_IN_FOV" \
    llm_fov_gate_state_fallback_enable:="$LLM_FOV_GATE_STATE_FALLBACK_ENABLE" \
    llm_fov_gate_state_fallback_after_s:="$LLM_FOV_GATE_STATE_FALLBACK_AFTER_S" \
    llm_fov_gate_soft_enable:="$LLM_FOV_GATE_SOFT_ENABLE" \
    llm_fov_gate_soft_after_s:="$LLM_FOV_GATE_SOFT_AFTER_S" \
    llm_fov_gate_soft_max_targets:="$LLM_FOV_GATE_SOFT_MAX_TARGETS" \
    llm_fov_gate_partial_after_s:="$LLM_FOV_GATE_PARTIAL_AFTER_S" \
    llm_fov_gate_bypass_distance:="$LLM_FOV_GATE_BYPASS_DISTANCE" \
    llm_trigger_distance_m:="$LLM_TRIGGER_DISTANCE_M" \
    sensor_front_hazard_enable:="$SENSOR_FRONT_HAZARD_ENABLE" \
    sensor_front_hazard_range_m:="$SENSOR_FRONT_HAZARD_RANGE_M" \
    sensor_front_hazard_bearing_deg:="$SENSOR_FRONT_HAZARD_BEARING_DEG" \
    sensor_front_hazard_min_forward_m:="$SENSOR_FRONT_HAZARD_MIN_FORWARD_M" \
    llm_pretrigger_enable:="$LLM_PRETRIGGER_ENABLE" \
    llm_pretrigger_distance_m:="$LLM_PRETRIGGER_DISTANCE_M" \
    llm_require_pointcloud_distance:="$LLM_REQUIRE_POINTCLOUD_DISTANCE" \
    llm_call_requires_trigger:="$LLM_CALL_REQUIRES_TRIGGER" \
    llm_require_trigger_low_before_call:="$LLM_REQUIRE_TRIGGER_LOW_BEFORE_CALL" \
    llm_once_per_trigger:="$LLM_ONCE_PER_TRIGGER" \
    llm_disable_after_first_success:="$LLM_DISABLE_AFTER_FIRST_SUCCESS" \
    llm_update_interval_s:="$LLM_UPDATE_INTERVAL_S" \
    pending_safety_enable:="$PENDING_SAFETY_ENABLE" \
    pending_safety_speed_scale:="$PENDING_SAFETY_SPEED_SCALE" \
    api_timeout:="$LLM_API_TIMEOUT" \
    llm_wait_image_on_entry:="$LLM_WAIT_IMAGE_ON_ENTRY" \
    llm_distance_init_grace_s:="$LLM_DISTANCE_INIT_GRACE_S" \
    > /dev/null 2>&1 &
  VLA_PID=$!
  sleep 2
  if [ "$VLA_STARTUP_STRICT" = "true" ] && ! kill -0 "$VLA_PID" >/dev/null 2>&1; then
    echo "[ERROR] VLA process exited during startup. VLA roslaunch stdout/stderr logging is disabled."
    return 1
  fi
}

wait_for_required_topics() {
  local timeout_s="$1"
  shift
  python3 - "$timeout_s" "$@" <<'PY'
import sys
import time

import rospy
from rospy.msg import AnyMsg

timeout_s = float(sys.argv[1])
topics = list(dict.fromkeys(sys.argv[2:]))
seen = {topic: False for topic in topics}

def make_cb(topic):
    def cb(_msg):
        seen[topic] = True
    return cb

rospy.init_node("monte_carlo_startup_probe", anonymous=True, disable_signals=True)
subs = [rospy.Subscriber(topic, AnyMsg, make_cb(topic), queue_size=1) for topic in topics]
start = time.monotonic()
while not rospy.is_shutdown() and time.monotonic() - start < timeout_s:
    if all(seen.values()):
        print("[INFO] Startup topics ready: %s" % " ".join(topics))
        raise SystemExit(0)
    time.sleep(0.1)

missing = [topic for topic, ok in seen.items() if not ok]
print("[ERROR] Startup topics not ready within %.1fs: %s" % (timeout_s, " ".join(missing)))
raise SystemExit(1)
PY
}

publish_start_trigger() {
  for _ in 1 2 3; do
    rostopic pub -1 /traj_start_trigger std_msgs/Empty "{}" >/dev/null 2>&1 || true
    sleep 0.25
  done
}

success_count=0
FIRST_SEED_VALUE=""
LAST_SEED_VALUE=""

echo "============================================================================"
echo "Monte Carlo runner"
echo "Scenario:  $SCENARIO_NAME"
echo "Mode:      $ALGORITHM"
echo "Sensor:    $MC_SENSOR ($SENSOR_POINTCLOUD_TOPIC)"
echo "VLM input: $VLA_IMAGE_SOURCE"
if [ "$VLA_IMAGE_SOURCE" = "global" ]; then
  echo "Global trajectory: history=${VLA_GLOBAL_HISTORY_S}s image=${VLA_GLOBAL_IMAGE_WIDTH}x${VLA_GLOBAL_IMAGE_HEIGHT}"
elif [ "$VLA_IMAGE_SOURCE" = "ais" ]; then
  echo "AIS trajectory: history=${VLA_GLOBAL_HISTORY_S}s image=${VLA_GLOBAL_IMAGE_WIDTH}x${VLA_GLOBAL_IMAGE_HEIGHT}"
fi
echo "Count:     $COUNT"
if [ "$SEED_MODE" = "fixed" ]; then
  echo "Seed:      fixed base=$SEED_BASE (trial seed=base+trial-1)"
else
  echo "Seed:      random time-based per trial"
fi
echo "Results:   $RESULT_CSV"
echo "============================================================================"

cleanup_ros

for trial in $(seq 1 "$COUNT"); do
  if [ -n "$MC_SEED" ]; then
    SEED_VALUE=$((MC_SEED + trial - 1))
  else
    SEED_VALUE=$(( $(date +%s%N) / 1000 + trial ))
  fi
  if [ -z "$FIRST_SEED_VALUE" ]; then
    FIRST_SEED_VALUE="$SEED_VALUE"
  fi
  LAST_SEED_VALUE="$SEED_VALUE"

  eval "$(scenario_random_vars "$SCENARIO_NAME" "$SEED_VALUE")"

  if [ "$ENABLE_VLA_MODE" = "1" ]; then
    EGO_SPEED=$(scale_float "$EGO_SPEED" "$VLA_SPEED_SCALE")
    TARGET_SPEED=$(scale_float "$TARGET_SPEED" "$VLA_SPEED_SCALE")
    TARGET2_SPEED=$(scale_float "$TARGET2_SPEED" "$VLA_SPEED_SCALE")
    TARGET3_SPEED=$(scale_float "$TARGET3_SPEED" "$VLA_SPEED_SCALE")
  fi
  TRIAL_TIMEOUT_S=$(effective_timeout "$MC_TIMEOUT_S" "$ENABLE_VLA_MODE" "$VLA_SPEED_SCALE" "$VLA_TIMEOUT_SCALE")

  TRAJECTORY_CSV="$RESULT_DIR/trajectory_trial_${trial}.csv"
  VLM_TRUTH_CSV="$RESULT_DIR/vla_truth_trial_${trial}.csv"
  PLANNER_MODE="mpc"
  PARALLEL_NAV_SEMANTIC_WEIGHT_ENABLE=0
  LAUNCH_WAIT_FOR_TRIGGER=false
  if [ "$ENABLE_VLA_MODE" = "1" ]; then
    PARALLEL_NAV_SEMANTIC_WEIGHT_ENABLE=1
    LAUNCH_WAIT_FOR_TRIGGER=true
  fi

  echo ""
  echo "[Trial $trial/$COUNT] seed=$SEED_VALUE scenario=$SCENARIO_NAME mode=$ALGORITHM"
  if [ "$ENABLE_VLA_MODE" = "1" ]; then
    echo "  VLA speed scale=$VLA_SPEED_SCALE"
    echo "  timeout=${TRIAL_TIMEOUT_S}s (base=$MC_TIMEOUT_S, scale=$VLA_TIMEOUT_SCALE)"
  fi
  echo "  ego=($EGO_X,$EGO_Y) speed=$EGO_SPEED goal=($EGO_GOAL_X,$EGO_GOAL_Y)"
  echo "  target1=($TARGET_X,$TARGET_Y,z=$TARGET_MODEL_Z) speed=$TARGET_SPEED yaw=$TARGET_YAW"
  if [ "$ENABLE_MULTI_TARGETS" = "1" ]; then
    echo "  target2=($TARGET2_X,$TARGET2_Y,z=$TARGET_MODEL_Z) speed=$TARGET2_SPEED yaw=$TARGET2_YAW"
    echo "  target3=($TARGET3_X,$TARGET3_Y,z=$TARGET_MODEL_Z) speed=$TARGET3_SPEED yaw=$TARGET3_YAW"
  fi
  echo "  sensor=$MC_SENSOR pointcloud=$SENSOR_POINTCLOUD_TOPIC"

  cleanup_ros

  roslaunch exploration_manager boat_collision.launch \
    gui:="$MC_GUI" \
    wait_for_trigger:="$LAUNCH_WAIT_FOR_TRIGGER" \
    enable_avoidance:="$ENABLE_AVOIDANCE" \
    planner_mode:="$PLANNER_MODE" \
    parallel_nav_enable:=0 \
    parallel_nav_semantic_weight_enable:="$PARALLEL_NAV_SEMANTIC_WEIGHT_ENABLE" \
    enable_trajectory_record:=true \
    trajectory_record_interval_s:="$TRAJECTORY_RECORD_INTERVAL_S" \
    trajectory_record_csv:="$TRAJECTORY_CSV" \
    sensor_pointcloud_topic:="$SENSOR_POINTCLOUD_TOPIC" \
    spawn_target_boat:="$SPAWN_TARGET_BOAT" \
    use_static_buoys:="$USE_STATIC_BUOYS" \
    enable_extra_buoys:="$ENABLE_EXTRA_BUOYS" \
    channel_constraints_enable:="$CHANNEL_CONSTRAINTS_ENABLE" \
    channel_left_y:="$CHANNEL_LEFT_Y" channel_right_y:="$CHANNEL_RIGHT_Y" \
    channel_x_min:="$CHANNEL_X_MIN" channel_x_max:="$CHANNEL_X_MAX" channel_x_step:="$CHANNEL_X_STEP" \
    enable_multi_targets:="$ENABLE_MULTI_TARGETS" \
    ego_x:="$EGO_X" ego_y:="$EGO_Y" ego_yaw:="$EGO_YAW" ego_speed:="$EGO_SPEED" ego_goal_x:="$EGO_GOAL_X" ego_goal_y:="$EGO_GOAL_Y" goal_tolerance_m:="$MC_GOAL_TOLERANCE_M" \
    target_x:="$TARGET_X" target_y:="$TARGET_Y" target_yaw:="$TARGET_YAW" target_speed:="$TARGET_SPEED" target_turn_rate:="$TARGET_TURN_RATE" \
    target_z:="$TARGET_MODEL_Z" \
    target2_x:="$TARGET2_X" target2_y:="$TARGET2_Y" target2_yaw:="$TARGET2_YAW" target2_speed:="$TARGET2_SPEED" target2_turn_rate:="$TARGET2_TURN_RATE" \
    target2_z:="$TARGET_MODEL_Z" \
    target3_x:="$TARGET3_X" target3_y:="$TARGET3_Y" target3_yaw:="$TARGET3_YAW" target3_speed:="$TARGET3_SPEED" target3_turn_rate:="$TARGET3_TURN_RATE" \
    target3_z:="$TARGET_MODEL_Z" \
    > /dev/null 2>&1 &
  GAZEBO_PID=$!

  sleep 3
  launch_vla_if_needed
  startup_topics=(/myboat/odom)
  if [ "$SPAWN_TARGET_BOAT" = "1" ] || [ "$SPAWN_TARGET_BOAT" = "true" ]; then
    startup_topics+=(/target_boat/odom)
    if [ "$ENABLE_MULTI_TARGETS" = "1" ] || [ "$ENABLE_MULTI_TARGETS" = "true" ]; then
      startup_topics+=(/target_boat_2/odom /target_boat_3/odom)
    fi
  fi
  if [ "$ENABLE_VLA_MODE" = "1" ]; then
    if [ "$VLA_IMAGE_SOURCE" = "camera" ]; then
      startup_topics+=("$VLA_IMAGE_TOPIC")
    elif [ "$VLA_IMAGE_SOURCE" = "global" ]; then
      startup_topics+=("$VLA_BEV_POINTCLOUD_TOPIC")
    elif [ "$VLA_IMAGE_SOURCE" = "ais" ]; then
      startup_topics+=("$AIS_TARGET_ODOM_TOPIC")
      if [ "$ENABLE_MULTI_TARGETS" = "1" ] || [ "$ENABLE_MULTI_TARGETS" = "true" ]; then
        startup_topics+=("$AIS_TARGET2_ODOM_TOPIC" "$AIS_TARGET3_ODOM_TOPIC")
      fi
    elif [ "$VLA_IMAGE_SOURCE" = "pointcloud_bev" ] || [ "$VLA_IMAGE_SOURCE" = "bev" ] || [ "$VLA_IMAGE_SOURCE" = "lidar_bev" ]; then
      startup_topics+=("$VLA_BEV_POINTCLOUD_TOPIC")
    fi
  fi

  if ! wait_for_required_topics "$MC_STARTUP_TIMEOUT_S" "${startup_topics[@]}"; then
    RESULT="FAIL"
    REASON="startup_failed"
    DURATION_S="0.000"
    GOAL_DISTANCE_M="-1.000"
    MIN_DISTANCE_M="-1.000"
    TRUTH_CPA_TIME_S="-1.000"
    MIN_TRUTH_TCPA_S="$TRUTH_CPA_TIME_S"
    MIN_TRUTH_DCPA_M="-1.000"
    MIN_TCPA_S="$MIN_TRUTH_TCPA_S"
    MIN_DCPA_M="$MIN_TRUTH_DCPA_M"
    COLREG_VIOLATION="false"
    NMF="0"
    VLM_TRUTH_CALLS="0"
    VLM_TRUTH_MATCHES="0"
    VLM_TRUTH_MISMATCHES="0"
    VLM_TRUTH_ACCURACY="-1.000"
    echo "  result=$RESULT reason=$REASON duration=${DURATION_S}s goal_dist=${GOAL_DISTANCE_M}m min_dist=${MIN_DISTANCE_M}m truth_cpa_time=${TRUTH_CPA_TIME_S}s min_truth_tcpa=${MIN_TRUTH_TCPA_S}s min_truth_dcpa=${MIN_TRUTH_DCPA_M}m colreg_violation=$COLREG_VIOLATION nmf=$NMF vlm_truth=${VLM_TRUTH_MATCHES}/${VLM_TRUTH_CALLS}"
    echo "$trial,$SCENARIO_NAME,$ALGORITHM,$SEED_VALUE,$RESULT,$REASON,$DURATION_S,$GOAL_DISTANCE_M,$MIN_DISTANCE_M,$TRUTH_CPA_TIME_S,$MIN_TRUTH_TCPA_S,$MIN_TRUTH_DCPA_M,$MIN_TCPA_S,$MIN_DCPA_M,$COLREG_VIOLATION,$NMF,$EGO_X,$EGO_Y,$EGO_YAW,$EGO_SPEED,$EGO_GOAL_X,$EGO_GOAL_Y,$TARGET_X,$TARGET_Y,$TARGET_YAW,$TARGET_SPEED,$TARGET_TURN_RATE,$TARGET2_X,$TARGET2_Y,$TARGET2_YAW,$TARGET2_SPEED,$TARGET2_TURN_RATE,$TARGET3_X,$TARGET3_Y,$TARGET3_YAW,$TARGET3_SPEED,$TARGET3_TURN_RATE,$TRAJECTORY_CSV,$VLM_TRUTH_CALLS,$VLM_TRUTH_MATCHES,$VLM_TRUTH_MISMATCHES,$VLM_TRUTH_ACCURACY,$VLM_TRUTH_CSV" >> "$RESULT_CSV"
    kill "${VLA_PID:-}" >/dev/null 2>&1 || true
    kill "${GAZEBO_PID:-}" >/dev/null 2>&1 || true
    cleanup_ros
    continue
  fi

  if [ "$ENABLE_VLA_MODE" = "1" ]; then
    sleep 1
    publish_start_trigger
  else
    sleep 2
  fi

  IFS=',' read -r RESULT REASON DURATION_S GOAL_DISTANCE_M MIN_DISTANCE_M TRUTH_CPA_TIME_S MIN_TRUTH_DCPA_M COLREG_VIOLATION NMF VLM_TRUTH_CALLS VLM_TRUTH_MATCHES VLM_TRUTH_MISMATCHES VLM_TRUTH_ACCURACY < <(
    monitor_trial "$EGO_GOAL_X" "$EGO_GOAL_Y" "$MC_GOAL_TOLERANCE_M" "$MC_COLLISION_DISTANCE_M" "$TRIAL_TIMEOUT_S" "$ENABLE_MULTI_TARGETS" "$MC_NEAR_MISS_DISTANCE_M" "$SCENARIO_NAME" "$MC_COLREG_PASS_CLEARANCE_M" "$MC_COLREG_SUBSTANTIAL_OFFSET_M" "$MC_COLREG_EVAL_DISTANCE_M" "$MC_COLREG_MULTI_ALLOWED_VIOLATIONS" "$MC_CLOCK_STALL_S" "$VLM_TRUTH_CSV" "$MC_VLM_TRUTH_ENABLE" "$MC_VLM_TRUTH_TCPA_HORIZON_S" "$MC_VLM_TRUTH_DCPA_DANGER_M" "$MC_VLM_TRUTH_CLOSE_RANGE_M" "$MC_TRUTH_CPA_WARMUP_S" "$MC_TRUTH_CPA_MIN_RANGE_M"
  )
  MIN_TRUTH_TCPA_S="$TRUTH_CPA_TIME_S"
  MIN_TCPA_S="$MIN_TRUTH_TCPA_S"
  MIN_DCPA_M="$MIN_TRUTH_DCPA_M"

  if [ "$RESULT" = "SUCCESS" ]; then
    success_count=$((success_count + 1))
  fi

  echo "  result=$RESULT reason=$REASON duration=${DURATION_S}s goal_dist=${GOAL_DISTANCE_M}m min_dist=${MIN_DISTANCE_M}m truth_cpa_time=${TRUTH_CPA_TIME_S}s min_truth_tcpa=${MIN_TRUTH_TCPA_S}s min_truth_dcpa=${MIN_TRUTH_DCPA_M}m colreg_violation=$COLREG_VIOLATION nmf=$NMF vlm_truth=${VLM_TRUTH_MATCHES}/${VLM_TRUTH_CALLS}"

  echo "$trial,$SCENARIO_NAME,$ALGORITHM,$SEED_VALUE,$RESULT,$REASON,$DURATION_S,$GOAL_DISTANCE_M,$MIN_DISTANCE_M,$TRUTH_CPA_TIME_S,$MIN_TRUTH_TCPA_S,$MIN_TRUTH_DCPA_M,$MIN_TCPA_S,$MIN_DCPA_M,$COLREG_VIOLATION,$NMF,$EGO_X,$EGO_Y,$EGO_YAW,$EGO_SPEED,$EGO_GOAL_X,$EGO_GOAL_Y,$TARGET_X,$TARGET_Y,$TARGET_YAW,$TARGET_SPEED,$TARGET_TURN_RATE,$TARGET2_X,$TARGET2_Y,$TARGET2_YAW,$TARGET2_SPEED,$TARGET2_TURN_RATE,$TARGET3_X,$TARGET3_Y,$TARGET3_YAW,$TARGET3_SPEED,$TARGET3_TURN_RATE,$TRAJECTORY_CSV,$VLM_TRUTH_CALLS,$VLM_TRUTH_MATCHES,$VLM_TRUTH_MISMATCHES,$VLM_TRUTH_ACCURACY,$VLM_TRUTH_CSV" >> "$RESULT_CSV"

  if [ "$ENABLE_VLA_MODE" = "1" ] && [ -n "${VLA_PID:-}" ] && kill -0 "$VLA_PID" >/dev/null 2>&1; then
    sleep "$VLA_SHUTDOWN_GRACE_S"
  fi

  kill "${VLA_PID:-}" >/dev/null 2>&1 || true
  kill "${GAZEBO_PID:-}" >/dev/null 2>&1 || true
  cleanup_ros
done

success_rate=$(python3 - "$success_count" "$COUNT" <<'PY'
import sys
succ = int(sys.argv[1])
total = int(sys.argv[2])
print("%.2f" % (100.0 * succ / max(1, total)))
PY
)

IFS=',' read -r AVG_MIN_TCPA AVG_MIN_DCPA AVG_SUCCESS_DURATION < <(
  python3 - "$RESULT_CSV" <<'PY'
import csv
import math
import sys

results_csv = sys.argv[1]

def as_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) and out >= 0.0 else None

def first_float(row, keys):
    for key in keys:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None

def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return -1.0
    return sum(values) / float(len(values))

min_tcpa_values = []
min_dcpa_values = []
success_durations = []

with open(results_csv, "r", newline="") as f:
    for row in csv.DictReader(f):
        min_tcpa_values.append(first_float(row, ("min_truth_tcpa_s", "truth_cpa_time_s", "min_tcpa_s")))
        min_dcpa_values.append(first_float(row, ("min_truth_dcpa_m", "min_dcpa_m")))
        if str(row.get("result", "")).strip().upper() == "SUCCESS":
            success_durations.append(as_float(row.get("duration_s")))

print(
    "%.3f,%.3f,%.3f"
    % (
        mean(min_tcpa_values),
        mean(min_dcpa_values),
        mean(success_durations),
    )
)
PY
)

PLOT_DIR="$RESULT_DIR/plot"
PLOT_LOG="not_saved"
PLOT_STATUS="disabled"
if [ "$MC_PLOT_TRAJECTORIES" = "true" ] || [ "$MC_PLOT_TRAJECTORIES" = "1" ]; then
  echo ""
  echo "Plotting trajectories..."
  mkdir -p "$PLOT_DIR"
  if python3 "$WORKSPACE/plot_boat_trajectories.py" \
      --dir "$RESULT_DIR" \
      --pattern "trajectory_trial_*.csv" \
      --max-files "$MC_PLOT_MAX_FILES" \
      --label-interval "$MC_PLOT_LABEL_INTERVAL_S" \
      --max-labels-per-track "$MC_PLOT_MAX_LABELS_PER_TRACK" \
      --min-label-gap "$MC_PLOT_MIN_LABEL_GAP_M" \
      --collision-distance "$MC_COLLISION_DISTANCE_M" \
      > /dev/null 2>&1; then
    PLOT_STATUS="ok"
    echo "Plots: $PLOT_DIR"
  else
    PLOT_STATUS="failed"
    echo "[WARN] Failed to plot trajectories. Plot stdout/stderr logging is disabled."
  fi
fi

{
  echo "scenario=$SCENARIO_NAME"
  echo "mode=$ALGORITHM"
  echo "vla_prompt_mode=${VLA_PROMPT_MODE:-none}"
  echo "vla_prompt_track_input_enable=$VLA_PROMPT_TRACK_INPUT_ENABLE"
  echo "vla_speed_scale=$([ "$ENABLE_VLA_MODE" = "1" ] && echo "$VLA_SPEED_SCALE" || echo "1.0")"
  echo "count=$COUNT"
  echo "seed_mode=$SEED_MODE"
  echo "seed_base=$SEED_BASE"
  echo "seed_first=$FIRST_SEED_VALUE"
  echo "seed_last=$LAST_SEED_VALUE"
  echo "success=$success_count"
  echo "success_rate_percent=$success_rate"
  echo "avg_min_tcpa_s=$AVG_MIN_TCPA"
  echo "avg_min_dcpa_m=$AVG_MIN_DCPA"
  echo "avg_success_duration_s=$AVG_SUCCESS_DURATION"
  echo "goal_tolerance_m=$MC_GOAL_TOLERANCE_M"
  echo "collision_distance_m=$MC_COLLISION_DISTANCE_M"
  echo "near_miss_distance_m=$MC_NEAR_MISS_DISTANCE_M"
  echo "colreg_pass_clearance_m=$MC_COLREG_PASS_CLEARANCE_M"
  echo "colreg_substantial_offset_m=$MC_COLREG_SUBSTANTIAL_OFFSET_M"
  echo "colreg_eval_distance_m=$MC_COLREG_EVAL_DISTANCE_M"
  echo "colreg_multi_allowed_violations=$MC_COLREG_MULTI_ALLOWED_VIOLATIONS"
  echo "vlm_truth_enable=$MC_VLM_TRUTH_ENABLE"
  echo "vlm_truth_tcpa_horizon_s=$MC_VLM_TRUTH_TCPA_HORIZON_S"
  echo "vlm_truth_dcpa_danger_m=$MC_VLM_TRUTH_DCPA_DANGER_M"
  echo "vlm_truth_close_range_m=$MC_VLM_TRUTH_CLOSE_RANGE_M"
  echo "truth_cpa_warmup_s=$MC_TRUTH_CPA_WARMUP_S"
  echo "truth_cpa_min_range_m=$MC_TRUTH_CPA_MIN_RANGE_M"
  echo "timeout_base_s=$MC_TIMEOUT_S"
  echo "vla_timeout_scale=$([ "$ENABLE_VLA_MODE" = "1" ] && echo "$VLA_TIMEOUT_SCALE" || echo "1.0")"
  echo "timeout_s=$([ "$ENABLE_VLA_MODE" = "1" ] && effective_timeout "$MC_TIMEOUT_S" "$ENABLE_VLA_MODE" "$VLA_SPEED_SCALE" "$VLA_TIMEOUT_SCALE" || echo "$MC_TIMEOUT_S")"
  echo "results_csv=$RESULT_CSV"
  echo "plot_status=$PLOT_STATUS"
  echo "plot_dir=$PLOT_DIR"
  echo "plot_log=$PLOT_LOG"
} > "$SUMMARY_TXT"

echo ""
echo "============================================================================"
echo "Done: $success_count/$COUNT success, success_rate=${success_rate}%"
echo "avg_min_tcpa=${AVG_MIN_TCPA}s avg_min_dcpa=${AVG_MIN_DCPA}m avg_success_duration=${AVG_SUCCESS_DURATION}s"
echo "Seeds: mode=$SEED_MODE base=${SEED_BASE:-none} first=$FIRST_SEED_VALUE last=$LAST_SEED_VALUE"
echo "Results: $RESULT_CSV"
echo "Summary: $SUMMARY_TXT"
echo "Plots: $PLOT_DIR ($PLOT_STATUS)"
echo "============================================================================"
