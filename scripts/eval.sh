#!/usr/bin/env bash

set -euo pipefail

usage() {
	cat >&2 <<'EOF'
Usage: eval.sh [task]

Generic AdaManip inference entrypoint.

Supported task keys:
  bottle, cm, door, lamp, microwave, pc, pen, safe, window

Environment overrides:
  ADA_MANIP_CFG_ENV          Config path relative to third_party/ada_manip
  ADA_MANIP_SEED             Eval seed
  ADA_MANIP_SIM_DEVICE       Simulation device, default cuda:0
  ADA_MANIP_RPYC_HOST        RPyC host, default localhost
  ADA_MANIP_RPYC_PORT        RPyC port, default 18861
  ADA_MANIP_SERVER_HEADLESS  Set to 1/true/yes/on to hide the IsaacGym server viewer
  ADA_MANIP_CLIENT_HEADLESS  Set to 0/false/no/off to pass no --headless to the client
  ADA_MANIP_DRY_RUN          Set to 1/true/yes/on to print commands without running
EOF
}

is_truthy() {
	case "${1:-}" in
		1|true|TRUE|yes|YES|on|ON) return 0 ;;
		*) return 1 ;;
	esac
}

is_falsey() {
	case "${1:-}" in
		0|false|FALSE|no|NO|off|OFF) return 0 ;;
		*) return 1 ;;
	esac
}

TASK_KEY="${1:-${ADA_MANIP_TASK:-microwave}}"

case "$TASK_KEY" in
	-h|--help)
		usage
		exit 0
		;;
	bottle|OpenBottle)
		TASK_NAME="OpenBottle"
		MANIPULATION_NAME="OpenBottleManipulation"
		DEFAULT_CFG_ENV="cfg/bottle/open_bottle_model.yaml"
		DEFAULT_SEED="0"
		;;
	cm|coffee_maker|coffee_machine|OpenCoffeeMachine)
		TASK_NAME="OpenCoffeeMachine"
		MANIPULATION_NAME="OpenCoffeeMachineManipulation"
		DEFAULT_CFG_ENV="cfg/cm/open_cm_model.yaml"
		DEFAULT_SEED="1"
		;;
	door|OpenDoor)
		TASK_NAME="OpenDoor"
		MANIPULATION_NAME="OpenDoorManipulation"
		DEFAULT_CFG_ENV="cfg/door/open_door_model.yaml"
		DEFAULT_SEED="0"
		;;
	lamp|OpenLamp)
		TASK_NAME="OpenLamp"
		MANIPULATION_NAME="OpenLampManipulation"
		DEFAULT_CFG_ENV="cfg/lamp/open_lamp_model.yaml"
		DEFAULT_SEED="0"
		;;
	microwave|OpenMicroWave)
		TASK_NAME="OpenMicroWave"
		MANIPULATION_NAME="OpenMicroWaveManipulation"
		DEFAULT_CFG_ENV="cfg/microwave/open_microwave_model.yaml"
		DEFAULT_SEED="0"
		;;
	pc|pressure_cooker|OpenPressureCooker)
		TASK_NAME="OpenPressureCooker"
		MANIPULATION_NAME="OpenPressureCookerManipulation"
		DEFAULT_CFG_ENV="cfg/pressure_cooker/open_pc_model.yaml"
		DEFAULT_SEED="0"
		;;
	pen|OpenPen)
		TASK_NAME="OpenPen"
		MANIPULATION_NAME="OpenPenManipulation"
		DEFAULT_CFG_ENV="cfg/pen/open_pen_model.yaml"
		DEFAULT_SEED="0"
		;;
	safe|OpenSafe)
		TASK_NAME="OpenSafe"
		MANIPULATION_NAME="OpenSafeManipulation"
		DEFAULT_CFG_ENV="cfg/safe/open_safe_model.yaml"
		DEFAULT_SEED="0"
		;;
	window|OpenWindow)
		TASK_NAME="OpenWindow"
		MANIPULATION_NAME="OpenWindowManipulation"
		DEFAULT_CFG_ENV="cfg/window/open_window_model.yaml"
		DEFAULT_SEED="0"
		;;
	*)
		echo "Unknown task key: $TASK_KEY" >&2
		usage
		exit 2
		;;
esac

CFG_ENV="${ADA_MANIP_CFG_ENV:-$DEFAULT_CFG_ENV}"
RPYC_HOST="${ADA_MANIP_RPYC_HOST:-localhost}"
RPYC_PORT="${ADA_MANIP_RPYC_PORT:-18861}"
SIM_DEVICE="${ADA_MANIP_SIM_DEVICE:-cuda:0}"
SEED="${ADA_MANIP_SEED:-$DEFAULT_SEED}"
CONTROLLER_NAME="${ADA_MANIP_CONTROLLER:-ModelController}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
ADA_MANIP_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$ADA_MANIP_ROOT/../.." && pwd)

SERVER_CMD=(
	pixi run --manifest-path "$REPO_ROOT/pyproject.toml" -e ada-data python run.py
	--task="$TASK_NAME"
	--controller="$CONTROLLER_NAME"
	--manipulation="$MANIPULATION_NAME"
	--sim_device="$SIM_DEVICE"
	--seed="$SEED"
	--pipeline=gpu
	--cfg_env="$CFG_ENV"
	--runtime_mode=rpyc-server
	--rpyc_host="$RPYC_HOST"
	--rpyc_port="$RPYC_PORT"
)

CLIENT_CMD=(
	pixi run --manifest-path "$REPO_ROOT/pyproject.toml" -e ada-manip python run.py
	--task="$TASK_NAME"
	--controller="$CONTROLLER_NAME"
	--manipulation="$MANIPULATION_NAME"
	--sim_device="$SIM_DEVICE"
	--seed="$SEED"
	--pipeline=gpu
	--cfg_env="$CFG_ENV"
	--runtime_mode=rpyc-client
	--rpyc_host="$RPYC_HOST"
	--rpyc_port="$RPYC_PORT"
)

if is_truthy "${ADA_MANIP_SERVER_HEADLESS:-0}"; then
	SERVER_CMD+=(--headless)
fi

if ! is_falsey "${ADA_MANIP_CLIENT_HEADLESS:-1}"; then
	CLIENT_CMD+=(--headless)
fi

print_command() {
	printf '  cd %q && ' "$ADA_MANIP_ROOT"
	printf '%q ' "$@"
	printf '\n'
}

if is_truthy "${ADA_MANIP_DRY_RUN:-0}"; then
	printf 'Task key: %s\n' "$TASK_KEY"
	printf 'Task class: %s\n' "$TASK_NAME"
	printf 'Manipulation class: %s\n' "$MANIPULATION_NAME"
	printf 'Config: %s\n' "$CFG_ENV"
	printf 'Seed: %s\n' "$SEED"
	printf 'Server command:\n'
	print_command "${SERVER_CMD[@]}"
	printf 'Client command:\n'
	print_command "${CLIENT_CMD[@]}"
	exit 0
fi

cleanup() {
	if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
		kill "$SERVER_PID" 2>/dev/null || true
		wait "$SERVER_PID" 2>/dev/null || true
	fi
}

trap cleanup EXIT INT TERM

(
	cd "$ADA_MANIP_ROOT"
	exec "${SERVER_CMD[@]}"
) &
SERVER_PID=$!

cd "$ADA_MANIP_ROOT"
"${CLIENT_CMD[@]}"