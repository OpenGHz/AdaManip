#!/usr/bin/env sh

set -eu

TASK_NAME="OpenMicroWave"
MANIPULATION_NAME="OpenMicroWaveManipulation"
CFG_ENV="${ADA_MANIP_CFG_ENV:-cfg/microwave/open_microwave_model.yaml}"

RPYC_HOST="${ADA_MANIP_RPYC_HOST:-localhost}"
RPYC_PORT="${ADA_MANIP_RPYC_PORT:-18861}"
SIM_DEVICE="${ADA_MANIP_SIM_DEVICE:-cuda:0}"
SEED="${ADA_MANIP_SEED:-0}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
ADA_MANIP_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$ADA_MANIP_ROOT/../.." && pwd)

cleanup() {
	if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
		kill "$SERVER_PID" 2>/dev/null || true
		wait "$SERVER_PID" 2>/dev/null || true
	fi
}

trap cleanup EXIT INT TERM

(
	cd "$ADA_MANIP_ROOT"
	pixi run --manifest-path "$REPO_ROOT/pyproject.toml" -e ada-data python run.py \
		--task=$TASK_NAME \
		--controller=ModelController \
		--manipulation=$MANIPULATION_NAME \
		--sim_device="$SIM_DEVICE" \
		--seed="$SEED" \
		--pipeline=gpu \
		--cfg_env="$CFG_ENV" \
		--runtime_mode=rpyc-server \
		--rpyc_host="$RPYC_HOST" \
		--rpyc_port="$RPYC_PORT" \
		# --headless
) &
SERVER_PID=$!

cd "$ADA_MANIP_ROOT"
pixi run --manifest-path "$REPO_ROOT/pyproject.toml" -e ada-manip python run.py \
	--task=$TASK_NAME \
	--controller=ModelController \
	--manipulation=$MANIPULATION_NAME \
	--sim_device="$SIM_DEVICE" \
	--seed="$SEED" \
	--pipeline=gpu \
	--cfg_env="$CFG_ENV" \
	--runtime_mode=rpyc-client \
	--rpyc_host="$RPYC_HOST" \
	--rpyc_port="$RPYC_PORT" \
	--headless
