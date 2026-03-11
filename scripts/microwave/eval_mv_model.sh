#!/usr/bin/env sh

set -eu

RPYC_HOST="${ADA_MANIP_RPYC_HOST:-localhost}"
RPYC_PORT="${ADA_MANIP_RPYC_PORT:-18861}"
CFG_ENV="${ADA_MANIP_CFG_ENV:-cfg/microwave/open_microwave_model.yaml}"
SIM_DEVICE="${ADA_MANIP_SIM_DEVICE:-cuda:0}"
SEED="${ADA_MANIP_SEED:-0}"

cleanup() {
	if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
		kill "$SERVER_PID" 2>/dev/null || true
		wait "$SERVER_PID" 2>/dev/null || true
	fi
}

trap cleanup EXIT INT TERM

python run.py \
	--task=OpenMicroWave \
	--controller=ModelController \
	--manipulation=OpenMicroWaveManipulation \
	--sim_device="$SIM_DEVICE" \
	--seed="$SEED" \
	--pipeline=gpu \
	--cfg_env="$CFG_ENV" \
	--runtime_mode=rpyc-server \
	--rpyc_host="$RPYC_HOST" \
	--rpyc_port="$RPYC_PORT" \
	--headless &
SERVER_PID=$!

python run.py \
	--task=OpenMicroWave \
	--controller=ModelController \
	--manipulation=OpenMicroWaveManipulation \
	--sim_device="$SIM_DEVICE" \
	--seed="$SEED" \
	--pipeline=gpu \
	--cfg_env="$CFG_ENV" \
	--runtime_mode=rpyc-client \
	--rpyc_host="$RPYC_HOST" \
	--rpyc_port="$RPYC_PORT" \
	--headless
