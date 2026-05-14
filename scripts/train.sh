#!/usr/bin/env bash

set -euo pipefail

usage() {
	cat >&2 <<'EOF'
Usage: train.sh <task> [role] [extra diffusion_train.py args...]

Generic AdaManip training entrypoint.

Supported task keys:
  bottle, cm, door, lamp, microwave, pc, pen, safe, window

Roles:
  manip, grasp

Environment overrides:
  ADA_MANIP_CFG_ENV   Config path relative to third_party/ada_manip
  ADA_MANIP_DRY_RUN   Set to 1/true/yes/on to print resolved training config
EOF
}

is_truthy() {
	case "${1:-}" in
		1|true|TRUE|yes|YES|on|ON) return 0 ;;
		*) return 1 ;;
	esac
}

TASK_KEY="${1:-}"
ROLE="${2:-manip}"

case "$TASK_KEY" in
	-h|--help|'')
		usage
		exit 0
		;;
	bottle) DEFAULT_CFG_ENV="cfg/bottle/open_bottle_model.yaml" ;;
	cm|coffee_maker|coffee_machine) DEFAULT_CFG_ENV="cfg/cm/open_cm_model.yaml" ;;
	door) DEFAULT_CFG_ENV="cfg/door/open_door_model.yaml" ;;
	lamp) DEFAULT_CFG_ENV="cfg/lamp/open_lamp_model.yaml" ;;
	microwave) DEFAULT_CFG_ENV="cfg/microwave/open_microwave_model.yaml" ;;
	pc|pressure_cooker) DEFAULT_CFG_ENV="cfg/pressure_cooker/open_pc_model.yaml" ;;
	pen) DEFAULT_CFG_ENV="cfg/pen/open_pen_model.yaml" ;;
	safe) DEFAULT_CFG_ENV="cfg/safe/open_safe_model.yaml" ;;
	window) DEFAULT_CFG_ENV="cfg/window/open_window_model.yaml" ;;
	*)
		echo "Unknown task key: $TASK_KEY" >&2
		usage
		exit 2
		;;
esac

case "$ROLE" in
	manip|grasp) ;;
	*)
		echo "Unknown training role: $ROLE" >&2
		usage
		exit 2
		;;
esac

CFG_ENV="${ADA_MANIP_CFG_ENV:-$DEFAULT_CFG_ENV}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
ADA_MANIP_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
if [ "$#" -ge 2 ]; then
	shift 2
elif [ "$#" -eq 1 ]; then
	shift 1
fi

CMD=(python diffusion_train.py --cfg_env "$CFG_ENV" --task_stage "$ROLE")
if is_truthy "${ADA_MANIP_DRY_RUN:-0}"; then
	CMD+=(--dry_run)
fi
CMD+=("$@")

cd "$ADA_MANIP_ROOT"
exec "${CMD[@]}"