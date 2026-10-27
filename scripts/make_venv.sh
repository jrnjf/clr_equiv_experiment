#!/usr/bin/env bash
# Create a working venv on Debian/Ubuntu hosts whose `python3 -m venv` omits the
# activation scripts (because the pythonX.Y-venv package is not installed).
#
# Usage:  ./scripts/make_venv.sh [cpu|cuda] [cuda_tag]
#   ./scripts/make_venv.sh cpu
#   ./scripts/make_venv.sh cuda cu124
set -euo pipefail

MODE="${1:-cpu}"
CUDA_TAG="${2:-cu124}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

rm -rf .venv
python3 -m venv .venv || true     # may fail at the pip step; handled below

# 1) pip: venv creation can leave us without it (no ensurepip on this host)
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  echo ">> bootstrapping pip"
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  .venv/bin/python /tmp/get-pip.py --no-warn-script-location
fi

# 2) activation script: absent when pythonX.Y-venv is not installed
if [ ! -f .venv/bin/activate ]; then
  echo ">> writing .venv/bin/activate"
  cat > .venv/bin/activate <<'ACTIVATE'
deactivate () {
    if [ -n "${_OLD_VIRTUAL_PATH:-}" ] ; then
        PATH="${_OLD_VIRTUAL_PATH}"; export PATH; unset _OLD_VIRTUAL_PATH
    fi
    if [ -n "${_OLD_VIRTUAL_PS1:-}" ] ; then
        PS1="${_OLD_VIRTUAL_PS1}"; export PS1; unset _OLD_VIRTUAL_PS1
    fi
    unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
    if [ ! "${1:-}" = "nondestructive" ] ; then unset -f deactivate ; fi
}
__D__="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"
VIRTUAL_ENV="$(dirname "$__D__")"; export VIRTUAL_ENV; unset __D__
_OLD_VIRTUAL_PATH="$PATH"; PATH="$VIRTUAL_ENV/bin:$PATH"; export PATH
VIRTUAL_ENV_PROMPT="$(basename "$VIRTUAL_ENV")"; export VIRTUAL_ENV_PROMPT
if [ -z "${VIRTUAL_ENV_DISABLE_PROMPT:-}" ] ; then
    _OLD_VIRTUAL_PS1="${PS1:-}"; PS1="(${VIRTUAL_ENV_PROMPT}) ${PS1:-}"; export PS1
fi
ACTIVATE
  chmod +x .venv/bin/activate
fi

# 3) dependencies
if [ "$MODE" = "cuda" ]; then
  URL="https://download.pytorch.org/whl/${CUDA_TAG}"
else
  URL="https://download.pytorch.org/whl/cpu"
fi
echo ">> installing torch/torchvision from $URL"
.venv/bin/python -m pip install --index-url "$URL" torch torchvision

# 4) verify
.venv/bin/python - <<'PY'
import torch, torchvision
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} "
      f"| cuda available: {torch.cuda.is_available()}")
PY
echo ">> done. Activate with: source .venv/bin/activate"
