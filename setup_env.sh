#!/usr/bin/env bash
# Environment setup — uv-based, entirely inside the repository.
#
#   ./setup_env.sh              # auto-detect CUDA, install the right extras
#   ./setup_env.sh --cpu        # force the CPU stack (no vLLM)
#   ./setup_env.sh --gpu        # force the GPU stack (torch + vLLM)
#   ./setup_env.sh --with-api   # also install the Anthropic SDK (H5 judge)
#
# Nothing is written outside the repo: the venv, all caches and all model
# weights live under this directory.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Print the leading comment block (stops at the first non-comment line).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

VENV="$REPO_ROOT/.venv"
MODE="auto"
WITH_API=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) MODE="cpu"; shift ;;
    --gpu) MODE="gpu"; shift ;;
    --with-api) WITH_API=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# --- HuggingFace auth (before HF_HOME moves; see run.sh for the rationale) ---
if [[ -z "${HF_TOKEN:-}" ]]; then
  for candidate in "${HF_HOME:+$HF_HOME/token}" \
                   "$HOME/.cache/huggingface/token" \
                   "$HOME/.huggingface/token"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      HF_TOKEN="$(tr -d '[:space:]' < "$candidate")"
      export HF_TOKEN
      break
    fi
  done
fi

# --- repo-local caches (mirrors src/scriptgap/env.py, for subprocesses) ------
export UV_CACHE_DIR="$REPO_ROOT/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$VENV"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$REPO_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache/triton"
export VLLM_CACHE_ROOT="$REPO_ROOT/.cache/vllm"
export TIKTOKEN_CACHE_DIR="$REPO_ROOT/.cache/tiktoken"
export MPLCONFIGDIR="$REPO_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$REPO_ROOT/.cache/xdg"
export TMPDIR="$REPO_ROOT/.cache/tmp"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$VLLM_CACHE_ROOT" "$TIKTOKEN_CACHE_DIR" "$MPLCONFIGDIR" \
         "$XDG_CACHE_HOME" "$TMPDIR" "$REPO_ROOT/models" "$REPO_ROOT/logs" \
         "$REPO_ROOT/data" "$REPO_ROOT/runs"

# --- uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "==> uv $(uv --version)"

# --- detect the accelerator -------------------------------------------------
if [[ "$MODE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    MODE="gpu"
  else
    MODE="cpu"
  fi
fi
echo "==> install mode: $MODE"

if [[ "$MODE" == "gpu" ]]; then
  echo "==> GPUs:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /'
fi

# --- virtualenv -------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  echo "==> creating $VENV (python 3.11)"
  uv venv --python 3.11 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

EXTRAS=".[$MODE]"
if [[ "$WITH_API" == "1" ]]; then
  EXTRAS=".[$MODE,api]"
fi

echo "==> installing $EXTRAS"
uv pip install --upgrade pip
uv pip install -e "$EXTRAS"

# --- verify -----------------------------------------------------------------
echo "==> verifying the install"
python - <<'PY'
import importlib, sys

print(f"    python              {sys.version.split()[0]}")
for module in ("datasets", "transformers", "aksharamukha", "indic_transliteration",
               "pandas", "pyarrow", "matplotlib", "tqdm", "tiktoken"):
    try:
        importlib.import_module(module)
        print(f"    {module:<20}ok")
    except Exception as exc:
        print(f"    {module:<20}MISSING ({exc})")

try:
    import torch
    print(f"    torch               {torch.__version__} (cuda={torch.cuda.is_available()})")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"      gpu{i}: {props.name} {props.total_memory / 1024**3:.0f} GiB")
except Exception as exc:
    print(f"    torch               not installed ({exc})")

try:
    import vllm
    print(f"    vllm                {vllm.__version__}")
except Exception:
    print("    vllm                not installed (CPU mode: use inference.backend=hf)")

try:
    import anthropic
    print(f"    anthropic           {anthropic.__version__}")
except Exception:
    print("    anthropic           not installed (needed for the H5 judge: ./setup_env.sh --with-api)")

import scriptgap
from scriptgap import env
print(f"    scriptgap           {scriptgap.__version__}")
print("    caches:")
for key, value in env.describe().items():
    print(f"      {key:<24}{value}")
PY

cat <<EOF

==> setup complete

  activate:   source .venv/bin/activate
  smoke test: ./run.sh --smoke
  full run:   ./run.sh

Gated models (Llama, Gemma) need a HuggingFace login once:
  source .venv/bin/activate && hf auth login
The H5 judge needs ANTHROPIC_API_KEY (or a saved \`ant auth login\` profile).
EOF
