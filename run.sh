#!/usr/bin/env bash
# End-to-end driver: download, data construction, inference, the two new
# measurements, metrics and figures.
#
#   ./run.sh                          # full pipeline in a fresh timestamped run
#   ./run.sh --smoke                  # tiny CPU-only end-to-end check
#   ./run.sh --resume latest          # continue the most recent run
#   ./run.sh --resume 20260730-1412   # continue a specific run
#   ./run.sh --stage inference        # one stage only
#   ./run.sh --model sarvam-m --model airavata-7b
#   ./run.sh --max-seconds 36000      # stop cleanly inside a 10h budget
#   ./run.sh --setup                  # re-run setup_env.sh first (see below)
#   ./run.sh --force                  # rebuild the suite instead of reusing it
#   ./run.sh --skip-decompose         # no LLM judge (no API key needed)
#   ./run.sh --stage calibrate        # fit the noise operators to Dakshina
#   ./run.sh --no-export              # skip the final archive step
#
# Everything — venv, caches, weights, datasets, runs, logs — stays inside this
# directory. Every stage is checkpointed: re-running with --resume picks up
# exactly where the previous invocation stopped.
#
# Dependency install is NOT run by default. `setup_env.sh` re-resolves the
# whole dependency set and then verifies it by importing torch and vLLM, which
# costs a minute or more on every invocation and changes nothing once the venv
# exists. Run it yourself when the environment actually needs to change:
#
#   ./setup_env.sh --with-api     # after editing pyproject.toml, or a fresh clone
#   ./run.sh --setup              # equivalent, as part of a run
#
# If .venv is missing, this script says so and stops rather than guessing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- defaults ---------------------------------------------------------------
STAGE="all"
RUN_ID=""
SMOKE=0
# Off by default: re-resolving and re-verifying an unchanged environment costs a
# minute per run. `--setup` opts back in.
DO_SETUP=0
DO_DOWNLOAD=1
# Rebuild base items and re-render instead of reusing data/suite/. Needed after
# anything that changes what the suite *contains* — new dataset access, or new
# transform probabilities from `scriptgap calibrate`.
FORCE=0
SKIP_DECOMPOSE=0
DO_EXPORT=1
MAX_SECONDS=""
LOG_LEVEL="INFO"
THEME="light"
CONFIGS=()
MODELS=()
TASKS=()
CONDITIONS=()
EXTRA=()

# Print the leading comment block (stops at the first non-comment line).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --resume|--run-id) RUN_ID="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --setup) DO_SETUP=1; shift ;;
    --force) FORCE=1; shift ;;
    # Now the default. Still accepted so existing commands and the README's
    # `--no-setup --resume` recipe keep working.
    --no-setup) DO_SETUP=0; shift ;;
    --no-download) DO_DOWNLOAD=0; shift ;;
    --skip-decompose) SKIP_DECOMPOSE=1; shift ;;
    --no-export) DO_EXPORT=0; shift ;;
    --max-seconds) MAX_SECONDS="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    --theme) THEME="$2"; shift 2 ;;
    -c|--config) CONFIGS+=("$2"); shift 2 ;;
    --model) MODELS+=("$2"); shift 2 ;;
    --task) TASKS+=("$2"); shift 2 ;;
    --condition) CONDITIONS+=("$2"); shift 2 ;;
    --set) EXTRA+=(--set "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# --- HuggingFace auth -------------------------------------------------------
# MUST run before HF_HOME is redirected below. `hf auth login` writes its token
# to $HF_HOME/token, so pointing HF_HOME at the repo hides a token that was
# saved under a different HF_HOME. The failure is confusing rather than
# obvious: `hf auth login` still reports "already logged in", while every gated
# repo 401s with "You must have access to it and be authenticated".
if [[ -z "${HF_TOKEN:-}" ]]; then
  for candidate in "${HF_HOME:+$HF_HOME/token}" \
                   "$HOME/.cache/huggingface/token" \
                   "$HOME/.huggingface/token"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      HF_TOKEN="$(tr -d '[:space:]' < "$candidate")"
      export HF_TOKEN
      echo "==> HF auth: token from $candidate"
      break
    fi
  done
  [[ -z "${HF_TOKEN:-}" ]] && echo "==> HF auth: no token found — gated models and MILU will fail (run \`hf auth login\`)" >&2
fi

# --- repo-local caches (belt-and-braces; src/scriptgap/env.py does this too) -
export UV_CACHE_DIR="$REPO_ROOT/.cache/uv"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$REPO_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache/triton"
export VLLM_CACHE_ROOT="$REPO_ROOT/.cache/vllm"
export VLLM_CONFIG_ROOT="$REPO_ROOT/.cache/vllm"
export TIKTOKEN_CACHE_DIR="$REPO_ROOT/.cache/tiktoken"
export MPLCONFIGDIR="$REPO_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$REPO_ROOT/.cache/xdg"
export TMPDIR="$REPO_ROOT/.cache/tmp"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" logs runs data models

# --- setup ------------------------------------------------------------------
if [[ "$DO_SETUP" == "1" ]]; then
  if [[ "$SKIP_DECOMPOSE" == "0" && "$SMOKE" == "0" ]]; then
    ./setup_env.sh --with-api
  else
    ./setup_env.sh
  fi
fi

if [[ ! -d .venv ]]; then
  echo "error: .venv is missing — run ./setup_env.sh (or ./run.sh --setup)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- smoke configuration ----------------------------------------------------
if [[ "$SMOKE" == "1" ]]; then
  echo "==> smoke mode: tiny CPU-only end-to-end check"
  CONFIGS=(configs/smoke.yaml configs/smoke-models.yaml)
  SKIP_DECOMPOSE=1
  [[ -z "$RUN_ID" ]] && RUN_ID=""
elif [[ ${#CONFIGS[@]} -eq 0 ]]; then
  CONFIGS=(configs/models.yaml)
fi

# --- assemble common flags --------------------------------------------------
COMMON=(--log-level "$LOG_LEVEL")
for c in ${CONFIGS[@]+"${CONFIGS[@]}"}; do COMMON+=(-c "$c"); done
COMMON+=(${EXTRA[@]+"${EXTRA[@]}"})

RUN_FLAGS=()
if [[ -n "$RUN_ID" ]]; then
  RUN_FLAGS+=(--run-id "$RUN_ID")
else
  # A fresh run id, reused by every stage below so they share one directory.
  RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
  RUN_FLAGS+=(--run-id "$RUN_ID")
fi

FILTERS=()
for m in ${MODELS[@]+"${MODELS[@]}"}; do FILTERS+=(--model "$m"); done
for t in ${TASKS[@]+"${TASKS[@]}"}; do FILTERS+=(--task "$t"); done
for c in ${CONDITIONS[@]+"${CONDITIONS[@]}"}; do FILTERS+=(--condition "$c"); done

BUDGET=()
[[ -n "$MAX_SECONDS" ]] && BUDGET+=(--max-seconds "$MAX_SECONDS")

echo "==> run id: $RUN_ID"
echo "==> configs: ${CONFIGS[*]}"
echo "==> stage: $STAGE"
echo "==> results will be under runs/$RUN_ID/ ; logs also append to logs/scriptgap.log"

sg() { python -m scriptgap.cli "$@"; }

# macOS ships bash 3.2, where "${arr[@]}" on an empty array is an unbound-variable
# error under `set -u`. Build argument lists explicitly instead of relying on
# array-expansion shorthand.
MODEL_ARGS=()
for m in ${MODELS[@]+"${MODELS[@]}"}; do MODEL_ARGS+=(--model "$m"); done
TASK_ARGS=()
for t in ${TASKS[@]+"${TASKS[@]}"}; do TASK_ARGS+=(--task "$t"); done

stage_download() {
  echo; echo "############ stage: download ############"
  sg download ${COMMON[@]+"${COMMON[@]}"} ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} \
    ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
}

stage_calibrate() {
  echo; echo "############ stage: calibrate ############"
  sg calibrate ${COMMON[@]+"${COMMON[@]}"} ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"}
}

stage_export() {
  echo; echo "############ stage: export ############"
  sg export ${COMMON[@]+"${COMMON[@]}"} ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"}
}

stage_build_data() {
  echo; echo "############ stage: build-data ############"
  FORCE_ARGS=()
  [[ "$FORCE" == "1" ]] && FORCE_ARGS+=(--force)
  sg build-data ${COMMON[@]+"${COMMON[@]}"} ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} \
    ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}
}

stage_inference() {
  echo; echo "############ stage: inference ############"
  sg run-inference ${COMMON[@]+"${COMMON[@]}"} ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} \
    ${FILTERS[@]+"${FILTERS[@]}"} ${BUDGET[@]+"${BUDGET[@]}"}
}

stage_decompose() {
  if [[ "$SKIP_DECOMPOSE" == "1" ]]; then
    echo; echo "############ stage: decompose (skipped) ############"
    return 0
  fi
  echo; echo "############ stage: decompose (H5) ############"
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && ! ant auth status >/dev/null 2>&1; then
    echo "!! no ANTHROPIC_API_KEY and no \`ant auth login\` profile — skipping H5." >&2
    echo "!! export ANTHROPIC_API_KEY=... then: ./run.sh --stage decompose --resume $RUN_ID" >&2
    return 0
  fi
  sg decompose ${COMMON[@]+"${COMMON[@]}"} ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} \
    ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}
}

stage_fertility() {
  echo; echo "############ stage: fertility (H6) ############"
  sg fertility ${COMMON[@]+"${COMMON[@]}"} ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} \
    ${TASK_ARGS[@]+"${TASK_ARGS[@]}"}
}

stage_metrics() {
  echo; echo "############ stage: metrics ############"
  sg metrics ${COMMON[@]+"${COMMON[@]}"} ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"}
}

stage_figures() {
  echo; echo "############ stage: figures ############"
  sg figures ${COMMON[@]+"${COMMON[@]}"} ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} --theme "$THEME"
}

case "$STAGE" in
  download)     stage_download ;;
  calibrate)    stage_calibrate ;;
  build-data)   stage_build_data ;;
  inference)    stage_inference ;;
  decompose)    stage_decompose ;;
  fertility)    stage_fertility ;;
  metrics)      stage_metrics ;;
  figures)      stage_figures ;;
  export)       stage_export ;;
  all)
    [[ "$DO_DOWNLOAD" == "1" ]] && stage_download
    stage_build_data
    set +e
    stage_inference
    INFERENCE_CODE=$?
    set -e
    if [[ "$INFERENCE_CODE" == "2" ]]; then
      echo
      echo "==> inference stopped on the time budget."
      echo "==> resume with: ./run.sh --resume $RUN_ID"
      exit 2
    elif [[ "$INFERENCE_CODE" != "0" ]]; then
      echo "==> inference failed with code $INFERENCE_CODE" >&2
      exit "$INFERENCE_CODE"
    fi
    stage_decompose
    stage_fertility
    stage_metrics
    stage_figures
    # Archive last, so results survive losing this machine.
    [[ "$DO_EXPORT" == "1" ]] && stage_export
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    echo "choose from: download calibrate build-data inference decompose fertility metrics figures export all" >&2
    exit 2
    ;;
esac

echo
echo "==> done. run directory: runs/$RUN_ID"
if [[ -d "runs/$RUN_ID/metrics" ]]; then
  echo "==> metrics:"; ls -1 "runs/$RUN_ID/metrics" 2>/dev/null | sed 's/^/      /'
fi
if [[ -d "runs/$RUN_ID/figures" ]]; then
  echo "==> figures:"; ls -1 "runs/$RUN_ID/figures" 2>/dev/null | sed 's/^/      /'
fi
if [[ -d "exports" ]]; then
  echo "==> archives:"; ls -1 exports 2>/dev/null | tail -3 | sed 's/^/      /'
fi
echo "==> resume/extend this run with: ./run.sh --resume $RUN_ID"
