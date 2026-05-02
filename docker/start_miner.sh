#!/usr/bin/env bash
set -euo pipefail

children=()
child_names=()

cleanup() {
  for pid in "${children[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

log() {
  printf '[start_miner] %s\n' "$*" >&2
}

wait_http() {
  local url="$1"
  local name="$2"
  local timeout="${3:-1800}"
  local start now
  start="$(date +%s)"
  while true; do
    check_children
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "$name is ready at $url"
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      log "timed out waiting for $name at $url"
      return 1
    fi
    sleep "${LOCAL_LLM_HEALTH_INTERVAL_SECONDS:-10}"
  done
}

check_children() {
  local index pid child_name
  for index in "${!children[@]}"; do
    pid="${children[$index]}"
    child_name="${child_names[$index]}"
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      wait "$pid" || true
      log "$child_name process exited before startup completed"
      return 1
    fi
  done
}

handle_startup_failure() {
  local name="$1"
  local status="$2"

  if [[ "${STARTUP_DIAGNOSTICS_ONLY:-0}" == "1" ]]; then
    log "$name startup failed with status $status ; keeping diagnostics server alive"
    wait -n
    exit $?
  fi

  return "$status"
}

start_vllm() {
  local name="$1"
  local model="$2"
  local port="$3"
  local gpus="$4"
  local tp="$5"
  local extra_args="$6"
  local log_file="/tmp/404-startup-logs/${name}.log"

  log "starting $name vLLM model=$model port=$port gpus=$gpus tp=$tp"
  mkdir -p /tmp/404-startup-logs
  {
    printf '[start_miner] command: CUDA_VISIBLE_DEVICES=%s vllm serve %s --port %s --tensor-parallel-size %s %s %s\n' "$gpus" "$model" "$port" "$tp" "${LOCAL_VLLM_COMMON_ARGS:-}" "$extra_args"
    printf '[start_miner] started_at: %s\n' "$(date -Is)"
  } >> "$log_file"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpus" vllm serve "$model" \
    --host 127.0.0.1 \
    --port "$port" \
    --served-model-name "$model" \
    --tensor-parallel-size "$tp" \
    --dtype "${LOCAL_VLLM_DTYPE:-auto}" \
    --trust-remote-code \
    --generation-config vllm \
    ${LOCAL_VLLM_COMMON_ARGS:-} \
    $extra_args >> "$log_file" 2>&1 &
  children+=("$!")
  child_names+=("$name vLLM")
}

export PRODUCTION_VISION_MODEL="${PRODUCTION_VISION_MODEL:-Qwen/Qwen3-VL-30B-A3B-Thinking}"
export PRODUCTION_CODE_MODEL="${PRODUCTION_CODE_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
export CHUTES_VISION_MODEL="${CHUTES_VISION_MODEL:-$PRODUCTION_VISION_MODEL}"
export CHUTES_CODE_MODEL="${CHUTES_CODE_MODEL:-$PRODUCTION_CODE_MODEL}"
export CHUTES_API_KEY="${CHUTES_API_KEY:-local-vllm}"
export LOCAL_VLLM_COMMON_ARGS="${LOCAL_VLLM_COMMON_ARGS:---moe-backend triton}"
export LOCAL_LLM_START_MODE="${LOCAL_LLM_START_MODE:-managed}"

if [[ "${STARTUP_DIAGNOSTICS:-0}" == "1" ]]; then
  log "STARTUP_DIAGNOSTICS=1 ; starting diagnostics server on port 10006"
  python -m miner_reference.startup_diagnostics &
  children+=("$!")
  child_names+=("startup diagnostics")
fi

if [[ "${MINER_INFERENCE_BACKEND:-local_vllm}" == "local_vllm" ]]; then
  export LOCAL_VISION_PORT="${LOCAL_VISION_PORT:-8001}"
  export LOCAL_CODE_PORT="${LOCAL_CODE_PORT:-8002}"
  export LOCAL_LLM_ROUTER_PORT="${LOCAL_LLM_ROUTER_PORT:-8000}"
  export LOCAL_MANAGED_PORT="${LOCAL_MANAGED_PORT:-8010}"
  export LOCAL_VISION_BASE_URL="http://127.0.0.1:${LOCAL_VISION_PORT}/v1"
  export LOCAL_CODE_BASE_URL="http://127.0.0.1:${LOCAL_CODE_PORT}/v1"
  export LOCAL_MANAGED_BASE_URL="http://127.0.0.1:${LOCAL_MANAGED_PORT}/v1"
  export CHUTES_BASE_URL="http://127.0.0.1:${LOCAL_LLM_ROUTER_PORT}/v1"
  export LOCAL_LLM_ROUTER_MODE="${LOCAL_LLM_ROUTER_MODE:-managed}"

  if [[ "$LOCAL_LLM_START_MODE" == "managed" ]]; then
    log "starting managed local LLM router on port ${LOCAL_LLM_ROUTER_PORT}"
    python -m miner_reference.llm_router &
    children+=("$!")
    child_names+=("local LLM router")
    wait_http "http://127.0.0.1:${LOCAL_LLM_ROUTER_PORT}/health" "local LLM router" 120

    if [[ -n "${LOCAL_LLM_PRELOAD_MODEL:-vision}" ]]; then
      log "preloading ${LOCAL_LLM_PRELOAD_MODEL:-vision} model through managed router"
      curl -fsS \
        -X POST "http://127.0.0.1:${LOCAL_LLM_ROUTER_PORT}/admin/preload" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${LOCAL_LLM_PRELOAD_MODEL:-vision}\"}" >/dev/null \
        || handle_startup_failure "managed local LLM preload" "$?"
    fi
  elif [[ "$LOCAL_LLM_START_MODE" == "both" || "$LOCAL_LLM_START_MODE" == "vision" ]]; then
    start_vllm \
      "vision" \
      "$CHUTES_VISION_MODEL" \
      "$LOCAL_VISION_PORT" \
      "${LOCAL_VISION_CUDA_DEVICES:-0,1,2,3}" \
      "${LOCAL_VISION_TENSOR_PARALLEL_SIZE:-4}" \
      "${LOCAL_VISION_VLLM_ARGS:-}"
  fi

  if [[ "$LOCAL_LLM_START_MODE" == "both" || "$LOCAL_LLM_START_MODE" == "code" ]]; then
    start_vllm \
      "code" \
      "$CHUTES_CODE_MODEL" \
      "$LOCAL_CODE_PORT" \
      "${LOCAL_CODE_CUDA_DEVICES:-0,1,2,3}" \
      "${LOCAL_CODE_TENSOR_PARALLEL_SIZE:-4}" \
      "${LOCAL_CODE_VLLM_ARGS:-}"
  fi

  if [[ "$LOCAL_LLM_START_MODE" == "both" || "$LOCAL_LLM_START_MODE" == "vision" ]]; then
    wait_http "http://127.0.0.1:${LOCAL_VISION_PORT}/v1/models" "vision vLLM" "${LOCAL_LLM_READY_TIMEOUT_SECONDS:-3600}" \
      || handle_startup_failure "vision vLLM" "$?"
  fi
  if [[ "$LOCAL_LLM_START_MODE" == "both" || "$LOCAL_LLM_START_MODE" == "code" ]]; then
    wait_http "http://127.0.0.1:${LOCAL_CODE_PORT}/v1/models" "code vLLM" "${LOCAL_LLM_READY_TIMEOUT_SECONDS:-3600}" \
      || handle_startup_failure "code vLLM" "$?"
  fi

  if [[ "${STARTUP_DIAGNOSTICS_ONLY:-0}" == "1" ]]; then
    log "STARTUP_DIAGNOSTICS_ONLY=1 ; keeping diagnostics server alive"
    wait -n
    exit $?
  fi

  if [[ "$LOCAL_LLM_START_MODE" != "managed" ]]; then
    log "starting local LLM router on port ${LOCAL_LLM_ROUTER_PORT}"
    python -m miner_reference.llm_router &
    children+=("$!")
    child_names+=("local LLM router")
    wait_http "http://127.0.0.1:${LOCAL_LLM_ROUTER_PORT}/health" "local LLM router" 120
  fi
else
  log "MINER_INFERENCE_BACKEND=${MINER_INFERENCE_BACKEND:-} ; using configured CHUTES_BASE_URL=${CHUTES_BASE_URL:-unset}"
fi

log "starting miner API on port 10006"
exec python main.py
