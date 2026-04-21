#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PORT="${APP_PORT:-7861}"
APP_HOST="${APP_HOST:-127.0.0.1}"
STATE_DIR="${ROOT_DIR}/.voice-studio"
APP_LOG="${STATE_DIR}/voice-studio.log"
TUNNEL_LOG="${STATE_DIR}/zrok.log"
APP_PID_FILE="${STATE_DIR}/voice-studio.pid"
TUNNEL_PID_FILE="${STATE_DIR}/zrok.pid"
RESERVED_NAME_FILE="${STATE_DIR}/zrok-reserved-name"
TARGET_URL="http://${APP_HOST}:${APP_PORT}"
APP_TARGET="${APP_HOST}:${APP_PORT}"
ZROK_UNIQUE_NAME="${ZROK_UNIQUE_NAME:-}"
ZROK_ENABLE_TOKEN="${ZROK_ENABLE_TOKEN:-}"

mkdir -p "${STATE_DIR}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -x "$1" ]]; then
    echo "Missing required executable: $1" >&2
    exit 1
  fi
}

cleanup_stale_pid() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      rm -f "${pid_file}"
    fi
  fi
}

is_http_ready() {
  curl -fsS "${TARGET_URL}" >/dev/null 2>&1
}

wait_for_http() {
  local attempts=0
  until is_http_ready; do
    attempts=$((attempts + 1))
    if [[ "${attempts}" -ge 90 ]]; then
      echo "Voice Studio did not become ready on ${TARGET_URL}" >&2
      exit 1
    fi
    sleep 1
  done
}

start_app_if_needed() {
  cleanup_stale_pid "${APP_PID_FILE}"

  if is_http_ready; then
    return
  fi

  if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    echo "Missing ${ROOT_DIR}/.venv/bin/python" >&2
    exit 1
  fi

  echo "Starting Voice Studio on ${TARGET_URL} ..."
  nohup env PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    F5_TTS_REPO_ROOT="${ROOT_DIR}" \
    "${ROOT_DIR}/.venv/bin/python" -m f5_tts.infer.voice_studio --host "${APP_HOST}" --port "${APP_PORT}" \
    >"${APP_LOG}" 2>&1 &
  echo $! >"${APP_PID_FILE}"
  wait_for_http
}

zrok_is_enabled() {
  [[ -f "${HOME}/.zrok/environment.json" ]]
}

enable_zrok_if_needed() {
  if zrok_is_enabled; then
    return
  fi

  if [[ -z "${ZROK_ENABLE_TOKEN}" ]]; then
    cat >&2 <<EOF
zrok is installed but not enabled on this Mac yet.

One-time setup:
  1. Create a free account at https://myzrok.io
  2. Copy your account token
  3. Re-run:
     ZROK_ENABLE_TOKEN=your_token ./scripts/voice_studio_zrok.sh

Optional:
  ZROK_UNIQUE_NAME=your-name ./scripts/voice_studio_zrok.sh
  This tries to reserve a stable public share name the first time.
EOF
    exit 1
  fi

  echo "Enabling zrok on this Mac ..."
  zrok enable "${ZROK_ENABLE_TOKEN}" --headless
}

stop_existing_tunnel() {
  cleanup_stale_pid "${TUNNEL_PID_FILE}"

  if [[ -f "${TUNNEL_PID_FILE}" ]]; then
    kill "$(cat "${TUNNEL_PID_FILE}")" >/dev/null 2>&1 || true
    rm -f "${TUNNEL_PID_FILE}"
  fi
}

reserve_name_if_needed() {
  if [[ -z "${ZROK_UNIQUE_NAME}" ]]; then
    if [[ -f "${RESERVED_NAME_FILE}" ]]; then
      ZROK_UNIQUE_NAME="$(cat "${RESERVED_NAME_FILE}")"
    fi
    return
  fi

  if [[ -f "${RESERVED_NAME_FILE}" ]] && [[ "$(cat "${RESERVED_NAME_FILE}")" == "${ZROK_UNIQUE_NAME}" ]]; then
    return
  fi

  echo "Reserving zrok share name '${ZROK_UNIQUE_NAME}' ..."
  if zrok reserve public "${APP_TARGET}" --unique-name "${ZROK_UNIQUE_NAME}" --json-output >/dev/null 2>&1; then
    printf '%s\n' "${ZROK_UNIQUE_NAME}" >"${RESERVED_NAME_FILE}"
    return
  fi

  echo "Could not reserve '${ZROK_UNIQUE_NAME}'. It may already exist or be unavailable." >&2
  exit 1
}

start_tunnel() {
  : >"${TUNNEL_LOG}"

  if [[ -n "${ZROK_UNIQUE_NAME}" ]]; then
    nohup script -q /dev/null zrok share reserved "${ZROK_UNIQUE_NAME}" --headless --force-local --override-endpoint "${TARGET_URL}" >"${TUNNEL_LOG}" 2>&1 &
  else
    nohup script -q /dev/null zrok share public "${APP_TARGET}" --headless --force-local >"${TUNNEL_LOG}" 2>&1 &
  fi

  echo $! >"${TUNNEL_PID_FILE}"
}

extract_public_url() {
  local attempts=0
  while [[ "${attempts}" -lt 60 ]]; do
    local url
    url="$(
      zrok overview 2>/dev/null | python3 - "${APP_PORT}" "${ZROK_UNIQUE_NAME}" <<'PY'
import json
import sys

target_port = sys.argv[1]
unique_name = sys.argv[2]

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)

matches = []
for env in payload.get("environments", []):
    for share in env.get("shares", []):
        backend = share.get("backendProxyEndpoint", "")
        frontend = share.get("frontendEndpoint", "")
        token = share.get("shareToken", "")
        if not frontend:
            continue
        if unique_name and token == unique_name:
            matches.append(frontend)
            continue
        if backend.endswith(f":{target_port}"):
            matches.append(frontend)

if matches:
    print(matches[-1])
PY
    )"
    if [[ -n "${url}" ]]; then
      printf '%s\n' "${url}"
      return 0
    fi
    attempts=$((attempts + 1))
    sleep 1
  done

  echo "Could not find a zrok public URL. Check ${TUNNEL_LOG}" >&2
  exit 1
}

require_command curl
require_command zrok
require_file /usr/bin/script

cd "${ROOT_DIR}"
start_app_if_needed
enable_zrok_if_needed
stop_existing_tunnel
reserve_name_if_needed
start_tunnel

PUBLIC_URL="$(extract_public_url)"

cat <<EOF
Voice Studio local app: ${TARGET_URL}
zrok URL: ${PUBLIC_URL}

Logs:
  app: ${APP_LOG}
  tunnel: ${TUNNEL_LOG}
EOF
