#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PORT="${APP_PORT:-7861}"
APP_HOST="${APP_HOST:-127.0.0.1}"
STATE_DIR="${ROOT_DIR}/.voice-studio"
APP_LOG="${STATE_DIR}/voice-studio.log"
TUNNEL_LOG="${STATE_DIR}/cloudflared.log"
APP_PID_FILE="${STATE_DIR}/voice-studio.pid"
TUNNEL_PID_FILE="${STATE_DIR}/cloudflared.pid"

mkdir -p "${STATE_DIR}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
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
  curl -fsS "http://${APP_HOST}:${APP_PORT}" >/dev/null 2>&1
}

wait_for_http() {
  local attempts=0
  until is_http_ready; do
    attempts=$((attempts + 1))
    if [[ "${attempts}" -ge 90 ]]; then
      echo "Voice Studio did not become ready on http://${APP_HOST}:${APP_PORT}" >&2
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

  echo "Starting Voice Studio on http://${APP_HOST}:${APP_PORT} ..."
  nohup env PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    F5_TTS_REPO_ROOT="${ROOT_DIR}" \
    "${ROOT_DIR}/.venv/bin/python" -m f5_tts.infer.voice_studio --host "${APP_HOST}" --port "${APP_PORT}" \
    >"${APP_LOG}" 2>&1 &
  echo $! >"${APP_PID_FILE}"
  wait_for_http
}

stop_existing_tunnel() {
  cleanup_stale_pid "${TUNNEL_PID_FILE}"

  if [[ -f "${TUNNEL_PID_FILE}" ]]; then
    kill "$(cat "${TUNNEL_PID_FILE}")" >/dev/null 2>&1 || true
    rm -f "${TUNNEL_PID_FILE}"
  fi
}

start_tunnel() {
  : >"${TUNNEL_LOG}"
  nohup cloudflared tunnel --url "http://${APP_HOST}:${APP_PORT}" >"${TUNNEL_LOG}" 2>&1 &
  echo $! >"${TUNNEL_PID_FILE}"
}

extract_tunnel_url() {
  local attempts=0
  while [[ "${attempts}" -lt 60 ]]; do
    if [[ -f "${TUNNEL_LOG}" ]]; then
      local url
      url="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "${TUNNEL_LOG}" | tail -n 1 || true)"
      if [[ -n "${url}" ]]; then
        printf '%s\n' "${url}"
        return 0
      fi
    fi
    attempts=$((attempts + 1))
    sleep 1
  done

  echo "Could not find a Quick Tunnel URL. Check ${TUNNEL_LOG}" >&2
  exit 1
}

require_command curl
require_command cloudflared

cd "${ROOT_DIR}"
start_app_if_needed
stop_existing_tunnel
start_tunnel

PUBLIC_URL="$(extract_tunnel_url)"

cat <<EOF
Voice Studio local app: http://${APP_HOST}:${APP_PORT}
Quick Tunnel URL: ${PUBLIC_URL}

Logs:
  app: ${APP_LOG}
  tunnel: ${TUNNEL_LOG}
EOF
