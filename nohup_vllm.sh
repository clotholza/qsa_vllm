#!/usr/bin/env bash
set -euo pipefail
qsa_runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
qsa_log_dir="${QSA_LOG_DIR:-${qsa_runtime_root}/runtime}"
mkdir -p "${qsa_log_dir}"
qsa_log_dir="$(cd -- "${qsa_log_dir}" && pwd)"
qsa_log_file="${QSA_LOG_FILE:-${qsa_log_dir}/server_$(date +%Y%m%d_%H%M%S).log}"
if [[ "${qsa_log_file}" != /* ]]; then
    qsa_log_file="${PWD}/${qsa_log_file}"
fi
mkdir -p "$(dirname -- "${qsa_log_file}")"
qsa_launch_script="${QSA_LAUNCH_SCRIPT:-${qsa_runtime_root}/serve.sh}"
nohup setsid bash "${qsa_launch_script}" "$@" > "${qsa_log_file}" 2>&1 < /dev/null &
qsa_server_pid=$!
echo "${qsa_server_pid}" > "${qsa_log_dir}/server.pid"
ln -sfn "${qsa_log_file}" "${qsa_log_dir}/server.latest.log"
printf 'PID: %s\nLog: %s\n' "${qsa_server_pid}" "${qsa_log_file}"
