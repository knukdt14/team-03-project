#!/usr/bin/env bash
# 수리된 벡터스토어 기준으로 무효화된 실험을 전부 순차 재실행한다.
# GPU가 1개라 동시 실행하면 VRAM이 충돌하므로 반드시 하나씩.
set -u
PY="/c/Users/Win11Pro/miniconda3/envs/CATIA/python.exe"
cd "/c/Users/Win11Pro/Documents/GitHub/team-03-project" || exit 1
LOG_DIR="/tmp/exp_logs"
mkdir -p "$LOG_DIR"

run_step () {
  local name="$1"; shift
  echo "===== [START] $name  $(date '+%H:%M:%S') ====="
  "$@" > "$LOG_DIR/$name.log" 2>&1
  local code=$?
  echo "===== [END]   $name  exit=$code  $(date '+%H:%M:%S') ====="
  return 0   # 한 단계가 실패해도 나머지는 계속 진행
}

run_step "01_qe_ab_easy"   "$PY" experiments/query_expansion_ab_test.py
run_step "02_rag_onoff_1.5b" "$PY" experiments/test_bigger_model.py "Qwen/Qwen2.5-1.5B-Instruct" "1.5b"
run_step "03_rag_onoff_3b"   "$PY" experiments/test_bigger_model.py "Qwen/Qwen2.5-3B-Instruct" "3b"
run_step "04_param_grid"     "$PY" experiments/tune_rag_params.py

echo "===== ALL DONE $(date '+%H:%M:%S') ====="
