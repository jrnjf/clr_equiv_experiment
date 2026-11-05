#!/usr/bin/env bash
# Weekend runbook: runs the four remaining experiment batteries in order.
#
#   bash scripts/weekend_run.sh            # everything (~38-42 h)
#   bash scripts/weekend_run.sh --from 2   # start at battery 2
#   bash scripts/weekend_run.sh --list     # show the plan
#
# Each battery is independent: if one crashes the rest still run, and a battery
# that already has all its result files is skipped on re-run.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=${PY:-.venv/bin/python}
OUT=results_weekend
mkdir -p "$OUT"

# name | config | sweep | expected hours | what it decides
BATTERIES=(
  "0_momentum|cifar|configs/wk0_momentum.json|7|THE STANDARD RECIPE we never ran: SGD momentum 0.9 -- can shrink the 4.1x"
  "1_adamw|cifar|configs/wk1_adamw.json|13|can delete the '+1.1pp over AdamW' claim"
  "2_cifar100|cifar100|configs/wk2_cifar100.json|13|main track vs workshop"
  "3_method_lr|cifar|configs/wk3_method_lr.json|7|answers 'why peak lr 0.8?' and tests the title's robustness claim"
  "4_step_decay|cifar|configs/wk4_step_decay.json|4|classic pre-cosine baseline (stretch)"
  "5_budget_seeds|cifar|configs/wk3_budget_seeds.json|6|peak-vs-budget claim to n=3 (stretch)"
  "6_headline_seeds|cifar|configs/wk4_headline_seeds.json|4|headline arms to n=5 (stretch)"
)

if [ "${1:-}" = "--list" ]; then
  total=0
  for b in "${BATTERIES[@]}"; do
    IFS='|' read -r name cfg sweep hrs why <<< "$b"
    printf '  %-18s %-10s %-34s ~%sh  %s\n' "$name" "$cfg" "$sweep" "$hrs" "$why"
    total=$((total + hrs))
  done
  echo "  ------------------------------------------------"
  echo "  total ~${total}h"
  exit 0
fi

FROM=1
if [ "${1:-}" = "--from" ]; then FROM=${2:-1}; fi

START=$(date +%s)
echo "started $(date '+%a %d %b %H:%M')  (log: $OUT/)"
echo

for b in "${BATTERIES[@]}"; do
  IFS='|' read -r name cfg sweep hrs why <<< "$b"
  idx=${name%%_*}
  if [ "$idx" -lt "$FROM" ]; then
    echo "--- skipping $name (before --from $FROM)"
    continue
  fi
  log="$OUT/$name.log"
  echo "=== [$(date +%H:%M)] START $name  ($why)"
  "$PY" experiments/run_equivalence.py --config "configs/$cfg.json" \
      --sweep "$sweep" >"$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== [$(date +%H:%M)] DONE  $name  ($(grep -c '^===' "$log" || true) runs)"
  else
    echo "=== [$(date +%H:%M)] FAILED $name (exit $rc) -- see $log; continuing"
    tail -5 "$log"
  fi
  echo
done

ELAPSED=$(( ($(date +%s) - START) / 60 ))
echo "finished $(date '+%a %d %b %H:%M')  total ${ELAPSED} min"
echo
echo "Next: python scripts/weekend_report.py"
