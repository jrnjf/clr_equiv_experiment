#!/usr/bin/env bash
# Supervisor-revision campaign: six batteries, one command, resumable.
#
#   bash scripts/supervisor_campaign.sh              # everything (~70 h)
#   bash scripts/supervisor_campaign.sh --list       # show the plan
#   bash scripts/supervisor_campaign.sh --from 2     # resume at battery 2
#   bash scripts/supervisor_campaign.sh --only a b   # run named batteries only
#
# Every run is resumable: --skip-existing means a finished run is loaded, not
# repeated, so an interruption costs only the run in flight. Output streams to the
# terminal (python -u) as well as to results_supervisor/<battery>.log.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=${PY:-.venv/bin/python}
OUT=results_supervisor
mkdir -p "$OUT"

# id | config | sweep | hours | what it answers | review item
BATTERIES=(
  "a_valsplit|cifar_val|configs/sv_a_valsplit.json|20|validation/test split, no test leakage|P0 3.4"
  "b_period|cifar|configs/sv_b_period.json|16|is the one-epoch period special?|P1 6.2"
  "c_peaklr|cifar|configs/sv_c_peaklr.json|13|peak-LR sensitivity of the method|P1 6.3"
  "d_decay|cifar|configs/sv_d_decay.json|7|decay strength vs decay profile|P1 4"
  "e_phase|cifar|configs/sv_e_phase.json|7|cycle phase vs data shuffling|P2 6.1"
  "f_seeds|cifar|configs/sv_f_seeds.json|7|five seeds on headline arms|3.3"
)

if [ "${1:-}" = "--list" ]; then
  total=0
  for b in "${BATTERIES[@]}"; do
    IFS='|' read -r id cfg sweep hrs why item <<< "$b"
    printf '  %-12s %-10s %-32s ~%2sh  %-34s [%s]\n' "$id" "$cfg" "$sweep" "$hrs" "$why" "$item"
    total=$((total + hrs))
  done
  echo "  ------------------------------"
  echo "  total ~${total}h   (P0 first; safe to stop after any battery)"
  exit 0
fi

FROM=1
ONLY=""
if [ "${1:-}" = "--from" ]; then FROM=${2:-1}; elif [ "${1:-}" = "--only" ]; then shift; ONLY="$*"; fi

echo "=== supervisor campaign: $(date '+%a %d %b %H:%M') ==="
echo "=== P0 analyses (no GPU, seconds) ==="
"$PY" scripts/verify_lr_math.py  > "$OUT/verify_lr_math.txt" 2>&1 && echo "  verify_lr_math.txt  ok"
"$PY" scripts/stats_report.py   > "$OUT/stats_report.txt"   2>&1 && echo "  stats_report.txt    ok"
echo

START=$(date +%s)
for b in "${BATTERIES[@]}"; do
  IFS='|' read -r id cfg sweep hrs why item <<< "$b"
  idx=${id%%_*}
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $idx "* ]]; then
    echo "--- skip $id (not in --only)"; continue
  fi
  # numeric --from check against battery order a=1..f=6
  order=$(printf '%s' "$idx" | tr 'abcdef' '123456')
  if [ -z "$ONLY" ] && [ "$order" -lt "$FROM" ]; then
    echo "--- skip $id (before --from $FROM)"; continue
  fi

  log="$OUT/$id.log"
  before=$(ls results*/*_seed*.json 2>/dev/null | wc -l)
  echo "=== [$(date +%H:%M)] START $id  ($why)  [$item]"
  "$PY" -u experiments/run_equivalence.py --config "configs/$cfg.json" \
       --sweep "$sweep" --skip-existing 2>&1 | tee -a "$log" \
       | grep -E '^===|^---|epoch [0-9]+/'
  # Capture PIPESTATUS IMMEDIATELY. Appending `|| true` here silently clobbered it
  # with the status of `true`, so a battery that crashed at step 0 was reported as
  # DONE -- which is exactly how battery a appeared to finish in 0 minutes.
  rc=${PIPESTATUS[0]}
  after=$(ls results*/*_seed*.json 2>/dev/null | wc -l)
  new_files=$((after - before))
  if [ $rc -ne 0 ]; then
    echo "=== [$(date +%H:%M)] FAILED $id (exit $rc), $new_files new result files -- continuing"
    echo "    last error lines:"; tail -4 "$log" | sed 's/^/      /'
  elif [ "$new_files" -eq 0 ]; then
    echo "=== [$(date +%H:%M)] SUSPECT $id: exit 0 but produced NO new result files."
    echo "    Everything was skipped by --skip-existing, or the config produced no runs."
    echo "    Check: tail -20 $log"
  else
    echo "=== [$(date +%H:%M)] DONE  $id ($new_files new result files)"
  fi
  echo
done

echo "=== [$(date +%H:%M)] analyses on the new results ==="
"$PY" scripts/stats_report.py    > "$OUT/stats_report_after.txt"    2>&1 && echo "  stats_report_after.txt"
"$PY" scripts/weekend_report.py  > "$OUT/results_after.txt" 2>&1 && echo "  results_after.txt"
"$PY" scripts/supervisor_report.py > "$OUT/supervisor_report.txt" 2>&1 \
  && echo "  supervisor_report.txt  <- batteries b-e" || echo "  (supervisor_report.py not present yet)"

ELAPSED=$(( ($(date +%s) - START) / 60 ))
echo
echo "=== finished $(date '+%a %d %b %H:%M')  total ${ELAPSED} min ==="
