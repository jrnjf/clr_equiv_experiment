#!/usr/bin/env bash
# Is the weekend run alive and making progress? Run this when it looks stuck.
cd "$(dirname "$0")/.." || exit 1
echo "=== now: $(date '+%a %d %b %H:%M') ==="
echo
echo "--- process alive? ---"
if pgrep -af "run_equivalence.py" >/dev/null; then
  pgrep -af "run_equivalence.py" | head -3
else
  echo "NO run_equivalence.py process running."
fi
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
           --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
echo
echo "--- log tail ---"
for f in results_weekend/*.log; do
  [ -e "$f" ] || continue
  echo "  [$f]  $(wc -l < "$f") lines, modified $(date -r "$f" '+%H:%M')"
  tail -2 "$f" | sed 's/^/    /'
done
[ -e results_weekend/*.log ] 2>/dev/null || echo "  (no logs yet in results_weekend/)"
echo
echo "--- result files written in the last 3 h ---"
find results* -name "*_seed*.json" -mmin -180 2>/dev/null | head -10
echo
echo "--- count per battery arm ---"
ls results/ 2>/dev/null | grep -cE "AdamW_cosine.*_wd" | sed 's/^/  AdamW wd-grid files: /'
ls results_cifar100/ 2>/dev/null | grep -c "_c100" | sed 's/^/  CIFAR-100 files: /'
