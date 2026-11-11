#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p logs
exec >>logs/analysis-job.log 2>&1
trap 's=$?; echo "$(date -Is) exit $s at line $LINENO" > analysis.failed; exit "$s"' ERR
echo "$(date -Is) tmux=$TMUX analysis waiting for eval.ready"
until test -f eval.ready; do sleep 15; done
test -f train.ready
sha256sum --quiet -c release.ready
"$HOME/mmdl_test/.venv/bin/python" -u analyze.py
test -s analysis/metrics.json
test -s analysis/report.md
echo "$(date -Is) automatic analysis complete, see analysis/report.md"
date -Is > analysis-job.done
