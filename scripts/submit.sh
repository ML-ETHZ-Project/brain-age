#!/usr/bin/env bash
# Submit a predictions CSV to the Kaggle competition.
# Usage: scripts/submit.sh submissions/my_submission.csv "short description of the approach"
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <path-to-csv> [\"message\"]" >&2
  exit 1
fi

FILE="$1"
MESSAGE="${2:-submission $(git rev-parse --short HEAD)}"
COMPETITION="eth-fdd-competition"

kaggle competitions submit -c "$COMPETITION" -f "$FILE" -m "$MESSAGE"
