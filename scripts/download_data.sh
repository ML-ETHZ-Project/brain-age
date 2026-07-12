#!/usr/bin/env bash
# Download and unzip the competition data into data/raw/.
# Requires ~/.kaggle/kaggle.json to be set up (see README).
set -euo pipefail

COMPETITION="eth-fdd-competition"
DEST="data/raw"

mkdir -p "$DEST"
kaggle competitions download -c "$COMPETITION" -p "$DEST"

for f in "$DEST"/*.zip; do
  [ -e "$f" ] || continue
  unzip -o "$f" -d "$DEST"
  rm "$f"
done

echo "Data downloaded to $DEST/"
