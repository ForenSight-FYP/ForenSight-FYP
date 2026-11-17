#!/usr/bin/env bash
# Fetch the official AASIST implementation and pretrained weights (Jung et al., 2022).
# Licensed by its authors (see third_party/aasist/LICENSE); kept out of our git history.
set -euo pipefail
DEST="${1:-third_party/aasist}"
if [ -d "$DEST/.git" ]; then
  echo "AASIST already present in $DEST"; exit 0
fi
mkdir -p "$(dirname "$DEST")"
git clone --depth 1 https://github.com/clovaai/aasist.git "$DEST"
ls "$DEST/models/weights" && echo "AASIST ready in $DEST"
