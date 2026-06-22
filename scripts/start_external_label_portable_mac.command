#!/bin/zsh
set -e

cd "$(dirname "$0")"
export PATH="$PWD/runtime/bin:$PATH"

if [ ! -x "runtime/bin/python" ]; then
  echo "runtime/bin/python not found. Please rebuild the portable package."
  read "?Press Enter to close..."
  exit 1
fi

if [ -x "runtime/bin/conda-unpack" ] && [ ! -f "runtime/.conda_unpacked" ]; then
  echo "Preparing runtime for this folder..."
  runtime/bin/conda-unpack
  echo ok > runtime/.conda_unpacked
fi

open "http://localhost:8501" >/dev/null 2>&1 || true
runtime/bin/python -m streamlit run external_label/app.py \
  --server.headless=true \
  --browser.gatherUsageStats=false

status=$?
if [ "$status" -ne 0 ]; then
  echo
  echo "Program exited with an error."
  read "?Press Enter to close..."
fi
exit "$status"
