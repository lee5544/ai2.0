#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_DIR="$PWD/build/external_label_env_mac"
PACKAGE_DIR="$PWD/dist/ExternalLabelPortable_mac_arm64"
RUNTIME_ZIP="$PWD/build/external_label_runtime_mac_arm64.zip"
OUT_ZIP="$PWD/dist/ExternalLabelPortable_mac_arm64.zip"
export CONDA_PKGS_DIRS="$PWD/.conda_pkgs"
export XDG_CACHE_HOME="$PWD/.cache"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please run this from a shell with conda initialized."
  exit 1
fi

rm -rf "$ENV_DIR" "$PACKAGE_DIR" "$RUNTIME_ZIP" "$OUT_ZIP"
mkdir -p "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME"

echo "Creating slim Python runtime..."
conda create -y -p "$ENV_DIR" -c conda-forge \
  python=3.12 streamlit pandas numpy plotly nptdms pyyaml zstandard librosa conda-pack

echo "Packing runtime..."
conda run -p "$ENV_DIR" conda-pack -p "$ENV_DIR" -o "$RUNTIME_ZIP" --force --zip-symlinks

mkdir -p "$PACKAGE_DIR/runtime"

echo "Unpacking runtime into portable package..."
ditto -x -k "$RUNTIME_ZIP" "$PACKAGE_DIR/runtime"

cp -R external_label "$PACKAGE_DIR/external_label"
cp -R cfg "$PACKAGE_DIR/cfg"
cp scripts/start_external_label_portable_mac.command "$PACKAGE_DIR/启动.command"
chmod +x "$PACKAGE_DIR/启动.command"

echo "Creating zip package..."
ditto -c -k --keepParent "$PACKAGE_DIR" "$OUT_ZIP"

echo
echo "Portable package complete:"
echo "$OUT_ZIP"
