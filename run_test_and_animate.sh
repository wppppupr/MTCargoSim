#!/bin/bash
set -e

# デフォルトのパラメータ設定 (引数で上書き可能)
# 使い方: ./run_test_and_animate.sh [CARGO_RADIUS] [PACKING_FRACTION] [A] [K_MT] [K_CARGO] [SEED]
CARGO_RADIUS=${1:-3.34}
PACKING_FRACTION=${2:-0.5}
A=${3:-0.5}
K_MT=${4:-0.0904}
K_CARGO=${5:-0.0226}
SEED=${6:-1000}

echo "=========================================="
echo "1. Running Julia simulation test..."
echo "Parameters: Radius=$CARGO_RADIUS, P=$PACKING_FRACTION, A=$A, k_MT=$K_MT, k_cargo=$K_CARGO, Seed=$SEED"
echo "=========================================="

# Juliaスクリプトを実行し、出力を一時ファイルに保存しつつ画面にも表示
TMP_LOG=$(mktemp)
pixi run julia --project=. src/julia/test_Mac.jl "$CARGO_RADIUS" "$PACKING_FRACTION" "$A" "$K_MT" "$K_CARGO" "$SEED" | tee "$TMP_LOG"

echo ""
echo "=========================================="
echo "2. Finding generated Zarr directory..."
echo "=========================================="
# Juliaの出力から実際に保存されたパスを抽出
ZARR_DIR=$(grep "SAVED_PATH:" "$TMP_LOG" | awk '{print $2}')
rm "$TMP_LOG"

if [ -z "$ZARR_DIR" ]; then
    echo "Error: Could not determine Zarr directory from Julia output."
    exit 1
fi

echo "Found: $ZARR_DIR"

echo ""
echo "=========================================="
echo "3. Generating animation..."
echo "=========================================="
pixi run python src/python/animate_radius.py "$ZARR_DIR"

echo ""
echo "=========================================="
echo "Done! Animation should be saved in $ZARR_DIR/animation.mov"
echo "=========================================="
