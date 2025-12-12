# --- デフォルトパラメータ (コマンドラインで上書き可能) ---
P ?= 0.5
A ?= 0.5
F ?= 0.001
SEED ?= 1
STEPS ?= 1000
WARMUP ?= 1000

# --- コマンド定義 ---

# --- バックアップ設定 ---
# NASのリモート名（git remote -v で確認したもの）
REMOTE_NAME = nas
BRANCH_NAME = main

# Macの標準的なpythonコマンド(venv等あれば適宜変更)
PYTHON_CMD = python3
JULIA_CMD = julia --project=.

# --- ファイルパスの定義 ---
# Juliaが出力するディレクトリパス (main.jlのロジックに合わせる)
DATA_DIR = data/P$(P)_A$(A)_F$(F)/seed$(SEED)
# ターゲットとなるデータファイル
DATA_FILE = $(DATA_DIR)/positions_history.npy

# Pythonが出力する動画ファイルパス (animate.pyのロジックに合わせる)
ANIM_DIR = animation/P$(P)_A$(A)_F$(F)
ANIM_FILE = $(ANIM_DIR)/seed$(SEED).mov

# --- ソースコード ---
JULIA_SRC = src/julia/main.jl
PYTHON_SRC = src/python/animate.py

# --- タスク ---

.PHONY: all clean help

# デフォルトターゲット: 動画を作成する
all: $(ANIM_FILE)
	@echo "✨ 全工程完了: $(ANIM_FILE)"

# ルール: 動画を作るには、データファイルとPythonコードが必要
$(ANIM_FILE): $(DATA_FILE) $(PYTHON_SRC)
	@echo "🎥 Pythonでアニメーション生成中..."
	$(PYTHON_CMD) $(PYTHON_SRC) --P $(P) --A $(A) --F $(F) --seed $(SEED)

# ルール: データファイルを作るには、Juliaコードが必要
$(DATA_FILE): $(JULIA_SRC)
	@echo "🧪 Juliaでシミュレーション計算中... (P=$(P), A=$(A), F=$(F))"
	$(JULIA_CMD) $(JULIA_SRC) --packing_fraction $(P) --A $(A) --force $(F) --seed $(SEED) --steps $(STEPS) --warmup $(WARMUP)

.PHONY: backup
backup:
	@echo "💾 NASへバックアップ中..."
	# コミットされていない変更があれば、自動コミットする（オプション）
	-git add . && git commit -m "Auto-backup via Makefile"
	# Push実行
	git push $(REMOTE_NAME) $(BRANCH_NAME)
	@echo "✅ バックアップ完了"

# 生成物を削除
clean:
	rm -rf data animation
	@echo "🗑️ データを削除しました"

# ヘルプ表示
help:
	@echo "使用方法:"
	@echo "  make              : デフォルト設定 (P=0.5, A=0.5...) で実行"
	@echo "  make P=0.8 A=1.0  : パラメータを指定して実行"
	@echo "  make clean        : 生成されたデータを削除"