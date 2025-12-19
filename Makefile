# --- デフォルトパラメータ (コマンドラインで上書き可能) ---
P ?= 0.5
A ?= 0.5
SEED ?= 1
STEPS ?= 10000
WARMUP ?= 10000

# 計算するSeedの最大値 (デフォルト: 1000)
MAX_SEED ?= 10

# --- コマンド定義 ---

# --- バックアップ設定 ---
# NASのリモート名（git remote -v で確認したもの）
REMOTE_NAME = NAS
BRANCH_NAME = main

# NASのバックアップ先フォルダパス (末尾にスラッシュをつけない)
NAS_PATH = /Volumes/data/Sasaki/backup_git/MTCargoSim

# Macの標準的なpythonコマンド(venv等あれば適宜変更)
PYTHON_CMD = python3
JULIA_CMD = julia --project=.

# --- ファイルパスの定義 ---
# Juliaが出力するディレクトリパス (main.jlのロジックに合わせる)
DATA_DIR = data/MTC/P$(P)_A$(A)/seed$(SEED)
# ターゲットとなるデータファイル
DATA_FILE = $(DATA_DIR)/positions_history.npy

# Pythonが出力する動画ファイルパス (animate.pyのロジックに合わせる)
ANIM_DIR = animation/MTC/P$(P)_A$(A)
ANIM_FILE = $(ANIM_DIR)/seed$(SEED).mov

# --- ソースコード ---
JULIA_SRC = src/julia/main.jl
MT_SRC = src/julia/only_MT.jl
PYTHON_SRC = src/python/animate.py

# --- タスク ---

.PHONY: all clean help

# デフォルトターゲット: 動画を作成する
all: $(ANIM_FILE)
	@echo "✨ 全工程完了: $(ANIM_FILE)"

# ルール: 動画を作るには、データファイルとPythonコードが必要
$(ANIM_FILE): $(DATA_FILE) $(PYTHON_SRC)
	@echo "🎥 Pythonでアニメーション生成中..."
	$(PYTHON_CMD) $(PYTHON_SRC) --P $(P) --A $(A) --seed $(SEED)

# ルール: データファイルを作るには、Juliaコードが必要
$(DATA_FILE): $(JULIA_SRC)
	@echo "🧪 Juliaでシミュレーション計算中... (P=$(P), A=$(A))"
	$(JULIA_CMD) $(JULIA_SRC) --packing_fraction $(P) --A $(A) --seed $(SEED) --steps $(STEPS) --warmup $(WARMUP)

.PHONY: backup
backup:
	@echo "💾 NASへバックアップ中..."
	# コミットされていない変更があれば、自動コミットする（オプション）
	-git add . && git commit -m "Auto-backup via Makefile"
	# Push実行
	git push $(REMOTE_NAME) $(BRANCH_NAME)
	@echo "✅ バックアップ完了"

# --- フルバックアップ設定 (rsync) ---

.PHONY: sync
sync:
	@echo "📦 プロジェクトを丸ごとNASに同期中..."
	# -a: アーカイブモード (属性維持)
	# -v: 詳細表示
	# --delete: ローカルで消したファイルはNASからも消す (完全同期)
	# --exclude: .gitフォルダは巨大になるので除外してもOK (Gitで管理してるなら)
	mkdir -p $(NAS_PATH)
	rsync -av --delete --exclude '.git' ./ $(NAS_PATH)
	@echo "✅ 全データの同期が完了しました: $(NAS_PATH)"

# --- 一括バックアップ設定 ---

.PHONY: save
save:
	@echo "🚀 プロジェクト全体の完全バックアップを開始します..."
	
	@echo "----------------------------------------"
	@echo "1. Git: ソースコードと履歴の保存"
	@echo "----------------------------------------"
	# 変更を全てステージング
	git add .
	# 日付入りで自動コミット (変更がない場合はエラーにせず通過させる '|| true')
	git commit -m "Auto-save: $$(date '+%Y-%m-%d %H:%M:%S')" || echo "⚠️ コミットする変更はありませんでした。"
	# GitHub (またはNASのGitリポジトリ) へ送信
	git push origin main
	
	@echo "----------------------------------------"
	@echo "2. NAS: データファイル(npy/mov)の同期"
	@echo "----------------------------------------"
	# 既存の sync タスクを呼び出す
	$(MAKE) sync
	
	@echo "✅ 全てのバックアップが完了しました！"


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
	@echo "  make MT P=0.5 A=0.0 MAX_SEED=? STEPS=10000 : MTだけシミュレーション

MT:
	@echo "🧪 MTだけシミュレーション... (P=$(P), A=$(A))"
	@seq 1 $(MAX_SEED) | xargs -I{} sh -c '\
		echo "  ... Seed {} 実行中"; \
		$(JULIA_CMD) $(MT_SRC) --seed {} --packing_fraction $(P) --A $(A) --steps $(STEPS); \
	'
