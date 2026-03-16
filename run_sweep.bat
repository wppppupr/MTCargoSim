@echo off
chcp 65001 > nul
REM ----------------------------------------------------
REM 複数の A の値で MTCargoSim を連続実行するバッチスクリプト
REM Windows上でダブルクリックまたはコマンドプロンプトから実行可能
REM ----------------------------------------------------

REM --- 実行パラメータの設定 ---
set PACKING_FRACTION=0.5
set A_START=0.1
set A_END=1.0
set A_STEP=0.1
set STEPS=1000
set SEED=1

echo ===================================================
echo   MTCargoSim シミュレーション実行スクリプト (A範囲指定)
echo ===================================================
echo Packing Fraction: %PACKING_FRACTION%
echo Aの範囲: %A_START% から %A_END% まで (ステップ: %A_STEP%)
echo 実行ステップ数: %STEPS%
echo シード値: %SEED%
echo.

REM Julia スクリプトの実行
echo 実行しています... しばらくお待ちください。
julia src/julia/only_MT.jl --packing_fraction %PACKING_FRACTION% --A_start %A_START% --A_end %A_END% --A_step %A_STEP% --steps %STEPS% --seed %SEED%

echo.
echo ===================================================
echo 全ての処理が完了しました。
pause
