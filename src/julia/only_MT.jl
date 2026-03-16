using ArgParse
# 作成したモジュールファイルを読み込む
include("MTC.jl")
# モジュールを使う宣言
using .MTC

BASE_PATH="D:\\Sasaki\\MTCargoSim\\MT"

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--packing_fraction", "-p"
            help = "Packing fraction (default: 0.5)"
            arg_type = Float64
            default = 0.5
        "--A", "-a"
            help = "Alignment interaction strength (default: 0.5). Used if A_start/A_end are not specified."
            arg_type = Float64
            default = 0.5
        "--A_start"
            help = "Start value of A for parameter sweep. If specified with A_end, sweeps A."
            arg_type = Float64
            default = NaN
        "--A_end"
            help = "End value of A for parameter sweep."
            arg_type = Float64
            default = NaN
        "--A_step"
            help = "Step size of A for parameter sweep."
            arg_type = Float64
            default = 0.1
        "--seed", "-s"
            help = "Random seed (default: 1)"
            arg_type = Int
            default = 1
        "--steps", "-n"
            help = "Number of simulation steps (default: 1000)"
            arg_type = Int
            default = 1000
    end

    return parse_args(s)
end

# シミュレーションの実行例（直接実行した場合のみ実行されます）
if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    
    # Aのリストを作成
    A_values = Float64[]
    if !isnan(args["A_start"]) && !isnan(args["A_end"])
        A_start = args["A_start"]
        A_end = args["A_end"]
        A_step = args["A_step"]
        # rangeオブジェクトをcollectして配列化
        A_values = collect(A_start:A_step:A_end)
    else
        push!(A_values, args["A"])
    end

    println("=== シミュレーション一括実行開始 ===")
    println("実行パラメータ設定:")
    println("  packing_fraction = $(args["packing_fraction"])")
    if length(A_values) > 1
        println("  対象とするAの値 = $(args["A_start"]) から $(args["A_end"]) まで (ステップ: $(args["A_step"]))")
    else
        println("  A = $(A_values[1])")
    end
    println("  seed = $(args["seed"])")
    println("  simulation steps = $(args["steps"])")
    println()

    for (i, current_A) in enumerate(A_values)
        if length(A_values) > 1
            println("--------------------------------------------------")
            println("[$i/$(length(A_values))] A = $current_A のシミュレーションを実行中...")
        end
        
        params = Parameters(
            packing_fraction=args["packing_fraction"],
            A=current_A,
            seed=args["seed"]
        )
        
        final_data = MT_simulation(params, args["steps"]; base_path = BASE_PATH)
        
        if length(A_values) > 1
            println("-> A = $current_A の計算完了")
        end
    end

    println("\nすべてのシミュレーション完了!")
end