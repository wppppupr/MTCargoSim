using ArgParse
# 作成したモジュールファイルを読み込む
include("MTC.jl")
# モジュールを使う宣言
using .MTC

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--packing_fraction", "-p"
            help = "Packing fraction (default: 0.5)"
            arg_type = Float64
            default = 0.5
        "--A", "-a"
            help = "Alignment interaction strength (default: 0.5)"
            arg_type = Float64
            default = 0.5
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
    
    println("実行パラメータ:")
    println("  packing_fraction = $(args["packing_fraction"])")
    println("  A = $(args["A"])")
    println("  seed = $(args["seed"])")
    println("  simulation steps = $(args["steps"])")
    println()

    params = Parameters(
        packing_fraction=args["packing_fraction"],
        A=args["A"],
        seed=args["seed"]
    )
    
    final_data = MT_simulation(params, args["steps"])

    println("\nシミュレーション完了!")
end