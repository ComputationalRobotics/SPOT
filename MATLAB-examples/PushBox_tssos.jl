using TSSOS
using DynamicPolynomials
using DelimitedFiles
using LinearAlgebra
using Printf
using ControlSystems
using Plots
using MAT

function get_poly(rpt, coeff, v)
    monomial_num = length(coeff)
    rpt = Int.(rpt)
    d = size(rpt, 2)
    poly_list = []
    for i in 1: monomial_num
        monomial = 1.0
        for j = 1: d
            idx = rpt[i, j]
            if idx > 0 
                monomial = monomial * v[idx]
            end
        end
        push!(poly_list, coeff[i] * monomial)
    end

    return sum(poly_list)
end

# prefix 
prefix = "2025-01-30_01-03-11"
prefix = prefix * "/"

# commands
ts = "MD"
cs = "MF"
if_self = false
kappa = 2
if_solution = false
if_MATLAB = false
MATLAB_filename = "v_opt_ordered"

# folders
prefix_tssos = prefix
prefix_MATLAB = prefix 

# load polynomials and params
polys_file = matread("./data/PushBox_MATLAB/" * prefix_MATLAB * "polys.mat")
params_file = matread("./data/PushBox_MATLAB/" * prefix_MATLAB * "params.mat")
params = params_file["params"]
total_var_num = Int(params["total_var_num"])
cliques = params["self_cliques"]
cliques_new = []
for arr in cliques
    tmp = []
    for idx in arr
        push!(tmp, Int(idx))
    end
    sort!(tmp)
    push!(cliques_new, Int.(tmp))
end
if if_self
    cliques = cliques_new
else
    cliques = []
end

@polyvar v[1: total_var_num]
eq_sys = [] # polynomial equality constraints
ineq_sys = [] # polynomial inequality constraints
obj_sys = 0 # polynomial objective

# read polynomials
coeff_f = polys_file["coeff_f"]
rpt_f = polys_file["supp_rpt_f"]
obj_sys = get_poly(rpt_f, coeff_f, v)

coeff_gs = polys_file["coeff_g"]
rpt_gs = polys_file["supp_rpt_g"]
for i in 1: length(coeff_gs)
    push!(ineq_sys, get_poly(rpt_gs[i], coeff_gs[i], v))
end

coeff_hs = polys_file["coeff_h"]
rpt_hs = polys_file["supp_rpt_h"]
for i in 1: length(coeff_hs)
    push!(eq_sys, get_poly(rpt_hs[i], coeff_hs[i], v))
end

F = [obj_sys]
for i in eachindex(ineq_sys)
    push!(F, ineq_sys[i])
end
for i in eachindex(eq_sys)
    push!(F, eq_sys[i])
end
opt, sol, data = cs_tssos_first(F, v, kappa, numeq=length(eq_sys), CS=cs, cliques=[], TS=ts, solution=false, QUIET=false)
println("Finish!")

open("julia_solution.txt", "a") do f
    println(f, "PushBox    TS=$ts, CS=$cs, result=$(string(round(opt, digits=20))), operation = , mosek = ")
end

if if_solution
    # recover solution
    v_opt = sol

    # save data
    if !isdir("./data/PushBox_tssos/" * prefix_tssos)
        mkpath("./data/PushBox_tssos/" * prefix_tssos)
    end
    file = matopen("./data/PushBox_tssos/" * prefix_tssos * "data.mat", "w")  
    write(file, "v_opt", v_opt)
end

# test MATLAB's result
if if_MATLAB && if_solution
    if MATLAB_filename == "v_opt_naive"
        println("Naive v_opt!")
        file = matread("./data/PushBox_MATLAB/" * prefix_MATLAB * "v_opt_naive.mat")
        sol_MATLAB = file["v_opt_naive"]
        file_saved = "data_naive_IPOPT.mat"
        ipopt_log_file = "naive_IPOPT.txt"
    elseif MATLAB_filename == "v_opt_robust"
        println("Robust v_opt!")
        file = matread("./data/PushBox_MATLAB/" * prefix_MATLAB * "v_opt_robust.mat")
        file_saved = "data_robust_IPOPT.mat"
        sol_MATLAB = file["v_opt_robust"]
        ipopt_log_file = "robust_IPOPT.txt"
    elseif MATLAB_filename == "v_opt_ordered"
        println("Ordered v_opt!")
        file = matread("./data/PushBox_MATLAB/" * prefix_MATLAB * "v_opt_ordered.mat")
        sol_MATLAB = file["v_opt_ordered"]
        file_saved = "data_ordered_IPOPT.mat"
        ipopt_log_file = "ordered_IPOPT.txt"
    end
    
    if !isdir("./logs/PushBox_MATLAB/" * prefix_MATLAB)
        mkpath("./logs/PushBox_MATLAB/" * prefix_MATLAB)
    end
    logfile = open("./logs/PushBox_MATLAB/" * prefix_MATLAB * ipopt_log_file, "w")
    redirect_stdout(logfile)
    redirect_stderr(logfile)

    # sol_MATLAB_refined = TSSOS.refine_sol(opt, sol_MATLAB, data)
    n = data.n
    nb = data.nb
    numeq = data.numeq
    m = data.m
    supp = data.supp
    coe = data.coe
    ub, sol_MATLAB_refined, status = TSSOS.local_solution(n, m, supp, coe, nb=nb, numeq=numeq, startpoint=sol_MATLAB, QUIET=false)
    println("MATLAB round finish!")
    # save data again 
    file = matopen("./data/PushBox_MATLAB/" * prefix_MATLAB * file_saved, "w") 
    v_opt = sol_MATLAB_refined
    write(file, "v_opt", v_opt)
    close(file)

    redirect_stdout()
    redirect_stderr()
    close(logfile)
end