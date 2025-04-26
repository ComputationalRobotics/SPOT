clear; close all;

addpath("../pathinfo/");
my_path;

if_mex = true; params.if_mex = if_mex;
kappa = 2; params.kappa = kappa;
relax_mode = "SOS"; params.relax_mode = relax_mode;
cs_mode = "MF"; params.cs_mode = cs_mode;
ts_mode = "MF"; params.ts_mode = ts_mode;
ts_mom_mode = "NON"; params.ts_mom_mode = ts_mom_mode;
ts_eq_mode = "NON"; params.ts_eq_mode = ts_eq_mode; 
if_solve = true; params.if_solve = if_solve;

total_var_num = 3;
v = msspoly('v', 3);
v1 = v(1); v2 = v(2); v3 = v(3);
f = v1 + v2 + v3;
g = [2 - v1; 2 - v2; 2 - v3];
h = [v1^2 + v2^2 - 1; v2^2 + v3^2 - 1; v2 - 0.5];
kappa = 2;

params.cliques = [];

[result, res, coeff_info, aux_info] = CSTSS_mex(f, g, h, kappa, v, params);

% extract solution
blk = cell(size(aux_info.clique_size, 1), 2);
for i = 1: size(aux_info.clique_size, 1)
    blk{i, 1} = 's';
    blk{i, 2} = aux_info.clique_size(i);
end
[Xopt, yopt, Sopt, obj] = recover_mosek_sol_blk(res, blk);
if relax_mode == "MOMENT"
    Xs = Xopt;
else
    Xs = Sopt;
    for i = 1: length(Xs)
        Xs{i} = -Xs{i};
    end
end

% get mom_mat_rpt: representation of vec (mat = vec * vec')
ts_info = aux_info.ts_info;
mom_mat_num = 0;
for i = 1: length(aux_info.cliques)
    mom_mat_num = mom_mat_num + length(ts_info{i});
end
mom_mat_rpt = cell(mom_mat_num, 1);
idx = 0;
for i = 1: length(aux_info.cliques)
    for j = 1: length(ts_info{i})
        idx = idx + 1;
        if if_mex 
            rpt = aux_info.mon_rpt{i}(ts_info{i}{j}, :);
            rpt = [zeros(size(rpt)), rpt];
        end
        mom_mat_rpt{idx} = rpt;
    end
end

% robust extraction only for CS
if ts_mode == "NON"
    [v_opt_robust, output_info_robust] = robust_extract_CS(Xs, mom_mat_rpt, total_var_num, 1e-2);
end
% naive extraction 
[v_opt_naive, output_info_naive] = naive_extract(Xs, mom_mat_rpt, total_var_num);
disp(v_opt_naive);

