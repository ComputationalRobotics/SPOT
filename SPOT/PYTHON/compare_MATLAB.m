clear; close all;

addpath("~/ksc/2024fall/2024-guorui/pathinfo");
ksc;

relax_mode = "SOS"; params.relax_mode = relax_mode;
cs_mode = "NON"; params.cs_mode = cs_mode;
ts_mode = "MD"; params.ts_mode = ts_mode;
ts_mom_mode = "NON"; params.ts_mom_mode = ts_mom_mode;
ts_eq_mode = "NON"; params.ts_eq_mode = ts_eq_mode; 
if_solve = true; params.if_solve = if_solve;

params.cliques = [];

n = 3;
kappa = 2;
x = msspoly('x', n);
x1 = x(1);
x2 = x(2);
x3 = x(3);
f = x1 + x2 + x3;
g = [2 - x1; 2 - x2; 2 - x3];
h = [x1^2 + x2^2 - 1; x2^2 + x3^2 - 1; x2 - 0.5];

[result, res, coeff_info, aux_info] = CSTSS_mex(f, g, h, kappa, x, params);

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

[v_opt, output_info] = naive_extract(Xs, aux_info.mon_rpt, aux_info.ts_info, n);

