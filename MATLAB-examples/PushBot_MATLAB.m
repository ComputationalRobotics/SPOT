clear; close all;

addpath("../pathinfo/");
my_path;

%% CSTSS parameters
if_mex = true; params.if_mex = if_mex;
kappa = 2; params.kappa = kappa;
relax_mode = "SOS"; params.relax_mode = relax_mode;
cs_mode = "MF"; params.cs_mode = cs_mode;
ts_mode = "NON"; params.ts_mode = ts_mode;
ts_mom_mode = "NON"; params.ts_mom_mode = ts_mom_mode;
ts_eq_mode = "NON"; params.ts_eq_mode = ts_eq_mode; 
if_solve = true; params.if_solve = if_solve;

%% hyper-parameters
N = 30; params.N = N;
dt = 0.05; params.dt = dt;
mc = 1.0; params.mc = mc;
mp = 0.1; params.mp = mp;
l  = 0.8; params.l = l;
g  = 9.8; params.g = g;
d1 = 1.0; params.d1 = d1;
d2 = 1.0; params.d2 = d2;
k1 = 200.0; params.k1 = k1;
k2 = 200.0; params.k2 = k2;
% variable maximums and minimums
a_max = 1.0; params.a_max = a_max;
u_max = 20.0; params.u_max = u_max;
lam1_max = 100.0; params.lam1_max = lam1_max;
lam2_max = 100.0; params.lam2_max = lam2_max;
fc_min = 0.7; params.fc_min = fc_min;
% init states
a_init = 0.9; params.a_init = a_init;
v_init = 0.0; params.v_init = v_init;
th_init = 0.0; params.th_init = th_init;
dth_init = 0.0; params.dth_init = dth_init;
% final states
a_final = 0.0; params.a_final = a_final;
v_final = 0.0; params.v_final = v_final;
th_final = pi; params.th_final = th_final;
dth_final = 0.0; params.dth_final = dth_final;
% objective coefficients
a_coeff = 1.0; params.a_coeff = a_coeff;
th_coeff = 1.0; params.th_coeff = th_coeff;
dth_coeff = 1.0; params.dth_coeff = dth_coeff;
u_coeff = 1.0; params.u_coeff = u_coeff;
final_penalty = 10.0; params.final_penalty = final_penalty;

% get id and var_mapping
total_var_num = 8*N+7; params.total_var_num = total_var_num;
v = msspoly('v', total_var_num);
id = @(prefix, k) get_id(prefix, k, N); params.id = id;
var_mapping = get_var_mapping(id, N); params.var_mapping = var_mapping;

% get remap info
ids_remap = get_remapped_ids(params);
params.ids_remap = ids_remap;

%% file management
current_time = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
current_time = string(current_time);
prefix = "PushBot_MATLAB/" + current_time + "/";
if ~exist("./data/" + prefix)
    mkdir("./data/" + prefix);
end
if ~exist("./markdown/" + prefix)
    mkdir("./markdown/" + prefix);
end
if ~exist("./figs/" + prefix)
    mkdir("./figs/" + prefix);
end
if ~exist("./logs/" + prefix)
    mkdir("./logs/" + prefix);
end
% start logging
diary("./logs/" + prefix + "log.txt");
diary on;
fprintf("params: \n");
disp(params);

%% formulate POP
% Define polynomials
v = msspoly('v', total_var_num);
eq_sys = []; % Polynomial equality constraints
ineq_sys = []; % Polynomial inequality constraints
eq_mask_sys = [];
obj_sys = 0; % Polynomial objective

% initial state constraints
[eqs, eq_mask] = get_eq_init(v(id("a", 0)) * a_max, v(id("a", 1)) * a_max,...
    v(id("rc", 0)), v(id("rs", 0)), v(id("rs", 1)), v(id("fc", 0)), v(id("fs", 0)),...
    a_init, v_init, th_init, dth_init,...
    dt, mc, mp, l, g, d1, d2, k1, k2, fc_min);
eq_sys = [eq_sys; eqs]; eq_mask_sys = [eq_mask_sys; eq_mask];

% dynamics equality constraints
for k = 1: N 
    [eqs, eq_mask] = get_eq_dyn(v(id("a", k-1)) * a_max, v(id("a", k)) * a_max, v(id("a", k+1)) * a_max,...
        v(id("rc", k-1)), v(id("rc", k)),... 
        v(id("rs", k-1)), v(id("rs", k)), v(id("rs", k+1)),...  
        v(id("fc", k-1)), v(id("fc", k)),...
        v(id("fs", k-1)), v(id("fs", k)),...
        v(id("u", k)) * u_max, v(id("lam1", k)) * lam1_max, v(id("lam2", k)) * lam2_max,...
        dt, mc, mp, l, g, d1, d2, k1, k2, fc_min);
    eq_sys = [eq_sys; eqs]; eq_mask_sys = [eq_mask_sys; eq_mask];
end

% dynamics inequality constraints
for k = 1: N
    ineqs = get_ineq_dyn(v(id("a", k-1)) * a_max, v(id("a", k)) * a_max, v(id("a", k+1)) * a_max,...
        v(id("rc", k-1)), v(id("rc", k)),...
        v(id("rs", k-1)), v(id("rs", k)), v(id("rs", k+1)),...  
        v(id("fc", k-1)), v(id("fc", k)),...
        v(id("fs", k-1)), v(id("fs", k)),...
        v(id("u", k)) * u_max, v(id("lam1", k)) * lam1_max, v(id("lam2", k)) * lam2_max,...
        dt, mc, mp, l, g, d1, d2, k1, k2, fc_min);
    ineq_sys = [ineq_sys; ineqs];

    % add bounds on variables
    if k == 1
        ineq_sys = [
            ineq_sys;
            [
                1 - v(id("a", k-1))^2;
                1 - v(id("a", k))^2;
                1 - v(id("rc", k-1))^2 - v(id("rs", k-1))^2;
                1 - v(id("fc", k-1))^2 - v(id("fs", k-1))^2
            ];
        ];
    end
    ineq_sys = [
        ineq_sys;
        [
            1 - v(id("a", k+1))^2;
            1 - v(id("rs", k+1))^2;
            1 - v(id("u", k))^2;
            1 - v(id("lam1", k))^2;
            1 - v(id("lam2", k))^2;
        ];
    ];
end

% objective 
a_final_scaled = a_final / a_max;
rc_final = cos(th_final);
rs_final = sin(th_final);
fc_final = cos(dt * dth_final);
fs_final = sin(dt * dth_final);
obj_list = [];
for k = 1: N 
    if k < N 
        obj_list = [
            obj_list;
            [
                a_coeff * (v(id("a", k+1)) - a_final_scaled)^2;
                th_coeff * (v(id("rc", k)) - rc_final)^2;
                th_coeff * (v(id("rs", k+1)) - rs_final)^2;
                dth_coeff * (v(id("fc", k)) - fc_final)^2;
                dth_coeff * (v(id("fs", k)) - fs_final)^2;
                u_coeff * v(id("u", k))^2;
            ]
        ];
    else
        obj_list = [
            obj_list;
            [
                final_penalty * a_coeff * (v(id("a", k+1)) - a_final_scaled)^2;
                final_penalty * th_coeff * (v(id("rc", k)) - rc_final)^2;
                final_penalty * th_coeff * (v(id("rs", k+1)) - rs_final)^2;
                final_penalty * dth_coeff * (v(id("fc", k)) - fc_final)^2;
                final_penalty * dth_coeff * (v(id("fs", k)) - fs_final)^2;
                u_coeff * v(id("u", k))^2;
            ]
        ];
    end
end
obj_sys = sum(obj_list);

% clean polynomials
[eq_sys, ~] = msspoly_clean(eq_sys, v, 1e-14, true);
[ineq_sys, ~] = msspoly_clean(ineq_sys, v, 1e-14, true);
[obj_sys, obj_scale_factor] = msspoly_clean(obj_sys, v, 1e-14, true);
disp('Construction Finish!');

% Cliques
cliques = cell(N, 1);
for k = 1:N
    clique = [
        id("a", k-1); id("a", k); id("a", k+1);
        id("rc", k-1); id("rc", k);
        id("rs", k-1); id("rs", k); id("rs", k+1);
        id("fc", k-1); id("fc", k);
        id("fs", k-1); id("fs", k);
        id("u", k);
        id("lam1", k); id("lam2", k);
    ];
    cliques{k} = clique;
end
params.cliques = cliques;

% Output variables
f = obj_sys;
g = ineq_sys;
h = eq_sys;
if if_mex 
    tic;
    [result, res, coeff_info, aux_info] = CSTSS_mex(f, g, h, kappa, v, params);
    toc;
end

fid = fopen('matlab_solution.txt', 'a');
fprintf(fid, 'PushBot  N=%d, Relax=%s, TS=%s, CS=%s, result=%.20f, operation=%.5f\n\n', N, relax_mode, ts_mode, cs_mode, result, aux_info.time);
fclose(fid);

%% remap clique ids
cliques_remapped = cell(size(aux_info.cliques));
aver_remapped = zeros(size(aux_info.cliques));
for i = 1: length(aux_info.cliques)
    cliques_remapped{i} = params.ids_remap(aux_info.cliques{i});
    aver_remapped(i) = mean(cliques_remapped{i});
end
[~, cliques_rank] = sort(aver_remapped);
params.cliques_rank = cliques_rank;

%% markdown debug
fid = fopen("./markdown/" + prefix + "opt_problem.md", "w");
% print out clique variables
clique_pop = msspoly();
for i = 1: length(aux_info.cliques)
    ii = cliques_rank(i);             
    v_clique = v(sort(aux_info.cliques{ii}));
    f = 0;
    for j = 1: length(v_clique)
        f = f + v_clique(j);
    end
    clique_pop = [clique_pop; f];
end
fprintf(fid, "equality constraints: \n");
msspoly_visualize(eq_sys, v, var_mapping, fid);
fprintf(fid, "inequality constraints: \n");
msspoly_visualize(ineq_sys, v, var_mapping, fid);
fprintf(fid, "objective: \n");
msspoly_visualize(obj_sys, v, var_mapping, fid);
fprintf(fid, "cliques: \n");
msspoly_visualize(clique_pop, v, var_mapping, fid);
fclose(fid);



if ~params.if_solve
    params.self_cliques = cliques;
    save("./data/" + prefix + "params.mat", "params");
    return;
end

%% extract solution
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
        if ~if_mex 
            s = length(ts_info{i}{j});
            rpt = find_rpt(idx * ones(1, s), ones(1, s), 1:s, aux_info);
        else
            rpt = aux_info.mon_rpt{i}(ts_info{i}{j}, :);
            rpt = [zeros(size(rpt)), rpt];
        end
        mom_mat_rpt{idx} = rpt;
    end
end

% robust extraction only for CS
if ts_mode == "NON"
    [v_opt_robust, output_info_robust] = robust_extract_CS(Xs, mom_mat_rpt, total_var_num, 1e-2);
    save("./data/" + prefix + "v_opt_robust.mat", "v_opt_robust");
    [v_opt_ordered, output_info_ordered] = ordered_extract_CS(Xs, mom_mat_rpt, total_var_num, 1e-2, cliques_rank);
    save("./data/" + prefix + "v_opt_ordered.mat", "v_opt_ordered");
end
% naive extraction 
[v_opt_naive, output_info_naive] = naive_extract(Xs, mom_mat_rpt, total_var_num);
save("./data/" + prefix + "v_opt_naive.mat", "v_opt_naive");
% save important data
save("./data/" + prefix + "data.mat", "v", "aux_info", "mom_mat_rpt", "mom_mat_num", "total_var_num");
supp_rpt_f = aux_info.supp_rpt_f;
supp_rpt_g = aux_info.supp_rpt_g;
supp_rpt_h = aux_info.supp_rpt_h;
coeff_f = aux_info.coeff_f;
coeff_g = aux_info.coeff_g;
coeff_h = aux_info.coeff_h;
% mask the redundant equality constraints for local solvers
% eq_mask_sys = logical(eq_mask_sys);
% supp_rpt_h = supp_rpt_h(eq_mask_sys);
% coeff_h = coeff_h(eq_mask_sys);
save("./data/" + prefix + "polys.mat", "supp_rpt_f", "supp_rpt_g", "supp_rpt_h", "coeff_f", "coeff_g", "coeff_h");
params.self_cliques = cliques;
save("./data/" + prefix + "params.mat", "params");

%% end logging
diary off;



%% helper functions
% re-map variable ids to recover chain-like sparsity pattern
function ids_remap = get_remapped_ids(params)
    N = params.N; total_var_num = params.total_var_num; id = params.id;
    ids_remap = zeros(total_var_num, 1);
    idx = 1;
    for k = 0: N+1
        ids_remap(id("a", k)) = idx; idx = idx + 1;
        ids_remap(id("rs", k)) = idx; idx = idx + 1;
        if k < N+1
            ids_remap(id("rc", k)) = idx; idx = idx + 1;
            ids_remap(id("fc", k)) = idx; idx = idx + 1;
            ids_remap(id("fs", k)) = idx; idx = idx + 1;
            if k > 0
                ids_remap(id("u", k)) = idx; idx = idx + 1;
                ids_remap(id("lam1", k)) = idx; idx = idx + 1;
                ids_remap(id("lam2", k)) = idx; idx = idx + 1;
            end
        end
    end
end

function id = get_id(prefix, k, N)
    % N: clique number
    % k: variable position (different variable has different meanings)
    % a: 0 : N+1 --> id: 1 : N+2 
    % rc: 0 : N --> id: N+2 + (1 : N+1)
    % rs: 0 : N+1 --> id: 2*N+3 + (1 : N+2)
    % fc: 0 : N --> id: 3*N+5 + (1 : N+1)
    % fs: 0 : N --> id: 4*N+6 + (1 : N+1)
    % u: 1 : N --> id: 5*N+7 + (1 : N)
    % lam1: 1 : N --> id: 6*N+7 + (1 : N)
    % lam2: 1 : N --> id: 7*N+7 + (1 : N)
    if prefix == "a"
        id = k + 1;
    elseif prefix == "rc"
        id = N+2 + k+1;
    elseif prefix == "rs"
        id = 2*N+3 + k+1;
    elseif prefix == "fc"
        id = 3*N+5 + k+1;
    elseif prefix == "fs"
        id = 4*N+6 + k+1;
    elseif prefix == "u"
        id = 5*N+7 + k;
    elseif prefix == "lam1"
        id = 6*N+7 + k;
    elseif prefix == "lam2"
        id = 7*N+7 + k;
    end
end

function var_mapping = get_var_mapping(id, N)
    var_mapping = strings(id("lam2", N), 1);
    for k = 0: N+1
        var_mapping(id("a", k)) = sprintf("a_{%d}", k);
        var_mapping(id("rs", k)) = sprintf("r_{s, %d}", k);
    end
    for k = 0: N
        var_mapping(id("rc", k)) = sprintf("r_{c, %d}", k);
        var_mapping(id("fc", k)) = sprintf("f_{c, %d}", k);
        var_mapping(id("fs", k)) = sprintf("f_{s, %d}", k);
    end
    for k = 1: N
        var_mapping(id("u", k)) = sprintf("u_{%d}", k);
        var_mapping(id("lam1", k)) = "\" + sprintf("\\lambda_{1, %d}", k);
        var_mapping(id("lam2", k)) = "\" + sprintf("\\lambda_{2, %d}", k);
    end
end

function [eq, eq_mask] = get_eq_init(a_0, a_1,...
    rc_0, rs_0, rs_1, fc_0, fs_0,...
    a_init, v_init, th_init, dth_init,... 
    dt, mc, mp, l, g, d1, d2, k1, k2, fc_min)
    eq = [
        a_0 - a_init;
        a_1 - (a_init + dt * v_init);
        rc_0 - cos(th_init);
        rs_0 - sin(th_init);
        fc_0 - cos(dt * dth_init);
        fs_0 - sin(dt * dth_init);
        rs_1 - (sin(th_init) * cos(dt * dth_init) + cos(th_init) * sin(dt * dth_init));
        rc_0^2 + rs_0^2 - 1;
        fc_0^2 + fs_0^2 - 1;
    ];
    eq_mask = ones(size(eq));
    eq_mask(end-1) = 0; eq_mask(end) = 0;
end

function [eq, eq_mask] = get_eq_dyn(a_km1, a_k, a_kp1,...
    rc_km1, rc_k, rs_km1, rs_k, rs_kp1, fc_km1, fc_k, fs_km1, fs_k,... 
    u_k, lam1_k, lam2_k,...
    dt, mc, mp, l, g, d1, d2, k1, k2, fc_min)
    eq1 = (mc + mp) * (a_km1 - 2*a_k + a_kp1) / (dt^2) + mp*l * (rs_km1 - 2*rs_k + rs_kp1) / (dt^2) - (u_k + lam1_k - lam2_k);
    eq2 = l * (fs_k - fs_km1) / (dt^2) + ( (a_km1 - 2*a_k + a_kp1) / (dt^2) + lam2_k - lam1_k ) * rc_k + g * rs_k;
    eq3 = lam1_k * (lam1_k / k1 + d1 + a_k + l * rs_k);
    eq4 = lam2_k * (lam2_k / k2 + d2 - a_k - l * rs_k);
    eq5 = rc_k - (rc_km1 * fc_km1 - rs_km1 * fs_km1);
    eq6 = rs_kp1 - (rs_k * fc_k + rc_k * fs_k);
    eq7 = rc_k^2 + rs_k^2 - 1;
    eq8 = fc_k^2 + fs_k^2 - 1;
    eq = [eq1; eq2; eq3; eq4; eq5; eq6; eq7; eq8];
    eq_mask = [1; 1; 1; 1; 1; 1; 0; 1];
end

function ineq = get_ineq_dyn(a_km1, a_k, a_kp1,... 
    rc_km1, rc_k, rs_km1, rs_k, rs_kp1, fc_km1, fc_k, fs_km1, fs_k,... 
    u_k, lam1_k, lam2_k,...
    dt, mc, mp, l, g, d1, d2, k1, k2, fc_min)
    ineq1 = lam1_k;
    ineq2 = lam1_k / k1 + d1 + a_k + l * rs_k;
    ineq3 = lam2_k;
    ineq4 = lam2_k / k2 + d2 - a_k - l * rs_k;
    ineq5 = fc_k - fc_min;
    ineq = [ineq1; ineq2; ineq3; ineq4; ineq5];
end

function traj = rescale_sol(sol, id, N, ...
    a_max, u_max, lam1_max, lam2_max)
    v_opt = sol;
    a_opt = zeros(1, N+1); % abandon the last a
    rot_opt = zeros(4, N+1); % abandon the last rs
    u_opt = zeros(1, N);
    lam_opt = zeros(2, N);
    
    % Loop for k = 0:N
    for k = 0:N
        a_opt(k+1) = a_max * v_opt(id("a", k));
        rot_opt(1, k+1) = v_opt(id("rc", k));
        rot_opt(2, k+1) = v_opt(id("rs", k));
        rot_opt(3, k+1) = v_opt(id("fc", k));
        rot_opt(4, k+1) = v_opt(id("fs", k));
    end
    
    % Loop for k = 1:N
    for k = 1:N
        u_opt(k) = u_max * v_opt(id("u", k));
        lam_opt(1, k) = lam1_max * v_opt(id("lam1", k));
        lam_opt(2, k) = lam2_max * v_opt(id("lam2", k));
    end

    traj.a = a_opt; 
    traj.rot = rot_opt;
    traj.u = u_opt;
    traj.lam = lam_opt;
end


