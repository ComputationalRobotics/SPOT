function aux_info = PushT_test(relax_mode, cs_mode, ts_mode, init_status)

addpath("../pathinfo/");
% ksc;
jordan;

%% CSTSS parameters
if_mex = true; params.if_mex = if_mex;
kappa = 2; params.kappa = kappa;
params.relax_mode = relax_mode;
params.cs_mode = cs_mode;
params.ts_mode = ts_mode;
ts_mom_mode = "NON"; params.ts_mom_mode = ts_mom_mode;
ts_eq_mode = "USE"; params.ts_eq_mode = ts_eq_mode; 
if_solve = true; params.if_solve = if_solve;

%% hyper-parameters
if_smart_loss = false; params.if_smart_loss = if_smart_loss;
if_sum_ineq = true; params.if_sum_ineq = if_sum_ineq;
if_sum_eq = if_sum_ineq; params.if_sum_eq = if_sum_eq;
% System parameters
N = 30; params.N = N;
dt = 0.1; params.dt = dt;
m = 1.0; params.m = m;
g = 9.8; params.g = g;
mu1 = 0.3; params.mu1 = mu1;
mu2 = 0.2; params.mu2 = mu2; % temporarily unused   
c = 0.6; params.c = c;
% Geometry information
l = 0.05; params.l = l;
dc = 37/14; params.dc = dc; % center of mass
r = l * max(sqrt(dc^2 + 0.5^2), sqrt((4-dc)^2 + 2^2)); params.r = r;
eta = 0.01; params.eta = eta; % ensure different modes have no overlaps
% Variable maximums and minimums
s_max = 0.6; params.s_max = s_max;
px_max = 2 * l; params.px_max = px_max;
py_max = max(4-dc, dc) * l; params.py_max = py_max;
F_max = 1.0; params.F_max = F_max;
fc_min = 0.7; params.fc_min = fc_min;
% Initial states
th_init = init_status(1); params.th_init = th_init;
sx_init = init_status(2); params.sx_init = sx_init;
sy_init = init_status(3); params.sy_init = sy_init;
% vertices of the box 
x1 = -2.0 * l; params.x1 = x1;
x2 = -0.5 * l; params.x2 = x2;
x3 = 0.5 * l; params.x3 = x3;
x4 = 2.0 * l; params.x4 = x4;
y1 = -dc * l; params.y1 = y1;
y2 = (3.0-dc) * l; params.y2 = y2;
y3 = (4.0-dc) * l; params.y3 = y3;
vertices = [
    x4, y3;
    x4, y2;
    x3, y2;
    x3, y1;
    x2, y1;
    x2, y2;
    x1, y2;
    x1, y3;
]; params.vertices = vertices;
% Final states
th_final = 0.0; params.th_final = th_final;
sx_final = 0.0; params.sx_final = sx_final;
sy_final = 0.0; params.sy_final = sy_final;
% Objective coefficients
th_coeff = 1.0; params.th_coeff = th_coeff;
s_coeff = 1.0; params.s_coeff = s_coeff;
F_coeff = 0.5; params.F_coeff = F_coeff;
vertex_coeff = 1.0; params.vertex_coeff = vertex_coeff;
continuous_coeff = 0.2; params.continuous_coeff = continuous_coeff;
final_penalty = 10.0; params.final_penalty = final_penalty;

% total variable number, id, variable-string mapping
total_var_num = 18*N+4; params.total_var_num = total_var_num;
id = @(prefix, k) get_id(prefix, k, N); params.id = id;
var_mapping = get_var_mapping(params); params.var_mapping = var_mapping;

% get remap info
ids_remap = get_remapped_ids(params);
params.ids_remap = ids_remap;

%% file management
current_time = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
current_time = string(current_time);
prefix = "PushT_MATLAB/" + current_time + "/";
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

% Initial state constraints
[eqs, eq_mask] = get_eq_init(...
    v(id("rc", 0)), v(id("rs", 0)), v(id("sx", 0)) * s_max, v(id("sy", 0)) * s_max, ...
    th_init, sx_init, sy_init, ...
    dt, m, g, mu1, mu2, c, l, dc, r, fc_min, eta ...
);

% Append to the equality constraint system
eq_sys = [eq_sys; eqs]; eq_mask_sys = [eq_mask_sys; eq_mask];

% Dynamics equality constraints
for k = 1:N
    % Local equality constraints
    [eqs, eq_mask] = get_eq_dyn(...
        v(id("rc", k-1)), v(id("rc", k)), ...
        v(id("rs", k-1)), v(id("rs", k)), ...
        v(id("sx", k-1)) * s_max, v(id("sx", k)) * s_max, ...
        v(id("sy", k-1)) * s_max, v(id("sy", k)) * s_max, ...
        v(id("fc", k)), v(id("fs", k)), ...
        v(id("px", k)) * px_max, v(id("py", k)) * py_max, ...
        v(id("Fx", k)) * F_max, v(id("Fy", k)) * F_max, ...
        v(id("lam1", k)), v(id("lam2", k)), v(id("lam3", k)), v(id("lam4", k)), ...
        v(id("lam5", k)), v(id("lam6", k)), v(id("lam7", k)), v(id("lam8", k)), ...
        dt, m, g, mu1, mu2, c, l, dc, r, fc_min, eta, ...
        if_sum_eq ...
    );
    eq_sys = [eq_sys; eqs]; eq_mask_sys = [eq_mask_sys; eq_mask];
end

% Dynamics inequality constraints
for k = 1:N
    % Local inequality constraints
    ineqs = get_ineq_dyn(...
        v(id("rc", k-1)), v(id("rc", k)), ...
        v(id("rs", k-1)), v(id("rs", k)), ...
        v(id("sx", k-1)) * s_max, v(id("sx", k)) * s_max, ...
        v(id("sy", k-1)) * s_max, v(id("sy", k)) * s_max, ...
        v(id("fc", k)), v(id("fs", k)), ...
        v(id("px", k)) * px_max, v(id("py", k)) * py_max, ...
        v(id("Fx", k)) * F_max, v(id("Fy", k)) * F_max, ...
        v(id("lam1", k)), v(id("lam2", k)), v(id("lam3", k)), v(id("lam4", k)), ...
        v(id("lam5", k)), v(id("lam6", k)), v(id("lam7", k)), v(id("lam8", k)), ...
        dt, m, g, mu1, mu2, c, l, dc, r, fc_min, eta, ...
        if_sum_ineq ...
    );
    ineq_sys = [ineq_sys; ineqs];

    % Add bounds on variables
    if k == 1
        ineq_sys = [ineq_sys; ...
            1 - v(id("sx", k-1))^2; ...
            1 - v(id("sy", k-1))^2];
    end
    ineq_sys = [ineq_sys; ...
        1 - v(id("sx", k))^2; ...
        1 - v(id("sy", k))^2; ...
        1 - v(id("px", k))^2; ...
        1 - v(id("py", k))^2; ...
        1 - v(id("Fx", k))^2; ...
        1 - v(id("Fy", k))^2];
end

% Objective
sx_final_scaled = sx_final / s_max;
sy_final_scaled = sy_final / s_max;
rc_final = cos(th_final);
rs_final = sin(th_final);
obj_list = [];

for k = 1:N
    if k < N
        if if_smart_loss
            % Append elements to obj_list for the "smart loss" case
            obj_list = [obj_list; ...
                vertex_coeff * vertex_loss(vertices, v(id('rc', k)), v(id('rs', k)), v(id('sx', k)) * s_max, v(id('sy', k)) * s_max); 
                F_coeff * (v(id('Fx', k))^2 + v(id('Fy', k))^2);
                continuous_coeff * (v(id('px', k+1)) - v(id('px', k)))^2; ...
                continuous_coeff * (v(id('py', k+1)) - v(id('py', k)))^2; ...
                ];
        else
            % Append elements to obj_list for the non-smart loss case
            obj_list = [obj_list; ...
                th_coeff * (v(id('rc', k)) - rc_final)^2; ...
                th_coeff * (v(id('rs', k)) - rs_final)^2; ...
                s_coeff * (v(id('sx', k)) - sx_final_scaled)^2; ...
                s_coeff * (v(id('sy', k)) - sy_final_scaled)^2; ...
                continuous_coeff * (v(id('px', k+1)) - v(id('px', k)))^2; ...
                continuous_coeff * (v(id('py', k+1)) - v(id('py', k)))^2; ...
                F_coeff * (v(id('Fx', k))^2 + v(id('Fy', k))^2)];
        end
    else
        if if_smart_loss
            % Append elements to obj_list for the "smart loss" case at the final step
            obj_list = [obj_list; ...
                final_penalty * vertex_coeff * vertex_loss(vertices, v(id('rc', k)), v(id('rs', k)), v(id('sx', k)) * s_max, v(id('sy', k)) * s_max); ...
                F_coeff * (v(id('Fx', k))^2 + v(id('Fy', k))^2)];
        else
            % Append elements to obj_list for the non-smart loss case at the final step
            obj_list = [obj_list; ...
                final_penalty * (v(id('rc', k)) - rc_final)^2; ...
                final_penalty * (v(id('rs', k)) - rs_final)^2; ...
                final_penalty * (v(id('sx', k)) - sx_final_scaled)^2; ...
                final_penalty * (v(id('sy', k)) - sy_final_scaled)^2; ...
                F_coeff * (v(id('Fx', k))^2 + v(id('Fy', k))^2)];
        end
    end
end

obj_sys = sum(obj_list);

% Scaling
[eq_sys, ~] = msspoly_clean(eq_sys, v, 1e-14, true);
[ineq_sys, ~] = msspoly_clean(ineq_sys, v, 1e-14, true);
[obj_sys, obj_scale_factor] = msspoly_clean(obj_sys, v, 1e-14, true);

% Initialize cliques
cliques = cell(N, 1);
for k = 1:N
    if k > 1 && abs(continuous_coeff) > 1e-6
        clique = [
            id("rc", k-1); id("rc", k);
            id("rs", k-1); id("rs", k);
            id("sx", k-1); id("sx", k); 
            id("sy", k-1); id("sy", k);
            id("fc", k); id("fs", k);
            id("px", k-1); id("px", k); 
            id("py", k-1); id("py", k);
            id("Fx", k); id("Fy", k);
            id("lam1", k); id("lam2", k); id("lam3", k); id("lam4", k);
            id("lam5", k); id("lam6", k); id("lam7", k); id("lam8", k);
        ];
    else
        clique = [
            id("rc", k-1); id("rc", k);
            id("rs", k-1); id("rs", k);
            id("sx", k-1); id("sx", k); 
            id("sy", k-1); id("sy", k);
            id("fc", k); id("fs", k);
            id("px", k); id("py", k);
            id("Fx", k); id("Fy", k);
            id("lam1", k); id("lam2", k); id("lam3", k); id("lam4", k);
            id("lam5", k); id("lam6", k); id("lam7", k); id("lam8", k);
        ];
    end
    cliques{k} = clique;
end
params.cliques = cliques;

% clean polynomials
[eq_sys, ~] = msspoly_clean(eq_sys, v, 1e-14, true);
[ineq_sys, ~] = msspoly_clean(ineq_sys, v, 1e-14, true);
[obj_sys, obj_scale_factor] = msspoly_clean(obj_sys, v, 1e-14, true);

% Output variables
f = obj_sys;
g = ineq_sys;
h = eq_sys;

disp('Construction Finish!');
%% run CSTSS
% Output variables
f = obj_sys;
g = ineq_sys;
h = eq_sys;
if if_mex 
    tic;
    [result, res, coeff_info, aux_info] = CSTSS_mex(f, g, h, kappa, v, params);
    toc;
    aux_info.result = result;
end
aux_info.result = result;
params.aux_info = aux_info;

fid = fopen('matlab_solution.txt', 'a');
fprintf(fid, 'PushT  CS=%s, TS=%s, result=%.5f, operation=%.5f, mosek=%.5f, prefix=%s\n\n', cs_mode, ts_mode, result, aux_info.mosek_time, aux_info.time, "./data/" + prefix + "data_ordered_YULIN/data.mat");
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

end


%% helper functions
% re-map variable ids to recover chain-like sparsity pattern
function ids_remap = get_remapped_ids(params)
    N = params.N; total_var_num = params.total_var_num; id = params.id;
    ids_remap = zeros(total_var_num, 1);
    idx = 1;
    for k = 0: N
        ids_remap(id("rc", k)) = idx; idx = idx + 1;
        ids_remap(id("rs", k)) = idx; idx = idx + 1;
        ids_remap(id("sx", k)) = idx; idx = idx + 1;
        ids_remap(id("sy", k)) = idx; idx = idx + 1;
        if k > 0
            ids_remap(id("fc", k)) = idx; idx = idx + 1;
            ids_remap(id("fs", k)) = idx; idx = idx + 1;
            ids_remap(id("px", k)) = idx; idx = idx + 1;
            ids_remap(id("py", k)) = idx; idx = idx + 1;
            ids_remap(id("Fx", k)) = idx; idx = idx + 1;
            ids_remap(id("Fy", k)) = idx; idx = idx + 1;
            ids_remap(id("lam1", k)) = idx; idx = idx + 1;
            ids_remap(id("lam2", k)) = idx; idx = idx + 1;
            ids_remap(id("lam3", k)) = idx; idx = idx + 1;
            ids_remap(id("lam4", k)) = idx; idx = idx + 1;
            ids_remap(id("lam5", k)) = idx; idx = idx + 1;
            ids_remap(id("lam6", k)) = idx; idx = idx + 1;
            ids_remap(id("lam7", k)) = idx; idx = idx + 1;
            ids_remap(id("lam8", k)) = idx; idx = idx + 1;
        end
    end
end

function id = get_id(prefix, k, N)
    % N: clique number
    % k: variable position (different variable has different meanings)
    % rc: 0: N --> id: 1: N+1
    % rs: 0: N --> id: (N+1) + 1: N+1
    % sx: 0: N --> id: (2*N+2) + 1: N+1
    % sy: 0: N --> id: (3*N+3) + 1: N+1
    % fc: 1: N --> id: (4*N+4) + 1: N 
    % fs: 1: N --> id: (5*N+4) + 1: N 
    % px: 1: N --> id: (6*N+4) + 1: N 
    % py: 1: N --> id: (7*N+4) + 1: N 
    % Fx: 1: N --> id: (8*N+4) + 1: N 
    % Fy: 1: N --> id: (9*N+4) + 1: N 
    % lam1: 1: N --> id: (10*N+4) + 1: N 
    % lam2: 1: N --> id: (11*N+4) + 1: N 
    % lam3: 1: N --> id: (12*N+4) + 1: N 
    % lam4: 1: N --> id: (13*N+4) + 1: N 
    % lam5: 1: N --> id: (14*N+4) + 1: N 
    % lam6: 1: N --> id: (15*N+4) + 1: N 
    % lam7: 1: N --> id: (16*N+4) + 1: N 
    % lam8: 1: N --> id: (17*N+4) + 1: N 
    if strcmp(prefix, "rc")
        id = k + 1;
    elseif strcmp(prefix, "rs")
        id = N+1 + k+1;
    elseif strcmp(prefix, "sx")
        id = 2*N+2 + k+1;
    elseif strcmp(prefix, "sy")
        id = 3*N+3 + k+1;
    elseif strcmp(prefix, "fc")
        id = 4*N+4 + k;
    elseif strcmp(prefix, "fs")
        id = 5*N+4 + k;
    elseif strcmp(prefix, "px")
        id = 6*N+4 + k;
    elseif strcmp(prefix, "py")
        id = 7*N+4 + k;
    elseif strcmp(prefix, "Fx")
        id = 8*N+4 + k;
    elseif strcmp(prefix, "Fy")
        id = 9*N+4 + k;
    elseif strcmp(prefix, "lam1")
        id = 10*N+4 + k;
    elseif strcmp(prefix, "lam2")
        id = 11*N+4 + k;
    elseif strcmp(prefix, "lam3")
        id = 12*N+4 + k;
    elseif strcmp(prefix, "lam4")
        id = 13*N+4 + k;
    elseif strcmp(prefix, "lam5")
        id = 14*N+4 + k;
    elseif strcmp(prefix, "lam6")
        id = 15*N+4 + k;
    elseif strcmp(prefix, "lam7")
        id = 16*N+4 + k;
    elseif strcmp(prefix, "lam8")
        id = 17*N+4 + k;
    end
end

function var_mapping = get_var_mapping(params)
    id = params.id; N = params.N;
    var_mapping = strings(id("lam8", N), 1);
    for k = 0: N
        var_mapping(id("rc", k)) = sprintf("r_{c, %d}", k);
        var_mapping(id("rs", k)) = sprintf("r_{s, %d}", k);
        var_mapping(id("sx", k)) = sprintf("s_{x, %d}", k);
        var_mapping(id("sy", k)) = sprintf("s_{y, %d}", k);
    end
    for k = 1: N
        var_mapping(id("fc", k)) = sprintf("f_{c, %d}", k);
        var_mapping(id("fs", k)) = sprintf("f_{s, %d}", k);
        var_mapping(id("px", k)) = sprintf("p_{x, %d}", k);
        var_mapping(id("py", k)) = sprintf("p_{y, %d}", k);
        var_mapping(id("Fx", k)) = sprintf("F_{x, %d}", k);
        var_mapping(id("Fy", k)) = sprintf("F_{y, %d}", k);
        var_mapping(id("lam1", k)) = "\" + sprintf("\\lambda_{1, %d}", k);
        var_mapping(id("lam2", k)) = "\" + sprintf("\\lambda_{2, %d}", k);
        var_mapping(id("lam3", k)) = "\" + sprintf("\\lambda_{3, %d}", k);
        var_mapping(id("lam4", k)) = "\" + sprintf("\\lambda_{4, %d}", k);
        var_mapping(id("lam5", k)) = "\" + sprintf("\\lambda_{5, %d}", k);
        var_mapping(id("lam6", k)) = "\" + sprintf("\\lambda_{6, %d}", k);
        var_mapping(id("lam7", k)) = "\" + sprintf("\\lambda_{7, %d}", k);
        var_mapping(id("lam8", k)) = "\" + sprintf("\\lambda_{8, %d}", k);
    end
end

function [eqs, eq_mask] = get_eq_init(rc_0, rs_0, sx_0, sy_0, ...
    th_init, sx_init, sy_init, ...
    dt, m, g, mu1, mu2, c, l, dc, r, fc_min, eta)
    % Compute the initial state equality constraints
    eqs = [
        rc_0 - cos(th_init);
        rs_0 - sin(th_init);
        sx_0 - sx_init;
        sy_0 - sy_init;
        rc_0^2 + rs_0^2 - 1;
    ];
    eq_mask = [1; 1; 1; 1; 0];
end

function [eq, eq_mask] = get_eq_dyn(rc_km1, rc_k, rs_km1, rs_k, sx_km1, sx_k, sy_km1, sy_k, ...
    fc_km1, fs_km1, px_km1, py_km1, Fx_km1, Fy_km1, ...
    lam1_km1, lam2_km1, lam3_km1, lam4_km1, lam5_km1, lam6_km1, lam7_km1, lam8_km1, ...
    dt, m, g, mu1, mu2, c, l, dc, r, fc_min, eta, ...
    if_sum_eq)
    eq_mask = [];

    % Dynamic equations
    x1 = -2.0 * l;
    x2 = -0.5 * l;
    x3 = 0.5 * l;
    x4 = 2.0 * l;
    y1 = -dc * l;
    y2 = (3.0-dc) * l;
    y3 = (4.0-dc) * l;

    eq1 = sx_k - sx_km1 - dt * (rc_km1 * Fx_km1 - rs_km1 * Fy_km1); eq_mask = [eq_mask; 1];
    eq2 = sy_k - sy_km1 - dt * (rs_km1 * Fx_km1 + rc_km1 * Fy_km1); eq_mask = [eq_mask; 1];
    eq3 = fs_km1 - dt / (c*r) * (-py_km1 * Fx_km1 + px_km1 * Fy_km1); eq_mask = [eq_mask; 1];
    eq4 = rc_k - (rc_km1 * fc_km1 - rs_km1 * fs_km1); eq_mask = [eq_mask; 1];
    eq5 = rs_k - (rs_km1 * fc_km1 + rc_km1 * fs_km1); eq_mask = [eq_mask; 1];
    eq6 = rc_k^2 + rs_k^2 - 1; eq_mask = [eq_mask; 0];
    eq7 = fc_km1^2 + fs_km1^2 - 1; eq_mask = [eq_mask; 1];

    % Contact modes
    eq_con_mode = [
        lam1_km1 * (1.0 - lam1_km1);
        lam2_km1 * (1.0 - lam2_km1);
        lam3_km1 * (1.0 - lam3_km1);
        lam4_km1 * (1.0 - lam4_km1);
        lam5_km1 * (1.0 - lam5_km1);
        lam6_km1 * (1.0 - lam6_km1);
        lam7_km1 * (1.0 - lam7_km1);
        lam8_km1 * (1.0 - lam8_km1);
        lam1_km1^2 + lam2_km1^2 + lam3_km1^2 + lam4_km1^2 + lam5_km1^2 + lam6_km1^2 + lam7_km1^2 + lam8_km1^2 - 1.0;
    ];
    
    if if_sum_eq
        eq_con_mode = [
            eq_con_mode;
            sum([ 
                lam1_km1 * (py_km1 - y3);
                lam2_km1 * (px_km1 - x4);
                lam3_km1 * (py_km1 - y2);
                lam4_km1 * (px_km1 - x3);
                lam5_km1 * (py_km1 - y1);
                lam6_km1 * (px_km1 - x2);
                lam7_km1 * (py_km1 - y2);
                lam8_km1 * (px_km1 - x1);
            ]);
        ];
        eq_con_mode = [
            eq_con_mode;
            sum([ 
                lam1_km1 * Fx_km1;
                lam2_km1 * Fy_km1;
                lam3_km1 * Fx_km1;
                lam4_km1 * Fy_km1;
                lam5_km1 * Fx_km1;
                lam6_km1 * Fy_km1;
                lam7_km1 * Fx_km1;
                lam8_km1 * Fy_km1;
            ]);
        ];
    else 
        eq_con_mode = [
            eq_con_mode;
            lam1_km1 * (py_km1 - y3);
            lam2_km1 * (px_km1 - x4);
            lam3_km1 * (py_km1 - y2);
            lam4_km1 * (px_km1 - x3);
            lam5_km1 * (py_km1 - y1);
            lam6_km1 * (px_km1 - x2);
            lam7_km1 * (py_km1 - y2);
            lam8_km1 * (px_km1 - x1);
        ];
        eq_con_mode = [
            eq_con_mode;
            lam1_km1 * Fx_km1;
            lam2_km1 * Fy_km1;
            lam3_km1 * Fx_km1;
            lam4_km1 * Fy_km1;
            lam5_km1 * Fx_km1;
            lam6_km1 * Fy_km1;
            lam7_km1 * Fx_km1;
            lam8_km1 * Fy_km1;
        ];
    end
    eq_mask = [eq_mask; ones(size(eq_con_mode))];

    eq = [eq1; eq2; eq3; eq4; eq5; eq6; eq7; eq_con_mode];
end
function ineq = get_ineq_dyn(rc_km1, rc_k, rs_km1, rs_k, sx_km1, sx_k, sy_km1, sy_k, ...
    fc_km1, fs_km1, px_km1, py_km1, Fx_km1, Fy_km1, ...
    lam1_km1, lam2_km1, lam3_km1, lam4_km1, lam5_km1, lam6_km1, lam7_km1, lam8_km1, ...
    dt, m, g, mu1, mu2, c, l, dc, r, fc_min, eta, ...
    if_sum_ineq)

    % Parameters
    x1 = -2.0 * l;
    x2 = -0.5 * l;
    x3 = 0.5 * l;
    x4 = 2.0 * l;
    y1 = -dc * l;
    y2 = (3.0-dc) * l;
    y3 = (4.0-dc) * l;

    % Inequality for minimum contact force
    ineq1 = fc_km1 - fc_min;

    % Contact modes
    if if_sum_ineq
        ineq_con_mode = sum([
            lam1_km1 * (px_km1 - enlarge(x1, eta));
            lam2_km1 * (py_km1 - enlarge(y2, eta));
            lam3_km1 * (px_km1 - enlarge(x3, eta));
            lam4_km1 * (py_km1 - enlarge(y1, eta));
            lam5_km1 * (px_km1 - enlarge(x2, eta));
            lam6_km1 * (py_km1 - enlarge(y1, eta));
            lam7_km1 * (px_km1 - enlarge(x1, eta));
            lam8_km1 * (py_km1 - enlarge(y2, eta));
        ]);
    else
        ineq_con_mode = [
            lam1_km1 * (px_km1 - enlarge(x1, eta));
            lam2_km1 * (py_km1 - enlarge(y2, eta));
            lam3_km1 * (px_km1 - enlarge(x3, eta));
            lam4_km1 * (py_km1 - enlarge(y1, eta));
            lam5_km1 * (px_km1 - enlarge(x2, eta));
            lam6_km1 * (py_km1 - enlarge(y1, eta));
            lam7_km1 * (px_km1 - enlarge(x1, eta));
            lam8_km1 * (py_km1 - enlarge(y2, eta));
        ];
    end

    % Append additional inequalities
    if if_sum_ineq
        ineq_con_mode = [
            ineq_con_mode;
            sum([ 
                lam1_km1 * (reduce(x4, eta) - px_km1);
                lam2_km1 * (reduce(y3, eta) - py_km1);
                lam3_km1 * (reduce(x4, eta) - px_km1);
                lam4_km1 * (reduce(y2, eta) - py_km1);
                lam5_km1 * (reduce(x3, eta) - px_km1);
                lam6_km1 * (reduce(y2, eta) - py_km1);
                lam7_km1 * (reduce(x2, eta) - px_km1);
                lam8_km1 * (reduce(y3, eta) - py_km1);
            ]);
        ];
    else
        ineq_con_mode = [
            ineq_con_mode;
            lam1_km1 * (reduce(x4, eta) - px_km1);
            lam2_km1 * (reduce(y3, eta) - py_km1);
            lam3_km1 * (reduce(x4, eta) - px_km1);
            lam4_km1 * (reduce(y2, eta) - py_km1);
            lam5_km1 * (reduce(x3, eta) - px_km1);
            lam6_km1 * (reduce(y2, eta) - py_km1);
            lam7_km1 * (reduce(x2, eta) - px_km1);
            lam8_km1 * (reduce(y3, eta) - py_km1);
        ];
    end

    % Append force-related inequalities
    if if_sum_ineq
        ineq_con_mode = [
            ineq_con_mode;
            sum([ 
                lam1_km1 * (-Fy_km1);
                lam2_km1 * (-Fx_km1);
                lam3_km1 * (Fy_km1);
                lam4_km1 * (-Fx_km1);
                lam5_km1 * (Fy_km1);
                lam6_km1 * (Fx_km1);
                lam7_km1 * (Fy_km1);
                lam8_km1 * (Fx_km1);
            ]);
        ];
    else
        ineq_con_mode = [
            ineq_con_mode;
            lam1_km1 * (-Fy_km1);
            lam2_km1 * (-Fx_km1);
            lam3_km1 * (Fy_km1);
            lam4_km1 * (-Fx_km1);
            lam5_km1 * (Fy_km1);
            lam6_km1 * (Fx_km1);
            lam7_km1 * (Fy_km1);
            lam8_km1 * (Fx_km1);
        ];
    end

    % Combine all inequalities
    ineq = [ineq1; ineq_con_mode];
end


function x = enlarge(x, eta)
    x = x + eta;
end

function x = reduce(x, eta)
    x = x - eta;
end

function total_loss = vertex_loss(vertices, rc, rs, sx, sy)
    % Initialize the loss array
    loss = [];
    
    % Loop through each vertex
    for i = 1:size(vertices, 1)
        vx = vertices(i, 1);
        vy = vertices(i, 2);
        
        % Calculate the loss components
        l1 = (sx + vx * rc - vy * rs - vx)^2;
        l2 = (sy + vx * rs + vy * rc - vy)^2;
        
        % Append the total loss for the current vertex
        loss = [loss, l1 + l2]; %#ok<AGROW>
    end
    
    % Return the sum of all losses
    total_loss = sum(loss);
end











