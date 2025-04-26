clear; close all; 

%% 
addpath("../pathinfo/");
my_path;

%% CSTSS parameters
if_mex = true; params.if_mex = if_mex;
kappa = 2; params.kappa = kappa;
relax_mode = "SOS"; params.relax_mode = relax_mode;
cs_mode = "SELF"; params.cs_mode = cs_mode;
ts_mode = "NON"; params.ts_mode = ts_mode;
ts_mom_mode = "NON"; params.ts_mom_mode = ts_mom_mode;
ts_eq_mode = "USE"; params.ts_eq_mode = ts_eq_mode; 
if_solve = true; params.if_solve = if_solve;

%% system parameters
N = 30; params.N = N;
params.dt = 0.05;
[var_mapping, var_start_dict, total_var_num] = get_var_mapping(params.N);
params.total_var_num = total_var_num;
v = msspoly('v', params.total_var_num);
params.var_mapping = var_mapping;
params.var_start_dict = var_start_dict;
id = @(prefix, k) get_id(prefix, k, var_start_dict);
params.id = id;
var = @(prefix, k) get_var(v, prefix, k, var_start_dict);
% geometric information
params.l = 0.02;
params.r = 0.01;
params.theta_0 = asin(params.r / (params.r + params.l));
params.Ld = 3*params.l + 4*params.r;
params.Lu = 2*params.l + 3*params.r;
params.H = 0.2;
params.R = 0.08;
% dynamics
params.mu = 0.3;
params.c = 0.6;
% limits
params.fc_min = 0.9;
params.fcrd_min = 0.9;
params.fcru_min = params.fcrd_min;
params.fcld_min = params.fcrd_min;
params.fclu_min = params.fcrd_min;
params.x_max = params.Ld + params.Lu + params.H/2;
params.y_max = params.x_max;
params.xr_max = params.x_max;
params.yr_max = params.y_max;
params.xl_max = params.x_max;
params.yl_max = params.y_max;
params.vxr_max = 1.0;
params.vyr_max = params.vxr_max;
params.vxl_max = params.vxr_max;
params.vyl_max = params.vxr_max;
params.vx_max = params.vxr_max;
params.vy_max = params.vxr_max;
params.lamnr_max = 1.0;
params.lamtr_max = params.lamnr_max;
params.lamnl_max = params.lamnr_max;
params.lamtl_max = params.lamnr_max;
params.dr_max = params.x_max * 2;
params.dl_max = params.dr_max;
params.vrelr_max = 3.0;
params.vrell_max = params.vrelr_max;

% initial states
% circle
params.x_init = 0.0;
params.y_init = params.Ld;
params.th_init = 0.0;
% left finger
params.th_ld_init = 2*pi/3;
params.th_lu_init = pi/3;
% right finger
params.th_rd_init = pi/3;
params.th_ru_init = 2*pi/3;

% loss
params.loss_track_coeff = 1.0;
params.loss_track_th_list = linspace(0.0, 2*pi, params.N);
params.loss_vel_finger_coeff = 1.0;
params.loss_vel_circle_coeff = 1.0;
params.loss_vel_relative_coeff = 0.0;
params.loss_force_coeff = 1.0;
params.loss_pos_finger_coeff = 0.1;

% get remap info
ids_remap = get_remapped_ids(params);
params.ids_remap = ids_remap;


%% file management
current_time = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
current_time = string(current_time);
prefix = "PlanarHand_MATLAB/" + current_time + "/";
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

%% generate POP: constraints
eq_sys = [];
ineq_sys = [];
eq_mask_sys = [];

% get initial constraints
[eqs, ineqs, eq_mask] = get_init_constraints( ...
    var("x", 0) * params.x_max, var("y", 0) * params.y_max, var("rc", 0), var("rs", 0), ...
    var("rcld", 0), var("rsld", 0), var("rclu", 0), var("rslu", 0), ...
    var("rcrd", 0), var("rsrd", 0), var("rcru", 0), var("rsru", 0), ...
    params ...
);
eq_sys = [eq_sys; eqs]; ineq_sys = [ineq_sys; ineqs]; eq_mask_sys = [eq_mask_sys; eq_mask];
% bound constraints
ineq_sys = [ineq_sys; 1.0 - var("x", 0)^2; 1.0 - var("y", 0)^2];

for k = 1: params.N 
    % get two fingers' kinematics
    [eqs, ineqs, eq_mask] = get_right_finger_kinematics(...
        var("rcrd", k-1), var("rcrd", k), var("rsrd", k-1), var("rsrd", k), var("fcrd", k-1), var("fsrd", k-1), ...
        var("rcru", k-1), var("rcru", k), var("rsru", k-1), var("rsru", k), var("fcru", k-1), var("fsru", k-1), ...
        var("xr", k-1) * params.xr_max, var("yr", k-1) * params.yr_max, var("vxr", k-1) * params.vxr_max, var("vyr", k-1) * params.vyr_max, ...
        params ...
    );
    eq_sys = [eq_sys; eqs]; ineq_sys = [ineq_sys; ineqs]; eq_mask_sys = [eq_mask_sys; eq_mask];
    [eqs, ineqs, eq_mask] = get_left_finger_kinematics(...
        var("rcld", k-1), var("rcld", k), var("rsld", k-1), var("rsld", k), var("fcld", k-1), var("fsld", k-1), ...
        var("rclu", k-1), var("rclu", k), var("rslu", k-1), var("rslu", k), var("fclu", k-1), var("fslu", k-1), ...
        var("xl", k-1) * params.xl_max, var("yl", k-1) * params.yl_max, var("vxl", k-1) * params.vxl_max, var("vyl", k-1) * params.vyl_max, ...
        params ...
    );
    eq_sys = [eq_sys; eqs]; ineq_sys = [ineq_sys; ineqs]; eq_mask_sys = [eq_mask_sys; eq_mask];

    % get two fingers' self collision information
    [eqs, ineqs, eq_mask] = get_self_collision_info(...
        var("rcld", k), var("rsld", k), var("rclu", k), var("rslu", k), ...
        var("rcrd", k), var("rsrd", k), var("rcru", k), var("rsru", k), ...
        params ...
    );
    eq_sys = [eq_sys; eqs]; ineq_sys = [ineq_sys; ineqs]; eq_mask_sys = [eq_mask_sys; eq_mask];

    % get two fingers' contact models
    [eqs, ineqs, eq_mask] = get_right_finger_contact(...
        var("x", k-1) * params.x_max, var("x", k) * params.x_max, var("y", k-1) * params.y_max, var("y", k) * params.y_max, var("fs", k-1), ...
        var("xr", k-1) * params.xr_max, var("yr", k-1) * params.yr_max, var("vxr", k-1) * params.vxr_max, var("vyr", k-1) * params.vyr_max, ...
        var("dr", k-1) * params.dr_max, var("vrelr", k-1) * params.vrelr_max, var("lamnr", k-1) * params.lamnr_max, var("lamtr", k-1) * params.lamtr_max, ...
        params ...
    );
    eq_sys = [eq_sys; eqs]; ineq_sys = [ineq_sys; ineqs]; eq_mask_sys = [eq_mask_sys; eq_mask];
    [eqs, ineqs, eq_mask] = get_left_finger_contact(...
        var("x", k-1) * params.x_max, var("x", k) * params.x_max, var("y", k-1) * params.y_max, var("y", k) * params.y_max, var("fs", k-1), ...
        var("xl", k-1) * params.xl_max, var("yl", k-1) * params.yl_max, var("vxl", k-1) * params.vxl_max, var("vyl", k-1) * params.vyl_max, ...
        var("dl", k-1) * params.dl_max, var("vrell", k-1) * params.vrell_max, var("lamnl", k-1) * params.lamnl_max, var("lamtl", k-1) * params.lamtl_max, ...
        params ...
    );
    eq_sys = [eq_sys; eqs]; ineq_sys = [ineq_sys; ineqs]; eq_mask_sys = [eq_mask_sys; eq_mask];

    % get circle dynamics
    [eqs, ineqs, eq_mask] = get_circle_dynamics( ...
        var("x", k-1) * params.x_max, var("x", k) * params.x_max, var("y", k-1) * params.y_max, var("y", k) * params.y_max, ...
        var("rc", k-1), var("rc", k), var("rs", k-1), var("rs", k), var("fc", k-1), var("fs", k-1), ...
        var("xl", k-1) * params.xl_max, var("yl", k-1) * params.yl_max, var("xr", k-1) * params.xr_max, var("yr", k-1) * params.yr_max, ...
        var("lamnl", k-1) * params.lamnl_max, var("lamtl", k-1) * params.lamtl_max, ...
        var("lamnr", k-1) * params.lamnr_max, var("lamtr", k-1) * params.lamtr_max, ...
        params ...
    );
    eq_sys = [eq_sys; eqs]; ineq_sys = [ineq_sys; ineqs]; eq_mask_sys = [eq_mask_sys; eq_mask];

    % get collision avoidance info
    [eqs, ineqs, eq_mask] = get_collision_avoidance_info(...
        var("x", k) * params.x_max, var("y", k) * params.y_max, ...
        var("rcld", k), var("rsld", k), var("rclu", k), var("rslu", k), ...
        var("rcrd", k), var("rsrd", k), var("rcru", k), var("rsru", k), ...
        params ...
    );
    eq_sys = [eq_sys; eqs]; ineq_sys = [ineq_sys; ineqs]; eq_mask_sys = [eq_mask_sys; eq_mask];

    % bound constraints
    ineq_sys = [
        ineq_sys;
        1.0 - var("x", k)^2; 1.0 - var("y", k)^2;
        1.0 - var("xl", k-1)^2; 1.0 - var("yl", k-1)^2; 1.0 - var("vxl", k-1)^2; 1.0 - var("vyl", k-1)^2; 
        1.0 - var("xr", k-1)^2; 1.0 - var("yr", k-1)^2; 1.0 - var("vxr", k-1)^2; 1.0 - var("vyr", k-1)^2; 
        1.0 - var("dl", k-1)^2; 1.0 - var("vrell", k-1)^2; 1.0 - var("lamnl", k-1)^2; 1.0 - var("lamtl", k-1)^2;
        1.0 - var("dr", k-1)^2; 1.0 - var("vrelr", k-1)^2; 1.0 - var("lamnr", k-1)^2; 1.0 - var("lamtr", k-1)^2;
    ];
end

%% generate POP: loss function
obj_sys = 0.0;
% tracking loss: (x, y) remain unchanged, track th
x_init_scaled = params.x_init / params.x_max;
y_init_scaled = params.y_init / params.y_max;
for k = 1: params.N 
    loss_x = params.loss_track_coeff * (var("x", k) - x_init_scaled)^2;
    loss_y = params.loss_track_coeff * (var("y", k) - y_init_scaled)^2;
    th_track = params.loss_track_th_list(k);
    loss_th = params.loss_track_coeff * (var("rc", k) - cos(th_track))^2 + params.loss_track_coeff * (var("rs", k) - sin(th_track))^2;
    obj_sys = obj_sys + loss_x + loss_y + loss_th;
end
% position loss: finger position
for k = 1: params.N 
    loss_rcld = params.loss_pos_finger_coeff * ( var("rcld", k) - cos(params.th_ld_init) )^2;
    loss_rsld = params.loss_pos_finger_coeff * ( var("rsld", k) - sin(params.th_ld_init) )^2;
    loss_rclu = params.loss_pos_finger_coeff * ( var("rclu", k) - cos(params.th_lu_init) )^2;
    loss_rslu = params.loss_pos_finger_coeff * ( var("rslu", k) - sin(params.th_lu_init) )^2;
    loss_rcrd = params.loss_pos_finger_coeff * ( var("rcrd", k) - cos(params.th_rd_init) )^2;
    loss_rsrd = params.loss_pos_finger_coeff * ( var("rsrd", k) - sin(params.th_rd_init) )^2;
    loss_rcru = params.loss_pos_finger_coeff * ( var("rcru", k) - cos(params.th_ru_init) )^2;
    loss_rsru = params.loss_pos_finger_coeff * ( var("rsru", k) - sin(params.th_ru_init) )^2;
    obj_sys = obj_sys + loss_rcld + loss_rsld + loss_rclu + loss_rslu + loss_rcrd + loss_rsrd + loss_rcru + loss_rsru;
end
% velocity loss: finger velocity
for k = 1: params.N 
    loss_fsld = params.loss_vel_finger_coeff * var("fsld", k-1)^2;
    loss_fslu = params.loss_vel_finger_coeff * var("fslu", k-1)^2;
    loss_fsrd = params.loss_vel_finger_coeff * var("fsrd", k-1)^2;
    loss_fsru = params.loss_vel_finger_coeff * var("fsru", k-1)^2;
    loss_vxl = params.loss_vel_finger_coeff * ( var("vxl", k-1) * params.vxl_max )^2;
    loss_vyl = params.loss_vel_finger_coeff * ( var("vyl", k-1) * params.vyl_max )^2;
    loss_vxr = params.loss_vel_finger_coeff * ( var("vxr", k-1) * params.vxr_max )^2;
    loss_vyr = params.loss_vel_finger_coeff * ( var("vyr", k-1) * params.vyr_max )^2;
    obj_sys = obj_sys + loss_fsld + loss_fslu + loss_fsrd + loss_fsru + loss_vxl + loss_vyl + loss_vxr + loss_vyr;
end
% velocity loss: circle velocity
for k = 1: params.N 
    loss_fs = params.loss_vel_circle_coeff * var("fs", k-1)^2;
    obj_sys = obj_sys + loss_fs;
    % loss_vx = params.loss_vel_circle_coeff * ( (var("x", k) - var("x", k-1)) / params.dt * params.vx_max )^2;
    % loss_vy = params.loss_vel_circle_coeff * ( (var("y", k) - var("y", k-1)) / params.dt * params.vy_max )^2;
    % obj_sys = obj_sys + loss_fs + loss_vx + loss_vy;
end
% force loss
for k = 1: params.N 
    loss_lamnl = params.loss_force_coeff * ( var("lamnl", k-1) * params.lamnl_max )^2;
    loss_lamtl = params.loss_force_coeff * ( var("lamtl", k-1) * params.lamtl_max )^2;
    loss_lamnr = params.loss_force_coeff * ( var("lamnr", k-1) * params.lamnr_max )^2;
    loss_lamtr = params.loss_force_coeff * ( var("lamtr", k-1) * params.lamtr_max )^2;
    obj_sys = obj_sys + loss_lamnl + loss_lamtl + loss_lamnr + loss_lamtr;
end

%% generate POP: rescale everything
[eq_sys, ~] = msspoly_clean(eq_sys, v, 1e-14, true);
[ineq_sys, ~] = msspoly_clean(ineq_sys, v, 1e-14, true);
[obj_sys, obj_scale_factor] = msspoly_clean(obj_sys, v, 1e-14, false);
disp('Construction Finish!');

%% generate heuristic CS clique decomposition
cliques = cell(N, 1);
idx = 1;
% cliques generated by initial constraints
cliques{idx} = [
    id("x", 0); id("y", 0); id("rc", 0); id("rs", 0); 
    id("rcld", 0); id("rsld", 0); id("rclu", 0); id("rslu", 0); 
    id("rcrd", 0); id("rsrd", 0); id("rcru", 0); id("rsru", 0);
]; idx = idx + 1;

for k = 1: N 
    % cliques generated by collision avoidance
    cliques{idx} = [
        id("x", k); id("y", k);
        id("rcld", k); id("rsld", k); id("rclu", k); id("rslu", k);
        id("rcrd", k); id("rsrd", k); id("rcru", k); id("rsru", k);
    ]; idx = idx + 1;

    % cliques generated by circle dynamics
    cliques{idx} = [
        id("x", k-1); id("x", k); id("y", k-1); id("y", k); id("fc", k-1); id("fs", k-1); 
        id("xl", k-1); id("yl", k-1); id("xr", k-1); id("yr", k-1);
        id("lamnl", k-1); id("lamtl", k-1);
        id("lamnr", k-1); id("lamtr", k-1); 
    ]; idx = idx + 1;
    cliques{idx} = [
        id("rc", k-1); id("rc", k); id("rs", k-1); id("rs", k); id("fc", k-1); id("fs", k-1);  
    ]; idx = idx + 1;

    % cliques generated by right finger's contact model
    cliques{idx} = [
        id("x", k-1); id("x", k); id("y", k-1); id("y", k); id("fc", k-1); id("fs", k-1); 
        id("xr", k-1); id("yr", k-1); id("vxr", k-1); id("vyr", k-1);
        id("dr", k-1); id("vrelr", k-1); id("lamnr", k-1); id("lamtr", k-1);
    ]; idx = idx + 1;
    cliques{idx} = [
        id("x", k-1); id("y", k-1); id("xr", k-1); id("yr", k-1); 
        id("dr", k-1); 
        id("rcrd", k-1); id("rsrd", k-1); id("rcru", k-1); id("rsru", k-1);
    ]; idx = idx + 1;

    % cliques generated by left finger's contact model
    cliques{idx} = [
        id("x", k-1); id("x", k); id("y", k-1); id("y", k); id("fc", k-1); id("fs", k-1); 
        id("xl", k-1); id("yl", k-1); id("vxl", k-1); id("vyl", k-1);
        id("dl", k-1); id("vrell", k-1); id("lamnl", k-1); id("lamtl", k-1);
    ]; idx = idx + 1;
    cliques{idx} = [
        id("x", k-1); id("y", k-1); id("xl", k-1); id("yl", k-1); 
        id("dl", k-1); 
        id("rcld", k-1); id("rsld", k-1); id("rclu", k-1); id("rslu", k-1);
    ]; idx = idx + 1;

    % cliques generated by right finger's kinematics
    cliques{idx} = [
        id("rcrd", k-1); id("rsrd", k-1); 
        id("rcru", k-1); id("rsru", k-1); 
        id("xr", k-1); id("yr", k-1);
    ]; idx = idx + 1;
    cliques{idx} = [
        id("rcrd", k-1); id("rsrd", k-1); id("fcrd", k-1); id("fsrd", k-1);
        id("rcru", k-1); id("rsru", k-1); id("fcru", k-1); id("fsru", k-1);
        id("vxr", k-1); id("vyr", k-1);
    ]; idx = idx + 1;
    cliques{idx} = [
        id("rcrd", k-1); id("rcrd", k); id("rsrd", k-1); id("rsrd", k); id("fcrd", k-1); id("fsrd", k-1);
        id("rcru", k-1); id("rcru", k); id("rsru", k-1); id("rsru", k); id("fcru", k-1); id("fsru", k-1);
    ]; idx = idx + 1;

    % cliques generated by left finger's kinematics
    cliques{idx} = [
        id("rcld", k-1); id("rsld", k-1); 
        id("rclu", k-1); id("rslu", k-1); 
        id("xl", k-1); id("yl", k-1);
    ]; idx = idx + 1;
    cliques{idx} = [
        id("rcld", k-1); id("rsld", k-1); id("fcld", k-1); id("fsld", k-1);
        id("rclu", k-1); id("rslu", k-1); id("fclu", k-1); id("fslu", k-1);
        id("vxl", k-1); id("vyl", k-1);
    ]; idx = idx + 1;
    cliques{idx} = [
        id("rcld", k-1); id("rcld", k); id("rsld", k-1); id("rsld", k); id("fcld", k-1); id("fsld", k-1);
        id("rclu", k-1); id("rclu", k); id("rslu", k-1); id("rslu", k); id("fclu", k-1); id("fslu", k-1);
    ]; idx = idx + 1;
end
params.cliques = cliques;

%% run CSTSS
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
fprintf(fid, 'PlanarHand  N=%d, Relax=%s, TS=%s, CS=%s, result=%.20f, operation=%.5f\n\n', N, relax_mode, ts_mode, cs_mode, result, aux_info.time);
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
params.self_cliques = cliques;

%% remap clique ids
cliques_remapped = cell(size(aux_info.cliques));
aver_remapped = zeros(size(aux_info.cliques));
for i = 1: size(aux_info.cliques, 1)
    cliques_remapped{i} = params.ids_remap(aux_info.cliques{i});
    aver_remapped(i) = mean(cliques_remapped{i});
end
[~, cliques_rank] = sort(aver_remapped);
params.cliques_rank = cliques_rank;
if ~params.if_solve
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

%% helper functions
% get initial constraints
function [eqs, ineqs, eq_mask] = get_init_constraints( ...
    x_0, y_0, rc_0, rs_0, ...
    rcld_0, rsld_0, rclu_0, rslu_0, ...
    rcrd_0, rsrd_0, rcru_0, rsru_0, ...
    params ...
)   
    eqs = [];
    ineqs = [];
    eq_mask = [];
    eqs = [eqs; x_0 - params.x_init]; eq_mask = [eq_mask; 1];
    eqs = [eqs; y_0 - params.y_init]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rc_0 - cos(params.th_init)]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rs_0 - sin(params.th_init)]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcld_0 - cos(params.th_ld_init)]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rsld_0 - sin(params.th_ld_init)]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rclu_0 - cos(params.th_lu_init)]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rslu_0 - sin(params.th_lu_init)]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcrd_0 - cos(params.th_rd_init)]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rsrd_0 - sin(params.th_rd_init)]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcru_0 - cos(params.th_ru_init)]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rsru_0 - sin(params.th_ru_init)]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rc_0^2 + rs_0^2 - 1]; eq_mask = [eq_mask; 0];
    eqs = [eqs; rcld_0^2 + rsld_0^2 - 1]; eq_mask = [eq_mask; 0];
    eqs = [eqs; rclu_0^2 + rslu_0^2 - 1]; eq_mask = [eq_mask; 0];
    eqs = [eqs; rcrd_0^2 + rsrd_0^2 - 1]; eq_mask = [eq_mask; 0];
    eqs = [eqs; rcru_0^2 + rsru_0^2 - 1]; eq_mask = [eq_mask; 0];
end

% get collision avoidance info
function [eqs, ineqs, eq_mask] = get_collision_avoidance_info(...
    x_k, y_k, ...
    rcld_k, rsld_k, rclu_k, rslu_k, ...
    rcrd_k, rsrd_k, rcru_k, rsru_k, ...
    params ...
)
    r = params.r; R = params.R; l = params.l; Ld = params.Ld; Lu = params.Lu; H = params.H;
    eqs = [];
    ineqs = [];
    eq_mask = [];
    ineqs = [ineqs; ( (l+r)*rcrd_k + H/2 - x_k )^2 + ( (l+r)*rsrd_k - y_k )^2 - (R+r)^2];
    ineqs = [ineqs; ( (2*l+3*r)*rcrd_k + H/2 - x_k )^2 + ( (2*l+3*r)*rsrd_k - y_k )^2 - (R+r)^2];
    ineqs = [ineqs; ( Ld*rcrd_k + (l+r)*rcru_k + H/2 - x_k )^2 + ( Ld*rsrd_k + (l+r)*rsru_k - y_k )^2 - (R+r)^2];
    ineqs = [ineqs; ( Ld*rcrd_k + (2*l+3*r)*rcru_k + H/2 - x_k )^2 + ( Ld*rsrd_k + (2*l+3*r)*rsru_k - y_k )^2 - (R+r)^2];
    ineqs = [ineqs; ( (l+r)*rcld_k - H/2 - x_k )^2 + ( (l+r)*rsld_k - y_k )^2 - (R+r)^2];
    ineqs = [ineqs; ( (2*l+3*r)*rcld_k - H/2 - x_k )^2 + ( (2*l+3*r)*rsld_k - y_k )^2 - (R+r)^2];
    ineqs = [ineqs; ( Ld*rcld_k + (l+r)*rclu_k - H/2 - x_k )^2 + ( Ld*rsld_k + (l+r)*rslu_k - y_k )^2 - (R+r)^2];
    ineqs = [ineqs; ( Ld*rcld_k + (2*l+3*r)*rclu_k - H/2 - x_k )^2 + ( Ld*rsld_k + (2*l+3*r)*rslu_k - y_k )^2 - (R+r)^2];
end

% get circle dynamics
function [eqs, ineqs, eq_mask] = get_circle_dynamics( ...
    x_km1, x_k, y_km1, y_k, ...
    rc_km1, rc_k, rs_km1, rs_k, fc_km1, fs_km1, ...
    xl_km1, yl_km1, xr_km1, yr_km1, ...
    lamnl_km1, lamtl_km1, lamnr_km1, lamtr_km1, ...
    params ...
)
    r = params.r; R = params.R; dt = params.dt; c = params.c;
    vx_km1 = (x_k - x_km1) / dt;
    vy_km1 = (y_k - y_km1) / dt;
    cos_etal = (xl_km1 - x_km1) / (R+r);
    sin_etal = (yl_km1 - y_km1) / (R+r);
    cos_etar = (xr_km1 - x_km1) / (R+r);
    sin_etar = (yr_km1 - y_km1) / (R+r);
    eqs = [];
    ineqs = [];
    eq_mask = [];
    eqs = [eqs; lamnr_km1 * cos_etar - lamtr_km1 * sin_etar + lamnl_km1 * cos_etal - lamtl_km1 * sin_etal - vx_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; lamnr_km1 * sin_etar + lamtr_km1 * cos_etar + lamnl_km1 * sin_etal + lamtl_km1 * cos_etal - vy_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; dt/(c*R) * (lamtr_km1 + lamtl_km1) - fs_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rc_km1 * fc_km1 - rs_km1 * fs_km1 - rc_k]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rc_km1 * fs_km1 + rs_km1 * fc_km1 - rs_k]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rc_k^2 + rs_k^2 - 1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; fc_km1^2 + fs_km1^2 - 1]; eq_mask = [eq_mask; 1];
    ineqs = [ineqs; fc_km1 - params.fc_min];
end

% get left finger's contact model
function [eqs, ineqs, eq_mask] = get_left_finger_contact(...
    x_km1, x_k, y_km1, y_k, fs_km1, ...
    xl_km1, yl_km1, vxl_km1, vyl_km1, ...
    dl_km1, vrell_km1, lamnl_km1, lamtl_km1, ...
    params ...
)
    r = params.r; R = params.R; dt = params.dt; mu = params.mu;
    vx_km1 = (x_k - x_km1) / dt;
    vy_km1 = (y_k - y_km1) / dt;
    cos_etal = (xl_km1 - x_km1) / (R+r);
    sin_etal = (yl_km1 - y_km1) / (R+r);
    eqs = [];
    ineqs = [];
    eq_mask = [];
    eqs = [eqs; (xl_km1 - x_km1)^2 + (yl_km1 - y_km1)^2 - dl_km1^2]; eq_mask = [eq_mask; 1];
    ineqs = [ineqs; dl_km1 - (R+r)];
    eqs = [eqs; -(vxl_km1 - vx_km1) * sin_etal + (vyl_km1 - vy_km1) * cos_etal - R/dt * fs_km1 - vrell_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; (dl_km1 - R - r) * lamnl_km1]; eq_mask = [eq_mask; 1];
    ineqs = [ineqs; -lamnl_km1];
    ineqs = [ineqs; mu^2 * lamnl_km1^2 - lamtl_km1^2];
    eqs = [eqs; vrell_km1 * (mu^2 * lamnl_km1^2 - lamtl_km1^2)]; eq_mask = [eq_mask; 1];
    ineqs = [ineqs; vrell_km1 * lamtl_km1];
end

% get right finger's contact model
function [eqs, ineqs, eq_mask] = get_right_finger_contact(...
    x_km1, x_k, y_km1, y_k, fs_km1, ...
    xr_km1, yr_km1, vxr_km1, vyr_km1, ...
    dr_km1, vrelr_km1, lamnr_km1, lamtr_km1, ...
    params ...
)
    r = params.r; R = params.R; dt = params.dt; mu = params.mu;
    vx_km1 = (x_k - x_km1) / dt;
    vy_km1 = (y_k - y_km1) / dt;
    cos_etar = (xr_km1 - x_km1) / (R+r);
    sin_etar = (yr_km1 - y_km1) / (R+r);
    eqs = [];
    ineqs = [];
    eq_mask = [];
    eqs = [eqs; (xr_km1 - x_km1)^2 + (yr_km1 - y_km1)^2 - dr_km1^2]; eq_mask = [eq_mask; 1];
    ineqs = [ineqs; dr_km1 - (R+r)];
    eqs = [eqs; -(vxr_km1 - vx_km1) * sin_etar + (vyr_km1 - vy_km1) * cos_etar - R/dt * fs_km1 - vrelr_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; (dr_km1 - R - r) * lamnr_km1]; eq_mask = [eq_mask; 1];
    ineqs = [ineqs; -lamnr_km1];
    ineqs = [ineqs; mu^2 * lamnr_km1^2 - lamtr_km1^2];
    eqs = [eqs; vrelr_km1 * (mu^2 * lamnr_km1^2 - lamtr_km1^2)]; eq_mask = [eq_mask; 1];
    ineqs = [ineqs; vrelr_km1 * lamtr_km1];
end

% get two fingers' self collision information
function [eqs, ineqs, eq_mask] = get_self_collision_info(...
    rcld_k, rsld_k, rclu_k, rslu_k, ...
    rcrd_k, rsrd_k, rcru_k, rsru_k, ...
    params ...
)
    th0 = params.theta_0; r = params.r; H = params.H; Ld = params.Ld; Lu = params.Lu; 
    eqs = [];
    ineqs = [];
    eq_mask = [];
    ineqs = [ineqs; rsld_k - sin(th0)]; 
    ineqs = [ineqs; rsrd_k - sin(th0)];
    ineqs = [ineqs; rsld_k * rclu_k - rcld_k * rslu_k];
    ineqs = [ineqs; rcld_k * rclu_k + rsld_k * rslu_k + cos(2 * th0)];
    ineqs = [ineqs; rsru_k * rcrd_k - rcru_k * rsrd_k];
    ineqs = [ineqs; rcru_k * rcrd_k + rsru_k * rsrd_k + cos(2 * th0)];
    ineqs = [ineqs; -(Ld * rcld_k + Lu * rclu_k - H/2) - r];
    ineqs = [ineqs; (Ld * rcrd_k + Lu * rcru_k + H/2) - r];
end

% get left finger's kinematics
function [eqs, ineqs, eq_mask] = get_left_finger_kinematics(...
    rcld_km1, rcld_k, rsld_km1, rsld_k, fcld_km1, fsld_km1, ...
    rclu_km1, rclu_k, rslu_km1, rslu_k, fclu_km1, fslu_km1, ...
    xl_km1, yl_km1, vxl_km1, vyl_km1, ...
    params ...
)   
    H = params.H; Ld = params.Ld; Lu = params.Lu; dt = params.dt;
    eqs = [];
    ineqs = [];
    eq_mask = [];
    eqs = [eqs; Ld * rcld_km1 + Lu * rclu_km1 - H/2 - xl_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; Ld * rsld_km1 + Lu * rslu_km1 - yl_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; -Ld/dt * rsld_km1 * fsld_km1 - Lu/dt * rslu_km1 * fslu_km1 - vxl_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; Ld/dt * rcld_km1 * fsld_km1 + Lu/dt * rclu_km1 * fslu_km1 - vyl_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcld_km1 * fcld_km1 - rsld_km1 * fsld_km1 - rcld_k]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcld_km1 * fsld_km1 + rsld_km1 * fcld_km1 - rsld_k]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rclu_km1 * fclu_km1 - rslu_km1 * fslu_km1 - rclu_k]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rclu_km1 * fslu_km1 + rslu_km1 * fclu_km1 - rslu_k]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcld_k^2 + rsld_k^2 - 1]; eq_mask = [eq_mask; 0];
    eqs = [eqs; rclu_k^2 + rslu_k^2 - 1]; eq_mask = [eq_mask; 0];
    eqs = [eqs; fcld_km1^2 + fsld_km1^2 - 1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; fclu_km1^2 + fslu_km1^2 - 1]; eq_mask = [eq_mask; 1];
    ineqs = [ineqs; fcld_km1 - params.fcld_min];
    ineqs = [ineqs; fclu_km1 - params.fclu_min];
end

% get right finger's kinematics
function [eqs, ineqs, eq_mask] = get_right_finger_kinematics(...
    rcrd_km1, rcrd_k, rsrd_km1, rsrd_k, fcrd_km1, fsrd_km1, ...
    rcru_km1, rcru_k, rsru_km1, rsru_k, fcru_km1, fsru_km1, ...
    xr_km1, yr_km1, vxr_km1, vyr_km1, ...
    params ...
)   
    H = params.H; Ld = params.Ld; Lu = params.Lu; dt = params.dt;
    eqs = [];
    ineqs = [];
    eq_mask = [];
    eqs = [eqs; Ld * rcrd_km1 + Lu * rcru_km1 + H/2 - xr_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; Ld * rsrd_km1 + Lu * rsru_km1 - yr_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; -Ld/dt * rsrd_km1 * fsrd_km1 - Lu/dt * rsru_km1 * fsru_km1 - vxr_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; Ld/dt * rcrd_km1 * fsrd_km1 + Lu/dt * rcru_km1 * fsru_km1 - vyr_km1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcrd_km1 * fcrd_km1 - rsrd_km1 * fsrd_km1 - rcrd_k]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcrd_km1 * fsrd_km1 + rsrd_km1 * fcrd_km1 - rsrd_k]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcru_km1 * fcru_km1 - rsru_km1 * fsru_km1 - rcru_k]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcru_km1 * fsru_km1 + rsru_km1 * fcru_km1 - rsru_k]; eq_mask = [eq_mask; 1];
    eqs = [eqs; rcrd_k^2 + rsrd_k^2 - 1]; eq_mask = [eq_mask; 0];
    eqs = [eqs; rcru_k^2 + rsru_k^2 - 1]; eq_mask = [eq_mask; 0];
    eqs = [eqs; fcrd_km1^2 + fsrd_km1^2 - 1]; eq_mask = [eq_mask; 1];
    eqs = [eqs; fcru_km1^2 + fsru_km1^2 - 1]; eq_mask = [eq_mask; 1];
    ineqs = [ineqs; fcrd_km1 - params.fcrd_min];
    ineqs = [ineqs; fcru_km1 - params.fcru_min];
end

% a light wrapper that returns v(id)
function var = get_var(v, prefix, k, var_start_dict)
    var = v(get_id(prefix, k, var_start_dict));
end

% get id based on pre-stored info for indexing
function id = get_id(prefix, k, var_start_dict)
    start = var_start_dict(prefix);
    switch prefix 
        case {"x", "y", "rc", "rs", "rcld", "rsld", "rclu", "rslu", "rcrd", "rsrd", "rcru", "rsru"}
            id = start + k;
        otherwise
            assert(k < var_start_dict("N"))
            id = start + k;
    end
end

% get var_mapping, and some pre-stored info for indexing
function [var_mapping, var_start_dict, total_var_num] = get_var_mapping(N)
    % variable layout:
    % only circle: (x, y), (rc, rs, fc, fs) (vx and vy can be represented by (x, y))
    % only left finger: (rcld, rsld, fcld, fsld), (rclu, rslu, fclu, fslu), (xl, yl, vxl, vyl)
    % only right finger: (rcrd, rsrd, fcrd, fsrd), (rcru, rsru, fcru, fsru), (xr, yr, vxr, vyr)
    % between left finger and the circle: dl, vrell, (lamnl, lamtl)
    % between right finger and the circle: dr, vrelr, (lamnr, lamtr)
    % variables that need (N+1) number: (x, y), (rc, rs), (rcld, rsld), (rclu, rslu), (rcrd, rsrd), (rcru, rsru)
    var_start_dict = dictionary();
    var_mapping = strings(N, 1);
    cnt = 1;
    short_list = 0: N-1;
    long_list = 0: N; 
    % only circle 
    var_start_dict("x") = cnt; for k = long_list, var_mapping(cnt) = sprintf("x_{%d}", k); cnt = cnt + 1; end 
    var_start_dict("y") = cnt; for k = long_list, var_mapping(cnt) = sprintf("y_{%d}", k); cnt = cnt + 1; end 
    var_start_dict("rc") = cnt; for k = long_list, var_mapping(cnt) = sprintf("r_{c, %d}", k); cnt = cnt + 1; end 
    var_start_dict("rs") = cnt; for k = long_list, var_mapping(cnt) = sprintf("r_{s, %d}", k); cnt = cnt + 1; end
    var_start_dict("fc") = cnt; for k = short_list, var_mapping(cnt) = sprintf("f_{c, %d}", k); cnt = cnt + 1; end 
    var_start_dict("fs") = cnt; for k = short_list, var_mapping(cnt) = sprintf("f_{s, %d}", k); cnt = cnt + 1; end
    % only left finger
    var_start_dict("rcld") = cnt; for k = long_list, var_mapping(cnt) = sprintf("r_{c, ld, %d}", k); cnt = cnt + 1; end 
    var_start_dict("rsld") = cnt; for k = long_list, var_mapping(cnt) = sprintf("r_{s, ld, %d}", k); cnt = cnt + 1; end
    var_start_dict("fcld") = cnt; for k = short_list, var_mapping(cnt) = sprintf("f_{c, ld, %d}", k); cnt = cnt + 1; end 
    var_start_dict("fsld") = cnt; for k = short_list, var_mapping(cnt) = sprintf("f_{s, ld, %d}", k); cnt = cnt + 1; end
    var_start_dict("rclu") = cnt; for k = long_list, var_mapping(cnt) = sprintf("r_{c, lu, %d}", k); cnt = cnt + 1; end 
    var_start_dict("rslu") = cnt; for k = long_list, var_mapping(cnt) = sprintf("r_{s, lu, %d}", k); cnt = cnt + 1; end
    var_start_dict("fclu") = cnt; for k = short_list, var_mapping(cnt) = sprintf("f_{c, lu, %d}", k); cnt = cnt + 1; end 
    var_start_dict("fslu") = cnt; for k = short_list, var_mapping(cnt) = sprintf("f_{s, lu, %d}", k); cnt = cnt + 1; end                         
    var_start_dict("xl") = cnt; for k = short_list, var_mapping(cnt) = sprintf("x_{l, %d}", k); cnt = cnt + 1; end 
    var_start_dict("yl") = cnt; for k = short_list, var_mapping(cnt) = sprintf("y_{l, %d}", k); cnt = cnt + 1; end
    var_start_dict("vxl") = cnt; for k = short_list, var_mapping(cnt) = sprintf("v_{x, l, %d}", k); cnt = cnt + 1; end 
    var_start_dict("vyl") = cnt; for k = short_list, var_mapping(cnt) = sprintf("v_{y, l, %d}", k); cnt = cnt + 1; end
    % only right finger
    var_start_dict("rcrd") = cnt; for k = long_list, var_mapping(cnt) = sprintf("r_{c, rd, %d}", k); cnt = cnt + 1; end 
    var_start_dict("rsrd") = cnt; for k = long_list, var_mapping(cnt) = sprintf("r_{s, rd, %d}", k); cnt = cnt + 1; end
    var_start_dict("fcrd") = cnt; for k = short_list, var_mapping(cnt) = sprintf("f_{c, rd, %d}", k); cnt = cnt + 1; end 
    var_start_dict("fsrd") = cnt; for k = short_list, var_mapping(cnt) = sprintf("f_{s, rd, %d}", k); cnt = cnt + 1; end
    var_start_dict("rcru") = cnt; for k = long_list, var_mapping(cnt) = sprintf("r_{c, ru, %d}", k); cnt = cnt + 1; end 
    var_start_dict("rsru") = cnt; for k = long_list, var_mapping(cnt) = sprintf("r_{s, ru, %d}", k); cnt = cnt + 1; end
    var_start_dict("fcru") = cnt; for k = short_list, var_mapping(cnt) = sprintf("f_{c, ru, %d}", k); cnt = cnt + 1; end 
    var_start_dict("fsru") = cnt; for k = short_list, var_mapping(cnt) = sprintf("f_{s, ru, %d}", k); cnt = cnt + 1; end                         
    var_start_dict("xr") = cnt; for k = short_list, var_mapping(cnt) = sprintf("x_{r, %d}", k); cnt = cnt + 1; end 
    var_start_dict("yr") = cnt; for k = short_list, var_mapping(cnt) = sprintf("y_{r, %d}", k); cnt = cnt + 1; end
    var_start_dict("vxr") = cnt; for k = short_list, var_mapping(cnt) = sprintf("v_{x, r, %d}", k); cnt = cnt + 1; end 
    var_start_dict("vyr") = cnt; for k = short_list, var_mapping(cnt) = sprintf("v_{y, r, %d}", k); cnt = cnt + 1; end
    % between left finger and the circle
    var_start_dict("dl") = cnt; for k = short_list, var_mapping(cnt) = sprintf("d_{l, %d}", k); cnt = cnt + 1; end 
    var_start_dict("vrell") = cnt; for k = short_list, var_mapping(cnt) = sprintf("v_{rel, l, %d}", k); cnt = cnt + 1; end
    var_start_dict("lamnl") = cnt; for k = short_list, var_mapping(cnt) = "\" + sprintf("\\lambda_{n, l, %d}", k); cnt = cnt + 1; end 
    var_start_dict("lamtl") = cnt; for k = short_list, var_mapping(cnt) = "\" + sprintf("\\lambda_{t, l, %d}", k); cnt = cnt + 1; end
    % between right finger and the circle
    var_start_dict("dr") = cnt; for k = short_list, var_mapping(cnt) = sprintf("d_{r, %d}", k); cnt = cnt + 1; end 
    var_start_dict("vrelr") = cnt; for k = short_list, var_mapping(cnt) = sprintf("v_{rel, r, %d}", k); cnt = cnt + 1; end
    var_start_dict("lamnr") = cnt; for k = short_list, var_mapping(cnt) = "\" + sprintf("\\lambda_{n, r, %d}", k); cnt = cnt + 1; end 
    var_start_dict("lamtr") = cnt; for k = short_list, var_mapping(cnt) = "\" + sprintf("\\lambda_{t, r, %d}", k); cnt = cnt + 1; end
    % total variable number
    total_var_num = cnt - 1;
    % store N
    var_start_dict("N") = N;
end

% re-map variable ids to recover chain-like sparsity pattern
function ids_remap = get_remapped_ids(params)
    N = params.N; total_var_num = params.total_var_num; id = params.id;
    ids_remap = zeros(total_var_num, 1);
    idx = 1;
    for k = 0: N
        ids_remap(id("x", k)) = idx; idx = idx + 1;
        ids_remap(id("y", k)) = idx; idx = idx + 1;
        ids_remap(id("rc", k)) = idx; idx = idx + 1;
        ids_remap(id("rs", k)) = idx; idx = idx + 1;
        if k < N 
            ids_remap(id("fc", k)) = idx; idx = idx + 1;
            ids_remap(id("fs", k)) = idx; idx = idx + 1;
        end
        ids_remap(id("rcld", k)) = idx; idx = idx + 1;
        ids_remap(id("rsld", k)) = idx; idx = idx + 1;
        ids_remap(id("rclu", k)) = idx; idx = idx + 1;
        ids_remap(id("rslu", k)) = idx; idx = idx + 1;
        ids_remap(id("rcrd", k)) = idx; idx = idx + 1;
        ids_remap(id("rsrd", k)) = idx; idx = idx + 1;
        ids_remap(id("rcru", k)) = idx; idx = idx + 1;
        ids_remap(id("rsru", k)) = idx; idx = idx + 1;
        if k < N 
            ids_remap(id("fcld", k)) = idx; idx = idx + 1;
            ids_remap(id("fsld", k)) = idx; idx = idx + 1;
            ids_remap(id("fclu", k)) = idx; idx = idx + 1;
            ids_remap(id("fslu", k)) = idx; idx = idx + 1;
            ids_remap(id("fcrd", k)) = idx; idx = idx + 1;
            ids_remap(id("fsrd", k)) = idx; idx = idx + 1;
            ids_remap(id("fcru", k)) = idx; idx = idx + 1;
            ids_remap(id("fsru", k)) = idx; idx = idx + 1;
            ids_remap(id("xl", k)) = idx; idx = idx + 1;
            ids_remap(id("yl", k)) = idx; idx = idx + 1;
            ids_remap(id("vxl", k)) = idx; idx = idx + 1;
            ids_remap(id("vyl", k)) = idx; idx = idx + 1;
            ids_remap(id("xr", k)) = idx; idx = idx + 1;
            ids_remap(id("yr", k)) = idx; idx = idx + 1;
            ids_remap(id("vxr", k)) = idx; idx = idx + 1;
            ids_remap(id("vyr", k)) = idx; idx = idx + 1;
            ids_remap(id("dl", k)) = idx; idx = idx + 1;
            ids_remap(id("vrell", k)) = idx; idx = idx + 1;
            ids_remap(id("lamnl", k)) = idx; idx = idx + 1;
            ids_remap(id("lamtl", k)) = idx; idx = idx + 1;
            ids_remap(id("dr", k)) = idx; idx = idx + 1;
            ids_remap(id("vrelr", k)) = idx; idx = idx + 1;
            ids_remap(id("lamnr", k)) = idx; idx = idx + 1;
            ids_remap(id("lamtr", k)) = idx; idx = idx + 1;
        end
    end
end








