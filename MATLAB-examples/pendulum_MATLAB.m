clear; close all;

addpath("../pathinfo/");
my_path;

%% CSTSS parameters
if_mex = true; params.if_mex = if_mex;
kappa = 2; params.kappa = kappa;
relax_mode = "SOS"; params.relax_mode = relax_mode;
cs_mode = "SELF"; params.cs_mode = cs_mode;
ts_mode = "NON"; params.ts_mode = ts_mode;
ts_mom_mode = "NON"; params.ts_mom_mode = ts_mom_mode;
ts_eq_mode = "NON"; params.ts_eq_mode = ts_eq_mode; 
if_solve = true; params.if_solve = if_solve;

%% fmincon options
options = optimoptions('fmincon',...
                        'Display', 'iter',...
                        'Algorithm', 'sqp',...
                        'MaxIterations', 1e3,... 
                        'MaxFunctionEvaluations', 1e4,...
                        'SpecifyConstraintGradient', true,...
                        'SpecifyObjectiveGradient', true,...
                        'CheckGradients', false);

%% pendulum parameters
N = 30;
dt = 0.1;
umax = 5;
kappa = 2;

terminal_coeff = 1;
if_final_state = false;

%% start simulation
subopt_gap_list = [];
u_list = [];
th_list = [];
dth_list = [];
obj_list = [];

th = 0.1;
dth = 0.0; 
th_list = [th_list, th];
dth_list = [dth_list, dth];

for iter = 1: 100
    env = P_env(N=N, dt=dt,...
                umax=umax,...
                th_init=th, dth_init=dth,...
                terminal_coeff=terminal_coeff,...
                if_final_state=if_final_state);
    [primal_info, dual_info, aux_info_pendulum] = P_lascsbe(env);
    f = dual_info.f(1);
    g = dual_info.g;
    h = dual_info.h;
    v = dual_info.x;
    params.cliques = dual_info.cliques;
    
    %% run CSTSS
    [result, res, coeff_info, aux_info] = CSTSS_mex(f, g, h, kappa, v, params);
    aux_info.result = result;
    params.aux_info = aux_info;
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
        total_var_num = length(v);
        [vopt, output_info_robust] = robust_extract_CS(Xs, mom_mat_rpt, total_var_num, 1e-2);
        vopt(aux_info_pendulum.var_ids.vec_u_id) = vopt(aux_info_pendulum.var_ids.vec_u_id) * env.umax;
        uopt = vopt(aux_info_pendulum.var_ids.vec_u_id);
        xopt = vopt(aux_info_pendulum.var_ids.vec_x_id);
        xopt = reshape(xopt, env.x_dim, []);
        xopt = xopt(:, 1: env.N);
        % get_debug_img(xopt, uopt, "");
    end
    
    [vfea, objval, fmincon_info] = P_fmincon(vopt, env, aux_info_pendulum.var_ids, options);
    vfea(aux_info_pendulum.var_ids.vec_u_id) = vfea(aux_info_pendulum.var_ids.vec_u_id) * env.umax;
    ufea = vfea(aux_info_pendulum.var_ids.vec_u_id);
    xfea = vfea(aux_info_pendulum.var_ids.vec_x_id);
    xfea = reshape(xfea, env.x_dim, []);
    xfea = xfea(:, 1: env.N);

    %%% test open loop control %%%
    [x_seq_opt, cost_opt] = open_loop_control(uopt, env, primal_info, dual_info, aux_info_pendulum);
    [x_seq_fea, cost_fea] = open_loop_control(ufea, env, primal_info, dual_info, aux_info_pendulum);
    
    % since we are using Mosek, there is no need to use modified dual certificate
    subopt_gap = abs(objval - result) / (1 + abs(objval) + abs(result));
    
    % simulate
    u = ufea(1);
    [th, dth] = simulate(u, dt, env);
    
    u_list = [u_list, u];
    th_list = [th_list, th];
    dth_list = [dth_list, dth];
    subopt_gap_list = [subopt_gap_list, subopt_gap];
    obj_list = [obj_list, objval];

    if abs(th - pi) < 0.1 || abs(th + pi) < 0.1
        fprintf("Pendulum convergent! \n");
        break;
    end
end









%% helper functions
function env = P_env(opinions)
    arguments
        opinions.N = 10
        opinions.dt = 0.1
        opinions.th_init = 0.1
        opinions.dth_init = 0
        opinions.m = 1
        opinions.l = 1
        opinions.b = 0.1
        opinions.g = 9.8
        opinions.umax = 1.8
        opinions.fc_max = 0.5
        opinions.terminal_coeff = 1
        opinions.if_final_state = false
        opinions.x_final_rtwo = [pi; 0]
        opinions.if_scale = true
        opinions.tol = 1e-12
        opinions.if_fmincon = false
    end

    env.N = opinions.N;
    env.dt = opinions.dt;
    env.th_init = opinions.th_init;
    env.dth_init = opinions.dth_init;
    env.x_init_rtwo = [env.th_init; env.dth_init];
    env.m = opinions.m;
    env.l = opinions.l;
    env.b = opinions.b;
    env.g = opinions.g;
    env.umax = opinions.umax;
    env.fc_max = opinions.fc_max;
    env.terminal_coeff = opinions.terminal_coeff;
    env.if_final_state = opinions.if_final_state;
    env.x_final_rtwo = opinions.x_final_rtwo;

    % convert x_rtwo to x_sotwo
    env.x_init_sotwo = [
        cos(env.x_init_rtwo(1));
        sin(env.x_init_rtwo(1));
        cos(env.x_init_rtwo(2) * env.dt);
        sin(env.x_init_rtwo(2) * env.dt);
    ];
    env.x_final_sotwo = [
        cos(env.x_final_rtwo(1));
        sin(env.x_final_rtwo(1));
        cos(env.x_final_rtwo(2) * env.dt);
        sin(env.x_final_rtwo(2) * env.dt);
    ];
    env.th_final = env.x_final_rtwo(1);
    env.dth_final = env.x_final_rtwo(2);

    env.x_dim = 4;
    env.u_dim = 1;

    env.if_scale = opinions.if_scale;
    env.tol = opinions.tol;
    env.if_fmincon = opinions.if_fmincon;
    env.eps_relax_rc = 0.1;
    env.eps_relax_rs = 0.1;

    env.u_penalty = 1;
end

function func_collection = liegroup_implicitbe_grad()
    % func_collection will contain three functions, 
    % with each returning eq, ineq, eq_grad, ineq_grad
    % func_collection.first: for the first clique
    % func_collection.middle: for the 2 ~ (N-1)'th clques
    % func_collection.last: for the last clique
    func_collection.first = @get_middle_clique;
    func_collection.middle = @get_middle_clique; 
    func_collection.last = @get_middle_clique;
end

function [eq_dyn, eq_dyn_grad] = get_middle_clique(v, k, env, var_ids)
    dt = env.dt; m = env.m; b = env.b; l = env.l; g = env.g;
    u_k_id = var_ids.u_id(k); u_k = v(u_k_id); 
    rc_k_id = var_ids.rc_id(k-1); rc_k = v(rc_k_id);
    rs_k_id = var_ids.rs_id(k-1); rs_k = v(rs_k_id);
    fc_k_id = var_ids.fc_id(k-1); fc_k = v(fc_k_id);
    fs_k_id = var_ids.fs_id(k-1); fs_k = v(fs_k_id);
    rc_kp1_id = var_ids.rc_id(k); rc_kp1 = v(rc_kp1_id);
    rs_kp1_id = var_ids.rs_id(k); rs_kp1 = v(rs_kp1_id);
    fc_kp1_id = var_ids.fc_id(k); fc_kp1 = v(fc_kp1_id);
    fs_kp1_id = var_ids.fs_id(k); fs_kp1 = v(fs_kp1_id);
 
    eq_dyn = [
        1/dt * m*l^2 * (fs_kp1 - fs_k) + dt * m*g*l * rs_kp1 + b * fs_kp1 - dt * u_k * env.umax;
        rc_kp1 - (rc_k * fc_k - rs_k * fs_k);
        rs_kp1 - (rs_k * fc_k + rc_k * fs_k);
        rc_kp1^2 + rs_kp1^2 - 1;
        fc_kp1^2 + fs_kp1^2 - 1;
    ];

    if nargout == 2
        eq_dyn_grad = cell(5, 1);
        % eq_dyn_grad = cell(4, 1);
        eq_dyn_grad{1} = [
            [u_k_id, -dt*env.umax];
            [fs_k_id, -(l^2*m)/dt];
            [rs_kp1_id, dt*g*l*m];
            [fs_kp1_id, b + (l^2*m)/dt];
        ];
        eq_dyn_grad{2} = [
            [rc_k_id, -fc_k];
            [rs_k_id, fs_k];
            [fc_k_id, -rc_k];
            [fs_k_id, rs_k];
            [rc_kp1_id, 1];
        ];
        eq_dyn_grad{3} = [
            [rc_k_id, -fs_k];
            [rs_k_id, -fc_k];
            [fc_k_id, -rs_k];
            [fs_k_id, -rc_k];
            [rs_kp1_id, 1];
        ];
        eq_dyn_grad{4} = [
            [rc_kp1_id, 2*rc_kp1];
            [rs_kp1_id, 2*rs_kp1];
        ];
        eq_dyn_grad{5} = [
            [fc_kp1_id, 2*fc_kp1];
            [fs_kp1_id, 2*fs_kp1];
        ];
        % eq_dyn_grad{4} = [
        %     [fc_kp1_id, 2*fc_kp1];
        %     [fs_kp1_id, 2*fs_kp1];
        % ];
    end
end

function [eq_dyn, eq_dyn_grad] = get_last_clique(v, k, env, var_ids)
    dt = env.dt; m = env.m; b = env.b; l = env.l; g = env.g;
    u_k_id = var_ids.u_id(k); u_k = v(u_k_id); 
    rc_k_id = var_ids.rc_id(k-1); rc_k = v(rc_k_id);
    rs_k_id = var_ids.rs_id(k-1); rs_k = v(rs_k_id);
    fc_k_id = var_ids.fc_id(k-1); fc_k = v(fc_k_id);
    fs_k_id = var_ids.fs_id(k-1); fs_k = v(fs_k_id);
    rc_kp1_id = var_ids.rc_id(k); rc_kp1 = v(rc_kp1_id);
    rs_kp1_id = var_ids.rs_id(k); rs_kp1 = v(rs_kp1_id);
    fc_kp1_id = var_ids.fc_id(k); fc_kp1 = v(fc_kp1_id);
    fs_kp1_id = var_ids.fs_id(k); fs_kp1 = v(fs_kp1_id);
 
    eq_dyn = [
        1/dt * m*l^2 * (fs_kp1 - fs_k) + dt * m*g*l * rs_kp1 + b * fs_kp1 - dt * u_k * env.umax;
        rc_kp1 - (rc_k * fc_k - rs_k * fs_k);
        rs_kp1 - (rs_k * fc_k + rc_k * fs_k);
    ];

    if nargout == 2
        eq_dyn_grad = cell(3, 1);
        eq_dyn_grad{1} = [
            [u_k_id, -dt * env.umax];
            [fs_k_id, -(l^2*m)/dt];
            [rs_kp1_id, dt*g*l*m];
            [fs_kp1_id, b + (l^2*m)/dt];
        ];
        eq_dyn_grad{2} = [
            [rc_k_id, -fc_k];
            [rs_k_id, fs_k];
            [fc_k_id, -rc_k];
            [fs_k_id, rs_k];
            [rc_kp1_id, 1];
        ];
        eq_dyn_grad{3} = [
            [rc_k_id, -fs_k];
            [rs_k_id, -fc_k];
            [fc_k_id, -rs_k];
            [fs_k_id, -rc_k];
            [rs_kp1_id, 1];
        ];
    end
end

function [loss, loss_grad] = P_get_unit_state_loss(v, k, env, var_ids, obj_scale_factor)
    % here k represents vertex if: 0 ~ N
    dt = env.dt;
    rc_k_id = var_ids.rc_id(k); rs_k_id = var_ids.rs_id(k);
    fc_k_id = var_ids.fc_id(k); fs_k_id = var_ids.fs_id(k);
    rc_k = v(rc_k_id); rs_k = v(rs_k_id);
    fc_k = v(fc_k_id); fs_k = v(fs_k_id);
    % loss = dt * (rc + 1)^2 + dt * rs^2 + 1/dt * (fc - 1)^2 + 1/dt * fs^2;
    loss = (rc_k + 1)^2 + rs_k^2 + (fc_k - 1)^2 + fs_k^2;
    loss = loss / obj_scale_factor;

    if nargout == 2
        loss_grad = [
            [rc_k_id, (2*rc_k + 2) / obj_scale_factor];
            [rs_k_id, 2*rs_k / obj_scale_factor];
            [fc_k_id, (2*fc_k - 2) / obj_scale_factor];
            [fs_k_id, 2*fs_k / obj_scale_factor];
        ];
    end
end


function [primal_info, dual_info, aux_info] = P_lascsbe(env)
    % primal info: return vars, objective, equality, inequality, z of sparse_sdp_relax.m
    % dual info: return a, f, g, h, x, cliques, I, J of sparsesos.m
    dyn_func = liegroup_implicitbe_grad;

    N = env.N; dt = env.dt; 
    if_scale = env.if_scale; tol = env.tol;
    x_init = env.x_init_sotwo; 
    x_final = env.x_final_sotwo;
    var_ids = get_var_ids(env);

    primal_info.vars = {};
    primal_info.inequality = {};
    primal_info.equality = {};
    dual_info.g = [];
    dual_info.h = [];
    dual_info.cliques = {};
    dual_info.I = {};
    dual_info.J = {};

    v = msspoly('v', var_ids.d); dual_info.x = v;

    for k = 1: N 
        [clique_ids, ~] = get_clique_info(k, env, var_ids); 
        dual_info.cliques{k} = clique_ids; primal_info.vars{k} = v(clique_ids);
        x_km1 = v(var_ids.x_id(k-1));
        u_km1 = v(var_ids.u_id(k));
        x_k = v(var_ids.x_id(k));
        rc_km1_id = var_ids.rc_id(k-1); rc_km1 = v(rc_km1_id);
        rs_km1_id = var_ids.rs_id(k-1); rs_km1 = v(rs_km1_id);
        fc_km1_id = var_ids.fc_id(k-1); fc_km1 = v(fc_km1_id);
        fs_km1_id = var_ids.fs_id(k-1); fs_km1 = v(fs_km1_id);
        rc_k_id = var_ids.rc_id(k); rc_k = v(rc_k_id);
        rs_k_id = var_ids.rs_id(k); rs_k = v(rs_k_id);
        fc_k_id = var_ids.fc_id(k); fc_k = v(fc_k_id);
        fs_k_id = var_ids.fs_id(k); fs_k = v(fs_k_id);
        if k == 1
            eq_dyn = dyn_func.first(v, k, env, var_ids);
            primal_info.equality{k} = [
                x_km1 - x_init;
                eq_dyn;
            ];
            primal_info.inequality{k} = [
                1 - u_km1^2;
                fc_k - env.fc_max;
            ];
        elseif k == N
            eq_dyn = dyn_func.last(v, k, env, var_ids);
            if env.if_final_state
                primal_info.equality{k} = [
                    x_k - x_final;
                    eq_dyn;
                    rc_km1^2 + rs_km1^2 - 1;
                    fc_km1^2 + fs_km1^2 - 1;
                ];
                primal_info.inequality{k} = [
                    % 1 - (x_k - x_final)' * (x_k - x_final);
                    1 - u_km1^2;
                ];
                % fprintf("\n Relax terminal constraint! \n");
            else
                primal_info.equality{k} = [
                    eq_dyn;
                    rc_km1^2 + rs_km1^2 - 1;
                    fc_km1^2 + fs_km1^2 - 1;
                ];
                primal_info.inequality{k} = [
                    1 - u_km1^2;
                    fc_k - env.fc_max;
                ];
            end  
            
        else
            eq_dyn = dyn_func.middle(v, k, env, var_ids);
            primal_info.equality{k} = [
                eq_dyn;
                rc_km1^2 + rs_km1^2 - 1;
                fc_km1^2 + fs_km1^2 - 1;
            ];
            primal_info.inequality{k} = [
                1 - u_km1^2;
                fc_k - env.fc_max;
            ];
        end
    end

    % from primal_info.equality and primal_info.inequality construct dual_info.h and dual_info.g,
    % and dual_info.J and dual_info.I
    h_num = 0;
    g_num = 0;
    for k = 1: N
        [primal_info.equality{k}, ~] = msspoly_clean(primal_info.equality{k}, v, tol, if_scale);
        [primal_info.inequality{k}, ~] = msspoly_clean(primal_info.inequality{k}, v, tol, if_scale);
        h_num_in_clique = length(primal_info.equality{k});
        g_num_in_clique = length(primal_info.inequality{k});
        dual_info.J{k} = h_num + (1: h_num_in_clique);
        dual_info.I{k} = g_num + (1: g_num_in_clique);
        h_num = h_num + h_num_in_clique;
        g_num = g_num + g_num_in_clique;
        dual_info.h = [dual_info.h; primal_info.equality{k}];
        dual_info.g = [dual_info.g; primal_info.inequality{k}];
    end

    %% get obj and obj_homo
    obj = 0; 
    obj_clique = msspoly();
    dt = env.dt;
    for k = 1: N
        u_km1 = v(var_ids.u_id(k)); 
        x_km1 = v(var_ids.x_id(k-1)); 
        x_k = v(var_ids.x_id(k));
        if k == N
            obj_in_each_clique = env.u_penalty * u_km1^2 + P_get_unit_state_loss(v, k-1, env, var_ids, 1)...
                                + P_get_unit_state_loss(v, k, env, var_ids, 1 / env.terminal_coeff);
        else
            obj_in_each_clique = env.u_penalty * u_km1^2 + P_get_unit_state_loss(v, k-1, env, var_ids, 1);
        end
        obj_clique = [obj_clique; obj_in_each_clique];
        obj = obj + obj_in_each_clique; 
    end
    % carefully scale obj_clique: each clique should share the same scaling factor!
    [obj, obj_scale_factor] = msspoly_clean(obj, v, tol, false);
    for k = 1: N
        [obj_clique(k), ~] = msspoly_clean(obj_clique(k) / obj_scale_factor, v, tol, false);
    end

    % get primal_info.objective, dual_info.a, and dual_info.f
    primal_info.objective = obj_clique;
    dual_info.a = -1;
    dual_info.f = [obj; -1];
    primal_info.rip_predecessor = 0: (N-1);

    % set aux_info
    aux_info.obj = obj;
    aux_info.var_ids = var_ids;
    aux_info.get_clique_info = @get_clique_info;
    aux_info.v = v;
    aux_info.var_x = v(var_ids.vec_x_id);
    aux_info.var_u = v(var_ids.vec_u_id);
    aux_info.obj_scale_factor = obj_scale_factor;
    aux_info.extract_solution = @extract_solution;
    aux_info.var_str_mapping = get_var_str_mapping(v, env, var_ids);
end




%% get variable ids
function var_ids = get_var_ids(env)
    % layout of variables
    % [u_0, u_1, ..., u_{N-1},
    %  x_0, x_1, ..., x_{N-1}, x_N]
    % where x_k = [rc_k, rs_k, fc_k, fs_k]
    N = env.N;
    var_ids.du = N; var_ids.dx = (N+1) * env.x_dim; 
    var_ids.d = var_ids.du + var_ids.dx;
    var_ids.u_id = @(k) k; % k is the clique id, k = 1 ... p
    var_ids.x_id = @(k) (var_ids.du + env.x_dim * k + (1:env.x_dim))'; % k is the clique id, k = 0 ... p

    % relative ids in x: [rc, rs, fc, fs]
    var_ids.rc_xid = 1;
    var_ids.rs_xid = 2;
    var_ids.fc_xid = 3;
    var_ids.fs_xid = 4;

    % for convenience, get x id quickly
    var_ids.rc_id = @(k) var_ids.du + env.x_dim * k + var_ids.rc_xid;
    var_ids.rs_id = @(k) var_ids.du + env.x_dim * k + var_ids.rs_xid;
    var_ids.fc_id = @(k) var_ids.du + env.x_dim * k + var_ids.fc_xid;
    var_ids.fs_id = @(k) var_ids.du + env.x_dim * k + var_ids.fs_xid;

    % vectorized u, x, s ids
    var_ids.vec_u_id = []; 
    var_ids.vec_x_id = []; 
    N = env.N;
    for k = 1: N
        var_ids.vec_u_id = [var_ids.vec_u_id; var_ids.u_id(k)];
    end
    for k = 0: N
        var_ids.vec_x_id = [var_ids.vec_x_id; var_ids.x_id(k)];
    end
end

%% get variable-string mapping for tex record
function var_str_mapping = get_var_str_mapping(v, env, var_ids)
    N = env.N;
    total_length = length(v);
    var_str_mapping = strings(total_length, 1);

    % k is the clique id, in the k'th clique, the subscript of u is (k-1)
    u_str = @(k) sprintf("u_{%d}", k); 
    x_str = @(k) [sprintf("r_{c, %d}", k); sprintf("r_{s, %d}", k);
                  sprintf("f_{c, %d}", k); sprintf("f_{s, %d}", k)];

    for k = 1: N
        var_str_mapping(var_ids.u_id(k)) = u_str(k-1);
    end
    for k = 0: N
        var_str_mapping(var_ids.x_id(k)) = x_str(k);
    end
end

%% get ids, vars, and their relative positions in each clique
function [clique_ids, clique_relpos] = get_clique_info(k, env, var_ids)
    x_km1_id = var_ids.x_id(k-1); x_k_id = var_ids.x_id(k);
    u_km1_id = var_ids.u_id(k);

    % u -> x
    N = env.N;
    clique_ids = [u_km1_id; x_km1_id; x_k_id];

    % get relative pos
    count = 0;
    clique_relpos.u_km1 = count + 1; count = count + 1; 
    clique_relpos.x_km1 = count + (1:env.x_dim); count = count + env.x_dim;
    clique_relpos.x_k = count + (1:env.x_dim); count = count + env.x_dim;
end

%% extract solution
function [uopt, xopt, vopt] = extract_solution(Xmoms, get_clique_info_func, env, var_ids)
    uopt = []; xopt = []; 
    for k = 1: env.N
        X = Xmoms{k}; 
        [clique_ids, clique_relpos] = get_clique_info_func(k, env, var_ids);
        u_km1_relpos = clique_relpos.u_km1;
        x_km1_relpos = clique_relpos.x_km1;
        x_k_relpos = clique_relpos.x_k;
        if k == 1
            X_small_index = [1, 1+u_km1_relpos, 1+x_km1_relpos, 1+x_k_relpos];
            X_small = X(X_small_index, X_small_index);
            [U, D] = sorteig(X_small);
            m = U(:, 1); m = m / m(1);
            count = 2;
            u_km1 = m(count); u_km1 = u_km1 * env.umax;
            uopt = [uopt; u_km1]; count = count + 1;
            x_km1 = m(count: count + env.x_dim - 1); xopt = x_km1; count = count + env.x_dim;
            x_k = m(count: count + env.x_dim - 1); xopt = [xopt, x_k];
        else
            X_small_index = [1, 1+u_km1_relpos, 1+x_k_relpos];
            X_small = X(X_small_index, X_small_index);
            [U, D] = sorteig(X_small);
            m = U(:, 1); m = m / m(1);
            count = 2;
            u_km1 = m(count); u_km1 = u_km1 * env.umax;
            uopt = [uopt; u_km1]; count = count + 1;
            x_k = m(count: count + env.x_dim - 1); xopt = [xopt, x_k];
        end
    end

    vopt = [
        uopt;
        reshape(xopt, [], 1);
    ];
end


function v = from_cell_to_array(c)
    v = [];
    for i = 1: length(c)
        v = [v; c{i}];
    end
end

function X_vec = purevec_to_vec(blk, X_purevec)
    X_vec = {};
    idx = 0;
    for i = 1: size(blk, 1)
        mat_size = blk{i, 2};
        vec_size = mat_size * (mat_size + 1) / 2;
        X_vec{i} = X_purevec((idx+1): (idx+vec_size));
        idx = idx+vec_size;
    end
end

function v0 = SDP_to_POP(X, env, primal_info, dual_info, aux_info, if_fmincon)
    N = env.N;
    var_ids = aux_info.var_ids;
    get_clique_info = aux_info.get_clique_info;
    Xmoms = [];
    idx = 1;
    for i = 1: env.N
        num_loc_per_clique = length(primal_info.inequality{i});
        Xmoms = [Xmoms; {X{idx}}];
        idx = idx + num_loc_per_clique + 1;
    end
    v_val = zeros(var_ids.d, 1);
    v_cnt = zeros(size(v_val));
    for k = 1: env.N
        [clique_ids, ~] = get_clique_info(k, env, var_ids);
        Xmom = Xmoms{k};
        [U, D] = sorteig(Xmom);
        D = diag(D);
        disp(D(1:7)');
        u = U(:, 1);
        clique_vars = u(2: length(clique_ids)+1) / u(1);
        for i = 1: length(clique_vars)
            val = clique_vars(i);
            pos = clique_ids(i);
            v_val(pos) = v_val(pos) + val;
            v_cnt(pos) = v_cnt(pos) + 1;
        end
    end
    v0 = v_val ./ v_cnt;
    if if_fmincon
        for k = 1: N
            v0(var_ids.u_id(k)) = v0(var_ids.u_id(k)) * env.umax;
        end
    end
end

function v0 = SDP_to_POP_feasible(X, env, primal_info, dual_info, aux_info, if_recover)
    N = env.N;
    var_ids = aux_info.var_ids;
    get_clique_info = aux_info.get_clique_info;
    Xmoms = [];
    idx = 1;
    for i = 1: env.N
        num_loc_per_clique = length(primal_info.inequality{i});
        Xmoms = [Xmoms; {X{idx}}];
        idx = idx + num_loc_per_clique + 1;
    end
    v_val = zeros(var_ids.d, 1);
    v_cnt = zeros(size(v_val));
    for k = 1: env.N
        [clique_ids, ~] = get_clique_info(k, env, var_ids);
        Xmom = Xmoms{k};
        [U, D] = sorteig(Xmom);
        D = diag(D);
        disp(D(1:7)');
        u = U(:, 1);
        clique_vars = u(2: length(clique_ids)+1) / u(1);
        for i = 1: length(clique_vars)
            val = clique_vars(i);
            pos = clique_ids(i);
            v_val(pos) = v_val(pos) + val;
            v_cnt(pos) = v_cnt(pos) + 1;
        end
    end
    % avoid NAN
    for i = 1: length(v_cnt)
        if v_cnt(i) == 0
            v_cnt(i) = 1;
        end
    end
    v0 = v_val ./ v_cnt;
    % recover a and u
    if if_recover
        for k = 1: N
            v0(var_ids.u_id(k)) = v0(var_ids.u_id(k)) * env.umax;
        end
    end

    % get a feasible trajectory
    v0(var_ids.x_id(0)) = env.x_init_sotwo;
    for k = 1: env.N
        u_km1 = v0(var_ids.u_id(k));
        % for numerical stability
        if u_km1 >= 1
            u_km1 = 1 - 1e-12;
        elseif u_km1 <= -1
            u_km1 = -1 + 1e-12;
        end
        v0(var_ids.u_id(k)) = u_km1;
        x_km1 = v0(var_ids.x_id(k-1));
        x_k = P_one_poly_step(u_km1, x_km1, env, var_ids);
        v0(var_ids.x_id(k)) = x_k;
    end
end

function get_debug_img(x, u, img_name)
    rc = x(1, :); rs = x(2, :);
    fc = x(3, :); fs = x(4, :);
    fig = figure;

    subplot(2, 2, 1);
    plot(rc, DisplayName="rc");
    hold on;
    plot(rs, DisplayName="rs");
    legend("location", "northoutside");
    title("angular position");

    subplot(2, 2, 2);
    plot(fc, DisplayName="fc");
    hold on;
    plot(fs, DisplayName="fs");
    title("angular velocity");
    legend("location", "northoutside");

    subplot(2, 2, 3);
    plot(rc.^2 + rs.^2, DisplayName="pos-err");
    hold on;
    plot(fc.^2 + fs.^2, DisplayName="vel-err");
    title("SO2 errors");
    legend("location", "northoutside");

    subplot(2, 2, 4);
    plot(u, DisplayName="u");
    title("control input");
    legend("location", "northoutside");

    % print("-dpng", "-r600", img_name);
    % close(fig);
end


function info_struct = info_cell_to_struct(info_cell)
    info_struct = struct();
    for i = 1: size(info_cell, 1)
        name = info_cell{i, 1};
        info_struct.(name) = info_cell{i, 2};
    end
end

function [vfea, objval, fmincon_info] = P_fmincon(v0, env, var_ids, options) 
    env.if_fmincon = true;
    dyn_func = liegroup_implicitbe_grad;
    var_ids.fmincon_d = var_ids.du + var_ids.dx;

    % A, b, Aeq, beq, lb, ub
    A = []; b = []; Aeq = []; beq = [];
    lb = zeros(size(v0)); ub = zeros(size(v0));
    for i = 1: length(v0)
        lb(i) = -1; ub(i) = 1;
    end

    % must add: (1) -umax <= u <= umax, -1 <= x <= 1
    % (2) fc >= fc_max 
    % (3) x_0 = x_init
    % (4) x_N = x_final (if env.if_final_state)
    N = env.N;
    % (1)
    for k = 1: N
        u_km1_id = var_ids.u_id(k);
        lb(u_km1_id) = -1; ub(u_km1_id) = 1;
    end
    % for k = 0: N
    %     rc_k_id = var_ids.rc_id(k); lb(rc_k_id) = -1; ub(rc_k_id) = 1;
    %     rs_k_id = var_ids.rs_id(k); lb(rs_k_id) = -1; ub(rs_k_id) = 1;
    %     fc_k_id = var_ids.fc_id(k); lb(fc_k_id) = -1; ub(fc_k_id) = 1;
    %     fs_k_id = var_ids.fs_id(k); lb(fs_k_id) = -1; ub(fs_k_id) = 1;
    % end
    % (2)
    for k = 0: N
        fc_k_id = var_ids.fc_id(k);
        if k == 0
            continue;
        elseif k == N
            if ~env.if_final_state
                lb(fc_k_id) = env.fc_max;
            end
        else
            lb(fc_k_id) = env.fc_max;
        end
    end
    x_init = env.x_init_sotwo; 
    x_final = env.x_final_sotwo;
    % (3)
    lb(var_ids.rc_id(0)) = x_init(var_ids.rc_xid); ub(var_ids.rc_id(0)) = x_init(var_ids.rc_xid); 
    lb(var_ids.rs_id(0)) = x_init(var_ids.rs_xid); ub(var_ids.rs_id(0)) = x_init(var_ids.rs_xid); 
    lb(var_ids.fc_id(0)) = x_init(var_ids.fc_xid); ub(var_ids.fc_id(0)) = x_init(var_ids.fc_xid); 
    lb(var_ids.fs_id(0)) = x_init(var_ids.fs_xid); ub(var_ids.fs_id(0)) = x_init(var_ids.fs_xid); 
    % (4)
    if env.if_final_state
        lb(var_ids.rc_id(N)) = x_final(var_ids.rc_xid); ub(var_ids.rc_id(N)) = x_final(var_ids.rc_xid); 
        lb(var_ids.rs_id(N)) = x_final(var_ids.rs_xid); ub(var_ids.rs_id(N)) = x_final(var_ids.rs_xid); 
        lb(var_ids.fc_id(N)) = x_final(var_ids.fc_xid); ub(var_ids.fc_id(N)) = x_final(var_ids.fc_xid); 
        lb(var_ids.fs_id(N)) = x_final(var_ids.fs_xid); ub(var_ids.fs_id(N)) = x_final(var_ids.fs_xid); 
    end

    v1 = max(lb, min(ub, v0));

    % get c and ceq
    nonlcon = @(v) get_nonlinear_con(v, env, var_ids, dyn_func);
    % get objective function
    fun = @(v) get_objective(v, env, var_ids);
    [vfea, objval, exitflag, output] = fmincon(fun, v1, A, b, Aeq, beq, lb, ub, nonlcon, options);

    % get fmincon_info
    disp(output);
    fmincon_info.get_nonlinear_con_func = @get_nonlinear_con;
    fmincon_info.get_objective_func = @get_objective;
    fmincon_info.output = output;
    fmincon_info.constrviolation = output.constrviolation;
    fmincon_info.firstorderopt = output.firstorderopt;
end

function [c, ceq, c_grad, ceq_grad] = get_nonlinear_con(v, env, var_ids, dyn_func)
    N = env.N;

    if nargout == 2
        % equality constraints
        ceq = [];
        for k = 1: N
            if k == 1
                eq_dyn = dyn_func.first(v, k, env, var_ids); 
            elseif k == N
                eq_dyn = dyn_func.last(v, k, env, var_ids); 
            else
                eq_dyn = dyn_func.middle(v, k, env, var_ids); 
            end
            ceq = [ceq; eq_dyn];
        end
        % inequality constraints: None!
        c = [];

    % include gradient
    elseif nargout == 4
        % equality constraints
        ceq = [];
        ceq_sparse_grad = [];
        for k = 1: N
            if k == 1
                [eq_dyn, tmp_sparse_grad] = dyn_func.first(v, k, env, var_ids); 
            elseif k == N
                [eq_dyn, tmp_sparse_grad] = dyn_func.last(v, k, env, var_ids); 
            else
                [eq_dyn, tmp_sparse_grad] = dyn_func.middle(v, k, env, var_ids); 
            end
            ceq = [ceq; eq_dyn];
            ceq_sparse_grad = [ceq_sparse_grad; tmp_sparse_grad];
        end
        % construct ceq_grad
        rows = []; cols = []; vals = [];
        for ii = 1: size(ceq, 1)
            pack = ceq_sparse_grad{ii};
            rows = [rows, pack(:, 1)']; 
            cols = [cols, ii * ones(1, size(pack, 1))];
            vals = [vals, pack(:, 2)'];
        end
        ceq_grad = sparse(rows, cols, vals, var_ids.fmincon_d, size(ceq, 1));
        % disp(svd(full(ceq_grad)));
        % inequality constraints: None!
        c = [];
        c_grad = [];
    end
end

function [obj, obj_grad] = get_objective(v, env, var_ids)
    obj = 0;
    N = env.N; dt = env.dt;

    % here k represesnts vertex id: 0 ~ N
    for k = 1: N
        u_km1 = v(var_ids.u_id(k)); 
        if k == N
            obj = obj + u_km1^2 * env.u_penalty + P_get_unit_state_loss(v, k-1, env, var_ids, 1)...
                                + P_get_unit_state_loss(v, k, env, var_ids, 1/env.terminal_coeff);
        else
            obj = obj + u_km1^2 * env.u_penalty + P_get_unit_state_loss(v, k-1, env, var_ids, 1);
        end
    end

    % include gradient
    if nargout == 2
        sparse_grad = [];
        for k = 1: N
            u_km1_id = var_ids.u_id(k);
            u_km1 = v(u_km1_id);
            sparse_grad = [
                sparse_grad;
                [u_km1_id, 2*u_km1 * env.u_penalty];
            ];
        end
        for k = 0: N
            if k == N
                [~, loss_grad] = P_get_unit_state_loss(v, k, env, var_ids, 1/env.terminal_coeff);
            else
                [~, loss_grad] = P_get_unit_state_loss(v, k, env, var_ids, 1);
            end
            sparse_grad = [
                sparse_grad;
                loss_grad;
            ];
        end
        num = size(sparse_grad, 1);
        obj_grad = sparse(sparse_grad(:, 1)', ones(1, num), sparse_grad(:, 2)', var_ids.fmincon_d, 1);
    end
end

function [th_new, dth_new] = simulate(u, dt, env, resolution)
    if nargin == 4
        resolution = resolution;
    else
        resolution = 1e-5;
    end

    step = round(dt / resolution);
    th = env.th_init;
    dth = env.dth_init;
    l = env.l;
    m = env.m;
    b = env.b;
    g = env.g;
    for i = 1: step 
        ddth = 1/(m*l^2) * (-m*g*l * sin(th) - b * dth + u);
        th = th + dth * resolution;
        dth = dth + ddth * resolution;
    end
    if th > pi 
        th = th - 2*pi;
    elseif th < -pi
        th = th + 2*pi;
    end
    th_new = th;
    dth_new = dth;
end

function x_kp1 = P_one_poly_step(u_k, x_k, env, var_ids)
    opts = optimoptions(@fsolve, 'Algorithm', 'levenberg-marquardt', 'Display', 'none');
    x_kp1_0 = x_k; % initial guess
    func = @(x_kp1) one_step_simulate(x_kp1, x_k, u_k, env, var_ids);
    x_kp1 = fsolve(func, x_kp1_0, opts);
end


function F = one_step_simulate(x_kp1, x_k, u_k, env, var_ids)
    rc_k = x_k(var_ids.rc_xid); rs_k = x_k(var_ids.rs_xid);
    fc_k = x_k(var_ids.fc_xid); fs_k = x_k(var_ids.fs_xid);
    rc_kp1 = x_kp1(var_ids.rc_xid); rs_kp1 = x_kp1(var_ids.rs_xid);
    fc_kp1 = x_kp1(var_ids.fc_xid); fs_kp1 = x_kp1(var_ids.fs_xid);
    
    F = zeros(5, 1);
    dt = env.dt; m = env.m; b = env.b; l = env.l; g = env.g;
    F(1) = 1/dt * m*l^2 * (fs_kp1 - fs_k) + dt * m*g*l * rs_kp1 + b * fs_kp1 - dt * u_k * env.umax;
    F(2) = rc_kp1 - (rc_k * fc_k - rs_k * fs_k);
    F(3) = rs_kp1 - (rs_k * fc_k + rc_k * fs_k);
    F(4) = rc_kp1^2 + rs_kp1^2 - 1;
    F(5) = fc_kp1^2 + fs_kp1^2 - 1;
end

function [x_seq, cost] = open_loop_control(u_seq, env, primal_info, dual_info, aux_info)
    x_seq = [cos(env.th_init); sin(env.th_init); cos(env.dt * env.dth_init); sin(env.dt * env.dth_init)];
    x = x_seq;
    assert(length(u_seq) == env.N, "env.N should be equal to length(u_seq)! \n");

    for k = 1: env.N
        u = u_seq(k);
        u = u / env.umax; % rescale back u!
        x_new = P_one_poly_step(u, x, env, aux_info.var_ids);
        x_seq = [x_seq, x_new];
        x = x_new;
    end

    x_seq_vec = x_seq(:);
    v_seq = zeros(size(dual_info.x));
    v_seq(aux_info.var_ids.vec_u_id) = u_seq / env.umax;
    v_seq(aux_info.var_ids.vec_x_id) = x_seq_vec;
    
    cost = double( subs(dual_info.f(1), dual_info.x, v_seq) );
end









