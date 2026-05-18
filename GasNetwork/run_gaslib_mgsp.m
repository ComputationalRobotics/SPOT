function out = run_gaslib_mgsp(net_file, scn_file, scn_id, opts)
%RUN_GASLIB_MGSP  Parse GasLib XML and solve the weighted-shedding NLP.
%
% This generalizes the toy MGSP script to GasLib .net/.scn files.
% It keeps the same basic optimization structure:
%   - minimize weighted load shedding at exits
%   - nodal mass balance
%   - bidirectional Weymouth on pipes using q*|q|
%   - short pipes enforce equal pressure
%   - simplified compressor constraints using ratio bounds from the .net file
%
% Important modeling assumptions in this file:
%   1) Pipe coefficient uses the corrected stationary isothermal form
%        K = 16 * lambda * Rs * z * T * L / (pi^2 * D^5)
%      with lambda = (2*log10(D/eps) + 1.138)^(-2).
%   2) Scenario pressure bounds are intersected with .net pressure bounds.
%   3) Entry lower bounds are relaxed to 0 <= supply <= cap, matching the
%      behavior of the Python parser/model you posted.
%   4) Supported arc types: pipe, shortPipe, compressorStation.
%      The script errors out on resistor / valve / controlValve because the
%      toy NLP does not contain the switching / pressure-loss logic needed.
%   5) Compressor stations are modeled only by simplified pressure-ratio
%      bounds from the .net file (not the full .cs machine model).
%   6) Nodal pressures are **absolute bar**. Weymouth uses the equivalent form
%      (pb_i^2 - pb_j^2) - (res_factor/Pa2_per_bar2)*q*q_abs = 0 so POP coeffs
%      avoid an explicit 1e10 scale (Pa2_per_bar2 = (1e5)^2); compressors use
%      ratio constraints on pb^2 directly (divide Pa^2 inequalities by Pa2>0).
%   7) NLP and POP use hat variables with x_phys(i)=opts.x_scale(i)*x_hat(i); see
%      opts.x_scale (auto from ub/2 or pop_ball_c_inf/2 for infinite ub).
%
% Usage:
%   out = run_gaslib_mgsp();
%   out = run_gaslib_mgsp('data/GasLib-40-v1-20211130.net', ...
%                         'data/GasLib-40-v1-20211130.scn', ...
%                         'nomination_1');
%
% Optional fields in opts:
%   opts.z                         scalar compressibility factor (default 1.0)
%   opts.passive_bidirectional     true/false, make pipe/short bounds symmetric
%                                  around zero (default true)
%   opts.enforce_compressor_forward true/false (default true)
%   opts.weight_default            default shedding weight (default 1)
%   opts.weights                   struct of per-node weights, e.g.
%                                  opts.weights.sink_7 = 3;
%   opts.run_sdp                   true/false, default false
%   opts.kappa                     SDP relaxation order (default 2)
%   opts.cs_mode                   correlative sparsity for CSTSS, e.g. 'MF','NON'
%                                  (default 'MF'). If the moment SDP is primal
%                                  infeasible, try 'NON' (single clique, larger SDP).
%   opts.pop_feas_tol              tolerance for POP feasibility check on NLP x0
%                                  vs CSTSS convention g>=0, h=0 (default 1e-7)
%   opts.report_pop_coeff_ranges   if true and msspoly is on path, report min/max
%                                  of coefficients in f, h, and g split by
%                                  constraint group (pressure/flow/qabs/supply/
%                                  shed bounds, compressor, ball); supp_rpt uses
%                                  degree cap 2*kappa (same as CSTSS_mex); default false
%   opts.pop_ball_constraints      if true, append per-variable SOS ball g_i:
%                                  c_i^2 - z(i)^2 >= 0 with c_i = ub(i) if finite
%                                  else opts.pop_ball_c_inf (default false)
%   opts.pop_ball_c_inf            radius c when ub(i) is not finite (default 100)
%   opts.x_scale                   per-variable diagonal scales (length nvar); if
%                                  empty (default), filled as ub(i)/2 when ub(i)
%                                  is finite, else pop_ball_c_inf/2 when ub(i) is not
%   opts.fmincon_options           custom fmincon options
%   opts.refine_robust_extract     if true (default), after SDP robust_extract_CS,
%                                  run hat-space fmincon from v_opt_robust (same
%                                  stack as run_gaslib_mgsp: obj_hat, Aeq_hat, bounds_hat)
%   opts.verify_nlp_pop_hat        if true (default false), sample random hat vectors
%                                  near x0_hat and compare NLP (fmincon stack) vs
%                                  gaslib_pop_numeric_fgh objective and equality stack
%   opts.verify_nlp_pop_hat_samples   number of random hat samples (default 5)
%   opts.verify_nlp_pop_hat_noise     std dev of Gaussian perturbation of x0_hat (default 0.05)
%   opts.verify_nlp_pop_hat_tol       pass tolerance on |df|, |dh|, |dg| (default 1e-8)
%   opts.verify_nlp_pop_hat_seed      RNG seed; empty (default) uses rng('shuffle')
%   opts.verbose                   true/false (default true)
%
% Output:
%   out.data      parsed GasLib data and variable maps
%   out.nlp       NLP solution / diagnostics (includes max_bound_violation,
%                 pop_x0, x_scale, x_hat (scaled NLP vars), and pop_coeff if
%                 opts.report_pop_coeff_ranges is true; nlp.x is physical units.
%                 pop_coeff.ball_c is per-var SOS radius c when pop_ball_constraints.
%                 nlp.verify_nlp_pop_hat (if opts.verify_nlp_pop_hat) holds random-check results.
%   out.sdp       SDP solution / diagnostics if opts.run_sdp (includes sdp.zopt,
%                 sdp.robust_extract_nlp, and when opts.refine_robust_extract succeeds:
%                 sdp.x_hat_refined, sdp.x_refined = x_hat.*x_scale, sdp.fval_refined,
%                 sdp.exitflag_refined, sdp.output_refined)
%
% Requires:
%   - Optimization Toolbox for fmincon
%   - SPOT / CSTSS only if opts.run_sdp = true
%
% -------------------------------------------------------------------------

    if nargin < 1 || isempty(net_file)
        net_file = fullfile('data', 'GasLib-40-v1-20211130.net');
    end
    if nargin < 2 || isempty(scn_file)
        scn_file = fullfile('data', 'GasLib-40-v1-20211130.scn');
    end
    if nargin < 3 || isempty(scn_id)
        scn_id = 'nomination_1';
    end
    if nargin < 4
        opts = struct();
    end

    opts = set_default_opts(opts);

    data = parse_gaslib_instance(net_file, scn_file, scn_id, opts);
    data = build_variable_indexing(data, opts);
    [Aeq, beq] = build_mass_balance(data);
    [lb, ub, x0] = build_bounds_and_x0(data);
    opts = ensure_x_scale(opts, ub, data);

    s = opts.x_scale(:);
    Aeq_hat = Aeq * diag(s);
    lb_hat = scale_bounds_for_hat(lb, s, 'down');
    ub_hat = scale_bounds_for_hat(ub, s, 'down');
    x0_hat = x0 ./ s;

    w_lin = zeros(data.var.nvar, 1);
    w_lin(data.var.i_shed) = data.shed_weights(:) .* s(data.var.i_shed);
    obj_hat = @(xh) w_lin(:).' * xh(:);
    nonlcon_hat = @(xh) gaslib_nonlcon(xh .* s, data);

    [f_pop0, g_pop0, h_pop0] = gaslib_pop_numeric_fgh(x0, data, Aeq, beq, lb, ub, opts);

    verify_nlp_pop_hat = [];
    if opts.verify_nlp_pop_hat
        try
            verify_nlp_pop_hat = gaslib_verify_nlp_pop_hat_suite( ...
                x0_hat, lb_hat, ub_hat, data, Aeq, beq, lb, ub, opts, w_lin);
        catch ME
            warning('run_gaslib_mgsp:VerifyNlpPopHat', ...
                'NLP vs POP hat-space verification failed: %s', ME.message);
        end
    end

    if isempty(opts.fmincon_options)
        fopts = optimoptions('fmincon', ...
            'Algorithm', 'interior-point', ...
            'Display', ternary(opts.verbose, 'iter', 'off'), ...
            'SpecifyObjectiveGradient', false, ...
            'SpecifyConstraintGradient', false, ...
            'MaxIterations', 1000, ...
            'MaxFunctionEvaluations', 50000, ...
            'OptimalityTolerance', 1e-8, ...
            'StepTolerance', 1e-10, ...
            'ConstraintTolerance', 1e-8);
    else
        fopts = opts.fmincon_options;
    end

    [x_hat, fval, exitflag, output] = fmincon(obj_hat, x0_hat, [], [], Aeq_hat, beq, lb_hat, ub_hat, nonlcon_hat, fopts);

    x = x_hat .* s;
    nonlcon = @(xphys) gaslib_nonlcon(xphys, data);
    [c, ceq] = nonlcon(x);
    p_bar = max(x(data.var.i_p2), 0);

    nlp = struct();
    nlp.x = x;
    nlp.fval = fval;
    nlp.exitflag = exitflag;
    nlp.output = output;
    nlp.lin_resid = norm(Aeq*x - beq, inf);
    nlp.nonlin_eq_resid = norm(ceq, inf);
    nlp.max_ineq_violation = max([c; 0]);

    mask_lb = isfinite(lb);
    mask_ub = isfinite(ub);
    if any(mask_lb)
        nlp.max_lb_bound_violation = max(max(0, lb(mask_lb) - x(mask_lb)));
    else
        nlp.max_lb_bound_violation = 0;
    end
    if any(mask_ub)
        nlp.max_ub_bound_violation = max(max(0, x(mask_ub) - ub(mask_ub)));
    else
        nlp.max_ub_bound_violation = 0;
    end
    nlp.max_bound_violation = max(nlp.max_lb_bound_violation, nlp.max_ub_bound_violation);

    nlp.p_bar = p_bar;
    nlp.q = x(data.var.i_q);
    nlp.g = x(data.var.i_g);
    nlp.shed = x(data.var.i_shed);
    nlp.qabs = x(data.var.i_qabs);
    nlp.x_scale = opts.x_scale;
    nlp.x_hat = x_hat;
    nlp.verify_nlp_pop_hat = verify_nlp_pop_hat;

    nlp.pop_x0 = struct();
    nlp.pop_x0.f = f_pop0;
    nlp.pop_x0.min_g = min(g_pop0);
    nlp.pop_x0.max_abs_h = norm(h_pop0, inf);
    nlp.pop_x0.max_ineq_violation = max(max(-g_pop0, 0));
    nlp.pop_x0.feasible = (nlp.pop_x0.min_g >= -opts.pop_feas_tol) && ...
        (nlp.pop_x0.max_abs_h <= opts.pop_feas_tol);
    nlp.pop_x0.tol = opts.pop_feas_tol;
    nlp.pop_x0.m_ineq = numel(g_pop0);
    nlp.pop_x0.m_eq = numel(h_pop0);

    nlp.pop_coeff = [];
    if opts.report_pop_coeff_ranges && exist('msspoly', 'class') == 8
        try
            [z_m, f_m, g_m, h_m] = build_gaslib_pop_msspoly(data, Aeq, beq, lb, ub, opts);
            nlp.pop_coeff = gaslib_pop_coeff_ranges(z_m, f_m, g_m, h_m, opts.kappa, data, opts, ub);
        catch ME
            warning('run_gaslib_mgsp:PopCoeff', 'Could not compute POP coefficient ranges: %s', ME.message);
        end
    end

    if opts.verbose
        fprintf('\n=== GasLib MGSP NLP solution ===\n');
        fprintf('Network  : %s\n', char(data.net_title));
        fprintf('Scenario : %s\n', scn_id);
        fprintf('Nodes    : %d\n', data.N);
        fprintf('Arcs     : %d  (pipes=%d, short=%d, comps=%d)\n', ...
            data.M, numel(data.pipe_idx), numel(data.short_idx), numel(data.comp_idx));
        fprintf('POP vars : %d hat z  (p2=%d, q=%d, qabs=%d, g=%d, shed=%d)\n', ...
            data.var.nvar, data.N, data.M, numel(data.var.i_qabs), ...
            numel(data.var.i_g), numel(data.var.i_shed));
        fprintf('rho0     : %.6f kg/m^3\n', data.rho0);
        fprintf('T        : %.6f K\n', data.Tk);
        fprintf('M        : %.6f kg/mol\n', data.molar_mass_kg_per_mol);
        fprintf('Rs       : %.6f J/(kg K)\n', data.Rs);
        fprintf('z        : %.6f\n', data.z);
        fprintf('Total demand target : %.6f kg/s\n', sum(data.demand));
        fprintf('Total supply cap    : %.6f kg/s\n', sum(data.supply_cap));
        fprintf('Objective (weighted shedding): %.9g\n', fval);
        fprintf('Max linear residual        : %.3e\n', nlp.lin_resid);
        fprintf('Max nonlinear eq residual  : %.3e\n', nlp.nonlin_eq_resid);
        fprintf('Max inequality violation   : %.3e\n', nlp.max_ineq_violation);
        fprintf('Max lb bound violation     : %.3e  (max(0, lb-x), finite lb)\n', nlp.max_lb_bound_violation);
        fprintf('Max ub bound violation     : %.3e  (max(0, x-ub), finite ub)\n', nlp.max_ub_bound_violation);
        fprintf('Max bound violation        : %.3e\n', nlp.max_bound_violation);
        fprintf('POP init x0 (CSTSS f,g,h): f=%.9g  min(g)=%.3e  max|h|=%.3e  max(0,-g)=%.3e\n', ...
            nlp.pop_x0.f, nlp.pop_x0.min_g, nlp.pop_x0.max_abs_h, nlp.pop_x0.max_ineq_violation);
        fprintf('POP init x0 feasible (tol=%.1e on g,h): %d\n', nlp.pop_x0.tol, nlp.pop_x0.feasible);
        if opts.verify_nlp_pop_hat && ~isempty(nlp.verify_nlp_pop_hat)
            v = nlp.verify_nlp_pop_hat;
            fprintf('\n--- NLP vs POP (random hat checks, tol=%.1e) ---\n', opts.verify_nlp_pop_hat_tol);
            fprintf('samples=%d  max|f_nlp-f_pop|=%.3e  max|h_pop-h_nlp|_inf=%.3e\n', ...
                v.n_samples, v.max_abs_df, v.max_dh_inf);
            fprintf('max |g_box+comp_pop - rebuild|_inf = %.3e\n', v.max_dg_bc_inf);
            if opts.pop_ball_constraints
                fprintf('max |g_ball_pop - rebuild|_inf     = %.3e\n', v.max_dg_ball_inf);
            end
            fprintf('aggregate: max sample max(0,-c)=%.3e  max(0,-g_pop)=%.3e\n', ...
                v.max_nlp_neg_c, v.max_pop_neg_g);
            fprintf('check %s\n', ternary(v.pass, 'PASSED', 'FAILED (see per-sample in nlp.verify_nlp_pop_hat.samples)'));
        end
        if opts.pop_ball_constraints
            fprintf('POP ball rows          : %d  (1 - (x_phys(i)/c_i)^2 >= 0; x_phys=s.*z, c_i=ub or pop_ball_c_inf)\n', data.var.nvar);
        end
        if ~isempty(nlp.pop_coeff)
            pc = nlp.pop_coeff;
            fprintf('POP coeff via supp_rpt (2*kappa=%d):\n', pc.d2);
            fprintf('  f (m=1)     c in [%.3e, %.3e]  |.|_nz in [%.3e, %.3e]  dyn=%.3e\n', ...
                pc.f.min_c, pc.f.max_c, pc.f.min_abs_nz, pc.f.max_abs, pc.f.dynamic_range);
            gg = pc.g_groups;
            pop_coeff_print_group('g box p   ', gg.pressure_bounds);
            pop_coeff_print_group('g box q   ', gg.arc_flow_bounds);
            pop_coeff_print_group('g box qabs', gg.qabs_bounds);
            pop_coeff_print_group('g box g   ', gg.supply_bounds);
            pop_coeff_print_group('g box shed', gg.shed_bounds);
            pop_coeff_print_group('g comp    ', gg.compressor);
            pop_coeff_print_group('g ball    ', gg.ball);
            if opts.pop_ball_constraints && isfield(pc, 'ball_c')
                bc = pc.ball_c(:);
                nball = numel(bc);
                fprintf('  g ball c : c_i = ub(i) if isfinite(ub(i)), else pop_ball_c_inf=%.6g\n', opts.pop_ball_c_inf);
                fprintf('           c min=%.6g  max=%.6g  max(c^2)=%.6g (largest const term in c^2 - z(i)^2)\n', ...
                    min(bc), max(bc), max(bc .^ 2));
                fprintf('           finite ub: %d  inf ub -> c_inf: %d\n', sum(isfinite(ub(1:nball))), sum(~isfinite(ub(1:nball))));
                [bc_sorted, ord] = sort(bc, 'descend');
                ktop = min(10, nball);
                fprintf('           largest c (i, c):');
                for k = 1:ktop
                    fprintf(' (%d,%.6g)', ord(k), bc_sorted(k));
                end
                fprintf('\n');
            end
            fprintf('  h (m=%d)    c in [%.3e, %.3e]  |.|_nz in [%.3e, %.3e]  dyn=%.3e\n', ...
                pc.h.n_rows, pc.h.min_c, pc.h.max_c, pc.h.min_abs_nz, pc.h.max_abs, pc.h.dynamic_range);
            fprintf('  f+g+h all c in [%.3e, %.3e]  |.|_nz in [%.3e, %.3e]  dyn=%.3e\n', ...
                pc.all.min_c, pc.all.max_c, pc.all.min_abs_nz, pc.all.max_abs, pc.all.dynamic_range);
        end
        fprintf('Exitflag                   : %d\n', exitflag);
    end

    nlp.node_table = table( ...
        string(data.node_ids(:)), string(data.node_types(:)), ...
        data.pmin_bar(:), data.pmax_bar(:), p_bar(:), ...
        data.supply_cap(:), data.demand(:), ...
        'VariableNames', {'node_id','node_type','pmin_bar','pmax_bar','p_bar','supply_cap_kgps','demand_kgps'});

    nlp.arc_table = table( ...
        string(data.arc_ids(:)), string(data.arc_types(:)), ...
        string(data.node_ids(data.from(:))), string(data.node_ids(data.to(:))), ...
        x(data.var.i_q), ...
        'VariableNames', {'arc_id','arc_type','from_node','to_node','flow_kgps'});

    if ~isempty(data.var.i_g)
        nlp.supply_table = table( ...
            string(data.node_ids(data.g_nodes(:))), x(data.var.i_g), data.supply_cap(data.g_nodes(:)), ...
            'VariableNames', {'node_id','supply_kgps','supply_cap_kgps'});
    else
        nlp.supply_table = table();
    end

    if ~isempty(data.var.i_shed)
        nlp.shed_table = table( ...
            string(data.node_ids(data.shed_nodes(:))), data.demand(data.shed_nodes(:)), ...
            x(data.var.i_shed), data.shed_weights(:), ...
            'VariableNames', {'node_id','demand_kgps','shed_kgps','weight'});
    else
        nlp.shed_table = table();
    end

    sdp = [];
    if opts.run_sdp
        sdp = solve_gaslib_sdp(data, Aeq, beq, lb, ub, nlp.fval, opts);
    end

    out = struct();
    out.data = data;
    out.nlp = nlp;
    out.sdp = sdp;
end

% =======================================================================
% Parsing
% =======================================================================

function data = parse_gaslib_instance(net_file, scn_file, scn_id, opts)
    doc_net = xmlread(net_file);
    root_net = doc_net.getDocumentElement();

    nodes_parent = find_first_child(root_net, 'nodes');
    conns_parent = find_first_child(root_net, 'connections');
    info_parent  = find_first_child(root_net, 'information');

    if isempty(nodes_parent) || isempty(conns_parent)
        error('Could not find <nodes> or <connections> in %s.', net_file);
    end

    title_node = find_first_child(info_parent, 'title');
    if isempty(title_node)
        net_title = string(net_file);
    else
        net_title = string(strtrim(char(title_node.getTextContent())));
    end

    node_elems = child_elements(nodes_parent);
    N = numel(node_elems);

    node_ids = cell(N,1);
    node_types = cell(N,1);
    pmin_bar = zeros(N,1);
    pmax_bar = zeros(N,1);
    height_m = zeros(N,1);
    node_flowmax_kgps = nan(N,1);
    node_flowmin_kgps = nan(N,1);
    tempk = nan(N,1);
    norm_density = nan(N,1);
    molar_mass_kg_per_mol = nan(N,1);
    calorific_MJ_per_m3 = nan(N,1);

    for i = 1:N
        ne = node_elems{i};
        node_ids{i} = get_attr(ne, 'id');
        node_types{i} = local_name(ne);

        tmp = get_value_unit(ne, 'height');
        if ~isempty(tmp)
            height_m(i) = convert_length_to_m(tmp.value, tmp.unit);
        end

        tmp = get_value_unit(ne, 'pressureMin');
        pmin_bar(i) = convert_pressure_to_bar_abs(tmp.value, tmp.unit, 1.01325);
        tmp = get_value_unit(ne, 'pressureMax');
        pmax_bar(i) = convert_pressure_to_bar_abs(tmp.value, tmp.unit, 1.01325);

        tmp = get_value_unit(ne, 'flowMin');
        if ~isempty(tmp)
            node_flowmin_kgps(i) = NaN; % filled after rho0 is known
        end
        tmp = get_value_unit(ne, 'flowMax');
        if ~isempty(tmp)
            node_flowmax_kgps(i) = NaN; % filled after rho0 is known
        end

        if strcmp(node_types{i}, 'source')
            tmp = get_value_unit(ne, 'gasTemperature');
            if ~isempty(tmp)
                tempk(i) = convert_temperature_to_K(tmp.value, tmp.unit);
            end
            tmp = get_value_unit(ne, 'normDensity');
            if ~isempty(tmp)
                norm_density(i) = convert_density_to_kgpm3(tmp.value, tmp.unit);
            end
            tmp = get_value_unit(ne, 'molarMass');
            if ~isempty(tmp)
                molar_mass_kg_per_mol(i) = convert_molarmass_to_kgpmol(tmp.value, tmp.unit);
            end
            tmp = get_value_unit(ne, 'calorificValue');
            if ~isempty(tmp)
                calorific_MJ_per_m3(i) = convert_calorific_to_MJpm3(tmp.value, tmp.unit);
            end
        end
    end

    src_mask = strcmp(node_types, 'source');
    if ~any(src_mask)
        error('No source nodes found in %s.', net_file);
    end

    rho0 = mean(norm_density(src_mask), 'omitnan');
    Tk = mean(tempk(src_mask), 'omitnan');
    Mkgpmol = mean(molar_mass_kg_per_mol(src_mask), 'omitnan');
    Hs_MJpm3 = mean(calorific_MJ_per_m3(src_mask), 'omitnan');
    Rs = 8.314462618 / Mkgpmol;
    z = opts.z;

    % fill node flow bounds now that rho0/Hs are known
    for i = 1:N
        ne = node_elems{i};
        tmp = get_value_unit(ne, 'flowMin');
        if ~isempty(tmp)
            node_flowmin_kgps(i) = convert_flow_to_kgps(tmp.value, tmp.unit, rho0, Hs_MJpm3);
        end
        tmp = get_value_unit(ne, 'flowMax');
        if ~isempty(tmp)
            node_flowmax_kgps(i) = convert_flow_to_kgps(tmp.value, tmp.unit, rho0, Hs_MJpm3);
        end
    end

    node_map = containers.Map(node_ids, num2cell(1:N));

    arc_elems = child_elements(conns_parent);
    M = numel(arc_elems);

    arc_ids = cell(M,1);
    arc_types = cell(M,1);
    from = zeros(M,1);
    to = zeros(M,1);
    qmin = zeros(M,1);
    qmax = zeros(M,1);
    raw_qmin = zeros(M,1);
    raw_qmax = zeros(M,1);
    res_factor = zeros(M,1);
    crmin = ones(M,1);
    crmax = ones(M,1);

    unsupported_types = {};

    for e = 1:M
        ae = arc_elems{e};
        typ = local_name(ae);
        arc_ids{e} = get_attr(ae, 'id');
        arc_types{e} = typ;
        from_id = get_attr(ae, 'from');
        to_id = get_attr(ae, 'to');

        if ~isKey(node_map, from_id) || ~isKey(node_map, to_id)
            error('Arc %s references unknown nodes (%s, %s).', arc_ids{e}, from_id, to_id);
        end
        from(e) = node_map(from_id);
        to(e) = node_map(to_id);

        tmp = get_value_unit(ae, 'flowMin');
        raw_qmin(e) = convert_flow_to_kgps(tmp.value, tmp.unit, rho0, Hs_MJpm3);
        tmp = get_value_unit(ae, 'flowMax');
        raw_qmax(e) = convert_flow_to_kgps(tmp.value, tmp.unit, rho0, Hs_MJpm3);

        switch typ
            case 'pipe'
                D = convert_length_to_m(get_value_unit(ae, 'diameter').value, get_value_unit(ae, 'diameter').unit);
                eps = convert_length_to_m(get_value_unit(ae, 'roughness').value, get_value_unit(ae, 'roughness').unit);
                L = convert_length_to_m(get_value_unit(ae, 'length').value, get_value_unit(ae, 'length').unit);
                lambda = (2.0 * log10(D / eps) + 1.138)^(-2);
                res_factor(e) = 16.0 * lambda * Rs * z * Tk * L / (pi^2 * D^5);

                if opts.passive_bidirectional
                    qcap = max(abs([raw_qmin(e), raw_qmax(e)]));
                    qmin(e) = -qcap;
                    qmax(e) =  qcap;
                else
                    qmin(e) = raw_qmin(e);
                    qmax(e) = raw_qmax(e);
                end

            case 'shortPipe'
                if opts.passive_bidirectional
                    qcap = max(abs([raw_qmin(e), raw_qmax(e)]));
                    qmin(e) = -qcap;
                    qmax(e) =  qcap;
                else
                    qmin(e) = raw_qmin(e);
                    qmax(e) = raw_qmax(e);
                end

            case 'compressorStation'
                pin_min = get_value_unit(ae, 'pressureInMin');
                pout_max = get_value_unit(ae, 'pressureOutMax');
                pin_min_bar = convert_pressure_to_bar_abs(pin_min.value, pin_min.unit, 1.01325);
                pout_max_bar = convert_pressure_to_bar_abs(pout_max.value, pout_max.unit, 1.01325);
                crmin(e) = 1.0;
                crmax(e) = pout_max_bar / pin_min_bar;

                if opts.enforce_compressor_forward
                    qmin(e) = max(0, raw_qmin(e));
                else
                    qmin(e) = raw_qmin(e);
                end
                qmax(e) = raw_qmax(e);

            otherwise
                unsupported_types{end+1} = typ; %#ok<AGROW>
                qmin(e) = raw_qmin(e);
                qmax(e) = raw_qmax(e);
        end
    end

    if ~isempty(unsupported_types)
        unsupported_types = unique(unsupported_types);
        error(['Unsupported arc type(s): %s. ', ...
            'This script supports only pipe, shortPipe, and compressorStation. ', ...
            'Resistors / valves / control valves need separate modeling.'], ...
            strjoin(unsupported_types, ', '));
    end

    % ------------------------------------------------------------------
    % Scenario parsing
    % ------------------------------------------------------------------
    doc_scn = xmlread(scn_file);
    root_scn = doc_scn.getDocumentElement();

    scn_elem = [];
    scn_children = child_elements(root_scn);
    for k = 1:numel(scn_children)
        if strcmp(local_name(scn_children{k}), 'scenario')
            if strcmp(get_attr(scn_children{k}, 'id'), scn_id)
                scn_elem = scn_children{k};
                break;
            end
        end
    end
    if isempty(scn_elem)
        error('Scenario id "%s" not found in %s.', scn_id, scn_file);
    end

    scn_p_lb = nan(N,1);
    scn_p_ub = nan(N,1);
    flow_lb = nan(N,1);
    flow_ub = nan(N,1);
    scn_entry = false(N,1);
    scn_exit = false(N,1);

    scn_nodes = child_elements(scn_elem);
    for k = 1:numel(scn_nodes)
        se = scn_nodes{k};
        if ~strcmp(local_name(se), 'node')
            continue;
        end

        node_id = get_attr(se, 'id');
        if ~isKey(node_map, node_id)
            error('Scenario references unknown node %s.', node_id);
        end
        i = node_map(node_id);
        ntype = get_attr(se, 'type');
        if strcmp(ntype, 'entry')
            scn_entry(i) = true;
        elseif strcmp(ntype, 'exit')
            scn_exit(i) = true;
        end

        se_children = child_elements(se);
        for j = 1:numel(se_children)
            ce = se_children{j};
            tag = local_name(ce);
            bound = get_attr(ce, 'bound');
            val = str2double(get_attr(ce, 'value'));
            unit = get_attr(ce, 'unit');

            switch tag
                case 'pressure'
                    pbar = convert_pressure_to_bar_abs(val, unit, 1.01325);
                    switch bound
                        case 'lower'
                            scn_p_lb(i) = pbar;
                        case 'upper'
                            scn_p_ub(i) = pbar;
                        case 'both'
                            scn_p_lb(i) = pbar;
                            scn_p_ub(i) = pbar;
                    end
                case 'flow'
                    fkgps = convert_flow_to_kgps(val, unit, rho0, Hs_MJpm3);
                    switch bound
                        case 'lower'
                            flow_lb(i) = fkgps;
                        case 'upper'
                            flow_ub(i) = fkgps;
                        case 'both'
                            flow_lb(i) = fkgps;
                            flow_ub(i) = fkgps;
                    end
                otherwise
                    % ignore contractPressure etc. in this simplified model
            end
        end
    end

    % Intersect pressure bounds
    for i = 1:N
        if ~isnan(scn_p_lb(i))
            pmin_bar(i) = max(pmin_bar(i), scn_p_lb(i));
        end
        if ~isnan(scn_p_ub(i))
            pmax_bar(i) = min(pmax_bar(i), scn_p_ub(i));
        end
        if pmin_bar(i) > pmax_bar(i)
            error('Pressure bounds inconsistent at node %s after .net/.scn intersection.', node_ids{i});
        end
    end

    supply_cap = zeros(N,1);
    demand = zeros(N,1);

    for i = 1:N
        if scn_entry(i)
            cap = pick_entry_cap(flow_lb(i), flow_ub(i));
            if ~isnan(node_flowmax_kgps(i))
                cap = min(cap, node_flowmax_kgps(i));
            end
            supply_cap(i) = max(cap, 0);
        end

        if scn_exit(i)
            dem = pick_exit_demand(flow_lb(i), flow_ub(i));
            if ~isnan(node_flowmax_kgps(i))
                dem = min(dem, node_flowmax_kgps(i));
            end
            demand(i) = max(dem, 0);
        end
    end

    weights = zeros(N,1);
    for i = 1:N
        if demand(i) > 0
            weights(i) = lookup_weight(opts.weights, node_ids{i}, opts.weight_default);
        end
    end

    data = struct();
    data.net_file = net_file;
    data.scn_file = scn_file;
    data.net_title = net_title;
    data.scn_id = scn_id;
    data.N = N;
    data.M = M;
    data.node_ids = node_ids;
    data.node_types = node_types;
    data.arc_ids = arc_ids;
    data.arc_types = arc_types;
    data.from = from;
    data.to = to;
    data.raw_qmin = raw_qmin;
    data.raw_qmax = raw_qmax;
    data.qmin = qmin;
    data.qmax = qmax;
    data.pmin_bar = pmin_bar;
    data.pmax_bar = pmax_bar;
    data.Pa_per_bar = 1e5;
    data.Pa2_per_bar2 = data.Pa_per_bar^2;
    data.height_m = height_m;
    data.node_flowmax_kgps = node_flowmax_kgps;
    data.node_flowmin_kgps = node_flowmin_kgps;
    data.rho0 = rho0;
    data.Tk = Tk;
    data.molar_mass_kg_per_mol = Mkgpmol;
    data.calorific_MJ_per_m3 = Hs_MJpm3;
    data.Rs = Rs;
    data.z = z;
    data.res_factor = res_factor;
    data.res_over_pa2 = res_factor / data.Pa2_per_bar2;
    data.crmin = crmin;
    data.crmax = crmax;
    data.scn_entry = scn_entry;
    data.scn_exit = scn_exit;
    data.supply_cap = supply_cap;
    data.demand = demand;
    data.weights = weights;
    data.pipe_idx = find(strcmp(arc_types, 'pipe'));
    data.short_idx = find(strcmp(arc_types, 'shortPipe'));
    data.comp_idx = find(strcmp(arc_types, 'compressorStation'));
end

% =======================================================================
% Variable maps / linear model / initialization
% =======================================================================

function data = build_variable_indexing(data, opts)
    data.g_nodes = find(data.supply_cap > 0);
    data.shed_nodes = find(data.demand > 0);
    data.shed_weights = data.weights(data.shed_nodes);

    N = data.N;
    M = data.M;
    P = numel(data.pipe_idx);
    G = numel(data.g_nodes);
    S = numel(data.shed_nodes);

    offs = 0;
    % Pressure: absolute bar (field name i_p2 kept for compatibility).
    data.var.i_p2 = offs + (1:N); offs = offs + N;
    data.var.i_q = offs + (1:M); offs = offs + M;
    data.var.i_qabs = offs + (1:P); offs = offs + P;
    data.var.i_g = offs + (1:G); offs = offs + G;
    data.var.i_shed = offs + (1:S); offs = offs + S;
    data.var.nvar = offs;

    data.var.qabs_arc = data.pipe_idx(:);
    data.var.qabs_of_arc = zeros(M,1);
    for k = 1:P
        data.var.qabs_of_arc(data.pipe_idx(k)) = data.var.i_qabs(k);
    end

    data.var.g_of_node = zeros(N,1);
    for k = 1:G
        data.var.g_of_node(data.g_nodes(k)) = data.var.i_g(k);
    end

    data.var.shed_of_node = zeros(N,1);
    for k = 1:S
        data.var.shed_of_node(data.shed_nodes(k)) = data.var.i_shed(k);
    end

    data.passive_bidirectional = opts.passive_bidirectional;
end

function [Aeq, beq] = build_mass_balance(data)
    Aeq = zeros(data.N, data.var.nvar);
    beq = -data.demand(:);

    for e = 1:data.M
        Aeq(data.from(e), data.var.i_q(e)) = Aeq(data.from(e), data.var.i_q(e)) + 1;
        Aeq(data.to(e),   data.var.i_q(e)) = Aeq(data.to(e),   data.var.i_q(e)) - 1;
    end

    for k = 1:numel(data.g_nodes)
        n = data.g_nodes(k);
        Aeq(n, data.var.i_g(k)) = -1;
    end

    for k = 1:numel(data.shed_nodes)
        n = data.shed_nodes(k);
        Aeq(n, data.var.i_shed(k)) = -1;
    end
end

function [lb, ub, x0] = build_bounds_and_x0(data)
    lb = -inf(data.var.nvar, 1);
    ub =  inf(data.var.nvar, 1);

    lb(data.var.i_p2) = data.pmin_bar;
    ub(data.var.i_p2) = data.pmax_bar;

    lb(data.var.i_q) = data.qmin;
    ub(data.var.i_q) = data.qmax;

    if ~isempty(data.var.i_qabs)
        qcaps = max(abs([data.qmin(data.pipe_idx), data.qmax(data.pipe_idx)]), [], 2);
        lb(data.var.i_qabs) = 0;
        ub(data.var.i_qabs) = qcaps;
    end

    if ~isempty(data.var.i_g)
        lb(data.var.i_g) = 0;
        ub(data.var.i_g) = data.supply_cap(data.g_nodes);
    end

    if ~isempty(data.var.i_shed)
        lb(data.var.i_shed) = 0;
        ub(data.var.i_shed) = data.demand(data.shed_nodes);
    end

    % Start from a mass-balance-feasible point: zero arc flows, zero supply,
    % full shedding at demand nodes. Choose a common pressure in bar if possible.
    x0 = zeros(data.var.nvar, 1);

    common_lb = max(data.pmin_bar);
    common_ub = min(data.pmax_bar);

    if common_lb <= common_ub
        p0_bar = 0.5 * (common_lb + common_ub);
        for i = 1:data.N
            ii = data.var.i_p2(i);
            x0(ii) = min(max(p0_bar, data.pmin_bar(i)), data.pmax_bar(i));
        end
    else
        for i = 1:data.N
            ii = data.var.i_p2(i);
            x0(ii) = 0.5 * (data.pmin_bar(i) + data.pmax_bar(i));
        end
    end

    % Respect any strictly positive lower flow bounds if they exist.
    x0(data.var.i_q) = max(0, lb(data.var.i_q));

    if ~isempty(data.var.i_qabs)
        for k = 1:numel(data.pipe_idx)
            e = data.pipe_idx(k);
            x0(data.var.i_qabs(k)) = abs(x0(data.var.i_q(e)));
        end
    end

    if ~isempty(data.var.i_g)
        x0(data.var.i_g) = 0;
    end
    if ~isempty(data.var.i_shed)
        x0(data.var.i_shed) = data.demand(data.shed_nodes);
    end
end

% =======================================================================
% Nonlinear constraints
% =======================================================================

function [c, ceq] = gaslib_nonlcon(x, data)
    pb = x(data.var.i_p2);  % absolute pressure [bar]
    q  = x(data.var.i_q);

    np = numel(data.pipe_idx);
    ns = numel(data.short_idx);
    nc = numel(data.comp_idx);

    c = zeros(np + 2*nc, 1);
    ceq = zeros(2*np + ns, 1);

    % Pipes: (pb_i^2 - pb_j^2) - (res_factor/Pa2_per_bar2)*q*q_abs = 0  (equiv. to Pa^2 form).
    for k = 1:np
        e = data.pipe_idx(k);
        i = data.from(e);
        j = data.to(e);
        qabs = x(data.var.i_qabs(k));

        ceq(k) = (pb(i)^2 - pb(j)^2) - data.res_over_pa2(e) * q(e) * qabs;
        ceq(np + k) = qabs^2 - q(e)^2;
        c(k) = -qabs;
    end

    % Short pipes: equal pb^2 (pb >= 0 from bounds => equal bar pressure).
    for k = 1:ns
        e = data.short_idx(k);
        i = data.from(e);
        j = data.to(e);
        ceq(2*np + k) = pb(i)^2 - pb(j)^2;
    end

    % Compressors: ratio bounds on pb^2 (equivalent to Pa^2 ratios / Pa2_per_bar2).
    for k = 1:nc
        e = data.comp_idx(k);
        i = data.from(e);
        j = data.to(e);
        c(np + 2*k - 1) = (data.crmin(e)^2) * pb(i)^2 - pb(j)^2;
        c(np + 2*k)     = pb(j)^2 - (data.crmax(e)^2) * pb(i)^2;
    end
end

% Numeric POP (CSTSS convention: g >= 0, h = 0) at point x. Must stay aligned
% with solve_gaslib_sdp stacking order for inequality / equality vectors.
function [f, g, h] = gaslib_pop_numeric_fgh(x, data, Aeq, beq, lb, ub, opts)
    if isempty(data.var.i_shed)
        f = 0;
    else
        f = data.shed_weights(:).' * x(data.var.i_shed);
    end

    pb = x(data.var.i_p2);
    q = x(data.var.i_q);

    h = Aeq * x - beq;

    for k = 1:numel(data.pipe_idx)
        e = data.pipe_idx(k);
        i = data.from(e);
        j = data.to(e);
        qabs = x(data.var.i_qabs(k));
        h = [h; ...
            (pb(i)^2 - pb(j)^2) - data.res_over_pa2(e) * q(e) * qabs; ...
            qabs^2 - q(e)^2]; %#ok<AGROW>
    end

    for k = 1:numel(data.short_idx)
        e = data.short_idx(k);
        i = data.from(e);
        j = data.to(e);
        h = [h; pb(i)^2 - pb(j)^2]; %#ok<AGROW>
    end

    g = [ ...
        pb - lb(data.var.i_p2); ...
        ub(data.var.i_p2) - pb; ...
        q - lb(data.var.i_q); ...
        ub(data.var.i_q) - q];

    if ~isempty(data.var.i_qabs)
        qa = x(data.var.i_qabs);
        g = [g; qa; ub(data.var.i_qabs) - qa];
    end

    if ~isempty(data.var.i_g)
        gv = x(data.var.i_g);
        g = [g; gv; ub(data.var.i_g) - gv];
    end

    if ~isempty(data.var.i_shed)
        sv = x(data.var.i_shed);
        g = [g; sv; ub(data.var.i_shed) - sv];
    end

    for k = 1:numel(data.comp_idx)
        e = data.comp_idx(k);
        i = data.from(e);
        j = data.to(e);
        g = [g; ...
            pb(j)^2 - (data.crmin(e)^2) * pb(i)^2; ...
            (data.crmax(e)^2) * pb(i)^2 - pb(j)^2]; %#ok<AGROW>
    end

    if opts.pop_ball_constraints
        n = data.var.nvar;
        for iv = 1:n
            if isfinite(ub(iv))
                cball = ub(iv);
            else
                cball = opts.pop_ball_c_inf;
            end
            g = [g; cball^2 - x(iv)^2]; %#ok<AGROW>
        end
    end
end

function xh = clip_xhat_to_bounds(xh, lb_hat, ub_hat)
    xh = xh(:);
    for i = 1:numel(xh)
        if isfinite(lb_hat(i))
            xh(i) = max(xh(i), lb_hat(i));
        end
        if isfinite(ub_hat(i))
            xh(i) = min(xh(i), ub_hat(i));
        end
    end
end

function gbc = gaslib_pop_g_box_compressor_only_physical(x, data, lb, ub)
    % Box + compressor g rows only (same as gaslib_pop_numeric_fgh), physical x.
    pb = x(data.var.i_p2);
    q = x(data.var.i_q);
    gbc = [ ...
        pb - lb(data.var.i_p2); ...
        ub(data.var.i_p2) - pb; ...
        q - lb(data.var.i_q); ...
        ub(data.var.i_q) - q];
    if ~isempty(data.var.i_qabs)
        qa = x(data.var.i_qabs);
        gbc = [gbc; qa; ub(data.var.i_qabs) - qa]; %#ok<AGROW>
    end
    if ~isempty(data.var.i_g)
        gv = x(data.var.i_g);
        gbc = [gbc; gv; ub(data.var.i_g) - gv]; %#ok<AGROW>
    end
    if ~isempty(data.var.i_shed)
        sv = x(data.var.i_shed);
        gbc = [gbc; sv; ub(data.var.i_shed) - sv]; %#ok<AGROW>
    end
    for k = 1:numel(data.comp_idx)
        e = data.comp_idx(k);
        i = data.from(e);
        j = data.to(e);
        gbc = [gbc; ...
            pb(j)^2 - (data.crmin(e)^2) * pb(i)^2; ...
            (data.crmax(e)^2) * pb(i)^2 - pb(j)^2]; %#ok<AGROW>
    end
end

function gb = gaslib_pop_g_ball_only_physical(x, lb, ub, opts)
    gb = zeros(0, 1);
    if ~opts.pop_ball_constraints
        return;
    end
    n = numel(x);
    for iv = 1:n
        if isfinite(ub(iv))
            cball = ub(iv);
        else
            cball = opts.pop_ball_c_inf;
        end
        gb = [gb; cball^2 - x(iv)^2]; %#ok<AGROW>
    end
end

function r = gaslib_verify_nlp_pop_hat_once(x_hat, data, Aeq, beq, lb, ub, opts, w_lin)
    s = opts.x_scale(:);
    x_hat = x_hat(:);
    x_phys = x_hat .* s;
    f_nlp = w_lin(:).' * x_hat;
    [f_pop, g_pop, h_pop] = gaslib_pop_numeric_fgh(x_phys, data, Aeq, beq, lb, ub, opts);
    r.df = f_nlp - f_pop;

    [c, ceq] = gaslib_nonlcon(x_phys, data);
    % gaslib_nonlcon stacks ceq as [Weymouth_1..np; qabs_eq_1..np; short_1..ns].
    % gaslib_pop_numeric_fgh stacks h as [mass; W1,Q1,W2,Q2,...; short...].
    np = numel(data.pipe_idx);
    ns = numel(data.short_idx);
    h_mass = Aeq * x_phys - beq;
    if np > 0
        W = ceq(1:np);
        Q = ceq(np + 1:2 * np);
        h_tail = zeros(2 * np + ns, 1);
        for kk = 1:np
            h_tail(2 * kk - 1) = W(kk);
            h_tail(2 * kk) = Q(kk);
        end
        if ns > 0
            h_tail(2 * np + 1:2 * np + ns) = ceq(2 * np + 1:2 * np + ns);
        end
        h_expected = [h_mass; h_tail];
    else
        h_expected = [h_mass; ceq(:)];
    end
    r.dh = h_pop(:) - h_expected(:);
    r.dh_inf = norm(r.dh, inf);

    N = data.N;
    M = data.M;
    Pq = numel(data.var.i_qabs);
    Gn = numel(data.var.i_g);
    Sn = numel(data.var.i_shed);
    nc = numel(data.comp_idx);
    n_bc = 2 * N + 2 * M + 2 * Pq + 2 * Gn + 2 * Sn + 2 * nc;
    gbc = gaslib_pop_g_box_compressor_only_physical(x_phys, data, lb, ub);
    r.dg_bc_inf = norm(g_pop(1:n_bc) - gbc, inf);

    if opts.pop_ball_constraints
        gb = gaslib_pop_g_ball_only_physical(x_phys, lb, ub, opts);
        r.dg_ball_inf = norm(g_pop(n_bc + 1:end) - gb, inf);
    else
        r.dg_ball_inf = 0;
    end

    r.nlp_max_neg_c = max(max(-c(:), 0));
    r.pop_max_neg_g = max(max(-g_pop(:), 0));
end

function rep = gaslib_verify_nlp_pop_hat_suite(x0_hat, lb_hat, ub_hat, data, Aeq, beq, lb, ub, opts, w_lin)
    rep = struct();
    rep.n_samples = opts.verify_nlp_pop_hat_samples;
    if isempty(opts.verify_nlp_pop_hat_seed)
        rng('shuffle');
    else
        rng(opts.verify_nlp_pop_hat_seed);
    end
    rep.samples = cell(rep.n_samples, 1);
    rep.max_abs_df = 0;
    rep.max_dh_inf = 0;
    rep.max_dg_bc_inf = 0;
    rep.max_dg_ball_inf = 0;
    rep.max_nlp_neg_c = 0;
    rep.max_pop_neg_g = 0;
    noise = opts.verify_nlp_pop_hat_noise;
    tol = opts.verify_nlp_pop_hat_tol;
    for k = 1:rep.n_samples
        xh = clip_xhat_to_bounds(x0_hat + noise * randn(data.var.nvar, 1), lb_hat, ub_hat);
        r = gaslib_verify_nlp_pop_hat_once(xh, data, Aeq, beq, lb, ub, opts, w_lin);
        rep.samples{k} = r;
        rep.max_abs_df = max(rep.max_abs_df, abs(r.df));
        rep.max_dh_inf = max(rep.max_dh_inf, r.dh_inf);
        rep.max_dg_bc_inf = max(rep.max_dg_bc_inf, r.dg_bc_inf);
        rep.max_dg_ball_inf = max(rep.max_dg_ball_inf, r.dg_ball_inf);
        rep.max_nlp_neg_c = max(rep.max_nlp_neg_c, r.nlp_max_neg_c);
        rep.max_pop_neg_g = max(rep.max_pop_neg_g, r.pop_max_neg_g);
    end
    rep.pass = (rep.max_abs_df <= tol) && (rep.max_dh_inf <= tol) && ...
        (rep.max_dg_bc_inf <= tol) && (rep.max_dg_ball_inf <= tol);
end

function [z, objective, inequality, equality] = build_gaslib_pop_msspoly(data, Aeq, beq, lb, ub, opts)
    % z = hat-space msspoly indeterminates (passed to CSTSS_mex); x_phys(i)=s(i)*z(i).
    z = msspoly('z', data.var.nvar);
    svec = opts.x_scale(:);
    x = z;
    for iv = 1:data.var.nvar
        x(iv) = svec(iv) * z(iv);
    end

    objective = data.shed_weights(:).' * x(data.var.i_shed);

    equality = Aeq * x - beq;

    for k = 1:numel(data.pipe_idx)
        e = data.pipe_idx(k);
        i = data.from(e);
        j = data.to(e);
        iqabs = data.var.i_qabs(k);
        equality = [equality; ...
            x(data.var.i_p2(i))^2 - x(data.var.i_p2(j))^2 - data.res_over_pa2(e) * x(data.var.i_q(e)) * x(iqabs); ...
            x(iqabs)^2 - x(data.var.i_q(e))^2]; %#ok<AGROW>
    end

    for k = 1:numel(data.short_idx)
        e = data.short_idx(k);
        i = data.from(e);
        j = data.to(e);
        equality = [equality; x(data.var.i_p2(i))^2 - x(data.var.i_p2(j))^2]; %#ok<AGROW>
    end

    inequality = [];

    inequality = [inequality; x(data.var.i_p2) - lb(data.var.i_p2)];
    inequality = [inequality; ub(data.var.i_p2) - x(data.var.i_p2)];
    inequality = [inequality; x(data.var.i_q) - lb(data.var.i_q)];
    inequality = [inequality; ub(data.var.i_q) - x(data.var.i_q)];

    if ~isempty(data.var.i_qabs)
        inequality = [inequality; x(data.var.i_qabs)];
        inequality = [inequality; ub(data.var.i_qabs) - x(data.var.i_qabs)];
    end

    if ~isempty(data.var.i_g)
        inequality = [inequality; x(data.var.i_g)];
        inequality = [inequality; ub(data.var.i_g) - x(data.var.i_g)];
    end

    if ~isempty(data.var.i_shed)
        inequality = [inequality; x(data.var.i_shed)];
        inequality = [inequality; ub(data.var.i_shed) - x(data.var.i_shed)];
    end

    for k = 1:numel(data.comp_idx)
        e = data.comp_idx(k);
        i = data.from(e);
        j = data.to(e);
        inequality = [inequality; ...
            x(data.var.i_p2(j))^2 - (data.crmin(e)^2) * x(data.var.i_p2(i))^2; ...
            (data.crmax(e)^2) * x(data.var.i_p2(i))^2 - x(data.var.i_p2(j))^2]; %#ok<AGROW>
    end

    if opts.pop_ball_constraints
        n = data.var.nvar;
        for iv = 1:n
            if isfinite(ub(iv))
                cball = ub(iv);
            else
                cball = opts.pop_ball_c_inf;
            end
            inequality = [inequality; 1 - (x(iv) / cball)^2];
        end
    end
end

function stats = gaslib_pop_coeff_ranges(z, objective, inequality, equality, kappa, data, opts, ub)
    stats = struct();
    stats.kappa = kappa;
    stats.d2 = 2 * kappa;
    d2 = stats.d2;

    [~, cf] = supp_rpt(objective, z, d2);
    cf = full(double(cf(:)));
    stats.f = local_coeff_block(cf, 1);

    N = data.N;
    M = data.M;
    Pq = numel(data.var.i_qabs);
    Gn = numel(data.var.i_g);
    Sn = numel(data.var.i_shed);
    nc = numel(data.comp_idx);

    idx = 1;
    stats.g_groups.pressure_bounds = pop_collect_g_rows(inequality, z, d2, idx, idx + 2 * N - 1);
    idx = idx + 2 * N;

    stats.g_groups.arc_flow_bounds = pop_collect_g_rows(inequality, z, d2, idx, idx + 2 * M - 1);
    idx = idx + 2 * M;

    if Pq > 0
        nq = 2 * Pq;
        stats.g_groups.qabs_bounds = pop_collect_g_rows(inequality, z, d2, idx, idx + nq - 1);
        idx = idx + nq;
    else
        stats.g_groups.qabs_bounds = pop_empty_coeff_stat();
    end

    if Gn > 0
        ng = 2 * Gn;
        stats.g_groups.supply_bounds = pop_collect_g_rows(inequality, z, d2, idx, idx + ng - 1);
        idx = idx + ng;
    else
        stats.g_groups.supply_bounds = pop_empty_coeff_stat();
    end

    if Sn > 0
        ns2 = 2 * Sn;
        stats.g_groups.shed_bounds = pop_collect_g_rows(inequality, z, d2, idx, idx + ns2 - 1);
        idx = idx + ns2;
    else
        stats.g_groups.shed_bounds = pop_empty_coeff_stat();
    end

    if nc > 0
        ncrows = 2 * nc;
        stats.g_groups.compressor = pop_collect_g_rows(inequality, z, d2, idx, idx + ncrows - 1);
        idx = idx + ncrows;
    else
        stats.g_groups.compressor = pop_empty_coeff_stat();
    end

    if opts.pop_ball_constraints
        nb = data.var.nvar;
        stats.g_groups.ball = pop_collect_g_rows(inequality, z, d2, idx, idx + nb - 1);
        idx = idx + nb;
    else
        stats.g_groups.ball = pop_empty_coeff_stat();
    end

    mg = size(inequality, 1);
    if idx - 1 ~= mg
        warning('run_gaslib_mgsp:PopCoeffGroupCount', ...
            'POP inequality row partition (%d) does not match size(inequality) (%d).', idx - 1, mg);
    end

    allg = [];
    for i = 1:mg
        [~, cgi] = supp_rpt(inequality(i), z, d2);
        allg = [allg; full(double(cgi(:)))]; %#ok<AGROW>
    end
    stats.g_pooled = local_coeff_block(allg, mg);

    allh = [];
    mh = size(equality, 1);
    for i = 1:mh
        [~, chi] = supp_rpt(equality(i), z, d2);
        allh = [allh; full(double(chi(:)))]; %#ok<AGROW>
    end
    stats.h = local_coeff_block(allh, mh);

    allv = [cf; allg; allh];
    stats.all = local_coeff_block(allv, 0);

    if opts.pop_ball_constraints
        n = data.var.nvar;
        cball = zeros(n, 1);
        for iv = 1:n
            if isfinite(ub(iv))
                cball(iv) = ub(iv);
            else
                cball(iv) = opts.pop_ball_c_inf;
            end
        end
        stats.ball_c = cball;
    end
end

function s = pop_collect_g_rows(inequality, z, d2, rowFirst, rowLast)
    if rowLast < rowFirst
        s = pop_empty_coeff_stat();
        return;
    end
    v = [];
    for i = rowFirst:rowLast
        [~, cgi] = supp_rpt(inequality(i), z, d2);
        v = [v; full(double(cgi(:)))]; %#ok<AGROW>
    end
    s = local_coeff_block(v, rowLast - rowFirst + 1);
end

function s = pop_empty_coeff_stat()
    s = struct('n_rows', 0, 'min_c', NaN, 'max_c', NaN, 'min_abs_nz', NaN, 'max_abs', 0, 'dynamic_range', NaN);
end

function pop_coeff_print_group(label, s)
    if s.n_rows == 0
        fprintf('  %-10s m=0   (none)\n', label);
        return;
    end
    fprintf('  %-10s m=%-4d c in [%.3e, %.3e]  |.|_nz in [%.3e, %.3e]  dyn=%.3e\n', ...
        label, s.n_rows, s.min_c, s.max_c, s.min_abs_nz, s.max_abs, s.dynamic_range);
end

function s = local_coeff_block(v, n_rows_report)
    v = v(:);
    s.n_rows = n_rows_report;
    if isempty(v)
        s.min_c = NaN;
        s.max_c = NaN;
        s.min_abs_nz = NaN;
        s.max_abs = 0;
        s.dynamic_range = NaN;
        return;
    end
    s.min_c = min(v);
    s.max_c = max(v);
    vn = v(v ~= 0);
    if isempty(vn)
        s.min_abs_nz = 0;
        s.max_abs = max(abs(v));
    else
        s.min_abs_nz = min(abs(vn));
        s.max_abs = max(abs(v));
    end
    s.dynamic_range = s.max_abs / max(s.min_abs_nz, eps);
end

function sdp = solve_gaslib_sdp(data, Aeq, beq, lb, ub, nlp_fval, opts)
    % Polynomial problem passed to CSTSS_mex (same convention as test_CSTSS_Python):
    %   minimize f(z)  s.t.  g(z) >= 0 (inequalities),  h(z) == 0 (equalities).
    % Here: f = weighted shedding; h stacks mass balance + pipe/short Weymouth
    % equalities + q_abs^2 - q^2 = 0; g stacks all variable box bounds, q_abs>=0,
    % and compressor ratio inequalities. Pressures z(i_p2) are **bar**; Weymouth
    % uses Pa2_per_bar2 only inside res_over_pa2 = res_factor/Pa2_per_bar2.
    % Optional: opts.pop_ball_constraints adds 1 - (x_phys(i)/c_i)^2 >= 0 per variable
    % (same as c_i^2 - x_phys(i)^2 >= 0 for c_i > 0), with x_phys(i)=opts.x_scale(i)*z(i).
    %
    % MOSEK "primal infeasible" on the moment SDP at fixed opts.kappa does *not*
    % certify that the NLP (or the POP) is infeasible: the truncated relaxation
    % can be infeasible while a higher kappa or different cs_mode still yields a
    % feasible relaxation; also check scaling / ill-conditioning.
    sdp = struct();

    if exist('msspoly', 'class') ~= 8
        warning('msspoly/SPOT not found. Skipping SDP relaxation.');
        sdp.status = 'skipped_no_msspoly';
        return;
    end
    if exist('CSTSS_mex', 'file') ~= 3 && exist('CSTSS_mex', 'file') ~= 2
        warning('CSTSS_mex not found. Skipping SDP relaxation.');
        sdp.status = 'skipped_no_cstss';
        return;
    end

    [z, objective, inequality, equality] = build_gaslib_pop_msspoly(data, Aeq, beq, lb, ub, opts);

    n_p2 = data.N;
    n_q = data.M;
    n_qabs = numel(data.var.i_qabs);
    n_g = numel(data.var.i_g);
    n_shed = numel(data.var.i_shed);
    fprintf(['POP decision variables (hat indeterminates z): %d total\n' ...
        '  p2=%d  q=%d  qabs=%d  g=%d  shed=%d  (x_phys(i)=opts.x_scale(i)*z(i))\n'], ...
        data.var.nvar, n_p2, n_q, n_qabs, n_g, n_shed);
    fprintf('POP constraints (before msspoly_clean): %d equalities, %d inequalities\n', ...
        numel(equality), numel(inequality));

    [equality, ~] = msspoly_clean(equality, z, 1e-14, true);
    [inequality, ~] = msspoly_clean(inequality, z, 1e-14, true);
    [objective, obj_scale_factor] = msspoly_clean(objective, z, 1e-14, true);
    disp('Construction Finish!');
    fprintf('obj_scale_factor: %.3e\n', obj_scale_factor);

    if_mex = true; params.if_mex = if_mex;
    kappa = opts.kappa; params.kappa = kappa;
    relax_mode = 'SOS'; params.relax_mode = relax_mode;
    cs_mode = char(opts.cs_mode);
    params.cs_mode = cs_mode;
    ts_mode = 'NON'; params.ts_mode = ts_mode;
    ts_mom_mode = 'NON'; params.ts_mom_mode = ts_mom_mode;
    ts_eq_mode = 'NON'; params.ts_eq_mode = ts_eq_mode;
    if_solve = true; params.if_solve = if_solve;
    params.cliques = [];

    [result, res, coeff_info, aux_info] = CSTSS_mex(objective, inequality, equality, kappa, z, params); %#ok<ASGLU>

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
    total_var_num = length(z);
    v_opt_robust = [];
    output_info_robust = [];
    if strcmp(ts_mode, 'NON')
        [v_opt_robust, output_info_robust] = robust_extract_CS(Xs, mom_mat_rpt, total_var_num, 1e-2);
    end
    % naive extraction
    [v_opt_naive, output_info_naive] = naive_extract(Xs, mom_mat_rpt, total_var_num);

    aux_info.total_var_num = total_var_num;
    sdp.relax_info = aux_info;
    sdp.status = 'solved';
    sdp.result = result;
    sdp.res = res;
    sdp.Xs = Xs;
    sdp.lower_bound = obj(1) * obj_scale_factor;
    sdp.nlp_upper_bound = nlp_fval;
    sdp.relative_gap = abs(nlp_fval - obj(1) * obj_scale_factor) / (1 + abs(nlp_fval) + abs(obj(1) * obj_scale_factor));
    sdp.v_opt_naive = v_opt_naive;
    sdp.output_info_naive = output_info_naive;

    if ~isempty(v_opt_robust)
        sdp.zopt = v_opt_robust(:);
        sdp.output_info_robust = output_info_robust;
        sdp.output_info = output_info_robust; %#ok<NASGU> legacy alias
        try
            sdp.robust_extract_nlp = gaslib_eval_fmincon_stack_at_zhat(v_opt_robust(:), data, Aeq, beq, lb, ub, opts);
        catch ME
            warning('run_gaslib_mgsp:RobustExtractEval', ...
                'Could not evaluate robust extract on NLP stack: %s', ME.message);
            sdp.robust_extract_nlp = [];
        end

        sdp.x_hat_refined = [];
        sdp.x_refined = [];
        sdp.fval_refined = [];
        sdp.exitflag_refined = [];
        sdp.output_refined = [];
        if opts.refine_robust_extract
            try
                [xh, fv, ef, out, xph] = gaslib_fmincon_refine_hat( ...
                    v_opt_robust(:), data, Aeq, beq, lb, ub, opts);
                sdp.x_hat_refined = xh;
                sdp.x_refined = xph;
                sdp.fval_refined = fv;
                sdp.exitflag_refined = ef;
                sdp.output_refined = out;
            catch ME
                warning('run_gaslib_mgsp:HatRefineFromExtract', ...
                    'Hat-space fmincon refinement from robust extract failed: %s', ME.message);
            end
        end
    else
        sdp.zopt = [];
        sdp.output_info_robust = [];
        sdp.output_info = [];
        sdp.robust_extract_nlp = [];
        sdp.x_hat_refined = [];
        sdp.x_refined = [];
        sdp.fval_refined = [];
        sdp.exitflag_refined = [];
        sdp.output_refined = [];
    end

    fprintf('\n=== SDP relaxation ===\n');
    fprintf('Upper bound from NLP : %.9g\n', nlp_fval);
    fprintf('Lower bound from SDP : %.9g\n', sdp.lower_bound);
    fprintf('Relative gap         : %.3e\n', sdp.relative_gap);
    if ~isempty(v_opt_robust) && ~isempty(sdp.robust_extract_nlp)
        r = sdp.robust_extract_nlp;
        fprintf('\n--- Robust extract vs NLP (same model as run_gaslib_mgsp fmincon) ---\n');
        fprintf('Objective (weighted shedding, physical): %.9g\n', r.fval_physical);
        fprintf('Linear |Aeq*x-beq|_inf                  : %.3e\n', r.lin_resid_physical);
        fprintf('Nonlinear max(0,-c) (gaslib_nonlcon)    : %.3e\n', r.max_ineq_nonlin);
        fprintf('Nonlinear max|ceq|                      : %.3e\n', r.max_abs_ceq);
        fprintf('Box max lb violation (finite lb)        : %.3e\n', r.max_lb_viol);
        fprintf('Box max ub violation (finite ub)        : %.3e\n', r.max_ub_viol);
        fprintf('Hat space |Aeq_hat*xhat-beq|_inf         : %.3e\n', r.lin_resid_hat);
        fprintf('Hat max lb / ub violation               : %.3e / %.3e\n', r.max_lb_hat_viol, r.max_ub_hat_viol);
        fprintf('POP (full CSTSS stack) f, min(g), max|h|: %.6g, %.3e, %.3e\n', ...
            r.pop_f, r.pop_min_g, r.pop_max_abs_h);
        fprintf('POP max(0,-g), feasible (tol=%.1e)      : %.3e, %d\n', ...
            opts.pop_feas_tol, r.pop_max_ineq_viol, r.pop_feasible);
        if r.fmincon_zero_iter_ok
            fprintf('fmincon MaxIter=0 at extract: f=%.9g exitflag=%d\n', r.fmincon_fval, r.fmincon_exitflag);
        end
    end
    if opts.refine_robust_extract && ~isempty(v_opt_robust) && ~isempty(sdp.x_hat_refined)
        fprintf('\n--- Hat-space fmincon refined from SDP robust extract ---\n');
        fprintf('Objective (same as main NLP fval)         : %.9g  exitflag=%d\n', ...
            sdp.fval_refined, sdp.exitflag_refined);
    end
end

function [x_hat, fval, exitflag, output, x_phys] = gaslib_fmincon_refine_hat(x0_hat, data, Aeq, beq, lb, ub, opts)
    % Refine SDP robust extract in hat space (identical NLP to run_gaslib_mgsp fmincon).
    s = opts.x_scale(:);
    Aeq_hat = Aeq * diag(s);
    lb_hat = scale_bounds_for_hat(lb, s, 'down');
    ub_hat = scale_bounds_for_hat(ub, s, 'down');
    n = data.var.nvar;
    w_lin = zeros(n, 1);
    if ~isempty(data.var.i_shed)
        w_lin(data.var.i_shed) = data.shed_weights(:) .* s(data.var.i_shed);
    end
    obj_hat = @(xh) w_lin(:).' * xh(:);
    nonlcon_hat = @(xh) gaslib_nonlcon(xh .* s, data);

    if isempty(opts.fmincon_options)
        fopts = optimoptions('fmincon', ...
            'Algorithm', 'interior-point', ...
            'Display', 'iter', ...
            'SpecifyObjectiveGradient', false, ...
            'SpecifyConstraintGradient', false, ...
            'MaxIterations', 1000, ...
            'MaxFunctionEvaluations', 50000, ...
            'OptimalityTolerance', 1e-8, ...
            'StepTolerance', 1e-10, ...
            'ConstraintTolerance', 1e-8);
    else
        fopts = opts.fmincon_options;
    end

    [x_hat, fval, exitflag, output] = fmincon( ...
        obj_hat, x0_hat(:), [], [], Aeq_hat, beq, lb_hat, ub_hat, nonlcon_hat, fopts);
    x_phys = x_hat .* s;
end

function r = gaslib_eval_fmincon_stack_at_zhat(x_hat, data, Aeq, beq, lb, ub, opts)
    % Same algebraic objective and constraints as run_gaslib_mgsp's fmincon call:
    %   x_phys = x_scale .* x_hat,  minimize w_lin'*x_hat  s.t.  Aeq_hat*x_hat=beq,
    %   lb_hat <= x_hat <= ub_hat,  nonlcon_hat(x_hat) = gaslib_nonlcon(x_phys).
    r = struct();
    x_hat = x_hat(:);
    s = opts.x_scale(:);
    n = data.var.nvar;
    if numel(x_hat) ~= n
        error('run_gaslib_mgsp:ZhatLength', 'Extracted vector length %d ~= nvar %d.', numel(x_hat), n);
    end

    x_phys = x_hat .* s;
    r.x_hat = x_hat;
    r.x_phys = x_phys;
    w_lin = zeros(n, 1);
    if ~isempty(data.var.i_shed)
        w_lin(data.var.i_shed) = data.shed_weights(:) .* s(data.var.i_shed);
    end
    r.fval_hat = w_lin(:).' * x_hat;
    if isempty(data.var.i_shed)
        r.fval_physical = 0;
    else
        r.fval_physical = data.shed_weights(:).' * x_phys(data.var.i_shed);
    end

    r.lin_resid_physical = norm(Aeq * x_phys - beq, inf);
    Aeq_hat = Aeq * diag(s);
    r.lin_resid_hat = norm(Aeq_hat * x_hat - beq, inf);

    [c, ceq] = gaslib_nonlcon(x_phys, data);
    r.max_ineq_nonlin = max([c(:); 0]);
    r.max_abs_ceq = norm(ceq(:), inf);

    mask_lb = isfinite(lb);
    mask_ub = isfinite(ub);
    r.max_lb_viol = 0;
    if any(mask_lb)
        r.max_lb_viol = max(max(0, lb(mask_lb) - x_phys(mask_lb)));
    end
    r.max_ub_viol = 0;
    if any(mask_ub)
        r.max_ub_viol = max(max(0, x_phys(mask_ub) - ub(mask_ub)));
    end

    lb_hat = scale_bounds_for_hat(lb, s, 'down');
    ub_hat = scale_bounds_for_hat(ub, s, 'down');
    r.max_lb_hat_viol = 0;
    if any(mask_lb)
        r.max_lb_hat_viol = max(max(0, lb_hat(mask_lb) - x_hat(mask_lb)));
    end
    r.max_ub_hat_viol = 0;
    if any(mask_ub)
        r.max_ub_hat_viol = max(max(0, x_hat(mask_ub) - ub_hat(mask_ub)));
    end

    [f_pop, g_pop, h_pop] = gaslib_pop_numeric_fgh(x_phys, data, Aeq, beq, lb, ub, opts);
    r.pop_f = f_pop;
    r.pop_min_g = min(g_pop);
    r.pop_max_abs_h = norm(h_pop, inf);
    r.pop_max_ineq_viol = max(max(-g_pop, 0));
    r.pop_feasible = (r.pop_min_g >= -opts.pop_feas_tol) && (r.pop_max_abs_h <= opts.pop_feas_tol);

    obj_hat = @(xh) w_lin(:).' * xh(:);
    nonlcon_hat = @(xh) gaslib_nonlcon(xh .* s, data);
    r.fmincon_zero_iter_ok = false;
    r.fmincon_fval = NaN;
    r.fmincon_exitflag = NaN;
    try
        o0 = optimoptions('fmincon', ...
            'Algorithm', 'interior-point', ...
            'Display', 'off', ...
            'MaxIterations', 0, ...
            'OptimalityTolerance', 1e100, ...
            'StepTolerance', 1e100, ...
            'ConstraintTolerance', 1e100, ...
            'MaxFunctionEvaluations', 1e9);
        [~, r.fmincon_fval, r.fmincon_exitflag] = fmincon( ...
            obj_hat, x_hat, [], [], Aeq_hat, beq, lb_hat, ub_hat, nonlcon_hat, o0);
        r.fmincon_zero_iter_ok = true;
    catch
        % Some MATLAB versions reject MaxIterations = 0; algebraic checks above still apply.
    end
end

% =======================================================================
% XML helpers
% =======================================================================

function kids = child_elements(parent)
    kids = {};
    if isempty(parent)
        return;
    end
    nodes = parent.getChildNodes();
    for k = 1:nodes.getLength()
        item = nodes.item(k-1);
        if item.getNodeType() == 1
            kids{end+1} = item; %#ok<AGROW>
        end
    end
end

function name = local_name(node)
    if isempty(node)
        name = '';
        return;
    end
    name = char(node.getNodeName());
    idx = find(name == ':', 1, 'last');
    if ~isempty(idx)
        name = name(idx+1:end);
    end
end

function child = find_first_child(parent, wanted_name)
    child = [];
    if isempty(parent)
        return;
    end
    kids = child_elements(parent);
    for k = 1:numel(kids)
        if strcmp(local_name(kids{k}), wanted_name)
            child = kids{k};
            return;
        end
    end
end

function val = get_attr(node, attr_name)
    val = '';
    if isempty(node)
        return;
    end
    if node.hasAttribute(attr_name)
        val = char(node.getAttribute(attr_name));
    end
end

function out = get_value_unit(parent, child_name)
    out = [];
    child = find_first_child(parent, child_name);
    if isempty(child)
        return;
    end
    out = struct();
    out.value = str2double(get_attr(child, 'value'));
    out.unit = get_attr(child, 'unit');
end

% =======================================================================
% Unit conversion helpers
% =======================================================================

function T = convert_temperature_to_K(val, unit)
    switch unit
        case {'K', 'Kelvin'}
            T = val;
        case 'Celsius'
            T = val + 273.15;
        otherwise
            error('Unsupported temperature unit: %s', unit);
    end
end

function rho = convert_density_to_kgpm3(val, unit)
    switch unit
        case 'kg_per_m_cube'
            rho = val;
        otherwise
            error('Unsupported density unit: %s', unit);
    end
end

function M = convert_molarmass_to_kgpmol(val, unit)
    switch unit
        case 'kg_per_kmol'
            M = 1e-3 * val;
        case 'kg_per_mol'
            M = val;
        otherwise
            error('Unsupported molar mass unit: %s', unit);
    end
end

function Hs = convert_calorific_to_MJpm3(val, unit)
    switch unit
        case 'MJ_per_m_cube'
            Hs = val;
        case 'kJ_per_m_cube'
            Hs = 1e-3 * val;
        otherwise
            error('Unsupported calorific value unit: %s', unit);
    end
end

function L = convert_length_to_m(val, unit)
    switch unit
        case 'm'
            L = val;
        case 'mm'
            L = 1e-3 * val;
        case 'km'
            L = 1e3 * val;
        otherwise
            error('Unsupported length unit: %s', unit);
    end
end

function pbar = convert_pressure_to_bar_abs(val, unit, p_atm_bar)
    switch unit
        case 'bar'
            pbar = val;
        case 'barg'
            pbar = val + p_atm_bar;
        case 'Pa'
            pbar = val / 1e5;
        case 'kPa'
            pbar = val / 100;
        case 'MPa'
            pbar = 10 * val;
        otherwise
            error('Unsupported pressure unit: %s', unit);
    end
end

function f = convert_flow_to_kgps(val, unit, rho0, Hs_MJpm3)
    switch unit
        case '1000m_cube_per_hour'
            f = rho0 * val / 3.6;
        case 'm_cube_per_hour'
            f = rho0 * val / 3600;
        case 'm_cube_per_s'
            f = rho0 * val;
        case 'kg_per_s'
            f = val;
        case 'kg_per_hour'
            f = val / 3600;
        case 'MW'
            if isnan(Hs_MJpm3)
                error('Power-based scenario flow encountered, but no calorificValue is available.');
            end
            f = rho0 * val / Hs_MJpm3;  % MW = MJ/s
        case 'kW'
            if isnan(Hs_MJpm3)
                error('Power-based scenario flow encountered, but no calorificValue is available.');
            end
            f = rho0 * (1e-3 * val) / Hs_MJpm3;  % kW = 1e-3 MJ/s
        otherwise
            error('Unsupported flow/power unit: %s', unit);
    end
end

% =======================================================================
% Misc helpers
% =======================================================================

function cap = pick_entry_cap(lb, ub)
    if ~isnan(ub)
        cap = ub;
    elseif ~isnan(lb)
        cap = lb;
    else
        cap = 0;
    end
end

function dem = pick_exit_demand(lb, ub)
    if ~isnan(lb)
        dem = lb;
    elseif ~isnan(ub)
        dem = ub;
    else
        dem = 0;
    end
end

function w = lookup_weight(weight_struct, node_id, default_w)
    fname = matlab.lang.makeValidName(node_id);
    if isstruct(weight_struct) && isfield(weight_struct, fname)
        w = weight_struct.(fname);
    else
        w = default_w;
    end
end

function opts = set_default_opts(opts)
    opts = set_opt(opts, 'z', 1.0);
    opts = set_opt(opts, 'passive_bidirectional', true);
    opts = set_opt(opts, 'enforce_compressor_forward', true);
    opts = set_opt(opts, 'weight_default', 1.0);
    opts = set_opt(opts, 'weights', struct());
    opts = set_opt(opts, 'run_sdp', false);
    opts = set_opt(opts, 'kappa', 2);
    opts = set_opt(opts, 'cs_mode', 'MF');
    opts = set_opt(opts, 'pop_feas_tol', 1e-7);
    opts = set_opt(opts, 'report_pop_coeff_ranges', false);
    opts = set_opt(opts, 'pop_ball_constraints', false);
    opts = set_opt(opts, 'pop_ball_c_inf', 100);
    opts = set_opt(opts, 'x_scale', []);
    opts = set_opt(opts, 'fmincon_options', []);
    opts = set_opt(opts, 'refine_robust_extract', true);
    opts = set_opt(opts, 'verify_nlp_pop_hat', false);
    opts = set_opt(opts, 'verify_nlp_pop_hat_samples', 5);
    opts = set_opt(opts, 'verify_nlp_pop_hat_noise', 0.05);
    opts = set_opt(opts, 'verify_nlp_pop_hat_tol', 1e-8);
    opts = set_opt(opts, 'verify_nlp_pop_hat_seed', []);
    opts = set_opt(opts, 'verbose', true);
end

function opts = ensure_x_scale(opts, ub, data)
    n = data.var.nvar;
    if ~isempty(opts.x_scale)
        if numel(opts.x_scale) ~= n
            error('run_gaslib_mgsp:xScaleSize', ...
                'opts.x_scale must have length nvar=%d (got %d).', n, numel(opts.x_scale));
        end
        opts.x_scale = opts.x_scale(:);
        return;
    end
    s = zeros(n, 1);
    for i = 1:n
        if isfinite(ub(i))
            s(i) = max(0.5 * ub(i), 100 * eps);
        else
            s(i) = 0.5 * opts.pop_ball_c_inf;
        end
    end
    opts.x_scale = s;
end

function b2 = scale_bounds_for_hat(b, s, dir)
    b = b(:);
    s = s(:);
    b2 = b;
    switch dir
        case 'down'
            fin = isfinite(b) & isfinite(s) & (s ~= 0);
            b2(fin) = b(fin) ./ s(fin);
        case 'up'
            fin = isfinite(b) & isfinite(s) & (s ~= 0);
            b2(fin) = b(fin) .* s(fin);
        otherwise
            error('run_gaslib_mgsp:scale_bounds', 'dir must be ''down'' or ''up''.');
    end
end

function s = set_opt(s, name, value)
    if ~isfield(s, name)
        s.(name) = value;
    end
end

function y = ternary(cond, a, b)
    if cond
        y = a;
    else
        y = b;
    end
end
